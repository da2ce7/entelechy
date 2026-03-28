# kernel_signatures/phase_2_learn_B_processing.py

"""
The Definitive, Executable Contracts for Gradient Processing (Nodes 11-13).

Jurisdictional Mandate:
This file is the canonical Python-side embodiment of the C-level kernel
contracts for the second, critical stage of the 'Learn' phase. Its
jurisdiction covers the transformation of raw gradients into numerically stable,
reduction-ready forms. This includes the foundational stability primitive
(Node 11, clipping) and the system's primary Item Synchronization Point
(Node 13, gather/permute).

Architectural Role:
The classes herein are not mere data containers; they are immutable 'artisans'
that encapsulate a single, verifiable dispatch. Their design upholds two core
architectural tenets:
  1. Polymorphic Correctness: By using a base-class hierarchy for Node 11, we
     encode the clipping strategy (global vs. per-item) into the type system,
     absolving higher-level recipes of managing complex state.
  2. Plan Validation: The `GatherAndPermute...` signature's primary role is
     to act as a final validation gate, asserting that the host's strategic
     ExecutionPlan is physically realizable within the allocated VRAM, thus
     preventing runtime errors through build-time verification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Primitives & Core Infrastructure ---
from ....shared.workload_primitives import WorkTile
from ..launcher_infra import BufferHandle, KernelSignature, BufferManager
from ....shared.memory_layout import pad_to_multiple
from ..context import DiscoveredArchConstants


# =========================================================================
# === API Clarification Primitives ===
# =========================================================================


@dataclass
class GradientHandles:
    """
    A helper dataclass to simplify the API for the complex `clip_partial_gradients`
    kernel. It groups the numerous, logically-related gradient buffers into a
    single, coherent unit, making the kernel signature's interface cleaner and
    its intent clearer.
    """

    # Raw, unclipped partials (inputs to the kernel)
    grad_weights_module: BufferHandle
    grad_biases_module: BufferHandle
    grad_temps: BufferHandle
    grad_hidden_activations_aos: BufferHandle
    # Clipped partials (outputs from the kernel)
    clipped_grad_weights_module: BufferHandle
    clipped_grad_biases_module: BufferHandle
    clipped_grad_temps: BufferHandle
    clipped_grad_hidden_activations_aos: BufferHandle


# =========================================================================
# === Node 11: The Foundational Stability Primitive (clip_partial_gradients)
# =========================================================================


@dataclass(frozen=True)
class _ClipTiledModuleGradsBase(KernelSignature):
    """
    (Internal) The unified base for all Node 11 clipping operations.

    This class embodies the core contract of the `clip_partial_gradients`
    kernel: to treat the complete set of partial gradients for a single logical
    work item (a 'tile') as one unified vector, compute its L2 norm, and
    conditionally apply a single scaling factor. This is the system's primary
    defense against numerical overflow and the guarantor of stability for the
    subsequent reduction engine.
    """

    # --- Injected System Context (The Architectural Mandate) ---
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    # --- Kernel-Specific Needs ---
    handles: GradientHandles
    tile: WorkTile
    epsilon: np.float32

    # --- Derived Fields ---
    padded_hidden_count: np.uint32 = field(init=False)
    padded_class_count: np.uint32 = field(init=False)
    total_batch_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives physical dimensions from the injected memory context."""
        super().__post_init__()
        grad_h_shape, _ = self._buffer_mgr.get_spec(self.handles.grad_hidden_activations_aos)
        grad_w_shape, _ = self._buffer_mgr.get_spec(self.handles.grad_weights_module)
        object.__setattr__(self, "total_tile_count", np.uint32(grad_h_shape[0]))
        object.__setattr__(self, "total_batch_count", np.uint32(grad_h_shape[2]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(grad_h_shape[3]))
        object.__setattr__(self, "padded_class_count", np.uint32(grad_w_shape[3]))

    @property
    def kernel_name(self) -> str:
        return "clip_partial_gradients"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """
        Calculates a grid size based on the total number of elements that
        constitute a single, complete logical tile. This ensures that one
        work-group is dispatched to atomically process the entire gradient vector
        for one parallel work item.
        """
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        total_elements_in_tile = (
            (self.padded_hidden_count * self.padded_class_count)  # weights
            + self.padded_class_count  # biases
            + 1  # temps
            + self.padded_hidden_count  # hidden_activations
        ) * self.tile.modules_per_chunk
        global_size = (pad_to_multiple(int(total_elements_in_tile), work_group_size),)
        local_size = (work_group_size,)
        return global_size, local_size


@dataclass(frozen=True)
class ClipPartialGradientsGlobalNormSignature(_ClipTiledModuleGradsBase):
    """(Node 11 - Global) The public signature for clipping with a single, batch-wide threshold."""

    clipping_threshold_global: np.float32

    def get_args(self) -> List:
        """Assembles arguments, setting the `use_per_item_norm` flag to FALSE (0)."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = work_group_size * scalar_size_bytes
        h = self.handles
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(h.grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.grad_temps),
            self._buffer_mgr.get_cl_buffer(h.grad_hidden_activations_aos),
            None,  # Pass NULL for the unused per-item norm buffer
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_temps),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_hidden_activations_aos),
            np.uint32(0),  # FLAG: use_per_item_norm = FALSE
            self.clipping_threshold_global,
            self.epsilon,
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.padded_hidden_count,
            self.padded_class_count,
            self.total_tile_count,
        ]


@dataclass(frozen=True)
class ClipPartialGradientsPerItemNormSignature(_ClipTiledModuleGradsBase):
    """(Node 11 - Per-Item) The public signature for clipping with a per-item threshold buffer."""

    clipping_threshold_per_item_ref: BufferHandle

    def get_args(self) -> List:
        """Assembles arguments, setting the `use_per_item_norm` flag to TRUE (1)."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = work_group_size * scalar_size_bytes
        h = self.handles
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(h.grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.grad_temps),
            self._buffer_mgr.get_cl_buffer(h.grad_hidden_activations_aos),
            self._buffer_mgr.get_cl_buffer(self.clipping_threshold_per_item_ref),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_temps),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_hidden_activations_aos),
            np.uint32(1),  # FLAG: use_per_item_norm = TRUE
            np.float32(0.0),  # Pass a dummy value for the unused global norm
            self.epsilon,
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.padded_hidden_count,
            self.padded_class_count,
            self.total_tile_count,
        ]


# =========================================================================
# === Node 13: The Canonical Item Synchronization Point
# =========================================================================


@dataclass(frozen=True)
class GatherAndPermuteGradHiddenActivationsSignature(KernelSignature):
    """
    (Node 13) Signature for the `gather_and_permute_grad_hidden_activations` kernel.

    This signature represents a critical architectural primitive: the Item
    Synchronization Point. It gathers the scattered, clipped partial `Grad_H`
    results and permutes them from an inefficient Array-of-Structs (AoS) memory
    layout to a reduction-ready Struct-of-Arrays (SoA) layout.

    Its `__post_init__` method serves as a mandatory validation gate, asserting
    that the host's high-level tiling plan is consistent with the physical
    memory dimensions allocated for its operation.
    """

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    clipped_partials_aos_ref: BufferHandle
    permuted_soa_out_ref: BufferHandle

    total_modules_count: np.uint32
    hidden_count: np.uint32
    total_batch_count: np.uint32
    num_module_chunks: np.uint32
    modules_per_chunk: np.uint32
    num_class_chunks: np.uint32

    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives physical dimensions and validates the host's plan."""
        super().__post_init__()
        aos_shape, _ = self._buffer_mgr.get_spec(self.clipped_partials_aos_ref)
        soa_shape, _ = self._buffer_mgr.get_spec(self.permuted_soa_out_ref)

        object.__setattr__(self, "total_tile_count", np.uint32(aos_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(aos_shape[3]))
        object.__setattr__(self, "padded_total_modules_count", np.uint32(soa_shape[1]))

        # --- Contractual Verification (Host-Side Assurance) ---
        # This block is the system's guarantee that the host's strategy is
        # physically realizable in the memory it has allocated.
        assert self.total_tile_count == self.num_module_chunks * self.num_class_chunks, (
            f"Host plan inconsistency: total_tiles ({self.total_tile_count}) does not match "
            f"num_module_chunks ({self.num_module_chunks}) * num_class_chunks ({self.num_class_chunks})."
        )
        assert aos_shape[1] == self.modules_per_chunk, (
            f"Buffer spec mismatch: Clipped partials buffer expects {aos_shape[1]} modules per chunk, "
            f"but plan requires {self.modules_per_chunk}."
        )
        assert aos_shape[2] == self.total_batch_count, (
            f"Buffer spec mismatch: Clipped partials buffer expects batch size {aos_shape[2]}, "
            f"but plan requires {self.total_batch_count}."
        )
        assert soa_shape[0] == self.total_batch_count * self.padded_hidden_count, (
            f"Buffer spec mismatch: Permuted SoA output expects {soa_shape[0]} rows, "
            f"but plan requires {self.total_batch_count * self.padded_hidden_count}."
        )

    @property
    def kernel_name(self) -> str:
        return "gather_and_permute_grad_hidden_activations"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (
            int(self.total_batch_count) * int(self.padded_hidden_count),
            int(self.padded_total_modules_count),
        )
        return global_size, None

    def get_args(self) -> List:
        """Returns all 11 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.clipped_partials_aos_ref),
            self._buffer_mgr.get_cl_buffer(self.permuted_soa_out_ref),
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_modules_count,
            self.padded_total_modules_count,
            self.num_module_chunks,
            self.modules_per_chunk,
            self.num_class_chunks,
            self.total_tile_count,
        ]
