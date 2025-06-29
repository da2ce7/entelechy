# kernel_signatures/phase_2_learn_B_processing.py

"""
Concrete KernelSignature Implementations for Gradient Processing (Nodes 11-13).

This file contains the final, canonical implementations for the kernel launch
signatures related to the critical second stage of the 'Learn' phase. These
kernels are responsible for stabilizing initial partial gradients (clipping),
performing general-purpose layout transformations (transpose), and executing
the architecturally-mandated synchronization point for the hidden layer
gradient (`Grad_H`).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..launcher_infra import BufferHandle, KernelSignature, WorkTile, SCALAR_NP_TYPE
from ..memory_layout import _pad_to_multiple


# === Node 11: Clip Partial Gradients (Stability Primitive) ===


@dataclass
class GradientHandles:
    """A helper dataclass to declutter the ClipPartialGradientsSignature constructor."""

    # Input Partial Gradients (from Nodes 8, 9, 10)
    grad_weights_module: BufferHandle
    grad_biases_module: BufferHandle
    grad_temps: BufferHandle
    grad_hidden_activations_aos: BufferHandle
    # Output Clipped Partial Gradients
    clipped_grad_weights_module: BufferHandle
    clipped_grad_biases_module: BufferHandle
    clipped_grad_temps: BufferHandle
    clipped_grad_hidden_activations_aos: BufferHandle


@dataclass(frozen=True)
class _ClipPartialGradientsBase(KernelSignature):
    """(Internal) Shared base for the Node 11 gradient clipping kernel."""

    work_group_size_0: int
    scalar_size_bytes: int
    handles: GradientHandles
    tile: WorkTile
    epsilon: SCALAR_NP_TYPE
    padded_hidden_count: np.uint32 = field(init=False)
    total_batch_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        grad_h_shape, _ = self._buffer_mgr.get_spec(self.handles.grad_hidden_activations_aos)
        object.__setattr__(self, "total_tile_count", np.uint32(grad_h_shape[0]))
        object.__setattr__(self, "total_batch_count", np.uint32(grad_h_shape[2]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(grad_h_shape[3]))

    @property
    def kernel_name(self) -> str:
        return "clip_partial_gradients"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # Grid Calculation Strategy: The kernel calculates the L2 norm for all
        # gradients associated with a single item (tile). We launch enough
        # work-items to cover all these elements, which then cooperate on the
        # reduction using local memory.
        total_elements_in_tile = (
            (self.padded_hidden_count * self.tile.classes_per_chunk)  # weights
            + self.tile.classes_per_chunk  # biases
            + 1  # temps
            + self.padded_hidden_count  # hidden_activations
        ) * self.tile.modules_per_chunk
        global_size = (_pad_to_multiple(int(total_elements_in_tile), self.work_group_size_0),)
        local_size = (self.work_group_size_0,)
        return global_size, local_size


@dataclass(frozen=True)
class ClipPartialGradientsGlobalNormSignature(_ClipPartialGradientsBase):
    """(Node 11 - Global) Type-safe signature for clipping with a single global norm."""

    max_norm_global: SCALAR_NP_TYPE

    def get_args(self) -> List:
        """Assembles all 20 arguments, injecting the '0' flag and the global norm value."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        h = self.handles
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(h.grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.grad_temps),
            self._buffer_mgr.get_cl_buffer(h.grad_hidden_activations_aos),
            None,  # per_item_norm buffer is NULL
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_temps),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_hidden_activations_aos),
            np.uint32(0),  # use_per_item_norm = FALSE
            self.max_norm_global,
            self.epsilon,
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.padded_hidden_count,
            self.total_tile_count,
        ]


@dataclass(frozen=True)
class ClipPartialGradientsPerItemNormSignature(_ClipPartialGradientsBase):
    """(Node 11 - Per-Item) Type-safe signature for clipping with a per-item norm buffer."""

    max_norm_per_item_ref: BufferHandle

    def get_args(self) -> List:
        """Assembles all 20 arguments, injecting the '1' flag and the per-item norm buffer."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        h = self.handles
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(h.grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.grad_temps),
            self._buffer_mgr.get_cl_buffer(h.grad_hidden_activations_aos),
            self._buffer_mgr.get_cl_buffer(self.max_norm_per_item_ref),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_weights_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_biases_module),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_temps),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_hidden_activations_aos),
            np.uint32(1),  # use_per_item_norm = TRUE
            SCALAR_NP_TYPE(0.0),  # global_norm value is ignored by kernel
            self.epsilon,
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.padded_hidden_count,
            self.total_tile_count,
        ]


# === Node 13: Gather & Permute Grad_H (Item Synchronization Point) ===


@dataclass(frozen=True)
class GatherAndPermuteGradHSignature(KernelSignature):
    """
    (REV 2) Signature for the `gather_and_permute_grad_hidden_activations` kernel.

    This version is architecturally rectified. It no longer attempts to derive
    the host's tiling plan. Instead, it accepts all dimensional parameters
    explicitly, fulfilling its sole contract of being a 1:1 representation of the
    C-level kernel interface. Its validation logic now serves as the final
    assurance check against the host's provided plan.
    """

    # --- Buffer Handles (from kernel contract) ---
    clipped_partials_aos_ref: BufferHandle
    permuted_soa_out_ref: BufferHandle

    # --- High-Level Dimensional Parameters (Now explicitly passed in) ---
    total_modules_count: np.uint32
    hidden_count: np.uint32
    total_batch_count: np.uint32
    num_module_chunks_count: np.uint32
    modules_per_chunk_count: np.uint32
    num_class_chunks_count: np.uint32

    # --- Derived *Padded* Dimensions & Total Counts ---
    # These are the only values derived, as they are properties of the memory, not the plan.
    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """
        Derives physical (padded) dimensions from buffer specs and performs
        host-side validation of the provided plan against memory allocations.
        """
        aos_shape, _ = self._buffer_mgr.get_spec(self.clipped_partials_aos_ref)
        soa_shape, _ = self._buffer_mgr.get_spec(self.permuted_soa_out_ref)

        # Derive physical memory properties
        object.__setattr__(self, "total_tile_count", np.uint32(aos_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(aos_shape[3]))
        object.__setattr__(self, "padded_total_modules_count", np.uint32(soa_shape[1]))

        # --- VALIDATION LOGIC (The Signature's True Responsibility) ---
        # 1. Validate the host's plan against itself for internal consistency.
        assert self.total_tile_count == self.num_module_chunks_count * self.num_class_chunks_count, (
            f"Host plan inconsistency: total_tiles ({self.total_tile_count}) does not match "
            f"num_module_chunks ({self.num_module_chunks_count}) * num_class_chunks ({self.num_class_chunks_count})."
        )

        # 2. Validate the host's plan against the physical buffer allocations.
        assert aos_shape[1] == self.modules_per_chunk_count, (
            f"Buffer spec mismatch: Clipped partials buffer expects {aos_shape[1]} modules per chunk, "
            f"but plan requires {self.modules_per_chunk_count}."
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
        # Dispatch one work-item per element in the output buffer
        global_size = (self.total_batch_count * self.padded_hidden_count, self.padded_total_modules_count)
        return global_size, None

    def get_args(self) -> List:
        """
        Returns all 11 arguments in exact contractual order. The arguments are
        now guaranteed to be consistent by the __post_init__ validation.
        """
        return [
            self._buffer_mgr.get_cl_buffer(self.clipped_partials_aos_ref),
            self._buffer_mgr.get_cl_buffer(self.permuted_soa_out_ref),
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_modules_count,
            self.padded_total_modules_count,
            self.num_module_chunks_count,
            self.modules_per_chunk_count,
            self.num_class_chunks_count,
            self.total_tile_count,
        ]
