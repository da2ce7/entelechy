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


# === Node 12: Transpose Chunk (Utility) ===


@dataclass(frozen=True)
class TransposeChunkSignature(KernelSignature):
    """(Node 12) Signature for the general-purpose `transpose_chunk` kernel."""

    c_tile_size: int
    local_mem_bank_padding: int
    scalar_size_bytes: int
    in_ref: BufferHandle
    out_ref: BufferHandle
    in_offset: np.uint32
    out_offset: np.uint32
    height: np.uint32
    width: np.uint32
    in_stride: np.uint32
    out_stride: np.uint32
    in_total_element_count: np.uint32 = field(init=False)
    out_total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        in_shape, _ = self._buffer_mgr.get_spec(self.in_ref)
        out_shape, _ = self._buffer_mgr.get_spec(self.out_ref)
        object.__setattr__(self, "in_total_element_count", np.uint32(np.prod(in_shape)))
        object.__setattr__(self, "out_total_element_count", np.uint32(np.prod(out_shape)))

    @property
    def kernel_name(self) -> str:
        return "transpose_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (_pad_to_multiple(self.width, self.c_tile_size), _pad_to_multiple(self.height, self.c_tile_size))
        local_size = (self.c_tile_size, self.c_tile_size)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 11 arguments in exact contractual order."""
        local_mem_size = self.c_tile_size * (self.c_tile_size + self.local_mem_bank_padding) * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.in_ref),
            self._buffer_mgr.get_cl_buffer(self.out_ref),
            self.in_offset,
            self.out_offset,
            self.height,
            self.width,
            self.in_stride,
            self.out_stride,
            self.in_total_element_count,
            self.out_total_element_count,
        ]


# === Node 13: Gather & Permute Grad_H (Item Synchronization Point) ===


@dataclass(frozen=True)
class GatherAndPermuteGradHSignature(KernelSignature):
    """(Node 13) Signature for the `gather_and_permute_grad_hidden_activations` kernel."""

    clipped_partials_aos_ref: BufferHandle
    permuted_soa_out_ref: BufferHandle
    total_modules_count: np.uint32
    hidden_count: np.uint32
    total_batch_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_modules_count: np.uint32 = field(init=False)
    num_module_chunks_count: np.uint32 = field(init=False)
    modules_per_chunk_count: np.uint32 = field(init=False)
    num_class_chunks_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        aos_shape, _ = self._buffer_mgr.get_spec(self.clipped_partials_aos_ref)
        soa_shape, _ = self._buffer_mgr.get_spec(self.permuted_soa_out_ref)

        object.__setattr__(self, "total_tile_count", np.uint32(aos_shape[0]))
        object.__setattr__(self, "modules_per_chunk_count", np.uint32(aos_shape[1]))
        object.__setattr__(self, "total_batch_count", np.uint32(aos_shape[2]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(aos_shape[3]))
        object.__setattr__(self, "padded_total_modules_count", np.uint32(soa_shape[1]))

        if self.modules_per_chunk_count > 0:
            num_mod_chunks = (
                self.total_modules_count + self.modules_per_chunk_count - 1
            ) // self.modules_per_chunk_count
            object.__setattr__(self, "num_module_chunks_count", np.uint32(num_mod_chunks))
            if self.total_tile_count > 0 and num_mod_chunks > 0:
                object.__setattr__(self, "num_class_chunks_count", np.uint32(self.total_tile_count // num_mod_chunks))
            else:
                object.__setattr__(self, "num_class_chunks_count", np.uint32(0))
        else:
            object.__setattr__(self, "num_module_chunks_count", np.uint32(0))
            object.__setattr__(self, "num_class_chunks_count", np.uint32(0))

        # Derivation-as-verification: Enforce dimensional consistency
        assert self.total_batch_count * self.padded_hidden_count == soa_shape[0], "SoA output dimension 0 mismatch!"
        assert (
            self.total_tile_count == self.num_module_chunks_count * self.num_class_chunks_count
        ), "Tile count mismatch!"

    @property
    def kernel_name(self) -> str:
        return "gather_and_permute_grad_hidden_activations"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # Dispatch one work-item per element in the output buffer
        global_size = (self.total_batch_count * self.padded_hidden_count, self.padded_total_modules_count)
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
            self.num_module_chunks_count,
            self.modules_per_chunk_count,
            self.num_class_chunks_count,
            self.total_tile_count,
        ]
