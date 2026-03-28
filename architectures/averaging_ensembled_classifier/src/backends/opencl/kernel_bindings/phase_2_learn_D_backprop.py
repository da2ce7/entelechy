# kernel_signatures/phase_2_learn_D_backprop.py

"""
The Definitive, Executable Contracts for Shared Layer Backpropagation (Nodes 17-19).

Jurisdictional Mandate:
This file is the canonical Python-side embodiment of the C-level kernel
contracts for the fourth stage of the 'Learn' phase. Its jurisdiction is
to define the immutable 'artisan' classes responsible for backpropagating the
unified error signal (`summed_grad_h`) through the shared network layer,
thereby computing the partial gradients for the shared weights and biases.

Architectural Role:
These signatures are designed to support the system's "True Streaming" model.
They operate on independent chunks of the input batch, producing partial
gradients which are then stabilized by the `ClipSharedGradientsChunkSignature`
before being consumed by the reduction engine. This module provides the host
with the precise, verifiable contracts needed to orchestrate this complex,
memory-efficient dataflow.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Primitives & Core Infrastructure ---
from ..launcher_infra import BufferHandle, KernelSignature, BufferManager
from ....shared.memory_layout import pad_to_multiple
from ..context import DiscoveredArchConstants


@dataclass
class SharedGradientHandles:
    """
    A helper dataclass to simplify the API for the `clip_shared_gradients_chunk`
    kernel. It cleanly separates the transient, single-chunk 'source' buffers from
    the persistent 'destination' collection buffers.
    """
    # Source: The raw gradient for a single chunk, held in transient scratch space.
    grad_weights_shared_chunk: BufferHandle
    grad_biases_shared_chunk: BufferHandle
    # Destination: The monolithic collection buffers for all clipped partials.
    clipped_grad_weights_shared_collection: BufferHandle
    clipped_grad_biases_shared_collection: BufferHandle


@dataclass(frozen=True)
class BackpropSharedWeightsChunkSignature(KernelSignature):
    """(Node 17) Signature for the `backprop_shared_weights_chunk` kernel."""

    # --- Injected System Context (The Architectural Mandate) ---
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    # --- Kernel-Specific Buffers ---
    input_ref: BufferHandle
    h_ref: BufferHandle
    grad_h_ref: BufferHandle
    mask_ref: BufferHandle
    partial_gsw_out_ref: BufferHandle

    # --- Kernel-Specific Control Scalars ---
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    batch_chunk_index: np.uint32
    num_batch_chunks_count: np.uint32

    # --- Derived Fields ---
    total_batch_count: np.uint32 = field(init=False)
    padded_input_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    final_grad_hidden_total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        input_shape, _ = self._buffer_mgr.get_spec(self.input_ref)
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        grad_h_shape, _ = self._buffer_mgr.get_spec(self.grad_h_ref)

        object.__setattr__(self, "total_batch_count", np.uint32(input_shape[0]))
        object.__setattr__(self, "padded_input_count", np.uint32(input_shape[1]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))
        object.__setattr__(
            self, "final_grad_hidden_total_element_count", np.uint32(np.prod(grad_h_shape))
        )

    @property
    def kernel_name(self) -> str:
        return "backprop_shared_weights_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Calculates a 2D grid, deriving its tiling dimensions from the arch constants."""
        work_group_size_1 = self._arch_consts.optimal_rectangular_tile_dim1
        global_size = (
            int(self.padded_input_count),
            pad_to_multiple(int(self.padded_hidden_count), work_group_size_1),
        )
        local_size = (1, work_group_size_1)
        return global_size, local_size

    def get_args(self) -> List:
        """Assembles all 14 arguments in the exact order mandated by the C contract."""
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        work_group_size_1 = self._arch_consts.optimal_rectangular_tile_dim1
        local_mem_size = work_group_size_1 * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.input_ref),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.grad_h_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_gsw_out_ref),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            self.batch_chunk_index,
            self.total_batch_count,
            self.num_batch_chunks_count,
            self.padded_input_count,
            self.padded_hidden_count,
            self.final_grad_hidden_total_element_count,
        ]


@dataclass(frozen=True)
class BackpropSharedBiasesChunkSignature(KernelSignature):
    """(Node 18) Signature for the `backprop_shared_biases_chunk` kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    h_ref: BufferHandle
    grad_h_ref: BufferHandle
    mask_ref: BufferHandle
    partial_gsb_out_ref: BufferHandle

    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    batch_chunk_index: np.uint32
    num_batch_chunks_count: np.uint32

    total_batch_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    final_grad_hidden_total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        grad_h_shape, _ = self._buffer_mgr.get_spec(self.grad_h_ref)

        object.__setattr__(self, "total_batch_count", np.uint32(h_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))
        object.__setattr__(
            self, "final_grad_hidden_total_element_count", np.uint32(np.prod(grad_h_shape))
        )

    @property
    def kernel_name(self) -> str:
        return "backprop_shared_biases_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Calculates a 1D grid for reduction over the hidden dimension."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        num_work_groups = self.padded_hidden_count
        global_size = (int(num_work_groups) * work_group_size,)
        local_size = (work_group_size,)
        return global_size, local_size

    def get_args(self) -> List:
        """Assembles all 12 arguments in the exact order mandated by the C contract."""
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        local_mem_size = work_group_size * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.grad_h_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_gsb_out_ref),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            self.batch_chunk_index,
            self.total_batch_count,
            self.num_batch_chunks_count,
            self.padded_hidden_count,
            self.final_grad_hidden_total_element_count,
        ]


@dataclass(frozen=True)
class ClipSharedGradientsChunkSignature(KernelSignature):
    """
    (Node 19) Signature for the `clip_shared_gradients_chunk` utility kernel.

    This serves as the mandatory stability primitive for the streaming backprop
    data path. It atomically clips the concatenated `Grad_SW` and `Grad_SB`
    vectors for a single chunk before they are written to the main collection
    buffer for subsequent reduction.
    """
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    handles: SharedGradientHandles

    clipping_threshold_global: np.float32
    epsilon: np.float32
    dest_weights_write_offset_elements: np.uint32
    dest_biases_write_offset_elements: np.uint32
    num_batch_chunks: np.uint32

    weights_param_count: np.uint32 = field(init=False)
    biases_param_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        w_shape, _ = self._buffer_mgr.get_spec(self.handles.grad_weights_shared_chunk)
        b_shape, _ = self._buffer_mgr.get_spec(self.handles.grad_biases_shared_chunk)
        object.__setattr__(self, "weights_param_count", np.uint32(np.prod(w_shape)))
        object.__setattr__(self, "biases_param_count", np.uint32(np.prod(b_shape)))

    @property
    def kernel_name(self) -> str:
        return "clip_shared_gradients_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Calculates grid to cover all elements in the concatenated gradient vector."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        total_elements = self.weights_param_count + self.biases_param_count
        global_size = (pad_to_multiple(int(total_elements), work_group_size),)
        local_size = (work_group_size,)
        return global_size, local_size

    def get_args(self) -> List:
        """Assembles all 12 arguments in exact C-level contractual order."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = work_group_size * scalar_size_bytes
        h = self.handles
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(h.grad_weights_shared_chunk),
            self._buffer_mgr.get_cl_buffer(h.grad_biases_shared_chunk),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_weights_shared_collection),
            self._buffer_mgr.get_cl_buffer(h.clipped_grad_biases_shared_collection),
            self.clipping_threshold_global,
            self.epsilon,
            self.weights_param_count,
            self.biases_param_count,
            self.dest_weights_write_offset_elements,
            self.dest_biases_write_offset_elements,
            self.num_batch_chunks,
        ]
