# kernel_signatures/phase_2_learn_D_backprop.py

"""
Concrete KernelSignature Implementations for Shared Layer Backpropagation (Nodes 17-18).

This file contains the final, canonical implementations for the kernel launch
signatures related to the fourth stage of the 'Learn' phase. These kernels
are responsible for backpropagating the unified error signal (`summed_grad_h`)
through the shared network layer to compute gradients for the shared weights
and biases.

These kernels operate on chunks of the input batch and therefore produce
partial gradients, which must be aggregated by the orchestrator in a subsequent
step using the tools from `phase_2_learn_C_reduction.py`.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..launcher_infra import BufferHandle, KernelSignature
from ..memory_layout import _pad_to_multiple


@dataclass(frozen=True)
class BackpropSharedWeightsChunkSignature(KernelSignature):
    """
    (Node 17) Signature for `backprop_shared_weights_chunk` kernel.

    (REV 2 - Rectified) This version corrects the previous implementation, which
    was missing two arguments and had incorrect argument ordering. This signature
    is now in full compliance with the kernel's 14-argument contract defined
    in `kernels.cl.h`.
    """

    # --- Injected Architectural Constants ---
    work_group_size_1: int
    scalar_size_bytes: int

    # --- Buffer Handles ---
    input_ref: BufferHandle
    h_ref: BufferHandle  # Post-activation hidden state from Node 4
    grad_h_ref: BufferHandle  # Summed upstream gradient for hidden layer from Node 16
    mask_ref: BufferHandle
    partial_gsw_out_ref: BufferHandle  # Partial gradient for shared weights

    # --- Control & Dimensional Scalars ---
    # These must be provided by the host orchestrator as part of the streaming plan.
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    batch_chunk_index: np.uint32  # CRITICAL FIX: Added missing index for Placement Contract
    num_batch_chunks_count: np.uint32

    # --- Derived Scalar Fields ---
    total_batch_count: np.uint32 = field(init=False)
    padded_input_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    # CRITICAL FIX: Add derived field for kernel-side validation
    final_grad_hidden_total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives dimensions from input and output buffer specifications."""
        super().__post_init__()
        input_shape, _ = self._buffer_mgr.get_spec(self.input_ref)
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        grad_h_shape, _ = self._buffer_mgr.get_spec(self.grad_h_ref)

        object.__setattr__(self, "total_batch_count", np.uint32(input_shape[0]))
        object.__setattr__(self, "padded_input_count", np.uint32(input_shape[1]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))
        object.__setattr__(self, "final_grad_hidden_total_element_count", np.uint32(np.prod(grad_h_shape)))

    @property
    def kernel_name(self) -> str:
        return "backprop_shared_weights_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Calculates 2D grid: (input_dim, hidden_dim)."""
        global_size = (self.padded_input_count, _pad_to_multiple(int(self.padded_hidden_count), self.work_group_size_1))
        local_size = (1, self.work_group_size_1)
        return global_size, local_size

    def get_args(self) -> List:
        """
        Returns all 14 arguments in the exact order mandated by `kernels.cl.h`.
        """
        local_mem_size = self.work_group_size_1 * self.scalar_size_bytes
        return [
            # Arg 1: Local Memory
            cl.LocalMemory(local_mem_size),
            # Arg 2-6: Buffers
            self._buffer_mgr.get_cl_buffer(self.input_ref),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.grad_h_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_gsw_out_ref),
            # Arg 7-14: Scalars in strict contractual order
            self.batch_chunk_offset,
            self.batch_chunk_count,
            self.batch_chunk_index,  # Corrected
            self.total_batch_count,  # Corrected
            self.num_batch_chunks_count,  # Corrected
            self.padded_input_count,  # Corrected
            self.padded_hidden_count,  # Corrected
            self.final_grad_hidden_total_element_count,  # Corrected
        ]


@dataclass(frozen=True)
class BackpropSharedBiasesChunkSignature(KernelSignature):
    """(Node 18) Signature for `backprop_shared_biases_chunk` kernel."""

    # --- Injected Architectural Constants ---
    work_group_size_0: int
    scalar_size_bytes: int

    # --- Buffer Handles ---
    h_ref: BufferHandle
    grad_h_ref: BufferHandle
    mask_ref: BufferHandle
    partial_gsb_out_ref: BufferHandle  # Partial gradient for shared biases

    # --- Control & Dimensional Scalars ---
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    num_batch_chunks: np.uint32

    # --- Derived Scalar Fields ---
    total_batch_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives dimensions from the hidden activation buffer spec."""
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        object.__setattr__(self, "total_batch_count", np.uint32(h_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))

    @property
    def kernel_name(self) -> str:
        return "backprop_shared_biases_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Calculates 1D grid for reduction over hidden dimension.

        The execution model is one work-group per output element of the bias
        gradient vector. The total number of work-groups is therefore equal
        to the number of hidden dimensions.
        """

        num_work_groups = self.padded_hidden_count
        global_size = (int(num_work_groups) * self.work_group_size_0,)
        local_size = (self.work_group_size_0,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 10 arguments in exact contractual order."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.grad_h_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_gsb_out_ref),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            self.num_batch_chunks,
            self.total_batch_count,
            self.padded_hidden_count,
        ]
