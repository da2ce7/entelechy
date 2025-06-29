# kernel_signatures/phase_2_learn_C_reduction.py

"""
Concrete KernelSignature Implementations for Reduction & Aggregation (Nodes 14-16, 19).

This file contains the final, canonical implementations for the kernel launch
signatures related to the third stage of the 'Learn' phase.

(REV 3): The generic aggregation signatures have been completely refactored to
support the superior "Device-Side Gather" model. They now accept an indirection
table (offset list) on the device, allowing the kernel to perform the gather
operation implicitly. This avoids a massive, host-orchestrated memory copy and
is a key performance optimization.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..launcher_infra import BufferHandle, KernelSignature
from ..memory_layout import _pad_to_multiple, SCALAR_NP_TYPE


# === Generic, Tiered Aggregation Engine Signatures (Nodes 14, 15, 19) ===


@dataclass(frozen=True)
class AggregateRegisterReduceSignature(KernelSignature):
    """(Node 15) Signature for `aggregate_register_reduce` using an indirection table."""

    partial_collection_ref: BufferHandle
    partial_offset_list_ref: BufferHandle  # The indirection table
    dest_ref: BufferHandle
    partial_offset_list_count: np.uint32  # Number of partials to reduce
    partial_width: np.uint32  # Number of elements per partial
    operation_type: np.uint32  # 0=SUM, 1=AVERAGE

    def __post_init__(self):
        super().__post_init__()

    @property
    def kernel_name(self) -> str:
        return "aggregate_register_reduce"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # One work-item per element of the final reduced partial
        return (int(self.partial_width),), None

    def get_args(self) -> List:
        """Returns all 6 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.partial_collection_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_offset_list_ref),
            self._buffer_mgr.get_cl_buffer(self.dest_ref),
            self.partial_offset_list_count,
            self.partial_width,
            self.operation_type,
        ]


@dataclass(frozen=True)
class AggregateLocalReduceSignature(KernelSignature):
    """(Node 19) Signature for `aggregate_local_reduce` using an indirection table."""

    work_group_size_0: int
    scalar_size_bytes: int
    partial_collection_ref: BufferHandle
    partial_offset_list_ref: BufferHandle  # The indirection table
    dest_ref: BufferHandle
    partial_offset_list_count: np.uint32  # Number of partials to reduce
    partial_width: np.uint32  # Number of elements per partial
    operation_type: np.uint32  # 0=SUM, 1=AVERAGE

    def __post_init__(self):
        super().__post_init__()

    @property
    def kernel_name(self) -> str:
        return "aggregate_local_reduce"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (_pad_to_multiple(int(self.partial_width), self.work_group_size_0),)
        local_size = (self.work_group_size_0,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 7 arguments, prepending local memory."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.partial_collection_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_offset_list_ref),
            self._buffer_mgr.get_cl_buffer(self.dest_ref),
            self.partial_offset_list_count,
            self.partial_width,
            self.operation_type,
        ]


# === Specialized Reduction Signature (Node 16) ===


@dataclass(frozen=True)
class ReduceGradHOverModulesSignature(KernelSignature):
    """(Node 16) Signature for the specialized `reduce_grad_h_over_modules` kernel."""

    # --- Injected Architectural Constants ---
    work_group_size_0: int
    scalar_size_bytes: int

    # --- Buffer Handles ---
    permuted_soa_in_ref: BufferHandle
    final_grad_h_out_ref: BufferHandle

    # --- Control & Dimensional Scalars ---
    total_modules_count: np.uint32  # unpadded

    # --- Derived Scalar Fields ---
    total_batch_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_modules_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives dimensions from the SoA input and final AoS output buffers."""
        soa_shape, _ = self._buffer_mgr.get_spec(self.permuted_soa_in_ref)
        final_shape, _ = self._buffer_mgr.get_spec(self.final_grad_h_out_ref)

        object.__setattr__(self, "total_batch_count", np.uint32(final_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(final_shape[1]))
        object.__setattr__(self, "padded_total_modules_count", np.uint32(soa_shape[1]))

        # Derivation-as-verification: Enforces the contract between Node 13 and Node 16.
        total_rows = self.total_batch_count * self.padded_hidden_count
        assert soa_shape[0] == total_rows, f"SoA input row count mismatch! Expected {total_rows}, got {soa_shape[0]}."

    @property
    def kernel_name(self) -> str:
        return "reduce_grad_h_over_modules"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Sets up a grid where each work-group reduces one row of the SoA matrix."""
        num_rows = self.total_batch_count * self.padded_hidden_count
        global_size = (int(num_rows) * self.work_group_size_0,)
        local_size = (self.work_group_size_0,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 7 arguments in exact contractual order."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.permuted_soa_in_ref),
            self._buffer_mgr.get_cl_buffer(self.final_grad_h_out_ref),
            self.total_batch_count,
            self.padded_hidden_count,
            self.total_modules_count,
            self.padded_total_modules_count,
        ]
