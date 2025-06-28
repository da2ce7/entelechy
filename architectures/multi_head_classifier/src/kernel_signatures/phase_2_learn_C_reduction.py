# kernel_signatures/phase_2_learn_C_reduction.py

"""
Concrete KernelSignature Implementations for Reduction & Aggregation (Nodes 14-16, 19).

This file contains the final, canonical implementations for the kernel launch
signatures related to the third stage of the 'Learn' phase. These kernels are
responsible for collapsing many partial gradient buffers into single,
summed/averaged results.

This implementation follows a rectified design where each kernel in the tiered
aggregation engine has its own simple, 'dumb' signature class. This removes
strategic logic from the signature layer and places it correctly in the
orchestration layer (within a dedicated `AggregationManager`).

This module provides:
- AggregateIdentitySignature (Node 14): The N=1 base case.
- AggregateRegisterReduceSignature (Node 15): The small-N, register-based case.
- AggregateLocalReduceSignature (Node 19): The large-N, local memory-based case.
- ReduceGradHOverModulesSignature (Node 16): A highly specialized signature for the
  final reduction of the unique `Grad_H` SoA buffer.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..launcher_infra import BufferHandle, KernelSignature
from ..memory_layout import _pad_to_multiple


# === Generic, Tiered Aggregation Engine Signatures (Nodes 14, 15, 19) ===


@dataclass(frozen=True)
class AggregateIdentitySignature(KernelSignature):
    """(Node 14) Signature for `aggregate_identity`, the N=1 reduction base case."""

    in_ref: BufferHandle
    out_ref: BufferHandle
    total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        shape, _ = self._buffer_mgr.get_spec(self.out_ref)
        object.__setattr__(self, "total_element_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "aggregate_identity"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        return (int(self.total_element_count),), None

    def get_args(self) -> List:
        """Returns all 3 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.in_ref),
            self._buffer_mgr.get_cl_buffer(self.out_ref),
            self.total_element_count,
        ]


@dataclass(frozen=True)
class AggregateRegisterReduceSignature(KernelSignature):
    """(Node 15) Signature for `aggregate_register_reduce`, for small N."""

    partials_collection_ref: BufferHandle
    out_ref: BufferHandle
    in_partials_count: np.uint32
    operation_type: np.uint32
    partial_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        shape, _ = self._buffer_mgr.get_spec(self.out_ref)
        object.__setattr__(self, "partial_element_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "aggregate_register_reduce"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        return (int(self.partial_element_count),), None

    def get_args(self) -> List:
        """Returns all 5 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.partials_collection_ref),
            self._buffer_mgr.get_cl_buffer(self.out_ref),
            self.in_partials_count,
            self.partial_element_count,
            self.operation_type,
        ]


@dataclass(frozen=True)
class AggregateLocalReduceSignature(KernelSignature):
    """(Node 19) Signature for `aggregate_local_reduce`, for large N."""

    work_group_size_0: int
    scalar_size_bytes: int
    partials_collection_ref: BufferHandle
    out_ref: BufferHandle
    in_partials_count: np.uint32
    operation_type: np.uint32
    partial_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        shape, _ = self._buffer_mgr.get_spec(self.out_ref)
        object.__setattr__(self, "partial_element_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "aggregate_local_reduce"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (_pad_to_multiple(int(self.partial_element_count), self.work_group_size_0),)
        local_size = (self.work_group_size_0,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 6 arguments, prepending the local memory buffer."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.partials_collection_ref),
            self._buffer_mgr.get_cl_buffer(self.out_ref),
            self.in_partials_count,
            self.partial_element_count,
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
