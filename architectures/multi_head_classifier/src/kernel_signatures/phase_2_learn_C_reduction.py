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


# === Generic, Tiered Aggregation Engine Signatures (Nodes 14, 15, 20) ===


@dataclass(frozen=True)
class AggregateRegisterReduceSignature(KernelSignature):
    """(Node 15a, 20a) Signature for `aggregate_register_reduce` using an indirection table."""

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
    """(Node 15a, 20a) Signature for `aggregate_local_reduce` using an indirection table."""

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


@dataclass(frozen=True)
class ClipIntermediateGradSignature(KernelSignature):
    """
    (Node 15b, 20b) Signature for the `clip_intermediate_grad` utility kernel.

    This class provides the host-side contract for the kernel that performs the
    "clip" half of the atomic `sum-then-clip` pattern. It is a critical component
    of the Recursive Clip-Aggregation Engine, used by recipes to apply the
    gradient stabilization policy to intermediate results at each stage of a
    reduction tree.
    """

    # --- Injected Architectural Constants ---
    work_group_size_0: int
    scalar_size_bytes: int

    # --- Buffer Handle ---
    # The kernel operates in-place on this buffer.
    intermediate_grad_ref: BufferHandle

    # --- Control & Policy Scalars ---
    clipping_threshold_t_j: SCALAR_NP_TYPE  # The threshold for this stage 'j'
    epsilon: SCALAR_NP_TYPE

    # --- Derived Scalar Field ---
    parameter_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives the parameter count from the buffer specification."""
        super().__post_init__()
        shape, _ = self._buffer_mgr.get_spec(self.intermediate_grad_ref)
        object.__setattr__(self, "parameter_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "clip_intermediate_grad"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """
        Calculates a 1D grid. The kernel computes one L2 norm over the entire
        buffer, so all work-items are dispatched to cover all elements in parallel
        for the initial sum-of-squares reduction.
        """
        num_elements = self.parameter_count
        global_size = (_pad_to_multiple(int(num_elements), self.work_group_size_0),)
        local_size = (self.work_group_size_0,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 5 arguments in the exact order mandated by kernels.cl.h."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            # Arg 1: Local memory for the norm reduction
            cl.LocalMemory(local_mem_size),
            # Arg 2: The global buffer to be clipped in-place
            self._buffer_mgr.get_cl_buffer(self.intermediate_grad_ref),
            # Arg 3-4: Policy and stability scalars
            self.clipping_threshold_t_j,
            self.epsilon,
            # Arg 5: The total element count for bounds checking
            self.parameter_count,
        ]


# === Specialized Reduction Signature (Node 16) ===


@dataclass(frozen=True)
class StabilizeAndReduceGradHiddenActivationsSignature(KernelSignature):
    """
    (Node 16) Signature for the kernel.

    This version is a direct, 1:1, executable embodiment of its C-level kernel
    contract. It accepts all mandated parameters, removes all extraneous ones,
    and delegates all computational responsibility to the device as specified.
    """

    # --- Injected Architectural Constants ---
    work_group_size_0: int
    scalar_size_bytes: int

    # --- Buffer Handles (from kernel contract) ---
    permuted_soa_in_ref: BufferHandle
    final_grad_h_out_ref: BufferHandle

    # === Contractual Policy & Control Scalars ===
    # These must be provided by the Host Orchestrator.
    fp_max: SCALAR_NP_TYPE
    policy_t_algorithmic: SCALAR_NP_TYPE
    policy_lambda: SCALAR_NP_TYPE
    policy_max_k: np.uint32
    epsilon: SCALAR_NP_TYPE

    # --- Explicit Dimensional Parameters (from kernel contract) ---
    # These are now passed directly, removing brittle internal derivations.
    total_batch_count: np.uint32
    padded_hidden_count: np.uint32
    total_modules_count: np.uint32
    padded_total_modules_count: np.uint32

    def __post_init__(self):
        """
        Performs host-side assurance checks, validating that the provided
        parameters are consistent with the physical buffer allocations.
        """
        super().__post_init__()
        soa_shape, _ = self._buffer_mgr.get_spec(self.permuted_soa_in_ref)
        final_shape, _ = self._buffer_mgr.get_spec(self.final_grad_h_out_ref)

        # --- Contractual Verification (Host-Side Assurance) ---
        total_rows_expected_from_params = self.total_batch_count * self.padded_hidden_count
        assert soa_shape[0] == total_rows_expected_from_params, (
            f"Buffer spec mismatch for permuted_soa_in_ref: Expected {total_rows_expected_from_params} rows, "
            f"but buffer has {soa_shape[0]}."
        )
        assert (
            soa_shape[1] == self.padded_total_modules_count
        ), f"Buffer spec mismatch: Expected {self.padded_total_modules_count} padded modules, but buffer has {soa_shape[1]}."
        assert final_shape == (self.total_batch_count, self.padded_hidden_count), (
            f"Buffer spec mismatch for final_grad_h_out_ref: Expected shape {(self.total_batch_count, self.padded_hidden_count)}, "
            f"but buffer has {final_shape}."
        )

    @property
    def kernel_name(self) -> str:
        return "stabilize_and_reduce_grad_hidden_activations"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """
        Sets up a grid where each work-group reduces one row of the SoA matrix.
        The kernel internally divides its work among its work-items.
        """
        num_rows = self.total_batch_count * self.padded_hidden_count
        global_size = (num_rows * self.work_group_size_0,)
        local_size = (self.work_group_size_0,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 13 arguments in exact contractual order."""
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            # Arg 1: Local Memory
            cl.LocalMemory(local_mem_size),
            # Arg 2-3: Buffers
            self._buffer_mgr.get_cl_buffer(self.permuted_soa_in_ref),
            self._buffer_mgr.get_cl_buffer(self.final_grad_h_out_ref),
            # Arg 4-8: Policy & Control Scalars
            self.fp_max,
            self.policy_t_algorithmic,
            self.policy_lambda,
            self.policy_max_k,
            self.epsilon,
            # Arg 9-12: Dimensional Parameters
            self.total_batch_count,
            self.padded_hidden_count,
            self.total_modules_count,
            self.padded_total_modules_count,
        ]
