# kernel_signatures/phase_2_learn_C_reduction.py

"""
The Definitive, Executable Contracts for Gradient Reduction & Aggregation.

Jurisdictional Mandate:
This file is the canonical Python-side embodiment of the C-level kernel
contracts for the third stage of the 'Learn' phase: Reduction. Its jurisdiction
covers the full spectrum of aggregation, from the generic, host-driven "sum-then-clip"
engine to the specialized, self-contained, policy-aware reduction of the `Grad_H`
vector.

Architectural Role:
This module provides the 'artisan' classes that physicalize the system's
"Primacy of Memory Strategy." The tiered aggregation signatures (`Aggregate*`)
are the core building blocks of the `log_K(N)` reduction tree, operating via an
indirection list to avoid costly intermediate memory copies. They work in tandem
with the `ClipIntermediateGradSignature`, which applies the host's dynamic
stabilization policy at each stage, transforming a simple sum into a robust,
numerically-stable learning primitive.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Primitives & Core Infrastructure ---
from ..launcher_infra import BufferHandle, KernelSignature, BufferManager
from ....shared.memory_layout import _pad_to_multiple
from ..context import DiscoveredArchConstants


# =========================================================================
# === Generic, Tiered Aggregation Engine Signatures (Nodes 14, 15a, 20a)
# =========================================================================


@dataclass(frozen=True)
class AggregateRegisterReduceSignature(KernelSignature):
    """(Tier 1) Signature for `aggregate_register_reduce` using an indirection table."""

    # --- Injected System Context (The Architectural Mandate) ---
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    # --- Kernel-Specific Buffers & Control Scalars (The Indirection Contract) ---
    partial_collection_ref: BufferHandle  # The memory pool of scattered partials
    partial_offset_list_ref: BufferHandle  # The indirection table for the gather
    dest_ref: BufferHandle
    partial_offset_list_count: np.uint32
    partial_width: np.uint32
    operation_type: np.uint32

    def __post_init__(self):
        super().__post_init__()

    @property
    def kernel_name(self) -> str:
        return "aggregate_register_reduce"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # This kernel is optimized for small `N`. It dispatches one work-item per
        # element of the final output, using registers for the summation.
        return (int(self.partial_width),), None

    def get_args(self) -> List:
        """Assembles all 6 arguments in exact contractual order."""
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
    """(Tier 2) Signature for `aggregate_local_reduce` using an indirection table."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    partial_collection_ref: BufferHandle
    partial_offset_list_ref: BufferHandle
    dest_ref: BufferHandle
    partial_offset_list_count: np.uint32
    partial_width: np.uint32
    operation_type: np.uint32

    def __post_init__(self):
        super().__post_init__()

    @property
    def kernel_name(self) -> str:
        return "aggregate_local_reduce"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # This kernel is for larger `N`. It dispatches a 1D grid of work-groups,
        # where each work-group is responsible for reducing one slice of the
        # output vector across all input partials, using local memory for scalability.
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        global_size = (_pad_to_multiple(int(self.partial_width), work_group_size),)
        local_size = (work_group_size,)
        return global_size, local_size

    def get_args(self) -> List:
        """Assembles arguments, including the required local memory allocation."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = work_group_size * scalar_size_bytes
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

    This signature represents the "clip" half of the atomic "sum-then-clip"
    pattern. It is the physical mechanism by which the host's `StabilizationPolicy`
    is enforced upon the intermediate results of the Recursive Clip-Aggregation Engine.
    """

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    intermediate_grad_ref: BufferHandle  # This buffer is modified in-place.

    # The threshold for this specific stage 'j' of the reduction tree.
    # This value is the final, authoritative output of the StabilizationPolicy.
    clipping_threshold_t_j: np.float32
    epsilon: np.float32

    parameter_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        shape, _ = self._buffer_mgr.get_spec(self.intermediate_grad_ref)
        object.__setattr__(self, "parameter_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "clip_intermediate_grad"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Dispatches enough work-items to compute a single L2 norm over the entire buffer."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        global_size = (_pad_to_multiple(int(self.parameter_count), work_group_size),)
        local_size = (work_group_size,)
        return global_size, local_size

    def get_args(self) -> List:
        """Assembles arguments in the exact order mandated by the kernel contract."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = work_group_size * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.intermediate_grad_ref),
            self.clipping_threshold_t_j,
            self.epsilon,
            self.parameter_count,
        ]


# =========================================================================
# === Specialized, Policy-Aware Reduction Signature (Node 16)
# =========================================================================


@dataclass(frozen=True)
class StabilizeAndReduceGradHiddenActivationsSignature(KernelSignature):
    """
    (Node 16) Signature for the specialized `stabilize_and_reduce_grad_hidden_activations` kernel.

    This is not a generic reduction. It is a specialized, self-contained engine
    for the critical upstream gradient, `Grad_H`. It operates on the pre-gathered,
    contiguous SoA buffer from Node 13 and has the system's stabilization policy
    baked directly into its C-level contract, ensuring maximum signal fidelity.
    """

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    permuted_soa_in_ref: BufferHandle
    final_grad_h_out_ref: BufferHandle

    # --- Contractual Policy & Control Scalars (The Host's Final Command) ---
    fp_max: np.float32
    policy_t_algorithmic: np.float32
    policy_lambda: np.float32
    policy_max_k: np.uint32
    epsilon: np.float32

    # --- Explicit Dimensional Parameters ---
    total_batch_count: np.uint32
    padded_hidden_count: np.uint32
    total_modules_count: np.uint32
    padded_total_modules_count: np.uint32

    def __post_init__(self):
        """Performs host-side assurance, validating the plan against physical memory."""
        super().__post_init__()
        soa_shape, _ = self._buffer_mgr.get_spec(self.permuted_soa_in_ref)
        final_shape, _ = self._buffer_mgr.get_spec(self.final_grad_h_out_ref)

        total_rows_expected = self.total_batch_count * self.padded_hidden_count
        assert soa_shape[0] == total_rows_expected, (
            f"Buffer spec mismatch for permuted_soa_in_ref: Expected {total_rows_expected} rows, "
            f"but buffer has {soa_shape[0]}."
        )
        assert soa_shape[1] == self.padded_total_modules_count, (
            f"Buffer spec mismatch: Expected {self.padded_total_modules_count} padded modules, "
            f"but buffer has {soa_shape[1]}."
        )
        assert final_shape == (self.total_batch_count, self.padded_hidden_count), (
            f"Buffer spec mismatch for final_grad_h_out_ref: Expected shape "
            f"{(self.total_batch_count, self.padded_hidden_count)}, but has {final_shape}."
        )

    @property
    def kernel_name(self) -> str:
        return "stabilize_and_reduce_grad_hidden_activations"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Dispatches one work-group per row of the SoA matrix for reduction."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        num_rows = int(self.total_batch_count) * int(self.padded_hidden_count)
        global_size = (num_rows * work_group_size,)
        local_size = (work_group_size,)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 13 arguments, translating the Host's policy into device primitives."""
        work_group_size = self._arch_consts.optimal_workgroup_size_1d_reduction
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = work_group_size * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.permuted_soa_in_ref),
            self._buffer_mgr.get_cl_buffer(self.final_grad_h_out_ref),
            self.fp_max,
            self.policy_t_algorithmic,
            self.policy_lambda,
            self.policy_max_k,
            self.epsilon,
            self.total_batch_count,
            self.padded_hidden_count,
            self.total_modules_count,
            self.padded_total_modules_count,
        ]
