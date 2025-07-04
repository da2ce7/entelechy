# compute_patterns.py

"""
A Module of Low-Level, Reusable Parallel Computing Components.

(REV 6 - REFACTORED) This module provides the definitive, low-level components
for implementing complex parallel patterns. The stateful, high-level
`ReductionTreeExecutor` has been removed, as its logic is now captured in a
stateless recipe in `graph_recipes.py`.

This module now contains only the tactical, reusable tools and configuration
objects consumed by those higher-level recipes.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from .launcher_infra import BufferHandle, BufferManager, KernelExecutor
from .kernel_signatures import (
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
)

# === Configuration Object for Reductions ===


@dataclass(frozen=True)
class ReductionPlan:
    """A simple configuration object defining the fan-in for the reduction tree."""

    k: int


# === Tactical Tool for Executing a Single Reduction Stage ===


class AggregationManager:
    """
    A stateless, tactical tool that executes a SINGLE stage of a reduction.

    This class is the 'dumb' executor in the reduction hierarchy. Its sole
    responsibility is to select the correct kernel tier (register vs. local)
    and dispatch it against a given set of partials, as defined by an

    indirection table (`offset_list`). It has no knowledge of the overall
    reduction tree and is used internally by the `execute_reduction_tree` recipe.
    """

    def __init__(self, ex: KernelExecutor, bm: BufferManager, arch_consts: Dict[str, int]):
        self.ex = ex
        self.bm = bm
        self.wgs0 = arch_consts.get("work_group_size_0", 256)
        self.max_reg_agg = arch_consts.get("max_register_aggregate_items", 16)

    def execute_stage(
        self,
        queue: cl.CommandQueue,
        collection_ref: BufferHandle,
        offset_list_ref: BufferHandle,
        num_partials_to_reduce: int,
        elements_per_partial: int,
        destination_ref: BufferHandle,
        wait_for: List[cl.Event],
    ) -> cl.Event:
        """Executes a single reduction stage, honoring the Indirection Contract."""
        if num_partials_to_reduce <= 1:
            raise ValueError("AggregationManager.execute_stage should only be called for N > 1 partials.")

        spec, dtype = self.bm.get_spec(destination_ref)
        scalar_byte_size = dtype().itemsize

        if num_partials_to_reduce <= self.max_reg_agg:
            sig = AggregateRegisterReduceSignature(
                partial_collection_ref=collection_ref,
                partial_offset_list_ref=offset_list_ref,
                dest_ref=destination_ref,
                partial_offset_list_count=np.uint32(num_partials_to_reduce),
                partial_width=np.uint32(elements_per_partial),
                operation_type=np.uint32(0),  # AGG_MODE_SUM
            )
        else:
            sig = AggregateLocalReduceSignature(
                work_group_size_0=self.wgs0,
                scalar_size_bytes=scalar_byte_size,
                partial_collection_ref=collection_ref,
                partial_offset_list_ref=offset_list_ref,
                dest_ref=destination_ref,
                partial_offset_list_count=np.uint32(num_partials_to_reduce),
                partial_width=np.uint32(elements_per_partial),
                operation_type=np.uint32(0),  # AGG_MODE_SUM
            )
        return self.ex.launch(queue, sig, wait_for=wait_for)
