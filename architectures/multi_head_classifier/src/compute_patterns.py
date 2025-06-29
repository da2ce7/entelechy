# compute_patterns.py

"""
A Module of High-Level, Reusable Parallel Computing Patterns.

This module provides high-level, reusable classes that implement common, complex
parallel computing patterns like workload partitioning (tiling) and hierarchical
reduction (aggregation).

(REV 4): This version formalizes the `log_K(N)` reduction process into a
first-class architectural primitive: the `ReductionTreeExecutor`. This new
class encapsulates the entire stateful process of executing a reduction tree,
leaving the `AggregationManager` as a stateless, tactical tool that only
executes single reduction stages. This rectifies a previous design flaw and
restores architectural elegance by moving complex loop and state management out
of the high-level orchestrator.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from .launcher_infra import BufferHandle, BufferManager, KernelExecutor, PingPongManager, SCALAR_UINT_TYPE
from .kernel_signatures import (
    AggregateIdentitySignature,
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
)


# === Workload Partitioning (Tiling) Pattern ===


@dataclass(frozen=True)
class WorkTile:
    """Defines a single, independent unit of work for tiled kernels."""

    module_chunk_index: int
    class_chunk_index: int
    flat_tile_index: int
    num_class_chunks: int
    module_chunk_offset: int
    modules_per_chunk: int
    class_chunk_offset: int
    classes_per_chunk: int


@dataclass(frozen=True)
class ExecutionGrid:
    """A factory for producing WorkTile objects that partition a 2D problem space."""

    num_module_chunks: int
    num_class_chunks: int
    total_modules: int
    total_classes: int

    @property
    def total_tiles(self) -> int:
        return self.num_module_chunks * self.num_class_chunks

    def __iter__(self):
        for m_idx in range(self.num_module_chunks):
            for c_idx in range(self.num_class_chunks):
                yield self.get_tile(m_idx, c_idx)

    def get_tile(self, m_idx: int, c_idx: int) -> "WorkTile":
        mod_chunk_size = (self.total_modules + self.num_module_chunks - 1) // self.num_module_chunks
        cls_chunk_size = (self.total_classes + self.num_class_chunks - 1) // self.num_class_chunks
        mod_offset = m_idx * mod_chunk_size
        num_mods = min(mod_chunk_size, self.total_modules - mod_offset)
        cls_offset = c_idx * cls_chunk_size
        num_cls = min(cls_chunk_size, self.total_classes - cls_offset)
        flat_idx = m_idx * self.num_class_chunks + c_idx
        return WorkTile(m_idx, c_idx, flat_idx, self.num_class_chunks, mod_offset, num_mods, cls_offset, num_cls)


# === Hierarchical Aggregation (Reduction) Pattern ===


@dataclass(frozen=True)
class ReductionPlan:
    """A simple configuration object defining the fan-in for the reduction tree."""

    k: int


class AggregationManager:
    """
    A stateless, tactical tool that executes a SINGLE stage of a reduction.

    This class is the 'dumb' executor in the reduction hierarchy. Its sole
    responsibility is to select the correct kernel tier (register vs. local)
    and dispatch it against a given set of partials, as defined by an
    indirection table (`offset_list`). It has no knowledge of the overall
    reduction tree.
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
        """
        Executes a single reduction stage, honoring the Indirection Contract.
        """
        if num_partials_to_reduce <= 1:
            raise ValueError("AggregationManager.execute_stage should only be called for N > 1 partials.")

        spec, dtype = self.bm.get_spec(destination_ref)
        scalar_byte_size = dtype().itemsize

        if num_partials_to_reduce <= self.max_reg_agg:
            sig = AggregateRegisterReduceSignature(
                self.bm,
                partial_collection_ref=collection_ref,
                partial_offset_list_ref=offset_list_ref,
                dest_ref=destination_ref,
                partial_offset_list_count=np.uint32(num_partials_to_reduce),
                partial_width=np.uint32(elements_per_partial),
                operation_type=np.uint32(0),  # AGG_MODE_SUM
            )
        else:
            sig = AggregateLocalReduceSignature(
                self.bm,
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


class ReductionTreeExecutor:
    """
    A stateful process manager for a complete, multi-stage log_K(N) reduction.

    This class is the embodiment of the hierarchical aggregation primitive. It
    encapsulates the entire complex process of looping, managing transient
    ping-pong buffers, and generating the necessary indirection tables
    for each stage of the reduction. It uses an `AggregationManager`
    as its tactical tool to execute each individual stage.
    """

    def __init__(
        self,
        queue: cl.CommandQueue,
        ex: KernelExecutor,
        bm: BufferManager,
        agg_mgr: AggregationManager,
        plan: ReductionPlan,
    ):
        self.q = queue
        self.ex = ex
        self.bm = bm
        self.agg_mgr = agg_mgr
        self.plan = plan

    def _create_offset_list(
        self, num_offsets: int, element_stride: int, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        """Creates and uploads an indirection table (offset list) to the device."""
        offsets_host = np.arange(num_offsets, dtype=SCALAR_UINT_TYPE) * element_stride
        offset_list_ref = self.bm.acquire_transient_buffer(offsets_host.nbytes)
        evt = cl.enqueue_copy(self.q, self.bm.get_cl_buffer(offset_list_ref), offsets_host, wait_for=wait_for)
        return offset_list_ref, evt

    def execute(
        self,
        partial_collection_ref: BufferHandle,
        final_dest_handle: BufferHandle,
        num_initial_partials: int,
        wait_for: Optional[List[cl.Event]] = None,
    ) -> cl.Event:
        """Executes the full reduction tree from scattered partials to a final dense buffer."""
        wait_for = wait_for or []
        if num_initial_partials <= 0:
            return cl.UserEvent(self.q.context)  # Return a completed event if no work

        spec, dtype = self.bm.get_spec(final_dest_handle)
        elements_per_partial = int(np.prod(spec)) if spec else 1
        scalar_byte_size = dtype().itemsize

        if num_initial_partials == 1:
            # Base case: A simple identity copy is sufficient. Assumes the
            # single partial is at offset 0 of the collection buffer.
            sig = AggregateIdentitySignature(
                self.bm,
                in_ref=partial_collection_ref,
                out_ref=final_dest_handle,
                total_element_count=np.uint32(elements_per_partial),
            )
            return self.ex.launch(self.q, sig, wait_for=wait_for)

        # --- Setup the Process Resources ---
        ppm = PingPongManager()
        transient_handles = []
        try:
            ppm.initialize(self.bm, max_bytes=(num_initial_partials * elements_per_partial * scalar_byte_size))

            # --- Stage 1: The Initial Gather from SCATTERED Partials ---
            offset_list_ref, upload_evt = self._create_offset_list(num_initial_partials, elements_per_partial, wait_for)
            transient_handles.append(offset_list_ref)

            current_n = num_initial_partials
            # The first stage reads from the original, scattered collection.
            current_collection_ref = partial_collection_ref
            loop_deps = [upload_evt]

            # --- Main Reduction Loop ---
            while current_n > 1:
                stage_dest_ref, _ = ppm.get_io()
                stage_event = self.agg_mgr.execute_stage(
                    self.q,
                    current_collection_ref,
                    offset_list_ref,
                    current_n,
                    elements_per_partial,
                    stage_dest_ref,
                    loop_deps,
                )
                loop_deps = [stage_event]

                # Prepare for the NEXT Iteration
                next_n = (current_n + self.plan.k - 1) // self.plan.k
                if next_n <= 1:
                    break  # Last stage has produced the final transient result

                # The output of the last stage is the now-contiguous input for the next.
                current_collection_ref = stage_dest_ref
                current_n = next_n

                # Subsequent offset lists are trivial, as the partials are now dense.
                offset_list_ref, upload_evt = self._create_offset_list(current_n, elements_per_partial, loop_deps)
                transient_handles.append(offset_list_ref)
                loop_deps = [upload_evt]
                ppm.swap()

            # --- Finalization: Copy the final result to its persistent destination ---
            final_transient_handle, _ = ppm.get_io()
            final_event = cl.enqueue_copy_buffer(
                self.q,
                src=self.bm.get_cl_buffer(final_transient_handle),
                dst=self.bm.get_cl_buffer(final_dest_handle),
                byte_count=(elements_per_partial * scalar_byte_size),
                wait_for=loop_deps,
            )
            return final_event

        finally:
            # Safely release all transient resources used during the process.
            ppm.release()
            for handle in transient_handles:
                self.bm.release_transient_buffer(handle)
