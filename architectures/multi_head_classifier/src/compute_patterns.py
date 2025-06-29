# compute_patterns.py

"""
A Module of High-Level, Reusable Parallel Computing Patterns.

(REV 5) This module provides the definitive, architecturally pure implementations
for common parallel computing patterns. This version formalizes the concept of
"gathering" partial results into a first-class declarative abstraction: the
`GatherPrimitive`.

This core change resolves all prior architectural discrepancies:
- The `ReductionTreeExecutor` is now fully decoupled from the host's high-level
  workload partitioning logic, consuming the `GatherPrimitive` as its sole source
  of truth for memory layout.
- The `N=1` reduction base case is now handled with a direct, efficient driver
  call, upholding the "Primacy of Memory Strategy".
- All stateful loop management, buffer sizing, and indirection list generation
  is now flawlessly encapsulated within the `ReductionTreeExecutor`, presenting
  a clean and declarative interface to the high-level orchestrator.
"""

import abc
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from .launcher_infra import BufferHandle, BufferManager, KernelExecutor, PingPongManager, SCALAR_UINT_TYPE
from .kernel_signatures import (
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
)


# === Section 1: Workload Partitioning (Tiling) Pattern ===


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


# === Section 2: The Gather Abstraction ===


class GatherPrimitive(abc.ABC):
    """
    An abstract contract representing a collection of partials to be gathered.

    This is a declarative primitive that tells the ReductionTreeExecutor *how* a
    set of partials is laid out, allowing the executor to derive the necessary
    indirection list without the orchestrator needing to manage low-level details.
    """

    @property
    @abc.abstractmethod
    def num_partials(self) -> int:
        """The total number of partials in the collection."""
        pass

    @property
    @abc.abstractmethod
    def elements_per_partial(self) -> int:
        """The number of scalar elements in a single partial."""
        pass

    @abc.abstractmethod
    def get_offsets(self) -> np.ndarray:
        """
        Returns a host-side numpy array of uints, where each element is the
        starting OFFSET (in elements, not bytes) of a partial relative to the
        start of its collection buffer.
        """
        pass


@dataclass(frozen=True)
class LinearlyChunkedGather(GatherPrimitive):
    """
    Represents partials scattered into a collection buffer according to a
    linear chunking of a primary dimension (e.g., batch).
    """

    num_chunks: int
    elements_per_chunk: int

    @property
    def num_partials(self) -> int:
        return self.num_chunks

    @property
    def elements_per_partial(self) -> int:
        return self.elements_per_chunk

    def get_offsets(self) -> np.ndarray:
        """The offset is the chunk index multiplied by the element stride."""
        return np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) * self.elements_per_partial


@dataclass(frozen=True)
class TiledGather(GatherPrimitive):
    """
    Represents partials scattered across a collection buffer according to a
    tiled placement strategy (e.g., `grid_mod_cls`).
    """

    grid: ExecutionGrid
    _elements_per_partial: int

    @property
    def num_partials(self) -> int:
        return self.grid.total_tiles

    @property
    def elements_per_partial(self) -> int:
        return self._elements_per_partial

    def get_offsets(self) -> np.ndarray:
        """The offset is simply the tile index multiplied by the element stride."""
        return np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) * self.elements_per_partial


@dataclass(frozen=True)
class ContiguousGather(GatherPrimitive):
    """

    Represents partials that are laid out contiguously in memory, such as the
    output of a previous reduction stage in a ping-pong buffer.
    """

    _num_partials: int
    _elements_per_partial: int

    @property
    def num_partials(self) -> int:
        return self._num_partials

    @property
    def elements_per_partial(self) -> int:
        return self._elements_per_partial

    def get_offsets(self) -> np.ndarray:
        """The offsets are a simple linear progression from the start of the buffer."""
        return np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) * self.elements_per_partial


# === Section 3: Hierarchical Aggregation (Reduction) Pattern ===


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
        """Executes a single reduction stage, honoring the Indirection Contract."""
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
    for each stage of the reduction, which it consumes via the declarative
    `GatherPrimitive` abstraction.
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

    def _create_offset_list(self, offsets_host: np.ndarray, wait_for: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        """Creates and uploads an indirection table (offset list) from a host array."""
        offset_list_ref = self.bm.acquire_transient_buffer(offsets_host.nbytes)
        evt = cl.enqueue_copy(self.q, self.bm.get_cl_buffer(offset_list_ref), offsets_host, wait_for=wait_for)
        return offset_list_ref, evt

    def execute(
        self,
        gather_primitive: GatherPrimitive,
        partial_collection_ref: BufferHandle,
        final_dest_handle: BufferHandle,
        wait_for: Optional[List[cl.Event]] = None,
    ) -> cl.Event:
        """Executes the full reduction tree from a collection of partials to a final dense buffer."""
        wait_for = wait_for or []
        n = gather_primitive.num_partials

        if n <= 0:
            user_event = cl.UserEvent(self.q.context)
            user_event.set_status(cl.command_execution_status.COMPLETE)
            return user_event

        elements_per_partial = gather_primitive.elements_per_partial
        scalar_byte_size = self.bm.get_spec(final_dest_handle)[1]().itemsize
        partial_byte_size = elements_per_partial * scalar_byte_size

        initial_offsets_host = gather_primitive.get_offsets()

        if n == 1:
            # Optimal N=1 case: A direct driver copy with the correct offset.
            src_offset_bytes = int(initial_offsets_host[0] * scalar_byte_size)
            return cl.enqueue_copy_buffer(
                self.q,
                src=self.bm.get_cl_buffer(partial_collection_ref),
                dst=self.bm.get_cl_buffer(final_dest_handle),
                byte_count=partial_byte_size,
                src_offset=src_offset_bytes,
                dst_offset=0,
                wait_for=wait_for,
            )

        # --- Setup Resources for N > 1 Reduction Tree ---
        ppm = PingPongManager()
        stage_output_partials = (n + self.plan.k - 1) // self.plan.k
        ppm_buffer_size_bytes = stage_output_partials * partial_byte_size
        ppm.initialize(self.bm, max_bytes=ppm_buffer_size_bytes)
        transient_handles = []
        try:
            # Stage 1: The Initial Gather from the source collection.
            offset_list_ref, upload_evt = self._create_offset_list(initial_offsets_host, wait_for)
            transient_handles.append(offset_list_ref)

            current_n = n
            current_collection_ref = partial_collection_ref
            current_deps = [upload_evt]

            # --- Main Reduction Loop ---
            while current_n > self.plan.k:
                stage_dest_ref, _ = ppm.get_io()
                stage_event = self.agg_mgr.execute_stage(
                    self.q,
                    current_collection_ref,
                    offset_list_ref,
                    current_n,
                    elements_per_partial,
                    stage_dest_ref,
                    current_deps,
                )
                current_deps = [stage_event]

                # Prepare for the NEXT Iteration
                next_n = (current_n + self.plan.k - 1) // self.plan.k

                # The output of the last stage is the now-contiguous input for the next.
                current_collection_ref = stage_dest_ref
                next_gather_primitive = ContiguousGather(next_n, elements_per_partial)
                next_offsets_host = next_gather_primitive.get_offsets()
                offset_list_ref, upload_evt = self._create_offset_list(next_offsets_host, current_deps)
                transient_handles.append(offset_list_ref)
                current_deps = [upload_evt]
                current_n = next_n
                ppm.swap()

            # --- Final Reduction Stage (writes to the persistent destination) ---
            final_stage_event = self.agg_mgr.execute_stage(
                self.q,
                current_collection_ref,
                offset_list_ref,
                current_n,
                elements_per_partial,
                final_dest_handle,
                current_deps,
            )
            return final_stage_event

        finally:
            ppm.release()
            for handle in transient_handles:
                self.bm.release_transient_buffer(handle)
