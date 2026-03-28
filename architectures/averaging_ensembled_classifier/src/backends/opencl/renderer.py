# src/backends/opencl/renderer.py
"""OpenCL plan renderer — DAG traversal and dispatch (ADR-001)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyopencl as cl

from ...shared.hardware_profile import HardwareProfile
from ...shared.plan_types import (
    BarrierNode,
    ExecutionPlan,
    KernelDispatchNode,
    ReductionTreeNode,
    RetrievalNode,
    StreamingLoopNode,
)
from ...shared.retrieval_future import RetrievalFuture
from .buffer_allocator import OpenCLBufferAllocator
from .retrieval import OpenCLRetrievalFuture

# Maximum fan-in for register-reduce (vs. local-reduce) tier selection.
MAX_REG_AGG = 16


class OpenCLPlanRenderer:
    """OpenCL implementation of PlanRenderer (ADR-001).

    Renders an ExecutionPlan by traversing its topological order and
    dispatching each node using PyOpenCL's imperative enqueue model.
    """

    def __init__(
        self,
        context: cl.Context,
        queue: cl.CommandQueue,
        program: cl.Program,
        kernel_bindings: dict[str, Any],
        hardware: HardwareProfile | None = None,
    ) -> None:
        self._context = context
        self._queue = queue
        self._program = program
        self._bindings = kernel_bindings
        self._hardware = hardware
        self._allocator = OpenCLBufferAllocator(context, queue)
        # Reduction engine bindings (set externally after construction)
        self._register_reduce_binding: Any | None = None
        self._local_reduce_binding: Any | None = None
        self._clip_intermediate_binding: Any | None = None

    def set_reduction_bindings(
        self,
        register_reduce: Any,
        local_reduce: Any,
        clip_intermediate: Any,
    ) -> None:
        """Inject reduction engine bindings (consumed by _render_reduction_tree)."""
        self._register_reduce_binding = register_reduce
        self._local_reduce_binding = local_reduce
        self._clip_intermediate_binding = clip_intermediate

    @property
    def allocator(self) -> OpenCLBufferAllocator:
        return self._allocator

    def render(self, plan: ExecutionPlan) -> dict[str, RetrievalFuture]:
        """Render an execution plan using PyOpenCL's imperative dispatch model."""
        # Store plan reference for streaming loop body lookup
        self._plan = plan

        # 1. Allocate all plan buffers
        self._allocator.allocate_plan_buffers(plan.buffers)

        # 2. Traverse topological order, dispatching each node
        event_map: dict[str, cl.Event] = {}
        futures: dict[str, RetrievalFuture] = {}

        for node_id in plan.topological_order:
            node = plan.nodes[node_id]
            wait_for = self._collect_dependency_events(node.depends_on, event_map)

            if isinstance(node, KernelDispatchNode):
                event = self._render_kernel_dispatch(node, wait_for)
                event_map[node_id] = event

            elif isinstance(node, ReductionTreeNode):
                event = self._render_reduction_tree(node, wait_for)
                event_map[node_id] = event

            elif isinstance(node, StreamingLoopNode):
                event = self._render_streaming_loop(node, wait_for)
                event_map[node_id] = event

            elif isinstance(node, BarrierNode):
                event = self._render_barrier(wait_for)
                event_map[node_id] = event

            else:  # RetrievalNode
                assert isinstance(node, RetrievalNode)
                future = self._render_retrieval(node, wait_for, plan)
                futures[node.event_name] = future
                event_map[node_id] = future.event

        return futures

    # ------------------------------------------------------------------
    # Barrier & retrieval (fully functional in Phase 2A)
    # ------------------------------------------------------------------

    def _render_barrier(self, wait_for: list[cl.Event]) -> cl.Event:
        """Render a BarrierNode as a marker event joining upstream events."""
        if not wait_for:
            marker = cl.UserEvent(self._context)
            marker.set_status(cl.command_execution_status.COMPLETE)
            return marker
        return cl.enqueue_marker(self._queue, wait_for=wait_for)

    def _render_retrieval(
        self,
        node: RetrievalNode,
        wait_for: list[cl.Event],
        plan: ExecutionPlan,
    ) -> OpenCLRetrievalFuture:
        """Render a RetrievalNode: enqueue async D2H read, return future."""
        desc = plan.buffers[node.source_buffer]
        padded_shape = desc.padded_shape
        total_elements = 1
        for d in padded_shape:
            total_elements *= d
        host_buffer = np.empty(total_elements, dtype=plan.precision.numpy_dtype)

        event = self._allocator.enqueue_read(
            node.source_buffer, host_buffer, wait_for=wait_for or None,
        )
        return OpenCLRetrievalFuture(
            node_id=node.node_id,
            event=event,
            host_buffer=host_buffer,
            logical_shape=node.logical_shape,
            padded_shape=padded_shape,
        )

    # ------------------------------------------------------------------
    # Kernel dispatch (Phase 2B fills in full logic)
    # ------------------------------------------------------------------

    def _render_kernel_dispatch(
        self,
        node: KernelDispatchNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        """Dispatch a KernelDispatchNode via per-tile imperative enqueue."""
        binding = self._bindings[node.kernel_name]
        kernel = getattr(self._program, binding.get_kernel_name())

        tile_events: list[cl.Event] = []
        for tile_idx in range(node.tile_count):
            args = binding.marshal_args(
                get_buffer=self._allocator.get_buffer,
                buffer_bindings=node.buffer_bindings,
                scalar_params=node.scalar_params,
                tile_index=tile_idx,
            )
            global_size, local_size = binding.compute_grid(
                tile_index=tile_idx,
                scalar_params=node.scalar_params,
                hardware_simd_width=self._hardware.simd_width if self._hardware else 16,
            )
            kernel.set_args(*args)
            event = cl.enqueue_nd_range_kernel(
                self._queue, kernel, global_size, local_size,
                wait_for=wait_for or None,
            )
            tile_events.append(event)

        if len(tile_events) == 1:
            return tile_events[0]
        return cl.enqueue_marker(self._queue, wait_for=tile_events)

    # ------------------------------------------------------------------
    # Reduction tree (Phase 2B fills in full logic)
    # ------------------------------------------------------------------

    def _render_reduction_tree(
        self,
        node: ReductionTreeNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        """Render a multi-stage log_K(N) reduction tree."""
        plan = node.reduction_plan
        hw_simd = self._hardware.simd_width if self._hardware else 16
        element_size = 4  # float32 default

        # 1. Upload initial offset list to device
        offset_array = np.array(plan.initial_offset_list, dtype=np.uint32)
        offset_buf = self._allocator.allocate_internal(offset_array.nbytes)
        upload_evt = cl.enqueue_copy(
            self._queue, offset_buf, offset_array,
            wait_for=wait_for or None, is_blocking=False,
        )

        # 2. Allocate ping-pong intermediate buffers
        intermed_size = plan.partial_width * element_size
        ping = self._allocator.allocate_internal(intermed_size)
        pong = self._allocator.allocate_internal(intermed_size)

        # 3. Stage loop
        source_buf = self._allocator.get_buffer(plan.source_buffer)
        current_N = plan.num_partials
        K = plan.fan_in_K
        prev_events = [upload_evt]

        for stage in range(plan.num_stages):
            if current_N <= 1:
                break

            output_N = math.ceil(current_N / K)

            # Select kernel tier
            if current_N <= MAX_REG_AGG and self._register_reduce_binding is not None:
                binding = self._register_reduce_binding
            elif self._local_reduce_binding is not None:
                binding = self._local_reduce_binding
            else:
                raise RuntimeError("Reduction bindings not registered")

            kernel = getattr(self._program, binding.get_kernel_name())

            # Marshal and dispatch reduction stage
            args = binding.marshal_args_reduction(
                source=source_buf,
                offset_list=offset_buf,
                dest=ping,
                offset_count=current_N,
                partial_width=plan.partial_width,
                operation_type=0,  # sum
            )
            global_size, local_size = binding.compute_grid_reduction(
                partial_width=plan.partial_width,
                hardware_simd_width=hw_simd,
            )
            kernel.set_args(*args)
            agg_event = cl.enqueue_nd_range_kernel(
                self._queue, kernel, global_size, local_size,
                wait_for=prev_events or None,
            )
            prev_events = [agg_event]

            # Optional clip for "sum_and_clip"
            if (plan.tree_variant == "sum_and_clip"
                    and stage < len(plan.threshold_schedule)
                    and plan.threshold_schedule[stage] is not None
                    and self._clip_intermediate_binding is not None):
                clip_binding = self._clip_intermediate_binding
                clip_kernel = getattr(self._program, clip_binding.get_kernel_name())
                clip_args = clip_binding.marshal_args_clip(
                    buffer=ping,
                    threshold=plan.threshold_schedule[stage],
                    epsilon=1e-7,
                    param_count=plan.partial_width,
                )
                clip_global, clip_local = clip_binding.compute_grid_clip(
                    param_count=plan.partial_width,
                    hardware_simd_width=hw_simd,
                )
                clip_kernel.set_args(*clip_args)
                clip_event = cl.enqueue_nd_range_kernel(
                    self._queue, clip_kernel, clip_global, clip_local,
                    wait_for=prev_events,
                )
                prev_events = [clip_event]

            # Swap ping-pong; source for next stage is the output ping
            source_buf = ping
            ping, pong = pong, ping

            # Compute contiguous offsets for next stage
            if output_N > 1:
                next_offsets = np.arange(output_N, dtype=np.uint32) * plan.partial_width
                new_offset_buf = self._allocator.allocate_internal(next_offsets.nbytes)
                ofs_evt = cl.enqueue_copy(
                    self._queue, new_offset_buf, next_offsets,
                    wait_for=prev_events, is_blocking=False,
                )
                offset_buf = new_offset_buf
                prev_events = [ofs_evt]

            current_N = output_N

        # 4. Copy final result to destination buffer
        dest_buf = self._allocator.get_buffer(plan.destination_buffer)
        copy_size = plan.partial_width * element_size
        copy_event = cl.enqueue_copy(
            self._queue, dest_buf, source_buf,  # type: ignore[arg-type]
            byte_count=copy_size,
            wait_for=prev_events, is_blocking=False,
        )
        return copy_event

    # ------------------------------------------------------------------
    # Streaming loop (Phase 2B fills in full logic)
    # ------------------------------------------------------------------

    def _render_streaming_loop(
        self,
        node: StreamingLoopNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        """Render a streaming loop: per-chunk parametric dispatch."""
        splan = node.streaming_plan

        # Allocate scratch buffers
        scratch_bufs: dict[str, cl.Buffer] = {}
        for spec in splan.scratch_buffers:
            scratch_bufs[spec.logical_name] = self._allocator.allocate_internal(spec.size_bytes)

        chunk_events = wait_for

        for chunk_idx in range(splan.iteration.chunk_count):
            # Compute per-chunk scalars from strides
            chunk_scalars: dict[str, int | float] = dict(splan.constant_scalars)
            for stride in splan.parameter_strides:
                chunk_scalars[stride.param_name] = stride.base + chunk_idx * stride.stride

            # Dispatch each body node in sequence
            body_events = chunk_events
            for body_node_id in splan.body:
                body_node = self._plan.nodes[body_node_id]
                assert isinstance(body_node, KernelDispatchNode)

                # Merge chunk-specific scalars over the body node's base scalars
                merged_scalars = {**body_node.scalar_params, **chunk_scalars}

                binding = self._bindings[body_node.kernel_name]
                kernel = getattr(self._program, binding.get_kernel_name())

                tile_events: list[cl.Event] = []
                for tile_idx in range(body_node.tile_count):
                    args = binding.marshal_args(
                        get_buffer=self._allocator.get_buffer,
                        buffer_bindings=body_node.buffer_bindings,
                        scalar_params=merged_scalars,
                        tile_index=tile_idx,
                    )
                    global_size, local_size = binding.compute_grid(
                        tile_index=tile_idx,
                        scalar_params=merged_scalars,
                        hardware_simd_width=self._hardware.simd_width if self._hardware else 16,
                    )
                    kernel.set_args(*args)
                    event = cl.enqueue_nd_range_kernel(
                        self._queue, kernel, global_size, local_size,
                        wait_for=body_events or None,
                    )
                    tile_events.append(event)

                if len(tile_events) == 1:
                    body_events = tile_events
                else:
                    body_events = [cl.enqueue_marker(self._queue, wait_for=tile_events)]

            chunk_events = body_events

        if chunk_events:
            return chunk_events[0]
        # Fallback: return a completed user event
        marker = cl.UserEvent(self._context)
        marker.set_status(cl.command_execution_status.COMPLETE)
        return marker

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_dependency_events(
        depends_on: frozenset[str],
        event_map: dict[str, cl.Event],
    ) -> list[cl.Event]:
        """Collect cl.Events for all dependency node IDs."""
        events: list[cl.Event] = []
        for dep_id in depends_on:
            if dep_id in event_map:
                events.append(event_map[dep_id])
        return events
