# src/backends/opencl/renderer.py
"""OpenCL plan renderer — DAG traversal and dispatch (ADR-001)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pyopencl as cl

from ...shared.hardware_profile import HardwareProfile
from ...shared.buffer_lifecycle import BufferRole
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
from .retrieval import OpenCLKernelError, OpenCLRetrievalFuture

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
        # Kernel cache to avoid repeated kernel retrieval (RepeatedKernelRetrieval warning)
        self._kernel_cache: dict[str, cl.Kernel] = {}
        # Reduction engine bindings (set externally after construction)
        # Storage-entry variants (existing)
        self._register_reduce_binding: Any | None = None
        self._local_reduce_binding: Any | None = None
        self._clip_intermediate_binding: Any | None = None
        self._k_fan_in_binding: Any | None = None
        # ADR-026: Compute-entry variants
        self._register_reduce_from_compute_binding: Any | None = None
        self._local_reduce_from_compute_binding: Any | None = None
        self._k_fan_in_from_compute_binding: Any | None = None

    def set_reduction_bindings(
        self,
        register_reduce: Any,
        local_reduce: Any,
        clip_intermediate: Any,
        k_fan_in: Any | None = None,
        *,
        register_reduce_from_compute: Any | None = None,
        local_reduce_from_compute: Any | None = None,
        k_fan_in_from_compute: Any | None = None,
    ) -> None:
        """Inject reduction engine bindings (consumed by _render_reduction_tree).

        ADR-026: Accepts both storage-entry and compute-entry variant bindings.
        Compute-entry variants are used when the source buffer's precision_role
        is "compute" (e.g., BCE loss partials, interior reduction stages).
        """
        self._register_reduce_binding = register_reduce
        self._local_reduce_binding = local_reduce
        self._clip_intermediate_binding = clip_intermediate
        self._k_fan_in_binding = k_fan_in
        # ADR-026: Compute-entry variants
        self._register_reduce_from_compute_binding = register_reduce_from_compute
        self._local_reduce_from_compute_binding = local_reduce_from_compute
        self._k_fan_in_from_compute_binding = k_fan_in_from_compute

    @property
    def allocator(self) -> OpenCLBufferAllocator:
        return self._allocator

    def _get_kernel(self, name: str) -> cl.Kernel:
        """Get a cached kernel instance, creating it if needed."""
        if name not in self._kernel_cache:
            self._kernel_cache[name] = cl.Kernel(self._program, name)
        return self._kernel_cache[name]

    # ---------------------------------------------------------------
    # MODEL_STATE initialization (must match CPU renderer for parity)
    # ---------------------------------------------------------------
    _WEIGHT_SUBSTRINGS = ("weight",)
    _UNIT_INIT_SUBSTRINGS = ("temperature",)

    @staticmethod
    def _buffer_rng(logical_name: str, padded_shape: tuple[int, ...]) -> np.random.Generator:
        """Deterministic per-buffer RNG seeded by buffer identity."""
        import hashlib
        h = hashlib.sha256(f"{logical_name}{padded_shape}".encode()).hexdigest()
        return np.random.default_rng(int(h[:16], 16))

    def _init_model_state_buffers(self, plan: ExecutionPlan) -> None:
        """Upload Xavier/unit-initialized values for MODEL_STATE buffers."""
        role_dtypes = {
            "storage": plan.precision.storage_dtype,
            "compute": plan.precision.compute_dtype,
            "state": plan.precision.state_dtype,
        }
        for descriptor in plan.buffers.values():
            if descriptor.role != BufferRole.MODEL_STATE:
                continue
            name = descriptor.logical_name
            total = int(np.prod(descriptor.padded_shape))
            dtype = role_dtypes[descriptor.precision_role]
            if any(s in name for s in self._WEIGHT_SUBSTRINGS):
                shape_for_fan = (
                    descriptor.logical_shape
                    if descriptor.logical_shape and len(descriptor.logical_shape) >= 2
                    else descriptor.padded_shape
                )
                if len(shape_for_fan) >= 2:
                    fan_in_plus_out = shape_for_fan[-2] + shape_for_fan[-1]
                else:
                    fan_in_plus_out = max(
                        shape_for_fan[0] if shape_for_fan else total, 2,
                    )
                limit = float(np.sqrt(6.0 / fan_in_plus_out))
                rng = self._buffer_rng(name, descriptor.padded_shape)
                host = rng.uniform(
                    -limit, limit, size=total,
                ).astype(dtype)
                cl.enqueue_copy(
                    self._queue,
                    self._allocator.get_buffer(descriptor.handle),
                    host,
                )
            elif any(s in name for s in self._UNIT_INIT_SUBSTRINGS):
                host = np.ones(total, dtype=dtype)
                cl.enqueue_copy(
                    self._queue,
                    self._allocator.get_buffer(descriptor.handle),
                    host,
                )

    def render(
        self,
        plan: ExecutionPlan,
        data_injections: dict[str, NDArray] | None = None,
    ) -> dict[str, RetrievalFuture]:
        """Render an execution plan using PyOpenCL's imperative dispatch model."""
        # Store plan reference for streaming loop body lookup
        self._plan = plan

        # 1. Drain previous work, release stale buffers, allocate for this plan
        self._queue.finish()
        self._allocator.release_all()
        self._allocator.allocate_plan_buffers(plan.buffers)

        # 1b. Initialize MODEL_STATE buffers (Xavier for weights, unit for temps)
        self._init_model_state_buffers(plan)

        # 1c. Host-to-device data injection
        if data_injections:
            name_to_desc = {
                desc.logical_name: desc
                for desc in plan.buffers.values()
            }
            for logical_name, host_data in data_injections.items():
                desc = name_to_desc.get(logical_name)
                if desc is None:
                    continue
                buf = self._allocator.get_buffer(desc.handle)
                padded = np.zeros(
                    int(np.prod(desc.padded_shape)),
                    dtype=host_data.dtype,
                )
                flat = host_data.ravel()
                padded[: len(flat)] = flat
                cl.enqueue_copy(self._queue, buf, padded)

        # 2. Traverse topological order, dispatching each node
        event_map: dict[str, cl.Event] = {}
        futures: dict[str, RetrievalFuture] = {}

        try:
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
        except OpenCLKernelError:
            # Drain the queue so the device is not left in a dirty state
            # before re-raising. finish() waits for all enqueued work.
            try:
                self._queue.finish()
            except cl.RuntimeError:
                pass  # queue drain is best-effort during error recovery
            raise

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
        host_buffer = np.empty(total_elements, dtype=plan.precision.storage_dtype)

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
        kernel = self._get_kernel(binding.get_kernel_name())

        # Inject hardware-derived scalars that bindings may need
        scalar_params = self._enrich_scalar_params(node.scalar_params)

        tile_events: list[cl.Event] = []
        for tile_idx in range(node.tile_count):
            args = binding.marshal_args(
                get_buffer=self._allocator.get_buffer,
                buffer_bindings=node.buffer_bindings,
                scalar_params=scalar_params,
                tile_index=tile_idx,
            )
            global_size, local_size = binding.compute_grid(
                tile_index=tile_idx,
                scalar_params=scalar_params,
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
    # Hardware scalar injection
    # ------------------------------------------------------------------

    def _enrich_scalar_params(
        self, scalar_params: dict[str, int | float],
    ) -> dict[str, int | float]:
        """Inject hardware-derived scalars that OpenCL bindings expect."""
        enriched = dict(scalar_params)
        simd = self._hardware.simd_width if self._hardware else 16
        enriched.setdefault("work_group_size_0", min(simd * 8, 256))
        enriched.setdefault("optimal_workgroup_size_1d_reduction", min(simd * 8, 256))
        enriched.setdefault("simd_width", simd)
        enriched.setdefault("element_size", 4)
        return enriched

    # ------------------------------------------------------------------
    # Reduction tree (Phase 2B fills in full logic)
    # ------------------------------------------------------------------

    def _render_reduction_tree(
        self,
        node: ReductionTreeNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        """Render a multi-stage log_K(N) reduction tree.

        ADR-026: Selects between storage-entry and compute-entry kernel
        variants based on the source buffer's precision_role.

        Single-stage trees (num_stages == 1): dispatch existing all-to-one
        aggregate kernel + clip_intermediate_grad (unchanged).

        Multi-stage trees (num_stages > 1, ADR-019): dispatch
        reduce_k_fan_in_and_clip at each stage — one dispatch per stage
        with fused per-node summation and L2 clip.
        """
        plan = node.reduction_plan
        element_size = 4  # float32 default

        # ADR-026 §3: Determine if source buffer is compute-role
        source_desc = self._plan.buffers[plan.source_buffer]
        use_compute_entry = source_desc.precision_role == "compute"

        if plan.num_stages > 1:
            return self._render_reduction_tree_multi_stage(
                plan, wait_for, element_size, use_compute_entry
            )
        return self._render_reduction_tree_single_stage(
            plan, wait_for, element_size, use_compute_entry
        )

    def _render_reduction_tree_single_stage(
        self,
        plan: Any,
        wait_for: list[cl.Event],
        element_size: int,
        use_compute_entry: bool,
    ) -> cl.Event:
        """Single-stage tree: existing all-to-one aggregate + clip path.

        ADR-026: Selects compute-entry variant when use_compute_entry is True.
        """
        hw_simd = self._hardware.simd_width if self._hardware else 16

        # 1. Upload initial offset list to device
        offset_array = np.array(plan.initial_offset_list, dtype=np.uint32)
        offset_buf = self._allocator.allocate_internal(offset_array.nbytes)
        upload_evt = cl.enqueue_copy(
            self._queue, offset_buf, offset_array,
            wait_for=wait_for or None, is_blocking=False,
        )

        # 2. Allocate output buffer
        intermed_size = plan.partial_width * element_size
        ping = self._allocator.allocate_internal(intermed_size)

        # 3. Select kernel tier and dispatch (ADR-026: select variant)
        current_N = plan.num_partials
        if use_compute_entry:
            # ADR-026: Compute-entry variant for compute-role source buffers
            if current_N <= MAX_REG_AGG and self._register_reduce_from_compute_binding is not None:
                binding = self._register_reduce_from_compute_binding
            elif self._local_reduce_from_compute_binding is not None:
                binding = self._local_reduce_from_compute_binding
            else:
                # Fallback: use storage-entry if compute-entry not registered
                if current_N <= MAX_REG_AGG and self._register_reduce_binding is not None:
                    binding = self._register_reduce_binding
                elif self._local_reduce_binding is not None:
                    binding = self._local_reduce_binding
                else:
                    raise RuntimeError("Reduction bindings not registered")
        else:
            # Storage-entry variant (existing behavior)
            if current_N <= MAX_REG_AGG and self._register_reduce_binding is not None:
                binding = self._register_reduce_binding
            elif self._local_reduce_binding is not None:
                binding = self._local_reduce_binding
            else:
                raise RuntimeError("Reduction bindings not registered")

        kernel = self._get_kernel(binding.get_kernel_name())
        args = binding.marshal_args_reduction(
            source=self._allocator.get_buffer(plan.source_buffer),
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
        prev_event = cl.enqueue_nd_range_kernel(
            self._queue, kernel, global_size, local_size,
            wait_for=[upload_evt],
        )

        # 4. Optional clip for "sum_and_clip"
        if (plan.tree_variant == "sum_and_clip"
                and plan.threshold_schedule
                and plan.threshold_schedule[0] is not None
                and self._clip_intermediate_binding is not None):
            clip_binding = self._clip_intermediate_binding
            clip_kernel = self._get_kernel(clip_binding.get_kernel_name())
            clip_args = clip_binding.marshal_args_clip(
                buffer=ping,
                threshold=plan.threshold_schedule[0],
                epsilon=1e-7,
                param_count=plan.partial_width,
            )
            clip_global, clip_local = clip_binding.compute_grid_clip(
                param_count=plan.partial_width,
                hardware_simd_width=hw_simd,
            )
            clip_kernel.set_args(*clip_args)
            prev_event = cl.enqueue_nd_range_kernel(
                self._queue, clip_kernel, clip_global, clip_local,
                wait_for=[prev_event],
            )

        # 5. Copy final result to destination buffer
        dest_buf = self._allocator.get_buffer(plan.destination_buffer)
        copy_size = plan.partial_width * element_size
        copy_event = cl.enqueue_copy(
            self._queue, dest_buf, ping,  # type: ignore[arg-type]
            byte_count=copy_size,
            wait_for=[prev_event],
        )
        return copy_event

    # Sentinel value matching SENTINEL_ABSENT_PARTIAL in kernels.cl.h
    _SENTINEL_ABSENT_PARTIAL = 0xFFFF_FFFF

    def _render_reduction_tree_multi_stage(
        self,
        plan: Any,
        wait_for: list[cl.Event],
        element_size: int,
        use_compute_entry: bool,
    ) -> cl.Event:
        """Multi-stage tree (ADR-019): dispatch reduce_k_fan_in_and_clip per stage.

        ADR-026 §3: Stage 0 uses storage-entry or compute-entry variant based
        on the source buffer's precision_role. Stages ≥ 1 always use
        compute-entry variant (prior stage output is COMPUTE_TYPE).
        """
        if self._k_fan_in_binding is None:
            raise RuntimeError(
                "Multi-stage reduction tree requires K-fan-in binding "
                "(ADR-019). Call set_reduction_bindings with k_fan_in=..."
            )

        K = plan.fan_in
        current_N = plan.num_partials
        source_buf = self._allocator.get_buffer(plan.source_buffer)

        # Build initial flat offset list padded to node_count * K with sentinels
        node_count = math.ceil(current_N / K)
        flat_offsets = list(plan.initial_offset_list)
        pad_count = node_count * K - len(flat_offsets)
        flat_offsets.extend([self._SENTINEL_ABSENT_PARTIAL] * pad_count)
        offset_array = np.array(flat_offsets, dtype=np.uint32)
        offset_buf = self._allocator.allocate_internal(offset_array.nbytes)
        upload_evt = cl.enqueue_copy(
            self._queue, offset_buf, offset_array,
            wait_for=wait_for or None, is_blocking=False,
        )

        # Allocate ping-pong intermediate buffers sized for max node output
        max_output_elems = math.ceil(plan.num_partials / K) * plan.partial_width
        ping = self._allocator.allocate_internal(max_output_elems * element_size)
        pong = self._allocator.allocate_internal(max_output_elems * element_size)

        # ADR-026 §3: Select binding variant based on stage and source role
        storage_binding = self._k_fan_in_binding
        compute_binding = (
            self._k_fan_in_from_compute_binding
            if self._k_fan_in_from_compute_binding is not None
            else storage_binding  # fallback if not registered
        )

        prev_events = [upload_evt]

        for stage in range(plan.num_stages):
            if current_N <= 1:
                break

            node_count = math.ceil(current_N / K)

            # ADR-026 §3: Stage 0 uses variant based on source role;
            # stages ≥ 1 always use compute-entry (prior output is COMPUTE_TYPE)
            if stage == 0 and not use_compute_entry:
                fan_in_binding = storage_binding
            else:
                fan_in_binding = compute_binding

            fan_in_kernel = self._get_kernel(fan_in_binding.get_kernel_name())

            # Determine clipping threshold for this stage
            # Negative value bypasses clipping (diagnostic mode); must not default to 0.0
            # which would clip all gradients to zero norm.
            threshold = -1.0
            if (plan.tree_variant == "sum_and_clip"
                    and stage < len(plan.threshold_schedule)
                    and plan.threshold_schedule[stage] is not None):
                threshold = plan.threshold_schedule[stage]

            args = fan_in_binding.marshal_args_fan_in(
                source=source_buf,
                offset_list=offset_buf,
                dest=ping,
                fan_in=K,
                node_count=node_count,
                partial_width=plan.partial_width,
                clipping_threshold=threshold,
                epsilon=1e-7,
            )
            global_size, local_size = fan_in_binding.compute_grid_fan_in(
                node_count=node_count,
            )
            fan_in_kernel.set_args(*args)
            stage_event = cl.enqueue_nd_range_kernel(
                self._queue, fan_in_kernel, global_size, local_size,
                wait_for=prev_events,
            )
            prev_events = [stage_event]

            # Prepare for next stage: source is the output ping
            source_buf = ping
            ping, pong = pong, ping

            # Build contiguous offsets for next stage
            current_N = node_count
            if current_N > 1:
                next_node_count = math.ceil(current_N / K)
                next_flat_count = next_node_count * K
                next_offsets = np.full(next_flat_count, self._SENTINEL_ABSENT_PARTIAL, dtype=np.uint32)
                for i in range(current_N):
                    next_offsets[i] = np.uint32(i * plan.partial_width)
                new_offset_buf = self._allocator.allocate_internal(next_offsets.nbytes)
                ofs_evt = cl.enqueue_copy(
                    self._queue, new_offset_buf, next_offsets,
                    wait_for=prev_events, is_blocking=False,
                )
                offset_buf = new_offset_buf
                prev_events = [ofs_evt]

        # Copy final result to destination buffer
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
                merged_scalars = self._enrich_scalar_params(
                    {**body_node.scalar_params, **chunk_scalars}
                )

                binding = self._bindings[body_node.kernel_name]
                kernel = self._get_kernel(binding.get_kernel_name())

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
    @staticmethod
    def _collect_dependency_events(
        depends_on: frozenset[str],
        event_map: dict[str, cl.Event],
    ) -> list[cl.Event]:
        """Collect cl.Events for all dependency node IDs.

        Raises OpenCLKernelError if any dependency event completed with
        an error status, preventing cascading dispatches against a
        failed event chain.
        """
        events: list[cl.Event] = []
        for dep_id in depends_on:
            if dep_id in event_map:
                evt = event_map[dep_id]
                try:
                    status = evt.command_execution_status
                except Exception:
                    status = 0  # treat query failure as non-error
                if status < 0:
                    raise OpenCLKernelError(
                        f"Upstream node '{dep_id}' completed with "
                        f"error status {status}. Aborting downstream "
                        f"dispatch to prevent cascading failures.",
                        node_id=dep_id,
                        event_status=status,
                    )
                events.append(evt)
        return events
