# src/backends/opencl/renderer.py
"""OpenCL plan renderer — DAG traversal and dispatch (ADR-001)."""
from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pyopencl as cl

from ...shared.hardware_profile import HardwareProfile
from ...shared.buffer_lifecycle import BufferDescriptor, BufferRole
from ...shared.precision_config import PrecisionConfig
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
        self._kernel_cache: dict[str, cl.Kernel] = {}
        self._plan: ExecutionPlan | None = None

        # Persistent MODEL_STATE store keyed by (logical_name, padded_shape,
        # dtype_str) so buffers survive across render() calls with different
        # plan handle namespaces.
        self._persistent_model_state: dict[
            tuple[str, tuple[int, ...], str], cl.Buffer
        ] = {}

        # Reduction engine bindings (set by set_reduction_bindings)
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
        """Inject reduction engine bindings for the reduction-tree renderer.

        ADR-026: Accepts both storage-entry and compute-entry variants.
        """
        self._register_reduce_binding = register_reduce
        self._local_reduce_binding = local_reduce
        self._clip_intermediate_binding = clip_intermediate
        self._k_fan_in_binding = k_fan_in
        self._register_reduce_from_compute_binding = register_reduce_from_compute
        self._local_reduce_from_compute_binding = local_reduce_from_compute
        self._k_fan_in_from_compute_binding = k_fan_in_from_compute

    @property
    def allocator(self) -> OpenCLBufferAllocator:
        return self._allocator

    def _get_kernel(self, name: str) -> cl.Kernel:
        if name not in self._kernel_cache:
            self._kernel_cache[name] = cl.Kernel(self._program, name)
        return self._kernel_cache[name]

    # ------------------------------------------------------------------
    # Hardware scalar injection
    # ------------------------------------------------------------------

    def _enrich_scalar_params(
        self, scalar_params: dict[str, Any],
    ) -> dict[str, Any]:
        """Inject hardware-derived and precision-derived metadata that
        OpenCL bindings expect alongside the plan's scalar parameters.
        """
        enriched: dict[str, Any] = dict(scalar_params)
        hw = self._hardware
        simd = hw.simd_width if hw else 16

        # Bindings access "SIMD_WIDTH" (uppercase), matching CONTRACT Article 6
        enriched.setdefault("SIMD_WIDTH", simd)

        # Compute-type metadata from the active plan's PrecisionConfig
        if self._plan is not None:
            pc = self._plan.precision
            enriched.setdefault("_compute_type_size_bytes", pc.compute_dtype.itemsize)
            enriched.setdefault("_compute_dtype", pc.compute_dtype)
            enriched.setdefault("_compute_fp_format_max", pc.compute_fp_format_max)
        else:
            enriched.setdefault("_compute_type_size_bytes", 4)
            enriched.setdefault("_compute_dtype", np.float32)

        return enriched

    # ------------------------------------------------------------------
    # Persistent key helper
    # ------------------------------------------------------------------

    @staticmethod
    def _persistent_key(
        desc: BufferDescriptor, precision: PrecisionConfig,
    ) -> tuple[str, tuple[int, ...], str]:
        """Identity key for a MODEL_STATE buffer across plan generations."""
        if desc.precision_role is not None:
            role_dtypes: dict[str, np.dtype[Any]] = {
                "storage": precision.storage_dtype,
                "compute": precision.compute_dtype,
                "state": precision.state_dtype,
            }
            dtype = role_dtypes[desc.precision_role]
        else:
            dtype = np.dtype(np.float32)
        return (desc.logical_name, desc.padded_shape, dtype.str)

    # ------------------------------------------------------------------
    # MODEL_STATE initialization
    # ------------------------------------------------------------------

    _WEIGHT_SUBSTRINGS = ("weight",)
    _UNIT_INIT_SUBSTRINGS = ("temperature",)

    @staticmethod
    def _buffer_rng(
        logical_name: str, padded_shape: tuple[int, ...],
    ) -> np.random.Generator:
        import hashlib
        h = hashlib.sha256(f"{logical_name}{padded_shape}".encode()).hexdigest()
        return np.random.default_rng(int(h[:16], 16))

    def _init_model_state_buffers(
        self,
        plan: ExecutionPlan,
        only_names: set[str],
    ) -> None:
        """Upload Xavier/unit-initialized values for MODEL_STATE buffers.

        Only buffers whose ``logical_name`` is in *only_names* are touched.
        """
        role_dtypes = {
            "storage": plan.precision.storage_dtype,
            "compute": plan.precision.compute_dtype,
            "state": plan.precision.state_dtype,
        }
        for descriptor in plan.buffers.values():
            if descriptor.role != BufferRole.MODEL_STATE:
                continue
            name = descriptor.logical_name
            if name not in only_names:
                continue
            prole = descriptor.precision_role
            if prole is None:
                continue
            dtype = role_dtypes.get(prole)
            if dtype is None:
                continue

            total = int(np.prod(descriptor.padded_shape))

            if any(s in name for s in self._WEIGHT_SUBSTRINGS):
                shape_for_fan = (
                    descriptor.logical_shape
                    if descriptor.logical_shape and len(descriptor.logical_shape) >= 2
                    else descriptor.padded_shape
                )
                if len(shape_for_fan) >= 2:
                    fan_in_plus_out = shape_for_fan[-2] + shape_for_fan[-1]
                else:
                    fan_in_plus_out = max(shape_for_fan[0] if shape_for_fan else total, 2)
                limit = float(np.sqrt(6.0 / fan_in_plus_out))
                rng = self._buffer_rng(name, descriptor.padded_shape)
                host = rng.uniform(-limit, limit, size=total).astype(dtype)
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

    # ------------------------------------------------------------------
    # Data injection
    # ------------------------------------------------------------------

    def _upload_data_injections(
        self,
        plan: ExecutionPlan,
        injections: dict[str, NDArray[Any]],
    ) -> None:
        """Upload host data arrays into their corresponding device buffers."""
        name_to_desc = {d.logical_name: d for d in plan.buffers.values()}
        role_dtypes = {
            "storage": plan.precision.storage_dtype,
            "compute": plan.precision.compute_dtype,
            "state": plan.precision.state_dtype,
        }

        for name, host_data in injections.items():
            desc = name_to_desc.get(name)
            if desc is None:
                continue

            # Determine the upload dtype from the buffer's precision role.
            # "flag-conditional", "exempt", or None → use the host data's
            # native type (preserves integer class indices, uint masks, etc.)
            prole = desc.precision_role
            if prole is not None and prole in role_dtypes:
                buf_dtype = role_dtypes[prole]
            else:
                buf_dtype = host_data.dtype

            ps = desc.padded_shape
            padded = np.zeros(ps, dtype=buf_dtype)
            src = host_data if host_data.dtype == buf_dtype else host_data.astype(buf_dtype)

            # Copy logical extent into padded array
            if src.ndim == len(ps) and src.ndim > 0:
                slices = tuple(
                    slice(0, min(src.shape[d], ps[d]))
                    for d in range(src.ndim)
                )
                padded[slices] = src[slices]
            elif src.size > 0:
                flat_dst = padded.ravel()
                flat_src = src.ravel()
                n = min(len(flat_src), len(flat_dst))
                flat_dst[:n] = flat_src[:n]

            cl.enqueue_copy(
                self._queue,
                self._allocator.get_buffer(desc.handle),
                padded.ravel(),
            )

    # ==================================================================
    # Main entry point
    # ==================================================================

    def render(
        self,
        plan: ExecutionPlan,
        data_injections: dict[str, NDArray[Any]] | None = None,
    ) -> dict[str, RetrievalFuture]:
        """Render an execution plan using PyOpenCL's imperative dispatch model."""
        self._plan = plan
        self._queue.finish()

        # --- Persistent MODEL_STATE management ---
        # 1. Extract MODEL_STATE buffers before release (keeps cl.Buffers alive)
        for _, cl_buf, desc in self._allocator.extract_buffers_by_role(
            BufferRole.MODEL_STATE,
        ):
            key = self._persistent_key(desc, plan.precision)
            self._persistent_model_state[key] = cl_buf

        # 2. Release everything still in the allocator
        self._allocator.release_all()

        # 3. Re-inject persistent buffers under the new plan's handles
        newly_needed: set[str] = set()
        current_keys: set[tuple[str, tuple[int, ...], str]] = set()
        for desc in plan.buffers.values():
            if desc.role != BufferRole.MODEL_STATE:
                continue
            key = self._persistent_key(desc, plan.precision)
            current_keys.add(key)
            if key in self._persistent_model_state:
                self._allocator.inject_buffer(
                    desc.handle, self._persistent_model_state[key], desc,
                )
            else:
                newly_needed.add(desc.logical_name)

        # 3b. Purge stale MODEL_STATE buffers no longer needed by this plan
        # (e.g., from config changes: different padded_shape, precision, etc.)
        for key in list(self._persistent_model_state):
            if key not in current_keys:
                self._persistent_model_state.pop(key).release()

        # 4. Allocate remaining buffers (skips pre-injected handles)
        self._allocator.allocate_plan_buffers(plan.buffers)

        # 5. Record newly-allocated MODEL_STATE for future render() calls
        for desc in plan.buffers.values():
            if desc.role == BufferRole.MODEL_STATE and desc.logical_name in newly_needed:
                key = self._persistent_key(desc, plan.precision)
                self._persistent_model_state[key] = self._allocator.get_buffer(
                    desc.handle,
                )

        # 6. Initialize only *new* MODEL_STATE buffers
        if newly_needed:
            self._init_model_state_buffers(plan, only_names=newly_needed)

        # 7. Host→device data injection
        if data_injections:
            self._upload_data_injections(plan, data_injections)

        # 8. DAG traversal
        event_map: dict[str, cl.Event] = {}
        futures: dict[str, RetrievalFuture] = {}

        try:
            for node_id in plan.topological_order:
                node = plan.nodes[node_id]
                wait_for = self._collect_dependency_events(
                    node.depends_on, event_map,
                )

                if isinstance(node, KernelDispatchNode):
                    event_map[node_id] = self._render_kernel_dispatch(
                        node, wait_for,
                    )
                elif isinstance(node, ReductionTreeNode):
                    event_map[node_id] = self._render_reduction_tree(
                        node, wait_for,
                    )
                elif isinstance(node, StreamingLoopNode):
                    event_map[node_id] = self._render_streaming_loop(
                        node, wait_for,
                    )
                elif isinstance(node, BarrierNode):
                    event_map[node_id] = self._render_barrier(wait_for)
                else:
                    # Exhaustive: only RetrievalNode remains in PlanNode union
                    assert isinstance(node, RetrievalNode), (
                        f"Unknown plan node type '{type(node).__name__}' "
                        f"for node '{node_id}'"
                    )
                    future = self._render_retrieval(node, wait_for, plan)
                    futures[node.event_name] = future
                    event_map[node_id] = future.event
        except Exception:
            # Flush the command queue on any exception to prevent stale commands
            # from affecting subsequent operations.
            try:
                self._queue.finish()
            except cl.RuntimeError:
                pass
            raise

        return futures

    # ------------------------------------------------------------------
    # Barrier & retrieval
    # ------------------------------------------------------------------

    def _render_barrier(self, wait_for: list[cl.Event]) -> cl.Event:
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
        desc = plan.buffers[node.source_buffer]
        ps = desc.padded_shape
        total_elements = int(np.prod(ps))

        # Use the source buffer's precision-role dtype for the host array.
        prole = desc.precision_role
        match prole:
            case "storage":
                dtype = plan.precision.storage_dtype
            case "state":
                dtype = plan.precision.state_dtype
            case _:  # "compute" or None
                dtype = plan.precision.compute_dtype

        host_buffer: NDArray[Any] = np.empty(total_elements, dtype=dtype)
        event = self._allocator.enqueue_read(
            node.source_buffer, host_buffer, wait_for=wait_for or None,
        )
        return OpenCLRetrievalFuture(
            node_id=node.node_id,
            event=event,
            host_buffer=host_buffer,
            logical_shape=node.logical_shape,
            padded_shape=ps,
        )

    # ------------------------------------------------------------------
    # Kernel dispatch
    # ------------------------------------------------------------------

    def _render_kernel_dispatch(
        self,
        node: KernelDispatchNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        binding = self._bindings[node.kernel_name]
        kernel = self._get_kernel(binding.get_kernel_name())
        scalar_params = self._enrich_scalar_params(node.scalar_params)
        hw_simd = self._hardware.simd_width if self._hardware else 16

        tile_events: list[cl.Event] = []
        for tile_idx in range(node.tile_count):
            # Two-phase binding protocol: prepare_dispatch may augment
            # scalar_params with binding-computed values (_prepared_* keys)
            # and perform device-side resource preparation (e.g., schedule
            # uploads).  marshal_args is purely functional.
            prepared_params = binding.prepare_dispatch(
                queue=self._queue,
                get_buffer=self._allocator.get_buffer,
                buffer_bindings=node.buffer_bindings,
                scalar_params=scalar_params,
                tile_index=tile_idx,
            )
            args = binding.marshal_args(
                get_buffer=self._allocator.get_buffer,
                buffer_bindings=node.buffer_bindings,
                scalar_params=prepared_params,
                tile_index=tile_idx,
            )
            global_size, local_size = binding.compute_grid(
                tile_index=tile_idx,
                scalar_params=prepared_params,
                hardware_simd_width=hw_simd,
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
    # Reduction tree
    # ------------------------------------------------------------------

    _SENTINEL_ABSENT_PARTIAL = 0xFFFF_FFFF

    def _render_reduction_tree(
        self,
        node: ReductionTreeNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        """Render a multi-stage log_K(N) reduction tree.

        ADR-026: Selects between storage-entry and compute-entry kernel
        variants based on the source buffer's precision_role.
        """
        assert self._plan is not None
        rplan = node.reduction_plan
        compute_elem_size = self._plan.precision.compute_dtype.itemsize

        source_desc = self._plan.buffers[rplan.source_buffer]
        use_compute_entry = source_desc.precision_role == "compute"

        if rplan.num_stages > 1:
            return self._render_reduction_tree_multi_stage(
                rplan, wait_for, compute_elem_size, use_compute_entry,
            )
        return self._render_reduction_tree_single_stage(
            rplan, wait_for, compute_elem_size, use_compute_entry,
        )

    def _render_reduction_tree_single_stage(
        self,
        rplan: Any,
        wait_for: list[cl.Event],
        compute_elem_size: int,
        use_compute_entry: bool,
    ) -> cl.Event:
        """Single-stage: all-to-one aggregate + optional clip."""
        assert self._plan is not None
        hw_simd = self._hardware.simd_width if self._hardware else 16
        N = rplan.num_partials
        W = rplan.partial_width
        # Operation type: 0=SUM (default), future-proofed for plan extensibility
        op_type: int = getattr(rplan, 'operation_type', 0)

        # 1. Upload offset list
        ofs_arr = np.array(rplan.initial_offset_list, dtype=np.uint32)
        ofs_buf = self._allocator.allocate_internal(ofs_arr.nbytes)
        up_evt = cl.enqueue_copy(
            self._queue, ofs_buf, ofs_arr,
            wait_for=wait_for or None, is_blocking=False,
        )

        # 2. Intermediate output (COMPUTE_TYPE)
        out_buf = self._allocator.allocate_internal(W * compute_elem_size)
        source_buf = self._allocator.get_buffer(rplan.source_buffer)

        # 3. Tier selection (register vs local) + variant (storage vs compute)
        # ADR-026: If source buffer is compute-role, we MUST use compute-entry
        # variant.  Falling back to storage-entry would reinterpret compute-
        # precision data as storage-precision — silent corruption.
        if N <= MAX_REG_AGG:
            if use_compute_entry:
                if self._register_reduce_from_compute_binding is None:
                    raise RuntimeError(
                        "Compute-entry register-reduce binding required for "
                        "compute-role source buffers, but not registered via "
                        "set_reduction_bindings()"
                    )
                binding = self._register_reduce_from_compute_binding
            else:
                binding = self._register_reduce_binding
            if binding is None:
                raise RuntimeError("Register-reduce binding not registered")
            # Register-reduce: no local memory, no compute_type_size_bytes arg
            args = binding.marshal_args_direct(
                source_buf, ofs_buf, out_buf, N, W, op_type,
            )
        else:
            if use_compute_entry:
                if self._local_reduce_from_compute_binding is None:
                    raise RuntimeError(
                        "Compute-entry local-reduce binding required for "
                        "compute-role source buffers, but not registered via "
                        "set_reduction_bindings()"
                    )
                binding = self._local_reduce_from_compute_binding
            else:
                binding = self._local_reduce_binding
            if binding is None:
                raise RuntimeError("Local-reduce binding not registered")
            # Local-reduce: needs compute_type_size_bytes for local alloc
            args = binding.marshal_args_direct(
                source_buf, ofs_buf, out_buf, N, W, op_type,
                compute_type_size_bytes=compute_elem_size,
            )

        kernel = self._get_kernel(binding.get_kernel_name())
        gs, ls = binding.compute_grid_direct(W, hw_simd)
        kernel.set_args(*args)
        prev = cl.enqueue_nd_range_kernel(
            self._queue, kernel, gs, ls, wait_for=[up_evt],
        )

        # 4. Optional clip for "sum_and_clip" trees
        if (
            rplan.tree_variant == "sum_and_clip"
            and rplan.threshold_schedule
            and rplan.threshold_schedule[0] is not None
            and self._clip_intermediate_binding is not None
        ):
            cb = self._clip_intermediate_binding
            ck = self._get_kernel(cb.get_kernel_name())
            c_args = cb.marshal_args_direct(
                out_buf,
                rplan.threshold_schedule[0],
                1e-7,
                W,
                compute_type_size_bytes=compute_elem_size,
                compute_dtype=self._plan.precision.compute_dtype,
            )
            c_gs, c_ls = cb.compute_grid_direct(W, hw_simd)
            ck.set_args(*c_args)
            prev = cl.enqueue_nd_range_kernel(
                self._queue, ck, c_gs, c_ls, wait_for=[prev],
            )

        # 5. Copy result to destination (cast needed: pyopencl stubs lack Buffer-to-Buffer typing)
        dest_buf = self._allocator.get_buffer(rplan.destination_buffer)
        return cl.enqueue_copy(
            self._queue, cast(Any, dest_buf), cast(Any, out_buf),
            byte_count=W * compute_elem_size,
            wait_for=[prev], is_blocking=False,
        )

    def _render_reduction_tree_multi_stage(
        self,
        rplan: Any,
        wait_for: list[cl.Event],
        compute_elem_size: int,
        use_compute_entry: bool,
    ) -> cl.Event:
        """Multi-stage (ADR-019): dispatch reduce_k_fan_in_and_clip per stage."""
        assert self._plan is not None
        if self._k_fan_in_binding is None:
            raise RuntimeError(
                "Multi-stage reduction requires K-fan-in binding (ADR-019)"
            )

        hw_simd = self._hardware.simd_width if self._hardware else 16
        c_dtype = self._plan.precision.compute_dtype
        K = rplan.fan_in
        N = rplan.num_partials
        W = rplan.partial_width
        source = self._allocator.get_buffer(rplan.source_buffer)

        # Pad initial offset list with sentinels to node_count × K
        node_count = math.ceil(N / K)
        flat = list(rplan.initial_offset_list)
        flat.extend([self._SENTINEL_ABSENT_PARTIAL] * (node_count * K - len(flat)))
        ofs_arr = np.array(flat, dtype=np.uint32)
        ofs_buf = self._allocator.allocate_internal(ofs_arr.nbytes)
        prev_evt = cl.enqueue_copy(
            self._queue, ofs_buf, ofs_arr,
            wait_for=wait_for or None, is_blocking=False,
        )

        # Ping-pong intermediate buffers
        max_elems = math.ceil(N / K) * W
        ping = self._allocator.allocate_internal(max_elems * compute_elem_size)
        pong = self._allocator.allocate_internal(max_elems * compute_elem_size)

        storage_k = self._k_fan_in_binding
        # ADR-026: Stages >=1 always read COMPUTE_TYPE intermediates, so compute-
        # entry variant is required.  Silently falling back to storage-entry would
        # reinterpret compute-precision data as storage-precision — fail-fast.
        if self._k_fan_in_from_compute_binding is None:
            raise RuntimeError(
                "Multi-stage reduction requires compute-entry K-fan-in binding "
                "(ADR-019, ADR-026) for interior stages, but not registered via "
                "set_reduction_bindings()"
            )
        compute_k = self._k_fan_in_from_compute_binding

        events: list[cl.Event] = [prev_evt]
        cur_N = N

        for stage in range(rplan.num_stages):
            if cur_N <= 1:
                break

            nc = math.ceil(cur_N / K)

            # ADR-026: stage 0 uses storage-entry iff source is storage-role;
            # stages >= 1 always use compute-entry (prior output is COMPUTE_TYPE)
            binding = (
                storage_k if (stage == 0 and not use_compute_entry) else compute_k
            )

            # Threshold: negative bypasses clipping (diagnostic mode)
            thresh = -1.0
            if (
                rplan.tree_variant == "sum_and_clip"
                and stage < len(rplan.threshold_schedule)
                and rplan.threshold_schedule[stage] is not None
            ):
                thresh = rplan.threshold_schedule[stage]

            kernel = self._get_kernel(binding.get_kernel_name())
            args = binding.marshal_args_direct(
                source, ofs_buf, ping,
                K, nc, W, thresh, 1e-7,
                compute_type_size_bytes=compute_elem_size,
                compute_dtype=c_dtype,
            )
            gs, ls = binding.compute_grid_direct(nc, hw_simd)
            kernel.set_args(*args)
            evt = cl.enqueue_nd_range_kernel(
                self._queue, kernel, gs, ls, wait_for=events,
            )
            events = [evt]

            # Swap for next stage
            source = ping
            ping, pong = pong, ping
            cur_N = nc

            # Build contiguous offsets for next stage
            if cur_N > 1:
                next_nc = math.ceil(cur_N / K)
                next_ofs = np.full(
                    next_nc * K, self._SENTINEL_ABSENT_PARTIAL, dtype=np.uint32,
                )
                for i in range(cur_N):
                    next_ofs[i] = np.uint32(i * W)
                new_buf = self._allocator.allocate_internal(next_ofs.nbytes)
                ofs_evt = cl.enqueue_copy(
                    self._queue, new_buf, next_ofs,
                    wait_for=events, is_blocking=False,
                )
                ofs_buf = new_buf
                events = [ofs_evt]

        # Copy final result to destination (cast needed: pyopencl stubs lack Buffer-to-Buffer typing)
        dest = self._allocator.get_buffer(rplan.destination_buffer)
        return cl.enqueue_copy(
            self._queue, cast(Any, dest), cast(Any, source),
            byte_count=W * compute_elem_size,
            wait_for=events, is_blocking=False,
        )

    # ------------------------------------------------------------------
    # Streaming loop
    # ------------------------------------------------------------------

    def _render_streaming_loop(
        self,
        node: StreamingLoopNode,
        wait_for: list[cl.Event],
    ) -> cl.Event:
        """Render a streaming loop: sequential per-chunk parametric dispatch."""
        assert self._plan is not None
        splan = node.streaming_plan
        hw_simd = self._hardware.simd_width if self._hardware else 16

        chunk_events = wait_for

        for chunk_idx in range(splan.iteration.chunk_count):
            # Compute per-chunk scalar overrides from strides
            chunk_scalars: dict[str, Any] = dict(splan.constant_scalars)
            for stride in splan.parameter_strides:
                chunk_scalars[stride.param_name] = (
                    stride.base + chunk_idx * stride.stride
                )

            # Dispatch each body node sequentially within the chunk
            body_events = chunk_events
            for body_node_id in splan.body:
                body_node = self._plan.nodes[body_node_id]
                if not isinstance(body_node, KernelDispatchNode):
                    raise TypeError(
                        f"StreamingLoopNode body node '{body_node_id}' is "
                        f"{type(body_node).__name__}, expected KernelDispatchNode"
                    )

                merged = self._enrich_scalar_params(
                    {**body_node.scalar_params, **chunk_scalars},
                )
                binding = self._bindings[body_node.kernel_name]
                kernel = self._get_kernel(binding.get_kernel_name())

                tile_events: list[cl.Event] = []
                for tile_idx in range(body_node.tile_count):
                    # Two-phase binding protocol (see _render_kernel_dispatch).
                    prepared_params = binding.prepare_dispatch(
                        queue=self._queue,
                        get_buffer=self._allocator.get_buffer,
                        buffer_bindings=body_node.buffer_bindings,
                        scalar_params=merged,
                        tile_index=tile_idx,
                    )
                    args = binding.marshal_args(
                        get_buffer=self._allocator.get_buffer,
                        buffer_bindings=body_node.buffer_bindings,
                        scalar_params=prepared_params,
                        tile_index=tile_idx,
                    )
                    gs, ls = binding.compute_grid(
                        tile_index=tile_idx,
                        scalar_params=prepared_params,
                        hardware_simd_width=hw_simd,
                    )
                    kernel.set_args(*args)
                    evt = cl.enqueue_nd_range_kernel(
                        self._queue, kernel, gs, ls,
                        wait_for=body_events or None,
                    )
                    tile_events.append(evt)

                body_events = (
                    tile_events if len(tile_events) == 1
                    else [cl.enqueue_marker(self._queue, wait_for=tile_events)]
                )

            chunk_events = body_events

        if chunk_events:
            return chunk_events[0]
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
        """Collect cl.Events for all dependency node IDs.

        Raises OpenCLKernelError if any dependency event completed with
        an error status.
        """
        events: list[cl.Event] = []
        for dep_id in depends_on:
            evt = event_map.get(dep_id)
            if evt is None:
                continue
            try:
                status = evt.command_execution_status
            except Exception:
                status = 0
            if status < 0:
                raise OpenCLKernelError(
                    f"Upstream node '{dep_id}' completed with error "
                    f"status {status}.",
                    node_id=dep_id,
                    event_status=status,
                )
            events.append(evt)
        return events
