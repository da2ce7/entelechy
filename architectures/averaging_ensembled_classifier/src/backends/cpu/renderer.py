"""CPUPlanRenderer — PlanRenderer implementation for the CPU backend (ADR-001).

Traverses an ExecutionPlan DAG in topological order and dispatches each
node via the CPU kernel library's pool_dispatch_and_wait.
"""
from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import Any

import numpy as np

from ...shared.plan_types import (
    BarrierNode,
    ExecutionPlan,
    KernelDispatchNode,
    ReductionTreeNode,
    StreamingLoopNode,
)
from ...shared.retrieval_future import RetrievalFuture
from ._dispatch_table import build_dispatch_table
from ._ffi_types import ReductionTreePlanFFI, c_float_p, c_int_p, c_uint_p
from ._loader import load_cpu_library
from .buffer_allocator import CPUBufferAllocator
from .discovery import detect_thread_count
from .retrieval import CPURetrievalFuture


class CPUPlanRenderer:
    """Plan renderer for the CPU backend (ADR-001, ADR-015).

    Traverses an ExecutionPlan DAG in topological order and dispatches
    each node via the CPU kernel library's pool_dispatch_and_wait.
    """

    def __init__(self, thread_count: int | None = None) -> None:
        self._lib = load_cpu_library()
        self._dispatch_table = build_dispatch_table(self._lib)

        if thread_count is None:
            thread_count = detect_thread_count()
        self._pool = self._lib.pool_create(thread_count)
        self._thread_count = thread_count

    def __del__(self) -> None:
        if hasattr(self, "_pool") and self._pool:
            self._lib.pool_destroy(self._pool)
            self._pool = None

    def render(
        self, plan: ExecutionPlan
    ) -> dict[str, RetrievalFuture]:
        """Render an execution plan via CPU dispatch.

        Allocates SIMD-aligned numpy buffers, traverses the plan's
        topological order, and dispatches each node type.
        """
        allocator = CPUBufferAllocator(
            simd_alignment=plan.hardware.cache_line_bytes,
            dtype=plan.precision.numpy_dtype,
        )
        for descriptor in plan.buffers.values():
            allocator.allocate(descriptor)

        futures: dict[str, RetrievalFuture] = {}

        for node_id in plan.topological_order:
            node = plan.nodes[node_id]

            if isinstance(node, KernelDispatchNode):
                self._render_kernel_dispatch(node, allocator)
            elif isinstance(node, ReductionTreeNode):
                self._render_reduction_tree(node, allocator, plan)
            elif isinstance(node, StreamingLoopNode):
                self._render_streaming_loop(node, allocator, plan)
            elif isinstance(node, BarrierNode):
                pass  # Implicit — blocking dispatch provides sync
            else:
                future = CPURetrievalFuture(
                    node_id=node.node_id,
                    padded_buffer=allocator.get_buffer(node.source_buffer),
                    logical_shape=node.logical_shape,
                )
                futures[node.event_name] = future

        return futures

    # -----------------------------------------------------------
    # Node-type dispatch methods
    # -----------------------------------------------------------

    # Per-element kernels (placement_strategy="linear_generic") need
    # their task count derived from scalar params, not tile_count.
    # Each maps kernel_name → callable(scalar_params) → int.
    _TASK_COUNT_RESOLVERS: dict[str, Any] = {
        "stabilize_reduce_grad_h": lambda s: (
            int(s["total_batch_count"]) * int(s["padded_hidden_count"])
        ),
        "normalize_gradients": lambda s: int(s["parameter_count"]),
        "adam_update": lambda s: int(s["parameter_count"]),
        "clamp_temperatures": lambda s: int(s["total_modules_count"]),
    }

    def _resolve_task_count(self, node: KernelDispatchNode) -> int:
        """Compute the actual task count for a KernelDispatchNode.

        For 'linear_generic' kernels each CPU task processes one element,
        so the task count must be derived from the kernel's scalar params.
        For all other placement strategies, tile_count is used directly.
        """
        resolver = self._TASK_COUNT_RESOLVERS.get(node.kernel_name)
        if resolver is not None:
            return resolver(node.scalar_params)
        return node.tile_count

    def _render_kernel_dispatch(
        self,
        node: KernelDispatchNode,
        allocator: CPUBufferAllocator,
    ) -> None:
        """Dispatch a KernelDispatchNode via pool_dispatch_and_wait."""
        fn_addr, struct_cls = self._dispatch_table[node.kernel_name]

        args = self._marshal_args(struct_cls, node, allocator)

        self._lib.pool_dispatch_and_wait(
            self._pool,
            fn_addr,
            ctypes.byref(args),
            self._resolve_task_count(node),
        )

    def _render_reduction_tree(
        self,
        node: ReductionTreeNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
    ) -> None:
        """Render a ReductionTreeNode via execute_reduction_tree."""
        rtp = node.reduction_plan

        # Build offset list and stage metadata arrays
        offsets = rtp.initial_offset_list
        fan_in = rtp.fan_in_K
        num_stages = rtp.num_stages

        # Flatten: for a simple fan-in tree, construct per-stage offset lists
        # The shared-layer ReductionTreePlan provides initial_offset_list
        offsets_array = (ctypes.c_uint32 * len(offsets))(*offsets)

        # Stage offsets: stage 0 starts at 0; single-stage tree
        stage_offsets = (ctypes.c_uint32 * num_stages)(
            *[i * fan_in for i in range(num_stages)]
        )
        stage_fan_in = (ctypes.c_uint32 * num_stages)(*([fan_in] * num_stages))

        # Node counts per stage
        n_partials = rtp.num_partials
        stage_counts: list[int] = []
        current_n = n_partials
        for _ in range(num_stages):
            nodes_at_stage = (current_n + fan_in - 1) // fan_in
            stage_counts.append(nodes_at_stage)
            current_n = nodes_at_stage
        stage_node_counts = (ctypes.c_uint32 * num_stages)(*stage_counts)

        # Allocate staging buffers (ping-pong)
        pw = rtp.partial_width
        max_intermediates: int = max(stage_counts) if stage_counts else 1
        staging_0 = np.zeros(max_intermediates * pw, dtype=np.float32)
        staging_1 = np.zeros(max_intermediates * pw, dtype=np.float32)

        # Build the C plan struct
        c_plan = ReductionTreePlanFFI()
        c_plan.partial_collection = ctypes.cast(
            allocator.get_data_pointer(rtp.source_buffer),
            ctypes.POINTER(ctypes.c_float),
        )
        c_plan.offset_lists_flat = ctypes.cast(
            offsets_array, ctypes.POINTER(ctypes.c_uint32)
        )
        c_plan.stage_offsets_into_list = ctypes.cast(
            stage_offsets, ctypes.POINTER(ctypes.c_uint32)
        )
        c_plan.stage_fan_in = ctypes.cast(
            stage_fan_in, ctypes.POINTER(ctypes.c_uint32)
        )
        c_plan.stage_node_counts = ctypes.cast(
            stage_node_counts, ctypes.POINTER(ctypes.c_uint32)
        )
        c_plan.staging_buffer_0 = staging_0.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        c_plan.staging_buffer_1 = staging_1.ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        )
        c_plan.output = ctypes.cast(
            allocator.get_data_pointer(rtp.destination_buffer),
            ctypes.POINTER(ctypes.c_float),
        )
        c_plan.partial_width = pw
        c_plan.num_stages = num_stages

        # Threshold schedule
        t_alg = rtp.threshold_schedule[0] if rtp.threshold_schedule else 0.0
        c_plan.t_algorithmic = t_alg if t_alg is not None else 0.0
        c_plan.lambda_ = (
            rtp.threshold_schedule[1]
            if len(rtp.threshold_schedule) > 1 and rtp.threshold_schedule[1] is not None
            else 0.0
        )
        c_plan.fp_max = plan.precision.fp_format_max
        c_plan.epsilon = plan.precision.epsilon

        self._lib.execute_reduction_tree(
            self._pool, ctypes.byref(c_plan)
        )

    def _render_streaming_loop(
        self,
        node: StreamingLoopNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
    ) -> None:
        """Render a StreamingLoopNode by iterating chunks."""
        sp = node.streaming_plan

        for chunk_index in range(sp.iteration.chunk_count):
            for body_node_id in sp.body:
                body_node = plan.nodes[body_node_id]
                if isinstance(body_node, KernelDispatchNode):
                    adjusted = self._apply_strides(
                        body_node, sp, chunk_index
                    )
                    self._render_kernel_dispatch(adjusted, allocator)

    # -----------------------------------------------------------
    # Argument marshalling
    # -----------------------------------------------------------

    _POINTER_TYPES = (c_float_p, c_uint_p, c_int_p, ctypes.c_void_p)

    def _marshal_args(
        self,
        struct_cls: type[ctypes.Structure],
        node: KernelDispatchNode,
        allocator: CPUBufferAllocator,
    ) -> ctypes.Structure:
        """Construct a ctypes argument struct from plan node parameters."""
        args = struct_cls()

        # Build a map of pointer field names -> field types for quick lookup
        pointer_fields: dict[str, type] = {}
        for field_name, field_type in struct_cls._fields_:  # type: ignore[reportAssignmentType]
            if field_type in self._POINTER_TYPES:
                pointer_fields[field_name] = field_type

        # Set buffer pointer fields from buffer_bindings
        for binding_name, handle in node.buffer_bindings.items():
            field_name = self._binding_to_field(binding_name)
            if field_name in pointer_fields:
                ptr = allocator.get_data_pointer(handle)
                setattr(args, field_name,
                        ctypes.cast(ptr, pointer_fields[field_name]))

        # Set scalar fields from scalar_params
        for param_name, value in node.scalar_params.items():
            field_name = self._param_to_field(param_name)
            if hasattr(args, field_name):
                setattr(args, field_name, value)

        return args

    @staticmethod
    def _binding_to_field(binding_name: str) -> str:
        """Map a buffer binding name to its struct field name.

        CONTRACT.md naming: src_buffer_GLOBAL_<field_name> or
        dest_buffer_GLOBAL_<field_name> or update_buffer_GLOBAL_<field_name>.
        Strip the prefix.
        """
        for prefix in (
            "src_buffer_GLOBAL_",
            "dest_buffer_GLOBAL_",
            "update_buffer_GLOBAL_",
            "src_buffer_GLOBAL_CONST_",
        ):
            if binding_name.startswith(prefix):
                return binding_name[len(prefix):]
        return binding_name

    @staticmethod
    def _param_to_field(param_name: str) -> str:
        """Map a scalar param name to its struct field name.

        CONTRACT.md naming: src_scalar_NATURAL_<field_name> or
        src_scalar_REAL_<field_name> or src_scalar_FLAG_<field_name> or
        dest_scalar_NATURAL_<field_name>.
        Strip the prefix.
        """
        for prefix in (
            "src_scalar_NATURAL_",
            "src_scalar_REAL_",
            "src_scalar_FLAG_",
            "dest_scalar_NATURAL_",
        ):
            if param_name.startswith(prefix):
                return param_name[len(prefix):]
        return param_name

    @staticmethod
    def _apply_strides(
        node: KernelDispatchNode,
        sp: Any,
        chunk_index: int,
    ) -> KernelDispatchNode:
        """Create a copy of a KernelDispatchNode with per-chunk strides applied."""
        adjusted_params = dict(node.scalar_params)
        for stride in sp.parameter_strides:
            adjusted_params[stride.param_name] = (
                stride.base + chunk_index * stride.stride
            )
        # Add constant scalars
        for name, value in sp.constant_scalars.items():
            adjusted_params[name] = value

        return replace(node, scalar_params=adjusted_params)
