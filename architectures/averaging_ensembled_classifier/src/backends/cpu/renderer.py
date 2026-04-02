"""CPUPlanRenderer — PlanRenderer implementation for the CPU backend (ADR-001).

Traverses an ExecutionPlan DAG in topological order and dispatches each
node via the CPU kernel library's pool_dispatch_and_wait.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, c_int32, c_uint32, c_void_p
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
from ._ffi_types import PRECISION_C_TYPES, PRECISION_STRUCTS, PRECISION_SUFFIXES
from ._loader import load_cpu_library
from .buffer_allocator import CPUBufferAllocator
from .discovery import detect_thread_count
from .retrieval import CPURetrievalFuture

c_uint_p = POINTER(c_uint32)
c_int_p = POINTER(c_int32)


def _to_compute_scalar(value: float, suffix: str) -> float | int:
    """Convert a Python float to the appropriate scalar for a compute-type field.

    For FP16 compute variants (s16c16*), ctypes uses c_uint16 because there
    is no native c_float16. We convert the float to its IEEE 754 binary16
    representation as an integer. For FP32/FP64 compute variants, return
    the float unchanged.
    """
    # FP16 compute suffixes start with s16c16 or s32c16 (hypothetically)
    # In practice, only s16c16* exists — check the middle part (compute axis).
    compute_axis = suffix.split("c")[1].split("x")[0]  # e.g., "16" from "s16c16x32"
    if compute_axis == "16":
        # Convert to FP16 bits: float -> np.float16 -> view as uint16
        return int(np.float16(value).view(np.uint16))
    return value


def _get_precision_suffix(
    storage_dtype: np.dtype,
    compute_dtype: np.dtype,
    state_dtype: np.dtype,
) -> str:
    """Map three-axis precision configuration to kernel suffix (ADR-024 §4.1).

    Returns one of the 11 valid s{s}c{c}x{x} suffixes.
    """
    _SUFFIX_MAP: dict[tuple[type, type, type], str] = {
        (np.float16, np.float16, np.float16): "s16c16x16",
        (np.float16, np.float16, np.float32): "s16c16x32",
        (np.float16, np.float16, np.float64): "s16c16x64",
        (np.float16, np.float32, np.float16): "s16c32x16",
        (np.float16, np.float32, np.float32): "s16c32x32",
        (np.float16, np.float32, np.float64): "s16c32x64",
        (np.float16, np.float64, np.float16): "s16c64x16",
        (np.float16, np.float64, np.float32): "s16c64x32",
        (np.float16, np.float64, np.float64): "s16c64x64",
        (np.float32, np.float32, np.float32): "s32c32x32",
        (np.float32, np.float32, np.float64): "s32c32x64",
        (np.float32, np.float64, np.float32): "s32c64x32",
        (np.float32, np.float64, np.float64): "s32c64x64",
        (np.float64, np.float64, np.float64): "s64c64x64",
    }
    key = (storage_dtype.type, compute_dtype.type, state_dtype.type)
    suffix = _SUFFIX_MAP.get(key)
    if suffix is None:
        raise ValueError(
            f"No CPU kernel instantiation for "
            f"storage={storage_dtype}, compute={compute_dtype}, "
            f"state={state_dtype}"
        )
    return suffix


class CPUPlanRenderer:
    """Plan renderer for the CPU backend (ADR-001, ADR-015).

    Traverses an ExecutionPlan DAG in topological order and dispatches
    each node via the CPU kernel library's pool_dispatch_and_wait.
    """

    def __init__(self, thread_count: int | None = None) -> None:
        self._lib = load_cpu_library()
        self._dispatch_tables = {
            suffix: build_dispatch_table(self._lib, suffix)
            for suffix in PRECISION_SUFFIXES
        }

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
        suffix = _get_precision_suffix(
            plan.precision.storage_dtype,
            plan.precision.compute_dtype,
            plan.precision.state_dtype,
        )
        allocator = CPUBufferAllocator(
            simd_alignment=plan.hardware.cache_line_bytes,
            role_dtypes={
                "storage": plan.precision.storage_dtype,
                "compute": plan.precision.compute_dtype,
                "state": plan.precision.state_dtype,
            },
        )
        for descriptor in plan.buffers.values():
            allocator.allocate(descriptor)

        futures: dict[str, RetrievalFuture] = {}

        for node_id in plan.topological_order:
            node = plan.nodes[node_id]

            if isinstance(node, KernelDispatchNode):
                self._render_kernel_dispatch(node, allocator, suffix)
            elif isinstance(node, ReductionTreeNode):
                self._render_reduction_tree(node, allocator, plan, suffix)
            elif isinstance(node, StreamingLoopNode):
                self._render_streaming_loop(node, allocator, plan, suffix)
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
        "stabilize_and_reduce_grad_hidden_activations": lambda s: (
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
        suffix: str,
    ) -> None:
        """Dispatch a KernelDispatchNode via pool_dispatch_and_wait."""
        fn_addr, struct_cls = self._dispatch_tables[suffix][node.kernel_name]

        args = self._marshal_args(struct_cls, node, allocator, suffix)

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
        suffix: str,
    ) -> None:
        """Render a ReductionTreeNode via execute_reduction_tree."""
        _, c_storage_p, _, c_compute_p, _, _ = PRECISION_C_TYPES[suffix]
        ReductionTreePlanFFI = PRECISION_STRUCTS[suffix]["ReductionTreePlanFFI"]

        rtp = node.reduction_plan

        offsets = rtp.initial_offset_list
        fan_in = rtp.fan_in_K
        num_stages = rtp.num_stages
        pw = rtp.partial_width
        SENTINEL = 0xFFFFFFFF

        # Compute node counts per stage
        n_partials = rtp.num_partials
        stage_counts: list[int] = []
        current_n = n_partials
        for _ in range(num_stages):
            nodes_at_stage = (current_n + fan_in - 1) // fan_in
            stage_counts.append(nodes_at_stage)
            current_n = nodes_at_stage

        # Build the complete flat offset list for all stages.
        # Each stage s needs stage_counts[s] * fan_in entries.
        # Stage 0: initial_offset_list, padded with sentinels.
        # Stages 1+: sequential offsets into the staging buffer.
        flat_offsets: list[int] = []
        stage_offset_values: list[int] = []

        for s in range(num_stages):
            stage_offset_values.append(len(flat_offsets))
            entries_needed = stage_counts[s] * fan_in
            if s == 0:
                # Use initial offsets, pad remainder with sentinel
                flat_offsets.extend(offsets)
                flat_offsets.extend([SENTINEL] * (entries_needed - len(offsets)))
            else:
                # Previous stage wrote contiguously: offset i -> i * pw
                prev_count = stage_counts[s - 1]
                for i in range(prev_count):
                    flat_offsets.append(i * pw)
                flat_offsets.extend([SENTINEL] * (entries_needed - prev_count))

        offsets_array = (ctypes.c_uint32 * len(flat_offsets))(*flat_offsets)
        stage_offsets = (ctypes.c_uint32 * num_stages)(*stage_offset_values)
        stage_fan_in = (ctypes.c_uint32 * num_stages)(*([fan_in] * num_stages))
        stage_node_counts = (ctypes.c_uint32 * num_stages)(*stage_counts)

        # Allocate staging buffers (ping-pong)
        max_intermediates: int = max(stage_counts) if stage_counts else 1
        staging_0 = np.zeros(max_intermediates * pw, dtype=plan.precision.storage_dtype)
        staging_1 = np.zeros(max_intermediates * pw, dtype=plan.precision.storage_dtype)

        # Build the C plan struct
        c_plan = ReductionTreePlanFFI()
        c_plan.partial_collection = ctypes.cast(
            allocator.get_data_pointer(rtp.source_buffer),
            c_storage_p,
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
        c_plan.staging_buffer_0 = staging_0.ctypes.data_as(c_storage_p)
        c_plan.staging_buffer_1 = staging_1.ctypes.data_as(c_storage_p)
        c_plan.output = ctypes.cast(
            allocator.get_data_pointer(rtp.destination_buffer),
            c_compute_p,
        )
        c_plan.partial_width = pw
        c_plan.num_stages = num_stages

        # Threshold schedule — convert to compute-type representation
        t_alg = rtp.threshold_schedule[0] if rtp.threshold_schedule else 0.0
        c_plan.t_algorithmic = _to_compute_scalar(
            t_alg if t_alg is not None else 0.0, suffix
        )
        lambda_val = (
            rtp.threshold_schedule[1]
            if len(rtp.threshold_schedule) > 1 and rtp.threshold_schedule[1] is not None
            else 0.0
        )
        c_plan.lambda_ = _to_compute_scalar(lambda_val, suffix)
        c_plan.fp_max = _to_compute_scalar(plan.precision.compute_fp_format_max, suffix)
        c_plan.epsilon = _to_compute_scalar(plan.precision.compute_epsilon, suffix)

        getattr(self._lib, f"execute_reduction_tree_{suffix}")(
            self._pool, ctypes.byref(c_plan)
        )

    def _render_streaming_loop(
        self,
        node: StreamingLoopNode,
        allocator: CPUBufferAllocator,
        plan: ExecutionPlan,
        suffix: str,
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
                    self._render_kernel_dispatch(adjusted, allocator, suffix)

    # -----------------------------------------------------------
    # Argument marshalling
    # -----------------------------------------------------------

    _POINTER_TYPES = (
        *(ct[1] for ct in PRECISION_C_TYPES.values()),  # storage pointers
        *(ct[3] for ct in PRECISION_C_TYPES.values()),  # compute pointers
        *(ct[5] for ct in PRECISION_C_TYPES.values()),  # state pointers
        c_uint_p, c_int_p, c_void_p,
    )

    def _marshal_args(
        self,
        struct_cls: type[ctypes.Structure],
        node: KernelDispatchNode,
        allocator: CPUBufferAllocator,
        suffix: str,
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
        # For FP16 compute variants, float scalars that map to c_uint16 fields
        # (i.e., c_compute for FP16) must be converted to FP16 binary representation.
        # We detect these by inspecting the struct field type.
        field_type_map = {name: ftype for name, ftype in struct_cls._fields_}  # type: ignore
        for param_name, value in node.scalar_params.items():
            field_name = self._param_to_field(param_name)
            if hasattr(args, field_name):
                # If value is float and field is c_uint16 (FP16 binary representation),
                # convert the float to FP16 bits
                if isinstance(value, float):
                    ftype = field_type_map.get(field_name)
                    if ftype is ctypes.c_uint16:
                        value = _to_compute_scalar(value, suffix)
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
