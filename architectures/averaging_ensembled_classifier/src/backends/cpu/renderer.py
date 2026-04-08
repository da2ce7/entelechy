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
from ...shared.buffer_lifecycle import BufferRole
from ...shared.retrieval_future import RetrievalFuture
from ._dispatch_table import build_dispatch_table
from ._ffi_types import ALL_PRECISION_SUFFIXES, PRECISION_C_TYPES, PRECISION_STRUCTS
from ._loader import load_cpu_library
from .buffer_allocator import CPUBufferAllocator
from .discovery import detect_thread_count
from .retrieval import CPURetrievalFuture

from ...shared.precision_config import FP8_DTYPES, FP8_E4M3, FP8_E5M2
from ...shared.precision_suffix import precision_to_suffix

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

    Delegates to the shared ``precision_to_suffix`` implementation.
    """
    return precision_to_suffix(storage_dtype, compute_dtype, state_dtype)


class CPUPlanRenderer:
    """Plan renderer for the CPU backend (ADR-001, ADR-015).

    Traverses an ExecutionPlan DAG in topological order and dispatches
    each node via the CPU kernel library's pool_dispatch_and_wait.
    """

    def __init__(self, thread_count: int | None = None) -> None:
        self._lib = load_cpu_library()
        self._dispatch_tables = {
            suffix: build_dispatch_table(self._lib, suffix)
            for suffix in ALL_PRECISION_SUFFIXES
        }

        if thread_count is None:
            thread_count = detect_thread_count()
        self._pool = self._lib.pool_create(thread_count)
        self._thread_count = thread_count

        # Persistent MODEL_STATE buffer store (ADR-009).
        # Keyed by (logical_name, padded_shape, dtype_name) so that
        # MODEL_STATE buffers survive across render() calls, enabling
        # multi-batch training where parameter updates accumulate.
        self._persistent_buffers: dict[
            tuple[str, tuple[int, ...], str], np.ndarray
        ] = {}

    def __del__(self) -> None:
        if hasattr(self, "_pool") and self._pool:
            self._lib.pool_destroy(self._pool)
            self._pool = None

    # ---------------------------------------------------------------
    # Weight initialization (symmetry breaking for MODEL_STATE buffers)
    # ---------------------------------------------------------------
    # Names containing "weight" get Xavier uniform initialization;
    # names containing "temperature" are set to 1.0 (unit scaling);
    # biases, optimizer moments (m1_*, m2_*) stay zero.
    _WEIGHT_SUBSTRINGS = ("weight",)
    _UNIT_INIT_SUBSTRINGS = ("temperature",)

    @staticmethod
    def _buffer_rng(logical_name: str, padded_shape: tuple[int, ...]) -> np.random.Generator:
        """Deterministic per-buffer RNG seeded by buffer identity."""
        import hashlib
        h = hashlib.sha256(f"{logical_name}{padded_shape}".encode()).hexdigest()
        return np.random.default_rng(int(h[:16], 16))

    def _init_model_state(
        self, logical_name: str, buf: np.ndarray, padded_shape: tuple[int, ...],
        logical_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Apply role-appropriate initialization to MODEL_STATE buffers."""
        if any(s in logical_name for s in self._WEIGHT_SUBSTRINGS):
            # Xavier uniform: fan_in + fan_out from last two dims of shape
            # Prefer logical_shape (unpadded) when available for correct scaling
            shape_for_fan = logical_shape if logical_shape and len(logical_shape) >= 2 else padded_shape
            if len(shape_for_fan) >= 2:
                fan_in_plus_out = shape_for_fan[-2] + shape_for_fan[-1]
            else:
                fan_in_plus_out = max(shape_for_fan[0] if shape_for_fan else buf.size, 2)
            limit = float(np.sqrt(6.0 / fan_in_plus_out))
            rng = self._buffer_rng(logical_name, padded_shape)
            values = rng.uniform(-limit, limit, size=buf.shape)
            buf[:] = values.astype(buf.dtype)
        elif any(s in logical_name for s in self._UNIT_INIT_SUBSTRINGS):
            buf[:] = buf.dtype.type(1.0)

    def render(
        self,
        plan: ExecutionPlan,
        data_injections: dict[str, np.ndarray] | None = None,
    ) -> dict[str, RetrievalFuture]:
        """Render an execution plan via CPU dispatch.

        Allocates SIMD-aligned numpy buffers, traverses the plan's
        topological order, and dispatches each node type.

        MODEL_STATE buffers (ADR-009) are persisted across render() calls
        so that parameter updates from the Learn phase carry forward to
        subsequent Act phases (multi-batch training).  Weight-role
        MODEL_STATE buffers are Xavier-initialized on first allocation
        to break symmetry.

        Args:
            plan: The execution plan to render.
            data_injections: Optional mapping of buffer logical_name -> host data.
                Each entry is copied into the allocated plan buffer with the
                matching logical_name. The host array is padded to match the
                buffer's padded shape.
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
            key = (
                descriptor.logical_name,
                descriptor.padded_shape,
                allocator._role_dtypes[descriptor.precision_role].str,
            )
            if descriptor.role == BufferRole.MODEL_STATE and key in self._persistent_buffers:
                # Reuse the persistent buffer
                allocator.set_buffer(descriptor.handle, self._persistent_buffers[key])
            else:
                allocator.allocate(descriptor)
                if descriptor.role == BufferRole.MODEL_STATE:
                    buf = allocator.get_buffer(descriptor.handle)
                    self._init_model_state(
                        descriptor.logical_name, buf, descriptor.padded_shape,
                        logical_shape=descriptor.logical_shape)
                    self._persistent_buffers[key] = buf

        # --- Host-to-device data injection ---
        if data_injections:
            # Build logical_name -> descriptor mapping
            name_to_desc = {
                desc.logical_name: desc
                for desc in plan.buffers.values()
            }
            for logical_name, host_data in data_injections.items():
                desc = name_to_desc.get(logical_name)
                if desc is None:
                    continue
                buf = allocator.get_buffer(desc.handle)
                padded_shape = desc.padded_shape
                buf[:] = 0  # Zero-fill including padding regions
                if (
                    logical_name == "targets_cce"
                    and np.issubdtype(host_data.dtype, np.integer)
                    and not np.issubdtype(buf.dtype, np.integer)
                ):
                    # CCE class indices: raw byte copy preserves int32 bit
                    # pattern so the C kernel can read the buffer as int*.
                    src_bytes = host_data.astype(np.int32).tobytes()
                    dst_bytes = buf.view(np.uint8)
                    n = min(len(src_bytes), len(dst_bytes))
                    dst_bytes[:n] = np.frombuffer(src_bytes[:n], dtype=np.uint8)
                elif host_data.ndim <= 1 or len(padded_shape) <= 1:
                    if len(padded_shape) > 1 and host_data.ndim == 1:
                        # 1D host into 2D+ padded buffer: place in first
                        # column so strided kernel access is correct (e.g.
                        # BCE targets at [b * padded_class_dim + 0]).
                        reshaped = buf.reshape(padded_shape)
                        n = min(len(host_data), padded_shape[0])
                        reshaped[:n, 0] = host_data[:n].astype(buf.dtype)
                    else:
                        # 1D host data or 1D buffer: flat copy into leading
                        # elements (handles sample_mask, etc.)
                        flat = host_data.astype(buf.dtype).flatten()
                        n = min(len(flat), len(buf))
                        buf[:n] = flat[:n]
                else:
                    # Multi-dimensional: copy host region into top-left
                    # corner of padded buffer so padding stays zero.
                    host = host_data.astype(buf.dtype)
                    reshaped = buf.reshape(padded_shape)
                    slices = tuple(
                        slice(0, min(host.shape[d], padded_shape[d]))
                        for d in range(min(host.ndim, len(padded_shape)))
                    )
                    reshaped[slices] = host[slices]

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

    # Per-element kernels need their task count derived from scalar
    # params, not tile_count.  Each entry maps kernel_name to a
    # callable(scalar_params) → int that computes the task count.
    _TASK_COUNT_RESOLVERS: dict[str, Any] = {
        # Phase 1 — Act
        "forward_pass": lambda s: int(s["batch_chunk_count"]),
        "render_logits_chunk": lambda s: (
            int(s["module_chunk_count"])
            * int(s["batch_chunk_count"])
            * int(s["class_chunk_count"])
        ),
        # Phase 2A — Production
        "compute_probs_loss_cce_chunk": lambda s: (
            int(s["modules_per_chunk"]) * int(s["total_batch_count"])
        ),
        "compute_probs_loss_bce_chunk": lambda s: (
            int(s["modules_per_chunk"]) * int(s["total_batch_count"])
        ),
        "calculate_module_param_grads_chunk": lambda s: (
            int(s["modules_per_chunk"]) * int(s["hidden_count"])
        ),
        # Phase 2B — Processing
        "backprop_error_to_hidden_chunk": lambda s: (
            int(s["modules_per_chunk"])
            * int(s["total_batch_count"])
            * int(s["padded_hidden_count"])
        ),
        "calculate_chunk_temp_gradients": lambda s: int(s["modules_per_chunk"]),
        # clip_partial_gradients: ignores task_index → tile_count=1 is correct
        # Phase 2C — Reduction
        "gather_and_permute_grad_hidden_activations": lambda s: (
            int(s["total_batch_count"])
            * int(s["padded_hidden_count"])
            * int(s["total_modules_count"])
        ),
        "stabilize_and_reduce_grad_hidden_activations": lambda s: (
            int(s["total_batch_count"]) * int(s["padded_hidden_count"])
        ),
        # clip_intermediate_grad: ignores task_index → tile_count=1 is correct
        # Phase 2D — Backprop
        "backprop_shared_weights_chunk": lambda s: (
            int(s["padded_input_count"]) * int(s["padded_hidden_count"])
        ),
        "backprop_shared_biases_chunk": lambda s: int(s["padded_hidden_count"]),
        # clip_shared_gradients_chunk: ignores task_index → tile_count=1 is correct
        # Phase 3 — Update
        "normalize_gradients": lambda s: int(s["parameter_count"]),
        "adam_update": lambda s: int(s["parameter_count"]),
        "clamp_temperatures": lambda s: int(s["parameter_count"]),
    }

    def _resolve_task_count(self, node: KernelDispatchNode) -> int:
        """Compute the actual task count for a KernelDispatchNode.

        Each kernel has a specific task decomposition: some dispatch one
        task per sample, others per element, others once for the entire
        buffer.  The resolvers map scalar params to the correct count.
        Kernels not listed fall through to tile_count.
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
        """Render a ReductionTreeNode via execute_reduction_tree.

        ADR-026: Selects between storage-entry and compute-entry variants
        based on the source buffer's precision_role.
        """
        _, c_storage_p, _, c_compute_p, _, _ = PRECISION_C_TYPES[suffix]

        rtp = node.reduction_plan

        # ADR-026 §3: Select variant based on source buffer's precision_role
        source_desc = plan.buffers[rtp.source_buffer]
        use_compute_entry = source_desc.precision_role == "compute"

        if use_compute_entry:
            PlanStruct = PRECISION_STRUCTS[suffix]["ReductionTreePlanComputeEntryFFI"]
            c_collection_p = c_compute_p
            staging_dtype = plan.precision.compute_dtype
            fn_name = f"execute_reduction_tree_from_compute_{suffix}"
        else:
            PlanStruct = PRECISION_STRUCTS[suffix]["ReductionTreePlanFFI"]
            c_collection_p = c_storage_p
            staging_dtype = plan.precision.storage_dtype
            fn_name = f"execute_reduction_tree_{suffix}"

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

        # Allocate staging buffers (ping-pong) — dtype matches variant
        max_intermediates: int = max(stage_counts) if stage_counts else 1
        staging_0 = np.zeros(max_intermediates * pw, dtype=staging_dtype)
        staging_1 = np.zeros(max_intermediates * pw, dtype=staging_dtype)

        # Build the C plan struct
        c_plan = PlanStruct()
        c_plan.partial_collection = ctypes.cast(
            allocator.get_data_pointer(rtp.source_buffer),
            c_collection_p,
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
        c_plan.staging_buffer_0 = staging_0.ctypes.data_as(c_collection_p)
        c_plan.staging_buffer_1 = staging_1.ctypes.data_as(c_collection_p)
        c_plan.output = ctypes.cast(
            allocator.get_data_pointer(rtp.destination_buffer),
            c_compute_p,
        )
        c_plan.partial_width = pw
        c_plan.num_stages = num_stages

        # Threshold schedule — convert to compute-type representation.
        # None entries mean "no clipping" → use fp_max as a passthrough
        # threshold so the clipping branch is never triggered.
        fp_max = plan.precision.compute_fp_format_max
        t_alg = rtp.threshold_schedule[0] if rtp.threshold_schedule else None
        c_plan.t_algorithmic = _to_compute_scalar(
            t_alg if t_alg is not None else fp_max, suffix
        )
        lambda_val = (
            rtp.threshold_schedule[1]
            if len(rtp.threshold_schedule) > 1 and rtp.threshold_schedule[1] is not None
            else 0.0
        )
        c_plan.lambda_ = _to_compute_scalar(lambda_val, suffix)
        c_plan.fp_max = _to_compute_scalar(fp_max, suffix)
        c_plan.epsilon = _to_compute_scalar(plan.precision.compute_epsilon, suffix)

        getattr(self._lib, fn_name)(
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
