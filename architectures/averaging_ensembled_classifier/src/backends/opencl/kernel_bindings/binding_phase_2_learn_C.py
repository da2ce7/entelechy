# src/backends/opencl/kernel_bindings/binding_phase_2_learn_C.py
"""KernelBinding adapters for Learn-C reduction & aggregation kernels.

Covers (in DAG / kernel-source order):

  - Nodes 14, 15a, 20a: Tiered aggregate reduction (register / local,
    storage-entry and ADR-026 compute-entry variants)
  - Nodes 15b, 20b: ``clip_intermediate_grad``
  - Node 16: ``stabilize_and_reduce_grad_hidden_activations``
  - ADR-019: K-fan-in reduction primitives (storage-entry and ADR-026
    compute-entry variants)

Each binding translates the backend-neutral plan node's buffer bindings and
scalar parameters into the concrete argument list required by
``clEnqueueNDRangeKernel``, including local-memory allocations, dispatch
geometry, and tile-index delivery (CONTRACT.md §7).

Bindings in this module additionally expose ``marshal_args_direct`` /
``compute_grid_direct`` convenience methods for the Orchestration tier's
reduction-tree renderer, which composes these kernels programmatically
rather than through plan nodes.

Reference specification: kernels.cl.h (ADR-013 designation).
Dispatch geometry source: phase_2_learn_C_reduction.cl.c implementation comments.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ....shared.stabilization_policy import render_node16_threshold_schedule
from ..type_mapping import compute_scalar
from .base import KernelBinding


# ---------------------------------------------------------------------------
# Nodes 14, 15a, 20a — aggregate_register_reduce (Tier 1, storage-entry)
# ---------------------------------------------------------------------------


class AggregateRegisterReduceBinding(KernelBinding):
    """Binding for ``aggregate_register_reduce`` (Tier 1 reduction engine).

    Dispatch geometry::

        global = (partial_width)
        local  = backend-selected (None)

    Each work-item computes one element of the output tensor by serially
    accumulating all N partials for that element in private registers — no
    local memory, no synchronization.  Selected by the Orchestration tier
    when the number of partials (N) is small enough for register-only
    accumulation.

    Precision Boundary Conversion: storage-role inputs widened via
    ``load_storage()``; compute-role output written directly in
    COMPUTE_TYPE.
    """

    def get_kernel_name(self) -> str:
        return "aggregate_register_reduce"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Reduction — no tiling.
        return (
            int(scalar_params["src_scalar_NATURAL_partial_width"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Reduction — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )

        return [
            buf("src_buffer_GLOBAL_partial_collection"),
            buf("src_buffer_GLOBAL_CONST_partial_offset_list"),
            buf("dest_buffer_GLOBAL_partial"),
            u32("src_scalar_NATURAL_partial_offset_list_count"),
            u32("src_scalar_NATURAL_partial_width"),
            u32("src_scalar_FLAG_operation_type"),
        ]

    # -- Direct-use interface for reduction-tree renderer -------------------

    def marshal_args_direct(
        self,
        source: cl.Buffer,
        offset_list: cl.Buffer,
        dest: cl.Buffer,
        offset_count: int,
        partial_width: int,
        operation_type: int,
    ) -> list[Any]:
        """Marshal arguments for direct invocation by the reduction renderer.

        Bypasses the plan-driven ``buffer_bindings`` / ``scalar_params``
        indirection for performance-critical reduction-tree composition.
        """
        return [
            source,
            offset_list,
            dest,
            np.uint32(offset_count),
            np.uint32(partial_width),
            np.uint32(operation_type),
        ]

    def compute_grid_direct(
        self,
        partial_width: int,
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Compute dispatch geometry for direct invocation."""
        _ = hardware_simd_width
        return (partial_width,), None


# ---------------------------------------------------------------------------
# Nodes 14, 15a, 20a — aggregate_local_reduce (Tier 2, storage-entry)
# ---------------------------------------------------------------------------


class AggregateLocalReduceBinding(KernelBinding):
    """Binding for ``aggregate_local_reduce`` (Tier 2 reduction engine).

    Dispatch geometry::

        global = (partial_width × work_group_size)
        local  = (work_group_size)

    A full work-group collaborates to reduce all N partials for a single
    output element.  Threads first accumulate strided slices into private
    registers, then perform a fast tree reduction in ``__local`` memory.
    One work-group per output element (``get_group_id(0)`` → element
    index).  Selected by the Orchestration tier when the number of partials
    (N) is large enough to benefit from parallel reduction.

    Precision Boundary Conversion: storage-role inputs widened via
    ``load_storage()``; compute-role output written directly in
    COMPUTE_TYPE.

    Parameters
    ----------
    workgroup_size : int
        Work-group size for the parallel reduction tree.  Must be a power
        of two (guaranteed by Orchestration-tier dispatch selection).
    """

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "aggregate_local_reduce"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Reduction — no tiling.
        partial_width = int(
            scalar_params["src_scalar_NATURAL_partial_width"]
        )
        wg = self._workgroup_size
        return (partial_width * wg,), (wg,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Reduction — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = self._workgroup_size * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_partial_collection"),
            buf("src_buffer_GLOBAL_CONST_partial_offset_list"),
            buf("dest_buffer_GLOBAL_partial"),
            u32("src_scalar_NATURAL_partial_offset_list_count"),
            u32("src_scalar_NATURAL_partial_width"),
            u32("src_scalar_FLAG_operation_type"),
        ]

    # -- Direct-use interface for reduction-tree renderer -------------------

    def marshal_args_direct(
        self,
        source: cl.Buffer,
        offset_list: cl.Buffer,
        dest: cl.Buffer,
        offset_count: int,
        partial_width: int,
        operation_type: int,
        compute_type_size_bytes: int = 4,
    ) -> list[Any]:
        """Marshal arguments for direct invocation by the reduction renderer.

        Bypasses the plan-driven ``buffer_bindings`` / ``scalar_params``
        indirection for performance-critical reduction-tree composition.
        """
        local_bytes = self._workgroup_size * compute_type_size_bytes
        return [
            cl.LocalMemory(local_bytes),
            source,
            offset_list,
            dest,
            np.uint32(offset_count),
            np.uint32(partial_width),
            np.uint32(operation_type),
        ]

    def compute_grid_direct(
        self,
        partial_width: int,
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Compute dispatch geometry for direct invocation."""
        _ = hardware_simd_width
        wg = self._workgroup_size
        return (partial_width * wg,), (wg,)


# ---------------------------------------------------------------------------
# ADR-026: aggregate_register_reduce_from_compute (Tier 1, compute-entry)
# ---------------------------------------------------------------------------


class AggregateRegisterReduceFromComputeBinding(
    AggregateRegisterReduceBinding,
):
    """Compute-entry variant of ``aggregate_register_reduce`` (ADR-026).

    Identical dispatch geometry and argument marshaling; only the kernel
    name differs.  Used when the source buffer's ``precision_role`` is
    ``"compute"`` (e.g., BCE loss partials, or interior stages of
    multi-stage reduction trees).

    All buffers are compute-role; no precision boundary conversion is
    required.  When ``STORAGE_TYPE == COMPUTE_TYPE``, both variants
    compile to identical machine code.
    """

    def get_kernel_name(self) -> str:
        return "aggregate_register_reduce_from_compute"


# ---------------------------------------------------------------------------
# ADR-026: aggregate_local_reduce_from_compute (Tier 2, compute-entry)
# ---------------------------------------------------------------------------


class AggregateLocalReduceFromComputeBinding(AggregateLocalReduceBinding):
    """Compute-entry variant of ``aggregate_local_reduce`` (ADR-026).

    Identical dispatch geometry and argument marshaling; only the kernel
    name differs.  Used when the source buffer's ``precision_role`` is
    ``"compute"``.

    All buffers are compute-role; no precision boundary conversion is
    required.  When ``STORAGE_TYPE == COMPUTE_TYPE``, both variants
    compile to identical machine code.
    """

    def get_kernel_name(self) -> str:
        return "aggregate_local_reduce_from_compute"


# ---------------------------------------------------------------------------
# Nodes 15b, 20b — clip_intermediate_grad
# ---------------------------------------------------------------------------


class ClipIntermediateGradBinding(KernelBinding):
    """Binding for ``clip_intermediate_grad`` (inter-stage clipping utility).

    Dispatch geometry::

        global = (work_group_size)
        local  = (work_group_size)

    A single work-group computes the L2 norm of a contiguous intermediate
    gradient buffer (the output of a preceding ``aggregate_*`` kernel) and
    conditionally scales it in-place.  An early-exit path bypasses the
    second global memory pass when the norm is within the threshold.  The
    kernel strides across the full ``parameter_count`` with
    ``get_local_size(0)``-stride loops — a single work-group handles
    arbitrarily large buffers.

    All buffers are compute-role; no precision boundary conversion is
    required.

    Parameters
    ----------
    workgroup_size : int
        Work-group size for the parallel L2-norm reduction tree.  Must be
        a power of two (guaranteed by Orchestration-tier dispatch selection).
    """

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "clip_intermediate_grad"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, scalar_params, hardware_simd_width
        # Single work-group dispatch: the kernel strides across the full
        # parameter_count with get_local_size(0) increments.
        wg = self._workgroup_size
        return (wg,), (wg,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Utility — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = self._workgroup_size * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("update_buffer_GLOBAL_intermediate_grad"),
            real("src_scalar_REAL_clipping_threshold_t_j"),
            real("src_scalar_REAL_epsilon"),
            u32("src_scalar_NATURAL_parameter_count"),
        ]

    # -- Direct-use interface for reduction-tree renderer -------------------

    def marshal_args_direct(
        self,
        buffer: cl.Buffer,
        threshold: float,
        epsilon: float,
        parameter_count: int,
        compute_type_size_bytes: int = 4,
        compute_dtype: np.dtype | None = None,
    ) -> list[Any]:
        """Marshal arguments for direct invocation by the reduction renderer.

        Bypasses the plan-driven ``buffer_bindings`` / ``scalar_params``
        indirection for performance-critical reduction-tree composition.
        """
        local_bytes = self._workgroup_size * compute_type_size_bytes
        dtype = compute_dtype if compute_dtype is not None else np.float32
        return [
            cl.LocalMemory(local_bytes),
            buffer,
            np.dtype(dtype).type(threshold),
            np.dtype(dtype).type(epsilon),
            np.uint32(parameter_count),
        ]

    def compute_grid_direct(
        self,
        parameter_count: int,
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Compute dispatch geometry for direct invocation."""
        _ = parameter_count, hardware_simd_width
        wg = self._workgroup_size
        return (wg,), (wg,)


# ---------------------------------------------------------------------------
# Precision Bridge — narrow_to_storage
# ---------------------------------------------------------------------------


class NarrowToStorageBinding(KernelBinding):
    """Binding for ``narrow_to_storage`` (Precision Bridge utility).

    Dispatch geometry::

        global = (element_count)
        local  = backend-selected (None)

    Embarrassingly parallel element-wise format conversion.  Each work-item
    narrows one COMPUTE_TYPE element to STORAGE_TYPE via store_storage().
    Elided entirely by the Orchestration tier when storage_dtype ==
    compute_dtype — the source buffer is bit-compatible with the consumer.
    """

    def get_kernel_name(self) -> str:
        return "narrow_to_storage"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width
        return (int(scalar_params["src_scalar_NATURAL_element_count"]),), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        return [
            buf("src_buffer_GLOBAL_input"),
            buf("dest_buffer_GLOBAL_output"),
            np.uint32(scalar_params["src_scalar_NATURAL_element_count"]),
        ]


# ---------------------------------------------------------------------------
# Node 16 — stabilize_and_reduce_grad_hidden_activations
# ---------------------------------------------------------------------------


class StabilizeReduceGradHBinding(KernelBinding):
    """Binding for ``stabilize_and_reduce_grad_hidden_activations`` (Node 16).

    Dispatch geometry::

        global = (total_rows × work_group_size)
                 where total_rows = total_batch_count × padded_hidden_count
        local  = (work_group_size)

    Specialized "work-group per row" reduction engine.  Each work-group
    reduces one row of the contiguous SoA buffer produced by the upstream
    Item Synchronization Point (Node 13), applying a host-prescribed
    stabilization schedule (CONCEPT.md §11, Execution-Tier Internal
    Amplification).

    The Orchestration tier renders the threshold schedule via
    ``render_node16_threshold_schedule()`` in ``prepare_dispatch()``,
    accounting for the GPU backend's workgroup-size-limited
    pre-accumulation.  The rendered schedule is uploaded to the per-stage
    threshold buffer as a documented side effect of ``prepare_dispatch``.

    In addition to the canonical kernel parameters (keyed by their
    ``src_scalar_*`` / ``src_buffer_*`` contract names), ``scalar_params``
    must carry the following Orchestration-tier metadata (underscore-
    prefixed, NOT kernel parameters):

    * ``_policy_t_algorithmic`` — Quadratic Scaling Policy anchor
    * ``_policy_lambda`` — Quadratic Scaling Policy curvature
    * ``_policy_max_k`` — maximum fan-in K for staged reduction
    * ``_compute_fp_format_max`` — COMPUTE_TYPE maximum finite value
    * ``_compute_type_size_bytes`` — sizeof(COMPUTE_TYPE)
    * ``_compute_dtype`` — numpy dtype for COMPUTE_TYPE scalars

    Parameters
    ----------
    workgroup_size : int
        Work-group size for the staged reduction tree.  Must be a power
        of two (guaranteed by Orchestration-tier dispatch selection).
    """

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "stabilize_and_reduce_grad_hidden_activations"

    def prepare_dispatch(
        self,
        queue: cl.CommandQueue,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> dict[str, Any]:
        """Render and upload the host-prescribed threshold schedule.

        Side effect: enqueues a single H2D copy of the rendered schedule
        into the ``_per_stage`` threshold buffer.  This is the sole
        device write in the binding lifecycle for this kernel.

        The rendered schedule depends on ``self._workgroup_size``, which
        is a binding-instance property unknown to the Policy tier.  This
        is the canonical use case for ``prepare_dispatch()``:
        Orchestration-tier adaptation of Policy-tier parameters to the
        backend's dispatch topology.
        """
        _ = tile_index  # Global Barrier — no tile placement.
        total_modules = int(
            scalar_params["src_scalar_NATURAL_total_modules_count"]
        )
        num_stages, t_pre, schedule = render_node16_threshold_schedule(
            t_algorithmic=float(scalar_params["_policy_t_algorithmic"]),
            lambda_=float(scalar_params["_policy_lambda"]),
            compute_fp_format_max=float(
                scalar_params["_compute_fp_format_max"]
            ),
            total_modules=total_modules,
            workgroup_size=self._workgroup_size,
            max_fan_in=int(scalar_params["_policy_max_k"]),
        )

        # Upload rendered schedule to device buffer.
        if schedule:
            compute_dtype: np.dtype[Any] = np.dtype(
                scalar_params.get("_compute_dtype", np.float32)
            )
            schedule_np = np.array(schedule, dtype=compute_dtype)
            schedule_buf = get_buffer(
                buffer_bindings[
                    "src_buffer_GLOBAL_CONST_clipping_threshold_per_stage"
                ]
            )
            cl.enqueue_copy(queue, schedule_buf, schedule_np)

        # Augment scalar_params with binding-computed values.
        return {
            **scalar_params,
            "_prepared_num_reduction_stages": num_stages,
            "_prepared_clipping_threshold_t_pre": t_pre,
        }

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Global Barrier — no tiling.
        total_batch = int(
            scalar_params["src_scalar_NATURAL_total_batch_count"]
        )
        padded_hidden = int(
            scalar_params["src_scalar_NATURAL_padded_hidden_count"]
        )
        wg = self._workgroup_size
        num_rows = total_batch * padded_hidden
        return (num_rows * wg,), (wg,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        """Pure argument translation — reads _prepared_* keys, no side effects."""
        _ = tile_index  # Global Barrier — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = self._workgroup_size * compute_bytes

        # Read binding-computed values from prepare_dispatch().
        num_stages = int(scalar_params["_prepared_num_reduction_stages"])
        t_pre = scalar_params["_prepared_clipping_threshold_t_pre"]

        return [
            cl.LocalMemory(local_bytes),
            buf(
                "src_buffer_GLOBAL_clipped_grad_hidden_activations"
                "_permuted_soa"
            ),
            buf("dest_buffer_GLOBAL_summed_grad_hidden_activations"),
            buf("src_buffer_GLOBAL_CONST_clipping_threshold_per_stage"),
            np.uint32(num_stages),
            compute_scalar(t_pre, scalar_params),
            compute_scalar(
                scalar_params["src_scalar_REAL_epsilon"], scalar_params
            ),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
            u32("src_scalar_NATURAL_padded_total_modules_count"),
        ]


# ---------------------------------------------------------------------------
# ADR-019: reduce_k_fan_in_and_clip (storage-entry)
# ---------------------------------------------------------------------------


class ReduceKFanInAndClipBinding(KernelBinding):
    """Binding for ``reduce_k_fan_in_and_clip`` (ADR-019 K-fan-in primitive).

    Dispatch geometry::

        global = (node_count × work_group_size)
        local  = (work_group_size)

    "One work-group per reduction node" fused reduce-and-clip.  Each
    work-group processes a single reduction node: element-parallel gather
    across K input partials, simultaneous sum-of-squares tracking for the
    L2 norm, and conditional uniform scaling.  Fusing summation and clipping
    into a single kernel eliminates intermediate global memory traffic.

    Clipping behavior:

    * ``threshold >= 0``: per-node L2 clip applied.
    * ``threshold < 0``: clip bypassed (diagnostic mode).
    * ``threshold == 0``: clips to zero norm (zeroes all gradients).

    Precision Boundary Conversion: storage-role partials widened via
    ``load_storage()``; compute-role output written directly in
    COMPUTE_TYPE.

    Parameters
    ----------
    workgroup_size : int
        Work-group size for the element-parallel gather and L2-norm
        reduction.  Must be a power of two.
    """

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "reduce_k_fan_in_and_clip"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, Any],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Reduction — no tiling.
        node_count = int(
            scalar_params["src_scalar_NATURAL_node_count"]
        )
        wg = self._workgroup_size
        return (node_count * wg,), (wg,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, Any],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Reduction — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = self._workgroup_size * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_partial_collection"),
            buf("src_buffer_GLOBAL_CONST_offset_list_flat"),
            buf("dest_buffer_GLOBAL_stage_partial"),
            u32("src_scalar_NATURAL_fan_in"),
            u32("src_scalar_NATURAL_node_count"),
            u32("src_scalar_NATURAL_partial_width"),
            real("src_scalar_REAL_clipping_threshold_t_j"),
            real("src_scalar_REAL_epsilon"),
        ]

    # -- Direct-use interface for reduction-tree renderer -------------------

    def marshal_args_direct(
        self,
        source: cl.Buffer,
        offset_list: cl.Buffer,
        dest: cl.Buffer,
        fan_in: int,
        node_count: int,
        partial_width: int,
        clipping_threshold: float,
        epsilon: float,
        compute_type_size_bytes: int = 4,
        compute_dtype: np.dtype | None = None,
    ) -> list[Any]:
        """Marshal arguments for direct invocation by the reduction renderer.

        Bypasses the plan-driven ``buffer_bindings`` / ``scalar_params``
        indirection for performance-critical reduction-tree composition.
        """
        local_bytes = self._workgroup_size * compute_type_size_bytes
        dtype = compute_dtype if compute_dtype is not None else np.float32
        return [
            cl.LocalMemory(local_bytes),
            source,
            offset_list,
            dest,
            np.uint32(fan_in),
            np.uint32(node_count),
            np.uint32(partial_width),
            np.dtype(dtype).type(clipping_threshold),
            np.dtype(dtype).type(epsilon),
        ]

    def compute_grid_direct(
        self,
        node_count: int,
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        """Compute dispatch geometry for direct invocation."""
        _ = hardware_simd_width
        wg = self._workgroup_size
        return (node_count * wg,), (wg,)


# ---------------------------------------------------------------------------
# ADR-019/026: reduce_k_fan_in_and_clip_from_compute (compute-entry)
# ---------------------------------------------------------------------------


class ReduceKFanInAndClipFromComputeBinding(ReduceKFanInAndClipBinding):
    """Compute-entry variant of ``reduce_k_fan_in_and_clip`` (ADR-026).

    Identical dispatch geometry and argument marshaling; only the kernel
    name differs.  Used for interior stages of multi-stage reduction trees
    (where the source is a prior stage's COMPUTE_TYPE output) and for leaf
    stages whose source collection is natively COMPUTE_TYPE (e.g., BCE loss
    partials from Node 7).

    All buffers are compute-role; no precision boundary conversion is
    required.  When ``STORAGE_TYPE == COMPUTE_TYPE``, both variants
    compile to identical machine code.
    """

    def get_kernel_name(self) -> str:
        return "reduce_k_fan_in_and_clip_from_compute"
