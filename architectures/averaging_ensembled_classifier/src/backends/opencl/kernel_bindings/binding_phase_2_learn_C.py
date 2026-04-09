# src/backends/opencl/kernel_bindings/binding_phase_2_learn_C.py
"""KernelBinding adapters for Learn-C reduction & aggregation kernels."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ....shared.memory_layout import pad_to_multiple
from ....shared.stabilization_policy import render_node16_threshold_schedule
from .base import KernelBinding


class AggregateRegisterReduceBinding(KernelBinding):
    """Binding for aggregate_register_reduce (reduction engine tier 1)."""

    def get_kernel_name(self) -> str:
        return "aggregate_register_reduce"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        partial_width = int(scalar_params["partial_width"])
        return (partial_width,), None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["partial_collection"]),
            get_buffer(buffer_bindings["partial_offset_list"]),
            get_buffer(buffer_bindings["dest"]),
            np.uint32(scalar_params["partial_offset_list_count"]),
            np.uint32(scalar_params["partial_width"]),
            np.uint32(scalar_params.get("operation_type", 0)),
        ]

    # Reduction-specific interface for renderer direct use
    def marshal_args_reduction(
        self,
        source: cl.Buffer,
        offset_list: cl.Buffer,
        dest: cl.Buffer,
        offset_count: int,
        partial_width: int,
        operation_type: int,
    ) -> list[Any]:
        return [
            source, offset_list, dest,
            np.uint32(offset_count),
            np.uint32(partial_width),
            np.uint32(operation_type),
        ]

    def compute_grid_reduction(
        self, partial_width: int, hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        return (partial_width,), None


class AggregateLocalReduceBinding(KernelBinding):
    """Binding for aggregate_local_reduce (reduction engine tier 2)."""

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "aggregate_local_reduce"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        partial_width = int(scalar_params["partial_width"])
        wg = self._workgroup_size
        global_size = (pad_to_multiple(partial_width, wg),)
        local_size = (wg,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["partial_collection"]),
            get_buffer(buffer_bindings["partial_offset_list"]),
            get_buffer(buffer_bindings["dest"]),
            np.uint32(scalar_params["partial_offset_list_count"]),
            np.uint32(scalar_params["partial_width"]),
            np.uint32(scalar_params.get("operation_type", 0)),
        ]

    # Reduction-specific interface for renderer direct use
    def marshal_args_reduction(
        self,
        source: cl.Buffer,
        offset_list: cl.Buffer,
        dest: cl.Buffer,
        offset_count: int,
        partial_width: int,
        operation_type: int,
        element_size: int = 4,
    ) -> list[Any]:
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            source, offset_list, dest,
            np.uint32(offset_count),
            np.uint32(partial_width),
            np.uint32(operation_type),
        ]

    def compute_grid_reduction(
        self, partial_width: int, hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg = self._workgroup_size
        global_size = (pad_to_multiple(partial_width, wg),)
        local_size = (wg,)
        return global_size, local_size


class ClipIntermediateGradBinding(KernelBinding):
    """Binding for clip_intermediate_grad (inter-stage clipping in sum_and_clip trees)."""

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "clip_intermediate_grad"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        param_count = int(scalar_params["parameter_count"])
        wg = self._workgroup_size
        global_size = (pad_to_multiple(param_count, wg),)
        local_size = (wg,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["intermediate_grad"]),
            np.float32(scalar_params["clipping_threshold"]),
            np.float32(scalar_params["epsilon"]),
            np.uint32(scalar_params["parameter_count"]),
        ]

    # Clip-specific interface for renderer direct use
    def marshal_args_clip(
        self,
        buffer: cl.Buffer,
        threshold: float,
        epsilon: float,
        param_count: int,
        element_size: int = 4,
    ) -> list[Any]:
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            buffer,
            np.float32(threshold),
            np.float32(epsilon),
            np.uint32(param_count),
        ]

    def compute_grid_clip(
        self, param_count: int, hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg = self._workgroup_size
        global_size = (pad_to_multiple(param_count, wg),)
        local_size = (wg,)
        return global_size, local_size


class StabilizeReduceGradHBinding(KernelBinding):
    """Binding for stabilize_and_reduce_grad_hidden_activations (Node 16)."""

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "stabilize_and_reduce_grad_hidden_activations"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        total_batch = int(scalar_params["total_batch_count"])
        padded_hidden = int(scalar_params["padded_hidden_count"])
        wg = self._workgroup_size
        num_rows = total_batch * padded_hidden
        global_size = (num_rows * wg,)
        local_size = (wg,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._workgroup_size * element_size

        # Render the threshold schedule for this backend's workgroup size.
        M = int(scalar_params["total_modules_count"])
        W = self._workgroup_size
        max_k = int(scalar_params["policy_max_k"])
        num_stages, t_pre, schedule = render_node16_threshold_schedule(
            t_algorithmic=float(scalar_params["policy_t_algorithmic"]),
            lambda_=float(scalar_params["policy_lambda"]),
            compute_fp_format_max=float(scalar_params["compute_fp_format_max"]),
            total_modules=M,
            workgroup_size=W,
            max_fan_in=max_k,
        )

        # Fill the schedule buffer.
        schedule_buf = get_buffer(buffer_bindings["clipping_threshold_per_stage"])
        if schedule:
            schedule_np = np.array(schedule, dtype=np.float32)
            cl.enqueue_copy(schedule_buf.context.queue, schedule_buf, schedule_np)

        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["clipped_grad_hidden_activations_permuted_soa"]),
            get_buffer(buffer_bindings["summed_grad_hidden_activations"]),
            get_buffer(buffer_bindings["clipping_threshold_per_stage"]),
            np.uint32(num_stages),
            np.float32(t_pre),
            np.float32(scalar_params["epsilon"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["padded_total_modules_count"]),
        ]


class ReduceKFanInAndClipBinding(KernelBinding):
    """Binding for reduce_k_fan_in_and_clip (ADR-019: multi-stage K-fan-in)."""

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "reduce_k_fan_in_and_clip"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        node_count = int(scalar_params["node_count"])
        wg = self._workgroup_size
        global_size = (node_count * wg,)
        local_size = (wg,)
        return global_size, local_size

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["partial_collection"]),
            get_buffer(buffer_bindings["offset_list_flat"]),
            get_buffer(buffer_bindings["stage_partial"]),
            np.uint32(scalar_params["fan_in"]),
            np.uint32(scalar_params["node_count"]),
            np.uint32(scalar_params["partial_width"]),
            np.float32(scalar_params["clipping_threshold"]),
            np.float32(scalar_params["epsilon"]),
        ]

    # Reduction-specific interface for renderer direct use
    def marshal_args_fan_in(
        self,
        source: cl.Buffer,
        offset_list: cl.Buffer,
        dest: cl.Buffer,
        fan_in: int,
        node_count: int,
        partial_width: int,
        clipping_threshold: float,
        epsilon: float,
        element_size: int = 4,
    ) -> list[Any]:
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            source,
            offset_list,
            dest,
            np.uint32(fan_in),
            np.uint32(node_count),
            np.uint32(partial_width),
            np.float32(clipping_threshold),
            np.float32(epsilon),
        ]

    def compute_grid_fan_in(
        self, node_count: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        wg = self._workgroup_size
        global_size = (node_count * wg,)
        local_size = (wg,)
        return global_size, local_size


# ══════════════════════════════════════════════════════════════════════════════
# ADR-026: Compute-entry variants for compute-role source buffers
# ══════════════════════════════════════════════════════════════════════════════


class AggregateRegisterReduceFromComputeBinding(AggregateRegisterReduceBinding):
    """Binding for aggregate_register_reduce_from_compute (ADR-026).

    Compute-entry variant of aggregate_register_reduce. Used when the source
    buffer's precision_role is "compute" (e.g., BCE loss partials, or interior
    stages of multi-stage reduction trees).
    """

    def get_kernel_name(self) -> str:
        return "aggregate_register_reduce_from_compute"


class AggregateLocalReduceFromComputeBinding(AggregateLocalReduceBinding):
    """Binding for aggregate_local_reduce_from_compute (ADR-026).

    Compute-entry variant of aggregate_local_reduce. Used when the source
    buffer's precision_role is "compute".
    """

    def get_kernel_name(self) -> str:
        return "aggregate_local_reduce_from_compute"


class ReduceKFanInAndClipFromComputeBinding(ReduceKFanInAndClipBinding):
    """Binding for reduce_k_fan_in_and_clip_from_compute (ADR-026).

    Compute-entry variant of reduce_k_fan_in_and_clip. Used for interior
    stages of multi-stage reduction trees (where the source is a prior
    stage's COMPUTE_TYPE output) and for leaf stages whose source collection
    is natively COMPUTE_TYPE (e.g., BCE loss partials).
    """

    def get_kernel_name(self) -> str:
        return "reduce_k_fan_in_and_clip_from_compute"
