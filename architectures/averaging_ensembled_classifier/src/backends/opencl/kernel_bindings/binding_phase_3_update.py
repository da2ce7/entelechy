# src/backends/opencl/kernel_bindings/binding_phase_3_update.py
"""KernelBinding adapters for Final Update Phase kernels (Nodes 21, 24, 25)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ..type_mapping import compute_scalar
from .base import KernelBinding


class NormalizeGradientsBinding(KernelBinding):
    """Binding for normalize_gradients (Node 21)."""

    def get_kernel_name(self) -> str:
        return "normalize_gradients"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        element_count = int(scalar_params["parameter_count"])
        return (element_count,), None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["summed_grad"]),
            get_buffer(buffer_bindings["final_grad"]),
            compute_scalar(scalar_params["effective_batch_size"], scalar_params),
            compute_scalar(scalar_params["epsilon"], scalar_params),
            np.uint32(scalar_params["parameter_count"]),
        ]


class AdamUpdateBinding(KernelBinding):
    """Binding for adam_update (Node 24).

    Host MUST pre-compute beta1_pow_t and beta2_pow_t in float64 to prevent
    on-device precision loss during long training runs.

    The kernel operates on a slice [parameter_offset, parameter_offset + parameter_count)
    within state buffers (m1, m2) of total_parameter_count length.
    """

    def get_kernel_name(self) -> str:
        return "adam_update"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        parameter_count = int(scalar_params["parameter_count"])
        return (parameter_count,), None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["final_grad"]),
            get_buffer(buffer_bindings["parameters"]),
            get_buffer(buffer_bindings["m1"]),
            get_buffer(buffer_bindings["m2"]),
            compute_scalar(scalar_params["learning_rate"], scalar_params),
            compute_scalar(scalar_params["beta1_pow_t"], scalar_params),
            compute_scalar(scalar_params["beta2_pow_t"], scalar_params),
            compute_scalar(scalar_params["beta1"], scalar_params),
            compute_scalar(scalar_params["beta2"], scalar_params),
            compute_scalar(scalar_params["epsilon"], scalar_params),
            np.uint32(scalar_params["parameter_offset"]),
            np.uint32(scalar_params["parameter_count"]),
            np.uint32(scalar_params["total_parameter_count"]),
        ]


class ClampTemperaturesBinding(KernelBinding):
    """Binding for clamp_temperatures (Node 25).

    The kernel operates on a slice [parameter_offset, parameter_offset + parameter_count)
    within the temperatures buffer of total_parameter_count length.
    """

    def get_kernel_name(self) -> str:
        return "clamp_temperatures"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        element_count = int(scalar_params["parameter_count"])
        return (element_count,), None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["temperatures"]),
            compute_scalar(scalar_params["min_value"], scalar_params),
            compute_scalar(scalar_params["max_value"], scalar_params),
            np.uint32(scalar_params["parameter_offset"]),
            np.uint32(scalar_params["parameter_count"]),
            np.uint32(scalar_params["total_parameter_count"]),
        ]
