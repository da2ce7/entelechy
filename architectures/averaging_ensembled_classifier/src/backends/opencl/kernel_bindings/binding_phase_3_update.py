# src/backends/opencl/kernel_bindings/binding_phase_3_update.py
"""KernelBinding adapters for Final Update Phase kernels (Nodes 21, 24, 25)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
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
            np.float32(scalar_params["effective_batch_size"]),
            np.float32(scalar_params["epsilon"]),
            np.uint32(scalar_params["parameter_count"]),
        ]


class AdamUpdateBinding(KernelBinding):
    """Binding for adam_update (Node 24).

    Host MUST pre-compute beta1_pow_t and beta2_pow_t in float64 to prevent
    on-device precision loss during long training runs.
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
            np.float32(scalar_params["learning_rate"]),
            np.float32(scalar_params["beta1_pow_t"]),
            np.float32(scalar_params["beta2_pow_t"]),
            np.float32(scalar_params["beta1"]),
            np.float32(scalar_params["beta2"]),
            np.float32(scalar_params["epsilon"]),
            np.uint32(scalar_params["parameter_count"]),
        ]


class ClampTemperaturesBinding(KernelBinding):
    """Binding for clamp_temperatures (Node 25)."""

    def get_kernel_name(self) -> str:
        return "clamp_temperatures"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        element_count = int(scalar_params["total_modules_count"])
        return (element_count,), None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["temperatures"]),
            np.float32(scalar_params["min_value"]),
            np.float32(scalar_params["max_value"]),
            np.uint32(scalar_params["total_modules_count"]),
        ]
