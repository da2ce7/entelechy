# src/backends/opencl/kernel_bindings/binding_phase_3_update.py
"""KernelBinding adapters for finalization and parameter update kernels
(Nodes 21, 24, 25).

Each binding translates the backend-neutral plan node's buffer bindings and
scalar parameters into the concrete argument list required by
``clEnqueueNDRangeKernel``, including dispatch geometry and scalar type
marshaling (CONTRACT.md §7).

Reference specification: kernels.cl.h (ADR-013 designation).
Dispatch geometry source: phase_3_update.cl.c implementation comments.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ..type_mapping import compute_scalar
from .base import KernelBinding


# ---------------------------------------------------------------------------
# Node 21 — normalize_gradients
# ---------------------------------------------------------------------------


class NormalizeGradientsBinding(KernelBinding):
    """Binding for ``normalize_gradients`` (Node 21).

    Dispatch geometry::

        global = (parameter_count)
        local  = backend-selected (None)

    Embarrassingly parallel "map" kernel.  Each work-item normalizes exactly
    one element of the summed gradient buffer by dividing by the effective
    batch size.  This converts the batch-wide gradient sum from the
    reduction engine into a true average gradient, ensuring that learning
    dynamics are independent of batch size.

    All buffers are compute-role; no precision boundary conversion is
    required.  The epsilon term prevents division by zero when the
    effective batch size is 0 (e.g., all samples were masked).
    """

    def get_kernel_name(self) -> str:
        return "normalize_gradients"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Map kernel — no tiling.
        return (
            int(scalar_params["src_scalar_NATURAL_parameter_count"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Map kernel — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        return [
            buf("src_buffer_GLOBAL_summed_grad"),
            buf("dest_buffer_GLOBAL_final_grad"),
            real("src_scalar_REAL_effective_batch_size"),
            real("src_scalar_REAL_epsilon"),
            u32("src_scalar_NATURAL_parameter_count"),
        ]


# ---------------------------------------------------------------------------
# Node 24 — adam_update
# ---------------------------------------------------------------------------


class AdamUpdateBinding(KernelBinding):
    """Binding for ``adam_update`` (Node 24).

    Dispatch geometry::

        global = (parameter_count)
        local  = backend-selected (None)

    Stateful, embarrassingly parallel "map" kernel.  Each work-item updates
    a single parameter and its corresponding first and second moment vectors
    according to the Adam optimizer algorithm.

    State-Precision Accumulation: EMA updates (m₁, m₂) and parameter
    subtraction are performed in ``ACCUM_TYPE = max(COMPUTE_TYPE,
    STATE_TYPE)``, preserving state fidelity across unbounded training
    steps.  Bias correction and parameter delta computation are
    transformative operations performed in COMPUTE_TYPE.

    The Host MUST pre-compute ``beta1_pow_t`` and ``beta2_pow_t`` in FP64
    to prevent on-device precision loss during long training runs.  These
    FP64 values are narrowed to COMPUTE_TYPE at the interface boundary.

    ADR-030: The kernel operates on a slice
    ``[parameter_offset, parameter_offset + parameter_count)`` within
    state-role buffers (parameters, m₁, m₂) of ``total_parameter_count``
    length.  The Host guarantees
    ``(parameter_offset + parameter_count) <= total_parameter_count``.
    """

    def get_kernel_name(self) -> str:
        return "adam_update"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Map kernel — no tiling.
        return (
            int(scalar_params["src_scalar_NATURAL_parameter_count"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Stateful update — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        return [
            buf("src_buffer_GLOBAL_final_grad"),
            buf("update_buffer_GLOBAL_parameters"),
            buf("update_buffer_GLOBAL_m1"),
            buf("update_buffer_GLOBAL_m2"),
            real("src_scalar_REAL_learning_rate"),
            real("src_scalar_REAL_beta1_pow_t"),
            real("src_scalar_REAL_beta2_pow_t"),
            real("src_scalar_REAL_beta1"),
            real("src_scalar_REAL_beta2"),
            real("src_scalar_REAL_epsilon"),
            u32("src_scalar_NATURAL_parameter_offset"),
            u32("src_scalar_NATURAL_parameter_count"),
            u32("src_scalar_NATURAL_total_parameter_count"),
        ]


# ---------------------------------------------------------------------------
# Node 25 — clamp_temperatures
# ---------------------------------------------------------------------------


class ClampTemperaturesBinding(KernelBinding):
    """Binding for ``clamp_temperatures`` (Node 25).

    Dispatch geometry::

        global = (parameter_count)
        local  = backend-selected (None)

    Embarrassingly parallel "map" kernel.  Each work-item operates on a
    single temperature parameter independently, enforcing domain-specific
    ``[min, max]`` constraints.  This "parameter governor" ensures learnable
    temperatures remain in a stable and meaningful range, preventing
    numerical instability in downstream Softmax/Sigmoid computations.

    Precision Boundary Conversion: state-role buffer accessed via
    ``load_state()`` / ``store_state_update()``; clamp arithmetic
    exclusively in COMPUTE_TYPE.

    ADR-030: The kernel operates on a slice
    ``[parameter_offset, parameter_offset + parameter_count)`` within the
    state-role temperatures buffer of ``total_parameter_count`` length.
    The Host guarantees
    ``(parameter_offset + parameter_count) <= total_parameter_count``.
    """

    def get_kernel_name(self) -> str:
        return "clamp_temperatures"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Map kernel — no tiling.
        return (
            int(scalar_params["src_scalar_NATURAL_parameter_count"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Finalizer utility — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(
            scalar_params[key]
        )
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        return [
            buf("update_buffer_GLOBAL_temps"),
            real("src_scalar_REAL_min_value"),
            real("src_scalar_REAL_max_value"),
            u32("src_scalar_NATURAL_parameter_offset"),
            u32("src_scalar_NATURAL_parameter_count"),
            u32("src_scalar_NATURAL_total_parameter_count"),
        ]
