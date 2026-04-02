# tests/tier2/test_multi_precision_config.py
"""Tier 2 multi-configuration precision tests (ADR-020 §4.6, Phase 7C Step 7C.15).

Validates that all three PrecisionConfig factories (float32, float16,
mixed_f16_f32) produce valid execution plans with correct buffer sizing,
and that the StabilizationPolicy safety ceiling uses compute_fp_format_max.

Includes "The Alchemist" validation scenario (ADR-020 §2.6).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.model_spec import ModelSpec
from src.shared.precision_config import PrecisionConfig
from src.shared.plan_builder import build_act_plan, build_learn_plan
from src.shared.hardware_profile import HardwareProfile
from src.shared.problem_type_strategy import PlanCceStrategy
from src.shared.stabilization_policy import StabilizationPolicy
from tests.tolerance_config import precision_label_from_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IRIS_KWARGS = dict(
    input_dim=4, hidden_dim=32, output_classes=3,
    num_modules=8, simd_width=4, cache_line_bytes=64,
)

_HW = HardwareProfile(
    simd_width=4, cache_line_bytes=64,
    max_reduce_fan_in=64, max_local_mem_bytes=32768, global_mem_bytes=2**30,
)


def _make_spec(precision: PrecisionConfig) -> ModelSpec:
    return ModelSpec(precision=precision, **_IRIS_KWARGS)


def _make_policy(precision: PrecisionConfig) -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        compute_fp_format_max=precision.compute_fp_format_max,
    )


ALL_PRECISIONS = [
    pytest.param(PrecisionConfig.float32(), id="fp32"),
    pytest.param(PrecisionConfig.float16(), id="fp16"),
    pytest.param(PrecisionConfig.mixed_f16_f32(), id="mixed_f16_f32"),
]


# =========================================================================
# Plan builder: parameterized over all precision configs
# =========================================================================


class TestPlanBuildingMultiPrecision:
    """Plan builder produces valid plans for all PrecisionConfig factories."""

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_act_plan_builds(self, precision: PrecisionConfig):
        """build_act_plan succeeds and sets plan.precision."""
        spec = _make_spec(precision)
        plan = build_act_plan(spec, _HW, PlanCceStrategy(), batch_size=16)
        assert plan.precision is precision

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_learn_plan_builds(self, precision: PrecisionConfig):
        """build_learn_plan succeeds and sets plan.precision."""
        spec = _make_spec(precision)
        policy = _make_policy(precision)
        plan = build_learn_plan(spec, _HW, PlanCceStrategy(), batch_size=16, policy=policy)
        assert plan.precision is precision

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_act_plan_storage_buffer_sizes(self, precision: PrecisionConfig):
        """Storage-role buffers use storage_dtype element size."""
        spec = _make_spec(precision)
        plan = build_act_plan(spec, _HW, PlanCceStrategy(), batch_size=16)
        elem_storage = precision.storage_dtype.itemsize

        for buf in plan.buffers.values():
            if buf.precision_role == "storage":
                assert buf.element_size_bytes == elem_storage, (
                    f"Buffer {buf.logical_name}: expected {elem_storage}B storage element, "
                    f"got {buf.element_size_bytes}B"
                )

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_act_plan_state_buffer_sizes(self, precision: PrecisionConfig):
        """State-role buffers use state_dtype element size."""
        spec = _make_spec(precision)
        plan = build_act_plan(spec, _HW, PlanCceStrategy(), batch_size=16)
        elem_state = precision.state_dtype.itemsize

        for buf in plan.buffers.values():
            if buf.precision_role == "state":
                assert buf.element_size_bytes == elem_state, (
                    f"Buffer {buf.logical_name}: expected {elem_state}B state element, "
                    f"got {buf.element_size_bytes}B"
                )

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_learn_plan_compute_buffer_sizes(self, precision: PrecisionConfig):
        """Compute-role floating-point buffers use compute_dtype element size."""
        spec = _make_spec(precision)
        policy = _make_policy(precision)
        plan = build_learn_plan(spec, _HW, PlanCceStrategy(), batch_size=16, policy=policy)
        elem_compute = precision.compute_dtype.itemsize

        for buf in plan.buffers.values():
            if buf.precision_role == "compute":
                # Integer targets buffers are tagged "compute" but use int32 (4B always)
                if "targets" in buf.logical_name:
                    continue
                assert buf.element_size_bytes == elem_compute, (
                    f"Buffer {buf.logical_name}: expected {elem_compute}B compute element, "
                    f"got {buf.element_size_bytes}B"
                )

    @pytest.mark.parametrize("precision", ALL_PRECISIONS)
    def test_precision_label_helper(self, precision: PrecisionConfig):
        """precision_label_from_config returns a valid label string."""
        label = precision_label_from_config(precision)
        assert label in ("fp32", "fp16", "mixed")


# =========================================================================
# The Alchemist: mixed vs fp32 fidelity (ADR-020 §2.6)
# =========================================================================


class TestAlchemistMixedPrecisionFidelity:
    """The Alchemist scenario: mixed_f16_f32 achieves FP32 compute fidelity
    with FP16 storage bandwidth savings."""

    @staticmethod
    def _buffers_by_name(plan):
        """Index plan buffers by name for cross-plan comparison."""
        return {buf.logical_name: buf for buf in plan.buffers.values()}

    def test_mixed_storage_element_size_is_half_fp32(self):
        """Storage-role element size in mixed is half of FP32 (FP16 vs FP32)."""
        fp32_plan = build_act_plan(
            _make_spec(PrecisionConfig.float32()), _HW, PlanCceStrategy(), 16,
        )
        mixed_plan = build_act_plan(
            _make_spec(PrecisionConfig.mixed_f16_f32()), _HW, PlanCceStrategy(), 16,
        )
        fp32_bufs = self._buffers_by_name(fp32_plan)
        mixed_bufs = self._buffers_by_name(mixed_plan)

        for name, fp32_buf in fp32_bufs.items():
            if fp32_buf.precision_role != "storage":
                continue
            mixed_buf = mixed_bufs[name]
            assert mixed_buf.element_size_bytes == fp32_buf.element_size_bytes // 2, (
                f"Buffer {name}: mixed storage element should be half FP32 "
                f"({mixed_buf.element_size_bytes} vs {fp32_buf.element_size_bytes})"
            )

    def test_mixed_state_element_size_equals_fp32(self):
        """State-role element size in mixed equals FP32 (both use FP32 state)."""
        fp32_plan = build_act_plan(
            _make_spec(PrecisionConfig.float32()), _HW, PlanCceStrategy(), 16,
        )
        mixed_plan = build_act_plan(
            _make_spec(PrecisionConfig.mixed_f16_f32()), _HW, PlanCceStrategy(), 16,
        )
        fp32_bufs = self._buffers_by_name(fp32_plan)
        mixed_bufs = self._buffers_by_name(mixed_plan)

        for name, fp32_buf in fp32_bufs.items():
            if fp32_buf.precision_role != "state":
                continue
            mixed_buf = mixed_bufs[name]
            assert mixed_buf.element_size_bytes == fp32_buf.element_size_bytes, (
                f"Buffer {name}: mixed state element should equal FP32 "
                f"({mixed_buf.element_size_bytes} vs {fp32_buf.element_size_bytes})"
            )

    def test_mixed_compute_element_size_equals_fp32(self):
        """Compute-role element size in mixed equals FP32 (both use FP32 compute)."""
        fp32_policy = _make_policy(PrecisionConfig.float32())
        mixed_policy = _make_policy(PrecisionConfig.mixed_f16_f32())

        fp32_plan = build_learn_plan(
            _make_spec(PrecisionConfig.float32()), _HW, PlanCceStrategy(), 16, fp32_policy,
        )
        mixed_plan = build_learn_plan(
            _make_spec(PrecisionConfig.mixed_f16_f32()), _HW, PlanCceStrategy(), 16, mixed_policy,
        )
        fp32_bufs = self._buffers_by_name(fp32_plan)
        mixed_bufs = self._buffers_by_name(mixed_plan)

        for name, fp32_buf in fp32_bufs.items():
            if fp32_buf.precision_role != "compute":
                continue
            mixed_buf = mixed_bufs[name]
            assert mixed_buf.element_size_bytes == fp32_buf.element_size_bytes, (
                f"Buffer {name}: mixed compute element should equal FP32 "
                f"({mixed_buf.element_size_bytes} vs {fp32_buf.element_size_bytes})"
            )

    def test_mixed_stabilization_uses_fp32_safety_ceiling(self):
        """StabilizationPolicy safety ceiling derives from compute_fp_format_max (FP32)."""
        mixed_prec = PrecisionConfig.mixed_f16_f32()
        mixed_policy = _make_policy(mixed_prec)

        # compute_fp_format_max should be FP32 max (~3.4e38), not FP16 max (~65504)
        assert mixed_policy.compute_fp_format_max == float(np.finfo(np.float32).max)
        assert mixed_prec.compute_fp_format_max > 1e30
        assert mixed_prec.storage_fp_format_max < 1e5  # FP16 max ~65504

    def test_mixed_precision_label(self):
        """Mixed config is identified as 'mixed' by the tolerance helper."""
        assert precision_label_from_config(PrecisionConfig.mixed_f16_f32()) == "mixed"
        assert precision_label_from_config(PrecisionConfig.float32()) == "fp32"
        assert precision_label_from_config(PrecisionConfig.float16()) == "fp16"
