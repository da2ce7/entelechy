# tests/tier2/cpu/test_cpu_crash_regressions.py
"""
Tier 2 CPU regression tests: crash reproducers for plan rendering.

These tests reproduce SIGSEGV/SIGABRT crashes discovered during benchmark
expansion (bench_cpu_backend.py, 2026-03-30).  Each test exercises the
minimal plan-build → render path that triggers the defect.

Failure Categories:
  1. Lexicon (output_classes=10_000) — heap corruption during tiled
     kernel dispatch when class-dimension tiling produces large offsets.
     All Act and Learn plans crash (SIGSEGV).
  2. Rodeo (FP16 precision) — struct layout mismatch or missing
     half-float support in CPU kernel sources.  Act crashes with
     SIGABRT, Learn crashes with SIGSEGV/SIGABRT.

Each test is intentionally minimal: build the plan, render it once.
If the native library corrupts the heap or violates memory, the process
will crash (caught by the test runner as a subprocess failure or by
ASAN if using the sanitized build).
"""
from __future__ import annotations

import pytest

from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]

if not BACKEND_CPU:
    pytest.skip("CPU backend not available", allow_module_level=True)

from src.backends.cpu.discovery import discover_hardware  # noqa: E402
from src.backends.cpu.renderer import CPUPlanRenderer  # noqa: E402
from src.shared.hardware_profile import HardwareProfile  # noqa: E402
from src.shared.model_spec import ModelSpec  # noqa: E402
from src.shared.precision_config import PrecisionConfig  # noqa: E402
from src.shared.plan_builder import build_act_plan, build_learn_plan  # noqa: E402
from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy  # noqa: E402
from src.shared.stabilization_policy import StabilizationPolicy  # noqa: E402


# =========================================================================
# Fixtures
# =========================================================================

_PREC_FP32 = PrecisionConfig.float32()
_PREC_FP16 = PrecisionConfig.float16()


@pytest.fixture(scope="module")
def hw() -> HardwareProfile:
    return discover_hardware()


@pytest.fixture(scope="module")
def renderer() -> CPUPlanRenderer:
    return CPUPlanRenderer()


def _lexicon_spec(hw: HardwareProfile) -> ModelSpec:
    """10,000 output classes — triggers class-dimension tiling overflow."""
    return ModelSpec(
        precision=_PREC_FP32,
        input_dim=8, hidden_dim=32, output_classes=10_000,
        num_modules=4, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


def _rodeo_spec(hw: HardwareProfile) -> ModelSpec:
    """FP16 — triggers struct layout / half-float crash."""
    return ModelSpec(
        precision=_PREC_FP16,
        input_dim=8, hidden_dim=64, output_classes=10,
        num_modules=16, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


def _fp32_policy() -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0, lambda_=1.0,
        compute_fp_format_max=_PREC_FP32.compute_fp_format_max,
    )


def _fp16_aggressive_policy() -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=0.1, lambda_=0.01,
        compute_fp_format_max=_PREC_FP16.compute_fp_format_max,
    )


# =========================================================================
# Category 1: Lexicon — large output_classes (SIGSEGV)
# =========================================================================


class TestLexiconCrashRegression:
    """Regression: output_classes=10_000 formerly caused heap corruption in CPU renderer."""

    def test_lexicon_act_cce(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _lexicon_spec(hw)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 16)
        renderer.render(plan)

    def test_lexicon_act_cce_bs64(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _lexicon_spec(hw)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 64)
        renderer.render(plan)

    def test_lexicon_learn_cce(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _lexicon_spec(hw)
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 16, _fp32_policy())
        renderer.render(plan)

    def test_lexicon_learn_cce_bs64(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _lexicon_spec(hw)
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 64, _fp32_policy())
        renderer.render(plan)

    def test_lexicon_learn_bce(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _lexicon_spec(hw)
        plan = build_learn_plan(spec, hw, PlanBceStrategy(), 16, _fp32_policy())
        renderer.render(plan)


# =========================================================================
# Category 2: Rodeo — FP16 precision (formerly SIGSEGV / SIGABRT)
# =========================================================================


class TestRodeoCrashRegression:
    """Regression: FP16 precision formerly caused buffer overflow in CPU renderer."""

    def test_rodeo_act_fp16(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _rodeo_spec(hw)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 64)
        renderer.render(plan)

    def test_rodeo_learn_fp16_cce(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _rodeo_spec(hw)
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 64, _fp16_aggressive_policy())
        renderer.render(plan)

    def test_rodeo_learn_fp16_bce(self, hw: HardwareProfile, renderer: CPUPlanRenderer) -> None:
        spec = _rodeo_spec(hw)
        plan = build_learn_plan(spec, hw, PlanBceStrategy(), 64, _fp16_aggressive_policy())
        renderer.render(plan)
