# tests/bench_cpu_backend.py
from __future__ import annotations

"""
Benchmarks: CPU Backend — Plan-Level Execution.

These benchmarks exercise the CPU backend's hot-paths through the plan
renderer: the code that allocates SIMD-aligned buffers, marshals FFI
argument structs, and dispatches kernels via pool_dispatch_and_wait.

The CPU backend operates at the ExecutionPlan level (not individual
kernel calls), so benchmarks are structured around Act and Learn plans
at various model scales.

A compiled CPU backend (libcpu_kernels.so) IS required.  All benchmarks
are skipped gracefully when the CPU backend is not available.

Run with:
    pytest tests/bench_cpu_backend.py --benchmark-only
    pytest tests/bench_cpu_backend.py --benchmark-only --benchmark-sort=fullname
    pytest tests/bench_cpu_backend.py --benchmark-only --benchmark-group-by=group
"""

import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional CPU backend import — skip entire module when absent.
# ---------------------------------------------------------------------------
try:
    from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]
    _has_cpu = BACKEND_CPU
except ImportError:
    _has_cpu = False

pytestmark = pytest.mark.skipif(
    not _has_cpu,
    reason="CPU backend not available (BACKEND_CPU=False or _build_config missing)",
)

if TYPE_CHECKING:
    from src.backends.cpu.renderer import CPUPlanRenderer
    from src.backends.cpu.discovery import discover_hardware, detect_thread_count
    from src.shared.hardware_profile import HardwareProfile
    from src.shared.model_spec import ModelSpec
    from src.shared.precision_config import PrecisionConfig
    from src.shared.plan_builder import build_act_plan, build_learn_plan
    from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy
    from src.shared.stabilization_policy import StabilizationPolicy
elif _has_cpu:
    from src.backends.cpu.renderer import CPUPlanRenderer  # noqa: E402
    from src.backends.cpu.discovery import discover_hardware, detect_thread_count  # noqa: E402
    from src.shared.hardware_profile import HardwareProfile  # noqa: E402
    from src.shared.model_spec import ModelSpec  # noqa: E402
    from src.shared.precision_config import PrecisionConfig  # noqa: E402
    from src.shared.plan_builder import build_act_plan, build_learn_plan  # noqa: E402
    from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy  # noqa: E402
    from src.shared.stabilization_policy import StabilizationPolicy  # noqa: E402


# =========================================================================
# Shared Fixtures
# =========================================================================


@pytest.fixture(scope="session")
def hardware_profile() -> HardwareProfile:
    return discover_hardware()


@pytest.fixture(scope="session")
def thread_count() -> int:
    return detect_thread_count()


@pytest.fixture(scope="session")
def renderer() -> CPUPlanRenderer:
    """Session-scoped renderer — amortises library load."""
    return CPUPlanRenderer()


@pytest.fixture(scope="session")
def renderer_single() -> CPUPlanRenderer:
    """Single-threaded renderer for serial baseline."""
    return CPUPlanRenderer(thread_count=1)


# =========================================================================
# Model Configurations
# =========================================================================

_PREC_FP32: PrecisionConfig = PrecisionConfig.float32()  # type: ignore[reportCallIssue]


def _make_hw(profile: HardwareProfile) -> HardwareProfile:
    """Use the real discovered hardware profile."""
    return profile


def _iris_spec(hw: HardwareProfile) -> ModelSpec:
    return ModelSpec(
        precision=_PREC_FP32,
        input_dim=4, hidden_dim=32, output_classes=3,
        num_modules=8, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


def _hydra_spec(hw: HardwareProfile) -> ModelSpec:
    return ModelSpec(
        precision=_PREC_FP32,
        input_dim=16, hidden_dim=64, output_classes=10,
        num_modules=256, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


def _lexicon_spec(hw: HardwareProfile) -> ModelSpec:
    return ModelSpec(
        precision=_PREC_FP32,
        input_dim=8, hidden_dim=32, output_classes=10_000,
        num_modules=4, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


_SPEC_FACTORIES = {
    "iris": _iris_spec,
    "hydra": _hydra_spec,
    "lexicon": _lexicon_spec,
}


def _make_policy() -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0, lambda_=1.0,
        fp_format_max=float(np.finfo(np.float32).max),
    )


# =========================================================================
# §1 — CPUPlanRenderer Construction
# =========================================================================


class TestBenchRendererInit:
    """Benchmark the cost of creating a CPUPlanRenderer (library load + pool)."""

    @pytest.mark.benchmark(group="renderer-init")
    def test_bench_renderer_construction(self, benchmark: Any) -> None:
        """Library load + thread pool creation."""
        def _create():
            r = CPUPlanRenderer()
            del r
        benchmark(_create)

    @pytest.mark.benchmark(group="renderer-init")
    def test_bench_renderer_single_thread(self, benchmark: Any) -> None:
        """Single-threaded renderer construction."""
        def _create():
            r = CPUPlanRenderer(thread_count=1)
            del r
        benchmark(_create)


# =========================================================================
# §2 — Act Phase Benchmarks
# =========================================================================


class TestBenchActPhase:
    """Benchmark complete Act-phase plan execution."""

    @pytest.mark.benchmark(group="act-phase-cce")
    @pytest.mark.parametrize(
        "spec_name,batch_size",
        [
            ("iris", 32),
            ("iris", 150),
            ("hydra", 32),
            ("hydra", 150),
        ],
        ids=["iris/BS=32", "iris/BS=150", "hydra/BS=32", "hydra/BS=150"],
    )
    def test_bench_act_cce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
        spec_name: str, batch_size: int,
    ) -> None:
        """Full Act phase: forward_pass → render_logits → probs/loss → retrieval."""
        spec = _SPEC_FACTORIES[spec_name](hardware_profile)
        hw = _make_hw(hardware_profile)
        strategy = PlanCceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size)

        # Warm-up: first render compiles any lazy init
        renderer.render(plan)

        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="act-phase-bce")
    @pytest.mark.parametrize(
        "spec_name,batch_size",
        [
            ("iris", 150),
            ("hydra", 150),
        ],
        ids=["iris/BS=150", "hydra/BS=150"],
    )
    def test_bench_act_bce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
        spec_name: str, batch_size: int,
    ) -> None:
        """Act phase with BCE loss."""
        spec = _SPEC_FACTORIES[spec_name](hardware_profile)
        hw = _make_hw(hardware_profile)
        strategy = PlanBceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size)

        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §3 — Learn Phase Benchmarks
# =========================================================================


class TestBenchLearnPhase:
    """Benchmark complete Learn-phase plan execution."""

    @pytest.mark.benchmark(group="learn-phase-cce")
    @pytest.mark.parametrize(
        "spec_name,batch_size",
        [
            ("iris", 32),
            ("iris", 150),
            ("hydra", 32),
            ("hydra", 150),
        ],
        ids=["iris/BS=32", "iris/BS=150", "hydra/BS=32", "hydra/BS=150"],
    )
    def test_bench_learn_cce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
        spec_name: str, batch_size: int,
    ) -> None:
        """Full Learn phase: gradient production → reduction → streaming backprop → update."""
        spec = _SPEC_FACTORIES[spec_name](hardware_profile)
        hw = _make_hw(hardware_profile)
        strategy = PlanCceStrategy()
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, strategy, batch_size, policy)

        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="learn-phase-bce")
    @pytest.mark.parametrize(
        "spec_name,batch_size",
        [
            ("iris", 150),
            ("hydra", 150),
        ],
        ids=["iris/BS=150", "hydra/BS=150"],
    )
    def test_bench_learn_bce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
        spec_name: str, batch_size: int,
    ) -> None:
        """Learn phase with BCE loss."""
        spec = _SPEC_FACTORIES[spec_name](hardware_profile)
        hw = _make_hw(hardware_profile)
        strategy = PlanBceStrategy()
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, strategy, batch_size, policy)

        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §4 — Threading Comparison
# =========================================================================


class TestBenchThreadingComparison:
    """Compare multi-threaded vs single-threaded plan execution."""

    @pytest.mark.benchmark(group="threading-act")
    def test_bench_act_multithread(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Act phase: multi-threaded (auto-detected thread count)."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 150)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="threading-act")
    def test_bench_act_singlethread(
        self, benchmark: Any, renderer_single: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Act phase: single-threaded baseline."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 150)
        renderer_single.render(plan)
        benchmark(renderer_single.render, plan)

    @pytest.mark.benchmark(group="threading-learn")
    def test_bench_learn_multithread(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase: multi-threaded (auto-detected thread count)."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 150, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="threading-learn")
    def test_bench_learn_singlethread(
        self, benchmark: Any, renderer_single: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase: single-threaded baseline."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 150, policy)
        renderer_single.render(plan)
        benchmark(renderer_single.render, plan)


# =========================================================================
# §5 — Scale Comparison
# =========================================================================


class TestBenchScaleComparison:
    """Benchmark plan execution across model scales."""

    @pytest.mark.benchmark(group="scale-act")
    @pytest.mark.parametrize(
        "spec_name", ["iris", "hydra"],
    )
    def test_bench_act_scale(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, spec_name: str,
    ) -> None:
        """Act phase at different model scales (BS=150)."""
        spec = _SPEC_FACTORIES[spec_name](hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 150)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="scale-learn")
    @pytest.mark.parametrize(
        "spec_name", ["iris", "hydra"],
    )
    def test_bench_learn_scale(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, spec_name: str,
    ) -> None:
        """Learn phase at different model scales (BS=150)."""
        spec = _SPEC_FACTORIES[spec_name](hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 150, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)
