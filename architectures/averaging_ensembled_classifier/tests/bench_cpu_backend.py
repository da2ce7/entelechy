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

Validation Scenarios from CONCEPT.md:
  §1 — Renderer Init           (infrastructure)
  §2 — Act Phase               (baseline Act benchmarks)
  §3 — Learn Phase             (baseline Learn benchmarks)
  §4 — Threading Comparison    (multi vs single thread)
  §5 — Scale Comparison        (model scale)
  §6 — The Iris Case           (graceful degradation, small batches)
  §7 — The Real-Time Trader    (Act/Learn temporal split)
  §8 — The Marathon            (multi-epoch Adam stability)
  §9 — The Hydra              (massive num_heads)
  §10 — The Behemoth          (massive hidden_dim, cache vs recompute)
  §11 — The Lexicon           (massive output_classes)
  §12 — The Rodeo             (FP16 numerical safety)
  §13 — The Scientist's Repeater (batch-size-independent dynamics)
  §14 — The Data Tsunami      (massive batch_size)
  §15 — The Colossus          (compound stress)
  §16 — The Cross-Backend Arbiter (CCE/BCE parity)

Run with:
    pytest tests/bench_cpu_backend.py --benchmark-only
    pytest tests/bench_cpu_backend.py --benchmark-only --benchmark-sort=fullname
    pytest tests/bench_cpu_backend.py --benchmark-only --benchmark-group-by=group
"""

from typing import TYPE_CHECKING, Any

import pytest

# ---------------------------------------------------------------------------
# Conditional CPU backend import — skip entire module when absent.
# ---------------------------------------------------------------------------
_has_cpu: bool = False
try:
    from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]
    _has_cpu = True if BACKEND_CPU else False  # type: ignore[possibly-undefined]
except ImportError:
    pass

pytestmark = [
    pytest.mark.skipif(
        not _has_cpu,
        reason="CPU backend not available (BACKEND_CPU=False or _build_config missing)",
    ),
    pytest.mark.timeout(20),
]

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
_PREC_FP16: PrecisionConfig = PrecisionConfig.mixed_f16_f32()  # type: ignore[reportCallIssue]


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


def _behemoth_spec(hw: HardwareProfile) -> ModelSpec:
    """Massive hidden_dim — stresses activation buffers."""
    return ModelSpec(
        precision=_PREC_FP32,
        input_dim=16, hidden_dim=2048, output_classes=10,
        num_modules=8, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


def _rodeo_spec_fp16(hw: HardwareProfile) -> ModelSpec:
    """FP16 spec for numerical safety stress testing."""
    return ModelSpec(
        precision=_PREC_FP16,
        input_dim=8, hidden_dim=64, output_classes=10,
        num_modules=16, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


def _colossus_spec(hw: HardwareProfile) -> ModelSpec:
    """Compound stress: many heads, large classes, decent hidden_dim."""
    return ModelSpec(
        precision=_PREC_FP32,
        input_dim=16, hidden_dim=512, output_classes=1_000,
        num_modules=64, simd_width=hw.simd_width,
        cache_line_bytes=hw.cache_line_bytes,
    )


_SPEC_FACTORIES = {
    "iris": _iris_spec,
    "hydra": _hydra_spec,
    "lexicon": _lexicon_spec,
    "behemoth": _behemoth_spec,
    "rodeo_fp16": _rodeo_spec_fp16,
    "colossus": _colossus_spec,
}


def _make_policy(prec: PrecisionConfig = _PREC_FP32) -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0, lambda_=1.0,
        compute_fp_format_max=prec.compute_fp_format_max,
    )


def _make_aggressive_policy(prec: PrecisionConfig = _PREC_FP32) -> StabilizationPolicy:
    """Tight funnel — stresses clipping at every reduction stage."""
    return StabilizationPolicy(
        t_algorithmic=0.1, lambda_=0.01,
        compute_fp_format_max=prec.compute_fp_format_max,
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


# =========================================================================
# §6 — The Iris Case (Sequential Execution Mode Validation)
#
# CONCEPT.md: Proves graceful degradation of parallel machinery.
# The Act/Learn split is not a costly abstraction on small problems.
# =========================================================================


class TestBenchIrisCase:
    """Benchmark the full Act→Learn cycle at Iris scale (N=150, 4→3)."""

    @pytest.mark.benchmark(group="iris-case")
    def test_bench_iris_act_learn_cycle(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """One complete Act→Learn render cycle at Iris scale."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        act_plan = build_act_plan(spec, hw, PlanCceStrategy(), 150)
        learn_plan = build_learn_plan(spec, hw, PlanCceStrategy(), 150, policy)

        renderer.render(act_plan)
        renderer.render(learn_plan)

        def _cycle():
            renderer.render(act_plan)
            renderer.render(learn_plan)
        benchmark(_cycle)

    @pytest.mark.benchmark(group="iris-case")
    def test_bench_iris_n1_degenerate(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """N=1: degenerate case — no meaningful reduction needed."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        act_plan = build_act_plan(spec, hw, PlanCceStrategy(), 1)
        learn_plan = build_learn_plan(spec, hw, PlanCceStrategy(), 1, policy)

        renderer.render(act_plan)
        renderer.render(learn_plan)

        def _cycle():
            renderer.render(act_plan)
            renderer.render(learn_plan)
        benchmark(_cycle)


# =========================================================================
# §7 — The Real-Time Trader (Event-Triggered Execution Mode)
#
# CONCEPT.md: Act and Learn plans are independently renderable.
# The temporal split between Act and Learn is a first-class concept.
# =========================================================================


class TestBenchRealTimeTrader:
    """Benchmark Act-only latency (prediction serving) independently of Learn."""

    @pytest.mark.benchmark(group="trader-act-only")
    @pytest.mark.parametrize("batch_size", [1, 16, 64], ids=["BS=1", "BS=16", "BS=64"])
    def test_bench_act_only_latency(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Act-only render: prediction serving latency at various batch sizes."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), batch_size)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="trader-deferred-learn")
    def test_bench_deferred_learn(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn-only render: deferred learning after prediction served."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 16, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §8 — The Marathon (Massive Epochs)
#
# CONCEPT.md: Guarantees long-term stability by delegating beta**t to
# the host. Benchmark measures per-epoch amortized cost over many
# sequential renders (5 epochs keeps wall-clock reasonable).
# =========================================================================


class TestBenchMarathon:
    """Benchmark repeated Act→Learn cycles simulating multi-epoch training."""

    @pytest.mark.benchmark(group="marathon")
    @pytest.mark.parametrize("num_epochs", [5], ids=["epochs=5"])
    def test_bench_marathon_epochs(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, num_epochs: int,
    ) -> None:
        """Multi-epoch Act→Learn cycle (amortized per-epoch cost)."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        act_plan = build_act_plan(spec, hw, PlanCceStrategy(), 150)
        learn_plan = build_learn_plan(spec, hw, PlanCceStrategy(), 150, policy)

        # Warm-up
        renderer.render(act_plan)
        renderer.render(learn_plan)

        def _epoch_loop():
            for _ in range(num_epochs):
                renderer.render(act_plan)
                renderer.render(learn_plan)
        benchmark(_epoch_loop)


# =========================================================================
# §9 — The Hydra (Massive num_heads)
#
# CONCEPT.md: Validates the Reduction Planner — massive num_modules
# produces an efficient log_K(N) reduction tree for gradient
# accumulation.
# =========================================================================


class TestBenchHydra:
    """Benchmark at massive num_modules (256 heads)."""

    @pytest.mark.benchmark(group="hydra-act")
    @pytest.mark.parametrize("batch_size", [32, 128], ids=["BS=32", "BS=128"])
    def test_bench_hydra_act(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Act phase with 256 modules."""
        spec = _hydra_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), batch_size)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="hydra-learn")
    @pytest.mark.parametrize("batch_size", [32, 128], ids=["BS=32", "BS=128"])
    def test_bench_hydra_learn(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Learn phase with 256 modules — exercises log_K(N) reduction tree."""
        spec = _hydra_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), batch_size, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="hydra-learn-bce")
    def test_bench_hydra_learn_bce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase (BCE) with 256 modules."""
        spec = _hydra_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanBceStrategy(), 64, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §10 — The Behemoth (Massive hidden_dim)
#
# CONCEPT.md: Proves scalability through strategic memory trade-offs.
# Cache vs Recompute activation lifecycle is a critical orthogonal
# optimization.
# =========================================================================


class TestBenchBehemoth:
    """Benchmark at massive hidden_dim (2048) — activation buffer pressure."""

    @pytest.mark.benchmark(group="behemoth-act")
    def test_bench_behemoth_act(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Act phase with hidden_dim=2048."""
        spec = _behemoth_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 32)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="behemoth-learn-recompute")
    def test_bench_behemoth_learn_recompute(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase with hidden_dim=2048, activation_lifecycle=recompute."""
        spec = _behemoth_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(
            spec, hw, PlanCceStrategy(), 32, policy,
            activation_lifecycle="recompute",
        )
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="behemoth-learn-cache")
    def test_bench_behemoth_learn_cache(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase with hidden_dim=2048, activation_lifecycle=cache."""
        spec = _behemoth_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(
            spec, hw, PlanCceStrategy(), 32, policy,
            activation_lifecycle="cache",
        )
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §11 — The Lexicon (Massive output_classes)
#
# CONCEPT.md: Achieves maximum hardware occupancy by interleaving
# partial gradient computation with reduction stages. 10,000 classes
# exercises class-dimension tiling and multi-stage reduction.
# =========================================================================


class TestBenchLexicon:
    """Benchmark at massive output_classes (10,000).

    KNOWN CRASH: All lexicon-scale tests segfault due to class-dimension
    tiling overflow in the CPU backend.  Regression tests live in
    tests/tier2/cpu/test_cpu_crash_regressions.py::TestLexiconCrashRegression.
    """

    @pytest.mark.skip(reason="SIGSEGV: lexicon spec crashes CPU backend (see test_cpu_crash_regressions.py)")
    @pytest.mark.benchmark(group="lexicon-act")
    @pytest.mark.parametrize("batch_size", [16, 64], ids=["BS=16", "BS=64"])
    def test_bench_lexicon_act(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Act phase with 10,000 output classes."""
        spec = _lexicon_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), batch_size)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.skip(reason="SIGSEGV: lexicon spec crashes CPU backend (see test_cpu_crash_regressions.py)")
    @pytest.mark.benchmark(group="lexicon-learn")
    @pytest.mark.parametrize("batch_size", [16, 64], ids=["BS=16", "BS=64"])
    def test_bench_lexicon_learn(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Learn phase with 10,000 output classes — exercises class-dim reduction."""
        spec = _lexicon_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), batch_size, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.skip(reason="SIGSEGV: lexicon spec crashes CPU backend (see test_cpu_crash_regressions.py)")
    @pytest.mark.benchmark(group="lexicon-learn-bce")
    def test_bench_lexicon_learn_bce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase (BCE) with 10,000 output classes — BCE partial loss reduction."""
        spec = _lexicon_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanBceStrategy(), 16, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §12 — The Rodeo (Extreme Instability Resilience)
#
# CONCEPT.md: Low-precision (FP16) with aggressive learning rate.
# Proves the bifurcated clipping stage (Nodes 11 & 19) as a
# non-optional stability primitive. Tight policy funnel stresses
# every reduction stage.
# =========================================================================


class TestBenchRodeo:
    """Benchmark FP16 plan execution — numerical safety under clipping pressure.

    KNOWN CRASH: All FP16 tests crash due to struct layout mismatch or
    missing half-float support in the CPU backend.  Regression tests live in
    tests/tier2/cpu/test_cpu_crash_regressions.py::TestRodeoCrashRegression.
    """

    @pytest.mark.skip(reason="SIGABRT: FP16 crashes CPU backend (see test_cpu_crash_regressions.py)")
    @pytest.mark.benchmark(group="rodeo-act-fp16")
    def test_bench_rodeo_act_fp16(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Act phase in FP16."""
        spec = _rodeo_spec_fp16(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 64)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.skip(reason="SIGSEGV: FP16 crashes CPU backend (see test_cpu_crash_regressions.py)")
    @pytest.mark.benchmark(group="rodeo-learn-fp16")
    def test_bench_rodeo_learn_fp16(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase in FP16 with aggressive clipping policy."""
        spec = _rodeo_spec_fp16(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_aggressive_policy(_PREC_FP16)
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 64, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.skip(reason="SIGABRT: FP16 crashes CPU backend (see test_cpu_crash_regressions.py)")
    @pytest.mark.benchmark(group="rodeo-learn-fp16-bce")
    def test_bench_rodeo_learn_fp16_bce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase (BCE) in FP16 with aggressive clipping policy."""
        spec = _rodeo_spec_fp16(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_aggressive_policy(_PREC_FP16)
        plan = build_learn_plan(spec, hw, PlanBceStrategy(), 64, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §13 — The Scientist's Repeater (Numerical Correctness Validation)
#
# CONCEPT.md: Two runs with different batch sizes should produce
# similar-magnitude updates thanks to normalize_gradients.  Benchmark
# exercises both batch sizes back-to-back.
# =========================================================================


class TestBenchScientistsRepeater:
    """Benchmark Learn at two batch sizes — validates normalize_gradients cost."""

    @pytest.mark.benchmark(group="repeater-learn")
    @pytest.mark.parametrize("batch_size", [16, 32], ids=["BS=16", "BS=32"])
    def test_bench_repeater_learn(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Learn phase at batch_size N — normalize_gradients ensures scale-invariance."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), batch_size, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §14 — The Data Tsunami (Massive batch_size)
#
# CONCEPT.md: Canonical test of the full parallel learning model.
# One ticket, one Act plan, one Learn plan. The ReductionTreeNode
# handles log_K(N) aggregation over the Placement & Indirection
# Contracts.
# =========================================================================


class TestBenchDataTsunami:
    """Benchmark at large batch_size — exercises reduction tree depth."""

    @pytest.mark.benchmark(group="tsunami-act")
    @pytest.mark.parametrize("batch_size", [256, 512], ids=["BS=256", "BS=512"])
    def test_bench_tsunami_act(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Act phase at large batch size."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), batch_size)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="tsunami-learn")
    @pytest.mark.parametrize("batch_size", [256, 512], ids=["BS=256", "BS=512"])
    def test_bench_tsunami_learn(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, batch_size: int,
    ) -> None:
        """Learn phase at large batch size — full clip→reduce→normalize→update pipeline."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), batch_size, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)


# =========================================================================
# §15 — The Colossus (Holistic Stress Test)
#
# CONCEPT.md: Synergy of all scaling strategies under compound memory
# pressure. Many heads + large classes + decent hidden_dim + large
# batch in both CCE and BCE modes.
# =========================================================================


class TestBenchColossus:
    """Benchmark under compound stress — all subsystems under simultaneous load."""

    @pytest.mark.benchmark(group="colossus-act")
    def test_bench_colossus_act(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Act phase: 64 modules × 1000 classes × hidden=512."""
        spec = _colossus_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        plan = build_act_plan(spec, hw, PlanCceStrategy(), 32)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="colossus-learn-cce")
    def test_bench_colossus_learn_cce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase (CCE): full clip→reduce→stream→normalize→update under compound load."""
        spec = _colossus_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanCceStrategy(), 32, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="colossus-learn-bce")
    def test_bench_colossus_learn_bce(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Learn phase (BCE): BCE partial loss reduction under compound load."""
        spec = _colossus_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        plan = build_learn_plan(spec, hw, PlanBceStrategy(), 32, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="colossus-full-cycle")
    def test_bench_colossus_full_cycle(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile,
    ) -> None:
        """Full Act→Learn cycle under compound stress (CCE)."""
        spec = _colossus_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        act_plan = build_act_plan(spec, hw, PlanCceStrategy(), 32)
        learn_plan = build_learn_plan(spec, hw, PlanCceStrategy(), 32, policy)

        renderer.render(act_plan)
        renderer.render(learn_plan)

        def _cycle():
            renderer.render(act_plan)
            renderer.render(learn_plan)
        benchmark(_cycle)


# =========================================================================
# §16 — The Cross-Backend Arbiter (CCE/BCE Parity)
#
# CONCEPT.md: Same plan rendered through both CCE and BCE strategies.
# Since this is CPU-only, we benchmark both strategies side-by-side
# to expose any strategy-dependent cost differential.
# =========================================================================


class TestBenchCrossStrategyParity:
    """Benchmark CCE vs BCE strategy rendering at identical model scales."""

    @pytest.mark.benchmark(group="strategy-parity-act")
    @pytest.mark.parametrize(
        "strategy_name", ["cce", "bce"], ids=["CCE", "BCE"],
    )
    def test_bench_strategy_parity_act(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, strategy_name: str,
    ) -> None:
        """Act phase: CCE vs BCE on identical iris spec."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        strategy = PlanCceStrategy() if strategy_name == "cce" else PlanBceStrategy()
        plan = build_act_plan(spec, hw, strategy, 150)
        renderer.render(plan)
        benchmark(renderer.render, plan)

    @pytest.mark.benchmark(group="strategy-parity-learn")
    @pytest.mark.parametrize(
        "strategy_name", ["cce", "bce"], ids=["CCE", "BCE"],
    )
    def test_bench_strategy_parity_learn(
        self, benchmark: Any, renderer: CPUPlanRenderer,
        hardware_profile: HardwareProfile, strategy_name: str,
    ) -> None:
        """Learn phase: CCE vs BCE on identical iris spec."""
        spec = _iris_spec(hardware_profile)
        hw = _make_hw(hardware_profile)
        policy = _make_policy()
        strategy = PlanCceStrategy() if strategy_name == "cce" else PlanBceStrategy()
        plan = build_learn_plan(spec, hw, strategy, 150, policy)
        renderer.render(plan)
        benchmark(renderer.render, plan)
