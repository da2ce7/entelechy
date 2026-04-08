# Phase 12: Convergence Testing on Standard Problems — Detailed Plan

**Status:** Not started  
**Phase:** 12 (Phase 4-adjacent quality assurance layer)  
**Objective:** Implement a convergence test suite that validates end-to-end training correctness by running multi-epoch training on canonical classification problems and asserting that the system achieves expected learning outcomes. This extends the ADR-016 test framework from *operational correctness* (does the machinery work?) to *functional correctness* (does training succeed?).  
**Governing ADRs:** ADR-028 (convergence testing — problem selection, convergence criteria, cross-backend validation, CI integration), ADR-016 (test strategy — layered pytest framework, CPU oracle), ADR-018 (user-facing API — WorkTicket/Engine entry point), ADR-008 (precision configuration — precision-aware thresholds)  
**Rollback gate:** All `convergence_fast` tests pass on CPU/FP32. Iris achieves ≥97% accuracy within 100 epochs. At least one BCE problem passes. `pytest -m convergence_fast` completes in <60s total.  
**Dependencies:** Phase 11 (Engine/WorkTicket API — complete), Phase 12A (Optimizer Hyperparameter Configuration — complete), Phase 4 (test harness — complete), Phase 3 (CPU backend — complete). Phase 12 exercises the full Engine → Plan → Renderer → Kernel pipeline; at least one functional backend (CPU) is required.

### Relationship to Other Phases

Phase 12 is positioned as a quality assurance layer that validates the entire training pipeline end-to-end. It sits above the correctness tiers (Phases 1–5) and consumes the user-facing API (Phase 11):

| Phase | Phase 12 interaction |
| :--- | :--- |
| Phase 1 (Plan Model) | Phase 12 exercises the plan builder indirectly through the Engine's `train_batch()` path. Plan structure correctness is assumed (Tier 1 gate passed). |
| Phase 3 (CPU Backend) | Phase 12's primary validation target. CPU/FP32 is the reference configuration for convergence baselines. |
| Phase 4 (Test Harness) | Phase 12 reuses the `_build_config`-driven skip logic, tolerance infrastructure, and conftest path-setup from Phase 4. Convergence tests live alongside but are distinct from Tiers 1–3. |
| Phase 7–10 (Precision) | Phase 12's precision-parameterized tests validate convergence across FP16, FP64, and FP8 configurations when those precisions are enabled. |
| Phase 11 (User-Facing API) | Phase 12 exercises the `Engine.train_batch()` convenience method as the primary training entry point. Convergence tests validate the API's fitness for real training workflows (ADR-018). |
| Phase 2/5 (OpenCL/Vulkan) | Phase 12 validates convergence on GPU backends when available. Cross-backend trajectory comparison detects gross algorithmic differences. |

### ADR-028 Decisions Summary

The following choices are **decided** and govern this phase:

| Choice | Decision | Implication |
| :--- | :--- | :--- |
| **Test Organization** | Option B: Separate category | Convergence tests live in `tests/convergence/` with dedicated markers, separate from Tiers 1–3 |
| **Problem Selection** | Option C: Curated canonical suite | Fixed set of 7 problems with known baselines (Iris, Digits, XOR, Two Moons, Breast Cancer, MNIST-1k, Fashion-1k) |
| **Convergence Criteria** | Option F: Multi-criteria | Accuracy threshold + loss threshold + epoch budget + NaN/Inf check + optional monotonicity |
| **Cross-Backend Validation** | Option H: Independent convergence + trajectory logging | Each backend validated independently; trajectory divergence logged as warning, not failure |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State Assessment](#2-current-state-assessment)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 12.1: Register convergence markers in `pyproject.toml`](#step-121-register-convergence-markers-in-pyprojecttoml)
   - [Step 12.2: Create `tests/convergence/` directory skeleton](#step-122-create-testsconvergence-directory-skeleton)
   - [Step 12.3: Implement convergence infrastructure (`conftest.py`)](#step-123-implement-convergence-infrastructure-conftestpy)
   - [Step 12.4: Implement `ConvergenceCriteria` and `TrainingHistory` dataclasses](#step-124-implement-convergencecriteria-and-traininghistory-dataclasses)
   - [Step 12.5: Implement precision adjustment utilities](#step-125-implement-precision-adjustment-utilities)
   - [Step 12.6: Implement problem loaders (sklearn-bundled)](#step-126-implement-problem-loaders-sklearn-bundled)
   - [Step 12.7: Implement problem loaders (optional dependencies)](#step-127-implement-problem-loaders-optional-dependencies)
   - [Step 12.8: Implement training loop harness](#step-128-implement-training-loop-harness)
   - [Step 12.9: Implement CCE convergence tests](#step-129-implement-cce-convergence-tests)
   - [Step 12.10: Implement BCE convergence tests](#step-1210-implement-bce-convergence-tests)
   - [Step 12.11: Implement precision-parameterized convergence tests](#step-1211-implement-precision-parameterized-convergence-tests)
   - [Step 12.12: Implement cross-backend trajectory comparison](#step-1212-implement-cross-backend-trajectory-comparison)
   - [Step 12.13: Validate `convergence_fast` gate on CPU/FP32](#step-1213-validate-convergence_fast-gate-on-cpufp32)
   - [Step 12.14: Validate rollback gate](#step-1214-validate-rollback-gate)
5. [Convergence Criteria Specification](#5-convergence-criteria-specification)
6. [Problem Suite Reference](#6-problem-suite-reference)
7. [Training Loop Architecture](#7-training-loop-architecture)
8. [Precision Adjustment Tables](#8-precision-adjustment-tables)
9. [Cross-Backend Trajectory Comparison](#9-cross-backend-trajectory-comparison)
10. [CI Integration](#10-ci-integration)
11. [CLI Usage Patterns](#11-cli-usage-patterns)
12. [Test Target Inventory](#12-test-target-inventory)
13. [Risk Register](#13-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating the `tests/convergence/` directory structure with `conftest.py`, problem loaders, and test modules.
- Registering `convergence`, `convergence_fast`, and `convergence_full` pytest markers.
- Implementing `ConvergenceCriteria` and `TrainingHistory` dataclasses for multi-criteria convergence validation.
- Implementing problem loader modules for 5 sklearn-bundled problems (Iris, Digits, XOR, Two Moons, Breast Cancer) and 2 optional-dependency problems (MNIST-1k, Fashion-1k).
- Implementing a reusable training loop harness that uses the `Engine.train_batch()` API to execute multi-epoch training and collect per-epoch metrics.
- Implementing CCE convergence tests (`test_convergence_cce.py`) for Iris, Digits, and (optionally) MNIST-1k / Fashion-1k.
- Implementing BCE convergence tests (`test_convergence_bce.py`) for XOR, Two Moons, and Breast Cancer.
- Implementing precision-parameterized tests (`test_convergence_precision.py`) that adjust convergence criteria per `PrecisionConfig`.
- Implementing cross-backend trajectory comparison with warning-level (not failure-level) divergence detection.
- Configuring `pyproject.toml` markers and implementing `pytest_collection_modifyitems` to exclude `convergence_full` from default runs.
- Implementing `BaselineHyperparameters` configurations for each problem (ADR-028 §Hyperparameter Baselines).
- Implementing reproducibility infrastructure: fixed random seeds, dataset integrity hashing.

### Out of scope

- Implementing or modifying any backend renderer — convergence tests consume existing backends.
- Implementing or modifying kernels — convergence tests exercise the training pipeline as-is.
- Event-Triggered mode convergence testing — only Sequential mode (`train_batch()`) is tested.
- Hyperparameter sensitivity analysis or automated sweeps — tests use fixed baseline hyperparameters.
- Generalization/test-set accuracy validation — tests measure training accuracy only.
- Regression detection (accuracy tracking across commits) — deferred to future infrastructure.
- Performance benchmarking — convergence tests measure correctness, not speed.

### Key constraint: Engine API as sole entry point

All convergence tests use `Engine.train_batch()` as the training entry point. This validates the user-facing API path (ADR-018) and ensures convergence tests exercise the full Engine → Plan Builder → PlanRenderer → Kernel pipeline. No direct renderer or plan builder calls are made in convergence test bodies.

### Key constraint: sklearn-only core suite

The 5 core problems (Iris, Digits, XOR, Two Moons, Breast Cancer) use only sklearn-bundled datasets or deterministic synthetic generation. These require no external downloads and no optional dependencies beyond `scikit-learn` (already a project dependency). MNIST and Fashion-MNIST are gated behind `pytest.importorskip("torchvision")` and excluded from the core convergence gate.

### Key constraint: reproducibility

Data shuffling and synthetic data generation are seeded with fixed seeds. Weight initialization is currently zero-initialized by the buffer allocator — `ParameterSpace` is a layout manifest (not an initializer) and carries no random state. If the renderer or plan evolves to support non-trivial weight initialization, a seed parameter must be added to `Engine` or the renderer factory. Tests must produce identical results across runs on the same platform. GPU backends with non-deterministic reduction ordering are validated independently (ADR-028 Option H) — trajectory divergence is logged as a warning, not a failure.

### Key constraint: optimizer hyperparameters via `OptimizerConfig`

Phase 12A (ADR-029, complete) introduced `OptimizerConfig`, a frozen dataclass that makes Adam optimizer hyperparameters (`learning_rate`, `beta1`, `beta2`, `epsilon`) configurable through the `Engine` constructor and `build_learn_plan()`. Convergence tests exercise per-problem learning rates by passing `OptimizerConfig(learning_rate=baseline.learning_rate)` to `make_engine()`. The `BaselineHyperparameters` fields `learning_rate`, `beta1`, `beta2`, and `epsilon` are now plumbed through to plan-node scalar parameters.

Weight decay (`weight_decay`) remains unplumbed — `OptimizerConfig` does not include a weight decay field. The `weight_decay` field in `BaselineHyperparameters` is retained for documentation only.

### Key constraint: CI runtime budget

`convergence_fast` tests must complete in <60s total (CPU/FP32). This constrains epoch budgets for fast tests to ~20–50 epochs depending on problem size. `convergence_full` tests have a relaxed budget of ~30 minutes total for the nightly suite.

---

## 2. Current State Assessment

### Existing infrastructure consumed by Phase 12

| Component | Status | Location | Phase 12 usage |
| :--- | :--- | :--- | :--- |
| `Engine.train_batch()` | ✅ Complete (Phase 11) | `src/shared/engine.py` | Primary training entry point |
| `WorkTicket` lifecycle | ✅ Complete (Phase 11) | `src/shared/ticket.py` | Exercised implicitly via `train_batch()` |
| CPU `PlanRenderer` | ✅ Complete (Phase 3) | `src/backends/cpu/` | Reference backend for convergence baselines |
| `_build_config` skip logic | ✅ Complete (Phase 4) | `tests/conftest.py` | Backend-gated test collection |
| `ModelSpec` factory methods | ✅ Complete | `src/shared/model_spec.py` | `.float32()`, `.mixed_f16_f32()` constructors |
| `PrecisionConfig` | ✅ Complete (Phase 7) | `src/shared/precision_config.py` | Precision-parameterized criteria adjustment |
| `ParameterSpace` | ✅ Complete | `src/shared/parameter_space.py` | Memory layout manifest for parameter buffers (not an initializer — carries no weight values or random state) |
| `HardwareProfile` | ✅ Complete | `src/shared/hardware_profile.py` | Backend configuration |
| `StabilizationPolicy` | ✅ Complete | `src/shared/stabilization_policy.py` | Gradient clipping during training |
| `PlanCceStrategy` / `PlanBceStrategy` | ✅ Complete | `src/shared/problem_type_strategy.py` | CCE vs BCE mode selection |
| `fp32_iris_spec` fixture | ✅ Complete | `tests/conftest.py` | Reusable for Iris convergence baseline |
| Tolerance config | ✅ Complete (Phase 4) | `tests/tolerance_config.py` | Not directly used (convergence uses accuracy/loss thresholds, not per-kernel numerical tolerances) |
| Pytest marker infrastructure | ✅ Complete (Phase 4) | `pyproject.toml` | Extended with convergence markers |

### Infrastructure not yet created

| Component | Phase 12 step |
| :--- | :--- |
| `tests/convergence/` directory tree | Step 12.2 |
| `convergence` / `convergence_fast` / `convergence_full` markers | Step 12.1 |
| `ConvergenceCriteria` / `TrainingHistory` dataclasses | Step 12.4 |
| Problem loader modules | Steps 12.6, 12.7 |
| Training loop harness | Step 12.8 |
| Convergence test files | Steps 12.9, 12.10, 12.11 |
| Cross-backend trajectory comparison | Step 12.12 |

---

## 3. Target Deliverables

After Phase 12 completes, the test directory gains:

```
tests/
├── convergence/
│   ├── __init__.py
│   ├── conftest.py                        # Engine fixtures, backend setup, skip logic
│   ├── criteria.py                        # ConvergenceCriteria, TrainingHistory, assert_convergence()
│   ├── precision_adjustments.py           # PRECISION_ADJUSTMENTS table, adjust_criteria_for_precision()
│   ├── training_harness.py                # run_training_loop() — multi-epoch Engine.train_batch() driver
│   ├── trajectory_comparison.py           # compare_trajectories() — cross-backend warning logic
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── iris.py                        # Iris dataset loader + baseline config
│   │   ├── digits.py                      # sklearn digits dataset + baseline config
│   │   ├── xor.py                         # XOR synthetic generator (seed=42) + baseline config
│   │   ├── two_moons.py                   # Two Moons synthetic generator + baseline config
│   │   ├── breast_cancer.py               # sklearn breast cancer dataset + baseline config
│   │   ├── mnist.py                       # MNIST-1k subset loader (torchvision optional) + baseline config
│   │   └── fashion_mnist.py               # Fashion-MNIST-1k loader (torchvision optional) + baseline config
│   ├── test_convergence_cce.py            # CCE-mode convergence tests
│   ├── test_convergence_bce.py            # BCE-mode convergence tests
│   └── test_convergence_precision.py      # Precision-parameterized convergence tests
```

Additionally:
- `pyproject.toml` gains `convergence`, `convergence_fast`, and `convergence_full` marker registrations.
- `tests/convergence/conftest.py` implements a `pytest_collection_modifyitems` hook that skips `convergence_full` tests unless explicitly selected via `-m`.

---

## 4. Task Breakdown

### Step 12.1: Register convergence markers in `pyproject.toml`

**Action:** Add convergence marker definitions to `pyproject.toml`. Default exclusion of `convergence_full` from runs is handled by the `pytest_collection_modifyitems` hook in Step 12.3, not by `addopts`.

**File:** `architectures/averaging_ensembled_classifier/pyproject.toml`

**Markers to add:**

| Marker | Description |
| :--- | :--- |
| `convergence` | All convergence tests (multi-epoch training) |
| `convergence_fast` | Fast convergence smoke tests (<30s each) |
| `convergence_full` | Full convergence tests (1–5 min each) |

**Configuration update:**

```toml
[tool.pytest.ini_options]
markers = [
    "tier1: Host-side plan correctness (no backend required)",
    "tier2: Per-backend kernel correctness (requires enabled backend)",
    "tier3: Cross-backend parity (requires >= 2 backends)",
    "cpu: Requires CPU backend",
    "opencl: Requires OpenCL backend",
    "vulkan: Requires Vulkan backend",
    "slow: Long-running test",
    "convergence: All convergence tests (multi-epoch training)",
    "convergence_fast: Fast convergence smoke tests (<30s each)",
    "convergence_full: Full convergence tests (1-5 min each)",
]
```

**`convergence_full` exclusion via collection hook (not `addopts`):**

The `addopts = "-m 'not convergence_full'"` approach was considered but rejected: pytest CLI `-m` flags *replace* rather than stack with `addopts`, so running `pytest -m tier1` would lose the exclusion filter and collect `convergence_full` tests. Instead, `convergence_full` exclusion is implemented as a `pytest_collection_modifyitems` hook in `tests/convergence/conftest.py`:

```python
def pytest_collection_modifyitems(config, items):
    """Skip convergence_full unless explicitly selected via -m."""
    # If the user explicitly selected convergence markers, respect that
    markexpr = config.getoption("-m", default="")
    if "convergence_full" in markexpr or "convergence" == markexpr.strip():
        return
    skip_full = pytest.mark.skip(reason="convergence_full excluded from default runs; use -m convergence_full")
    for item in items:
        if "convergence_full" in item.keywords and "convergence_fast" not in item.keywords:
            item.add_marker(skip_full)
```

This approach is compatible with all existing `-m` usage patterns and does not require modifying `addopts`.

**Validation:** `pytest --strict-markers --collect-only` reports zero marker warnings. `convergence_full` tests show as SKIPPED in default runs.

---

### Step 12.2: Create `tests/convergence/` directory skeleton

**Action:** Create the convergence test directory structure with empty `__init__.py` files.

**Files to create:**

```
tests/convergence/
├── __init__.py
├── problems/
│   └── __init__.py
```

All other files are created in subsequent steps. The `__init__.py` files are empty package markers.

---

### Step 12.3: Implement convergence infrastructure (`conftest.py`)

**Action:** Create `tests/convergence/conftest.py` with Engine construction fixtures, backend parameterization, and convergence-specific skip logic.

**File:** `tests/convergence/conftest.py`

**Design:**

The convergence conftest provides:

1. **Engine factory fixture** — constructs an `Engine` for a given problem configuration (ModelSpec, ParameterSpace, HardwareProfile, strategy) on the requested backend.
2. **Backend parameterization** — fixture that yields each available backend name for cross-backend tests.
3. **Reproducibility setup** — fixture that sets numpy random seed before each test.
4. **Convergence skip logic** — skip convergence tests if no backend is available.

```python
# tests/convergence/conftest.py
"""Convergence test infrastructure (ADR-028)."""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.engine import Engine
from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.precision_config import PrecisionConfig
from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy
from src.shared.stabilization_policy import StabilizationPolicy


# ---------------------------------------------------------------------------
# Backend availability (mirrors root conftest pattern — ADR-014, ADR-016)
# ---------------------------------------------------------------------------
def _load_build_config() -> dict[str, bool]:
    """Load the build manifest; return a dict of backend availability."""
    try:
        from src._build_config import BACKEND_CPU, BACKEND_OPENCL, BACKEND_VULKAN  # type: ignore[import-not-found]
        return {
            "cpu": BACKEND_CPU,
            "opencl": BACKEND_OPENCL,
            "vulkan": BACKEND_VULKAN,
        }
    except ImportError:
        raise RuntimeError(
            "_build_config.py not found. Run 'meson setup builddir' before testing."
        )


BUILD_CONFIG = _load_build_config()

AVAILABLE_BACKENDS: list[str] = [
    name for name, available in BUILD_CONFIG.items() if available
]


def pytest_collection_modifyitems(config, items):
    """Skip convergence tests based on backend availability and marker selection."""
    # 1. Skip all convergence tests if no backend is available
    if not AVAILABLE_BACKENDS:
        skip = pytest.mark.skip(reason="No backends available for convergence testing")
        for item in items:
            if "convergence" in item.keywords:
                item.add_marker(skip)
        return

    # 2. Skip convergence_full unless explicitly selected via -m
    markexpr = config.getoption("-m", default="")
    if "convergence_full" not in markexpr and markexpr.strip() != "convergence":
        skip_full = pytest.mark.skip(
            reason="convergence_full excluded from default runs; use -m convergence_full"
        )
        for item in items:
            if "convergence_full" in item.keywords and "convergence_fast" not in item.keywords:
                item.add_marker(skip_full)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reproducibility_seed():
    """Set deterministic random state for data shuffling/synthetic generation."""
    np.random.seed(42)
    yield


@pytest.fixture
def cpu_hardware_profile() -> HardwareProfile:
    """Hardware profile for CPU backend convergence testing."""
    return HardwareProfile(
        simd_width=4,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=None,  # CPU has no local memory concept
        global_mem_bytes=4 * 1024**3,
    )


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend_name(request) -> str:
    """Yield each available backend name for cross-backend tests."""
    return request.param


def make_engine(
    *,
    input_dim: int,
    hidden_dim: int,
    output_classes: int,
    num_modules: int,
    mode: str,
    backend: str = "cpu",
    precision: PrecisionConfig | None = None,
    gradient_clip_threshold: float = 1.0,
    optimizer: OptimizerConfig | None = None,
) -> Engine:
    """Construct an Engine for convergence testing.

    Not a fixture — called directly by test functions to allow
    per-problem parameterization.

    Optimizer hyperparameters are forwarded via OptimizerConfig (ADR-029).
    """
    hardware = HardwareProfile(
        simd_width=4,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=None,
        global_mem_bytes=4 * 1024**3,
    )
    spec = ModelSpec.float32(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_classes=output_classes,
        num_modules=num_modules,
        simd_width=4,
        cache_line_bytes=64,
    )
    param_space = ParameterSpace(spec)

    if mode == "CCE":
        strategy = PlanCceStrategy()
    elif mode == "BCE":
        strategy = PlanBceStrategy()
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    policy = StabilizationPolicy(
        t_algorithmic=gradient_clip_threshold,
        lambda_=0.1,
        compute_fp_format_max=spec.precision.compute_fp_format_max,
    )

    return Engine(
        model_spec=spec,
        parameter_space=param_space,
        hardware_profile=hardware,
        precision=precision,
        backend=backend,
        strategy=strategy,
        policy=policy,
        optimizer=optimizer,
    )
```

**Key decisions:**

1. **`make_engine` as a function, not a fixture.** Each problem has different ModelSpec dimensions. A fixture would require complex parameterization chains. A plain function is more flexible — each test constructs its own Engine with problem-specific dimensions.

2. **`ParameterSpace(spec)`** — constructs the layout manifest. `ParameterSpace` is a declarative manifest that generates buffer name mappings from a `ModelSpec`; it does not initialize weight values. Weight initialization is controlled by the buffer allocator (currently zero-initialized). Reproducibility of initial weights is therefore automatic (deterministic zero state).

3. **`HardwareProfile` direct construction** — all 5 fields are required (frozen dataclass). Values match those used in existing test fixtures and `discover_hardware()` defaults. `max_local_mem_bytes=None` reflects CPU's lack of local memory; `max_reduce_fan_in=256` matches `cpu/discovery.py`.

4. **`_build_config` wrapped import** — follows the root `tests/conftest.py` pattern with `ImportError` handling and clear error message, rather than top-level import that produces an inscrutable traceback.

5. **Optimizer hyperparameters are configurable via `OptimizerConfig` (ADR-029, Phase 12A).** The `make_engine` helper constructs an `OptimizerConfig` from `BaselineHyperparameters` fields (`learning_rate`, `beta1`, `beta2`, `epsilon`) and passes it to the `Engine` constructor, which forwards it to `build_learn_plan()`. Weight decay is not yet plumbed.

6. **`convergence_full` exclusion** is handled by the `pytest_collection_modifyitems` hook in this file, not by `pyproject.toml` `addopts`. This avoids the pytest `-m` replacement problem (CLI `-m` replaces `addopts` `-m` rather than stacking).

**Validation:** `pytest tests/convergence/ --collect-only` collects convergence tests with correct skip behavior.

---

### Step 12.4: Implement `ConvergenceCriteria` and `TrainingHistory` dataclasses

**Action:** Create `tests/convergence/criteria.py` containing the multi-criteria convergence validation logic.

**File:** `tests/convergence/criteria.py`

**Types to implement:**

```python
@dataclass(frozen=True)
class ConvergenceCriteria:
    """Multi-criteria convergence specification for a problem (ADR-028 Option F)."""
    problem_id: str
    mode: Literal["CCE", "BCE"]
    accuracy_threshold: float          # minimum final accuracy
    loss_threshold: float              # maximum final loss
    epoch_budget: int                  # must converge within this many epochs
    warmup_epochs: int                 # epochs before monotonicity check begins
    monotonicity_tolerance: float | None  # relative tolerance for loss decrease; None disables check
    nan_inf_allowed: bool = False      # always False for production

@dataclass
class TrainingHistory:
    """Per-epoch training metrics for convergence validation."""
    loss_curve: list[float]            # loss at end of each epoch
    accuracy_curve: list[float]        # accuracy at end of each epoch
    has_nan_inf: bool                  # True if any NaN/Inf encountered

    @property
    def final_loss(self) -> float: ...

    @property
    def final_accuracy(self) -> float: ...

    def epochs_to_threshold(self, accuracy_threshold: float) -> int: ...
```

**Assertion function:**

```python
def assert_convergence(history: TrainingHistory, criteria: ConvergenceCriteria) -> None:
    """Assert that training history meets all convergence criteria.

    Checks (in order):
    1. No NaN/Inf in any intermediate value.
    2. Final accuracy >= accuracy_threshold.
    3. Final loss <= loss_threshold.
    4. Convergence achieved within epoch_budget.
    5. Loss monotonicity after warmup (if monotonicity_tolerance is not None).

    Raises:
        AssertionError: If any criterion is not met.
    """
```

The implementation follows ADR-028 §Convergence Criteria verbatim. The monotonicity check uses relative tolerance: `loss[i+1] ≤ loss[i] * (1 + monotonicity_tolerance)`.

**Validation:** Unit tests for `assert_convergence()` with synthetic `TrainingHistory` objects exercise each failure mode.

---

### Step 12.5: Implement precision adjustment utilities

**Action:** Create `tests/convergence/precision_adjustments.py` containing the precision-specific criteria adjustment logic.

**File:** `tests/convergence/precision_adjustments.py`

**Adjustment table (ADR-028 §Precision-Specific Adjustments):**

```python
@dataclass(frozen=True)
class PrecisionAdjustment:
    accuracy_delta: float     # added to accuracy_threshold
    loss_delta: float         # added to loss_threshold
    epoch_multiplier: float   # multiplied with epoch_budget

PRECISION_ADJUSTMENTS: dict[str, PrecisionAdjustment] = {
    "float32": PrecisionAdjustment(accuracy_delta=0.0, loss_delta=0.0, epoch_multiplier=1.0),
    "float16": PrecisionAdjustment(accuracy_delta=-0.02, loss_delta=0.05, epoch_multiplier=1.2),
    "float64": PrecisionAdjustment(accuracy_delta=0.0, loss_delta=-0.01, epoch_multiplier=0.9),
    "float8_e4m3": PrecisionAdjustment(accuracy_delta=-0.05, loss_delta=0.1, epoch_multiplier=1.5),
}
```

**Adjustment function:**

```python
def adjust_criteria_for_precision(
    base: ConvergenceCriteria,
    precision: PrecisionConfig,
) -> ConvergenceCriteria:
    """Return a new ConvergenceCriteria with precision-specific adjustments applied."""
    adjustments = PRECISION_ADJUSTMENTS[precision.storage_dtype.name]
    return ConvergenceCriteria(
        problem_id=base.problem_id,
        mode=base.mode,
        accuracy_threshold=base.accuracy_threshold + adjustments.accuracy_delta,
        loss_threshold=base.loss_threshold + adjustments.loss_delta,
        epoch_budget=int(base.epoch_budget * adjustments.epoch_multiplier),
        warmup_epochs=base.warmup_epochs,
        monotonicity_tolerance=base.monotonicity_tolerance,
        nan_inf_allowed=base.nan_inf_allowed,
    )
```

**Validation:** Unit tests verify that FP16 criteria are relaxed relative to FP32, and FP64 criteria are tightened.

---

### Step 12.6: Implement problem loaders (sklearn-bundled)

**Action:** Create problem loader modules for the 5 sklearn-bundled problems. Each module exports a `load()` function returning `(X, y)` and a `BASELINE` configuration constant.

**Files to create:**

| File | Problem | Mode | Source |
| :--- | :--- | :--- | :--- |
| `tests/convergence/problems/iris.py` | Iris | CCE | `sklearn.datasets.load_iris()` |
| `tests/convergence/problems/digits.py` | Digits (8×8) | CCE | `sklearn.datasets.load_digits()` |
| `tests/convergence/problems/xor.py` | XOR | BCE | Synthetic (seed=42) |
| `tests/convergence/problems/two_moons.py` | Two Moons | BCE | `sklearn.datasets.make_moons(seed=42)` |
| `tests/convergence/problems/breast_cancer.py` | Breast Cancer | BCE | `sklearn.datasets.load_breast_cancer()` |

**Common module structure:**

Each problem module provides:

1. **`load() → tuple[NDArray, NDArray]`** — returns `(X, y)` with X as float32 and y as appropriate integer type. Applies standard preprocessing (feature scaling to [0, 1] or standardization as appropriate).
2. **`BASELINE: BaselineHyperparameters`** — frozen dataclass with hyperparameters known to converge.
3. **`FAST_CRITERIA: ConvergenceCriteria`** — relaxed smoke-test criteria.
4. **`FULL_CRITERIA: ConvergenceCriteria`** — strict production-quality criteria.
5. **`PROBLEM_ID: str`** — unique identifier for the problem.
6. **`MODE: Literal["CCE", "BCE"]`** — problem type.

**Example — `tests/convergence/problems/iris.py`:**

```python
"""Iris convergence problem (ADR-028)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "iris"
MODE: Literal["CCE", "BCE"] = "CCE"

BASELINE = BaselineHyperparameters(
    hidden_size=16,
    learning_rate=0.01,         # exercised via OptimizerConfig (ADR-029)
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.0,
    gradient_clip_threshold=1.0,
    batch_size=32,
)

FAST_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.90,
    loss_threshold=0.5,
    epoch_budget=20,
    warmup_epochs=3,
    monotonicity_tolerance=None,  # disabled for fast tests
)

FULL_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.97,
    loss_threshold=0.15,
    epoch_budget=100,
    warmup_epochs=10,
    monotonicity_tolerance=0.05,
)

def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load the Iris dataset, scaled to [0, 1]."""
    data = load_iris()
    X = data.data.astype(np.float32)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    y = data.target.astype(np.int64)
    return X, y
```

**XOR synthetic generator (`tests/convergence/problems/xor.py`):**

```python
def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Generate XOR classification problem (seed=42, 200 samples)."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.int64)
    return X, y
```

**Preprocessing decisions:**

| Problem | Preprocessing | Rationale |
| :--- | :--- | :--- |
| Iris | Min-max scale to [0, 1] | Features have different scales; normalization prevents gradient magnitude imbalance |
| Digits | Divide by 16.0 (pixels in [0, 16]) | Pre-scaled to [0, 1] range |
| XOR | Already in [-1, 1] | Synthetic, no scaling needed |
| Two Moons | Standardize (zero mean, unit variance) | sklearn output has arbitrary scale |
| Breast Cancer | Standardize (zero mean, unit variance) | Features span orders of magnitude (mean radius ~14 vs. fractal dimension ~0.06) |

**`BaselineHyperparameters` dataclass** (defined in `criteria.py`):

```python
@dataclass(frozen=True)
class BaselineHyperparameters:
    """Hyperparameter configuration for a problem (ADR-028).

    ``learning_rate``, ``beta1``, ``beta2``, and ``epsilon`` are
    exercised by convergence tests through ``make_engine()`` via
    ``OptimizerConfig`` (ADR-029, Phase 12A).  ``weight_decay`` is
    retained for documentation only — ``OptimizerConfig`` does not
    include weight decay.
    """
    hidden_size: int
    learning_rate: float                # Exercised via OptimizerConfig (ADR-029)
    beta1: float                        # Exercised via OptimizerConfig (ADR-029)
    beta2: float                        # Exercised via OptimizerConfig (ADR-029)
    epsilon: float                      # Exercised via OptimizerConfig (ADR-029)
    weight_decay: float                 # NOT YET PLUMBED — hardcoded at 0.0
    gradient_clip_threshold: float      # Exercised via StabilizationPolicy.t_algorithmic
    batch_size: int                     # Exercised via training_harness batch iteration
```

**Validation:** Each loader is tested in isolation — `load()` returns arrays with expected shapes and dtypes; `BASELINE`, `FAST_CRITERIA`, and `FULL_CRITERIA` are well-formed.

---

### Step 12.7: Implement problem loaders (optional dependencies)

**Action:** Create problem loader modules for MNIST-1k and Fashion-MNIST-1k, gated behind `torchvision`.

**Files to create:**

| File | Problem | Mode | Source |
| :--- | :--- | :--- | :--- |
| `tests/convergence/problems/mnist.py` | MNIST (1k subset) | CCE | `torchvision.datasets.MNIST` |
| `tests/convergence/problems/fashion_mnist.py` | Fashion-MNIST (1k subset) | CCE | `torchvision.datasets.FashionMNIST` |

**Skip logic:**

```python
def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load MNIST 1k subset; skip test if torchvision unavailable."""
    pytest = __import__("pytest")
    torchvision = pytest.importorskip("torchvision", reason="MNIST requires torchvision")
    ...
```

**Dataset integrity hashing (ADR-028 §Reproducibility Guarantees):**

Each optional loader computes a SHA-256 hash of the loaded subset and compares against a known hash. The known hashes are computed during initial implementation and hardcoded into the loader modules. This detects:
- `torchvision` version changes that alter dataset contents.
- Corruption or tampering with cached data files.

```python
def _verify_integrity(X: NDArray, y: NDArray, expected_hash: str) -> None:
    import hashlib
    data_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    assert f"sha256:{data_hash}" == expected_hash, (
        f"Dataset hash mismatch — expected {expected_hash}, got sha256:{data_hash}"
    )
```

The hashes are left as `"sha256:<computed-during-implementation>"` placeholders in the plan; their actual values are determined at implementation time by running the loaders once and recording the output hashes.

**Validation:** Loaders produce arrays with expected shapes `(1000, 784)` and `(1000,)`. Tests are skipped cleanly when `torchvision` is absent.

---

### Step 12.8: Implement training loop harness

**Action:** Create `tests/convergence/training_harness.py` containing the reusable multi-epoch training driver.

**File:** `tests/convergence/training_harness.py`

**Function signature:**

```python
def run_training_loop(
    engine: Engine,
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    epochs: int,
    batch_size: int,
    shuffle_seed: int = 42,
) -> TrainingHistory:
    """Execute multi-epoch training using Engine.train_batch().

    For each epoch:
    1. Shuffle training data (deterministic, seeded per epoch).
    2. Iterate over mini-batches, calling engine.train_batch(X_batch, y_batch).
    3. After all batches, compute epoch-level metrics:
       - Loss: mean batch loss across the epoch.
       - Accuracy: fraction of correct predictions across the epoch.
    4. Check for NaN/Inf in predictions.

    Returns a TrainingHistory recording per-epoch loss and accuracy curves.
    """
```

**Implementation details:**

1. **Epoch-level shuffling.** A per-epoch RNG is created as `np.random.default_rng(shuffle_seed + epoch)` to ensure deterministic shuffling that varies across epochs but is reproducible across runs.

2. **Batch iteration.** Training data is split into contiguous mini-batches of `batch_size`. The final batch may be smaller than `batch_size` if the dataset size is not evenly divisible.

3. **Prediction collection.** `train_batch()` returns prediction probabilities for each batch. The harness collects these to compute per-epoch accuracy.

4. **Loss collection.** The architecture's `train_batch()` returns predictions, not loss. The harness computes loss externally using numpy:
   - CCE: `−Σ y_onehot * log(pred + ε)` averaged over samples
   - BCE: `−Σ [y * log(pred + ε) + (1−y) * log(1−pred + ε)]` averaged over samples

   This external loss computation is consistent with the architecture's internal loss computation but computed in FP64 (numpy default) for numerical stability. It serves as an independent verification that the loss is decreasing.

5. **Accuracy computation.**
   - CCE: `(argmax(pred, axis=1) == y).mean()`
   - BCE: `((pred > 0.5).astype(int).squeeze() == y).mean()`

6. **NaN/Inf detection.** After each `train_batch()` call, the prediction is checked for NaN/Inf. If detected, `TrainingHistory.has_nan_inf` is set to `True` and training is terminated early.

**Design rationale — external loss computation:**

The `Engine.train_batch()` API (ADR-018) returns predictions only; loss is an internal quantity computed by the loss kernel (`compute_probs_loss_cce_chunk` / `compute_probs_loss_bce_chunk`) and consumed by the gradient kernels. The harness recomputes loss externally to provide an independent convergence signal. This is intentional: it validates that the *observable behavior* (predictions improving over epochs) is consistent with the *internal behavior* (loss decreasing within the kernel pipeline).

**Validation:** Run the harness on Iris with known-good hyperparameters; verify that accuracy increases and loss decreases over 20 epochs.

---

### Step 12.9: Implement CCE convergence tests

**Action:** Create `tests/convergence/test_convergence_cce.py` containing convergence tests for all CCE-mode problems.

**File:** `tests/convergence/test_convergence_cce.py`

**Test structure:**

```python
# tests/convergence/test_convergence_cce.py
"""CCE-mode convergence tests (ADR-028)."""
import pytest

from .conftest import make_engine
from .criteria import assert_convergence
from .training_harness import run_training_loop
from .problems import iris, digits

# -------------------------------------------------------------------------
# Fast tests — every commit
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_iris_convergence_fast():
    """Iris CCE smoke test: basic training progress on CPU/FP32."""
    X, y = iris.load()
    engine = make_engine(
        input_dim=4,
        hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3,
        num_modules=8,
        mode="CCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=iris.FAST_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
    )
    assert_convergence(history, iris.FAST_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_digits_convergence_fast():
    """Digits CCE smoke test: mid-scale training progress on CPU/FP32."""
    X, y = digits.load()
    engine = make_engine(
        input_dim=64,
        hidden_dim=digits.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=digits.FAST_CRITERIA.epoch_budget,
        batch_size=digits.BASELINE.batch_size,
    )
    assert_convergence(history, digits.FAST_CRITERIA)


# -------------------------------------------------------------------------
# Full tests — nightly/release
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_iris_convergence_full():
    """Iris CCE full convergence: 97% accuracy within 100 epochs on CPU/FP32."""
    X, y = iris.load()
    engine = make_engine(
        input_dim=4,
        hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3,
        num_modules=8,
        mode="CCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=iris.FULL_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
    )
    assert_convergence(history, iris.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_digits_convergence_full():
    """Digits CCE full convergence: 95% accuracy within 150 epochs on CPU/FP32."""
    X, y = digits.load()
    engine = make_engine(
        input_dim=64,
        hidden_dim=digits.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=digits.FULL_CRITERIA.epoch_budget,
        batch_size=digits.BASELINE.batch_size,
    )
    assert_convergence(history, digits.FULL_CRITERIA)


# -------------------------------------------------------------------------
# Optional: MNIST and Fashion-MNIST (torchvision required)
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_mnist_convergence_full():
    """MNIST-1k CCE: 92% accuracy within 100 epochs on CPU/FP32."""
    from .problems import mnist
    X, y = mnist.load()  # skips if torchvision unavailable
    engine = make_engine(
        input_dim=784,
        hidden_dim=mnist.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=mnist.FULL_CRITERIA.epoch_budget,
        batch_size=mnist.BASELINE.batch_size,
    )
    assert_convergence(history, mnist.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_fashion_mnist_convergence_full():
    """Fashion-MNIST-1k CCE: 85% accuracy within 150 epochs on CPU/FP32."""
    from .problems import fashion_mnist
    X, y = fashion_mnist.load()  # skips if torchvision unavailable
    engine = make_engine(
        input_dim=784,
        hidden_dim=fashion_mnist.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=fashion_mnist.FULL_CRITERIA.epoch_budget,
        batch_size=fashion_mnist.BASELINE.batch_size,
    )
    assert_convergence(history, fashion_mnist.FULL_CRITERIA)
```

**Cross-backend CCE tests:**

In addition to the CPU-only tests above, each CCE problem has a cross-backend variant that uses the `backend_name` parameterized fixture:

```python
@pytest.mark.convergence
@pytest.mark.convergence_full
def test_iris_convergence_cross_backend(backend_name):
    """Iris CCE convergence on each available backend."""
    X, y = iris.load()
    engine = make_engine(
        input_dim=4,
        hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3,
        num_modules=8,
        mode="CCE",
        backend=backend_name,
    )
    history = run_training_loop(
        engine, X, y,
        epochs=iris.FULL_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
    )
    assert_convergence(history, iris.FULL_CRITERIA)
```

**Validation:** `test_iris_convergence_fast` passes on CPU/FP32 with ≥90% accuracy within 20 epochs.

---

### Step 12.10: Implement BCE convergence tests

**Action:** Create `tests/convergence/test_convergence_bce.py` containing convergence tests for all BCE-mode problems.

**File:** `tests/convergence/test_convergence_bce.py`

**Test structure mirrors Step 12.9** with BCE-specific problems:

```python
# tests/convergence/test_convergence_bce.py
"""BCE-mode convergence tests (ADR-028)."""
import pytest

from .conftest import make_engine
from .criteria import assert_convergence
from .training_harness import run_training_loop
from .problems import xor, two_moons, breast_cancer


# -------------------------------------------------------------------------
# Fast tests — every commit
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_xor_convergence_fast():
    """XOR BCE smoke test: basic non-linear convergence on CPU/FP32."""
    X, y = xor.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=xor.BASELINE.hidden_size,
        output_classes=1,  # BCE mode: single sigmoid output
        num_modules=8,
        mode="BCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=xor.FAST_CRITERIA.epoch_budget,
        batch_size=xor.BASELINE.batch_size,
    )
    assert_convergence(history, xor.FAST_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_breast_cancer_convergence_fast():
    """Breast Cancer BCE smoke test: moderate-dim convergence on CPU/FP32."""
    X, y = breast_cancer.load()
    engine = make_engine(
        input_dim=30,
        hidden_dim=breast_cancer.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=breast_cancer.FAST_CRITERIA.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
    )
    assert_convergence(history, breast_cancer.FAST_CRITERIA)


# -------------------------------------------------------------------------
# Full tests — nightly/release
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_xor_convergence_full():
    """XOR BCE full: 99% accuracy within 200 epochs on CPU/FP32."""
    X, y = xor.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=xor.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=xor.FULL_CRITERIA.epoch_budget,
        batch_size=xor.BASELINE.batch_size,
    )
    assert_convergence(history, xor.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_two_moons_convergence_full():
    """Two Moons BCE full: 98% accuracy within 150 epochs on CPU/FP32."""
    X, y = two_moons.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=two_moons.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=two_moons.FULL_CRITERIA.epoch_budget,
        batch_size=two_moons.BASELINE.batch_size,
    )
    assert_convergence(history, two_moons.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_breast_cancer_convergence_full():
    """Breast Cancer BCE full: 96% accuracy within 100 epochs on CPU/FP32."""
    X, y = breast_cancer.load()
    engine = make_engine(
        input_dim=30,
        hidden_dim=breast_cancer.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
    )
    history = run_training_loop(
        engine, X, y,
        epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
    )
    assert_convergence(history, breast_cancer.FULL_CRITERIA)
```

**Cross-backend BCE tests** follow the same pattern as CCE (Step 12.9), parameterized over `backend_name`.

**Validation:** `test_xor_convergence_fast` passes on CPU/FP32 with ≥95% accuracy within 50 epochs. This validates the BCE mode training path end-to-end.

---

### Step 12.11: Implement precision-parameterized convergence tests

**Action:** Create `tests/convergence/test_convergence_precision.py` containing convergence tests parameterized over `PrecisionConfig`.

**File:** `tests/convergence/test_convergence_precision.py`

**Design:**

Precision-parameterized tests use the representative subset of problems (Iris for CCE, Breast Cancer for BCE) and adjust convergence criteria using `adjust_criteria_for_precision()`:

```python
# tests/convergence/test_convergence_precision.py
"""Precision-parameterized convergence tests (ADR-028)."""
import pytest

from src.shared.model_spec import ModelSpec
from src.shared.precision_config import PrecisionConfig

from .conftest import make_engine
from .criteria import assert_convergence
from .precision_adjustments import adjust_criteria_for_precision
from .training_harness import run_training_loop
from .problems import iris, breast_cancer


PRECISION_CONFIGS = [
    pytest.param(PrecisionConfig.float32(), id="fp32"),
    pytest.param(PrecisionConfig.mixed_f16_f32(), id="fp16"),
    # pytest.param(PrecisionConfig.float64(), id="fp64"),     # enabled when double-precision lands
    # pytest.param(PrecisionConfig.float8_e4m3(), id="fp8"),  # enabled when FP8 lands
]


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_iris_convergence_precision(precision):
    """Iris CCE convergence across precision configurations."""
    X, y = iris.load()
    criteria = adjust_criteria_for_precision(iris.FULL_CRITERIA, precision)
    engine = make_engine(
        input_dim=4,
        hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3,
        num_modules=8,
        mode="CCE",
        backend="cpu",
        precision=precision,
    )
    history = run_training_loop(
        engine, X, y,
        epochs=criteria.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
    )
    assert_convergence(history, criteria)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_breast_cancer_convergence_precision(precision):
    """Breast Cancer BCE convergence across precision configurations."""
    X, y = breast_cancer.load()
    criteria = adjust_criteria_for_precision(breast_cancer.FULL_CRITERIA, precision)
    engine = make_engine(
        input_dim=30,
        hidden_dim=breast_cancer.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
        precision=precision,
    )
    history = run_training_loop(
        engine, X, y,
        epochs=criteria.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
    )
    assert_convergence(history, criteria)
```

**Precision configs** — only FP32 and mixed-FP16 are initially enabled. FP64 and FP8 configurations are commented out with references to Phase 8 and Phase 9 respectively. They are enabled by uncommenting when the corresponding backend precision support lands.

**`make_engine` precision integration** — the `precision` parameter is passed through to `ModelSpec` construction. When `precision` is not FP32, the `make_engine` helper must use the appropriate `ModelSpec` factory (`ModelSpec.mixed_f16_f32()`, `ModelSpec.float64()`, etc.) instead of `ModelSpec.float32()`. This requires the `make_engine` helper to dispatch on precision. The updated helper constructs `ParameterSpace(spec)` and `HardwareProfile(...)` consistently:

```python
# Updated make_engine (in conftest.py) — precision-aware ModelSpec construction
def make_engine(
    *,
    input_dim: int,
    hidden_dim: int,
    output_classes: int,
    num_modules: int,
    mode: str,
    backend: str = "cpu",
    precision: PrecisionConfig | None = None,
    gradient_clip_threshold: float = 1.0,
    optimizer: OptimizerConfig | None = None,
) -> Engine:
    hw_kwargs = dict(input_dim=input_dim, hidden_dim=hidden_dim,
                     output_classes=output_classes, num_modules=num_modules,
                     simd_width=4, cache_line_bytes=64)
    if precision is None or precision.storage_dtype == np.dtype("float32"):
        spec = ModelSpec.float32(**hw_kwargs)
    elif precision.storage_dtype == np.dtype("float16"):
        spec = ModelSpec.mixed_f16_f32(**hw_kwargs)
    else:
        # Defer to generic constructor; assumes ModelSpec supports the precision
        spec = ModelSpec(precision=precision, **hw_kwargs)

    hardware = HardwareProfile(
        simd_width=4, cache_line_bytes=64,
        max_reduce_fan_in=256, max_local_mem_bytes=None,
        global_mem_bytes=4 * 1024**3,
    )
    param_space = ParameterSpace(spec)
    ...
```

**Validation:** FP32 tests pass as baseline. FP16 tests pass with relaxed criteria (−2% accuracy, +0.05 loss, +20% epochs).

---

### Step 12.12: Implement cross-backend trajectory comparison

**Action:** Create `tests/convergence/trajectory_comparison.py` containing the trajectory comparison logic, and integrate it into cross-backend test functions.

**File:** `tests/convergence/trajectory_comparison.py`

**Implementation (ADR-028 §Cross-Backend Validation):**

```python
"""Cross-backend trajectory comparison (ADR-028 Option H)."""
from __future__ import annotations

import warnings

from .criteria import TrainingHistory

TRAJECTORY_DIVERGENCE_THRESHOLD = 0.10  # 10% relative accuracy difference


def compare_trajectories(
    oracle_history: TrainingHistory,
    comparison_history: TrainingHistory,
    *,
    oracle_name: str = "oracle",
    comparison_name: str = "comparison",
    divergence_threshold: float = TRAJECTORY_DIVERGENCE_THRESHOLD,
) -> None:
    """Warn (do not fail) if training trajectories diverge between backends.

    Emits UserWarning for each epoch where relative accuracy difference
    exceeds the threshold. These warnings are informational — CI captures
    them via 'pytest --tb=short -rw' for trend analysis.
    """
    n_epochs = min(len(oracle_history.accuracy_curve),
                   len(comparison_history.accuracy_curve))
    for epoch in range(n_epochs):
        oracle_acc = oracle_history.accuracy_curve[epoch]
        comp_acc = comparison_history.accuracy_curve[epoch]
        if oracle_acc > 0:
            rel_diff = abs(oracle_acc - comp_acc) / oracle_acc
            if rel_diff > divergence_threshold:
                warnings.warn(
                    f"Trajectory divergence at epoch {epoch + 1}: "
                    f"{oracle_name}={oracle_acc:.2%}, {comparison_name}={comp_acc:.2%}, "
                    f"rel_diff={rel_diff:.1%}",
                    stacklevel=2,
                )
```

**Integration into cross-backend tests:**

Cross-backend tests (Steps 12.9, 12.10) that use the `backend_name` fixture are augmented to collect trajectories and compare them against the CPU oracle:

```python
@pytest.mark.convergence
@pytest.mark.convergence_full
def test_iris_convergence_cross_backend(backend_name):
    """Iris CCE convergence on each available backend, with trajectory comparison."""
    X, y = iris.load()

    # Run on requested backend
    engine = make_engine(
        input_dim=4, hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3, num_modules=8, mode="CCE",
        backend=backend_name,
    )
    history = run_training_loop(
        engine, X, y,
        epochs=iris.FULL_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
    )
    assert_convergence(history, iris.FULL_CRITERIA)

    # Trajectory comparison against CPU oracle (if not already CPU)
    if backend_name != "cpu" and "cpu" in AVAILABLE_BACKENDS:
        oracle_engine = make_engine(
            input_dim=4, hidden_dim=iris.BASELINE.hidden_size,
            output_classes=3, num_modules=8, mode="CCE",
            backend="cpu",
        )
        oracle_history = run_training_loop(
            oracle_engine, X, y,
            epochs=iris.FULL_CRITERIA.epoch_budget,
            batch_size=iris.BASELINE.batch_size,
        )
        compare_trajectories(
            oracle_history, history,
            oracle_name="cpu", comparison_name=backend_name,
        )
```

**CI configuration note:** `pytest -W error::UserWarning` must NOT be used for convergence tests — trajectory divergence warnings are intentionally informational. CI runs convergence tests with `pytest --tb=short -rw` to capture warnings in the log artifact.

**Validation:** Verify that CPU-vs-CPU trajectory comparison produces zero warnings (identical trajectories). If GPU backends are available, verify warnings are emitted only when divergence exceeds threshold.

---

### Step 12.13: Validate `convergence_fast` gate on CPU/FP32

**Action:** Run the `convergence_fast` test suite on CPU/FP32 and validate the rollback gate criteria.

**Validation commands:**

```bash
cd architectures/averaging_ensembled_classifier
pytest -m convergence_fast --tb=short -v 2>&1 | tee /tmp/convergence_fast.txt
```

**Gate criteria (ADR-028 §Success Metrics):**

| Criterion | Expected |
| :--- | :--- |
| `test_iris_convergence_fast` passes | ≥90% accuracy within 20 epochs |
| `test_digits_convergence_fast` passes | ≥85% accuracy within 30 epochs |
| `test_xor_convergence_fast` passes | ≥95% accuracy within 50 epochs |
| `test_breast_cancer_convergence_fast` passes | ≥90% accuracy within 30 epochs |
| Total `convergence_fast` runtime | <60s |
| No NaN/Inf in any test | `has_nan_inf == False` for all histories |

If any fast test fails, investigate the cause before proceeding:
1. If accuracy does not reach the threshold, adjust baseline hyperparameters (per CONCEPT.md §1, this signals incomplete architectural modeling of the problem's convergence requirements).
2. If NaN/Inf occurs, investigate numerical stability in the kernel pipeline (may trace back to stabilization policy or gradient clipping configuration).
3. If runtime exceeds 60s, reduce epoch budgets for fast tests or investigate training loop overhead.

---

### Step 12.14: Validate rollback gate

**Action:** Verify all Phase 12 rollback gate criteria are satisfied.

**Gate criteria:**

| # | Criterion | Validation command |
| :--- | :--- | :--- |
| 1 | `tests/convergence/` directory structure exists with all files | `find tests/convergence/ -type f \| sort` |
| 2 | `convergence` markers registered in `pyproject.toml` | `pytest --strict-markers --collect-only -m convergence` |
| 3 | All `convergence_fast` tests pass on CPU/FP32 | `pytest -m convergence_fast --tb=short` |
| 4 | Iris achieves ≥97% accuracy within 100 epochs (full) | `pytest -m convergence_full -k iris --tb=short` |
| 5 | At least one BCE problem passes (full) | `pytest -m convergence_full -k "xor or two_moons or breast_cancer" --tb=short` |
| 6 | `pytest -m convergence_fast` completes in <60s | Timed run |
| 7 | `convergence_full` skipped in default `pytest` run | `pytest --collect-only -q \| grep convergence_full` shows SKIPPED |
| 8 | Optional MNIST/Fashion tests skipped cleanly without torchvision | `pytest -m convergence_full -k "mnist or fashion" --tb=short` reports `SKIPPED` |

---

## 5. Convergence Criteria Specification

### Multi-Criteria Check Order

The `assert_convergence()` function checks criteria in a specific order chosen to produce the most informative error messages:

| # | Check | Failure message example |
| :--- | :--- | :--- |
| 1 | NaN/Inf absence | `"Training produced NaN/Inf values"` |
| 2 | Accuracy threshold | `"Final accuracy 0.82 < threshold 0.90"` |
| 3 | Loss threshold | `"Final loss 0.6234 > threshold 0.15"` |
| 4 | Epoch budget | `"Required 150 epochs to reach 97%; budget was 100"` |
| 5 | Loss monotonicity | `"Non-monotonic loss at epoch 35: 0.1234 → 0.1567 (+27.0%, tolerance=5.0%)"` |

NaN/Inf is checked first because it indicates a fundamental numerical failure — all subsequent checks would be meaningless. Accuracy is checked before loss because accuracy is the user-interpretable metric.

### Monotonicity Tolerance Design

The monotonicity check operates after a warmup period to allow loss to increase during the initial learning rate warmup / Adam moment initialization phase. After warmup:

- **Tolerance = `None`**: Monotonicity check is disabled. Used for `convergence_fast` tests and noisy problems.
- **Tolerance = `0.05`** (5%): Each epoch's loss may increase by at most 5% relative to the previous epoch. This allows mini-batch noise while catching systematic divergence.
- **Tolerance = `0.10`** (10%): More permissive; used for harder problems (Fashion-MNIST) or lower-precision configurations.

### Per-Problem Criteria Table

> **Note:** These criteria should be calibrated against each problem's intended learning rate (now exercisable via `OptimizerConfig`, ADR-029). Previous calibration assumed a fixed lr=0.001 for all problems; re-validate after switching to per-problem learning rates.

| Problem | Mode | Fast: acc / epochs | Full: acc / loss / epochs / warmup / mono | Max Epochs |
| :--- | :--- | :--- | :--- | :--- |
| `iris` | CCE | 90% / 20 | 97% / 0.15 / 100 / 10 / 0.05 | 200 |
| `digits` | CCE | 85% / 30 | 95% / 0.25 / 150 / 15 / 0.05 | 300 |
| `xor` | BCE | 95% / 50 | 99% / 0.05 / 200 / 20 / 0.05 | 500 |
| `two_moons` | BCE | 90% / 30 | 98% / 0.10 / 150 / 10 / 0.05 | 300 |
| `breast_cancer` | BCE | 90% / 30 | 96% / 0.15 / 100 / 10 / 0.05 | 200 |
| `mnist_1k` | CCE | 80% / 20 | 92% / 0.35 / 100 / 10 / 0.10 | 200 |
| `fashion_1k` | CCE | 70% / 30 | 85% / 0.50 / 150 / 15 / 0.10 | 300 |

---

## 6. Problem Suite Reference

### Dataset Properties

| Problem | Source | Samples | Features | Output Dim | Linearity | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Iris | sklearn | 150 | 4 | 3 (CCE) | ~Linear | Canonical small-scale, existing fixture compatibility |
| Digits | sklearn | 1,797 | 64 | 10 (CCE) | Non-linear | Mid-scale, 8×8 pixel images |
| XOR | Synthetic | 200 | 2 | 1 (BCE) | Non-linear | Minimal non-linear problem; requires hidden layer |
| Two Moons | sklearn | 1,000 | 2 | 1 (BCE) | Non-linear | Curved decision boundary |
| Breast Cancer | sklearn | 569 | 30 | 1 (BCE) | ~Linear | Real-world, moderate dimensionality |
| MNIST-1k | torchvision | 1,000 | 784 | 10 (CCE) | Non-linear | High-dimensional; single-hidden-layer ceiling |
| Fashion-1k | torchvision | 1,000 | 784 | 10 (CCE) | Non-linear | Harder than MNIST; validates generalization |

### Hyperparameter Baselines

> **Optimizer note:** LR, β₁, β₂, and ε are exercised by convergence tests via `OptimizerConfig` (ADR-029, Phase 12A). WD (weight decay) is not yet plumbed. **Hidden**, **Clip**, **Batch**, **LR**, **β₁**, **β₂**, and **ε** are all exercised. Problems marked with † have a LR that differs from the Adam default (0.001).

| Problem | Hidden | LR | β₁ | β₂ | ε | WD | Clip | Batch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `iris` | 16 | 0.01† | 0.9 | 0.999 | 1e-8 | 0.0 | 1.0 | 32 |
| `digits` | 128 | 0.001 | 0.9 | 0.999 | 1e-8 | 1e-4 | 1.0 | 64 |
| `xor` | 8 | 0.1† | 0.9 | 0.999 | 1e-8 | 0.0 | 5.0 | 32 |
| `two_moons` | 16 | 0.01† | 0.9 | 0.999 | 1e-8 | 0.0 | 1.0 | 64 |
| `breast_cancer` | 32 | 0.001 | 0.9 | 0.999 | 1e-8 | 1e-4 | 1.0 | 32 |
| `mnist_1k` | 256 | 0.001 | 0.9 | 0.999 | 1e-8 | 1e-4 | 1.0 | 64 |
| `fashion_1k` | 256 | 0.001 | 0.9 | 0.999 | 1e-8 | 1e-4 | 1.0 | 64 |

These baselines are validated during ADR-028 acceptance (Step 12.13) using each problem's intended learning rate, forwarded via `OptimizerConfig` (ADR-029, Phase 12A). If a baseline fails to converge at its intended LR, investigate: (a) increase `hidden_size`, (b) adjust gradient clip threshold, (c) relax convergence criteria. Per CONCEPT.md §1, if the architecture cannot converge on a standard problem with tuned hyperparameters, extend the architecture rather than workaround.

---

## 7. Training Loop Architecture

### `run_training_loop()` Data Flow

```
For each epoch in range(epochs):
    ┌─────────────────────────────────────────────┐
    │ 1. Shuffle (X, y) with per-epoch seed       │
    │ 2. Split into mini-batches of batch_size    │
    │ 3. For each batch:                          │
    │    ├─ pred = engine.train_batch(X_b, y_b)   │
    │    ├─ Check pred for NaN/Inf                │
    │    ├─ Accumulate correct predictions         │
    │    └─ Accumulate batch loss (numpy)          │
    │ 4. Compute epoch accuracy = correct / total  │
    │ 5. Compute epoch loss = mean batch loss      │
    │ 6. Append to TrainingHistory curves          │
    └─────────────────────────────────────────────┘
```

### Engine.train_batch() Internal Path

```
engine.train_batch(X_batch, y_batch)
    │
    ├─ submit(X_batch)
    │   ├─ build_act_plan(model_spec, hardware, strategy, batch_size)
    │   ├─ renderer.render(act_plan) → {"inference_event": future}
    │   └─ return WorkTicket(X_batch, future, engine)
    │
    ├─ ticket.get_prediction()
    │   ├─ future.result() → prediction_matrix
    │   ├─ future.release() → free device buffers
    │   └─ cache + return prediction
    │
    ├─ ticket.resolve(y_batch)
    │   ├─ build_learn_plan(model_spec, hardware, strategy, batch_size, policy)
    │   ├─ renderer.render(learn_plan) → {"final_batch_event": future}
    │   └─ return LearnHandle(future, ticket)
    │
    └─ learn_handle.wait()
        ├─ future.result() → updated parameters
        ├─ future.release()
        └─ ticket.state → CONSUMED
```

### External Loss Computation

The training harness computes loss independently of the kernel pipeline:

**CCE (Categorical Cross-Entropy):**
```python
def _compute_cce_loss(predictions: NDArray, labels: NDArray, n_classes: int) -> float:
    """Compute CCE loss using numpy (FP64 accumulation)."""
    eps = 1e-12
    one_hot = np.zeros((len(labels), n_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    pred_clipped = np.clip(predictions.astype(np.float64), eps, 1 - eps)
    return float(-np.sum(one_hot * np.log(pred_clipped)) / len(labels))
```

**BCE (Binary Cross-Entropy):**
```python
def _compute_bce_loss(predictions: NDArray, labels: NDArray) -> float:
    """Compute BCE loss using numpy (FP64 accumulation)."""
    eps = 1e-12
    pred_clipped = np.clip(predictions.astype(np.float64).squeeze(), eps, 1 - eps)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(pred_clipped) + (1 - y) * np.log(1 - pred_clipped)))
```

Both use FP64 accumulation and epsilon clipping to avoid `log(0)`.

---

## 8. Precision Adjustment Tables

### Adjustment Values

| Precision | Key | Accuracy Δ | Loss Δ | Epoch × | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | `float32` | 0.0 | 0.0 | 1.0× | Reference baseline |
| FP16 (mixed) | `float16` | −2% | +0.05 | 1.2× | Reduced gradient precision; Adam moment estimates may lose fine structure |
| FP64 | `float64` | 0.0 | −0.01 | 0.9× | Higher precision; may converge faster due to reduced rounding errors |
| FP8 (E4M3) | `float8_e4m3` | −5% | +0.1 | 1.5× | Significant quantization; validates basic convergence only |

### Adjustment Interaction Example

Iris Full criteria (FP32 baseline): accuracy=0.97, loss=0.15, epochs=100

| Precision | Adjusted accuracy | Adjusted loss | Adjusted epochs |
| :--- | :--- | :--- | :--- |
| FP32 | 0.97 | 0.15 | 100 |
| FP16 | 0.95 | 0.20 | 120 |
| FP64 | 0.97 | 0.14 | 90 |
| FP8 | 0.92 | 0.25 | 150 |

---

## 9. Cross-Backend Trajectory Comparison

### Design (ADR-028 Option H + trajectory logging)

Cross-backend validation uses independent convergence — each backend must pass the convergence criteria independently. No cross-backend parameter comparison is performed (floating-point divergence over thousands of steps would exceed reasonable tolerances).

**Trajectory comparison is warning-level, not failure-level.** A `UserWarning` is emitted for each epoch where the relative accuracy difference between two backends exceeds `TRAJECTORY_DIVERGENCE_THRESHOLD` (default 10%).

### When trajectory comparison runs

| Test type | Trajectory comparison |
| :--- | :--- |
| CPU-only (`convergence_fast`) | No — single backend |
| Cross-backend (`convergence_full`) | Yes — compare each non-CPU backend against CPU oracle |
| CPU-only (`convergence_full`) | No — single backend |

### CI warning capture

Convergence tests run with `pytest --tb=short -rw` (report warnings summary). The `-W error::UserWarning` flag is NOT used. Trajectory divergence warnings appear in the warnings summary section of the pytest output and are written to the CI log artifact.

**Rationale:** GPU backends with non-deterministic reduction ordering (e.g., OpenCL with atomic operations, Vulkan with subgroup shuffles) may produce slightly different trajectories even with identical algorithms. This is expected floating-point behavior and should not cause CI failures. Repeated trajectory divergence across commits may indicate a backend-specific issue worth investigation.

---

## 10. CI Integration

### Execution Matrix (ADR-028 §CI Integration)

| CI Trigger | Scope | Backends | Precisions | Estimated Runtime |
| :--- | :--- | :--- | :--- | :--- |
| Every commit | `convergence_fast` | CPU only | FP32 | <60s |
| Pull request | `convergence_fast` | All enabled | FP32, FP16 | <3 min |
| Nightly | `convergence_fast` + `convergence_full` | All enabled | FP32, FP16, FP8 | <30 min |
| Release | Full suite | All enabled | All supported | <45 min |

### Pytest Invocations

| CI Trigger | Command |
| :--- | :--- |
| Every commit | `pytest -m convergence_fast --tb=short -rw` |
| Pull request | `pytest -m "convergence_fast" --tb=short -rw` |
| Nightly | `pytest -m convergence --tb=short -rw` |
| Release | `pytest -m convergence --tb=short -rw --strict-markers` |

### Default Exclusion via Collection Hook

`convergence_full` exclusion is implemented via `pytest_collection_modifyitems` in `tests/convergence/conftest.py` (not `addopts`) — see Step 12.1. This means:

- Running `pytest` without marker filters: runs all tests including `convergence_fast`; `convergence_full` tests are SKIPPED.
- Running `pytest -m convergence`: runs all convergence tests (both fast and full — the hook detects `convergence` in the markexpr and allows full tests).
- Running `pytest -m convergence_fast`: runs only fast convergence tests.
- Running `pytest -m convergence_full`: runs only full convergence tests (explicitly opted in).

---

## 11. CLI Usage Patterns

### Running convergence tests

```bash
cd architectures/averaging_ensembled_classifier

# Fast convergence smoke tests (every commit)
pytest -m convergence_fast --tb=short -v

# All convergence tests (nightly)
pytest -m convergence --tb=short -rw

# Full convergence only
pytest -m convergence_full --tb=short -rw

# Specific problem
pytest -m convergence -k iris --tb=short -v

# Specific mode
pytest tests/convergence/test_convergence_bce.py --tb=short -v

# Cross-backend (all available)
pytest tests/convergence/ -k cross_backend --tb=short -rw

# Precision-specific
pytest tests/convergence/test_convergence_precision.py --tb=short -v

# Collect only (verify skip logic)
pytest -m convergence --collect-only -q

# With strict markers (CI validation)
pytest -m convergence --strict-markers --collect-only
```

### Debugging convergence failures

```bash
# Verbose output with full traceback
pytest -m convergence_fast -k iris --tb=long -v -s

# Print training progress (add --capture=no to see epoch-by-epoch output)
pytest -m convergence_fast -k iris -s
```

---

## 12. Test Target Inventory

### CCE convergence tests

| Test | Marker | Backend | Problem | Criteria |
| :--- | :--- | :--- | :--- | :--- |
| `test_iris_convergence_fast` | fast | CPU | Iris | 90% / 20 ep |
| `test_digits_convergence_fast` | fast | CPU | Digits | 85% / 30 ep |
| `test_iris_convergence_full` | full | CPU | Iris | 97% / 100 ep |
| `test_digits_convergence_full` | full | CPU | Digits | 95% / 150 ep |
| `test_mnist_convergence_full` | full | CPU | MNIST-1k | 92% / 100 ep |
| `test_fashion_mnist_convergence_full` | full | CPU | Fashion-1k | 85% / 150 ep |
| `test_iris_convergence_cross_backend` | full | all | Iris | 97% / 100 ep |
| `test_digits_convergence_cross_backend` | full | all | Digits | 95% / 150 ep |

### BCE convergence tests

| Test | Marker | Backend | Problem | Criteria |
| :--- | :--- | :--- | :--- | :--- |
| `test_xor_convergence_fast` | fast | CPU | XOR | 95% / 50 ep |
| `test_breast_cancer_convergence_fast` | fast | CPU | Breast Cancer | 90% / 30 ep |
| `test_xor_convergence_full` | full | CPU | XOR | 99% / 200 ep |
| `test_two_moons_convergence_full` | full | CPU | Two Moons | 98% / 150 ep |
| `test_breast_cancer_convergence_full` | full | CPU | Breast Cancer | 96% / 100 ep |
| `test_xor_convergence_cross_backend` | full | all | XOR | 99% / 200 ep |
| `test_breast_cancer_convergence_cross_backend` | full | all | Breast Cancer | 96% / 100 ep |

### Precision-parameterized tests

| Test | Marker | Backend | Problem | Precisions |
| :--- | :--- | :--- | :--- | :--- |
| `test_iris_convergence_precision` | full | CPU | Iris | FP32, FP16 (+ FP64, FP8 when enabled) |
| `test_breast_cancer_convergence_precision` | full | CPU | Breast Cancer | FP32, FP16 (+ FP64, FP8 when enabled) |

### Test count summary

| Category | Fast | Full (CPU) | Full (cross-backend) | Full (precision) | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| CCE | 2 | 4 | 2 × N_backends | 2 | 8 + 2N |
| BCE | 2 | 3 | 2 × N_backends | 2 | 7 + 2N |
| **Total** | **4** | **7** | **4 × N_backends** | **4** | **15 + 4N** |

Where N = number of available backends. With CPU only: **19 tests**. With CPU + OpenCL + Vulkan: **27 tests**.

---

## 13. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **~~Hardcoded optimizer hyperparameters limit convergence.~~** *(Resolved by Phase 12A / ADR-029.)* Optimizer hyperparameters are now configurable via `OptimizerConfig`. Per-problem learning rates are exercised through `make_engine()`. | ~~High~~ N/A | ~~High~~ N/A | Resolved. `make_engine()` constructs `OptimizerConfig` from `BaselineHyperparameters` and passes it to the `Engine` constructor. |
| **Baseline hyperparameters fail to converge.** Even with the correct learning rate, a problem's configuration may not achieve target accuracy due to architecture limitations (single hidden layer, averaging ensemble structure) or numerical precision erosion. | Medium | High | Validate all baselines during Step 12.13 before committing. If a baseline fails, investigate: (a) increase hidden size, (b) adjust gradient clip threshold, (c) relax convergence criteria. Per CONCEPT.md §1, if the architecture cannot converge on a standard problem, extend the architecture rather than workaround. |
| **Training loop overhead exceeds CI runtime budget.** Python-level overhead in the training harness (numpy loss computation, data shuffling, Engine construction) may dominate for small problems, pushing `convergence_fast` beyond 60s. | Medium | Medium | Profile the fast suite during Step 12.13. If overhead is excessive: (a) reduce epoch budgets, (b) cache `Engine` construction across epochs (already session-scoped), (c) consider removing the external loss computation from fast tests (check accuracy only). |
| **Zero-initialized weights prevent convergence.** Buffer allocator zero-initializes all buffers including model parameters. Zero weights produce zero gradients in many architectures, preventing learning. The system may require a dedicated weight initialization mechanism that does not yet exist. | Medium | High | During Step 12.13, if zero-init weights cause training to stall (zero accuracy across all epochs), investigate: (a) whether the kernel pipeline applies its own initialization internally, (b) whether the renderer injects initial parameter values, (c) if not, weight initialization may need to be added as a plan-level primitive (CONCEPT.md §1 signal). |
| **BCE output dimension mismatch.** The architecture may expect `output_classes > 1` even for BCE mode, or may use a different output encoding (e.g., 2-class softmax instead of 1-class sigmoid). | Medium | Medium | Verify BCE output dimension conventions by reading `PlanBceStrategy` and the BCE loss kernel contract. If the architecture uses 2-class CCE for binary problems, adjust problem loaders to use 2-class encoding. |
| **External loss computation disagrees with kernel loss.** The numpy-based loss computation in the training harness may diverge from the kernel's internal loss due to different numerical implementations (e.g., log-sum-exp vs. direct log). | Low | Low | The external loss is an independent verification signal, not a ground truth oracle. Divergence is expected but should be bounded. If divergence exceeds 10% relative after convergence, investigate the kernel's loss computation for potential numerical issues. |
| **sklearn version changes dataset contents.** A sklearn update could change the bundled dataset (e.g., Iris feature values, Breast Cancer sample count), causing hash mismatches or threshold failures. | Low | Low | sklearn datasets are historically stable. Pin sklearn version in `requirements.txt` for CI. Dataset integrity hashing (Step 12.7) detects changes for optional datasets; bundled datasets rely on sklearn's stability guarantee. |
| **Cross-backend trajectory warnings flood CI logs.** With non-deterministic GPU backends, every epoch may trigger a divergence warning, producing unreadable test output. | Medium | Low | If excessive warnings are observed, reduce warning frequency: emit at most one warning per test (aggregate divergent epochs into a summary) rather than per-epoch. This is an implementation-time adjustment that does not change the architectural design. |
| **CONCEPT.md §1 trigger: convergence failure exposes incomplete architectural modeling.** A standard problem fails to converge despite correct hyperparameters, revealing that the architecture's abstractions (e.g., `StabilizationPolicy`, `ModelSpec`, `PrecisionConfig`) are insufficient. | Low | High | This is a feature, not a bug. Per CONCEPT.md §1 (Architectural Elegance Feedback): (1) suspend implementation of the failing test, (2) formalize the missing pattern as a documented architectural primitive, (3) reify through revised contracts and extensions. The convergence test suite's purpose is precisely to surface such gaps. |

---

## References

- [ADR-028: Convergence Testing on Standard Problems](../adr/ADR-028-convergence-testing-standard-problems.md) — governing ADR for this phase
- [ADR-016: Test Strategy](../adr/ADR-016-test-strategy.md) — three-tier test framework, CPU oracle, pytest markers
- [ADR-018: User-Facing API](../adr/ADR-018-user-facing-api.md) — WorkTicket/Engine pattern, `train_batch()` convenience method
- [ADR-008: Precision Configuration](../adr/ADR-008-precision-configuration.md) — `PrecisionConfig` composition, per-precision tolerances
- [ADR-011: CCE/BCE Strategy Delegation](../adr/ADR-011-cce-bce-strategy-delegation.md) — `PlanCceStrategy` / `PlanBceStrategy` selection
- [ADR-014: Build System Integration](../adr/ADR-014-build-system-integration.md) — `_build_config.py` backend flags
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback, §5 Unified Execution Model
