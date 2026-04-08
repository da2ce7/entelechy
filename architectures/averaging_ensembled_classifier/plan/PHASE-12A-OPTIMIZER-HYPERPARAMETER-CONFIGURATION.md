# Phase 12A: Optimizer Hyperparameter Configuration — Detailed Plan

**Status:** Complete (2026-04-07)  
**Phase:** 12A (Phase 12 prerequisite)  
**Objective:** Make Adam optimizer hyperparameters (learning rate, β₁, β₂, ε) configurable through the user-facing API by introducing an `OptimizerConfig` frozen dataclass, plumbing it through `Engine` constructor injection and `build_learn_plan()` signature extension, and validating backward compatibility with all existing tests.  
**Governing ADR:** ADR-029 (Optimizer Hyperparameter Configuration)  
**Rollback gate:** All existing Tier 1–3 tests pass without modification. `build_learn_plan()` called without `optimizer` produces byte-identical `ExecutionPlan` scalar parameters. `Engine(optimizer=OptimizerConfig(learning_rate=0.01))` produces a plan with `learning_rate=0.01` in all adam_update nodes.  
**Dependencies:** Phase 11 (Engine/WorkTicket API — complete), Phase 3 (CPU backend — complete). No kernel or backend changes required — the adam_update kernel contract already accepts optimizer hyperparameters as scalar parameters.  
**Enables:** Phase 12 (Convergence Testing) — allows `BaselineHyperparameters.learning_rate` to be exercised through `make_engine()`.

### Relationship to Other Phases

Phase 12A is a narrow, focused prerequisite for Phase 12. It modifies only the host orchestration layer — no kernels, renderers, or backends are touched.

| Phase | Phase 12A interaction |
| :--- | :--- |
| Phase 1 (Plan Model) | Phase 12A modifies `build_learn_plan()` to accept an `OptimizerConfig` parameter. Plan node types and buffer lifecycle are unchanged. Adam update nodes already accept scalar optimizer parameters. |
| Phase 3 (CPU Backend) | No change. The CPU backend renders scalar parameters injected by the plan builder; the source of those values is transparent to the renderer. |
| Phase 4 (Test Harness) | Phase 12A adds targeted unit tests for `OptimizerConfig` and integration tests for the plumbing. Existing Tier 1–3 tests are not modified. |
| Phase 11 (User-Facing API) | Phase 12A extends `Engine.__init__()` with an optional `optimizer` parameter and updates `_dispatch_learn_plan()` to forward it. The `train_batch()` convenience method signature is unchanged. |
| Phase 12 (Convergence Testing) | Phase 12A is the direct prerequisite. After it completes, Phase 12's `make_engine()` can pass `OptimizerConfig(learning_rate=baseline.learning_rate)` and convergence tests exercise per-problem learning rates. |
| Phase 6 (Legacy Removal) | `TrainingOrchestrator` can optionally adopt `OptimizerConfig` by constructing it from `TrainingHyperparams` fields. This is not required by Phase 12A. |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 12A.1: Create `OptimizerConfig` frozen dataclass](#step-12a1-create-optimizerconfig-frozen-dataclass)
   - [Step 12A.2: Extend `build_learn_plan()` signature](#step-12a2-extend-build_learn_plan-signature)
   - [Step 12A.3: Extend `Engine` constructor with optimizer injection](#step-12a3-extend-engine-constructor-with-optimizer-injection)
   - [Step 12A.4: Forward optimizer through `_dispatch_learn_plan()`](#step-12a4-forward-optimizer-through-_dispatch_learn_plan)
   - [Step 12A.5: Export `OptimizerConfig` from `src/shared/__init__.py`](#step-12a5-export-optimizerconfig-from-srcshared__init__py)
   - [Step 12A.6: Write unit tests for `OptimizerConfig`](#step-12a6-write-unit-tests-for-optimizerconfig)
   - [Step 12A.7: Write integration tests for optimizer plumbing](#step-12a7-write-integration-tests-for-optimizer-plumbing)
   - [Step 12A.8: Validate backward compatibility (rollback gate)](#step-12a8-validate-backward-compatibility-rollback-gate)
5. [Data Type Reference](#5-data-type-reference)
6. [Backward Compatibility Matrix](#6-backward-compatibility-matrix)
7. [Risk Register](#7-risk-register)

---

## 1. Scope & Constraints

### In scope

- Creating `src/shared/optimizer_config.py` containing the `OptimizerConfig` frozen dataclass with default values matching the current hardcoded constants.
- Extending `build_learn_plan()` in `src/shared/plan_builder.py` with an optional `optimizer: OptimizerConfig | None = None` parameter.
- Replacing the hardcoded `_adam_scalars` dict in `build_learn_plan()` with values derived from the `OptimizerConfig`.
- Extending `Engine.__init__()` in `src/shared/engine.py` with an optional `optimizer: OptimizerConfig | None = None` parameter.
- Updating `Engine._dispatch_learn_plan()` to forward the optimizer config to `build_learn_plan()`.
- Exporting `OptimizerConfig` from `src/shared/__init__.py`.
- Writing unit tests for `OptimizerConfig` validation, defaults, and `resolve_epsilon()`.
- Writing integration tests verifying that custom optimizer values appear in plan node scalar parameters.
- Validating that all existing tests pass without modification.

### Out of scope

- Modifying any kernel source, kernel contract, or backend renderer — the kernel layer already accepts optimizer hyperparameters as scalar parameters.
- Implementing learning rate scheduling or per-step optimizer parameter updates — the `OptimizerConfig` is set once per Engine lifecycle (ADR-018 Choice 5A: ephemeral).
- Migrating `TrainingOrchestrator` to use `OptimizerConfig` — optional future work (Phase 6 adjacent).
- Modifying Phase 12 convergence test infrastructure — that is Phase 12's scope. Phase 12A only provides the plumbing.
- Adding weight decay support — `OptimizerConfig` covers only the four Adam parameters that currently appear in the plan as scalar values (`learning_rate`, `beta1`, `beta2`, `epsilon`).
- Modifying the host-side bias correction stepping mechanism — Phase 12A ensures the initial `beta1_pow_t` and `beta2_pow_t` values derive from `OptimizerConfig`, but the epoch-stepping logic remains the caller's responsibility.

### Key constraint: backward compatibility

**All existing code that does not specify an `optimizer` parameter must produce identical behavior.** This is non-negotiable. The mechanism:

1. `OptimizerConfig()` default constructor produces `learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=None`.
2. `OptimizerConfig.resolve_epsilon(precision)` returns `precision.compute_epsilon` when `epsilon is None`.
3. `build_learn_plan(..., optimizer=None)` is equivalent to `build_learn_plan(...)` at the current call sites — the `None` default falls through to `OptimizerConfig()`.
4. `Engine(..., optimizer=None)` behaves identically to `Engine(...)` as currently implemented.

The hardcoded `_adam_scalars` dict in `build_learn_plan()` currently uses:

```python
{
    "learning_rate": 0.001,
    "beta1_pow_t": 0.9,       # beta1 ** 1
    "beta2_pow_t": 0.999,     # beta2 ** 1
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": model_spec.precision.compute_epsilon,
}
```

The replacement using `OptimizerConfig()` defaults must produce **exactly these same floating-point values** (not approximately equal — exactly equal, since these are Python float literals that compare with `==`).

### Key constraint: shared-layer purity

`OptimizerConfig` resides in `src/shared/` and imports only from Python stdlib and other `src/shared/` modules. It does not import from any backend package.

---

## 2. Pre-Condition Inventory

| File | Relevant current state | Phase 12A action |
| :--- | :--- | :--- |
| `src/shared/plan_builder.py` | `build_learn_plan()` accepts 6 parameters (+ `activation_lifecycle` default). Hardcoded `_adam_scalars` dict at ~line 1104. | Add `optimizer` parameter; replace `_adam_scalars` with config-derived values (Step 12A.2). |
| `src/shared/engine.py` | `Engine.__init__()` accepts `model_spec`, `parameter_space`, `hardware_profile`, `precision`, `backend`, `strategy`, `policy`, `renderer_factory` (all keyword after first 3). `_dispatch_learn_plan()` calls `build_learn_plan()` with 6 positional + 1 keyword arg. | Add `optimizer` to constructor; forward from `_dispatch_learn_plan()` (Steps 12A.3, 12A.4). |
| `src/shared/__init__.py` | Exports plan types, buffer types, configuration types, builder functions, ticket API. No `OptimizerConfig`. | Add `OptimizerConfig` export (Step 12A.5). |
| `src/shared/optimizer_config.py` | Does not exist. | Create (Step 12A.1). |
| `src/shared/precision_config.py` | `PrecisionConfig.compute_epsilon` property — returns the smallest positive normal for the compute dtype. | Consumed by `OptimizerConfig.resolve_epsilon()`. No change. |
| `src/main_orchestrator.py` | `TrainingHyperparams` dataclass with `learning_rate`, `adam_beta1`, `adam_beta2`, `adam_epsilon` fields. These are not forwarded to `build_learn_plan()`. | No change (out of scope). |
| `src/shared/kernel_contracts/phase_3_update.py` | adam_update contract defines `ScalarParamSpec` for `learning_rate`, `beta1_pow_t`, `beta2_pow_t`, `beta1`, `beta2`. | No change (kernel layer already parameterized). |
| `tests/tier2/test_multi_precision_config.py` | Calls `build_learn_plan(spec, _HW, PlanCceStrategy(), batch_size=16, policy=policy)`. | No change required (uses positional args + `policy` keyword; `optimizer` default is `None`). |
| `tests/bench_cpu_backend.py` | Multiple calls to `build_learn_plan()` with 5–6 positional args. | No change required (new `optimizer` parameter has a default). |

---

## 3. Target Deliverables

After Phase 12A completes, the shared layer gains:

```
src/shared/
├── optimizer_config.py              # NEW — OptimizerConfig frozen dataclass
├── engine.py                        # MODIFIED — optimizer parameter added
├── plan_builder.py                  # MODIFIED — optimizer parameter, _adam_scalars derived
├── __init__.py                      # MODIFIED — OptimizerConfig exported
└── ... (all other files unchanged)
```

Test additions:

```
tests/
├── tier1/
│   ├── test_optimizer_config.py     # NEW — OptimizerConfig unit tests
│   └── test_optimizer_plumbing.py   # NEW — plan builder + Engine integration
└── ... (all existing tests unchanged)
```

---

## 4. Task Breakdown

### Step 12A.1: Create `OptimizerConfig` frozen dataclass

**Governing authority:** ADR-029 §1  
**File:** `src/shared/optimizer_config.py` (new file)

**Action:** Create the `OptimizerConfig` frozen dataclass with validation, defaults, and the `resolve_epsilon()` method.

```python
# src/shared/optimizer_config.py
"""Optimizer hyperparameter configuration (ADR-029)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .precision_config import PrecisionConfig


@dataclass(frozen=True)
class OptimizerConfig:
    """Adam optimizer hyperparameters for plan construction.

    All fields have defaults matching the previously hardcoded values in
    ``build_learn_plan()``, ensuring backward compatibility for code that
    does not specify optimizer configuration.

    When ``epsilon`` is ``None``, the precision-aware default
    (``PrecisionConfig.compute_epsilon``) is used — preserving the
    existing behavior where epsilon depends on the compute dtype.
    """
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if not (0 <= self.beta1 < 1):
            raise ValueError(
                f"beta1 must be in [0, 1), got {self.beta1}"
            )
        if not (0 <= self.beta2 < 1):
            raise ValueError(
                f"beta2 must be in [0, 1), got {self.beta2}"
            )
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be positive (or None for precision-aware default), "
                f"got {self.epsilon}"
            )

    def resolve_epsilon(self, precision: PrecisionConfig) -> float:
        """Return epsilon, falling back to the precision-aware default.

        When epsilon is None, returns ``precision.compute_epsilon`` —
        the smallest representable value for the compute dtype that
        prevents division by zero in the Adam update denominator.
        """
        if self.epsilon is not None:
            return self.epsilon
        return precision.compute_epsilon
```

**Design decisions:**

1. **`TYPE_CHECKING` guard for `PrecisionConfig` import.** The `resolve_epsilon` method's type annotation uses a string literal or `TYPE_CHECKING`-guarded import to avoid a circular dependency risk. At runtime, `PrecisionConfig` is passed as an argument — no module-level import is needed.

2. **`__post_init__` validation.** Catches invalid values at construction time rather than at plan-construction time. This follows the pattern established by `StabilizationPolicy`, which validates its fields in `__post_init__`.

3. **`epsilon: float | None = None` sentinel.** The `None` default means "use the precision-aware epsilon from `PrecisionConfig.compute_epsilon`." This preserves the current behavior where different precision configurations produce different epsilon values. A user can override this with a concrete float (e.g., `epsilon=1e-7`) if they want a fixed epsilon across precisions.

4. **Default values are exact float literals.** `0.001`, `0.9`, `0.999` are the exact Python float representations of the current hardcoded values. No rounding or conversion occurs.

**Validation:** `OptimizerConfig()` == `OptimizerConfig(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=None)`. Construction with invalid values raises `ValueError`.

---

### Step 12A.2: Extend `build_learn_plan()` signature

**Governing authority:** ADR-029 §3  
**File:** `src/shared/plan_builder.py`

**Action:** Add an optional `optimizer` parameter to `build_learn_plan()` and replace the hardcoded `_adam_scalars` dict with values derived from the config.

**Current signature (line ~442):**

```python
def build_learn_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
    policy: StabilizationPolicy,
    activation_lifecycle: Literal["cache", "recompute"] = "recompute",
) -> ExecutionPlan:
```

**New signature:**

```python
def build_learn_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
    policy: StabilizationPolicy,
    activation_lifecycle: Literal["cache", "recompute"] = "recompute",
    optimizer: OptimizerConfig | None = None,
) -> ExecutionPlan:
```

**Current `_adam_scalars` dict (line ~1104):**

```python
    # Default optimizer hyperparameters
    _adam_scalars = {
        "learning_rate": 0.001,
        "beta1_pow_t": 0.9,
        "beta2_pow_t": 0.999,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": model_spec.precision.compute_epsilon,
    }
```

**Replacement:**

```python
    # Optimizer hyperparameters (ADR-029)
    _opt = optimizer if optimizer is not None else OptimizerConfig()
    _adam_scalars = {
        "learning_rate": _opt.learning_rate,
        "beta1_pow_t": _opt.beta1,         # beta1 ** 1 for step t=1
        "beta2_pow_t": _opt.beta2,         # beta2 ** 1 for step t=1
        "beta1": _opt.beta1,
        "beta2": _opt.beta2,
        "epsilon": _opt.resolve_epsilon(model_spec.precision),
    }
```

**Import addition** (at top of `plan_builder.py`, with existing shared-layer imports):

```python
from .optimizer_config import OptimizerConfig
```

**Why `optimizer if optimizer is not None else OptimizerConfig()` instead of `optimizer or OptimizerConfig()`:** Both are equivalent here since `OptimizerConfig` is always truthy (frozen dataclass). The explicit `is not None` check is preferred for clarity — it makes the sentinel semantics unambiguous and avoids any theoretical confusion with falsy custom objects.

**Backward compatibility proof:**

| Current value | `OptimizerConfig()` equivalent | Match? |
| :--- | :--- | :--- |
| `"learning_rate": 0.001` | `OptimizerConfig().learning_rate` → `0.001` | ✅ Exact |
| `"beta1_pow_t": 0.9` | `OptimizerConfig().beta1` → `0.9` | ✅ Exact |
| `"beta2_pow_t": 0.999` | `OptimizerConfig().beta2` → `0.999` | ✅ Exact |
| `"beta1": 0.9` | `OptimizerConfig().beta1` → `0.9` | ✅ Exact |
| `"beta2": 0.999` | `OptimizerConfig().beta2` → `0.999` | ✅ Exact |
| `"epsilon": model_spec.precision.compute_epsilon` | `OptimizerConfig().resolve_epsilon(model_spec.precision)` → `precision.compute_epsilon` | ✅ Exact |

All values are float literals that pass through without arithmetic. No floating-point rounding can occur.

**Validation:** Call `build_learn_plan()` with and without `optimizer=None`; assert that the resulting `ExecutionPlan` has identical adam_update node scalar parameters.

---

### Step 12A.3: Extend `Engine` constructor with optimizer injection

**Governing authority:** ADR-029 §2  
**File:** `src/shared/engine.py`

**Action:** Add an optional `optimizer` parameter to `Engine.__init__()` and store it as instance state.

**Current `__init__` signature (line ~133):**

```python
    def __init__(
        self,
        model_spec: ModelSpec,
        parameter_space: "ParameterSpace",
        hardware_profile: HardwareProfile,
        *,
        precision: PrecisionConfig | None = None,
        backend: str = "auto",
        strategy: PlanProblemTypeStrategy | None = None,
        policy: StabilizationPolicy | None = None,
        renderer_factory: Callable[[str, HardwareProfile, PrecisionConfig], PlanRenderer] | None = None,
    ) -> None:
```

**New `__init__` signature:**

```python
    def __init__(
        self,
        model_spec: ModelSpec,
        parameter_space: "ParameterSpace",
        hardware_profile: HardwareProfile,
        *,
        precision: PrecisionConfig | None = None,
        backend: str = "auto",
        strategy: PlanProblemTypeStrategy | None = None,
        policy: StabilizationPolicy | None = None,
        optimizer: OptimizerConfig | None = None,
        renderer_factory: Callable[[str, HardwareProfile, PrecisionConfig], PlanRenderer] | None = None,
    ) -> None:
```

**Import addition** (at top of `engine.py`, with existing shared-layer imports):

```python
from .optimizer_config import OptimizerConfig
```

**Instance storage** (added after `self._policy` assignment):

```python
        # Configure optimizer (ADR-029)
        self._optimizer = optimizer if optimizer is not None else OptimizerConfig()
```

**Parameter positioning rationale:** `optimizer` is placed after `policy` and before `renderer_factory`. The `renderer_factory` is the most advanced/rare parameter — most users will never touch it. `optimizer` is more commonly useful (convergence tests, hyperparameter experiments) and belongs alongside other configuration objects (`precision`, `strategy`, `policy`).

**Docstring updates:** Add `optimizer` to the `__init__` docstring's `Args:` section:

```
            optimizer: Optimizer hyperparameters (default: Adam with lr=0.001,
                β₁=0.9, β₂=0.999, precision-aware ε). See OptimizerConfig.
```

**Validation:** `Engine(model_spec, param_space, hw)` produces the same `self._optimizer` as `Engine(model_spec, param_space, hw, optimizer=OptimizerConfig())`.

---

### Step 12A.4: Forward optimizer through `_dispatch_learn_plan()`

**Governing authority:** ADR-029 §§2–3  
**File:** `src/shared/engine.py`

**Action:** Update `Engine._dispatch_learn_plan()` to pass `self._optimizer` to `build_learn_plan()`.

**Current call (line ~260):**

```python
        plan = build_learn_plan(
            self._model_spec,
            self._hardware_profile,
            self._strategy,
            batch_size,
            self._policy,
            activation_lifecycle="recompute",
        )
```

**Updated call:**

```python
        plan = build_learn_plan(
            self._model_spec,
            self._hardware_profile,
            self._strategy,
            batch_size,
            self._policy,
            activation_lifecycle="recompute",
            optimizer=self._optimizer,
        )
```

This is the single point where `Engine` connects to `build_learn_plan()`. No other `Engine` methods call `build_learn_plan()`.

**Validation:** Construct an `Engine` with `optimizer=OptimizerConfig(learning_rate=0.05)`, call `train_batch()`, and inspect the rendered plan's scalar parameters (requires integration test with CPU backend — Step 12A.7).

---

### Step 12A.5: Export `OptimizerConfig` from `src/shared/__init__.py`

**Governing authority:** ADR-029 §1  
**File:** `src/shared/__init__.py`

**Action:** Import `OptimizerConfig` and add it to `__all__`.

**Import addition** (alongside existing configuration imports, before the User-facing API block):

```python
from .optimizer_config import OptimizerConfig
```

**`__all__` addition** (in the "Configuration types" section):

```python
    "OptimizerConfig",
```

Place it after `"StabilizationPolicy"` and before the problem type strategy exports, grouping it with the other optimizer-adjacent configuration.

**Validation:** `from src.shared import OptimizerConfig` succeeds. `OptimizerConfig` appears in `src.shared.__all__`.

---

### Step 12A.6: Write unit tests for `OptimizerConfig`

**File:** `tests/tier1/test_optimizer_config.py` (new file)

**Action:** Create unit tests validating `OptimizerConfig` construction, defaults, validation, and `resolve_epsilon()`.

**Test cases:**

```python
# tests/tier1/test_optimizer_config.py
"""Unit tests for OptimizerConfig (ADR-029)."""
import pytest

from src.shared.optimizer_config import OptimizerConfig


@pytest.mark.tier1
class TestOptimizerConfigDefaults:
    """Verify default values match the previously hardcoded constants."""

    def test_default_learning_rate(self):
        cfg = OptimizerConfig()
        assert cfg.learning_rate == 0.001

    def test_default_beta1(self):
        cfg = OptimizerConfig()
        assert cfg.beta1 == 0.9

    def test_default_beta2(self):
        cfg = OptimizerConfig()
        assert cfg.beta2 == 0.999

    def test_default_epsilon_is_none(self):
        cfg = OptimizerConfig()
        assert cfg.epsilon is None

    def test_frozen(self):
        cfg = OptimizerConfig()
        with pytest.raises(AttributeError):
            cfg.learning_rate = 0.01  # type: ignore[misc]


@pytest.mark.tier1
class TestOptimizerConfigValidation:
    """Verify __post_init__ rejects invalid values."""

    def test_negative_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=-0.001)

    def test_zero_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=0.0)

    def test_beta1_negative(self):
        with pytest.raises(ValueError, match="beta1 must be in"):
            OptimizerConfig(beta1=-0.1)

    def test_beta1_one(self):
        with pytest.raises(ValueError, match="beta1 must be in"):
            OptimizerConfig(beta1=1.0)

    def test_beta2_negative(self):
        with pytest.raises(ValueError, match="beta2 must be in"):
            OptimizerConfig(beta2=-0.1)

    def test_beta2_one(self):
        with pytest.raises(ValueError, match="beta2 must be in"):
            OptimizerConfig(beta2=1.0)

    def test_epsilon_zero(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            OptimizerConfig(epsilon=0.0)

    def test_epsilon_negative(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            OptimizerConfig(epsilon=-1e-8)

    def test_valid_custom_values(self):
        """Accepts valid non-default values without raising."""
        cfg = OptimizerConfig(
            learning_rate=0.1,
            beta1=0.95,
            beta2=0.9999,
            epsilon=1e-7,
        )
        assert cfg.learning_rate == 0.1
        assert cfg.beta1 == 0.95
        assert cfg.beta2 == 0.9999
        assert cfg.epsilon == 1e-7

    def test_beta_zero_allowed(self):
        """beta=0 is valid (disables momentum)."""
        cfg = OptimizerConfig(beta1=0.0, beta2=0.0)
        assert cfg.beta1 == 0.0
        assert cfg.beta2 == 0.0


@pytest.mark.tier1
class TestResolveEpsilon:
    """Verify resolve_epsilon() returns correct epsilon values."""

    def test_none_epsilon_uses_precision_default(self):
        """When epsilon is None, delegate to PrecisionConfig.compute_epsilon."""
        from src.shared.model_spec import ModelSpec

        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=8, output_classes=3,
            num_modules=4, simd_width=4, cache_line_bytes=64,
        )
        cfg = OptimizerConfig()  # epsilon=None
        resolved = cfg.resolve_epsilon(spec.precision)
        assert resolved == spec.precision.compute_epsilon

    def test_explicit_epsilon_overrides_precision(self):
        """When epsilon is set, the precision default is ignored."""
        from src.shared.model_spec import ModelSpec

        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=8, output_classes=3,
            num_modules=4, simd_width=4, cache_line_bytes=64,
        )
        cfg = OptimizerConfig(epsilon=1e-7)
        resolved = cfg.resolve_epsilon(spec.precision)
        assert resolved == 1e-7
```

**Marker:** `@pytest.mark.tier1` on each test class (host-side, no backend required).

**Validation:** `pytest tests/tier1/test_optimizer_config.py -v` — all tests pass.

---

### Step 12A.7: Write integration tests for optimizer plumbing

**File:** `tests/tier1/test_optimizer_plumbing.py` (new file)

**Action:** Create integration tests verifying that custom optimizer values flow through to plan node scalar parameters.

**Test approach:** These tests exercise `build_learn_plan()` directly (Tier 1 — no backend required). They inspect the `ExecutionPlan` to verify that adam_update nodes contain the expected scalar parameters.

```python
# tests/tier1/test_optimizer_plumbing.py
"""Integration tests for optimizer hyperparameter plumbing (ADR-029)."""
import pytest

from src.shared.model_spec import ModelSpec
from src.shared.hardware_profile import HardwareProfile
from src.shared.optimizer_config import OptimizerConfig
from src.shared.plan_builder import build_learn_plan
from src.shared.problem_type_strategy import PlanCceStrategy
from src.shared.stabilization_policy import StabilizationPolicy


_HW = HardwareProfile(
    simd_width=4,
    cache_line_bytes=64,
    max_reduce_fan_in=256,
    max_local_mem_bytes=None,
    global_mem_bytes=4 * 1024**3,
)


def _make_spec() -> ModelSpec:
    return ModelSpec.float32(
        input_dim=4, hidden_dim=8, output_classes=3,
        num_modules=4, simd_width=4, cache_line_bytes=64,
    )


def _make_policy(spec: ModelSpec) -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=0.1,
        compute_fp_format_max=spec.precision.compute_fp_format_max,
    )


def _get_adam_scalars(plan, node_id: str) -> dict:
    """Extract scalar parameters from an adam_update plan node."""
    node = plan.nodes[node_id]
    return node.scalar_params


class TestBuildLearnPlanDefaults:
    """Verify build_learn_plan() without optimizer produces legacy values."""

    @pytest.mark.tier1
    def test_default_scalars_match_hardcoded(self):
        """Default OptimizerConfig produces identical scalars to legacy."""
        spec = _make_spec()
        policy = _make_policy(spec)

        plan = build_learn_plan(spec, _HW, PlanCceStrategy(), 16, policy)

        for node_id in ("adam_update_shared", "adam_update_module", "adam_update_temps"):
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["learning_rate"] == 0.001
            assert scalars["beta1"] == 0.9
            assert scalars["beta2"] == 0.999
            assert scalars["beta1_pow_t"] == 0.9
            assert scalars["beta2_pow_t"] == 0.999
            assert scalars["epsilon"] == spec.precision.compute_epsilon

    @pytest.mark.tier1
    def test_explicit_none_matches_omitted(self):
        """optimizer=None produces same plan as omitting the argument."""
        spec = _make_spec()
        policy = _make_policy(spec)

        plan_omitted = build_learn_plan(spec, _HW, PlanCceStrategy(), 16, policy)
        plan_none = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=None,
        )

        for node_id in ("adam_update_shared", "adam_update_module", "adam_update_temps"):
            if node_id not in plan_omitted.nodes:
                continue
            assert (
                _get_adam_scalars(plan_omitted, node_id)
                == _get_adam_scalars(plan_none, node_id)
            )


class TestBuildLearnPlanCustomOptimizer:
    """Verify custom OptimizerConfig values appear in plan nodes."""

    @pytest.mark.tier1
    def test_custom_learning_rate(self):
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(learning_rate=0.01)

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in ("adam_update_shared", "adam_update_module", "adam_update_temps"):
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["learning_rate"] == 0.01
            # beta values remain default
            assert scalars["beta1"] == 0.9
            assert scalars["beta2"] == 0.999

    @pytest.mark.tier1
    def test_custom_betas(self):
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(beta1=0.95, beta2=0.9999)

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in ("adam_update_shared", "adam_update_module", "adam_update_temps"):
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["beta1"] == 0.95
            assert scalars["beta2"] == 0.9999
            assert scalars["beta1_pow_t"] == 0.95   # beta1 ** 1
            assert scalars["beta2_pow_t"] == 0.9999  # beta2 ** 1

    @pytest.mark.tier1
    def test_custom_epsilon(self):
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(epsilon=1e-7)

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in ("adam_update_shared", "adam_update_module", "adam_update_temps"):
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["epsilon"] == 1e-7

    @pytest.mark.tier1
    def test_all_custom_values(self):
        """All four optimizer parameters overridden simultaneously."""
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(
            learning_rate=0.1, beta1=0.85, beta2=0.9995, epsilon=1e-6,
        )

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in ("adam_update_shared", "adam_update_module", "adam_update_temps"):
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["learning_rate"] == 0.1
            assert scalars["beta1"] == 0.85
            assert scalars["beta2"] == 0.9995
            assert scalars["beta1_pow_t"] == 0.85
            assert scalars["beta2_pow_t"] == 0.9995
            assert scalars["epsilon"] == 1e-6
```

**Note on adam_update node IDs:** The plan builder creates adam_update nodes with IDs `adam_update_shared`, `adam_update_module`, and `adam_update_temps`. The test iterates over all three, skipping any that don't exist in the plan (some model configurations may not produce all three).

**Validation:** `pytest tests/tier1/test_optimizer_plumbing.py -v` — all tests pass.

---

### Step 12A.8: Validate backward compatibility (rollback gate)

**Action:** Run the full existing test suite and confirm zero regressions.

**Commands:**

```bash
cd architectures/averaging_ensembled_classifier

# 1. Run all Tier 1 tests (plan correctness — no backend required)
pytest tests/tier1/ -v 2>&1 | tee /tmp/phase12a-tier1.log

# 2. Run all Tier 2 tests (kernel correctness — requires backends)
pytest tests/tier2/ -v 2>&1 | tee /tmp/phase12a-tier2.log

# 3. Run Tier 3 tests (cross-backend parity)
pytest tests/tier3/ -v 2>&1 | tee /tmp/phase12a-tier3.log

# 4. Run new Phase 12A tests
pytest tests/tier1/test_optimizer_config.py tests/tier1/test_optimizer_plumbing.py -v 2>&1 | tee /tmp/phase12a-new.log

# 5. Grep for failures
grep -E "FAILED|ERROR" /tmp/phase12a-tier1.log /tmp/phase12a-tier2.log /tmp/phase12a-tier3.log /tmp/phase12a-new.log
```

**Pass criteria (rollback gate):**

| Criterion | Required result |
| :--- | :--- |
| All existing Tier 1 tests | PASS (zero changes to existing test files) |
| All existing Tier 2 tests | PASS |
| All existing Tier 3 tests | PASS |
| `test_optimizer_config.py` | PASS |
| `test_optimizer_plumbing.py` | PASS |
| `build_learn_plan()` without `optimizer` | Identical scalars to pre-change |

**If any existing test fails:** The change has broken backward compatibility. Diagnose the failure — likely causes:
1. Import error from new `optimizer_config` module — check circular import.
2. `build_learn_plan()` signature change broke a positional-argument call — check that `optimizer` is keyword-only (positioned after `activation_lifecycle` default).
3. Floating-point value mismatch — verify `OptimizerConfig` defaults match the exact literals.

---

## 5. Data Type Reference

### `OptimizerConfig`

```
OptimizerConfig (frozen dataclass)
├── learning_rate: float = 0.001     # Adam step size
├── beta1: float = 0.9              # First moment decay rate
├── beta2: float = 0.999            # Second moment decay rate
├── epsilon: float | None = None    # Adam denominator stabilizer (None → precision-aware)
│
├── __post_init__()                 # Validates all fields
└── resolve_epsilon(precision: PrecisionConfig) -> float
                                    # Returns epsilon or precision.compute_epsilon
```

### Scalar parameter mapping

```
OptimizerConfig           →  _adam_scalars dict        →  adam_update plan node
─────────────────────────────────────────────────────────────────────────
learning_rate             →  "learning_rate"           →  ScalarParamSpec("learning_rate")
beta1                     →  "beta1"                   →  ScalarParamSpec("beta1")
beta1                     →  "beta1_pow_t"             →  ScalarParamSpec("beta1_pow_t")  [beta1**1]
beta2                     →  "beta2"                   →  ScalarParamSpec("beta2")
beta2                     →  "beta2_pow_t"             →  ScalarParamSpec("beta2_pow_t")  [beta2**1]
resolve_epsilon(prec)     →  "epsilon"                 →  ScalarParamSpec("epsilon")      [REAL type]
```

The `beta1_pow_t` and `beta2_pow_t` values are initialized to `beta1**1` and `beta2**1` respectively (i.e., just `beta1` and `beta2`). The host-side bias correction logic is responsible for computing `beta1**t` and `beta2**t` for subsequent training steps.

---

## 6. Backward Compatibility Matrix

| Call site | Current code | After Phase 12A | Behavior change |
| :--- | :--- | :--- | :--- |
| `build_learn_plan(spec, hw, strat, bs, pol)` | Hardcoded scalars | `optimizer=None` → `OptimizerConfig()` → same scalars | None |
| `build_learn_plan(spec, hw, strat, bs, pol, "recompute")` | Hardcoded scalars | `optimizer=None` → same | None |
| `Engine(spec, ps, hw)` | No optimizer | `optimizer=None` → `OptimizerConfig()` | None |
| `Engine(spec, ps, hw, policy=pol)` | No optimizer | Same | None |
| `test_multi_precision_config.py` | `build_learn_plan(spec, _HW, PlanCceStrategy(), batch_size=16, policy=policy)` | `optimizer=None` default | None |
| `bench_cpu_backend.py` | Multiple `build_learn_plan()` calls with 5–6 positional args | `optimizer=None` default | None |
| `main_orchestrator.py` | `build_learn_plan()` without `optimizer` | `optimizer=None` default | None |

**Critical observation:** All existing callers use positional args for the first 5 parameters and `activation_lifecycle` either as positional or keyword. No existing caller passes an `optimizer` argument. The new parameter is keyword-argument-safe because it follows the existing keyword-defaulted `activation_lifecycle`.

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Circular import from `optimizer_config.py` importing `PrecisionConfig`.** The `resolve_epsilon` method needs `PrecisionConfig`. If `precision_config.py` transitively imports something that imports `optimizer_config.py`, a circular dependency occurs. | Low | Medium | Use `TYPE_CHECKING` guard for the type annotation. The runtime import is deferred to the method body, or `PrecisionConfig` is accepted as a duck-typed argument (any object with `.compute_epsilon`). Verify at Step 12A.1 by running `python -c "from src.shared.optimizer_config import OptimizerConfig"`. |
| **Positional argument breakage.** A caller passes `build_learn_plan()` arguments positionally in a way that the new `optimizer` parameter shifts expected positions. | Low | High | Verify that `optimizer` is positioned after `activation_lifecycle` (which has a default). All existing callers either use ≤6 positional args (the first 5 mandatory + `activation_lifecycle`) or use keyword args. Since `optimizer` is the 8th parameter and `activation_lifecycle` is the 7th with a default, no positional call can accidentally target `optimizer`. |
| **Float literal precision mismatch.** `OptimizerConfig().beta2` might not be bitwise-identical to the literal `0.999` in the current hardcoded dict due to Python float representation. | Negligible | High | Python float literals are parsed deterministically. `OptimizerConfig().beta2` is the same float object as `0.999`. Validated in Step 12A.7 with `assert scalars["beta2"] == 0.999`. |
| **Existing test imports break.** Adding a new import to `plan_builder.py` or `engine.py` could cause import errors if the test environment's `sys.path` doesn't include the right directories. | Low | Medium | The import `from .optimizer_config import OptimizerConfig` is a relative import within `src/shared/`, identical in form to existing imports (`from .precision_config import PrecisionConfig`). If existing shared-layer imports work, this one will too. |
| **Phase 12 convergence criteria need re-validation.** Once per-problem learning rates are plumbed, the convergence criteria calibrated against lr=0.001 may be too strict or too lenient. | High | Medium | This is expected and documented in ADR-029 §6. Phase 12 explicitly states: "Convergence criteria should be re-validated with their intended learning rates." Phase 12A provides the plumbing; Phase 12 handles the calibration. |

---

## References

- [ADR-029: Optimizer Hyperparameter Configuration](../adr/ADR-029-optimizer-hyperparameter-configuration.md) — governing ADR for this phase
- [ADR-018: User-Facing API](../adr/ADR-018-user-facing-api.md) — Engine constructor, `train_batch()` convenience method, Choice 5A (ephemeral)
- [ADR-002: Plan Node Types](../adr/ADR-002-plan-node-types-and-synchronization-structure.md) — closed five-node taxonomy, scalar parameter passing
- [ADR-008: Precision Configuration](../adr/ADR-008-precision-configuration.md) — `compute_epsilon` derivation
- [ADR-028: Convergence Testing](../adr/ADR-028-convergence-testing-standard-problems.md) — `BaselineHyperparameters`, per-problem learning rates
- [Phase 11: User-Facing API](PHASE-11-USER-FACING-API.md) — Engine implementation, `_dispatch_learn_plan()` call site
- [Phase 12: Convergence Testing](PHASE-12-CONVERGENCE-TESTING.md) — "Key constraint: hardcoded optimizer hyperparameters", risk register
