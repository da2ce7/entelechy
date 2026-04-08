# ADR-029: Optimizer Hyperparameter Configuration

**Status:** ACCEPTED  
**Date:** 2026-04-07  
**Deciders:** —  
**Triggered by:** Phase 12 convergence testing reveals that hardcoded Adam hyperparameters (lr=0.001, β₁=0.9, β₂=0.999) prevent per-problem tuning and force convergence criteria to compensate for suboptimal learning rates  
**Depends on:** ADR-018 (User-Facing API), ADR-002 (Plan Node Types), ADR-008 (Precision Configuration)  
**Constrains:** `build_learn_plan()`, `Engine` constructor and `train_batch()` API, Phase 12 convergence baselines  
**Enables:** ADR-028 (Convergence Testing) — allows `BaselineHyperparameters.learning_rate` to be exercised

---

## Context

### The Current State

`build_learn_plan()` constructs adam_update plan nodes with hardcoded optimizer hyperparameters:

```python
# src/shared/plan_builder.py (current)
_adam_scalars = {
    "learning_rate": 0.001,
    "beta1_pow_t": 0.9,
    "beta2_pow_t": 0.999,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": model_spec.precision.compute_epsilon,
}
```

These values are embedded at plan-construction time and propagated as scalar parameters to every adam_update node in the Learn plan. Neither `build_learn_plan()` nor the `Engine` API accepts user-specified optimizer parameters.

The `Engine` constructor (ADR-018) accepts `ModelSpec`, `ParameterSpace`, `HardwareProfile`, `PrecisionConfig`, backend selector, `PlanProblemTypeStrategy`, and `StabilizationPolicy` — but no optimizer configuration. The convenience method `train_batch()` delegates to `build_learn_plan()` with no mechanism to forward hyperparameters.

### The Legacy Surface

The older `TrainingOrchestrator` defines a `TrainingHyperparams` dataclass:

```python
@dataclass(frozen=True)
class TrainingHyperparams:
    epochs: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    temp_min: float
    temp_max: float
    stabilization: StabilizationConfig
```

However, `TrainingOrchestrator` also does not forward these values to `build_learn_plan()`. The `TrainingHyperparams` fields are consumed only for FP8 scaling logic and host-side bias correction — the plan builder never sees them.

### The Phase 12 Signal

Phase 12 (Convergence Testing, ADR-028) defines per-problem `BaselineHyperparameters` with intended learning rates that differ from the hardcoded 0.001:

| Problem | Intended lr | Hardcoded lr | Impact |
|:---|:---|:---|:---|
| Iris | 0.01 | 0.001 | 10× slower convergence; requires relaxed epoch budget |
| XOR | 0.1 | 0.001 | 100× slower; may not converge within practical epoch budget |
| Digits | 0.001 | 0.001 | No impact (matches) |
| Two Moons | 0.01 | 0.001 | Slower convergence |
| Breast Cancer | 0.001 | 0.001 | No impact |

Phase 12's risk register identifies this as a High likelihood / High impact risk:

> "Problems whose ideal learning rate differs (Iris at 0.01, XOR at 0.1) will train at a suboptimal lr, potentially requiring relaxed accuracy thresholds or greatly increased epoch budgets."

The phase plan explicitly defers optimizer parameterization to a future ADR:

> "When `build_learn_plan()` is extended to accept optimizer hyperparameters (tracked as future work beyond Phase 12's scope), the baselines and criteria should be re-validated with their intended learning rates."

Per CONCEPT.md §1 (Architectural Elegance Feedback), the convergence testing gap is a signal that the architecture's abstractions are incomplete. The optimizer hyperparameters are a first-class training concern that must be represented as a documented architectural primitive rather than left as hardcoded constants.

### The Kernel Contract Surface

The adam_update kernel contract (Phase 3 Update contracts) already accepts optimizer hyperparameters as scalar parameters:

```python
ScalarParamSpec("learning_rate", "src", "REAL"),
ScalarParamSpec("beta1_pow_t", "src", "REAL"),
ScalarParamSpec("beta2_pow_t", "src", "REAL"),
ScalarParamSpec("beta1", "src", "REAL"),
ScalarParamSpec("beta2", "src", "REAL"),
```

The kernel layer is already parameterized. The gap is entirely in the host orchestration layer: the path from user intent to plan-node scalar parameters has no configurable surface.

### Host-Side Bias Correction

The plan builder computes initial values for `beta1_pow_t` and `beta2_pow_t` (the power terms for Adam bias correction). These are currently set to the beta values themselves (i.e., `beta1**1` and `beta2**1` for step t=1). The host is responsible for updating these across training steps. This computation must use the user-specified beta values, not hardcoded constants.

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** Hardcoded optimizer hyperparameters are an ad-hoc shortcut. The convergence testing gap confirms that this shortcut has become a constraint. The optimization pressure (different problems need different learning rates) must be formalized as a first-class primitive.

2. **ADR-018 (User-Facing API).** The Engine is the user-facing entry point. Optimizer configuration must be expressible through the Engine's construction or method API without requiring users to construct plan nodes directly.

3. **ADR-002 (Plan Node Types).** The five-node taxonomy is a closed set. Optimizer configuration must flow as scalar parameters within existing adam_update nodes, not as new node types.

4. **ADR-008 (Precision Configuration).** The epsilon value interacts with precision: `compute_epsilon` is derived from `PrecisionConfig.compute_dtype`. The optimizer configuration must respect this relationship — users may specify epsilon, but the default must remain precision-aware.

5. **Minimal disruption to existing tests.** Tiers 1–3 tests (ADR-016) construct plans with the current hardcoded values. The change must preserve backward compatibility: existing test code that does not specify optimizer parameters must continue to produce identical plans.

6. **Phase 12 unblocking.** The primary consumer is Phase 12 convergence testing, which needs per-problem learning rates. The design must allow `BaselineHyperparameters` values to flow through `make_engine()` to `build_learn_plan()`.

---

## Options Considered

### Representation of Optimizer Configuration

#### Option A: Frozen dataclass — `OptimizerConfig`

A new shared-layer frozen dataclass encapsulating all Adam hyperparameters:

```python
@dataclass(frozen=True)
class OptimizerConfig:
    """Adam optimizer hyperparameters for plan construction."""
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float | None = None  # None → derive from PrecisionConfig.compute_epsilon

    def resolve_epsilon(self, precision: PrecisionConfig) -> float:
        """Return epsilon, falling back to precision-aware default."""
        if self.epsilon is not None:
            return self.epsilon
        return precision.compute_epsilon
```

**Advantages:**
- Type-safe, immutable, self-documenting.
- Defaults preserve backward compatibility (all existing code gets lr=0.001).
- Sentinel `None` for epsilon preserves the precision-aware default while allowing override.
- Consistent with existing shared-layer dataclass patterns (`ModelSpec`, `PrecisionConfig`, `StabilizationPolicy`).

**Disadvantages:**
- New type in the shared layer; one more constructor argument for `Engine`.

#### Option B: Keyword arguments on `build_learn_plan()`

Add `learning_rate`, `beta1`, `beta2`, `epsilon` as keyword arguments with defaults:

```python
def build_learn_plan(
    model_spec, hardware, strategy, batch_size, policy,
    activation_lifecycle="recompute",
    learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=None,
) -> ExecutionPlan:
```

**Advantages:**
- No new types. Minimal change surface.

**Disadvantages:**
- Four new parameters on an already long signature.
- No single object to pass through the Engine; the Engine must forward each parameter individually.
- No reusable representation for convergence test baselines.
- Harder to extend if additional optimizer parameters are needed in the future (e.g., weight decay, learning rate schedules).

#### Option C: Extend `StabilizationPolicy`

Add optimizer hyperparameters to the existing `StabilizationPolicy` dataclass, since both concern training dynamics.

**Advantages:**
- No new types; reuses an existing plumbing path (Engine already accepts `StabilizationPolicy`).

**Disadvantages:**
- Violates single-responsibility: `StabilizationPolicy` concerns gradient clipping thresholds, not optimizer step-size configuration. The two are conceptually orthogonal.
- `StabilizationPolicy` is already consumed by plan nodes that don't need optimizer hyperparameters; adding unused fields creates confusing coupling.

### Plumbing Through the Engine API

#### Option D: Constructor injection

`Engine.__init__()` accepts an optional `OptimizerConfig` (or individual kwargs), stored as instance state and forwarded to every `build_learn_plan()` call:

```python
class Engine:
    def __init__(self, ..., optimizer: OptimizerConfig | None = None):
        self._optimizer = optimizer or OptimizerConfig()
```

**Advantages:**
- Set-once semantics: all training calls use the same optimizer configuration.
- Consistent with how `StabilizationPolicy` and `PrecisionConfig` are plumbed.
- `train_batch()` signature unchanged — no new per-call parameters.

**Disadvantages:**
- Cannot change optimizer parameters mid-training session without constructing a new Engine. (This is consistent with the ephemeral design — ADR-018 Choice 5A — but forecloses learning rate scheduling within a single Engine lifecycle.)

#### Option E: Per-call injection via `train_batch()`

`train_batch()` accepts an optional `OptimizerConfig` that overrides the Engine default:

```python
def train_batch(self, x_data, y_data, optimizer=None):
    opt = optimizer or self._optimizer
    ...
```

**Advantages:**
- Allows per-batch optimizer configuration (useful for learning rate scheduling).

**Disadvantages:**
- Complicates the convenience method's minimal-ceremony design (ADR-018).
- Learning rate scheduling is out of scope for Phase 12 and the current architecture.
- Per-batch configuration is unusual; most frameworks configure the optimizer once per training run.

---

## Decision

### §1: Representation — Option A: `OptimizerConfig` frozen dataclass

A new frozen dataclass `OptimizerConfig` is added to the shared orchestration layer at `src/shared/optimizer_config.py`.

```python
@dataclass(frozen=True)
class OptimizerConfig:
    """Adam optimizer hyperparameters for plan construction.

    All fields have defaults matching the previously hardcoded values,
    ensuring backward compatibility for code that does not specify
    optimizer configuration.
    """
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float | None = None

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

**Rationale:** The dataclass pattern is consistent with existing shared-layer primitives. The default values ensure that all existing code — including Tiers 1–3 tests — produces identical plans without modification. The `None` sentinel for epsilon preserves the precision-aware computation from ADR-008.

**Validation constraints:**
- `learning_rate > 0`
- `0 ≤ beta1 < 1`
- `0 ≤ beta2 < 1`
- `epsilon > 0` (when not None)

Validation is performed at construction via `__post_init__`.

### §2: Plumbing — Option D: Constructor injection on `Engine`

The `Engine.__init__()` signature is extended with an optional `optimizer` parameter:

```python
class Engine:
    def __init__(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware_profile: HardwareProfile,
        precision: PrecisionConfig | None = None,
        backend: str = "auto",
        strategy: PlanProblemTypeStrategy | None = None,
        policy: StabilizationPolicy | None = None,
        optimizer: OptimizerConfig | None = None,
        renderer_factory: Callable[..., PlanRenderer] | None = None,
    ):
        self._optimizer = optimizer or OptimizerConfig()
```

When `optimizer` is `None`, the default `OptimizerConfig()` provides the same values as the current hardcoded constants — backward compatible.

`Engine._dispatch_learn_plan()` forwards the optimizer config to `build_learn_plan()`.

### §3: `build_learn_plan()` Signature Extension

`build_learn_plan()` gains an optional `optimizer` parameter:

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

Inside the function, the hardcoded `_adam_scalars` dict is replaced:

```python
from .optimizer_config import OptimizerConfig

opt = optimizer or OptimizerConfig()
_adam_scalars = {
    "learning_rate": opt.learning_rate,
    "beta1_pow_t": opt.beta1,        # beta1**1 for step t=1
    "beta2_pow_t": opt.beta2,        # beta2**1 for step t=1
    "beta1": opt.beta1,
    "beta2": opt.beta2,
    "epsilon": opt.resolve_epsilon(model_spec.precision),
}
```

When `optimizer` is `None`, the default `OptimizerConfig()` produces identical scalar values to the current hardcoded constants. Existing callers require no changes.

### §4: Host-Side Bias Correction Consistency

The host-side bias correction logic (computing `beta1_pow_t` and `beta2_pow_t` across training steps) must use the same beta values as the plan. Currently, the `TrainingOrchestrator` hardcodes `beta1=0.9, beta2=0.999` in its bias correction loop. When migrating to the Engine API, bias correction must derive beta values from `self._optimizer.beta1` and `self._optimizer.beta2`.

The Engine's current design is ephemeral (ADR-018 Choice 5A) — each `train_batch()` call constructs a fresh plan. Bias correction power terms (`beta1**t`, `beta2**t`) are currently embedded in the plan at construction time. The epoch-stepping logic that increments these power terms across training steps is the caller's responsibility (or, in the `TrainingOrchestrator`, handled by the training loop). This ADR does not change the bias correction stepping mechanism — it only ensures that the initial values and the base (beta1, beta2) are derived from _the same_ `OptimizerConfig`.

### §5: Backward Compatibility

| Consumer | Change required | Rationale |
|:---|:---|:---|
| `Engine` users (Phase 11 API) | None | `optimizer=None` default preserves current behavior |
| `build_learn_plan()` callers | None | `optimizer=None` default preserves current behavior |
| Tier 1–3 tests (ADR-016) | None | Tests that don't specify optimizer get identical plans |
| Phase 12 convergence tests | Update `make_engine()` | Pass `OptimizerConfig(learning_rate=baseline.learning_rate)` |
| `TrainingOrchestrator` (legacy) | Optional migration | Can construct `OptimizerConfig` from `TrainingHyperparams` and pass to `build_learn_plan()` |

### §6: Phase 12 Integration

Phase 12's `make_engine()` helper gains an optional `optimizer` parameter:

```python
def make_engine(*, ..., optimizer: OptimizerConfig | None = None) -> Engine:
    return Engine(..., optimizer=optimizer)
```

Each problem's `BaselineHyperparameters` is consumed at test time:

```python
engine = make_engine(
    ...,
    optimizer=OptimizerConfig(
        learning_rate=problem.BASELINE.learning_rate,
        beta1=problem.BASELINE.beta1,
        beta2=problem.BASELINE.beta2,
    ),
)
```

After this ADR is implemented, Phase 12's `BaselineHyperparameters.learning_rate` field transitions from "NOT YET PLUMBED" documentation to an exercised value. Convergence criteria should be re-validated against per-problem intended learning rates rather than the fixed lr=0.001.

> **Implementation status:** Phase 12A (2026-04-07) implemented this ADR. `OptimizerConfig` is available at `src/shared/optimizer_config.py`, exported from `src.shared`, and accepted by both `Engine.__init__()` and `build_learn_plan()`. All Tier 1–3 tests pass without modification. The Phase 12 plan has been updated to exercise per-problem learning rates via `OptimizerConfig`.

---

## Consequences

### Positive

- **Convergence testing unblocked.** Phase 12 can exercise per-problem learning rates, removing the primary risk from the convergence testing risk register.
- **Zero-disruption backward compatibility.** All defaults match current hardcoded values. No existing code changes behavior.
- **Consistent with existing patterns.** `OptimizerConfig` follows the same frozen-dataclass, constructor-injection pattern as `StabilizationPolicy`, `PrecisionConfig`, and `ModelSpec`.
- **Kernel layer unchanged.** The adam_update kernel contract already accepts these values as scalar parameters. No kernel or backend changes required.

### Negative

- **One more constructor argument on `Engine`.** The Engine's `__init__` signature grows by one parameter. This is mitigated by the optional default.
- **Learning rate scheduling not addressed.** Constructor injection sets optimizer parameters once per Engine lifecycle. Per-step learning rate schedules require either reconstructing the Engine or a future extension (out of scope).

### Neutral

- **`TrainingOrchestrator` migration is optional.** The legacy orchestrator can adopt `OptimizerConfig` at its convenience. It continues to work with its existing `TrainingHyperparams` until deprecated.

---

## References

- [CONCEPT.md §1](../CONCEPT.md) — Architectural Elegance Feedback
- [ADR-018: User-Facing API](ADR-018-user-facing-api.md) — Engine constructor, `train_batch()` convenience method
- [ADR-002: Plan Node Types](ADR-002-plan-node-types-and-synchronization-structure.md) — closed five-node taxonomy, scalar parameter passing
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `compute_epsilon` derivation
- [ADR-028: Convergence Testing](ADR-028-convergence-testing-standard-problems.md) — `BaselineHyperparameters`, per-problem learning rates
- [Phase 12 Plan](../plan/PHASE-12-CONVERGENCE-TESTING.md) — "Key constraint: hardcoded optimizer hyperparameters"
