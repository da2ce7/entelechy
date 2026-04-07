# ADR-028: Convergence Testing on Standard Problems

**Status:** PROPOSED  
**Date:** 2026-04-07  
**Deciders:** —  
**Triggered by:** Need to validate end-to-end training correctness beyond numerical parity  
**Depends on:** ADR-016 (Test Strategy), ADR-018 (User-Facing API), ADR-008 (Precision Configuration)  
**Constrains:** CI pipeline configuration, test runtime budgets  
**Extends:** ADR-016 Tier 3 (Cross-Backend Parity)

---

## Decision Summary

| Decision Axis | Selected Option | Rationale |
|:---|:---|:---|
| Test Organization | **Option B** — Separate category | Preserves tier model for correctness; convergence is distinct quality layer |
| Problem Selection | **Option C** — Curated canonical suite | Known baselines; sklearn-bundled datasets; manageable CI runtime |
| Convergence Criteria | **Option F** — Multi-criteria | Catches slow convergence, instability, and numerical blowup |
| Cross-Backend Validation | **Option H** — Independent convergence | Robust to floating-point divergence; trajectory divergence logged as warning |

---

## Context

### The Gap Between Correctness and Convergence

ADR-016 establishes a three-tier test framework that validates:

1. **Tier 1 — Plan correctness:** The shared orchestration layer produces structurally valid execution plans.
2. **Tier 2 — Kernel correctness:** Each backend's kernels produce numerically correct outputs for known inputs.
3. **Tier 3 — Cross-backend parity:** All backends produce equivalent outputs for the same execution plan.

These tiers verify *operational correctness* — that the machinery works as specified. They do not verify *functional correctness* — that the system, when used for its intended purpose (training classifiers), achieves the expected learning outcomes.

A system passing all three tiers could still fail to converge on real problems due to:

- **Hyperparameter sensitivity:** Default learning rate, momentum, or regularization values that are technically valid but practically pathological.
- **Numerical accumulation patterns:** Precision erosion or gradient magnitude scaling issues that emerge only over many training steps — not visible in single-kernel correctness tests.
- **Phase interaction bugs:** Subtle Act/Learn phase sequencing errors that produce correct single-batch outputs but corrupt state across epochs.
- **Stabilization policy misconfiguration:** Threshold schedules that either clip too aggressively (preventing learning) or too permissively (allowing gradient explosion over extended training).
- **Adam optimizer pathologies:** Moment estimate initialization, bias correction timing, or learning rate scheduling edge cases.

### The Role of Standard Problems

Standard machine learning benchmark problems provide a controlled validation surface:

| Property | Value for Convergence Testing |
|:---|:---|
| **Known optimal performance** | Published baselines establish the expected accuracy ceiling |
| **Reproducible data** | Canonical datasets eliminate data-loading variance |
| **Diverse problem characteristics** | Linear separability, non-linearity, class imbalance, dimensionality variation |
| **Cross-framework comparison** | Results can be compared against PyTorch/TensorFlow/scikit-learn implementations |

The architecture's single-hidden-layer constraint (CONCEPT.md §Core Components) limits the problem difficulty ceiling: deep learning benchmarks requiring convolutional or attention layers are out of scope. The target is: *problems solvable by a well-tuned single-hidden-layer network should converge when trained by this system*.

### Relationship to ADR-016 Tier 3

ADR-016 Tier 3 validates cross-backend parity at two granularities:

1. **Per-kernel parity:** Same kernel, same inputs, different backends → same outputs.
2. **End-to-end parity:** Same plan (Act + Learn), different backends → same final parameters.

Convergence testing extends this model:

3. **Multi-epoch convergence parity:** Same problem, same hyperparameters, different backends → same training trajectory and final accuracy (within tolerance).
4. **Convergence quality:** Each backend, when trained in isolation, achieves the expected accuracy for the problem.

The distinction: ADR-016 Tier 3 asks "do backends agree?" — convergence testing asks "do backends succeed?"

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** If convergence tests reveal patterns that violate current abstractions (e.g., a problem that requires hyperparameter auto-tuning outside the `ModelSpec` vocabulary), the response is to extend the architecture — not to workaround with ad-hoc test fixtures.

2. **CONCEPT.md §5 (Unified Execution Model).** Sequential and Event-Triggered execution modes must both produce convergent training. Convergence tests validate this for the Sequential case (pre-labeled batches); Event-Triggered validation requires distinct test infrastructure (deferred to future ADR if needed).

3. **ADR-008 (Precision Configuration).** Convergence behavior varies with precision: FP16 training may converge more slowly or to different final accuracy than FP32. Convergence criteria must be precision-aware.

4. **ADR-016 Decision Drivers 1, 3, 8.** Test framework consistency (Option A — layered pytest), per-precision tolerance tables, and CPU as reference oracle all apply to convergence tests.

5. **ADR-018 (User-Facing API).** Convergence tests exercise the `WorkTicket` API (`train_batch()`, `submit()`, `finalize()`) as the user-facing entry point. Tests validate the API's fitness for real training workflows.

6. **CI execution time budget.** Convergence tests are inherently slow (multi-epoch training). Test design must balance coverage against CI runtime constraints. The framework must support tiered execution: fast smoke tests for every commit, full convergence suites for nightly/release.

7. **Reproducibility.** Convergence tests must produce deterministic results across runs. This requires: fixed random seeds for weight initialization and data shuffling, deterministic backend execution (CPU; or GPU with deterministic mode where supported), and version-pinned datasets.

8. **Problem diversity.** The test suite must cover: (a) CCE vs. BCE operating modes, (b) varying input dimensionality, (c) varying class counts, (d) varying dataset sizes, (e) linearly separable and non-linearly separable problems.

---

## Options Considered

### Test Suite Organization

#### Option A: Convergence as Tier 4

Add a fourth tier to ADR-016's framework. Tier 4 tests are multi-epoch training runs that assert convergence criteria. Marker: `@pytest.mark.tier4`. Skip logic extends `_build_config` with `CONVERGENCE_TESTS_ENABLED`.

**Advantages:**
- Natural extension of the existing tier vocabulary.
- Tier 3 oracle model (CPU reference) directly applicable.
- Consistent skip/collection logic.

**Disadvantages:**
- Tier numbering implies dependency ordering (Tier 4 depends on Tier 1-3 passing), which is conceptually accurate but may confuse the distinction between *correctness* (Tiers 1-3) and *quality* (Tier 4).

#### Option B: Convergence as separate test category

Convergence tests live in `tests/convergence/` with their own markers (`@pytest.mark.convergence`, `@pytest.mark.convergence_fast`, `@pytest.mark.convergence_full`). No tier number.

**Advantages:**
- Clear conceptual separation: correctness vs. quality.
- Supports independent execution (`pytest -m convergence`) without tier-number confusion.
- Fast/full distinction is first-class rather than parameterized.

**Disadvantages:**
- Duplicates some fixture infrastructure from Tier 3 (backend setup, tolerance tables).
- Less integrated with the tier skip logic.

### Problem Selection

#### Option C: Curated canonical suite

A fixed set of problems with published baselines:

| Problem | Mode | Features | Classes | Samples | Baseline Accuracy | Rationale |
|:---|:---|:---|:---|:---|:---|:---|
| Iris | CCE | 4 | 3 | 150 | 97% | Canonical small-scale CCE; existing fixture (`fp32_iris_spec`) |
| MNIST (subset) | CCE | 784 | 10 | 10,000 | 92% | High-dimensional input; single-hidden-layer ceiling |
| Fashion-MNIST (subset) | CCE | 784 | 10 | 10,000 | 85% | Harder than MNIST; validates generalization |
| XOR | BCE | 2 | 1 | 200 | 99% | Minimal non-linear BCE; synthetic, deterministic |
| Two Moons | BCE | 2 | 1 | 1,000 | 98% | Curved decision boundary; validates non-linearity |
| Breast Cancer Wisconsin | BCE | 30 | 1 | 569 | 96% | Real-world BCE with moderate dimensionality |
| Digits (8×8) | CCE | 64 | 10 | 1,797 | 95% | Mid-scale CCE; sklearn bundled dataset |

**Advantages:**
- Known baselines provide clear pass/fail criteria.
- Covers both CCE and BCE modes.
- Dataset sizes are manageable for CI.

**Disadvantages:**
- Fixed suite may miss edge cases.
- MNIST/Fashion-MNIST require external data download (or bundling).

#### Option D: Parameterized synthetic suite

Generate synthetic classification problems with controlled properties:

```python
@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
@pytest.mark.parametrize("n_features", [4, 64, 784])
@pytest.mark.parametrize("n_classes", [2, 5, 10])
@pytest.mark.parametrize("separability", ["linear", "nonlinear"])
def test_convergence_synthetic(n_samples, n_features, n_classes, separability, ...):
    X, y = generate_synthetic_problem(n_samples, n_features, n_classes, separability, seed=42)
    ...
```

**Advantages:**
- Exhaustive coverage of the problem-space dimensions.
- No external data dependencies.

**Disadvantages:**
- Synthetic problems may not expose real-world numerical edge cases.
- Combinatorial explosion: 3×3×3×2 = 54 configurations per precision per backend.
- No published baselines — requires self-generated accuracy targets.

### Convergence Criteria

#### Option E: Accuracy threshold only

Define a minimum accuracy for each problem. Test passes if final accuracy ≥ threshold.

```python
CONVERGENCE_THRESHOLDS = {
    "iris": {"accuracy": 0.95, "epochs": 100},
    "mnist": {"accuracy": 0.90, "epochs": 50},
    ...
}
```

**Advantages:**
- Simple, interpretable criteria.
- Matches ML practitioner intuition.

**Disadvantages:**
- Does not detect slow convergence (100 epochs to reach 95% when 20 should suffice).
- Does not detect training instability (oscillating loss).
- Does not validate loss monotonicity.

#### Option F: Multi-criteria convergence

Define accuracy threshold, loss threshold, and epoch budget. Test passes if:
1. Final accuracy ≥ accuracy threshold, AND
2. Final loss ≤ loss threshold, AND
3. Convergence achieved within epoch budget, AND
4. No NaN/Inf in any intermediate value.

Optionally: loss must decrease monotonically (within noise tolerance) after warmup epochs.

**Advantages:**
- Catches more failure modes: slow convergence, instability, numerical blowup.
- Monotonicity check detects oscillation.

**Disadvantages:**
- More complex criteria require more problem-specific tuning.
- Monotonicity may be too strict for noisy problems.

### Cross-Backend Convergence Validation

#### Option G: Final-state parity

Train on each backend independently, compare final accuracy and final parameters. Backends must produce the same accuracy (within tolerance) and similar final parameters (within higher tolerance due to floating-point accumulation over many steps).

**Advantages:**
- Validates that backends converge to the *same* solution, not just *a* solution.
- Natural extension of ADR-016 Tier 3.

**Disadvantages:**
- Floating-point divergence across thousands of steps may exceed reasonable tolerances.
- Requires deterministic backends; non-deterministic GPU execution makes parity assertion fragile.

#### Option H: Independent convergence

Train on each backend independently. Each must meet the convergence criteria for the problem. No cross-backend parameter comparison.

**Advantages:**
- Robust to floating-point divergence.
- Each backend is validated in isolation.

**Disadvantages:**
- Does not detect systematic backend-specific bugs that still converge but to different (possibly wrong) solutions.

---

## Decision

### Test Organization: Option B (separate category)

Convergence tests are organized under `tests/convergence/` with dedicated markers. This preserves the ADR-016 tier model for correctness testing while establishing convergence as a distinct quality assurance layer.

**Markers:**
- `@pytest.mark.convergence` — all convergence tests
- `@pytest.mark.convergence_fast` — smoke tests (~10s per test); runs on every commit (CPU/FP32)
- `@pytest.mark.convergence_full` — comprehensive tests (~1-5 min per test); nightly/release only

**File organization:**
```
tests/
├── convergence/
│   ├── __init__.py
│   ├── conftest.py                        # Problem fixtures, convergence criteria, backend setup
│   ├── problems/
│   │   ├── __init__.py
│   │   ├── iris.py                        # Iris dataset loader + baseline config
│   │   ├── digits.py                      # sklearn digits dataset
│   │   ├── xor.py                         # XOR synthetic generator (seed=42)
│   │   ├── two_moons.py                   # Two Moons synthetic generator
│   │   ├── breast_cancer.py               # sklearn breast cancer dataset
│   │   ├── mnist.py                       # MNIST subset loader (optional dependency)
│   │   └── fashion_mnist.py               # Fashion-MNIST subset loader (optional)
│   ├── test_convergence_cce.py            # CCE-mode convergence tests
│   ├── test_convergence_bce.py            # BCE-mode convergence tests
│   └── test_convergence_precision.py      # Precision-specific convergence validation
```

### Problem Selection: Option C (curated canonical suite)

The canonical problem suite provides known baselines and deterministic validation:

| Problem ID | Mode | Source | Samples | Features | Out Dim† | Fast Acc (ep) | Full Acc (ep) | Loss ≤ | Max Epochs |
|:---|:---|:---|---:|---:|---:|:---|:---|---:|---:|
| `iris` | CCE | sklearn | 150 | 4 | 3 | 90% (20) | 97% (100) | 0.15 | 200 |
| `digits` | CCE | sklearn | 1,797 | 64 | 10 | 85% (30) | 95% (150) | 0.25 | 300 |
| `xor` | BCE | synthetic (seed=42) | 200 | 2 | 1 | 95% (50) | 99% (200) | 0.05 | 500 |
| `two_moons` | BCE | sklearn | 1,000 | 2 | 1 | 90% (30) | 98% (150) | 0.10 | 300 |
| `breast_cancer` | BCE | sklearn | 569 | 30 | 1 | 90% (30) | 96% (100) | 0.15 | 200 |
| `mnist_1k` | CCE | torchvision‡ | 1,000 | 784 | 10 | 80% (20) | 92% (100) | 0.35 | 200 |
| `fashion_1k` | CCE | torchvision‡ | 1,000 | 784 | 10 | 70% (30) | 85% (150) | 0.50 | 300 |

† **Out Dim** = output layer dimension. For CCE mode, equals class count. For BCE mode, equals 1 (binary classification uses single sigmoid output).

‡ Optional dependency — requires `torchvision` or manual data download. Tests are skipped if unavailable. Core convergence validation uses only sklearn-bundled datasets.

**Column semantics:**
- **Fast Acc (ep):** Minimum accuracy and epoch budget for `convergence_fast` tests.
- **Full Acc (ep):** Minimum accuracy and epoch budget for `convergence_full` tests (maps to `ConvergenceCriteria.epoch_budget`).
- **Max Epochs:** Absolute ceiling before test failure; used only if convergence is not reached within the Full epoch budget.

**Fast vs. Full:**
- Fast tests (`convergence_fast`) run for limited epochs with relaxed thresholds — sufficient to verify that training progresses without numerical failure.
- Full tests (`convergence_full`) run to convergence with strict thresholds — validate production-quality training.

```python
# tests/convergence/problems/mnist.py

def load_mnist_subset(n_samples: int = 1000, seed: int = 42):
    """Load MNIST subset; skip test if torchvision unavailable."""
    torchvision = pytest.importorskip("torchvision", reason="MNIST requires torchvision")
    from torchvision.datasets import MNIST
    dataset = MNIST(root="~/.cache/torchvision", train=True, download=True)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    X = dataset.data[indices].numpy().reshape(n_samples, -1).astype(np.float32) / 255.0
    y = dataset.targets[indices].numpy()
    return X, y
```

### Convergence Criteria: Option F (multi-criteria)

Each problem defines a `ConvergenceCriteria` dataclass:

```python
@dataclass(frozen=True)
class ConvergenceCriteria:
    problem_id: str
    mode: Literal["CCE", "BCE"]
    accuracy_threshold: float          # minimum final accuracy
    loss_threshold: float              # maximum final loss
    epoch_budget: int                  # must converge within this many epochs
    warmup_epochs: int                 # epochs before monotonicity check
    monotonicity_tolerance: float | None  # relative tolerance: loss[i+1] ≤ loss[i] * (1 + tol); None disables check
    nan_inf_allowed: bool              # always False for production; True only for debugging
```

The `TrainingHistory` dataclass records per-epoch metrics for assertion and trajectory comparison:

```python
@dataclass
class TrainingHistory:
    loss_curve: list[float]            # loss at end of each epoch
    accuracy_curve: list[float]        # accuracy at end of each epoch
    has_nan_inf: bool                  # True if any NaN/Inf encountered
    
    @property
    def final_loss(self) -> float:
        return self.loss_curve[-1]
    
    @property
    def final_accuracy(self) -> float:
        return self.accuracy_curve[-1]
    
    def epochs_to_threshold(self, accuracy_threshold: float) -> int:
        """Return first epoch where accuracy >= threshold, or len(curve) if never reached."""
        for i, acc in enumerate(self.accuracy_curve):
            if acc >= accuracy_threshold:
                return i + 1  # 1-indexed epoch count
        return len(self.accuracy_curve)
```

**Test assertion logic:**

```python
def assert_convergence(history: TrainingHistory, criteria: ConvergenceCriteria):
    # 1. No NaN/Inf at any point
    assert not history.has_nan_inf, "Training produced NaN/Inf values"
    
    # 2. Final accuracy meets threshold
    assert history.final_accuracy >= criteria.accuracy_threshold, (
        f"Final accuracy {history.final_accuracy:.2%} < threshold {criteria.accuracy_threshold:.2%}"
    )
    
    # 3. Final loss meets threshold
    assert history.final_loss <= criteria.loss_threshold, (
        f"Final loss {history.final_loss:.4f} > threshold {criteria.loss_threshold:.4f}"
    )
    
    # 4. Converged within epoch budget
    epochs_needed = history.epochs_to_threshold(criteria.accuracy_threshold)
    assert epochs_needed <= criteria.epoch_budget, (
        f"Required {epochs_needed} epochs to reach {criteria.accuracy_threshold:.0%}; budget was {criteria.epoch_budget}"
    )
    
    # 5. Monotonicity after warmup (optional, may be disabled for noisy problems)
    # Uses relative tolerance: loss[i+1] ≤ loss[i] * (1 + monotonicity_tolerance)
    if criteria.monotonicity_tolerance is not None:
        for i in range(criteria.warmup_epochs, len(history.loss_curve) - 1):
            max_allowed = history.loss_curve[i] * (1 + criteria.monotonicity_tolerance)
            if history.loss_curve[i + 1] > max_allowed:
                rel_increase = (history.loss_curve[i + 1] - history.loss_curve[i]) / history.loss_curve[i]
                raise AssertionError(
                    f"Non-monotonic loss at epoch {i + 1}: "
                    f"{history.loss_curve[i]:.4f} → {history.loss_curve[i + 1]:.4f} "
                    f"(+{rel_increase:.1%}, tolerance={criteria.monotonicity_tolerance:.1%})"
                )
```

### Cross-Backend Validation: Option H (independent convergence) + trajectory logging

Each backend is validated independently against the convergence criteria. Cross-backend parity is not required for final parameters (floating-point divergence over thousands of steps is expected).

**However:** training trajectories (accuracy per epoch, loss per epoch) are logged and compared qualitatively. A warning (not failure) is raised if backends diverge in trajectory shape by more than `TRAJECTORY_DIVERGENCE_THRESHOLD` (default: 10% relative accuracy) at any epoch. This catches gross algorithmic differences without asserting bit-exact parity.

```python
# Default trajectory divergence threshold for cross-backend warnings
TRAJECTORY_DIVERGENCE_THRESHOLD = 0.10  # 10% relative accuracy difference

def compare_trajectories(oracle_history: TrainingHistory, 
                         comparison_history: TrainingHistory,
                         divergence_threshold: float = TRAJECTORY_DIVERGENCE_THRESHOLD):
    """Warn if trajectories diverge; do not fail."""
    for epoch in range(min(len(oracle_history.accuracy_curve), 
                           len(comparison_history.accuracy_curve))):
        oracle_acc = oracle_history.accuracy_curve[epoch]
        comp_acc = comparison_history.accuracy_curve[epoch]
        if oracle_acc > 0:
            rel_diff = abs(oracle_acc - comp_acc) / oracle_acc
            if rel_diff > divergence_threshold:
                warnings.warn(
                    f"Trajectory divergence at epoch {epoch}: "
                    f"oracle={oracle_acc:.2%}, comparison={comp_acc:.2%}, diff={rel_diff:.1%}"
                )
```

**CI warning capture:** Pytest's `-W error::UserWarning` flag is *not* used for convergence tests — trajectory divergence warnings are informational, not failures. Instead, CI captures warnings via `pytest --tb=short -rw` (report warnings summary) and writes them to the test log artifact. Repeated trajectory divergence across commits may indicate a backend-specific issue worth investigation, but is not a blocking failure.

---

## Precision-Specific Adjustments

Convergence criteria are parameterized by `PrecisionConfig`:

| Precision | Key | Accuracy Adjustment | Loss Adjustment | Epoch Adjustment | Rationale |
|:---|:---|:---|:---|:---|:---|
| FP32 | `float32` | Baseline | Baseline | Baseline | Reference precision |
| FP16 | `float16` | −2% | +0.05 | +20% | Reduced gradient precision affects fine tuning |
| FP64 | `float64` | +0% | −0.01 | −10% | Higher precision may converge faster |
| FP8 (E4M3) | `float8_e4m3` | −5% | +0.1 | +50% | Significant quantization; validate basic convergence |

The **Key** column shows the `PrecisionConfig.storage_dtype.name` value used for lookup in `PRECISION_ADJUSTMENTS`.

```python
def adjust_criteria_for_precision(base: ConvergenceCriteria, 
                                   precision: PrecisionConfig) -> ConvergenceCriteria:
    adjustments = PRECISION_ADJUSTMENTS[precision.storage_dtype.name]  # e.g., "float32", "float16"
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

---

## Hyperparameter Baselines

Each problem defines a baseline hyperparameter configuration that is known to converge. These baselines are not necessarily optimal — they are *reliable defaults* validated during ADR acceptance.

```python
@dataclass(frozen=True)
class BaselineHyperparameters:
    hidden_size: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    gradient_clip_threshold: float
    batch_size: int

PROBLEM_BASELINES = {
    "iris": BaselineHyperparameters(
        hidden_size=16,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        gradient_clip_threshold=1.0,
        batch_size=32,
    ),
    "digits": BaselineHyperparameters(
        hidden_size=128,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-4,
        gradient_clip_threshold=1.0,
        batch_size=64,
    ),
    "xor": BaselineHyperparameters(
        hidden_size=8,
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        gradient_clip_threshold=5.0,
        batch_size=32,
    ),
    "two_moons": BaselineHyperparameters(
        hidden_size=16,
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        gradient_clip_threshold=1.0,
        batch_size=64,
    ),
    "breast_cancer": BaselineHyperparameters(
        hidden_size=32,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-4,
        gradient_clip_threshold=1.0,
        batch_size=32,
    ),
    "mnist_1k": BaselineHyperparameters(
        hidden_size=256,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-4,
        gradient_clip_threshold=1.0,
        batch_size=64,
    ),
    "fashion_1k": BaselineHyperparameters(
        hidden_size=256,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=1e-4,
        gradient_clip_threshold=1.0,
        batch_size=64,
    ),
}
```

---

## CI Integration

### Test Execution Matrix

| CI Trigger | Convergence Scope | Backends Tested | Precision Scope |
|:---|:---|:---|:---|
| Every commit | `convergence_fast` only | CPU only | FP32 only |
| Pull request | `convergence_fast` | All enabled | FP32, FP16 |
| Nightly | `convergence_fast` + `convergence_full` | All enabled | FP32, FP16, FP8 |
| Release | Full suite | All enabled | All supported |

### Pytest Configuration

```ini
# pyproject.toml or pytest.ini

[tool.pytest.ini_options]
markers = [
    "convergence: All convergence tests",
    "convergence_fast: Fast convergence smoke tests (<30s each)",
    "convergence_full: Full convergence tests (1-5 min each)",
]

# Default: skip slow convergence tests
addopts = "-m 'not convergence_full'"
```

### Skip Logic

```python
# tests/convergence/conftest.py

import pytest
from averaging_ensembled_classifier import _build_config

def pytest_configure(config):
    # Register convergence markers
    config.addinivalue_line("markers", "convergence: all convergence tests")
    config.addinivalue_line("markers", "convergence_fast: fast smoke tests")
    config.addinivalue_line("markers", "convergence_full: full training tests")

@pytest.fixture(scope="session")
def available_backends():
    return {
        "cpu": getattr(_build_config, "BACKEND_CPU", False),
        "opencl": getattr(_build_config, "BACKEND_OPENCL", False),
        "vulkan": getattr(_build_config, "BACKEND_VULKAN", False),
    }

def skip_if_backend_unavailable(backend: str):
    return pytest.mark.skipif(
        not getattr(_build_config, f"BACKEND_{backend.upper()}", False),
        reason=f"Backend '{backend}' not available"
    )
```

---

## Reproducibility Guarantees

### Random Seed Management

All sources of randomness are seeded deterministically:

```python
@dataclass(frozen=True)
class ReproducibilityConfig:
    weight_init_seed: int = 42
    data_shuffle_seed: int = 42
    numpy_seed: int = 42
    # Note: GPU backends may require additional deterministic mode configuration

def set_reproducibility(config: ReproducibilityConfig):
    import numpy as np
    np.random.seed(config.numpy_seed)
    # Backend-specific deterministic mode (if supported)
```

### Dataset Versioning

sklearn-bundled datasets are version-stable (tied to sklearn version). For optional datasets (MNIST, Fashion-MNIST), tests record the dataset hash and assert consistency:

```python
# TODO: Compute actual hashes during implementation (seed=42, n_samples=1000)
DATASET_HASHES = {
    "mnist_1k": "sha256:<computed-during-implementation>",
    "fashion_1k": "sha256:<computed-during-implementation>",
}

def verify_dataset_integrity(problem_id: str, X: np.ndarray, y: np.ndarray):
    import hashlib
    data_hash = hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest()
    expected = DATASET_HASHES.get(problem_id)
    if expected:
        assert f"sha256:{data_hash}" == expected, f"Dataset hash mismatch for {problem_id}"
```

---

## Success Metrics for ADR Acceptance

This ADR is accepted when:

1. **Infrastructure exists:** `tests/convergence/` directory structure with `conftest.py`, problem loaders, and test stubs.

2. **Iris convergence validated:** `test_convergence_cce.py::test_iris_convergence` passes on the CPU backend with FP32 precision, achieving ≥97% accuracy within 100 epochs.

3. **BCE mode validated:** At least one BCE problem (`xor` or `two_moons`) passes on the CPU backend.

4. **Fast/Full distinction operational:** `pytest -m convergence_fast` completes in <60s total; `pytest -m convergence_full` runs extended tests.

5. **CI integration:** GitHub Actions (or equivalent) runs `convergence_fast` on every push, `convergence_full` nightly.

---

## Limitations

Convergence testing validates that the system can successfully train classifiers, but has inherent scope boundaries:

1. **Single-hidden-layer ceiling.** The architecture (CONCEPT.md §Core Components) is constrained to single-hidden-layer networks. Problems requiring deeper architectures, convolutional layers, or attention mechanisms are out of scope — convergence tests cannot validate capabilities the architecture does not have.

2. **Hyperparameter sensitivity not tested.** Tests use fixed baseline hyperparameters. A model that converges with baselines but fails with reasonable perturbations (e.g., 2× learning rate) may still pass all convergence tests.

3. **Generalization not validated.** Tests measure training accuracy, not test-set generalization. A model that overfits dramatically would pass convergence tests.

4. **Event-Triggered mode excluded.** Only Sequential execution mode (pre-labeled batches) is tested. Event-Triggered mode (external readiness signals) requires distinct infrastructure.

5. **Non-deterministic GPU execution.** GPU backends with non-deterministic reduction ordering may exhibit run-to-run variance. Trajectory comparison warnings account for this, but systematic GPU-specific bugs may be masked.

---

## Future Extensions

The following are explicitly deferred to future ADRs:

- **Event-Triggered mode convergence:** Testing the Act/Learn split with external readiness signals.
- **Multi-backend ensemble averaging:** Convergence testing where multiple backends contribute to the ensemble.
- **Hyperparameter sensitivity analysis:** Automated sweeps to validate robustness around baseline configurations.
- **Regression detection:** Tracking accuracy/loss across commits to detect performance regressions.
- **Larger-scale problems:** ImageNet-scale or text classification (requires convolutional/attention layers beyond current architecture scope).

---

## References

- [ADR-016](ADR-016-test-strategy.md) — Test Strategy (three-tier framework)
- [ADR-018](ADR-018-user-facing-api.md) — User-Facing API (WorkTicket model)
- [ADR-008](ADR-008-precision-configuration.md) — Precision Configuration
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §5 Unified Execution Model; Validation Scenarios
- [sklearn datasets](https://scikit-learn.org/stable/datasets.html) — Bundled dataset documentation
