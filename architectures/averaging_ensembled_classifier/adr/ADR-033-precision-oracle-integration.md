# ADR-033: Precision Oracle Integration

**Status:** PROPOSED
**Date:** 2026-04-11
**Deciders:** —
**Triggered by:** Need to validate precision-limited convergence independently of engine decomposition artifacts; Oracle D (NumPy Precision Oracle) implementation completed without test integration
**Depends on:** ADR-008 (Precision Configuration), ADR-016 (Test Strategy), ADR-020 (Mixed-Precision Execution Model), ADR-028 (Convergence Testing Standard Problems), ADR-031 (Mask Buffer Rationalization), ADR-032 (Oracle Testing)
**Constrains:** `tests/oracle/` test matrix, `tests/convergence/` precision baselines
**Extends:** ADR-016 Oracle hierarchy (A/B/C → A/B/C/D), ADR-028 convergence test infrastructure

---

## Decision Summary

| Decision Axis | Selected Option | Rationale |
|:--------------|:----------------|:----------|
| Oracle D integration scope | **Dual integration** — oracle/ and convergence/ | D validates precision modeling (oracle/) AND provides convergence baselines (convergence/) |
| Cross-oracle validation strategy | **D vs C at FP64, then D sweeps precisions** | Establishes D's correctness via C agreement, then uses D's authority for precision parameterization |
| Convergence baseline strategy | **Trajectory comparison (warn, not fail)** | Engine applies clipping/tiling that D does not model; divergence is informative, not necessarily incorrect |
| Config bridging | **Explicit adapter + dtype-keyed mapping** | `OracleDConfig` ↔ `OracleConfig` via adapter functions; `NumpyPrecisionSpec` → convergence criteria via dtype-name keying (no direct `NumpyPrecisionSpec` ↔ `PrecisionConfig` converter) |
| Precision adjustment derivation | **Oracle-D-derived, static pre-computation** | Replace hand-tuned `PRECISION_ADJUSTMENTS` with Oracle D trajectory deltas; base criteria remain hand-authored design targets; refresh script + CI staleness gate |

---

## Context

### The Oracle Hierarchy Before This ADR

The test oracle hierarchy established by `doc_archive/Oracle.md` and integrated through ADR-016 comprises three oracles:

| Oracle | Implementation | Dependency | Authority |
|:-------|:--------------|:-----------|:----------|
| **A** (Faithful) | Manual FP64 PyTorch | torch | Gradient calculus + tiling + clipping fidelity |
| **B** (Autograd) | torch.autograd FP64 | torch | Gradient correctness (B1=flat, B2=tiled) |
| **C** (Convergence) | Standard PyTorch FP64 | torch | Multi-step convergence, optimizer state management |

All three oracles operate at FP64 throughout.  They validate the *mathematical correctness* of the training pipeline — gradient formulas, tiled decomposition, clipping policy, Adam optimizer.  They do **not** model precision effects: storage narrowing, compute-precision arithmetic, state-precision accumulation.

### The Gap

The engine executes with three-role precision (ADR-008, ADR-020):

- **Storage:** Bandwidth optimization (FP16, FP8, or FP32)
- **Compute:** Arithmetic fidelity (FP16, FP32, or FP64)
- **State:** Optimizer stability (FP32 or FP64)

When a convergence test fails under mixed precision but passes at FP32, the diagnosis question is: *"Is this a precision effect (expected) or a decomposition bug (unexpected)?"*  Without a precision-aware reference, this question is unanswerable — every mixed-precision failure requires manual analysis.

### Oracle D

Oracle D (`NumPyPrecisionOracle`) implements the same mathematical model as Oracles A/B/C:

```
Input → ReLU(X @ W_shared.T + b_shared)
      → per-module (H @ W_module[m] + b_module[m])
      → temperature scaling (logits / T[m])
      → softmax (CCE) or sigmoid (BCE)
      → loss
```

with faithful three-role precision simulation at every storage/compute/state boundary.  Key properties:

1. **Zero torch dependency.**  Pure NumPy.  No shared code with Oracles A/B/C.
2. **Configurable precision boundaries:**
   - Hidden activation storage (always modeled — dominant effect)
   - Gradient storage round-trips (configurable hops)
   - Logit storage (optional, secondary)
   - Probability storage (optional, tertiary)
3. **Mask strategy support:**  `"explicit"` (compute-precision derivative truth) and `"recompute"` (derived from stored activations), matching ADR-031's `MaskStrategy` policy.
4. **FP8 support:**  Software scalar quantizer matching `kernels.cl.h`'s `store_storage_fp8` logic, with `ml_dtypes` acceleration when available.
5. **State-precision-accumulation-aware Adam:**  EMA in `ACCUM_TYPE = max(compute, state)`, bias correction in `COMPUTE_TYPE`, parameter update in `ACCUM_TYPE`.

Oracle D's authority: **precision-limited convergence rate.**  It answers: *"Given only the mathematical model and the precision configuration, what convergence trajectory should we expect?"*

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).**  Oracle D's precision modeling must track the architecture's three-role precision contract.  If Oracle D's precision boundaries don't match the engine's, this signals incomplete architectural modeling — not a reason for ad-hoc workarounds.

2. **ADR-008 (Precision Configuration).**  The `NumpyPrecisionSpec` dataclass mirrors `PrecisionConfig`'s three-role structure (storage/compute/state).  The two types are not directly convertible — `NumpyPrecisionSpec` uses numpy dtypes while `PrecisionConfig` uses engine-native types — but `precision_adjustments.py` maps between them via dtype-name keying to share the same adjustment table.

3. **ADR-016 (Test Strategy).**  The oracle hierarchy is the arbitration backbone.  Oracle D extends it with a new dimension (precision) orthogonal to the existing correctness dimensions (A=manual, B=autograd, C=convergence).

4. **ADR-028 (Convergence Testing Standard Problems).**  Convergence tests need precision-parameterized baselines.  Currently, `precision_adjustments.py` uses hand-tuned deltas; Oracle D provides computed baselines.

5. **ADR-031 (Mask Buffer Rationalization).**  Oracle D implements both mask strategies (`explicit`/`recompute`), enabling direct validation that the strategies agree where they should (FP32/FP64) and diverge where expected (FP8).

---

## Decisions

### 1. Oracle Test Integration (`tests/oracle/`)

Oracle D is integrated into the oracle test package with the following test axes:

| Test Class | Comparison | Purpose |
|:-----------|:-----------|:--------|
| `TestDvsCForwardParity` | D (FP64) vs C | Validates D's NumPy implementation matches C's PyTorch at FP64 |
| `TestDvsCMultiStep` | D (FP64) vs C over N steps | Validates multi-step trajectory parity (accumulated drift tolerance) |
| `TestKnownSolutionConvergencePrecision` | D alone, per precision | D must converge on separable problems at each precision level |
| `TestPrecisionGapIsolation` | D FP16 vs D FP64 | Confirms precision modeling is non-trivial — trajectories must differ |
| `TestMaskStrategy` | D explicit vs D recompute | Strategies agree at FP32/FP64 (tested); FP8 divergence test deferred until FP8 convergence is validated |
| `TestGradientStorageModeling` | D with/without gradient storage | No-op at FP64, measurable at FP16 |

**Config bridging:** `oracle_config_to_d_config()`, `sync_oracle_d_from_torch()`, and `torch_state_from_oracle_d()` in `tests/oracle/conftest.py` handle the `OracleConfig` ↔ `OracleDConfig` mapping and bidirectional torch ↔ numpy state conversions.

**Tolerance model:** Multi-step D-vs-C parity (`TestDvsCMultiStep`) uses absolute tolerance `1e-9 × (1 + √steps)`.  The sqrt-growth models accumulated per-step accumulation-order divergence between NumPy and PyTorch transcendental implementations — linear growth would be too conservative for short runs, while constant tolerance would be too tight for long runs.

### 2. Convergence Test Integration (`tests/convergence/`)

Oracle D serves as a precision-aware baseline via the `oracle_baseline` module:

| Component | File | Purpose |
|:----------|:-----|:--------|
| `OracleDBaselineConfig` | `oracle_baseline.py` | Maps convergence problem hyperparameters to `OracleDConfig` |
| `run_oracle_d_baseline()` | `oracle_baseline.py` | Runs Oracle D through the same epoch/batch loop as the engine, returns `TrainingHistory` |
| `TestOracleDSelfConvergence` | `test_convergence_oracle_d.py` | Oracle D must converge on standard problems (precondition for baseline use) |
| `TestOracleDPrecisionConvergence` | `test_convergence_oracle_d.py` | Oracle D converges across precision configs on standard problems |
| `TestEngineVsOracleD` | `test_convergence_oracle_d.py` | Engine trajectory compared to Oracle D baseline (warn on divergence) |
| `adjust_criteria_for_oracle_d_precision()` | `precision_adjustments.py` | Maps `NumpyPrecisionSpec` to convergence criteria adjustments |

**Trajectory comparison semantics:**  Engine vs Oracle D divergence uses `trajectory_comparison.compare_trajectories()` which *warns* rather than *fails*.  The comparison operates on **accuracy curves** (not loss curves) — relative accuracy difference exceeding 10% per epoch triggers a `UserWarning`.  The engine applies staged clipping (Quadratic Scaling Policy) and tiled reduction that Oracle D deliberately does not model — these are expected divergence sources.  The comparison's value is diagnostic: large or unexpected divergence patterns guide investigation.

**Current engine-vs-D precision scope:**  `TestEngineVsOracleD` currently runs at default FP32 precision only.  Precision-parameterized engine-vs-D comparisons (FP16, FP8) are deferred until the engine's mixed-precision convergence is validated independently via `TestEngineConvergence`.

### 3. Cross-Validation Flow

The validation chain for a new precision configuration proceeds as:

```
Step 1:  D (FP64) vs C (FP64)        — D's implementation correctness
         ↓ must agree within 1e-9 absolute tolerance
Step 2:  D (target precision) alone   — convergence baseline established
         ↓ must converge (loss decreases)
Step 3:  Engine (target precision) vs D — decomposition correctness under precision
         ↓ warn on trajectory divergence
Step 4:  Engine meets convergence criteria (adjusted for precision)
```

Steps 1–2 are independent of the engine.  Step 3 is diagnostic.  Step 4 is the hard pass/fail gate.

### 4. Oracle-D-Derived Convergence Criteria

#### Problem

The convergence test suite contains two layers of hand-coded numerical expectations:

1. **Per-problem base criteria** (`problems/*.py`): `FAST_CRITERIA` and `FULL_CRITERIA` with hand-picked `accuracy_threshold`, `loss_threshold`, `epoch_budget` per problem.
2. **Precision adjustment table** (`precision_adjustments.py`): `PRECISION_ADJUSTMENTS` dict maps dtype names to hand-tuned `(accuracy_delta, loss_delta, epoch_multiplier)` triples.

These values were empirically chosen during initial development.  When Oracle D's precision model or the engine's pipeline evolves, the table drifts — silently making tests either too lenient (missing regressions) or too strict (false failures).

#### Replaceable Values — Precision Adjustments

The `PRECISION_ADJUSTMENTS` table is fully replaceable by Oracle D.  Instead of hand-tuning "FP16 loses 2% accuracy," the system:

1. Runs Oracle D at FP32 (baseline) and the target precision on the same problem, seed, and hyperparameters.
2. Computes the trajectory delta: `accuracy_delta = D_target.final_accuracy - D_fp32.final_accuracy`, `loss_delta = D_target.final_loss - D_fp32.final_loss`, `epoch_multiplier = D_target.epochs_to_threshold / D_fp32.epochs_to_threshold`.
3. Uses the derived delta as the precision adjustment.

This eliminates hand-tuning and automatically adapts when Oracle D's precision model improves or when new precision configurations are added.

| Component | File | Purpose |
|:----------|:-----|:--------|
| `derive_precision_adjustment()` | `precision_adjustments.py` | Runs Oracle D at a reference and target precision, returns `PrecisionAdjustment` |
| `derive_all_adjustments()` | `precision_adjustments.py` | Iterates over precisions × problems, returns the adjustment table |

#### Non-Replaceable Values — Base Criteria

The base `FULL_CRITERIA` and `FAST_CRITERIA` in each problem module should **not** be replaced by Oracle D.  These are **design targets** — the architect's contract for what the system must achieve.  Oracle D predicts what the mathematical model *can* achieve; the criteria specify what the system *must* achieve.  These are different concerns:

- Oracle D might converge to 99% on Iris at FP32, but the architect sets the target at 97% to allow headroom for decomposition artifacts.
- `FAST_CRITERIA` are intentionally loose smoke-test gates — deriving them from Oracle D would defeat their purpose (fast, lenient, never flaky).

However, Oracle D **validates** the base criteria: if Oracle D at FP32 cannot reach a problem's `accuracy_threshold`, either the target is unreachable or Oracle D has a bug.  `TestOracleDSelfConvergence` already serves this role.

#### Execution Model — Static Pre-Computation with Refresh Script

Oracle D derivation runs at **maintenance time**, not test time:

1. A script (`scripts/refresh_precision_adjustments.py`) runs `derive_all_adjustments()` across the standard problem × precision matrix.
2. The script emits a Python dict literal that replaces `PRECISION_ADJUSTMENTS` in `precision_adjustments.py`.
3. The refreshed table is committed alongside any Oracle D model changes.

This avoids adding Oracle D runtime cost to every test run while keeping the table synchronized with Oracle D's current model.  A CI check can compare the committed table against a fresh derivation and fail if they diverge (staleness gate).

#### Transition

The transition is incremental:

| Phase | Change |
|:------|:-------|
| **Phase 1 (current)** | Hand-tuned `PRECISION_ADJUSTMENTS` table, Oracle D comparison is diagnostic only |
| **Phase 2** | Add `derive_precision_adjustment()` and refresh script; validate committed table against derivation in CI |
| **Phase 3** | CI staleness gate enforced; hand-tuned values fully replaced by Oracle D-derived values |

---

## Consequences

### Positive

- **Precision failures become diagnosable.**  When the engine fails to converge under FP16, Oracle D shows whether the mathematical model with FP16 precision would converge.  If Oracle D converges but the engine doesn't, it's a decomposition or pipeline bug.  If Oracle D also fails, it's a fundamental precision limitation.
- **New precision configs get computed baselines.**  Adding a new `NumpyPrecisionSpec` factory (e.g., `fp8_e4m3_f64`) and including it in the test parametrization lists enables the full convergence test matrix for that configuration — no hand-tuned criteria needed.  (The factory alone is insufficient; it must be added to `PRECISION_CONFIGS` / `PRECISION_CONFIGS_CONVERGENCE` in the test modules.)
- **Precision adjustments are derived, not guessed.**  The `PRECISION_ADJUSTMENTS` table becomes a computed artifact of Oracle D's precision model rather than a hand-tuned constant.  When Oracle D's model evolves (e.g., additional gradient storage hops), the refresh script propagates the change to convergence criteria automatically.  New precision configurations (FP8 variants, mixed compute/state) get correct adjustments on first derivation — no trial-and-error tuning cycle.
- **Staleness is detectable.**  The CI staleness gate catches drift between Oracle D's current model and the committed adjustment table.  This prevents the silent failure mode where the table becomes too lenient after an Oracle D improvement.
- **Mask strategy validation is precision-parameterized.**  ADR-031's `explicit`/`recompute` equivalence for non-FP8 configurations now has a direct test via Oracle D.
- **Zero new torch dependency.**  Oracle D uses only NumPy.  The convergence baseline can run in environments without PyTorch.

### Negative

- **Oracle D does not model the engine's clipping.**  Trajectory comparison between engine and Oracle D has a structural divergence source.  This is by design — D's authority is "precision-only" — but it means engine-vs-D comparison is diagnostic, not definitive.
- **Two config types with a latent `normalize_epsilon` gap.**  `OracleDConfig` and `OracleConfig` are structurally similar but not identical (`OracleDConfig` has `normalize_epsilon`; `OracleConfig` has `t_algorithmic`, `lambda_`, `compute_fp_format_max`).  The `oracle_config_to_d_config` adapter maps shared fields but does not set `normalize_epsilon`, relying on `OracleDConfig`'s default (1e-7).  Because `OracleConfig` uses its single `epsilon` field (also defaulting to 1e-7) for both Adam and Node 21 normalization, the two defaults coincide today.  However, any change to `OracleConfig.epsilon` will silently break this coincidence — Oracle D's normalization divisor will diverge from C's, producing false D-vs-C parity failures.  The adapter should explicitly set `normalize_epsilon=cfg.epsilon`.
- **Convergence oracle baseline adds test runtime.**  Oracle D training on standard problems is pure Python/NumPy — significantly slower than the C/OpenCL engine for large problems.  Mitigated by running Oracle D baselines only in `convergence_full` tests.
- **Refresh script maintenance.**  The `scripts/refresh_precision_adjustments.py` script must be kept synchronized with the problem module interface (`load()`, `BASELINE`, criteria fields).  A new problem module requires a corresponding entry in the refresh script's problem registry.

### Risks

- **Oracle D's precision model may not capture all engine effects.**  The engine's multi-hop gradient pipeline (write → clip → write → reduce) applies more storage round-trips than Oracle D's default `gradient_storage_hops=1`.  Setting `gradient_storage_hops=2` improves fidelity but only approximates the full pipeline.  Residual divergence must be analyzed case-by-case.
- **FP8 software quantizer accuracy.**  The scalar FP8 round-trip functions match `kernels.cl.h`'s logic by construction, but edge cases (subnormal rounding) have bounded error.  This is documented in the quantizer's inline comments.
- **Oracle D bugs propagate to criteria.**  If Oracle D's precision model has a systematic error (e.g., over-optimistic FP8 convergence), the derived adjustments will be too lenient, and the engine tests will pass incorrectly.  Mitigated by the cross-validation chain (Steps 1–2): D-vs-C parity at FP64 catches implementation bugs; precision sweep non-triviality catches broken precision modeling.  The base criteria (hand-authored design targets) remain as an independent safety net.

---

## Files Changed

| File | Change |
|:-----|:-------|
| `tests/oracle/conftest.py` | Added Oracle D imports, `oracle_config_to_d_config()`, `sync_oracle_d_from_torch()`, `torch_state_from_oracle_d()`, numpy data fixtures |
| `tests/oracle/test_precision_oracle.py` | **New.** Oracle D test matrix (6 test classes) |
| `tests/convergence/oracle_baseline.py` | **New.** Oracle D baseline runner producing `TrainingHistory` |
| `tests/convergence/test_convergence_oracle_d.py` | **New.** Oracle D self-convergence + engine vs Oracle D comparison |
| `tests/convergence/precision_adjustments.py` | Added `adjust_criteria_for_oracle_d_precision()` for `NumpyPrecisionSpec`; `derive_precision_adjustment()` and `derive_all_adjustments()` (Phase 2) |
| `scripts/refresh_precision_adjustments.py` | **New (Phase 2).** Runs Oracle D derivation across problem × precision matrix, emits updated `PRECISION_ADJUSTMENTS` table |

---

## Related ADRs

- **ADR-008** — Defines the three-role precision model that Oracle D simulates.
- **ADR-016** — Establishes the oracle hierarchy that Oracle D extends.
- **ADR-020** — Defines mixed-precision execution semantics that Oracle D faithfully models.
- **ADR-028** — Defines the convergence test infrastructure that Oracle D plugs into.
- **ADR-031** — Defines mask strategies that Oracle D validates per-precision.
