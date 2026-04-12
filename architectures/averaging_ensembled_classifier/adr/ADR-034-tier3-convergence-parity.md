# ADR-034: Tier 3 Multi-Step Convergence Parity

**Status:** PROPOSED
**Date:** 2026-04-11
**Deciders:** —
**Triggered by:** Tier 3 parity tests are single-step (Act/Learn plan); multi-step convergence drift between backends is untested
**Depends on:** ADR-016 (Test Strategy), ADR-028 (Convergence Testing Standard Problems), ADR-032 (Oracle Testing), ADR-033 (Precision Oracle Integration)
**Constrains:** `tests/tier3/` test surface, Tier 3 tolerance model
**Extends:** ADR-016 Tier 3 (Cross-Backend Parity), ADR-028 Cross-Backend Validation

---

## Decision Summary

| Decision Axis | Selected Option | Rationale |
|:--------------|:----------------|:----------|
| Comparison topology | **CPU-oracle vs each GPU backend** (existing Tier 3 model) | CPU is Oracle-D-validated; reuses ADR-016's oracle selection |
| Failure semantics | **Hard-fail parity assertion** on loss curves | Multi-step drift is a genuine backend bug, not a tolerance ambiguity |
| Tolerance model | **Per-step cumulative tolerance** with step-scaled growth | Floating-point divergence compounds across steps; constant tolerance is too tight |
| Problem scope | **Iris-scale CCE + BCE** (reuse Tier 3 geometry) | Small enough for CI; large enough to exercise the full pipeline |
| Step count | **Configurable, default 50** | Enough to surface accumulation drift; short enough for CI |

---

## Context

### The Single-Step Gap

Tier 3 currently validates cross-backend parity at two granularities (ADR-016):

1. **Per-kernel parity** (`test_parity_per_kernel.py`): Same kernel, same inputs, different backends → same outputs.
2. **End-to-end plan parity** (`test_parity_e2e.py`): Same Act/Learn plan, different backends → same final outputs from one plan execution.

Both are single-step comparisons.  They verify that a single forward pass or a single learning step produces matching results across backends.  They do **not** verify that repeated application of the Learn plan — the actual training loop — produces matching trajectories.

Multi-step divergence can arise from:

- **Reduction order non-determinism.**  GPU backends may reduce partial gradients in a different tree order than the CPU backend.  Per-step rounding differences are within tolerance individually, but they compound across optimizer state (Adam m1/m2 moments) over many steps.
- **Transcendental function approximation.**  GPU `exp()`, `log()`, `sqrt()` implementations may differ from CPU IEEE 754 results by 1–2 ULP.  These differences feed into Adam's bias correction and parameter updates, accumulating in state-precision moment buffers.
- **Fused multiply-add (FMA) availability.**  GPU backends may use FMA where the CPU does not (or vice versa), producing different intermediate rounding.  Over N steps, the moment vectors drift.

A system passing all current Tier 3 tests could still exhibit backends that converge at measurably different rates — or where one backend converges and another diverges — due to accumulated floating-point differences in optimizer state.

### The Oracle Chain

ADR-032 and ADR-033 establish the oracle validation chain:

```
Oracle D (NumPy, precision-aware)
    ↓ validates (D vs C at FP64: implementation correctness)
Oracle C (PyTorch, FP64)
    ↓ validates (C vs A/B: gradient and convergence correctness)
Oracles A/B (PyTorch, FP64)
    ↓ validates (Tier 2: per-kernel correctness)
CPU Backend
    ↓ validates (Tier 3: cross-backend parity)  ← THIS ADR
GPU Backends (OpenCL, Vulkan)
```

The CPU backend is the natural Tier 3 oracle (ADR-016).  Oracle D validates the CPU backend's precision-limited convergence behavior (ADR-033).  This ADR closes the final link: the CPU backend, validated by Oracle D, serves as the multi-step convergence reference for GPU backends.

### Relationship to `tests/convergence/`

ADR-028's convergence tests (`tests/convergence/`) ask: *"Does the engine converge on standard problems?"*  They run each backend independently and assert convergence criteria (accuracy thresholds, loss thresholds, epoch budgets).  Cross-backend trajectory comparison uses warn-not-fail semantics (`trajectory_comparison.py`) because the convergence tests exercise multi-epoch training with shuffled mini-batches, where per-batch divergence is expected and informative but not necessarily a bug.

This ADR's Tier 3 convergence parity tests ask a different question: *"Do backends agree on the same training trajectory?"*  They present **identical data in identical order** on every step — no shuffling, no mini-batching — and assert that the resulting loss curves match within cumulative tolerance.  This is a strict parity test, not a convergence quality test.

| Concern | `tests/convergence/` (ADR-028) | `tests/tier3/` (this ADR) |
|:--------|:-------------------------------|:--------------------------|
| Question | "Does each backend converge?" | "Do backends agree step-by-step?" |
| Data presentation | Shuffled mini-batches per epoch | Fixed batch, repeated N times |
| Failure semantics | Hard-fail on convergence criteria | Hard-fail on parity violation |
| Cross-backend comparison | Warn on trajectory divergence | Assert trajectory parity |
| Oracle D role | Precision-aware baseline (diagnostic) | None (CPU backend is the oracle) |

---

## Decision Drivers

1. **ADR-016 (Test Strategy).**  Tier 3's mandate is cross-backend parity.  Single-step parity is necessary but not sufficient — accumulated state divergence is a distinct failure mode that single-step tests cannot detect.

2. **ADR-028 (Convergence Testing).**  Convergence tests validate that each backend *succeeds* independently.  Tier 3 convergence parity validates that backends *agree* on the same trajectory.  These are complementary, not redundant.

3. **CONCEPT.md §1 (Architectural Elegance Feedback).**  The gap between single-step parity and multi-step convergence parity is a missing architectural primitive in the Tier 3 test surface.  This ADR formalizes it rather than relying on convergence tests' warn-level cross-backend comparison as an indirect proxy.

4. **ADR-033 (Precision Oracle Integration).**  Oracle D validates the CPU backend's precision behavior.  The CPU backend, thus validated, is trustworthy as the multi-step parity oracle.  This completes the validation chain from mathematical model to GPU backend.

5. **Debugging cost.**  When a user reports "OpenCL converges slower than CPU," the current test suite cannot reproduce the issue — convergence tests run backends independently, and Tier 3 tests are single-step.  A multi-step parity test catches this class of bug directly.

---

## Decisions

### 1. Test Structure

A new test module `tests/tier3/test_convergence_parity.py` implements multi-step cross-backend convergence parity tests.  The test:

1. Builds an Iris-scale model (reusing Tier 3's existing `_IRIS` geometry).
2. Generates deterministic input data and targets.
3. Initializes both backends with identical parameter state.
4. Runs N training steps, presenting the same batch on every step.
5. After each step, retrieves loss and parameter state from both backends.
6. Asserts that loss values match within cumulative tolerance.
7. After all steps, asserts that final parameter state matches within cumulative tolerance.

The test uses the existing Tier 3 oracle selection model (`conftest.py:_select_oracle`) — CPU as oracle when available, all-pairs fallback otherwise.

### 2. Tolerance Model

Single-step Tier 3 tolerances (`tolerance_config.py`) are insufficient for multi-step comparison.  Floating-point divergence accumulates across steps due to:

- Per-step rounding differences (1–2 ULP per transcendental operation)
- Moment EMA compounding (each step's rounding error persists in m1/m2)
- Parameter drift feeding back into the next step's forward pass

The tolerance model for step `t`:

$$\text{atol}(t) = \text{atol}_{\text{base}} \cdot (1 + \alpha \sqrt{t})$$

where:
- $\text{atol}_{\text{base}}$ is the single-step Tier 3 tolerance for `learn_plan_e2e` (currently `1e-4` for CPU)
- $\alpha$ is the growth coefficient (default `1.0`)
- $\sqrt{t}$ models sub-linear error accumulation (tighter than linear, validated empirically)

For loss comparison, the tolerance is:

$$\text{atol}_{\text{loss}}(t) = \text{atol}_{\text{base,loss}} \cdot (1 + \alpha \sqrt{t})$$

where $\text{atol}_{\text{base,loss}} = 10^{-4}$ — matching the softmax/loss kernel tolerance.

**Rationale for $\sqrt{t}$ growth:**  Each step contributes an independent rounding perturbation of magnitude $\sim \epsilon_{\text{mach}}$.  Under the assumption that per-step errors are uncorrelated (different gradient values, different optimizer states), the cumulative error after $t$ steps is $O(\sqrt{t} \cdot \epsilon_{\text{mach}})$ by the random walk model.  If empirical testing shows super-$\sqrt{t}$ growth (indicating correlated errors), the growth model should be revised — not the tolerance loosened.

### 3. Step Count and CI Budget

| Marker | Steps | Purpose |
|:-------|:------|:--------|
| `@pytest.mark.tier3` | 50 | Default: catches accumulation drift within CI budget |
| `@pytest.mark.slow` | 500 | Extended: catches slow-onset divergence (nightly CI) |

50 steps with Iris-scale geometry on CPU + one GPU backend completes in seconds.  500 steps adds ~10× runtime; suitable for nightly runs.

### 4. Data Presentation

The test presents the **same fixed batch** on every step.  This is deliberately different from `tests/convergence/` (shuffled mini-batches per epoch):

- **Fixed-batch repetition** isolates optimizer state accumulation as the sole divergence source.  Shuffled data introduces per-step input variance that masks small accumulation differences.
- **Identical data order** ensures that any loss curve divergence is attributable to backend implementation differences, not data presentation differences.
- **Overfitting is the goal.**  The test checks backend agreement on the trajectory, not convergence quality.  Both backends should overfit the fixed batch identically.

### 5. Parameter State Comparison

In addition to loss curve parity, the test compares parameter norms after every step.  This catches a failure mode where loss values coincidentally agree (flat region of the loss landscape) but underlying parameter state has diverged — a latent bug that would manifest on different data.

Final parameter state comparison uses element-wise `assert_allclose` with the cumulative tolerance at step N, providing the definitive parity verdict.

### 6. Scope Exclusions

- **Mixed-precision parity** is out of scope.  FP16/FP8 backends have wider per-step tolerances and faster accumulation growth, requiring a separate tolerance calibration.  Deferred to a future extension once FP32 parity is validated.
- **Multi-epoch with shuffled data** is out of scope.  That's ADR-028's domain (`tests/convergence/`).
- **Oracle D comparison** is out of scope.  Oracle D validates the CPU backend (ADR-033); it does not participate in Tier 3 backend-vs-backend comparison.

---

## Consequences

### Positive

- **Closes the validation chain.**  Oracle D → CPU backend → GPU backends.  Every link is tested: Oracle D validates CPU precision behavior (ADR-033); CPU validates GPU multi-step parity (this ADR).
- **Catches accumulation drift.**  The most likely cross-backend divergence mode under FP32 — slow moment drift from reduction order or transcendental approximation differences — is now directly tested.
- **Distinguishes parity from convergence.**  Tier 3 convergence parity (this ADR) and convergence quality (ADR-028) are separate test surfaces with separate failure semantics.  A backend can pass convergence quality tests while failing parity tests (converges correctly but via a different trajectory).
- **Reuses existing infrastructure.**  Oracle selection, backend pairs, renderer factory, and tolerance tables from `tests/tier3/conftest.py` and `tests/tolerance_config.py` are reused directly.

### Negative

- **CI runtime increase.**  50 training steps × 2 backends × 2 problem types = 200 plan executions per test run.  Manageable for Iris-scale, but scales poorly if the geometry grows.
- **Tolerance calibration required.**  The $\sqrt{t}$ growth model is theoretically motivated but needs empirical validation.  Initial runs may require $\alpha$ tuning per backend pair.
- **FP32-only initially.**  Mixed-precision parity requires a more sophisticated tolerance model.  This ADR explicitly defers it.

### Risks

- **Systematic reduction-order differences.**  If a GPU backend consistently reduces in a different order (e.g., reverse tree vs forward tree), per-step errors may be *correlated*, producing growth faster than $\sqrt{t}$.  The test would require $\alpha$ increase or a linear growth model.  Mitigation: start with $\alpha = 1.0$; if empirical runs show consistent failures at high step counts, diagnose the correlation structure before loosening.
- **Platform-dependent FMA behavior.**  CPU FMA availability varies by hardware and compiler flags.  A test passing on one CI machine may fail on another due to different FMA codegen.  Mitigation: the CPU backend's Meson build controls `-mfma`; Tier 3 tolerance already accounts for this (see `CPU_KERNEL_TOLERANCES`).

---

## Files Changed

| File | Change |
|:-----|:-------|
| `tests/tier3/test_convergence_parity.py` | **New.** Multi-step convergence parity tests (CCE + BCE, 50/500 steps) |
| `tests/tolerance_config.py` | Add `get_tier3_convergence_tolerance(step, precision_label)` with $\sqrt{t}$-scaled growth model |

---

## Related ADRs

- **ADR-016** — Establishes the Tier 3 parity framework that this ADR extends from single-step to multi-step.
- **ADR-028** — Convergence quality testing (complementary, not overlapping).
- **ADR-032** — Oracle hierarchy; CPU backend's trustworthiness derives from oracle validation.
- **ADR-033** — Oracle D validates the CPU backend; this ADR uses the validated CPU backend as the Tier 3 multi-step oracle.
