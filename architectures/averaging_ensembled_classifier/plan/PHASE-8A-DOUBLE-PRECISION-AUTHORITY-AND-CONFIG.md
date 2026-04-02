# Phase 8A: Double Precision — Authority Documents, PrecisionConfig Extension & Shared Infrastructure

**Status: COMPLETE** (implemented 2 April 2026)  
**Phase:** 8A of 8  
**Prerequisite:** Phase 7C complete and rollback gate passed.  
**Objective:** Amend the authority documents (CONCEPT.md, CONTRACT.md) with FP64 additions. Extend `PrecisionConfig` with four new factory classmethods and update the `__post_init__` invariants. Cascade the FP64 derived constants (`_IS_DOUBLE` flags, `compute_epsilon` for FP64) through the shared configuration, plan, and host layers. Extend `StabilizationPolicy` safety ceiling commentary for FP64. Update test fixtures. No kernel sources, no backend-native code, no build system changes. The phase ends when all existing tests pass against the new `PrecisionConfig` interface and the four new factories construct without error.  
**Governing ADR:** ADR-024 (§§1–2, §7, §8.1)  
**Rollback gate:** All tests that passed before Phase 8A must pass after. `PrecisionConfig.float32()`, `.float16()`, `.mixed_f16_f32()` remain identical. The new factories (`float64()`, `mixed_f32_f64_state()`, `mixed_f16_f64_state()`, `mixed_f32_f64()`) construct without error and satisfy the amended `__post_init__` invariants. Failing any existing Tier 2 test is a blocking regression.  
**Dependencies:** Phase 7C (mixed-precision CPU/Vulkan/alias removal) complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 8A.1: Amend CONCEPT.md](#step-8a1-amend-conceptmd)
   - [Step 8A.2: Amend CONTRACT.md](#step-8a2-amend-contractmd)
   - [Step 8A.3: Extend `PrecisionConfig`](#step-8a3-extend-precisionconfig)
   - [Step 8A.4: Update `StabilizationPolicy` commentary](#step-8a4-update-stabilizationpolicy-commentary)
   - [Step 8A.5: Update `HardwareProfile` SIMD width documentation](#step-8a5-update-hardwareprofile-simd-width-documentation)
   - [Step 8A.6: Update test fixtures](#step-8a6-update-test-fixtures)
   - [Step 8A.7: Validate rollback gate](#step-8a7-validate-rollback-gate)
4. [Migration Order Rationale](#4-migration-order-rationale)
5. [Risk Register](#5-risk-register)

---

## 1. Scope & Constraints

### In scope

- Amending `CONCEPT.md` with the FP64 state-stability rationale and the extended "Alchemist II" validation scenario (ADR-024 §8.2).
- Amending `CONTRACT.md` Article 6 with two new derived build-time symbols: `COMPUTE_TYPE_IS_DOUBLE` and `STATE_TYPE_IS_DOUBLE` (ADR-024 §2).
- Extending `PrecisionConfig` with `np.float64` support in all three role fields, plus four new factory classmethods: `float64()`, `mixed_f32_f64_state()`, `mixed_f16_f64_state()`, `mixed_f32_f64()` (ADR-024 §1).
- Amending the `__post_init__` invariants to permit `state_dtype > compute_dtype` (state is independent of compute) while retaining `storage_dtype.itemsize <= compute_dtype.itemsize` and `storage_dtype.itemsize <= state_dtype.itemsize` (ADR-024 §1.2).
- Updating derived constant values for FP64: `compute_epsilon = 1e-15` when `compute_dtype = np.float64` (ADR-024 §1.3).
- Updating `StabilizationPolicy` documentation to note that the FP64 safety ceiling ($\sim 1.8 \times 10^{308} / K_j$) makes overflow a non-practical concern (ADR-024 §7).
- Updating `HardwareProfile` documentation to note that `simd_width` is defined in elements of `COMPUTE_TYPE`, so FP64 compute halves the effective SIMD width on the same hardware (ADR-024 §6.1).
- Extending test fixtures in `conftest.py` and `test_mixed_precision.py` to cover the new factories.

### Out of scope

- Any change to `kernels/kernels.cl.h` or `kernels/phase_*.cl.c` — deferred to Phase 8B.
- Any change to `src/backends/opencl/type_mapping.py` — deferred to Phase 8B.
- Any change to CPU backend kernel sources (`cpu_kernels.h`, `cpu_kernels.c`, `cpu_precision.h`, `.inc` files) — deferred to Phase 8C.
- Any change to Vulkan backend shader sources (`common.glsl`, `.comp` files) — deferred to Phase 8D.
- SIMD implementation variants for FP64 (`simd_load_state_fp64`, etc.) — deferred to Phase 8C; scalar-only for initial FP64 support on CPU.
- Meson build changes for FP64 compilation flags — deferred to Phase 8B/8C/8D.
- `HardwareProfile` runtime querying of FP64 SIMD width — deferred to Phase 8C (the profile value is backend-populated at plan construction time; no shared-layer change needed).

### Key constraint: no existing behaviour changes

The four new factories are additions. No existing `PrecisionConfig` factory changes its return values. The `__post_init__` invariant is relaxed (removing a constraint that was never triggered by existing factories), but no existing factory violates the prior invariant, so existing construction paths are unaffected.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/shared/precision_config.py` | `PrecisionConfig` frozen dataclass: fields `storage_dtype`, `compute_dtype`, `state_dtype`, `storage_fp_format_max`, `compute_fp_format_max`, `compute_epsilon`. Three factories: `float32()`, `float16()`, `mixed_f16_f32()`. `__post_init__` enforces `storage_dtype.itemsize <= compute_dtype.itemsize` and `storage_dtype.itemsize <= state_dtype.itemsize`. |
| `src/shared/stabilization_policy.py` | `StabilizationPolicy` dataclass: field `compute_fp_format_max: float`. Used in Quadratic Scaling Policy safety ceiling. Docstring references FP32. |
| `src/shared/hardware_profile.py` | `HardwareProfile` frozen dataclass: `simd_width: int`. No documentation of its relationship to compute precision. |
| `src/shared/buffer_lifecycle.py` | `BufferDescriptor` with `precision_role` field. Unchanged by this phase. |
| `src/shared/kernel_contracts/phase_*.py` | Six files; `BufferParamSpec` objects have `precision_role`. Unchanged by this phase. |
| `src/tests/conftest.py` | `PRECISION_CONFIGS` list used for test parametrization. Contains `float32()`, `float16()`, `mixed_f16_f32()`. |
| `src/tests/test_mixed_precision.py` | "The Alchemist" test and multi-config parametrized tests. |
| `CONCEPT.md` | Post-Phase-7A state: three-role precision model documented, Alchemist scenario present. |
| `CONTRACT.md` | Post-Phase-7A state: Article 6 symbols include `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`. No `_IS_DOUBLE` flags. |

---

## 3. Task Breakdown

---

### Step 8A.1: Amend `CONCEPT.md`

**Governing authority:** ADR-024 Context §§1–3, §8.2  
**File:** `CONCEPT.md` (architecture root)

Three amendments. Insert in section order.

#### Amendment 8A.1.1 — Extend §2 (Primacy of Memory Strategy) with FP64 storage note

Append to the existing mixed-precision paragraph (added by Phase 7A, Amendment 2.1):

> **Storage-role FP64 is permitted but not optimised.** FP64 storage doubles FP32's bandwidth cost with no storage-compression benefit. The architecture permits storage-role FP64 for configurations where uniformity is preferred over bandwidth (e.g., `PrecisionConfig.float64()` for validation reference), but does not optimize for it. The expected production configurations place FP64 only in compute or state roles — e.g., `PrecisionConfig.mixed_f32_f64_state()` (FP32 storage, FP32 compute, FP64 state) for extended-stability training.

#### Amendment 8A.1.2 — Extend §11 (Host Orchestrator) with FP64 state rationale

Append to the existing `PrecisionConfig` description (amended by Phase 7A, Amendment 2.5):

> The state role supports FP64 precision for unbounded training stability. Adam's EMA update requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. For $\beta_1 = 0.999$, the gradient contributes only $0.001$ of its magnitude per step. FP32's 23-bit mantissa (approximately 7 decimal digits) allows precision erosion to accumulate over long training runs. FP64's 52-bit mantissa (approximately 16 decimal digits) provides an order-of-magnitude more headroom. The architecture's existing practice of computing Adam bias correction in FP64 on the host is recognized as a de facto instance of the state-role FP64 pattern; `PrecisionConfig.mixed_f32_f64_state()` formalizes it.

#### Amendment 8A.1.3 — Add Validation Scenario: The Alchemist II

Add to the *Validation Scenarios* section, after the existing Alchemist scenario:

> **Scenario: The Alchemist II (FP64 State Stability Validation)**
>
> - **Description:** A training task is executed for $10^6$ steps using three state precision configurations: `PrecisionConfig.float32()` (FP32 state), `PrecisionConfig.mixed_f32_f64_state()` (FP64 state), and a numpy reference implementation using FP64 throughout. All use identical hyperparameters ($\beta_1 = 0.999$, $\beta_2 = 0.9999$).
> - **Validation Focus:** Confirms that the FP64-state configuration's moment vectors track the FP64 reference within FP64 tolerance ($< 10^{-14}$ relative error), while the FP32-state configuration diverges measurably (relative error grows with step count). Validates that state precision is architecturally independent of compute precision.
> - **Key Insight:** Proves that the state role's FP64 extension achieves its stated goal: unbounded training stability without FP32 precision erosion in moment vectors.

**Verification:** After editing, confirm no dangling references to `fp64` variant (the retired single-axis name). All FP64 references should use three-role terminology: `state_dtype=float64`, `compute_dtype=float64`, or factory names.

---

### Step 8A.2: Amend `CONTRACT.md`

**Governing authority:** ADR-024 §2  
**File:** `CONTRACT.md` (architecture root)

Two amendments to Article 6 (Build-Time Symbols).

#### Amendment 8A.2.1 — Add `_IS_DOUBLE` derived symbols

In the Article 6 build-time symbol table, add two new rows after the `_IS_HALF` entries:

| Symbol | Type | Meaning |
|:---|:---|:---|
| `COMPUTE_TYPE_IS_DOUBLE` | `int` (0 or 1) | 1 when `COMPUTE_TYPE == double`; gates `cl_khr_fp64` extension and FP64 arithmetic paths |
| `STATE_TYPE_IS_DOUBLE` | `int` (0 or 1) | 1 when `STATE_TYPE == double`; gates FP64 load/store mechanics for state buffers |

#### Amendment 8A.2.2 — Add mutual-exclusivity note

Append to the symbol table or its notes section:

> The `_IS_HALF` and `_IS_DOUBLE` flags are mutually exclusive for the same role type. When `COMPUTE_TYPE = float`, both `COMPUTE_TYPE_IS_HALF = 0` and `COMPUTE_TYPE_IS_DOUBLE = 0`. A `_IS_HALF = 1` and `_IS_DOUBLE = 1` combination for the same role is a build-system error.

#### Amendment 8A.2.3 — Extend §5 PrecisionConfig contract

Update the `PrecisionConfig` paragraph to reference the new factories:

> Seven factory classmethods are defined: `float32()` (all FP32), `float16()` (all FP16), `mixed_f16_f32()` (FP16 storage/FP32 compute/FP32 state), `float64()` (all FP64), `mixed_f32_f64_state()` (FP32 storage/FP32 compute/FP64 state), `mixed_f16_f64_state()` (FP16 storage/FP32 compute/FP64 state), `mixed_f32_f64()` (FP32 storage/FP64 compute/FP64 state). The state role has no ordering constraint relative to compute — `state_dtype.itemsize` may be greater than, equal to, or (when storage is narrower than state) less than `compute_dtype.itemsize`.

---

### Step 8A.3: Extend `PrecisionConfig`

**Governing authority:** ADR-024 §1  
**File:** `src/shared/precision_config.py`

#### 8A.3.1: Verify `__post_init__` invariant compatibility

The current `__post_init__` enforces:
```python
assert self.storage_dtype.itemsize <= self.compute_dtype.itemsize
assert self.storage_dtype.itemsize <= self.state_dtype.itemsize
```

ADR-024 §1.2 retains both of these. There is **no constraint** between `compute_dtype` and `state_dtype` — they are independent roles. The new factory `mixed_f32_f64_state()` has `state_dtype (8 bytes) > compute_dtype (4 bytes)`, which is valid because the invariant only constrains storage relative to compute and state.

Confirm: the current code has no implicit constraint between compute and state. If any assertion or comparison between `compute_dtype.itemsize` and `state_dtype.itemsize` exists, it must be removed.

#### 8A.3.2: Add four factory classmethods

Append the following four factories after the existing `mixed_f16_f32()`:

```python
@classmethod
def float64(cls) -> "PrecisionConfig":
    return cls(
        storage_dtype=np.dtype(np.float64),
        compute_dtype=np.dtype(np.float64),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(np.finfo(np.float64).max),
        compute_fp_format_max=float(np.finfo(np.float64).max),
        compute_epsilon=1e-15,
    )

@classmethod
def mixed_f32_f64_state(cls) -> "PrecisionConfig":
    return cls(
        storage_dtype=np.dtype(np.float32),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(np.finfo(np.float32).max),
        compute_fp_format_max=float(np.finfo(np.float32).max),
        compute_epsilon=float(np.finfo(np.float32).eps),
    )

@classmethod
def mixed_f16_f64_state(cls) -> "PrecisionConfig":
    return cls(
        storage_dtype=np.dtype(np.float16),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(np.finfo(np.float16).max),
        compute_fp_format_max=float(np.finfo(np.float32).max),
        compute_epsilon=float(np.finfo(np.float32).eps),
    )

@classmethod
def mixed_f32_f64(cls) -> "PrecisionConfig":
    return cls(
        storage_dtype=np.dtype(np.float32),
        compute_dtype=np.dtype(np.float64),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(np.finfo(np.float32).max),
        compute_fp_format_max=float(np.finfo(np.float64).max),
        compute_epsilon=1e-15,
    )
```

**Note on `compute_epsilon`:** ADR-024 §1.3 specifies `compute_epsilon = 1e-15` for FP64 compute. This is a conservative stability guard, not `np.finfo(np.float64).eps` (which is `~2.2e-16`). The `1e-15` value mirrors the design pattern of `1e-7` for FP32 (also conservative relative to `np.finfo(np.float32).eps ≈ 1.19e-7`). For factories where `compute_dtype = np.float32`, `compute_epsilon` remains `float(np.finfo(np.float32).eps)` — the FP64 state dtype does not affect the compute epsilon.

#### 8A.3.3: Postcondition checks

- `PrecisionConfig.float64()` produces `compute_fp_format_max ≈ 1.7976931348623157e+308` and `compute_epsilon = 1e-15`.
- `PrecisionConfig.mixed_f32_f64_state()` passes `__post_init__` (storage=4, compute=4, state=8 → 4≤4 ✔, 4≤8 ✔).
- `PrecisionConfig.mixed_f16_f64_state()` passes (storage=2, compute=4, state=8 → 2≤4 ✔, 2≤8 ✔).
- `PrecisionConfig.mixed_f32_f64()` passes (storage=4, compute=8, state=8 → 4≤8 ✔, 4≤8 ✔).
- The hypothetical `PrecisionConfig(storage_dtype=np.float64, compute_dtype=np.float32, ...)` fails `__post_init__` (8 ≤ 4 → ✘).

---

### Step 8A.4: Update `StabilizationPolicy` commentary

**Governing authority:** ADR-024 §7  
**File:** `src/shared/stabilization_policy.py`

No structural changes to `StabilizationPolicy`. The field `compute_fp_format_max` already accepts any `float` value — FP64's `~1.8e+308` is valid.

Update the field docstring only:

```python
# WHY: This parameter represents a non-negotiable physical system boundary.
# The maximum representable value of the compute precision format. Governs
# overflow safety during reduction tree summation. Under the three-role model,
# the safety ceiling is bounded by arithmetic precision, not storage precision.
# At FP64 compute precision, the safety ceiling (~1.8e+308 / K_j) makes
# overflow a non-practical concern for any realistic fan-in K. The
# stabilization machinery remains active (it costs nothing when not
# triggered) but will never fire under FP64 compute.
compute_fp_format_max: float
```

---

### Step 8A.5: Update `HardwareProfile` SIMD width documentation

**Governing authority:** ADR-024 §6.1  
**File:** `src/shared/hardware_profile.py`

No structural changes to `HardwareProfile`. The `simd_width` field is an integer populated by backends at plan construction time.

Update the field docstring only:

```python
simd_width: int  # SIMD lane count in elements of COMPUTE_TYPE.
                 # When COMPUTE_TYPE = double, the effective SIMD width
                 # halves on the same hardware (e.g., AVX2: 8 for float,
                 # 4 for double). Backends report the width appropriate
                 # to the active PrecisionConfig.compute_dtype.
```

---

### Step 8A.6: Update test fixtures

**Governing authority:** ADR-024 §8.1  
**Files:** `src/tests/conftest.py`, `src/tests/test_mixed_precision.py`

#### 8A.6.1: Extend `PRECISION_CONFIGS` in `conftest.py`

Add the four new factories to the `PRECISION_CONFIGS` list:

```python
PRECISION_CONFIGS = [
    PrecisionConfig.float32(),
    PrecisionConfig.float16(),
    PrecisionConfig.mixed_f16_f32(),
    PrecisionConfig.float64(),
    PrecisionConfig.mixed_f32_f64_state(),
    PrecisionConfig.mixed_f16_f64_state(),
    PrecisionConfig.mixed_f32_f64(),
]
```

**Important:** Adding FP64 configs to `PRECISION_CONFIGS` may cause existing parametrized Tier 2 tests that exercise backend code (kernel compilation, FFI dispatch) to fail, because the backend layers do not yet support FP64. Two strategies:

**Strategy A (recommended):** Create a separate fixture list `FP64_PRECISION_CONFIGS` for the four new factories. Mark all tests using this fixture with `@pytest.mark.skip(reason="FP64 backend support not yet implemented (Phase 8B–8D)")`. The existing `PRECISION_CONFIGS` list is unchanged. Tests that only exercise shared-layer code (PrecisionConfig construction, StabilizationPolicy, ModelSpec) can use the full list.

**Strategy B:** Add all to `PRECISION_CONFIGS` and accept that backend-exercising tests will fail. This creates noise. Strategy A is preferred.

#### 8A.6.2: Add `PrecisionConfig` construction tests

Add a test function `test_fp64_precision_config_factories()` that:

1. Constructs each of the four new factories.
2. Asserts `__post_init__` does not raise.
3. Asserts derived constants match expected values:
   - `float64().compute_fp_format_max == float(np.finfo(np.float64).max)`
   - `float64().compute_epsilon == 1e-15`
   - `mixed_f32_f64_state().state_dtype == np.dtype(np.float64)`
   - `mixed_f32_f64_state().compute_dtype == np.dtype(np.float32)`
   - `mixed_f16_f64_state().storage_dtype == np.dtype(np.float16)`
   - `mixed_f32_f64().compute_fp_format_max == float(np.finfo(np.float64).max)`
4. Asserts that an invalid construction (`storage=float64, compute=float32`) raises `AssertionError`.

#### 8A.6.3: Add `StabilizationPolicy` FP64 test

Add a test that constructs a `StabilizationPolicy` with `compute_fp_format_max=float(np.finfo(np.float64).max)` and verifies the safety ceiling computation does not overflow or produce `inf` for realistic fan-in values (e.g., $K = 32, 64, 128$).

---

### Step 8A.7: Validate rollback gate

```bash
cd architectures/averaging_ensembled_classifier
python -m pytest src/tests/ -q 2>&1 | tee /tmp/phase8a_tests.txt
grep -E "passed|failed|error" /tmp/phase8a_tests.txt
```

All pre-existing tests must pass. New tests must pass. Zero regressions.

---

## 4. Migration Order Rationale

Phase 8A establishes the semantic foundation (authority documents, config types, test scaffolding) without touching any compilable backend code. This mirrors the Phase 7A pattern: the shared Python layer is self-contained and testable independently of the kernel sources. Phases 8B–8D then cascade FP64 through each backend's native code, build system, and SPIR-V compilation.

The authority document amendments (Steps 8A.1–8A.2) come first because subsequent phases' code changes are governed by, and must reference, the amended authority text. The `PrecisionConfig` extension (Step 8A.3) is the root change from which all backend work derives.

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| Adding FP64 factories to parametrized test fixtures causes cascading failures in backend-exercising tests | High | Use Strategy A (separate `FP64_PRECISION_CONFIGS` list, skip-marked backend tests). Shared-layer-only tests can use the full list safely. |
| `compute_epsilon = 1e-15` is too conservative or too aggressive for FP64 kernel stability guards | Low | The value is conservative (FP64 machine epsilon is `~2.2e-16`). Validate in Phase 8B–8C by running the Alchemist II scenario; adjust if numerical issues arise. |
| FP64 `compute_fp_format_max` value (`~1.8e+308`) causes floating-point issues in Python `StabilizationPolicy` arithmetic | Low | All policy computations are in Python `float` (which is FP64 natively). The value is exactly representable. Verify with the new StabilizationPolicy test. |
| The amended `__post_init__` invariant (no constraint between compute and state) is too permissive — allows nonsensical configs like `compute=fp16, state=fp64` | Low | This configuration is technically valid (narrow arithmetic, wide state). It's unusual but not harmful. If needed, add a warning (not assertion) for extreme width ratios. |
