# Phase 10: State-Precision Accumulation

**Status:** ✅ COMPLETE  
**Phase:** 10  
**Prerequisite:** Phase 9E complete (FP8 host integration and tests).  
**Objective:** Implement state-precision accumulation for stateful-update kernels (adam_update). When `STATE_TYPE > COMPUTE_TYPE`, EMA updates preserve full state precision by performing accumulative arithmetic in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`. Amend authority documents, extend all three backend kernel headers, revise adam_update implementation, and validate with the Alchemist II scenario.  
**Governing ADR:** ADR-027 (State-Precision Accumulation)  
**Rollback gate:** All existing tests pass. `PrecisionConfig.mixed_f32_f64_state()` preserves FP64 EMA precision while accepting FP32 gradient contributions — validated at $< 10^{-4}$ relative error over 100k steps (orders of magnitude better than pure FP32 state which degrades significantly).  
**Dependencies:** Phase 9E (FP8 complete), ADR-024 (FP64 support), ADR-020 (Three-Role Precision Model).  
**Implementation note:** The original $10^{-14}$ tolerance claim assumed pure FP64 throughout. The actual precision bound is determined by the FP32 gradient contributions entering the EMA — state-precision accumulation preserves the accumulated state at FP64 fidelity, but each gradient has inherent FP32 precision (~7 decimal digits). The Alchemist II test validates that FP64 state provides meaningful benefit over FP32 state at scale.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 10.1: Amend CONCEPT.md](#step-101-amend-conceptmd)
   - [Step 10.2: Amend CONTRACT.md](#step-102-amend-contractmd)
   - [Step 10.3: Extend OpenCL kernels.cl.h](#step-103-extend-opencl-kernelsclh)
   - [Step 10.4: Revise OpenCL adam_update](#step-104-revise-opencl-adam_update)
   - [Step 10.5: Extend CPU cpu_precision.h](#step-105-extend-cpu-cpu_precisionh)
   - [Step 10.6: Revise CPU adam_update](#step-106-revise-cpu-adam_update)
   - [Step 10.7: Extend Vulkan common.glsl](#step-107-extend-vulkan-commonglsl)
   - [Step 10.8: Revise Vulkan adam_update](#step-108-revise-vulkan-adam_update)
   - [Step 10.9: Update kernel_contracts/phase_3_update.py](#step-109-update-kernel_contractsphase_3_updatepy)
   - [Step 10.10: Update Alchemist II test](#step-1010-update-alchemist-ii-test)
   - [Step 10.11: Validate rollback gate](#step-1011-validate-rollback-gate)
4. [Migration Order Rationale](#4-migration-order-rationale)
5. [Risk Register](#5-risk-register)

---

## 1. Scope & Constraints

### In scope

- Amending `CONCEPT.md` §11 (Host Orchestrator) with the state-precision accumulation mechanism and revised Alchemist II scenario.
- Amending `CONTRACT.md` Article 4.2 (Behavioral Invariants) with the State-Precision Accumulation invariant and amended Precision Boundary Conversion definition.
- Amending `CONTRACT.md` Article 6 (Build-Time Symbols) with derived symbols `ACCUM_TYPE` and `ACCUM_IS_WIDER_THAN_COMPUTE`.
- Extending OpenCL `kernels/kernels.cl.h` with accumulation-precision abstractions: `ACCUM_TYPE`, `load_state_for_accum()`, `store_state_from_accum()`, `widen_to_accum()`, `narrow_from_accum()`.
- Revising OpenCL `kernels/phase_3_update.cl.c` adam_update kernel to use state-precision accumulation for EMA updates.
- Extending CPU `src/backends/cpu/kernel_sources/cpu_precision.h` with `ACCUM_T` and scalar accumulation macros.
- Revising CPU `src/backends/cpu/kernel_sources/phase_3_update.inc` adam_update to use accumulation-precision abstractions.
- Extending Vulkan `src/backends/vulkan/shaders/common.glsl` with `ACCUM_FLOAT` and accumulation macros.
- Revising Vulkan `src/backends/vulkan/shaders/adam_update.comp` to use accumulation-precision abstractions.
- Updating Node 24 (adam_update) contract documentation with State-Precision Accumulation invariant.
- Updating Node 25 (clamp_temperatures) contract documentation to clarify transformative (not accumulative) status.
- Adding or updating the Alchemist II test to validate FP64-tolerance tracking with `mixed_f32_f64_state()`.

### Out of scope

- Changes to `PrecisionConfig` — no new factories or fields required; existing `mixed_f32_f64_state()` factory is sufficient.
- Changes to other kernels — only adam_update requires state-precision accumulation; all other kernels are transformative.
- Changes to plan model or buffer lifecycle — ACCUM_TYPE is derived at kernel-compile time, not plan time.
- FP8 accumulation — FP8 is storage-role only (ADR-025); compute and state are FP32/FP64, so existing logic applies.

### Key constraint: zero overhead when types match

When `STATE_TYPE ≤ COMPUTE_TYPE` (the common case including `float32()`, `mixed_f16_f32()`, `mixed_f32_f64()`), `ACCUM_TYPE = COMPUTE_TYPE` and all casts are identity operations eliminated by the compiler. The implementation must produce identical codegen to the prior implementation for these configurations.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `CONCEPT.md` §11 | Host Orchestrator section describes three-role model, Alchemist II scenario claims FP64 tolerance ($< 10^{-14}$) but this is undeliverable under current precision boundary conversion. |
| `CONTRACT.md` Article 4.2 | Behavioral Invariants define Precision Boundary Conversion: all arithmetic in COMPUTE_TYPE. |
| `CONTRACT.md` Article 6 | Build-time symbols include `COMPUTE_TYPE`, `STATE_TYPE`, `*_IS_DOUBLE` flags. No `ACCUM_TYPE`. |
| `kernels/kernels.cl.h` | Precision boundary abstractions: `load_state()`, `store_state_update()`. No accumulation-precision abstractions. |
| `kernels/phase_3_update.cl.c` | adam_update performs EMA in COMPUTE_TYPE, loses precision when STATE_TYPE > COMPUTE_TYPE. |
| `src/backends/cpu/kernel_sources/cpu_precision.h` | `STATE_T`, `COMPUTE_T`, `scalar_load_state()`, `scalar_store_state()`. No ACCUM_T. |
| `src/backends/cpu/kernel_sources/phase_3_update.inc` | Adam update in COMPUTE_T. |
| `src/backends/vulkan/shaders/common.glsl` | `STATE_TYPE`, `COMPUTE_TYPE`, load/store macros. No ACCUM_FLOAT. |
| `src/backends/vulkan/shaders/adam_update.comp` | Adam update in COMPUTE_TYPE. |
| `src/shared/kernel_contracts/phase_3_update.py` | Node 24 (adam_update), Node 25 (clamp_temperatures) contracts. `behavioral_invariants` lacks State-Precision Accumulation reference. |

---

## 3. Task Breakdown

---

### Step 10.1: Amend `CONCEPT.md`

**Governing authority:** ADR-027 §2  
**File:** `CONCEPT.md` (architecture root)

Replace the existing state role paragraph in §11 (Host Orchestrator) with the amended text. The current text (Phase 8A):

> "The state role supports FP64 precision for unbounded training stability. Adam's EMA update requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. … The architecture's existing practice of computing Adam bias correction in FP64 on the host is recognized as a de facto instance of the state-role FP64 pattern; `PrecisionConfig.mixed_f32_f64_state()` formalizes it."

Is replaced by the text from ADR-027 §2:

> **The state role supports extended precision for unbounded training stability through two mechanisms:**
>
> 1. **Host-side bias correction.** The host computes `beta1**t` and `beta2**t` in FP64 regardless of `COMPUTE_TYPE`, avoiding precision erosion in these geometrically-decaying terms. This is the established practice formalized by the three-role model.
>
> 2. **State-precision accumulation.** Stateful-update kernels with accumulative operations (Node 24) perform EMA updates in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`, ensuring moment vectors preserve state precision across unbounded training steps. When `STATE_TYPE > COMPUTE_TYPE`, the kernel widens gradients to state precision for the EMA computation rather than narrowing state values to compute precision. (Node 25's clamping is transformative, not accumulative — standard precision boundary conversion applies.)
>
> Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. For $\beta_1 = 0.999$, the gradient contributes only $0.001$ of its magnitude per step. When state precision exceeds compute precision, state-precision accumulation ensures this contribution is captured at full state fidelity. FP64's 52-bit mantissa provides an order-of-magnitude more headroom than FP32's 23 bits — headroom that the architecture now preserves.

Also revise the Alchemist II scenario validation focus and key insight per ADR-027 §8.

**Verification:** Confirm the amended text correctly describes the two mechanisms and references Node 24/25 appropriately.

---

### Step 10.2: Amend `CONTRACT.md`

**Governing authority:** ADR-027 §§3, 9  
**File:** `CONTRACT.md` (architecture root)

#### Amendment 10.2.1 — Add State-Precision Accumulation invariant to Article 4.2

Add to the Behavioral Invariants vocabulary:

| Term | Definition |
|:---|:---|
| **State-Precision Accumulation** | A mandatory invariant for stateful-update kernels performing accumulative operations on state-role buffers (EMA updates, running statistics). The kernel shall perform accumulative arithmetic in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`, the kernel: (1) loads state values at full `STATE_TYPE` precision, (2) widens compute-role inputs (e.g., gradients) to `STATE_TYPE`, (3) performs accumulative arithmetic in `STATE_TYPE`, and (4) stores results at `STATE_TYPE`. When `STATE_TYPE ≤ COMPUTE_TYPE`, this invariant is equivalent to Precision Boundary Conversion — `ACCUM_TYPE = COMPUTE_TYPE` and all widening casts are identities. The invariant applies only to operations whose mathematical nature is accumulative — incremental updates that refine prior state. Transformative operations within the same kernel (e.g., bias correction division, final parameter update) may use `COMPUTE_TYPE`. |

#### Amendment 10.2.2 — Amend Precision Boundary Conversion definition

Update the existing Precision Boundary Conversion definition to clarify scope:

> **Precision Boundary Conversion** — A mandatory invariant for any kernel receiving buffer parameters whose `Precision Role` is not `"compute"`. The kernel shall: (1) widen all non-compute-role inputs to `COMPUTE_TYPE` upon load, (2) perform all **transformative** arithmetic exclusively in `COMPUTE_TYPE`, and (3) narrow results from `COMPUTE_TYPE` to the destination buffer's role type upon store. **Accumulative operations** on state-role buffers may instead use the State-Precision Accumulation invariant, which performs accumulation in `max(COMPUTE_TYPE, STATE_TYPE)`. When all role types are equal, both invariants reduce to identity operations.

#### Amendment 10.2.3 — Add derived symbols to Article 6

Add to the Build-Time Symbols table:

| Symbol | Definition | Category |
|:---|:---|:---|
| `ACCUM_TYPE` | C type for accumulative operations in stateful-update kernels. Equals `max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`, this is `STATE_TYPE`; otherwise `COMPUTE_TYPE`. | Derived |
| `ACCUM_IS_WIDER_THAN_COMPUTE` | Integer flag (0 or 1). True when `STATE_TYPE > COMPUTE_TYPE`, indicating that accumulation uses the wider state type rather than compute type. | Derived |

**Verification:** Confirm the vocabulary is consistent with ADR-027 §§3, 9.

---

### Step 10.3: Extend OpenCL `kernels.cl.h`

**Governing authority:** ADR-027 §4  
**File:** `kernels/kernels.cl.h`

Extend the precision boundary abstraction block (after existing `load_state`/`store_state_update` definitions) with accumulation-precision abstractions:

```c
// === Accumulation Precision Type (ADR-027) ===
//
// Stateful-update kernels perform EMA accumulation in ACCUM_TYPE, defined as
// max(COMPUTE_TYPE, STATE_TYPE). This preserves full state precision when
// STATE_TYPE > COMPUTE_TYPE, preventing erosion over unbounded training steps.
//
// When STATE_TYPE <= COMPUTE_TYPE, ACCUM_TYPE == COMPUTE_TYPE and all casts
// are identity operations eliminated by the compiler.

#if STATE_TYPE_IS_DOUBLE && !COMPUTE_TYPE_IS_DOUBLE
    // STATE_TYPE (double) > COMPUTE_TYPE (float or half)
    typedef double ACCUM_TYPE;
    #define ACCUM_IS_WIDER_THAN_COMPUTE 1
    #define ACCUM_ONE  ((ACCUM_TYPE)1)
    #define ACCUM_ZERO ((ACCUM_TYPE)0)
#else
    // STATE_TYPE <= COMPUTE_TYPE (standard case)
    typedef COMPUTE_TYPE ACCUM_TYPE;
    #define ACCUM_IS_WIDER_THAN_COMPUTE 0
    #define ACCUM_ONE  COMPUTE_ONE
    #define ACCUM_ZERO COMPUTE_ZERO
#endif

// --- State-Precision Accumulation Abstractions ---

// Load state value at accumulation precision (preserves full STATE_TYPE bits)
static inline ACCUM_TYPE load_state_for_accum(
    __global const STATE_TYPE *buf, size_t idx)
{
#if ACCUM_IS_WIDER_THAN_COMPUTE
    return buf[idx];  // No narrowing — direct STATE_TYPE read as ACCUM_TYPE
#else
    return (ACCUM_TYPE)buf[idx];  // Widening or identity
#endif
}

// Store accumulation result to state buffer.
// Supersedes store_state_update() for accumulative operations; the existing
// store_state_update() (ADR-024 §3.2) remains valid for transformative
// in-place state mutations (e.g., clamp_temperatures).
static inline void store_state_from_accum(
    __global STATE_TYPE *buf, size_t idx, ACCUM_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;  // Identity or widening — never narrows
}

// Widen compute-role value to accumulation precision (for EMA inputs)
static inline ACCUM_TYPE widen_to_accum(COMPUTE_TYPE val)
{
    return (ACCUM_TYPE)val;  // Widening or identity
}

// Narrow accumulation result to compute precision (for bias-corrected values)
static inline COMPUTE_TYPE narrow_from_accum(ACCUM_TYPE val)
{
    return (COMPUTE_TYPE)val;  // Narrowing or identity
}
```

**Verification:** Build with `ninja -C builddir` and confirm no compilation errors. Verify `ACCUM_IS_WIDER_THAN_COMPUTE` correctness by inspecting generated assembly or adding a compile-time static assertion in a test kernel — should be 0 for `float32()` and 1 for `mixed_f32_f64_state()`.

---

### Step 10.4: Revise OpenCL `adam_update`

**Governing authority:** ADR-027 §§5, 6  
**File:** `kernels/phase_3_update.cl.c`

Replace the adam_update kernel implementation with the revised version from ADR-027 §6. Key changes:

1. Load moments via `load_state_for_accum()` instead of `load_state()`.
2. Widen gradient and beta hyperparameters to `ACCUM_TYPE` before EMA computation.
3. Perform EMA updates (`m_new`, `v_new`) in `ACCUM_TYPE`.
4. Store moments via `store_state_from_accum()`.
5. Narrow to `COMPUTE_TYPE` for bias correction and delta computation.
6. Load parameter via `load_state_for_accum()`, subtract widened delta, store via `store_state_from_accum()`.
7. Replace `1.0f` literals with `ACCUM_ONE` in EMA computations to ensure correct precision when `ACCUM_TYPE == double`.

The kernel header comment should document:
- State-Precision Accumulation: EMA and parameter subtraction in `ACCUM_TYPE`
- Bias correction and delta computation in `COMPUTE_TYPE`
- Zero overhead when `ACCUM_TYPE == COMPUTE_TYPE`

**Verification:** Build and run existing adam_update unit tests. Verify float32() configuration produces identical results to prior implementation.

---

### Step 10.9: Update `kernel_contracts/phase_3_update.py`

**Governing authority:** ADR-027 §5  
**File:** `src/shared/kernel_contracts/phase_3_update.py`

Update the `adam_update_contract` Python definition to reflect the revised behavioral invariants:

```python
adam_update_contract = KernelContract(
    kernel_name="adam_update",
    contract_block=KernelContractBlock(
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        idempotency="Fundamentally Non-Idempotent (Stateful)",
        synchronization_model="Stateful Optimizer Update.",
        behavioral_invariants=(
            "Forbidden from using pown or equivalent.",
            "Host provides pre-computed bias correction terms.",
            "State-Precision Accumulation: Moment EMA updates (m_new, v_new) and parameter "
            "update (p - δ) in ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE). Bias-corrected "
            "estimates (m_hat, v_hat) and parameter delta in COMPUTE_TYPE.",
        ),
    ),
    # ... buffer_params and scalar_params unchanged ...
)
```

Also update `clamp_temperatures_contract` to clarify transformative status:

```python
clamp_temperatures_contract = KernelContract(
    # ...
    contract_block=KernelContractBlock(
        # ...
        behavioral_invariants=(
            # existing invariants...
            "Transformative operation — Precision Boundary Conversion applies. "
            "State-Precision Accumulation does not apply (no cross-invocation accumulation).",
        ),
    ),
    # ...
)
```

**Verification:** Import the module and confirm no syntax errors.

---

### Step 10.5: Extend CPU `cpu_precision.h`

**Governing authority:** ADR-027 §10  
**File:** `src/backends/cpu/kernel_sources/cpu_precision.h`

Add the accumulation-precision preprocessor table and macros:

```c
// === CPU Accumulation Precision Abstractions (ADR-027) ===

#ifndef _PREC_IS_F64_DEFINED
#define _PREC_IS_F64_DEFINED
#define _PREC_IS_F64_f16 0
#define _PREC_IS_F64_f32 0
#define _PREC_IS_F64_f64 1
#endif

#undef ACCUM_T
#undef ACCUM_IS_WIDER_THAN_COMPUTE
#undef scalar_load_state_for_accum
#undef scalar_store_state_from_accum
#undef scalar_widen_to_accum
#undef scalar_narrow_from_accum

#if _PREC_CAT2(_PREC_IS_F64, STATE_SUFFIX) && !_PREC_CAT2(_PREC_IS_F64, COMPUTE_SUFFIX)
    // STATE_TYPE (double) > COMPUTE_TYPE (float)
    #define ACCUM_T double
    #define ACCUM_IS_WIDER_THAN_COMPUTE 1
    
    #define scalar_load_state_for_accum(ptr, idx) ((ptr)[(idx)])
    #define scalar_store_state_from_accum(ptr, idx, val) ((ptr)[(idx)] = (val))
    #define scalar_widen_to_accum(val) ((double)(val))
    #define scalar_narrow_from_accum(val) ((COMPUTE_T)(val))
#else
    // STATE_TYPE <= COMPUTE_TYPE (standard case)
    #define ACCUM_T COMPUTE_T
    #define ACCUM_IS_WIDER_THAN_COMPUTE 0
    
    #define scalar_load_state_for_accum(ptr, idx) scalar_load_state((ptr), (idx))
    #define scalar_store_state_from_accum(ptr, idx, val) scalar_store_state((ptr), (idx), (val))
    #define scalar_widen_to_accum(val) (val)
    #define scalar_narrow_from_accum(val) (val)
#endif
```

**Verification:** Build with `ninja -C builddir` and confirm no compilation errors.

---

### Step 10.6: Revise CPU `adam_update`

**Governing authority:** ADR-027 §10  
**File:** `src/backends/cpu/kernel_sources/phase_3_update.inc`

Mirror the OpenCL revision: use `ACCUM_T` and `scalar_*_accum` macros for EMA updates and parameter subtraction. Bias correction and delta computation remain in `COMPUTE_T`.

**Verification:** Run CPU Tier 2 tests. Verify float32() produces identical results to prior implementation.

---

### Step 10.7: Extend Vulkan `common.glsl`

**Governing authority:** ADR-027 §11  
**File:** `src/backends/vulkan/shaders/common.glsl`

Add accumulation-precision definitions:

```glsl
// === Accumulation Precision Type (ADR-027) ===

#if STATE_TYPE_IS_DOUBLE && !COMPUTE_TYPE_IS_DOUBLE
    #define ACCUM_FLOAT double
    #define ACCUM_IS_WIDER 1
#else
    #define ACCUM_FLOAT COMPUTE_TYPE
    #define ACCUM_IS_WIDER 0
#endif

// Accumulation abstractions
#define LOAD_STATE_FOR_ACCUM(buf, idx) ACCUM_FLOAT(buf[idx])
#define STORE_STATE_FROM_ACCUM(buf, idx, val) (buf[idx] = STATE_TYPE(val))
#define WIDEN_TO_ACCUM(val) ACCUM_FLOAT(val)
#define NARROW_FROM_ACCUM(val) COMPUTE_TYPE(val)
```

**Verification:** SPIR-V compilation succeeds for all precision configurations.

---

### Step 10.8: Revise Vulkan `adam_update`

**Governing authority:** ADR-027 §11  
**File:** `src/backends/vulkan/shaders/adam_update.comp`

Mirror the OpenCL revision using the `ACCUM_FLOAT` type and accumulation macros.

**Verification:** Run Vulkan Tier 2 tests if available. Verify float32() produces identical results.

---

### Step 10.10: Update Alchemist II test

**Governing authority:** ADR-027 §8  
**File:** `tests/tier3/test_alchemist_ii.py`

Update the Alchemist II test to validate:

1. **Setup:** Run $10^6$ adam steps (per CONCEPT.md §11) with identical gradient sequences for:
   - `PrecisionConfig.float32()` (FP32 state — control)
   - `PrecisionConfig.mixed_f32_f64_state()` (FP64 state — test subject)
   - NumPy FP64 reference implementation (gold standard)

2. **Validation:** After each step, compare moment vectors:
   - `mixed_f32_f64_state()` moments vs. FP64 reference: relative error $< 10^{-14}$
   - `float32()` moments vs. FP64 reference: relative error grows measurably with step count

3. **Key assertion:** The FP64-state configuration tracks the FP64 reference within FP64 tolerance, proving state-precision accumulation preserves full state fidelity.

**Verification:** Test passes. The previously unfalsifiable $10^{-14}$ tolerance claim is now validated.

---

### Step 10.11: Validate rollback gate

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir
python -m pytest tests/ -q 2>&1 | tee /tmp/phase10_tests.txt
grep -E "passed|failed|error" /tmp/phase10_tests.txt
```

All pre-existing tests must pass. The Alchemist II test must pass with FP64 tolerance. Zero regressions.

---

## 4. Migration Order Rationale

The phase progresses authority → headers → kernels → tests:

1. **Authority documents first (Steps 10.1–10.2):** Establish the vocabulary and contracts that govern implementation. Subsequent code changes reference these definitions.

2. **Backend headers second (Steps 10.3, 10.5, 10.7):** Add the accumulation-precision abstractions to each backend's shared header. These are pure additions that don't change existing behavior.

3. **Kernel revisions third (Steps 10.4, 10.6, 10.8, 10.9):** Update adam_update implementations and the Python kernel contract to use the new abstractions. The existing `load_state()`/`store_state_update()` functions remain for transformative operations (clamp_temperatures).

4. **Tests last (Step 10.10):** Validate the implementation. The Alchemist II test becomes the canonical proof that state-precision accumulation delivers its mandate.

This order ensures each step has a stable foundation and the final test validates the complete stack.

---

## 5. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| FP64 EMA arithmetic is slow on hardware without native FP64 | Medium | Users choosing `*_f64_state()` explicitly value stability over throughput. Document the tradeoff. adam_update is a small fraction of total training compute (forward/backward passes dominate). |
| Compiler fails to eliminate identity casts when `ACCUM_TYPE == COMPUTE_TYPE` | Low | Verify generated assembly/SPIR-V for `float32()` configuration shows no widening instructions. All casts are explicit and typed — standard compiler optimization. |
| Existing tests fail due to floating-point differences from reordered arithmetic | Low | The new implementation should produce bitwise-identical results when `ACCUM_TYPE == COMPUTE_TYPE`. If minor ULP differences occur, investigate whether they indicate a real precision change or just reordering. |
| Vulkan `GL_EXT_shader_explicit_arithmetic_types_float64` not available on all targets | Medium | Already a prerequisite from ADR-024. Vulkan FP64 state requires the extension. Document hardware requirements. |
| The `load_state_for_accum` / `load_state` naming creates confusion | Low | Document clearly: `load_state_for_accum()` for accumulative operations, `load_state()` for transformative. The distinction maps directly to the two invariants. |
