# ADR-027: State-Precision Accumulation

**Status:** PROPOSED  
**Date:** 2026-04-05  
**Deciders:** —  
**Triggered by:** Discovered contradiction between the state role's stability mandate (CONCEPT §11, ADR-024) and the Precision Boundary Conversion invariant (ADR-020 §3.5) in accumulative operations  
**Depends on:** ADR-020 (Three-Role Precision Model), ADR-024 (FP64 Support)  
**Amends:** ADR-020 §3.5 (Behavioral Invariants Vocabulary), ADR-024 §8.2 (Alchemist II scenario)  
**Constrains:** Node 24 (`adam_update`) and all future accumulative stateful-update kernels  
**Classifies:** Node 25 (`clamp_temperatures`) as transformative (unchanged)

---

## Context

### The State Role's Mandate

ADR-020 introduced the state role with a clear mandate:

> "**State precision is a stability concern.** Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. … FP32's 23-bit mantissa allows precision erosion to accumulate over unbounded training runs. FP64's 52-bit mantissa provides an order-of-magnitude more headroom."

This mandate led ADR-024 to introduce `PrecisionConfig.mixed_f32_f64_state()` — FP32 storage, FP32 compute, FP64 state — with the Alchemist II scenario claiming:

> "Confirms that the FP64-state configuration's moment vectors track the FP64 reference within FP64 tolerance ($< 10^{-14}$ relative error)."

### The Precision Boundary Invariant

ADR-020 §3.5 established the Precision Boundary Conversion invariant:

> "The kernel shall: (1) widen all non-compute-role inputs to `COMPUTE_TYPE` upon load, (2) **perform all arithmetic exclusively in `COMPUTE_TYPE`**, and (3) narrow results from `COMPUTE_TYPE` to the destination buffer's role type upon store."

### The Contradiction

When `STATE_TYPE > COMPUTE_TYPE` (e.g., `mixed_f32_f64_state()`), the adam_update kernel, faithfully implementing the invariant, performs:

```c
// 1. Load: FP64 → FP32 (45 mantissa bits destroyed)
COMPUTE_TYPE m_prev = load_state(m1_buf, i);

// 2. Arithmetic: FP32 (~7 decimal digits)
COMPUTE_TYPE m_new = β₁ * m_prev + (1 - β₁) * g;

// 3. Store: FP32 → FP64 (no information recovered)
store_state_update(m1_buf, i, m_new);
```

The FP64 state buffer becomes a wider container holding values with FP32 precision. The Alchemist II scenario's $10^{-14}$ tolerance claim is unfalsifiable — the architecture cannot deliver it under the current contract.

### The Architectural Pattern

This is not a bug in the kernel implementation. The kernel correctly implements its contract. The issue is that **the state role's mandate and the precision boundary invariant express different optimization targets that conflict for accumulative operations:**

| Concern | Precision Boundary Conversion | State Role Mandate |
|:---|:---|:---|
| Optimization target | **Throughput** — narrower compute = faster | **Stability** — wider state = less erosion |
| Governing principle | Primacy of Memory (bandwidth) | Unbounded training stability |
| Applicable to | Transformative operations | Accumulative operations |

The three-role model correctly separates *storage* (bandwidth) from *compute* (fidelity). But it conflates *transformative compute* (bounded by input precision) with *accumulative compute* (erosion compounds across steps). The state role was intended to solve the latter, but the precision boundary invariant forces all arithmetic through the former.

Per CONCEPT.md §1 (Architectural Elegance Feedback):

> 1. **Suspend** — The Alchemist II scenario's FP64-tolerance claim is suspended.
> 2. **Formalize** — Establish accumulation precision as a behavioral distinction within stateful-update kernels.
> 3. **Reify** — Amend the precision boundary invariant with a state-preserving accumulation path.

---

## Decision Drivers

1. **The state role's raison d'être is precision erosion prevention.** A state role that doesn't prevent erosion under configurations designed for erosion prevention (`*_f64_state()`) is semantically hollow. If FP64 state with FP32 compute degrades to FP32 precision, the factories should not exist — users should use `mixed_f32_f64()` (FP64 compute).

2. **Transformative vs. accumulative operations have different precision semantics.** A Softmax reading FP64-state weights into FP32 compute loses no information the FP32 arithmetic couldn't represent anyway — the output precision is bounded by compute. But an EMA update reading FP64 state, computing in FP32, and writing back loses information *that was present* and *could have been preserved*.

3. **The pattern is local to stateful-update kernels.** Only kernels with `Idempotency: "Fundamentally Non-Idempotent (Stateful)"` require the accumulation-precision path. Transformative kernels (the majority) are unchanged.

4. **ADR-026 established the variant-rather-than-branch pattern.** Rather than conditionals in existing code, introduce a behavioral distinction at the contract level. The kernel source uses different abstractions; the compiler selects the appropriate implementation.

5. **Generalization across precision configurations.** The pattern must work uniformly for all valid `(storage, compute, state)` combinations:
   - FP16/FP32/FP64 storage configurations (standard hierarchy)
   - FP8 storage configurations (ADR-025) where compute and state are always FP32 or FP64
   - Future precision formats

---

## Decision

### §1: Foundational Distinction — Accumulation vs. Transformation

A new distinction is established in the precision model's vocabulary:

| Operation Class | Definition | Precision Implication |
|:---|:---|:---|
| **Transformative** | Reads input, computes output, writes result. Each invocation is independent. | Arithmetic precision bounded by `COMPUTE_TYPE`. State inputs narrow on load; state outputs widen on store. No cross-invocation precision concern. |
| **Accumulative** | Reads prior state, applies incremental update, writes new state. Each invocation refines the prior value. | Arithmetic precision matches `max(COMPUTE_TYPE, STATE_TYPE)`, preserving full state precision across unbounded invocations. |

The existing Precision Boundary Conversion invariant governs transformative operations. A new **State-Precision Accumulation** invariant governs accumulative operations.

### §1.1: Accumulation Precision Type

The accumulation precision is defined as:

$$\text{ACCUM\_TYPE} = \max(\text{COMPUTE\_TYPE}, \text{STATE\_TYPE})$$

When `STATE_TYPE > COMPUTE_TYPE`:
- `ACCUM_TYPE = STATE_TYPE` (wider type preserves precision)
- EMA arithmetic uses the state type's full mantissa
- Gradients widen to state precision before accumulation

When `STATE_TYPE ≤ COMPUTE_TYPE`:
- `ACCUM_TYPE = COMPUTE_TYPE` (standard behavior)
- No widening required; accumulation matches existing semantics
- All casts are identities, eliminated by the compiler

### §1.2: Applicability Across Configurations

| Configuration | Storage | Compute | State | ACCUM_TYPE | Behavior |
|:---|:---|:---|:---|:---|:---|
| `float32()` | FP32 | FP32 | FP32 | FP32 | Identity — unchanged |
| `mixed_f16_f32()` | FP16 | FP32 | FP32 | FP32 | Identity — unchanged |
| `mixed_f32_f64_state()` | FP32 | FP32 | FP64 | **FP64** | EMA in FP64 |
| `mixed_f16_f64_state()` | FP16 | FP32 | FP64 | **FP64** | EMA in FP64 |
| `mixed_f32_f64()` | FP32 | FP64 | FP64 | FP64 | Identity — unchanged |
| `float64()` | FP64 | FP64 | FP64 | FP64 | Identity — unchanged |
| `fp8_e4m3()` | FP8 | FP32 | FP32 | FP32 | Identity — unchanged |
| `fp8_e4m3_f64()` | FP8 | FP64 | FP64 | FP64 | Identity — unchanged |
| FP8 + FP32c + FP64x | FP8 | FP32 | FP64 | **FP64** | EMA in FP64 |

The pattern activates only when `state_dtype.itemsize > compute_dtype.itemsize`. For all other configurations — including the common cases — the behavior is identical to the prior implementation with zero overhead.

---

### §2: CONCEPT.md Amendment — §11 Host Orchestrator

The paragraph describing the state role (currently at line ~305) is amended. The current text:

> "The state role supports FP64 precision for unbounded training stability. Adam's EMA update requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. … The architecture's existing practice of computing Adam bias correction in FP64 on the host is recognized as a de facto instance of the state-role FP64 pattern; `PrecisionConfig.mixed_f32_f64_state()` formalizes it."

Is replaced by:

> **The state role supports extended precision for unbounded training stability through two mechanisms:**
>
> 1. **Host-side bias correction.** The host computes `beta1**t` and `beta2**t` in FP64 regardless of `COMPUTE_TYPE`, avoiding precision erosion in these geometrically-decaying terms. This is the established practice formalized by the three-role model.
>
> 2. **State-precision accumulation.** Stateful-update kernels with accumulative operations (Node 24) perform EMA updates in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`, ensuring moment vectors preserve state precision across unbounded training steps. When `STATE_TYPE > COMPUTE_TYPE`, the kernel widens gradients to state precision for the EMA computation rather than narrowing state values to compute precision. (Node 25's clamping is transformative, not accumulative — standard precision boundary conversion applies.)
>
> Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. For $\beta_1 = 0.999$, the gradient contributes only $0.001$ of its magnitude per step. When state precision exceeds compute precision, state-precision accumulation ensures this contribution is captured at full state fidelity. FP64's 52-bit mantissa provides an order-of-magnitude more headroom than FP32's 23 bits — headroom that the architecture now preserves.

---

### §3: CONTRACT.md Amendment — Article 4.2 (Behavioral Invariants)

A new invariant is added to the recognized vocabulary:

| Term | Definition |
|:---|:---|
| **State-Precision Accumulation** | A mandatory invariant for stateful-update kernels performing accumulative operations on state-role buffers (EMA updates, running statistics). The kernel shall perform accumulative arithmetic in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`, the kernel: (1) loads state values at full `STATE_TYPE` precision, (2) widens compute-role inputs (e.g., gradients) to `STATE_TYPE`, (3) performs accumulative arithmetic in `STATE_TYPE`, and (4) stores results at `STATE_TYPE`. When `STATE_TYPE ≤ COMPUTE_TYPE`, this invariant is equivalent to Precision Boundary Conversion — `ACCUM_TYPE = COMPUTE_TYPE` and all widening casts are identities. The invariant applies only to operations whose mathematical nature is accumulative — incremental updates that refine prior state. Transformative operations within the same kernel (e.g., bias correction division, final parameter update) may use `COMPUTE_TYPE`. |

The Precision Boundary Conversion invariant definition is amended to clarify scope:

> **Precision Boundary Conversion** — A mandatory invariant for any kernel receiving buffer parameters whose `Precision Role` is not `"compute"`. The kernel shall: (1) widen all non-compute-role inputs to `COMPUTE_TYPE` upon load, (2) perform all **transformative** arithmetic exclusively in `COMPUTE_TYPE`, and (3) narrow results from `COMPUTE_TYPE` to the destination buffer's role type upon store. **Accumulative operations** on state-role buffers may instead use the State-Precision Accumulation invariant, which performs accumulation in `max(COMPUTE_TYPE, STATE_TYPE)`. When all role types are equal, both invariants reduce to identity operations.

---

### §4: kernels.cl.h Amendment — Accumulation Precision Abstractions

The precision boundary abstraction block (currently lines 140–180) is extended:

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
    return (ACCUM_TYPE)buf[idx];  // Widening or identity — STATE_TYPE ≤ COMPUTE_TYPE here
#endif
}

// Store accumulation result to state buffer.
// Supersedes store_state_update() for accumulative operations; the existing
// store_state_update() (ADR-024 §3.2) remains valid for transformative
// in-place state mutations (e.g., clamp_temperatures).
static inline void store_state_from_accum(
    __global STATE_TYPE *buf, size_t idx, ACCUM_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;  // Identity or widening — never narrows (requires STATE_TYPE ≥ COMPUTE_TYPE)
}

// Widen compute-role value to accumulation precision (for EMA inputs)
static inline ACCUM_TYPE widen_to_accum(COMPUTE_TYPE val)
{
    return (ACCUM_TYPE)val;  // Widening or identity — never narrows
}

// Narrow accumulation result to compute precision (for bias-corrected values)
static inline COMPUTE_TYPE narrow_from_accum(ACCUM_TYPE val)
{
    return (COMPUTE_TYPE)val;  // Narrowing or identity
}
```

**Rationale for abstraction design:**

1. `load_state_for_accum`: When `ACCUM_IS_WIDER_THAN_COMPUTE`, reads the state value at its native precision (no narrowing). The existing `load_state()` narrows to `COMPUTE_TYPE` — correct for transformative operations, incorrect for accumulative.

2. `store_state_from_accum`: Stores the accumulated value. When `ACCUM_TYPE == STATE_TYPE`, this is identity. When `ACCUM_TYPE == COMPUTE_TYPE < STATE_TYPE`, this widens (no precision loss).

3. `widen_to_accum`: Promotes compute-role values (gradients) to accumulation precision for EMA arithmetic.

4. `narrow_from_accum`: Converts accumulated values back to compute precision for transformative operations (bias correction, parameter update delta).

---

### §5: Node 24 Contract Amendment — adam_update

The kernel contract's Behavioral Invariants are revised.

**Current:**
> "EMA arithmetic exclusively in COMPUTE_TYPE."

**Revised:**
> "State-Precision Accumulation: Moment EMA updates (`m_new`, `v_new`) and parameter update (`p - δ`) performed exclusively in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`. Bias-corrected estimates (`m_hat`, `v_hat`) and parameter update delta computed in `COMPUTE_TYPE`. Precision Boundary Conversion: gradient consumed in `COMPUTE_TYPE`, widened to `ACCUM_TYPE` for EMA; parameter delta widened from `COMPUTE_TYPE` to `ACCUM_TYPE` for accumulative subtraction."

---

### §6: adam_update Implementation Revision

The implementation in `phase_3_update.cl.c` is revised:

```c
// --- Implementation: adam_update (Node 24) ---
// Strategy: A stateful, embarrassingly parallel "map" kernel. Each work-item
// updates a single parameter and its corresponding moment vectors.
//
// Key behavioral contract:
// - State-Precision Accumulation: EMA updates and parameter subtraction in
//   ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)
// - Bias correction and parameter delta computation in COMPUTE_TYPE
// - Host provides pre-computed beta powers for numerical stability
//
// When ACCUM_TYPE == COMPUTE_TYPE (the common case), all widen/narrow casts
// are identity operations eliminated by the compiler — zero overhead.

__kernel void adam_update(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_final_grad,
    __global STATE_TYPE         *update_buffer_GLOBAL_parameters,
    __global STATE_TYPE         *update_buffer_GLOBAL_m1,
    __global STATE_TYPE         *update_buffer_GLOBAL_m2,
    COMPUTE_TYPE                 src_scalar_REAL_learning_rate,
    COMPUTE_TYPE                 src_scalar_REAL_beta1_pow_t,
    COMPUTE_TYPE                 src_scalar_REAL_beta2_pow_t,
    COMPUTE_TYPE                 src_scalar_REAL_beta1,
    COMPUTE_TYPE                 src_scalar_REAL_beta2,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_parameter_count)
{
    // --- 1. Work-Item to Parameter Mapping ---
    const uint i = get_global_id(0);
    if (i >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // --- 2. Load Inputs ---
    // Gradient in COMPUTE_TYPE (precision role: compute)
    const COMPUTE_TYPE g = src_buffer_GLOBAL_final_grad[i];
    
    // State-Precision Accumulation: load moments at full state precision
    // When ACCUM_TYPE > COMPUTE_TYPE, preserves FP64 fidelity
    const ACCUM_TYPE m_prev = load_state_for_accum(update_buffer_GLOBAL_m1, i);
    const ACCUM_TYPE v_prev = load_state_for_accum(update_buffer_GLOBAL_m2, i);

    // --- 3. EMA Updates in ACCUM_TYPE (preserves state precision) ---
    // Widen gradient and hyperparameters to accumulation precision
    const ACCUM_TYPE g_accum     = widen_to_accum(g);
    const ACCUM_TYPE beta1_accum = widen_to_accum(src_scalar_REAL_beta1);
    const ACCUM_TYPE beta2_accum = widen_to_accum(src_scalar_REAL_beta2);
    
    // First moment: m_new = β₁ · m_prev + (1 - β₁) · g
    const ACCUM_TYPE m_new = beta1_accum * m_prev + 
                             (ACCUM_ONE - beta1_accum) * g_accum;
    
    // Second moment: v_new = β₂ · v_prev + (1 - β₂) · g²
    const ACCUM_TYPE v_new = beta2_accum * v_prev + 
                             (ACCUM_ONE - beta2_accum) * (g_accum * g_accum);

    // Store updated moments at state precision
    store_state_from_accum(update_buffer_GLOBAL_m1, i, m_new);
    store_state_from_accum(update_buffer_GLOBAL_m2, i, v_new);

    // --- 4. Bias Correction in COMPUTE_TYPE ---
    // Transformative operations — bounded by compute precision, not state
    const COMPUTE_TYPE m_hat = narrow_from_accum(m_new) / 
                               (COMPUTE_ONE - src_scalar_REAL_beta1_pow_t);
    const COMPUTE_TYPE v_hat = narrow_from_accum(v_new) / 
                               (COMPUTE_ONE - src_scalar_REAL_beta2_pow_t);

    // --- 5. Parameter Update in ACCUM_TYPE (accumulative) ---
    // The delta is transformative (computed fresh each step), but the subtraction
    // p -= delta is accumulative — p refines over unbounded training steps.
    // Bias correction and delta computation stay in COMPUTE_TYPE; the subtraction
    // from the parameter is performed in ACCUM_TYPE to preserve state precision.
    const COMPUTE_TYPE param_delta = src_scalar_REAL_learning_rate * m_hat / 
                                     (MATH_FN sqrt(v_hat) + src_scalar_REAL_epsilon);

    const ACCUM_TYPE current_param = load_state_for_accum(update_buffer_GLOBAL_parameters, i);
    store_state_from_accum(update_buffer_GLOBAL_parameters, i, 
                           current_param - widen_to_accum(param_delta));
}
```

**Behavioral analysis by configuration:**

| Configuration | ACCUM_TYPE | EMA Precision | Overhead |
|:---|:---|:---|:---|
| `float32()` | FP32 | FP32 | Zero — all casts are identity |
| `mixed_f16_f32()` | FP32 | FP32 | Zero — ACCUM = COMPUTE |
| `mixed_f32_f64_state()` | FP64 | **FP64** | FP64 EMA arithmetic |
| `mixed_f32_f64()` | FP64 | FP64 | Zero — ACCUM = COMPUTE |
| `fp8_e4m3()` | FP32 | FP32 | Zero — ACCUM = COMPUTE |

---

### §7: Node 25 Contract Amendment — clamp_temperatures

The `clamp_temperatures` kernel also performs in-place state modification. Its contract is amended:

**Current:**
> "Enforces `temps = clamp(temps, min_value, max_value)` for each element. Precision Boundary Conversion: state-role buffer accessed via load_state()/store_state_update(); clamp arithmetic exclusively in COMPUTE_TYPE."

**Revised:**
> "Enforces `temps = clamp(temps, min_value, max_value)` for each element. **Note:** Clamping is a transformative operation (not accumulative) — precision boundary conversion applies. State-role buffer accessed via load_state()/store_state_update(); clamp arithmetic in COMPUTE_TYPE; result widened on store. State-Precision Accumulation does not apply — clamp has no cross-invocation precision accumulation."

The distinction is deliberate: `clamp_temperatures` reads a value, transforms it, and writes back. It does not accumulate state across invocations. The value is bounded by the clamp range, not by prior state history. Standard precision boundary conversion applies.

---

### §8: Alchemist II Scenario Revision

The scenario in CONCEPT.md §11 is revised:

**Current Validation Focus:**
> "Confirms that the FP64-state configuration's moment vectors track the FP64 reference within FP64 tolerance ($< 10^{-14}$ relative error)."

**Revised Validation Focus:**
> - **Validation Focus:** Confirms that the FP64-state configuration's moment vectors track the FP64 reference within FP64 tolerance ($< 10^{-14}$ relative error) **for the EMA update step** under State-Precision Accumulation. The test feeds identical pre-computed gradient sequences to all three configurations, isolating accumulation precision from input-precision divergence. The bias-corrected values (`m_hat`, `v_hat`) and parameter updates are bounded by `COMPUTE_TYPE` precision. Validates that state-precision accumulation correctly isolates moment fidelity from compute-path throughput.

**Revised Key Insight:**
> - **Key Insight:** Proves that the state role, combined with state-precision accumulation, achieves its stated goal: unbounded training stability through extended-precision moment vectors, while allowing narrower `COMPUTE_TYPE` for throughput in transformative operations. The configuration `mixed_f32_f64_state()` provides a distinct tradeoff from `mixed_f32_f64()`: the former preserves FP64 only where erosion compounds (moments), the latter uses FP64 throughout (higher fidelity, lower throughput).

---

### §9: Build System Symbol Amendment — Article 6

New derived symbols are added:

| Symbol | Definition | Category |
|:---|:---|:---|
| `ACCUM_TYPE` | C type for accumulative operations in stateful-update kernels. Equals `max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`, this is `STATE_TYPE`; otherwise `COMPUTE_TYPE`. | Derived |
| `ACCUM_IS_WIDER_THAN_COMPUTE` | Integer flag (0 or 1). True when `STATE_TYPE > COMPUTE_TYPE`, indicating that accumulation uses the wider state type rather than compute type. | Derived |

These symbols are derived from the existing `COMPUTE_TYPE`, `STATE_TYPE`, `COMPUTE_TYPE_IS_DOUBLE`, and `STATE_TYPE_IS_DOUBLE` flags. No additional host-provided symbols are required.

---

### §10: CPU Backend Amendment

The CPU backend's `cpu_precision.h` is extended with accumulation-precision macros:

```c
// === CPU Accumulation Precision Abstractions (ADR-027) ===
//
// Derives a boolean from the existing suffix tokens via preprocessor lookup.
// The _PREC_IS_F64_* table goes inside the include guard (defined once);
// the per-inclusion macros below are outside the guard, re-evaluated each
// time STATE_SUFFIX / COMPUTE_SUFFIX change — mirroring the existing
// storage-role and state-role pattern.

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

The CPU `adam_update` implementation in `phase_3_update.inc` is revised to mirror the OpenCL kernel's structure: EMA updates and the parameter subtraction (`p -= delta`) use `ACCUM_T` via the `scalar_*_accum` macros; bias correction and delta computation remain in `COMPUTE_T`. When `ACCUM_IS_WIDER_THAN_COMPUTE == 0`, all macros expand to identity operations — the generated code is identical to the prior implementation.

---

### §11: Vulkan Backend Amendment

The Vulkan backend's `common.glsl` is extended:

```glsl
// === Accumulation Precision Type (ADR-027) ===
//
// Uses the boolean flags STATE_TYPE_IS_DOUBLE and COMPUTE_TYPE_IS_DOUBLE
// established by ADR-024 §5.3 for the Vulkan backend.

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

When `STATE_TYPE == double` requires `GL_EXT_shader_explicit_arithmetic_types_float64` (already mandated by ADR-024 §5.3).

---

## Consequences

### Positive

- **The state role delivers its mandate.** `mixed_f32_f64_state()` and similar configurations genuinely provide extended-precision moment tracking. The Alchemist II scenario becomes falsifiable and valid.

- **The tradeoff space is complete.** Users can now meaningfully choose:
  - `mixed_f32_f64()` — FP64 compute throughout (highest fidelity, lowest throughput)
  - `mixed_f32_f64_state()` — FP64 moments only (stability without sacrificing forward-pass throughput)
  - `float32()` — FP32 throughout (maximum throughput, bounded stability)

- **The pattern localizes to stateful kernels.** The vast majority of kernels (forward pass, loss computation, gradient computation, reduction) are unchanged. Only `adam_update` requires the new abstraction.

- **Zero overhead when types match.** When `STATE_TYPE == COMPUTE_TYPE` (the common case), `ACCUM_TYPE == COMPUTE_TYPE` and all casts are identity operations eliminated by the compiler. The implementation is identical to the prior version.

- **Generalizes to FP8 and future formats.** FP8 configurations with FP32 compute + FP64 state (constructed directly via `PrecisionConfig(...)`) benefit from state-precision accumulation. The pattern is format-agnostic.

### Negative

- **FP64 EMA arithmetic on FP32-compute configurations.** `adam_update` performs FP64 arithmetic when `STATE_TYPE > COMPUTE_TYPE`. On hardware without native FP64, this may be emulated and slow. However:
  - Users choosing `*_f64_state()` explicitly value stability over throughput
  - The EMA computation is a small fraction of total kernel work
  - Forward-pass kernels (the throughput-critical path) remain at FP32

- **Additional precision abstraction complexity.** The vocabulary grows: `ACCUM_TYPE`, `load_state_for_accum`, `widen_to_accum`, `narrow_from_accum`. This is justified by the semantic distinction (accumulative vs. transformative) that the prior model conflated.

- **Backend implementation burden.** CPU, OpenCL, and Vulkan backends must implement the `ACCUM_*` abstractions. The pattern mirrors existing state-role abstractions and is straightforward.

### Neutral

- **Transformative vs. accumulative is a kernel-contract concern.** The Precision Boundary Conversion invariant continues to govern transformative operations. State-Precision Accumulation governs accumulative operations. Both invariants reduce to identical behavior when `STATE_TYPE == COMPUTE_TYPE`.

- **No change to `PrecisionConfig` factories.** The existing factories remain valid. The behavioral change is in the kernel contract, not the configuration.

---

## References

- CONCEPT.md §1 — Architectural Elegance Feedback
- CONCEPT.md §11 — Host Orchestrator, Alchemist II scenario
- ADR-020 — Three-Role Precision Model, §3.5 Behavioral Invariants
- ADR-024 — Double Precision Support, §3.2 load_state/store_state, §8.2 Alchemist II  
- ADR-025 — FP8 Support (FP8 is storage-role only; compute may be FP16/FP32/FP64; state may be FP16/FP32/FP64)
- ADR-026 — Precision-Typed Kernel Variants (establishes variant-rather-than-branch pattern)
