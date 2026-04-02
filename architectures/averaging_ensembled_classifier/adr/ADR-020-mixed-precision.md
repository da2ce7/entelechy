# ADR-020: Three-Role Precision Model

**Status:** ACCEPTED
**Date:** 2026-04-01
**Deciders:** —
**Supersedes:** ADR-008 PrecisionConfig design (§3.5 of DESIGN.md)
**Extends:** ADR-007, ADR-009, ADR-016
**Constrains:** ADR-013, ADR-014
**Depends on:** ADR-001 (three-tier jurisdictional model)
**Triggered by:** Recognition that the uniform `SCALAR_TYPE` model conflates three independent precision concerns; the Architectural Elegance Feedback protocol (CONCEPT.md §1) requires formalization

---

## Context

The system's current precision model treats numeric precision as a single axis. One `SCALAR_TYPE` at the kernel level, one `numpy_dtype` in `PrecisionConfig`, one `fp_format_max`, one `epsilon`. Every buffer, every arithmetic operation, every optimizer state variable uses the same type.

This uniform model conflates three independent concerns with different optimization targets:

**1. Storage precision is a bandwidth concern.** The buffers flowing through the DAG — activations, gradient partials, weight reads — are the dominant consumers of memory bandwidth. Narrowing the storage format halves bandwidth cost per halving of bit-width. The Primacy of Memory Strategy (CONCEPT.md §2) identifies bandwidth as the singular optimization target of host-side orchestration. Storage precision is the most direct lever for this — and it has nothing to do with arithmetic fidelity.

**2. Compute precision is a fidelity concern.** Reduction tree accumulation, Softmax log-sum-exp, gradient scaling — these operations consume mantissa bits. The $\log_K(N)$ Recursive Reduction Engine requires $\log_2(K)$ additional mantissa bits beyond the summands' precision for exact accumulation. The Quadratic Scaling Policy's safety ceiling ($T_{\text{safety}_j} = \text{FP\_FORMAT\_MAX} / K_j$) is bounded by the *arithmetic* format's range, not the *stored* format's.

**3. State precision is a stability concern.** Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires that the format can represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. Moment vectors persist across unbounded training steps. The architecture already acknowledges this concern: the host computes `beta1**t` in FP64 regardless of `SCALAR_TYPE` (CONCEPT.md §11). This is ad-hoc mixed-precision computation — the right pattern, without the right abstraction.

Under the uniform model, these three concerns rise and fall together. A user choosing FP16 for bandwidth also accepts FP16 for accumulation fidelity and FP16 for optimizer stability. A user choosing FP32 for fidelity forfeits FP16's bandwidth advantage. The mixed configuration — FP16 storage with FP32 arithmetic — is strictly superior for bandwidth-bound workloads, yet is architecturally inexpressible.

FP8 (E4M3/E5M2) makes the incompleteness catastrophic. Uniform FP8 produces a safety ceiling of $448/K$ (the policy funnel degenerates), Softmax over ~256 distinct positive values, EMA updates that round to either $m$ or $g$, and reduction mantissa saturation after one stage. FP8 as a uniform `SCALAR_TYPE` produces an engine that runs but does not train.

Per CONCEPT.md §1 (Architectural Elegance Feedback):

> 1. **Suspend** — Do not implement FP8 as a uniform `SCALAR_TYPE`.
> 2. **Formalize** — Establish the three-role precision model as a first-class architectural primitive.
> 3. **Reify** — Amend the authority documents and let implications flow through the architecture.

---

## Decision Drivers

1. **Top-down authority flow.** CONCEPT.md is the sovereign authority; CONTRACT.md is the binding interface law. An architectural primitive that does not exist in these documents does not exist in the architecture. The decision must amend the root authorities first, with downstream implications derived from those amendments — not the reverse.

2. **No special cases.** The architecture's principles reject modal complexity: Principle §5 unifies Act/Learn under one execution model; Principle §7 eliminates silent assumptions. A precision model with a "uniform mode" and a "mixed mode" introduces the same conditional complexity these principles reject. If three roles is the correct model, then a configuration where all three happen to share a type is a parameterization, not a separate mode.

3. **CONCEPT.md §1 (Architectural Elegance Feedback).** The three-step protocol is mandatory. The uniform `SCALAR_TYPE` is the incomplete constraint. This ADR is the formalization and reification.

4. **CONCEPT.md §2 (Primacy of Memory Strategy).** Storage precision is the most direct bandwidth lever. The architecture must decouple storage bandwidth from arithmetic fidelity.

5. **CONCEPT.md §6 (Architectural Hierarchy).** Design-level artifacts are subordinate to conceptual and contractual authorities. They never influence higher layers. Therefore, precision-role vocabulary must be established in CONCEPT.md and CONTRACT.md before any design-level dataclass or build-system change can be specified.

---

## Options Considered

### Option A: Extend the uniform model with more dtype choices

Add new dtypes (BFloat16, FP8) as uniform `SCALAR_TYPE` values.

**Rejected.** Does not address the root problem. Adding values to a single axis does not create the missing dimension. FP16/FP32 mixed precision remains inexpressible. Uniform FP8 is numerically non-functional.

### Option B: Dual precision model — uniform coexists with role-based

Introduce `MixedPrecisionConfig` alongside `PrecisionConfig`. Annotate buffers with `precision_role` where `None` means "use uniform `PrecisionConfig`." Mixed-precision machinery is dormant when all roles are `None`.

**Rejected.** Creates two parallel configuration types, two validation paths (role-annotated and role-absent), conditional compilation in kernel sources (`#if STORAGE_IS_NARROW`), and a deferred unification burden. The `precision_role = None` dormancy is a mode flag. This violates the "no special cases" driver: if three roles is the correct model, the uniform case is a parameterization of that model, not a different model.

### Option C: Native three-role model

Precision is *always* three roles: storage, compute, state. `PrecisionConfig` is redesigned with three dtype fields. Every buffer always carries a precision role. Every kernel always operates through precision-boundary abstractions. The configuration where all three roles share a type is a parameterization that the compiler optimizes to identity conversions. No dormancy. No mode flags. No conditional compilation.

### Option D: Per-buffer precision

Each `BufferDescriptor` carries an arbitrary `numpy_dtype` independent of any global configuration.

**Rejected.** Combinatorial validation explosion across 40–60 buffers. No semantic structure explaining *why* a buffer has its precision. No consistency guarantee between buffers of the same logical role. The practical mixed-precision configurations in use across the ML ecosystem are all expressible as 2–3 global role assignments.

---

## Analysis

### Eliminating Option B

Option B was the previous revision of this ADR. It correctly identified the three roles but introduced them as an *extension* to the uniform model rather than a *replacement* of it. This creates architectural debt:

1. **Two configuration types.** Code consuming precision configuration must dispatch on type (`isinstance(config, MixedPrecisionConfig)`) or rely on `precision_role = None` as a sentinel. This is the kind of conditional branching the architecture's principles reject at the kernel level (Principle §3) and at the execution model level (Principle §5).

2. **Conditional kernel sources.** `#if STORAGE_IS_NARROW` gates whether kernels perform widening/narrowing, creating two codepaths through every kernel. When the flag is 0, the mixed-precision path is dead code. When 1, a different path activates. Under Option C, there is one codepath — the compiler eliminates identity conversions when types are equal.

3. **Incomplete authority amendments.** Option B specified downstream extensions (dataclass fields, build symbols) but did not amend CONCEPT.md or fully amend CONTRACT.md. A precision model that exists in design-level dataclasses but not in the sovereign authority is an orphaned abstraction — it has no conceptual mandate and no contractual vocabulary.

4. **Deferred unification.** Option B acknowledged that "a future ADR may unify the two." Deferring unification is deferring the correct design.

Option C eliminates all four issues: one configuration type, one kernel codepath, complete authority-document amendments, no deferred work.

### Choosing Option C

Option C captures the actual structure of numeric computation: every value has a precision role determined by its function. The three roles are exhaustive:

| Role | Governs | Optimization target | Example values |
|:---|:---|:---|:---|
| **Storage** | Buffer memory layout, transfer bandwidth | Bandwidth — narrower is better | Forward-pass activations, gradient partials, weight read-views |
| **Compute** | Arithmetic accumulation, transcendentals, stability guards | Fidelity — wider is better | Reduction tree outputs, Softmax intermediates, loss values |
| **State** | Persistent model parameters, optimizer moments | Stability — wider is better | Adam `m1`/`m2` vectors, master weight copies |

Every buffer in the execution plan maps to exactly one role. Every arithmetic operation executes at compute precision. The uniform configuration `PrecisionConfig.float32()` is three roles that happen to be `float32` — not a different model, not a fallback, not a dormancy state. The kernel source is identical across all configurations; only type parameters differ.

The key implementation insight: **kernels always operate through the same precision-boundary pattern.** Every read widens from the buffer's role-type to `COMPUTE_TYPE`. Every write narrows from `COMPUTE_TYPE` to the buffer's role-type. When types are equal, these are identity operations that any compiler eliminates completely. No `#ifdef`, no mode flag, no conditional codepath.

---

## Decision

**Option C: Native three-role precision model.** Precision is always three independent roles. This decision is structured as amendments to the authority hierarchy, flowing top-down from conceptual to contractual to design-level implications.

---

### §1: Foundational Principle

The system's precision model is decomposed into three independent roles — **storage** (bandwidth), **compute** (fidelity), and **state** (stability) — that partition all values in the execution plan. This decomposition is the native precision model. There is no uniform-precision mode. A configuration where all three roles share a type is a parameterization of the three-role model, not an alternative to it.

---

### §2: Conceptual Authority Amendments (CONCEPT.md)

The following amendments are binding on CONCEPT.md and constitute first-class sovereign-authority content. Each amendment specifies the target section and the precise change.

#### §2.1: Amendment to Principle §2 — Primacy of Memory Strategy

The following paragraph is appended:

> **Storage precision is a first-class bandwidth lever.** The system decomposes numeric precision into three independent roles — storage, compute, and state — reflecting the distinct optimization targets of bandwidth, arithmetic fidelity, and long-term stability. Narrowing the storage format reduces buffer sizes and transfer bandwidth proportionally (FP16 halves FP32's footprint; FP8 halves again) without constraining the arithmetic precision used for computation or the precision of persistent optimizer state. This decoupling is a direct expression of the Primacy of Memory Strategy: data is stored in the narrowest format that preserves sufficient information for the downstream operation, and widened to the arithmetic format only at the point of computation.

#### §2.2: Amendment to §3.4 — Orthogonal Enforcement of Numerical Safety

The safety ceiling formula is amended. The current text:

> **`T_safety_j = FP_FORMAT_MAX / K_j`**
>
> **`FP_FORMAT_MAX`**: A conservative, high value representing the maximum representable number for the current precision (e.g., `~6.5e4` for FP16, `~3.4e38` for FP32).

Is replaced by:

> **`T_safety_j = COMPUTE_FP_FORMAT_MAX / K_j`**
>
> **`COMPUTE_FP_FORMAT_MAX`**: The maximum representable value of the **compute precision** format (e.g., `~3.4e38` for FP32 compute, `~6.5e4` for FP16 compute). The safety ceiling reflects the arithmetic precision that performs the summation, not the storage precision of the values being summed. In a configuration where all precision roles share a format, `COMPUTE_FP_FORMAT_MAX` equals that format's maximum.

#### §2.3: Amendment to §7 — Kernel Contract Model

The following is appended to the `KernelContract` description:

> Each buffer parameter in a `KernelContract` carries a `precision_role` declaration — one of `"storage"`, `"compute"`, or `"state"` — that identifies the buffer's element type under the active precision configuration. When a kernel reads a buffer whose precision role differs from `"compute"`, the kernel widens the value to compute precision before any arithmetic. When writing, the kernel narrows from compute precision to the buffer's role precision. These conversions are declared in the contract's `Behavioral Invariants` as a mandatory `Precision Boundary Conversion` invariant. The conversion pattern is structurally identical regardless of whether the role types happen to be equal — the kernel source has one codepath, and the compiler eliminates identity conversions.

#### §2.4: Amendment to §8 — Buffer Lifecycle

The `BufferDescriptor` description is amended. The sentence introducing `BufferDescriptor` is extended with:

> Every device-side buffer's `BufferDescriptor` carries a `precision_role` field — one of `"storage"`, `"compute"`, or `"state"` — that determines the buffer's element type and allocation size under the active `PrecisionConfig`. The `element_size_bytes` and `size_bytes` fields derive from the role-appropriate dtype. A storage-role activation buffer allocated at FP16 is half the size of a compute-role accumulation buffer at FP32, enabling the bandwidth benefits of narrow formats without affecting the fidelity of arithmetic operations.

#### §2.5: Amendment to §11 — Host Orchestrator, Item 7

The current sentence:

> "Precision is uniformly configured through a `PrecisionConfig` frozen dataclass carrying `numpy_dtype`, `fp_format_max`, and `epsilon`."

Is replaced by:

> Precision is configured through a `PrecisionConfig` frozen dataclass carrying three independent role assignments: `storage_dtype` (bandwidth optimization), `compute_dtype` (arithmetic fidelity), and `state_dtype` (optimizer stability), along with derived constants `storage_fp_format_max`, `compute_fp_format_max`, and `compute_epsilon`. A configuration where all three roles share a type (e.g., `PrecisionConfig.float32()`) is a parameterization — not a distinct mode. The host's existing practice of computing Adam bias correction terms in FP64 is recognized as an instance of state-precision independence; the three-role model formalizes this pattern as a first-class concern rather than an ad-hoc exception.

#### §2.6: New Validation Scenario

The following is added to *Validation Scenarios*:

> **Scenario: The Alchemist (Mixed-Precision Fidelity Validation)**
>
> - **Description:** The same training task is executed in three precision configurations: `PrecisionConfig.float32()` (all roles FP32), `PrecisionConfig.float16()` (all roles FP16), and `PrecisionConfig.mixed_f16_f32()` (storage FP16, compute FP32, state FP32). All three use the same `PrecisionConfig` with three roles — there is no mode switch.
> - **Validation Focus:** Confirms that the mixed configuration produces loss curves and final parameters tracking the FP32 baseline within tolerance, while achieving memory footprint comparable to the all-FP16 configuration. Validates that buffer allocation sizes reflect `storage_dtype`, that accumulation uses `compute_dtype`, that the stabilization policy's safety ceiling derives from `compute_fp_format_max`, and that optimizer state maintains `state_dtype` fidelity.
> - **Key Insight:** Proves the three-role decomposition achieves its stated goal: narrow-storage bandwidth without narrow-compute fidelity loss. The architectural machinery is identical across all three configurations — only the type parameters differ. This scenario validates not a "mixed-precision mode" but the generality of the precision model itself.

---

### §3: Contractual Authority Amendments (CONTRACT.md)

The following amendments are binding on CONTRACT.md. They flow from the conceptual amendments in §2 and establish the machine-verifiable interface vocabulary for the three-role precision model.

#### §3.1: Amendment to Article 2.1 — Buffer Name Grammar

The buffer name grammar is **unchanged**. Precision role is not a naming component — it is a commentary and contract concern exclusively. The C type declaration (`STORAGE_TYPE*`, `COMPUTE_TYPE*`, `STATE_TYPE*`) distinguishes precision roles at the kernel source level; the parameter name identifies the buffer's *logical function*, not its *physical type*.

The following note is added to Article 2.1:

> A buffer's precision role is determined by its C type declaration (`STORAGE_TYPE`, `COMPUTE_TYPE`, or `STATE_TYPE`) and formally specified in the parameter's `Precision Role` commentary key (Article 3). It is not encoded in the buffer name's `[ContextAndUsage]` component.
>
> **Rationale (Article 1.2 — Axiom of Semantic Uniqueness):** The C declaration encodes the precision role via the type symbol. Duplicating this information in the parameter name violates the prohibition on cross-jurisdictional redundancy.

#### §3.2: Amendment to Article 3 — Parameter Commentary Contract

A new commentary key is added to the mandatory key list, positioned after `Padding Contract` and before `Calculability Proof`:

| Key | Definition | Status |
|:---|:---|:---|
| **`Precision Role`** | One of `"storage"`, `"compute"`, or `"state"`. Declares which precision-role dtype from the active `PrecisionConfig` governs this buffer's element type and allocation size. | **Mandatory** for all buffer parameters |

This key closes the gap between the human-readable `@param` block and the machine-verifiable `BufferParamSpec` dataclass, maintaining the closed-system property of Article 1.4: all information needed to validate a buffer's interface is present in its parameter specification.

Scalar parameters do not carry a `Precision Role` key. Their types are declared directly in the kernel signature and are not role-dependent — they are passed by value from the host at whatever type the interface specifies.

#### §3.3: Amendment to Article 3.1 — Padding Contract

The following clarification is appended:

> Element size for all padding calculations (SIMD alignment, cache line alignment) derives from the buffer's declared `Precision Role` dtype. An FP16-storage buffer with `SIMD_WIDTH=8` (in compute-type elements) pads to a different byte boundary than an FP32-compute buffer with the same logical dimension.
>
> When computing the padded dimension in elements, the buffer's role-type element size is used: `padded_elements = ceil(logical_elements, alignment_elements)` where `alignment_elements` depends on both the hardware alignment requirement and `sizeof(ROLE_TYPE)`.

#### §3.4: Amendment to Article 3.2 — Placement Contract

The following clarification is appended to Article 3.2.1:

> Write offset calculations in all placement strategies use the element size of the destination buffer's `Precision Role` dtype. The physical byte displacement for a partial of `partial_width` elements at precision role `R` is: `flat_tile_index × partial_width × sizeof(R_dtype)`.

#### §3.5: New Addition to Article 4.2 — Behavioral Invariants Vocabulary

The following term is added to Article 4.2's recognized `Behavioral Invariants` vocabulary:

| Term | Definition |
|:---|:---|
| **`Precision Boundary Conversion`** | A mandatory invariant for any kernel receiving buffer parameters whose `Precision Role` is not `"compute"`. The kernel shall: (1) widen all non-compute-role inputs to `COMPUTE_TYPE` upon load, (2) perform all arithmetic exclusively in `COMPUTE_TYPE`, and (3) narrow results from `COMPUTE_TYPE` to the destination buffer's role type upon store. This invariant is *structural* — it applies identically regardless of whether the role types happen to be equal. The kernel source does not branch on type equality; the compiler eliminates identity conversions. Widening and narrowing are performed through platform-appropriate load/store abstractions defined in the shared kernel header. |

When a kernel's `@kernel_contract` block includes `Precision Boundary Conversion` in its `Behavioral Invariants`, the host's proof obligation (Article 1.4a) extends to verifying that:
- Every buffer parameter with `Precision Role` other than `"compute"` is declared with the role-appropriate type symbol in the kernel signature.
- The kernel's `Behavioral Invariants` explicitly states the conversion obligation.

#### §3.6: Replacement of Article 6 — Mandatory Build-Time Symbols

Article 6 is replaced in its entirety. The following symbols are mandatory for all kernel compilations:

| Symbol | Definition | Category |
|:---|:---|:---|
| `STORAGE_TYPE` | C type for buffers with `Precision Role: "storage"`. Defines the element type for bandwidth-optimized transient DAG data (activations, gradient partials, intermediate results). | Primary |
| `COMPUTE_TYPE` | C type for all kernel-internal arithmetic — accumulation, transcendentals, numerical stability guards. Every kernel's algorithmic core operates exclusively in this type. | Primary |
| `STATE_TYPE` | C type for buffers with `Precision Role: "state"`. Defines the element type for persistent model parameters and optimizer state vectors. | Primary |
| `STORAGE_TYPE_IS_HALF` | Integer flag (0 or 1). Required because the C preprocessor cannot perform type-name comparison. Governs platform-specific load/store mechanics for half-precision storage (e.g., `vload_half`/`vstore_half` on OpenCL). | Derived |
| `COMPUTE_TYPE_IS_HALF` | Integer flag (0 or 1). Governs precision-sensitive algorithm selection in the compute path (e.g., transcendental function implementation, stability guard thresholds). | Derived |
| `NUMERICAL_STABILITY_EPSILON` | The minimum epsilon for numerical stability guards (division-by-zero prevention, log-domain clamping). Derived from `COMPUTE_TYPE` precision — it governs arithmetic stability, which is a compute-precision property. | Derived |
| `SIMD_WIDTH` | The target SIMD vector width in elements of `COMPUTE_TYPE`. | Hardware |
| `C_TILE_SIZE` | The block/tile dimension for tiled algorithms. | Configuration |

**Superseded symbols.** The following symbols from the prior Article 6 are retired:

| Retired Symbol | Replacement | Migration |
|:---|:---|:---|
| `SCALAR_TYPE` | `STORAGE_TYPE`, `COMPUTE_TYPE`, or `STATE_TYPE` per the buffer's precision role | During migration, the build system may provide `SCALAR_TYPE` as a transitional alias for `COMPUTE_TYPE`. This alias shall be removed upon migration completion. |
| `SCALAR_IS_HALF` | `STORAGE_TYPE_IS_HALF` or `COMPUTE_TYPE_IS_HALF` as appropriate | Same transitional treatment |

**Uniform-configuration property.** When all three roles share a type (e.g., `PrecisionConfig.float32()`), `STORAGE_TYPE == COMPUTE_TYPE == STATE_TYPE`. The build system provides all three symbols with identical values. Kernel sources are identical — the compiler eliminates identity conversions in the precision-boundary abstractions. No conditional compilation for "uniform mode" is required or permitted.

#### §3.7: Amendment to Article 7 — Canonical Interface Instantiation

The canonical example is extended to demonstrate precision-role declarations and multi-type buffer signatures:

```c
/**
* @brief Demonstrates a kernel interface with precision-role-typed buffers.
* @kernel_contract
*        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
*        - Idempotency: "Strictly Idempotent"
*        - Behavioral Invariants: "Precision Boundary Conversion: All STORAGE_TYPE and STATE_TYPE
*          inputs are widened to COMPUTE_TYPE upon load. All STORAGE_TYPE outputs are narrowed from
*          COMPUTE_TYPE upon store. No arithmetic is performed outside COMPUTE_TYPE."
*/
__kernel void illustrative_mixed_precision_kernel(
    /**
    * @param src_buffer_GLOBAL_hidden_activations The post-activation hidden layer output.
    *        - Precision Role: "storage"
    *        - Tensor Shape: (src_scalar_NATURAL_batch_size, src_scalar_NATURAL_padded_hidden_width)
    *        - Padding Contract: {Type: SIMD, Formula: "Pad hidden_width to SIMD boundary at sizeof(STORAGE_TYPE)"}
    *        - Calculability Proof: [src_scalar_NATURAL_batch_size, src_scalar_NATURAL_padded_hidden_width]
    *        - Validation Preconditions: [src_scalar_NATURAL_padded_hidden_width >= src_scalar_NATURAL_hidden_width]
    */
    __global const STORAGE_TYPE* src_buffer_GLOBAL_hidden_activations,

    /**
    * @param src_buffer_GLOBAL_CONST_weights The learnable weight matrix (persistent model state).
    *        - Precision Role: "state"
    *        - Tensor Shape: (src_scalar_NATURAL_output_height, src_scalar_NATURAL_padded_input_width)
    *        - Padding Contract: {Type: SIMD, Formula: "Pad input_width to SIMD boundary at sizeof(STATE_TYPE)"}
    *        - Calculability Proof: [src_scalar_NATURAL_output_height, src_scalar_NATURAL_padded_input_width]
    *        - Validation Preconditions: None.
    */
    __global const STATE_TYPE* src_buffer_GLOBAL_CONST_weights,

    /**
    * @param dest_buffer_GLOBAL_summed_gradient The accumulated gradient after reduction.
    *        - Precision Role: "compute"
    *        - Tensor Shape: (dest_scalar_NATURAL_gradient_width)
    *        - Padding Contract: {Type: SIMD, Formula: "Pad to SIMD boundary at sizeof(COMPUTE_TYPE)"}
    *        - Calculability Proof: [dest_scalar_NATURAL_gradient_width]
    *        - Validation Preconditions: None.
    */
    __global COMPUTE_TYPE* dest_buffer_GLOBAL_summed_gradient,

    uint src_scalar_NATURAL_batch_size,
    uint src_scalar_NATURAL_padded_hidden_width,
    uint src_scalar_NATURAL_hidden_width,
    uint src_scalar_NATURAL_output_height,
    uint src_scalar_NATURAL_padded_input_width,
    uint dest_scalar_NATURAL_gradient_width
);
```

The existing uniform-precision canonical example in Article 7 is retained as valid — under the three-role model, a kernel where all buffer parameters share a precision role and all role types are equal produces the same signature (with `STORAGE_TYPE`, `COMPUTE_TYPE`, or `STATE_TYPE` replacing `SCALAR_TYPE`).

#### §3.8: Amendment to Article 8 — Canonical Lexicon

No new `[ContextAndUsage]` terms are required. Precision role is expressed through the C type declaration and the `Precision Role` commentary key, not through the buffer name.

The following clarification is added to Section 5.0, Generic Matrix Ops:

> `stride` — The physical displacement, in elements of the *buffer's declared precision-role type*, required to move from the start of one row to the start of the next. When computing byte offsets from element strides, the element count is multiplied by `sizeof(ROLE_TYPE)` where `ROLE_TYPE` corresponds to the buffer's `Precision Role`.

---

### §4: Design-Level Implications

The following changes to design-level artifacts are direct consequences of the §2 and §3 amendments. They are binding but subordinate — they exist because the authority documents require them, not as independent decisions.

#### §4.1: PrecisionConfig Redesign (supersedes ADR-008)

`PrecisionConfig` is redesigned from the ground up. The prior single-dtype model is retired.

```python
@dataclass(frozen=True)
class PrecisionConfig:
    """Three-role precision configuration.

    Every precision configuration specifies three independent roles.
    The configuration where all roles share a type is a parameterization,
    not a distinct mode.
    """
    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype

    storage_fp_format_max: float
    compute_fp_format_max: float
    compute_epsilon: float
```

Factory classmethods:

| Factory | storage | compute | state |
|:---|:---|:---|:---|
| `PrecisionConfig.float32()` | float32 | float32 | float32 |
| `PrecisionConfig.float16()` | float16 | float16 | float16 |
| `PrecisionConfig.mixed_f16_f32()` | float16 | float32 | float32 |

Construction invariants at `__post_init__`:
- `storage_dtype.itemsize <= compute_dtype.itemsize`
- `storage_dtype.itemsize <= state_dtype.itemsize`
- `compute_fp_format_max >= storage_fp_format_max`

There is no `MixedPrecisionConfig`. There is no `precision_role = None`. The prior `numpy_dtype`, `fp_format_max`, and `epsilon` fields are removed. `ModelSpec` factories delegate to the new `PrecisionConfig` factories.

#### §4.2: BufferDescriptor Amendment (extends ADR-009)

`BufferDescriptor.precision_role` is always one of `"storage"`, `"compute"`, or `"state"`. The field is mandatory, not optional. `element_size_bytes` and `size_bytes` derive from the role-appropriate dtype via the active `PrecisionConfig`.

The plan builder assigns precision roles based on each buffer's semantic function:

| Buffer classification | Precision role | Rationale |
|:---|:---|:---|
| Forward-pass activations, gradient partials, logit intermediates | `"storage"` | Transient DAG data; bandwidth-bound |
| Reduction tree accumulation outputs, loss values | `"compute"` | Arithmetic fidelity required during summation |
| Learnable parameters (weights, biases, temperatures), optimizer moments (`m1`, `m2`) | `"state"` | Persist across batches; require long-term numerical stability |

#### §4.3: KernelContract Amendment (extends ADR-007)

`BufferParamSpec.precision_role` is always one of `"storage"`, `"compute"`, or `"state"`. Plan-construction validation verifies that each buffer binding's precision role matches the contract's declaration, and that each kernel accessing non-`"compute"`-role buffers declares `Precision Boundary Conversion` in its `Behavioral Invariants`.

#### §4.4: Kernel Source Strategy (constrains ADR-013)

Kernel sources are migrated from `SCALAR_TYPE` to the role-appropriate type symbols (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`). The algorithmic core of every kernel operates exclusively in `COMPUTE_TYPE`. Data boundary conversions use platform-appropriate abstractions defined in the shared kernel header:

```c
// kernels.cl.h — precision boundary abstractions
static inline COMPUTE_TYPE load_storage(__global const STORAGE_TYPE* buf, size_t idx);
static inline void store_storage(__global STORAGE_TYPE* buf, size_t idx, COMPUTE_TYPE val);
static inline COMPUTE_TYPE load_state(__global const STATE_TYPE* buf, size_t idx);
static inline void store_state(__global STATE_TYPE* buf, size_t idx, COMPUTE_TYPE val);
```

When `STORAGE_TYPE == COMPUTE_TYPE`, `load_storage` and `store_storage` are identity operations that the compiler eliminates. When they differ, they perform the appropriate type conversion — using `vload_half`/`vstore_half` on OpenCL when `STORAGE_TYPE_IS_HALF == 1`, F16C intrinsics on CPU, or standard casts otherwise.

The `#if STORAGE_IS_NARROW` conditional compilation pattern is **not used**. There is no branching on type equality. The abstractions produce correct code for all configurations uniformly. Each backend's implementation of these abstractions resides in its `kernel_sources/` directory per ADR-013.

#### §4.5: Build System Implications (constrains ADR-014)

The build configuration provides `STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`, and the derived `_IS_HALF` flags to every kernel compilation unit. The specific Meson option structure — whether a single `aec_precision` option expands to three type assignments, or factory-named presets are used — is a build-system design concern resolved during implementation within the constraints of this ADR.

During the `SCALAR_TYPE` migration period, the build system may provide `SCALAR_TYPE` and `SCALAR_IS_HALF` as transitional aliases (mapped to `COMPUTE_TYPE` and `COMPUTE_TYPE_IS_HALF` respectively). These aliases are removed when the migration completes.

**Host-mode stubs.** The kernel header's host/C++ mode stub definitions (`#else` branch when `__OPENCL_VERSION__` is not defined) also define `SCALAR_TYPE` as a transitional alias for `COMPUTE_TYPE`. This enables the host-mode validation build (`libkernel_host_validate.a`) to compile kernel signatures that have not yet been migrated to the three-role model. The host stubs mirror the build-system alias for consistency.

#### §4.6: Test Strategy Implications (extends ADR-016)

- **Tier 2 tests** execute against all active `PrecisionConfig` factory configurations, including at least one configuration where roles diverge (`mixed_f16_f32`).
- **Tier 3 cross-backend parity tests** include mixed-precision configurations to validate that role-appropriate type handling is consistent across backends.
- **Tolerance configuration** gains a role-aware dimension: numerical closeness tolerances for mixed-precision runs are derived from the compute precision (which governs arithmetic fidelity), not the storage precision.
- **The Alchemist** validation scenario (§2.6) is the conceptual Tier 3 mandate; its implementation is a Tier 3 test suite that compares FP32, FP16, and mixed-F16/F32 runs.

#### §4.7: Stabilization Policy (consequence of §2.2)

The Quadratic Scaling Policy's safety ceiling computation uses `PrecisionConfig.compute_fp_format_max`:

$$T_{\text{safety}_j} = \frac{\text{config.compute\_fp\_format\_max}}{K_j}$$

This follows directly from the §2.2 CONCEPT.md amendment. The safety ceiling prevents overflow during summation — a compute-precision operation — regardless of the storage precision of the values being summed.

---

### §5: Narrow-Format Instantiation Preconditions

The three-role model is immediately applicable to all configurations expressible with FP16 and FP32 on all three backends. For formats narrower than FP16 (FP8 E4M3/E5M2), instantiation as a `storage_dtype` value in `PrecisionConfig` is deferred until:

| Precondition | Rationale |
|:---|:---|
| A C storage type for E4M3/E5M2 in a released GCC or Clang | The CPU backend requires a concrete `STORAGE_TYPE`. Software emulation via `uint8_t` with manual bit manipulation hides the type from compiler verification. |
| `ml_dtypes` (or equivalent) FP8 dtype stable and interoperable with NumPy | `PrecisionConfig.storage_dtype` must support `np.dtype().itemsize` and array construction. |
| At least one GPU backend has a viable FP8 load/store dispatch path | FP8 storage usability requires platform support for typed 8-bit float memory access (e.g., `VK_KHR_8bit_storage`, an OpenCL extension, or a demonstrated cast-from-uint8 pattern). |

These preconditions gate FP8-specific *code*. The architectural primitives decided in §§1–4 are binding, format-independent, and impose no FP8 toolchain requirement. When the preconditions are met, FP8 slots into `storage_dtype` with FP16 or FP32 as `compute_dtype` — no new architectural vocabulary is needed.

---

## Consequences

### Positive

- **Authority documents are complete.** The three-role model exists as sovereign-authority content in CONCEPT.md (§2) and as binding interface vocabulary in CONTRACT.md (§3). No downstream artifact introduces precision vocabulary that the authority documents do not define. The top-down authority flow (CONCEPT.md §6) is preserved.
- **No special cases.** One `PrecisionConfig`. One precision model. One kernel codepath. The "uniform FP32" configuration is `PrecisionConfig.float32()` — three roles that share the type `float32`. No mode switch, no dormancy, no conditional compilation.
- **CONCEPT.md §1 fully satisfied.** The Architectural Elegance Feedback protocol is complete: the uniform `SCALAR_TYPE` constraint is identified as incomplete, the three-role model is formalized in the sovereign authority, and the reification flows through the contractual and design layers.
- **Immediately useful.** `PrecisionConfig.mixed_f16_f32()` delivers bandwidth improvement with full arithmetic fidelity on all three backends, using FP16 and FP32 types that are already fully supported.
- **Kernel sources are uniform.** Every kernel uses precision-boundary load/store abstractions. No `#ifdef` on type equality. One codepath serves all configurations. Identity conversions are eliminated by the compiler — zero overhead for the equal-type case.
- **FP8 pathway is architectural, not aspirational.** The load/store abstractions, the `Precision Role` commentary key, the build-time type symbols, and the role-aware `BufferDescriptor` are all in place. FP8 instantiation requires only new platform-specific load/store implementations and a new `PrecisionConfig` factory — no new primitives.
- **Contract completeness.** The `Precision Role` commentary key, the `Precision Boundary Conversion` behavioral invariant, the role-typed canonical example, the padding and placement clarifications, and the complete build-time symbol replacement ensure that CONTRACT.md has full, self-consistent vocabulary for verifying precision-aware kernel interfaces.

### Negative

- **Breaking change to PrecisionConfig.** The redesign removes `numpy_dtype`, `fp_format_max`, and `epsilon`. All code depending on these fields must migrate. This is a one-time cost confined to the shared layer, backend discovery modules, and `ModelSpec`.
- **SCALAR_TYPE migration across all kernel sources.** Every `.cl.c`, `.comp`, `.inc` file must be migrated from `SCALAR_TYPE` to role-appropriate symbols and load/store abstractions. The migration is mechanical (find-replace plus abstraction wrapping) but touches 32+ source files.
- **Every buffer parameter gains a mandatory commentary key.** The `Precision Role` key must be added to every buffer `@param` block in the kernel specification corpus. This is mechanical but non-trivial in scope.
- **New correctness obligation on every kernel.** `Precision Boundary Conversion` must be declared and implemented by every kernel accessing non-`"compute"`-role buffers. Verified by contract validation and Tier 2 tests, but adds to the verification surface.
- **Build system surface grows.** Three primary type symbols and two derived flags replace one primary symbol and one derived flag. Factory-named presets reduce the user-facing complexity, but the build system internals must manage the mapping.

### Risk Assessment

| Risk | Impact | Mitigation |
|:---|:---|:---|
| `SCALAR_TYPE` migration introduces regressions | Medium | Gated by existing Tier 2 tests; transitional alias during migration prevents broken builds |
| Precision-boundary abstractions have overhead for uniform configs | Negligible | Identity casts are eliminated by all production C/GLSL compilers. Verified by benchmarking before transitional alias removal. |
| `Precision Role` commentary key creates maintenance burden | Low | Role assignment is mechanical — each buffer's role follows from its logical function, which the existing contracts already describe. |
| Three build-time type symbols complicate backend build logic | Low | Factory presets (e.g., "float32", "mixed_f16_f32") reduce the configuration to a single named choice. The expansion to three symbols is the build system's concern, not the user's. |

---

## References

- CONCEPT.md §1 — Architectural Elegance Feedback: the protocol that mandates this ADR's structure
- CONCEPT.md §2 — Primacy of Memory Strategy: storage precision as bandwidth optimization (amended by §2.1)
- CONCEPT.md §3.4 — Orthogonal Enforcement of Numerical Safety: safety ceiling formula (amended by §2.2)
- CONCEPT.md §6 — Architectural Hierarchy: top-down authority flow that structures this ADR
- CONCEPT.md §7 — Kernel Contract Model (amended by §2.3)
- CONCEPT.md §8 — Buffer Lifecycle (amended by §2.4)
- CONCEPT.md §11 — Host Orchestrator (amended by §2.5)
- CONTRACT.md Article 1.2 — Axiom of Semantic Uniqueness (rationale for §3.1)
- CONTRACT.md Article 1.4 — Axiom of Collaborative Interface Verifiability (extended by §3.5)
- CONTRACT.md Articles 2–8 — Full interface grammar, commentary, build symbols, lexicon (amended by §3.1–§3.8)
- [ADR-001](adr/ADR-001-backend-abstraction-boundary.md) — Three-tier jurisdictional model: precision roles are a Policy-tier concern
- [ADR-007](adr/ADR-007-kernel-signature-contract-binding-split.md) — KernelContract/KernelBinding split: extended by §4.3
- [ADR-008](adr/ADR-008-precision-configuration.md) — PrecisionConfig: superseded by §4.1
- [ADR-009](adr/ADR-009-buffer-lifecycle-in-the-plan-model.md) — BufferDescriptor: extended by §4.2
- [ADR-013](adr/ADR-013-kernel-source-strategy.md) — Kernel source strategy: constrained by §4.4
- [ADR-014](adr/ADR-014-build-system-integration.md) — Build system: constrained by §4.5
- [ADR-016](adr/ADR-016-test-strategy.md) — Test strategy: extended by §4.6

