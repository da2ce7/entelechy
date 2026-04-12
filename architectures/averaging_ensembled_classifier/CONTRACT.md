### **System Contract: Host-Device Kernel Interface (Revision 8)**

#### **Preamble**

This document constitutes the definitive and inviolable contract governing all interactions across the Host-Device boundary. Its articles are not guidelines but fundamental laws of the system architecture. Adherence is mandatory and absolute. This contract supersedes all prior conventions and informal agreements.

---

### **Article 1: Foundational Axioms**

The architecture is immutably founded upon the following axioms.

#### 1.1. Axiom of Jurisdictional Separation.

A parameter's syntactic structure (**Name**) defines its machine-enforced contract. A parameter's semantic block (**Commentary**) defines its human-verifiable and logical contract. These two jurisdictions are distinct and exhaustive.

#### 1.2. Axiom of Semantic Uniqueness.

Information encoded within the syntactic jurisdiction (Name) is prohibited from being duplicated within the semantic jurisdiction (Commentary), and vice versa. There shall exist no redundancy between the two.

#### 1.3. Axiom of Memory Layout.

All multi-dimensional buffers are contractually obligated to be stored in a **row-major memory layout**. The physical address of an element is calculated accordingly. Any deviation from this layout must be explicitly signaled by a canonical layout suffix (e.g., `_soa`).

#### 1.4: Axiom of Collaborative Interface Verifiability

A kernel's public interface constitutes a **closed logical system** for verification. Validation follows a layered responsibility model:

- **a. Host Proof Obligation** - The host's `KernelSignature` abstraction performs rigorous pre-flight validation of all buffer dimensions, scalar constraints, and placement strategies _before dispatch_.
- **b. Device Assurance Fallback** - Kernels may include verification parameters (even redundant ones) solely for final lightweight sanity checks. This defensive redundancy serves as a fail-safe mechanism against host-side validation failures, not as primary verification.

> _Illustration:_ A kernel calculating `dest_buffer[padded_width * row + col]` may receive both `padded_width` and `total_rows` as parameters, though `total_rows` could be derived from other values. This satisfies both:
>
> - Host pre-validates dimensional consistency via `Calculability Proofs`
> - Device verifies `row < total_rows` via explicit parameter

- **1.4.1. Inviolable Constraint:** All parameters used in each `Calculability Proof` and `Validation Preconditions` must exist in the interface, preserving the closed system. Redundancy for assurance doesn't violate minimalism when serving this axiom.

### **Article 2: Parameter Lexical Mandate**

The structure of a parameter name is formally specified. This grammar is not a convention but a mandatory syntactical requirement for interface validity.

**2.1. Buffer Name Grammar**
A buffer identifier shall be constructed as:
`[Flow] :: "buffer" :: [MemoryScope] :: [ContextAndUsage]`

- **`[Flow]`**: `src_` | `dest_` | `update_` | `sync_`
- **`[MemoryScope]`**: `GLOBAL_` | `LOCAL_` | `GLOBAL_CONST_` | `DEVICE_CONST_`
- **`[ContextAndUsage]`**: A canonical identifier defined exclusively in **Article 8: Canonical Lexicon**.

**2.1.1. Memory Scope Token Definitions**

- **`GLOBAL_`**: Standard `__global` device memory for pipeline data. Buffers in this scope represent transient, per-dispatch data flowing through the computational DAG (inputs, activations, masks, partials, intermediates). They may carry the `const` qualifier in the C declaration when used as a read-only source.
- **`LOCAL_`**: Work-group exclusive `__local` memory.
- **`GLOBAL_CONST_`**: Read-only `__global` device memory holding host-provided parametric data that configures the computational transformation and is invariant for the duration of a kernel dispatch. This includes learnable model state (weights, biases, temperatures), host-computed structural metadata (offset lists, indirection tables), and host-computed policy values (clipping thresholds).
- **`DEVICE_CONST_`**: The hardware-specific, read-only `__constant` address space.

> **Note (Article 1.2 — Axiom of Semantic Uniqueness):** A buffer's precision role is determined by its C type declaration (`STORAGE_TYPE`, `COMPUTE_TYPE`, or `STATE_TYPE`) and formally specified in the parameter's `Precision Role` commentary key (Article 3). It is not encoded in the buffer name's `[ContextAndUsage]` component. The C declaration encodes the precision role via the type symbol. Duplicating this in the name would violate the prohibition on cross-jurisdictional redundancy.

**2.2. Scalar Name Grammar**
A scalar identifier shall be constructed as:
`[Flow] :: "scalar" :: [NumberType] :: [ContextAndUsage]`

- **`[Flow]`**: `src_` | `dest_`
- **`[NumberType]`**: A mandatory prefix defining the parameter's abstract numerical domain.
- **`[ContextAndUsage]`**: A canonical identifier defined exclusively in **Article 8: Canonical Lexicon**.

**2.3. Scalar `[NumberType]` Taxonomy**
The `[NumberType]` component defines the set of valid values for a scalar.

| Token      | Abstract Numerical Domain            | Core Constraint                                                      |
| :--------- | :----------------------------------- | :------------------------------------------------------------------- |
| `NATURAL_` | Natural Numbers (ℕ₀: {0, 1, 2, ...}) | Value must be non-negative. Defines counts, indices, and sizes.      |
| `INTEGER_` | Integers (ℤ: {..., -1, 0, 1, ...})   | Value may be negative. Defines offsets and signed quantities.        |
| `REAL_`    | Real Numbers (ℝ)                     | Value represents a continuous quantity. Defines thresholds and data. |
| `FLAG_`    | Boolean Set ({0, 1})                 | Value must be `0` or `1`. Defines logical switches.                  |

### **Article 3: Parameter Commentary Contract**

The `@param` block constitutes the complete logical specification for a parameter. It shall contain the following keys as required:

- **`Tensor Shape`**: The logical dimensions of the tensor.
- **`Padding Contract`**: A key-value object literal specifying padding strategy. > Under the three-role precision model, padding byte counts use the element size of the buffer's actual `precision_role` type: `sizeof(STORAGE_TYPE)`, `sizeof(COMPUTE_TYPE)`, or `sizeof(STATE_TYPE)` as appropriate. Using a different role's `sizeof` in a `Padding Contract` expression is a contract violation.
- **`Precision Role`**: One of `"storage"`, `"compute"`, `"state"`, or `"flag-conditional"`. Declares which precision-role dtype from the active `PrecisionConfig` governs this buffer's element type and allocation size. **Mandatory** for all floating-point buffer parameters. Integer-typed buffers (`int`, `uint`, `atomic_uint`) whose element size is format-independent are exempt. When the role is `"flag-conditional"`, the buffer's element type depends on a FLAG scalar parameter's value; the commentary block MUST enumerate each flag value and its corresponding interpretation (role+type). Host-side validation operates on the runtime flag value.
- **`Calculability Proof`**: Defines the derivation of buffer dimensions or scalar values through a constructive arithmetic expression composed solely of parameters present within the kernel's interface. All terms in this expression shall correspond to kernel arguments, satisfying the Axiom of Interface Verifiability (1.4).
- **`Initialization Contract`**: For `dest_` flow buffers, declares whether the Host must pre-initialize the buffer contents before the producing kernel(s) are dispatched. Omission is equivalent to `{Type: NONE}`.
- **`Validation Preconditions`**: Mandatory conditions the host must meet.
- **`Performance Notes`**: Optional, non-binding performance optimization hints.

**3.1. Padding Contract Specification**
The `Padding Contract` field `Type` key accepts the following string literals:

| Type Token                | Definition                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| :------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CACHE`                   | Padding to align data to a hardware cache line boundary.                                                                                                                                                                                                                                                                                                                                                                                                                |
| `BANK_CONFLICT_AVOIDANCE` | Padding to the stride of a local memory array to prevent bank conflicts.                                                                                                                                                                                                                                                                                                                                                                                                |
| `SIMD`                    | Padding to align a dimension to the natural SIMD vector width.                                                                                                                                                                                                                                                                                                                                                                                                          |
| `NONE`                    | No independent padding strategy is applied to this buffer. The buffer's allocation dimensions may incorporate padding from `padded_*` dimension parameters, which are host-computed scalars satisfying the union of alignment constraints across all buffers sharing those dimensions. When `padded_*` parameters appear in the Tensor Shape, the padding is fully determined by those parameter values; no additional buffer-specific padding calculation is required. |

When a buffer has multiple independently padded dimensions, the `Padding Contract` field SHALL use a per-dimension dictionary format that specifies the padding `Type` for each dimension individually:

```
- Padding Contract: {
    dim[0] ("total_modules_count"): {Type: NONE},
    dim[1] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
    dim[2] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
  }
```

Each dimension entry uses the notation `dim[N] ("natural_name" → "padded_name")` for padded dimensions, or `dim[N] ("name")` for unpadded dimensions. The `Type` and `Formula` keys within each dimension entry follow the same vocabulary defined in the table above. This eliminates the "primary padding motivation" ambiguity while staying within the existing vocabulary. It is a natural evolution of the `padded_*` scalar convention that already carries the padding values — this extension carries the padding rationale per dimension.

Buffers with a single padded dimension (or no padding) continue to use the flat `{Type: X}` format.

**3.1.1. Initialization Contract Specification**

The `Initialization Contract` field declares whether the Host must pre-initialize a destination buffer before the producing kernel(s) are dispatched.

| Type Token               | Definition                                                                                                                                                                                                                         |
| :----------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ZERO_REQUIRED`          | Host must zero-fill the entire buffer before any kernel dispatch writes to it. The producing kernel uses partial-write or scatter-write patterns that leave positions unwritten; downstream consumers read the full buffer extent. |
| `ZERO_REQUIRED_ADDITIVE` | Host must zero-fill the buffer before the _first_ dispatch of a streaming series. The producing kernel adds to existing values on each dispatch; downstream consumers read only after the complete series.                         |
| `NONE`                   | No initialization required. The producing kernel(s) guarantee that all positions read by downstream consumers are written before consumption.                                                                                      |

**Default:** When a `dest_` buffer's commentary block omits the `Initialization Contract` field, the contract is implicitly `{Type: NONE}`. This field is not applicable to `src_` flow parameters (produced by prior pipeline stages), `update_buffer_GLOBAL_` parameters (persisted state), or `update_buffer_LOCAL_` parameters (transient work-group scratch).

**3.2. Partial Renderer Contract**

**3.2.1. Principle**

A kernel designated a "Partial Renderer" writes its output to a discrete, non-overlapping slice of a larger collection buffer. This is governed by a **`Placement Contract`**, which defines the precise strategy for calculating a write offset from a host-provided key.

**3.2.2. Specification**

The contract is specified within a parameter's commentary block using the `Placement Contract` key. The value of this key **shall** be a string literal adhering to a function-like grammar:

`strategy_name(key_parameter)`

- `strategy_name`: A canonical, lowercase identifier for the placement strategy, as defined in Article 3.2.3.
- `key_parameter`: The full, canonical name of the scalar parameter that serves as the unique placement key for the kernel invocation.

**3.2.3. Canonical Placement Strategies**

The following strategy names are exhaustive. Their use contractually binds the implementation to the specified calculation logic.

| `strategy_name`  | Definition                                                                        | Mandatory Key Parameter                  | Required Context Parameters            |
| :--------------- | :-------------------------------------------------------------------------------- | :--------------------------------------- | :------------------------------------- |
| `grid_mod_cls`   | Decomposes a 2D logical grid of (Module, Class) chunks from a flattened 1D index. | `src_scalar_NATURAL_flat_tile_index`     | `src_scalar_NATURAL_num_class_chunks`  |
| `linear_batch`   | Decomposes by linear chunking of the batch dimension.                             | `src_scalar_NATURAL_batch_chunk_index`   | None, beyond buffer/stride dimensions. |
| `linear_generic` | Decomposes by linear chunking of an arbitrary dimension.                          | An appropriate `..._chunk_index` scalar. | None, beyond buffer/stride dimensions. |

**3.3. Conditional Buffer Contract**

When a buffer parameter's access is gated by a FLAG scalar (the _controlling flag_), the parameter's commentary block SHALL include:

1. A `[CONDITIONAL]` annotation in the `@param` description line, identifying the controlling flag.
2. Validation Preconditions specifying: (a) the flag value under which the buffer is accessed, (b) the full allocation and bounds requirements when active, and (c) the permissible stub-buffer behavior when inactive (_"the Host MAY pass a minimal stub buffer"_).

The `[CONDITIONAL]` annotation is a documentation convention; it does not introduce a new flow prefix or memory scope. The buffer's `[Flow]` prefix (`src_` or `dest_`) reflects its role when active.

### **Article 4: The Kernel Contract Block**

**4.1. Mandate of Inclusion.** Every kernel interface specification **shall** begin with a `@kernel_contract` block. This block is mandatory and must precede the parameter list. Its purpose is to declare holistic constraints that apply to the kernel as a single unit.

**4.2. Formal Structure.** The block shall be a key-value list. The following keys are recognized:

| Key                         | Definition                                                                                 | Status        |
| :-------------------------- | :----------------------------------------------------------------------------------------- | :------------ |
| **`Holistic Constraints`**  | See note [1].                                                                              | **Mandatory** |
| **`Idempotency`**           | See note [2].                                                                              | **Mandatory** |
| **`Synchronization Model`** | Describes the kernel's role within the global DAG, using terms defined in **Article 4.4**. | Optional      |
| **`Behavioral Invariants`** | See note [3].                                                                              | Optional      |
| **`Precision Variant`**     | See note [4].                                                                              | Optional      |
| **`Kernel Bifurcation`**    | See note [5].                                                                              | Optional      |

**Notes:**

1. **Holistic Constraints**: This key is a tool of last resort, to be used only when a constraint truly has no logical owner in the parameter list. If none exist, this key **shall** contain the exact string: _"All constraints are defined by the parameter commentary blocks."_

2. **Idempotency**: Declares the kernel's precise deterministic and state-modifying behavior. It **shall** be one of the following string literals:
   - `Strictly Idempotent` — the kernel's output is solely a deterministic function of its inputs; re-execution with identical inputs and execution geometry overwrites the output with identical values; internal FP reductions do not disqualify, as the criterion is output-level determinism for fixed inputs and fixed dispatch geometry, not bit-exact reproducibility across geometries.
   - `Associatively Non-Idempotent` — the kernel accumulates results into a shared buffer whose final value depends on the composition of multiple dispatches, OR its output is a reduction whose result is input-order-sensitive in a composition context.
   - `Fundamentally Non-Idempotent (Stateful)` — the kernel modifies persistent model state in-place; re-execution corrupts the model.

3. **Behavioral Invariants**: Defines strict rules governing the kernel's internal implementation (e.g., "Forbidden from using `pown`"). The recognized values include:
   - `Precision Boundary Conversion` — required for any kernel that accesses buffers whose `precision_role` is `"storage"` or `"state"`. The kernel shall: (1) widen all non-compute-role inputs to `COMPUTE_TYPE` upon load, (2) perform all **transformative** arithmetic exclusively in `COMPUTE_TYPE`, and (3) narrow results from `COMPUTE_TYPE` to the destination buffer's role type upon store. **Accumulative operations** on state-role buffers may instead use the State-Precision Accumulation invariant, which performs accumulation in `max(COMPUTE_TYPE, STATE_TYPE)`. When all role types are equal, both invariants reduce to identity operations. Kernels that access only `"compute"`-role and integer buffers do not require this invariant.
   - `State-Precision Accumulation` — a mandatory invariant for stateful-update kernels performing accumulative operations on state-role buffers (EMA updates, running statistics). The kernel shall perform accumulative arithmetic in `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`, the kernel: (1) loads state values at full `STATE_TYPE` precision, (2) widens compute-role inputs (e.g., gradients) to `STATE_TYPE`, (3) performs accumulative arithmetic in `STATE_TYPE`, and (4) stores results at `STATE_TYPE`. When `STATE_TYPE ≤ COMPUTE_TYPE`, this invariant is equivalent to Precision Boundary Conversion — `ACCUM_TYPE = COMPUTE_TYPE` and all widening casts are identities. The invariant applies only to operations whose mathematical nature is accumulative — incremental updates that refine prior state. Transformative operations within the same kernel (e.g., bias correction division, final parameter update) may use `COMPUTE_TYPE`.
   - `Padding Zero-Establishment` — the kernel SHALL actively write zero to all positions at indices ≥ the logical extent, within its write footprint. The kernel is the **sole guarantor** that these positions contain zero. Applies when the destination buffer's `Initialization Contract` is `NONE` and downstream consumers read the full padded extent.
   - `Padding Zero-Preservation` — the kernel SHALL NOT write non-zero values to positions at indices ≥ the logical extent. The host initialization is the **primary guarantor**; the kernel's obligation is non-corruption. Applies when the destination buffer's `Initialization Contract` is `ZERO_REQUIRED` or `ZERO_REQUIRED_ADDITIVE`.

4. **Precision Variant**: Declares this kernel as a precision-typed variant of a named base kernel (ADR-026). Documents the specific buffer role divergence (storage-entry vs. compute-entry) and the selection criterion. The Orchestration tier selects the variant based on the source buffer's `precision_role`. When `STORAGE_TYPE == COMPUTE_TYPE`, both variants compile to identical machine code.

5. **Kernel Bifurcation**: Documents that this kernel is one half of a CONCEPT.md Principle 3(B) bifurcated pair. The value SHALL include: (1) an explicit reference to Principle 3(B), (2) identification of the peer kernel, and (3) a summary of the structural incompatibility (differing buffer types, shapes, or DAG edges) that prevents unification under a single interface with a FLAG parameter. This key is applicable only when the §7.0 Exception criteria are met.

**4.3. Padding Invariant Mandate.**

Every kernel whose output buffer has a padded dimension consumed by a downstream kernel at the full padded extent **SHALL** declare exactly one of: `Padding Zero-Establishment`, `Padding Zero-Preservation`, or `Padding Zero Propagation (Emergent)` with its preconditions in the `Behavioral Invariants` key. Omission is not equivalent to "no padding concern" — it is a contract violation if the Zero-Propagation Theorem's proof chain (CONCEPT.md §3.6) traverses that kernel.

**4.4. Canonical Behavioral Vocabulary.**
This section defines the canonical terms used to describe a kernel's behavior or its role in the system DAG, typically within the `Synchronization Model` key.

| Term                 | Definition                                                                                                                                                                                                                                  |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Streamable`         | A kernel that can operate on any independent chunk of a larger problem without requiring access to other chunks. This is the default, most parallelizable behavior.                                                                         |
| `Partial Renderer`   | A kernel that writes its output to a discrete, non-overlapping slice of a larger collection buffer, governed by a `Placement Contract`. The outputs require a subsequent aggregation step.                                                  |
| `Slice Renderer`     | A kernel that computes and writes a contiguous slice of a pre-allocated, monolithic destination buffer. The key distinction from a `Partial Renderer` is that its outputs do not require aggregation.                                       |
| `Global Barrier`     | A kernel that cannot execute until all of its inputs, potentially produced by many parallel upstream kernels, are fully available. It represents a major synchronization point in the DAG.                                                  |
| `Stateful`           | A kernel that modifies its inputs in-place or has internal state that persists across invocations (e.g., via atomics or buffer updates). This is a property of its `Idempotency`.                                                           |
| `Conditional Writer` | A kernel that writes to a destination buffer only when a data-dependent predicate (specified in Behavioral Invariants) is satisfied for the current work item. Positions not satisfying the predicate remain at their initialization state. |
| `Utility`            | A generic, reusable kernel that performs a common, low-level operation (e.g., copy, transpose, element-wise scaling) and is not specific to the core learning algorithm.                                                                    |

### **Article 5: Architectural Constants**

This article defines fixed, system-wide constants that are contractually binding on both host and device implementations. The device implementation shall enforce these values at compile-time.

- `LOCAL_MEM_BANK_PADDING`: Defined with a mandatory value of **`1`**.

`PrecisionConfig` is a frozen dataclass with three independent dtype fields — `storage_dtype`, `compute_dtype`, `state_dtype` — and five derived scalar constants: `storage_fp_format_max`, `storage_fp_min_positive`, `storage_mantissa_bits`, `compute_fp_format_max`, and `compute_epsilon`. The invariant `storage_dtype.itemsize ≤ compute_dtype.itemsize` and `storage_dtype.itemsize ≤ state_dtype.itemsize` is enforced by `__post_init__`. FP8 compute and FP8 state are architecturally prohibited and raise `ValueError` at construction time. Twelve factory classmethods are defined: `float32()` (all FP32), `mixed_f16_f32()` (FP16 storage, FP32 compute, FP32 state), `float64()` (all FP64), `mixed_f32_f64_state()` (FP32 storage, FP32 compute, FP64 state), `mixed_f16_f64_state()` (FP16 storage, FP32 compute, FP64 state), `mixed_f32_f64()` (FP32 storage, FP64 compute, FP64 state), `fp8_e4m3()` (E4M3 storage, FP32 compute, FP32 state), `fp8_e5m2()` (E5M2 storage, FP32 compute, FP32 state), `fp8_e4m3_f16()` (E4M3 storage, FP16 compute, FP32 state), `fp8_e5m2_f16()` (E5M2 storage, FP16 compute, FP32 state), `fp8_e4m3_f64()` (E4M3 storage, FP64 compute, FP64 state), `fp8_e5m2_f64()` (E5M2 storage, FP64 compute, FP64 state). Additional FP8 combinations (e.g., FP16 compute + FP64 state) are constructed directly. The state role has no ordering constraint relative to compute — `state_dtype.itemsize` may be greater than, equal to, or (when storage is narrower than state) less than `compute_dtype.itemsize`. The retired fields `numpy_dtype`, `fp_format_max`, and `epsilon` do not exist in this type. The uniform `float16()` factory (all FP16) was deleted in Phase 9A — FP16 state provides insufficient mantissa for EMA stability (10 bits vs. FP32's 23); for β₁ = 0.999, the per-step gradient contribution 0.001 × g rounds to zero for small gradients. Use `mixed_f16_f32()` instead. Direct construction with `state_dtype=np.float16` remains valid for experimental use.

### **Article 6: Mandatory Build-Time Symbols**

This article defines symbols that must be provided by the host build environment at compile time (e.g., via `-D` flags). Their values constitute the "hardware target profile" for a given compilation.

| Symbol                        | Type               | Meaning                                                                                        |
| :---------------------------- | :----------------- | :--------------------------------------------------------------------------------------------- |
| `STORAGE_TYPE`                | OpenCL/C type name | Element type for storage-role buffers. See note [1].                                           |
| `COMPUTE_TYPE`                | OpenCL/C type name | Element type for arithmetic and compute-role buffers                                           |
| `STATE_TYPE`                  | OpenCL/C type name | Element type for state-role buffers                                                            |
| `STORAGE_TYPE_IS_FP8`         | `int` (0 or 1)     | 1 when `STORAGE_TYPE` is an FP8 format (E4M3 or E5M2); gates FP8-specific load/store mechanics |
| `STORAGE_TYPE_IS_E4M3`        | `int` (0 or 1)     | 1 when `STORAGE_TYPE` is specifically E4M3; selects E4M3 conversion logic and lookup tables    |
| `STORAGE_TYPE_IS_E5M2`        | `int` (0 or 1)     | 1 when `STORAGE_TYPE` is specifically E5M2; selects E5M2 conversion logic and lookup tables    |
| `STORAGE_TYPE_IS_HALF`        | `int` (0 or 1)     | 1 when `STORAGE_TYPE == half`                                                                  |
| `STORAGE_TYPE_IS_FLOAT`       | `int` (0 or 1)     | 1 when `STORAGE_TYPE == float`                                                                 |
| `STORAGE_TYPE_IS_DOUBLE`      | `int` (0 or 1)     | 1 when `STORAGE_TYPE == double`                                                                |
| `COMPUTE_TYPE_IS_HALF`        | `int` (0 or 1)     | 1 when `COMPUTE_TYPE == half`                                                                  |
| `COMPUTE_TYPE_IS_FLOAT`       | `int` (0 or 1)     | 1 when `COMPUTE_TYPE == float`                                                                 |
| `COMPUTE_TYPE_IS_DOUBLE`      | `int` (0 or 1)     | 1 when `COMPUTE_TYPE == double`                                                                |
| `STATE_TYPE_IS_HALF`          | `int` (0 or 1)     | 1 when `STATE_TYPE == half`                                                                    |
| `STATE_TYPE_IS_FLOAT`         | `int` (0 or 1)     | 1 when `STATE_TYPE == float`                                                                   |
| `STATE_TYPE_IS_DOUBLE`        | `int` (0 or 1)     | 1 when `STATE_TYPE == double`                                                                  |
| `SIMD_WIDTH`                  | `int`              | Hardware SIMD lane count from `HardwareProfile`                                                |
| `C_TILE_SIZE`                 | `int`              | Column tile size for the module-chunking strategy                                              |
| `NUMERICAL_STABILITY_EPSILON` | float literal      | Epsilon for numerical stability guards; derived from `compute_epsilon`                         |

**Notes:**

1. **STORAGE_TYPE FP8 handling**: When `STORAGE_TYPE_IS_FP8 = 1`, this shall be `uchar` (unsigned 8-bit integer). FP8 values are represented as raw bytes; conversion to/from `COMPUTE_TYPE` is performed by the `load_storage()`/`store_storage()` functions in the kernel header, which dispatch to the software FP8 codec when `STORAGE_TYPE_IS_FP8` is set.

**Precision Flag Semantics.** The `_IS_*` flags form a complete, declarative type taxonomy for each precision role. Exactly one flag per role must be set to 1; all others must be 0. The kernel header enforces this at compile time via `#if` guards. This design makes the type taxonomy explicit and allows compile-time enforcement of `PrecisionConfig` invariants (e.g., `storage.itemsize ≤ compute.itemsize`).

**Kernel-Internal Derived Constants.** The following symbols are derived _within the kernel source_ (`kernels.cl.h`) from the primary precision-role symbols above. **The build system MUST NOT provide these via `-D` flags.** They are listed here for documentation completeness:

| Symbol                        | Type               | Meaning                                                                                                                                                                                        |
| :---------------------------- | :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ACCUM_TYPE`                  | OpenCL/C type name | Element type for accumulative operations in stateful-update kernels. Equals `max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`, this is `STATE_TYPE`; otherwise `COMPUTE_TYPE`. |
| `ACCUM_IS_WIDER_THAN_COMPUTE` | `int` (0 or 1)     | 1 when `STATE_TYPE > COMPUTE_TYPE`, indicating that accumulation uses the wider state type rather than compute type; otherwise 0.                                                              |

The symbols `SCALAR_TYPE` and `SCALAR_IS_HALF` are **retired**. They do not appear in any kernel source file; all kernel signatures use the three-role precision model (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`).

**Compile-time invariant enforcement.** The kernel header validates the following invariants via `#if` guards, producing `#error` on violation:

1. **Exactly-one selection**: For each role, exactly one `_IS_*` flag must be 1 (e.g., `STORAGE_TYPE_IS_FP8 + STORAGE_TYPE_IS_HALF + STORAGE_TYPE_IS_FLOAT + STORAGE_TYPE_IS_DOUBLE == 1`).
2. **Storage ≤ Compute**: FP64 storage requires FP64 compute; FP32 storage requires FP32 or FP64 compute.
3. **Storage ≤ State**: FP64 storage requires FP64 state; FP32 storage requires FP32 or FP64 state.

These guards provide defense-in-depth against build-system misconfigurations, complementing `PrecisionConfig.__post_init__` runtime validation.

**FP8 is storage-role only.** The symbols `COMPUTE_TYPE_IS_FP8` and `STATE_TYPE_IS_FP8` are **not defined** because FP8 compute and FP8 state are architecturally prohibited. The build system does not emit these symbols. Attempting to configure FP8 for compute or state roles raises `ValueError` at `PrecisionConfig` construction time.

**Cross-backend naming:** All backends (OpenCL, CPU, and Vulkan) use the canonical `STORAGE_TYPE_IS_*` / `COMPUTE_TYPE_IS_*` / `STATE_TYPE_IS_*` flag names from this article. The Vulkan backend's GLSL type macros (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`) and boolean flags (`STORAGE_TYPE_IS_FP8`, `STORAGE_TYPE_IS_E4M3`, `STORAGE_TYPE_IS_E5M2`) use the same canonical names as OpenCL, injected via `glslc -D`. No backend-specific shortened names are used.

**CPU backend FP8 implementation (Phase 9C):** The CPU backend represents FP8 values as C struct wrappers (`cpu_fp8_e4m3`, `cpu_fp8_e5m2`) containing a `uint8_t bits` field. Conversion between FP8 and compute types uses LUT-based decode (256-entry lookup tables from `cpu_fp8_lut.gen.h`) and algorithmic round-to-nearest-even encode (in `cpu_fp8.h`). The precision macro system uses `STORAGE_SUFFIX` token pasting: `STORAGE_SUFFIX=fp8e4m3` generates `scalar_load_real_fp8e4m3()` etc. FP16 compute variants (`s8e4c16x32`, `s8e4c16x64`, `s8e5c16x32`, `s8e5c16x64`) are conditionally compiled when `_Float16` is available, detected by Meson and exposed via `HAS_FLOAT16`.

### **Article 7: Canonical Interface Instantiation**

The following formal notation illustrates the sole valid method for specifying a kernel interface in adherence to this contract.

```
/**
* @brief Performs a tiled matrix transpose, demonstrating full contract compliance.
* @kernel_contract
*        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
*        - Idempotency: "Fundamentally Non-Idempotent (Stateful)"
*/
__kernel void illustrative_kernel_name(
    /**
    * @param src_buffer_GLOBAL_input_stream The primary data source for the computational unit.
    *        - Tensor Shape: (src_scalar_NATURAL_total_item_count)
    *        - Padding Contract: {Type: CACHE, Formula: "Post-pad to 128-byte alignment"}
    *        - Precision Role: "storage"
    *        - Calculability Proof: [src_scalar_NATURAL_total_item_count]
    *        - Validation Preconditions: [src_scalar_NATURAL_item_offset + src_scalar_NATURAL_item_count <= src_scalar_NATURAL_total_item_count]
    */
    __global const STORAGE_TYPE* src_buffer_GLOBAL_input_stream,

    /**
    * @param src_buffer_DEVICE_CONST_lookup_table A read-only, device-constant memory resource.
    *        - Tensor Shape: (LUT_CAPACITY)
    *        - Padding Contract: {Type: NONE}
    *        - Precision Role: "compute"
    *        - Calculability Proof: [Compile-time constant: LUT_CAPACITY]
    *        - Validation Preconditions: None.
    */
    __constant const COMPUTE_TYPE* src_buffer_DEVICE_CONST_lookup_table,

    /**
    * @param dest_buffer_GLOBAL_partial_results The sole collection resource for this unit's partial output.
    *        - Tensor Shape: (dest_scalar_NATURAL_total_chunks, RESULT_ELEMENTS_PER_CHUNK)
    *        - Padding Contract: {Type: NONE}
    *        - Initialization Contract: {Type: ZERO_REQUIRED}
    *        - Precision Role: "storage"
    *        - Calculability Proof: [dest_scalar_NATURAL_total_chunks, Compile-time constant: RESULT_ELEMENTS_PER_CHUNK]
    *        - Validation Preconditions: Host shall zero-initialize this buffer prior to dispatch.
    */
    __global STORAGE_TYPE* dest_buffer_GLOBAL_partial_results,

    /**
    * @param update_buffer_LOCAL_transpose_tile A work-group exclusive memory resource for a tiled matrix transpose.
    *        - Tensor Shape: (TILE_DIM, TILE_DIM + LOCAL_MEM_BANK_PADDING)
    *        - Padding Contract: {Type: BANK_CONFLICT_AVOIDANCE, Formula: "Pad row stride to (TILE_DIM + LOCAL_MEM_BANK_PADDING) elements"}
    *        - Precision Role: "compute"
    *        - Validation Preconditions: Host shall allocate size according to the formula derived from this contract, using the value of `LOCAL_MEM_BANK_PADDING` defined in System Contract Article 5.
    */
    __local COMPUTE_TYPE* update_buffer_LOCAL_transpose_tile,

    /**
    * @param sync_buffer_GLOBAL_atomic_counter A global resource for cross-group atomic synchronization. This buffer makes the kernel stateful.
    *        - Tensor Shape: (1)
    *        - Padding Contract: {Type: NONE}
    *        - Calculability Proof: [Implicit size: atomic_uint]
    *        - Validation Preconditions: Host shall initialize this resource to 0.
    */
    __global atomic_uint* sync_buffer_GLOBAL_atomic_counter,

    uint src_scalar_NATURAL_item_offset,
    uint src_scalar_NATURAL_item_count,
    uint src_scalar_NATURAL_total_item_count,
    float src_scalar_REAL_processing_threshold,
    uint dest_scalar_NATURAL_output_chunk_index,
    uint dest_scalar_NATURAL_total_chunks
);
```

### **Article 8: Canonical Lexicon for `[ContextAndUsage]`**

#### **1.0 Mandate**

This Lexicon establishes the sole binding definitions for the `[ContextAndUsage]` component. Usage of any term not defined herein is a violation.

#### **1.1 Canonical Abbreviations**

The following abbreviations are formally blessed as equivalent short-form representations of their corresponding Lexicon terms. They may be used interchangeably in `[ContextAndUsage]` components.

| Lexicon Term    | Abbreviation |
| :-------------- | :----------- |
| `temperatures`  | `temps`      |
| `probabilities` | `probs`      |
| `gradient`      | `grad`       |

#### **1.2 Pluralization Rule**

The Canonical Lexicon defines terms in their singular form. Plural forms of defined terms are implicitly valid when they appear as dimensional components — specifically when combined with `num_`, `total_`, `_count`, `_per_*`, or analogous cardinality modifiers. The plural is the natural morphological form for expressing "how many of this entity." Singular and plural forms carry no semantic distinction beyond grammatical number; they refer to the same Lexicon entry.

#### **2.0 Core Data Role Primitives**

#### **Group 1: Foundational Inputs & Ground Truth**

_These are the primary external data sources for a complete Act/Learn cycle._

| Term      | Definition                                                                    |
| :-------- | :---------------------------------------------------------------------------- |
| `input`   | The initial, untransformed data set for a complete computation. See note [1]. |
| `targets` | The ground truth labels for a supervised learning task.                       |

**Notes:**

1. **`input` dimensional usage**: When used as a dimensional qualifier (e.g., `padded_input_count`, `input_count`), denotes the feature dimensionality of the primary input data. This dimensional usage is distinct from the data-role usage, which denotes the data buffer itself.

#### **Group 2: Learnable Model Parameters**

_These are the stateful, learnable components of the model._

| Term           | Definition                                                     |
| :------------- | :------------------------------------------------------------- |
| `weights`      | The set of learnable weight parameters for a model layer.      |
| `biases`       | The set of learnable bias parameters for a model layer.        |
| `temperatures` | The set of learnable temperature parameters for logit scaling. |
| `parameters`   | A generic learnable parameter buffer (e.g., for optimizers).   |

#### **Group 3: Forward Pass Data Flow**

_These represent data as it is transformed during the `Act` (inference) phase._

| Term                 | Definition                                                                    |
| :------------------- | :---------------------------------------------------------------------------- |
| `hidden_activations` | The post-activation output tensor of an intermediate system layer.            |
| `logits`             | The pre-activation, real-valued output tensor of the final system layer.      |
| `probabilities`      | The post-activation, normalized probability tensor of the final system layer. |

#### **Group 4: Learning Process Artifacts**

_These are the primary data structures generated and consumed during the `Learn` phase._

| Term            | Definition                                                  |
| :-------------- | :---------------------------------------------------------- |
| `loss`          | The final computed loss value or tensor.                    |
| `gradient`      | The gradient tensor derived from a specified parameter.     |
| `normalization` | An L2 norm, typically of a gradient vector or its partials. |

#### **Group 5: Parallel Processing & Reduction Primitives**

_These terms define the core mechanics of the distributed and scalable computation model._

| Term                 | Definition                                                                                    |
| :------------------- | :-------------------------------------------------------------------------------------------- |
| `partial`            | A single, intermediate, un-aggregated result. Used as a noun.                                 |
| `partial_collection` | A generic memory pool containing multiple, potentially non-contiguous, `partial` results.     |
| `offset_list`        | An indirection table containing a list of memory offsets, used for scatter/gather operations. |

#### **Group 6: Control, Scoping & Dimensionality Primitives**

_These terms define the scope, validity, or dimension of other data structures._

| Term                   | Definition                                                                                                                                 |
| :--------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| `module`               | A parameter specific to a single classifier module (head).                                                                                 |
| `shared`               | A parameter that is shared across multiple modules or layers.                                                                              |
| `output_class`         | A dimension or count related to the output classes of a classifier.                                                                        |
| `sample_mask`          | A packed bitmask buffer encoding sample validity. See note [1].                                                                            |
| `hidden_mask`          | A derivative mask for the hidden layer activation function. See note [2].                                                                  |
| `effective_batch_size` | A scalar value representing the effective number of samples in a batch, computed by the host.                                              |
| `batch`                | The primary sample dimension of a training batch. Used as a dimensional qualifier in counts, offsets, and chunk decompositions.            |
| `hidden`               | The dimensionality of the intermediate (hidden) representation layer. Used as a dimensional qualifier in counts and padding parameters.    |
| `tile`                 | A discrete, indivisible unit of parallel work. See note [3].                                                                               |
| `height`               | The extent of a logical dimension, typically the slower-moving one (e.g., the number of rows).                                             |
| `width`                | The number of scalar elements that constitute a single logical row or a 1D vector. For a 2D entity, this represents the number of columns. |

**Notes:**

1. **`sample_mask` format**: A `uint*` buffer, 32 samples per word, LSB-first. Bit `1` = valid, `0` = invalid/padding. Accessed via `load_sample_mask(mask_words, sample_index)`. Host allocation shape: `((batch_size + 31) // 32,)` of `uint32` elements.

2. **`hidden_mask` strategies**: When the mask strategy is `explicit`, this buffer is produced by the forward pass kernel (Node 4) at compute precision and stored at storage precision, preserving derivative information that may be lost during activation storage narrowing. When the mask strategy is `recompute`, consuming kernels derive the mask from stored activations internally. The mask strategy is a Policy-tier decision based on the active `PrecisionConfig`.

3. **`tile` vs `flat_tile`**: Produced by the flattening of a logical problem grid. Distinguished from `flat_tile` (§3.0) which denotes the specific decomposition strategy; `tile` is the resulting work unit itself.

#### **Group 7: Utility Primitives**

_Generic terms for special cases._

| Term      | Definition                                                      |
| :-------- | :-------------------------------------------------------------- |
| `generic` | A type-punned buffer whose interpretation is context-dependent. |

#### **3.0 Decomposition Strategy Primitives**

| Term           | Definition                                                           |
| :------------- | :------------------------------------------------------------------- |
| `flat_tile`    | A unit of work from the flattening of a logical grid to a 1D index.  |
| `batch_chunk`  | A contiguous 1D partition of the primary batch dimension.            |
| `class_chunk`  | A contiguous 1D partition of the output class dimension.             |
| `module_chunk` | A contiguous 1D partition of the module (classifier head) dimension. |

#### **4.0 Context Modifiers**

| Type   | Term            | Function                                                                                                                                                                 |
| :----- | :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefix | `total_`        | Denotes the total logical cardinality of a dimension.                                                                                                                    |
| Prefix | `padded_`       | Denotes the physical, in-memory cardinality of a dimension.                                                                                                              |
| Prefix | `num_`          | Denotes the number of partitions a dimension has been decomposed into.                                                                                                   |
| Prefix | `partial_`      | Denotes an intermediate, un-aggregated result requiring further reduction.                                                                                               |
| Prefix | `leaf_`         | Denotes a raw, un-aggregated result at the entry point of a reduction process. It is the most granular form of a `partial_` result.                                      |
| Prefix | `clipped_`      | Denotes a buffer whose elements have undergone a norm-clipping transformation. This is a transitional state, typically applied to `partial_` gradients before reduction. |
| Prefix | `final_`        | Denotes a fully processed, normalized result ready for consumption by a final state-modifying kernel (e.g., optimizer).                                                  |
| Prefix | `in_`           | Pertaining to a source buffer.                                                                                                                                           |
| Prefix | `out_`          | Pertaining to a destination buffer.                                                                                                                                      |
| Prefix | `summed_`       | Denotes a buffer whose elements are the result of a batch-wide reduction (summation) of `partial_` or `clipped_` precursor elements.                                     |
| Suffix | `_index`        | A logical, ordinal position within a sequence or grid.                                                                                                                   |
| Suffix | `_offset`       | A physical displacement for direct memory address calculation.                                                                                                           |
| Suffix | `_count`        | The number of elements to process, relative to a corresponding `_offset`.                                                                                                |
| Suffix | `_global`       | Denotes a value that applies uniformly across an entire batch or dispatch.                                                                                               |
| Suffix | `_pow_t`        | A value representing a base raised to the power of the current time-step `t`.                                                                                            |
| Suffix | `_t_pre`        | A value that initial pre-process stage of a multi-stage process (e.g., processing the leaf in a reduction tree layer).                                                   |
| Suffix | `_t_j`          | A value that is dependent on the stage j of a multi-stage process (e.g., reduction tree layer).                                                                          |
| Suffix | `_per_item`     | Denotes a per-item parameterization of a scalar quantity, providing one value per logical work-item (tile) rather than a single global scalar.                           |
| Suffix | `_per_chunk`    | Denotes a per-chunk parameterization, providing one value per decomposition chunk rather than a single global scalar.                                                    |
| Suffix | `_per_stage`    | Denotes a per-stage parameterization of a scalar quantity, providing one value per reduction tree stage rather than a single global scalar.                              |
| Prefix | `intermediate_` | Denotes a buffer at a transitional reduction stage. See note [1].                                                                                                        |
| Prefix | `write_`        | Pertaining to the computed write position within a destination buffer. See note [2].                                                                                     |

**Notes:**

1. **`intermediate_` semantics**: Past `partial_` (raw/clipped) but before `summed_` (fully reduced). Applicable to buffers that are the output of one aggregation stage and the input to a subsequent clip or aggregation stage.

2. **`write_` vs `out_`**: Distinct from `out_` (which marks buffer affinity) in that `write_` qualifies an address offset calculated by the host for placement within a collection buffer.

#### **5.0 Domain and Utility Primitives**

| Domain               | Term                                         | Type           | Definition                                                                                                       |
| :------------------- | :------------------------------------------- | :------------- | :--------------------------------------------------------------------------------------------------------------- |
| Specialized Layout   | `_simd_major`                                | Suffix         | A Struct-of-Arrays (SoA) layout aligned to SIMD vector width.                                                    |
| Specialized Layout   | `_aos`                                       | Suffix         | An Array-of-Structs layout.                                                                                      |
| Specialized Layout   | `_soa`                                       | Suffix         | A Struct-of-Arrays layout.                                                                                       |
| Specialized Layout   | `_permuted`                                  | Suffix         | A buffer whose elements have undergone a non-trivial permutation.                                                |
| Specialized Layout   | `_flat`                                      | Suffix         | A logically multi-dimensional structure serialized into a contiguous 1D representation. See note [1].            |
| Local Memory Pattern | `simd_tile`                                  | Data Role      | A local memory tile used for SIMD optimization.                                                                  |
| Local Memory Pattern | `reduction_tile`                             | Data Role      | A local memory tile used for parallel reduction.                                                                 |
| Local Memory Pattern | `transpose_tile`                             | Data Role      | A local memory tile used for matrix transpose.                                                                   |
| Optimizer State      | `m1`, `m2`                                   | Data Role      | The first and second moment vectors.                                                                             |
| Generic Matrix Ops   | `stride`                                     | Suffix         | The physical displacement in elements between consecutive rows. See note [2].                                    |
| Optimizer State      | `learning_rate`, `beta1`, `beta2`, `epsilon` | Hyperparameter | Optimizer hyperparameters.                                                                                       |
| Constraint Value     | `min_value`                                  | Scalar Context | The inclusive minimum boundary for a value.                                                                      |
| Constraint Value     | `max_value`                                  | Scalar Context | The inclusive maximum boundary for a value.                                                                      |
| Constraint Value     | `fp_max`                                     | Scalar Context | The maximum finite value representable in the current floating-point format.                                     |
| Constraint Value     | `clipping_threshold`                         | Data Role      | The maximum permissible L2 norm for gradient clipping.                                                           |
| Policy Parameter     | `policy_t_algorithmic`                       | Hyperparameter | Time-dependent policy threshold from the adaptive gradient clipping schedule.                                    |
| Policy Parameter     | `policy_lambda`                              | Hyperparameter | Scaling coefficient for the gradient clipping policy.                                                            |
| Policy Parameter     | `policy_max_k`                               | Hyperparameter | Maximum fan-in K for staged reduction policy selection.                                                          |
| Reduction Engine     | `fan_in`                                     | Topology       | The number of input partials consumed by a single reduction node (the K in K-fan-in).                            |
| Reduction Engine     | `node`                                       | Topology       | An independent unit of work in the reduction tree; each node aggregates K partials into one output.              |
| Reduction Engine     | `stage`                                      | Topology       | A level in the multi-stage reduction tree; stage 0 is the leaf layer, higher stages consume prior stage outputs. |

**Notes:**

1. **`_flat` flattening convention**: For example, node × K offset pairs. The flattening convention (e.g., row-major) follows Article 1.3.

2. **`stride` full definition**: The physical displacement, in elements, required to move from the start of one row to the start of the next consecutive row.

#### **6.0 Canonical Flag Identifiers**

| Term                       | Definition                                                                                                  |
| :------------------------- | :---------------------------------------------------------------------------------------------------------- |
| `problem_type`             | Selects the primary loss calculation path (e.g., CCE vs. BCE).                                              |
| `operation_type`           | Selects the specific mathematical operation for a generic kernel (e.g., SUM vs. AVERAGE for an aggregator). |
| `use_per_item_norm`        | Selects the source for the gradient clipping threshold between a `_global` scalar and a `_per_item` buffer. |
| `produce_hidden_mask`      | Controls whether the forward pass kernel writes the ReLU derivative mask. See note [1].                     |
| `use_explicit_hidden_mask` | Selects the source for the ReLU derivative in consuming kernels. See note [2].                              |

**Notes:**

1. **`produce_hidden_mask`**: When 0, mask writes are skipped and the Host MAY pass a minimal stub buffer.

2. **`use_explicit_hidden_mask`**: When 0, the mask is derived internally from stored activations (`activation > 0`). When 1, the mask is read from an explicit `hidden_mask` buffer.

#### **7.0 Forbidden & Deprecated Terms**

The following terms are contractually forbidden and must be refactored if found in existing code.
| Term | Reason | Replacement |
| :--- | :--- | :--- |
| `param` | Too generic. | Use `parameters`, or a specific learnable (`weights`, `biases`).|
| `h` | Ambiguous abbreviation. | Use the full canonical term `hidden_activations`. |
| `elements` | Redundant with `_count`. | Standardize on the canonical `_count` suffix. |
| `_leading_dim` | Ambiguous library-specific term. | `stride` |
| `cce`/`bce` | Problem-specific type in name. | Use generic terms (`loss`, `targets`); type is handled by a `FLAG` param. |

**Exception: Architecturally-Mandated Kernel Bifurcation.**
When CONCEPT.md Principle 3(B) requires separate kernels due to incompatible type signatures, memory layouts, or downstream DAG topologies, those kernels may use otherwise-forbidden terms to distinguish the variant. This exception applies only when:

1. The kernel pair cannot share a unified interface (differing buffer types, shapes, or DAG edges).
2. The distinction is documented in the kernel's `@kernel_contract` block with an explicit reference to Principle 3(B).
3. No `FLAG` parameter could eliminate the interface divergence without producing a "smart kernel" with complex internal branching over incompatible memory patterns.

**Current applicants:** `compute_probs_loss_cce_chunk` (Node 6), `compute_probs_loss_bce_chunk` (Node 7).
