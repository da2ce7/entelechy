# System Contract: Host-Device Kernel Interface (Revision 10)

## Preamble

This document constitutes the definitive and inviolable contract governing all interactions across the Host-Device boundary. Its articles are fundamental laws of the system architecture. Adherence is mandatory and absolute. This contract supersedes all prior revisions and informal agreements.

---

## Article 1: Foundational Axioms

The architecture rests upon the following immutable axioms.

### 1.1. Axiom of Jurisdictional Separation

A parameter's syntactic structure (**Name**) defines its machine-enforced contract. A parameter's semantic block (**Commentary**) defines its human-verifiable and logical contract. These two jurisdictions are distinct and exhaustive.

### 1.2. Axiom of Semantic Uniqueness

Information encoded within the syntactic jurisdiction (Name) is prohibited from being duplicated within the semantic jurisdiction (Commentary), and vice versa. No redundancy shall exist between the two.

### 1.3. Axiom of Memory Layout

All multi-dimensional buffers are stored in **row-major memory layout**. The physical address of element `[i][j]` in a buffer of stride `S` is `i * S + j`. Any deviation from this layout must be explicitly signaled by a canonical layout suffix (e.g., `_soa`, `_simd_major`).

### 1.4. Axiom of Collaborative Interface Verifiability

A kernel's public interface constitutes a **closed logical system** for verification. Validation follows a layered responsibility model:

- **a. Host Proof Obligation.** The host's `KernelSignature` abstraction performs rigorous pre-flight validation of all buffer dimensions, scalar constraints, and placement strategies *before dispatch*.
- **b. Device Assurance Fallback.** Kernels may include verification parameters — even redundant ones — solely for final lightweight sanity checks. This defensive redundancy serves as a fail-safe against host-side validation failures, not as primary verification.

> *Illustration:* A kernel computing `dest_buffer[padded_width * row + col]` may receive both `padded_width` and `total_rows`, though `total_rows` could be derived from other values. The host pre-validates dimensional consistency via `Calculability Proofs`; the device verifies `row < total_rows` via the explicit parameter.

**1.4.1. Inviolable Constraint.** All parameters referenced in each `Calculability Proof` and `Validation Preconditions` entry must exist in the kernel interface, preserving the closed system. Redundancy for assurance does not violate minimalism when it serves this axiom.

---

## Article 2: Parameter Lexical Mandate

The structure of a parameter name is formally specified. This grammar is a mandatory requirement for interface validity.

### 2.1. Flow Prefix Semantics

Every buffer and scalar parameter begins with a **flow prefix** that declares the parameter's directional role in the kernel's data flow:

| Prefix | Semantics | Host Obligation |
| :----- | :-------- | :-------------- |
| `src_` | **Read-only input.** The kernel reads this parameter but never modifies it. | Provide a valid, fully-populated buffer or scalar. |
| `dest_` | **Write-only output.** The kernel writes to this parameter. It does not read prior contents unless its `Initialization Contract` prescribes host-provided initial values. | Provide a buffer of the declared size; the kernel is the sole producer. |
| `out_` | **Output-disposition input.** (Scalars only.) The kernel reads this scalar to configure a write operation on a `dest_` buffer. The kernel never writes to it. Applicable to scalars only. | Provide a valid value consistent with the corresponding `dest_` buffer's Placement, Initialization, or Conditional Buffer Contract. |
| `update_` | **Read-write mutation.** The kernel reads existing contents and modifies them in-place. Used for `LOCAL` scratch memory and stateful parameter updates (e.g., optimizer moment vectors). | Provide a buffer with valid prior contents (for `GLOBAL_`) or a correctly-sized allocation (for `LOCAL_`). |
| `sync_` | **Atomic synchronization.** The kernel uses atomic operations for cross-work-group coordination. Implies `update_` semantics with additional atomicity guarantees. | Initialize before dispatch (typically to zero). |

### 2.2. Buffer Name Grammar

A buffer identifier shall be constructed as:

`[Flow] :: "buffer" :: [MemoryScope] :: [ContextAndUsage]`

- **`[Flow]`**: `src_` \| `dest_` \| `update_` \| `sync_`
- **`[MemoryScope]`**: One of the tokens defined in §2.2.1.
- **`[ContextAndUsage]`**: A canonical identifier defined exclusively in **Article 8: Canonical Lexicon**.

#### 2.2.1. Memory Scope Token Definitions

| Token | Address Space | Definition |
| :---- | :------------ | :--------- |
| `GLOBAL_` | `__global` | Standard device memory for pipeline data. Buffers represent transient, per-dispatch data flowing through the computational DAG. May carry the `const` qualifier in the C declaration when used as a read-only source. |
| `LOCAL_` | `__local` | Work-group exclusive memory. Allocated per dispatch; lifetime does not extend beyond the kernel invocation. |
| `GLOBAL_CONST_` | `__global` | Read-only device memory holding host-provided parametric data that configures the computational transformation and is invariant for the duration of a kernel dispatch. Includes learnable model state, host-computed structural metadata (offset lists, indirection tables), and host-computed policy values (clipping thresholds). |
| `DEVICE_CONST_` | `__constant` | The hardware-specific, read-only constant address space. |

> **Precision Role and Buffer Names (Article 1.2).** A buffer's precision role is determined by its C type declaration (`STORAGE_TYPE`, `COMPUTE_TYPE`, or `STATE_TYPE`) and formally specified in the parameter's `Precision Role` commentary key (Article 3). The precision role is *not* encoded in the buffer name's `[ContextAndUsage]` component. Duplicating type information in the name would violate the Axiom of Semantic Uniqueness.

### 2.3. Scalar Name Grammar

A scalar identifier shall be constructed as:

`[Flow] :: "scalar" :: [NumberType] :: [ContextAndUsage]`

- **`[Flow]`**: `src_` \| `dest_` \| `out_`
- **`[NumberType]`**: A mandatory prefix defining the abstract numerical domain (§2.4).
- **`[ContextAndUsage]`**: A canonical identifier defined exclusively in **Article 8: Canonical Lexicon**.

> **Note on `out_` Flow.** The `out_` flow prefix is reserved for scalars that configure a `dest_` buffer's write geometry, activation predicate, or placement. Parameters with `out_` flow SHOULD carry `Validation Preconditions` identifying the associated `dest_` buffer and the cross-parameter constraint (e.g., `out_scalar_NATURAL_write_offset + element_count <= buffer_extent`).

### 2.4. Scalar `[NumberType]` Taxonomy

| Token | Abstract Domain | Core Constraint |
| :---- | :-------------- | :-------------- |
| `NATURAL_` | ℕ₀ = {0, 1, 2, …} | Non-negative. Defines counts, indices, and sizes. |
| `INTEGER_` | ℤ = {…, −1, 0, 1, …} | May be negative. Defines signed offsets and quantities. |
| `REAL_` | ℝ | Continuous quantity. Defines thresholds, hyperparameters, and data values. |
| `FLAG_` | {0, 1} | Boolean. Defines logical switches. |

---

## Article 3: Parameter Commentary Contract

The `@param` block constitutes the complete logical specification for a parameter.

### 3.1. Scope Dichotomy

The set of applicable commentary keys depends on the parameter's memory scope.

**Global-scope parameters** (`GLOBAL_`, `GLOBAL_CONST_`, `DEVICE_CONST_`) use the full key set defined in §3.2.

**Local-scope parameters** (`LOCAL_`) use **only** the following restricted key set:

| Key | Status | Purpose |
| :-- | :----- | :------ |
| `Allocation Formula` | **Mandatory** | A constructive arithmetic expression yielding the required allocation in **bytes**. All terms must be kernel interface parameters, compile-time symbols (Article 6), or OpenCL work-group query functions (e.g., `get_local_size(0)`). This formula is the **complete** host obligation for the local buffer: the host computes the value, allocates that many bytes, and passes the pointer. |
| `Precision Role` | **Mandatory** | Documents the element type used in the formula's `sizeof()` term. |
| `Internal Layout Note` | Optional | Free-text description of the kernel's internal access patterns (sub-array partitioning, bank-conflict avoidance stride padding, reduction tree structure). **Purely informational** — imposes no obligation on the host and is not subject to machine validation. |

The keys `Tensor Shape`, `Padding Contract`, `Initialization Contract`, `Placement Contract`, and `Calculability Proof` are **inapplicable** to local-scope parameters; their presence on a `LOCAL_` parameter is a contract violation. This restriction reflects the jurisdictional boundary: the host's sole obligation for local memory is to allocate the correct number of bytes. Internal structure is exclusively an Execution-tier concern (CONCEPT.md §5).

**Scalar parameters** may carry the keys `Calculability Proof`, `Validation Preconditions`, and `Performance Notes`. The buffer-specific keys (`Tensor Shape`, `Padding Contract`, `Initialization Contract`, `Placement Contract`, `Precision Role`) are inapplicable to scalars.

### 3.2. Commentary Key Definitions

The following keys are recognized for global-scope buffer parameters:

| Key | Status | Definition |
| :-- | :----- | :--------- |
| **`Tensor Shape`** | Mandatory | The logical dimensions of the tensor, expressed as a parenthesized tuple of scalar parameters or compile-time constants. |
| **`Padding Contract`** | Mandatory | The padding strategy applied to the buffer's dimensions (§3.3). |
| **`Precision Role`** | Mandatory (FP buffers) | The precision-role dtype governing this buffer's element type and allocation size (§3.2.1). Integer-typed buffers (`int`, `uint`, `atomic_uint`) are exempt. |
| **`Calculability Proof`** | Mandatory | A constructive expression deriving the buffer's dimensions from interface parameters, satisfying Axiom 1.4. |
| **`Initialization Contract`** | `dest_` only | Pre-initialization requirements (§3.4). Omission is equivalent to `{Type: NONE}`. |
| **`Placement Contract`** | `dest_` only | The placement strategy for Partial Renderers (§3.5). |
| **`Validation Preconditions`** | Recommended | Conditions the host must satisfy before dispatch. |
| **`Performance Notes`** | Optional | Non-binding optimization hints. |

#### 3.2.1. Precision Role Specification

The `Precision Role` key declares which precision-role dtype from the active `PrecisionConfig` governs the buffer's element type and allocation size:

| Role Value | C Type Symbol | Governs |
| :--------- | :------------ | :------ |
| `"storage"` | `STORAGE_TYPE` | Bandwidth-optimized buffers (activations, stored gradients). |
| `"compute"` | `COMPUTE_TYPE` | Arithmetic-fidelity buffers (accumulators, intermediate results). |
| `"state"` | `STATE_TYPE` | Long-term stability buffers (optimizer moments, parameters). |
| `"flag-conditional"` | Varies | Element type depends on a FLAG scalar's runtime value. |

**Flag-conditional buffers.** When the role is `"flag-conditional"`, the commentary block MUST enumerate each flag value and its corresponding interpretation (role + type). Host-side validation operates on the runtime flag value.

**Padding byte calculations.** Under the three-role precision model, padding byte counts use the element size of the buffer's *actual* precision-role type: `sizeof(STORAGE_TYPE)`, `sizeof(COMPUTE_TYPE)`, or `sizeof(STATE_TYPE)` as appropriate. Using a different role's `sizeof` in a `Padding Contract` expression is a contract violation.

### 3.3. Padding Contract Specification

The `Padding Contract` field (global-scope buffers only) specifies the padding strategy applied to each buffer dimension.

#### 3.3.1. Padding Type Vocabulary

| Type Token | Definition |
| :--------- | :--------- |
| `CACHE` | Padding to align data to a hardware cache line boundary. |
| `SIMD` | Padding to align a dimension to the natural SIMD vector width. |
| `NONE` | No independent padding strategy is applied to this buffer. When `padded_*` dimension parameters appear in the `Tensor Shape`, the padding is fully determined by those parameter values; no additional buffer-specific padding calculation is required. |

#### 3.3.2. Single-Dimension Format

Buffers with a single padded dimension (or no padding) use the flat format:

```
- Padding Contract: {Type: CACHE, Formula: "128-byte alignment"}
```

#### 3.3.3. Per-Dimension Format

When a buffer has multiple independently padded dimensions, the `Padding Contract` field uses a per-dimension dictionary:

```
- Padding Contract: {
    dim[0] ("total_modules_count"): {Type: NONE},
    dim[1] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
    dim[2] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
  }
```

Each dimension entry uses the notation `dim[N] ("natural_name" → "padded_name")` for padded dimensions, or `dim[N] ("name")` for unpadded dimensions. The `Type` and `Formula` keys within each entry follow the vocabulary of §3.3.1. This format is a natural extension of the `padded_*` scalar convention that already carries the padding values — it additionally carries the padding *rationale* per dimension.

### 3.4. Initialization Contract Specification

The `Initialization Contract` field (global-scope `dest_` buffers only) declares whether the host must pre-initialize the buffer before the producing kernel(s) are dispatched.

| Type Token | Definition |
| :--------- | :--------- |
| `ZERO_REQUIRED` | Host must zero-fill the entire buffer before any kernel dispatch writes to it. The producing kernel uses partial-write or scatter-write patterns that leave positions unwritten; downstream consumers read the full buffer extent. |
| `ZERO_REQUIRED_ADDITIVE` | Host must zero-fill the buffer before the *first* dispatch of a streaming series. The producing kernel adds to existing values on each dispatch; downstream consumers read only after the complete series. |
| `NONE` | No initialization required. The producing kernel(s) guarantee that all positions read by downstream consumers are written before consumption. |

**Default.** When a `dest_` buffer's commentary block omits the `Initialization Contract` field, the contract is implicitly `{Type: NONE}`.

**Applicability.** This field is not applicable to `src_` flow parameters (produced by prior pipeline stages), `update_` flow parameters (whose prior contents are meaningful), or `sync_` flow parameters (which carry their own initialization requirements in `Validation Preconditions`).

> **Guidance — Sparse-Write Partial Renderers.** When a Partial Renderer writes to a destination buffer whose allocated extent per tile is wider than the tile's logical write footprint (e.g., when the buffer is deliberately over-allocated to enable contiguous downstream access), `{Type: NONE}` is inappropriate unless the kernel actively fills all unwritten positions. Use `{Type: ZERO_REQUIRED}` with `Padding Zero-Preservation` to delegate unwritten-position responsibility to the host. See Node 8's class-chunk amplification pattern for the canonical example.

### 3.5. Partial Renderer Contract

#### 3.5.1. Principle

A kernel designated a "Partial Renderer" writes its output to a discrete, non-overlapping slice of a larger collection buffer, governed by a **Placement Contract** that defines the strategy for calculating a write offset from a host-provided key. The Placement Contract is applicable only to global-scope destination buffers.

#### 3.5.2. Grammar

The contract is specified via the `Placement Contract` commentary key. The value shall be a string literal:

`strategy_name(key_parameter)`

- `strategy_name`: A canonical, lowercase strategy identifier from §3.5.3.
- `key_parameter`: The full canonical name of the scalar parameter serving as the unique placement key.

#### 3.5.3. Canonical Placement Strategies

| Strategy | Definition | Mandatory Key Parameter(s) | Required Context Parameters |
| :------- | :--------- | :------------------------- | :-------------------------- |
| `grid_mod_cls` | Decomposes a 2D (Module, Class) grid from a flattened 1D index. | `src_scalar_NATURAL_flat_tile_index` | `src_scalar_NATURAL_num_class_chunks` |
| `grid_mod_cls_batch` | Decomposes a 3D (Module, Class, Batch) grid from a flat tile index and a batch chunk index. | `src_scalar_NATURAL_flat_tile_index`, `src_scalar_NATURAL_batch_chunk_index` | `src_scalar_NATURAL_num_class_chunks`, `src_scalar_NATURAL_num_batch_chunks` |
| `linear_batch` | Decomposes by linear chunking of the batch dimension. | `src_scalar_NATURAL_batch_chunk_index` | None beyond buffer/stride dimensions. |
| `linear_generic` | Decomposes by linear chunking of an arbitrary dimension. | An appropriate `..._chunk_index` scalar. | None beyond buffer/stride dimensions. |

### 3.6. Conditional Buffer Contract

When a buffer parameter's access is gated by a FLAG scalar (the *controlling flag*), the parameter's commentary block SHALL include:

1. A `[CONDITIONAL]` annotation in the `@param` description line, identifying the controlling flag.
2. `Validation Preconditions` specifying: (a) the flag value under which the buffer is accessed, (b) the full allocation and bounds requirements when active, and (c) the permissible stub-buffer behavior when inactive (*"the Host MAY pass a minimal stub buffer"*).

The `[CONDITIONAL]` annotation is a documentation convention; it does not introduce a new flow prefix or memory scope. The buffer's `[Flow]` prefix reflects its role when active. This contract applies exclusively to global-scope buffer parameters.

---

## Article 4: The Kernel Contract Block

### 4.1. Mandate of Inclusion

Every kernel interface specification **shall** begin with a `@kernel_contract` block. This block is mandatory, must precede the parameter list, and declares holistic constraints that apply to the kernel as a single unit.

### 4.2. Formal Structure

The block shall be a key-value list. The following keys are recognized:

| Key | Status | Definition |
| :-- | :----- | :--------- |
| **`Holistic Constraints`** | **Mandatory** | Constraints with no logical owner among the individual parameters. If none exist, the value **shall** be the exact string: *"All constraints are defined by the parameter commentary blocks."* |
| **`Idempotency`** | **Mandatory** | The kernel's deterministic and state-modifying behavior (§4.2.1). |
| **`Synchronization Model`** | Optional | The kernel's role within the global DAG, using terms from §4.4. |
| **`Behavioral Invariants`** | Optional | Strict rules governing internal implementation (§4.2.2). |
| **`Precision Variant`** | Optional | Declares this kernel as a precision-typed variant of a named base kernel (ADR-026). Documents the buffer role divergence and the Orchestration-tier selection criterion. |
| **`Kernel Bifurcation`** | Optional | Documents that this kernel is one half of a CONCEPT.md Principle 3(B) bifurcated pair (§4.2.3). |

#### 4.2.1. Idempotency Values

The `Idempotency` key shall be one of the following string literals:

| Value | Semantics |
| :---- | :-------- |
| `Strictly Idempotent` | Output is solely a deterministic function of inputs; re-execution with identical inputs and execution geometry overwrites output with identical values. Internal FP reductions do not disqualify — the criterion is output-level determinism for fixed inputs and fixed dispatch geometry, not bit-exact reproducibility across geometries. |
| `Associatively Non-Idempotent` | The kernel accumulates results into a shared buffer whose final value depends on the composition of multiple dispatches, OR its output is a reduction whose result is input-order-sensitive in a composition context. |
| `Fundamentally Non-Idempotent (Stateful)` | The kernel modifies persistent model state in-place; re-execution corrupts the model. |

#### 4.2.2. Behavioral Invariant Vocabulary

The following named invariants are recognized in the `Behavioral Invariants` key:

| Invariant | Applicability | Definition |
| :-------- | :------------ | :--------- |
| `Precision Boundary Conversion` | Kernels accessing buffers whose `precision_role` is `"storage"` or `"state"`. | The kernel shall: (1) widen all non-compute-role inputs to `COMPUTE_TYPE` upon load, (2) perform all **transformative** arithmetic exclusively in `COMPUTE_TYPE`, and (3) narrow results from `COMPUTE_TYPE` to the destination buffer's role type upon store. When all role types are equal, all conversions are identity operations eliminated by the compiler. Kernels accessing only `"compute"`-role and integer buffers do not require this invariant. |
| `State-Precision Accumulation` | Stateful-update kernels performing **accumulative** operations on state-role buffers (EMA updates, running statistics). | Accumulative arithmetic shall use `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`. When `STATE_TYPE > COMPUTE_TYPE`: (1) load state values at full `STATE_TYPE` precision, (2) widen compute-role inputs to `STATE_TYPE`, (3) perform accumulative arithmetic in `STATE_TYPE`, (4) store results at `STATE_TYPE`. When `STATE_TYPE ≤ COMPUTE_TYPE`, this invariant is equivalent to `Precision Boundary Conversion`. Applies **only** to mathematically accumulative operations — transformative operations within the same kernel (e.g., bias correction, final parameter update) use `COMPUTE_TYPE`. |
| `Padding Zero-Establishment` | Kernels whose destination buffer has `Initialization Contract: NONE` and whose downstream consumers read the full padded extent. | The kernel SHALL actively write zero to all positions at indices ≥ the logical extent within its write footprint. The kernel is the **sole guarantor** of zeros at these positions. |
| `Padding Zero-Preservation` | Kernels whose destination buffer has `Initialization Contract: ZERO_REQUIRED` or `ZERO_REQUIRED_ADDITIVE`. | The kernel SHALL NOT write non-zero values to positions at indices ≥ the logical extent. The host initialization is the **primary guarantor**; the kernel's obligation is non-corruption. |
| `Padding Zero Propagation (Emergent)` | Kernels where zero padding is a mathematical consequence of upstream invariants, not an active zeroing step. | When declared, its preconditions must be stated explicitly. The kernel is not required to distinguish padding from logical positions — the output carries zero at padding positions by mathematical necessity. |

#### 4.2.3. Kernel Bifurcation

The `Kernel Bifurcation` key documents membership in a CONCEPT.md Principle 3(B) bifurcated pair. The value SHALL include:

1. An explicit reference to Principle 3(B).
2. Identification of the peer kernel.
3. A summary of the structural incompatibility (differing buffer types, shapes, or DAG edges) that prevents unification under a single interface with a FLAG parameter.

This key is applicable only when no `FLAG` parameter could eliminate the interface divergence without producing a "smart kernel" with complex internal branching over incompatible memory patterns.

### 4.3. Padding Invariant Mandate

Every kernel whose output buffer has a padded dimension consumed by a downstream kernel at the full padded extent **SHALL** declare exactly one of `Padding Zero-Establishment`, `Padding Zero-Preservation`, or `Padding Zero Propagation (Emergent)` with its preconditions in the `Behavioral Invariants` key. Omission is a contract violation if the Zero-Propagation Theorem's proof chain (CONCEPT.md §3.6) traverses that kernel.

### 4.4. Canonical Behavioral Vocabulary

This section defines terms used to describe a kernel's role in the system DAG, typically within the `Synchronization Model` key.

| Term | Definition |
| :--- | :--------- |
| `Streamable` | Operates on any independent chunk of a larger problem without requiring access to other chunks. The default, most parallelizable behavior. |
| `Partial Renderer` | Writes output to a discrete, non-overlapping slice of a larger collection buffer, governed by a `Placement Contract`. Outputs require subsequent aggregation. |
| `Slice Renderer` | Writes a contiguous slice of a pre-allocated, monolithic destination buffer. Outputs do *not* require aggregation. |
| `Global Barrier` | Cannot execute until all inputs — potentially produced by many parallel upstream kernels — are fully available. A major synchronization point. |
| `Stateful` | Modifies inputs in-place or maintains internal state that persists across invocations. A property of its `Idempotency`. |
| `Conditional Writer` | Writes to a destination buffer only when a data-dependent predicate (specified in `Behavioral Invariants`) is satisfied. Positions not satisfying the predicate remain at their initialization state. |
| `Utility` | A generic, reusable kernel performing a common operation (element-wise scaling, transpose, reduction) not specific to the core learning algorithm. |

---

## Article 5: System Configuration

### 5.1. Architectural Constants

System-wide constants shared across the host-device boundary are registered in this section.

*No system-wide architectural constants are currently defined.* `LOCAL_MEM_BANK_PADDING` was retired — it is a kernel-internal constant defined in `kernels.cl.h` (value: `1`), a consequence of the bank-conflict avoidance technique, not a hardware-dependent parameter. Future constants requiring host-device agreement are registered here.

### 5.2. The Precision Configuration Model

Numeric precision is decomposed into three independent roles — **storage**, **compute**, and **state** — reflecting the distinct optimization targets of bandwidth, arithmetic fidelity, and long-term stability (CONCEPT.md §2). This decomposition is reified as `PrecisionConfig`, a frozen dataclass.

#### 5.2.1. Primary Fields

| Field | Type | Role |
| :---- | :--- | :--- |
| `storage_dtype` | NumPy dtype | Element type for bandwidth-optimized buffers. Narrowing reduces buffer sizes and transfer bandwidth proportionally. |
| `compute_dtype` | NumPy dtype | Element type for arithmetic operations and intermediate accumulators. |
| `state_dtype` | NumPy dtype | Element type for persistent optimizer state (moment vectors, learnable parameters). |

A configuration where all three roles share a type (e.g., `PrecisionConfig.float32()`) is a parameterization — not a distinct mode.

#### 5.2.2. Derived Scalar Constants

These fields are computed from the primary fields and are read-only:

| Field | Source | Purpose |
| :---- | :----- | :------ |
| `storage_fp_format_max` | `storage_dtype` | Maximum finite value representable in the storage format. |
| `storage_fp_min_positive` | `storage_dtype` | Minimum positive subnormal in the storage format. Used for quantization floor analysis. |
| `storage_mantissa_bits` | `storage_dtype` | Mantissa bit count in the storage format. |
| `compute_fp_format_max` | `compute_dtype` | Maximum finite value in the compute format. Used for safety ceiling calculations. |
| `compute_epsilon` | `compute_dtype` | Machine epsilon for the compute format. Used for numerical stability guards. |

#### 5.2.3. Mask Strategy

`PrecisionConfig` derives a `mask_strategy` field — one of `"explicit"` or `"recompute"` — from whether `storage_dtype == compute_dtype`:

| Condition | Strategy | Behavior |
| :-------- | :------- | :------- |
| `storage_dtype != compute_dtype` | `explicit` | Node 4 writes the `hidden_mask` buffer capturing compute-precision derivative truth before activation storage narrowing. Consuming kernels read the mask directly. |
| `storage_dtype == compute_dtype` | `recompute` | The mask is bit-identical to `activation > 0`. No mask buffer is allocated; consumers derive the mask internally. |

The mask strategy is orthogonal to the activation Cache/Recompute strategy; all four combinations produce correct results (see ADR-031 §1.4).

#### 5.2.4. Construction Invariants

The following invariants are enforced by `__post_init__`:

1. **Storage ≤ Compute:** `storage_dtype.itemsize ≤ compute_dtype.itemsize`.
2. **Storage ≤ State:** `storage_dtype.itemsize ≤ state_dtype.itemsize`.
3. **FP8 is storage-only:** Attempting to set `compute_dtype` or `state_dtype` to an FP8 format raises `ValueError`. There is no user-discipline escape hatch for non-functional training.

The state role has **no ordering constraint** relative to compute — `state_dtype.itemsize` may be greater than, equal to, or less than `compute_dtype.itemsize`.

#### 5.2.5. Factory Methods

| Factory | Storage | Compute | State | Primary Use Case |
| :------ | :------ | :------ | :---- | :--------------- |
| `float32()` | FP32 | FP32 | FP32 | Default and reference configuration. |
| `float64()` | FP64 | FP64 | FP64 | Validation reference; full FP64 uniformity. |
| `mixed_f16_f32()` | FP16 | FP32 | FP32 | Production mixed-precision. |
| `mixed_f32_f64()` | FP32 | FP64 | FP64 | High-fidelity scientific computation. |
| `mixed_f32_f64_state()` | FP32 | FP32 | FP64 | Extended-stability training (FP64 moments only). |
| `mixed_f16_f64_state()` | FP16 | FP32 | FP64 | Bandwidth efficiency + extended-stability moments. |
| `fp8_e4m3()` | E4M3 | FP32 | FP32 | Maximum bandwidth compression (4× vs. FP32). |
| `fp8_e5m2()` | E5M2 | FP32 | FP32 | Maximum dynamic range in FP8 storage. |
| `fp8_e4m3_f16()` | E4M3 | FP16 | FP32 | FP8 storage with FP16 compute throughput. |
| `fp8_e5m2_f16()` | E5M2 | FP16 | FP32 | FP8 E5M2 storage with FP16 compute throughput. |
| `fp8_e4m3_f64()` | E4M3 | FP64 | FP64 | FP8 storage with full FP64 arithmetic fidelity. |
| `fp8_e5m2_f64()` | E5M2 | FP64 | FP64 | FP8 E5M2 storage with full FP64 arithmetic fidelity. |

Additional combinations (e.g., FP16 compute + FP64 state) are constructed directly via the `PrecisionConfig` constructor.

#### 5.2.6. FP8 Format Reference

| Variant | Exponent Bits | Mantissa Bits | Max Finite | Min Subnormal | Special Values |
| :------ | :------------ | :------------ | :--------- | :------------ | :------------- |
| E4M3 (`float8_e4m3fn`) | 4 | 3 | 448 | 2⁻⁹ ≈ 0.00195 | No infinity; 2 NaN bit patterns (0x7F, 0xFF). 254 finite values. |
| E5M2 | 5 | 2 | 57344 | 2⁻¹⁶ ≈ 1.53×10⁻⁵ | IEEE-like: exponent 0x1F encodes ±∞ (mantissa=0) and NaN (mantissa≠0). 248 finite values. |

The store path saturates to the maximum finite value in both formats; infinity and NaN bit patterns are never written to storage buffers.

#### 5.2.7. Deleted and Retired Items

**Deleted factory.** The uniform `float16()` factory (all FP16) was removed — FP16's 10-bit mantissa provides insufficient precision for EMA stability. For β₁ = 0.999, the per-step gradient contribution `0.001 × g` rounds to zero for small gradients. Use `mixed_f16_f32()` instead. Direct construction with `state_dtype=np.float16` continues to be supported for expert use.

**Retired fields.** The fields `numpy_dtype`, `fp_format_max`, and `epsilon` do not exist in this type. They are superseded by the three-role field model.

---

## Article 6: Mandatory Build-Time Symbols

Symbols that must be provided by the host build environment at compile time (e.g., via `-D` flags). Their values constitute the "hardware target profile" for a given compilation.

### 6.1. Primary Symbol Table

| Symbol | Type | Meaning |
| :----- | :--- | :------ |
| `STORAGE_TYPE` | OpenCL/C type name | Element type for storage-role buffers. When FP8 is active, this shall be `uchar` (§6.5). |
| `COMPUTE_TYPE` | OpenCL/C type name | Element type for arithmetic and compute-role buffers. |
| `STATE_TYPE` | OpenCL/C type name | Element type for state-role buffers. |
| `STORAGE_TYPE_IS_FP8` | `int` (0\|1) | 1 when `STORAGE_TYPE` is an FP8 format; gates FP8-specific load/store mechanics. |
| `STORAGE_TYPE_IS_E4M3` | `int` (0\|1) | 1 when `STORAGE_TYPE` is specifically E4M3; selects E4M3 conversion logic and lookup tables. |
| `STORAGE_TYPE_IS_E5M2` | `int` (0\|1) | 1 when `STORAGE_TYPE` is specifically E5M2; selects E5M2 conversion logic and lookup tables. |
| `STORAGE_TYPE_IS_HALF` | `int` (0\|1) | 1 when `STORAGE_TYPE == half`. |
| `STORAGE_TYPE_IS_FLOAT` | `int` (0\|1) | 1 when `STORAGE_TYPE == float`. |
| `STORAGE_TYPE_IS_DOUBLE` | `int` (0\|1) | 1 when `STORAGE_TYPE == double`. |
| `COMPUTE_TYPE_IS_HALF` | `int` (0\|1) | 1 when `COMPUTE_TYPE == half`. |
| `COMPUTE_TYPE_IS_FLOAT` | `int` (0\|1) | 1 when `COMPUTE_TYPE == float`. |
| `COMPUTE_TYPE_IS_DOUBLE` | `int` (0\|1) | 1 when `COMPUTE_TYPE == double`. |
| `STATE_TYPE_IS_HALF` | `int` (0\|1) | 1 when `STATE_TYPE == half`. |
| `STATE_TYPE_IS_FLOAT` | `int` (0\|1) | 1 when `STATE_TYPE == float`. |
| `STATE_TYPE_IS_DOUBLE` | `int` (0\|1) | 1 when `STATE_TYPE == double`. |
| `SIMD_WIDTH` | `int` | Hardware SIMD lane count from `HardwareProfile`. |
| `C_TILE_SIZE` | `int` | Column tile size for the module-chunking strategy. |
| `NUMERICAL_STABILITY_EPSILON` | float literal | Epsilon for numerical stability guards; derived from `compute_epsilon`. |

### 6.2. Optional Build-Time Symbols

| Symbol | Type | Default | Meaning |
| :----- | :--- | :------ | :------ |
| `USE_FAST_MATH` | `int` (0\|1) | 0 | When 1 *and* `COMPUTE_TYPE_IS_FLOAT`, selects `native_*` intrinsics for `exp`, `log`, `sqrt`. No effect when `COMPUTE_TYPE` is `half` or `double` (no `native_*` variants exist in those precisions). |

### 6.3. Precision Flag Semantics

The `_IS_*` flags form a complete, declarative type taxonomy for each precision role. **Exactly one flag per role must be set to 1; all others must be 0.** The kernel header enforces this at compile time via `#if` guards.

**FP8 mutual exclusivity.** `STORAGE_TYPE_IS_E4M3` and `STORAGE_TYPE_IS_E5M2` are mutually exclusive. When `STORAGE_TYPE_IS_FP8 = 1`, exactly one of the two must be set. When `STORAGE_TYPE_IS_FP8 = 0`, both must be 0.

**FP8 is storage-role only.** The symbols `COMPUTE_TYPE_IS_FP8` and `STATE_TYPE_IS_FP8` are **not defined** because FP8 compute and FP8 state are architecturally prohibited. The build system does not emit these symbols.

### 6.4. Compile-Time Invariant Enforcement

The kernel header validates the following invariants via `#if` guards, producing `#error` on violation:

1. **Exactly-one selection.** For each role, the sum of `_IS_*` flags must equal 1.
2. **Storage ≤ Compute.** FP64 storage requires FP64 compute; FP32 storage requires ≥FP32 compute.
3. **Storage ≤ State.** FP64 storage requires FP64 state; FP32 storage requires ≥FP32 state.

These guards provide defense-in-depth against build-system misconfigurations, complementing `PrecisionConfig.__post_init__` runtime validation.

### 6.5. FP8 Storage Implementation

When `STORAGE_TYPE_IS_FP8 = 1`, `STORAGE_TYPE` shall be `uchar` (unsigned 8-bit integer). FP8 values are represented as raw bytes; conversion to/from `COMPUTE_TYPE` is performed by the `load_storage()` / `store_storage()` functions in the kernel header, which dispatch to the software FP8 codec (LUT-based decode, algorithmic round-to-nearest-even encode) when `STORAGE_TYPE_IS_FP8` is set.

### 6.6. Kernel-Internal Derived Constants

The following symbols are derived *within the kernel source* (`kernels.cl.h`) from the primary symbols above. **The build system MUST NOT provide these via `-D` flags.** They are documented here for reference completeness.

| Symbol | Type | Derivation |
| :----- | :--- | :--------- |
| `ACCUM_TYPE` | OpenCL/C type name | `max(COMPUTE_TYPE, STATE_TYPE)` by element size. Used for accumulative operations in stateful-update kernels. |
| `ACCUM_IS_WIDER_THAN_COMPUTE` | `int` (0\|1) | 1 when `STATE_TYPE > COMPUTE_TYPE`, indicating accumulation uses the wider state type; 0 otherwise. |

### 6.7. Cross-Backend Naming

All backends (OpenCL, CPU, Vulkan) use the canonical flag names from this article (`STORAGE_TYPE_IS_*`, `COMPUTE_TYPE_IS_*`, `STATE_TYPE_IS_*`). The Vulkan backend's GLSL type macros and boolean flags use the same canonical names, injected via `glslc -D`. No backend-specific shortened names are used.

### 6.8. CPU Backend FP8 Implementation

The CPU backend represents FP8 values as C struct wrappers (`cpu_fp8_e4m3`, `cpu_fp8_e5m2`) containing a `uint8_t bits` field. Conversion between FP8 and compute types uses LUT-based decode (256-entry lookup tables from `cpu_fp8_lut.gen.h`) and algorithmic round-to-nearest-even encode (in `cpu_fp8.h`). The precision macro system uses `STORAGE_SUFFIX` token pasting: `STORAGE_SUFFIX=fp8e4m3` generates `scalar_load_real_fp8e4m3()` etc. FP16 compute variants (`s8e4c16x32`, `s8e4c16x64`, `s8e5c16x32`, `s8e5c16x64`) are conditionally compiled when `_Float16` is available, detected by Meson and exposed via `HAS_FLOAT16`.

### 6.9. Retired Build-Time Symbols

| Symbol | Reason for Retirement | Current Status |
| :----- | :-------------------- | :------------- |
| `LOCAL_MEM_BANK_PADDING` | Invariant value (`1`) across all hardware targets; not a hardware profile parameter. | Kernel-internal `#define` in `kernels.cl.h`. Build system MUST NOT provide via `-D`. |
| `SCALAR_TYPE` | Replaced by three-role precision model (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`). | Does not exist in any kernel source. |
| `SCALAR_IS_HALF` | Replaced by three-role precision model. | Does not exist in any kernel source. |

---

## Article 7: Canonical Interface Instantiation

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
     * @param src_buffer_GLOBAL_input_stream The primary data source.
     *        - Tensor Shape: (src_scalar_NATURAL_total_item_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Post-pad to 128-byte alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_item_count]
     *        - Validation Preconditions: [src_scalar_NATURAL_item_offset
     *          + src_scalar_NATURAL_item_count
     *          <= src_scalar_NATURAL_total_item_count]
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input_stream,

    /**
     * @param src_buffer_DEVICE_CONST_lookup_table A read-only device-constant resource.
     *        - Tensor Shape: (LUT_CAPACITY)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [Compile-time constant: LUT_CAPACITY]
     *        - Validation Preconditions: None.
     */
    __constant const COMPUTE_TYPE *src_buffer_DEVICE_CONST_lookup_table,

    /**
     * @param dest_buffer_GLOBAL_partial_results The collection buffer
     *        for this unit's partial output.
     *        - Tensor Shape: (out_scalar_NATURAL_total_chunks,
     *          RESULT_ELEMENTS_PER_CHUNK)
     *        - Padding Contract: {Type: NONE}
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [out_scalar_NATURAL_total_chunks,
     *          Compile-time constant: RESULT_ELEMENTS_PER_CHUNK]
     *        - Validation Preconditions: Host shall zero-initialize
     *          prior to dispatch.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_results,

    /**
     * @param update_buffer_LOCAL_transpose_tile Work-group exclusive memory
     *        for a tiled matrix transpose.
     *        - Allocation Formula: TILE_DIM * (TILE_DIM + LOCAL_MEM_BANK_PADDING)
     *          * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "2D tile with padded stride (TILE_DIM + 1) to
     *          avoid bank conflicts during column-wise access.
     *          LOCAL_MEM_BANK_PADDING is a kernel-internal constant (value: 1),
     *          not a build-time symbol."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_transpose_tile,

    /**
     * @param sync_buffer_GLOBAL_atomic_counter A global resource for
     *        cross-group atomic synchronization.
     *        - Tensor Shape: (1)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [Implicit size: atomic_uint]
     *        - Validation Preconditions: Host shall initialize to 0.
     */
    __global atomic_uint *sync_buffer_GLOBAL_atomic_counter,

    uint src_scalar_NATURAL_item_offset,
    uint src_scalar_NATURAL_item_count,
    uint src_scalar_NATURAL_total_item_count,
    float src_scalar_REAL_processing_threshold,
    uint out_scalar_NATURAL_output_chunk_index,
    uint out_scalar_NATURAL_total_chunks
);
```

---

## Article 8: Canonical Lexicon for `[ContextAndUsage]`

### 1.0 Mandate

This Lexicon establishes the sole binding definitions for the `[ContextAndUsage]` component. Usage of any term not defined herein is a contract violation.

### 1.1 Canonical Abbreviations

The following abbreviations are formally blessed as interchangeable short-form representations:

| Lexicon Term | Abbreviation |
| :----------- | :----------- |
| `temperatures` | `temps` |
| `probabilities` | `probs` |
| `gradient` | `grad` |

### 1.2 Pluralization Rule

Terms are defined in singular form. Plural forms are implicitly valid when combined with cardinality modifiers (`num_`, `total_`, `_count`, `_per_*`). Singular and plural forms carry no semantic distinction beyond grammatical number; they refer to the same Lexicon entry.

### 2.0 Core Data Role Primitives

#### Group 1: Foundational Inputs & Ground Truth

| Term | Definition |
| :--- | :--------- |
| `input` | The initial, untransformed data set for a complete computation. **Dimensional usage:** when qualifying a dimension (e.g., `padded_input_count`, `input_count`), denotes the feature dimensionality of the primary input data. |
| `targets` | The ground truth labels for a supervised learning task. |

#### Group 2: Learnable Model Parameters

| Term | Definition |
| :--- | :--------- |
| `weights` | The set of learnable weight parameters for a model layer. |
| `biases` | The set of learnable bias parameters for a model layer. |
| `temperatures` | The set of learnable temperature parameters for logit scaling. |
| `parameters` | A generic learnable parameter buffer (e.g., for optimizer dispatch across parameter groups). |

#### Group 3: Forward Pass Data Flow

| Term | Definition |
| :--- | :--------- |
| `hidden_activations` | The post-activation output tensor of an intermediate system layer. |
| `logits` | The pre-activation, real-valued output tensor of the final system layer. |
| `probabilities` | The post-activation, normalized probability tensor of the final system layer. |

#### Group 4: Learning Process Artifacts

| Term | Definition |
| :--- | :--------- |
| `loss` | The final computed loss value or tensor. |
| `gradient` | The gradient tensor derived from a specified parameter. |
| `normalization` | An L2 norm, typically of a gradient vector or its partials. |

#### Group 5: Parallel Processing & Reduction Primitives

| Term | Definition |
| :--- | :--------- |
| `partial` | A single, intermediate, un-aggregated result. Used as a noun. |
| `partial_collection` | A generic memory pool containing multiple, potentially non-contiguous, `partial` results. |
| `offset_list` | An indirection table containing a list of memory offsets for scatter/gather operations. |

#### Group 6: Control, Scoping & Dimensionality Primitives

| Term | Definition |
| :--- | :--------- |
| `module` | A parameter specific to a single classifier module (head). |
| `shared` | A parameter that is shared across multiple modules or layers. |
| `output_class` | A dimension or count related to the output classes of a classifier. |
| `sample_mask` | A packed `uint*` bitmask encoding sample validity (32 samples per word, LSB-first; bit 1 = valid, 0 = invalid/padding). Accessed via `load_sample_mask()`. Host allocation shape: `((batch_size + 31) // 32,)` of `uint32`. |
| `hidden_mask` | A derivative mask for the hidden layer activation function. Under `explicit` strategy: produced by Node 4 at compute precision, stored at storage precision, preserving derivative truth across storage narrowing. Under `recompute` strategy: derived internally by consuming kernels from stored activations (`activation > 0`). |
| `effective_batch_size` | The number of valid (non-padding) samples in a batch, computed by host-side popcount over the sample mask. |
| `batch` | The primary sample dimension of a training batch. Used as a dimensional qualifier in counts, offsets, and chunk decompositions. |
| `hidden` | The dimensionality of the intermediate (hidden) representation layer. Used as a dimensional qualifier. |
| `tile` | A discrete, indivisible unit of parallel work. Distinguished from `flat_tile` (§3.0), which denotes the specific flattening strategy; `tile` is the resulting work unit itself. |
| `height` | The extent of a logical dimension, typically the slower-moving one (number of rows). |
| `width` | The number of scalar elements constituting a single logical row or 1D vector. For a 2D entity, the number of columns. |

#### Group 7: Utility Primitives

| Term | Definition |
| :--- | :--------- |
| `generic` | A type-punned buffer whose interpretation is context-dependent. |

### 3.0 Decomposition Strategy Primitives

| Term | Definition |
| :--- | :--------- |
| `flat_tile` | A unit of work from the flattening of a logical grid to a 1D index. |
| `batch_chunk` | A contiguous 1D partition of the primary batch dimension. |
| `class_chunk` | A contiguous 1D partition of the output class dimension. |
| `module_chunk` | A contiguous 1D partition of the module (classifier head) dimension. |

### 4.0 Context Modifiers

#### 4.1. Prefixes

| Prefix | Function |
| :----- | :------- |
| `total_` | The total logical cardinality of a dimension. |
| `padded_` | The physical, in-memory cardinality of a dimension. |
| `num_` | The number of partitions a dimension has been decomposed into. |
| `partial_` | An intermediate, un-aggregated result requiring further reduction. |
| `leaf_` | A raw, un-aggregated result at the entry point of a reduction process — the most granular form of a `partial_`. |
| `clipped_` | A buffer whose elements have undergone norm-clipping. A transitional state applied to `partial_` gradients before reduction. |
| `summed_` | A buffer whose elements are the result of batch-wide reduction (summation) of `partial_` or `clipped_` precursors. |
| `intermediate_` | A buffer at a transitional reduction stage — past `partial_` (raw/clipped) but before `summed_` (fully reduced). |
| `final_` | A fully processed, normalized result ready for consumption by a final state-modifying kernel (e.g., optimizer). |
| `in_` | Pertaining to a source buffer. |
| `write_` | Pertaining to a computed write position within a destination buffer. Qualifies an address *offset* calculated by the host for placement within a collection buffer. |

#### 4.2. Suffixes

| Suffix | Function |
| :----- | :------- |
| `_index` | A logical, ordinal position within a sequence or grid. |
| `_offset` | A physical displacement for direct memory address calculation. |
| `_count` | The number of elements to process, relative to a corresponding `_offset`. |
| `_global` | A value applying uniformly across an entire batch or dispatch. |
| `_pow_t` | A value representing a base raised to the power of the current time-step *t*. |
| `_t_pre` | A value for the initial pre-process stage of a multi-stage process (e.g., the leaf clip in a reduction tree). |
| `_t_j` | A value dependent on stage *j* of a multi-stage process (e.g., reduction tree layer). |
| `_per_item` | One value per logical work-item (tile) rather than a single global scalar. |
| `_per_chunk` | One value per decomposition chunk. |
| `_per_stage` | One value per reduction tree stage. |

### 5.0 Domain and Utility Primitives

#### 5.1. Specialized Layout Suffixes

| Suffix | Definition |
| :----- | :--------- |
| `_simd_major` | A Struct-of-Arrays (SoA) layout aligned to SIMD vector width. |
| `_aos` | An Array-of-Structs layout. |
| `_soa` | A Struct-of-Arrays layout. |
| `_permuted` | A buffer whose elements have undergone a non-trivial permutation. |
| `_flat` | A logically multi-dimensional structure serialized into a contiguous 1D representation. Flattening convention (row-major) follows Article 1.3. |

#### 5.2. Local Memory Patterns

| Term | Type | Definition |
| :--- | :--- | :--------- |
| `simd_tile` | Data Role | A local memory tile used for SIMD optimization. |
| `reduction_tile` | Data Role | A local memory tile used for parallel reduction. |
| `transpose_tile` | Data Role | A local memory tile used for matrix transpose. |

#### 5.3. Optimizer State & Hyperparameters

| Term | Type | Definition |
| :--- | :--- | :--------- |
| `m1`, `m2` | Data Role | The first and second moment vectors (Adam optimizer). |
| `learning_rate` | Hyperparameter | Optimizer learning rate. |
| `beta1`, `beta2` | Hyperparameter | Exponential decay rates for moment estimates. |
| `epsilon` | Hyperparameter | Small constant for numerical stability in denominator terms. |

#### 5.4. Generic Matrix & Constraint Primitives

| Term | Type | Definition |
| :--- | :--- | :--------- |
| `stride` | Suffix | The physical displacement, in elements, between the start of consecutive rows. |
| `min_value` | Scalar Context | The inclusive minimum boundary for a value. |
| `max_value` | Scalar Context | The inclusive maximum boundary for a value. |
| `fp_max` | Scalar Context | The maximum finite value representable in the current floating-point format. |

#### 5.5. Gradient Stabilization Primitives

| Term | Type | Definition |
| :--- | :--- | :--------- |
| `clipping_threshold` | Data Role | The maximum permissible L2 norm for gradient clipping. |
| `policy_t_algorithmic` | Hyperparameter | Time-dependent policy threshold from the adaptive gradient clipping schedule. |
| `policy_lambda` | Hyperparameter | Scaling coefficient for the gradient clipping policy. |
| `policy_max_k` | Hyperparameter | Maximum fan-in *K* for staged reduction policy selection. |

#### 5.6. Reduction Engine Topology

| Term | Type | Definition |
| :--- | :--- | :--------- |
| `fan_in` | Topology | The number of input partials consumed by a single reduction node (the *K* in K-fan-in). |
| `node` | Topology | An independent unit of work in the reduction tree; each node aggregates *K* partials into one output. |
| `stage` | Topology | A level in the multi-stage reduction tree. Stage 0 is the leaf layer; higher stages consume prior stage outputs. |

### 6.0 Canonical Flag Identifiers

| Term | Definition |
| :--- | :--------- |
| `problem_type` | Selects the primary loss calculation path (CCE vs. BCE). |
| `operation_type` | Selects the mathematical operation for a generic kernel (e.g., SUM vs. AVERAGE for an aggregator). |
| `use_per_item_norm` | Selects the gradient clipping threshold source between a `_global` scalar and a `_per_item` buffer. |
| `produce_hidden_mask` | Controls whether the forward pass kernel writes the ReLU derivative mask. When 0, mask writes are skipped and the Host MAY pass a minimal stub buffer. |
| `use_explicit_hidden_mask` | Selects the ReLU derivative source in consuming kernels. When 0, derived internally from stored activations (`activation > 0`). When 1, read from an explicit `hidden_mask` buffer. |

### 7.0 Forbidden & Deprecated Terms

The following terms are contractually forbidden and must be refactored if found in existing code:

| Term | Reason | Replacement |
| :--- | :----- | :---------- |
| `param` | Too generic. | `parameters`, or a specific learnable term (`weights`, `biases`). |
| `h` | Ambiguous abbreviation. | `hidden_activations` |
| `elements` | Redundant with `_count`. | Standardize on the `_count` suffix. |
| `_leading_dim` | Ambiguous library-specific term. | `stride` |
| `cce` / `bce` | Problem-specific type encoded in name. | Generic terms (`loss`, `targets`); type is handled by a `FLAG` parameter. |

**Exception: Architecturally-Mandated Kernel Bifurcation.**
When CONCEPT.md Principle 3(B) requires separate kernels due to incompatible type signatures, memory layouts, or downstream DAG topologies, those kernels may use otherwise-forbidden terms to distinguish the variant. This exception applies only when:

1. The kernel pair cannot share a unified interface (differing buffer types, shapes, or DAG edges).
2. The distinction is documented in the kernel's `@kernel_contract` block with an explicit reference to Principle 3(B).
3. No `FLAG` parameter could eliminate the interface divergence without producing a "smart kernel" with complex internal branching over incompatible memory patterns.

**Current applicants:** `compute_probs_loss_cce_chunk` (Node 6), `compute_probs_loss_bce_chunk` (Node 7).