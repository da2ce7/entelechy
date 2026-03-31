# ADR-020: Mixed-Precision Execution Model

**Status:** PROPOSED  
**Date:** 2026-03-31  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-008 (extends), ADR-007 (extends), ADR-009 (extends), ADR-013 (constrains), ADR-014 (constrains), ADR-001  
**Blocks:** —  
**Triggered by:** Recognition that the uniform `SCALAR_TYPE` model is an incomplete abstraction of how numeric computation actually requires precision

---

## Context

The system's current precision model (ADR-008) is built on a single assumption: **one datatype governs everything**. `PrecisionConfig` carries one `numpy_dtype`, one `fp_format_max`, and one `epsilon`. This single type — `SCALAR_TYPE` at the kernel level — is used for buffer storage, arithmetic accumulation, optimizer state, and numerical stability guards uniformly throughout the entire execution plan.

This uniform model is an architectural simplification, not a reflection of how numeric computation actually works. In practice, different phases of a numerical pipeline have fundamentally different precision requirements. The uniform model conflates three independent concerns:

### 1. Storage precision is a bandwidth concern

The buffers flowing through the execution DAG — activations, weight matrices, gradient partials — are the dominant consumers of device memory and memory bandwidth. Their precision determines buffer size and transfer throughput. Narrower storage formats reduce memory pressure proportionally: FP16 halves FP32's footprint; FP8 halves again.

The Primacy of Memory Strategy (CONCEPT.md §2) identifies bandwidth as the singular optimization target of host-side orchestration: "The singular goal of the host-side orchestration is to ensure the core computation executes in the fastest possible memory tier." Storage precision is the most direct lever for this — a narrower storage type moves more data through the same bandwidth.

Under the uniform model, narrowing storage precision also narrows arithmetic precision. This coupling is not inherent to the problem — it is an artifact of the single-`SCALAR_TYPE` abstraction.

### 2. Compute precision is an accuracy concern

Arithmetic operations — accumulation in reductions, transcendental functions in Softmax and Sigmoid, gradient scaling — need mantissa headroom to produce numerically faithful results. The precision requirements differ by operation:

- **Reduction tree accumulation.** The $\log_K(N)$ Recursive Reduction Engine (CONCEPT.md §2) sums $K$ partials at each stage. Exact summation of $K$ values requires $\log_2(K)$ additional mantissa bits beyond the summands' precision. For $K=4$, an FP16 summand (10 mantissa bits) needs 12 bits for exact accumulation — feasible within FP16's format. An FP8 E4M3 summand (3 mantissa bits) needs 5 bits — exceeding its format. FP32 accumulation (23 mantissa bits) handles $K$ up to $2^{23}$ without bit exhaustion.

- **Softmax log-sum-exp.** Node (6) `compute_probs_loss_cce_chunk` computes numerically stable Softmax internally (CONCEPT.md §Kernel & Synchronization Contracts). The four-step sequence — row-max subtraction, exponentiation, summation, logarithm — amplifies quantization error at each step. With FP32 arithmetic, the 23-bit mantissa provides sufficient resolution for smooth probability distributions. With FP16, the 10-bit mantissa introduces measurable but tolerable quantization. With narrow formats (≤8 bits mantissa), the `exp()` output quantizes to a sparse set of distinct values, and the Softmax degenerates.

- **Gradient stabilization safety ceiling.** The Quadratic Scaling Policy (CONCEPT.md §3.3) schedules thresholds as $T_j = T_{\text{algorithmic}} + \lambda j^2$, clamped by $T_{\text{safety}_j} = \text{FP\_FORMAT\_MAX} / K_j$. The safety ceiling operates on the *arithmetic* format — it must prevent overflow during summation, which is a compute-precision property. Coupling this to storage precision means that narrowing storage for bandwidth also narrows the safety ceiling, constricting the policy funnel.

Under the uniform model, the user must choose: narrow precision for bandwidth (and accept degraded arithmetic fidelity), or wide precision for accuracy (and forfeit bandwidth). This is a false dilemma imposed by the single-type abstraction, not by the underlying mathematics.

### 3. State precision is a stability concern

The Adam optimizer (Node 24) maintains exponential moving averages `m1` and `m2` with the `MODEL_STATE` lifecycle role (ADR-009) — these persist across batches for the lifetime of training. The EMA update $\beta_1 \cdot m + (1 - \beta_1) \cdot g$ requires that the format can represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. With typical $\beta_1 = 0.9$, the gradient contribution is scaled by 0.1 — the format needs enough mantissa to distinguish $m$ from $m + 0.1g$.

In FP32 (machine epsilon $\sim 10^{-7}$), this works for all practical gradient magnitudes. In FP16 (machine epsilon $\sim 10^{-3}$), the EMA can track gradients whose magnitude is at least $\sim 10^{-3} \cdot |m|$ — adequate for most training, but not all. In narrower formats, the EMA degenerates: moment values quantize to a sparse grid where the update rounds to either $m$ or $g$, destroying the smooth averaging Adam depends on.

State precision is independent of both storage and compute precision. A pipeline may store activations in FP16 for bandwidth, compute reductions in FP32 for accuracy, and maintain optimizer moments in FP32 for stability — or any other combination where each role's precision is chosen for its specific requirement.

### 4. The existing architecture already separates these concerns — in one place

The host computes `beta1**t` and `beta2**t` bias correction terms "in high precision (FP64)" (CONCEPT.md §7, §11). This is explicit mixed-precision computation: the bias correction scalars are computed in FP64 regardless of `SCALAR_TYPE`, then passed to the kernel as `SCALAR_TYPE` parameters.

The architecture already acknowledges that some computations require wider precision than the pipeline's primary type. But this acknowledgment is ad-hoc — hardcoded in the host orchestrator for one specific case (Adam bias correction), with no formal model. The `PrecisionConfig`, the `KernelContract`, and the plan's `BufferDescriptor` have no vocabulary to express that different roles need different precisions. The FP64 bias correction works only because it is computed on the host before kernel dispatch; there is no mechanism for expressing mixed precision *within* the device-side execution plan.

### 5. The immediate value: FP16 storage with FP32 compute

The most immediate beneficiary of a mixed-precision model is not a future format — it is the combination of formats the system already supports. FP16-storage/FP32-compute is a well-established configuration in the ML ecosystem:

- **Storage:** Weights and activations stored in FP16 for 2× memory reduction and 2× bandwidth improvement vs. FP32.
- **Accumulation:** Reduction tree summations, Softmax log-sum-exp, and loss computation performed in FP32 for numerical fidelity.
- **Optimizer state:** Adam moment vectors maintained in FP32 for stable EMA tracking.

This configuration requires zero toolchain extensions — FP16 and FP32 are fully supported by all three backends (OpenCL `half`/`float`, GLSL `float16_t`/`float`, C `_Float16`/`float`). It requires zero hardware prerequisites. It delivers a concrete benefit: the bandwidth advantage of FP16 with the numerical headroom of FP32 for operations that need it.

Under the current uniform model, this configuration is inexpressible. A user must choose `PrecisionConfig.float16()` (fast, less precise) or `PrecisionConfig.float32()` (precise, slower). The mixed configuration — which is strictly superior to either uniform choice for bandwidth-bound workloads — cannot be requested.

### 6. The extreme case: FP8

FP8 (E4M3 and E5M2) is the format that made the uniform model's incompleteness impossible to ignore. Unlike FP16, where uniform precision produces degraded but functional results, uniform FP8 is **numerically non-functional**:

| Failure mode | Mechanism | Severity |
|:---|:---|:---|
| **Safety ceiling collapse** | $T_{\text{safety}_j} = 448 / K_j$. At $K=8$: $T_{\text{safety}} = 56$. The Quadratic Scaling Policy funnel degenerates to hard-clipping at every stage. | Complete loss of gradient regulation structure. |
| **Softmax quantization** | E4M3 has ~256 distinct positive values. `exp()` maps the post-max-subtraction logits to a sparse quantized set. The probability distribution is a step function. | Cross-entropy loss is dominated by quantization noise. |
| **Adam moment degeneration** | Machine epsilon ≈ 0.125 (E4M3). The EMA rounds to either $m$ or $g$ — no smooth averaging. | Optimizer state is meaningless. |
| **Reduction tree mantissa saturation** | Summing $K=4$ partials with 3 mantissa bits requires 5 bits. Exhausted after 1 stage. | Accumulated gradients are random values within the clipped range. |

An attempt to support FP8 by adding it as a uniform `SCALAR_TYPE` — extending `_DTYPE_EPSILON_MAP`, adding FP8 to `cpu_precision.h` — would produce an engine that "runs" but does not "train." The result violates the system's most fundamental guarantee: that the mathematical algorithm specified in `kernels.cl.h` is faithfully executed.

FP8's practical value in the ML ecosystem is as a **storage and bandwidth format** in a mixed-precision pipeline: FP8 inputs with FP16/FP32 accumulation (hardware tensor cores perform this natively). The mixed-precision model is not an FP8 accommodation — it is the correct abstraction that FP8 demands and that FP16/FP32 already benefits from.

### 7. Toolchain landscape for narrow formats

| Layer | FP32 | FP16 | FP8 (E4M3/E5M2) |
|:---|:---|:---|:---|
| **C type** | `float` | `_Float16` (C23) | No standard type. |
| **NumPy dtype** | `numpy.float32` | `numpy.float16` | Requires `ml_dtypes.float8_e4m3fn` / `float8_e5m2`. |
| **OpenCL** | `float` | `half` | No FP8 type in OpenCL 1.2 or 3.0. |
| **GLSL/SPIR-V** | `float` | `float16_t` | No FP8 type in Vulkan 1.3. |
| **CPU SIMD** | AVX2 `__m256` | F16C `_mm256_cvtph_ps` | No standard FP8 SIMD intrinsics. |

FP16/FP32 mixed precision has zero toolchain gaps. FP8 has comprehensive gaps. The mixed-precision model must be designed to accommodate both — delivering immediate value for FP16/FP32 today while establishing the architectural pathway for FP8 when toolchains mature.

### 8. Architectural Elegance Feedback

Per CONCEPT.md §1:

> *When emergent efficiency gains contradict current constraints:*
> 1. *Suspend implementation of the optimization*
> 2. *Formalize the pattern as a documented architectural primitive*
> 3. *Reify the optimization through revised contracts & DAG extensions*

The uniform `SCALAR_TYPE` model is the constraint. The mixed-precision pattern — using narrower types for bandwidth and wider types for arithmetic — is the optimization that contradicts it. The contradiction was always latent: FP16 uniform pipelines accept degraded accumulation fidelity as a trade-off, when mixed FP16/FP32 would eliminate the trade-off entirely. FP8 made the contradiction acute by producing non-functional results under uniform precision.

The Architectural Elegance Feedback protocol applies:

1. **Suspend:** Do not implement FP8 as a uniform `SCALAR_TYPE`. Do not implement FP8-specific narrow-format code until toolchain preconditions are met.
2. **Formalize:** Establish mixed-precision as an independent, first-class architectural primitive — the recognition that storage precision, compute precision, and state precision are three distinct concerns with different optimization targets.
3. **Reify:** Extend the precision configuration (ADR-008), kernel contract vocabulary (ADR-007), buffer lifecycle annotations (ADR-009), build-time symbols (CONTRACT.md Article 6), and kernel source strategy (ADR-013) with concrete, binding vocabulary for the three precision roles.

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** The protocol is mandatory. The contradiction between the uniform-precision constraint and the mixed-precision optimization has been identified. The architecturally required response is formalization and reification — not acknowledgment and deferral.

2. **CONCEPT.md §2 (Primacy of Memory Strategy).** Storage precision is the most direct bandwidth lever. The architecture must be able to decouple storage bandwidth from arithmetic fidelity. A narrow storage type with a wide compute type directly serves the "fastest possible memory tier" goal without sacrificing numerical correctness.

3. **ADR-008 (PrecisionConfig — composition-based, dtype-indexed).** `PrecisionConfig` is the sole precision authority in the shared layer. The mixed-precision extension must compose alongside it per the same composition-over-inheritance principle ADR-008 established. The existing uniform model must remain unchanged for uniform pipelines.

4. **ADR-001 (Three-Tier Jurisdictional Model).** The assignment of precision roles to buffers and operations is a Policy-tier concern — it determines numerical correctness. The mechanism by which a backend implements type widening (hardware tensor cores, software conversion) is an Orchestration/Execution concern. The boundary must separate these.

5. **ADR-007 (KernelContract/KernelBinding Split).** Kernel contracts specify buffer parameter types symbolically. A mixed-precision model requires expressing that some buffers carry a narrow storage type while the kernel computes in a wider type. The contract vocabulary must distinguish storage precision from compute precision — this is an interface property, not a binding detail.

6. **ADR-009 (BufferDescriptor — plan-prescribed lifetimes and roles).** `BufferDescriptor` annotates each buffer with shape, lifecycle role, and lifetime. Precision role is a natural extension of this annotation: it tells the backend what native type to allocate the buffer in, just as `element_size_bytes` tells it how large the allocation must be.

7. **ADR-013 (Kernel Source Strategy — `SCALAR_TYPE` uniformity).** Kernel sources use a single `SCALAR_TYPE` for all variables. Mixed-precision requires distinguishing storage access types from arithmetic types at the kernel source level. The extension must preserve the property that kernel algorithmic cores operate in a single arithmetic type — widening/narrowing occurs only at data access boundaries.

8. **CONTRACT.md Article 6 (Mandatory Build-Time Symbols).** The current symbols (`SCALAR_TYPE`, `SCALAR_IS_HALF`, `NUMERICAL_STABILITY_EPSILON`) assume uniform precision. Additional symbols are needed to express the storage/compute distinction.

9. **ADR-014 (Build System Integration — Meson).** The `aec_cpu_kernel_precision` combo option governs the C `REAL_T` typedef. Mixed-precision may require an additional option dimension or a different instantiation strategy for backends that support it.

---

## Options Considered

### Option A: Extend the uniform model — add more `SCALAR_TYPE` choices

Add new dtypes (BFloat16, FP8 variants) to `PrecisionConfig`'s `_DTYPE_EPSILON_MAP`. Each new format becomes a new uniform `SCALAR_TYPE` value. The system handles FP8 the same way it handles FP16: one type everywhere.

**Advantages:**
- Zero architectural changes. `PrecisionConfig`, `KernelContract`, `BufferDescriptor`, build system — all unchanged.
- Smallest scope.

**Disadvantages:**
- **Does not address the actual problem.** The gap is not "we need more uniform types" — it is "different operations need different precisions." Adding FP8 as a uniform type produces numerically non-functional results (§Context item 6). Adding BFloat16 as a uniform type forfeits its loss precision (7-bit mantissa) in Softmax/loss kernels.
- **FP16/FP32 mixed precision remains inexpressible.** The immediate value case (§Context item 5) is inaccessible.
- **Perpetuates a known-incomplete abstraction.** The uniform model conflates storage bandwidth, arithmetic fidelity, and state stability — three independent concerns with different optimization targets. Extending it adds more values to an axis that should not exist as a single axis.

### Option B: Precision roles — formalize storage, compute, and state as independent precision dimensions

Introduce `MixedPrecisionConfig` as a frozen dataclass that decomposes precision into three roles: storage (bandwidth), compute (arithmetic), and state (optimizer persistence). Extend `KernelContract`, `BufferDescriptor`, and build-time symbols with a `precision_role` vocabulary that maps each buffer and each kernel's arithmetic domain to the appropriate precision.

The three roles form a closed taxonomy:

| Role | Governs | Optimization target | Example |
|:---|:---|:---|:---|
| **Storage** | Buffer memory layout, transfer bandwidth | Bandwidth | FP16 weights, FP8 activations |
| **Compute** | Arithmetic accumulation, transcendentals | Numerical fidelity | FP32 reduction sums, FP32 Softmax |
| **State** | Persistent optimizer buffers | Stability over unbounded time | FP32 Adam moments |

**Advantages:**
- **Addresses the root cause.** The three roles capture the actual precision requirements of the pipeline — independent dimensions with independent optimization targets.
- **Immediately useful.** FP16-storage/FP32-compute/FP32-state requires no toolchain extensions and delivers bandwidth improvement with full arithmetic fidelity.
- **Correctly models FP8.** FP8-storage/FP16-compute/FP32-state matches hardware capabilities (tensor cores accumulate in wider types natively) and avoids all five FP8-uniform failure modes.
- **Backward compatible.** Uniform pipelines are the degenerate case where `storage == compute == state`. The `precision_role = None` annotation means "use uniform `PrecisionConfig`." Existing code is unaffected.
- **Policy-tier visibility.** Precision role assignments are plan data — inspectable, validatable, and backend-neutral. The plan builder can verify that reduction tree accumulation uses `compute_dtype`, not `storage_dtype`. The backend is told what precision each buffer carries; it does not infer it.
- **Extensible.** New formats (BFloat16, future FP formats) slot into the existing role taxonomy without new abstractions. The question for each format is: "which roles does it serve?" — not "should we add a new class."

**Disadvantages:**
- **Architectural extension across multiple subsystems.** `PrecisionConfig` (ADR-008), `KernelContract` (ADR-007), `BufferDescriptor` (ADR-009), build-time symbols (CONTRACT.md Article 6), kernel source strategy (ADR-013), and the build system (ADR-014) all gain new vocabulary.
- **Two precision configurations coexist.** `PrecisionConfig` (uniform) and `MixedPrecisionConfig` (role-based) are parallel concepts until a future unification.
- **Kernel complexity.** Kernels that access storage-type buffers must perform explicit widening/narrowing at data boundaries. This is a new correctness obligation that must be specified in `Behavioral Invariants` and verified.
- **Validation surface.** The plan builder must verify cross-role consistency: `storage_dtype.itemsize <= compute_dtype.itemsize`, all `"storage"` role buffers are accessed only by kernels that declare widening conversion, etc.

### Option C: Per-buffer precision — each buffer independently declares its own dtype

Instead of a closed role taxonomy, allow each `BufferDescriptor` to carry an arbitrary `numpy_dtype` independent of any global precision configuration. Kernels declare per-parameter dtypes in their contracts. No roles, no global configuration — fully decentralized precision.

**Advantages:**
- Maximum flexibility. Any buffer can be any type. A pipeline could use FP8 for activations, FP16 for gradients, FP32 for moments, and BFloat16 for weight storage — all independently specified.
- No closed taxonomy to maintain or extend.

**Disadvantages:**
- **Combinatorial explosion.** With $B$ buffers and $P$ possible precisions, the plan builder must validate $B \times P$ individual choices rather than 3 global role assignments. The validation becomes intractable for the 40–60 buffers in a typical plan.
- **No structural guarantee of consistency.** Two buffers that serve the same logical role (e.g., `clipped_partial_grad_shared_weights` and `clipped_partial_grad_shared_biases`) could have different precisions with no mechanism to enforce uniformity within a role.
- **Kernel contracts become per-buffer.** Every `BufferParamSpec` needs its own dtype, and every kernel implementation must handle arbitrary type combinations. The kernel source strategy (ADR-013) cannot provide a single `COMPUTE_TYPE` — it must provide per-parameter types, multiplying the kernel source complexity.
- **No policy-tier semantics.** The precision assignment carries no information about *why* a buffer has its precision. The distinction between "this buffer is narrow for bandwidth" and "this buffer is narrow because the value doesn't need precision" is lost. Without semantic roles, the plan builder cannot make structural guarantees about numerical correctness.
- **Over-engineers the problem.** The practical mixed-precision configurations in use (FP16/FP32, FP8/FP16, FP8/FP32) are all expressible as 2–3 global role assignments. The per-buffer model provides flexibility that no known workload requires, at the cost of structural clarity.

---

## Analysis

### Eliminating Option A

Option A adds more values to the wrong axis. The problem is not a missing dtype — it is a missing *dimension*. The system needs to express that storage precision and compute precision are independent, not that the single uniform precision has more choices.

The FP16/FP32 mixed-precision case proves this independently of FP8: a user today who wants FP16 bandwidth with FP32 accumulation accuracy has no way to express this, regardless of how many uniform types are available. Option A, by its nature, cannot express this — it only adds more choices along the one axis it has.

For FP8 specifically, Option A produces a numerically non-functional engine (§Context item 6). But the elimination does not depend on FP8 — it depends on the architectural incompleteness of a single-axis precision model.

### Eliminating Option C

Option C replaces one extreme (fully uniform) with the opposite extreme (fully decentralized). The practical mixed-precision patterns in use across the ML ecosystem all follow a small, fixed set of role-based configurations:

| Configuration | Storage | Compute | State |
|:---|:---|:---|:---|
| PyTorch `autocast` FP16 | FP16 | FP32 | FP32 |
| TransformerEngine FP8 | E4M3 / E5M2 | FP16 or FP32 | FP32 |
| NVIDIA AMP | FP16 | FP32 | FP32 (master weights) |
| BFloat16 training | BF16 | FP32 | FP32 |

In every case, the configuration is expressed as 2–3 global type choices mapped to semantic roles. No production framework assigns per-buffer or per-tensor precision from an open set — they use role-based policies.

Option C solves a problem that does not exist (arbitrary per-buffer precision), introduces a validation problem that the role-based model avoids (combinatorial consistency checking), and loses the semantic structure that makes the mixed-precision model architecturally meaningful (why a buffer has its precision, not just what it is).

### Choosing Option B

Option B captures the actual structure of mixed-precision computation: three semantic roles with different optimization targets, each assigned a precision independently. This is precisely what the ML ecosystem has converged on, and it is the minimal extension that resolves the uniform model's incompleteness.

The three-role taxonomy (storage, compute, state) is a closed set because it reflects the three distinct *reasons* a value has a precision:

- **Storage:** bandwidth optimization. Narrower is better, subject to information loss tolerance.
- **Compute:** arithmetic fidelity. Wider is better, subject to throughput cost.
- **State:** persistent stability. Wider is better, subject to memory cost.

These three roles partition all values in the execution plan:

| Plan entity | Role | Rationale |
|:---|:---|:---|
| Forward-pass activations | Storage | Bandwidth-bound intermediate data; consumed once, then dead. |
| Weight matrices (`MODEL_STATE`, read-only in the DAG) | Storage | Bandwidth-bound; loaded once per dispatch. Master copy may be at state precision. |
| Gradient partial collection buffers | Storage | Bandwidth-bound intermediates produced by `Partial Renderer` kernels. |
| Reduction tree accumulation outputs | Compute | Precision-critical: mantissa bits consumed during summation. |
| Softmax/loss intermediate values | Compute | Precision-critical: `exp()` and `log()` amplify quantization error. |
| Adam moment vectors (`m1`, `m2`) | State | Persist across unbounded training steps; EMA requires smooth averaging. |
| Adam bias correction scalars | State (host-computed in FP64) | Already mixed-precision — host computes in FP64, passes as kernel scalar. |

No value in the plan requires a precision role outside this taxonomy. If such a value were discovered, per CONCEPT.md §1, the response would be to extend the taxonomy — not to abandon it for per-buffer anarchy.

---

## Decision

**Option B: Precision roles — formalize storage, compute, and state as independent precision dimensions.**

The mixed-precision execution model is established as a first-class architectural primitive. The system's precision vocabulary is extended from one dimension (uniform `SCALAR_TYPE`) to three independent dimensions (storage, compute, state), each governing a distinct class of values in the execution plan.

### 1. `MixedPrecisionConfig` — binding extension to ADR-008

```python
@dataclass(frozen=True)
class MixedPrecisionConfig:
    """Precision configuration for mixed-precision execution pipelines.

    Decomposes the uniform PrecisionConfig model into three independent
    precision roles, each governing a distinct class of values.
    """
    storage_dtype: type        # Narrow format for buffers and bandwidth
    compute_dtype: type        # Wider format for arithmetic and accumulation
    state_dtype: type          # Format for persistent optimizer state

    storage_fp_format_max: float   # Range limit for storage-side safety checks
    compute_fp_format_max: float   # Range limit for stabilization policy (T_safety_j)
    compute_epsilon: float         # Numerical stability guard for arithmetic
```

**Relationship to `PrecisionConfig`.** `MixedPrecisionConfig` does not replace `PrecisionConfig`. The two coexist:

- **Uniform pipelines** (FP32, FP16) use `PrecisionConfig`. All `precision_role` annotations are `None`. No changes to existing functionality.
- **Mixed-precision pipelines** (FP16/FP32, future FP8/FP16, FP8/FP32) use `MixedPrecisionConfig`. The plan builder annotates buffers and kernels with precision roles.

A future ADR may unify the two by generalizing `PrecisionConfig` as the degenerate case where `storage_dtype == compute_dtype == state_dtype`. This unification is deferred to the implementing phase, as it depends on implementation experience.

**Stabilization policy integration.** When `MixedPrecisionConfig` is active, the `StabilizationPolicy` uses `compute_fp_format_max` for the safety ceiling: $T_{\text{safety}_j} = \text{compute\_fp\_format\_max} / K_j$. The safety ceiling reflects the precision of the arithmetic that performs the summation, not the precision of the stored values being summed. This preserves the Quadratic Scaling Policy funnel's headroom regardless of storage format.

**Construction invariants.** The following are enforced at construction:

- `storage_dtype.itemsize <= compute_dtype.itemsize` — storage is no wider than compute.
- `storage_dtype.itemsize <= state_dtype.itemsize` — storage is no wider than state.
- `compute_fp_format_max >= storage_fp_format_max` — compute range encompasses storage range.

### 2. `precision_role` — binding extension to ADR-007 (`KernelContract`)

`BufferParamSpec` (the contract's per-buffer-parameter specification) gains a `precision_role` field:

| `precision_role` | Buffer is at | Kernel obligation |
|:---|:---|:---|
| `"storage"` | `storage_dtype` | Widen to `compute_dtype` before arithmetic. Narrow on write. |
| `"compute"` | `compute_dtype` | Standard arithmetic applies. |
| `"state"` | `state_dtype` | Used for `MODEL_STATE` lifecycle buffers. |
| `None` | Uniform `PrecisionConfig.numpy_dtype` | Current behavior — mixed-precision machinery is dormant. |

When `precision_role` is `"storage"`, the kernel contract's `Behavioral Invariants` (CONTRACT.md Article 4.2) must specify widening conversion as a contractual obligation — not an optional optimization. This extends the existing verification model:

- **Host proof obligation (Article 1.4a):** The plan builder validates that every kernel receiving `"storage"` role buffers has a contract declaring the widening invariant.
- **Device assurance (Article 1.4b):** Kernel implementations perform the widening as their first operation on each storage-type value, and narrowing as their last operation before writing back.

When all `precision_role` fields are `None` — the current state for every existing kernel — the mixed-precision system is fully dormant. No existing contract requires modification.

### 3. Buffer precision annotation — binding extension to ADR-009 (`BufferDescriptor`)

`BufferDescriptor` gains a `precision_role` field:

```python
@dataclass(frozen=True)
class BufferDescriptor:
    handle: BufferHandle
    logical_name: str
    padded_shape: tuple[int, ...]
    element_size_bytes: int
    size_bytes: int
    role: BufferRole
    producing_node: str | None
    consumers: frozenset[str]
    last_consumer: str | None
    precision_role: Literal["storage", "compute", "state"] | None  # None for uniform precision
```

The plan builder annotates each buffer based on the active precision configuration:

- **When `PrecisionConfig` is active:** All `precision_role` values are `None`. `element_size_bytes` derives from `PrecisionConfig.numpy_dtype`.
- **When `MixedPrecisionConfig` is active:** Each buffer is annotated according to its semantic function:

| Buffer classification | `precision_role` | `element_size_bytes` derived from |
|:---|:---|:---|
| Activations, weight read-views, gradient partials | `"storage"` | `storage_dtype` |
| Reduction accumulation outputs, Softmax intermediates | `"compute"` | `compute_dtype` |
| Adam moment vectors (`m1`, `m2`), master weight copies | `"state"` | `state_dtype` |

The `element_size_bytes` and `size_bytes` fields now derive from the role-appropriate dtype, not from a single global dtype. This ensures that buffer allocation sizes reflect the actual storage format. Backends use `precision_role` plus the active `MixedPrecisionConfig` to select the correct native type for each buffer allocation.

### 4. Build-time symbol extensions — binding amendment to CONTRACT.md Article 6

In addition to the existing mandatory symbols (`SCALAR_TYPE`, `SCALAR_IS_HALF`, `NUMERICAL_STABILITY_EPSILON`, `SIMD_WIDTH`, `C_TILE_SIZE`), the following symbols are established for mixed-precision kernels:

| Symbol | Role | Uniform-mode value |
|:---|:---|:---|
| `STORAGE_TYPE` | C type for narrow storage buffers | `== SCALAR_TYPE` |
| `COMPUTE_TYPE` | C type for arithmetic and accumulation | `== SCALAR_TYPE` |
| `STORAGE_IS_NARROW` | Boolean flag (`0` or `1`): whether `STORAGE_TYPE` differs from `COMPUTE_TYPE` | `0` |

When `STORAGE_IS_NARROW == 0`, `STORAGE_TYPE == COMPUTE_TYPE == SCALAR_TYPE`, and the existing uniform codepath is unchanged. No kernel source modification is needed for uniform-precision pipelines.

`NUMERICAL_STABILITY_EPSILON` derives from `compute_epsilon` when `MixedPrecisionConfig` is active — the epsilon governs arithmetic stability, which is a compute-precision property.

### 5. Kernel source strategy — binding constraint on ADR-013

Kernels that operate on `"storage"` role buffers must perform explicit widening and narrowing at their **data access boundaries** — the point where values cross from memory into registers and back:

```c
#if STORAGE_IS_NARROW
    // Load: widen from storage to compute precision
    COMPUTE_TYPE val = (COMPUTE_TYPE)storage_buffer[idx];
#else
    COMPUTE_TYPE val = storage_buffer[idx];  // identity when types are the same
#endif

    // ... all arithmetic in COMPUTE_TYPE, identical to existing kernel code ...

#if STORAGE_IS_NARROW
    // Store: narrow from compute to storage precision
    storage_buffer[idx] = (STORAGE_TYPE)result;
#else
    storage_buffer[idx] = result;
#endif
```

This preserves ADR-013's key property: **each kernel's algorithmic core operates in a single arithmetic type** (`COMPUTE_TYPE`). The mixed-precision boundary is at data access, not within the algorithm. The widening/narrowing is structurally analogous to the existing padding/unpadding that kernels already perform at data boundaries — a format conversion at the edge, not a change to the core computation.

When `STORAGE_IS_NARROW == 0`, the preprocessor eliminates the conditional branches and the casts are identity operations. No runtime overhead for uniform-precision pipelines. No source changes to existing kernels until they are explicitly extended for mixed-precision.

### 6. Narrow-format instantiation preconditions

The mixed-precision model is immediately applicable to FP16-storage/FP32-compute on all three backends. For formats narrower than FP16 (FP8 E4M3/E5M2), instantiation is deferred until the following preconditions are satisfied:

| Precondition | Rationale |
|:---|:---|
| ADR-017 Phase 6 complete (Tier 3 parity green for FP32 and FP16). | The baseline must be stable before introducing a new precision axis. |
| A C storage type for E4M3 or E5M2 in released GCC or Clang. | The CPU backend requires a concrete `STORAGE_TYPE`. Software emulation via `uint8_t` with manual bit manipulation is not acceptable — it hides the type from compiler verification. |
| `ml_dtypes` (or equivalent) FP8 dtype stable and interoperable with NumPy. | `MixedPrecisionConfig.storage_dtype` must support `numpy.dtype().itemsize` and array construction. |
| At least one GPU backend has a viable FP8 dispatch path. | Either an OpenCL extension for 8-bit float load/store, a Vulkan `VK_KHR_8bit_storage`-compatible compute dispatch, or a demonstrated pattern for FP8 storage with FP16 compute via existing format support. |

These preconditions gate FP8-specific code. The architectural primitives decided in §§1–5 are binding and format-independent.

---

## Consequences

### Positive

- **CONCEPT.md §1 fully satisfied.** The three-step protocol is followed for the mixed-precision pattern itself — not as a response to FP8, but as the independent architectural primitive it is. Uniform `SCALAR_TYPE` is the incomplete abstraction; the three-role precision model is the formalization; the extensions to ADR-007, ADR-008, ADR-009, ADR-013, ADR-014, and CONTRACT.md Article 6 are the reification.
- **Immediately useful.** FP16-storage/FP32-compute/FP32-state can be implemented on all three backends with zero toolchain extensions. This eliminates the false FP16-vs-FP32 trade-off for bandwidth-bound workloads.
- **FP8 pathway is a natural consequence.** When FP8 toolchains mature, FP8 slots into the `storage_dtype` role with FP16 or FP32 as `compute_dtype`. No new architectural primitives are needed — only new format-specific kernel source instantiations.
- **Backward compatible.** `PrecisionConfig` is unchanged. `precision_role = None` means "uniform precision." Every existing kernel, contract, buffer descriptor, and build configuration continues to work identically.
- **Design vocabulary is binding.** `MixedPrecisionConfig`, `precision_role`, `storage_dtype`/`compute_dtype`/`state_dtype`, `STORAGE_TYPE`/`COMPUTE_TYPE`/`STORAGE_IS_NARROW` are now contractual vocabulary. Future ADRs and implementations use these terms, not ad-hoc alternatives.
- **Kernel algorithmic cores unchanged.** Mixed-precision affects data access boundaries only. Each kernel's algorithmic core continues to operate in a single arithmetic type (`COMPUTE_TYPE`), preserving the existing kernel structure and cross-backend fidelity model.

### Negative

- **Architectural scope.** Six subsystems gain new vocabulary: `PrecisionConfig` (ADR-008), `KernelContract` (ADR-007), `BufferDescriptor` (ADR-009), build-time symbols (CONTRACT.md), kernel sources (ADR-013), and build system (ADR-014). Implementation spans the shared layer and all three backends.
- **Two precision models coexist.** `PrecisionConfig` (uniform) and `MixedPrecisionConfig` (role-based) are parallel until a future unification. Code consuming precision configuration must handle both or rely on the `precision_role = None` dormancy.
- **New correctness obligation on kernels.** Kernels with `"storage"` role buffers must widen before arithmetic and narrow before writing. This is specified in `Behavioral Invariants` and verified by the plan builder, but adds to the verification surface.
- **Implementation gated by ADR-017.** The primitives are decided, but code implementing them is gated by migration phases. The FP16/FP32 mixed-precision benefit is deferred until the relevant phase is reached.
- **FP8 remains toolchain-blocked.** The architectural machinery is ready, but FP8-specific instantiation waits on compiler types, NumPy-compatible dtypes, and GPU backend dispatch paths.

### References

- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` frozen dataclass; extended by this ADR
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — Three-tier jurisdictional model; Policy/Orchestration/Execution separation
- [ADR-007: Kernel Signature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract`/`KernelBinding` separation; `BufferParamSpec` extended by this ADR
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` lifecycle roles; extended by this ADR
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — `SCALAR_TYPE` build-time symbols; kernel data-boundary pattern constrained by this ADR
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — Meson precision options
- [ADR-017: Incremental Migration Path](ADR-017-incremental-migration-path.md) — Phased implementation gates
