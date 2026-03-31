# ADR-020: FP8 Mixed-Precision Execution Model

**Status:** PROPOSED  
**Date:** 2026-03-31  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-008 (constrains), ADR-001, ADR-007, ADR-013, ADR-014  
**Blocks:** —  
**Triggered by:** Architectural feasibility assessment of FP8 (E4M3/E5M2) precision support

---

## Context

There is interest in extending the system to support 8-bit floating-point (FP8) formats for increased throughput and reduced memory bandwidth. Two IEEE-adjacent FP8 encodings are in practical use:

| Format | Exponent bits | Mantissa bits | `FP_FORMAT_MAX` | Machine epsilon | Notable hardware support |
|:---|:---|:---|:---|:---|:---|
| E4M3 (FP8) | 4 | 3 | ≈ 448 | ≈ 0.125 | NVIDIA Hopper (FP8 Tensor Cores), AMD MI300 |
| E5M2 (FP8) | 5 | 2 | ≈ 57344 | ≈ 0.25 | Same accelerators; wider range, less precision |

The current precision model (ADR-008) assumes a **uniform `SCALAR_TYPE`**: a single datatype is used for storage, arithmetic, accumulation, and optimizer state throughout the execution plan. `PrecisionConfig` carries one `numpy_dtype`, one `fp_format_max`, and one `epsilon`. Every buffer in the plan, every kernel scalar parameter, and every intermediate accumulation register operates at this single precision.

This uniform model has served FP32 and FP16 well. FP8 breaks the assumption fundamentally.

### 1. Dynamic range collapse in the stabilization policy

CONCEPT.md §3.4 specifies the safety ceiling as:

$$T_{\text{safety}_j} = \frac{\text{FP\_FORMAT\_MAX}}{K_j}$$

For E4M3 with `FP_FORMAT_MAX ≈ 448`:

| Fan-in K | $T_{\text{safety}_j}$ | Consequence |
|:---|:---|:---|
| 2 | 224 | |
| 4 | 112 | Quadratic Scaling Policy funnel severely constricted |
| 8 | 56 | Most gradient vectors already clip at safety ceiling |
| 16 | 28 | Policy threshold $T_{\text{algorithmic}} + \lambda j^2$ is dominated by safety clamp |
| 64 | 7 | Nearly all gradient information destroyed |

The Quadratic Scaling Policy (CONCEPT.md §3.3) requires headroom between $T_{\text{algorithmic}}$ and $T_{\text{safety}}$ to shape the threshold funnel across reduction stages. At E4M3 precision, the safety ceiling is so low that the policy degenerates to uniform hard-clipping at every stage — exactly the "single, compromised threshold" the architecture was designed to avoid.

E5M2 alleviates range pressure (`FP_FORMAT_MAX ≈ 57344`, comparable to FP16's `65504`) but at the cost of mantissa resolution — only ~2 decimal digits of precision, exacerbating accumulation error.

### 2. In-kernel Softmax and loss computation

Node (6) `compute_probs_loss_cce_chunk` performs a "complete, temperature-aware, numerically stable Softmax calculation internally" (CONCEPT.md §Kernel & Synchronization Contracts). The numerically stable log-sum-exp formulation requires:

1. **Subtraction of the row-maximum** from each logit — this preserves relative magnitudes but the result must be representable. With only 3–4 mantissa bits, quantization error dominates the subtracted values.
2. **Exponentiation** of the shifted logits — `exp(x)` maps small differences in input to large differences in output. With E4M3's representable range of only 256 distinct positive values across the dynamic range, the exp outputs quantize to a tiny set of distinct values.
3. **Summation** for normalization — the denominator $\sum \exp(x_i - x_{\max})$ accumulates quantized values, compounding error.
4. **Logarithm** for cross-entropy — $-\log(p)$ amplifies errors in small probabilities. With FP8 probabilities quantized to ~7-bit resolution, the log output is dominated by quantization noise.

Node (7) `compute_probs_loss_bce_chunk` has the same issue with its two-branch stable Sigmoid implementation. The positive/negative logit split that prevents `exp()` overflow in FP16/FP32 still produces meaningful results because those formats have enough mantissa bits to represent the Sigmoid's smooth transition. In FP8, the Sigmoid output is a step function over most of its range.

These kernels perform their computations "internally" per their contracts — the intermediate precision is an Execution-tier concern. But the current kernel source strategy (ADR-013) uses `SCALAR_TYPE` for all intermediate variables. No mechanism exists to specify that a kernel's internal arithmetic should use a wider type than its storage type.

### 3. Adam optimizer moment tracking

The Adam update kernel (Node 24) maintains exponential moving averages `m1` and `m2` on-device at `SCALAR_TYPE` precision. These moment vectors have the lifecycle role `MODEL_STATE` (persisting across batches per ADR-009).

In FP8, the exponential moving average degenerates:

- **E4M3:** With machine epsilon ≈ 0.125 (12.5% relative error per operation), the EMA cannot track slow gradient trends. Moment values quantize to a sparse grid where $\beta_1 \cdot m_1 + (1 - \beta_1) \cdot g$ rounds to either $m_1$ or $g$ for most inputs — the smooth averaging that Adam depends on is lost.
- **E5M2:** Machine epsilon ≈ 0.25 (25% relative error). Worse — the moment vectors are essentially 2-bit quantized noise.

The host computes `beta1**t` and `beta2**t` in FP64 (CONCEPT.md §7), protecting bias correction. But the bias-corrected moment estimates are still applied to FP8 moment vectors — the correction is precise, but the data it corrects is not.

### 4. Reduction tree arithmetic

The `log_K(N)` Recursive Reduction Engine (CONCEPT.md §2) compounds mantissa loss at each stage. For FP8 E4M3 with 3 mantissa bits:

- **Stage 0:** Sum K partials → up to $\log_2(K)$ bits of precision needed for exact sum. For $K = 4$, this requires 5 bits (3 mantissa + 2 from summation) — already exceeding E4M3's capacity.
- **Stage 1+:** Each subsequent stage sums previously-summed values whose mantissa bits are already exhausted.

The clipping at each stage prevents overflow but cannot prevent mantissa saturation. After 2–3 stages, the accumulated gradient is effectively a random value within the clipped range.

### 5. Toolchain gaps

| Layer | FP32 | FP16 | FP8 (E4M3/E5M2) |
|:---|:---|:---|:---|
| **C type** | `float` | `_Float16` (C23) | No standard type. GCC/Clang trunk have `__bf16` but not E4M3/E5M2. |
| **NumPy dtype** | `numpy.float32` | `numpy.float16` | Not in NumPy. Requires `ml_dtypes.float8_e4m3fn` / `float8_e5m2`. |
| **OpenCL** | `float` | `half` | No FP8 type in OpenCL 1.2 or 3.0 spec. |
| **GLSL/SPIR-V** | `float` | `float16_t` | No FP8 type in Vulkan compute shaders (as of Vulkan 1.3). |
| **CPU SIMD** | AVX2 `__m256` | F16C `_mm256_cvtph_ps` | No standard FP8 SIMD intrinsics. AMX-FP8 exists but is matrix-multiply only and requires Granite Rapids+. |

The build system (ADR-014) uses Meson with `aec_cpu_kernel_precision` as a `combo` option with choices `['fp32', 'fp16']`. The multi-precision instantiation in `cpu_precision.h` defines `REAL_T` and `PRECISION_SUFFIX` for each supported type — adding FP8 requires a concrete C storage type and SIMD load/store functions.

The `PrecisionConfig` factory's `_DTYPE_EPSILON_MAP` (ADR-008 Decision §Factory function) rejects unknown dtypes. FP8 dtypes from `ml_dtypes` are not in this map. This is a trivial extension point, but the factory's epsilon assignment requires a deliberate choice — there is no established "operational epsilon" convention for FP8.

### 6. The mixed-precision opportunity

FP8's practical value in the ML ecosystem is not as a uniform compute type but as a **storage and bandwidth format** in a mixed-precision pipeline:

- **Storage:** Weights and activations stored in FP8 for 2× memory reduction vs. FP16.
- **Matmul:** FP8 inputs, FP16/FP32 accumulation (hardware tensor cores perform this natively).
- **Softmax/loss/gradient norms:** FP16 or FP32 intermediate precision.
- **Optimizer state:** FP32 (or at minimum FP16) for moment vectors.

This pattern is standard in frameworks (PyTorch `torch.float8_e4m3fn` with `torch.autocast`, TransformerEngine's FP8 recipe). The architecture currently has no mechanism to express this: `PrecisionConfig` carries one dtype, and every buffer/kernel uses it uniformly.

### 7. Architectural Elegance Feedback

Per CONCEPT.md §1:

> *When emergent efficiency gains contradict current constraints:*
> 1. *Suspend implementation of the optimization*
> 2. *Formalize the pattern as a documented architectural primitive*
> 3. *Reify the optimization through revised contracts & DAG extensions*

FP8 support cannot be achieved by substituting `SCALAR_TYPE = float8` into the existing uniform pipeline — the result is numerically non-functional. The optimization (FP8's bandwidth and memory benefits) contradicts the current constraint (uniform `SCALAR_TYPE` throughout). Per §1, this triggers formalization of a mixed-precision execution model as a new first-class primitive.

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** FP8 support that violates the system's numerical contracts is not an acceptable optimization. The mixed-precision pattern must be formalized before implementation proceeds.

2. **CONCEPT.md §2 (Primacy of Memory Strategy).** FP8's primary benefit is bandwidth reduction — directly aligned with the Primacy of Memory Strategy. The architecture should be able to exploit narrower data paths for bandwidth-bound operations while preserving wider arithmetic where precision is required.

3. **ADR-008 (PrecisionConfig — composition-based, dtype-indexed).** The `PrecisionConfig` frozen dataclass is the sole precision authority in the shared layer. Any FP8 extension must either extend this dataclass or introduce a supplementary configuration that composes alongside it. The composition-over-inheritance principle established by ADR-008 must be preserved.

4. **ADR-001 (Three-Tier Jurisdictional Model — Policy/Orchestration/Execution separation).** The decision of which operations use FP8 storage vs. FP16/FP32 arithmetic is a Policy-tier concern (it affects numerical correctness). The mechanism by which a backend implements mixed-precision dispatch (e.g., hardware tensor cores, software widening) is an Orchestration/Execution-tier concern.

5. **ADR-007 (KernelContract/KernelBinding Split).** Kernel contracts specify parameter types symbolically (`SCALAR_TYPE`, `REAL`). A mixed-precision model requires expressing that some buffers carry a narrow storage type while the kernel computes in a wider type. The contract vocabulary must be extended to distinguish storage precision from compute precision.

6. **ADR-013 (Kernel Source Strategy).** Kernel sources use `SCALAR_TYPE` uniformly. A mixed-precision model requires either (a) additional type macros (e.g., `STORAGE_TYPE` vs. `COMPUTE_TYPE`) or (b) explicit widening/narrowing conversion functions at kernel boundaries.

7. **CONTRACT.md Article 6 (Mandatory Build-Time Symbols).** `SCALAR_TYPE`, `SCALAR_IS_HALF`, and `NUMERICAL_STABILITY_EPSILON` are the current build-time type symbols. A mixed-precision model may require additional symbols (e.g., `STORAGE_TYPE`, `COMPUTE_TYPE`, `STORAGE_IS_FP8`).

8. **ADR-014 (Build System Integration — Meson).** The `aec_cpu_kernel_precision` combo option governs the C `REAL_T` typedef. FP8 requires either a new combo value backed by a concrete C type or a fundamentally different instantiation strategy.

---

## Options Considered

### Option A: Uniform FP8 — add E4M3/E5M2 as `PrecisionConfig` dtypes alongside FP32/FP16

Extend the `_DTYPE_EPSILON_MAP` to include `ml_dtypes.float8_e4m3fn` and `ml_dtypes.float8_e5m2`. Add FP8 instantiation blocks in the CPU backend's `.c` files. Let `SCALAR_TYPE` be the FP8 type throughout, identical to how FP16 is supported today.

```python
_DTYPE_EPSILON_MAP: Dict[type, float] = {
    numpy.float32: 1e-7,
    numpy.float16: 1e-3,
    ml_dtypes.float8_e4m3fn: 0.25,
    ml_dtypes.float8_e5m2: 0.5,
}
```

**Advantages:**
- **Zero architectural changes.** `PrecisionConfig`, `KernelContract`, `KernelBinding`, build system, plan model — all unchanged. FP8 is "just another dtype."
- **Smallest implementation scope.** Only the precision map, factory function, and per-backend type mapping need updating (plus the toolchain prerequisites).
- **Uniform mental model.** No new concepts for users or developers to learn.

**Disadvantages:**
- **Numerically non-functional** for the reasons detailed in §Context items 1–4. The safety ceiling collapses, Softmax produces quantized noise, Adam moments degenerate, and reduction tree arithmetic loses all mantissa information. This option produces an engine that "runs" but does not "train."
- **`epsilon` is meaningless.** With E4M3 machine epsilon at 0.125, an operational epsilon of 0.25 (wider than machine epsilon) means the stability guard is larger than many legitimate gradient values. The epsilon concept breaks down when the format cannot represent small differences.
- **Toolchain prerequisites unsatisfied.** No standard C type, no OpenCL type, no GLSL type, no standard SIMD intrinsics. Each backend would need software emulation or format-specific conversion logic — all hidden inside what appears to be a "uniform" precision, creating substantial hidden complexity.

### Option B: `MixedPrecisionConfig` — formalize storage/compute/state precision roles

Introduce a new `MixedPrecisionConfig` frozen dataclass that extends the precision model with three distinct roles:

```python
@dataclass(frozen=True)
class MixedPrecisionConfig:
    storage_dtype: type        # Buffer/bandwidth type (e.g., float8_e4m3fn)
    compute_dtype: type        # Arithmetic accumulation type (e.g., float16)
    state_dtype: type          # Optimizer state type (e.g., float32)
    storage_fp_format_max: float
    compute_fp_format_max: float
    compute_epsilon: float
```

The plan model gains awareness of per-buffer precision roles. `KernelContract` extends its buffer parameter specs with a `precision_role` field. Kernels that receive FP8 storage buffers are contractually obligated to widen to `compute_dtype` before arithmetic and narrow on write.

**Advantages:**
- **Correct formalization of FP8's actual use pattern.** Storage, compute, and state precisions are independently specified. The Softmax kernel computes in FP16/FP32 even when its inputs are stored in FP8. Adam moment vectors live at `state_dtype` regardless of the forward pass's storage format.
- **Safety ceiling operates on `compute_dtype`.** The stabilization policy uses `compute_fp_format_max` (FP16's 65504 or FP32's 3.4e38), not the storage format's collapsed range. The Quadratic Scaling Policy funnel operates normally.
- **Policy-tier concern.** The decision of which buffers use narrow storage vs. wide compute is captured in the plan — inspectable, validatable, and backend-neutral.
- **Extensible.** BFloat16 storage with FP32 compute, or FP16 storage with FP32 state, are naturally expressed. The model supports any combination of precisions without combinatorial subclass explosion.
- **Aligns with hardware reality.** NVIDIA Hopper's FP8 tensor cores natively perform `FP8×FP8→FP16/FP32` accumulation. AMD's MI300 does the same. The mixed-precision model maps directly to hardware capabilities.

**Disadvantages:**
- **Significant architectural extension.** The plan model, `KernelContract` vocabulary (ADR-007), kernel source macros (ADR-013), build-time symbols (CONTRACT.md Article 6), and every backend's rendering/binding layer must all be extended.
- **Two precision configurations coexist.** Existing FP32/FP16 uniform pipelines would use the original `PrecisionConfig`; FP8 mixed pipelines would use `MixedPrecisionConfig`. Alternatively, `PrecisionConfig` is generalized to subsume both models, but this changes its interface for existing consumers.
- **Kernel interface complexity.** Kernels must now distinguish between storage-precision buffer parameters and compute-precision scalar parameters. Widening/narrowing conversion functions become a contract obligation, adding verification surface area.
- **Validation complexity.** The plan builder must verify consistency between storage, compute, and state dtypes (e.g., `storage_dtype.itemsize <= compute_dtype.itemsize`). Cross-role consistency is a new class of plan-time validation that does not exist today.
- **Toolchain prerequisites remain.** The storage-side FP8 type still needs a C representation, a NumPy-compatible dtype, and per-backend load/store functions. The compute side uses existing FP16/FP32 types, but the conversion boundary is new.

### Option C: Deferred — document constraints, do not extend

Formally document that FP8 is outside the system's supported precision range. Record the specific numerical, toolchain, and architectural blockers. Define the minimal preconditions under which FP8 support could be reconsidered (toolchain maturity, mixed-precision model formalization, hardware survey). Take no implementation action.

**Advantages:**
- **Zero risk.** No architectural changes, no new code, no new failure modes.
- **Honest scope.** The system's precision support is FP16–FP64, covering the vast majority of inference and training workloads for the classification task this architecture targets.
- **Preserves development bandwidth.** The ADR-017 incremental migration path has six phases, many not yet complete. Introducing a mixed-precision model before the baseline architecture is fully realized would compound implementation risk.
- **Toolchains may mature.** By the time the base architecture is complete, C23 `_Float8` types, OpenCL FP8 extensions, or Vulkan FP8 format support may materialize — reducing the implementation burden of a future FP8 ADR.

**Disadvantages:**
- **Misses the formalization opportunity.** Per CONCEPT.md §1, the architectural response to "optimization contradicts constraints" is to formalize the pattern — not to decline all action. Even without implementing FP8, formalizing the mixed-precision model as a future extension point documents the architectural pathway.
- **No forward compatibility.** Without even a provisional `MixedPrecisionConfig` concept, future precision extensions must retroactively extend the plan model, contracts, and build system with no established design vocabulary.

---

## Analysis

### Eliminating Option A

Option A is numerically non-functional. The detailed analysis in §Context items 1–4 demonstrates that uniform FP8 breaks the system's core numerical contracts:

- The stabilization policy's safety ceiling collapses (§1), destroying the Quadratic Scaling funnel.
- Softmax and Sigmoid lose representational fidelity (§2), producing step-function outputs.
- Adam moment tracking degenerates to quantized noise (§3).
- Reduction tree arithmetic exhausts mantissa bits within 2–3 stages (§4).

An engine configured with uniform FP8 would produce gradient updates that are numerically indistinguishable from random noise. This violates the system's most fundamental guarantee — that the mathematical algorithm specified in `kernels.cl.h` is faithfully executed. Option A is eliminated.

### Comparing Options B and C

The choice between B and C is a question of timing, not direction. Both acknowledge that FP8 requires a mixed-precision model. Option B builds that model now; Option C defers it.

**Arguments for deferring (favoring C):**

1. **Incomplete base architecture.** ADR-017's migration path is the active development track. Phases 1–6 cover plan construction, per-backend kernel migration, renderer integration, and cross-backend parity validation. The baseline FP32/FP16 uniform pipeline is not yet fully realized across all backends. Introducing a second precision axis before the first is stable creates a multiplicative testing burden ($\text{backends} \times \text{precision\_models} \times \text{kernel\_inventory}$).

2. **Toolchain immaturity.** No standard C type for E4M3/E5M2 exists in released compilers. The `ml_dtypes` Python package provides NumPy-compatible FP8 dtypes, but the CPU backend's native C kernels (`cpu_precision.h`, `cpu_simd.h`) have no FP8 load/store path. Implementing software FP8↔FP32 conversion in C without hardware intrinsics is possible but non-trivial and performance-negative — undermining the bandwidth motivation for FP8 in the first place.

3. **Hardware coverage.** FP8 tensor core support exists on NVIDIA Hopper and AMD MI300 — both are GPU architectures. This system's three backends are OpenCL (GPU), Vulkan (GPU), and CPU (SIMD). The CPU backend, which serves as the deterministic Tier 3 parity reference oracle (CONCEPT.md §9), has no FP8 hardware support on any mainstream ISA. A mixed-precision CPU backend would require software widening on every buffer access — correct but slow, reducing the CPU backend's value as a practical testing oracle.

**Arguments for formalizing now (favoring B):**

1. **CONCEPT.md §1 compliance.** The principle is explicit: when an optimization contradicts constraints, "formalize the pattern as a documented architectural primitive." FP8's bandwidth optimization contradicts the uniform-precision constraint. The architecturally correct response is formalization, even if implementation is deferred.

2. **Design vocabulary establishment.** If `MixedPrecisionConfig` is formalized as a concept — even without implementation — subsequent ADRs and contract extensions can reference it. Future work on BFloat16, or on FP16-storage/FP32-compute mixed pipelines (which are relevant even without FP8), has a design vocabulary to build on.

3. **Forward compatibility in ADR-008.** ADR-008's `PrecisionConfig` is currently a closed design — `numpy_dtype`, `fp_format_max`, `epsilon`, no extension points. Documenting the mixed-precision extension path now ensures that future changes to `PrecisionConfig` are anticipated rather than disruptive.

---

## Decision

**Option C, with the formalization provisions of Option B recorded as a forward-looking design sketch.**

FP8 support is not implemented at this time. The system's supported precision range remains FP16–FP64 as configured by ADR-008's `PrecisionConfig`. This decision is driven by three concrete blockers:

1. **The base architecture (ADR-017) is not yet fully realized.** Adding a second precision axis before Tier 3 cross-backend parity is achieved for the existing FP32/FP16 pipeline would compound risk without delivering a functional FP8 capability.
2. **Toolchain prerequisites are unmet.** No standard C FP8 type, no OpenCL/GLSL FP8 type, no CPU SIMD FP8 intrinsics.
3. **FP8 is numerically non-functional under the current uniform-precision model** (§Analysis, eliminating Option A).

### Recorded preconditions for FP8 enablement

The following preconditions, when satisfied, constitute the trigger for a follow-up ADR that implements the mixed-precision model:

| Precondition | Rationale |
|:---|:---|
| ADR-017 Phase 6 complete (Tier 3 parity gate green for FP32 and FP16). | The baseline must be stable before introducing a new precision axis. |
| A C storage type for E4M3 or E5M2 is available in released GCC or Clang. | The CPU backend requires a concrete `REAL_T` for multi-precision instantiation. Software-only emulation is acceptable for correctness but must use a real type, not `uint8_t` with manual bit manipulation. |
| `ml_dtypes` (or equivalent) FP8 dtype is stable and interoperable with NumPy. | `PrecisionConfig.numpy_dtype` must be a type that supports `numpy.finfo()`, `numpy.dtype().itemsize`, and array construction. |
| At least one GPU backend (OpenCL or Vulkan) has a viable FP8 dispatch path. | Either an OpenCL extension for 8-bit float load/store, a Vulkan `VK_KHR_8bit_storage`-compatible compute dispatch, or a demonstrated pattern for FP8 storage with FP16 compute via existing format support. |

### Mixed-precision model design sketch

The following design sketch records the anticipated shape of the mixed-precision extension. It is not binding — a future ADR may revise it based on toolchain and hardware developments. It is recorded here to establish design vocabulary and ensure forward compatibility.

#### `MixedPrecisionConfig` concept

```python
@dataclass(frozen=True)
class MixedPrecisionConfig:
    """Precision configuration for mixed-precision execution pipelines.

    Extends the uniform PrecisionConfig model with three distinct precision roles.
    """
    storage_dtype: type        # Narrow format for buffers/bandwidth (e.g., float8_e4m3fn)
    compute_dtype: type        # Wider format for arithmetic (e.g., float16, float32)
    state_dtype: type          # Format for persistent optimizer state (e.g., float32)

    storage_fp_format_max: float   # For storage-side safety bounds
    compute_fp_format_max: float   # For stabilization policy (T_safety_j)
    compute_epsilon: float         # For numerical stability guards
```

#### Kernel contract extension concept

`KernelContract` buffer parameter specs gain an optional `precision_role` field:

| `precision_role` | Semantics |
|:---|:---|
| `"storage"` | Buffer is at `storage_dtype`. Kernel must widen to `compute_dtype` before arithmetic. |
| `"compute"` | Buffer is at `compute_dtype`. Standard arithmetic applies. |
| `"state"` | Buffer is at `state_dtype`. Used for `MODEL_STATE` lifecycle buffers (Adam moments). |
| `None` | Uniform precision — current behavior. `PrecisionConfig.numpy_dtype` applies. |

When `precision_role` is `"storage"`, the kernel's behavioral invariants must specify the widening conversion as a contractual obligation — not an optional optimization.

#### Build-time symbol extension concept

In addition to the existing `SCALAR_TYPE`, `SCALAR_IS_HALF`, `NUMERICAL_STABILITY_EPSILON`:

| Symbol | Role |
|:---|:---|
| `STORAGE_TYPE` | C type for narrow storage buffers (e.g., `__fp8_e4m3`) |
| `COMPUTE_TYPE` | C type for arithmetic (e.g., `float`, `_Float16`) |
| `STORAGE_IS_FP8` | Boolean flag for conditional conversion logic |

#### Plan model impact

The `BufferDescriptor` (ADR-009) gains an optional `precision_role` field matching the kernel contract extension. The plan builder annotates each buffer with its precision role based on the `MixedPrecisionConfig`. Backends use this annotation to select the correct native type for buffer allocation.

#### Relationship to `PrecisionConfig`

`MixedPrecisionConfig` does not replace `PrecisionConfig`. The two coexist:

- **Uniform pipelines** (FP16, FP32, FP64) continue to use `PrecisionConfig`. All `precision_role` annotations are `None`. This is the current model — no changes.
- **Mixed-precision pipelines** (FP8 storage + FP16/FP32 compute) use `MixedPrecisionConfig`. The plan builder annotates buffers and kernels with precision roles. Backends implement widening/narrowing at buffer access boundaries.

A future ADR may generalize `PrecisionConfig` to subsume `MixedPrecisionConfig` (e.g., by making `PrecisionConfig` a special case where `storage_dtype == compute_dtype == state_dtype`). This unification is deferred to the implementing ADR, as it depends on implementation experience.

---

## Consequences

### Positive

- **Numerically correct by non-action.** The system does not gain an FP8 mode that produces incorrect results. Users are not exposed to a "supported" precision that fails silently.
- **CONCEPT.md §1 compliance.** The architectural response to FP8's contradiction of the uniform-precision model is formalized — the mixed-precision pattern is documented as the recognized extension path, not an ad-hoc future improvisation.
- **Design vocabulary established.** `MixedPrecisionConfig`, `precision_role`, `storage_dtype`/`compute_dtype`/`state_dtype` — these terms are now part of the project's architectural lexicon. Future ADRs, discussions, and designs can reference them without re-deriving the concept.
- **ADR-008 preserved.** The existing `PrecisionConfig` is unchanged. FP32 and FP16 uniform pipelines are unaffected. No refactoring of existing code is triggered.
- **Preconditions are concrete and testable.** The four enablement preconditions provide a clear checklist — not vague "when toolchains mature" language. Each precondition can be evaluated mechanically.

### Negative

- **No FP8 capability.** Users or workloads that could benefit from FP8's bandwidth reduction cannot use this system for that purpose until the preconditions are met and the follow-up ADR is implemented.
- **Design sketch is non-binding.** The `MixedPrecisionConfig` concept may prove inadequate when actual implementation begins. The sketch captures current understanding but not implementation experience.
- **Does not address FP16-storage/FP32-compute.** The mixed-precision model is valuable even without FP8 — FP16 storage with FP32 accumulation is a common pattern that improves numerical stability while preserving bandwidth benefits. This ADR does not implement that mode either, though the design sketch's formalization makes it a natural first step for a future implementing ADR.

### References

- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` frozen dataclass; precision-derived tolerances
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — Three-tier jurisdictional model
- [ADR-007: Kernel Signature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract`/`KernelBinding` separation
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` lifecycle roles
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — `SCALAR_TYPE` build-time symbols
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — Meson precision options
- [ADR-017: Incremental Migration Path](ADR-017-incremental-migration-path.md) — Phased implementation gates
