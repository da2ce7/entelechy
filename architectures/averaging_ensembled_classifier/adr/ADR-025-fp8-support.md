# ADR-025: FP8 (E4M3/E5M2) Storage Precision Support

**Status:** PROPOSED
**Date:** 2026-04-02
**Deciders:** —
**Triggered by:** ADR-020 identified FP8 as the motivating case for the three-role precision model
**Depends on:** ADR-020 (Three-Role Precision Model), ADR-021, ADR-022, ADR-023, ADR-024
**Extends:** ADR-020 §4.1 (PrecisionConfig factories), ADR-023 (backend implementations)
**Constrains:** FP8 is storage-role only; FP8 compute and FP8 state are architecturally prohibited

---

## Context

ADR-020 identified FP8 as the inflection point that made the uniform `SCALAR_TYPE` model untenable:

> FP8 (E4M3/E5M2) makes the incompleteness catastrophic. Uniform FP8 produces a safety ceiling of 448/K (the policy funnel degenerates), Softmax over ~256 distinct positive values, EMA updates that round to either m or g, and reduction mantissa saturation after one stage. FP8 as a uniform SCALAR_TYPE produces an engine that runs but does not train.

The three-role precision model (ADR-020) was designed specifically to enable FP8. Now that the model is formalized and the backend infrastructure supports role-specific precision (ADR-021 through ADR-024), FP8 can be added as the ultimate expression of the Primacy of Memory Strategy: **maximum bandwidth compression for storage-role buffers while preserving compute and state fidelity.**

### FP8 Format Variants

IEEE 754 does not define 8-bit floating-point. Two de facto standards have emerged from ML accelerator implementations:

| Format | Sign | Exponent | Mantissa | Bias | Max Value | Min Subnormal | Use Case |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **E4M3** | 1 | 4 | 3 | 7 | 448 | 2⁻⁹ ≈ 0.00195 | Activations, gradients (range-limited) |
| **E5M2** | 1 | 5 | 2 | 15 | 57344 | 2⁻¹⁶ ≈ 0.0000153 | Wider dynamic range, lower precision |

E4M3 provides 3 mantissa bits (8 distinct values per exponent binade) with a maximum representable value of 448. E5M2 provides 2 mantissa bits (4 distinct values per binade) but extends the dynamic range to 57344. Neither format supports infinity or NaN in their standard ML definitions — the bit patterns are repurposed for additional finite values.

**E4M3 is the primary target.** Its higher precision (3 mantissa bits vs. 2) better preserves gradient direction vectors and activation magnitudes. The 448 maximum is manageable under the architecture's existing Quadratic Scaling Policy — gradients are already scaled to prevent overflow at `COMPUTE_FP_FORMAT_MAX / K`, and post-scaling values fit within E4M3's range for practical K values.

**E5M2 is a secondary variant.** Its wider dynamic range is useful for gradient partials that span many orders of magnitude before aggregation. The architecture supports both variants as distinct storage dtypes.

### Why FP8 is Storage-Only

ADR-020's role definitions make the constraint obvious:

| Role | Optimization Target | FP8 Viability |
|:---|:---|:---|
| **Storage** | Bandwidth — narrower is better | ✓ FP8 halves FP16's bandwidth |
| **Compute** | Fidelity — wider is better | ✗ 3-bit mantissa saturates in one reduction stage |
| **State** | Stability — wider is better | ✗ EMA updates round to either m or g |

**FP8 compute is numerically non-functional.** The $\log_K(N)$ Recursive Reduction Engine requires $\log_2(K)$ additional mantissa bits beyond the summands' precision for exact accumulation. E4M3's 3-bit mantissa provides ~1 decimal digit of precision. After one reduction stage with K=8, the accumulated error exceeds the signal. Softmax over FP8 intermediates produces a categorical distribution with at most 8 distinguishable probability levels per exponent band — insufficient for gradient computation.

**FP8 state is numerically erosive.** Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires representing $(1 - \beta_1) \cdot g$ as a distinguishable increment. For $\beta_1 = 0.9$ and a gradient of magnitude 1.0, the increment is 0.1 — representable in E4M3. For $\beta_1 = 0.999$, the increment is 0.001 — below E4M3's minimum subnormal (0.00195), rounding to zero. Moment vectors would stagnate after initial convergence.

The three-role model exists precisely to decouple these concerns. FP8 storage with FP32 compute and FP32 state achieves:
- 4× bandwidth reduction vs. FP32 storage (8 bits vs. 32 bits)
- Full FP32 arithmetic fidelity in reductions and transcendentals
- Full FP32 optimizer stability across unbounded training steps

---

## Decision Drivers

1. **The Primacy of Memory Strategy demands the narrowest viable storage format.** FP8 is 2× narrower than FP16 and 4× narrower than FP32. For bandwidth-bound workloads (which the architecture explicitly targets), FP8 storage provides the maximum benefit.

2. **The three-role model was designed for this.** ADR-020's motivating example was FP8. The infrastructure exists; this ADR completes the original intent.

3. **FP8 compute and state produce non-functional training.** This is not a performance tradeoff — it is a correctness boundary. The architecture must enforce the storage-only constraint at the configuration level, not rely on user discipline.

4. **Hardware support is emerging but not universal.** NVIDIA H100/H200, AMD MI300, Intel Gaudi provide native FP8 ALUs. Older hardware requires software emulation. The architecture must support both paths through the backend abstraction boundary.

5. **Two FP8 variants require explicit selection.** E4M3 and E5M2 have different tradeoffs. The user must choose; the architecture does not auto-select.

---

## Decision

FP8 (E4M3 and E5M2) is added as a supported precision format for the **storage role only**. Attempts to configure FP8 for compute or state roles are rejected at construction time. The following sections specify the amendments.

---

### §1: NumPy dtype Representation

NumPy 2.0+ provides `np.float8_e4m3fn` and `np.float8_e5m2` dtypes via the ml_dtypes package. The architecture adopts these as the canonical FP8 representations:

```python
import ml_dtypes
import numpy as np

# E4M3 (no infinity, no NaN — all bit patterns are finite)
FP8_E4M3 = ml_dtypes.float8_e4m3fn

# E5M2 (standard ML definition)
FP8_E5M2 = ml_dtypes.float8_e5m2
```

The `ml_dtypes` package is added as a runtime dependency. It provides:
- NumPy dtype objects compatible with array operations
- Python scalar arithmetic (with FP32 promotion)
- Serialization via standard NumPy `.npy` format

---

### §2: PrecisionConfig Extension (amends ADR-020 §4.1)

#### §2.1: Storage dtype Expansion

The `storage_dtype` field accepts FP8 variants in addition to the existing FP16/FP32/FP64:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    storage_dtype: np.dtype   # float8_e4m3fn, float8_e5m2, float16, float32, float64
    compute_dtype: np.dtype   # float32, float64 (FP8/FP16 compute prohibited on CPU)
    state_dtype: np.dtype     # float32, float64 (FP8/FP16 state prohibited)
```

#### §2.2: Construction Invariants (amended)

The `__post_init__` validation is amended to enforce the storage-only constraint:

```python
def __post_init__(self) -> None:
    # FP8 is storage-role only
    if self.compute_dtype in (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2):
        raise ValueError(
            "FP8 compute is architecturally prohibited: "
            "3-bit mantissa saturates in one reduction stage"
        )
    if self.state_dtype in (ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2):
        raise ValueError(
            "FP8 state is architecturally prohibited: "
            "EMA updates round to zero for β > 0.9"
        )
    
    # Existing invariants (storage ≤ compute, storage ≤ state) still apply
    assert self.storage_dtype.itemsize <= self.compute_dtype.itemsize
    assert self.storage_dtype.itemsize <= self.state_dtype.itemsize
```

Note: The itemsize invariant naturally holds (FP8 = 1 byte, FP32/FP64 = 4/8 bytes).

#### §2.3: New Factory Classmethods

| Factory | `storage_dtype` | `compute_dtype` | `state_dtype` | Primary Use Case |
|:---|:---|:---|:---|:---|
| `PrecisionConfig.fp8_e4m3()` | `float8_e4m3fn` | `float32` | `float32` | Maximum bandwidth, standard training |
| `PrecisionConfig.fp8_e5m2()` | `float8_e5m2` | `float32` | `float32` | Maximum bandwidth, wider gradient range |
| `PrecisionConfig.fp8_e4m3_f64_state()` | `float8_e4m3fn` | `float32` | `float64` | Maximum bandwidth + extended stability |

The uniform `PrecisionConfig.fp8()` factory is **not provided**. There is no such thing as uniform FP8 — compute and state are always wider. Attempting to create a config with FP8 compute or state raises `ValueError`.

#### §2.4: Derived Constants

| Field | E4M3 Value | E5M2 Value | Source |
|:---|:---|:---|:---|
| `storage_fp_format_max` | `448.0` | `57344.0` | Format-specific maximum finite value |
| `storage_fp_format_min_subnormal` | `0.001953125` | `0.0000152588` | Smallest representable positive value |
| `storage_mantissa_bits` | `3` | `2` | Precision for quantization analysis |

These values do not affect compute or state operations — they govern storage buffer sizing and the host's quantization/dequantization logic.

---

### §3: CONTRACT.md Article 6 Extension (amends ADR-020 §3.6, ADR-024 §2)

The build-time symbol vocabulary is extended:

| Symbol | Definition | Category |
|:---|:---|:---|
| `STORAGE_TYPE_IS_FP8` | Integer flag (0 or 1). True when `STORAGE_TYPE` is an 8-bit float format. Governs FP8-specific load/store mechanics. | Derived |
| `STORAGE_TYPE_IS_E4M3` | Integer flag (0 or 1). True when `STORAGE_TYPE` is specifically E4M3. | Derived |
| `STORAGE_TYPE_IS_E5M2` | Integer flag (0 or 1). True when `STORAGE_TYPE` is specifically E5M2. | Derived |

**Note:** `COMPUTE_TYPE_IS_FP8` and `STATE_TYPE_IS_FP8` are **not defined** because FP8 compute and state are architecturally prohibited. The build system does not emit these symbols.

The complete mandatory symbol set (extending ADR-024 §2) becomes:

| Symbol | Category | FP8 Notes |
|:---|:---|:---|
| `STORAGE_TYPE` | Primary | May be `__nv_fp8_e4m3` (CUDA), software struct (CPU), etc. |
| `COMPUTE_TYPE` | Primary | Always FP32 or FP64 when storage is FP8 |
| `STATE_TYPE` | Primary | Always FP32 or FP64 when storage is FP8 |
| `STORAGE_TYPE_IS_HALF` | Derived | 0 when FP8 |
| `STORAGE_TYPE_IS_FP8` | Derived | 1 when E4M3 or E5M2 |
| `STORAGE_TYPE_IS_E4M3` | Derived | 1 when E4M3 specifically |
| `STORAGE_TYPE_IS_E5M2` | Derived | 1 when E5M2 specifically |
| `COMPUTE_TYPE_IS_HALF` | Derived | Always 0 when storage is FP8 (FP32/FP64 compute) |
| `COMPUTE_TYPE_IS_DOUBLE` | Derived | May be 1 for validation configs |
| `STATE_TYPE_IS_DOUBLE` | Derived | May be 1 for extended stability |
| `NUMERICAL_STABILITY_EPSILON` | Derived | Derived from COMPUTE_TYPE, unchanged |
| `SIMD_WIDTH` | Hardware | In COMPUTE_TYPE elements |
| `C_TILE_SIZE` | Configuration | Unchanged |

---

### §4: Precision Boundary Abstractions for FP8

#### §4.1: `kernels.cl.h` Extension

The precision boundary abstraction declarations (ADR-021 §1.3) are extended:

```c
// --- FP8 Precision Boundary Abstractions (ADR-025) ---
// FP8 is storage-role only. These functions convert between FP8 storage
// and COMPUTE_TYPE (FP32 or FP64) arithmetic.

#if STORAGE_TYPE_IS_FP8

static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx);

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val);

#endif
```

The function signatures are identical to FP16/FP32 — the precision boundary abstraction is type-agnostic. The implementation differs per backend.

#### §4.2: Quantization Semantics

**Load (FP8 → COMPUTE_TYPE):** Widening conversion. No precision loss. The 8-bit value is exactly representable in FP32/FP64.

**Store (COMPUTE_TYPE → FP8):** Narrowing conversion with round-to-nearest-even. Values exceeding `STORAGE_FP_FORMAT_MAX` saturate to `±MAX` (no infinity in E4M3/E5M2). Values below minimum subnormal round to zero.

The host is responsible for scaling gradients to fit within the FP8 range *before* kernel execution. The Quadratic Scaling Policy already scales gradients to `COMPUTE_FP_FORMAT_MAX / K`; for FP8 storage, an additional scaling factor is applied such that post-scaling values fit within `STORAGE_FP_FORMAT_MAX`.

---

### §5: CPU Backend — FP8 Software Implementation

#### §5.1: Storage Type Representation

The CPU backend uses a software struct for FP8:

```c
// cpu_precision.h — FP8 representation
typedef struct { uint8_t bits; } cpu_fp8_e4m3;
typedef struct { uint8_t bits; } cpu_fp8_e5m2;
```

The struct wrapper prevents accidental arithmetic on the raw bits. All FP8 operations go through explicit load/store functions.

#### §5.2: Conversion Functions

```c
// cpu_fp8.h — E4M3 conversion functions
float cpu_fp8_e4m3_to_float(cpu_fp8_e4m3 val);
cpu_fp8_e4m3 cpu_float_to_fp8_e4m3(float val);

double cpu_fp8_e4m3_to_double(cpu_fp8_e4m3 val);
cpu_fp8_e4m3 cpu_double_to_fp8_e4m3(double val);

// cpu_fp8.h — E5M2 conversion functions  
float cpu_fp8_e5m2_to_float(cpu_fp8_e5m2 val);
cpu_fp8_e5m2 cpu_float_to_fp8_e5m2(float val);

double cpu_fp8_e5m2_to_double(cpu_fp8_e5m2 val);
cpu_fp8_e5m2 cpu_double_to_fp8_e5m2(double val);
```

These functions implement the IEEE-adjacent FP8 semantics:
- **E4M3:** bias=7, no inf/nan, max=448
- **E5M2:** bias=15, standard inf/nan interpretation optional (ML variant omits them)

The implementation uses lookup tables for FP8→FP32 (256 entries × 4 bytes = 1KB per format) and bit manipulation for FP32→FP8 (extract exponent/mantissa, bias-adjust, round, saturate).

#### §5.3: Three-Axis Suffix Extension (amends ADR-024 §4.1)

New valid CPU suffixes for FP8 storage:

| Suffix | `STORAGE_T` | `COMPUTE_T` | `STATE_T` | Use Case |
|:---|:---|:---|:---|:---|
| `s8e4c32x32` | `cpu_fp8_e4m3` | `float` | `float` | E4M3 bandwidth, standard training |
| `s8e4c32x64` | `cpu_fp8_e4m3` | `float` | `double` | E4M3 bandwidth + extended stability |
| `s8e4c64x64` | `cpu_fp8_e4m3` | `double` | `double` | E4M3 bandwidth, FP64 compute/state |
| `s8e5c32x32` | `cpu_fp8_e5m2` | `float` | `float` | E5M2 bandwidth, standard training |
| `s8e5c32x64` | `cpu_fp8_e5m2` | `float` | `double` | E5M2 bandwidth + extended stability |
| `s8e5c64x64` | `cpu_fp8_e5m2` | `double` | `double` | E5M2 bandwidth, FP64 compute/state |

The `8e4` and `8e5` shorthand distinguishes the two FP8 variants. The `DECLARE_PRECISION_STRUCTS` macro is extended:

```c
/* FP8 E4M3 storage variants */
DECLARE_PRECISION_STRUCTS(s8e4c32x32, cpu_fp8_e4m3, float, float)
DECLARE_PRECISION_STRUCTS(s8e4c32x64, cpu_fp8_e4m3, float, double)
DECLARE_PRECISION_STRUCTS(s8e4c64x64, cpu_fp8_e4m3, double, double)

/* FP8 E5M2 storage variants */
DECLARE_PRECISION_STRUCTS(s8e5c32x32, cpu_fp8_e5m2, float, float)
DECLARE_PRECISION_STRUCTS(s8e5c32x64, cpu_fp8_e5m2, float, double)
DECLARE_PRECISION_STRUCTS(s8e5c64x64, cpu_fp8_e5m2, double, double)
```

#### §5.4: Load/Store Macro Extension

```c
// cpu_precision.h — FP8 storage role load/store
#define scalar_load_storage_fp8_e4m3(ptr) cpu_fp8_e4m3_to_float(*(ptr))
#define scalar_store_storage_fp8_e4m3(ptr, val) (*(ptr) = cpu_float_to_fp8_e4m3(val))

#define scalar_load_storage_fp8_e5m2(ptr) cpu_fp8_e5m2_to_float(*(ptr))
#define scalar_store_storage_fp8_e5m2(ptr, val) (*(ptr) = cpu_float_to_fp8_e5m2(val))
```

SIMD variants are deferred. AVX-512 provides `VCVTPH2PS`/`VCVTPS2PH` for FP16 but no native FP8 instructions. Software SIMD (processing 8 FP8 values as a 64-bit word, converting in parallel via vectorized lookup) is a future optimization.

---

### §6: OpenCL Backend — FP8 Implementation

#### §6.1: Extension Requirements

No standard OpenCL extension defines FP8. Implementations fall into two categories:

1. **Vendor extensions** (e.g., `cl_intel_fp8`, hypothetical): Native FP8 types and conversion functions. The backend queries extension availability at device initialization.

2. **Software emulation**: FP8 values stored as `uchar`. Conversion functions implemented in OpenCL C using bit manipulation. This is the fallback path.

```c
// kernels.cl.h — FP8 software types (fallback)
#if STORAGE_TYPE_IS_FP8 && !defined(CL_FP8_NATIVE)
typedef uchar fp8_e4m3;
typedef uchar fp8_e5m2;

#if STORAGE_TYPE_IS_E4M3
#define STORAGE_TYPE fp8_e4m3
#elif STORAGE_TYPE_IS_E5M2
#define STORAGE_TYPE fp8_e5m2
#endif

// Conversion via constant-memory lookup table
__constant float fp8_e4m3_to_float_lut[256] = { /* ... */ };
__constant float fp8_e5m2_to_float_lut[256] = { /* ... */ };

static inline float load_storage_fp8(__global const uchar *buf, size_t idx) {
    #if STORAGE_TYPE_IS_E4M3
    return fp8_e4m3_to_float_lut[buf[idx]];
    #else
    return fp8_e5m2_to_float_lut[buf[idx]];
    #endif
}

static inline void store_storage_fp8(__global uchar *buf, size_t idx, float val) {
    // Bit manipulation: extract sign/exp/mantissa, bias-adjust, round, saturate
    // Implementation elided for brevity
}
#endif
```

#### §6.2: `load_storage` / `store_storage` Implementation

When `STORAGE_TYPE_IS_FP8 == 1`:

```c
static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx) {
    #if defined(CL_FP8_NATIVE)
    return convert_float(buf[idx]);  // Vendor extension
    #else
    return load_storage_fp8((__global const uchar*)buf, idx);
    #endif
}

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val) {
    #if defined(CL_FP8_NATIVE)
    buf[idx] = convert_fp8(val);  // Vendor extension
    #else
    store_storage_fp8((__global uchar*)buf, idx, val);
    #endif
}
```

---

### §7: Vulkan Backend — FP8 Shader Variants

#### §7.1: Extension Requirements

Vulkan FP8 support depends on vendor extensions:
- **NVIDIA:** `VK_NV_fp8` (hypothetical, modeled on CUDA `__nv_fp8_e4m3`)
- **AMD:** Part of `VK_AMD_shader_float16_int8` (extended interpretation)

When extensions are unavailable, FP8 buffers are treated as `uint8_t` arrays with software conversion in the shader.

#### §7.2: Specialization Constant Extension

The three-axis specialization constant scheme (ADR-024 §5.1) is extended:

```glsl
// Existing constants
layout(constant_id = 0) const int STORAGE_FLOAT = 1;  // 0=fp8, 1=fp16, 2=fp32, 3=fp64
layout(constant_id = 1) const int COMPUTE_FLOAT = 2;  // 2=fp32, 3=fp64
layout(constant_id = 2) const int STATE_FLOAT = 2;    // 2=fp32, 3=fp64

// New FP8 variant selector (only valid when STORAGE_FLOAT == 0)
layout(constant_id = 3) const int FP8_VARIANT = 0;    // 0=E4M3, 1=E5M2
```

#### §7.3: Software Conversion (Fallback)

```glsl
#if STORAGE_FLOAT == 0 && !defined(VK_FP8_NATIVE)
// FP8 as uint8 with software conversion
float load_storage_fp8(uint8_t bits) {
    // E4M3: sign(1) + exp(4) + mantissa(3), bias=7, max=448
    // Implementation via bit extraction
}

uint8_t store_storage_fp8(float val) {
    // Saturating conversion with round-to-nearest-even
}
#endif
```

---

### §8: Host-Side Scaling for FP8 Storage

FP8's limited dynamic range (448 for E4M3, 57344 for E5M2) requires the host to ensure values fit before kernel execution.

#### §8.1: Activation Scaling

Forward-pass activations (post-ReLU, post-sigmoid, etc.) are inherently bounded:
- ReLU outputs: [0, max_input] — controlled by upstream scaling
- Sigmoid outputs: [0, 1] — always in FP8 range
- Softmax outputs: [0, 1] — always in FP8 range

The primary concern is hidden layer pre-activations (before nonlinearity). The host applies a scale factor `activation_scale` such that `max(abs(pre_activation)) × activation_scale ≤ STORAGE_FP_FORMAT_MAX`. This scale is tracked as DAG metadata and inverted after loading.

#### §8.2: Gradient Scaling

The existing Quadratic Scaling Policy scales gradients to `COMPUTE_FP_FORMAT_MAX / K` for overflow prevention. For FP8 storage, an additional `storage_scale` factor is applied:

```python
storage_scale = STORAGE_FP_FORMAT_MAX / COMPUTE_FP_FORMAT_MAX
# E4M3: 448 / 3.4e38 ≈ 1.3e-36 (extreme narrowing)
# This is inverted: we scale gradients DOWN to fit FP8 range
```

In practice, gradients are already scaled for stability. The FP8 storage path:
1. Computes gradients in `COMPUTE_TYPE` (FP32/FP64)
2. Applies loss scaling (standard mixed-precision technique)
3. Clips to `[−STORAGE_FP_FORMAT_MAX, +STORAGE_FP_FORMAT_MAX]`
4. Stores via `store_storage()` with saturating conversion

The host tracks cumulative scaling for correct weight updates.

#### §8.3: Integration with Stabilization Policy

The Quadratic Scaling Policy's safety ceiling (CONCEPT.md §3.4) is unchanged:

> `T_safety_j = COMPUTE_FP_FORMAT_MAX / K_j`

This governs the *compute* path. FP8 storage does not change the compute precision or the safety ceiling. The storage narrowing happens *after* the stability-guaranteed computation completes.

---

### §9: Validation and Testing (extends ADR-016)

#### §9.1: New Validation Scenario

> **Scenario: The Bandwidth Extremist (FP8 Storage Fidelity)**
>
> - **Description:** A training task is executed with `PrecisionConfig.fp8_e4m3()` and compared against `PrecisionConfig.mixed_f16_f32()` baseline. Both configurations use identical compute (FP32) and state (FP32) precision.
> - **Validation Focus:** Confirms that FP8 storage produces convergent training within tolerance of the FP16 baseline. Validates that storage-role buffer sizes are halved (8 bits vs. 16 bits), that quantization error does not prevent convergence, and that the host's scaling machinery maintains numerical correctness.
> - **Key Insight:** Proves the architectural claim that storage precision is independent of training fidelity. The 2× bandwidth reduction vs. FP16 (4× vs. FP32) is achieved without degrading the loss curve, because compute and state remain at FP32.

#### §9.2: FP8-Specific Test Cases

| Test | Assertion |
|:---|:---|
| `test_fp8_e4m3_roundtrip` | Values in E4M3 range survive FP32→FP8→FP32 roundtrip with ≤1 ULP error |
| `test_fp8_e5m2_roundtrip` | Values in E5M2 range survive FP32→FP8→FP32 roundtrip with ≤1 ULP error |
| `test_fp8_saturation` | Values exceeding 448 (E4M3) saturate to 448, not NaN/Inf |
| `test_fp8_compute_rejection` | `PrecisionConfig(..., compute_dtype=float8_e4m3fn)` raises `ValueError` |
| `test_fp8_state_rejection` | `PrecisionConfig(..., state_dtype=float8_e4m3fn)` raises `ValueError` |
| `test_fp8_buffer_sizing` | Activation buffer with FP8 storage is 1/4 size of FP32 baseline |

---

### §10: Future Work

This ADR establishes FP8 as a first-class storage format. The following items are deferred:

1. **Block-scaled FP8 (FP8 with per-block exponent):** Improves dynamic range by sharing an exponent across 32-128 elements. Requires additional metadata buffers and modified quantization logic. Deferred pending hardware support standardization.

2. **SIMD FP8 on CPU:** AVX-512 provides no native FP8 instructions. Software SIMD (vectorized lookup table, parallel bit manipulation) is a performance optimization, not a correctness requirement.

3. **Mixed E4M3/E5M2 within a single configuration:** Using E4M3 for activations and E5M2 for gradients. Requires per-buffer format specification beyond the single `storage_dtype`. Deferred as a potential future `PrecisionConfig` extension.

4. **Native FP8 ALU utilization:** When hardware provides FP8 tensor cores (H100, MI300), the architecture could theoretically route certain compute operations through FP8 ALUs. This violates the "FP8 is storage-only" constraint and would require re-evaluation of the fidelity analysis. Deferred indefinitely — the three-role model is correct as specified.

---

## Consequences

### Positive

- **Maximum bandwidth efficiency.** FP8 storage achieves 4× compression vs. FP32, 2× vs. FP16, directly benefiting bandwidth-bound workloads per the Primacy of Memory Strategy.

- **No fidelity loss.** Compute and state remain at FP32 (or FP64), preserving reduction accuracy and optimizer stability.

- **Clean architectural integration.** FP8 fits naturally into the three-role model. No new abstraction layers or mode flags required.

- **Future-proof.** As FP8 hardware support matures, backends can adopt native instructions without architectural changes.

### Negative

- **Additional conversion overhead.** Every storage-role load/store incurs FP8↔FP32 conversion. For non-native backends, this is ~10-20 cycles per value. Bandwidth savings typically dominate, but compute-bound edge cases may see regression.

- **Increased complexity.** Two FP8 variants (E4M3/E5M2), software fallback paths, and scaling machinery add implementation and testing burden.

- **Dependency on ml_dtypes.** The NumPy FP8 representation requires an external package. This is a stable, widely-used package, but it is a new dependency.

### Neutral

- **No change to existing configurations.** `PrecisionConfig.float32()`, `PrecisionConfig.mixed_f16_f32()`, etc. are unchanged. FP8 is additive.

---

## References

- [OCP FP8 Specification](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) — Industry-standard E4M3/E5M2 definitions
- [ml_dtypes](https://github.com/jax-ml/ml_dtypes) — NumPy-compatible FP8 implementation
- ADR-020 — Three-Role Precision Model (motivating document)
- CONCEPT.md §2 — Primacy of Memory Strategy
