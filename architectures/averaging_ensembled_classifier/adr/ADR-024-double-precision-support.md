# ADR-024: Double Precision (FP64) Support

**Status:** PROPOSED
**Date:** 2026-04-02
**Deciders:** —
**Triggered by:** Extended precision requirements for state stability and validation scenarios
**Depends on:** ADR-020 (Three-Role Precision Model), ADR-021, ADR-022, ADR-023
**Extends:** ADR-020 §4.1 (PrecisionConfig factories), ADR-023 (backend implementations)
**Supersedes:** ADR-023 §2.4 (two-axis suffix scheme), ADR-023 §3.3 (Vulkan `COMPUTE_TYPE = float` invariant)
**Revisits:** ADR-023 §4.3 (`fp64` variant retirement)

---

## Context

ADR-023 retired the `fp64` CPU kernel variant with the rationale that "FP64 was never a deployed production configuration and has no `PrecisionConfig` factory." This retirement was premature. The three-role precision model (ADR-020) explicitly acknowledges that state precision is an independent stability concern, and the architecture already computes Adam bias correction terms (`beta1**t`, `beta2**t`) in FP64 on the host — a de facto recognition that FP64 has a role in ensuring numerical stability across unbounded training steps.

Three use cases warrant formal FP64 support:

### 1. State-Role FP64 for Extreme Training Stability

Adam's EMA update ($\beta_1 \cdot m + (1 - \beta_1) \cdot g$) requires that the format represent the small difference $(1 - \beta_1) \cdot g$ without rounding it away. For $\beta_1 = 0.9$, the gradient contributes $(1 - 0.9) = 0.1$ of its magnitude per step. For $\beta_1 = 0.999$ (common in large-model training), the gradient contributes only $0.001$.

At FP32 precision, the 23-bit mantissa provides approximately 7 decimal digits of precision. When $m$ is large and $(1 - \beta_1) \cdot g$ is small, the update rounds away. Over unbounded training steps, this precision erosion accumulates — exactly the stability concern ADR-020 §1 identified as the motivation for the state role. FP64's 52-bit mantissa (approximately 16 decimal digits) provides an order-of-magnitude more headroom.

The architecture's existing practice of computing bias correction in FP64 on the host acknowledges this concern. Extending FP64 to the moment vectors themselves (`m1`, `m2`) and master weight copies eliminates the remaining precision erosion pathways.

### 2. Compute-Role FP64 for Validation and Scientific Computing

Reduction tree accumulation fidelity depends on mantissa bits. The $\log_K(N)$ Recursive Reduction Engine (CONCEPT.md §4) requires additional mantissa bits beyond the summands' precision for exact accumulation. When validating the correctness of FP32 or FP16 implementations, an FP64 reference computation provides a ground-truth comparison that isolates algorithmic errors from precision artifacts.

Scientific computing workloads — particularly those operating on high-dynamic-range data or requiring reproducible results for numerical certification — may require FP64 arithmetic throughout the forward pass. This is not the primary use case of the Averaging Ensembled Classifier architecture (which prioritizes bandwidth under the Primacy of Memory Strategy), but the architecture should not *prevent* it.

### 3. Architectural Completeness

ADR-020's three-role precision model is semantically complete: it decomposes precision into orthogonal concerns (bandwidth, fidelity, stability) with independent type assignments. A model that can express `(storage=FP16, compute=FP32, state=FP32)` but not `(storage=FP32, compute=FP32, state=FP64)` is artificially constrained. The state role *should* support wider precision than compute — this is precisely what "stability" means. The prior `fp64` retirement created this asymmetry.

Per CONCEPT.md §1 (Architectural Elegance Feedback):

> 1. **Suspend** — The `fp64` retirement is suspended pending formalization.
> 2. **Formalize** — Establish FP64 as a first-class citizen of the three-role model.
> 3. **Reify** — Define `PrecisionConfig` factories, backend implications, and constraints.

---

## Decision Drivers

1. **The state role is defined by stability, not bandwidth.** ADR-020 §1 states: "State precision is a stability concern." FP64 is the natural expression of maximum stability within IEEE 754 standard floating-point formats. Excluding FP64 from the state role contradicts the role's definition.

2. **The compute role is defined by fidelity.** When fidelity requirements demand FP64 arithmetic — for validation, certification, or scientific computing — the compute role should support it. This is not the common case, and hardware support varies, but the architecture should not forbid it.

3. **Storage-role FP64 violates the Primacy of Memory Strategy.** FP64 storage doubles FP32's bandwidth cost with no storage-compression benefit. The architecture may *permit* storage-role FP64 for uniformity, but should not *optimize* for it. The expected configurations place FP64 only in compute or state roles.

4. **Backend hardware support determines availability.** FP64 requires `cl_khr_fp64` on OpenCL, `VK_KHR_shader_float64` on Vulkan, and is natively available on all modern CPUs. The architecture expresses FP64 configurations; backends that cannot support them fail at plan-construction time with a clear capability check, not silently.

5. **ADR-020's transitional alias protocol is closed.** The `SCALAR_TYPE` → `COMPUTE_TYPE` migration is complete. FP64 support is added to the three-role model, not the retired single-axis model.

---

## Decision

FP64 (`float64`, `double`) is added as a supported precision format across all three precision roles. The following sections specify the amendments to existing authority documents and design artifacts.

---

### §1: PrecisionConfig Extension (amends ADR-020 §4.1)

The `PrecisionConfig` dataclass supports `np.float64` as a valid dtype for any of its three role fields:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    storage_dtype: np.dtype   # np.float16, np.float32, np.float64
    compute_dtype: np.dtype   # np.float32, np.float64
    state_dtype: np.dtype     # np.float32, np.float64
    # ... derived fields unchanged
```

#### §1.1: New Factory Classmethods

| Factory | `storage_dtype` | `compute_dtype` | `state_dtype` | Primary Use Case |
|:---|:---|:---|:---|:---|
| `PrecisionConfig.float64()` | `float64` | `float64` | `float64` | Validation reference, scientific computing |
| `PrecisionConfig.mixed_f32_f64_state()` | `float32` | `float32` | `float64` | Extended training stability (FP64 state) |
| `PrecisionConfig.mixed_f16_f64_state()` | `float16` | `float32` | `float64` | Maximum bandwidth + maximum stability |
| `PrecisionConfig.mixed_f32_f64()` | `float32` | `float64` | `float64` | High-fidelity compute with FP32 storage |

The existing factories are unchanged:

| Factory | `storage_dtype` | `compute_dtype` | `state_dtype` |
|:---|:---|:---|:---|
| `PrecisionConfig.float32()` | `float32` | `float32` | `float32` |
| `PrecisionConfig.float16()` | `float16` | `float16` | `float16` |
| `PrecisionConfig.mixed_f16_f32()` | `float16` | `float32` | `float32` |

#### §1.2: Construction Invariants (amended)

The `__post_init__` validation is amended:

```python
def __post_init__(self) -> None:
    # Narrowing from storage to compute/state is always permitted
    assert self.storage_dtype.itemsize <= self.compute_dtype.itemsize
    assert self.storage_dtype.itemsize <= self.state_dtype.itemsize
    # Compute cannot be narrower than storage (no precision loss on load)
    assert self.compute_dtype.itemsize >= self.storage_dtype.itemsize
    # No constraint between compute and state — they are independent roles
```

The prior constraint `compute_fp_format_max >= storage_fp_format_max` is preserved implicitly by the itemsize invariant. The state role has no ordering constraint relative to compute — `PrecisionConfig.mixed_f32_f64_state()` has `state_dtype > compute_dtype`, which is valid (state is more precise than compute).

#### §1.3: Derived Constants

| Field | FP64 Value | Source |
|:---|:---|:---|
| `storage_fp_format_max` | `1.7976931348623157e+308` | `np.finfo(np.float64).max` |
| `compute_fp_format_max` | `1.7976931348623157e+308` | `np.finfo(np.float64).max` |
| `compute_epsilon` | `1e-15` | Conservative stability guard for FP64 arithmetic |

---

### §2: CONTRACT.md Article 6 Extension (amends ADR-020 §3.6)

The build-time symbol vocabulary is extended with a new derived flag:

| Symbol | Definition | Category |
|:---|:---|:---|
| `STATE_TYPE_IS_DOUBLE` | Integer flag (0 or 1). True when `STATE_TYPE` is `double`. Governs high-precision load/store mechanics for state buffers. | Derived |
| `COMPUTE_TYPE_IS_DOUBLE` | Integer flag (0 or 1). True when `COMPUTE_TYPE` is `double`. Governs precision-sensitive algorithm selection. | Derived |

The complete mandatory symbol set becomes:

| Symbol | Category |
|:---|:---|
| `STORAGE_TYPE` | Primary |
| `COMPUTE_TYPE` | Primary |
| `STATE_TYPE` | Primary |
| `STORAGE_TYPE_IS_HALF` | Derived |
| `COMPUTE_TYPE_IS_HALF` | Derived |
| `COMPUTE_TYPE_IS_DOUBLE` | Derived |
| `STATE_TYPE_IS_DOUBLE` | Derived |
| `NUMERICAL_STABILITY_EPSILON` | Derived |
| `SIMD_WIDTH` | Hardware |
| `C_TILE_SIZE` | Configuration |

The `_IS_HALF` and `_IS_DOUBLE` flags are mutually exclusive for the same role type. When `COMPUTE_TYPE = float`, both `COMPUTE_TYPE_IS_HALF = 0` and `COMPUTE_TYPE_IS_DOUBLE = 0`.

---

### §3: OpenCL Backend — FP64 Precision Boundary Implementations

#### §3.1: Extension Enable

When `COMPUTE_TYPE_IS_DOUBLE == 1` or `STATE_TYPE_IS_DOUBLE == 1`, the OpenCL kernel sources require the `cl_khr_fp64` extension:

```c
#if COMPUTE_TYPE_IS_DOUBLE || STATE_TYPE_IS_DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
```

This pragma is placed immediately after the existing `cl_khr_fp16` extension guard in `kernels.cl.h`.

#### §3.2: `load_state` / `store_state` Amendments

The state-role precision boundary abstractions (ADR-023 §1) are unchanged in signature. When `STATE_TYPE = double` and `COMPUTE_TYPE = float`, the implementations perform implicit narrowing/widening via C-style casts:

```c
static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx) {
    return (COMPUTE_TYPE)buf[idx];  // double → float narrowing if needed
}

static inline void store_state(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) {
    buf[idx] = (STATE_TYPE)val;  // float → double widening if needed
}
```

When `STATE_TYPE > COMPUTE_TYPE`, `load_state` narrows (precision loss accepted — the value is about to enter lower-precision arithmetic). When storing, `store_state` widens (no precision loss — the narrower compute value is stored in the wider state format).

This is the correct direction: state *stores* precision, compute *uses* precision. A value computed at FP32 and stored at FP64 preserves all FP32 bits. A value loaded from FP64 state into FP32 compute loses precision, but this is acceptable because the compute role's fidelity is bounded by `compute_dtype`.

#### §3.3: `COMPUTE_ZERO` Extension

```c
#if COMPUTE_TYPE_IS_DOUBLE
    #define COMPUTE_ZERO 0.0
#elif COMPUTE_TYPE_IS_HALF
    #define COMPUTE_ZERO 0.0h
#else
    #define COMPUTE_ZERO 0.0f
#endif
```

---

### §4: CPU Backend — FP64 Struct Instantiations (amends ADR-023 §2)

#### §4.1: Three-Axis Suffix Scheme

The two-axis scheme from ADR-023 §2.4 (`s{storage}x{state}`) is insufficient — it cannot express configurations where compute precision differs from both storage and state. This ADR replaces it with a three-axis scheme:

```
s{storage}c{compute}x{state}
```

Where each axis uses the shorthand: `16` = FP16, `32` = FP32, `64` = FP64.

The valid combinations are constrained by the `PrecisionConfig` invariants (§1.2):
- `storage_dtype.itemsize <= compute_dtype.itemsize`
- `storage_dtype.itemsize <= state_dtype.itemsize`

Additionally, on the CPU backend, compute is restricted to FP32 or FP64 — FP16 compute requires native hardware support that is not universally available on CPUs.

**Note:** `PrecisionConfig.float16()` specifies FP16 for all roles including compute. On GPU backends (OpenCL, Vulkan) with native FP16 ALU support, this is honored. On the CPU backend, FP16 compute is not supported — `PrecisionConfig.float16()` maps to suffix `s16c32x16` (FP32 compute). This is a backend-specific promotion, not an architectural prohibition.

**Complete enumeration of valid CPU suffixes:**

| Suffix | `STORAGE_T` | `COMPUTE_T` | `STATE_T` | Use Case |
|:---|:---|:---|:---|:---|
| `s16c32x16` | `_Float16` | `float` | `_Float16` | `PrecisionConfig.float16()` on CPU |
| `s16c32x32` | `_Float16` | `float` | `float` | Mixed FP16/FP32 (bandwidth) |
| `s16c32x64` | `_Float16` | `float` | `double` | Maximum bandwidth + maximum stability |
| `s16c64x16` | `_Float16` | `double` | `_Float16` | FP64 compute validation, FP16 state |
| `s16c64x32` | `_Float16` | `double` | `float` | FP64 compute validation, FP32 state |
| `s16c64x64` | `_Float16` | `double` | `double` | FP64 compute + stability, FP16 bandwidth |
| `s32c32x32` | `float` | `float` | `float` | Uniform FP32 |
| `s32c32x64` | `float` | `float` | `double` | FP64 state (stability) |
| `s32c64x32` | `float` | `double` | `float` | FP64 compute validation |
| `s32c64x64` | `float` | `double` | `double` | High-fidelity compute + stability |
| `s64c64x64` | `double` | `double` | `double` | Uniform FP64 (reference) |

All 11 valid combinations are instantiated. Compile time is trivial; completeness eliminates runtime discovery of missing instantiations.

The `DECLARE_PRECISION_STRUCTS` macro gains a third parameter:

```c
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, COMPUTE_T, STATE_T)
```

Complete instantiation block:

```c
/* All valid s{storage}c{compute}x{state} combinations */
/* storage=16 */
DECLARE_PRECISION_STRUCTS(s16c32x16, _Float16, float, _Float16)
DECLARE_PRECISION_STRUCTS(s16c32x32, _Float16, float, float)
DECLARE_PRECISION_STRUCTS(s16c32x64, _Float16, float, double)
DECLARE_PRECISION_STRUCTS(s16c64x16, _Float16, double, _Float16)
DECLARE_PRECISION_STRUCTS(s16c64x32, _Float16, double, float)
DECLARE_PRECISION_STRUCTS(s16c64x64, _Float16, double, double)
/* storage=32 */
DECLARE_PRECISION_STRUCTS(s32c32x32, float, float, float)
DECLARE_PRECISION_STRUCTS(s32c32x64, float, float, double)
DECLARE_PRECISION_STRUCTS(s32c64x32, float, double, float)
DECLARE_PRECISION_STRUCTS(s32c64x64, float, double, double)
/* storage=64 */
DECLARE_PRECISION_STRUCTS(s64c64x64, double, double, double)
```

#### §4.2: `cpu_compute_t` Removal

ADR-023 §2.1 established `cpu_compute_t = float` as a CPU backend invariant. This ADR **removes** that invariant — compute type is now parameterized via `COMPUTE_T` in the struct instantiation, not a global typedef.

Each kernel template (`.inc` file) receives `COMPUTE_T` as a macro parameter alongside `STORAGE_T` and `STATE_T`. All arithmetic within kernels uses `COMPUTE_T`, not a hardcoded `float`.

```c
/* cpu_kernels.h — COMPUTE_T is now a per-instantiation parameter */
/* The prior cpu_compute_t typedef is removed. */
```

This eliminates the need for separate library variants (`libcpu_kernels_f32.so` vs `libcpu_kernels_f64.so`). A single `libcpu_kernels.so` exports all instantiated struct suffixes; the Python FFI layer selects the appropriate suffix based on `PrecisionConfig`.

#### §4.3: FP64 Load/Store Variants

The role-aware load/store macros from ADR-023 §2.2 are extended with FP64 variants:

```c
// cpu_precision.h extension — state role
double scalar_load_state_f64(const double *ptr);
void scalar_store_state_f64(double *ptr, double val);

// cpu_precision.h extension — storage role (for s64c64x64)
double scalar_load_storage_f64(const double *ptr);
void scalar_store_storage_f64(double *ptr, double val);

// No SIMD variants for FP64 — see rationale below
```

For FP64-role buffers, scalar operations are preferred. The Adam kernel's access pattern (independent element-wise EMA) does not benefit from SIMD when the element type is `double` — the AVX2 vector width (4 × `double`) provides minimal latency hiding compared to 8 × `float`, and the memory bandwidth is the bottleneck, not ALU throughput. AVX-512 (8 × `double`) may justify SIMD variants in the future; this is deferred.

---

### §5: Vulkan Backend — FP64 Shader Variants (amends ADR-023 §3)

ADR-023 §3.3 established "COMPUTE_TYPE = float on Vulkan backend, invariant." This ADR removes that invariant — compute type is now parameterized via `COMPUTE_FLOAT`, completing the three-axis scheme.

#### §5.1: Three-Axis Macro Scheme

The Vulkan backend's `glslc -D` parameterization is extended from two axes to three:

| Macro | Values | Injected by |
|:---|:---|:---|
| `STORAGE_FLOAT` | `float16_t`, `float`, `double` | `glslc -DSTORAGE_FLOAT=...` |
| `COMPUTE_FLOAT` | `float`, `double` | `glslc -DCOMPUTE_FLOAT=...` |
| `STATE_FLOAT` | `float16_t`, `float`, `double` | `glslc -DSTATE_FLOAT=...` |

The `common.glsl` macro block (ADR-023 §3.2.1) is extended:

```glsl
#ifndef STORAGE_FLOAT
#define STORAGE_FLOAT float
#endif
#ifndef COMPUTE_FLOAT
#define COMPUTE_FLOAT float
#endif
#ifndef STATE_FLOAT
#define STATE_FLOAT float
#endif

#define WIDEN_STORAGE(x)   COMPUTE_FLOAT(x)
#define NARROW_STORAGE(x)  STORAGE_FLOAT(x)
#define WIDEN_STATE(x)     COMPUTE_FLOAT(x)
#define NARROW_STATE(x)    STATE_FLOAT(x)
```

Note: `WIDEN_*` now casts to `COMPUTE_FLOAT` rather than hardcoded `float`.

#### §5.2: Workgroup Scratch Type

The workgroup reduction scratch (ADR-023 §3.2.3) changes from hardcoded `float` to `COMPUTE_FLOAT`:

```glsl
// Compute-role scratch for cross-subgroup bridge.
shared COMPUTE_FLOAT _compute_scratch[32];
```

#### §5.3: Extension Requirements

| Extension | Required when |
|:---|:---|
| `GL_EXT_shader_explicit_arithmetic_types_float16` | `STORAGE_FLOAT=float16_t` or `STATE_FLOAT=float16_t` |
| `GL_EXT_shader_explicit_arithmetic_types_float64` | `COMPUTE_FLOAT=double` or `STATE_FLOAT=double` or `STORAGE_FLOAT=double` |

The extension guard in `common.glsl` is extended:

```glsl
#ifdef ENABLE_FP16_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
#ifdef ENABLE_FP64_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif
```

The Meson build injects `-DENABLE_FP64_EXTENSION=1` when any role uses `double`.

#### §5.4: Capability Check

Vulkan FP64 support requires `VkPhysicalDeviceFeatures::shaderFloat64`. The backend capability check (`src/backends/vulkan/capabilities.py`) is extended:

```python
def supports_float64(self) -> bool:
    return self.physical_device_features.shaderFloat64
```

Plan construction for a `PrecisionConfig` with FP64 roles fails fast if `supports_float64() == False`.

#### §5.5: SPIR-V Variant Enumeration

All valid `(STORAGE_FLOAT, COMPUTE_FLOAT, STATE_FLOAT)` combinations are compiled to SPIR-V at build time. The enumeration follows the same constraints as the CPU backend (§4.1):

| STORAGE | COMPUTE | STATE | Count |
|:---|:---|:---|:---|
| `float16_t` | `float`, `double` | `float16_t`, `float`, `double` | 6 |
| `float` | `float`, `double` | `float`, `double` | 4 |
| `double` | `double` | `double` | 1 |
| **Total** | | | **11** |

Each `.comp` shader is compiled 11 times with the appropriate `-D` flags, producing 11 SPIR-V modules per shader. The runtime selects the correct module based on `PrecisionConfig`.

#### §5.6: GLSL Type Mapping Summary

| Python dtype | GLSL type | SPIR-V type |
|:---|:---|:---|
| `np.float64` | `double` | `OpTypeFloat 64` |
| `np.float32` | `float` | `OpTypeFloat 32` |
| `np.float16` | `float16_t` | `OpTypeFloat 16` |

---

### §6: SIMD Width Implications

#### §6.1: FP64 SIMD Width

Under the three-role model, `SIMD_WIDTH` is defined in elements of `COMPUTE_TYPE`. When `COMPUTE_TYPE = double`:

| Hardware | FP32 SIMD Width | FP64 SIMD Width |
|:---|:---|:---|
| AVX2 (256-bit) | 8 | 4 |
| AVX-512 | 16 | 8 |
| OpenCL (typical GPU) | 32–64 | 16–32 |

`HardwareProfile.simd_width` is queried at the target `COMPUTE_TYPE`. A `PrecisionConfig.float64()` configuration produces a different `simd_width` than `PrecisionConfig.float32()` on the same hardware.

#### §6.2: Padding Implications

Buffer padding is computed in elements of the buffer's precision role dtype. An FP64-state buffer pads to `SIMD_WIDTH` elements of `STATE_TYPE = double`, not `SIMD_WIDTH` elements of `COMPUTE_TYPE`. This is unchanged from ADR-020 §3.3 — the padding contract already uses role-appropriate element sizes.

---

### §7: StabilizationPolicy at FP64

The Quadratic Scaling Policy's safety ceiling is bounded by `COMPUTE_FP_FORMAT_MAX` (ADR-020 §2.2). At FP64 compute precision:

$$T_{\text{safety}_j} = \frac{1.797 \times 10^{308}}{K_j}$$

For practical $K$ values (e.g., $K = 32$), the ceiling is astronomically large. Overflow is not a practical concern at FP64 compute precision. The stabilization machinery remains active (it costs nothing when not triggered), but the dynamic range of FP64 makes gradient explosion effectively unconstrained by format limits.

---

### §8: Test Strategy Extension (amends ADR-016)

#### §8.1: New Factory Configurations in Tier 2

Tier 2 tests are extended to cover the new `PrecisionConfig` factories:

| Factory | Tests |
|:---|:---|
| `PrecisionConfig.float64()` | Full Tier 2 suite (validation reference) |
| `PrecisionConfig.mixed_f32_f64_state()` | Adam stability tests, long-training-run validation |
| `PrecisionConfig.mixed_f16_f64_state()` | End-to-end mixed-precision training, bandwidth measurement |

#### §8.2: The Alchemist Validation Scenario Extension

The validation scenario from ADR-020 §2.6 is extended:

> **Scenario: The Alchemist II (FP64 State Stability Validation)**
>
> - **Description:** A training task is executed for $10^6$ steps using three state precision configurations: `PrecisionConfig.float32()` (FP32 state), `PrecisionConfig.mixed_f32_f64_state()` (FP64 state), and a numpy reference implementation using FP64 throughout. All use identical hyperparameters ($\beta_1 = 0.999$, $\beta_2 = 0.9999$).
> - **Validation Focus:** Confirms that the FP64-state configuration's moment vectors track the FP64 reference within FP64 tolerance ($< 10^{-14}$ relative error), while the FP32-state configuration diverges measurably (relative error grows with step count). Validates that state precision is architecturally independent of compute precision.
> - **Key Insight:** Proves that the state role's FP64 extension achieves its stated goal: unbounded training stability without FP32 precision erosion in moment vectors.

---

## Consequences

### Positive

- **State-role FP64 eliminates precision erosion in Adam.** Moment vectors and master weight copies at FP64 can sustain unbounded training steps without accumulating rounding errors.
- **Compute-role FP64 enables validation.** FP64 reference computations isolate numerical errors from algorithmic errors, improving test confidence.
- **Architectural completeness.** The three-role precision model now spans the full IEEE 754 standard floating-point hierarchy (FP16 → FP32 → FP64) in each role independently.
- **The retired `fp64` variant is superseded, not restored.** The old single-axis `fp64` is not resurrected. The new FP64 support is native to the three-role model: `s32c32x64`, `s16c32x64`, `s64c64x64` struct suffixes express role-specific FP64, not uniform `fp64`.

### Negative

- **CPU library struct instantiations increase.** 11 struct-suffix instantiations (from three under ADR-023). Build time remains trivial; completeness eliminates runtime discovery of missing instantiations. All variants compile into a single `libcpu_kernels.so`.
- **Vulkan SPIR-V artifact count increases.** 11 SPIR-V modules per shader (from ~3 under ADR-023). Each `.comp` source is compiled with all valid precision combinations. Build time increases proportionally but remains acceptable for ahead-of-time compilation.
- **FP64 hardware support is not universal.** Mobile GPUs and some consumer GPUs lack `cl_khr_fp64` or `shaderFloat64`. Configurations requiring FP64 will fail capability checks on such hardware. This is acceptable — the architecture does not promise FP64 on all devices; it promises that FP64 configurations fail fast on incompatible hardware.
- **FP64 compute is 2–4× slower than FP32 on most hardware.** Users enabling `COMPUTE_TYPE = double` must accept the throughput cost. This is a known tradeoff, not a bug.
- **Storage-role FP64 has no bandwidth benefit.** `PrecisionConfig.float64()` stores activations at double width. This is permitted for uniformity but violates the Primacy of Memory Strategy. Users should prefer `PrecisionConfig.mixed_f32_f64()` for high-fidelity workloads.

### Migration Path

1. **PrecisionConfig extension.** Add new factories to `src/shared/precision_config.py`.
2. **Backend type mappings.** Update `build_compiler_flags()` in each backend to emit `_IS_DOUBLE` flags.
3. **CPU struct macro.** Extend `DECLARE_PRECISION_STRUCTS` to accept three type parameters (`STORAGE_T`, `COMPUTE_T`, `STATE_T`).
4. **CPU struct instantiations.** Migrate existing suffixes to three-axis scheme (`s32x32` → `s32c32x32`) and add all 11 valid variants to `cpu_kernels.h`.
5. **CPU precision macros.** Extend `cpu_precision.h` to accept `COMPUTE_T` alongside `STORAGE_T` and `STATE_T`.
6. **OpenCL extension guard.** Add `cl_khr_fp64` pragma to `kernels.cl.h`.
7. **Vulkan `common.glsl`.** Add `COMPUTE_FLOAT` macro; update `WIDEN_*` to cast to `COMPUTE_FLOAT`; update `_compute_scratch` type.
8. **Vulkan extension guards.** Add `GL_EXT_shader_explicit_arithmetic_types_float64` conditional enablement.
9. **Vulkan SPIR-V compilation.** Extend `meson.build` to compile all 11 precision variants per shader.
10. **Vulkan capability check.** Add `supports_float64()` to `capabilities.py`.
11. **Tier 2 test coverage.** Extend parametrized tests to new factories.

---

## References

- [ADR-020](ADR-020-mixed-precision.md) — Three-Role Precision Model (extended)
- [ADR-021](ADR-021-kernels-precision-role-migration.md) — Kernel specification migration
- [ADR-022](ADR-022-host-code-precision-role-implications.md) — Host code implications
- [ADR-023](ADR-023-backend-kernel-precision-role-implications.md) — Backend kernel code (§4.3 `fp64` retirement revisited)
- CONCEPT.md §3.4 — Quadratic Scaling Policy safety ceiling
- CONCEPT.md §6 — Architectural Hierarchy (host computes `beta1**t` in FP64)
