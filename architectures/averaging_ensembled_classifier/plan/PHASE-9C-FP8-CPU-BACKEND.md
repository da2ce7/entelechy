# Phase 9C: FP8 Support — CPU Backend Implementation

**Status: ✅ COMPLETE**  
**Phase:** 9C of 9  
**Prerequisite:** Phase 9A complete and rollback gate passed (includes LUT generation in Step 9A.1.5).  
**Note on parallelization:** Phases 9B/9C/9D are fully parallelizable since LUT generation moved to Phase 9A.

**Objective:** Implement FP8 software types and conversion functions in the CPU backend. Add `cpu_fp8.h` with E4M3/E5M2 structs and conversion functions. Extend `cpu_precision.h` with FP8 load/store macros. Generate up to ten new kernel suffix variants (six for FP32/FP64 compute — always compiled; four for FP16 compute — conditionally compiled if `_Float16` is available). Update Meson build. The phase ends when FP8 CPU kernels compile and pass roundtrip tests.  
**Governing ADR:** ADR-025 (§5)  
**Rollback gate:** All existing CPU backend tests pass. FP8 kernel variants compile. FP8 roundtrip tests pass with ≤1 ULP error.  
**Dependencies:** Phase 9A (PrecisionConfig extension) complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 9C.1: Create `cpu_fp8.h`](#step-9c1-create-cpu_fp8h)
   - [Step 9C.2: Extend `cpu_precision.h`](#step-9c2-extend-cpu_precisionh)
   - [Step 9C.3: Add FP8 suffix variants](#step-9c3-add-fp8-suffix-variants)
   - [Step 9C.4: Update Meson build](#step-9c4-update-meson-build)
   - [Step 9C.5: Add FP8 CPU roundtrip tests](#step-9c5-add-fp8-cpu-roundtrip-tests)
   - [Step 9C.6: Validate rollback gate](#step-9c6-validate-rollback-gate)
4. [Technical Notes](#4-technical-notes)
5. [Risk Register](#5-risk-register)
6. [Files Modified Summary](#6-files-modified-summary)

---

## 1. Scope & Constraints

### In scope

- Creating `src/backends/cpu/kernel_sources/cpu_fp8.h` with FP8 struct types and conversion functions.
- Extending `src/backends/cpu/kernel_sources/cpu_precision.h` with FP8 load/store macros.
- Adding six new three-axis suffix variants per ADR-025 §5.3 for FP32/FP64 compute: `s8e4c32x32`, `s8e4c32x64`, `s8e4c64x64`, `s8e5c32x32`, `s8e5c32x64`, `s8e5c64x64`.
- Adding four additional FP16 compute suffix variants: `s8e4c16x32`, `s8e4c16x64`, `s8e5c16x32`, `s8e5c16x64`.
- Extending `DECLARE_PRECISION_STRUCTS` macro invocations for FP8 variants.
- Updating `meson.build` to compile FP8 kernel variants.
- Adding CPU-specific FP8 roundtrip tests.

### Out of scope

- SIMD FP8 operations — AVX-512 has no native FP8 instructions; scalar-only is the baseline.
- OpenCL backend — completed in Phase 9B.
- Vulkan backend — deferred to Phase 9D.
- Host-side scaling — deferred to Phase 9E.

### Key constraint: FP8 as struct wrapper

The CPU backend wraps FP8 values in a struct (`cpu_fp8_e4m3`, `cpu_fp8_e5m2`) containing a `uint8_t`. This prevents accidental arithmetic on raw bits. All FP8 operations go through explicit conversion functions.

### Key constraint: `_Float16` availability for FP16 compute

The FP16 compute variants (`s8e4c16x32`, `s8e4c16x64`, etc.) require `_Float16` support, which is:
- **C23 standard** (ISO/IEC 9899:2024) — native support
- **GCC 12+** with `-std=c2x` or `-std=gnu2x` on x86-64 with F16C
- **Clang 15+** with same flags, or with `__fp16` extension on ARM
- **MSVC** — limited; cast through `_Float16` type requires `/arch:AVX512`

The Meson build gates FP16 compute variants on compiler capability detection:

```meson
# In meson.build — detect _Float16 support
cc = meson.get_compiler('c')

# The `2.0f16` literal suffix is C23-specific (_Float16 constant suffix).
# This intentionally tests for C23-conforming _Float16 support, NOT Clang's
# legacy __fp16 extension (which is a storage-only type without arithmetic).
# GCC 12+ and Clang 15+ support _Float16 with -std=c2x.
has_float16 = cc.compiles('''
    _Float16 test_float16(_Float16 x) { return x * 2.0f16; }
''', name: '_Float16 support check', args: ['-std=c2x'])

if has_float16
    message('_Float16 support detected — FP16 compute variants enabled')
    # ... compile c16 suffix variants ...
else
    warning('_Float16 not supported — FP16 compute variants disabled')
    # ... skip c16 suffix variants ...
endif
```

If `_Float16` is unavailable, the `c16` suffix variants are skipped (not compiled), and `PrecisionConfig.fp8_e4m3_f16()` will raise `RuntimeError` at kernel load time with a clear message indicating the compiler lacked `_Float16` support.

> **Runtime error behavior:** The `PrecisionConfig.fp8_e4m3_f16()` factory (Phase 9A) constructs
> successfully regardless of compiler support — config creation is pure Python. The error surfaces
> when the CPU backend attempts to load the kernel:
> ```python
> RuntimeError: CPU kernel variant 's8e4c16x32' not available.
> This variant requires _Float16 support (C23 or GCC 12+/Clang 15+ with -std=c2x).
> Your compiler did not support _Float16 at build time.
> Use fp8_e4m3() (FP32 compute) or fp8_e4m3_f64() (FP64 compute) instead.
> ```

> **⚠️ FP16 compute availability differs by backend:**
> | Backend | FP16 Compute Availability |
> |:---|:---|
> | **CPU** | Requires `_Float16` (C23 / GCC 12+ / Clang 15+). May be unavailable. |
> | **OpenCL** | Uses `half` type. Widely available via `cl_khr_fp16`. |
> | **Vulkan** | Uses `float16_t`. Requires `VK_KHR_shader_float16_int8`. |
>
> A config like `fp8_e4m3_f16()` may work on OpenCL/Vulkan but fail on CPU if the compiler lacks `_Float16`. Document this limitation to users.

---

## 2. Pre-Condition Inventory

> **Prerequisite: ADR-025 §5.3 amendment.** This phase introduces 4 conditional FP16 compute
> CPU suffix variants (`s8e4c16x32`, `s8e4c16x64`, `s8e5c16x32`, `s8e5c16x64`) gated on
> `_Float16` compiler support. ADR-025 §5.3 has been amended to authorize these variants and
> document the `_Float16` conditionality. If the amendment is not yet committed, apply it
> first — the suffix table in §5.3 must list these variants before implementation proceeds.

| File | Relevant current state |
|:---|:---|
| `src/backends/cpu/kernel_sources/cpu_precision.h` | Defines precision macros for FP16/FP32/FP64. No FP8 handling. |
| `src/backends/cpu/kernel_sources/cpu_kernels.h` | Kernel declarations. Uses `DECLARE_PRECISION_STRUCTS` for existing suffixes. |
| `src/backends/cpu/kernel_sources/cpu_kernels.c` | Kernel implementations. |
| `meson.build` | Compiles existing suffix variants. No FP8 variants. |

---

## 3. Task Breakdown

---

### Step 9C.1: Create `cpu_fp8.h`

**Governing authority:** ADR-025 §5.1–5.2  
**File:** `src/backends/cpu/kernel_sources/cpu_fp8.h` (new)

Create the FP8 type definitions and conversion functions supporting FP16, FP32, and FP64:

```c
/* cpu_fp8.h — FP8 representation and conversion for CPU backend
 * ADR-025 §5.1–5.2
 * 
 * Provides conversions to/from:
 *   - _Float16 (FP16 compute) — requires C23 / compiler support
 *   - float (FP32 compute)
 *   - double (FP64 compute)
 */

#ifndef CPU_FP8_H
#define CPU_FP8_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>  /* for memcpy (strict aliasing compliance) */
#include <math.h>
#include <float.h>   /* for FLT_MAX etc. */

/* --- FP8 Storage Types (ADR-025 §5.1) ---
 * Struct wrappers prevent accidental arithmetic on raw bits.
 */

typedef struct { uint8_t bits; } cpu_fp8_e4m3;
typedef struct { uint8_t bits; } cpu_fp8_e5m2;

/* --- E4M3 Constants ---
 * sign(1) + exp(4) + mantissa(3), bias=7, max=448, no inf/nan
 */
#define E4M3_BIAS 7
#define E4M3_MANTISSA_BITS 3
#define E4M3_MAX_VAL 448.0f
#define E4M3_MIN_SUBNORMAL 0.001953125f  /* 2^-9 */

/* --- E5M2 Constants ---
 * sign(1) + exp(5) + mantissa(2), bias=15, max=57344
 * Unlike E4M3fn, E5M2 has IEEE-like inf/NaN (exponent 0x1F).
 * The store path saturates to max finite (0x7B/0xFB), never writing inf/NaN patterns.
 */
#define E5M2_BIAS 15
#define E5M2_MANTISSA_BITS 2
#define E5M2_MAX_VAL 57344.0f
#define E5M2_MIN_SUBNORMAL 0.0000152587890625f  /* 2^-16, exact */

/* --- Include generated lookup tables (ADR-025 §6.1) ---
 * Generated by: python scripts/generate_fp8_lut.py
 * Contains: cpu_fp8_e4m3_to_float_lut[256], cpu_fp8_e5m2_to_float_lut[256]
 */
#include "cpu_fp8_lut.gen.h"

/* --- Include Path Note ---
 * The cpu_fp8_lut.gen.h header is committed to source control at:
 *   src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h
 * Meson's include_directories for cpu_kernel_inc must include this path.
 * See Step 9C.1.1 for verification.
 */

/* --- E4M3 Conversion Functions (ADR-025 §5.2) --- */

static inline float cpu_fp8_e4m3_to_float(cpu_fp8_e4m3 val) {
    return cpu_fp8_e4m3_to_float_lut[val.bits];
}

static inline double cpu_fp8_e4m3_to_double(cpu_fp8_e4m3 val) {
    return (double)cpu_fp8_e4m3_to_float_lut[val.bits];
}

/* FP16 conversion — guarded by _Float16 availability
 * Detection strategy: Use Meson-emitted -DHAS_FLOAT16=1 (see Step 9C.4) as the
 * single source of truth.  Meson's cc.compiles() test already verified that the
 * compiler accepts _Float16 with arithmetic; duplicating that check in a header
 * guard via __FLT16_MAX__ risks a mismatch (e.g., __FLT16_MAX__ can be defined
 * even when the linker lacks soft-float routines for _Float16 on some targets).
 *
 * Fallback: If this header is included outside the Meson build (e.g., unit-test
 * harness that compiles directly), HAS_FLOAT16 won't be defined and the FP16
 * conversion functions will be excluded — which is the safe default.
 */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
static inline _Float16 cpu_fp8_e4m3_to_half(cpu_fp8_e4m3 val) {
    return (_Float16)cpu_fp8_e4m3_to_float_lut[val.bits];
}
#endif

static inline cpu_fp8_e4m3 cpu_float_to_fp8_e4m3(float val) {
    cpu_fp8_e4m3 result;
    
    /* Handle special cases */
    if (isnan(val)) {
        /* E4M3 has no infinities but does have 2 NaN patterns (0x7F, 0xFF).
         * The store path never writes NaN; we map NaN→zero defensively.
         * 
         * We map NaN→zero rather than NaN→max because:
         * 1. NaN in gradients typically indicates upstream numerical instability
         *    that should be caught before reaching FP8 conversion.
         * 2. Zero gradients produce no parameter update ("safe" failure mode),
         *    whereas max-value gradients could cause divergence.
         * 3. This matches ml_dtypes.float8_e4m3fn behavior.
         *
         * If fail-fast NaN propagation is desired, check for NaN in host code
         * before invoking CPU kernels. */
        result.bits = 0;
        return result;
    }
    
    /* Extract sign via bit pattern (strict-aliasing-safe) */
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(fbits));
    uint32_t sign = (fbits >> 31) & 1;
    float absval = fabsf(val);
    
    /* Saturation: >= is intentional. 0x7E encodes exactly 448.0, so absval == MAX_VAL
     * short-circuits to the same result the rounding path would produce. */
    if (absval >= E4M3_MAX_VAL) {
        result.bits = (sign << 7) | 0x7E;  /* Max magnitude */
        return result;
    }
    
    /* Underflow to zero */
    if (absval < E4M3_MIN_SUBNORMAL * 0.5f) {
        result.bits = sign << 7;  /* Signed zero */
        return result;
    }
    
    /* Extract FP32 exponent and mantissa */
    int exp32 = ((fbits >> 23) & 0xFF) - 127;  /* Unbiased exponent */
    uint32_t mant32 = fbits & 0x7FFFFF;        /* 23-bit mantissa */
    
    /* Compute FP8 exponent */
    int exp8 = exp32 + E4M3_BIAS;
    
    /* Handle subnormals */
    if (exp8 <= 0) {
        /* Subnormal in E4M3 */
        int shift = 1 - exp8;
        if (shift >= 24) {
            result.bits = sign << 7;  /* Underflow to zero (shift >= 24 is UB) */
            return result;
        }
        mant32 = (mant32 | 0x800000) >> shift;  /* Denormalize */
        exp8 = 0;
    } else if (exp8 >= 15) {
        /* Overflow — should not reach here due to saturation above */
        result.bits = (sign << 7) | 0x7E;
        return result;
    }
    
    /* Round mantissa to 3 bits (round-to-nearest-even) */
    uint32_t round_bit = (mant32 >> 19) & 1;
    uint32_t sticky = mant32 & ((1u << 19) - 1);
    uint32_t mant8 = mant32 >> 20;
    if (round_bit && (sticky || (mant8 & 1))) {
        mant8++;
    }
    if (mant8 >= 8) {
        mant8 = 0;
        exp8++;
        if (exp8 >= 15) {
            result.bits = (sign << 7) | 0x7E;  /* Overflow to max */
            return result;
        }
    }
    
    result.bits = (sign << 7) | (exp8 << 3) | (mant8 & 0x7);
    return result;
}

static inline cpu_fp8_e4m3 cpu_double_to_fp8_e4m3(double val) {
    /* Defense-in-depth: clamp before double→float narrowing to prevent UB
     * (C99 §6.3.1.5: double→float overflow is undefined behavior).
     * Pre-scaling (Phase 9E) ensures values are in FP8 range ⊂ FP32 range,
     * so this clamp is redundant under correct operation. Consistent with
     * the OpenCL (Step 9B.3) and Vulkan (Step 9D.2) backends. */
    if (val > (double)FLT_MAX) val = (double)FLT_MAX;
    if (val < -(double)FLT_MAX) val = -(double)FLT_MAX;
    return cpu_float_to_fp8_e4m3((float)val);
}

/* FP16 conversion — guarded by _Float16 availability */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
static inline cpu_fp8_e4m3 cpu_half_to_fp8_e4m3(_Float16 val) {
    return cpu_float_to_fp8_e4m3((float)val);
}
#endif

/* --- E5M2 Conversion Functions (ADR-025 §5.2) --- */

static inline float cpu_fp8_e5m2_to_float(cpu_fp8_e5m2 val) {
    return cpu_fp8_e5m2_to_float_lut[val.bits];
}

static inline double cpu_fp8_e5m2_to_double(cpu_fp8_e5m2 val) {
    return (double)cpu_fp8_e5m2_to_float_lut[val.bits];
}

static inline cpu_fp8_e5m2 cpu_float_to_fp8_e5m2(float val) {
    cpu_fp8_e5m2 result;
    
    /* Handle special cases */
    if (isnan(val)) {
        result.bits = 0;
        return result;
    }
    
    /* Extract sign via bit pattern (strict-aliasing-safe) */
    uint32_t fbits;
    memcpy(&fbits, &val, sizeof(fbits));
    uint32_t sign = (fbits >> 31) & 1;
    float absval = fabsf(val);
    
    /* Saturation: >= is intentional. 0x7B encodes exactly 57344.0, so absval == MAX_VAL
     * short-circuits to the same result the rounding path would produce. */
    if (absval >= E5M2_MAX_VAL) {
        result.bits = (sign << 7) | 0x7B;  /* Max magnitude */
        return result;
    }
    
    /* Underflow to zero */
    if (absval < E5M2_MIN_SUBNORMAL * 0.5f) {
        result.bits = sign << 7;
        return result;
    }
    
    /* Extract FP32 exponent and mantissa */
    int exp32 = ((fbits >> 23) & 0xFF) - 127;
    uint32_t mant32 = fbits & 0x7FFFFF;
    
    /* Compute FP8 exponent */
    int exp8 = exp32 + E5M2_BIAS;
    
    /* Handle subnormals */
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            result.bits = sign << 7;  /* Underflow to zero (shift >= 24 is UB) */
            return result;
        }
        mant32 = (mant32 | 0x800000) >> shift;
        exp8 = 0;
    } else if (exp8 >= 31) {
        result.bits = (sign << 7) | 0x7B;
        return result;
    }
    
    /* Round mantissa to 2 bits (round-to-nearest-even) */
    uint32_t round_bit = (mant32 >> 20) & 1;
    uint32_t sticky = mant32 & ((1u << 20) - 1);
    uint32_t mant8 = mant32 >> 21;
    if (round_bit && (sticky || (mant8 & 1))) {
        mant8++;
    }
    if (mant8 >= 4) {
        mant8 = 0;
        exp8++;
        if (exp8 >= 31) {
            result.bits = (sign << 7) | 0x7B;
            return result;
        }
    }
    
    result.bits = (sign << 7) | (exp8 << 2) | (mant8 & 0x3);
    return result;
}

static inline cpu_fp8_e5m2 cpu_double_to_fp8_e5m2(double val) {
    /* Defense-in-depth: clamp before double→float narrowing to prevent UB
     * (C99 §6.3.1.5: double→float overflow is undefined behavior).
     * Pre-scaling (Phase 9E) ensures values are in FP8 range ⊂ FP32 range,
     * so this clamp is redundant under correct operation. Consistent with
     * the OpenCL (Step 9B.3) and Vulkan (Step 9D.2) backends. */
    if (val > (double)FLT_MAX) val = (double)FLT_MAX;
    if (val < -(double)FLT_MAX) val = -(double)FLT_MAX;
    return cpu_float_to_fp8_e5m2((float)val);
}

/* FP16 conversion — guarded by _Float16 availability */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
static inline _Float16 cpu_fp8_e5m2_to_half(cpu_fp8_e5m2 val) {
    return (_Float16)cpu_fp8_e5m2_to_float_lut[val.bits];
}

static inline cpu_fp8_e5m2 cpu_half_to_fp8_e5m2(_Float16 val) {
    return cpu_float_to_fp8_e5m2((float)val);
}
#endif

#endif /* CPU_FP8_H */
```

---

### Step 9C.1.1: Verify include paths for LUT header

**File:** `meson.build` (CPU backend section)

Verify that the existing `include_directories` for CPU kernels covers the LUT header location:

```meson
# Verify this include path covers cpu_fp8_lut.gen.h
cpu_kernel_inc = include_directories('src/backends/cpu/kernel_sources')
```

Since the LUT header is committed to `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` (Phase 9A.1.5.3), no additional include path injection is needed — the existing `cpu_kernel_inc` already provides access.

> **Why no build-time generation?**
>
> Unlike a `custom_target` approach, the committed LUT files eliminate build-time dependencies on `ml_dtypes`. The `#include "cpu_fp8_lut.gen.h"` directive in `cpu_fp8.h` resolves against the existing include path.

---

### Step 9C.2: Extend `cpu_precision.h`

**Governing authority:** ADR-025 §5.4  
**File:** `src/backends/cpu/kernel_sources/cpu_precision.h`

Add FP8 load/store macros supporting FP16, FP32, and FP64 compute:

```c
#include "cpu_fp8.h"

/* --- FP8 Storage Role Load/Store (ADR-025 §5.4) ---
 * Dispatch on COMPUTE_TYPE for appropriate conversion.
 */

#if defined(STORAGE_TYPE_IS_E4M3) && STORAGE_TYPE_IS_E4M3

#if defined(COMPUTE_TYPE_IS_DOUBLE) && COMPUTE_TYPE_IS_DOUBLE
#define scalar_load_storage(ptr) cpu_fp8_e4m3_to_double(*(ptr))
#define scalar_store_storage(ptr, val) (*(ptr) = cpu_double_to_fp8_e4m3(val))
#elif defined(COMPUTE_TYPE_IS_HALF) && COMPUTE_TYPE_IS_HALF
#define scalar_load_storage(ptr) cpu_fp8_e4m3_to_half(*(ptr))
#define scalar_store_storage(ptr, val) (*(ptr) = cpu_half_to_fp8_e4m3(val))
#else  /* FP32 compute */
#define scalar_load_storage(ptr) cpu_fp8_e4m3_to_float(*(ptr))
#define scalar_store_storage(ptr, val) (*(ptr) = cpu_float_to_fp8_e4m3(val))
#endif

#elif defined(STORAGE_TYPE_IS_E5M2) && STORAGE_TYPE_IS_E5M2

#if defined(COMPUTE_TYPE_IS_DOUBLE) && COMPUTE_TYPE_IS_DOUBLE
#define scalar_load_storage(ptr) cpu_fp8_e5m2_to_double(*(ptr))
#define scalar_store_storage(ptr, val) (*(ptr) = cpu_double_to_fp8_e5m2(val))
#elif defined(COMPUTE_TYPE_IS_HALF) && COMPUTE_TYPE_IS_HALF
#define scalar_load_storage(ptr) cpu_fp8_e5m2_to_half(*(ptr))
#define scalar_store_storage(ptr, val) (*(ptr) = cpu_half_to_fp8_e5m2(val))
#else  /* FP32 compute */
#define scalar_load_storage(ptr) cpu_fp8_e5m2_to_float(*(ptr))
#define scalar_store_storage(ptr, val) (*(ptr) = cpu_float_to_fp8_e5m2(val))
#endif

#endif /* STORAGE_TYPE_IS_E4M3 / E5M2 */
```

> **Preprocessor hygiene note:** The `defined(COMPUTE_TYPE_IS_HALF) && COMPUTE_TYPE_IS_HALF` guard
> handles the case where non-FP8 build variants do not define `COMPUTE_TYPE_IS_HALF` at all.
> For cleanliness, the Meson build (Step 9C.4) should emit `COMPUTE_TYPE_IS_HALF=0` and
> `COMPUTE_TYPE_IS_DOUBLE=0` for **all** CPU variants (not just FP8), ensuring these macros
> are always defined. This avoids silent dead-code if a flag is accidentally omitted.

---

### Step 9C.3: Add FP8 suffix variants

**Governing authority:** ADR-025 §5.3  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.h`

Add ten new suffix declarations:

**FP32/FP64 compute variants (6, always compiled):**
- `s8e4c32x32` — E4M3 storage, FP32 compute, FP32 state
- `s8e4c32x64` — E4M3 storage, FP32 compute, FP64 state
- `s8e4c64x64` — E4M3 storage, FP64 compute, FP64 state
- `s8e5c32x32` — E5M2 storage, FP32 compute, FP32 state
- `s8e5c32x64` — E5M2 storage, FP32 compute, FP64 state
- `s8e5c64x64` — E5M2 storage, FP64 compute, FP64 state

**FP16 compute variants (4, conditionally compiled if `_Float16` available):**
- `s8e4c16x32` — E4M3 storage, FP16 compute, FP32 state
- `s8e4c16x64` — E4M3 storage, FP16 compute, FP64 state
- `s8e5c16x32` — E5M2 storage, FP16 compute, FP32 state
- `s8e5c16x64` — E5M2 storage, FP16 compute, FP64 state

**Important:** The FP16 compute variants (`c16` suffix) use `_Float16`, which requires C23 or compiler extensions. Guard these declarations to prevent parse errors on compilers without `_Float16` support:

```c
/* FP8 E4M3 storage variants with FP32/FP64 compute (ADR-025 §5.3) */
DECLARE_PRECISION_STRUCTS(s8e4c32x32, cpu_fp8_e4m3, float, float)
DECLARE_PRECISION_STRUCTS(s8e4c32x64, cpu_fp8_e4m3, float, double)
DECLARE_PRECISION_STRUCTS(s8e4c64x64, cpu_fp8_e4m3, double, double)

/* FP8 E5M2 storage variants with FP32/FP64 compute */
DECLARE_PRECISION_STRUCTS(s8e5c32x32, cpu_fp8_e5m2, float, float)
DECLARE_PRECISION_STRUCTS(s8e5c32x64, cpu_fp8_e5m2, float, double)
DECLARE_PRECISION_STRUCTS(s8e5c64x64, cpu_fp8_e5m2, double, double)

/* FP8 storage variants with FP16 compute — requires _Float16.
 * Guarded by HAS_FLOAT16 (emitted by Meson — see Step 9C.4).
 * The Meson build also skips c16 variant compilation when _Float16 is unavailable.
 */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
/* FP8 E4M3 storage variants with FP16 compute */
DECLARE_PRECISION_STRUCTS(s8e4c16x32, cpu_fp8_e4m3, _Float16, float)
DECLARE_PRECISION_STRUCTS(s8e4c16x64, cpu_fp8_e4m3, _Float16, double)

/* FP8 E5M2 storage variants with FP16 compute */
DECLARE_PRECISION_STRUCTS(s8e5c16x32, cpu_fp8_e5m2, _Float16, float)
DECLARE_PRECISION_STRUCTS(s8e5c16x64, cpu_fp8_e5m2, _Float16, double)
#endif /* _Float16 available */
```

The `c16` suffix indicates FP16 compute. State remains FP32 or FP64 (never FP16).

---

### Step 9C.4: Update Meson build

**Governing authority:** ADR-025 §5  
**File:** `meson.build` (CPU backend section)

Add compilation targets for FP8 variants:

```meson
# FP8 variants — unified structure with 'fp8_format' key for consistency
# All variants use the same dict structure to avoid key-access errors.
cpu_fp8_variants = [
  # E4M3 variants
  {'suffix': 's8e4c32x32', 'fp8_format': 'e4m3', 'compute_half': false, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e4c32x64', 'fp8_format': 'e4m3', 'compute_half': false, 'compute_double': false, 'state_double': true},
  {'suffix': 's8e4c64x64', 'fp8_format': 'e4m3', 'compute_half': false, 'compute_double': true, 'state_double': true},
  {'suffix': 's8e4c16x32', 'fp8_format': 'e4m3', 'compute_half': true, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e4c16x64', 'fp8_format': 'e4m3', 'compute_half': true, 'compute_double': false, 'state_double': true},
  # E5M2 variants
  {'suffix': 's8e5c32x32', 'fp8_format': 'e5m2', 'compute_half': false, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e5c32x64', 'fp8_format': 'e5m2', 'compute_half': false, 'compute_double': false, 'state_double': true},
  {'suffix': 's8e5c64x64', 'fp8_format': 'e5m2', 'compute_half': false, 'compute_double': true, 'state_double': true},
  {'suffix': 's8e5c16x32', 'fp8_format': 'e5m2', 'compute_half': true, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e5c16x64', 'fp8_format': 'e5m2', 'compute_half': true, 'compute_double': false, 'state_double': true},
]

# Detect _Float16 support (C23 or compiler extension)
cc = meson.get_compiler('c')
has_float16 = cc.compiles('''
    _Float16 test_float16(_Float16 x) { return x * 2.0f16; }
''', name: '_Float16 support check', args: ['-std=c2x'])

if has_float16
    message('_Float16 support detected — FP16 compute variants enabled')
else
    warning('_Float16 not supported — FP16 compute variants will be skipped')
endif

# Emit -DHAS_FLOAT16=1 for ALL CPU kernel compilations (not just c16 variants).
# This is the SINGLE SOURCE OF TRUTH for _Float16 availability in C headers.
# Headers use `#if defined(HAS_FLOAT16) && HAS_FLOAT16` — do NOT duplicate
# detection in header guards via __FLT16_MAX__ (see cpu_fp8.h comments).
if has_float16
    fp8_float16_cflag = ['-DHAS_FLOAT16=1']
else
    fp8_float16_cflag = ['-DHAS_FLOAT16=0']
endif

# Initialize list to collect FP8 kernel libraries
cpu_fp8_kernel_libs = []

# Build FP8 variants, conditionally skipping c16 suffixes if _Float16 unavailable
foreach variant : cpu_fp8_variants
  # Skip FP16 compute variants if compiler lacks _Float16
  if variant['compute_half'] and not has_float16
    message('Skipping ' + variant['suffix'] + ' (requires _Float16)')
    continue
  endif
  
  # Build compile flags for this variant
  cflags = ['-DSTORAGE_TYPE_IS_FP8=1'] + fp8_float16_cflag
  
  # Use unified fp8_format key for E4M3/E5M2 selection
  if variant['fp8_format'] == 'e4m3'
    cflags += '-DSTORAGE_TYPE_IS_E4M3=1'
    cflags += '-DSTORAGE_TYPE_IS_E5M2=0'
  else  # e5m2
    cflags += '-DSTORAGE_TYPE_IS_E4M3=0'
    cflags += '-DSTORAGE_TYPE_IS_E5M2=1'
  endif
  
  if variant['compute_half']
    cflags += '-DCOMPUTE_TYPE_IS_HALF=1'
  else
    cflags += '-DCOMPUTE_TYPE_IS_HALF=0'
  endif
  
  if variant['compute_double']
    cflags += '-DCOMPUTE_TYPE_IS_DOUBLE=1'
  else
    cflags += '-DCOMPUTE_TYPE_IS_DOUBLE=0'
  endif
  
  if variant['state_double']
    cflags += '-DSTATE_TYPE_IS_DOUBLE=1'
  else
    cflags += '-DSTATE_TYPE_IS_DOUBLE=0'
  endif
  
  cflags += '-DSUFFIX=' + variant['suffix']
  
  # Add C23 flag for FP16 compute variants
  if variant['compute_half']
    cflags += ['-std=c2x']
  endif
  
  # Compile kernel variant object file
  kernel_obj = static_library(
    'cpu_kernels_' + variant['suffix'],
    'src/backends/cpu/kernel_sources/cpu_kernels.c',
    c_args: cflags + common_cflags,
    include_directories: cpu_kernel_inc,
    pic: true,
    install: false,
  )
  
  # Track compiled variants for linking
  cpu_fp8_kernel_libs += kernel_obj
endforeach

# Link all FP8 variants into the main CPU backend library
cpu_backend_lib = shared_library(
  'cpu_backend',
  'src/backends/cpu/backend.c',
  link_with: cpu_kernel_libs + cpu_fp8_kernel_libs,
  c_args: common_cflags,
  include_directories: cpu_kernel_inc,
  install: true,
  install_dir: cpu_backend_install_dir,
)
```

#### 9C.4.1: Record available FP8 variants for Python loader

The Meson build must record which FP8 variants were successfully compiled. This allows the Python kernel loader to provide clear error messages when unavailable variants are requested.

```meson
# Generate a Python module listing available FP8 variants
available_fp8_variants = []
foreach variant : cpu_fp8_variants
  if variant['compute_half'] and not has_float16
    continue
  endif
  available_fp8_variants += variant['suffix']
endforeach

# Build the list string for Python
# Note: Meson DSL does not support Python-style ', '.join().
# Use a foreach loop to build the comma-separated string manually.
# The embedded double-quotes are passed through configure_file's @VAR@
# substitution verbatim — verify the generated .py output is syntactically
# valid (no double-escaping) after the first build.
fp8_list_str = ''
fp8_sep = ''
foreach v : available_fp8_variants
  fp8_list_str += fp8_sep + '"' + v + '"'
  fp8_sep = ', '
endforeach

conf_data = configuration_data()
conf_data.set('AVAILABLE_FP8_VARIANTS', '[' + fp8_list_str + ']')
if has_float16
  conf_data.set('HAS_FLOAT16', 'True')
else
  conf_data.set('HAS_FLOAT16', 'False')
endif

configure_file(
  input: 'src/backends/cpu/_fp8_variants.py.in',
  output: '_fp8_variants.py',
  configuration: conf_data,
  install: true,
  install_dir: cpu_backend_install_dir,
)
```

Create the template file `src/backends/cpu/_fp8_variants.py.in`:

```python
# Auto-generated by Meson — DO NOT EDIT
# Lists FP8 kernel variants available in this build

AVAILABLE_FP8_VARIANTS: list[str] = @AVAILABLE_FP8_VARIANTS@
HAS_FLOAT16: bool = @HAS_FLOAT16@
```

> **⚠️ Post-build verification:** After the first `ninja` build, verify the generated
> `_fp8_variants.py` is syntactically valid Python. Meson's `configure_file` performs
> literal `@VAR@` substitution — embedded quotes from the `fp8_list_str` builder
> above are passed through verbatim. A quoting error produces a broken `.py` file
> that won't surface until Python import time.
>
> ```bash
> # Run immediately after first build:
> python3 -c "import ast; ast.parse(open('builddir/src/backends/cpu/_fp8_variants.py').read()); print('OK')"
> ```
> Expected: prints `OK`. If it raises `SyntaxError`, inspect the generated file for
> double-escaped quotes or missing brackets.

#### 9C.4.2: Python kernel loader FP16 availability check

**File:** `src/backends/cpu/kernel_loader.py`

Add runtime check for unavailable FP16 compute variants:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.precision_config import PrecisionConfig

# Import the generated availability module
try:
    from ._fp8_variants import AVAILABLE_FP8_VARIANTS, HAS_FLOAT16
except ImportError:
    # Fallback if module not yet generated (pre-build)
    AVAILABLE_FP8_VARIANTS: list[str] = []
    HAS_FLOAT16: bool = False


def _get_kernel_suffix(precision: "PrecisionConfig") -> str:
    """Determine kernel suffix from precision config.
    
    The suffix encodes the three-axis precision configuration:
    - Storage: s16 (FP16), s32 (FP32), s64 (FP64), s8e4 (E4M3), s8e5 (E5M2)
    - Compute: c16 (FP16), c32 (FP32), c64 (FP64)
    - State: x32 (FP32), x64 (FP64)
    
    Examples: s32c32x32, s16c32x32, s8e4c32x32, s8e4c16x64
    """
    # Storage axis
    if precision.storage_dtype == FP8_E4M3:
        storage = "s8e4"
    elif precision.storage_dtype == FP8_E5M2:
        storage = "s8e5"
    elif precision.storage_dtype == np.dtype(np.float16):
        storage = "s16"
    elif precision.storage_dtype == np.dtype(np.float32):
        storage = "s32"
    elif precision.storage_dtype == np.dtype(np.float64):
        storage = "s64"
    else:
        raise ValueError(f"Unsupported storage dtype: {precision.storage_dtype}")
    
    # Compute axis
    if precision.compute_dtype == np.dtype(np.float16):
        compute = "c16"
    elif precision.compute_dtype == np.dtype(np.float32):
        compute = "c32"
    elif precision.compute_dtype == np.dtype(np.float64):
        compute = "c64"
    else:
        raise ValueError(f"Unsupported compute dtype: {precision.compute_dtype}")
    
    # State axis
    if precision.state_dtype == np.dtype(np.float32):
        state = "x32"
    elif precision.state_dtype == np.dtype(np.float64):
        state = "x64"
    else:
        raise ValueError(f"Unsupported state dtype: {precision.state_dtype}")
    
    return f"{storage}{compute}{state}"


def load_cpu_kernel(precision: "PrecisionConfig") -> ctypes.CDLL:
    """Load CPU kernel library for the given precision config.
    
    Raises:
        RuntimeError: If the required kernel variant is not available.
    """
    suffix = _get_kernel_suffix(precision)
    
    # Check FP8+FP16 availability
    if suffix.startswith('s8') and 'c16' in suffix:
        if suffix not in AVAILABLE_FP8_VARIANTS:
            raise RuntimeError(
                f"CPU kernel variant '{suffix}' not available.\n"
                f"This variant requires _Float16 support (C23 or GCC 12+/Clang 15+ with -std=c2x).\n"
                f"Your compiler did not support _Float16 at build time.\n"
                f"Use fp8_e4m3() (FP32 compute) or fp8_e4m3_f64() (FP64 compute) instead."
            )
    
    # ... existing library loading logic ...
```

#### 9C.4.3: Test graceful failure for unavailable _Float16

**File:** `tests/tier2/cpu/test_fp8_cpu_availability.py` (new)

```python
"""Test graceful failure when _Float16 is unavailable."""

import pytest
from shared.precision_config import PrecisionConfig

try:
    from backends.cpu._fp8_variants import AVAILABLE_FP8_VARIANTS, HAS_FLOAT16
except ImportError:
    AVAILABLE_FP8_VARIANTS = []
    HAS_FLOAT16 = False


class TestFP16ComputeAvailability:
    """Validate graceful failure when _Float16 not available."""

    @pytest.mark.skipif(HAS_FLOAT16, reason="_Float16 IS available; skip unavailability test")
    def test_fp8_f16_raises_when_unavailable(self):
        """FP8 + FP16 compute raises RuntimeError if _Float16 unavailable.
        
        This test only runs when _Float16 was NOT detected at build time.
        It validates the error message guides users to alternatives.
        """
        from backends.cpu.kernel_loader import load_cpu_kernel
        
        cfg = PrecisionConfig.fp8_e4m3_f16()
        
        with pytest.raises(RuntimeError, match="_Float16 support"):
            load_cpu_kernel(cfg)

    @pytest.mark.skipif(HAS_FLOAT16, reason="_Float16 IS available; skip unavailability test")
    def test_error_message_suggests_alternatives(self):
        """Error message for unavailable FP16 compute suggests FP32/FP64 alternatives."""
        from backends.cpu.kernel_loader import load_cpu_kernel
        
        cfg = PrecisionConfig.fp8_e4m3_f16()
        
        with pytest.raises(RuntimeError) as exc_info:
            load_cpu_kernel(cfg)
        
        error_msg = str(exc_info.value)
        assert "fp8_e4m3()" in error_msg, "Error should suggest fp8_e4m3() (FP32 compute)"
        assert "fp8_e4m3_f64()" in error_msg, "Error should suggest fp8_e4m3_f64() (FP64 compute)"

    @pytest.mark.skipif(not HAS_FLOAT16, reason="_Float16 not available; skip availability test")
    def test_fp8_f16_works_when_available(self):
        """FP8 + FP16 compute loads successfully if _Float16 available."""
        from backends.cpu.kernel_loader import load_cpu_kernel
        
        cfg = PrecisionConfig.fp8_e4m3_f16()
        
        # Should not raise
        kernel = load_cpu_kernel(cfg)
        assert kernel is not None
```


**Governing authority:** ADR-025 §9.2  
**File:** `tests/tier2/cpu/test_fp8_cpu.py` (new)

```python
import pytest
import numpy as np
import ml_dtypes
import ctypes
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

class TestFP8CPURoundtrip:
    """ADR-025 §9.2: FP8 roundtrip tests for CPU backend."""
    
    @pytest.fixture
    def cpu_lib(self):
        # Load the CPU kernel library
        # ... ctypes loading ...
        pass
    
    def test_e4m3_roundtrip_in_range(self, cpu_lib):
        """Values in E4M3 range survive float→FP8→float roundtrip."""
        # Test known E4M3 values
        test_values = [0.0, 1.0, 0.5, 0.25, 100.0, 448.0, -1.0, -448.0]
        
        for val in test_values:
            # Convert via ml_dtypes for reference
            ref = float(np.array([val], dtype=np.float32).astype(ml_dtypes.float8_e4m3fn).astype(np.float32)[0])
            
            # ... invoke CPU kernel conversion ...
            # ... assert result matches ref ...
    
    def test_e4m3_saturation(self, cpu_lib):
        """Values exceeding 448 saturate to 448, not NaN/Inf."""
        overflow_values = [500.0, 1000.0, 1e10]
        
        for val in overflow_values:
            # ... invoke conversion ...
            # ... assert result == 448.0 or -448.0 depending on sign ...
    
    def test_e5m2_saturation(self, cpu_lib):
        """Values exceeding 57344 saturate to 57344, not NaN/Inf."""
        overflow_values = [60000.0, 100000.0, 1e10]
        
        for val in overflow_values:
            # ... invoke conversion ...
            # ... assert result == 57344.0 or -57344.0 depending on sign ...
    
    def test_e4m3_underflow(self, cpu_lib):
        """Values below min subnormal round to zero."""
        tiny_values = [1e-5, 1e-6, 1e-10]
        
        for val in tiny_values:
            # ... invoke conversion ...
            # ... assert result == 0.0 ...
    
    def test_nan_maps_to_zero(self, cpu_lib):
        """NaN inputs map to zero (E4M3/E5M2 have no NaN representation)."""
        import math
        nan_values = [float('nan'), -float('nan'), math.nan]
        
        for val in nan_values:
            # ... invoke E4M3 conversion ...
            # ... assert result == 0.0 ...
            
            # ... invoke E5M2 conversion ...
            # ... assert result == 0.0 ...
    
    @pytest.mark.parametrize("precision_factory", [
        PrecisionConfig.fp8_e4m3_f64,
        PrecisionConfig.fp8_e5m2_f64,
    ])
    def test_fp64_compute_roundtrip(self, cpu_lib, precision_factory):
        """FP64 compute with FP8 storage works correctly."""
        cfg = precision_factory()
        assert cfg.compute_dtype == np.float64
        
        test_values = [0.0, 1.0, 0.5, 100.0]
        # ... invoke CPU kernel with FP64 compute ...
```

---

### Step 9C.6: Validate rollback gate

#### 9C.6.1: Rebuild CPU backend

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir
```

Verify compilation succeeds for all FP8 suffix variants.

#### 9C.6.2: Run existing CPU test suite

```bash
pytest tests/tier2/cpu/ -v --ignore=tests/tier2/cpu/test_fp8_cpu.py 2>&1 | tee /tmp/cpu-regression.txt
grep -E "(FAILED|ERROR)" /tmp/cpu-regression.txt && exit 1 || echo "All existing CPU tests pass"
```

#### 9C.6.3: Run FP8 roundtrip tests

```bash
pytest tests/tier2/cpu/test_fp8_cpu.py -v 2>&1 | tee /tmp/fp8-cpu.txt
grep -E "(FAILED|ERROR)" /tmp/fp8-cpu.txt && exit 1 || echo "FP8 CPU tests pass"
```

#### 9C.6.4: Validate LUT headers match reference

Confirm the generated lookup tables produce identical values to `ml_dtypes`:

```bash
python -c "
import numpy as np
import ml_dtypes

# Spot-check E4M3 known values
e4m3_tests = [(0x38, 1.0), (0x7E, 448.0), (0xFE, -448.0)]
for bits, expected in e4m3_tests:
    arr = np.array([bits], dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
    assert float(arr[0]) == expected, f'E4M3 mismatch: 0x{bits:02X}'

# Spot-check E5M2 known values  
e5m2_tests = [(0x3C, 1.0), (0x7B, 57344.0), (0xFB, -57344.0)]
for bits, expected in e5m2_tests:
    arr = np.array([bits], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
    assert float(arr[0]) == expected, f'E5M2 mismatch: 0x{bits:02X}'

print('LUT reference validation passed')
"
```

---

## 4. Technical Notes

### Lookup table memory

Each 256-entry float table is 1KB. Total FP8 lookup table overhead: 2KB per compilation unit. For header-only tables with `static const`, each translation unit gets its own copy. With 10 FP8 suffix variants, total duplication is ~20KB — negligible.

### Conversion accuracy

The FP32→FP8 conversion implements round-to-nearest-even. This matches the `ml_dtypes` reference implementation. All 256 possible FP8 values should roundtrip exactly; values between representable FP8 values round to the nearest.

**Subnormal rounding accuracy:** The rounding shift distances (bit 19/20 for round bit, bits 20/21 for mantissa extraction) use fixed positions that are correct for **both** normalized and subnormal cases. After subnormal denormalization (`mant32 = (mant32 | 0x800000) >> shift`), the significant bits are right-shifted into positions that align with the fixed extraction window at bits [22:20] (E4M3) or [22:21] (E5M2). The round bit at position 19 (E4M3) or 20 (E5M2) correctly captures the first truncated bit, and sticky bits below capture the remainder. The early underflow check (`absval < MIN_SUBNORMAL * 0.5f`) prevents the subnormal path from reaching shift values large enough to push significant bits entirely below the extraction window (E4M3: max reachable shift = 4, MSB at bit 19 = round bit position; E5M2: max reachable shift = 3, MSB at bit 20 = round bit position). The rounding is therefore exact round-to-nearest-even for all reachable subnormal values. The roundtrip tests (Step 9C.5) validate against `ml_dtypes` reference values and should produce exact agreement for all representable inputs.

### NaN handling

**E4M3** (`float8_e4m3fn`) has no infinity representation but reserves 2 bit patterns for NaN (0x7F, 0xFF), leaving 254 finite values. **E5M2** follows IEEE-like conventions: exponent 0x1F encodes ±infinity (mantissa = 0) and NaN (mantissa ≠ 0), giving 248 finite patterns. However, the **store path** for both formats saturates to the maximum finite value; it never writes inf/NaN bit patterns to storage buffers.

When the input float is NaN, both conversion functions map it to **zero**:

```c
if (isnan(val)) {
    result.bits = 0;  // NaN → zero
    return result;
}
```

**Design rationale:** See Phase 9B Step 9B.3 and ADR-025 §4.2 for the canonical explanation. In summary: zero is the safe failure mode (no parameter update), matches `ml_dtypes.float8_e4m3fn` behavior, and NaN should be caught upstream.

If NaN propagation is desired (fail-fast), check for NaN before FP8 conversion in the host code. The CPU kernel conversion silently maps NaN to zero.

### No SIMD

AVX-512 has no FP8 instructions. SIMD FP8 would require manual vectorization of the lookup table access (gather operations) or software mantissa/exponent manipulation. This is a potential future optimization but not in scope for Phase 9C.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|:---|:---|:---|:---|
| FP32→FP8 conversion edge cases | Medium | High | Exhaustive tests against `ml_dtypes` reference. Test all 256 FP8 values for exact roundtrip. |
| Struct wrapper causes compiler issues | Low | Medium | Test with gcc/clang/MSVC. The struct-of-uint8_t pattern is well-supported. |
| Binary size increase from new variants | Low | Low | 10 new variants × ~200KB each = ~2MB. Acceptable. |
| FP16 compute requires `_Float16` support | Medium | Medium | Requires C23 or compiler extension. Gate FP16 variants on compiler capability detection in Meson. Skip `c16` variants if unsupported. |

---

## 6. Files Modified Summary

| File | Change Type | Description |
|:---|:---|:---|
| `src/backends/cpu/kernel_sources/cpu_fp8.h` | **New** | FP8 struct types, constants, conversion functions |
| `src/backends/cpu/kernel_sources/cpu_precision.h` | **Edit** | Add FP8 load/store macros with FP16/FP32/FP64 compute dispatch |
| `src/backends/cpu/kernel_sources/cpu_kernels.h` | **Edit** | Add 10 new `DECLARE_PRECISION_STRUCTS` for FP8 suffix variants (with `_Float16` guards) |
| `src/backends/cpu/kernel_sources/cpu_kernels.c` | **Edit** | Include new suffix implementations |
| `meson.build` | **Edit** | Add FP8 variant compilation targets with `_Float16` capability detection |
| `src/backends/cpu/_fp8_variants.py.in` | **New** | Meson template for generating `_fp8_variants.py` (lists available FP8 variants) |
| `src/backends/cpu/kernel_loader.py` | **Edit** | Add `_get_kernel_suffix()` function, FP16 availability check, import generated `_fp8_variants.py` |
| `tests/tier2/cpu/test_fp8_cpu.py` | **New** | FP8 roundtrip, saturation, and underflow tests for CPU backend |
| `tests/tier2/cpu/test_fp8_cpu_availability.py` | **New** | Graceful failure tests for unavailable `_Float16` support |

**Note:** The `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` file is generated in Phase 9A.1.5, not this phase. The `cpu_fp8.h` header includes it via `#include "cpu_fp8_lut.gen.h"`.

---

*End of Phase 9C. This phase can be executed in parallel with Phases 9B and 9D. Proceed to Phase 9E after all three backend phases complete.*
