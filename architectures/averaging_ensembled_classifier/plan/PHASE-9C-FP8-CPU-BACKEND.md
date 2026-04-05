# Phase 9C: FP8 Support — CPU Backend Implementation

**Status: NOT STARTED**  
**Phase:** 9C of 9  
**Prerequisite:** Phase 9A complete and rollback gate passed (includes LUT generation in Step 9A.1.5).  
**Note on parallelization:** Phases 9B/9C/9D are fully parallelizable since LUT generation moved to Phase 9A.

**Objective:** Implement FP8 software types and conversion functions in the CPU backend. Add `cpu_fp8.h` with E4M3/E5M2 structs and conversion functions. Extend `cpu_precision.h` with FP8 load/store macros. Generate ten new kernel suffix variants (six for FP32/FP64 compute, four for FP16 compute). Update Meson build. The phase ends when FP8 CPU kernels compile and pass roundtrip tests.  
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
#include <float.h>   /* for __FLT16_MAX__ detection */

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
 * sign(1) + exp(5) + mantissa(2), bias=15, max=57344, no inf/nan
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

/* --- E4M3 Conversion Functions (ADR-025 §5.2) --- */

static inline float cpu_fp8_e4m3_to_float(cpu_fp8_e4m3 val) {
    return cpu_fp8_e4m3_to_float_lut[val.bits];
}

static inline double cpu_fp8_e4m3_to_double(cpu_fp8_e4m3 val) {
    return (double)cpu_fp8_e4m3_to_float_lut[val.bits];
}

/* FP16 conversion — guarded by _Float16 availability */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L) || defined(__FLT16_MAX__)
static inline _Float16 cpu_fp8_e4m3_to_half(cpu_fp8_e4m3 val) {
    return (_Float16)cpu_fp8_e4m3_to_float_lut[val.bits];
}
#endif

static inline cpu_fp8_e4m3 cpu_float_to_fp8_e4m3(float val) {
    cpu_fp8_e4m3 result;
    
    /* Handle special cases */
    if (isnan(val)) {
        /* E4M3 has no NaN representation — all 256 bit patterns are finite values.
         * This is a design choice in the ml_dtypes FP8 formats (ADR-025 §1).
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
    
    /* Saturation (no infinity in E4M3) */
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
        mant32 = (mant32 | 0x800000) >> shift;  /* Denormalize */
        exp8 = 0;
    } else if (exp8 >= 15) {
        /* Overflow — should not reach here due to saturation above */
        result.bits = (sign << 7) | 0x7E;
        return result;
    }
    
    /* Round mantissa to 3 bits (round-to-nearest-even) */
    uint32_t mant8 = (mant32 + (1 << 19)) >> 20;  /* Round and shift */
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
    return cpu_float_to_fp8_e4m3((float)val);
}

/* FP16 conversion — guarded by _Float16 availability */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L) || defined(__FLT16_MAX__)
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
    
    /* Saturation */
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
        mant32 = (mant32 | 0x800000) >> shift;
        exp8 = 0;
    } else if (exp8 >= 31) {
        result.bits = (sign << 7) | 0x7B;
        return result;
    }
    
    /* Round mantissa to 2 bits */
    uint32_t mant8 = (mant32 + (1 << 20)) >> 21;
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
    return cpu_float_to_fp8_e5m2((float)val);
}

/* FP16 conversion — guarded by _Float16 availability */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L) || defined(__FLT16_MAX__)
static inline _Float16 cpu_fp8_e5m2_to_half(cpu_fp8_e5m2 val) {
    return (_Float16)cpu_fp8_e5m2_to_float_lut[val.bits];
}

static inline cpu_fp8_e5m2 cpu_half_to_fp8_e5m2(_Float16 val) {
    return cpu_float_to_fp8_e5m2((float)val);
}
#endif

#endif /* CPU_FP8_H */
```

#endif /* CPU_FP8_H */
```

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

---

### Step 9C.3: Add FP8 suffix variants

**Governing authority:** ADR-025 §5.3  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.h`

Add ten new suffix declarations (6 for FP32/FP64 compute, 4 for FP16 compute).

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

/* FP8 storage variants with FP16 compute — requires _Float16 (C23 / GCC 12+ / Clang 15+)
 * These are conditionally compiled to prevent parse errors on unsupported compilers.
 * The Meson build (Step 9C.4) also skips c16 variant compilation when _Float16 is unavailable.
 */
#if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L) || defined(__FLT16_MAX__)
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
# FP8 E4M3 variants with FP32/FP64 compute
cpu_fp8_e4m3_variants = [
  {'suffix': 's8e4c32x32', 'storage_e4m3': true, 'compute_half': false, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e4c32x64', 'storage_e4m3': true, 'compute_half': false, 'compute_double': false, 'state_double': true},
  {'suffix': 's8e4c64x64', 'storage_e4m3': true, 'compute_half': false, 'compute_double': true, 'state_double': true},
  {'suffix': 's8e4c16x32', 'storage_e4m3': true, 'compute_half': true, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e4c16x64', 'storage_e4m3': true, 'compute_half': true, 'compute_double': false, 'state_double': true},
]

# FP8 E5M2 variants
cpu_fp8_e5m2_variants = [
  {'suffix': 's8e5c32x32', 'storage_e5m2': true, 'compute_half': false, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e5c32x64', 'storage_e5m2': true, 'compute_half': false, 'compute_double': false, 'state_double': true},
  {'suffix': 's8e5c64x64', 'storage_e5m2': true, 'compute_half': false, 'compute_double': true, 'state_double': true},
  {'suffix': 's8e5c16x32', 'storage_e5m2': true, 'compute_half': true, 'compute_double': false, 'state_double': false},
  {'suffix': 's8e5c16x64', 'storage_e5m2': true, 'compute_half': true, 'compute_double': false, 'state_double': true},
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

# Initialize list to collect FP8 kernel libraries
cpu_fp8_kernel_libs = []

# Build FP8 variants, conditionally skipping c16 suffixes if _Float16 unavailable
foreach variant : cpu_fp8_e4m3_variants + cpu_fp8_e5m2_variants
  # Skip FP16 compute variants if compiler lacks _Float16
  if variant.get('compute_half', false) and not has_float16
    message('Skipping ' + variant['suffix'] + ' (requires _Float16)')
    continue
  endif
  
  # Build compile flags for this variant
  cflags = ['-DSTORAGE_TYPE_IS_FP8=1']
  
  if variant.get('storage_e4m3', false)
    cflags += '-DSTORAGE_TYPE_IS_E4M3=1'
    cflags += '-DSTORAGE_TYPE_IS_E5M2=0'
  else
    cflags += '-DSTORAGE_TYPE_IS_E4M3=0'
    cflags += '-DSTORAGE_TYPE_IS_E5M2=1'
  endif
  
  if variant.get('compute_half', false)
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
  if variant.get('compute_half', false)
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
foreach variant : cpu_fp8_e4m3_variants + cpu_fp8_e5m2_variants
  if variant.get('compute_half', false) and not has_float16
    continue
  endif
  available_fp8_variants += variant['suffix']
endforeach

# Build the list string for Python
fp8_list_items = []
foreach v : available_fp8_variants
  fp8_list_items += '"' + v + '"'
endforeach

conf_data = configuration_data()
conf_data.set('AVAILABLE_FP8_VARIANTS', '[' + ', '.join(fp8_list_items) + ']')
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
    """Determine kernel suffix from precision config."""
    # ... existing suffix logic ...
    return suffix


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

---

### Step 9C.5: Add FP8 CPU roundtrip tests

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

Each 256-entry float table is 1KB. Total FP8 lookup table overhead: 2KB per compilation unit. For header-only tables with `static const`, each translation unit gets its own copy. Consider moving to a shared object init function if binary size becomes a concern.

### Conversion accuracy

The FP32→FP8 conversion implements round-to-nearest-even. This matches the `ml_dtypes` reference implementation. All 256 possible FP8 values should roundtrip exactly; values between representable FP8 values round to the nearest.

### NaN handling

E4M3 and E5M2 have no NaN representation — all 256 bit patterns map to finite values. When the input float is NaN, the conversion functions map it to **zero**:

```c
if (isnan(val)) {
    result.bits = 0;  // NaN → zero
    return result;
}
```

**Design rationale:** Zero is chosen over max-saturation because:
1. NaN in gradients typically indicates numerical instability that should be caught upstream
2. Zero gradients are "safe" (no parameter update) whereas max-value gradients could cause divergence
3. This matches `ml_dtypes.float8_e4m3fn` behavior

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
| `tests/tier2/cpu/test_fp8_cpu.py` | **New** | FP8 roundtrip, saturation, and underflow tests for CPU backend |

**Note:** The `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` file is generated in Phase 9A.1.5, not this phase. The `cpu_fp8.h` header includes it via `#include "cpu_fp8_lut.gen.h"`.
