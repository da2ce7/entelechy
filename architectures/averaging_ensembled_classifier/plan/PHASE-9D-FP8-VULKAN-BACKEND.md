# Phase 9D: FP8 Support — Vulkan Backend Implementation

**Status: NOT STARTED**  
**Phase:** 9D of 9  
**Prerequisite:** Phase 9A complete and rollback gate passed (Phases 9B/9C/9D are fully parallelizable since LUT generation is in 9A.1.5).  
**Objective:** Implement FP8 software conversion in Vulkan shaders. Extend specialization constants, add FP8 conversion functions to `common.glsl` supporting both FP16 and FP32 compute, generate shader variants. The phase ends when FP8 shader variants compile and pass roundtrip tests.  
**Governing ADR:** ADR-025 (§7)  
**Rollback gate:** All existing Vulkan tests pass. FP8 shader compilation succeeds. FP8 roundtrip tests pass with ≤1 ULP error.  
**Dependencies:** Phase 9A (PrecisionConfig extension) complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 9D.1: Extend compile-time flags](#step-9d1-extend-compile-time-flags)
   - [Step 9D.2: Add FP8 conversion functions to `common.glsl`](#step-9d2-add-fp8-conversion-functions-to-commonglsl)
   - [Step 9D.3: Extend storage buffer declarations](#step-9d3-extend-storage-buffer-declarations)
   - [Step 9D.4: Generate FP8 SPIR-V variants](#step-9d4-generate-fp8-spir-v-variants)
   - [Step 9D.5: Add FP8 Vulkan roundtrip tests](#step-9d5-add-fp8-vulkan-roundtrip-tests)
   - [Step 9D.6: Validate rollback gate](#step-9d6-validate-rollback-gate)
4. [Technical Notes](#4-technical-notes)
5. [Risk Register](#5-risk-register)
6. [Files Modified Summary](#6-files-modified-summary)

---

## 1. Scope & Constraints

### In scope

- Extending Vulkan specialization constants with `FP8_VARIANT` selector (ADR-025 §7.2).
- Adding FP8→float conversion functions in `common.glsl` (ADR-025 §7.3).
- Extending storage buffer declarations to handle `uint8_t` for FP8.
- Generating SPIR-V shader variants for FP8 configurations.
- Adding Vulkan-specific FP8 roundtrip tests.

### Out of scope

- Native Vulkan FP8 extension support (`VK_FP8_NATIVE`) — no such extension exists.
- OpenCL backend — completed in Phase 9B.
- CPU backend — completed in Phase 9C.
- Host-side scaling — deferred to Phase 9E.

### Key constraint: FP8 as `uint8_t`

GLSL and SPIR-V have no native FP8 type. FP8 storage buffers use `uint8_t` (via `VK_KHR_8bit_storage` extension). Conversion to `COMPUTE_TYPE` (either `float16_t` via `VK_KHR_shader_float16_int8`, or `float`) happens in shader code via bit manipulation.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/backends/vulkan/kernel_sources/common.glsl` | Precision boundary abstractions for FP16/FP32/FP64. `-D` macro injection for precision roles. No FP8 handling. |
| `src/backends/vulkan/shader_builder.py` | Compiles shaders with `-D` flag injection. No FP8 variants. |
| `src/backends/vulkan/renderer.py` | Buffer allocation and shader dispatch. Uses `VkBuffer` with appropriate format. |
| `tests/tier2/vulkan/` | Vulkan backend tests. No FP8-specific tests. |

---

## 3. Task Breakdown

---

### Step 9D.1: Extend compile-time flags

**Governing authority:** ADR-025 §7.2  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Extend the compile-time `-D` flag scheme. The GLSL preprocessor cannot evaluate specialization constant values, so FP8 dispatch uses compile-time flags injected via `glslc -D`:

```glsl
// FP8 compile-time flags (ADR-025 §7.2)
// Injected by glslc: -DSTORAGE_TYPE_IS_FP8=1 -DSTORAGE_TYPE_IS_E4M3=1 (or -DSTORAGE_TYPE_IS_E5M2=1)
// When FP8: -DCOMPUTE_TYPE_IS_HALF=1 or -DCOMPUTE_TYPE_IS_FLOAT=1

#ifndef STORAGE_TYPE_IS_FP8
#define STORAGE_TYPE_IS_FP8 0
#endif
#ifndef STORAGE_TYPE_IS_E4M3
#define STORAGE_TYPE_IS_E4M3 0
#endif
#ifndef STORAGE_TYPE_IS_E5M2
#define STORAGE_TYPE_IS_E5M2 0
#endif
#ifndef COMPUTE_TYPE_IS_HALF
#define COMPUTE_TYPE_IS_HALF 0
#endif
#ifndef COMPUTE_TYPE_IS_FLOAT
#define COMPUTE_TYPE_IS_FLOAT 0
#endif
#ifndef COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_TYPE_IS_DOUBLE 0
#endif

// Enable required extensions
#if STORAGE_TYPE_IS_FP8
// VK_KHR_8bit_storage (Vulkan extension) → GL_EXT_shader_8bit_storage (GLSL pragma)
#extension GL_EXT_shader_8bit_storage : require
#endif

#if COMPUTE_TYPE_IS_HALF
// VK_KHR_shader_float16_int8 (Vulkan extension) → GL_EXT_shader_explicit_arithmetic_types_float16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#if COMPUTE_TYPE_IS_DOUBLE
// FP64 requires shaderFloat64 device feature
// Note: Vulkan GLSL uses GL_EXT, not GL_ARB (which is for desktop GL)
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif
```

**Note:** FP64 compute configurations (`fp8_e4m3_f64()`, `fp8_e5m2_f64()`) require the `shaderFloat64` Vulkan device feature. The renderer must check `VkPhysicalDeviceFeatures.shaderFloat64` at device selection and raise `RuntimeError` if FP64 compute is requested on unsupported hardware.

This mirrors the existing `-D` macro pattern used for `STORAGE_TYPE`, `COMPUTE_TYPE`, and `STATE_TYPE` (see [common.glsl](src/backends/vulkan/kernel_sources/common.glsl)).

> **Naming convention note:** All backends (OpenCL, CPU, and Vulkan) use the canonical CONTRACT.md Article 6 flag names: `STORAGE_TYPE_IS_FP8`, `STORAGE_TYPE_IS_E4M3`, `STORAGE_TYPE_IS_E5M2`, `COMPUTE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_DOUBLE`, etc. The Vulkan GLSL type macros (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`) also use the same canonical names as OpenCL, injected via `glslc -D`. No backend-specific shortened names are used.

---

### Step 9D.2: Add FP8 conversion functions to `common.glsl`

**Governing authority:** ADR-025 §7.3  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

> **Arithmetic vs. Lookup Table approach:**
> Unlike the OpenCL and CPU backends which use 256-entry lookup tables (LUTs) for FP8→float conversion, Vulkan uses **arithmetic conversion** via bit manipulation and `exp2()`. This is deliberate:
> - GLSL lacks OpenCL's `__constant` memory space for kernel-scope constant arrays
> - SPIR-V specialization constants could embed tables, but inflate shader binaries
> - Arithmetic conversion uses ~10-15 ALU ops — negligible overhead for bandwidth-bound workloads
> - The 4× storage bandwidth savings dominate any conversion overhead
>
> See [Technical Notes §4](#4-technical-notes) for detailed comparison.

Add software FP8↔float/float16 conversion:

```glsl
// --- FP8 Conversion Constants ---
#if STORAGE_TYPE_IS_FP8

#if STORAGE_TYPE_IS_E4M3
#define FP8_BIAS 7
#define FP8_MANTISSA_BITS 3
#define FP8_MAX_VAL 448.0
#define FP8_MIN_SUBNORMAL 0.001953125  // 2^-9
#elif STORAGE_TYPE_IS_E5M2
#define FP8_BIAS 15
#define FP8_MANTISSA_BITS 2
#define FP8_MAX_VAL 57344.0
#define FP8_MIN_SUBNORMAL 0.0000152587890625  // 2^-16, exact
#endif

// --- FP8 → float conversion (intermediate, lossless for FP8 values) ---
// All compute types (FP16/FP32/FP64) use float as the intermediate.
// This is exact: all 256 FP8 bit patterns map to exactly representable float values.
// FP16 compute narrows (lossless for FP8 values since FP8 ⊂ FP16); FP64 compute widens (lossless).
float fp8_to_float_internal(uint bits) {
    // Extract components
    uint sign = (bits >> 7) & 1u;
    
#if STORAGE_TYPE_IS_E4M3
    uint exp8 = (bits >> 3) & 0xFu;   // 4-bit exponent
    uint mant = bits & 0x7u;          // 3-bit mantissa
    
    float result;
    if (exp8 == 0u) {
        // Subnormal: value = mant * 2^(1-bias-mantissa_bits) = mant * 2^(-9)
        result = float(mant) * 0.001953125;  // 2^-9, exact constant
    } else {
        // Normal: value = (1 + mant/8) * 2^(exp8-bias)
        // Use standard GLSL exp2() for power-of-2 scaling
        float m = 1.0 + float(mant) * 0.125;  // mant/8
        result = m * exp2(float(int(exp8) - FP8_BIAS));
    }
    
#elif STORAGE_TYPE_IS_E5M2
    uint exp8 = (bits >> 2) & 0x1Fu;  // 5-bit exponent
    uint mant = bits & 0x3u;          // 2-bit mantissa
    
    float result;
    if (exp8 == 0x1Fu) {
        // IEEE special exponent: inf (mant==0) or NaN (mant!=0)
        // The store path never writes these patterns, but handle for safety.
        if (mant != 0u) {
            // NaN → zero: Returns BEFORE sign application. This is intentional —
            // NaN has no meaningful sign, and +0.0 is the correct "safe" value
            // regardless of the sign bit in the NaN bit pattern.
            return 0.0;
        }
        // ±inf → ±MAX: Saturate to max finite E5M2 value. Sign is applied below
        // by the shared `sign == 1u ? -result : result` return at the end of the
        // function. This produces correctly-signed ±57344.0.
        result = 57344.0;
    } else if (exp8 == 0u) {
        // Subnormal: value = mant * 2^(1-bias-mantissa_bits) = mant * 2^(-16)
        result = float(mant) * 0.0000152587890625;  // 2^-16, exact constant
    } else {
        // Normal: value = (1 + mant/4) * 2^(exp8-bias)
        float m = 1.0 + float(mant) * 0.25;  // mant/4
        result = m * exp2(float(int(exp8) - FP8_BIAS));
    }
#endif
    
    return sign == 1u ? -result : result;
}

// --- COMPUTE_TYPE-aware load ---
// Returns float16_t, float, or double depending on compute precision
#if COMPUTE_TYPE_IS_HALF
float16_t load_storage_fp8(uint bits) {
    return float16_t(fp8_to_float_internal(bits));
}
#elif COMPUTE_TYPE_IS_DOUBLE
double load_storage_fp8(uint bits) {
    return double(fp8_to_float_internal(bits));  // Widen to FP64 (lossless)
}
#else  // FP32 compute (default)
float load_storage_fp8(uint bits) {
    return fp8_to_float_internal(bits);
}
#endif

// --- float → FP8 conversion ---
// Always works in float, then quantizes to FP8
//
// NaN handling: NaN input → zero for both E4M3 and E5M2.
// See Phase 9B Step 9B.3 and ADR-025 §4.2 for the canonical rationale
// (safe failure mode, matches ml_dtypes behavior, upstream should catch NaN).
uint store_storage_fp8_internal(float val) {
    // NaN → zero (same rationale as OpenCL/CPU backends)
    if (isnan(val)) return 0u;
    
    uint sign = val < 0.0 ? 1u : 0u;
    val = abs(val);
    
    // Saturation: >= is intentional. FP8_MAX_VAL is exactly representable and
    // encodes to max bit pattern, so == short-circuits to the same result the
    // rounding path would produce. Consistent with OpenCL/CPU backends.
    if (val >= FP8_MAX_VAL) {
#if STORAGE_TYPE_IS_E4M3
        return (sign << 7) | 0x7Eu;  // Max E4M3
#elif STORAGE_TYPE_IS_E5M2
        return (sign << 7) | 0x7Bu;  // Max E5M2
#endif
    }
    
    // Underflow
    if (val < FP8_MIN_SUBNORMAL * 0.5) {
        return sign << 7;  // Zero
    }
    
    // Extract FP32 components via bit cast
    uint fbits = floatBitsToUint(val);
    int exp32 = int((fbits >> 23) & 0xFFu) - 127;
    uint mant32 = fbits & 0x7FFFFFu;
    
    // Compute FP8 exponent
    int exp8 = exp32 + FP8_BIAS;
    
#if STORAGE_TYPE_IS_E4M3
    // Handle subnormals and rounding for E4M3
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            return sign << 7;  // Underflow to zero (shift >= 24 is UB)
        }
        mant32 = (mant32 | 0x800000u) >> shift;
        exp8 = 0;
    } else if (exp8 >= 15) {
        return (sign << 7) | 0x7Eu;
    }
    
    // Round to 3 bits (round-to-nearest-even)
    uint round_bit = (mant32 >> 19) & 1u;
    uint sticky = mant32 & ((1u << 19) - 1u);
    uint mant8 = mant32 >> 20;
    if (round_bit != 0u && (sticky != 0u || (mant8 & 1u) != 0u)) {
        mant8++;
    }
    if (mant8 >= 8u) {
        mant8 = 0u;
        exp8++;
        if (exp8 >= 15) return (sign << 7) | 0x7Eu;
    }
    
    return (sign << 7) | (uint(exp8) << 3) | (mant8 & 0x7u);
    
#elif STORAGE_TYPE_IS_E5M2
    // Handle subnormals and rounding for E5M2
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            return sign << 7;  // Underflow to zero (shift >= 24 is UB)
        }
        mant32 = (mant32 | 0x800000u) >> shift;
        exp8 = 0;
    } else if (exp8 >= 31) {
        return (sign << 7) | 0x7Bu;
    }
    
    // Round to 2 bits (round-to-nearest-even)
    uint round_bit = (mant32 >> 20) & 1u;
    uint sticky = mant32 & ((1u << 20) - 1u);
    uint mant8 = mant32 >> 21;
    if (round_bit != 0u && (sticky != 0u || (mant8 & 1u) != 0u)) {
        mant8++;
    }
    if (mant8 >= 4u) {
        mant8 = 0u;
        exp8++;
        if (exp8 >= 31) return (sign << 7) | 0x7Bu;
    }
    
    return (sign << 7) | (uint(exp8) << 2) | (mant8 & 0x3u);
#endif
}

// --- COMPUTE_TYPE-aware store ---
// All paths narrow to float for bit manipulation, then quantize to FP8
#if COMPUTE_TYPE_IS_HALF
uint store_storage_fp8(float16_t val) {
    return store_storage_fp8_internal(float(val));
}
#elif COMPUTE_TYPE_IS_DOUBLE
uint store_storage_fp8(double val) {
    // NaN check BEFORE clamp: GLSL clamp(NaN, lo, hi) returns lo per spec,
    // so NaN would be silently converted to -FLT_MAX by the clamp below.
    if (isnan(val)) return 0u;  // NaN → zero (see rationale in store_storage_fp8_internal)
    // Defense-in-depth: clamp before double→float narrowing to prevent UB
    // (GLSL spec §4.1.4: double→float overflow is implementation-defined/UB).
    // Pre-scaling (Phase 9E) ensures values are in FP8 range ⊂ FP32 range,
    // so this clamp is redundant under correct operation. See "Note on FP64
    // precision loss" below.
    double clamped = clamp(val, -double(3.4028235e+38), double(3.4028235e+38));
    return store_storage_fp8_internal(float(clamped));
}
#else  // FP32 compute (default)
uint store_storage_fp8(float val) {
    return store_storage_fp8_internal(val);
}
#endif

#endif // STORAGE_TYPE_IS_FP8
```

**Note on FP64 precision loss:** When storing FP64 compute results to FP8, values are first narrowed to float (which may lose precision for values outside float's range), then quantized to FP8. This is acceptable because:
1. FP8's range (max 448 or 57344) is well within float's range
2. Values destined for FP8 storage should already be scaled to fit
3. The narrowing step is lossless for typical activation/gradient magnitudes

> **FP64 narrowing is safe because values are pre-scaled.** The host-side FP8 scaling path
> (Phase 9E) ensures all values are within `storage_fp_format_max` (448 or 57344) before
> shader invocation. Since both FP8 max values are well within float's range (~3.4e38),
> the intermediate `float(val)` cast is exact for all values that survive the scaling step.
> The `#elif COMPUTE_TYPE_IS_DOUBLE` defense-in-depth clamp in the code above prevents UB
> unconditionally, even if upstream scaling is bypassed or buggy.

The key insight: FP8↔FP16 conversion routes through FP32 intermediate since all FP8 values are exactly representable in FP32.

---

#### Step 9D.2.1: Pre-flight audit of `.comp` files for unguarded `STORAGE_TYPE` references

Before modifying any `.comp` shader file, audit for existing `STORAGE_TYPE` references that assume FP16/FP32/FP64 and would break when `STORAGE_TYPE` becomes `uint8_t`:

```bash
# Find all unguarded STORAGE_TYPE references in .comp files
grep -rn 'STORAGE_TYPE' src/backends/vulkan/kernel_sources/*.comp \
    | grep -v 'STORAGE_TYPE_IS_FP8' \
    | grep -v 'STORAGE_TYPE_IS_E4M3' \
    | grep -v 'STORAGE_TYPE_IS_E5M2' \
    | grep -v '#if\|#elif\|#ifdef\|#ifndef\|#define' \
    | tee /tmp/storage-type-audit.txt
```

Each match in `/tmp/storage-type-audit.txt` is a location that may need an `#if STORAGE_TYPE_IS_FP8` guard or a switch to the `LOAD_STORAGE` / `STORE_STORAGE` macros. Categorize each match:

- **Buffer declarations** (`layout(...) buffer ... { STORAGE_TYPE data[]; }`) — these get `#if`/`#else` branching in Step 9D.3
- **Direct buffer reads** (`buf.data[i]` cast to `COMPUTE_TYPE`) — these must use `LOAD_STORAGE(i)` macro
- **Direct buffer writes** (`buf.data[i] = val`) — these must use `STORE_STORAGE(i, val)` macro
- **Sizeof/stride calculations** — these must use `1` when `STORAGE_TYPE_IS_FP8` (since `uint8_t` is 1 byte)

**This audit is blocking** — do not proceed to Step 9D.3 without reviewing every match. Unguarded `STORAGE_TYPE` usage with `uint8_t` will produce corrupted reads (interpreting raw bytes as floats).

### Step 9D.3: Extend storage buffer declarations

**Governing authority:** ADR-025 §7  
**File:** `src/backends/vulkan/kernel_sources/*.comp`

Modify storage buffer declarations to use `uint8_t` when FP8:

```glsl
// Example buffer declaration with FP8 support
#if STORAGE_TYPE_IS_FP8
// Requires VK_KHR_8bit_storage extension
layout(set = 0, binding = 0) buffer StorageBuffer {
    uint8_t data[];
} storage_buf;

#define LOAD_STORAGE(idx) load_storage_fp8(uint(storage_buf.data[idx]))
#define STORE_STORAGE(idx, val) storage_buf.data[idx] = uint8_t(store_storage_fp8(val))

#else
// Existing FP16/FP32/FP64 declarations
// ...
#endif
```

#### 9D.3.1: Extension requirements

The Vulkan renderer must enable appropriate extensions for FP8 + FP16 configs:

```python
# In renderer initialization (during VkDevice creation)
def _check_precision_requirements(self, precision: PrecisionConfig) -> None:
    """Validate device supports required features for precision config."""
    
    # FP8 storage requires VK_KHR_8bit_storage
    if precision.storage_dtype in FP8_DTYPES:
        if "VK_KHR_8bit_storage" not in self._enabled_extensions:
            raise RuntimeError(
                f"FP8 storage requires VK_KHR_8bit_storage extension. "
                f"This device does not support it. "
                f"Use float32() or mixed_f16_f32() instead."
            )
    
    # FP16 compute requires VK_KHR_shader_float16_int8
    if precision.compute_dtype == np.float16:
        if "VK_KHR_shader_float16_int8" not in self._enabled_extensions:
            raise RuntimeError(
                f"FP16 compute requires VK_KHR_shader_float16_int8 extension. "
                f"Use fp8_e4m3() (FP32 compute) instead of fp8_e4m3_f16()."
            )
    
    # FP64 compute requires shaderFloat64 device feature (checked at device enumeration)
    if precision.compute_dtype == np.float64:
        if not self._physical_device_features.shaderFloat64:
            raise RuntimeError(
                f"FP64 compute requested but shaderFloat64 feature not supported. "
                f"Use fp8_e4m3() (FP32 compute) instead of fp8_e4m3_f64()."
            )
```

**Device enumeration integration:** The `shaderFloat64` feature is queried via `vkGetPhysicalDeviceFeatures()` during device enumeration (before `VkDevice` creation). Store the result in `_physical_device_features` for runtime checks. Extensions are checked via `vkEnumerateDeviceExtensionProperties()`.

FP16 compute with FP8 storage requires both `VK_KHR_8bit_storage` (for uint8_t buffers) and `VK_KHR_shader_float16_int8` (for float16_t arithmetic). FP64 compute requires the `shaderFloat64` Vulkan device feature, which is checked via `VkPhysicalDeviceFeatures` at device selection.

---

### Step 9D.4: Generate FP8 SPIR-V variants

**Governing authority:** ADR-025 §7  
**File:** `src/backends/vulkan/shader_builder.py`

Extend shader compilation to generate FP8 variants using `-D` compile flags:

```python
def compile_shader_variant(self, shader_path: Path, precision: PrecisionConfig) -> bytes:
    """Compile shader with precision-specific -D flags."""
    
    flags = []
    
    # FP8 storage flags (ADR-025 §7.2)
    is_fp8 = precision.storage_dtype in FP8_DTYPES
    is_e4m3 = precision.storage_dtype == FP8_E4M3
    is_e5m2 = precision.storage_dtype == FP8_E5M2
    
    flags.extend([
        f"-DSTORAGE_TYPE_IS_FP8={1 if is_fp8 else 0}",
        f"-DSTORAGE_TYPE_IS_E4M3={1 if is_e4m3 else 0}",
        f"-DSTORAGE_TYPE_IS_E5M2={1 if is_e5m2 else 0}",
    ])
    
    # When FP8: STORAGE_TYPE is NOT emitted (FP8 uses uint8_t buffers, not a
    # floating-point type). All existing code that references STORAGE_TYPE must
    # be guarded with `#if !STORAGE_TYPE_IS_FP8`. Storage buffer access in FP8 mode
    # goes exclusively through load_storage_fp8/store_storage_fp8.
    
    # Compute type flags for FP8 dispatch
    is_compute_fp16 = precision.compute_dtype == np.float16
    is_compute_fp32 = precision.compute_dtype == np.float32
    
    flags.extend([
        f"-DCOMPUTE_TYPE_IS_HALF={1 if is_compute_fp16 else 0}",
        f"-DCOMPUTE_TYPE_IS_FLOAT={1 if is_compute_fp32 else 0}",
        f"-DCOMPUTE_TYPE_IS_DOUBLE={1 if precision.compute_dtype == np.float64 else 0}",
    ])
    
    # Enable extensions as needed
    if is_fp8:
        flags.append("-DENABLE_8BIT_STORAGE=1")
    if is_compute_fp16:
        flags.append("-DENABLE_FP16_EXTENSION=1")
    
    # Existing STORAGE_TYPE/COMPUTE_TYPE/STATE_TYPE macros for non-FP8 paths
    # Note: FP8 storage uses uint8_t buffers, not a floating-point type.
    # STORAGE_TYPE is used only for FP16/FP32/FP64 storage; FP8 paths use STORAGE_TYPE_IS_FP8 flag.
    if not is_fp8:
        storage_glsl_type = {
            np.float16: "float16_t",
            np.float32: "float",
            np.float64: "double",
        }[precision.storage_dtype.type]
        flags.append(f"-DSTORAGE_TYPE={storage_glsl_type}")
    # For FP8, STORAGE_TYPE is not emitted — shaders use STORAGE_TYPE_IS_FP8 conditional
    # ... additional flags ...
    
    # Compile with glslc
    cmd = ["glslc", "-fshader-stage=compute", *flags, str(shader_path), "-o", "-"]
    # ...
```

---

### Step 9D.5: Add FP8 Vulkan roundtrip tests

**Governing authority:** ADR-025 §9.2  
**File:** `tests/tier2/vulkan/test_fp8_vulkan.py` (new)

```python
import pytest
import numpy as np
import ml_dtypes
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

# --- Test Helper Functions ---

def _get_vulkan_extensions() -> tuple[set[str], int]:
    """Query available Vulkan device extensions and API version.
    
    Uses vulkan module to enumerate extensions on the first physical device.
    Returns (extensions set, api_version) or (empty set, 0) if Vulkan is unavailable.
    """
    try:
        import vulkan as vk
        
        # Create instance
        app_info = vk.VkApplicationInfo(
            pApplicationName='extension_query',
            applicationVersion=1,
            pEngineName='test',
            engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_2,
        )
        instance_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        instance = vk.vkCreateInstance(instance_info, None)
        
        # Get first physical device
        devices = vk.vkEnumeratePhysicalDevices(instance)
        if not devices:
            vk.vkDestroyInstance(instance, None)
            return set(), 0
        
        # Query device properties for API version
        props = vk.vkGetPhysicalDeviceProperties(devices[0])
        api_version = props.apiVersion
        
        # Query extensions
        extensions = vk.vkEnumerateDeviceExtensionProperties(devices[0], None)
        ext_names = {ext.extensionName for ext in extensions}
        
        vk.vkDestroyInstance(instance, None)
        return ext_names, api_version
        
    except (ImportError, RuntimeError, OSError, SystemError):
        # SystemError: some Vulkan ICDs raise this on initialization failure
        return set(), 0

def _get_vulkan_features() -> dict[str, bool]:
    """Query Vulkan physical device features.
    
    Returns dict with feature names and availability, e.g. {'shaderFloat64': True}.
    Returns empty dict if Vulkan is not available.
    """
    try:
        import vulkan as vk
        
        app_info = vk.VkApplicationInfo(
            pApplicationName='feature_query',
            applicationVersion=1,
            pEngineName='test',  
            engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_2,
        )
        instance_info = vk.VkInstanceCreateInfo(pApplicationInfo=app_info)
        instance = vk.vkCreateInstance(instance_info, None)
        
        devices = vk.vkEnumeratePhysicalDevices(instance)
        if not devices:
            vk.vkDestroyInstance(instance, None)
            return {}
        
        features = vk.vkGetPhysicalDeviceFeatures(devices[0])
        result = {
            'shaderFloat64': bool(features.shaderFloat64),
            'shaderInt16': bool(features.shaderInt16),
        }
        
        vk.vkDestroyInstance(instance, None)
        return result
        
    except (ImportError, RuntimeError, OSError, SystemError):
        # SystemError: some Vulkan ICDs raise this on initialization failure
        return {}

# Cache extension query results (avoid repeated Vulkan instance creation)
_VULKAN_EXTENSIONS: set[str] | None = None
_VULKAN_API_VERSION: int = 0
_VULKAN_FEATURES: dict[str, bool] | None = None

def _vulkan_has_8bit_storage() -> bool:
    """Check if VK_KHR_8bit_storage is available.
    
    This extension is required for FP8 storage buffers (uint8_t in GLSL).
    Promoted to Vulkan 1.2 core, so also available if API version >= 1.2.
    """
    global _VULKAN_EXTENSIONS, _VULKAN_API_VERSION
    if _VULKAN_EXTENSIONS is None:
        _VULKAN_EXTENSIONS, _VULKAN_API_VERSION = _get_vulkan_extensions()
    
    # VK_MAKE_API_VERSION(0, 1, 2, 0) = (0 << 29) | (1 << 22) | (2 << 12) | 0 = 0x00402000
    VK_API_VERSION_1_2 = 0x00402000
    
    # Available via extension OR Vulkan 1.2+ (where it's core)
    return "VK_KHR_8bit_storage" in _VULKAN_EXTENSIONS or _VULKAN_API_VERSION >= VK_API_VERSION_1_2

def _vulkan_has_float16() -> bool:
    """Check if VK_KHR_shader_float16_int8 is available.
    
    Required for FP16 compute (float16_t arithmetic in shaders).
    Promoted to Vulkan 1.2 core.
    """
    global _VULKAN_EXTENSIONS, _VULKAN_API_VERSION
    if _VULKAN_EXTENSIONS is None:
        _VULKAN_EXTENSIONS, _VULKAN_API_VERSION = _get_vulkan_extensions()
    
    VK_API_VERSION_1_2 = 0x00402000
    return "VK_KHR_shader_float16_int8" in _VULKAN_EXTENSIONS or _VULKAN_API_VERSION >= VK_API_VERSION_1_2

def _vulkan_has_float64() -> bool:
    """Check if shaderFloat64 device feature is supported.
    
    Required for FP64 compute (double precision in shaders).
    """
    global _VULKAN_FEATURES
    if _VULKAN_FEATURES is None:
        _VULKAN_FEATURES = _get_vulkan_features()
    return _VULKAN_FEATURES.get('shaderFloat64', False)


# --- Test Class ---

class TestFP8VulkanRoundtrip:
    """ADR-025 §9.2: FP8 roundtrip tests for Vulkan backend."""
    
    @pytest.fixture
    def vulkan_context(self):
        """Create Vulkan context with VK_KHR_8bit_storage for testing.
        
        Yields the context and handles cleanup.
        """
        from backends.vulkan.renderer import VulkanRenderer
        
        # Create renderer with FP8 extension enabled
        try:
            renderer = VulkanRenderer(
                required_extensions=["VK_KHR_8bit_storage"],
                enable_validation=True,
            )
        except RuntimeError as e:
            pytest.skip(f"Vulkan setup failed: {e}")
        
        yield {"renderer": renderer}
        
        # Cleanup
        renderer.destroy()
    
    @pytest.mark.skipif(
        not _vulkan_has_8bit_storage(),
        reason="VK_KHR_8bit_storage not available"
    )
    @pytest.mark.parametrize("precision_factory", [
        PrecisionConfig.fp8_e4m3,      # FP32 compute
        PrecisionConfig.fp8_e4m3_f16,  # FP16 compute
        PrecisionConfig.fp8_e4m3_f64,  # FP64 compute
    ])
    def test_e4m3_roundtrip(self, vulkan_context, precision_factory):
        """Values in E4M3 range survive COMPUTE_TYPE→FP8→COMPUTE_TYPE roundtrip."""
        cfg = precision_factory()
        
        test_values = np.array([0.0, 1.0, 0.5, 100.0, 448.0, -1.0], dtype=np.float32)
        
        # ... create storage buffer, run compute shader, read back ...
        # ... assert roundtrip matches ml_dtypes reference ...
    
    @pytest.mark.skipif(
        not _vulkan_has_float16(),
        reason="VK_KHR_shader_float16_int8 not available"
    )
    def test_fp8_fp16_compute(self, vulkan_context):
        """FP8 storage with FP16 compute works correctly."""
        cfg = PrecisionConfig.fp8_e4m3_f16()
        
        # Verify FP16 compute produces expected results
        test_values = np.array([0.0, 0.5, 1.0, 64.0], dtype=np.float32)
        # ...
    
    def test_e4m3_saturation(self, vulkan_context):
        """Overflow saturates to max (448), not inf/nan."""
        test_values = np.array([500.0, 1000.0, 1e6], dtype=np.float32)
        # ... assert all saturate to 448.0 ...
    
    def test_e5m2_saturation(self, vulkan_context):
        """Overflow saturates to max (57344), not inf/nan."""
        test_values = np.array([60000.0, 100000.0, 1e10], dtype=np.float32)
        # ... assert all saturate to 57344.0 ...
    
    @pytest.mark.parametrize("precision_factory", [
        PrecisionConfig.fp8_e4m3_f64,
        PrecisionConfig.fp8_e5m2_f64,
    ])
    def test_fp64_compute_roundtrip(self, vulkan_context, precision_factory):
        """FP64 compute with FP8 storage works correctly."""
        cfg = precision_factory()
        assert cfg.compute_dtype == np.float64
        
        test_values = np.array([0.0, 1.0, 0.5, 100.0], dtype=np.float64)
        # ... invoke shader with FP64 compute ...
```

---

### Step 9D.6: Validate rollback gate

#### 9D.6.1: Run existing Vulkan test suite

All pre-FP8 Vulkan tests must continue to pass:

```bash
cd architectures/averaging_ensembled_classifier
pytest tests/tier2/vulkan/ -v --ignore=tests/tier2/vulkan/test_fp8_vulkan.py 2>&1 | tee /tmp/vulkan-regression.txt
grep -E "(FAILED|ERROR)" /tmp/vulkan-regression.txt && exit 1 || echo "All existing Vulkan tests pass"
```

#### 9D.6.2: Validate FP8 shader compilation

Verify FP8 shader variants compile without errors on systems with `VK_KHR_8bit_storage`:

```bash
# This test is skipped automatically on systems without the extension
pytest tests/tier2/vulkan/test_fp8_vulkan.py::test_shader_compilation -v
```

#### 9D.6.3: Run FP8 roundtrip tests

```bash
pytest tests/tier2/vulkan/test_fp8_vulkan.py -v 2>&1 | tee /tmp/fp8-vulkan.txt
grep -E "(FAILED|ERROR)" /tmp/fp8-vulkan.txt && exit 1 || echo "FP8 Vulkan tests pass"
```

#### 9D.6.4: Verify extension requirement checking

Verify that FP8 configs produce clear errors on systems lacking required extensions:

```python
# Manual verification (run interactively on a system without VK_KHR_8bit_storage)
from shared.precision_config import PrecisionConfig
cfg = PrecisionConfig.fp8_e4m3()

# This should raise RuntimeError with message:
# "FP8 storage requires VK_KHR_8bit_storage extension..."
from backends.vulkan import VulkanRenderer
renderer = VulkanRenderer(precision=cfg)  # Should fail with clear message
```

---

## 4. Technical Notes

### Arithmetic Conversion vs. Lookup Tables

Unlike the OpenCL and CPU backends which use 256-entry lookup tables for FP8→float conversion, the Vulkan backend uses **arithmetic conversion** via bit manipulation and `exp2()`. This is a deliberate design choice:

| Approach | OpenCL/CPU (LUT) | Vulkan (Arithmetic) |
|:---|:---|:---|
| **Memory** | 1KB per table (2KB total) | No memory footprint |
| **Latency** | Single memory access | ~10-15 ALU ops |
| **GLSL limitation** | N/A | No `__constant` equivalent for kernel-scope arrays¹ |
| **Portability** | Requires constant memory | Pure computation |

¹ GLSL lacks OpenCL's `__constant` memory space. While SPIR-V specialization constants could embed tables, the arithmetic approach is simpler and avoids shader binary bloat.

**Performance note:** FP8 conversion is not on the critical path—the bandwidth savings from 4× smaller storage buffers dominate. The arithmetic overhead is negligible for bandwidth-bound workloads.

### Vulkan Extensions vs. GLSL Pragmas

The Vulkan driver exposes capabilities via named extensions; GLSL shaders enable them via `#extension` pragmas. The naming differs:

| Vulkan Extension | GLSL Pragma | Purpose |
|:---|:---|:---|
| `VK_KHR_8bit_storage` | `GL_EXT_shader_8bit_storage` | `uint8_t` in storage buffers |
| `VK_KHR_shader_float16_int8` | `GL_EXT_shader_explicit_arithmetic_types_float16` | `float16_t` arithmetic |
| `shaderFloat64` device feature¹ | `GL_EXT_shader_explicit_arithmetic_types_float64` | `double` arithmetic |

¹ The `shaderFloat64` feature is a device feature (not an extension), queried via `VkPhysicalDeviceFeatures.shaderFloat64` during device enumeration.

The shader builder checks Vulkan extension availability at device init; the GLSL pragmas are injected via `-D` flags at compile time.

### VK_KHR_8bit_storage: Required extension handling

This extension is **required** for FP8 storage buffers. It allows `uint8_t` in storage buffer declarations. The extension is widely supported on:
- Desktop Vulkan 1.2+ (promoted to core)
- Most mobile GPUs (Adreno 6xx+, Mali-G7x+)

**GPUs lacking VK_KHR_8bit_storage:**
- Intel HD Graphics 5xx/6xx (Skylake/Kaby Lake integrated)—pre-Vulkan-1.2
- AMD GCN 1st/2nd gen (HD 7000, R9 200 series)—Vulkan 1.0 only²
- Mali-G5x and earlier (Bifrost 1st gen)—limited Vulkan 1.0
- Adreno 5xx (Snapdragon 6xx/7xx)—partial Vulkan 1.0 support

² These GPUs have limited or no driver updates; they represent <5% of active Vulkan devices (per vulkan.gpuinfo.org, March 2026).

**Behavior when extension is unavailable:**

| Scenario | Behavior |
|:---|:---|
| Extension missing, FP8 config requested | `RuntimeError` at shader load time with clear message |
| Capability check in tests | `pytest.mark.skipif` skips FP8 tests with reason "VK_KHR_8bit_storage not available" |
| User creates FP8 PrecisionConfig | Config creates successfully (Phase 9A); error deferred to backend initialization |

Without this extension, FP8 buffers would need to pack four FP8 values into a `uint32_t` and unpack in the shader — adding complexity and reducing performance. This fallback is **not implemented**; the extension is treated as mandatory for FP8.

### VK_KHR_shader_float16_int8 for FP16 compute

When using FP16 compute with FP8 storage (`fp8_e4m3_f16()`, `fp8_e5m2_f16()`), **two extensions** are required:

1. **`VK_KHR_8bit_storage`** — for `uint8_t` storage buffers (FP8 data)
2. **`VK_KHR_shader_float16_int8`** — for `float16_t` arithmetic in shaders

Both must be enabled at device creation. The GLSL shader enables the corresponding pragmas:

```glsl
#extension GL_EXT_shader_8bit_storage : require                      // From VK_KHR_8bit_storage
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require // From VK_KHR_shader_float16_int8
```

**Fallback:** If FP16 compute is not available, use the FP32 compute variants (`fp8_e4m3()`, `fp8_e5m2()`). The renderer checks extension availability at init and raises `RuntimeError` with a clear message if FP16 compute configs are used on unsupported hardware.

### No lookup tables in shaders

The arithmetic path is preferred over lookup tables for Vulkan FP8 conversion:

1. **Binding overhead avoided:** A 256-entry lookup table would require an additional SSBO binding (1KB buffer), adding descriptor set complexity. The FP8 storage buffer already uses one binding.

2. **Memory access vs. ALU:** The arithmetic path (~12-15 ALU operations) trades memory latency for compute. On modern GPUs with high ALU throughput and limited memory bandwidth, this is favorable for bandwidth-bound workloads.

3. **Consistency with conversion direction:** FP32→FP8 conversion *must* be arithmetic (no reverse lookup is practical). Using arithmetic for both directions keeps the codepath symmetric.

The OpenCL backend uses lookup tables because OpenCL's `__constant` memory is optimized for broadcast access patterns. GLSL's `const` arrays compile to literal SPIR-V constants that bloat shader size, or require explicit buffer bindings.

### SPIR-V compatibility

The FP8 conversion functions use only standard GLSL operations (`floatBitsToUint`, `exp2`, `abs`, bit operations). No SPIR-V extensions beyond `VK_KHR_8bit_storage` are required. The `exp2()` function is standard GLSL (core since GLSL 1.30) and compiles to portable SPIR-V.

### `exp2()` accuracy for FP8 exponent range

The FP8→float conversion uses `exp2(float(exp8 - FP8_BIAS))` for power-of-2 scaling. This is **exact** (no rounding error) for all FP8 exponent values because:

| Format | Exponent Range | Bias | `exp2()` Argument Range | Result Range |
|:---|:---|:---|:---|:---|
| E4M3 | [0, 14] | 7 | [-7, 7] | [2⁻⁷, 2⁷] = [0.0078125, 128] |
| E5M2 | [0, 30] | 15 | [-15, 15] | [2⁻¹⁵, 2¹⁵] = [~0.000031, 32768] |

All these values are exactly representable in float (powers of 2 within float's exponent range). The `exp2()` result introduces zero error. The only rounding possible is in the mantissa multiplication (`(1 + mant/M) * exp2(...)`) when the product rounds to the nearest float, but since we're reconstructing an FP8 value that was originally quantized from float, this reconstruction is exact—the product always equals a float value that was the source of the FP8 encoding.

### FP64 NaN-before-clamp ordering

The FP64 store path must check `isnan()` **before** the `clamp()` call. GLSL `clamp(NaN, lo, hi)` returns `lo` per spec, so a NaN input would be silently converted to `-FLT_MAX` by the clamp, then stored as a negative saturated value instead of zero. The NaN check is placed at the top of the `COMPUTE_TYPE_IS_DOUBLE` `store_storage_fp8()` overload. The FP16 and FP32 paths route through `store_storage_fp8_internal()`, which checks `isnan()` after the `float(val)` widening/identity cast (which preserves NaN). This is the same pattern and rationale as the OpenCL backend (Phase 9B).

### Subnormal conversion accuracy (store path)

The FP32→FP8 store path uses fixed rounding shift distances that are correct for **both** normalized and subnormal cases. After subnormal denormalization (`mant32 = (mant32 | 0x800000u) >> shift`), the significant bits are right-shifted into positions that align with the fixed extraction window at bits [22:20] (E4M3) or [22:21] (E5M2). The round bit at position 19 (E4M3) or 20 (E5M2) correctly captures the first truncated bit, and sticky bits below capture the remainder. The early underflow check (`val < FP8_MIN_SUBNORMAL * 0.5`) prevents the subnormal path from reaching shift values large enough to push significant bits entirely below the extraction window (E4M3: max reachable shift = 4, MSB at bit 19 = round bit position; E5M2: max reachable shift = 3, MSB at bit 20 = round bit position). The rounding is therefore exact round-to-nearest-even for all reachable subnormal values. The roundtrip tests (Step 9D.5) validate against `ml_dtypes` reference values and should produce exact agreement for all representable inputs.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|:---|:---|:---|:---|
| `VK_KHR_8bit_storage` not available | Medium | Medium | Check extension at renderer init. Skip FP8 tests on unsupported hardware. Provide clear error message. |
| `VK_KHR_shader_float16_int8` not available | Medium | Low | FP16 compute is optional. Fall back to FP32 compute variants. |
| Shader ALU overhead for conversion | Low | Low | FP8's bandwidth savings exceed conversion overhead for bandwidth-bound workloads. Profile if concerns arise. |
| SPIR-V compilation differs across vendors | Low | Medium | Test on NVIDIA, AMD, Intel, mobile GPUs. Use SPIR-V validation layer. |

---

## 6. Files Modified Summary

| File | Change Type | Description |
|:---|:---|:---|
| `src/backends/vulkan/kernel_sources/common.glsl` | **Edit** | Add FP8 compile-time flags, extension enables, `fp8_to_float_internal`, `load_storage_fp8`, `store_storage_fp8` functions |
| `src/backends/vulkan/kernel_sources/*.comp` | **Edit** | Update storage buffer declarations for FP8 (`uint8_t` via `VK_KHR_8bit_storage`) |
| `src/backends/vulkan/shader_builder.py` | **Edit** | Add FP8/FP16/FP64 compute flag emission (`-DSTORAGE_TYPE_IS_FP8`, etc.) |
| `src/backends/vulkan/renderer.py` | **Edit** | Check `VK_KHR_8bit_storage` and `VK_KHR_shader_float16_int8` availability at init |
| `tests/tier2/vulkan/test_fp8_vulkan.py` | **New** | FP8 roundtrip and saturation tests for Vulkan backend |

---

*End of Phase 9D. This phase can be executed in parallel with Phases 9B and 9C. Proceed to Phase 9E after all three backend phases complete.*
