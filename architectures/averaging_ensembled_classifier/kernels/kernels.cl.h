// kernels.cl.h
//
// --- ADR-013 Designation ---
// This file is the algorithmic specification document for the
// averaging_ensembled_classifier architecture.  Its @kernel_contract
// blocks and @param annotations constitute the authoritative,
// language-neutral reference that all backend implementations
// (OpenCL, Vulkan SPIR-V, CPU SIMD) must implement against.
// See: adr/ADR-013-kernel-source-strategy.md

#ifndef KERNELS_CL_H
#define KERNELS_CL_H

// --- Architectural Contract Note ---
// This header constitutes the sole and sufficient technical contract
// for interaction between host and device implementations:
//
// 1. **Device Specification:**
//    Kernels are defined as stateless computational units. Their behavior,
//    memory layouts, and interface constraints are fully specified here.
//    Device implementations require no external context beyond this document.
//
// 2. **Host Interface:**
//    Kernel invocation parameters, buffer semantics, and synchronization
//    requirements are exhaustively defined. Host code requires no knowledge
//    of device internals or optimization strategies beyond these specifications.
//
// Explicitly out of scope:
// - Host orchestration logic (e.g., task graphs, reduction strategies)
// - Device hardware optimizations (e.g., register allocation, vectorization)
//
// Adherence to this contract ensures strict separation of concerns and
// bidirectional implementation independence.

// =====================================================================
// OpenCL Environment Detection
// =====================================================================

#ifdef __OPENCL_VERSION__

#if __OPENCL_VERSION__ < 120
#error "OpenCL 1.2 or newer is required. Please use a compatible device/driver."
#endif

// =====================================================================
// 1. Mandatory Build-Time Symbol Validation
//    (CONTRACT.md Article 6, amended by ADR-020 §3.6, ADR-025 §4)
//
//    All symbols must be provided by the host build system via -D flags.
//    Missing symbols produce immediate compilation failure.
// =====================================================================

// --- Precision-Role Type Symbols ---
#ifndef STORAGE_TYPE
#error "System Contract Violation: STORAGE_TYPE must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE
#error "System Contract Violation: COMPUTE_TYPE must be defined by the host build system."
#endif
#ifndef STATE_TYPE
#error "System Contract Violation: STATE_TYPE must be defined by the host build system."
#endif

// --- Storage-Role Type Flags ---
#if !defined(STORAGE_TYPE_IS_FP8)
#error "System Contract Violation: STORAGE_TYPE_IS_FP8 must be defined by the build system."
#endif
#if !defined(STORAGE_TYPE_IS_E4M3)
#error "System Contract Violation: STORAGE_TYPE_IS_E4M3 must be defined by the build system."
#endif
#if !defined(STORAGE_TYPE_IS_E5M2)
#error "System Contract Violation: STORAGE_TYPE_IS_E5M2 must be defined by the build system."
#endif
#ifndef STORAGE_TYPE_IS_HALF
#error "System Contract Violation: STORAGE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef STORAGE_TYPE_IS_FLOAT
#error "System Contract Violation: STORAGE_TYPE_IS_FLOAT must be defined by the host build system."
#endif
#ifndef STORAGE_TYPE_IS_DOUBLE
#error "System Contract Violation: STORAGE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif

// --- Compute-Role Type Flags ---
#ifndef COMPUTE_TYPE_IS_HALF
#error "System Contract Violation: COMPUTE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE_IS_FLOAT
#error "System Contract Violation: COMPUTE_TYPE_IS_FLOAT must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE_IS_DOUBLE
#error "System Contract Violation: COMPUTE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif

// --- State-Role Type Flags ---
#ifndef STATE_TYPE_IS_HALF
#error "System Contract Violation: STATE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef STATE_TYPE_IS_FLOAT
#error "System Contract Violation: STATE_TYPE_IS_FLOAT must be defined by the host build system."
#endif
#ifndef STATE_TYPE_IS_DOUBLE
#error "System Contract Violation: STATE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif

// --- Hardware Profile & Tuning Symbols ---
#ifndef SIMD_WIDTH
#error "System Contract Violation: SIMD_WIDTH must be defined by the host build system."
#endif
#ifndef C_TILE_SIZE
#error "System Contract Violation: C_TILE_SIZE must be defined by the host build system."
#endif
#ifndef NUMERICAL_STABILITY_EPSILON
#error "System Contract Violation: NUMERICAL_STABILITY_EPSILON must be defined by the host build system."
#endif

// =====================================================================
// 2. Compile-Time Invariant Enforcement (CONTRACT.md Article 6.4)
//
//    These guards provide defense-in-depth against build-system
//    misconfigurations, complementing PrecisionConfig.__post_init__
//    runtime validation.
// =====================================================================

// --- FP8 Format Mutual Exclusivity (CONTRACT.md Article 6.3) ---
#if STORAGE_TYPE_IS_E4M3 && STORAGE_TYPE_IS_E5M2
#error "System Contract Violation: E4M3 and E5M2 are mutually exclusive."
#endif
#if STORAGE_TYPE_IS_FP8 && !STORAGE_TYPE_IS_E4M3 && !STORAGE_TYPE_IS_E5M2
#error "System Contract Violation: STORAGE_TYPE_IS_FP8 requires exactly one of E4M3 or E5M2."
#endif
#if !STORAGE_TYPE_IS_FP8 && (STORAGE_TYPE_IS_E4M3 || STORAGE_TYPE_IS_E5M2)
#error "System Contract Violation: E4M3/E5M2 flags require STORAGE_TYPE_IS_FP8=1."
#endif

// --- Precision Role Exactly-One Selection ---
#if (STORAGE_TYPE_IS_FP8 + STORAGE_TYPE_IS_HALF + STORAGE_TYPE_IS_FLOAT + STORAGE_TYPE_IS_DOUBLE) != 1
#error "System Contract Violation: Exactly one STORAGE_TYPE_IS_* flag must be set."
#endif
#if (COMPUTE_TYPE_IS_HALF + COMPUTE_TYPE_IS_FLOAT + COMPUTE_TYPE_IS_DOUBLE) != 1
#error "System Contract Violation: Exactly one COMPUTE_TYPE_IS_* flag must be set."
#endif
#if (STATE_TYPE_IS_HALF + STATE_TYPE_IS_FLOAT + STATE_TYPE_IS_DOUBLE) != 1
#error "System Contract Violation: Exactly one STATE_TYPE_IS_* flag must be set."
#endif

// --- PrecisionConfig Invariant: storage.itemsize <= compute.itemsize ---
#if STORAGE_TYPE_IS_DOUBLE && !COMPUTE_TYPE_IS_DOUBLE
#error "System Contract Violation: storage=double requires compute=double (storage <= compute)"
#endif
#if STORAGE_TYPE_IS_FLOAT && COMPUTE_TYPE_IS_HALF
#error "System Contract Violation: storage=float requires compute >= float (storage <= compute)"
#endif

// --- PrecisionConfig Invariant: storage.itemsize <= state.itemsize ---
#if STORAGE_TYPE_IS_DOUBLE && !STATE_TYPE_IS_DOUBLE
#error "System Contract Violation: storage=double requires state=double (storage <= state)"
#endif
#if STORAGE_TYPE_IS_FLOAT && STATE_TYPE_IS_HALF
#error "System Contract Violation: storage=float requires state >= float (storage <= state)"
#endif

// =====================================================================
// 3. Extension Enablement
// =====================================================================

// Enable FP16 extension if any precision role uses half.
#if STORAGE_TYPE_IS_HALF
#if !defined(cl_khr_fp16)
#error "FP16 extension (cl_khr_fp16) required for half precision storage but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#if COMPUTE_TYPE_IS_HALF && !STORAGE_TYPE_IS_HALF
#if !defined(cl_khr_fp16)
#error "FP16 extension (cl_khr_fp16) required for half precision compute but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#if STATE_TYPE_IS_HALF && !STORAGE_TYPE_IS_HALF && !COMPUTE_TYPE_IS_HALF
#if !defined(cl_khr_fp16)
#error "FP16 extension (cl_khr_fp16) required for half precision state but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Enable FP64 extension if any precision role uses double.
#if STORAGE_TYPE_IS_DOUBLE || COMPUTE_TYPE_IS_DOUBLE || STATE_TYPE_IS_DOUBLE
#if !defined(cl_khr_fp64)
#error "FP64 extension (cl_khr_fp64) required for double precision but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// =====================================================================
// 4. Compile-Time Constants
// =====================================================================

// --- Compute-Role Literal Constants ---
#if COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_ZERO 0.0
#define COMPUTE_ONE  1.0
#elif COMPUTE_TYPE_IS_HALF
#define COMPUTE_ZERO ((COMPUTE_TYPE)0.0h)
#define COMPUTE_ONE  ((COMPUTE_TYPE)1.0h)
#else
#define COMPUTE_ZERO ((COMPUTE_TYPE)0.0f)
#define COMPUTE_ONE  ((COMPUTE_TYPE)1.0f)
#endif

// --- Compute-Precision Extremal Constant ---
// Maximum finite value representable in COMPUTE_TYPE. Used for
// reduction sentinels (e.g., softmax max-finding initialization)
// and numerical stability guards. Derived from the same
// COMPUTE_TYPE_IS_* flags that govern COMPUTE_ZERO/COMPUTE_ONE.
//
// Corresponds to PrecisionConfig.compute_fp_format_max on the host.
// The two are guaranteed to agree because they derive from the same
// underlying type.
#if COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_FP_MAX DBL_MAX
#elif COMPUTE_TYPE_IS_HALF
#define COMPUTE_FP_MAX HALF_MAX
#else
#define COMPUTE_FP_MAX FLT_MAX
#endif

// --- Kernel Work-Group Attribute ---
// KERNEL_ATTR provides a work-group size hint for backends that choose to
// apply it. This header specifies interfaces only; actual application of
// the attribute is a backend rendering-tier concern (OpenCL may apply it,
// Vulkan uses specialization constants, CPU backend has no work-groups).
#define KERNEL_ATTR __attribute__((work_group_size_hint(SIMD_WIDTH, 1, 1)))

// =====================================================================
// 5. Precision Boundary Abstractions
//    (ADR-020 §4.4, ADR-023 §1, ADR-025 §6, ADR-027)
//
//    These functions are the sole mechanism for crossing precision role
//    boundaries. When STORAGE_TYPE == COMPUTE_TYPE, all conversions
//    compile to identity casts that any OpenCL compiler eliminates.
//    No #ifdef on type equality is used anywhere in the kernel sources.
// =====================================================================

// --- 5a. FP8 Software Codec (ADR-025 §4, §6) ---
//
// Software emulation path for FP8 storage. Tables store FP32 values;
// FP16 compute narrows via (half) cast, FP64 widens.
// Generated by: python scripts/generate_fp8_lut.py --backend opencl

#if STORAGE_TYPE_IS_FP8
#include "fp8_lut.gen.h"

// FP8 Load (storage → compute)
// Returns COMPUTE_TYPE (half, float, or double).
// LUT stores float values. FP8→float is lossless (all 256 FP8 values
// are exactly representable in float). FP16 compute narrows; FP64 widens.
static inline COMPUTE_TYPE load_storage_fp8(__global const uchar *buf, size_t idx) {
#if STORAGE_TYPE_IS_E4M3
    float f32_val = fp8_e4m3_to_float_lut[buf[idx]];
#elif STORAGE_TYPE_IS_E5M2
    float f32_val = fp8_e5m2_to_float_lut[buf[idx]];
#endif

#if COMPUTE_TYPE_IS_HALF
    return (half)f32_val;
#elif COMPUTE_TYPE_IS_DOUBLE
    return (double)f32_val;
#else
    return f32_val;
#endif
}

// FP8 Store (compute → storage)
// Accepts COMPUTE_TYPE, converts to FP8.
//
// NaN handling: NaN input maps to zero for both E4M3 and E5M2.
// E4M3 (float8_e4m3fn) has no NaN representation. E5M2 has IEEE-like
// inf/NaN but the store path saturates to max finite instead.
// NaN → zero is the safe failure mode (no parameter update) matching
// ml_dtypes.float8_e4m3fn behavior and the CPU backend (cpu_fp8.h).
static inline void store_storage_fp8(__global uchar *buf, size_t idx, COMPUTE_TYPE val) {
#if COMPUTE_TYPE_IS_DOUBLE
    // NaN check BEFORE clamp: fmin/fmax(NaN, x) == x per IEEE 754
    // minNum/maxNum, so NaN would be silently converted to ±FLT_MAX
    // by the clamp below.
    double dval = (double)val;
    if (isnan(dval)) {
        buf[idx] = 0x00;
        return;
    }
    // Defense-in-depth: clamp to FLT_MAX range before double→float narrowing
    // to prevent infinity from reaching the FP8 encoder.
    dval = fmin(fmax(dval, -(double)FLT_MAX), (double)FLT_MAX);
    float fval = (float)dval;
#else
    // Narrow to float for bit manipulation (FP8 fits in float exactly)
    float fval = (float)val;
    if (isnan(fval)) {
        buf[idx] = 0x00;
        return;
    }
#endif

    uint sign = (fval < 0.0f) ? 1 : 0;
    fval = fabs(fval);

#if STORAGE_TYPE_IS_E4M3
    // E4M3: bias=7, max=448, min_subnormal=2^-9
    const float MAX_VAL = 448.0f;
    const float MIN_SUBNORMAL = 0.001953125f;  // 2^-9

    if (fval >= MAX_VAL) {
        buf[idx] = sign ? 0xFE : 0x7E;  // Max magnitude (saturation, no inf)
        return;
    }
    if (fval < MIN_SUBNORMAL * 0.5f) {
        buf[idx] = sign ? 0x80 : 0x00;  // Zero
        return;
    }

    // Extract FP32 exponent and mantissa via bit cast
    uint fbits = as_uint(fval);
    int exp32 = ((fbits >> 23) & 0xFF) - 127;
    uint mant32 = fbits & 0x7FFFFF;

    // Compute FP8 exponent
    int exp8 = exp32 + 7;  // E4M3 bias = 7

    // Handle subnormals
    // NOTE: Subnormal rounding omits shifted-out bits from sticky calculation.
    // Max error: 1 ULP of FP8 subnormal (2^-9). Below quantization floor.
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            buf[idx] = sign ? 0x80 : 0x00;
            return;
        }
        mant32 = (mant32 | 0x800000) >> shift;
        exp8 = 0;
    } else if (exp8 > 15) {
        // Defense-in-depth: unreachable for well-formed floats.
        buf[idx] = sign ? 0xFE : 0x7E;
        return;
    }

    // Round mantissa to 3 bits (round-to-nearest-even)
    uint round_bit = (mant32 >> 19) & 1u;
    uint sticky = mant32 & ((1u << 19) - 1u);
    uint mant8 = mant32 >> 20;
    if (round_bit && (sticky || (mant8 & 1u))) {
        mant8++;
    }
    if (mant8 >= 8) {
        mant8 = 0;
        exp8++;
        if (exp8 > 15) {
            buf[idx] = sign ? 0xFE : 0x7E;
            return;
        }
    }
    // E4M3fn: exp=15, mant=7 is NaN — clamp to max finite (exp=15, mant=6)
    if (exp8 == 15 && mant8 >= 7) {
        buf[idx] = sign ? 0xFE : 0x7E;
        return;
    }

    buf[idx] = (sign << 7) | (exp8 << 3) | (mant8 & 0x7);

#elif STORAGE_TYPE_IS_E5M2
    // E5M2: bias=15, max=57344, min_subnormal=2^-16
    const float MAX_VAL = 57344.0f;
    const float MIN_SUBNORMAL = 0.0000152587890625f;  // 2^-16, exact

    if (fval >= MAX_VAL) {
        buf[idx] = sign ? 0xFB : 0x7B;  // Max magnitude
        return;
    }
    if (fval < MIN_SUBNORMAL * 0.5f) {
        buf[idx] = sign ? 0x80 : 0x00;  // Zero
        return;
    }

    // Extract FP32 exponent and mantissa via bit cast
    uint fbits = as_uint(fval);
    int exp32 = ((fbits >> 23) & 0xFF) - 127;
    uint mant32 = fbits & 0x7FFFFF;

    // Compute FP8 exponent
    int exp8 = exp32 + 15;  // E5M2 bias = 15

    // Handle subnormals
    // NOTE: Subnormal rounding omits shifted-out bits from sticky calculation.
    // Max error: 1 ULP of FP8 E5M2 subnormal (2^-16). Below quantization floor.
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            buf[idx] = sign ? 0x80 : 0x00;
            return;
        }
        mant32 = (mant32 | 0x800000) >> shift;
        exp8 = 0;
    } else if (exp8 >= 31) {
        buf[idx] = sign ? 0xFB : 0x7B;
        return;
    }

    // Round mantissa to 2 bits (round-to-nearest-even)
    uint round_bit = (mant32 >> 20) & 1u;
    uint sticky = mant32 & ((1u << 20) - 1u);
    uint mant8 = mant32 >> 21;
    if (round_bit && (sticky || (mant8 & 1u))) {
        mant8++;
    }
    if (mant8 >= 4) {
        mant8 = 0;
        exp8++;
        if (exp8 >= 31) {
            buf[idx] = sign ? 0xFB : 0x7B;
            return;
        }
    }

    buf[idx] = (sign << 7) | (exp8 << 2) | (mant8 & 0x3);

#endif
}

#endif // STORAGE_TYPE_IS_FP8

// --- 5b. Storage-Role Load/Store ---

static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx)
{
#if STORAGE_TYPE_IS_FP8
    return load_storage_fp8((__global const uchar *)buf, idx);
#elif STORAGE_TYPE_IS_HALF
    // Explicit cast for self-documentation; STORAGE_TYPE is half here.
    return (COMPUTE_TYPE)vload_half(idx, (__global const half *)buf);
#else
    return (COMPUTE_TYPE)buf[idx];
#endif
}

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
#if STORAGE_TYPE_IS_FP8
    store_storage_fp8((__global uchar *)buf, idx, val);
#elif STORAGE_TYPE_IS_HALF
    // vstore_half accepts float/double and performs round-to-nearest-even
    // narrowing internally; no explicit (half) pre-cast is needed.
    // Explicit cast for self-documentation; STORAGE_TYPE is half here.
    vstore_half(val, idx, (__global half *)buf);
#else
    buf[idx] = (STORAGE_TYPE)val;
#endif
}

// --- 5c. State-Role Load/Store ---
//
// FP64 Precision Boundary Notes (ADR-024 §3.2):
// When STATE_TYPE = double and COMPUTE_TYPE = float:
//   load_state: double → float narrowing (precision loss accepted;
//               the value is about to enter lower-precision arithmetic)
//   store_state_update: float → double widening (no precision loss;
//                       the narrower compute value preserves all its bits)
// When STATE_TYPE = double and COMPUTE_TYPE = double:
//   Both casts are identity operations, eliminated by the compiler.
//
// STATE_TYPE = half Note (ADR-024):
//   When STATE_TYPE = half, cl_khr_fp16 must be active. This is guaranteed
//   because the extension is enabled when any role type is half (storage,
//   compute, or state). Plain array access on half* is valid when the
//   extension is active, so no vload_half/vstore_half path is required here.
//
// Intentional Asymmetry Note:
//   load_storage() uses vload_half() for half-precision storage, while
//   load_state() uses plain array access. This asymmetry is intentional:
//   vload_half() was historically required because some OpenCL 1.x
//   implementations supported half storage but not half arithmetic —
//   vload_half() returns float without requiring full cl_khr_fp16 support.
//   With cl_khr_fp16 active (required when any role type is half), plain
//   access works identically for state buffers. Both approaches are correct.

static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx)
{
    return (COMPUTE_TYPE)buf[idx];
}

static inline void store_state(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}

// Semantically distinct from store_state(): marks an in-place optimizer
// state mutation. Currently identical; exists for future extensibility.
static inline void store_state_update(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}

// --- 5d. Sample Mask Bitmask Access (ADR-031) ---
// Packed uint bitmask with 32 samples per word, LSB-first.
// Compiles to 3 integer ALU operations.
static inline uint load_sample_mask(
    __global const uint *mask_words, uint sample_index)
{
    return (mask_words[sample_index >> 5u] >> (sample_index & 31u)) & 1u;
}

// --- 5e. Accumulation Precision Type System (ADR-027) ---
//
// Stateful-update kernels perform EMA accumulation in ACCUM_TYPE, defined
// as max(COMPUTE_TYPE, STATE_TYPE). This preserves full state precision
// when STATE_TYPE > COMPUTE_TYPE, preventing erosion over unbounded
// training steps.
//
// When STATE_TYPE <= COMPUTE_TYPE, ACCUM_TYPE == COMPUTE_TYPE and all
// casts are identity operations eliminated by the compiler.
//
// ACCUM_TYPE coverage matrix (S=STATE, C=COMPUTE):
//   S=double, C=double → else branch:  ACCUM=double (COMPUTE) ✓
//   S=double, C=float  → branch 1:     ACCUM=double (STATE)   ✓
//   S=double, C=half   → branch 1:     ACCUM=double (STATE)   ✓
//   S=float,  C=double → else branch:  ACCUM=double (COMPUTE) ✓
//   S=float,  C=float  → else branch:  ACCUM=float  (COMPUTE) ✓
//   S=float,  C=half   → branch 2:     ACCUM=float  (STATE)   ✓
//   S=half,   C=double → else branch:  ACCUM=double (COMPUTE) ✓
//   S=half,   C=float  → else branch:  ACCUM=float  (COMPUTE) ✓
//   S=half,   C=half   → else branch:  ACCUM=half   (COMPUTE) ✓

#if STATE_TYPE_IS_DOUBLE && !COMPUTE_TYPE_IS_DOUBLE
    // STATE_TYPE (double) > COMPUTE_TYPE (float or half)
    typedef double ACCUM_TYPE;
    #define ACCUM_IS_WIDER_THAN_COMPUTE 1
    #define ACCUM_ONE  1.0
    #define ACCUM_ZERO 0.0
#elif !STATE_TYPE_IS_HALF && !STATE_TYPE_IS_DOUBLE && COMPUTE_TYPE_IS_HALF
    // STATE_TYPE (float) > COMPUTE_TYPE (half) — float by exclusion
    typedef STATE_TYPE ACCUM_TYPE;
    #define ACCUM_IS_WIDER_THAN_COMPUTE 1
    #define ACCUM_ONE  ((ACCUM_TYPE)1.0f)
    #define ACCUM_ZERO ((ACCUM_TYPE)0.0f)
#else
    // STATE_TYPE <= COMPUTE_TYPE (standard case)
    typedef COMPUTE_TYPE ACCUM_TYPE;
    #define ACCUM_IS_WIDER_THAN_COMPUTE 0
    #define ACCUM_ONE  COMPUTE_ONE
    #define ACCUM_ZERO COMPUTE_ZERO
#endif

// --- 5f. Accumulation-Precision Abstractions ---

// Load state value at accumulation precision (preserves full STATE_TYPE bits)
static inline ACCUM_TYPE load_state_for_accum(
    __global const STATE_TYPE *buf, size_t idx)
{
#if ACCUM_IS_WIDER_THAN_COMPUTE
    return buf[idx];  // No narrowing — direct STATE_TYPE read as ACCUM_TYPE
#else
    return (ACCUM_TYPE)buf[idx];  // Widening or identity
#endif
}

// Store accumulation result to state buffer.
// Supersedes store_state_update() for accumulative operations; the existing
// store_state_update() (ADR-024 §3.2) remains valid for transformative
// in-place state mutations (e.g., clamp_temperatures).
static inline void store_state_from_accum(
    __global STATE_TYPE *buf, size_t idx, ACCUM_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}

// Widen compute-role value to accumulation precision (for EMA inputs)
static inline ACCUM_TYPE widen_to_accum(COMPUTE_TYPE val)
{
    return (ACCUM_TYPE)val;
}

// Narrow accumulation result to compute precision (for bias-corrected values)
static inline COMPUTE_TYPE narrow_from_accum(ACCUM_TYPE val)
{
    return (COMPUTE_TYPE)val;
}

// =====================================================================
// End of OpenCL-Specific Infrastructure
// =====================================================================

#else // !__OPENCL_VERSION__

// =====================================================================
// Host / C++ Mode Stub Definitions
//
// Provides minimal type shims and function stubs so that header
// parsing, IDE tooling, and host-mode unit tests can process this
// file without an OpenCL compiler.
// =====================================================================

#include <float.h>
#include <math.h>
#include <stdio.h>

#ifndef __kernel
#define __kernel
#endif
#ifndef __local
#define __local
#endif
#ifndef __global
#define __global
#endif
#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#endif
#ifndef uint
#define uint unsigned int
#endif
#define KERNEL_ATTR

// --- Precision-Role Type Defaults ---
#ifndef STORAGE_TYPE
#define STORAGE_TYPE float
#endif
#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE float
#endif
#ifndef STATE_TYPE
#define STATE_TYPE float
#endif

// --- Storage-Role Type Flags ---
#ifndef STORAGE_TYPE_IS_HALF
#define STORAGE_TYPE_IS_HALF 0
#endif
#ifndef STORAGE_TYPE_IS_FLOAT
#define STORAGE_TYPE_IS_FLOAT 1
#endif
#ifndef STORAGE_TYPE_IS_DOUBLE
#define STORAGE_TYPE_IS_DOUBLE 0
#endif
#ifndef STORAGE_TYPE_IS_FP8
#define STORAGE_TYPE_IS_FP8 0
#endif
#ifndef STORAGE_TYPE_IS_E4M3
#define STORAGE_TYPE_IS_E4M3 0
#endif
#ifndef STORAGE_TYPE_IS_E5M2
#define STORAGE_TYPE_IS_E5M2 0
#endif

// Host-mode FP8 guard: FP8 storage requires the full OpenCL backend's
// LUT-based decode and algorithmic encode paths. Host-mode stubs do not
// support FP8.
#if STORAGE_TYPE_IS_FP8
#error "FP8 storage is not supported in host-mode stubs. Use the CPU backend."
#endif

// --- Compute-Role Type Flags ---
#ifndef COMPUTE_TYPE_IS_HALF
#define COMPUTE_TYPE_IS_HALF 0
#endif
#ifndef COMPUTE_TYPE_IS_FLOAT
#define COMPUTE_TYPE_IS_FLOAT 1
#endif
#ifndef COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_TYPE_IS_DOUBLE 0
#endif

// --- State-Role Type Flags ---
#ifndef STATE_TYPE_IS_HALF
#define STATE_TYPE_IS_HALF 0
#endif
#ifndef STATE_TYPE_IS_FLOAT
#define STATE_TYPE_IS_FLOAT 1
#endif
#ifndef STATE_TYPE_IS_DOUBLE
#define STATE_TYPE_IS_DOUBLE 0
#endif

// --- Compute-Role Constants ---
#ifndef COMPUTE_ZERO
#define COMPUTE_ZERO 0.0f
#endif
#ifndef COMPUTE_ONE
#define COMPUTE_ONE 1.0f
#endif
#ifndef COMPUTE_FP_MAX
#define COMPUTE_FP_MAX FLT_MAX
#endif

// --- Hardware Profile Defaults ---
#ifndef SIMD_WIDTH
#define SIMD_WIDTH 1
#endif
#ifndef C_TILE_SIZE
#define C_TILE_SIZE 1
#endif
#ifndef NUMERICAL_STABILITY_EPSILON
#define NUMERICAL_STABILITY_EPSILON 1
#endif

// --- OpenCL Built-in Stubs ---
#define CLK_LOCAL_MEM_FENCE 0x01
#define CLK_GLOBAL_MEM_FENCE 0x02
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

inline int          get_global_id(int dim) { return 0; }
inline int          get_local_id(int dim) { return 0; }
inline int          get_group_id(int dim) { return 0; }
inline int          get_local_size(int dim) { return 1; }
inline int          get_global_size(int dim) { return 1; }
inline int          get_num_groups(int dim) { return 1; }
inline void         barrier(int flags) { (void)flags; }
inline COMPUTE_TYPE clamp(COMPUTE_TYPE val, COMPUTE_TYPE min_val, COMPUTE_TYPE max_val) { return fmin(fmax(val, min_val), max_val); }
inline COMPUTE_TYPE select(COMPUTE_TYPE a, COMPUTE_TYPE b, int c) { return (c) ? b : a; }
// OpenCL pown: use powf to avoid conflict with C23 pown declaration
#define pown(base, exp) ((COMPUTE_TYPE)powf((float)(base), (float)(exp)))

// --- Precision Boundary Stubs (identity operations) ---
inline COMPUTE_TYPE load_storage(const STORAGE_TYPE *buf, size_t idx) { return (COMPUTE_TYPE)buf[idx]; }
inline void store_storage(STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val) { buf[idx] = (STORAGE_TYPE)val; }
inline COMPUTE_TYPE load_state(const STATE_TYPE *buf, size_t idx) { return (COMPUTE_TYPE)buf[idx]; }
inline void store_state(STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) { buf[idx] = (STATE_TYPE)val; }
inline void store_state_update(STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) { buf[idx] = (STATE_TYPE)val; }

// --- Sample Mask Stub (ADR-031) ---
inline unsigned int load_sample_mask(const unsigned int *mask_words, unsigned int sample_index) {
    return (mask_words[sample_index >> 5u] >> (sample_index & 31u)) & 1u;
}

// --- Accumulation-Precision Stubs (ADR-027) ---
// In host mode, STATE_TYPE == COMPUTE_TYPE == float, so ACCUM_TYPE = COMPUTE_TYPE.
#ifndef ACCUM_TYPE
#define ACCUM_TYPE COMPUTE_TYPE
#endif
#ifndef ACCUM_IS_WIDER_THAN_COMPUTE
#define ACCUM_IS_WIDER_THAN_COMPUTE 0
#endif
#ifndef ACCUM_ONE
#define ACCUM_ONE 1.0f
#endif
#ifndef ACCUM_ZERO
#define ACCUM_ZERO 0.0f
#endif

inline ACCUM_TYPE   load_state_for_accum(const STATE_TYPE *buf, size_t idx) { return (ACCUM_TYPE)buf[idx]; }
inline void         store_state_from_accum(STATE_TYPE *buf, size_t idx, ACCUM_TYPE val) { buf[idx] = (STATE_TYPE)val; }
inline ACCUM_TYPE   widen_to_accum(COMPUTE_TYPE val) { return (ACCUM_TYPE)val; }
inline COMPUTE_TYPE narrow_from_accum(ACCUM_TYPE val) { return (COMPUTE_TYPE)val; }

#endif // __OPENCL_VERSION__

// =====================================================================
// Common Constants, Flags & Math Macros
//
// Shared between OpenCL and host-mode compilation.
// =====================================================================

// --- Kernel-Internal Constants ---
// Bank-conflict avoidance: +1 element per row stride in local memory.
// This is a well-known constant of the bank-conflict avoidance technique,
// not a hardware-dependent parameter. It is kernel-internal and is NOT a
// build-time symbol — the build system does not provide it via -D flags.
// Retained for any kernel source that uses the stride-padding pattern in
// local memory (e.g., tiled matrix transposes).
#ifndef LOCAL_MEM_BANK_PADDING
#define LOCAL_MEM_BANK_PADDING 1
#endif

// --- Host-Configurable Flags and Enums ---
#define PROBLEM_TYPE_CCE 0 // Selects Softmax/Cross-Entropy Loss math path
#define PROBLEM_TYPE_BCE 1 // Selects Sigmoid/Binary Cross-Entropy math path
#define AGG_MODE_SUM 0     // Selects summation for aggregation
#define AGG_MODE_AVERAGE 1 // Selects averaging for aggregation

// --- Precision-Gated Math Function Macros ---
// native_* intrinsics exist only for float in OpenCL.
// When COMPUTE_TYPE is half or double, fast-math is not applicable.
#ifndef USE_FAST_MATH
#define USE_FAST_MATH 0
#endif
#ifdef __OPENCL_VERSION__
#if USE_FAST_MATH && COMPUTE_TYPE_IS_FLOAT
#define MATH_EXP  native_exp
#define MATH_LOG  native_log
#define MATH_SQRT native_sqrt
#else
#define MATH_EXP  exp
#define MATH_LOG  log
#define MATH_SQRT sqrt
#endif
#else
// Host-mode stubs (fast math is never applicable):
#define MATH_EXP  exp
#define MATH_LOG  log
#define MATH_SQRT sqrt
#endif

// #####################################################################
// #####################################################################
// ##                                                                 ##
// ##                     KERNEL DECLARATIONS                         ##
// ##                                                                 ##
// #####################################################################
// #####################################################################

// =====================================================================
// Act Phase Kernels (Nodes 4–7)
//
// Forward pass computation: shared-layer activations, module-layer
// logit rendering, probability & loss calculation (CCE/BCE).
// =====================================================================

// --- Node 4: Shared Layer Forward Pass --------------------------------

/**
 * @brief (Node 4) Computes hidden activations for a slice of the input batch.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable"
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs loaded via load_storage(); state-role inputs loaded via load_state(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. Sample Masking (ADR-031): When load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; the kernel writes zero to all hidden activation outputs for that sample (and to the corresponding hidden mask positions when out_scalar_FLAG_produce_hidden_mask == 1) and skips further computation. This is a correctness requirement — not merely a performance optimization. A masked sample's input data may contain arbitrary values; without the mask check, the affine transform W × x + b produces non-zero activations whose downstream consequences include: (a) non-zero ReLU mask values propagating to Nodes 17 and 18, and (b) non-zero hidden activation values propagating to Node 8's weight gradient path. Downstream consumers cite this invariant explicitly: Nodes 17 and 18 rely on 'relu_derivative = 0 from Node 4's activation zeroing' as one of two independent guarantees that masked samples contribute zero to shared-layer gradients; Node 8's weight gradient path relies on hidden_activations = 0 to annihilate the non-zero loss derivative at masked samples. Hidden mask production is controlled by `out_scalar_FLAG_produce_hidden_mask`. When 1: the ReLU derivative mask capturing compute-precision truth is written to `dest_buffer_GLOBAL_hidden_mask`. When 0: mask writes are skipped; the Host MAY pass a minimal stub buffer. This FLAG enables policy-tier control over mask lifecycle based on the precision configuration. Padding Zero Propagation (Emergent): Hidden-dimension padding positions (indices >= padded_hidden_count's logical extent) in dest_buffer_GLOBAL_hidden_activations and dest_buffer_GLOBAL_hidden_mask carry zero when all three conditions hold: (a) input-dimension padding in src_buffer_GLOBAL_input is zero, (b) hidden-dimension padding in weights is zero, (c) hidden-dimension padding in biases is zero. This is a mathematical consequence of the affine transform (W*x + b) and ReLU(0) = 0, not an active zeroing step. The kernel is not required to distinguish padding from logical positions — it receives only padded dimensions (padded_input_count, padded_hidden_count). Implementations MAY skip computation at padding indices as a performance optimization; both approaches satisfy Padding Zero-Propagation. The host initialization and optimizer (Node 24) are the co-guarantors of the three preconditions."
 */
__kernel void forward_pass(
    /**
     * @param update_buffer_LOCAL_simd_tile Local memory for tiled matrix-vector multiplication.
     *        - Allocation Formula: (SIMD_WIDTH + SIMD_WIDTH * SIMD_WIDTH) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Partitioned into two contiguous sub-arrays:
     *          tile_input[SIMD_WIDTH] (broadcast tile for the input vector slice)
     *          at offset 0, followed by tile_weights[SIMD_WIDTH][SIMD_WIDTH]
     *          (weight matrix tile, row-major, stride = SIMD_WIDTH) at offset
     *          SIMD_WIDTH. Access pattern is stride-1 across threads for both
     *          sub-arrays; no bank-conflict avoidance padding is required."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_simd_tile,

    /**
     * @param src_buffer_GLOBAL_input The primary data source for the computational unit.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Pad row stride to 128-byte alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: [1] The access slice defined by chunk parameters must be within the buffer's bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset +
     * src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_input_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_shared_simd_major The learnable shared weights in a SIMD-friendly layout.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count/SIMD_WIDTH, src_scalar_NATURAL_padded_input_count, SIMD_WIDTH)
     *        - Padding Contract: {
     *            dim[0] ("hidden_count/SIMD_WIDTH" → "padded_hidden_count/SIMD_WIDTH"): {Type: SIMD, Formula: "SIMD_WIDTH-multiple alignment on hidden_count ensures exact division"},
     *            dim[1] ("input_count" → "padded_input_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[2] ("SIMD_WIDTH"): {Type: UNPADDED}
     *          }
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_input_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_weights_shared_simd_major,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_shared The learnable shared biases.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: SIMD, Formula: "Padded to SIMD_WIDTH"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_padded_hidden_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_biases_shared,

    /**
     * @param dest_buffer_GLOBAL_hidden_activations The output tensor of the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The write slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_hidden_activations,

    /**
     * @param dest_buffer_GLOBAL_hidden_mask [CONDITIONAL on out_scalar_FLAG_produce_hidden_mask] Derivative mask from ReLU operation (1 if activation > 0, else 0).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] This buffer is written to ONLY IF `out_scalar_FLAG_produce_hidden_mask` == 1. [2] If the flag is set, the write slice must be within bounds, as proven
     * by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count. [3] If the flag is set, Host shall allocate exactly
     * [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes. [4] If the flag is not set, the Host MAY pass a minimal stub buffer.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_hidden_mask,

    /**
     * @param out_scalar_FLAG_produce_hidden_mask A flag controlling mask production.
     *        - Validation Preconditions: Must be 0 or 1. If 1, the ReLU derivative mask is written to `dest_buffer_GLOBAL_hidden_mask`. If 0, mask writes are skipped.
     */
    uint out_scalar_FLAG_produce_hidden_mask,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_input_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_padded_hidden_count);

// --- Node 5: Module Layer Logit Rendering -----------------------------

/**
 * @brief (Node 5) Renders a contiguous slice of the monolithic logits buffer.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Slice Renderer. Consumes chunked inputs to render a final slice of a monolithic output buffer."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; logit output narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. Sample Masking (ADR-031): If load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; all logit outputs for that sample are set to 0 and further computation is skipped. ReLU derivative source is controlled by `src_scalar_FLAG_use_explicit_hidden_mask`. When 0: mask is derived internally from stored activations (mask = load_storage(hidden_activations) > 0). When 1: mask is read from `src_buffer_GLOBAL_hidden_mask`. The Host MAY pass a minimal stub buffer when the flag is 0. Sparsity-Aware Dot Product: The mask value is used as a branch predicate to elide dot-product terms corresponding to ReLU-zeroed hidden units. For each hidden dimension where mask < 0.5, the activation and weight reads are skipped entirely. This is a performance optimization exploiting upstream ReLU sparsity; it does not affect mathematical correctness."
 */
__kernel void render_logits_chunk(
    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask [CONDITIONAL on src_scalar_FLAG_use_explicit_hidden_mask] The ReLU mask corresponding to the hidden activations.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] This buffer is read from ONLY IF `src_scalar_FLAG_use_explicit_hidden_mask` == 1. [2] If the flag is set, the batch slice must be within bounds, as
     * proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count. [3] If the flag is set, Host must ensure this buffer was
     * allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes. [4] If the flag is not set, the Host MAY pass a minimal stub
     * buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_scalar_FLAG_use_explicit_hidden_mask A flag to select the ReLU derivative source.
     *        - Validation Preconditions: Must be 0 or 1. If 0, mask is derived from stored activations (mask = activation > 0). If 1, mask is read from `src_buffer_GLOBAL_hidden_mask`.
     */
    uint src_scalar_FLAG_use_explicit_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {
     *            dim[0] ("total_modules_count"): {Type: UNPADDED},
     *            dim[1] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[2] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
     *          }
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The module and class slices must be within bounds, as proven by: [(src_scalar_NATURAL_module_chunk_offset + src_scalar_NATURAL_module_chunk_count) <=
     * src_scalar_NATURAL_total_modules_count] AND [(src_scalar_NATURAL_class_chunk_offset + src_scalar_NATURAL_class_chunk_count) <= src_scalar_NATURAL_total_output_class_count]. [2] Host shall
     * allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_module The learnable biases for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: SIMD, Formula: "output_class_count padded for SIMD alignment"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The module and class slices must be within bounds, as proven by: [(src_scalar_NATURAL_module_chunk_offset + src_scalar_NATURAL_module_chunk_count) <=
     * src_scalar_NATURAL_total_modules_count] AND [(src_scalar_NATURAL_class_chunk_offset + src_scalar_NATURAL_class_chunk_count) <= src_scalar_NATURAL_total_output_class_count]. [2] Host shall
     * allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_biases_module,

    /**
     * @param dest_buffer_GLOBAL_logits The raw, pre-activation output tensor for all modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes. [2] Padding positions beyond `total_output_class_count` within each `padded_total_output_class_count` stride
     * are architecturally unwritten. Downstream consumers (Nodes 6, 7, 10) are contractually bounded by `total_output_class_count` and DO NOT access padding.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_logits,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_module_chunk_offset,
    uint src_scalar_NATURAL_module_chunk_count,
    uint src_scalar_NATURAL_class_chunk_offset,
    uint src_scalar_NATURAL_class_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count);

// --- Node 6: CCE Probability & Loss ----------------------------------

/**
 * @brief (Node 6) Fused kernel to compute probabilities and final CCE loss for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "The implementation is a fused, indivisible unit for numerically stable Softmax calculation. Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; partial_probs narrowed via store_storage(); final_loss written directly in COMPUTE_TYPE (no narrowing). All arithmetic exclusively in COMPUTE_TYPE. Sample Masking (ADR-031): When load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; the kernel writes zero to all probability and loss outputs for that sample and skips further computation. This ensures the diagnostic loss aggregation (Node 14) reflects only valid samples. Loss Write Predicate: The kernel writes to dest_buffer_GLOBAL_final_loss[module][sample] only when the target class index for the sample falls within the current tile's class chunk range [class_chunk_offset, class_chunk_offset + classes_per_chunk). All other tiles skip the loss write, relying on the ZERO_REQUIRED initialization."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Partial Renderer for probabilities output. Conditional Writer for loss (predicate: target class index ∈ [class_chunk_offset, class_chunk_offset + classes_per_chunk))."
 *        - Kernel Bifurcation: "CONCEPT.md Principle 3(B) — separate kernel required due to incompatible type signatures,
 *          memory layouts, and DAG topology vs. Node 7 (BCE path). CONTRACT §7.0 Exception applies."
 */
__kernel void compute_probs_loss_cce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes for this buffer. [2] All values must be strictly positive and finite (guaranteed by host initialization and Node 25 post-update enforcement).
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (class indices).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] Values must be in [0, total_output_class_count - 1]. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(int)] bytes for this
     * buffer.
     */
    __global const int *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes for this buffer.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_final_loss The monolithic buffer for final loss values.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] Host shall zero-initialize this buffer. [2] The kernel populates this buffer via a direct scatter-write; no host-side aggregation is required. [3] Host
     * shall allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_final_loss,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    /**
     * @param src_scalar_NATURAL_total_tile_count The total number of tiles in the flattened (module_chunk, class_chunk) grid.
     *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
     */
    uint src_scalar_NATURAL_total_tile_count);

// --- Node 7: BCE Probability & Loss ----------------------------------

/**
 * @brief (Node 7) Computes probabilities and PARTIAL BCE loss for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "The implementation shall employ a numerically stable Sigmoid computation (two-branch form: for positive logits, σ(x) = 1/(1+exp(-x)); for negative logits, σ(x) = exp(x)/(1+exp(x))) to prevent exp() overflow. BCE loss shall use a numerically stable formulation that avoids log(0) (e.g., max(x,0) - x*y + log(1+exp(-|x|))). Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; partial_probs narrowed via store_storage(); partial_loss written in COMPUTE_TYPE. All arithmetic exclusively in COMPUTE_TYPE. Sample Masking (ADR-031): When load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; the kernel writes zero to all probability and loss outputs for that sample and skips further computation. This ensures the diagnostic loss aggregation (Node 14) reflects only valid samples."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Dual Partial Renderer. Uses flat_tile_index for both probability and loss outputs."
 *        - Kernel Bifurcation: "CONCEPT.md Principle 3(B) — separate kernel required due to incompatible type signatures,
 *          memory layouts, and DAG topology vs. Node 6 (CCE path). CONTRACT §7.0 Exception applies."
 */
__kernel void compute_probs_loss_bce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] The tile access must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate exactly
     * [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes for this buffer. [3] All values must be strictly positive and finite (guaranteed by host initialization and Node 25 post-update enforcement).
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (multi-hot encoded).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes for this buffer.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_partial_loss The collection buffer for this tile's computed partial loss.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial_loss,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    /**
     * @param src_scalar_NATURAL_total_tile_count The total number of tiles in the flattened (module_chunk, class_chunk) grid.
     *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
     */
    uint src_scalar_NATURAL_total_tile_count);

// =====================================================================
// Learn Phase I: Gradient Generation & Clipping (Nodes 8–11)
//
// Parallel gradient computation for module parameters, hidden-layer
// error, and temperature parameters, followed by per-tile clipping.
// =====================================================================

// --- Node 8: Module Parameter Gradients -------------------------------

/**
 * @brief (Node 8) Computes partial module param gradients (Weights, Biases) for a tile and batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE. Padding Zero-Preservation: The kernel writes computed gradients only within its assigned class chunk range [class_offset, class_offset + classes_per_chunk) for each (module, hidden) position within its tile and batch chunk. Positions outside this range — including other tiles' class ranges and class-dimension padding — are never written; the ZERO_REQUIRED initialization of the destination buffer is the primary guarantor of zeros at these positions. For hidden-dimension padding (h >= hidden_count) within the class chunk range, the computation produces zero because upstream hidden activations at padding indices are zero (Zero-Propagation Theorem precondition (b): weight padding is zero, precondition (c): bias padding is zero, therefore hidden activation padding is zero, therefore 0 × (prob - target) = 0). The kernel is not required to special-case these positions — the zero output is a mathematical consequence of the upstream invariant chain. Implementations MAY skip computation at padding indices as a performance optimization; both approaches satisfy Padding Zero-Preservation. Sample Masking (ADR-031): When load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; the kernel skips the entire gradient contribution for that sample. This is a correctness requirement for bias gradients — not merely a performance optimization. Upstream invariants (Nodes 6/7) write zero probabilities for masked samples, but the loss derivative d_loss/d_logit remains non-zero when the target signal is non-zero (CCE: (prob − 1)/τ = −1/τ at the true class; BCE: prob − target = −target for non-zero targets). Without the mask check, bias gradients accumulate these spurious contributions directly (grad_bias[c] += d_loss/d_logit[sample][c]). Weight gradients are independently protected by the upstream hidden_activations = 0 invariant (Node 4's sample masking guarantees zero hidden activations for masked samples, so d_loss/d_logit × 0 = 0), but the unified per-sample skip is the sole guarantor of correct bias gradients."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Dual Partial Renderer. Uses (flat_tile_index, batch_chunk_index) composite placement for both weight and bias gradient outputs. Per Principle §3 Inter-Dispatch Reduction Delegation, each dispatch writes to a unique slot; batch-chunk summation is delegated to a ReductionTreeNode inserted before Node 11."
 */
__kernel void calculate_module_param_grads_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction of per-sample gradient
     *          contributions across the work-group."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Precision Role: "flag-conditional" — When src_scalar_FLAG_problem_type == BCE: storage-role (STORAGE_TYPE multi-hot targets), widened to COMPUTE_TYPE upon load. When src_scalar_FLAG_problem_type == CCE: integer-typed (exempt), interpreted as int class indices.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout, type, and total size
     * correspond to the value of `src_scalar_FLAG_problem_type`. [3] The kernel implementation will cast this pointer internally based on the flag.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] The tile access must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate exactly
     * [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_module The collection buffer for this (tile, batch_chunk) pair's computed weight gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {
     *            dim[0] ("total_tile_count"): {Type: UNPADDED},
     *            dim[1] ("num_batch_chunks"): {Type: UNPADDED},
     *            dim[2] ("modules_per_chunk"): {Type: UNPADDED},
     *            dim[3] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[4] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
     *          }
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls_batch(src_scalar_NATURAL_flat_tile_index, src_scalar_NATURAL_batch_chunk_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] The batch chunk index must be valid, as proven by: src_scalar_NATURAL_batch_chunk_index < src_scalar_NATURAL_num_batch_chunks. [3] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_num_batch_chunks * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     * [4] Host shall zero-initialize this buffer prior to dispatch. Each (tile, batch_chunk) writes only `classes_per_chunk` positions within the `padded_total_output_class_count`-wide innermost dimension; the ZERO_REQUIRED initialization covers all unwritten positions. A batch-chunk ReductionTreeNode sums across batch chunks before Node 11.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_module The collection buffer for this (tile, batch_chunk) pair's computed bias gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {
     *            dim[0] ("total_tile_count"): {Type: UNPADDED},
     *            dim[1] ("num_batch_chunks"): {Type: UNPADDED},
     *            dim[2] ("modules_per_chunk"): {Type: UNPADDED},
     *            dim[3] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
     *          }
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls_batch(src_scalar_NATURAL_flat_tile_index, src_scalar_NATURAL_batch_chunk_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] The batch chunk index must be valid, as proven by: src_scalar_NATURAL_batch_chunk_index < src_scalar_NATURAL_num_batch_chunks. [3] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_num_batch_chunks * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     * [4] Host shall zero-initialize this buffer prior to dispatch. Each (tile, batch_chunk) writes only `classes_per_chunk` positions within the `padded_total_output_class_count`-wide innermost dimension; the ZERO_REQUIRED initialization covers all unwritten positions. A batch-chunk ReductionTreeNode sums across batch chunks before Node 11.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_biases_module,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_batch_chunk_index,
    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_num_batch_chunks,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    /**
     * @param src_scalar_NATURAL_total_tile_count The total number of tiles in the flattened (module_chunk, class_chunk) grid.
     *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
     */
    uint src_scalar_NATURAL_total_tile_count);

// --- Node 9: Hidden Layer Error Backpropagation -----------------------

/**
 * @brief (Node 9) Computes the partial upstream gradient for the hidden layer (Grad_H) for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel processes the complete batch dimension in a single dispatch. Batch-chunking parameters (batch_chunk_offset, batch_chunk_count) are intentionally absent because the downstream Item Synchronization Point (Node 13) requires a monolithic collection buffer."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened upon load; storage-role output narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. Padding Zero-Establishment: For the padded_hidden_count dimension, the kernel SHALL write zero for all positions at indices >= hidden_count. The kernel is the sole guarantor of zeros at padding positions (Initialization Contract: NOT_REQUIRED). This guarantees that downstream L2 norm computations (Node 11) over the full padded extent are mathematically equivalent to norms over the logical extent. Sample Masking (ADR-031): When load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; the kernel writes zero to all gradient outputs for that sample and skips further computation. This is a correctness requirement — not merely a performance optimization. Upstream invariants (Nodes 6/7) write zero probabilities for masked samples, but the loss derivative d_loss/d_logit remains non-zero when the target signal is non-zero (CCE: −1/τ at the true class; BCE: −target for non-zero targets). Without the mask check, Grad_H[sample][h] = Σ_c (d_loss/d_logit[sample][c] × W[h][c]) produces non-zero gradients because module weights are non-zero learnable parameters. These spurious gradients would propagate through Nodes 11 → 13 → 16 into summed_grad_h. Downstream consumers (Nodes 17, 18) cite this masking as an upstream invariant ('summed_grad_h = 0 from Node 9's masking') — one of two independent guarantees protecting shared-layer gradient correctness."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Partial Renderer for a monolithic intermediate buffer."
 */
__kernel void backprop_error_to_hidden_chunk(
    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Precision Role: "flag-conditional" — When src_scalar_FLAG_problem_type == BCE: storage-role (STORAGE_TYPE multi-hot targets), widened to COMPUTE_TYPE upon load. When src_scalar_FLAG_problem_type == CCE: integer-typed (exempt), interpreted as int class indices.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout, type, and total size
     * correspond to the value of `src_scalar_FLAG_problem_type`.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes for this buffer.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {
     *            dim[0] ("total_modules_count"): {Type: UNPADDED},
     *            dim[1] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[2] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
     *          }
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The overarching tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] The tile access must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate exactly
     * [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_hidden_activations_aos The collection buffer for this tile's computed upstream gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     * [3] [ARCHITECTURAL SYNCHRONIZATION POINT] The consumer (Node 13) requires a monolithic input collection for its gather operation. Therefore, the Host Orchestrator MUST NOT stream the batch
     * dimension when populating this buffer.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    /**
     * @param src_scalar_NATURAL_total_tile_count The total number of tiles in the flattened (module_chunk, class_chunk) grid.
     *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
     */
    uint src_scalar_NATURAL_total_tile_count);

// --- Node 10: Temperature Gradients -----------------------------------

/**
 * @brief (Node 10) Computes partial temperature gradients for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened upon load; partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE. Sample Masking (ADR-031): When load_sample_mask(sample_mask, sample) == 0, the sample is invalid/padding; the kernel skips the entire contribution for that sample. This is a performance optimization — not a correctness requirement. Upstream invariants (zeroed logits from Node 5, uniform probabilities from Node 6/7) guarantee zero temperature gradient contribution for masked samples regardless of whether the early exit is applied."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Partial Renderer for temperature gradients."
 */
__kernel void calculate_chunk_temp_gradients(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction of per-sample
     *          temperature gradient contributions across the work-group."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Precision Role: "flag-conditional" — When src_scalar_FLAG_problem_type == BCE: storage-role (STORAGE_TYPE multi-hot targets), widened to COMPUTE_TYPE upon load. When src_scalar_FLAG_problem_type == CCE: integer-typed (exempt), interpreted as int class indices.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout, type, and total size
     * correspond to the value of `src_scalar_FLAG_problem_type`.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes for this buffer.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] The tile access must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate exactly
     * [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_temps The collection buffer for this tile's computed temperature gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_temps,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    /**
     * @param src_scalar_NATURAL_total_tile_count The total number of tiles in the flattened (module_chunk, class_chunk) grid.
     *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
     */
    uint src_scalar_NATURAL_total_tile_count);

// --- Node 11: Partial Gradient Clipping -------------------------------

/**
 * @brief (Node 11) [Utility Kernel] Computes the total L2 Norm for a single item's partial gradients and conditionally scales them. Supports both a single batch-wide clipping norm and per-item norms.
 * @kernel_contract
 *        - Holistic Constraints: "The kernel processes the complete set of partial gradients for a single logical work item (`flat_tile_index`). The clipping threshold is determined by
 * `src_scalar_FLAG_use_per_item_norm`."
 *        - Behavioral Invariants: "[1] Implements a two-pass algorithm: Norm calculation followed by conditional scaling. [2] An epsilon term shall be used to prevent division by zero when
 * calculating the scaling factor. [3] The L2 norm is computed over the logical concatenation of all four input gradient buffers (weights, biases, temperatures, hidden activations). A single derived scaling factor is applied uniformly to all four output buffers. Independent per-buffer norms are a contract violation. Precision Boundary Conversion: storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. FP8 Quantization Note: When STORAGE_TYPE is FP8, the clip-then-store sequence introduces re-quantization error. Gradient components scaled below the FP8 quantization floor (2^-9 for E4M3, 2^-16 for E5M2) may round to zero, effectively zeroing a subset of the gradient. This is an accepted consequence of the Primacy of Memory Strategy — the architecture trades gradient fidelity for 4x bandwidth compression. The Quadratic Scaling Policy's threshold schedule accounts for this by maintaining gradients well above the quantization floor. Padding Zero-Preservation: The uniform-scaling algorithm applies a single multiplicative factor derived from the joint L2 norm to all positions in all four output buffers. No per-element additive term exists. Zero-valued positions in the input (established by Node 9's Padding Zero-Establishment for Grad_H, and by Node 8's Padding Zero-Preservation for Grad_ModW and Grad_ModB) are mapped to zero in the output for all finite scaling factors. This is a mathematical consequence of the single-scale-factor design, not an active zeroing step. Implementations MAY skip computation at padding indices as a performance optimization."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Utility / Stability Primitive. Acts as a barrier for a single item's partial results before reduction."
 */
__kernel void clip_partial_gradients(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction of the sum-of-squares.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction to compute the L2 norm
     *          (sum of squares) across the concatenated gradient vector. Each
     *          thread accumulates partial sums of squares for its assigned
     *          elements, then the work-group reduces to a single scalar."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_grad_weights_module Source buffer: output of Node 8's collection buffer (after batch-chunk reduction when `num_batch_chunks > 1` and, when `storage_dtype != compute_dtype`, a `narrow_to_storage` Precision Bridge; direct from Node 8 otherwise).
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {
     *            dim[0] ("total_tile_count"): {Type: UNPADDED},
     *            dim[1] ("modules_per_chunk"): {Type: UNPADDED},
     *            dim[2] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[3] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
     *          }
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_weights_module,

    /**
     * @param src_buffer_GLOBAL_partial_grad_biases_module Source buffer: output of Node 8's collection buffer (after batch-chunk reduction when `num_batch_chunks > 1` and, when `storage_dtype != compute_dtype`, a `narrow_to_storage` Precision Bridge; direct from Node 8 otherwise).
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: SIMD, Formula: "SIMD_WIDTH alignment via padded_total_output_class_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_biases_module,

    /**
     * @param src_buffer_GLOBAL_partial_grad_temps Source buffer from Node 10.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_temps,

    /**
     * @param src_buffer_GLOBAL_partial_grad_hidden_activations_aos Source buffer from Node 9.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_hidden_activations_aos,

    /**
     * @param src_buffer_GLOBAL_CONST_clipping_threshold_per_item [CONDITIONAL on src_scalar_FLAG_use_per_item_norm] A buffer containing a distinct clipping threshold for each item.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count]
     *        - Validation Preconditions: [1] This buffer is read from ONLY IF `src_scalar_FLAG_use_per_item_norm` == 1. [2] If the flag is set, the Host MUST provide a valid buffer of size
     * [src_scalar_NATURAL_total_tile_count * sizeof(COMPUTE_TYPE)]. [3] If the flag is not set, the Host MAY pass a minimal stub buffer.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_CONST_clipping_threshold_per_item,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_weights_module Output for clipped weight gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {
     *            dim[0] ("total_tile_count"): {Type: UNPADDED},
     *            dim[1] ("modules_per_chunk"): {Type: UNPADDED},
     *            dim[2] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[3] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
     *          }
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_weights_module`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_weights_module,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_biases_module Output for clipped bias gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: SIMD, Formula: "SIMD_WIDTH alignment via padded_total_output_class_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_biases_module`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_biases_module,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_temps Output for clipped temperature gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_temps`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_temps,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos Output for clipped upstream gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_hidden_activations_aos`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,

    /**
     * @param src_scalar_FLAG_use_per_item_norm A flag to select the clipping threshold source.
     *        - Validation Preconditions: Must be 0 or 1. If 0, `src_scalar_REAL_clipping_threshold_t_pre` is used. If 1, the value is sourced from
     * `src_buffer_GLOBAL_CONST_clipping_threshold_per_item`.
     */
    uint src_scalar_FLAG_use_per_item_norm,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_pre [CONDITIONAL] The maximum permissible L2 norm, applied to all items if the controlling flag is 0.
     *        - Validation Preconditions: [1] Must be a positive real number. This value is IGNORED if `src_scalar_FLAG_use_per_item_norm` == 1.
     *          [2] The Host MUST ensure this value satisfies the Pre-Summation Amplification constraint:
     *          (src_scalar_REAL_clipping_threshold_t_pre * src_scalar_NATURAL_num_class_chunks) <= COMPUTE_FP_FORMAT_MAX,
     *          because the downstream Node 13 sums num_class_chunks clipped tiles per element (see CONCEPT.md §3.4).
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_pre,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    /**
     * @param src_scalar_NATURAL_total_tile_count The total number of tiles in the flattened (module_chunk, class_chunk) grid.
     *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
     */
    uint src_scalar_NATURAL_total_tile_count);

// =====================================================================
// Learn Phase II: Aggregation, Reduction & Specialized Processing
//                 (Nodes 13, 14–15, 16, 20)
//
// Gradient gather/permutation (Item Synchronization Point), the
// Recursive Clip-Aggregation Engine, and the specialized Grad_H
// reduction.
// =====================================================================

// --- Node 13: Gradient Gather & Permutation (Item Synchronization) ----

/**
 * @brief (Node 13) Specialized Kernel: Gathers scattered partial gradients into a single, reduction-ready buffer.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a specialized architectural primitive designed to solve the 'Transpose Illusion' by gathering scattered partial results into a dense, reduction-ready
 * SoA layout."
 *        - Behavioral Invariants: "The gather operation performs an implicit reduction (summation) over the `class_chunk` dimension. Precision Boundary Conversion: storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. Padding Zero-Preservation: The kernel writes only to destination positions corresponding to tiles within the logical (module, batch, hidden) domain. For the module dimension: positions at indices >= total_modules_count within the padded_total_modules_count stride are never written; the ZERO_REQUIRED initialization of the destination buffer is the primary guarantor of zeros at these positions. For the hidden dimension: the input's zero-valued padding positions (indices >= hidden_count, established by Node 9 and preserved by Node 11) are read, summed across class chunks (sum of zeros = zero), and stored via store_storage() — preserving zero at the corresponding output positions."
 *        - Synchronization Model: "Global Barrier. This kernel cannot execute until all its clipped partial inputs from Node 11 are fully rendered."
 *        - Idempotency: "Strictly Idempotent"
 */
__kernel void gather_and_permute_grad_hidden_activations(
    /**
     * @param src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos The full collection of *clipped* partial upstream gradients, produced by Node 11.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes. [2] [ARCHITECTURAL SYNCHRONIZATION POINT] The consumer (this kernel) requires a monolithic input fully populated by its
     * preceding dependency, Node (11).
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,

    /**
     * @param dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa The final, contiguous, SoA-layout buffer ready for reduction by Node 16.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count)
     *        - Padding Contract: {
     *            dim[0] ("total_batch_count * hidden_count" → "total_batch_count * padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment on hidden_count stride"},
     *            dim[1] ("total_modules_count" → "padded_total_modules_count"): {Type: CACHE, Formula: "128-byte alignment"}
     *          }
     *        - Precision Role: "storage"
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly [(src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count) * src_scalar_NATURAL_padded_total_modules_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,

    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_padded_total_modules_count,
    uint src_scalar_NATURAL_num_module_chunks,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_total_tile_count);

// --- Recursive Clip-Aggregation Engine (Nodes 14, 15, 20) ------------
//
// This section contains all kernels composing the host-orchestrated
// log_K(N) reduction tree. The engine is not a monolithic kernel but
// an emergent property of composing these single-purpose primitives.
//
// Each kernel exists in two precision variants (ADR-026):
//   Storage-entry: reads STORAGE_TYPE partials via load_storage()
//   Compute-entry: reads COMPUTE_TYPE intermediates directly
//
// The Orchestration tier selects the variant based on the source
// buffer's precision_role from its BufferDescriptor.

// --- Single-Stage Reduction: Register Reduce (Storage-Entry, Tier 1) --

/**
 * @brief (Node 14, 15a & 20a) Tier 1 (N is small): Reduces scattered partial results using registers and an indirection list.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a collection buffer, located via an explicit offset list. This avoids host-side staging
 * copies. When used as a stage in a multi-stage reduction tree, operation_type MUST be AGG_MODE_SUM; AGG_MODE_AVERAGE is valid only as a single-stage terminal reduction, as partial-count
 * division at interior stages produces silently incorrect results."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 */
__kernel void aggregate_register_reduce(
    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing all partial results for this stage.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and
     * `src_scalar_NATURAL_partial_width`.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

// --- Single-Stage Reduction: Register Reduce (Compute-Entry, Tier 1) --

/**
 * @brief (Node 14, 15a & 20a) Compute-entry variant of aggregate_register_reduce.
 *        Identical algorithm reading COMPUTE_TYPE intermediates directly.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a COMPUTE_TYPE collection buffer, located via an explicit offset list.
 * When used as a stage in a multi-stage reduction tree, operation_type MUST be AGG_MODE_SUM; AGG_MODE_AVERAGE is valid only as a single-stage terminal reduction, as partial-count division
 * at interior stages produces silently incorrect results."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. All buffers are compute-role; no precision boundary conversion is required. Reduction accumulation in COMPUTE_TYPE; compute-role output written directly. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 *        - Precision Variant: "Compute-entry variant of aggregate_register_reduce. Used for interior stages of multi-stage reduction trees (where the source is a prior stage's COMPUTE_TYPE output) and for leaf stages whose source collection is natively COMPUTE_TYPE."
 */
__kernel void aggregate_register_reduce_from_compute(
    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing COMPUTE_TYPE intermediate results.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and `src_scalar_NATURAL_partial_width`.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

// --- Single-Stage Reduction: Local Reduce (Storage-Entry, Tier 2) -----

/**
 * @brief (Node 14, 15a & 20a) Tier 2 (N is large): Reduces scattered partial results using local memory and an indirection list.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a collection buffer, located via an explicit offset list. This avoids host-side staging
 * copies. When used as a stage in a multi-stage reduction tree, operation_type MUST be AGG_MODE_SUM; AGG_MODE_AVERAGE is valid only as a single-stage terminal reduction, as partial-count
 * division at interior stages produces silently incorrect results."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage / Work-group Parallel"
 */
__kernel void aggregate_local_reduce(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel summation of per-element
     *          contributions from scattered partials. Each thread loads and
     *          accumulates its share of the partial_offset_list entries, then
     *          the work-group reduces to produce one output element per
     *          reduction round."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing all partial results for this stage.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and
     * `src_scalar_NATURAL_partial_width`.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

// --- Single-Stage Reduction: Local Reduce (Compute-Entry, Tier 2) -----

/**
 * @brief (Node 14, 15a & 20a) Compute-entry variant of aggregate_local_reduce.
 *        Identical algorithm reading COMPUTE_TYPE intermediates directly.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a COMPUTE_TYPE collection buffer, located via an explicit offset list.
 * When used as a stage in a multi-stage reduction tree, operation_type MUST be AGG_MODE_SUM; AGG_MODE_AVERAGE is valid only as a single-stage terminal reduction, as partial-count division
 * at interior stages produces silently incorrect results."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. All buffers are compute-role; no precision boundary conversion is required. Reduction accumulation in COMPUTE_TYPE; compute-role output written directly. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage / Work-group Parallel"
 *        - Precision Variant: "Compute-entry variant of aggregate_local_reduce. Used for interior stages of multi-stage reduction trees (where the source is a prior stage's COMPUTE_TYPE output) and for leaf stages whose source collection is natively COMPUTE_TYPE."
 */
__kernel void aggregate_local_reduce_from_compute(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Identical access pattern to aggregate_local_reduce; the only
     *          difference is that source reads are COMPUTE_TYPE rather than
     *          STORAGE_TYPE."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing COMPUTE_TYPE intermediate results.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and `src_scalar_NATURAL_partial_width`.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

// --- Single-Stage Clip: clip_intermediate_grad ------------------------

/**
 * @brief (Node 15b, 20b) [Utility Kernel] Applies partial-group-wise clipping to a single, contiguous, intermediate gradient buffer.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a core component of the host-driven, recursive clip-aggregation engine. It atomically computes an L2 norm over its entire input buffer and
 * conditionally scales that buffer in-place."
 *        - Behavioral Invariants: "An epsilon term shall be used to prevent division by zero when calculating the scaling factor. The implementation must use local memory for the norm reduction to be
 * scalable. All buffers are compute-role; no precision boundary conversion is required."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage Clip Primitive"
 */
__kernel void clip_intermediate_grad(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction of the sum-of-squares for the L2 norm.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction to compute a single
     *          L2 norm over the entire intermediate gradient buffer. Each thread
     *          accumulates partial sums of squares, then the work-group reduces
     *          to a single scalar used for the conditional scaling decision."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param update_buffer_GLOBAL_intermediate_grad The buffer to be clipped in-place. This is typically the output of a preceding `aggregate_*` kernel.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host must provide a valid buffer containing exactly `src_scalar_NATURAL_parameter_count` elements.
     */
    __global COMPUTE_TYPE *update_buffer_GLOBAL_intermediate_grad,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_j The clipping threshold for this specific reduction stage `j`.
     *        - Calculability Proof: [Host-side calculation based on the active stabilization policy (e.g., Quadratic Scaling Policy)]
     *        - Validation Preconditions: The value must be a positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_j,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent division by zero during norm calculation.
     *        - Validation Preconditions: Must be a small, positive real number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    /**
     * @param src_scalar_NATURAL_parameter_count The total number of elements in the `update_buffer_GLOBAL_intermediate_grad` buffer.
     *        - Calculability Proof: [Known by Host Orchestrator based on the parameter group being processed]
     *        - Validation Preconditions: Must match the element count of the `update_buffer_GLOBAL_intermediate_grad` buffer.
     */
    uint src_scalar_NATURAL_parameter_count);

// --- Multi-Stage K-Fan-In Reduction Primitive (ADR-019) ---------------
//
// Sentinel value indicating an absent partial in the tail node of a
// K-fan-in reduction stage. When num_partials is not divisible by K,
// the final node's offset list is padded with this sentinel.
#define SENTINEL_ABSENT_PARTIAL 0xFFFFFFFFu

// --- Multi-Stage Reduction: K-Fan-In (Storage-Entry) ------------------

/**
 * @brief (Node 14, 15a, 20a — multi-stage) Reduces groups of K scattered
 *        partials into independent output nodes with optional per-node L2 clip.
 * @kernel_contract
 *        - Holistic Constraints: "Each work-group processes one reduction node.
 *          The kernel reads K partials per node from the source buffer via an
 *          offset list, sums them, optionally clips the result per-node, and
 *          writes one output vector of partial_width elements. Supports absent
 *          partials via sentinel offset 0xFFFFFFFF for the tail node."
 *        - Behavioral Invariants: "When clipping_threshold >= 0, per-node L2
 *          clip is applied: scale = threshold / (norm + epsilon). When
 *          clipping_threshold < 0, clip is bypassed (diagnostic mode).
 *          Zero threshold clips to zero norm (zeroes all gradients).
 *          Epsilon prevents division by zero. Precision Boundary Conversion: storage-role partials widened via load_storage(); compute-role outputs written directly; LOCAL scratch uses COMPUTE_TYPE. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 */
__kernel void reduce_k_fan_in_and_clip(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel L2 norm reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction to compute the per-node
     *          L2 norm (sum of squares) over partial_width elements. Each thread
     *          accumulates partial sums of squares for its assigned elements of
     *          the summed K-partial vector, then the work-group reduces to a
     *          single scalar for the conditional scaling decision. One work-group
     *          per reduction node."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing all
     *        partial results referenced by the offset list.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that
     *          encompasses all memory regions referenced by the combination of
     *          `src_buffer_GLOBAL_CONST_offset_list_flat` and
     *          `src_scalar_NATURAL_partial_width`.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_offset_list_flat Flat offset list with K
     *        consecutive entries per node. Sentinel SENTINEL_ABSENT_PARTIAL
     *        (0xFFFFFFFF) indicates an absent partial in the tail node.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_fan_in]
     *        - Validation Preconditions: Host must provide a buffer containing exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in` uint entries.
     */
    __global const uint *src_buffer_GLOBAL_CONST_offset_list_flat,

    /**
     * @param dest_buffer_GLOBAL_stage_partial Contiguous output buffer. Node n writes
     *        at `[n * partial_width, (n+1) * partial_width)`.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)` bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_stage_partial,

    /**
     * @param src_scalar_NATURAL_fan_in Number of partials to reduce per node.
     *        - Validation Preconditions: Must be >= 2.
     */
    uint src_scalar_NATURAL_fan_in,

    /**
     * @param src_scalar_NATURAL_node_count Number of independent reduction nodes.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_node_count,

    /**
     * @param src_scalar_NATURAL_partial_width Number of elements per partial vector.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_partial_width,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_j The clipping threshold for this stage j.
     *        Value < 0 disables clip (diagnostic mode). Value 0 clips to zero norm.
     *        - Validation Preconditions: Negative values bypass clipping; non-negative values enable it.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_j,

    /**
     * @param src_scalar_REAL_epsilon Small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon);

// --- Multi-Stage Reduction: K-Fan-In (Compute-Entry) ------------------

/**
 * @brief (Node 14, 15a, 20a — interior stages and compute-role leaf stages)
 *        Compute-entry variant of reduce_k_fan_in_and_clip. Identical algorithm,
 *        but reads COMPUTE_TYPE intermediates rather than STORAGE_TYPE partials.
 * @kernel_contract
 *        - Holistic Constraints: "Each work-group processes one reduction node.
 *          The kernel reads K partials per node from the source buffer via an
 *          offset list, sums them, optionally clips the result per-node, and
 *          writes one output vector of partial_width elements. Supports absent
 *          partials via sentinel offset 0xFFFFFFFF for the tail node."
 *        - Behavioral Invariants: "When clipping_threshold >= 0, per-node L2
 *          clip is applied: scale = threshold / (norm + epsilon). When
 *          clipping_threshold < 0, clip is bypassed (diagnostic mode).
 *          Zero threshold clips to zero norm (zeroes all gradients).
 *          Epsilon prevents division by zero. All buffers are compute-role;
 *          no precision boundary conversion is required. Reduction accumulation
 *          in COMPUTE_TYPE; compute-role output written directly. All arithmetic
 *          exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 *        - Precision Variant: "Compute-entry variant of reduce_k_fan_in_and_clip.
 *          Used for interior stages of multi-stage reduction trees (where the
 *          source is a prior stage's COMPUTE_TYPE output) and for leaf stages
 *          whose source collection is natively COMPUTE_TYPE (e.g., BCE loss
 *          partials from Node 7)."
 */
__kernel void reduce_k_fan_in_and_clip_from_compute(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel L2 norm reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Identical access pattern to reduce_k_fan_in_and_clip; the only
     *          difference is that source reads are COMPUTE_TYPE rather than
     *          STORAGE_TYPE."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing
     *        COMPUTE_TYPE intermediate results from a prior reduction stage or
     *        a natively compute-role partial collection.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that
     *          encompasses all memory regions referenced by the combination of
     *          `src_buffer_GLOBAL_CONST_offset_list_flat` and
     *          `src_scalar_NATURAL_partial_width`.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_offset_list_flat Flat offset list with K
     *        consecutive entries per node. Sentinel SENTINEL_ABSENT_PARTIAL
     *        (0xFFFFFFFF) indicates an absent partial in the tail node.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_fan_in]
     *        - Validation Preconditions: Host must provide a buffer containing exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in` uint entries.
     */
    __global const uint *src_buffer_GLOBAL_CONST_offset_list_flat,

    /**
     * @param dest_buffer_GLOBAL_stage_partial Contiguous output buffer. Node n writes
     *        at `[n * partial_width, (n+1) * partial_width)`.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)` bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_stage_partial,

    /**
     * @param src_scalar_NATURAL_fan_in Number of partials to reduce per node.
     *        - Validation Preconditions: Must be >= 2.
     */
    uint src_scalar_NATURAL_fan_in,

    /**
     * @param src_scalar_NATURAL_node_count Number of independent reduction nodes.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_node_count,

    /**
     * @param src_scalar_NATURAL_partial_width Number of elements per partial vector.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_partial_width,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_j The clipping threshold for this stage j.
     *        Value < 0 disables clip (diagnostic mode). Value 0 clips to zero norm.
     *        - Validation Preconditions: Negative values bypass clipping; non-negative values enable it.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_j,

    /**
     * @param src_scalar_REAL_epsilon Small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon);

// --- Precision Bridge Primitive ---------------------------------------
//
// Inserted by the Orchestration tier at inter-kernel DAG edges where a
// compute-role producer feeds a storage-role consumer and no intermediate
// kernel naturally performs the conversion (see CONCEPT.md §5, Inter-Kernel
// Precision Boundary Resolution). The canonical use case is the batch-chunk
// ReductionTreeNode output (compute-role) feeding Node 11 (storage-role
// inputs). When storage_dtype == compute_dtype, the Orchestration tier
// elides this dispatch entirely — the source buffer is bit-compatible.

/**
 * @brief [Utility Kernel] Element-wise precision narrowing from compute
 *        role to storage role.
 * @kernel_contract
 *        - Holistic Constraints: "Generic, element-wise precision bridge.
 *          Inserted by the Orchestration tier at DAG edges where a
 *          compute-role producer feeds a storage-role consumer and no
 *          intermediate kernel naturally performs the conversion. When
 *          storage_dtype == compute_dtype, the Orchestration tier elides
 *          this dispatch entirely — the source buffer is bit-compatible
 *          with the consumer's expectation."
 *        - Behavioral Invariants: "Precision Boundary Conversion: reads
 *          COMPUTE_TYPE input directly; narrows to storage precision via
 *          store_storage(). When STORAGE_TYPE == COMPUTE_TYPE, the
 *          operation is an identity copy eliminated by compiler
 *          optimization (and the dispatch itself is elided at plan
 *          construction). When STORAGE_TYPE is FP8, the output is
 *          subject to the FP8 quantization floor and saturation
 *          semantics of store_storage_fp8(). No arithmetic is performed
 *          on the values — this is a pure format conversion.
 *          Padding Zero Propagation (Emergent): Zero-valued positions
 *          in the input map to zero-valued positions in the output
 *          because store_storage(0) = 0 for all supported precision
 *          formats — including FP8, where 0x00 represents zero in
 *          both E4M3 and E5M2. The bridge is padding-agnostic; zero
 *          propagation is a consequence of the identity property of
 *          zero under format conversion."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Utility"
 */
__kernel void narrow_to_storage(
    /**
     * @param src_buffer_GLOBAL_input The compute-role source buffer.
     *        - Tensor Shape: (src_scalar_NATURAL_element_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_element_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [src_scalar_NATURAL_element_count *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param dest_buffer_GLOBAL_output The storage-role destination
     *        buffer.
     *        - Tensor Shape: (src_scalar_NATURAL_element_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_element_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [src_scalar_NATURAL_element_count *
     *          sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_output,

    /**
     * @param src_scalar_NATURAL_element_count Total number of elements
     *        to convert.
     *        - Calculability Proof: [Known by Host Orchestrator from
     *          the source buffer's BufferDescriptor]
     *        - Validation Preconditions: Must match the element count
     *          of both src and dest buffers.
     */
    uint src_scalar_NATURAL_element_count);

// --- Node 16: Specialized Grad_H Reduction ----------------------------

/**
 * @brief (Node 16) Specialized Kernel: Reduces the permuted Grad_H buffer
 *        using a host-prescribed, multi-stage, numerically-stable reduction.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel performs a complete, row-wise
 *          reduction on the monolithic, contiguous SoA buffer produced by the
 *          upstream Item Synchronization Point (Node 13). The stabilization
 *          schedule is prescribed by the Orchestration tier; the kernel applies
 *          it without independent policy computation."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role
 *          inputs widened via load_storage(); reduction accumulation in
 *          COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE.
 *          The kernel's behavior is contractually bound to:
 *
 *          1. **Pre-accumulation Phase:**
 *             Each thread accumulates
 *             ceil(total_modules_count / get_local_size(0)) elements from its
 *             assigned row. Each thread's accumulator is clipped to
 *             src_scalar_REAL_clipping_threshold_t_pre after accumulation.
 *             When total_modules_count <= get_local_size(0), each thread loads
 *             at most one element and the clip is a no-op (the host sets the
 *             threshold sufficiently high).
 *
 *          2. **Staged Reduction Phase:**
 *             num_reduction_stages rounds of work-group-parallel tree reduction.
 *             After each round's summation, the per-element result is clipped
 *             to src_buffer_GLOBAL_CONST_clipping_threshold_per_stage[s]
 *             (where s indexes the round from 0 to num_reduction_stages - 1).
 *             The kernel determines the internal binary tree structure and
 *             fan-in distribution among stages.
 *             Thread 0 writes the final single-element result to the
 *             destination buffer.
 *
 *          When num_reduction_stages is 0, the kernel bypasses the Staged
 *          Reduction Phase entirely: pre-accumulation produces at most one
 *          value per row, and thread 0 writes it directly.
 *
 *          Padding Zero Propagation (Emergent): Hidden-dimension padding
 *          positions (indices h >= hidden_count within padded_hidden_count)
 *          in the output buffer carry zero when the input buffer has zero
 *          at the corresponding module-column positions for all modules.
 *          This is a mathematical consequence of summing zeros across the
 *          module reduction dimension. Preconditions: Node 13's Padding
 *          Zero-Preservation ensures zero at padding positions in the
 *          input SoA buffer."
 *        - Synchronization Model: "Specialized Reduction Kernel / Global
 *          Barrier"
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Architectural Justification: "This kernel is a specialized,
 *          single-launch reduction engine optimized for the contiguous SoA
 *          buffer produced by the upstream Node 13 synchronization point.
 *          The Orchestration tier prescribes the threshold schedule, and the
 *          kernel applies it — consistent with the threshold injection pattern
 *          used by the generic reduction engine (Nodes 14/15/20). The kernel
 *          retains sole control of its internal execution geometry (thread
 *          assignment, pre-accumulation fan-in, binary tree structure), while
 *          the Host retains sole control of stabilization policy through the
 *          prescribed schedule."
 */
__kernel void stabilize_and_reduce_grad_hidden_activations(
    /**
     * @param update_buffer_LOCAL_reduction_tile Work-group exclusive memory
     *        for staged parallel reduction with interleaved clipping.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Serves dual purpose: (1) holds each thread's pre-accumulation
     *          result (sum of ceil(total_modules_count / get_local_size(0))
     *          elements), then (2) is reused across multiple staged reduction
     *          rounds with interleaved per-element clipping. Thread 0 reads the
     *          final scalar result after the last reduction stage."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa
     *        The pre-gathered, contiguous input data in SoA layout.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count,
     *          src_scalar_NATURAL_padded_total_modules_count)
     *        - Padding Contract: {
     *            dim[0] ("total_batch_count * hidden_count" →
     *              "total_batch_count * padded_hidden_count"):
     *              {Type: CACHE, Formula: "128-byte alignment on
     *              hidden_count stride"},
     *            dim[1] ("total_modules_count" →
     *              "padded_total_modules_count"):
     *              {Type: CACHE, Formula: "128-byte alignment"}
     *          }
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count,
     *          src_scalar_NATURAL_padded_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [(src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count) *
     *          src_scalar_NATURAL_padded_total_modules_count *
     *          sizeof(STORAGE_TYPE)] bytes. This buffer must be fully
     *          populated by Node 13 before dispatch.
     */
    __global const STORAGE_TYPE
        *src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,

    /**
     * @param dest_buffer_GLOBAL_summed_grad_hidden_activations The
     *        destination for the final, summed hidden layer gradient vector.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [(src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count) *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE
        *dest_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_CONST_clipping_threshold_per_stage
     *        Host-prescribed threshold schedule for the staged reduction.
     *        Entry s (0-indexed) is the clip threshold applied after staged
     *        reduction round s. Index 0 is the first clip after
     *        pre-accumulation (leaf); index num_reduction_stages - 1 is the
     *        final clip (root). Each entry is the min of the Quadratic
     *        Scaling Policy threshold and the safety ceiling for that stage,
     *        pre-computed by the Orchestration tier.
     *        - Tensor Shape:
     *          (src_scalar_NATURAL_num_reduction_stages)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_num_reduction_stages]
     *        - Validation Preconditions: [1] Host shall allocate exactly
     *          [src_scalar_NATURAL_num_reduction_stages *
     *          sizeof(COMPUTE_TYPE)] bytes. [2] When
     *          num_reduction_stages == 0, the Host MAY pass a minimal stub
     *          buffer. [3] All entries must be positive real numbers.
     *          [4] The schedule must be monotonically non-increasing
     *          (schedule[0] >= schedule[1] >= ... >= schedule[num-1]),
     *          reflecting the Quadratic Scaling Policy's funnel property.
     */
    __global const COMPUTE_TYPE
        *src_buffer_GLOBAL_CONST_clipping_threshold_per_stage,

    /**
     * @param src_scalar_NATURAL_num_reduction_stages Number of clip stages
     *        in the staged reduction. Zero when total_modules_count <= 1.
     *        - Validation Preconditions: [1] Must be 0 when
     *          total_modules_count <= 1. [2] Must be >= 1 when
     *          total_modules_count > 1.
     */
    uint src_scalar_NATURAL_num_reduction_stages,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_pre Clip threshold for
     *        each thread's pre-accumulation phase. Set to
     *        compute_fp_format_max / K by the Orchestration tier when
     *        pre-accumulation occurs (total_modules_count >
     *        dispatch workgroup size); set to compute_fp_format_max when
     *        each thread loads at most one element.
     *        - Validation Preconditions: Must be a positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_pre,

    /**
     * @param src_scalar_REAL_epsilon Small constant to prevent division by
     *        zero during norm calculation.
     *        - Validation Preconditions: Must be a small, positive real
     *          number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_padded_total_modules_count);

// =====================================================================
// Learn Phase III: Streaming Shared-Layer Backpropagation (Nodes 17–19)
//
// True Streaming model: recompute inputs, compute partial gradients,
// clip immediately, and feed clipped partials into the reduction
// engine — all within a single parametric loop.
// =====================================================================

// --- Node 17: Shared Weight Gradients ---------------------------------

/**
 * @brief (Node 17) Computes partial gradients for shared layer weights from a batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); compute-role gradient consumed directly; partial gradient outputs narrowed via store_storage(). Integer-typed sample_mask accessed via load_sample_mask(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE. ReLU derivative source is controlled by `src_scalar_FLAG_use_explicit_hidden_mask`. When 0: mask is derived internally from stored activations (mask = load_storage(hidden_activations) > 0). When 1: mask is read from `src_buffer_GLOBAL_hidden_mask`. The Host MAY pass a minimal stub buffer when the flag is 0. Sample-Level Early Exit: When load_sample_mask() returns 0 for a sample, the kernel skips the entire contribution for that sample. This is a performance optimization — not a correctness requirement. Both upstream invariants (summed_grad_h = 0 from Node 9's masking, relu_derivative = 0 from Node 4's activation zeroing) independently guarantee zero contribution for masked samples regardless of whether the early exit is applied. SIMD-Major Write Pattern: The kernel writes gradient elements at flat indices corresponding to the SIMD-major (SoA) layout (padded_hidden_count/SIMD_WIDTH, padded_input_count, SIMD_WIDTH), matching the persistent shared weight parameter layout consumed by Node 4 and updated by Node 24. This ensures flat-index correspondence between the gradient and parameter buffers through the layout-agnostic reduction pipeline. Padding Zero-Establishment: For SIMD-major positions corresponding to logical indices input_count <= i < padded_input_count or hidden_count <= h < padded_hidden_count, the kernel SHALL write zero. The kernel is the sole guarantor of zeros at padding positions (Initialization Contract: NOT_REQUIRED). This guarantees that downstream L2 norm computations (Node 19) over the full padded extent are mathematically equivalent to norms over the logical extent."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Streamable. Designed for the 'True Streaming' backpropagation model."
 */
__kernel void backprop_shared_weights_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction of per-sample outer
     *          product contributions (input[i] * grad_h[h] * relu_mask[h])
     *          across the batch chunk to produce a single partial weight
     *          gradient element."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_input The initial, untransformed input data for the batch.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_input_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer (Node 4).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask [CONDITIONAL on src_scalar_FLAG_use_explicit_hidden_mask] The ReLU derivative mask from Node 4.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] This buffer is read from ONLY IF `src_scalar_FLAG_use_explicit_hidden_mask` == 1. [2] If the flag is set, the batch access slice must be within bounds,
     * as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count. [3] If the flag is set, Host must ensure this buffer was
     * allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes. [4] If the flag is not set, the Host MAY pass a minimal stub
     * buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_scalar_FLAG_use_explicit_hidden_mask A flag to select the ReLU derivative source.
     *        - Validation Preconditions: Must be 0 or 1. If 0, mask is derived from stored activations (mask = activation > 0). If 1, mask is read from `src_buffer_GLOBAL_hidden_mask`.
     */
    uint src_scalar_FLAG_use_explicit_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_summed_grad_hidden_activations The final, consolidated upstream gradient from Node 16.
     *        - Tensor Shape: (src_scalar_NATURAL_final_grad_hidden_activations_total_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_final_grad_hidden_activations_total_count]
     *        - Validation Preconditions: The logical shape assumed by this kernel must match the physical size of the provided buffer, as proven by: (src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count) == src_scalar_NATURAL_final_grad_hidden_activations_total_count.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major Single-slot scratch buffer for this chunk's computed weight gradients in SIMD-major layout.
     *        Consumed by Node 19 within the same streaming loop iteration.
     *        - Tensor Shape: (1, src_scalar_NATURAL_padded_hidden_count / SIMD_WIDTH, src_scalar_NATURAL_padded_input_count, SIMD_WIDTH)
     *        - Padding Contract: {
     *            dim[0] ("1"): {Type: UNPADDED},
     *            dim[1] ("hidden_count/SIMD_WIDTH" → "padded_hidden_count/SIMD_WIDTH"): {Type: SIMD, Formula: "SIMD_WIDTH-multiple alignment on hidden_count ensures exact division"},
     *            dim[2] ("input_count" → "padded_input_count"): {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[3] ("SIMD_WIDTH"): {Type: UNPADDED}
     *          }
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count / SIMD_WIDTH, src_scalar_NATURAL_padded_input_count, SIMD_WIDTH]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_padded_input_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_input_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_final_grad_hidden_activations_total_count);

// --- Node 18: Shared Bias Gradients -----------------------------------

/**
 * @brief (Node 18) Computes partial gradients for shared layer biases from a batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); compute-role gradient consumed directly; partial gradient outputs narrowed via store_storage(). Integer-typed sample_mask accessed via load_sample_mask(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE. ReLU derivative source is controlled by `src_scalar_FLAG_use_explicit_hidden_mask`. When 0: mask is derived internally from stored activations (mask = load_storage(hidden_activations) > 0). When 1: mask is read from `src_buffer_GLOBAL_hidden_mask`. The Host MAY pass a minimal stub buffer when the flag is 0. Sample-Level Early Exit: When load_sample_mask() returns 0 for a sample, the kernel skips the entire contribution for that sample. This is a performance optimization — not a correctness requirement. Both upstream invariants (summed_grad_h = 0 from Node 9's masking, relu_derivative = 0 from Node 4's activation zeroing) independently guarantee zero contribution for masked samples regardless of whether the early exit is applied. Padding Zero-Establishment: For the padded_hidden_count dimension, the kernel SHALL write zero for all positions at indices >= hidden_count. The kernel is the sole guarantor of zeros at padding positions (Initialization Contract: NOT_REQUIRED). This guarantees that downstream L2 norm computations (Node 19) over the full padded extent are mathematically equivalent to norms over the logical extent."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Streamable. Designed for the 'True Streaming' backpropagation model."
 */
__kernel void backprop_shared_biases_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction of per-sample bias
     *          gradient contributions (grad_h[h] * relu_mask[h]) across the
     *          batch chunk to produce a single partial bias gradient element."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer (Node 4).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask [CONDITIONAL on src_scalar_FLAG_use_explicit_hidden_mask] The ReLU derivative mask from Node 4.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] This buffer is read from ONLY IF `src_scalar_FLAG_use_explicit_hidden_mask` == 1. [2] If the flag is set, the batch access slice must be within bounds,
     * as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count. [3] If the flag is set, Host must ensure this buffer was
     * allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes. [4] If the flag is not set, the Host MAY pass a minimal stub
     * buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_scalar_FLAG_use_explicit_hidden_mask A flag to select the ReLU derivative source.
     *        - Validation Preconditions: Must be 0 or 1. If 0, mask is derived from stored activations (mask = activation > 0). If 1, mask is read from `src_buffer_GLOBAL_hidden_mask`.
     */
    uint src_scalar_FLAG_use_explicit_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_summed_grad_hidden_activations The final, consolidated upstream gradient from Node 16.
     *        - Tensor Shape: (src_scalar_NATURAL_final_grad_hidden_activations_total_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_final_grad_hidden_activations_total_count]
     *        - Validation Preconditions: The logical shape assumed by this kernel must match the physical size of the provided buffer, as proven by: (src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count) == src_scalar_NATURAL_final_grad_hidden_activations_total_count.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer encoding the validity (1) or padding (0) status of each sample.
     *        Bit i of word j encodes sample (32*j + i), LSB-first. Accessed via load_sample_mask() utility (ADR-031).
     *        - Tensor Shape: (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [ceil(src_scalar_NATURAL_total_batch_count / 32) * sizeof(uint)] bytes.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_shared Single-slot scratch buffer for this chunk's computed bias gradients.
     *        Consumed by Node 19 within the same streaming loop iteration.
     *        - Tensor Shape: (1, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_biases_shared,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_final_grad_hidden_activations_total_count);

// --- Node 19: Shared Gradient Clipping --------------------------------

/**
 * @brief (Node 19) [Utility Kernel] Computes the L2 Norm for a single SHARED GRADIENT
 * chunk, conditionally scales it, and writes the result to a destination memory address
 * provided by the host.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "[1] Implements a two-pass algorithm: Norm calculation followed
 *          by conditional scaling. [2] The L2 norm is computed over the concatenated
 *          vector of both weight and bias gradients for the chunk. Precision Boundary Conversion: storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. FP8 Quantization Note: When STORAGE_TYPE is FP8, the clip-then-store sequence introduces re-quantization error. Gradient components scaled below the FP8 quantization floor (2^-9 for E4M3, 2^-16 for E5M2) may round to zero, effectively zeroing a subset of the gradient. This is an accepted consequence of the Primacy of Memory Strategy — the architecture trades gradient fidelity for 4x bandwidth compression. The Quadratic Scaling Policy's threshold schedule accounts for this by maintaining gradients well above the quantization floor. Padding Zero-Preservation: Identical mathematical property to Node 11. The uniform-scaling algorithm applies a single multiplicative factor to all positions in both output buffers. Zero-valued positions established by Node 17's Padding Zero-Establishment (for weights: indices >= input_count * hidden_count) and Node 18's Padding Zero-Establishment (for biases: indices >= hidden_count) are mapped to zero in the output for all finite scaling factors."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable Utility / Stability Primitive. The responsibility
 *          for calculating the write offset is delegated entirely to the host, making this kernel
 *          a 'dumb' numerical primitive that writes to an explicitly provided memory location."
 */
__kernel void clip_shared_gradients_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group parallel reduction of the sum-of-squares.
     *        - Allocation Formula: get_local_size(0) * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Flat 1D array indexed by local thread ID.
     *          Used for tree-structured parallel reduction to compute a single
     *          L2 norm over the concatenated (weights, biases) gradient vector
     *          for one batch chunk. Each thread accumulates partial sums of
     *          squares for its assigned elements across both sub-buffers, then
     *          the work-group reduces to a single scalar."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_grad_weights_shared_simd_major The partial weight gradients for a single
     *        data chunk, produced by Node 17.
     *        - Tensor Shape: (src_scalar_NATURAL_weights_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_weights_parameter_count]
     *        - Validation Preconditions: Host shall ensure this buffer is a contiguous memory
     *          region containing the complete partial weight gradient for the chunk being processed.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_weights_shared_simd_major,

    /**
     * @param src_buffer_GLOBAL_partial_grad_biases_shared The partial bias gradients for a single
     *        data chunk, produced by Node 18.
     *        - Tensor Shape: (src_scalar_NATURAL_biases_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_biases_parameter_count]
     *        - Validation Preconditions: Host shall ensure this buffer is a contiguous memory
     *          region containing the complete partial bias gradient for the chunk being processed.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_biases_shared,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major The COLLECTION buffer for all clipped
     *        partial weight gradients, ready for consumption by an aggregate_* kernel (Node 20).
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_weights_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_weights_parameter_count]
     *        - Validation Preconditions: [1] The Host is responsible for providing a valid
     *          `out_scalar_NATURAL_weights_write_offset` such that the write operation
     *          remains within the bounds of this collection buffer: (out_scalar_NATURAL_weights_write_offset +
     *          src_scalar_NATURAL_weights_parameter_count) <=
     *          (src_scalar_NATURAL_num_batch_chunks * src_scalar_NATURAL_weights_parameter_count).
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_biases_shared The COLLECTION buffer for all clipped
     *        partial bias gradients, ready for consumption by an aggregate_* kernel (Node 20).
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_biases_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_biases_parameter_count]
     *        - Validation Preconditions: [1] The Host is responsible for providing a valid
     *          `out_scalar_NATURAL_biases_write_offset` such that the write operation
     *          remains within the bounds of this collection buffer: (out_scalar_NATURAL_biases_write_offset +
     *          src_scalar_NATURAL_biases_parameter_count) <=
     *          (src_scalar_NATURAL_num_batch_chunks * src_scalar_NATURAL_biases_parameter_count).
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_biases_shared,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_pre The maximum
     *        permissible L2 norm for this streaming chunk's shared
     *        gradient vector.
     *        - Validation Preconditions: Must be a positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_pre,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent
     *        division by zero during norm calculation.
     *        - Validation Preconditions: Must be a small, positive real
     *          number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    uint        src_scalar_NATURAL_weights_parameter_count,
    uint        src_scalar_NATURAL_biases_parameter_count,
    uint        out_scalar_NATURAL_weights_write_offset,
    uint        out_scalar_NATURAL_biases_write_offset,
    uint        src_scalar_NATURAL_num_batch_chunks);

// =====================================================================
// Learn Phase IV–V: Finalization & Parameter Updates (Nodes 21, 24–25)
//
// Gradient normalization, Adam optimizer update, and temperature
// clamping. Executes after the Batch Synchronization Point.
// =====================================================================

// --- Node 21: Gradient Normalization ----------------------------------

/**
 * @brief (Node 21) [Utility Kernel] Normalizes a buffer of summed gradients by dividing each element by the effective batch size.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a generic, element-wise scaling utility designed to operate on any parameter group's summed gradient buffer."
 *        - Behavioral Invariants: "Performs element-wise division: `output[i] = input[i] / (effective_batch_size + epsilon)`. An epsilon term MUST be used to prevent division by zero if the
 * effective_batch_size is 0. All buffers are compute-role; no precision boundary conversion is required. Padding Zero Propagation (Emergent): Positions in the output buffer corresponding to padding indices carry zero when the input buffer has zero at those positions. The operation output[i] = input[i] / divisor maps zero to zero for all finite positive divisors. Preconditions: the input buffer's zeros at padding positions are guaranteed by the upstream reduction chain (Nodes 11, 15/20) which preserves the zero invariant established by Nodes 8, 9, 17, and 18."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Finalizer Utility / Batch-wide Normalizer. Executes after the reduction engine and before the optimizer update."
 */
__kernel void normalize_gradients(
    /**
     * @param src_buffer_GLOBAL_summed_grad The buffer of aggregated, batch-wide gradients from the reduction engine.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host shall ensure this buffer contains the complete, summed gradients for a parameter group before dispatch.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad,

    /**
     * @param dest_buffer_GLOBAL_final_grad The output buffer containing the normalized, average gradients ready for the optimizer.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_summed_grad`.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_final_grad,

    /**
     * @param src_scalar_REAL_effective_batch_size The normalization factor.
     *        - Calculability Proof: [Host-side calculation: popcount over packed `sample_mask` bitmask words]
     *        - Validation Preconditions: [1] The Host is contractually obligated to calculate this value by performing a popcount over the packed `sample_mask` bitmask (ADR-031 §2.5). [2] The
     * value must be >= 0.
     */
    COMPUTE_TYPE src_scalar_REAL_effective_batch_size,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    /**
     * @param src_scalar_NATURAL_parameter_count The total number of elements in the gradient buffers.
     *        - Calculability Proof: [Dependent on the specific parameter group being processed]
     *        - Validation Preconditions: Must match the element count of the src/dest buffers.
     */
    uint src_scalar_NATURAL_parameter_count);

// --- Node 24: Adam Optimizer Update -----------------------------------

/**
 * @brief (Node 24) Applies Adam optimizer update to an entire parameter group. Single dispatch.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "The implementation is strictly forbidden from using `pown` or any equivalent function. The host is solely responsible for providing pre-computed bias correction terms (`beta1_pow_t`, `beta2_pow_t`) to ensure long-term numerical stability. State-Precision Accumulation: EMA updates on m1 and m2 use ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE). Moment vectors loaded via load_state_for_accum(); gradients widened via widen_to_accum(); EMA arithmetic in ACCUM_TYPE; results stored via store_state_from_accum(). Bias-corrected values and the final parameter update delta are transformative operations using COMPUTE_TYPE (narrowed via narrow_from_accum()). Parameter buffer subtraction is accumulative in ACCUM_TYPE. Hyperparameter Precision Note: Hyperparameter scalars (β₁, β₂, ε, lr) are received in COMPUTE_TYPE and widened to ACCUM_TYPE for EMA arithmetic. The widening preserves only COMPUTE_TYPE precision for these constants. For β₁ = 0.999 with COMPUTE_TYPE = float, the contribution factor (1 − β₁) carries ~7 significant digits regardless of ACCUM_TYPE. Bias Correction Precision Ceiling: The `beta1_pow_t` and `beta2_pow_t` scalars are computed by the Host in FP64 and narrowed to COMPUTE_TYPE at the interface boundary. Under mixed_f32_f64_state() (FP32 compute, FP64 state), beta1^t values below ~1.4e-45 round to FP32 zero, losing FP64 precision. This is currently sound — at t ≈ 100K, 1/(1-beta1^t) ≈ 1.0, so the loss is negligible. A future revision may accept these scalars in STATE_TYPE for full consistency with the state role's unbounded-training-stability guarantee. This interface typing is the last remaining non-state-precision bottleneck in the Adam path; see CONCEPT.md §11 (state-precision accumulation design). ADR-030: State-role buffers are indexed via [parameter_offset + i]. The slice access invariant (parameter_offset + parameter_count) <= total_parameter_count ensures no out-of-bounds access. Padding Zero-Preservation (Inductive): The kernel processes all positions in [parameter_offset, parameter_offset + parameter_count). At padding positions where the gradient is zero and moment vectors are zero (both host-initialized), the Adam recurrence produces zero moment updates and zero parameter change: m1 <- beta1*0 + (1-beta1)*0 = 0, m2 <- beta2*0 + (1-beta2)*0 = 0, delta_w = 0. This preserves padding zeros by mathematical induction over training steps — the kernel is not required to distinguish padding from logical positions. Implementations MAY skip computation at padding indices as a performance optimization; both approaches satisfy Padding Zero-Preservation."
 *        - Idempotency: "Fundamentally Non-Idempotent (Stateful). Modifies multiple state buffers in-place."
 *        - Synchronization Model: "Stateful Optimizer Update. Consumes final gradients after the Batch Synchronization Point."
 */
__kernel void adam_update(
    /**
     * @param src_buffer_GLOBAL_final_grad The buffer containing the final, normalized, batch-averaged gradients from Node 21.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_parameter_count * sizeof(COMPUTE_TYPE)] bytes for this buffer.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_final_grad,

    /**
     * @param update_buffer_GLOBAL_parameters The parameter buffer to be updated in-place (e.g., weights, biases).
     *        - Tensor Shape: (src_scalar_NATURAL_total_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_parameter_count * sizeof(STATE_TYPE)] bytes. [2] The physical memory layout must be identical to
     * `m1` and `m2` buffers. [3] (src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <= src_scalar_NATURAL_total_parameter_count.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_parameters,

    /**
     * @param update_buffer_GLOBAL_m1 The first moment vector buffer to be updated in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_total_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_parameter_count * sizeof(STATE_TYPE)] bytes. [2] The physical memory layout must be identical to other
     * state buffers. [3] (src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <= src_scalar_NATURAL_total_parameter_count.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_m1,

    /**
     * @param update_buffer_GLOBAL_m2 The second moment vector buffer to be updated in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_total_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_parameter_count * sizeof(STATE_TYPE)] bytes. [2] The physical memory layout must be identical to other
     * state buffers. [3] (src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <= src_scalar_NATURAL_total_parameter_count.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_m2,

    COMPUTE_TYPE src_scalar_REAL_learning_rate,
    COMPUTE_TYPE src_scalar_REAL_beta1_pow_t,
    COMPUTE_TYPE src_scalar_REAL_beta2_pow_t,
    COMPUTE_TYPE src_scalar_REAL_beta1,
    COMPUTE_TYPE src_scalar_REAL_beta2,
    COMPUTE_TYPE src_scalar_REAL_epsilon,
    uint        src_scalar_NATURAL_parameter_offset,
    uint        src_scalar_NATURAL_parameter_count,
    uint        src_scalar_NATURAL_total_parameter_count);

// --- Node 25: Temperature Clamping ------------------------------------

/**
 * @brief (Node 25) Clamps temperature parameters within a [min, max] range.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Enforces `temps = clamp(temps, min_value, max_value)` for each element. Precision Boundary Conversion: state-role buffer accessed via load_state()/store_state_update(); clamp arithmetic exclusively in COMPUTE_TYPE. ADR-030: Buffer is indexed via [parameter_offset + i]. The slice access invariant (parameter_offset + parameter_count) <= total_parameter_count ensures no out-of-bounds access."
 *        - Idempotency: "Fundamentally Non-Idempotent (Stateful). Modifies the temps buffer in-place."
 *        - Synchronization Model: "Finalizer Utility"
 */
__kernel void clamp_temperatures(
    /**
     * @param update_buffer_GLOBAL_temps The temperature parameter buffer to be clamped in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_total_parameter_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_parameter_count * sizeof(STATE_TYPE)] bytes for this buffer. [2] (src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <= src_scalar_NATURAL_total_parameter_count.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_temps,

    COMPUTE_TYPE src_scalar_REAL_min_value,
    COMPUTE_TYPE src_scalar_REAL_max_value,
    uint        src_scalar_NATURAL_parameter_offset,
    uint        src_scalar_NATURAL_parameter_count,
    uint        src_scalar_NATURAL_total_parameter_count);

// 
// #####################################################################
// ##                                                                 ##
// ##            EXPERIMENTAL KERNEL DECLARATIONS                     ##
// ##                                                                 ##
// ##  These kernels are provided for experimentation with multi-     ##
// ##  layer shared-layer stacks and branch-point gradient routing.   ##
// ##  They are NOT yet referenced by the canonical DAG node          ##
// ##  numbering, do not have assigned node IDs, and are not          ##
// ##  covered by the existing Validation Scenarios. Integration      ##
// ##  into the execution plan requires Policy-tier plan-             ##
// ##  construction extensions and new ADR(s) formalizing the         ##
// ##  multi-layer backpropagation topology.                          ##
// ##                                                                 ##
// #####################################################################
// 

// --- Transpose SIMD-Major Matrix-Vector (Structural Inverse of Node 4) ---

/**
 * @brief [EXPERIMENTAL] Computes input-dimension gradients via masked
 *        transposed SIMD-major matrix-vector multiplication.
 *        Structural inverse of Node 4 (forward_pass).
 *
 *        Computes:
 *          dest[b][i] = Σ_h W_simd[h][i] × grad_h[b][h] × mask[b][h]
 *
 *        where the sum reduces over the hidden dimension and the output
 *        spans the input dimension — the transpose of Node 4's
 *        input→hidden forward multiply.
 *
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the
 *          parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable"
 *        - Behavioral Invariants: "Precision Boundary Conversion:
 *          state-role weights loaded via load_state(); storage-role
 *          hidden_activations and hidden_mask loaded via load_storage();
 *          compute-role upstream gradient consumed directly; storage-role
 *          output narrowed via store_storage(). All arithmetic
 *          exclusively in COMPUTE_TYPE.
 *
 *          ReLU derivative source controlled by
 *          src_scalar_FLAG_use_explicit_hidden_mask (same pattern as
 *          Nodes 5, 17, 18). When 0: mask derived from stored
 *          activations (mask = load_storage(hidden_activations) > 0).
 *          When 1: mask read from src_buffer_GLOBAL_hidden_mask. Host
 *          MAY pass a minimal stub buffer when flag is 0.
 *
 *          Sample Masking (ADR-031): When load_sample_mask() returns 0
 *          for a sample, all gradient outputs for that sample are set
 *          to 0 and further computation is skipped.
 *
 *          Padding Zero-Establishment: For the padded_input_count
 *          dimension, the kernel SHALL write zero for all positions at
 *          indices >= input_count. The kernel is the sole guarantor of
 *          zeros at padding positions (Initialization Contract: NOT_REQUIRED).
 *          Hidden-dimension padding contributes zero to the reduction
 *          sum because upstream gradient padding positions are zero
 *          (guaranteed by upstream Padding Zero Propagation) and weight
 *          padding positions are zero (Zero-Propagation Theorem
 *          state-role origin); implementations MAY skip computation at
 *          hidden padding indices as a performance optimization."
 *        - Kernel Bifurcation: "CONCEPT.md Principle 3(B) — structurally
 *          incompatible with Node 4 (forward_pass). Node 4 reduces over
 *          the input dimension to produce hidden activations and a ReLU
 *          mask with bias addition; this kernel reduces over the hidden
 *          dimension consuming a mask and an upstream gradient to produce
 *          input-dimension gradients. The reduction axis swap,
 *          opposite-direction data flow, and differing ancillary
 *          operations (ReLU application + bias addition vs. mask-gated
 *          transposed multiply without bias) prevent unification under
 *          a single interface with a FLAG parameter."
 */
__kernel void transpose_matvec_masked_simd_major(
    /**
     * @param update_buffer_LOCAL_simd_tile Local memory for tiled
     *        transposed matrix-vector multiplication.
     *        - Allocation Formula: (SIMD_WIDTH + SIMD_WIDTH * SIMD_WIDTH)
     *          * sizeof(COMPUTE_TYPE)
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Internal Layout Note: "Partitioned into two contiguous
     *          sub-arrays: tile_grad[SIMD_WIDTH] (broadcast tile for the
     *          masked upstream gradient slice, holding grad_h[b][h] *
     *          mask[b][h] for SIMD_WIDTH consecutive hidden indices) at
     *          offset 0, followed by tile_weights[SIMD_WIDTH][SIMD_WIDTH]
     *          (weight matrix tile loaded from the SIMD-major layout,
     *          stride = SIMD_WIDTH) at offset SIMD_WIDTH. The weight tile
     *          is accessed in transposed order relative to Node 4's
     *          forward-pass access pattern: each thread reads across the
     *          input dimension for a fixed hidden-SIMD-lane, accumulating
     *          into an input-dimension output register."
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_simd_tile,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_shared_simd_major The
     *        learnable shared weights in SIMD-friendly layout. Identical
     *        buffer and layout as consumed by Node 4 (forward_pass).
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count
     *          / SIMD_WIDTH, src_scalar_NATURAL_padded_input_count,
     *          SIMD_WIDTH)
     *        - Padding Contract: {
     *            dim[0] ("hidden_count/SIMD_WIDTH" →
     *              "padded_hidden_count/SIMD_WIDTH"):
     *              {Type: SIMD, Formula: "SIMD_WIDTH-multiple alignment
     *              on hidden_count ensures exact division"},
     *            dim[1] ("input_count" → "padded_input_count"):
     *              {Type: CACHE, Formula: "128-byte alignment"},
     *            dim[2] ("SIMD_WIDTH"): {Type: UNPADDED}
     *          }
     *        - Precision Role: "state"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_padded_hidden_count,
     *          src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [src_scalar_NATURAL_padded_hidden_count *
     *          src_scalar_NATURAL_padded_input_count *
     *          sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE
        *src_buffer_GLOBAL_CONST_weights_shared_simd_major,

    /**
     * @param src_buffer_GLOBAL_grad_hidden_activations The upstream
     *        gradient with respect to this layer's hidden activations.
     *        May originate from Node 16 (single-layer), from
     *        elementwise_add (branch-point combination), or from a
     *        higher layer's transpose_matvec_masked_simd_major output
     *        (multi-layer chain).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice
     *          must be within bounds, as proven by:
     *          (src_scalar_NATURAL_batch_chunk_offset +
     *          src_scalar_NATURAL_batch_chunk_count) <=
     *          src_scalar_NATURAL_total_batch_count. [2] Host shall
     *          allocate exactly
     *          [src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global const COMPUTE_TYPE
        *src_buffer_GLOBAL_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate
     *        activations from this layer's forward pass (Node 4 output).
     *        Used for ReLU derivative recomputation when
     *        src_scalar_FLAG_use_explicit_hidden_mask == 0.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula:
     *          "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice
     *          must be within bounds, as proven by:
     *          (src_scalar_NATURAL_batch_chunk_offset +
     *          src_scalar_NATURAL_batch_chunk_count) <=
     *          src_scalar_NATURAL_total_batch_count. [2] Host must
     *          ensure this buffer was allocated to exactly
     *          [src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count *
     *          sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE
        *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask [CONDITIONAL on
     *        src_scalar_FLAG_use_explicit_hidden_mask] The ReLU
     *        derivative mask from Node 4.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula:
     *          "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] This buffer is read from
     *          ONLY IF src_scalar_FLAG_use_explicit_hidden_mask == 1.
     *          [2] If the flag is set, the batch access slice must be
     *          within bounds, as proven by:
     *          (src_scalar_NATURAL_batch_chunk_offset +
     *          src_scalar_NATURAL_batch_chunk_count) <=
     *          src_scalar_NATURAL_total_batch_count. [3] If the flag
     *          is set, Host must ensure this buffer was allocated to
     *          exactly [src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count *
     *          sizeof(STORAGE_TYPE)] bytes. [4] If the flag is not
     *          set, the Host MAY pass a minimal stub buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_sample_mask A packed bitmask buffer
     *        encoding the validity (1) or padding (0) status of each
     *        sample. Bit i of word j encodes sample (32*j + i),
     *        LSB-first. Accessed via load_sample_mask() utility
     *        (ADR-031).
     *        - Tensor Shape:
     *          (ceil(src_scalar_NATURAL_total_batch_count / 32))
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "exempt (integer bitmask)"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice
     *          must be within bounds, as proven by:
     *          (src_scalar_NATURAL_batch_chunk_offset +
     *          src_scalar_NATURAL_batch_chunk_count) <=
     *          src_scalar_NATURAL_total_batch_count. [2] Host shall
     *          allocate exactly
     *          [ceil(src_scalar_NATURAL_total_batch_count / 32) *
     *          sizeof(uint)] bytes.
     */
    __global const uint *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_grad_input The output gradient with
     *        respect to this layer's input. Each batch chunk writes to
     *        its disjoint row-slice; the full buffer is valid for
     *        downstream consumption only after all batch chunks
     *        complete.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula:
     *          "Pad row stride to 128-byte alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: [1] The write slice must be
     *          within bounds, as proven by:
     *          (src_scalar_NATURAL_batch_chunk_offset +
     *          src_scalar_NATURAL_batch_chunk_count) <=
     *          src_scalar_NATURAL_total_batch_count. [2] Host shall
     *          allocate exactly
     *          [src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_input_count *
     *          sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_grad_input,

    /**
     * @param src_scalar_FLAG_use_explicit_hidden_mask A flag to select
     *        the ReLU derivative source.
     *        - Validation Preconditions: Must be 0 or 1. If 0, mask is
     *          derived from stored activations
     *          (mask = load_storage(hidden_activations) > 0). If 1,
     *          mask is read from src_buffer_GLOBAL_hidden_mask.
     */
    uint src_scalar_FLAG_use_explicit_hidden_mask,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_input_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count);

// --- Element-Wise Addition (Branch-Point Gradient Combination) --------

/**
 * @brief [EXPERIMENTAL] Element-wise addition of two compute-role
 *        buffers. Combines gradient contributions at DAG branch points
 *        where classifier heads and deeper layers converge.
 *
 *        Computes:
 *          dest[i] = src_a[i] + src_b[i]  for i in [0, element_count)
 *
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the
 *          parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Utility"
 *        - Behavioral Invariants: "All buffers are compute-role; no
 *          precision boundary conversion is required. All arithmetic
 *          exclusively in COMPUTE_TYPE.
 *          Padding Zero Propagation (Emergent): When both input
 *          buffers carry zero at a position, the output carries zero
 *          at that position. This is a mathematical consequence of
 *          0 + 0 = 0. Precondition: both input buffers must carry
 *          zero at padding positions, guaranteed by their respective
 *          upstream invariant chains."
 */
__kernel void elementwise_add(
    /**
     * @param src_buffer_GLOBAL_input_a The first addend buffer.
     *        - Tensor Shape: (src_scalar_NATURAL_element_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_element_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [src_scalar_NATURAL_element_count *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_input_a,

    /**
     * @param src_buffer_GLOBAL_input_b The second addend buffer.
     *        - Tensor Shape: (src_scalar_NATURAL_element_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_element_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [src_scalar_NATURAL_element_count *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_input_b,

    /**
     * @param dest_buffer_GLOBAL_output The element-wise sum.
     *        - Tensor Shape: (src_scalar_NATURAL_element_count)
     *        - Padding Contract: {Type: UNPADDED}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_element_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [src_scalar_NATURAL_element_count *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_output,

    /**
     * @param src_scalar_NATURAL_element_count Total number of elements
     *        to process.
     *        - Calculability Proof: [Known by Host Orchestrator from
     *          the source buffers' BufferDescriptors — must be
     *          identical across all three buffers]
     *        - Validation Preconditions: Must match the element count
     *          of all three buffers.
     */
    uint src_scalar_NATURAL_element_count);

#endif // KERNELS_CL_H
