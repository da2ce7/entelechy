// common.glsl — Shared declarations for all Vulkan compute shaders
//
// Consumed via #include "common.glsl" by every .comp file.
// Provides: specialization constants, subgroup-accelerated workgroup
// reduction helpers, and common type/constant definitions.
//
// Three-axis precision model (ADR-024):
//   STORAGE_TYPE — element type in memory (bandwidth lever)
//   COMPUTE_TYPE — element type for arithmetic (fidelity lever)
//   STATE_TYPE   — element type for optimizer state (accumulation lever)

#ifndef COMMON_GLSL
#define COMMON_GLSL

// ── FP8 Compile-Time Flags (ADR-025 §7.2) ─────────────────────────────
// Injected by glslc: -DSTORAGE_TYPE_IS_FP8=1 -DSTORAGE_TYPE_IS_E4M3=1
// (or -DSTORAGE_TYPE_IS_E5M2=1).
// When FP8: -DCOMPUTE_TYPE_IS_HALF=1 or -DCOMPUTE_TYPE_IS_FLOAT=1
// or -DCOMPUTE_TYPE_IS_DOUBLE=1.
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
#ifndef STATE_TYPE_IS_DOUBLE
#define STATE_TYPE_IS_DOUBLE 0
#endif
#ifndef STATE_TYPE_IS_HALF
#define STATE_TYPE_IS_HALF 0
#endif
#ifndef STATE_TYPE_IS_FLOAT
#define STATE_TYPE_IS_FLOAT 0
#endif

// ── Extension Enables ──────────────────────────────────────────────────

// VK_KHR_8bit_storage → GL_EXT_shader_8bit_storage (uint8_t in storage buffers)
// GL_EXT_shader_explicit_arithmetic_types_int8 enables uint8_t in function params/returns
#if STORAGE_TYPE_IS_FP8
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#endif

// Conditionally enable FP16 extension when a role type requires it.
// Injected by glslc -DENABLE_FP16_EXTENSION=1 when any role is float16_t.
#ifdef ENABLE_FP16_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_subgroup_extended_types_float16   : require
#endif

// Conditionally enable FP64 extension when a role type requires it.
// Injected by glslc -DENABLE_FP64_EXTENSION=1 when any role is double.
#ifdef ENABLE_FP64_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot     : require

// ── Specialization Constants (Contract Article 6) ──────────────────────
layout(constant_id = 0) const uint SPEC_SIMD_WIDTH            = 8;
layout(constant_id = 1) const uint SPEC_LOCAL_MEM_BANK_PADDING = 1;
layout(constant_id = 2) const uint SPEC_C_TILE_SIZE            = 8;
layout(constant_id = 3) const uint SPEC_PROBLEM_TYPE           = 0; // 0=CCE, 1=BCE

// ── Precision-Role Type Macros (ADR-024 §5.1) ─────────────────────────
// STORAGE_TYPE, COMPUTE_TYPE, and STATE_TYPE are injected by the build
// system via glslc -D flags. Default to float (FP32) when not injected.
// When FP8, STORAGE_TYPE is uint8_t (no -DSTORAGE_TYPE flag emitted).
// When a role type == float, WIDEN/NARROW are identity operations
// eliminated by the SPIR-V compiler.
#ifndef STORAGE_TYPE
#if STORAGE_TYPE_IS_FP8
#define STORAGE_TYPE uint8_t
#else
#define STORAGE_TYPE float
#endif
#endif
#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE float
#endif
#ifndef STATE_TYPE
#define STATE_TYPE float
#endif

#if !STORAGE_TYPE_IS_FP8
#define WIDEN_STORAGE(x)   COMPUTE_TYPE(x)
#define NARROW_STORAGE(x)  STORAGE_TYPE(x)
#endif
#define WIDEN_STATE(x)     COMPUTE_TYPE(x)
#define NARROW_STATE(x)    STATE_TYPE(x)

// ── Derived Constants ──────────────────────────────────────────────────
// NUMERICAL_EPSILON_VALUE is injected by the build system for FP64
// compute variants (-DNUMERICAL_EPSILON_VALUE=1e-15). Defaults to
// FP32-appropriate 1e-7.
#ifndef NUMERICAL_EPSILON_VALUE
#define NUMERICAL_EPSILON_VALUE 1e-7
#endif
const COMPUTE_TYPE NUMERICAL_STABILITY_EPSILON = COMPUTE_TYPE(NUMERICAL_EPSILON_VALUE);

const uint  PROBLEM_TYPE_CCE            = 0;
const uint  PROBLEM_TYPE_BCE            = 1;
const uint  AGG_MODE_SUM                = 0;
const uint  AGG_MODE_AVERAGE            = 1;

// ── FP8 Conversion Functions (ADR-025 §7.3) ───────────────────────────
// Software FP8↔float conversion via arithmetic bit manipulation.
// Unlike the OpenCL/CPU backends (which use 256-entry LUTs), Vulkan uses
// arithmetic conversion: GLSL lacks __constant memory, and ~12-15 ALU ops
// are negligible for bandwidth-bound workloads.
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
    uint sign = (bits >> 7) & 1u;

#if STORAGE_TYPE_IS_E4M3
    uint exp8 = (bits >> 3) & 0xFu;   // 4-bit exponent
    uint mant = bits & 0x7u;           // 3-bit mantissa

    float result;
    if (exp8 == 0u) {
        // Subnormal: value = mant * 2^(1-bias-mantissa_bits) = mant * 2^(-9)
        result = float(mant) * 0.001953125;  // 2^-9, exact constant
    } else {
        // Normal: value = (1 + mant/8) * 2^(exp8-bias)
        float m = 1.0 + float(mant) * 0.125;  // mant/8
        result = m * exp2(float(int(exp8) - FP8_BIAS));
    }

#elif STORAGE_TYPE_IS_E5M2
    uint exp8 = (bits >> 2) & 0x1Fu;  // 5-bit exponent
    uint mant = bits & 0x3u;           // 2-bit mantissa

    float result;
    if (exp8 == 0x1Fu) {
        // IEEE special exponent: inf (mant==0) or NaN (mant!=0)
        if (mant != 0u) {
            // NaN → zero: Returns BEFORE sign application.
            return 0.0;
        }
        // ±inf → ±MAX: Saturate to max finite E5M2 value.
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
// NaN handling: NaN input → zero for both E4M3 and E5M2.
uint store_storage_fp8_internal(float val) {
    if (isnan(val)) return 0u;

    uint sign = val < 0.0 ? 1u : 0u;
    val = abs(val);

    // Saturation: >= is intentional (FP8_MAX_VAL is exactly representable).
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
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            return sign << 7;
        }
        mant32 = (mant32 | 0x800000u) >> shift;
        exp8 = 0;
    } else if (exp8 > 15) {
        // exp8=15 is valid in E4M3fn (not reserved for inf/NaN); only 16+ overflows.
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
        if (exp8 > 15) return (sign << 7) | 0x7Eu;
    }

    // Guard against encoding NaN (exp=15, mant=7 = 0x7F/0xFF).
    // The saturation check (val >= 448.0) prevents reaching mant=7 at exp=15,
    // but belt-and-suspenders defense: clamp to max finite if it occurs.
    if (exp8 == 15 && mant8 >= 7u) return (sign << 7) | 0x7Eu;

    return (sign << 7) | (uint(exp8) << 3) | (mant8 & 0x7u);

#elif STORAGE_TYPE_IS_E5M2
    if (exp8 <= 0) {
        int shift = 1 - exp8;
        if (shift >= 24) {
            return sign << 7;
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
#if COMPUTE_TYPE_IS_HALF
uint store_storage_fp8(float16_t val) {
    return store_storage_fp8_internal(float(val));
}
#elif COMPUTE_TYPE_IS_DOUBLE
uint store_storage_fp8(double val) {
    // NaN check BEFORE clamp: GLSL clamp(NaN, lo, hi) returns lo per spec.
    if (isnan(val)) return 0u;
    // Defense-in-depth: clamp before double→float narrowing to prevent UB.
    double clamped = clamp(val, -double(3.4028235e+38), double(3.4028235e+38));
    return store_storage_fp8_internal(float(clamped));
}
#else  // FP32 compute (default)
uint store_storage_fp8(float val) {
    return store_storage_fp8_internal(val);
}
#endif

#endif // STORAGE_TYPE_IS_FP8

// ── Precision Boundary Helpers (ADR-024 §5.1) ─────────────────────────
// Value-semantics wrappers that cross the storage/state → compute boundary.
// When a role type matches COMPUTE_TYPE, these are identity operations.
// The SPIR-V compiler eliminates them.
// FP8 overrides route through load_storage_fp8/store_storage_fp8.

#if STORAGE_TYPE_IS_FP8
COMPUTE_TYPE read_storage(uint8_t val) { return load_storage_fp8(uint(val)); }
uint8_t write_storage(COMPUTE_TYPE val) { return uint8_t(store_storage_fp8(val)); }
#else
COMPUTE_TYPE read_storage(STORAGE_TYPE val) { return WIDEN_STORAGE(val); }
STORAGE_TYPE write_storage(COMPUTE_TYPE val) { return NARROW_STORAGE(val); }
#endif

COMPUTE_TYPE read_state(STATE_TYPE val) { return WIDEN_STATE(val); }
STATE_TYPE write_state(COMPUTE_TYPE val) { return NARROW_STATE(val); }

// ── Accumulation Precision Type (ADR-027) ─────────────────────────────
// ACCUM_FLOAT = max(COMPUTE_TYPE, STATE_TYPE). When STATE_TYPE > COMPUTE_TYPE
// (e.g., FP64 state + FP32 compute), EMA arithmetic uses STATE_TYPE precision
// to prevent erosion over unbounded training steps.
//
// When STATE_TYPE <= COMPUTE_TYPE, ACCUM_FLOAT == COMPUTE_TYPE and all casts
// are identity operations eliminated by the SPIR-V compiler.

#if STATE_TYPE_IS_DOUBLE && !COMPUTE_TYPE_IS_DOUBLE
    // STATE_TYPE (double) > COMPUTE_TYPE (float or half)
    #define ACCUM_FLOAT double
    #define ACCUM_IS_WIDER 1
#elif !STATE_TYPE_IS_HALF && !STATE_TYPE_IS_DOUBLE && COMPUTE_TYPE_IS_HALF
    // STATE_TYPE (float) > COMPUTE_TYPE (half) — float by exclusion
    #define ACCUM_FLOAT float
    #define ACCUM_IS_WIDER 1
#else
    // STATE_TYPE <= COMPUTE_TYPE (standard case)
    #define ACCUM_FLOAT COMPUTE_TYPE
    #define ACCUM_IS_WIDER 0
#endif

// Accumulation abstractions (value-semantics wrappers for state-precision accumulation)
#define LOAD_STATE_FOR_ACCUM(val) ACCUM_FLOAT(val)
#define STORE_STATE_FROM_ACCUM(val) STATE_TYPE(val)
#define WIDEN_TO_ACCUM(val) ACCUM_FLOAT(val)
#define NARROW_FROM_ACCUM(val) COMPUTE_TYPE(val)

// ── Transcendental Wrappers (ADR-024 §5.1) ────────────────────────────
// GLSL exp/log only accept float.  We always round-trip through float:
// identity for COMPUTE_TYPE=float, narrowing for float16_t, widening
// for double.  Accumulation surrounding these calls still benefits from
// the full compute precision; only the transcendental itself is FP32.
COMPUTE_TYPE COMPUTE_EXP(COMPUTE_TYPE x) { return COMPUTE_TYPE(exp(float(x))); }
COMPUTE_TYPE COMPUTE_LOG(COMPUTE_TYPE x) { return COMPUTE_TYPE(log(float(x))); }

// ── Subgroup-Accelerated Workgroup Reduction ───────────────────────────
// Compute-role scratch for cross-subgroup bridge.
// Type is COMPUTE_TYPE, parameterized by the active precision configuration.
shared COMPUTE_TYPE _compute_scratch[32];

// workgroup_reduce_add — returns the sum of `value` across the entire
// workgroup using subgroup intrinsics.
//
// Three-level pattern (VULKAN_BACKEND.md §Subgroup Operations):
//   1. Intra-subgroup: subgroupAdd(value) — single instruction
//   2. Cross-subgroup: elected lane writes to shared memory; barrier
//   3. Final reduction: first subgroup reduces representatives; broadcast
COMPUTE_TYPE workgroup_reduce_add(COMPUTE_TYPE value) {
    // Level 1: native subgroup reduction
    COMPUTE_TYPE subgroup_sum = subgroupAdd(value);

    // Level 2: cross-subgroup bridge via shared memory
    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_sum;
    }
    barrier();

    // Level 3: first subgroup reduces all subgroup representatives
    COMPUTE_TYPE total = COMPUTE_TYPE(0.0);
    if (gl_SubgroupID == 0) {
        COMPUTE_TYPE val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : COMPUTE_TYPE(0.0);
        total = subgroupAdd(val);
    }

    // Broadcast the final result to all invocations
    if (gl_SubgroupID == 0 && subgroupElect()) {
        _compute_scratch[0] = total;
    }
    barrier();
    return _compute_scratch[0];
}

// workgroup_reduce_max — returns the maximum of `value` across the entire
// workgroup.  Same three-level pattern with subgroupMax.
COMPUTE_TYPE workgroup_reduce_max(COMPUTE_TYPE value) {
    COMPUTE_TYPE subgroup_max = subgroupMax(value);

    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_max;
    }
    barrier();

    COMPUTE_TYPE total = COMPUTE_TYPE(-1.0 / 0.0); // -Inf
    if (gl_SubgroupID == 0) {
        COMPUTE_TYPE val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : COMPUTE_TYPE(-1.0 / 0.0);
        total = subgroupMax(val);
    }

    if (gl_SubgroupID == 0 && subgroupElect()) {
        _compute_scratch[0] = total;
    }
    barrier();
    return _compute_scratch[0];
}

#endif // COMMON_GLSL
