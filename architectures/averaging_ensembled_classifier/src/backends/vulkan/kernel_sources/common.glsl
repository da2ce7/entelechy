// common.glsl — Shared declarations for all Vulkan compute shaders
//
// Consumed via #include "common.glsl" by every .comp file.
// Provides: specialization constants, subgroup-accelerated workgroup
// reduction helpers, and common type/constant definitions.
//
// Three-axis precision model (ADR-024):
//   STORAGE_FLOAT — element type in memory (bandwidth lever)
//   COMPUTE_FLOAT — element type for arithmetic (fidelity lever)
//   STATE_FLOAT   — element type for optimizer state (accumulation lever)

#ifndef COMMON_GLSL
#define COMMON_GLSL

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
// STORAGE_FLOAT, COMPUTE_FLOAT, and STATE_FLOAT are injected by the build
// system via glslc -D flags. Default to float (FP32) when not injected.
// When a role type == float, WIDEN/NARROW are identity operations
// eliminated by the SPIR-V compiler.
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

// ── Derived Constants ──────────────────────────────────────────────────
// NUMERICAL_EPSILON_VALUE is injected by the build system for FP64
// compute variants (-DNUMERICAL_EPSILON_VALUE=1e-15). Defaults to
// FP32-appropriate 1e-7.
#ifndef NUMERICAL_EPSILON_VALUE
#define NUMERICAL_EPSILON_VALUE 1e-7
#endif
const COMPUTE_FLOAT NUMERICAL_STABILITY_EPSILON = COMPUTE_FLOAT(NUMERICAL_EPSILON_VALUE);

const uint  PROBLEM_TYPE_CCE            = 0;
const uint  PROBLEM_TYPE_BCE            = 1;
const uint  AGG_MODE_SUM                = 0;
const uint  AGG_MODE_AVERAGE            = 1;

// ── Precision Boundary Helpers (ADR-024 §5.1) ─────────────────────────
// Value-semantics wrappers that cross the storage/state → compute boundary.
// When a role type matches COMPUTE_FLOAT, these are identity operations.
// The SPIR-V compiler eliminates them.

COMPUTE_FLOAT read_storage(STORAGE_FLOAT val) { return WIDEN_STORAGE(val); }
STORAGE_FLOAT write_storage(COMPUTE_FLOAT val) { return NARROW_STORAGE(val); }

COMPUTE_FLOAT read_state(STATE_FLOAT val) { return WIDEN_STATE(val); }
STATE_FLOAT write_state(COMPUTE_FLOAT val) { return NARROW_STATE(val); }

// ── Transcendental Wrappers (ADR-024 §5.1) ────────────────────────────
// GLSL exp/log only accept float.  We always round-trip through float:
// identity for COMPUTE_FLOAT=float, narrowing for float16_t, widening
// for double.  Accumulation surrounding these calls still benefits from
// the full compute precision; only the transcendental itself is FP32.
COMPUTE_FLOAT COMPUTE_EXP(COMPUTE_FLOAT x) { return COMPUTE_FLOAT(exp(float(x))); }
COMPUTE_FLOAT COMPUTE_LOG(COMPUTE_FLOAT x) { return COMPUTE_FLOAT(log(float(x))); }

// ── Subgroup-Accelerated Workgroup Reduction ───────────────────────────
// Compute-role scratch for cross-subgroup bridge.
// Type is COMPUTE_FLOAT, parameterized by the active precision configuration.
shared COMPUTE_FLOAT _compute_scratch[32];

// workgroup_reduce_add — returns the sum of `value` across the entire
// workgroup using subgroup intrinsics.
//
// Three-level pattern (VULKAN_BACKEND.md §Subgroup Operations):
//   1. Intra-subgroup: subgroupAdd(value) — single instruction
//   2. Cross-subgroup: elected lane writes to shared memory; barrier
//   3. Final reduction: first subgroup reduces representatives; broadcast
COMPUTE_FLOAT workgroup_reduce_add(COMPUTE_FLOAT value) {
    // Level 1: native subgroup reduction
    COMPUTE_FLOAT subgroup_sum = subgroupAdd(value);

    // Level 2: cross-subgroup bridge via shared memory
    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_sum;
    }
    barrier();

    // Level 3: first subgroup reduces all subgroup representatives
    COMPUTE_FLOAT total = COMPUTE_FLOAT(0.0);
    if (gl_SubgroupID == 0) {
        COMPUTE_FLOAT val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : COMPUTE_FLOAT(0.0);
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
COMPUTE_FLOAT workgroup_reduce_max(COMPUTE_FLOAT value) {
    COMPUTE_FLOAT subgroup_max = subgroupMax(value);

    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_max;
    }
    barrier();

    COMPUTE_FLOAT total = COMPUTE_FLOAT(-1.0 / 0.0); // -Inf
    if (gl_SubgroupID == 0) {
        COMPUTE_FLOAT val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : COMPUTE_FLOAT(-1.0 / 0.0);
        total = subgroupMax(val);
    }

    if (gl_SubgroupID == 0 && subgroupElect()) {
        _compute_scratch[0] = total;
    }
    barrier();
    return _compute_scratch[0];
}

#endif // COMMON_GLSL
