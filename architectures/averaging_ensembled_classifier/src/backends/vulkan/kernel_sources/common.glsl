// common.glsl — Shared declarations for all Vulkan compute shaders
//
// Consumed via #include "common.glsl" by every .comp file.
// Provides: specialization constants, subgroup-accelerated workgroup
// reduction helpers, and common type/constant definitions.

#ifndef COMMON_GLSL
#define COMMON_GLSL

// Conditionally enable FP16 extension when a role type requires it.
// Injected by glslc -DENABLE_FP16_EXTENSION=1 when any role is float16_t.
#ifdef ENABLE_FP16_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot     : require

// ── Specialization Constants (Contract Article 6) ──────────────────────
layout(constant_id = 0) const uint SPEC_SIMD_WIDTH            = 8;
layout(constant_id = 1) const uint SPEC_LOCAL_MEM_BANK_PADDING = 1;
layout(constant_id = 2) const uint SPEC_C_TILE_SIZE            = 8;
layout(constant_id = 3) const uint SPEC_PROBLEM_TYPE           = 0; // 0=CCE, 1=BCE

// ── Precision-Role Type Macros (ADR-023 §3.2) ─────────────────────────
// STORAGE_FLOAT and STATE_FLOAT are injected by the build system via
// glslc -D flags. Default to float (FP32) when not injected.
// When STORAGE_FLOAT == float, WIDEN_STORAGE/NARROW_STORAGE are identity
// operations eliminated by the SPIR-V compiler.
#ifndef STORAGE_FLOAT
#define STORAGE_FLOAT float
#endif
#ifndef STATE_FLOAT
#define STATE_FLOAT float
#endif

#define WIDEN_STORAGE(x)   float(x)
#define NARROW_STORAGE(x)  STORAGE_FLOAT(x)
#define WIDEN_STATE(x)     float(x)
#define NARROW_STATE(x)    STATE_FLOAT(x)

// ── Derived Constants ──────────────────────────────────────────────────
const float NUMERICAL_STABILITY_EPSILON = 1e-7;
const uint  PROBLEM_TYPE_CCE            = 0;
const uint  PROBLEM_TYPE_BCE            = 1;
const uint  AGG_MODE_SUM                = 0;
const uint  AGG_MODE_AVERAGE            = 1;

// ── Precision Boundary Helpers (ADR-023 §3.2.4) ───────────────────────
// Value-semantics wrappers that cross the storage/state → compute boundary.
// When STORAGE_FLOAT == float, these are identity operations. The SPIR-V
// compiler eliminates them. One codepath — no #ifdef on type equality.

float read_storage(STORAGE_FLOAT val) { return WIDEN_STORAGE(val); }
STORAGE_FLOAT write_storage(float val) { return NARROW_STORAGE(val); }

float read_state(STATE_FLOAT val) { return WIDEN_STATE(val); }
STATE_FLOAT write_state(float val) { return NARROW_STATE(val); }

// ── Subgroup-Accelerated Workgroup Reduction ───────────────────────────
// Compute-role scratch for cross-subgroup bridge.
// Always float (COMPUTE_TYPE = float on Vulkan backend, invariant per ADR-023 §3.3).
shared float _compute_scratch[32];

// workgroup_reduce_add — returns the sum of `value` across the entire
// workgroup using subgroup intrinsics.
//
// Three-level pattern (VULKAN_BACKEND.md §Subgroup Operations):
//   1. Intra-subgroup: subgroupAdd(value) — single instruction
//   2. Cross-subgroup: elected lane writes to shared memory; barrier
//   3. Final reduction: first subgroup reduces representatives; broadcast
float workgroup_reduce_add(float value) {
    // Level 1: native subgroup reduction
    float subgroup_sum = subgroupAdd(value);

    // Level 2: cross-subgroup bridge via shared memory
    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_sum;
    }
    barrier();

    // Level 3: first subgroup reduces all subgroup representatives
    float total = 0.0;
    if (gl_SubgroupID == 0) {
        float val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : 0.0;
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
float workgroup_reduce_max(float value) {
    float subgroup_max = subgroupMax(value);

    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_max;
    }
    barrier();

    float total = -1.0 / 0.0; // -Inf
    if (gl_SubgroupID == 0) {
        float val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : (-1.0 / 0.0);
        total = subgroupMax(val);
    }

    if (gl_SubgroupID == 0 && subgroupElect()) {
        _compute_scratch[0] = total;
    }
    barrier();
    return _compute_scratch[0];
}

#endif // COMMON_GLSL
