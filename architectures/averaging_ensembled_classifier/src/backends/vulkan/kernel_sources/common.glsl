// common.glsl — Shared declarations for all Vulkan compute shaders
//
// Consumed via #include "common.glsl" by every .comp file.
// Provides: specialization constants, subgroup-accelerated workgroup
// reduction helpers, and common type/constant definitions.

#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot     : require

// ── Specialization Constants (Contract Article 6) ──────────────────────
layout(constant_id = 0) const uint SPEC_SIMD_WIDTH            = 8;
layout(constant_id = 1) const uint SPEC_LOCAL_MEM_BANK_PADDING = 1;
layout(constant_id = 2) const uint SPEC_C_TILE_SIZE            = 8;
layout(constant_id = 3) const uint SPEC_PROBLEM_TYPE           = 0; // 0=CCE, 1=BCE

// ── Derived Constants ──────────────────────────────────────────────────
const float NUMERICAL_STABILITY_EPSILON = 1e-7;
const uint  PROBLEM_TYPE_CCE            = 0;
const uint  PROBLEM_TYPE_BCE            = 1;
const uint  AGG_MODE_SUM                = 0;
const uint  AGG_MODE_AVERAGE            = 1;

// ── Subgroup-Accelerated Workgroup Reduction ───────────────────────────
// Scratch array for cross-subgroup bridge.  32 is the maximum number of
// subgroups per workgroup on any current Vulkan implementation.
shared float _cross_subgroup_scratch[32];

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
        _cross_subgroup_scratch[gl_SubgroupID] = subgroup_sum;
    }
    barrier();

    // Level 3: first subgroup reduces all subgroup representatives
    float total = 0.0;
    if (gl_SubgroupID == 0) {
        float val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _cross_subgroup_scratch[gl_SubgroupInvocationID] : 0.0;
        total = subgroupAdd(val);
    }

    // Broadcast the final result to all invocations
    if (gl_SubgroupID == 0 && subgroupElect()) {
        _cross_subgroup_scratch[0] = total;
    }
    barrier();
    return _cross_subgroup_scratch[0];
}

// workgroup_reduce_max — returns the maximum of `value` across the entire
// workgroup.  Same three-level pattern with subgroupMax.
float workgroup_reduce_max(float value) {
    float subgroup_max = subgroupMax(value);

    if (subgroupElect()) {
        _cross_subgroup_scratch[gl_SubgroupID] = subgroup_max;
    }
    barrier();

    float total = -1.0 / 0.0; // -Inf
    if (gl_SubgroupID == 0) {
        float val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _cross_subgroup_scratch[gl_SubgroupInvocationID] : (-1.0 / 0.0);
        total = subgroupMax(val);
    }

    if (gl_SubgroupID == 0 && subgroupElect()) {
        _cross_subgroup_scratch[0] = total;
    }
    barrier();
    return _cross_subgroup_scratch[0];
}

#endif // COMMON_GLSL
