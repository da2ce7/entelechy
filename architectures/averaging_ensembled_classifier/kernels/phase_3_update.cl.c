// phase_3_update.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: normalize_gradients (Node 21) ---
// Strategy: A simple and highly efficient "map" kernel. Each work-item is assigned
// to normalize exactly one element of the summed gradient buffer. Its architectural
// role is critical: it converts the batch-wide gradient *sum* from the reduction
// engine into a true *average* gradient. This ensures that the learning dynamics
// are independent of the batch size, a fundamental requirement for stable and
// reproducible training.
__kernel void normalize_gradients(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_final_grad,
    COMPUTE_TYPE                 src_scalar_REAL_effective_batch_size,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_parameter_count) {

    // --- 1. Work-Item to Element Mapping ---
    // A 1D dispatch where each thread operates on one gradient component. This is an
    // "embarrassingly parallel" problem, allowing for maximum GPU throughput.
    const uint i = get_global_id(0);

    // Standard boundary check.
    if (i >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // --- 2. Normalization Calculation ---
    // Pre-calculate the reciprocal of the divisor. Multiplication is often faster
    // than division on GPU hardware. The epsilon term prevents division by zero
    // if the effective batch size is 0 (e.g., all samples were masked).
    const COMPUTE_TYPE normalizer = 1.0f / (src_scalar_REAL_effective_batch_size + src_scalar_REAL_epsilon);

    // Apply the normalization.
    dest_buffer_GLOBAL_final_grad[i] = src_buffer_GLOBAL_summed_grad[i] * normalizer;
}

// --- Implementation: adam_update (Node 24) ---
// Strategy: A stateful, embarrassingly parallel "map" kernel. Each work-item
// updates a single parameter and its corresponding moment vectors.
//
// Key behavioral contract:
// - State-Precision Accumulation: EMA updates and parameter subtraction in
//   ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)
// - Bias correction and parameter delta computation in COMPUTE_TYPE
// - Host provides pre-computed beta powers for numerical stability
//
// ADR-030: State-role buffers are indexed via [parameter_offset + i].
// The slice access invariant ensures no out-of-bounds access.
//
// When ACCUM_TYPE == COMPUTE_TYPE (the common case), all widen/narrow casts
// are identity operations eliminated by the compiler — zero overhead.
__kernel void adam_update(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_final_grad,
    __global STATE_TYPE         *update_buffer_GLOBAL_parameters,
    __global STATE_TYPE         *update_buffer_GLOBAL_m1,
    __global STATE_TYPE         *update_buffer_GLOBAL_m2,
    COMPUTE_TYPE                 src_scalar_REAL_learning_rate,
    COMPUTE_TYPE                 src_scalar_REAL_beta1_pow_t,
    COMPUTE_TYPE                 src_scalar_REAL_beta2_pow_t,
    COMPUTE_TYPE                 src_scalar_REAL_beta1,
    COMPUTE_TYPE                 src_scalar_REAL_beta2,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_parameter_offset,
    uint                         src_scalar_NATURAL_parameter_count,
    uint                         src_scalar_NATURAL_total_parameter_count) {

    // --- 1. Work-Item to Parameter Mapping ---
    const uint i = get_global_id(0);
    if (i >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // ADR-030: Compute the actual index into state-role buffers
    const uint state_idx = src_scalar_NATURAL_parameter_offset + i;

    // --- 2. Load Inputs ---
    // Gradient in COMPUTE_TYPE (precision role: compute) — zero-indexed per-dispatch
    const COMPUTE_TYPE g = src_buffer_GLOBAL_final_grad[i];

    // State-Precision Accumulation: load moments at full state precision
    // When ACCUM_TYPE > COMPUTE_TYPE, preserves FP64 fidelity
    const ACCUM_TYPE m_prev = load_state_for_accum(update_buffer_GLOBAL_m1, state_idx);
    const ACCUM_TYPE v_prev = load_state_for_accum(update_buffer_GLOBAL_m2, state_idx);

    // --- 3. EMA Updates in ACCUM_TYPE (preserves state precision) ---
    // Widen gradient and hyperparameters to accumulation precision
    const ACCUM_TYPE g_accum     = widen_to_accum(g);
    const ACCUM_TYPE beta1_accum = widen_to_accum(src_scalar_REAL_beta1);
    const ACCUM_TYPE beta2_accum = widen_to_accum(src_scalar_REAL_beta2);

    // First moment: m_new = β₁ · m_prev + (1 - β₁) · g
    const ACCUM_TYPE m_new = beta1_accum * m_prev +
                             (ACCUM_ONE - beta1_accum) * g_accum;

    // Second moment: v_new = β₂ · v_prev + (1 - β₂) · g²
    const ACCUM_TYPE v_new = beta2_accum * v_prev +
                             (ACCUM_ONE - beta2_accum) * (g_accum * g_accum);

    // Store updated moments at state precision
    store_state_from_accum(update_buffer_GLOBAL_m1, state_idx, m_new);
    store_state_from_accum(update_buffer_GLOBAL_m2, state_idx, v_new);

    // --- 4. Bias Correction in COMPUTE_TYPE ---
    // Transformative operations — bounded by compute precision, not state
    const COMPUTE_TYPE m_hat = narrow_from_accum(m_new) /
                               (COMPUTE_ONE - src_scalar_REAL_beta1_pow_t);
    const COMPUTE_TYPE v_hat = narrow_from_accum(v_new) /
                               (COMPUTE_ONE - src_scalar_REAL_beta2_pow_t);

    // --- 5. Parameter Update in ACCUM_TYPE (accumulative) ---
    // The delta is transformative (computed fresh each step), but the subtraction
    // p -= delta is accumulative — p refines over unbounded training steps.
    const COMPUTE_TYPE param_delta = src_scalar_REAL_learning_rate * m_hat /
                                     (MATH_FN sqrt(v_hat) + src_scalar_REAL_epsilon);

    const ACCUM_TYPE current_param = load_state_for_accum(update_buffer_GLOBAL_parameters, state_idx);
    store_state_from_accum(update_buffer_GLOBAL_parameters, state_idx,
                           current_param - widen_to_accum(param_delta));
}

// --- Implementation: clamp_temperatures (Node 25) ---
// Strategy: An embarrassingly parallel "map" kernel. This is the simplest and most
// efficient parallel pattern, as each work-item operates on a single temperature
// parameter independently, with no need for communication or synchronization.
// Its architectural role is that of a "parameter governor," applying a final,
// domain-specific constraint to ensure the learnable temperatures remain in a
// stable and meaningful range.
//
// ADR-030: Buffer is indexed via [parameter_offset + i].
__kernel void clamp_temperatures(
    __global STATE_TYPE *update_buffer_GLOBAL_temps,
    COMPUTE_TYPE         src_scalar_REAL_min_value,
    COMPUTE_TYPE         src_scalar_REAL_max_value,
    uint                 src_scalar_NATURAL_parameter_offset,
    uint                 src_scalar_NATURAL_parameter_count,
    uint                 src_scalar_NATURAL_total_parameter_count) {

    // --- 1. Work-Item to Parameter Mapping ---
    // A 1D dispatch where each thread operates on one temperature parameter.
    const uint idx = get_global_id(0);

    // Standard boundary check.
    if (idx >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // ADR-030: Compute the actual index into the state buffer
    const uint state_idx = src_scalar_NATURAL_parameter_offset + idx;

    // --- 2. In-Place Clamping Operation ---
    // This single operation enforces the physical constraints on the temperature
    // parameter. It prevents the value from becoming negative or excessively large,
    // which could lead to numerical instability in the Softmax/Sigmoid functions.
    // The `clamp` intrinsic is a highly optimized, standard OpenCL function.
    const COMPUTE_TYPE current_val = load_state(update_buffer_GLOBAL_temps, state_idx);
    store_state_update(update_buffer_GLOBAL_temps, state_idx, clamp(current_val, src_scalar_REAL_min_value, src_scalar_REAL_max_value));
}
