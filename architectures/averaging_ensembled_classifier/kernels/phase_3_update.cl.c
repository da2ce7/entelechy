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
    __global const SCALAR_TYPE *src_buffer_GLOBAL_summed_grad,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_final_grad,
    SCALAR_TYPE                 src_scalar_REAL_effective_batch_size,
    SCALAR_TYPE                 src_scalar_REAL_epsilon,
    uint                        src_scalar_NATURAL_parameter_count) {

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
    const SCALAR_TYPE normalizer = 1.0f / (src_scalar_REAL_effective_batch_size + src_scalar_REAL_epsilon);

    // Apply the normalization.
    dest_buffer_GLOBAL_final_grad[i] = src_buffer_GLOBAL_summed_grad[i] * normalizer;
}

// --- Implementation: adam_update (Node 24) ---
// Strategy: A stateful, embarrassingly parallel "map" kernel. Each work-item is assigned
// to update a single parameter and its corresponding moment vectors. Its most critical
// feature is its strict adherence to the behavioral contract forbidding on-device power
// calculations. By accepting pre-computed bias correction terms from the host, this kernel
// guarantees long-term numerical stability for training runs of any length.
__kernel void adam_update(
    __global const SCALAR_TYPE *src_buffer_GLOBAL_final_grad,
    __global SCALAR_TYPE       *update_buffer_GLOBAL_parameters,
    __global SCALAR_TYPE       *update_buffer_GLOBAL_m1,
    __global SCALAR_TYPE       *update_buffer_GLOBAL_m2,
    SCALAR_TYPE                 src_scalar_REAL_learning_rate,
    SCALAR_TYPE                 src_scalar_REAL_beta1_pow_t,
    SCALAR_TYPE                 src_scalar_REAL_beta2_pow_t,
    SCALAR_TYPE                 src_scalar_REAL_beta1,
    SCALAR_TYPE                 src_scalar_REAL_beta2,
    SCALAR_TYPE                 src_scalar_REAL_epsilon,
    uint                        src_scalar_NATURAL_parameter_count) {

    // --- 1. Work-Item to Parameter Mapping ---
    // A 1D dispatch where each thread operates on one parameter. This is the most
    // efficient parallelization strategy for this independent operation.
    const uint i = get_global_id(0);
    if (i >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // --- 2. Load Current State ---
    const SCALAR_TYPE g      = src_buffer_GLOBAL_final_grad[i];
    const SCALAR_TYPE m_prev = update_buffer_GLOBAL_m1[i];
    const SCALAR_TYPE v_prev = update_buffer_GLOBAL_m2[i];

    // --- 3. Update Biased Moment Estimates ---
    // Update the first moment (moving average of the gradients).
    const SCALAR_TYPE m_new = src_scalar_REAL_beta1 * m_prev + (1.0f - src_scalar_REAL_beta1) * g;
    // Update the second moment (moving average of the squared gradients).
    const SCALAR_TYPE v_new = src_scalar_REAL_beta2 * v_prev + (1.0f - src_scalar_REAL_beta2) * (g * g);

    // --- 4. Compute Bias-Corrected Estimates ---
    // The kernel performs the final division using pre-computed powers of beta.
    // This offloads the sensitive `beta**t` calculation to the host, which can use
    // high-precision arithmetic to prevent underflow, thus guaranteeing stability.
    const SCALAR_TYPE m_hat = m_new / (1.0f - src_scalar_REAL_beta1_pow_t);
    const SCALAR_TYPE v_hat = v_new / (1.0f - src_scalar_REAL_beta2_pow_t);

    // --- 5. Compute Final Parameter Update ---
    const SCALAR_TYPE param_update = src_scalar_REAL_learning_rate * m_hat / (MATH_FN sqrt(v_hat) + src_scalar_REAL_epsilon);

    // --- 6. Atomically Apply Updates ---
    // Update the parameter and its corresponding moment vectors in-place.
    update_buffer_GLOBAL_parameters[i] -= param_update;
    update_buffer_GLOBAL_m1[i] = m_new;
    update_buffer_GLOBAL_m2[i] = v_new;
}

// --- Implementation: clamp_temperatures (Node 25) ---
// Strategy: An embarrassingly parallel "map" kernel. This is the simplest and most
// efficient parallel pattern, as each work-item operates on a single temperature
// parameter independently, with no need for communication or synchronization.
// Its architectural role is that of a "parameter governor," applying a final,
// domain-specific constraint to ensure the learnable temperatures remain in a
// stable and meaningful range.
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *update_buffer_GLOBAL_temps,
    SCALAR_TYPE src_scalar_REAL_min_value,
    SCALAR_TYPE src_scalar_REAL_max_value,
    uint src_scalar_NATURAL_total_modules_count) {

    // --- 1. Work-Item to Parameter Mapping ---
    // A 1D dispatch where each thread operates on one temperature parameter.
    const uint idx = get_global_id(0);

    // Standard boundary check.
    if (idx >= src_scalar_NATURAL_total_modules_count) {
        return;
    }

    // --- 2. In-Place Clamping Operation ---
    // This single operation enforces the physical constraints on the temperature
    // parameter. It prevents the value from becoming negative or excessively large,
    // which could lead to numerical instability in the Softmax/Sigmoid functions.
    // The `clamp` intrinsic is a highly optimized, standard OpenCL function.
    update_buffer_GLOBAL_temps[idx] = clamp(
        update_buffer_GLOBAL_temps[idx],
        src_scalar_REAL_min_value,
        src_scalar_REAL_max_value);
}
