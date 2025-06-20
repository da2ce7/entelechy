// parameter_optim.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 12) Implements the Adam optimization step as an embarrassingly parallel map kernel.
 *
 * This kernel is generic and can be applied to any flattened parameter buffer. Each work-item is
 * assigned to a single parameter and is completely independent.
 *
 * It supports sliced updates via `param_offset` and `num_params_to_update`, making it suitable for
 * both streaming updates (e.g., exit parameters inside the chunk loop) and global updates
 * (e.g., shared parameters after aggregation).
 */
__kernel void adam_update(
    // Inputs (Read-Only)
    __global const SCALAR_TYPE *__restrict grad,
    SCALAR_TYPE beta1,
    SCALAR_TYPE beta2,
    SCALAR_TYPE beta1_t,
    SCALAR_TYPE beta2_t,
    SCALAR_TYPE learning_rate,
    SCALAR_TYPE epsilon,

    // Inputs/Outputs (Read and Write)
    __global SCALAR_TYPE *__restrict param,
    __global SCALAR_TYPE *__restrict m1,
    __global SCALAR_TYPE *__restrict m2,

    // Dimensions
    int param_offset,
    int num_params_to_update) {

    // `local_idx` is the index within the *current slice* of work (0 to num_params_to_update-1).
    const int local_idx = get_global_id(0);

    // Bounds check for the current dispatch.
    if (local_idx >= num_params_to_update) {
        return;
    }

    // `global_idx` is the absolute index into the full parameter buffers, calculated using the offset.
    // This is the key change that enables sliced updates for the streaming architecture.
    const int global_idx = param_offset + local_idx;

    const SCALAR_TYPE g      = grad[global_idx];
    const SCALAR_TYPE m_prev = m1[global_idx];
    const SCALAR_TYPE v_prev = m2[global_idx];

    // Update biased first moment estimate
    const SCALAR_TYPE m_new = beta1 * m_prev + (1.0f - beta1) * g;

    // Update biased second raw moment estimate
    const SCALAR_TYPE v_new = beta2 * v_prev + (1.0f - beta2) * (g * g);

    // Compute bias-corrected first moment estimate
    const SCALAR_TYPE m_hat = m_new / (1.0f - beta1_t);

    // Compute bias-corrected second raw moment estimate
    const SCALAR_TYPE v_hat = v_new / (1.0f - beta2_t);

    // Update the parameter
    const SCALAR_TYPE param_update = learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    param[global_idx] -= param_update;

    // Store the updated momentum values
    m1[global_idx] = m_new;
    m2[global_idx] = v_new;
}

/**
 * @brief (Node 13) Implements an element-wise clamp operation for the temperature parameters.
 *
 * This is a simple, embarrassingly parallel kernel where each work-item is assigned to a single
 * temperature value. It reads the value, clamps it to the specified min/max range using the
 * built-in `clamp()` function, and writes the result back. This is the final operation in the training step.
 */
__kernel void clamp_temperatures(
    // Inputs/Outputs
    __global SCALAR_TYPE *__restrict temps_buf,

    // Inputs
    SCALAR_TYPE min_temp,
    SCALAR_TYPE max_temp,
    int         num_exits) {

    const int idx = get_global_id(0);

    if (idx >= num_exits) {
        return;
    }

    // Read, clamp, and write back in one operation.
    temps_buf[idx] = clamp(temps_buf[idx], min_temp, max_temp);
}
