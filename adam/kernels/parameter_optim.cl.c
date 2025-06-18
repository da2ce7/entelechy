// parameter_optim.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 15) Implements the Adam optimization step as an embarrassingly parallel map kernel.
 *
 * This kernel is generic and can be applied to any flattened parameter buffer. Each work-item is
 * assigned to a single parameter and is completely independent. It loads the parameter, its gradient,
 * and its momentum vectors; performs the standard Adam update equations; and writes the new values back.
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
    int total_params) {

    const int idx = get_global_id(0);

    if (idx >= total_params) {
        return;
    }

    const SCALAR_TYPE g      = grad[idx];
    const SCALAR_TYPE m_prev = m1[idx];
    const SCALAR_TYPE v_prev = m2[idx];

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
    param[idx] -= param_update;

    // Store the updated momentum values
    m1[idx] = m_new;
    m2[idx] = v_new;
}

/**
 * @brief (Node 16) Implements an element-wise clamp operation for the temperature parameters.
 *
 * This is a simple, embarrassingly parallel kernel where each work-item is assigned to a single
 * temperature value. It reads the value, clamps it to the specified min/max range using the
 * built-in `clamp()` function, and writes the result back.
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
