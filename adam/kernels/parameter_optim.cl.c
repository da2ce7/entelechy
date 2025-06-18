// parameter_optim.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief Performs an element-wise Adam optimization step.
 *
 * This kernel is generic and can be applied to any parameter buffer and its corresponding
 * gradient and Adam moment buffers. It is launched with a 1D NDRange where each work-item
 * is responsible for updating a single parameter.
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

    // Each work-item processes one element of the flattened parameter array
    const int idx = get_global_id(0);

    // Boundary check to prevent processing padded elements if not desired,
    // or if the global size is rounded up.
    if (idx >= total_params) {
        return;
    }

    // Load the current gradient and moment values for this parameter
    const SCALAR_TYPE g      = grad[idx];
    const SCALAR_TYPE m_prev = m1[idx];
    const SCALAR_TYPE v_prev = m2[idx];

    // --- Adam Update Equations ---

    // 1. Update biased first moment estimate
    const SCALAR_TYPE m_new = beta1 * m_prev + (1.0f - beta1) * g;

    // 2. Update biased second raw moment estimate
    const SCALAR_TYPE v_new = beta2 * v_prev + (1.0f - beta2) * (g * g);

    // 3. Compute bias-corrected first moment estimate
    const SCALAR_TYPE m_hat = m_new / (1.0f - beta1_t);

    // 4. Compute bias-corrected second raw moment estimate
    const SCALAR_TYPE v_hat = v_new / (1.0f - beta2_t);

    // 5. Update the parameter
    const SCALAR_TYPE param_update = learning_rate * m_hat / (sqrt(v_hat) + epsilon);
    const SCALAR_TYPE param_new    = param[idx] - param_update;

    // --- Store the updated values back to global memory ---
    param[idx] = param_new;
    m1[idx]    = m_new;
    m2[idx]    = v_new;
}

/**
 * @brief Clamps the temperature values to a safe range.
 *
 * This kernel is run after the Adam optimizer updates the temperatures to prevent them
 * from becoming zero, negative, or excessively large, which would cause numerical
 * instability in subsequent training steps. Each work-item clamps a single temperature.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [shape: NUM_EXITS]
    SCALAR_TYPE min_temp,                       // Minimum temperature value
    SCALAR_TYPE max_temp,                       // Maximum temperature value
    int         num_exits) {                            // HOST parameter NUM_EXITS

    // Each work-item processes one temperature value
    const int idx = get_global_id(0);

    // Boundary check, although host should launch with global_size == num_exits
    if (idx >= num_exits) {
        return;
    }

    // Load the current temperature
    SCALAR_TYPE temp = temps_buf[idx];

    // Clamp the value using the built-in OpenCL function
    temp = clamp(temp, min_temp, max_temp);

    // Write the clamped value back to global memory
    temps_buf[idx] = temp;
}
