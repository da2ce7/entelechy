// parameter_optim.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: adam_update (Node 17) ---
// Strategy: An embarrassingly parallel map kernel. Each work-item is assigned
// to a single parameter and performs the update completely independently.
// The `param_offset` enables this generic kernel to be dispatched multiple times,
// applying updates to distinct slices of the overall parameter set.
//
// By accepting the raw global step `t`, this kernel guarantees numerical stability
// for training runs of any length. It performs the sensitive bias correction
// power calculation (`beta**t`) on the device, avoiding potential host-side
// precision loss when `t` becomes very large, as described in the
// "Marathon" validation scenario.
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad,
    SCALAR_TYPE beta1,
    SCALAR_TYPE beta2,
    SCALAR_TYPE learning_rate,
    SCALAR_TYPE epsilon,
    uint        t,
    __global SCALAR_TYPE *__restrict param,
    __global SCALAR_TYPE *__restrict m1,
    __global SCALAR_TYPE *__restrict m2,
    int param_offset,
    int num_params_to_update) {

    const int local_idx = get_global_id(0);

    if (local_idx >= num_params_to_update) {
        return;
    }

    // Map the work-item's local index within this dispatch to the global index
    // into the full parameter and momentum buffers.
    const int global_idx = param_offset + local_idx;

    const SCALAR_TYPE g      = grad[global_idx];
    const SCALAR_TYPE m_prev = m1[global_idx];
    const SCALAR_TYPE v_prev = m2[global_idx];

    // Update biased first moment estimate (m_t).
    const SCALAR_TYPE m_new = beta1 * m_prev + (1.0f - beta1) * g;

    // Update biased second raw moment estimate (v_t).
    const SCALAR_TYPE v_new = beta2 * v_prev + (1.0f - beta2) * (g * g);

    // Perform bias correction calculation internally for maximum numerical stability.
    const SCALAR_TYPE beta1_t = pown(beta1, (int)t);
    const SCALAR_TYPE beta2_t = pown(beta2, (int)t);

    // Compute bias-corrected first moment estimate (m_hat_t).
    const SCALAR_TYPE m_hat = m_new / (1.0f - beta1_t);

    // Compute bias-corrected second raw moment estimate (v_hat_t).
    const SCALAR_TYPE v_hat = v_new / (1.0f - beta2_t);

    // Update the parameter.
    const SCALAR_TYPE param_update = learning_rate * m_hat / (MATH_FN sqrt(v_hat) + epsilon);
    param[global_idx] -= param_update;

    // Store the updated momentum values.
    m1[global_idx] = m_new;
    m2[global_idx] = v_new;
}

// --- Implementation: clamp_temperatures (Node 18) ---
// Strategy: An embarrassingly parallel map kernel. Each work-item is assigned to
// a single temperature parameter and performs the clamp operation independently.
// This is the final operation in the training graph.
__kernel void clamp_temperatures(__global SCALAR_TYPE *__restrict temps_buf, SCALAR_TYPE min_temp, SCALAR_TYPE max_temp, int total_exits) {

    const int idx = get_global_id(0);

    // Use the standardized `total_exits` parameter name.
    if (idx >= total_exits) {
        return;
    }

    // Read, clamp, and write back in one operation.
    temps_buf[idx] = clamp(temps_buf[idx], min_temp, max_temp);
}
