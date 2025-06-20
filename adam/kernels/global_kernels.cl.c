// global_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 9) Backpropagates the final aggregated gradient through the hidden layer's activation function.
 *
 * Architectural Insight:
 * - This is an embarrassingly parallel "map" kernel. Each work-item is responsible for one element
 *   of the output `grad_pre_activation_out` matrix at coordinate (batch_idx, hidden_idx).
 * - It applies the chain rule, multiplying the final aggregated upstream gradient (dL/dH) by the
 *   local derivative of the activation function (dH/d(pre-activation)).
 * - CRITICAL DEPENDENCY: It must read the `full_hidden_buf` using the same SIMD-aware physical
 *   layout that `forward_pass` (Node 4) used to write it. The other buffers use a simple,
 *   logical row-major layout.
 *
 * WORK DISPATCH: A 2D grid of (padded_batch_size, padded_hidden_dim).
 */
__kernel void finalize_backprop_activation(
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,
    __global const SCALAR_TYPE *__restrict full_hidden_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global SCALAR_TYPE *__restrict grad_pre_activation_out,
    int padded_batch_size,
    int padded_hidden_dim) {

    const uint b_idx = get_global_id(0); // Batch index
    const uint h_idx = get_global_id(1); // Hidden dimension index (logical)

    // Bounds check
    if (b_idx >= padded_batch_size || h_idx >= padded_hidden_dim) {
        return;
    }

    const uint out_idx = b_idx * padded_hidden_dim + h_idx;

    // Early exit for padded samples, ensuring a zero gradient.
    if (hidden_mask[b_idx] < 0.5f) {
        grad_pre_activation_out[out_idx] = SCALAR_ZERO;
        return;
    }

    // --- Chain Rule: dL/d(pre-act) = dL/dH * dH/d(pre-act) ---

    // 1. Load dL/dH (the final aggregated gradient from Node 8).
    // This buffer has a simple logical layout.
    const SCALAR_TYPE grad_h = final_grad_h_buf[out_idx];

    // 2. Calculate dH/d(pre-act) (the derivative of ReLU).
    // This is 1 if hidden_val > 0, and 0 otherwise.
    // To get hidden_val, we MUST decode the physical SIMD-aware layout of full_hidden_buf.
    const uint        h_block                  = h_idx / SIMD_WIDTH;
    const uint        h_lane                   = h_idx % SIMD_WIDTH;
    const uint        padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
    const uint        physical_hidden_idx      = b_idx * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
    const SCALAR_TYPE hidden_val               = full_hidden_buf[physical_hidden_idx];

    const SCALAR_TYPE d_activation = select(SCALAR_ZERO, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);

    // 3. Compute the final gradient and store it.
    grad_pre_activation_out[out_idx] = grad_h * d_activation;
}

// in global_kernels.cl.c

/**
 * @brief (Node 10) Calculates gradients for a dense layer's parameters (weights and biases).
 *
 * Architectural Insight:
 * - Implements gradient calculation using the "work-group per gradient" strategy, a highly
 *   efficient pattern for this type of problem.
 * - WORK DISPATCH: A 2D grid of work-groups is launched, sized (input_dim, hidden_dim). The work-group
 *   at `(group_id.x, group_id.y)` is responsible for computing a single gradient element for the weight
 *   at `grad_weights[x][y]`.
 * - REDUCTION: The summation over the batch dimension is parallelized by the threads within each
 *   work-group. Each thread computes a partial sum, which is then finalized using a fast parallel
 *   reduction in `__local` memory.
 * - The bias gradient `grad_biases` calculation is efficiently handled only by the work-groups in
 *   the first row of the grid (`group_id.x == 0`).
 *
 * Host Assumptions:
 * - `grad_weights` and `grad_biases` output buffers must be zeroed by the host before this kernel is launched.
 * - The host must provide two __local memory arguments, one for weight reduction and one for bias reduction.
 */
__kernel void calculate_dense_layer_gradients(
    __local SCALAR_TYPE *local_grad_w,
    __local SCALAR_TYPE *local_grad_b,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf,
    __global SCALAR_TYPE *__restrict grad_weights,
    __global SCALAR_TYPE *__restrict grad_biases,
    int padded_batch_size,
    int input_dim,
    int hidden_dim) {

    // A work-group (i_idx, j_idx) computes the gradient for weight W[i_idx][j_idx].
    const uint i_idx = get_group_id(0); // Input dimension index
    const uint j_idx = get_group_id(1); // Hidden dimension index
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Bounds check for the work-group.
    if (i_idx >= input_dim || j_idx >= hidden_dim) {
        return;
    }

    // Private accumulators for each thread's partial sum.
    SCALAR_TYPE p_grad_w = SCALAR_ZERO;
    SCALAR_TYPE p_grad_b = SCALAR_ZERO;

    // Parallel loop over the batch dimension.
    for (int b = lid; b < padded_batch_size; b += lsize) {
        // Skip padded samples.
        if (input_mask[b] < 0.5f) {
            continue;
        }

        const SCALAR_TYPE grad_pre_act = grad_pre_activation_buf[b * hidden_dim + j_idx];

        // Accumulate for weight gradient: dL/dW_ij += (dL/dZ_j) * X_i
        p_grad_w += grad_pre_act * input_buf[b * input_dim + i_idx];

        // Accumulate for bias gradient: dL/dB_j += dL/dZ_j
        // Only the first row of work-groups (i_idx=0) needs to do this.
        if (i_idx == 0) {
            p_grad_b += grad_pre_act;
        }
    }

    // --- Reduction for Weight Gradient ---
    local_grad_w[lid] = p_grad_w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_grad_w[lid] += local_grad_w[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes the final reduced value.
    if (lid == 0) {
        grad_weights[i_idx * hidden_dim + j_idx] = local_grad_w[0];
    }

    // --- Reduction for Bias Gradient (only for the first row of work-groups) ---
    if (i_idx == 0) {
        local_grad_b[lid] = p_grad_b;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = lsize / 2; s > 0; s >>= 1) {
            if (lid < s) {
                local_grad_b[lid] += local_grad_b[lid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // First thread writes the final reduced value.
        if (lid == 0) {
            grad_biases[j_idx] = local_grad_b[0];
        }
    }
}

// in global_kernels.cl.c

/**
 * @brief (Node 11) Propagates the gradient back to the layer's input via a dot product (W.T * grad).
 *
 * Architectural Insight:
 * - This is an embarrassingly parallel "map" kernel, a fundamental building block for backpropagation.
 * - WORK DISPATCH: A 2D grid of (padded_batch_size, input_dim) is launched. Each work-item at
 *   `(b_idx, i_idx)` is independent and calculates one element of the `grad_input_buf` output.
 * - The core operation for each thread is a dot product between a row of the upstream gradient
 *   (`grad_pre_activation_buf`) and a row of the weight matrix (`weights_standard_buf`), which
 *   mathematically computes `grad_input = grad_pre_activation * W.T`.
 *
 * Host Assumptions:
 * - The `weights_standard_buf` must be in a standard, logical row-major layout (input_dim x hidden_dim).
 */
__kernel void backprop_input_gradient(
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf,
    __global const SCALAR_TYPE *__restrict weights_standard_buf,
    __global SCALAR_TYPE *__restrict grad_input_buf,
    int padded_batch_size,
    int input_dim,
    int hidden_dim) {

    const uint b_idx = get_global_id(0); // Batch index
    const uint i_idx = get_global_id(1); // Input dimension index

    // Bounds check
    if (b_idx >= padded_batch_size || i_idx >= input_dim) {
        return;
    }

    // Initialize a private accumulator for the dot product.
    SCALAR_TYPE sum = SCALAR_ZERO;

    // Perform the dot product:
    // sum = grad_pre_activation_buf[b_idx, :] · weights_standard_buf[i_idx, :]
    for (int j = 0; j < hidden_dim; ++j) {
        sum += grad_pre_activation_buf[b_idx * hidden_dim + j] * weights_standard_buf[i_idx * hidden_dim + j];
    }

    // Store the final computed value in the output buffer.
    grad_input_buf[b_idx * input_dim + i_idx] = sum;
}
