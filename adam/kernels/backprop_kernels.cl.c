// backprop_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 9) Calculates PARTIAL gradients for the shared layer by processing a CHUNK of the BATCH.
 *
 * Architectural Insight:
 * - This kernel is the heart of the scalable backpropagation (Phase 5). It streams over the batch
 *   dimension, which is essential for managing the memory footprint of large input and hidden activation buffers.
 * - Its contract is to compute the PARTIAL gradients for the shared layer's weights (SW) and biases (SB)
 *   for a single chunk of the batch data. These partial results are then fed to the final aggregation
 *   stage (Node 10).
 * - CRITICAL DEPENDENCY: It consumes the FINAL aggregated `grad_h` from Phase 4. Each work-item
 *   in this kernel reads the relevant slice of `final_grad_h_buf` to backpropagate through the ReLU activation.
 *
 * WORK DISPATCH: A "work-group per gradient" strategy. Work-group (group_id.x, group_id.y) computes
 * the PARTIAL gradient contribution for weight SW[x][y] from this specific batch chunk.
 */
__kernel void backprop_shared_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,
    int batch_offset,
    int num_batch_samples,
    int chunk_id, // The ID of THIS batch chunk, used for writing to the output buffer
    int padded_input_dim,
    int padded_hidden_dim) {
    // Work-group (i_idx, j_idx) computes the gradient for shared weight SW[i_idx][j_idx].
    const uint i_idx = get_group_id(0); // Input dimension index (0 to padded_input_dim-1)
    const uint j_idx = get_group_id(1); // Hidden dimension index (0 to padded_hidden_dim-1)
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Bounds check for the entire work-group
    if (i_idx >= padded_input_dim || j_idx >= padded_hidden_dim) {
        return;
    }

    // Private accumulators for this thread's partial sum over its slice of the chunk.
    SCALAR_TYPE p_grad_sw = SCALAR_ZERO;
    SCALAR_TYPE p_grad_sb = SCALAR_ZERO;

    // Parallel loop over the batch CHUNK assigned to this kernel launch.
    for (int b_local = lid; b_local < num_batch_samples; b_local += lsize) {
        const uint b_global = batch_offset + b_local; // Map local chunk index to global batch index

        if (input_mask[b_global] < 0.5f) {
            continue;
        }

        // --- Backprop through ReLU activation function ---
        // 1. Get the final aggregated upstream gradient dL/dH_j
        const SCALAR_TYPE grad_h = final_grad_h_buf[b_global * padded_hidden_dim + j_idx];

        // 2. Calculate dH_j/dZ_j (derivative of ReLU)
        // Must decode the SIMD-aware physical layout of the hidden_buf.
        const uint        h_block                  = j_idx / SIMD_WIDTH;
        const uint        h_lane                   = j_idx % SIMD_WIDTH;
        const uint        padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
        const uint        physical_hidden_idx      = b_global * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
        const SCALAR_TYPE hidden_val               = hidden_buf[physical_hidden_idx];
        const SCALAR_TYPE d_activation             = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);

        // 3. Compute the gradient w.r.t pre-activation: dL/dZ_j = dL/dH_j * dH_j/dZ_j
        const SCALAR_TYPE dL_dZ_j = grad_h * d_activation;

        // --- Accumulate gradient contributions from this sample ---
        // dL/dSW_ij += (dL/dZ_j) * X_i
        p_grad_sw += dL_dZ_j * input_buf[b_global * padded_input_dim + i_idx];

        // dL/dSB_j += dL/dZ_j
        // Only the first row of work-groups (i_idx=0) needs to do this.
        if (i_idx == 0) {
            p_grad_sb += dL_dZ_j;
        }
    }

    // --- Reduction Part 1: Shared Weight Gradient (p_grad_sw) ---
    local_mem[lid] = p_grad_sw;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s)
            local_mem[lid] += local_mem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        // The final element is a partial gradient for one batch chunk. We store it at index chunk_id.
        const uint grad_w_out_idx           = chunk_id * padded_input_dim * padded_hidden_dim + i_idx * padded_hidden_dim + j_idx;
        partial_grad_sw_out[grad_w_out_idx] = local_mem[0];
    }

    // --- Reduction Part 2: Shared Bias Gradient (p_grad_sb) ---
    if (i_idx == 0) {
        local_mem[lid] = p_grad_sb;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint s = lsize / 2; s > 0; s >>= 1) {
            if (lid < s)
                local_mem[lid] += local_mem[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            const uint grad_b_out_idx           = chunk_id * padded_hidden_dim + j_idx;
            partial_grad_sb_out[grad_b_out_idx] = local_mem[0];
        }
    }
}
