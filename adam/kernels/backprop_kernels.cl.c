// backprop_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 10) Calculates PARTIAL gradients for the shared layer WEIGHTS by processing a CHUNK of the BATCH.
 *
 * This kernel is the first part of the scalable backpropagation (Phase 6). It streams over the batch
 * dimension to compute the partial gradients for the shared layer's weights (SW). The result is
 * fed to the final aggregation stage (Node 11).
 *
 * WORK DISPATCH: A 2D "work-group per gradient" strategy. Work-group (group_id.x, group_id.y) computes
 * the PARTIAL gradient contribution for weight SW[x][y] from this specific batch chunk.
 */
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,
    int batch_offset,
    int num_batch_samples,
    int chunk_id,
    int padded_input_dim,
    int padded_hidden_dim) {
    // Work-group (i_idx, j_idx) computes the gradient for shared weight SW[i_idx][j_idx].
    const uint i_idx = get_group_id(0); // Input dimension index (0 to padded_input_dim-1)
    const uint j_idx = get_group_id(1); // Hidden dimension index (0 to padded_hidden_dim-1)
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (i_idx >= padded_input_dim || j_idx >= padded_hidden_dim) {
        return;
    }

    SCALAR_TYPE p_grad_sw = SCALAR_ZERO;

    // Parallel loop over the batch CHUNK.
    for (int b_local = lid; b_local < num_batch_samples; b_local += lsize) {
        const uint b_global = batch_offset + b_local;

        if (input_mask[b_global] < 0.5f) {
            continue;
        }

        const SCALAR_TYPE grad_h = final_grad_h_buf[b_global * padded_hidden_dim + j_idx];

        const uint        h_block                  = j_idx / SIMD_WIDTH;
        const uint        h_lane                   = j_idx % SIMD_WIDTH;
        const uint        padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
        const uint        physical_hidden_idx      = b_global * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
        const SCALAR_TYPE hidden_val               = hidden_buf[physical_hidden_idx];
        const SCALAR_TYPE d_activation             = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);
        const SCALAR_TYPE dL_dZ_j                  = grad_h * d_activation;

        p_grad_sw += dL_dZ_j * input_buf[b_global * padded_input_dim + i_idx];
    }

    // Reduction for PARTIAL shared weight gradient
    local_mem[lid] = p_grad_sw;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s)
            local_mem[lid] += local_mem[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const uint grad_w_out_idx           = chunk_id * padded_input_dim * padded_hidden_dim + i_idx * padded_hidden_dim + j_idx;
        partial_grad_sw_out[grad_w_out_idx] = local_mem[0];
    }
}

/**
 * @brief (Node 11) Calculates PARTIAL gradients for the shared layer BIASES by processing a CHUNK of the BATCH.
 *
 * This kernel is the second part of the scalable backpropagation (Phase 6). It performs a dedicated,
 * more efficient 1D reduction to compute the partial gradients for the shared layer's biases (SB).
 * The result is fed to the final aggregation stage (Node 11).
 *
 * WORK DISPATCH: A 1D "work-group per gradient" strategy. Work-group `group_id(0)` computes the
 * PARTIAL gradient contribution for bias SB[j] from this specific batch chunk.
 */
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,
    int batch_offset,
    int num_batch_samples,
    int chunk_id,
    int padded_hidden_dim) {
    // Work-group j_idx computes the gradient for shared bias SB[j_idx].
    const uint j_idx = get_group_id(0);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (j_idx >= padded_hidden_dim) {
        return;
    }

    SCALAR_TYPE p_grad_sb = SCALAR_ZERO;

    // Parallel loop over the batch CHUNK.
    for (int b_local = lid; b_local < num_batch_samples; b_local += lsize) {
        const uint b_global = batch_offset + b_local;

        if (input_mask[b_global] < 0.5f) {
            continue;
        }

        const SCALAR_TYPE grad_h = final_grad_h_buf[b_global * padded_hidden_dim + j_idx];

        const uint        h_block                  = j_idx / SIMD_WIDTH;
        const uint        h_lane                   = j_idx % SIMD_WIDTH;
        const uint        padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
        const uint        physical_hidden_idx      = b_global * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
        const SCALAR_TYPE hidden_val               = hidden_buf[physical_hidden_idx];
        const SCALAR_TYPE d_activation             = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);
        const SCALAR_TYPE dL_dZ_j                  = grad_h * d_activation;

        p_grad_sb += dL_dZ_j;
    }

    // Reduction for PARTIAL shared bias gradient
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
