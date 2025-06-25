// backprop_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: backprop_shared_weights_chunk (Node 14) ---
// Strategy: A 2D "work-group per gradient" reduction. Each work-group, identified
// by `(group_id.x, group_id.y)`, computes the partial gradient for a single shared
// weight `SW[i][j]`. Threads within the group parallelize the summation over the
// assigned batch chunk, with a final reduction in __local memory.
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,
    int batch_offset,
    int num_batch_samples,
    int batch_chunk_index,
    int padded_input_dim,
    int padded_hidden_dim) {
    // Work-group (i_idx, j_idx) computes the gradient for shared weight SW[i_idx][j_idx].
    const uint i_idx = get_group_id(0); // Input dimension index
    const uint j_idx = get_group_id(1); // Hidden dimension index
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (i_idx >= padded_input_dim || j_idx >= padded_hidden_dim) {
        return;
    }

    SCALAR_TYPE p_grad_sw = SCALAR_ZERO;

    // Parallel loop over the assigned batch CHUNK.
    for (int b_local = lid; b_local < num_batch_samples; b_local += lsize) {
        const uint b_global = batch_offset + b_local;

        if (sample_mask[b_global] < 0.5f) {
            continue;
        }

        // Upstream gradient for this hidden neuron activation (A).
        const SCALAR_TYPE grad_h = final_grad_h_buf[b_global * padded_hidden_dim + j_idx];

        // Use the macro to reliably get the hidden activation value.
        const SCALAR_TYPE hidden_val = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(b_global, j_idx, padded_hidden_dim)];

        // Derivative of the ReLU activation function (dA/dZ).
        const SCALAR_TYPE d_activation = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);

        // Gradient w.r.t the pre-activation value Z (dL/dZ = dL/dA * dA/dZ).
        const SCALAR_TYPE dL_dZ_j = grad_h * d_activation;

        // Final gradient contribution for this weight (dL/dW_ij = dL/dZ_j * dZ_j/dW_ij = dL/dZ_j * X_i).
        p_grad_sw += dL_dZ_j * input_buf[b_global * padded_input_dim + i_idx];
    }

    // Intra-workgroup reduction for the partial shared weight gradient.
    local_mem[lid] = p_grad_sw;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_mem[lid] += local_mem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // Use the renamed `batch_chunk_index` for clarity.
        const uint grad_w_out_idx           = batch_chunk_index * padded_input_dim * padded_hidden_dim + i_idx * padded_hidden_dim + j_idx;
        partial_grad_sw_out[grad_w_out_idx] = local_mem[0];
    }
}

// --- Implementation: backprop_shared_biases_chunk (Node 15) ---
// Strategy: A 1D "work-group per gradient" reduction. Each work-group `group_id(0)`
// computes the partial gradient for a single bias term `SB[j]`. This 1D dispatch
// is more efficient than a 2D dispatch for a 1D output. Threads sum over the
// batch chunk, followed by a local memory reduction.
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,
    int batch_offset,
    int num_batch_samples,
    int batch_chunk_index,
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

        if (sample_mask[b_global] < 0.5f) {
            continue;
        }

        const SCALAR_TYPE grad_h = final_grad_h_buf[b_global * padded_hidden_dim + j_idx];

        // Use the macro to reliably get the hidden activation value.
        const SCALAR_TYPE hidden_val   = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(b_global, j_idx, padded_hidden_dim)];
        const SCALAR_TYPE d_activation = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);
        const SCALAR_TYPE dL_dZ_j      = grad_h * d_activation;

        // Gradient for bias is simply dL/dZ_j, since dZ_j/dB_j = 1.
        p_grad_sb += dL_dZ_j;
    }

    // Intra-workgroup reduction for the partial shared bias gradient.
    local_mem[lid] = p_grad_sb;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_mem[lid] += local_mem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        // Use the renamed `batch_chunk_index` for clarity.
        const uint grad_b_out_idx           = batch_chunk_index * padded_hidden_dim + j_idx;
        partial_grad_sb_out[grad_b_out_idx] = local_mem[0];
    }
}
