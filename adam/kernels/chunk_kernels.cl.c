// chunk_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: forward_pass (Node 4) ---
// Strategy: A tiled matrix-vector multiplication.
// The `batch_offset` parameter allows this single kernel to handle both full-batch
// precomputation and smaller, streamed chunks for memory-constrained scenarios.
// Tiling with __local memory provides data reuse for inputs and weights within
// a work-group, improving performance.
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf,
    __global const SCALAR_TYPE *__restrict biases_buf,
    __global SCALAR_TYPE *__restrict hidden_out_buf,
    __global SCALAR_TYPE *__restrict hidden_mask_out,
    int batch_offset,
    int num_batch_samples,
    int padded_input_dim,
    int padded_hidden_dim) {
    const uint bid     = get_global_id(0); // Index within the current chunk
    const uint h_block = get_global_id(1); // Hidden dimension block
    const uint lid     = get_local_id(0);  // SIMD-lane index

    // Guard against out-of-bounds work-items for the chunk
    if (bid >= num_batch_samples) {
        return;
    }

    // Map chunk-local GID to global batch index
    const uint effective_bid = batch_offset + bid;

    // Propagate validity mask for this sample (single-thread write).
    if (lid == 0) {
        hidden_mask_out[effective_bid] = input_mask[effective_bid];
    }

    // Early exit for padded samples
    if (input_mask[effective_bid] < (SCALAR_TYPE)0.5f) {
        return;
    }

    // --- Tiled Matrix Multiplication using __local memory for data reuse ---
    const uint           TILE_SIZE    = SIMD_WIDTH;
    __local SCALAR_TYPE *tile_input   = local_mem;
    __local SCALAR_TYPE *tile_weights = local_mem + TILE_SIZE;

    SCALAR_TYPE accum = biases_buf[h_block * SIMD_WIDTH + lid];

    for (uint t = 0; t < padded_input_dim; t += TILE_SIZE) {
        // Coordinated load of inputs and weights into __local memory
        const uint input_idx = effective_bid * padded_input_dim + t + lid;
        if (t + lid < padded_input_dim) {
            tile_input[lid] = input_buf[input_idx];
        } else {
            tile_input[lid] = SCALAR_ZERO;
        }

        for (uint i = 0; i < TILE_SIZE; ++i) {
            // Manually linearizing the index for the documented (h_block, p_in_dim, SW) SIMD-major layout.
            const uint weight_idx = h_block * padded_input_dim * SIMD_WIDTH + (t + i) * SIMD_WIDTH + lid;
            if (t + i < padded_input_dim) {
                tile_weights[i * SIMD_WIDTH + lid] = weights_simd_major_buf[weight_idx];
            } else {
                tile_weights[i * SIMD_WIDTH + lid] = SCALAR_ZERO;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Computation from fast __local memory
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accum += tile_input[k] * tile_weights[k * SIMD_WIDTH + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const uint padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
    const uint hidden_idx               = effective_bid * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + lid;
    // Apply ReLU activation, fulfilling the kernel's contract.
    hidden_out_buf[hidden_idx] = fmax(accum, SCALAR_ZERO);
}

// --- Implementation: compute_chunk_outputs (Node 5) ---
// Strategy: A "map" kernel where each work-item computes one (exit, batch_sample)
// pair. A single branch on `problem_type_flag` handles both CCE/BCE logic.
// A stack-allocated array `p_logits` is used, leveraging the contract that
// `output_classes <= C_TILE_SIZE` to avoid local memory.
__kernel void compute_chunk_outputs(
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global const SCALAR_TYPE *__restrict exit_biases_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global SCALAR_TYPE *__restrict partial_logits_out,
    __global SCALAR_TYPE *__restrict partial_probs_out,
    __global SCALAR_TYPE *__restrict partial_loss_out,
    int problem_type_flag,
    int chunk_id,
    int param_offset,
    int batch_size,
    int hidden_dim,
    int output_classes,
    int chunk_size,
    int padded_hidden_dim) {
    const uint exit_local_idx = get_global_id(0); // Index within this chunk
    const uint batch_idx      = get_global_id(1); // Index within the batch

    const uint exit_global_idx = param_offset + exit_local_idx;

    if (exit_local_idx >= chunk_size || batch_idx >= batch_size) {
        return;
    }

    if (hidden_mask[batch_idx] < 0.5f || targets_mask[batch_idx] < 0.5f) {
        const uint out_base_idx = (chunk_id * chunk_size + exit_local_idx) * batch_size * output_classes + batch_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            partial_logits_out[out_base_idx + c] = SCALAR_ZERO;
            partial_probs_out[out_base_idx + c]  = SCALAR_ZERO;
        }
        partial_loss_out[(chunk_id * chunk_size + exit_local_idx) * batch_size + batch_idx] = SCALAR_ZERO;
        return;
    }

    SCALAR_TYPE p_logits[C_TILE_SIZE];

    for (int c = 0; c < output_classes; c++) {
        SCALAR_TYPE logit = exit_biases_buf[exit_global_idx * output_classes + c];
        for (int h = 0; h < hidden_dim; h++) {
            // Decode the physical layout of the SIMD-aware hidden buffer to find h_val.
            const uint h_block                  = h / SIMD_WIDTH;
            const uint h_lane                   = h % SIMD_WIDTH;
            const uint padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
            const uint physical_hidden_idx      = batch_idx * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
            logit += hidden_buf[physical_hidden_idx] * exit_weights_buf[exit_global_idx * hidden_dim * output_classes + h * output_classes + c];
        }
        p_logits[c] = logit;
    }

    const uint out_base_idx = (chunk_id * chunk_size + exit_local_idx) * batch_size * output_classes + batch_idx * output_classes;
    for (int c = 0; c < output_classes; c++) {
        partial_logits_out[out_base_idx + c] = p_logits[c];
    }

    SCALAR_TYPE       total_loss = SCALAR_ZERO;
    const SCALAR_TYPE temp_inv   = 1.0f / temps_buf[exit_global_idx];

    if (problem_type_flag == PROBLEM_TYPE_CCE) {
        const __global int *targets_cce = (__global int *)targets_buf;
        const int           true_class  = targets_cce[batch_idx];

        SCALAR_TYPE max_logit = -FLT_MAX;
        for (int c = 0; c < output_classes; c++)
            max_logit = fmax(max_logit, p_logits[c]);
        SCALAR_TYPE sum_exp = SCALAR_ZERO;
        for (int c = 0; c < output_classes; c++)
            sum_exp += exp((p_logits[c] - max_logit) * temp_inv);
        sum_exp = fmax(sum_exp, (SCALAR_TYPE)1e-7f); // Epsilon for numerical stability.

        for (int c = 0; c < output_classes; c++) {
            SCALAR_TYPE prob                    = exp((p_logits[c] - max_logit) * temp_inv) / sum_exp;
            partial_probs_out[out_base_idx + c] = prob;
            if (c == true_class) {
                total_loss = -log(fmax(prob, (SCALAR_TYPE)1e-7f)); // Epsilon for stability.
            }
        }
    } else { // Implicitly PROBLEM_TYPE_BCE
        const __global SCALAR_TYPE *targets_bce     = (__global SCALAR_TYPE *)targets_buf;
        const uint                  target_base_idx = batch_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            SCALAR_TYPE prob                    = 1.0f / (1.0f + MATH_FN exp(-p_logits[c] * temp_inv));
            partial_probs_out[out_base_idx + c] = prob;
            const SCALAR_TYPE target_val        = targets_bce[target_base_idx + c];
            total_loss -= (target_val * MATH_FN log(fmax(prob, (SCALAR_TYPE)1e-9f)) + (1.0f - target_val) * MATH_FN log(fmax(1.0f - prob, (SCALAR_TYPE)1e-9f)));
        }
    }

    partial_loss_out[(chunk_id * chunk_size + exit_local_idx) * batch_size + batch_idx] = total_loss;
}

// --- Implementation: calculate_chunk_gradients (Node 6) ---
// Strategy: "Work-group per gradient" reduction. Work-group (gx, gy) computes
// gradients for exit `gx` and hidden unit `gy`. Threads parallelize summation
// over the batch dimension, followed by an intra-workgroup reduction using
// __local memory to produce the final partial gradient values.
__kernel void calculate_chunk_gradients(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global SCALAR_TYPE *__restrict partial_grad_h_out,
    __global SCALAR_TYPE *__restrict partial_grad_exit_w_out,
    __global SCALAR_TYPE *__restrict partial_grad_exit_b_out,
    int problem_type_flag,
    int chunk_id,
    int param_offset,
    int batch_size,
    int hidden_dim,
    int output_classes,
    int chunk_size,
    int padded_hidden_dim) {
    const uint exit_local_idx = get_group_id(0);
    const uint h_idx          = get_group_id(1);
    const uint lid            = get_local_id(0);
    const uint lsize          = get_local_size(0);

    if (exit_local_idx >= chunk_size || h_idx >= hidden_dim) {
        return;
    }

    const uint exit_global_idx         = param_offset + exit_local_idx;
    const uint chunk_relative_exit_idx = chunk_id * chunk_size + exit_local_idx;

    for (int c_base = 0; c_base < output_classes; c_base += C_TILE_SIZE) {
        SCALAR_TYPE p_grad_w[C_TILE_SIZE] = {SCALAR_ZERO};
        SCALAR_TYPE p_grad_b[C_TILE_SIZE] = {SCALAR_ZERO};

        for (int b = lid; b < batch_size; b += lsize) {
            const uint        h_block                   = h_idx / SIMD_WIDTH;
            const uint        h_lane                    = h_idx % SIMD_WIDTH;
            const uint        padded_hidden_dim_blocks  = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
            const uint        physical_hidden_idx       = b * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
            const SCALAR_TYPE h_val                     = hidden_buf[physical_hidden_idx];
            SCALAR_TYPE       grad_h_contribution_for_b = SCALAR_ZERO;

            for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
                const int c_global = c_base + c_local;
                if (c_global >= output_classes)
                    continue;

                const uint  prob_idx = chunk_relative_exit_idx * batch_size * output_classes + b * output_classes + c_global;
                SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
                SCALAR_TYPE d_loss_d_logit;

                if (problem_type_flag == PROBLEM_TYPE_CCE) {
                    const __global int *targets_cce = (__global int *)targets_buf;
                    // This implements `p-y` for CCE: `p-1` for the target class, `p` otherwise.
                    d_loss_d_logit = select(prob, prob - 1.0f, c_global == targets_cce[b]);
                } else {
                    const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
                    d_loss_d_logit                          = prob - targets_bce[b * output_classes + c_global];
                }

                p_grad_w[c_local] += d_loss_d_logit * h_val;
                // Optimization: Bias gradients are independent of h_idx, so only one thread group (h_idx=0) calculates them.
                if (h_idx == 0) {
                    p_grad_b[c_local] += d_loss_d_logit;
                }

                const uint weight_idx = exit_global_idx * hidden_dim * output_classes + h_idx * output_classes + c_global;
                grad_h_contribution_for_b += d_loss_d_logit * exit_weights_buf[weight_idx];
            }
            const uint grad_h_out_idx          = chunk_relative_exit_idx * batch_size * hidden_dim + b * hidden_dim + h_idx;
            partial_grad_h_out[grad_h_out_idx] = grad_h_contribution_for_b;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
            const int c_global = c_base + c_local;
            if (c_global >= output_classes)
                continue;

            local_mem[lid] = p_grad_w[c_local];
            barrier(CLK_LOCAL_MEM_FENCE);
            for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
                if (lid < stride)
                    local_mem[lid] += local_mem[lid + stride];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (lid == 0) {
                const uint out_idx               = chunk_relative_exit_idx * hidden_dim * output_classes + h_idx * output_classes + c_global;
                partial_grad_exit_w_out[out_idx] = local_mem[0];
            }

            if (h_idx == 0) {
                local_mem[lid] = p_grad_b[c_local];
                barrier(CLK_LOCAL_MEM_FENCE);
                for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
                    if (lid < stride)
                        local_mem[lid] += local_mem[lid + stride];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (lid == 0) {
                    const uint out_idx               = chunk_relative_exit_idx * output_classes + c_global;
                    partial_grad_exit_b_out[out_idx] = local_mem[0];
                }
            }
        }
    }
}

// --- Implementation: calculate_chunk_temp_gradients (Node 7) ---
// Strategy: "Work-group per exit" reduction. Threads sum contributions over the
// batch dimension, with a final reduction pass in __local memory.
// Math: Implements dL/dT = sum(dL/d_logit * d_logit/dT) over the batch, where
// d_logit/dT = -unscaled_logit / T^2.
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict unscaled_logits_buf,
    __global const SCALAR_TYPE *__restrict probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,
    int problem_type_flag,
    int chunk_id,
    int param_offset,
    int batch_size,
    int output_classes,
    int chunk_size) {
    const uint exit_local_idx = get_group_id(0);
    const uint lid            = get_local_id(0);
    const uint lsize          = get_local_size(0);

    if (exit_local_idx >= chunk_size) {
        return;
    }

    const uint  exit_global_idx = param_offset + exit_local_idx;
    SCALAR_TYPE p_grad_sum      = SCALAR_ZERO;

    for (int b = lid; b < batch_size; b += lsize) {
        if (targets_mask[b] < 0.5f) {
            continue;
        }

        SCALAR_TYPE grad_contribution_for_sample = SCALAR_ZERO;
        const uint  chunk_relative_exit_idx      = chunk_id * chunk_size + exit_local_idx;

        for (int c = 0; c < output_classes; c++) {
            const uint        base_idx = chunk_relative_exit_idx * batch_size * output_classes + b * output_classes + c;
            const SCALAR_TYPE prob     = probs_buf[base_idx];
            SCALAR_TYPE       d_loss_d_logit;

            if (problem_type_flag == PROBLEM_TYPE_CCE) {
                const __global int *targets_cce = (__global int *)targets_buf;
                d_loss_d_logit                  = select(prob, prob - 1.0f, c == targets_cce[b]);
            } else { // PROBLEM_TYPE_BCE
                const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
                d_loss_d_logit                          = prob - targets_bce[b * output_classes + c];
            }

            const SCALAR_TYPE unscaled_logit = unscaled_logits_buf[base_idx];
            grad_contribution_for_sample += d_loss_d_logit * unscaled_logit;
        }
        p_grad_sum += grad_contribution_for_sample;
    }

    local_mem[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_mem[lid] += local_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const SCALAR_TYPE total_sum = local_mem[0];
        const SCALAR_TYPE temp      = temps_buf[exit_global_idx];
        // Apply the d_logit/dT term, which is -logit/T^2.
        const SCALAR_TYPE final_grad    = total_sum * (-1.0f / (temp * temp));
        const uint        out_idx       = chunk_id * chunk_size + exit_local_idx;
        partial_grad_temps_out[out_idx] = final_grad;
    }
}
