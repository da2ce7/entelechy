// network_operations.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 4) Implements a high-performance, tiled matrix-vector multiplication.
 *
 * Each work-group computes a single SIMD-wide vector of hidden activations for one sample.
 * It tiles the input vector and weight matrix into `__local` memory to maximize data reuse
 * and ensure coalesced global memory access. Masking is handled as a pre-check to allow
 * invalid samples to exit early after correctly propagating their mask state.
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf,
    __global const SCALAR_TYPE *__restrict biases_buf,
    __global SCALAR_TYPE *__restrict hidden_buf,
    __global SCALAR_TYPE *__restrict hidden_mask,
    int padded_input_dim,
    int padded_hidden_dim) {

    const uint bid     = get_global_id(0);
    const uint h_block = get_global_id(1);
    const uint lid     = get_local_id(0);

    // Propagate the mask before any early exit. Only one thread per sample needs to do this.
    if (lid == 0) {
        hidden_mask[bid] = input_mask[bid];
    }

    // Early exit for padded samples.
    if (input_mask[bid] < (SCALAR_TYPE)0.5f) {
        return;
    }

    const uint           TILE_SIZE    = SIMD_WIDTH;
    __local SCALAR_TYPE *tile_input   = local_mem;
    __local SCALAR_TYPE *tile_weights = local_mem + TILE_SIZE;

    SCALAR_TYPE accum = biases_buf[h_block * SIMD_WIDTH + lid];

    for (uint t = 0; t < padded_input_dim; t += TILE_SIZE) {
        // Coordinated load into local memory
        const uint input_idx = bid * padded_input_dim + t + lid;
        if (t + lid < padded_input_dim) {
            tile_input[lid] = input_buf[input_idx];
        } else {
            tile_input[lid] = SCALAR_ZERO;
        }

        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint weight_idx = h_block * padded_input_dim * SIMD_WIDTH + (t + i) * SIMD_WIDTH + lid;
            if (t + i < padded_input_dim) {
                tile_weights[i * SIMD_WIDTH + lid] = weights_simd_major_buf[weight_idx];
            } else {
                tile_weights[i * SIMD_WIDTH + lid] = SCALAR_ZERO;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Computation from local memory
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accum += tile_input[k] * tile_weights[k * SIMD_WIDTH + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result with inlined ReLU
    const uint hidden_idx  = bid * padded_hidden_dim + h_block * SIMD_WIDTH + lid;
    hidden_buf[hidden_idx] = fmax(accum, SCALAR_ZERO);
}

/**
 * @brief (Node 5) Implements a straightforward "map" kernel where each work-item handles one (exit, sample) pair.
 *
 * The implementation first computes unscaled logits via a serial dot-product. It then applies
 * the numerically stable log-sum-exp trick for the softmax calculation before writing out all
 * three results: the raw logits, the final probabilities, and the per-exit loss.
 */
__kernel void compute_all_exits(
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global const SCALAR_TYPE *__restrict exit_biases_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const int *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict unscaled_logits_buf,
    __global SCALAR_TYPE *__restrict exit_probs_buf,
    __global SCALAR_TYPE *__restrict per_exit_losses_buf,
    int padded_batch_size,
    int hidden_dim,
    int output_classes,
    int num_exits) {

    const uint exit_idx  = get_global_id(0);
    const uint batch_idx = get_global_id(1);

    if (exit_idx >= num_exits || batch_idx >= padded_batch_size || hidden_mask[batch_idx] < 0.5f || targets_mask[batch_idx] < 0.5f) {
        // To ensure determinism, explicitly zero out outputs for invalid items.
        for (int c = 0; c < output_classes; ++c) {
            const uint out_base_idx           = exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c;
            unscaled_logits_buf[out_base_idx] = SCALAR_ZERO;
            exit_probs_buf[out_base_idx]      = SCALAR_ZERO;
        }
        per_exit_losses_buf[exit_idx * padded_batch_size + batch_idx] = SCALAR_ZERO;
        return;
    }

    // Compute unscaled logits (W*h + b)
    for (int c = 0; c < output_classes; ++c) {
        SCALAR_TYPE logit = exit_biases_buf[exit_idx * output_classes + c];
        for (int h = 0; h < hidden_dim; ++h) {
            // Must read from hidden_buf using its physical SIMD-major layout
            const uint h_block                  = h / SIMD_WIDTH;
            const uint h_lane                   = h % SIMD_WIDTH;
            const uint padded_hidden_dim_blocks = hidden_dim / SIMD_WIDTH;
            const uint physical_hidden_idx      = batch_idx * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;

            logit += hidden_buf[physical_hidden_idx] * exit_weights_buf[exit_idx * hidden_dim * output_classes + h * output_classes + c];
        }
        unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] = logit;
    }

    // Apply temperature and compute numerically stable softmax
    const SCALAR_TYPE temp_inv  = 1.0f / temps_buf[exit_idx];
    SCALAR_TYPE       max_logit = -INFINITY;
    for (int c = 0; c < output_classes; ++c) {
        max_logit = fmax(max_logit, unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c]);
    }

    SCALAR_TYPE sum_exp = 0.0f;
    for (int c = 0; c < output_classes; ++c) {
        sum_exp += exp((unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] - max_logit) * temp_inv);
    }
    sum_exp = fmax(sum_exp, (SCALAR_TYPE)1e-7f);

    const int true_class = targets_buf[batch_idx];
    for (int c = 0; c < output_classes; ++c) {
        SCALAR_TYPE scaled_logit = (unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] - max_logit) * temp_inv;
        SCALAR_TYPE prob         = exp(scaled_logit) / sum_exp;
        exit_probs_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] = prob;

        if (c == true_class) {
            per_exit_losses_buf[exit_idx * padded_batch_size + batch_idx] = -log(fmax(prob, (SCALAR_TYPE)1e-7f));
        }
    }
}

/**
 * @brief (Node 6, Tier 1) The "F1 Car": Implements the fastest path for a small number of exits (`N <= 64`).
 *
 * Each thread handles one sample and performs the entire calculation—finding confidences,
 * summing their exponents, and normalizing—within a fast, serial loop using only private
 * registers. This avoids all synchronization overhead.
 */
__kernel void ensemble_weights_reg_reduce(
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict ensemble_weights,
    int padded_batch_size,
    int output_classes,
    int num_exits) {

    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size)
        return;

    if (targets_mask[b_idx] < 0.5f) {
        for (int e = 0; e < num_exits; ++e)
            ensemble_weights[b_idx * num_exits + e] = SCALAR_ZERO;
        return;
    }

    SCALAR_TYPE p_confidences[MAX_EXITS_ENSEMBLE];
    SCALAR_TYPE sum_exp_confidences = SCALAR_ZERO;
    for (int e = 0; e < num_exits; ++e) {
        SCALAR_TYPE max_prob = SCALAR_ZERO;
        const uint  base_idx = e * padded_batch_size * output_classes + b_idx * output_classes;
        for (int c = 0; c < output_classes; ++c) {
            max_prob = fmax(max_prob, exit_probs[base_idx + c]);
        }
        p_confidences[e] = exp(max_prob);
        sum_exp_confidences += p_confidences[e];
    }
    sum_exp_confidences = fmax(sum_exp_confidences, (SCALAR_TYPE)1e-7f);

    for (int e = 0; e < num_exits; ++e) {
        ensemble_weights[b_idx * num_exits + e] = p_confidences[e] / sum_exp_confidences;
    }
}

/**
 * @brief (Node 6, Tier 2) The "Cargo Van": Implements the strategy where one work-group is assigned to one sample.
 *
 * Threads first calculate a partial sum of exponentiated confidences in private registers. These partial
 * sums are then combined using a fast, parallel reduction in `__local` memory to find the total denominator
 * for the sample before the final weights are calculated and written.
 */
__kernel void ensemble_weights_local_reduce(
    __local SCALAR_TYPE *l_reduction_mem,
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global SCALAR_TYPE *__restrict ensemble_weights,
    int num_exits,
    int output_classes,
    int padded_batch_size) {
    const uint b_idx = get_group_id(0);
    const uint tid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    SCALAR_TYPE p_sum = SCALAR_ZERO;
    for (int e = tid; e < num_exits; e += lsize) {
        SCALAR_TYPE max_prob      = SCALAR_ZERO;
        const uint  prob_base_idx = e * padded_batch_size * output_classes + b_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            max_prob = fmax(max_prob, exit_probs[prob_base_idx + c]);
        }
        p_sum += exp(max_prob);
    }
    l_reduction_mem[tid] = p_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            l_reduction_mem[tid] += l_reduction_mem[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const SCALAR_TYPE denominator = fmax(l_reduction_mem[0], (SCALAR_TYPE)1e-7f);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int e = tid; e < num_exits; e += lsize) {
        SCALAR_TYPE max_prob      = SCALAR_ZERO;
        const uint  prob_base_idx = e * padded_batch_size * output_classes + b_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            max_prob = fmax(max_prob, exit_probs[prob_base_idx + c]);
        }
        ensemble_weights[b_idx * num_exits + e] = exp(max_prob) / denominator;
    }
}

/**
 * @brief (Node 6, Tier 3, Map Stage) The "Cargo Ship" (Part 1): Implements the 'map' stage of the scalable pattern.
 *
 * This is an embarrassingly parallel kernel where each work-item independently computes a single
 * exponentiated confidence score for one (sample, exit) pair and writes it to a large intermediate buffer.
 */
__kernel void ensemble_weights_map_exp_conf(
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict temp_exp_conf_buf,
    int padded_batch_size,
    int num_exits,
    int output_classes) {
    const uint b_idx = get_global_id(0);
    const uint e_idx = get_global_id(1);

    if (b_idx >= padded_batch_size || e_idx >= num_exits)
        return;

    if (targets_mask[b_idx] < 0.5f) {
        temp_exp_conf_buf[b_idx * num_exits + e_idx] = SCALAR_ZERO;
        return;
    }

    SCALAR_TYPE max_prob      = SCALAR_ZERO;
    const uint  prob_base_idx = e_idx * padded_batch_size * output_classes + b_idx * output_classes;
    for (int c = 0; c < output_classes; c++) {
        max_prob = fmax(max_prob, exit_probs[prob_base_idx + c]);
    }
    temp_exp_conf_buf[b_idx * num_exits + e_idx] = exp(max_prob);
}

/**
 * @brief (Node 6, Tier 3, Reduce Stage) The "Cargo Ship" (Part 2): Reduces a large buffer into partial sums.
 *
 * Each work-group is assigned a large chunk of the intermediate buffer from the 'map' stage
 * and performs a parallel reduction in `__local` memory to produce a single partial sum.
 */
__kernel void
reduce_partial_sums(__local SCALAR_TYPE *l_reduction_mem, __global const SCALAR_TYPE *__restrict temp_exp_conf_buf, __global SCALAR_TYPE *__restrict temp_partial_sums_buf, int num_exits) {
    const uint b_idx      = get_group_id(0);
    const uint chunk_idx  = get_group_id(1);
    const uint num_chunks = get_num_groups(1);
    const uint tid        = get_local_id(1);
    const uint lsize      = get_local_size(1);

    const uint items_per_group = (num_exits + num_chunks - 1) / num_chunks;
    const uint chunk_start_idx = chunk_idx * items_per_group;
    const uint chunk_end_idx   = min(chunk_start_idx + items_per_group, (uint)num_exits);

    SCALAR_TYPE p_sum = SCALAR_ZERO;
    for (int e = chunk_start_idx + tid; e < chunk_end_idx; e += lsize) {
        p_sum += temp_exp_conf_buf[b_idx * num_exits + e];
    }
    l_reduction_mem[tid] = p_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            l_reduction_mem[tid] += l_reduction_mem[tid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (tid == 0) {
        temp_partial_sums_buf[b_idx * num_chunks + chunk_idx] = l_reduction_mem[0];
    }
}

/**
 * @brief (Node 6, Tier 3, Finalize Stage) The "Cargo Ship" (Part 3): Aggregates partial sums and computes final weights.
 *
 * Each thread handles one sample, performs a fast serial reduction of the few partial sums
 * to get the final denominator, and then scatters the final calculated weights back to global memory.
 */
__kernel void ensemble_weights_finalize(
    __global const SCALAR_TYPE *__restrict temp_exp_conf_buf,
    __global const SCALAR_TYPE *__restrict temp_partial_sums_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict ensemble_weights,
    int padded_batch_size,
    int num_exits,
    int num_chunks) {
    const uint b_idx = get_global_id(0);

    if (b_idx >= padded_batch_size)
        return;

    if (targets_mask[b_idx] < 0.5f) {
        for (int e = 0; e < num_exits; e++)
            ensemble_weights[b_idx * num_exits + e] = SCALAR_ZERO;
        return;
    }

    SCALAR_TYPE total_denominator = SCALAR_ZERO;
    const uint  partials_base_idx = b_idx * num_chunks;
    for (int i = 0; i < num_chunks; ++i) {
        total_denominator += temp_partial_sums_buf[partials_base_idx + i];
    }
    total_denominator = fmax(total_denominator, (SCALAR_TYPE)1e-7f);

    const uint numerators_base_idx = b_idx * num_exits;
    for (int e = 0; e < num_exits; ++e) {
        ensemble_weights[numerators_base_idx + e] = temp_exp_conf_buf[numerators_base_idx + e] / total_denominator;
    }
}

/**
 * @brief (Node 7) Implements a high-performance weighted average using a tiling strategy.
 *
 * Each thread handles one sample. By processing classes in tiles, it ensures that memory reads
 * from the large `exit_probs` buffer are fully coalesced, maximizing memory bandwidth
 * regardless of the number of exits or output classes.
 */
__kernel void blend_ensemble_probabilities(
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict ensemble_weights,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict ensemble_probs,
    int padded_batch_size,
    int output_classes,
    int num_exits) {
    const uint b_idx = get_global_id(0);

    if (b_idx >= padded_batch_size)
        return;

    if (targets_mask[b_idx] < 0.5f) {
        for (int c = 0; c < output_classes; ++c)
            ensemble_probs[b_idx * output_classes + c] = SCALAR_ZERO;
        return;
    }

    for (int c_base = 0; c_base < output_classes; c_base += C_TILE_SIZE) {
        SCALAR_TYPE p_final_probs_tile[C_TILE_SIZE] = {SCALAR_ZERO};
        for (int e = 0; e < num_exits; ++e) {
            const SCALAR_TYPE w = ensemble_weights[b_idx * num_exits + e];
            if (w == SCALAR_ZERO)
                continue;
            const uint prob_base_idx = e * padded_batch_size * output_classes + b_idx * output_classes + c_base;
            for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
                if (c_base + c_local < output_classes) {
                    p_final_probs_tile[c_local] += w * exit_probs[prob_base_idx + c_local];
                }
            }
        }
        for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
            if (c_base + c_local < output_classes) {
                ensemble_probs[b_idx * output_classes + c_base + c_local] = p_final_probs_tile[c_local];
            }
        }
    }
}
