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
 * @brief (Node 6, Hierarchical Tier, Stage 1) Processes chunks of exits to find the max logit and a stable sum-of-exponentials.
 *
 * The massively parallel 'map' stage of the reduction. It initiates the distributed log-sum-exp
 * trick for numerical stability. The two-pass approach first finds a chunk's true `max`,
 * then calculates a sum of exponentials relative to it. Re-computing the per-exit max in
 * the second pass is an intentional design choice, trading cheaper arithmetic operations
 * for reduced memory traffic—a common and effective GPU optimization.
 */
__kernel void compute_exit_chunk(
    __global const SCALAR_TYPE *__restrict unscaled_logits,
    __global SCALAR_TYPE *__restrict chunk_max_out,
    __global SCALAR_TYPE *__restrict chunk_sum_out,
    int chunk_id,
    int chunk_size,
    int total_exits,
    int padded_batch_size,
    int output_classes) {
    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size) {
        return;
    }

    const uint start_exit = chunk_id * chunk_size;
    const uint end_exit   = min((uint)(start_exit + chunk_size), (uint)total_exits);

    SCALAR_TYPE local_max_logit = -FLT_MAX;
    for (uint e_idx = start_exit; e_idx < end_exit; e_idx++) {
        SCALAR_TYPE current_exit_max_logit = -FLT_MAX;
        const uint  logit_base_idx         = e_idx * padded_batch_size * output_classes + b_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            current_exit_max_logit = fmax(current_exit_max_logit, unscaled_logits[logit_base_idx + c]);
        }
        local_max_logit = fmax(local_max_logit, current_exit_max_logit);
    }

    SCALAR_TYPE sum_of_exps = 0.0f;
    for (uint e_idx = start_exit; e_idx < end_exit; e_idx++) {
        SCALAR_TYPE current_exit_max_logit = -FLT_MAX;
        const uint  logit_base_idx         = e_idx * padded_batch_size * output_classes + b_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            current_exit_max_logit = fmax(current_exit_max_logit, unscaled_logits[logit_base_idx + c]);
        }
        sum_of_exps += exp(current_exit_max_logit - local_max_logit);
    }

    const uint out_idx     = chunk_id * padded_batch_size + b_idx;
    chunk_max_out[out_idx] = local_max_logit;
    chunk_sum_out[out_idx] = sum_of_exps;
}

/**
 * @brief (Node 6, Hierarchical Tier, Stage 2) Reduces a group of intermediate chunk results into a single, stable result.
 *
 * The recursive 'reduce' workhorse of the reduction tree. It stably merges (`max`, `sum`)
 * pairs from a previous level by first finding a new `max_of_maxes` and then re-basing
 * each input sum relative to it before accumulation. This preserves numerical precision
 * across the entire tree. This kernel is intentionally unaware of the full tree structure;
 * the host orchestrates its execution in a chained sequence to form the complete reduction DAG.
 */
__kernel void reduce_chunk_pair(
    __global const SCALAR_TYPE *__restrict input_max,
    __global const SCALAR_TYPE *__restrict input_sum,
    __global SCALAR_TYPE *__restrict output_max,
    __global SCALAR_TYPE *__restrict output_sum,
    int start_chunk_idx,
    int num_chunks_to_reduce,
    int total_input_chunks,
    int padded_batch_size) {
    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size) {
        return;
    }

    const uint end_chunk = min((uint)(start_chunk_idx + num_chunks_to_reduce), (uint)total_input_chunks);

    SCALAR_TYPE max_of_maxes = -FLT_MAX;
    for (uint c_idx = start_chunk_idx; c_idx < end_chunk; c_idx++) {
        max_of_maxes = fmax(max_of_maxes, input_max[c_idx * padded_batch_size + b_idx]);
    }

    SCALAR_TYPE combined_sum = 0.0f;
    for (uint c_idx = start_chunk_idx; c_idx < end_chunk; c_idx++) {
        SCALAR_TYPE chunk_max = input_max[c_idx * padded_batch_size + b_idx];
        SCALAR_TYPE chunk_sum = input_sum[c_idx * padded_batch_size + b_idx];
        combined_sum += chunk_sum * exp(chunk_max - max_of_maxes);
    }

    const uint out_group_idx = start_chunk_idx / num_chunks_to_reduce;
    const uint out_idx       = out_group_idx * padded_batch_size + b_idx;
    output_max[out_idx]      = max_of_maxes;
    output_sum[out_idx]      = combined_sum;
}

/**
 * @brief (Node 6, Hierarchical Tier, Stage 3) Computes the final normalized ensemble weights using globally reduced values.

 *
 * The embarrassingly parallel 'finalize' stage, consuming the single root (`global_max`,
 * `global_sum`) of the reduction tree. To conserve memory bandwidth throughout the preceding
 * reduction stages, this kernel re-computes each exit's max logit. This recompute-vs-store
 * strategy is a conscious design choice, leveraging the fact that arithmetic on modern GPUs
 * is significantly cheaper than additional global memory access.
 */
__kernel void normalize_weights(
    __global const SCALAR_TYPE *__restrict unscaled_logits,
    __global const SCALAR_TYPE *__restrict global_max,
    __global const SCALAR_TYPE *__restrict global_sum,
    __global SCALAR_TYPE *__restrict ensemble_weights,
    int total_exits,
    int padded_batch_size,
    int output_classes) {
    const uint e_idx = get_global_id(0);
    const uint b_idx = get_global_id(1);

    if (e_idx >= total_exits || b_idx >= padded_batch_size) {
        return;
    }

    SCALAR_TYPE exit_max_logit = -FLT_MAX;
    const uint  logit_base_idx = e_idx * padded_batch_size * output_classes + b_idx * output_classes;
    for (int c = 0; c < output_classes; c++) {
        exit_max_logit = fmax(exit_max_logit, unscaled_logits[logit_base_idx + c]);
    }

    const SCALAR_TYPE g_max = global_max[b_idx];
    const SCALAR_TYPE g_sum = global_sum[b_idx];

    const SCALAR_TYPE numerator   = exp(exit_max_logit - g_max);
    const SCALAR_TYPE denominator = g_sum;

    const uint out_idx        = b_idx * total_exits + e_idx;
    ensemble_weights[out_idx] = (denominator > 1e-9f) ? (numerator / denominator) : 0.0f;
}

/**
 * @brief (Node 6, Hierarchical Tier, Stage 1) Processes chunks of exits to find the max logit and a stable sum-of-exponentials.
 *
 * This kernel is the first, massively parallel stage of the hierarchical reduction tree. It replaces the old, simpler 'map'
 * stage. Each work-item is responsible for processing a single chunk of exits for a single batch item.
 *
 * To ensure numerical stability (implementing the log-sum-exp trick in a distributed manner), it uses a two-pass approach:
 * 1.  **Pass 1:** It iterates through its assigned exits to find the true maximum logit value for that chunk.
 * 2.  **Pass 2:** It re-iterates, calculating sum(exp(exit_max - chunk_max)) for all exits in its chunk.
 *
 * The resulting (max, sum) pair is written out, ready to be consumed by the `reduce_chunk_pair` kernel in the next level of the tree.
 */
__kernel void compute_exit_chunk(
    __global const SCALAR_TYPE *__restrict unscaled_logits,
    __global SCALAR_TYPE *__restrict chunk_max_out,
    __global SCALAR_TYPE *__restrict chunk_sum_out,
    int chunk_id,
    int chunk_size,
    int total_exits,
    int padded_batch_size,
    int output_classes) {
    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size) {
        return;
    }

    const uint start_exit = chunk_id * chunk_size;
    const uint end_exit   = min((uint)(start_exit + chunk_size), (uint)total_exits);

    // --- Pass 1: Find the true maximum logit value within this batch item's chunk ---
    // This is the core of the log-sum-exp trick for numerical stability.
    SCALAR_TYPE local_max_logit = -FLT_MAX;
    for (uint e_idx = start_exit; e_idx < end_exit; e_idx++) {
        SCALAR_TYPE current_exit_max_logit = -FLT_MAX;
        const uint  logit_base_idx         = e_idx * padded_batch_size * output_classes + b_idx * output_classes;

        for (int c = 0; c < output_classes; c++) {
            current_exit_max_logit = fmax(current_exit_max_logit, unscaled_logits[logit_base_idx + c]);
        }
        local_max_logit = fmax(local_max_logit, current_exit_max_logit);
    }

    // --- Pass 2: Calculate sum of exponentials relative to the true local max ---
    SCALAR_TYPE sum_of_exps = 0.0f;
    for (uint e_idx = start_exit; e_idx < end_exit; e_idx++) {
        SCALAR_TYPE current_exit_max_logit = -FLT_MAX;
        const uint  logit_base_idx         = e_idx * padded_batch_size * output_classes + b_idx * output_classes;

        // Note: This re-calculation can be optimized by using local memory to store
        // the max logit for each exit from Pass 1 if chunk_size is small and fits.
        // For a generic, robust implementation, re-calculating is safe and correct.
        for (int c = 0; c < output_classes; c++) {
            current_exit_max_logit = fmax(current_exit_max_logit, unscaled_logits[logit_base_idx + c]);
        }

        sum_of_exps += exp(current_exit_max_logit - local_max_logit);
    }

    // Write the stable, intermediate results for the next reduction stage.
    const uint out_idx     = chunk_id * padded_batch_size + b_idx;
    chunk_max_out[out_idx] = local_max_logit;
    chunk_sum_out[out_idx] = sum_of_exps;
}

/**
 * @brief (Node 6, Hierarchical Tier, Stage 2) Reduces a group of intermediate chunk results into a single, stable result.
 *
 * This kernel is the recursive workhorse of the reduction tree. It replaces the simple local-memory reduction of the old system.
 * Instead of just summing values, this kernel intelligently merges a group of (max, sum) pairs from a previous level into a new,
 * combined (max, sum) pair for the next level. This process is orchestrated by the host, which calls this kernel
 * repeatedly in a loop, creating a chain of dependencies that form the Merkle-style reduction tree.
 *
 * The key operation is a numerically stable merge:
 * 1.  It finds the absolute maximum (`max_of_maxes`) from all incoming chunk `max` values.
 * 2.  It then scales each incoming `sum` relative to this new `max_of_maxes` before accumulating them.
 *     (i.e., `combined_sum += chunk_sum * exp(chunk_max - max_of_maxes)`)
 *
 * This preserves numerical precision across the entire reduction tree, preventing intermediate values from becoming INF or zero.
 */
__kernel void reduce_chunk_pair(
    __global const SCALAR_TYPE *__restrict input_max,
    __global const SCALAR_TYPE *__restrict input_sum,
    __global SCALAR_TYPE *__restrict output_max,
    __global SCALAR_TYPE *__restrict output_sum,
    int start_chunk_idx,
    int num_chunks_to_reduce,
    int total_input_chunks,
    int padded_batch_size) {
    // Each work-item is responsible for one batch item (e.g., one sample in the batch).
    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size) {
        return;
    }

    // Determine the range of input chunks this work-item needs to read and reduce.
    const uint end_chunk = min((uint)(start_chunk_idx + num_chunks_to_reduce), (uint)total_input_chunks);

    // --- Pass 1: Find the maximum of all incoming maxes for this reduction group ---
    // This will become the new reference point for the combined sum, ensuring stability.
    SCALAR_TYPE max_of_maxes = -FLT_MAX;
    for (uint c_idx = start_chunk_idx; c_idx < end_chunk; c_idx++) {
        max_of_maxes = fmax(max_of_maxes, input_max[c_idx * padded_batch_size + b_idx]);
    }

    // --- Pass 2: Combine the sums relative to the new `max_of_maxes` ---
    // This is the core of the distributed log-sum-exp trick. Each chunk's sum, which was
    // relative to its own local max, is now adjusted to be relative to the group's max.
    SCALAR_TYPE combined_sum = 0.0f;
    for (uint c_idx = start_chunk_idx; c_idx < end_chunk; c_idx++) {
        SCALAR_TYPE chunk_max = input_max[c_idx * padded_batch_size + b_idx];
        SCALAR_TYPE chunk_sum = input_sum[c_idx * padded_batch_size + b_idx];
        combined_sum += chunk_sum * exp(chunk_max - max_of_maxes);
    }

    // Write the newly reduced (max, sum) pair to the correct location in the output buffers
    // for the next level of the reduction tree.
    const uint out_group_idx = start_chunk_idx / num_chunks_to_reduce;
    const uint out_idx       = out_group_idx * padded_batch_size + b_idx;
    output_max[out_idx]      = max_of_maxes;
    output_sum[out_idx]      = combined_sum;
}

/**
 * @brief (Node 6, Hierarchical Tier, Stage 3) Computes the final normalized ensemble weights using globally reduced values.
 *
 * This kernel is the final stage of the hierarchical reduction pipeline, replacing the old `finalize` kernel.
 * Its purpose is to perform the last, correctly-scaled normalization step for every exit.
 *
 * Unlike the old kernel, which had to perform a final serial reduction over partial sums, this kernel benefits
 * from the globally consistent `global_max` and `global_sum` values produced by the reduction tree.
 *
 * For each (exit, batch_item) pair, it:
 * 1.  Re-computes the maximum logit for that single exit.
 * 2.  Uses the pre-computed global values to perform the final, numerically stable division:
 *     `weight = exp(exit_max - global_max) / global_sum`.
 *
 * This design is highly parallel and avoids the numerical precision issues of the previous approach.
 */
__kernel void normalize_weights(
    __global const SCALAR_TYPE *__restrict unscaled_logits,
    __global const SCALAR_TYPE *__restrict global_max,
    __global const SCALAR_TYPE *__restrict global_sum,
    __global SCALAR_TYPE *__restrict ensemble_weights,
    int total_exits,
    int padded_batch_size,
    int output_classes) {
    // This kernel uses a 2D grid where each work-item handles one exit for one batch item.
    const uint e_idx = get_global_id(0); // Exit index
    const uint b_idx = get_global_id(1); // Batch index

    if (e_idx >= total_exits || b_idx >= padded_batch_size) {
        return;
    }

    // --- Step 1: Find the max logit for this specific exit ---
    // This value needs to be re-calculated as it was not stored during the reduction to save memory.
    SCALAR_TYPE exit_max_logit = -FLT_MAX;
    const uint  logit_base_idx = e_idx * padded_batch_size * output_classes + b_idx * output_classes;
    for (int c = 0; c < output_classes; c++) {
        exit_max_logit = fmax(exit_max_logit, unscaled_logits[logit_base_idx + c]);
    }

    // --- Step 2: Perform the final, numerically stable normalization ---
    const SCALAR_TYPE g_max = global_max[b_idx];
    const SCALAR_TYPE g_sum = global_sum[b_idx];

    // This is the final, stable calculation. The numerator is scaled relative to the global max,
    // and the denominator (g_sum) is already correctly scaled from the reduction tree.
    const SCALAR_TYPE numerator   = exp(exit_max_logit - g_max);
    const SCALAR_TYPE denominator = g_sum;

    // Guard against division by zero. This can happen if all logits for a sample were -INF,
    // resulting in a global_sum of 0. In this case, the weights should be zero.
    const uint out_idx        = b_idx * total_exits + e_idx;
    ensemble_weights[out_idx] = (denominator > 1e-9f) ? (numerator / denominator) : 0.0f;
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
