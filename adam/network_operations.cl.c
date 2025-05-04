// network_operations.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

__kernel void forward_pass(
    __local SCALAR_TYPE        *local_mem,
    __global const SCALAR_TYPE *input,
    __global const SCALAR_TYPE *input_mask,
    __global const SCALAR_TYPE *weights,
    __global const SCALAR_TYPE *biases,
    __global SCALAR_TYPE       *hidden,
    __global SCALAR_TYPE       *hidden_mask,
    int                         padded_input_dim,
    int                         padded_hidden_dim) {

    const uint bid  = get_global_id(0); // Batch index
    const uint h    = get_global_id(1); // Hidden block index
    const uint lid  = get_local_id(0);  // SIMD lane in workgroup
    const uint SIMD = SIMD_WIDTH;

    const uint           tile_stride  = get_local_size(0);
    __local SCALAR_TYPE *tile_input   = local_mem;
    __local SCALAR_TYPE *tile_weights = local_mem + tile_stride;

    if (h * SIMD >= padded_hidden_dim) {
        return;
    }

    // Process valid samples using mask
    if (input_mask[bid] < SCALAR_TYPE(0.5)) {
        hidden_mask[bid] = SCALAR_ZERO;
        for (uint l = 0; l < SIMD; l++) {
            hidden[bid * padded_hidden_dim + h * SIMD + l] = SCALAR_ZERO; // Updated indexing
        }
        return;
    }
    hidden_mask[bid] = SCALAR_TYPE(1.0);

    SCALAR_TYPE accum = SCALAR_ZERO;

    // Process input in tiles using padded dimension
    for (uint t = 0; t < padded_input_dim; t += tile_stride) {
        const uint copy_len = min(tile_stride, padded_input_dim - t);

        // Load input tile from global to local memory
        if (lid < copy_len) {
            tile_input[lid] = input[bid * padded_input_dim + t + lid];
        }

        // Coalesced weight load with 3D indexing
        if (lid < copy_len) {
            uint weight_idx   = h * padded_input_dim * SIMD + (t + lid) * SIMD + lid;
            tile_weights[lid] = weights[weight_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

// Vectorized accumulation
#if SIMD_WIDTH >= 4
#pragma unroll
        for (uint l = 0; l < copy_len; l += 4) {
            SCALAR_TYPE4 in_vec = vload4(l, tile_input);
            SCALAR_TYPE4 wt_vec = vload4(l, tile_weights);
            accum += dot(in_vec, wt_vec);
        }
#else
        for (uint l = 0; l < copy_len; l++) {
            accum += tile_input[l] * tile_weights[l];
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result with ReLU
    const uint hidden_idx = bid * padded_hidden_dim + h * SIMD + lid; // padded_hidden_dim used
    accum += biases[h * SIMD + lid];
    hidden[hidden_idx] = scalar_relu(accum);
}

__kernel void compute_exit_probabilities(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict exit_weights,
    __global const SCALAR_TYPE *__restrict exit_biases,
    __global SCALAR_TYPE *__restrict exit_probs,
    __global SCALAR_TYPE *__restrict exit_probs_mask,
    __global SCALAR_TYPE *__restrict losses,
    __global SCALAR_TYPE *__restrict losses_mask,
    __global const int *__restrict targets,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict temperatures,
    int padded_batch_size,
    int hidden_dim,
    int output_classes,
    int padded_hidden_dim,
    int padded_output_classes,
    int num_exits) {
    const uint exit_idx  = get_global_id(0);
    const uint batch_idx = get_global_id(1);
    const uint lid       = get_local_id(0);
    const uint lsize     = get_local_size(0);

    const bool valid_sample    = (hidden_mask[batch_idx] > 0.5f) && (targets_mask[batch_idx] > 0.5f);
    exit_probs_mask[batch_idx] = valid_sample ? (SCALAR_TYPE)1.0 : SCALAR_ZERO;
    losses_mask[batch_idx]     = valid_sample ? (SCALAR_TYPE)1.0 : SCALAR_ZERO;

    if (!valid_sample || exit_idx >= num_exits) {
        for (uint c = lid; c < padded_output_classes; c += lsize) {
            exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c] = SCALAR_ZERO;
        }
        return;
    }

    const int         true_class = targets[batch_idx];
    const SCALAR_TYPE temp       = temperatures[exit_idx];
    SCALAR_TYPE       max_logit  = -INFINITY;
    SCALAR_TYPE       sum_exp    = SCALAR_ZERO;
    SCALAR_TYPE       loss       = SCALAR_ZERO;

    // Phase 1: Compute logits & find global max
    for (uint c = lid; c < output_classes; c += lsize) {
        SCALAR_TYPE logit = exit_biases[exit_idx * padded_output_classes + c];

        const uint weight_block = exit_idx * padded_hidden_dim * padded_output_classes + (c / SIMD_WIDTH) * padded_hidden_dim * SIMD_WIDTH;

        for (uint k = 0; k < hidden_dim; k++) {
            logit += hidden[batch_idx * padded_hidden_dim + k] * exit_weights[weight_block + k * SIMD_WIDTH + (batch_idx % SIMD_WIDTH)];
        }

        logit *= native_recip(temp);
        exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c] = logit;

        max_logit = fmax(max_logit, logit);
    }

    // Single barrier for max reduction
    local_mem[lid] = max_logit;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction for max (no tree)
    for (uint s = lsize; s > 1; s = (s + 1) / 2) {
        const uint half = s / 2;
        if (lid < half && lid + half < s) {
            local_mem[lid] = fmax(local_mem[lid], local_mem[lid + half]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const SCALAR_TYPE global_max = local_mem[0];

    // Phase 2: Compute exps and sum (on previous data)
    for (uint c = lid; c < output_classes; c += lsize) {
        SCALAR_TYPE val = exp(exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c] - global_max);
        sum_exp += val;
        exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c] = val;
    }

    // Single barrier for sum reduction
    local_mem[lid] = sum_exp;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Reduction for sum (horizontal collapse)
    for (uint s = lsize; s > 1; s = (s + 1) / 2) {
        const uint half = s / 2;
        if (lid < half && lid + half < s) {
            local_mem[lid] += local_mem[lid + half];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const SCALAR_TYPE total_sum = fmax(local_mem[0], SAFE_LOG_MIN);

    // Phase 3: Final probabilities and loss
    if (lid == 0) {
        losses[exit_idx * padded_batch_size + batch_idx] = SCALAR_ZERO;
    }

    for (uint c = lid; c < output_classes; c += lsize) {
        exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c] /= total_sum;

        if ((int)c == true_class) {
            loss = -native_log(fmax(exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c], SAFE_LOG_MIN));
        }
    }

    // Final loss accumulation (single write)
    if (lid == 0 && output_classes > 0) {
        losses[exit_idx * padded_batch_size + batch_idx] = loss;
    }

    // Handle padded classes
    for (uint c = output_classes + lid; c < padded_output_classes; c += lsize) {
        exit_probs[exit_idx * padded_batch_size * padded_output_classes + batch_idx * padded_output_classes + c] = SCALAR_ZERO;
    }
}
