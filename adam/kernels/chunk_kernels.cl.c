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

    const uint effective_bid = batch_offset + bid;

    // Propagate validity mask (single-thread write).
    if (lid == 0) {
        hidden_mask_out[effective_bid] = input_mask[effective_bid];
    }
    if (input_mask[effective_bid] < (SCALAR_TYPE)0.5f) {
        return;
    }

    const uint           TILE_SIZE    = SIMD_WIDTH;
    __local SCALAR_TYPE *tile_input   = local_mem;
    __local SCALAR_TYPE *tile_weights = local_mem + TILE_SIZE;

    // Each thread accumulates its partial dot product for one output neuron.
    SCALAR_TYPE accum = biases_buf[h_block * SIMD_WIDTH + lid];

    // Loop over the input dimension in tiles
    for (uint t = 0; t < padded_input_dim; t += TILE_SIZE) {
        // --- Coordinated Load from Global to Local Memory ---

        // 1. Load a tile of the input vector. All threads in the work-group cooperate.
        //    This input tile is correctly reused by all threads for the computation below.
        const uint input_idx = effective_bid * padded_input_dim + t + lid;
        if (t + lid < padded_input_dim) {
            tile_input[lid] = input_buf[input_idx];
        } else {
            tile_input[lid] = SCALAR_ZERO;
        }

        // 2. THE FIX: Load a tile of the weight matrix.
        //    The flawed inner loop is REMOVED. Each thread `lid` now cooperatively loads
        //    the column of the weight tile corresponding to its output neuron.
        //    This completely avoids redundant reads from global memory within the 't' loop.
        for (int i = 0; i < TILE_SIZE; ++i) {
            const uint weight_idx = h_block * padded_input_dim * SIMD_WIDTH + (t + i) * SIMD_WIDTH + lid;
            if (t + i < padded_input_dim) {
                tile_weights[i * SIMD_WIDTH + lid] = weights_simd_major_buf[weight_idx];
            } else {
                tile_weights[i * SIMD_WIDTH + lid] = SCALAR_ZERO;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Computation from Fast __local Memory ---
        // Each thread computes its partial sum using the shared input tile
        // and its own column from the weight tile.
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accum += tile_input[k] * tile_weights[k * SIMD_WIDTH + lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result with ReLU activation.
    const uint hidden_idx      = GET_PHYSICAL_HIDDEN_IDX(effective_bid, h_block * SIMD_WIDTH + lid, padded_hidden_dim);
    hidden_out_buf[hidden_idx] = fmax(accum, SCALAR_ZERO);
}

// --- Implementation: compute_logits_chunk (Node 5) ---
// Strategy: A pure "map" kernel dispatched with a 3D grid. Each work-item is
// responsible for computing exactly one logit value. The three dimensions of the
// grid map to (exit, batch_sample, class). This is the first, fully streamable
// stage in the three-part softmax pipeline.
__kernel void compute_logits_chunk(
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global const SCALAR_TYPE *__restrict exit_biases_buf,
    __global SCALAR_TYPE *__restrict full_logits_out,
    int chunk_id,
    int param_offset,
    int chunk_size,
    int class_offset,
    int num_classes_in_chunk,
    int full_batch_size,
    int hidden_dim,
    int padded_hidden_dim,
    int total_output_classes) {

    // Map the 3D work-item grid to the logical problem space.
    const uint exit_local_idx  = get_global_id(0); // Index within the current EXIT chunk
    const uint batch_idx       = get_global_id(1); // Global batch index
    const uint class_local_idx = get_global_id(2); // Index within the current CLASS chunk

    // --- Boundary Checks ---
    if (exit_local_idx >= chunk_size || batch_idx >= full_batch_size || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    // Convert local chunk indices to global indices.
    const uint exit_global_idx  = param_offset + exit_local_idx;
    const uint class_global_idx = class_offset + class_local_idx;

    // The output index for this specific work-item's logit.
    const uint out_idx = exit_global_idx * full_batch_size * total_output_classes + batch_idx * total_output_classes + class_global_idx;

    // --- Masking ---
    // For padded or invalid samples, skip computation and write a zero logit.
    if (hidden_mask[batch_idx] < 0.5f) {
        full_logits_out[out_idx] = SCALAR_ZERO;
        return;
    }

    // --- Core Computation: Dot Product ---
    // Initialize the accumulator with the bias for the assigned (exit, class).
    SCALAR_TYPE logit = exit_biases_buf[exit_global_idx * total_output_classes + class_global_idx];

    // Compute the dot product of the hidden activation vector and the corresponding weight vector.
    for (int h = 0; h < hidden_dim; h++) {
        // Decode the physical layout of the SIMD-aware hidden buffer to get the h-th value.
        const uint        h_block                  = h / SIMD_WIDTH;
        const uint        h_lane                   = h % SIMD_WIDTH;
        const uint        padded_hidden_dim_blocks = (padded_hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
        const uint        physical_hidden_idx      = batch_idx * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
        const SCALAR_TYPE h_val                    = hidden_buf[physical_hidden_idx];

        // Get the corresponding weight.
        const uint        weight_idx = exit_global_idx * hidden_dim * total_output_classes + h * total_output_classes + class_global_idx;
        const SCALAR_TYPE w_val      = exit_weights_buf[weight_idx];

        logit += h_val * w_val;
    }

    // Write the final computed logit to the full global buffer.
    full_logits_out[out_idx] = logit;
}

// --- Implementation: reduce_logits_for_softmax (Node 6) ---
// Strategy: A 2D "map" kernel where each work-item (exit, batch_sample) performs
// an independent reduction over the entire class dimension for that sample. This
// kernel is the essential synchronization point in the class-streaming pipeline.
// A two-pass algorithm is used for numerical stability when calculating the sum
// of exponentials.
__kernel void reduce_logits_for_softmax(
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global SCALAR_TYPE *__restrict softmax_params_out,
    int num_total_exits,
    int full_batch_size,
    int total_output_classes) {

    // Map the 2D work-item grid to the (exit, batch) problem space.
    const uint exit_idx  = get_global_id(0);
    const uint batch_idx = get_global_id(1);

    // Boundary checks
    if (exit_idx >= num_total_exits || batch_idx >= full_batch_size) {
        return;
    }

    // Base index for all data related to this specific (exit, batch) pair.
    const uint base_in_idx  = exit_idx * full_batch_size * total_output_classes + batch_idx * total_output_classes;
    const uint base_out_idx = exit_idx * full_batch_size * 2 + batch_idx * 2;

    // Fetch the temperature for this exit once.
    const SCALAR_TYPE temp     = temps_buf[exit_idx];
    const SCALAR_TYPE temp_inv = 1.0f / temp;

    // --- Pass 1: Find the maximum of the temperature-scaled logits ---
    // This is essential for numerical stability of the `exp` function in the next pass.
    SCALAR_TYPE max_scaled_logit = -FLT_MAX;
    for (int c = 0; c < total_output_classes; c++) {
        const SCALAR_TYPE logit = full_logits_buf[base_in_idx + c];
        max_scaled_logit        = fmax(max_scaled_logit, logit * temp_inv);
    }

    // For cases where all logits are -inf (e.g., masked sample), prevent max_scaled_logit from being -FLT_MAX.
    // This can happen if a sample is valid but produces no valid logits.
    if (max_scaled_logit == -FLT_MAX) {
        max_scaled_logit = 0.0f;
    }

    // --- Pass 2: Calculate the sum of exponentials using the max_scaled_logit for stability ---
    SCALAR_TYPE sum_exp = SCALAR_ZERO;
    for (int c = 0; c < total_output_classes; c++) {
        const SCALAR_TYPE logit = full_logits_buf[base_in_idx + c];
        // Subtracting the max before exponentiating prevents overflow to +inf and improves precision for small values.
        sum_exp += MATH_FN exp((logit * temp_inv) - max_scaled_logit);
    }

    // Write the two resulting parameters to the output buffer.
    // The next kernel will use these to compute the final probabilities.
    softmax_params_out[base_out_idx + 0] = max_scaled_logit;
    softmax_params_out[base_out_idx + 1] = sum_exp;
}

// --- Implementation: compute_probs_loss_cce_chunk (Node 7 - CCE Path) ---
// Strategy: A pure 3D "map" kernel dispatched with a grid corresponding to
// (exit, batch_sample, class). Each work-item is responsible for one probability
// value. If a work-item's class matches the target class for its sample, it
// performs a "scatter" write of the final CCE loss value.
__kernel void compute_probs_loss_cce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict softmax_params_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const int *__restrict targets_cce_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict partial_probs_out,
    __global SCALAR_TYPE *__restrict final_loss_out,
    int chunk_id,
    int param_offset,
    int chunk_size,
    int class_offset,
    int num_classes_in_chunk,
    int full_batch_size,
    int total_output_classes) {

    // Map the 3D work-item grid to the logical problem space.
    const uint exit_local_idx  = get_global_id(0);
    const uint batch_idx       = get_global_id(1);
    const uint class_local_idx = get_global_id(2);

    // --- Boundary Checks ---
    if (exit_local_idx >= chunk_size || batch_idx >= full_batch_size || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    // Convert local chunk indices to global indices.
    const uint exit_global_idx  = param_offset + exit_local_idx;
    const uint class_global_idx = class_offset + class_local_idx;
    const uint prob_out_idx     = exit_global_idx * full_batch_size * total_output_classes + batch_idx * total_output_classes + class_global_idx;

    // --- Masking ---
    // If the sample is masked, all its outputs are zero.
    if (targets_mask[batch_idx] < 0.5f) {
        partial_probs_out[prob_out_idx] = SCALAR_ZERO;
        return;
    }

    // --- Probability Calculation ---
    // Read the pre-computed normalization terms from Node 6.
    const uint        params_base_idx  = exit_global_idx * full_batch_size * 2 + batch_idx * 2;
    const SCALAR_TYPE max_scaled_logit = softmax_params_buf[params_base_idx + 0];
    const SCALAR_TYPE sum_exp          = softmax_params_buf[params_base_idx + 1];

    // Read this work-item's specific logit.
    const SCALAR_TYPE logit = full_logits_buf[prob_out_idx];

    // Read temperature.
    const SCALAR_TYPE temp     = temps_buf[exit_global_idx];
    const SCALAR_TYPE temp_inv = 1.0f / temp;

    // Calculate final probability using the pre-computed stable softmax parameters.
    SCALAR_TYPE prob = SCALAR_ZERO;
    if (sum_exp > (SCALAR_TYPE)1e-9f) { // Avoid division by zero for empty samples
        prob = MATH_FN exp((logit * temp_inv) - max_scaled_logit) / sum_exp;
    }

    // Write out the final probability.
    partial_probs_out[prob_out_idx] = prob;

    // --- CCE Loss Calculation (Scatter Operation) ---
    // Read the single integer target class for this sample.
    const int true_class = targets_cce_buf[batch_idx];

    // Check if THIS work-item is responsible for the true class.
    if (class_global_idx == true_class) {
        // Calculate the CCE loss.
        const SCALAR_TYPE loss = -MATH_FN log(fmax(prob, (SCALAR_TYPE)1e-9f));

        // Write the single loss value for this sample. The host is responsible for
        // zero-initializing this buffer, so this write is safe.
        const uint loss_out_idx      = exit_global_idx * full_batch_size + batch_idx;
        final_loss_out[loss_out_idx] = loss;
    }
}

// --- Implementation: compute_probs_loss_bce_chunk (Node 7 - BCE Path) ---
// Strategy: A 2D "map-reduce" kernel dispatched with a grid corresponding to
// (exit, batch_sample). Each work-item is responsible for one sample of one exit.
// It performs two tasks: 1) it maps sigmoid probabilities to the output buffer for its
// assigned class chunk, and 2) it reduces the BCE loss contributions from that
// class chunk into a single partial loss value.
__kernel void compute_probs_loss_bce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const SCALAR_TYPE *__restrict targets_bce_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict partial_probs_out,
    __global SCALAR_TYPE *__restrict partial_loss_out,
    int chunk_id,
    int param_offset,
    int chunk_size,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int full_batch_size,
    int total_output_classes,
    int num_total_exits) {

    // Map the 2D work-item grid to the (exit, batch) problem space.
    const uint exit_local_idx = get_global_id(0);
    const uint batch_idx      = get_global_id(1);

    // --- Boundary Checks ---
    if (exit_local_idx >= chunk_size || batch_idx >= full_batch_size) {
        return;
    }

    const uint exit_global_idx = param_offset + exit_local_idx;
    const uint loss_out_idx    = class_chunk_id * num_total_exits * full_batch_size + exit_global_idx * full_batch_size + batch_idx;

    // --- Masking ---
    // If the sample is masked, its partial loss is zero.
    if (targets_mask[batch_idx] < 0.5f) {
        partial_loss_out[loss_out_idx] = SCALAR_ZERO;
        // Also zero out the corresponding probabilities for this chunk.
        for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
            const int  c_global             = class_offset + c_local;
            const uint prob_out_idx         = exit_global_idx * full_batch_size * total_output_classes + batch_idx * total_output_classes + c_global;
            partial_probs_out[prob_out_idx] = SCALAR_ZERO;
        }
        return;
    }

    // --- Core Logic: Map (Probs) and Reduce (Loss) ---
    // Each work-item computes a partial sum of the loss for its (exit, sample) pair over the class chunk.
    SCALAR_TYPE       partial_loss_sum = SCALAR_ZERO;
    const SCALAR_TYPE temp_inv         = 1.0f / temps_buf[exit_global_idx];

    // This loop performs the reduction.
    for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
        const int  c_global       = class_offset + c_local;
        const uint logit_prob_idx = exit_global_idx * full_batch_size * total_output_classes + batch_idx * total_output_classes + c_global;

        // 1. Read the logit.
        const SCALAR_TYPE logit = full_logits_buf[logit_prob_idx];

        // 2. Compute the sigmoid probability.
        const SCALAR_TYPE prob = 1.0f / (1.0f + MATH_FN exp(-logit * temp_inv));

        // 3. Write out the probability (Map operation).
        partial_probs_out[logit_prob_idx] = prob;

        // 4. Read the target value.
        const SCALAR_TYPE target_val = targets_bce_buf[batch_idx * total_output_classes + c_global];

        // 5. Calculate the BCE loss term and accumulate it (Reduce operation).
        partial_loss_sum -= (target_val * MATH_FN log(fmax(prob, 1e-9f)) + (1.0f - target_val) * MATH_FN log(fmax(1.0f - prob, 1e-9f)));
    }

    // Write the final partial loss sum for this chunk to its unique memory location.
    // This will be aggregated later by the aggregation engine.
    partial_loss_out[loss_out_idx] = partial_loss_sum;
}

// --- Implementation: calculate_exit_param_grads_chunk (Node 8) ---
// Strategy: "Work-group per gradient component" reduction. A 3D dispatch grid
// is used, where each work-group is responsible for computing a single gradient
// component for dL/dW_ehc and dL/dB_ec (where e=exit, h=hidden, c=class).
// Threads within a work-group parallelize the summation over the entire batch
// dimension, followed by a final, fast reduction using __local memory.
__kernel void calculate_exit_param_grads_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global SCALAR_TYPE *__restrict partial_grad_exit_w_out,
    __global SCALAR_TYPE *__restrict partial_grad_exit_b_out,
    int problem_type_flag,
    int chunk_id,
    int param_offset,
    int chunk_size,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int full_batch_size,
    int hidden_dim,
    int padded_hidden_dim,
    int total_output_classes,
    int num_total_exits) {

    // Map the 3D work-group grid to the logical problem space of a single gradient component.
    const uint exit_local_idx  = get_group_id(0);
    const uint h_idx           = get_group_id(1);
    const uint class_local_idx = get_group_id(2);

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- Boundary Checks for Padded Dispatch Grid ---
    if (exit_local_idx >= chunk_size || h_idx >= hidden_dim || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    // --- Convert Local/Chunk Indices to Global Indices ---
    const uint exit_global_idx  = param_offset + exit_local_idx;
    const uint class_global_idx = class_offset + class_local_idx;

    // --- Private Accumulators for each Thread ---
    // These will hold the thread's partial sum over its slice of the batch.
    SCALAR_TYPE p_grad_w = SCALAR_ZERO;
    SCALAR_TYPE p_grad_b = SCALAR_ZERO;

    // --- Parallel Summation over the Batch Dimension ---
    // Each thread handles a strided slice of the batch samples.
    for (int b = lid; b < full_batch_size; b += lsize) {
        // dL/dW_ehc = Sum_b [ (prob_ebc - target_ebc) * h_b_h ]
        // dL/dB_ec  = Sum_b [ (prob_ebc - target_ebc) ]

        // Get the common error signal: d_loss_d_logit = (prob - target)
        const uint        prob_idx = exit_global_idx * full_batch_size * total_output_classes + b * total_output_classes + class_global_idx;
        const SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
        SCALAR_TYPE       d_loss_d_logit;

        if (problem_type_flag == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (__global int *)targets_buf;
            d_loss_d_logit                  = select(prob, prob - 1.0f, class_global_idx == targets_cce[b]);
        } else { // PROBLEM_TYPE_BCE
            const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
            d_loss_d_logit                          = prob - targets_bce[b * total_output_classes + class_global_idx];
        }

        // Get the hidden activation value for this sample and hidden neuron.
        const SCALAR_TYPE h_val = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(b, h_idx, padded_hidden_dim)];

        // Accumulate gradient for the weight.
        p_grad_w += d_loss_d_logit * h_val;

        // The bias gradient is independent of the hidden dimension 'h'.
        // To avoid redundant computation and writes, we designate only the
        // work-groups with h_idx == 0 to calculate it.
        if (h_idx == 0) {
            p_grad_b += d_loss_d_logit;
        }
    }

    // --- Intra-Workgroup Reduction for Weight Gradient ---
    local_mem[lid] = p_grad_w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_mem[lid] += local_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes the final reduced value for this component.
    if (lid == 0) {
        const uint out_idx
            = class_chunk_id * (num_total_exits * hidden_dim * total_output_classes) + exit_global_idx * (hidden_dim * total_output_classes) + h_idx * (total_output_classes) + class_global_idx;
        partial_grad_exit_w_out[out_idx] = local_mem[0];
    }

    // --- Intra-Workgroup Reduction for Bias Gradient (only for h_idx == 0) ---
    if (h_idx == 0) {
        local_mem[lid] = p_grad_b;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                local_mem[lid] += local_mem[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            const uint out_idx               = class_chunk_id * (num_total_exits * total_output_classes) + exit_global_idx * (total_output_classes) + class_global_idx;
            partial_grad_exit_b_out[out_idx] = local_mem[0];
        }
    }
}

// --- Implementation: backprop_error_to_hidden_chunk (Node 9) ---
// Strategy: A pure 3D "map" kernel. Each work-item is dispatched to compute
// one output value in the partial Grad_H tensor. The grid dimensions map to
// (exit_chunk, batch_sample, hidden_neuron). Each work-item performs a small,
// independent reduction (a sum) over the classes within its assigned class_chunk.
// This kernel generates one component of the full Grad_H, which will be
// aggregated later.
__kernel void backprop_error_to_hidden_chunk(
    __local SCALAR_TYPE *local_mem, // Unused in this map-style kernel, present for signature compatibility.
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,
    int problem_type_flag,
    int chunk_id,
    int param_offset,
    int chunk_size,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int full_batch_size,
    int hidden_dim,
    int total_output_classes,
    int num_total_exits) {

    // Map the 3D work-item grid to the logical problem space.
    const uint exit_local_idx = get_global_id(0); // Index within the current EXIT chunk
    const uint batch_idx      = get_global_id(1); // Global batch index
    const uint h_idx          = get_global_id(2); // Global hidden dimension index

    // --- Boundary Checks for Padded Dispatch Grid ---
    if (exit_local_idx >= chunk_size || batch_idx >= full_batch_size || h_idx >= hidden_dim) {
        return;
    }

    // --- Convert Local/Chunk Index to Global Index ---
    const uint exit_global_idx = param_offset + exit_local_idx;

    // --- Core Logic: Reduction over the Class Chunk ---
    // dL/dH_beh = Sum_c [ dL/dLogit_ebc * dLogit_ebc/dH_beh ]
    //           = Sum_c [ (prob_ebc - target_ebc) * W_ehc ]
    // Each work-item computes this sum for its assigned (e, b, h) over the classes in the chunk.
    SCALAR_TYPE grad_h_accum = SCALAR_ZERO;

    for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
        const int c_global = class_offset + c_local;

        // 1. Get the error signal: d_loss_d_logit = (prob - target)
        const uint        prob_idx = exit_global_idx * full_batch_size * total_output_classes + batch_idx * total_output_classes + c_global;
        const SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
        SCALAR_TYPE       d_loss_d_logit;

        if (problem_type_flag == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (__global int *)targets_buf;
            d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[batch_idx]);
        } else { // PROBLEM_TYPE_BCE
            const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
            d_loss_d_logit                          = prob - targets_bce[batch_idx * total_output_classes + c_global];
        }

        // 2. Get the corresponding weight: W_ehc
        const uint        weight_idx = exit_global_idx * hidden_dim * total_output_classes + h_idx * total_output_classes + c_global;
        const SCALAR_TYPE weight_val = exit_weights_buf[weight_idx];

        // 3. Accumulate the product.
        grad_h_accum += d_loss_d_logit * weight_val;
    }

    // --- Write Final Partial Gradient to Global Memory ---
    // The output buffer stores the partial gradient contribution from this specific
    // (class_chunk, exit, batch, hidden) combination.
    const uint out_idx = class_chunk_id * (num_total_exits * full_batch_size * hidden_dim) + exit_global_idx * (full_batch_size * hidden_dim) + batch_idx * hidden_dim + h_idx;

    partial_grad_h_aos_out[out_idx] = grad_h_accum;
}

// --- Implementation: calculate_chunk_temp_gradients (Node 10) ---
// Strategy: "Work-group per exit" reduction. A work-group is responsible for
// computing the partial temperature gradient for one exit, based on the contribution
// from the current class chunk. Threads parallelize the summation over the batch,
// followed by a final reduction using __local memory.
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,
    int problem_type_flag,
    int chunk_id,
    int param_offset,
    int chunk_size,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int full_batch_size,
    int total_output_classes,
    int num_total_exits) {

    // A whole work-group computes the gradient for a single exit.
    const uint exit_local_idx = get_group_id(0);
    const uint lid            = get_local_id(0);
    const uint lsize          = get_local_size(0);

    // Guard against work-groups for padded exits in the dispatch grid.
    if (exit_local_idx >= chunk_size) {
        return;
    }

    const uint  exit_global_idx = param_offset + exit_local_idx;
    SCALAR_TYPE p_grad_sum      = SCALAR_ZERO; // Private accumulator for each thread.

    // Each thread sums contributions for a strided slice of the batch.
    for (int b = lid; b < full_batch_size; b += lsize) {
        if (targets_mask[b] < 0.5f) {
            continue;
        }

        SCALAR_TYPE grad_contribution_for_sample = SCALAR_ZERO;

        // Sum contributions from the assigned CLASS CHUNK.
        for (int c_local = 0; c_local < num_classes_in_chunk; c_local++) {
            const int c_global = class_offset + c_local;

            const uint        base_idx = exit_global_idx * full_batch_size * total_output_classes + b * total_output_classes + c_global;
            const SCALAR_TYPE prob     = partial_probs_buf[base_idx];
            SCALAR_TYPE       d_loss_d_logit;

            // d_loss_d_logit = prob - target
            if (problem_type_flag == PROBLEM_TYPE_CCE) {
                const __global int *targets_cce = (__global int *)targets_buf;
                d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[b]);
            } else { // PROBLEM_TYPE_BCE
                const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
                d_loss_d_logit                          = prob - targets_bce[b * total_output_classes + c_global];
            }

            // The 'unscaled_logit' is just the value from full_logits_buf.
            const SCALAR_TYPE unscaled_logit = full_logits_buf[base_idx];
            grad_contribution_for_sample += d_loss_d_logit * unscaled_logit;
        }
        p_grad_sum += grad_contribution_for_sample;
    }

    // --- Intra-Workgroup Reduction ---
    // Sum the partial sums from each thread in the work-group.
    local_mem[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_mem[lid] += local_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes the final partial result for this class chunk.
    if (lid == 0) {
        // This is the complete sum over the batch for this exit and this class chunk.
        const SCALAR_TYPE total_sum_for_chunk = local_mem[0];
        const SCALAR_TYPE temp                = temps_buf[exit_global_idx];

        // Apply the d(logit)/dT term, which is (-1 / T^2). This factor is common
        // to all class contributions, so we can apply it after summing them.
        const SCALAR_TYPE final_partial_grad = total_sum_for_chunk * (-1.0f / (temp * temp));

        // Write to the unique location for this (class_chunk, exit) pair.
        const uint out_idx              = class_chunk_id * num_total_exits + exit_global_idx;
        partial_grad_temps_out[out_idx] = final_partial_grad;
    }
}