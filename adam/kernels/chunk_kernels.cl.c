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
        const uint input_idx = effective_bid * padded_input_dim + t + lid;
        if (t + lid < padded_input_dim) {
            tile_input[lid] = input_buf[input_idx];
        } else {
            tile_input[lid] = SCALAR_ZERO;
        }

        // 2. Load a tile of the weight matrix. Each thread `lid` cooperatively loads
        //    the column of the weight tile corresponding to its output neuron.
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
    int exit_chunk_id,
    int exit_param_offset,
    int num_exits_in_chunk,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int hidden_dim,
    int padded_hidden_dim,
    int total_output_classes) {

    // Map the 3D work-item grid to the logical problem space.
    const uint exit_local_idx  = get_global_id(0); // Index within the current EXIT chunk
    const uint batch_idx       = get_global_id(1); // Global batch index
    const uint class_local_idx = get_global_id(2); // Index within the current CLASS chunk

    // --- Boundary Checks ---
    if (exit_local_idx >= num_exits_in_chunk || batch_idx >= total_batch_size || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    // Convert local chunk indices to global indices.
    const uint exit_global_idx  = exit_param_offset + exit_local_idx;
    const uint class_global_idx = class_offset + class_local_idx;

    // The output index for this specific work-item's logit.
    const uint out_idx = exit_global_idx * total_batch_size * total_output_classes + batch_idx * total_output_classes + class_global_idx;

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
        // HARMONIZED: Use the macro for consistent, safe access to the hidden buffer.
        const SCALAR_TYPE h_val = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(batch_idx, h, padded_hidden_dim)];

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
    int total_exits,
    int total_batch_size,
    int total_output_classes) {

    // Map the 2D work-item grid to the (exit, batch) problem space.
    const uint exit_idx  = get_global_id(0);
    const uint batch_idx = get_global_id(1);

    // Boundary checks
    if (exit_idx >= total_exits || batch_idx >= total_batch_size) {
        return;
    }

    // Base index for all data related to this specific (exit, batch) pair.
    const uint base_in_idx  = exit_idx * total_batch_size * total_output_classes + batch_idx * total_output_classes;
    const uint base_out_idx = exit_idx * total_batch_size * 2 + batch_idx * 2;

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
    int exit_chunk_id,
    int exit_param_offset,
    int num_exits_in_chunk,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int total_output_classes) {

    // Map the 3D work-item grid to the logical problem space.
    const uint exit_local_idx  = get_global_id(0);
    const uint batch_idx       = get_global_id(1);
    const uint class_local_idx = get_global_id(2);

    // --- Boundary Checks ---
    if (exit_local_idx >= num_exits_in_chunk || batch_idx >= total_batch_size || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    // Convert local chunk indices to global indices.
    const uint exit_global_idx  = exit_param_offset + exit_local_idx;
    const uint class_global_idx = class_offset + class_local_idx;
    const uint prob_out_idx     = exit_global_idx * total_batch_size * total_output_classes + batch_idx * total_output_classes + class_global_idx;

    // --- Masking ---
    if (targets_mask[batch_idx] < 0.5f) {
        partial_probs_out[prob_out_idx] = SCALAR_ZERO;
        return;
    }

    // --- Probability Calculation ---
    const uint        params_base_idx  = exit_global_idx * total_batch_size * 2 + batch_idx * 2;
    const SCALAR_TYPE max_scaled_logit = softmax_params_buf[params_base_idx + 0];
    const SCALAR_TYPE sum_exp          = softmax_params_buf[params_base_idx + 1];
    const SCALAR_TYPE logit            = full_logits_buf[prob_out_idx];
    const SCALAR_TYPE temp_inv         = 1.0f / temps_buf[exit_global_idx];

    SCALAR_TYPE prob = SCALAR_ZERO;
    if (sum_exp > (SCALAR_TYPE)1e-9f) {
        prob = MATH_FN exp((logit * temp_inv) - max_scaled_logit) / sum_exp;
    }
    partial_probs_out[prob_out_idx] = prob;

    // --- CCE Loss Calculation (Scatter Operation) ---
    const int true_class = targets_cce_buf[batch_idx];
    if (class_global_idx == true_class) {
        const SCALAR_TYPE loss                         = -MATH_FN log(fmax(prob, (SCALAR_TYPE)1e-9f));
        const uint                        loss_out_idx = exit_global_idx * total_batch_size + batch_idx;
        final_loss_out[loss_out_idx]                   = loss;
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
    int exit_chunk_id,
    int exit_param_offset,
    int num_exits_in_chunk,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int total_output_classes,
    int total_exits) {

    // Map the 2D work-item grid to the (exit, batch) problem space.
    const uint exit_local_idx = get_global_id(0);
    const uint batch_idx      = get_global_id(1);

    // --- Boundary Checks ---
    if (exit_local_idx >= num_exits_in_chunk || batch_idx >= total_batch_size) {
        return;
    }

    const uint exit_global_idx = exit_param_offset + exit_local_idx;
    const uint loss_out_idx    = class_chunk_id * total_exits * total_batch_size + exit_global_idx * total_batch_size + batch_idx;

    // --- Masking ---
    if (targets_mask[batch_idx] < 0.5f) {
        partial_loss_out[loss_out_idx] = SCALAR_ZERO;
        for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
            const int  c_global             = class_offset + c_local;
            const uint prob_out_idx         = exit_global_idx * total_batch_size * total_output_classes + batch_idx * total_output_classes + c_global;
            partial_probs_out[prob_out_idx] = SCALAR_ZERO;
        }
        return;
    }

    // --- Core Logic: Map (Probs) and Reduce (Loss) ---
    SCALAR_TYPE       partial_loss_sum = SCALAR_ZERO;
    const SCALAR_TYPE temp_inv         = 1.0f / temps_buf[exit_global_idx];

    for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
        const int         c_global       = class_offset + c_local;
        const uint        logit_prob_idx = exit_global_idx * total_batch_size * total_output_classes + batch_idx * total_output_classes + c_global;
        const SCALAR_TYPE logit          = full_logits_buf[logit_prob_idx];
        const SCALAR_TYPE prob           = 1.0f / (1.0f + MATH_FN exp(-logit * temp_inv));

        partial_probs_out[logit_prob_idx] = prob;

        const SCALAR_TYPE target_val = targets_bce_buf[batch_idx * total_output_classes + c_global];
        partial_loss_sum -= (target_val * MATH_FN log(fmax(prob, 1e-9f)) + (1.0f - target_val) * MATH_FN log(fmax(1.0f - prob, 1e-9f)));
    }
    partial_loss_out[loss_out_idx] = partial_loss_sum;
}

// --- Implementation: calculate_exit_param_grads_chunk (Node 8) ---
// Strategy: "Work-group per gradient component" reduction. A 3D dispatch grid
// is used, where each work-group is responsible for computing a single gradient
// component for dL/dW_ehc and dL/dB_ec (where e=exit, h=hidden, c=class).
// Threads within a work-group parallelize the summation over the entire batch
// dimension. Two sequential reductions using __local memory are performed to
// safely compute bias and weight gradients without race conditions.
__kernel void calculate_exit_param_grads_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global SCALAR_TYPE *__restrict partial_grad_exit_w_out,
    __global SCALAR_TYPE *__restrict partial_grad_exit_b_out,
    int problem_type_flag,
    int exit_chunk_id,
    int exit_param_offset,
    int num_exits_in_chunk,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int hidden_dim,
    int padded_hidden_dim,
    int total_output_classes,
    int total_exits) {

    const uint exit_local_idx  = get_group_id(0);
    const uint h_idx           = get_group_id(1);
    const uint class_local_idx = get_group_id(2);
    const uint lid             = get_local_id(0);
    const uint lsize           = get_local_size(0);

    // --- Boundary Checks ---
    if (exit_local_idx >= num_exits_in_chunk || h_idx >= hidden_dim || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    const uint exit_global_idx  = exit_param_offset + exit_local_idx;
    const uint class_global_idx = class_offset + class_local_idx;

    SCALAR_TYPE p_grad_w = SCALAR_ZERO;
    SCALAR_TYPE p_grad_b = SCALAR_ZERO;

    // --- Parallel Summation over the Batch Dimension ---
    // Each thread calculates its local contribution to the gradients.
    for (int b = lid; b < total_batch_size; b += lsize) {
        const uint        prob_idx = exit_global_idx * total_batch_size * total_output_classes + b * total_output_classes + class_global_idx;
        const SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
        SCALAR_TYPE       d_loss_d_logit;

        if (problem_type_flag == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (__global int *)targets_buf;
            d_loss_d_logit                  = select(prob, prob - 1.0f, class_global_idx == targets_cce[b]);
        } else { // PROBLEM_TYPE_BCE
            const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
            d_loss_d_logit                          = prob - targets_bce[b * total_output_classes + class_global_idx];
        }

        const SCALAR_TYPE h_val = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(b, h_idx, padded_hidden_dim)];
        p_grad_w += d_loss_d_logit * h_val;

        // Bias gradient component is only needed once per (exit, class).
        // It's arbitrarily but deterministically computed by threads with h_idx=0.
        if (h_idx == 0) {
            p_grad_b += d_loss_d_logit;
        }
    }

    // --- serialized Intra-Workgroup Reductions ---

    // STAGE 1: Reduce Bias Gradient (only h_idx=0 threads participate).
    // These threads use the shared local_mem first to calculate their result.
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
            const uint out_idx               = class_chunk_id * (total_exits * total_output_classes) + exit_global_idx * (total_output_classes) + class_global_idx;
            partial_grad_exit_b_out[out_idx] = local_mem[0];
        }
    }

    // SYNCHRONIZATION POINT: All threads in the work-group must wait here.
    // This ensures Stage 1 is complete before anyone starts Stage 2, preventing
    // the weight gradient calculation from corrupting the bias gradient calculation.
    barrier(CLK_LOCAL_MEM_FENCE);

    // STAGE 2: Reduce Weight Gradient (all threads participate).
    // Now that local_mem is free, all threads use it to reduce the weight gradient.
    local_mem[lid] = p_grad_w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_mem[lid] += local_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const uint out_idx
            = class_chunk_id * (total_exits * hidden_dim * total_output_classes) + exit_global_idx * (hidden_dim * total_output_classes) + h_idx * (total_output_classes) + class_global_idx;
        partial_grad_exit_w_out[out_idx] = local_mem[0];
    }
}

// --- Implementation: backprop_error_to_hidden_chunk (Node 9) ---
// Strategy: A pure 3D "map" kernel. Each work-item is dispatched to compute
// one output value in the partial Grad_H tensor. The grid dimensions map to
// (exit_chunk, batch_sample, hidden_neuron). Each work-item performs a small,
// independent reduction (a sum) over the classes within its assigned class_chunk.
__kernel void backprop_error_to_hidden_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,
    int problem_type_flag,
    int exit_chunk_id,
    int exit_param_offset,
    int num_exits_in_chunk,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int hidden_dim,
    int total_output_classes,
    int total_exits) {

    // Map the 3D work-item grid to the logical problem space.
    const uint exit_local_idx = get_global_id(0);
    const uint batch_idx      = get_global_id(1);
    const uint h_idx          = get_global_id(2);

    // --- Boundary Checks ---
    if (exit_local_idx >= num_exits_in_chunk || batch_idx >= total_batch_size || h_idx >= hidden_dim) {
        return;
    }

    const uint exit_global_idx = exit_param_offset + exit_local_idx;

    // --- Core Logic: Reduction over the Class Chunk ---
    SCALAR_TYPE grad_h_accum = SCALAR_ZERO;
    for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
        const int c_global = class_offset + c_local;

        const uint        prob_idx = exit_global_idx * total_batch_size * total_output_classes + batch_idx * total_output_classes + c_global;
        const SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
        SCALAR_TYPE       d_loss_d_logit;

        if (problem_type_flag == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (__global int *)targets_buf;
            d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[batch_idx]);
        } else { // PROBLEM_TYPE_BCE
            const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
            d_loss_d_logit                          = prob - targets_bce[batch_idx * total_output_classes + c_global];
        }

        const uint        weight_idx = exit_global_idx * hidden_dim * total_output_classes + h_idx * total_output_classes + c_global;
        const SCALAR_TYPE weight_val = exit_weights_buf[weight_idx];
        grad_h_accum += d_loss_d_logit * weight_val;
    }

    // --- Write Final Partial Gradient ---
    const uint out_idx              = class_chunk_id * (total_exits * total_batch_size * hidden_dim) + exit_global_idx * (total_batch_size * hidden_dim) + batch_idx * hidden_dim + h_idx;
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
    int exit_chunk_id,
    int exit_param_offset,
    int num_exits_in_chunk,
    int class_chunk_id,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int total_output_classes,
    int total_exits) {

    const uint exit_local_idx = get_group_id(0);
    const uint lid            = get_local_id(0);
    const uint lsize          = get_local_size(0);

    if (exit_local_idx >= num_exits_in_chunk) {
        return;
    }

    const uint  exit_global_idx = exit_param_offset + exit_local_idx;
    SCALAR_TYPE p_grad_sum      = SCALAR_ZERO;

    for (int b = lid; b < total_batch_size; b += lsize) {
        if (targets_mask[b] < 0.5f) {
            continue;
        }

        SCALAR_TYPE grad_contribution_for_sample = SCALAR_ZERO;
        for (int c_local = 0; c_local < num_classes_in_chunk; c_local++) {
            const int         c_global = class_offset + c_local;
            const uint        base_idx = exit_global_idx * total_batch_size * total_output_classes + b * total_output_classes + c_global;
            const SCALAR_TYPE prob     = partial_probs_buf[base_idx];
            SCALAR_TYPE       d_loss_d_logit;

            if (problem_type_flag == PROBLEM_TYPE_CCE) {
                const __global int *targets_cce = (__global int *)targets_buf;
                d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[b]);
            } else { // PROBLEM_TYPE_BCE
                const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
                d_loss_d_logit                          = prob - targets_bce[b * total_output_classes + c_global];
            }

            const SCALAR_TYPE unscaled_logit = full_logits_buf[base_idx];
            grad_contribution_for_sample += d_loss_d_logit * unscaled_logit;
        }
        p_grad_sum += grad_contribution_for_sample;
    }

    // --- Intra-Workgroup Reduction ---
    local_mem[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_mem[lid] += local_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const SCALAR_TYPE total_sum_for_chunk = local_mem[0];
        const SCALAR_TYPE temp                = temps_buf[exit_global_idx];
        const SCALAR_TYPE final_partial_grad  = total_sum_for_chunk * (-1.0f / (temp * temp));
        const uint        out_idx             = class_chunk_id * total_exits + exit_global_idx;
        partial_grad_temps_out[out_idx]       = final_partial_grad;
    }
}
