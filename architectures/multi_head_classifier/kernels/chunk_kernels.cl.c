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
    __global const SCALAR_TYPE *__restrict sample_mask,
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

    // Propagate validity mask. Masked samples do not need the barrier, so they exit early.
    if (sample_mask[effective_bid] < (SCALAR_TYPE)0.5f) {
        if (lid == 0) {
            hidden_mask_out[effective_bid] = sample_mask[effective_bid];
        }
        return;
    }
    // Only one thread writes the valid mask to avoid a race condition.
    if (lid == 0) {
        hidden_mask_out[effective_bid] = sample_mask[effective_bid];
    }

    const uint TILE_SIZE = SIMD_WIDTH;
    // This partitioning dedicates local memory to a shared input tile (reused by all threads) and the locally-owned weight columns.
    __local SCALAR_TYPE *tile_input   = local_mem;             // Size: TILE_SIZE
    __local SCALAR_TYPE *tile_weights = &local_mem[TILE_SIZE]; // Size: TILE_SIZE * TILE_SIZE

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

// --- Implementation: render_logits_chunk (Node 5) ---
// Strategy: A pure 3D "map" kernel where each work-item computes one logit. It
// is designed to be padding-aware, using the physical stride of the class
// dimension (`padded_total_output_classes`) for all memory index calculations.
// This ensures correct addressing into buffers that are padded for performance.
__kernel void render_logits_chunk(
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict module_weights_buf,
    __global const SCALAR_TYPE *__restrict module_biases_buf,
    __global SCALAR_TYPE *__restrict full_logits_out,
    int module_batch_chunk_index,
    int module_param_offset,
    int num_modules_in_chunk,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int hidden_dim,
    int padded_hidden_dim,
    int total_output_classes,
    int padded_total_output_classes) {

    // Map the 3D work-item grid to the logical (module, batch, class) space.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);
    const uint class_local_idx  = get_global_id(2);

    // Boundary check against the logical chunk dimensions.
    if (module_local_idx >= num_modules_in_chunk || batch_idx >= total_batch_size || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    // Establish global (module, class) coordinates for this work-item.
    const uint module_global_idx = module_param_offset + module_local_idx;
    const uint class_global_idx  = class_offset + class_local_idx;

    // CRITICAL: Use the physical stride `padded_total_output_classes` for correct addressing.
    const long out_idx = (long)module_global_idx * total_batch_size * padded_total_output_classes + (long)batch_idx * padded_total_output_classes + class_global_idx;

    // Skip computation for masked-out samples.
    if (hidden_mask[batch_idx] < 0.5f) {
        full_logits_out[out_idx] = SCALAR_ZERO;
        return;
    }

    // Initialize accumulator with the bias, using the physical stride.
    SCALAR_TYPE logit = module_biases_buf[(long)module_global_idx * padded_total_output_classes + class_global_idx];

    // Compute the dot product using the physical stride for the weight lookup.
    for (int h = 0; h < hidden_dim; h++) {
        const SCALAR_TYPE h_val      = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(batch_idx, h, padded_hidden_dim)];
        const long        weight_idx = (long)module_global_idx * hidden_dim * padded_total_output_classes + (long)h * padded_total_output_classes + class_global_idx;
        const SCALAR_TYPE w_val      = module_weights_buf[weight_idx];
        logit += h_val * w_val;
    }

    full_logits_out[out_idx] = logit;
}

// --- Implementation: reduce_logits_for_softmax (Node 6) ---
// Strategy: A 2D "map" kernel where each work-item (module, batch_sample) performs
// a stable two-pass reduction over the class dimension. It is padding-aware, using
// the physical class stride for correct memory addressing into the logit buffer.
__kernel void reduce_logits_for_softmax(
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global SCALAR_TYPE *__restrict softmax_params_out,
    int total_modules,
    int total_batch_size,
    int total_output_classes,
    int padded_total_output_classes) {

    // Map work-item to a (module, batch) pair.
    const uint module_idx = get_global_id(0);
    const uint batch_idx  = get_global_id(1);

    if (module_idx >= total_modules || batch_idx >= total_batch_size) {
        return;
    }

    // CRITICAL: Calculate base input index using the physical stride.
    const long base_in_idx = (long)module_idx * total_batch_size * padded_total_output_classes + (long)batch_idx * padded_total_output_classes;
    // Output index is unaffected by class padding as its last dimension is 2.
    const uint base_out_idx = module_idx * total_batch_size * 2 + batch_idx * 2;

    const SCALAR_TYPE temp     = temps_buf[module_idx];
    const SCALAR_TYPE temp_inv = 1.0f / temp;

    // --- Pass 1: Find max logit for numerical stability ---
    // This is essential for the stability of the `exp` function in the next pass.
    SCALAR_TYPE max_scaled_logit = -FLT_MAX;
    // The loop iterates over the LOGICAL number of classes.
    for (int c = 0; c < total_output_classes; c++) {
        // The indexing is now correct because base_in_idx used the physical stride.
        const SCALAR_TYPE logit = full_logits_buf[base_in_idx + c];
        max_scaled_logit        = fmax(max_scaled_logit, logit * temp_inv);
    }

    // For cases where all logits are -inf (e.g., a masked sample), this prevents `max_scaled_logit`
    // from propagating a -FLT_MAX, which could lead to NaNs in later calculations.
    if (max_scaled_logit == -FLT_MAX) {
        max_scaled_logit = 0.0f;
    }

    // --- Pass 2: Calculate sum of exponentials using the max for stability ---
    SCALAR_TYPE sum_exp = SCALAR_ZERO;
    for (int c = 0; c < total_output_classes; c++) {
        const SCALAR_TYPE logit = full_logits_buf[base_in_idx + c];
        // The log-sum-exp trick: subtracting the max before `exp` prevents overflow to +inf
        // and improves precision for values that would otherwise underflow to zero.
        sum_exp += MATH_FN exp((logit * temp_inv) - max_scaled_logit);
    }

    softmax_params_out[base_out_idx + 0] = max_scaled_logit;
    softmax_params_out[base_out_idx + 1] = sum_exp;
}

// --- Implementation: compute_probs_loss_cce_chunk (Node 7a - CCE Path) ---
// Strategy: A 3D "map" kernel where each work-item computes one probability. It's
// padding-aware, using the physical class stride for correct addressing. The
// work-item matching the target class performs a race-free "scatter" write of
// the final CCE loss, eliminating the need for a separate reduction.
__kernel void compute_probs_loss_cce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict softmax_params_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const int *__restrict targets_cce_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global SCALAR_TYPE *__restrict partial_probs_out,
    __global SCALAR_TYPE *__restrict final_loss_out,
    int module_batch_chunk_index,
    int module_param_offset,
    int num_modules_in_chunk,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int total_output_classes,
    int padded_total_output_classes) {

    // Map the 3D work-item grid to the logical (module, batch, class) space.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);
    const uint class_local_idx  = get_global_id(2);

    // Boundary check against the logical chunk dimensions.
    if (module_local_idx >= num_modules_in_chunk || batch_idx >= total_batch_size || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    const uint module_global_idx = module_param_offset + module_local_idx;
    const uint class_global_idx  = class_offset + class_local_idx;

    // CRITICAL: Use the physical stride for correct index calculation into class-dimensioned buffers.
    const long prob_out_idx = (long)module_global_idx * total_batch_size * padded_total_output_classes + (long)batch_idx * padded_total_output_classes + class_global_idx;

    // Skip computation for masked-out samples.
    if (sample_mask[batch_idx] < 0.5f) {
        partial_probs_out[prob_out_idx] = SCALAR_ZERO;
        return;
    }

    // --- Probability Calculation (using pre-computed Softmax parameters) ---
    const uint        params_base_idx  = module_global_idx * total_batch_size * 2 + batch_idx * 2;
    const SCALAR_TYPE max_scaled_logit = softmax_params_buf[params_base_idx + 0]; // For stability
    const SCALAR_TYPE sum_exp          = softmax_params_buf[params_base_idx + 1]; // Normalizer
    const SCALAR_TYPE logit            = full_logits_buf[prob_out_idx];
    const SCALAR_TYPE temp_inv         = 1.0f / temps_buf[module_global_idx];

    SCALAR_TYPE prob = SCALAR_ZERO;
    if (sum_exp > (SCALAR_TYPE)1e-9f) {
        prob = MATH_FN exp((logit * temp_inv) - max_scaled_logit) / sum_exp;
    }
    partial_probs_out[prob_out_idx] = prob;

    // --- CCE Loss Calculation (Scatter Write) ---
    // This kernel fuses probability calculation with a non-atomic scatter-write for the loss, avoiding a separate reduction step.
    // The single work-item matching the true class writes the loss for its sample. This is a safe, race-free operation
    // because each (module, batch_sample) pair has exactly one true class, ensuring no two work-items write to the same address.
    const int true_class = targets_cce_buf[batch_idx];
    if (class_global_idx == true_class) {
        const SCALAR_TYPE loss                         = -MATH_FN log(fmax(prob, (SCALAR_TYPE)1e-9f));
        const uint                        loss_out_idx = module_global_idx * total_batch_size + batch_idx;
        final_loss_out[loss_out_idx]                   = loss;
    }
}

// --- Implementation: compute_probs_loss_bce_chunk (Node 7b - BCE Path) ---
// Strategy: A tile-based kernel where each work-item processes one
// (module, batch_sample) pair. It is a "Partial Renderer" for both outputs.
// It uses the host-provided `flat_tile_index` to calculate the base write offset for
// this tile's results, ensuring that parallel invocations write to unique,
// non-overlapping regions of the `partial_probs_out` and `partial_loss_out`
// collection buffers.
__kernel void compute_probs_loss_bce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const SCALAR_TYPE *__restrict targets_bce_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global SCALAR_TYPE *__restrict partial_probs_out,
    __global SCALAR_TYPE *__restrict partial_loss_out,
    int module_batch_chunk_index,
    int module_param_offset,
    int num_modules_in_chunk,
    int class_batch_chunk_index,
    int flat_tile_index,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int total_output_classes,
    int padded_total_output_classes,
    int total_modules) {

    // Map work-item to a (module, batch) pair within the current module chunk.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);

    // Boundary check against the logical work dimensions.
    if (module_local_idx >= num_modules_in_chunk || batch_idx >= total_batch_size) {
        return;
    }

    const uint module_global_idx = module_param_offset + module_local_idx;

    // --- Calculate base write offsets for this tile using the flat_tile_index ---
    const long loss_tile_base_offset = (long)flat_tile_index * num_modules_in_chunk * total_batch_size;
    const long prob_tile_base_offset = (long)flat_tile_index * num_modules_in_chunk * total_batch_size * num_classes_in_chunk;

    const long loss_out_idx = loss_tile_base_offset + (long)module_local_idx * total_batch_size + batch_idx;

    // Skip computation for masked samples.
    if (sample_mask[batch_idx] < 0.5f) {
        partial_loss_out[loss_out_idx] = SCALAR_ZERO;
        for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
            // Write zero to the correct slice of the partial probabilities collection buffer.
            const long prob_out_idx         = prob_tile_base_offset + ((long)module_local_idx * total_batch_size + batch_idx) * num_classes_in_chunk + c_local;
            partial_probs_out[prob_out_idx] = SCALAR_ZERO;
        }
        return;
    }

    SCALAR_TYPE       partial_loss_accum = SCALAR_ZERO;
    const SCALAR_TYPE temp_inv           = 1.0f / temps_buf[module_global_idx];

    // Each work-item iterates through its assigned slice of classes, fusing two operations:
    // 1. MAP: Computing and writing the Sigmoid probability for each class to its tile-local slot.
    // 2. REDUCE: Accumulating the BCE loss contributions from each class.
    for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
        const int c_global = class_offset + c_local;

        // READ indices are unchanged; they read from full input buffers.
        const long logit_read_idx  = (long)module_global_idx * total_batch_size * padded_total_output_classes + (long)batch_idx * padded_total_output_classes + c_global;
        const long target_read_idx = (long)batch_idx * padded_total_output_classes + c_global;

        const SCALAR_TYPE logit = full_logits_buf[logit_read_idx];

        // 1. MAP: Compute and write the Sigmoid probability.
        const SCALAR_TYPE prob = 1.0f / (1.0f + MATH_FN exp(-logit * temp_inv));

        // --- Write to the unique slot for this tile. ---
        const long prob_out_idx         = prob_tile_base_offset + ((long)module_local_idx * total_batch_size + batch_idx) * num_classes_in_chunk + c_local;
        partial_probs_out[prob_out_idx] = prob;

        // 2. REDUCE: Accumulate the BCE loss.
        const SCALAR_TYPE target_val = targets_bce_buf[target_read_idx];
        const SCALAR_TYPE term1      = target_val * MATH_FN log(fmax(prob, 1e-9f));
        const SCALAR_TYPE term2      = (1.0f - target_val) * MATH_FN log(fmax(1.0f - prob, 1e-9f));
        partial_loss_accum += term1 + term2;
    }

    // Write the final, negated partial loss sum to the unique slot for this tile.
    partial_loss_out[loss_out_idx] = -partial_loss_accum;
}

// --- Implementation: calculate_module_param_grads_chunk (Node 8) ---
// Strategy: A "work-group per gradient component" kernel that computes the partial
// gradients for a single tile. It accepts a unique `flat_tile_index` from the host,
// which it uses to calculate a base offset into the large `partial_grad_*_out`
// collection buffers. Each work-group computes one scalar gradient value for the
// current tile (e.g., dL/dW_ehc) by reducing over the batch dimension. This ensures
// each tile's complete gradient result is written to a unique, non-overlapping block.
__kernel void calculate_module_param_grads_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global SCALAR_TYPE *__restrict partial_grad_module_w_out,
    __global SCALAR_TYPE *__restrict partial_grad_module_b_out,
    int problem_type_flag,
    int module_batch_chunk_index,
    int module_param_offset,
    int num_modules_in_chunk,
    int class_batch_chunk_index,
    int flat_tile_index,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int hidden_dim,
    int padded_hidden_dim,
    int total_output_classes,
    int padded_total_output_classes,
    int total_modules) {

    const uint module_local_idx = get_group_id(0);
    const uint h_idx            = get_group_id(1);
    const uint class_local_idx  = get_group_id(2);
    const uint lid              = get_local_id(0);
    const uint lsize            = get_local_size(0);

    // Boundary check against the logical chunk dimensions.
    if (module_local_idx >= num_modules_in_chunk || h_idx >= hidden_dim || class_local_idx >= num_classes_in_chunk) {
        return;
    }

    const uint module_global_idx = module_param_offset + module_local_idx;
    const uint class_global_idx  = class_offset + class_local_idx;

    SCALAR_TYPE p_grad_w = SCALAR_ZERO;
    SCALAR_TYPE p_grad_b = SCALAR_ZERO;

    // Threads in the work-group sum contributions over the batch dimension.
    for (int b = lid; b < total_batch_size; b += lsize) {
        if (sample_mask[b] < 0.5f) {
            continue;
        }
        // CRITICAL: Use physical stride for class-dimensioned buffer access.
        const long        prob_idx = (long)module_global_idx * total_batch_size * padded_total_output_classes + (long)b * padded_total_output_classes + class_global_idx;
        const SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
        SCALAR_TYPE       d_loss_d_logit;

        if (problem_type_flag == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (__global int *)targets_buf;
            d_loss_d_logit                  = select(prob, prob - 1.0f, class_global_idx == targets_cce[b]);
        } else { // PROBLEM_TYPE_BCE
            const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
            d_loss_d_logit                          = prob - targets_bce[(long)b * padded_total_output_classes + class_global_idx];
        }

        const SCALAR_TYPE h_val = hidden_buf[GET_PHYSICAL_HIDDEN_IDX(b, h_idx, padded_hidden_dim)];
        p_grad_w += d_loss_d_logit * h_val;

        if (h_idx == 0) {
            p_grad_b += d_loss_d_logit;
        }
    }

    // --- Calculate base offsets for this tile's results ---
    const long bias_tile_size     = (long)num_modules_in_chunk * num_classes_in_chunk;
    const long weight_tile_size   = (long)num_modules_in_chunk * hidden_dim * num_classes_in_chunk;
    const long bias_base_offset   = (long)flat_tile_index * bias_tile_size;
    const long weight_base_offset = (long)flat_tile_index * weight_tile_size;

    // --- Two-Stage Local Memory Reduction ---

    // STAGE 1: Reduce bias gradient (only h_idx=0 threads participate).
    if (h_idx == 0) {
        local_mem[lid] = p_grad_b;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
                local_mem[lid] += local_mem[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            // --- Use flat_tile_index to find the correct output slot. ----
            const long out_idx                 = bias_base_offset + (long)module_local_idx * num_classes_in_chunk + class_local_idx;
            partial_grad_module_b_out[out_idx] = local_mem[0];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // STAGE 2: Reduce weight gradient (all threads participate).
    local_mem[lid] = p_grad_w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            local_mem[lid] += local_mem[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        // --- Use flat_tile_index to find the correct output slot. ----
        const long out_idx                 = weight_base_offset + (long)module_local_idx * (hidden_dim * num_classes_in_chunk) + (long)h_idx * num_classes_in_chunk + class_local_idx;
        partial_grad_module_w_out[out_idx] = local_mem[0];
    }
}

// --- Implementation: backprop_error_to_hidden_chunk (Node 9) ---
// Strategy: A 3D "map" kernel that computes the partial Grad_H tensor for
// a single tile. It receives a unique `flat_tile_index` from the host, which it
// uses to compute a base write offset into the `partial_grad_h_aos_out`
// collection buffer. Each work-item computes one scalar value in this tile's
// Grad_H by reducing over its assigned class chunk.
__kernel void backprop_error_to_hidden_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global const SCALAR_TYPE *__restrict module_weights_buf,
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,
    int problem_type_flag,
    int module_batch_chunk_index,
    int module_param_offset,
    int num_modules_in_chunk,
    int class_batch_chunk_index,
    int flat_tile_index,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int hidden_dim,
    int total_output_classes,
    int padded_total_output_classes,
    int total_modules) {

    // Map the 3D work-item grid to the logical (module, batch, hidden) space for this tile.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);
    const uint h_idx            = get_global_id(2);

    // Boundary check against the logical work dimensions.
    if (module_local_idx >= num_modules_in_chunk || batch_idx >= total_batch_size || h_idx >= hidden_dim) {
        return;
    }

    // --- Use flat_tile_index to find the correct output block. ---
    const long tile_size        = (long)num_modules_in_chunk * total_batch_size * hidden_dim;
    const long tile_base_offset = (long)flat_tile_index * tile_size;
    const long local_offset     = (long)module_local_idx * (total_batch_size * hidden_dim) + (long)batch_idx * hidden_dim + h_idx;
    const long out_idx          = tile_base_offset + local_offset;

    // Skip computation for masked-out samples.
    if (sample_mask[batch_idx] < 0.5f) {
        partial_grad_h_aos_out[out_idx] = SCALAR_ZERO;
        return;
    }

    const uint module_global_idx = module_param_offset + module_local_idx;

    // --- Core Logic: Reduction over the Class Chunk ---
    SCALAR_TYPE grad_h_accum = SCALAR_ZERO;
    for (int c_local = 0; c_local < num_classes_in_chunk; ++c_local) {
        const int c_global = class_offset + c_local;
        // CRITICAL: Use physical stride for all class-dimensioned buffer access.
        const long        prob_idx = (long)module_global_idx * total_batch_size * padded_total_output_classes + (long)batch_idx * padded_total_output_classes + c_global;
        const SCALAR_TYPE prob     = partial_probs_buf[prob_idx];
        SCALAR_TYPE       d_loss_d_logit;

        if (problem_type_flag == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (__global int *)targets_buf;
            d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[batch_idx]);
        } else { // PROBLEM_TYPE_BCE
            const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
            d_loss_d_logit                          = prob - targets_bce[(long)batch_idx * padded_total_output_classes + c_global];
        }

        const long        weight_idx = (long)module_global_idx * hidden_dim * padded_total_output_classes + (long)h_idx * padded_total_output_classes + c_global;
        const SCALAR_TYPE weight_val = module_weights_buf[weight_idx];
        grad_h_accum += d_loss_d_logit * weight_val;
    }

    partial_grad_h_aos_out[out_idx] = grad_h_accum;
}

// --- Implementation: calculate_chunk_temp_gradients (Node 10) ---
// Strategy: A tile-based reduction kernel where each work-group computes a
// partial temperature gradient for one module within its assigned tile. It uses the
// host-provided `flat_tile_index` to calculate a base write offset, ensuring each
// tile's results are placed in a unique block within the `partial_grad_temps_out`
// collection buffer.
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict full_logits_buf,
    __global const SCALAR_TYPE *__restrict partial_probs_buf,
    __global const void *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict sample_mask,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,
    int problem_type_flag,
    int module_batch_chunk_index,
    int module_param_offset,
    int num_modules_in_chunk,
    int class_batch_chunk_index,
    int flat_tile_index,
    int class_offset,
    int num_classes_in_chunk,
    int total_batch_size,
    int total_output_classes,
    int padded_total_output_classes,
    int total_modules) {

    const uint module_local_idx = get_group_id(0);
    const uint lid              = get_local_id(0);
    const uint lsize            = get_local_size(0);

    if (module_local_idx >= num_modules_in_chunk) {
        return;
    }

    const uint  module_global_idx = module_param_offset + module_local_idx;
    SCALAR_TYPE p_grad_sum        = SCALAR_ZERO;

    // Threads in the work-group sum contributions over the batch dimension.
    for (int b = lid; b < total_batch_size; b += lsize) {
        if (sample_mask[b] < 0.5f) {
            continue;
        }

        // For each sample, sum the gradient contribution across the class chunk.
        SCALAR_TYPE grad_contribution_for_sample = SCALAR_ZERO;
        for (int c_local = 0; c_local < num_classes_in_chunk; c_local++) {
            const int         c_global = class_offset + c_local;
            const long        base_idx = (long)module_global_idx * total_batch_size * padded_total_output_classes + (long)b * padded_total_output_classes + c_global;
            const SCALAR_TYPE prob     = partial_probs_buf[base_idx];
            SCALAR_TYPE       d_loss_d_logit;

            if (problem_type_flag == PROBLEM_TYPE_CCE) {
                const __global int *targets_cce = (__global int *)targets_buf;
                d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[b]);
            } else { // PROBLEM_TYPE_BCE
                const __global SCALAR_TYPE *targets_bce = (__global SCALAR_TYPE *)targets_buf;
                d_loss_d_logit                          = prob - targets_bce[(long)b * padded_total_output_classes + c_global];
            }

            const SCALAR_TYPE unscaled_logit = full_logits_buf[base_idx];
            grad_contribution_for_sample += d_loss_d_logit * unscaled_logit;
        }
        p_grad_sum += grad_contribution_for_sample;
    }

    // Reduce the partial sums from each thread using local memory.
    local_mem[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_mem[lid] += local_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes the final result for the work-group.
    if (lid == 0) {
        const SCALAR_TYPE total_sum_for_chunk = local_mem[0];
        const SCALAR_TYPE temp                = temps_buf[module_global_idx];
        const SCALAR_TYPE final_partial_grad  = total_sum_for_chunk * (-1.0f / (temp * temp));

        // --- Use flat_tile_index to find the correct output slot. ---
        // Each tile produces `num_modules_in_chunk` partial gradient values.
        const uint out_idx              = (uint)flat_tile_index * num_modules_in_chunk + module_local_idx;
        partial_grad_temps_out[out_idx] = final_partial_grad;
    }
}
