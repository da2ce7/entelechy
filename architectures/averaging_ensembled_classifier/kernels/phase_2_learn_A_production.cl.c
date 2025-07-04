// phase_2_learn_A_production.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif



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
