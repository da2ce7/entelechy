// phase_2_learn_A_production.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: calculate_module_param_grads_chunk (Node 8) ---
// Strategy: A "work-group per gradient component" reduction. Each work-group is assigned
// the task of computing a single scalar value in the output gradient tensor (e.g., dL/dW_m,h,c).
// Threads within the group then collaborate to perform the reduction (summation) over the
// entire batch dimension. This model is highly efficient as it maximizes parallelism and
// correctly utilizes the local memory hierarchy for the final reduction step.
__kernel void calculate_module_param_grads_chunk(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,
    __global const void         *src_buffer_GLOBAL_targets,
    __global const uint          *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_weights_module,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_biases_module,
    uint                         src_scalar_FLAG_problem_type,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // --- 1. Work-Group to Gradient Component Mapping ---
    // The 3D work-group ID maps directly to a coordinate in the gradient tensor space.
    const uint module_local_idx = get_group_id(0);
    const uint h_idx            = get_group_id(1);
    const uint class_local_idx  = get_group_id(2);
    // The local thread ID maps to the reduction dimension (batch).
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Boundary check for the gradient component this work-group is assigned.
    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk || h_idx >= src_scalar_NATURAL_hidden_count || class_local_idx >= src_scalar_NATURAL_classes_per_chunk) {
        return;
    }

    // --- 2. Decompose Flat Index to find Global Coordinates ---
    const uint module_chunk_idx  = src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx = module_chunk_idx * src_scalar_NATURAL_modules_per_chunk + module_local_idx;
    const COMPUTE_TYPE inv_temp  = 1.0f / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);
    const uint class_chunk_idx   = src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint class_global_idx  = class_chunk_idx * src_scalar_NATURAL_classes_per_chunk + class_local_idx;

    // Padding Zero-Preservation: skip work-groups in final class chunk that exceed logical extent
    if (class_global_idx >= src_scalar_NATURAL_total_output_class_count) {
        return;
    }

    // --- 3. Parallel Reduction over Batch Dimension ---
    COMPUTE_TYPE p_grad_w = COMPUTE_ZERO;
    COMPUTE_TYPE p_grad_b = COMPUTE_ZERO;

    // Use the flat_tile_index to find the correct slice of the partial probabilities produced by the upstream kernel.
    const long prob_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;

    // Each thread sums a strided slice of the batch dimension.
    for (uint b = lid; b < src_scalar_NATURAL_total_batch_count; b += lsize) {
        if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, b)) {
            continue;
        }

        // --- Calculate d_loss_d_logit (upstream gradient from the loss function) ---
        const long prob_read_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                   + (long)b * src_scalar_NATURAL_classes_per_chunk + class_local_idx;
        const COMPUTE_TYPE prob = load_storage(src_buffer_GLOBAL_partial_probs, prob_read_idx);

        COMPUTE_TYPE d_loss_d_logit;
        if (src_scalar_FLAG_problem_type == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce = (const __global int *)src_buffer_GLOBAL_targets;
            d_loss_d_logit                  = select(prob, prob - 1.0f, class_global_idx == targets_cce[b]);
        } else { // PROBLEM_TYPE_BCE
            const __global STORAGE_TYPE *targets_bce = (const __global STORAGE_TYPE *)src_buffer_GLOBAL_targets;
            d_loss_d_logit                           = prob - load_storage(targets_bce, (long)b * src_scalar_NATURAL_padded_total_output_class_count + class_global_idx);
        }

        // Apply temperature chain rule: dL/dz = dL/dz_s * (1/tau)
        d_loss_d_logit *= inv_temp;

        // --- Accumulate Weight and Bias Gradients ---
        const COMPUTE_TYPE h_val = load_storage(src_buffer_GLOBAL_hidden_activations, (long)b * src_scalar_NATURAL_padded_hidden_count + h_idx);
        p_grad_w += d_loss_d_logit * h_val; // dL/dW = (dL/dLogit) * h_val

        // Optimization: The bias gradient (dL/dB) is simply dL/dLogit and does not depend on `h_idx`.
        // We compute it only once per work-group (when h_idx == 0) to avoid redundant computation.
        if (h_idx == 0) {
            p_grad_b += d_loss_d_logit;
        }
    }

    // --- 4. Two-Stage Intra-Workgroup Reduction & Write ---
    // This two-stage process efficiently handles the fused computation of both bias and weight gradients.

    // Stage 1: Reduce and write the bias gradient. Only work-groups with h_idx=0 participate fully.
    if (h_idx == 0) {
        update_buffer_LOCAL_reduction_tile[lid] = p_grad_b;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride)
                update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            const long bias_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_total_output_class_count;
            const long out_idx               = bias_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_padded_total_output_class_count + class_global_idx;
            store_storage(dest_buffer_GLOBAL_partial_grad_biases_module, out_idx, update_buffer_LOCAL_reduction_tile[0]);
        }
    }

    // Stage 2: Reduce and write the weight gradient. All work-groups participate.
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride)
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        const long weight_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count;
        const long out_idx                 = weight_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count
                             + (long)h_idx * src_scalar_NATURAL_padded_total_output_class_count + class_global_idx;
        store_storage(dest_buffer_GLOBAL_partial_grad_weights_module, out_idx, update_buffer_LOCAL_reduction_tile[0]);
    }
}

// --- Implementation: backprop_error_to_hidden_chunk (Node 9) ---
// Strategy: A 3D "map-reduce" kernel. Each work-item is assigned to compute a single
// scalar value in the partial upstream gradient tensor (Grad_H). It achieves this by
// performing a serial reduction (a dot product) over its assigned chunk of the
// class dimension. This structure effectively parallelizes what is mathematically a
// matrix-vector multiplication, mapping it efficiently to the GPU architecture.
__kernel void backprop_error_to_hidden_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,
    __global const void         *src_buffer_GLOBAL_targets,
    __global const uint          *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_weights_module,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,
    uint                         src_scalar_FLAG_problem_type,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // --- 1. Work-Item to Output Coordinate Mapping ---
    // The 3D dispatch grid maps directly to the (module, batch, hidden) coordinates
    // of the output tensor this work-item will compute.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);
    const uint h_idx            = get_global_id(2);

    // Boundary check for the output component this work-item is assigned.
    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk || batch_idx >= src_scalar_NATURAL_total_batch_count || h_idx >= src_scalar_NATURAL_padded_hidden_count) {
        return;
    }

    // --- 2. Calculate Final Write Address (Placement Contract) ---
    // This calculation translates the 3D logical coordinate into a 1D physical memory address
    // within the destination collection buffer, respecting the Array-of-Structs (AoS) layout.
    const long tile_size        = (long)src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count;
    const long tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * tile_size;
    const long local_offset = (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count + (long)batch_idx * src_scalar_NATURAL_padded_hidden_count + h_idx;
    const long out_idx      = tile_base_offset + local_offset;

    // Early exit for padded samples, writing zero to the output to maintain correctness.
    if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, batch_idx)) {
        store_storage(dest_buffer_GLOBAL_partial_grad_hidden_activations_aos, out_idx, COMPUTE_ZERO);
        return;
    }

    // Padding Zero-Establishment: write zero for positions in the padded hidden dimension.
    if (h_idx >= src_scalar_NATURAL_hidden_count) {
        store_storage(dest_buffer_GLOBAL_partial_grad_hidden_activations_aos, out_idx, COMPUTE_ZERO);
        return;
    }

    // --- 3. Reduction over Class Dimension ---
    // Each thread accumulates the gradient contributions from its assigned chunk of the class dimension.
    const uint   module_chunk_idx  = src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint   module_global_idx = module_chunk_idx * src_scalar_NATURAL_modules_per_chunk + module_local_idx;
    const COMPUTE_TYPE inv_temp    = 1.0f / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);
    COMPUTE_TYPE grad_h_accum      = COMPUTE_ZERO;

    // Decompose the flat index to find the correct chunk of classes and probabilities.
    const uint class_chunk_idx       = src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint class_offset          = class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;
    const long prob_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;

    // This loop performs the dot product between a row of the module weights and the d_loss_d_logit vector.
    for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk; ++c_local) {
        const uint c_global = class_offset + c_local;

        // Gracefully handle ragged chunks where the number of classes is not a multiple of the chunk size.
        if (c_global < src_scalar_NATURAL_total_output_class_count) {
            // Read the pre-computed probability for this class from the correct tile-local slice.
            const long prob_read_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                       + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk + c_local;
            const COMPUTE_TYPE prob = load_storage(src_buffer_GLOBAL_partial_probs, prob_read_idx);

            // Compute the upstream gradient from the loss function (d_loss_d_logit) for this specific class.
            COMPUTE_TYPE d_loss_d_logit;
            if (src_scalar_FLAG_problem_type == PROBLEM_TYPE_CCE) {
                const __global int *targets_cce = (const __global int *)src_buffer_GLOBAL_targets;
                d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[batch_idx]);
            } else { // PROBLEM_TYPE_BCE
                const __global STORAGE_TYPE *targets_bce = (const __global STORAGE_TYPE *)src_buffer_GLOBAL_targets;
                d_loss_d_logit                           = prob - load_storage(targets_bce, (long)batch_idx * src_scalar_NATURAL_padded_total_output_class_count + c_global);
            }

            // Apply temperature chain rule: dL/dz = dL/dz_s * (1/tau)
            d_loss_d_logit *= inv_temp;

            // Read the corresponding weight and accumulate the gradient contribution.
            const long weight_idx = (long)module_global_idx * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count
                                    + (long)h_idx * src_scalar_NATURAL_padded_total_output_class_count + c_global;

            const COMPUTE_TYPE weight_val = load_state(src_buffer_GLOBAL_CONST_weights_module, weight_idx);
            grad_h_accum += d_loss_d_logit * weight_val;
        }
    }

    // Write the final, reduced gradient value to its unique destination.
    store_storage(dest_buffer_GLOBAL_partial_grad_hidden_activations_aos, out_idx, grad_h_accum);
}

// --- Implementation: calculate_chunk_temp_gradients (Node 10) ---
// Strategy: A "Work-Group Per Module" reduction. Each work-group is assigned to compute a
// partial temperature gradient for a single module. It performs a hierarchical, three-level
// reduction to achieve this. Crucially, it only computes the gradient contribution from its
// single, assigned chunk of classes, making it a "partial-partial" renderer.
__kernel void calculate_chunk_temp_gradients(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,
    __global const void         *src_buffer_GLOBAL_targets,
    __global const uint          *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_temps,
    uint                         src_scalar_FLAG_problem_type,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // --- 1. Work-Group to Module Mapping ---
    // A 1D dispatch is used, where each work-group computes the gradient for one module.
    const uint module_local_idx = get_group_id(0);
    // Threads within the work-group collaborate on the reduction over the batch dimension.
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk) {
        return;
    }

    // --- 2. Decompose Flat Index to find Global Coordinates & Read Offsets ---
    // This is the core of the placement contract logic, translating the 1D tile index
    // into the 2D grid coordinates needed to find our data.
    const uint module_chunk_idx  = src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx = module_chunk_idx * src_scalar_NATURAL_modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint class_offset      = class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;

    // --- 3. Hierarchical Three-Level Reduction ---
    COMPUTE_TYPE p_grad_sum = COMPUTE_ZERO; // This thread's partial sum over its slice of the batch.

    // Calculate the base offset to read from the correct slice of the upstream partial probability buffer.
    const long prob_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;

    // Level 1 Reduction (over batch): Each thread sums contributions from a strided slice of the batch.
    for (uint b = lid; b < src_scalar_NATURAL_total_batch_count; b += lsize) {
        if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, b)) {
            continue;
        }

        // Level 2 Reduction (over class chunk): For each sample, we sum the gradient
        // contributions from this tile's assigned chunk of classes.
        COMPUTE_TYPE grad_contribution_for_sample = COMPUTE_ZERO;
        for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk; c_local++) {
            const uint c_global = class_offset + c_local;

            if (c_global < src_scalar_NATURAL_total_output_class_count) {
                // Calculate dL/dLogit for this specific class.
                const long prob_read_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                           + (long)b * src_scalar_NATURAL_classes_per_chunk + c_local;
                const COMPUTE_TYPE prob = load_storage(src_buffer_GLOBAL_partial_probs, prob_read_idx);

                COMPUTE_TYPE d_loss_d_logit;
                if (src_scalar_FLAG_problem_type == PROBLEM_TYPE_CCE) {
                    const __global int *targets_cce = (const __global int *)src_buffer_GLOBAL_targets;
                    d_loss_d_logit                  = select(prob, prob - 1.0f, c_global == targets_cce[b]);
                } else {
                    const __global STORAGE_TYPE *targets_bce = (const __global STORAGE_TYPE *)src_buffer_GLOBAL_targets;
                    d_loss_d_logit                           = prob - load_storage(targets_bce, (long)b * src_scalar_NATURAL_padded_total_output_class_count + c_global);
                }

                // Accumulate the gradient contribution: (dL/dLogit) * Logit.
                const long logit_read_idx = (long)module_global_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count
                                            + (long)b * src_scalar_NATURAL_padded_total_output_class_count + c_global;
                grad_contribution_for_sample += d_loss_d_logit * load_storage(src_buffer_GLOBAL_logits, logit_read_idx);
            }
        }
        p_grad_sum += grad_contribution_for_sample;
    }

    // --- 4. Final Reduction and Gradient Calculation ---
    // Level 3 Reduction (over work-group): Sum the partial sums from all threads.
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread applies the final chain rule step and writes the result.
    if (lid == 0) {
        const COMPUTE_TYPE total_sum_for_chunk = update_buffer_LOCAL_reduction_tile[0];
        const COMPUTE_TYPE temp                = load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);

        // Apply final step of the chain rule: dL/dTemp = (dL/dScaledLogit) * (-Logit / Temp^2)
        // The loop calculated sum[(dL/dLogit) * Logit]. This final multiplication completes the gradient.
        const COMPUTE_TYPE final_partial_grad = total_sum_for_chunk * (-1.0f / (temp * temp));

        // Write to the unique slot for this tile in the collection buffer.
        const long tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk;
        const uint out_idx          = tile_base_offset + module_local_idx;
        store_storage(dest_buffer_GLOBAL_partial_grad_temps, out_idx, final_partial_grad);
    }
}
