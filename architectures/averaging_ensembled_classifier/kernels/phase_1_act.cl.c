// phase_1_act.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: forward_pass (Node 4) ---
// Strategy: A tiled matrix-vector multiplication. Each work-group computes a tile of
// the output hidden activations. The core optimization is the use of local memory
// to broadcast a slice of an input vector to all threads in a work-group,
// drastically reducing global memory bandwidth requirements.
__kernel void forward_pass(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_simd_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_weights_shared_simd_major,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_biases_shared,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_hidden_activations,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_hidden_mask,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_padded_input_count,
    uint                         src_scalar_NATURAL_padded_hidden_count) {

    // --- 1. Work-Item to Logical Coordinate Mapping ---
    // Each work-item is responsible for computing one element of the hidden activation tensor.
    const uint bid     = get_global_id(0); // This thread's index along the batch dimension of the current chunk.
    const uint h_block = get_global_id(1); // The block of hidden neurons this work-group is processing.
    const uint lid     = get_local_id(0);  // The SIMD lane index within the work-group.

    // Boundary check for the current chunk of work.
    if (bid >= src_scalar_NATURAL_batch_chunk_count) {
        return;
    }

    // --- 2. Handle Padded Samples ---
    // Convert the chunk-local batch index to a global index.
    const uint effective_bid = src_scalar_NATURAL_batch_chunk_offset + bid;
    // Calculate the final destination index for this work-item's output.
    const uint hidden_idx = effective_bid * src_scalar_NATURAL_padded_hidden_count + h_block * SIMD_WIDTH + lid;

    // Early exit for padded samples to avoid wasted computation and synchronization.
    // Both activation and mask are set to zero to propagate the invalid state.
    if (load_storage(src_buffer_GLOBAL_sample_mask, effective_bid) < (COMPUTE_TYPE)0.5f) {
        store_storage(dest_buffer_GLOBAL_hidden_activations, hidden_idx, COMPUTE_ZERO);
        store_storage(dest_buffer_GLOBAL_hidden_mask, hidden_idx, COMPUTE_ZERO);
        return;
    }

    // --- 3. Tiled Computation using Local Memory ---
    const uint TILE_SIZE = SIMD_WIDTH;

    // Partition the work-group's local memory. This is a manual allocation scheme.
    // 'tile_input' is shared by all threads in the work-group to enable data reuse.
    __local COMPUTE_TYPE *tile_input = update_buffer_LOCAL_simd_tile;
    // 'tile_weights' holds the portion of the weight matrix relevant to this work-group's tile.
    __local COMPUTE_TYPE *tile_weights = &update_buffer_LOCAL_simd_tile[TILE_SIZE];

    // Initialize accumulator with the bias value for this specific hidden neuron.
    COMPUTE_TYPE accum = load_state(src_buffer_GLOBAL_CONST_biases_shared, h_block * TILE_SIZE + lid);

    // Loop over the input dimension in tiles of size SIMD_WIDTH.
    for (uint t = 0; t < src_scalar_NATURAL_padded_input_count; t += TILE_SIZE) {
        // --- Coordinated Load from Global to Local Memory ---
        // Load a tile of the input vector. All threads in the work-group cooperate.
        const uint input_idx = effective_bid * src_scalar_NATURAL_padded_input_count + t + lid;
        if (t + lid < src_scalar_NATURAL_padded_input_count) {
            tile_input[lid] = load_storage(src_buffer_GLOBAL_input, input_idx);
        } else {
            tile_input[lid] = COMPUTE_ZERO; // Zero-out padding within the tile.
        }

        // Cooperatively load a tile of the weight matrix. The SIMD-major layout ensures
        // that accesses by adjacent threads (different `lid`) are coalesced.
        for (int i = 0; i < TILE_SIZE; ++i) {
            const uint weight_idx = h_block * src_scalar_NATURAL_padded_input_count * TILE_SIZE + (t + i) * TILE_SIZE + lid;
            if (t + i < src_scalar_NATURAL_padded_input_count) {
                tile_weights[i * TILE_SIZE + lid] = load_state(src_buffer_GLOBAL_CONST_weights_shared_simd_major, weight_idx);
            } else {
                tile_weights[i * TILE_SIZE + lid] = COMPUTE_ZERO;
            }
        }
        // Synchronize work-group to ensure all local memory is populated before proceeding.
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Computation from Fast Local Memory ---
        // Each thread computes its partial sum using the shared input tile. This is the
        // core of the optimization: 'tile_input' is read from global memory once per
        // tile but read from fast local memory SIMD_WIDTH times.
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accum += tile_input[k] * tile_weights[k * TILE_SIZE + lid];
        }
        // Synchronize before loading the next tile to prevent race conditions.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. Final Activation and Concurrent Mask Generation ---
    // Apply the ReLU activation function.
    const COMPUTE_TYPE activation = fmax(accum, COMPUTE_ZERO);
    store_storage(dest_buffer_GLOBAL_hidden_activations, hidden_idx, activation);

    // Concurrently compute the derivative mask for ReLU (0 or 1). This is a fused operation that
    // avoids a separate kernel launch, saving overhead. The `select` intrinsic is often
    // more efficient than an if/else block.
    store_storage(dest_buffer_GLOBAL_hidden_mask, hidden_idx, select((COMPUTE_TYPE)0.0f, (COMPUTE_TYPE)1.0f, activation > COMPUTE_ZERO));
}

// --- Implementation: render_logits_chunk (Node 5) ---
// Strategy: A pure, 3D "map" kernel. Each work-item is assigned the task of computing
// exactly one logit value. This embarrassingly parallel structure is highly scalable
// and maps perfectly to the GPU's execution model. The implementation is fully
// compliant with the modern, orthogonal chunking contract.
__kernel void render_logits_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_weights_module,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_biases_module,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_logits,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_module_chunk_offset,
    uint                         src_scalar_NATURAL_module_chunk_count,
    uint                         src_scalar_NATURAL_class_chunk_offset,
    uint                         src_scalar_NATURAL_class_chunk_count,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count) {

    // --- 1. Work-Item to Logical Coordinate Mapping ---
    // The 3D global work-size maps directly to the logical (module, batch, class)
    // dimensions of the computational chunk.
    const uint module_local_idx = get_global_id(0);
    const uint batch_local_idx  = get_global_id(1);
    const uint class_local_idx  = get_global_id(2);

    // Standard boundary check to discard excess work-items from incomplete work-groups.
    if (module_local_idx >= src_scalar_NATURAL_module_chunk_count || batch_local_idx >= src_scalar_NATURAL_batch_chunk_count || class_local_idx >= src_scalar_NATURAL_class_chunk_count) {
        return;
    }

    // --- 2. Global Index Calculation ---
    // Translate the local, chunk-relative index of this work-item into the
    // absolute global index within the full tensor space.
    const uint module_global_idx = src_scalar_NATURAL_module_chunk_offset + module_local_idx;
    const uint batch_global_idx  = src_scalar_NATURAL_batch_chunk_offset + batch_local_idx;
    const uint class_global_idx  = src_scalar_NATURAL_class_chunk_offset + class_local_idx;

    // --- 3. Padding-Aware Address Calculation ---
    // Calculate the final 1D memory address for the output logit.
    // Casting to `long` is a forward-thinking robustness measure against integer
    // overflow for models with extremely large tensor dimensions.
    const long out_idx = (long)module_global_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count
                         + (long)batch_global_idx * src_scalar_NATURAL_padded_total_output_class_count + (long)class_global_idx;

    // Initialize the accumulator with the corresponding bias.
    COMPUTE_TYPE logit = load_state(src_buffer_GLOBAL_CONST_biases_module, (long)module_global_idx * src_scalar_NATURAL_padded_total_output_class_count + class_global_idx);

    // --- 4. Sparse Dot Product ---
    // Sum contributions over the hidden dimension to compute the dot product.
    for (uint h = 0; h < src_scalar_NATURAL_hidden_count; ++h) {
        // Calculate the base index for the hidden layer outputs for this batch item.
        const long         hidden_base_idx = (long)batch_global_idx * src_scalar_NATURAL_padded_hidden_count;
        const COMPUTE_TYPE h_mask          = load_storage(src_buffer_GLOBAL_hidden_mask, hidden_base_idx + h);

        // This `if` check is a critical, sparsity-aware optimization. The upstream ReLU
        // activation (in Node 4) zeroes out many hidden activations. By checking the
        // mask first, this kernel avoids two expensive global memory reads (for the
        // activation and weight) and a multiplication for every zeroed-out neuron.
        if (h_mask > (COMPUTE_TYPE)0.5f) {
            const COMPUTE_TYPE h_val = load_storage(src_buffer_GLOBAL_hidden_activations, hidden_base_idx + h);

            // The weight index calculation is complex but correctly follows row-major layout
            // while respecting the physical padding of all dimensions.
            const long weight_idx = (long)module_global_idx * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count
                                    + (long)h * src_scalar_NATURAL_padded_total_output_class_count + (long)class_global_idx;

            const COMPUTE_TYPE w_val = load_state(src_buffer_GLOBAL_CONST_weights_module, weight_idx);
            logit += h_val * w_val;
        }
    }

    // Write the final computed logit to its destination.
    store_storage(dest_buffer_GLOBAL_logits, out_idx, logit);
}

// --- Implementation: compute_probs_loss_cce_chunk (Node 6) ---
// Strategy: A fused kernel using a 2D dispatch grid. Each work-item is a self-contained
// reduction engine for a single (module, sample) pair. It serially computes the numerically
// stable Softmax parameters (max_logit, sum_exp) over the entire class dimension.
// It then fulfills its dual contract by acting as a "Partial Renderer" for its assigned
// probability chunk and performing a direct "scatter-write" of the final loss.
__kernel void compute_probs_loss_cce_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global const int          *src_buffer_GLOBAL_targets,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_probs,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_final_loss,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // --- 1. Work-Item to Logical Coordinate Mapping ---
    // A 2D dispatch is used: each thread processes one (module, batch_idx) pair.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk || batch_idx >= src_scalar_NATURAL_total_batch_count) {
        return;
    }

    // --- 2. Decompose Flat Tile Index into Logical Grid Coordinates ---
    // The host provides a flat 1D index encoding a 2D tile grid. We de-flatten it
    // here to find the global module index for this work-item.
    const uint module_chunk_idx  = src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx = module_chunk_idx * src_scalar_NATURAL_modules_per_chunk + module_local_idx;

    // --- 3. Handle Padded Samples ---
    const long loss_out_idx = (long)module_global_idx * src_scalar_NATURAL_total_batch_count + batch_idx;

    if (load_storage(src_buffer_GLOBAL_sample_mask, batch_idx) < (COMPUTE_TYPE)0.5f) {
        store_storage(dest_buffer_GLOBAL_final_loss, loss_out_idx, COMPUTE_ZERO);
        // Also zero out the partial probabilities this tile is responsible for. This ensures
        // downstream gradient kernels receive correct zero inputs for padded samples.
        const uint class_chunk_idx = src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
        const long prob_tile_base_offset
            = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;

        for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk; ++c_local) {
            const long prob_write_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                        + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk + c_local;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx, COMPUTE_ZERO);
        }
        return;
    }

    // --- 4. Fused, Numerically Stable Softmax (Internal Reduction) ---
    // This thread performs a full two-pass reduction over the class dimension.
    const COMPUTE_TYPE temp_inv = 1.0f / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);
    const long         base_logits_idx
        = (long)module_global_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count + (long)batch_idx * src_scalar_NATURAL_padded_total_output_class_count;

    // Pass 1: Find max for numerical stability (part of log-sum-exp trick).
    COMPUTE_TYPE max_scaled_logit = -FLT_MAX;
    for (uint c = 0; c < src_scalar_NATURAL_total_output_class_count; ++c) {
        max_scaled_logit = fmax(max_scaled_logit, load_storage(src_buffer_GLOBAL_logits, base_logits_idx + c) * temp_inv);
    }
    if (max_scaled_logit == -FLT_MAX) {
        max_scaled_logit = 0.0f;
    }

    // Pass 2: Calculate sum of exponentials.
    COMPUTE_TYPE sum_exp = COMPUTE_ZERO;
    for (uint c = 0; c < src_scalar_NATURAL_total_output_class_count; ++c) {
        sum_exp += MATH_FN exp((load_storage(src_buffer_GLOBAL_logits, base_logits_idx + c) * temp_inv) - max_scaled_logit);
    }
    const COMPUTE_TYPE inv_sum_exp = (sum_exp > (COMPUTE_TYPE)NUMERICAL_STABILITY_EPSILON) ? (1.0f / sum_exp) : 0.0f;

    // --- 5. Partial Probability Rendering ---
    // Now, write out only the chunk of probabilities this tile is responsible for.
    const uint class_chunk_idx       = src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint class_offset          = class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;
    const long prob_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;

    for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk; ++c_local) {
        const uint c_global = class_offset + c_local;
        if (c_global < src_scalar_NATURAL_total_output_class_count) {
            const COMPUTE_TYPE logit = load_storage(src_buffer_GLOBAL_logits, base_logits_idx + c_global);
            const COMPUTE_TYPE prob  = MATH_FN exp((logit * temp_inv) - max_scaled_logit) * inv_sum_exp;

            const long prob_write_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                        + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk + c_local;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx, prob);
        }
    }

    // --- 6. Final Loss Calculation (Scatter-Write with Write Predicate) ---
    // Loss Write Predicate: Only the tile whose class chunk contains the target class
    // index writes to the loss buffer. All other tiles skip the write.
    // This prevents data races under OpenCL/Vulkan memory models when num_class_chunks > 1.
    // The ZERO_REQUIRED initialization contract ensures unwritten positions are zero.
    const int  true_class_idx       = src_buffer_GLOBAL_targets[batch_idx];
    const uint class_chunk_end      = class_offset + src_scalar_NATURAL_classes_per_chunk;
    const int  target_in_this_chunk = ((uint)true_class_idx >= class_offset) && ((uint)true_class_idx < class_chunk_end);

    if (target_in_this_chunk) {
        // Re-calculate the probability for the true class. This is cheap as the
        // expensive reduction parameters (max_logit, inv_sum_exp) are already computed.
        const COMPUTE_TYPE logit_true_class = load_storage(src_buffer_GLOBAL_logits, base_logits_idx + true_class_idx);
        const COMPUTE_TYPE prob_true_class  = MATH_FN exp((logit_true_class * temp_inv) - max_scaled_logit) * inv_sum_exp;

        // The argument to log is bounded by the system's epsilon.
        store_storage(dest_buffer_GLOBAL_final_loss, loss_out_idx, -MATH_FN log(fmax(prob_true_class, (COMPUTE_TYPE)NUMERICAL_STABILITY_EPSILON)));
    }
}

// --- Implementation: compute_probs_loss_bce_chunk (Node 7) ---
// Strategy: A 2D dispatch kernel that acts as a "Dual Partial Renderer". Each work-item
// is responsible for a single (module, sample) pair within its assigned tile. It calculates
// a slice of the Sigmoid probabilities and reduces the BCE loss over that same slice.
// Both results are written to unique, non-overlapping regions of their respective
// collection buffers, governed by the `flat_tile_index`.
__kernel void compute_probs_loss_bce_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_targets,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_probs,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_loss,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // --- 1. Work-Item to Logical Coordinate Mapping ---
    // A 2D dispatch is used: each thread processes one (module, batch_idx) pair.
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk || batch_idx >= src_scalar_NATURAL_total_batch_count) {
        return;
    }

    // --- 2. Decompose Flat Tile Index & Calculate Write Offsets ---
    // De-flatten the tile index provided by the host to determine our logical grid position.
    const uint module_chunk_idx  = src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx = module_chunk_idx * src_scalar_NATURAL_modules_per_chunk + module_local_idx;
    // Calculate the base write offset into the partial loss collection buffer for this specific tile.
    const long loss_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count;
    const long loss_write_idx        = loss_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count + batch_idx;

    // --- 3. Handle Padded Samples ---
    if (load_storage(src_buffer_GLOBAL_sample_mask, batch_idx) < (COMPUTE_TYPE)0.5f) {
        // For padded samples, we must zero out both outputs this tile is responsible for.
        store_storage(dest_buffer_GLOBAL_partial_loss, loss_write_idx, COMPUTE_ZERO);

        // Calculate the base offset for this tile's slice of the probability buffer.
        const long prob_tile_base_offset
            = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;
        for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk; ++c_local) {
            const long prob_write_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                        + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk + c_local;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx, COMPUTE_ZERO);
        }
        return;
    }

    // --- 4. Fused Probability Calculation and Loss Reduction ---
    // Each thread accumulates the loss contributions from its assigned chunk of classes.
    COMPUTE_TYPE       partial_loss_accum = COMPUTE_ZERO;
    const COMPUTE_TYPE temp_inv           = 1.0f / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);

    // Decompose the tile index again to find this tile's specific chunk of classes.
    const uint class_chunk_idx       = src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint class_offset          = class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;
    const long prob_tile_base_offset = (long)src_scalar_NATURAL_flat_tile_index * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk;

    // Loop over this tile's assigned slice of the class dimension.
    for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk; ++c_local) {
        const uint c_global = class_offset + c_local;

        // This check gracefully handles cases where the total number of classes is not
        // a multiple of the chunk size, preventing out-of-bounds access on the last chunk.
        if (c_global < src_scalar_NATURAL_total_output_class_count) {
            const long base_read_idx = (long)module_global_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count
                                       + (long)batch_idx * src_scalar_NATURAL_padded_total_output_class_count + c_global;

            // Compute Sigmoid probability (numerically stable two-branch form).
            const COMPUTE_TYPE logit        = load_storage(src_buffer_GLOBAL_logits, base_read_idx);
            const COMPUTE_TYPE scaled_logit = logit * temp_inv;
            COMPUTE_TYPE prob;
            if (scaled_logit >= 0.0f) {
                const COMPUTE_TYPE e = MATH_FN exp(-scaled_logit);
                prob = 1.0f / (1.0f + e);
            } else {
                const COMPUTE_TYPE e = MATH_FN exp(scaled_logit);
                prob = e / (1.0f + e);
            }

            // Write the probability to its unique, tile-local slot in the collection buffer.
            const long prob_write_idx = prob_tile_base_offset + (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk
                                        + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk + c_local;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx, prob);

            // Accumulate the BCE loss contribution from this class, the log is bounded by the system's epsilon.
            const COMPUTE_TYPE target_val = load_storage(src_buffer_GLOBAL_targets, (long)batch_idx * src_scalar_NATURAL_padded_total_output_class_count + c_global);
            const COMPUTE_TYPE term1      = target_val * MATH_FN log(fmax(prob, (COMPUTE_TYPE)NUMERICAL_STABILITY_EPSILON));
            const COMPUTE_TYPE term2      = (1.0f - target_val) * MATH_FN log(fmax(1.0f - prob, (COMPUTE_TYPE)NUMERICAL_STABILITY_EPSILON));
            partial_loss_accum += term1 + term2;
        }
    }

    // --- 5. Final Partial Loss Write ---
    // Write the final summed partial loss for this tile to its unique collection slot.
    // The outputs of this kernel are contractually obligated to be summed by a
    // subsequent reduction stage to get the final loss for the sample.
    store_storage(dest_buffer_GLOBAL_partial_loss, loss_write_idx, -partial_loss_accum);
}
