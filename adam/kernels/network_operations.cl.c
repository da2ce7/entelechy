// network_operations.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// network_operations.cl.c

/**
 * @brief (Node 4 - CORRECTED & FINAL) Performs a tiled, SIMD-aware feed-forward pass.
 *
 * This is a corrected, high-performance implementation. It uses a 2D NDRange where each
 * work-group is responsible for computing one (1 x SIMD_WIDTH) vector of hidden-layer
 * activations for a single sample.
 *
 * It overcomes the indexing flaws of the original by correctly tiling both the input
 * vector and the weight matrix sub-block into local memory, ensuring maximum data reuse
 * and coalesced global memory access.
 *
 * This final version also corrects a critical logic bug by ensuring the `hidden_mask`
 * is always written, even for masked-out samples, before any early exit.
 *
 * Host Assumptions:
 * - Launch grid is (padded_batch_size, padded_hidden_dim / SIMD_WIDTH).
 * - Local work size MUST be (SIMD_WIDTH) for this kernel to function correctly.
 * - Weights are in the pre-transposed SIMD-major format.
 */
__kernel void forward_pass(
    // Local memory must be sized by host: (TILE_SIZE + TILE_SIZE * SIMD_WIDTH) * sizeof(SCALAR_TYPE)
    // We will use a TILE_SIZE equal to SIMD_WIDTH for simplicity and efficiency.
    __local SCALAR_TYPE *local_mem,

    // Inputs
    __global const SCALAR_TYPE *__restrict input,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global const SCALAR_TYPE *__restrict weights,
    __global const SCALAR_TYPE *__restrict biases,

    // Outputs
    __global SCALAR_TYPE *__restrict hidden,
    __global SCALAR_TYPE *__restrict hidden_mask,

    // Dimensions
    int padded_input_dim,
    int padded_hidden_dim) {

    // --- Global and Local IDs ---
    const uint bid     = get_global_id(0);  // Batch index (which sample)
    const uint h_block = get_global_id(1);  // Hidden block index (which group of SIMD neurons)
    const uint lane    = get_local_id(0);   // This thread's lane in the SIMD vector (0...SIMD_WIDTH-1)
    const uint lsize   = get_local_size(0); // Should be == SIMD_WIDTH

    // --- Masking and Boundary Checks ---
    // CORRECTED LOGIC: Write the mask FIRST to ensure it's always propagated.
    // One thread per work-group (sample) is sufficient.
    if (lane == 0) {
        hidden_mask[bid] = input_mask[bid];
    }

    // Now, check the mask and perform an early exit if the sample is invalid.
    if (input_mask[bid] < (SCALAR_TYPE)0.5f) {
        // Zero out the entire hidden vector for this masked sample
        if (lane == 0) { // Only one thread needs to do this
            for (int i = 0; i < padded_hidden_dim; ++i) {
                hidden[bid * padded_hidden_dim + i] = (SCALAR_TYPE)0.0f;
            }
        }
        return; // This return is now safe as the hidden_mask has been written.
    }

    // --- Tiling Setup ---
    // A work-group computes one (1 x SIMD_WIDTH) output vector.
    // We tile across the input_dim dimension. Let's use a tile size equal to the work-group size.
    const uint           TILE_SIZE    = lsize;
    __local SCALAR_TYPE *tile_input   = local_mem;             // Size: TILE_SIZE
    __local SCALAR_TYPE *tile_weights = local_mem + TILE_SIZE; // Size: TILE_SIZE * SIMD_WIDTH

    // --- Accumulation ---
    // Each thread in the work-group computes one element of the final SIMD vector.
    SCALAR_TYPE accum = biases[h_block * lsize + lane];

    // --- Tiled Matrix-Vector Multiplication ---
    for (uint t = 0; t < padded_input_dim; t += TILE_SIZE) {

        // --- Step 1: Coordinated Load from Global to Local Memory ---
        // Load one tile of the input vector. Each thread loads one element.
        const uint input_idx = bid * padded_input_dim + t + lane;
        if (t + lane < padded_input_dim) {
            tile_input[lane] = input[input_idx];
        } else {
            tile_input[lane] = (SCALAR_TYPE)0.0f; // Pad with zero if out of bounds
        }

        // Load one tile of the weight matrix. This is the crucial part.
        // Each thread loads a column of the (TILE_SIZE x SIMD_WIDTH) weight tile.
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint weight_idx = h_block * padded_input_dim * lsize + (t + i) * lsize + lane;
            if (t + i < padded_input_dim) {
                tile_weights[i * lsize + lane] = weights[weight_idx];
            } else {
                tile_weights[i * lsize + lane] = (SCALAR_TYPE)0.0f;
            }
        }

        // Synchronize to ensure all data is loaded into the local memory tile.
        barrier(CLK_LOCAL_MEM_FENCE);

        // --- Step 2: Computation using Local Memory ---
        // Each thread accumulates its dot product contribution from the tile.
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accum += tile_input[k] * tile_weights[k * lsize + lane];
        }

        // Synchronize before the next tile iteration to prevent overwriting local memory prematurely.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Step 3: Write Final Result with Inlined ReLU ---
    const uint hidden_idx = bid * padded_hidden_dim + h_block * lsize + lane;
    hidden[hidden_idx]    = fmax(accum, (SCALAR_TYPE)0.0f);
}

/**
 * @brief (REFACTORED & FIXED) Computes all necessary outputs for each exit in the forward pass.
 *
 * This is a high-performance refactoring of the original exit kernel. It implements the same
 * optimizations (vectorized math, log-sum-exp stabilization) while fulfilling the new API
 * contract required by the advanced backpropagation kernels.
 *
 * The key change is the explicit output of the `unscaled_logits` buffer, which is critical
 * for calculating temperature gradients in the backward pass.
 */
__kernel void compute_all_exits(
    // Inputs (Signature now matches kernels.cl.h)
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,
    __global const SCALAR_TYPE *__restrict exit_biases_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const int *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,

    // Outputs
    __global SCALAR_TYPE *__restrict unscaled_logits_buf,
    __global SCALAR_TYPE *__restrict exit_probs_buf,
    __global SCALAR_TYPE *__restrict per_exit_losses_buf,

    // Dimensions
    int padded_batch_size,
    int hidden_dim,
    int output_classes,
    int num_exits) {

    const uint exit_idx  = get_global_id(0);
    const uint batch_idx = get_global_id(1);

    if (exit_idx >= num_exits || batch_idx >= padded_batch_size || hidden_mask[batch_idx] < 0.5f || targets_mask[batch_idx] < 0.5f) {
        return;
    }

    // --- Phase 1: Compute Unscaled Logits (W*h + b) and write them ---
    for (int c = 0; c < output_classes; ++c) {
        SCALAR_TYPE logit = exit_biases_buf[exit_idx * output_classes + c];
        for (int h = 0; h < hidden_dim; ++h) {
            logit += hidden_buf[batch_idx * hidden_dim + h] * exit_weights_buf[exit_idx * hidden_dim * output_classes + h * output_classes + c];
        }
        unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] = logit;
    }

    // --- Phase 2: Apply Temperature and compute Softmax ---
    const SCALAR_TYPE temp_inv = 1.0f / temps_buf[exit_idx];

    // 1. Find max_logit for stability
    SCALAR_TYPE max_logit = -INFINITY;
    for (int c = 0; c < output_classes; ++c) {
        SCALAR_TYPE scaled_logit = unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] * temp_inv;
        max_logit                = fmax(max_logit, scaled_logit);
    }

    // 2. Compute sum of exponentials
    SCALAR_TYPE sum_exp = 0.0f;
    for (int c = 0; c < output_classes; ++c) {
        SCALAR_TYPE scaled_logit = unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] * temp_inv;
        sum_exp += exp(scaled_logit - max_logit);
    }
    sum_exp = fmax(sum_exp, (SCALAR_TYPE)1e-7f);

    // --- Phase 3: Final Probabilities and Per-Exit Loss ---
    const int   true_class = targets_buf[batch_idx];
    SCALAR_TYPE loss       = 0.0f;
    for (int c = 0; c < output_classes; ++c) {
        SCALAR_TYPE scaled_logit = unscaled_logits_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] * temp_inv;
        SCALAR_TYPE prob         = exp(scaled_logit - max_logit) / sum_exp;
        exit_probs_buf[exit_idx * padded_batch_size * output_classes + batch_idx * output_classes + c] = prob;

        if (c == true_class) {
            loss = -log(fmax(prob, (SCALAR_TYPE)1e-7f));
        }
    }
    per_exit_losses_buf[exit_idx * padded_batch_size + batch_idx] = loss;
}

/**
 * @brief (Tier 1: Small-Scale) Computes ensemble weights via a fast, in-register serial reduction.
 *
 * This kernel is the "F1 Car" in our toolbox, designed for maximum performance when `num_exits`
 * is small (e.g., <= 64). It is launched with one thread per sample. Each thread performs the
 * entire weight calculation for its sample by looping serially over the exits, using a small
 * private array ("workbench") that fits entirely in fast registers.
 *
 * This approach has minimal overhead, requires no barriers or shared memory, and achieves
 * 100% warp utilization, making it the optimal solution for the most common use cases. It
 * executes the conceptual Node 6 of the main execution graph.
 *
 * Host Launch: A 1D NDRange with global_size = (padded_batch_size).
 * Constraint: The `num_exits` passed by the host MUST be <= MAX_EXITS_ENSEMBLE.
 */
__kernel void ensemble_weights_reg_reduce(
    // Inputs
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict targets_mask,
    // Output
    __global SCALAR_TYPE *__restrict ensemble_weights,
    // Dimensions
    int padded_batch_size,
    int output_classes,
    int num_exits) {

    // Each thread is responsible for one sample in the batch.
    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size || targets_mask[b_idx] < 0.5f) {
        return;
    }

    // --- STEP 1: The Private Register "Workbench" ---
    // A small, fixed-size private array to hold intermediate values.
    // This is the core reason this kernel is fast and has a size limit.
    SCALAR_TYPE p_confidences[MAX_EXITS_ENSEMBLE];

    // --- STEP 2: Calculate All Confidence Values and the Denominator ---
    // This serial loop is extremely fast for small `num_exits`.
    SCALAR_TYPE sum_exp_confidences = 0.0f;
    for (int e = 0; e < num_exits; ++e) {
        // Find max probability (confidence score) for this exit
        SCALAR_TYPE max_prob = 0.0f;
        const uint  base_idx = e * padded_batch_size * output_classes + b_idx * output_classes;
        // This inner read is fully coalesced and efficient.
        for (int c = 0; c < output_classes; ++c) {
            max_prob = fmax(max_prob, exit_probs[base_idx + c]);
        }

        // Store the exponentiated confidence on our workbench and add to the sum
        p_confidences[e] = exp(max_prob);
        sum_exp_confidences += p_confidences[e];
    }

    sum_exp_confidences = fmax(sum_exp_confidences, (SCALAR_TYPE)1e-7f);

    // --- STEP 3: Final Scatter and Write to Global Memory ---
    // Loop over the workbench one last time to calculate and write the final weights.
    for (int e = 0; e < num_exits; ++e) {
        ensemble_weights[b_idx * num_exits + e] = p_confidences[e] / sum_exp_confidences;
    }

    // The logic for blending probabilities has been DELETED.
    // It will be handled by the separate `blend_ensemble_probabilities` kernel.
}

/**
 * @brief (Tier 2: Medium-Scale - CORRECTED) Computes ensemble weights for one sample using a single work-group.
 *
 * This kernel represents the "Cargo Van": a powerful, self-contained team that can solve
 * a medium-sized problem (`64 < num_exits <= 512`) with high efficiency. It avoids slow
 * global memory writes by performing a fast, parallel reduction within its `__local` memory
 * workbench.
 *
 * Host Launch: Must be launched with a 1D NDRange where each WORK-GROUP is responsible for
 * one sample. Global size = (batch_size * local_size), Local size = (e.g., 256).
 */
__kernel void ensemble_weights_local_reduce(
    // Local memory, dynamically allocated by the host. Size = local_size * sizeof(SCALAR_TYPE)
    __local SCALAR_TYPE *l_reduction_mem,

    // Inputs
    __global const SCALAR_TYPE *__restrict exit_probs, // Data layout is [num_exits, batch_size, output_classes]
    // Outputs
    __global SCALAR_TYPE *__restrict ensemble_weights, // [batch_size, num_exits]
    // Dimensions
    int num_exits,
    int output_classes,
    int padded_batch_size // CORRECTED: Added padded_batch_size for correct stride calculation
) {
    // --- Step 0: Get IDs and Dimensions ---
    // b_idx is the sample this entire work-group is responsible for.
    const uint b_idx = get_group_id(0);
    const uint tid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- Step 1: Thread-Local Partial Sum Calculation ---
    // Each thread calculates a partial sum of exp(confidence) for its assigned exits.
    SCALAR_TYPE p_sum = (SCALAR_TYPE)0.0f; // This sum is stored in a private register

    for (int e = tid; e < num_exits; e += lsize) {
        // Find max confidence for this specific exit (e)
        SCALAR_TYPE max_prob = (SCALAR_TYPE)0.0f;

        // CORRECTED INDEXING: This now correctly navigates the (exit, batch, class) memory layout.
        const uint prob_base_idx = e * padded_batch_size * output_classes + b_idx * output_classes;

        for (int c = 0; c < output_classes; c++) {
            max_prob = fmax(max_prob, exit_probs[prob_base_idx + c]);
        }
        p_sum += exp(max_prob);
    }

    // --- Step 2: Parallel Reduction using Shared Local Memory ---
    // Each thread places its private partial sum into the shared workbench.
    l_reduction_mem[tid] = p_sum;

    // Synchronize to ensure all threads have finished writing their p_sum.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the reduction in-place in local memory.
    // This is a standard, highly optimized reduction tree.
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            l_reduction_mem[tid] += l_reduction_mem[tid + stride];
        }
        // A barrier is required at each level of the tree to prevent race conditions.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Step 3: Broadcast the Final Denominator ---
    // At this point, l_reduction_mem[0] holds the final, total sum for this sample.
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure reduction is visible to all threads.
    const SCALAR_TYPE denominator = fmax(l_reduction_mem[0], (SCALAR_TYPE)1e-7f);

    // --- Step 4: Final Weight Calculation (The "Scatter") ---
    // Each thread re-traverses its assigned exits, but now with the final denominator.
    for (int e = tid; e < num_exits; e += lsize) {
        // Re-calculate this thread's exp(confidence) for this exit
        SCALAR_TYPE max_prob = (SCALAR_TYPE)0.0f;
        // CORRECTED INDEXING (must be the same as above)
        const uint prob_base_idx = e * padded_batch_size * output_classes + b_idx * output_classes;
        for (int c = 0; c < output_classes; c++) {
            max_prob = fmax(max_prob, exit_probs[prob_base_idx + c]);
        }
        const SCALAR_TYPE exp_confidence = exp(max_prob);

        // Calculate and write the final weight to global memory
        ensemble_weights[b_idx * num_exits + e] = exp_confidence / denominator;
    }
}

/**
 * @brief (Tier 3, Kernel 1/3: The "Map" Stage - CORRECTED) Calculates exp(confidence) for every exit.
 *
 * This is the first kernel in the scalable "Cargo Ship" chain. It is an embarrassingly
 * parallel "map" operation. Each thread is assigned a single (sample, exit) pair and is
 * responsible for calculating one value: the exponentiated confidence score.
 *
 * It reads from the main `exit_probs` buffer and writes its single result to a temporary
 * global buffer, which will be consumed by the next kernel in the chain (`reduce_partial_sums`).
 *
 * Host Launch: A 2D NDRange with global_size = (padded_batch_size, num_exits).
 */
__kernel void ensemble_weights_map_exp_conf(
    // Inputs
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict targets_mask, // Assuming we still mask invalid samples
    // Output
    __global SCALAR_TYPE *__restrict temp_exp_conf_buf, // [batch_size, num_exits]
    // Dimensions
    int padded_batch_size,
    int num_exits,
    int output_classes) {
    // --- Step 1: Get Thread Assignment from 2D Grid ---
    const uint b_idx = get_global_id(0); // Which sample we are working on
    const uint e_idx = get_global_id(1); // Which exit we are working on

    // --- Step 2: Boundary Checks and Masking ---
    // Standard check to ensure we don't process padded items
    if (b_idx >= padded_batch_size || e_idx >= num_exits) {
        return;
    }
    // Masking check for valid samples
    if (targets_mask[b_idx] < 0.5f) {
        // Explicitly zero out the result for masked samples to ensure correctness
        temp_exp_conf_buf[b_idx * num_exits + e_idx] = 0.0f;
        return;
    }

    // --- Step 3: Find Max Confidence (The "Map" Operation) ---
    SCALAR_TYPE max_prob = 0.0f;
    // CORRECTED INDEXING: This now correctly navigates the (exit, batch, class) memory layout.
    const uint prob_base_idx = e_idx * padded_batch_size * output_classes + b_idx * output_classes;

    // This loop performs a fully coalesced read from global memory, making it highly efficient.
    for (int c = 0; c < output_classes; c++) {
        max_prob = fmax(max_prob, exit_probs[prob_base_idx + c]);
    }

    // --- Step 4: Calculate and Write Single Result to Global Memory ---
    const SCALAR_TYPE exp_confidence             = exp(max_prob);
    temp_exp_conf_buf[b_idx * num_exits + e_idx] = exp_confidence;
}

/**
 * @brief (Tier 3, Kernel 2/3: The "Reduce" Stage) Reduces a large buffer of values into a small one.
 *
 * This kernel is the parallel workhorse of the scalable "Cargo Ship" chain. It takes the large
 * temporary buffer of exponentiated confidences and reduces it to a much smaller buffer of
 * partial sums. Each work-group is responsible for reducing one large chunk of the input buffer.
 * It performs its reduction efficiently using __local memory, writing only a single value as output.
 *
 * Host Launch: A 2D grid of work-groups. Global size = (batch_size * local_size_x, num_chunks * local_size_y),
 * Local size depends on the reduction strategy, but conceptually it's one work-group per chunk.
 * A simpler launch is global_size=(batch_size, num_chunks*workgroup_size), local_size=(1,workgroup_size).
 */
__kernel void reduce_partial_sums(
    // Local memory, dynamically allocated by the host. Size = local_size * sizeof(SCALAR_TYPE)
    __local SCALAR_TYPE *l_reduction_mem,

    // Inputs
    __global const SCALAR_TYPE *__restrict temp_exp_conf_buf, // [batch_size, num_exits]
    // Output
    __global SCALAR_TYPE *__restrict temp_partial_sums_buf, // [batch_size, num_chunks]
    // Dimensions
    int num_exits) {

    // --- Step 1: Get Work-Group and Thread Assignment ---
    // This work-group is responsible for one sample (b_idx) and one chunk (chunk_idx) of its exits.
    const uint b_idx      = get_group_id(0);
    const uint chunk_idx  = get_group_id(1);
    const uint num_chunks = get_num_groups(1); // Total number of chunks per sample

    const uint tid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Each work-group will process a fixed number of items, e.g., 2048.
    // The last chunk may process fewer.
    const uint items_per_group = (num_exits + num_chunks - 1) / num_chunks;
    const uint chunk_start_idx = chunk_idx * items_per_group;
    const uint chunk_end_idx   = min(chunk_start_idx + items_per_group, (uint)num_exits);

    // --- Step 2: Thread-Local Partial Sum from Global Memory ---
    // Each thread loops through its assigned portion of this GROUP's chunk.
    SCALAR_TYPE p_sum = 0.0f;
    for (int e = chunk_start_idx + tid; e < chunk_end_idx; e += lsize) {
        p_sum += temp_exp_conf_buf[b_idx * num_exits + e];
    }

    // --- Step 3: Parallel Reduction using Shared Local Memory ---
    // This logic is identical to the Tier 2 kernel.
    l_reduction_mem[tid] = p_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            l_reduction_mem[tid] += l_reduction_mem[tid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Step 4: Write Single Partial Sum Result ---
    // Only the leader of each work-group writes the final sum for its chunk.
    if (tid == 0) {
        const uint output_idx             = b_idx * num_chunks + chunk_idx;
        temp_partial_sums_buf[output_idx] = l_reduction_mem[0];
    }
}

/**
 * @brief (Tier 3, Kernel 3/3: The "Finalize" Stage) Aggregates partial sums and computes final weights.
 *
 * This is the final kernel in the scalable "Cargo Ship" chain. It is launched with one thread
 * per sample. Its job is to perform the last, small reduction on the partial sums to calculate
 * the final denominator. It then loops over the large buffer of pre-calculated numerators
 * (from the first kernel) to compute and write the final `ensemble_weights`.
 *
 * Host Launch: A 1D NDRange with global_size = (padded_batch_size).
 */
__kernel void ensemble_weights_finalize(
    // Inputs (the two temporary buffers from the previous kernels)
    __global const SCALAR_TYPE *__restrict temp_exp_conf_buf,     // [batch_size, num_exits]
    __global const SCALAR_TYPE *__restrict temp_partial_sums_buf, // [batch_size, num_chunks]
    __global const SCALAR_TYPE *__restrict targets_mask,          // To mask invalid samples
    // Output
    __global SCALAR_TYPE *__restrict ensemble_weights, // [batch_size, num_exits]
    // Dimensions
    int padded_batch_size,
    int num_exits,
    int num_chunks // The number of work-groups/chunks from the previous kernel
) {
    // --- Step 1: Get Thread Assignment ---
    // Each thread is responsible for one sample in the batch.
    const uint b_idx = get_global_id(0);

    // --- Step 2: Boundary Checks and Masking ---
    if (b_idx >= padded_batch_size) {
        return;
    }
    // For masked samples, the weights will simply be calculated as 0/denominator.
    // If temp_exp_conf_buf was zeroed for this sample, this path is safe.
    if (targets_mask[b_idx] < 0.5f) {
        // Can optionally zero out the entire weight vector here for clarity
        for (int e = 0; e < num_exits; e++) {
            ensemble_weights[b_idx * num_exits + e] = 0.0f;
        }
        return;
    }

    // --- Step 3: Final Aggregation (fast serial reduction) ---
    // Each thread reads the small list of partial sums for its sample.
    SCALAR_TYPE total_denominator = 0.0f;
    const uint  partials_base_idx = b_idx * num_chunks;
    for (int i = 0; i < num_chunks; ++i) {
        total_denominator += temp_partial_sums_buf[partials_base_idx + i];
    }
    total_denominator = fmax(total_denominator, (SCALAR_TYPE)1e-7f);

    // --- Step 4: Final Scatter Operation ---
    // Each thread now loops over all exits for its sample to calculate the final weight.
    const uint numerators_base_idx = b_idx * num_exits;
    for (int e = 0; e < num_exits; ++e) {
        // Read the pre-calculated numerator from the first kernel's output buffer
        const SCALAR_TYPE numerator = temp_exp_conf_buf[numerators_base_idx + e];
        // Perform the final division and write to the final output buffer
        ensemble_weights[numerators_base_idx + e] = numerator / total_denominator;
    }
}

/**
 * @brief (Node 7 - CORRECTED) Blends exit probabilities using pre-calculated weights to get the final distribution.
 *
 * This kernel executes the "Blend Probabilities" node in the main execution graph. It is fully
 * decoupled from weight calculation. Using a tiling strategy over the `output_classes` dimension,
 * it performs a weighted average of all exit probabilities for each sample. This ensures high
 * performance through coalesced memory reads, regardless of the size of `num_exits` or
 * `output_classes`.
 *
 * Host Launch: A 1D NDRange with global_size = (padded_batch_size).
 */
__kernel void blend_ensemble_probabilities(
    // Inputs
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict ensemble_weights,
    __global const SCALAR_TYPE *__restrict targets_mask,
    // Output
    __global SCALAR_TYPE *__restrict ensemble_probs,
    // Dimensions
    int padded_batch_size,
    int output_classes,
    int num_exits) {
    // Each thread is responsible for one sample in the batch.
    const uint b_idx = get_global_id(0);
    if (b_idx >= padded_batch_size || targets_mask[b_idx] < 0.5f) {
        return;
    }

    // --- Tiled and Coalesced Computation of Final Probabilities ---
    // The outer loop iterates through the final probability vector in "tiles".
    for (int c_base = 0; c_base < output_classes; c_base += C_TILE_SIZE) {

        // A small, fixed-size private array to hold the blended sums for the current tile.
        SCALAR_TYPE p_final_probs_tile[C_TILE_SIZE] = {0.0f};

        // For this tile, loop over all exits to accumulate the weighted sum.
        for (int e = 0; e < num_exits; ++e) {
            // Read the single weight for this sample and exit.
            const SCALAR_TYPE w = ensemble_weights[b_idx * num_exits + e];

            // If a weight is zero, we can skip the expensive memory reads for that exit.
            if (w == (SCALAR_TYPE)0.0f)
                continue;

            // CORRECTED INDEXING: This now correctly navigates the (exit, batch, class) memory layout.
            const uint prob_base_idx = e * padded_batch_size * output_classes + b_idx * output_classes + c_base;

            // Inner loop over the TILE of classes performs a COALESCED READ.
            for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
                // Boundary check for the very last tile.
                if (c_base + c_local < output_classes) {
                    p_final_probs_tile[c_local] += w * exit_probs[prob_base_idx + c_local];
                }
            }
        }

        // Write the completed tile from the private workbench to global memory.
        for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
            if (c_base + c_local < output_classes) {
                ensemble_probs[b_idx * output_classes + c_base + c_local] = p_final_probs_tile[c_local];
            }
        }
    }
}
