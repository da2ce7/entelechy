// autograd.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (FULLY DYNAMIC & MEMORY EFFICIENT) Calculates gradients for exit-specific parameters.
 *
 * This version is the most robust, flexible, and high-performance implementation. It adapts
 * the high-performance reduction pattern to be memory-efficient even with a very large
 * `output_classes` dimension (e.g., 1000+).
 *
 * It processes the output classes in "tiles", using a combination of private register arrays
 * and a small, reused local memory buffer for reduction. This keeps local memory usage low and
 * constant, solving the scaling problem of the previous implementation.
 */
__kernel void calculate_exit_gradients(
    // Local Memory (Passed dynamically, usage is now small and constant)
    __local SCALAR_TYPE *local_grad_w,
    __local SCALAR_TYPE *local_grad_b,

    // Inputs
    __global const SCALAR_TYPE *__restrict hidden,
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict ensemble_weights,
    __global const int *__restrict targets,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict exit_weights,

    // Outputs
    __global SCALAR_TYPE *__restrict grad_exit_weights,
    __global SCALAR_TYPE *__restrict grad_exit_biases,
    __global SCALAR_TYPE *__restrict grad_hidden_contributions,

    // Dims
    int padded_batch_size,
    int hidden_dim,
    int output_classes,
    int num_exits) {

    // Global and local IDs
    const uint e_idx = get_group_id(0);
    const uint h_idx = get_group_id(1);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- Main loop to process output_classes in tiles (for parameter gradients) ---
    for (int c_base = 0; c_base < output_classes; c_base += C_TILE_SIZE) {

        // --- STEP 1: Accumulate into PRIVATE register arrays ---
        // This is the key change. We use fast private memory (registers) for the
        // accumulation over the batch dimension. The array is small and fixed-size.
        SCALAR_TYPE p_grad_w[C_TILE_SIZE] = {0.0f};
        SCALAR_TYPE p_grad_b[C_TILE_SIZE] = {0.0f};

        // Each thread loops through its assigned portion of the batch
        for (int b = lid; b < padded_batch_size; b += lsize) {
            if (targets_mask[b] < (SCALAR_TYPE)0.5f)
                continue;

            const int         true_class = targets[b];
            const SCALAR_TYPE weight_b_e = ensemble_weights[b * num_exits + e_idx];
            const SCALAR_TYPE hidden_val = hidden[b * hidden_dim + h_idx];

// Loop over the *current tile* of output classes
#pragma unroll
            for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
                const int c_global = c_base + c_local;
                if (c_global >= output_classes)
                    continue; // Boundary check for the last tile

                SCALAR_TYPE is_target        = (c_global == true_class) ? 1.0f : 0.0f;
                SCALAR_TYPE prob             = exit_probs[e_idx * padded_batch_size * output_classes + b * output_classes + c_global];
                SCALAR_TYPE d_loss_d_logit_c = weight_b_e * (prob - is_target);

                // Accumulate into the private arrays
                p_grad_w[c_local] += d_loss_d_logit_c * hidden_val;
                if (h_idx == 0) {
                    p_grad_b[c_local] += d_loss_d_logit_c;
                }
            }
        }

        // --- STEP 2: Reduce each element of the tile in Local Memory ---
        // This reduction is done once per element in the tile.
        for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
            const int c_global = c_base + c_local;
            if (c_global >= output_classes)
                continue;

            // Transfer from private to local memory for this specific class index
            local_grad_w[lid] = p_grad_w[c_local];
            if (h_idx == 0) {
                local_grad_b[lid] = p_grad_b[c_local];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Perform the parallel reduction across work-items
            for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_grad_w[lid] += local_grad_w[lid + stride];
                    if (h_idx == 0) {
                        local_grad_b[lid] += local_grad_b[lid + stride];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            // --- STEP 3: Write final result for this class from thread 0 ---
            if (lid == 0) {
                grad_exit_weights[e_idx * hidden_dim * output_classes + h_idx * output_classes + c_global] = local_grad_w[0];
                if (h_idx == 0) {
                    grad_exit_biases[e_idx * output_classes + c_global] = local_grad_b[0];
                }
            }
        }
    }

    // --- STEP 4: Calculate grad_hidden_contributions (must be done separately) ---
    // The tiling logic for parameter gradients does not apply here, as this calculation
    // requires a full sum over all output classes for each sample.
    for (int b = lid; b < padded_batch_size; b += lsize) {
        if (targets_mask[b] < (SCALAR_TYPE)0.5f)
            continue;

        const int         true_class          = targets[b];
        const SCALAR_TYPE weight_b_e          = ensemble_weights[b * num_exits + e_idx];
        SCALAR_TYPE       grad_h_contribution = (SCALAR_TYPE)0.0f;

        for (int c = 0; c < output_classes; ++c) {
            SCALAR_TYPE is_target        = (c == true_class) ? 1.0f : 0.0f;
            SCALAR_TYPE prob             = exit_probs[e_idx * padded_batch_size * output_classes + b * output_classes + c];
            SCALAR_TYPE d_loss_d_logit_c = weight_b_e * (prob - is_target);
            grad_h_contribution += d_loss_d_logit_c * exit_weights[e_idx * hidden_dim * output_classes + h_idx * output_classes + c];
        }
        grad_hidden_contributions[e_idx * padded_batch_size * hidden_dim + b * hidden_dim + h_idx] = grad_h_contribution;
    }
}

/**
 * @brief (Stage 2 - UPDATED) Aggregates per-exit contributions and calculates final gradients for the shared layer.
 *
 * This kernel executes the second stage of the main backpropagation chain. It has been updated
 * to use fully dynamic __local memory supplied by the host, removing any compile-time
 * assumptions about work-group size.
 */
__kernel void calculate_shared_gradients(
    // Local Memory (UPDATED: Passed dynamically from host)
    __local SCALAR_TYPE *local_grad_w,
    __local SCALAR_TYPE *local_grad_b,

    // Inputs
    __global const SCALAR_TYPE *__restrict input,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global const SCALAR_TYPE *__restrict hidden,
    __global const SCALAR_TYPE *__restrict grad_hidden_contributions,

    // Outputs
    __global SCALAR_TYPE *__restrict grad_weights,
    __global SCALAR_TYPE *__restrict grad_biases,

    // Dimensions
    int padded_batch_size,
    int input_dim,
    int hidden_dim,
    int num_exits) {

    // Global work-group identifiers for which final gradient to calculate
    const uint i_idx = get_group_id(0); // Index into input_dim
    const uint j_idx = get_group_id(1); // Index into hidden_dim

    // Local thread identifiers for intra-group parallelism
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Private registers for accumulating sums over the batch dimension
    SCALAR_TYPE p_grad_w = (SCALAR_TYPE)0.0f;
    SCALAR_TYPE p_grad_b = (SCALAR_TYPE)0.0f; // Only used by groups where i_idx = 0

    // Each thread loops through its assigned portion of the batch
    for (int b = lid; b < padded_batch_size; b += lsize) {
        if (input_mask[b] < (SCALAR_TYPE)0.5f)
            continue;

        // --- STEP 1: Aggregate contributions from all exits ---
        SCALAR_TYPE aggregated_grad_h = (SCALAR_TYPE)0.0f;
        for (int e = 0; e < num_exits; ++e) {
            aggregated_grad_h += grad_hidden_contributions[e * padded_batch_size * hidden_dim + b * hidden_dim + j_idx];
        }

        // --- STEP 2: Backpropagate through the activation function's derivative ---
        // We assume a ReLU activation function, so the derivative is 1 if hidden > 0, and 0 otherwise.
        SCALAR_TYPE hidden_val          = hidden[b * hidden_dim + j_idx];
        SCALAR_TYPE grad_pre_activation = aggregated_grad_h * (hidden_val > (SCALAR_TYPE)0.0f ? (SCALAR_TYPE)1.0f : (SCALAR_TYPE)0.0f);

        // --- STEP 3: Calculate gradient contribution for weights and biases ---
        SCALAR_TYPE input_val = input[b * input_dim + i_idx];

        // Accumulate into private registers
        p_grad_w += grad_pre_activation * input_val;
        if (i_idx == 0) { // To avoid redundant work, only the first column of work-groups handles bias gradients
            p_grad_b += grad_pre_activation;
        }
    }

    // --- Start of Reduction ---
    local_grad_w[lid] = p_grad_w;
    if (i_idx == 0) {
        local_grad_b[lid] = p_grad_b;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_grad_w[lid] += local_grad_w[lid + stride];
            if (i_idx == 0) {
                local_grad_b[lid] += local_grad_b[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Final Write to Global Memory ---
    if (lid == 0) {
        grad_weights[i_idx * hidden_dim + j_idx] = local_grad_w[0];
        if (i_idx == 0) {
            grad_biases[j_idx] = local_grad_b[0];
        }
    }
}

/**
 * @brief (Branch B - UPDATED) Calculates gradients for the temperature parameters.
 *
 * This kernel has been updated to use fully dynamic __local memory supplied by the host,
 * removing any compile-time assumptions about work-group size and aligning it with the
 * other high-performance gradient kernels.
 */
__kernel void calculate_temp_gradients(
    // Local Memory (UPDATED: Passed dynamically from host)
    __local SCALAR_TYPE *local_grad_sum,

    // Inputs
    __global const SCALAR_TYPE *__restrict unscaled_logits,
    __global const SCALAR_TYPE *__restrict exit_probs,
    __global const SCALAR_TYPE *__restrict ensemble_weights,
    __global const SCALAR_TYPE *__restrict temperatures,
    __global const int *__restrict targets,
    __global const SCALAR_TYPE *__restrict targets_mask,

    // Output
    __global SCALAR_TYPE *__restrict grad_temps,

    // Dimensions
    int padded_batch_size,
    int output_classes,
    int num_exits) {

    // Global work-group ID determines which temperature to process
    const uint e_idx = get_group_id(0);

    // Local thread identifiers for intra-group parallelism
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Private register for accumulating the gradient sum over the batch
    SCALAR_TYPE p_grad_sum = (SCALAR_TYPE)0.0f;

    // Pre-calculate for efficiency inside the loop
    const SCALAR_TYPE temp        = temperatures[e_idx];
    const SCALAR_TYPE temp_sq_inv = -1.0f / (temp * temp); // This is part of the d(logit)/dT derivative

    // Each thread loops through its assigned portion of the batch
    for (int b = lid; b < padded_batch_size; b += lsize) {
        if (targets_mask[b] < (SCALAR_TYPE)0.5f)
            continue;

        /* ... inner loop logic is unchanged ... */

        const int         true_class                    = targets[b];
        const SCALAR_TYPE weight_b_e                    = ensemble_weights[b * num_exits + e_idx];
        SCALAR_TYPE       grad_contribution_from_sample = (SCALAR_TYPE)0.0f;

        for (int c = 0; c < output_classes; ++c) {
            SCALAR_TYPE is_target          = (c == true_class) ? 1.0f : 0.0f;
            SCALAR_TYPE prob               = exit_probs[e_idx * padded_batch_size * output_classes + b * output_classes + c];
            SCALAR_TYPE d_loss_d_logit_c   = weight_b_e * (prob - is_target);
            SCALAR_TYPE unscaled_logit_val = unscaled_logits[e_idx * padded_batch_size * output_classes + b * output_classes + c];
            SCALAR_TYPE d_logit_d_temp     = unscaled_logit_val * temp_sq_inv;
            grad_contribution_from_sample += d_loss_d_logit_c * d_logit_d_temp;
        }

        p_grad_sum += grad_contribution_from_sample;
    }

    // --- PARALLEL REDUCTION ---
    local_grad_sum[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_grad_sum[lid] += local_grad_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Only thread 0 writes the final, reduced result for this temperature
    if (lid == 0) {
        grad_temps[e_idx] = local_grad_sum[0];
    }
}

/**
 * @brief (Node 8 - Stage 1) Calculates partial cross-entropy losses for the batch.
 *
 * This is the first stage of a two-stage parallel reduction for the total batch loss. Each
 * work-group is responsible for a subset of the samples in the batch. It computes the
 * cross-entropy loss for each of its assigned samples and then performs an efficient
 * reduction within its own __local memory to produce a single "partial loss" sum.
 *
 * This partial sum is then written to a temporary global buffer, which will be consumed
 * by the `aggregate_partial_losses` kernel in the next stage.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange over the batch dimension, e.g., global_size=(padded_batch_size,).
 * - Host must provide a __local memory buffer of size: local_size[0] * sizeof(float).
 */
__kernel void calculate_partial_losses(
    // Local memory, dynamically allocated by the host
    __local float *l_loss_sums,

    // Inputs
    __global const SCALAR_TYPE *__restrict ensemble_probs_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const int *__restrict targets_buf,

    // Output
    __global float *__restrict partial_loss_buf, // Buffer for one float per work-group

    // Dimensions
    int padded_batch_size,
    int output_classes) {

    // --- Get IDs and Sizes ---
    const uint gid      = get_global_id(0);
    const uint lid      = get_local_id(0);
    const uint lsize    = get_local_size(0);
    const uint group_id = get_group_id(0);
    const uint gsize    = get_global_size(0);

    // --- Phase 1: Private Accumulation ---
    // Each thread calculates a sum of losses for its assigned samples.
    // This sum is stored in a fast private register.
    float p_loss_sum = 0.0f;

    // This loop structure ensures that all samples are covered, even if the number of
    // work-items (gsize) is smaller than the number of samples (padded_batch_size).
    for (uint b = gid; b < padded_batch_size; b += gsize) {
        // Skip any samples that are part of the padding mask
        if (targets_mask[b] < (SCALAR_TYPE)0.5f) {
            continue;
        }

        // Get the ground-truth label for this sample
        const int true_class = targets_buf[b];

        // Find the probability assigned to the true class by the ensemble
        const uint  prob_idx = b * output_classes + true_class;
        SCALAR_TYPE prob     = ensemble_probs_buf[prob_idx];

        // Calculate cross-entropy loss for this single sample.
        // Use fmax to prevent log(0) which would result in -inf.
        p_loss_sum += -log(fmax((float)prob, 1e-7f));
    }

    // --- Phase 2: Parallel Reduction in Local Memory ---
    // Each thread places its privately computed sum into the shared local "workbench".
    l_loss_sums[lid] = p_loss_sum;

    // Synchronize to ensure all private sums are written to local memory before reduction begins.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform the reduction in-place. Each step halves the number of active threads.
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            l_loss_sums[lid] += l_loss_sums[lid + stride];
        }
        // A barrier is essential here to prevent race conditions between reduction levels.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Phase 3: Final Write ---
    // Only the first thread in each work-group (the "leader") writes the final,
    // reduced partial sum to global memory.
    if (lid == 0) {
        partial_loss_buf[group_id] = l_loss_sums[0];
    }
}

/**
 * @brief (Node 9 - Stage 2) Aggregates partial loss sums into a final total loss.
 *
 * This is the second and final stage of the loss reduction. This kernel is launched with
 * a single work-group to sum the small number of partial results from the previous stage.
 * It uses the same efficient parallel reduction pattern to produce the single, final
 * floating-point value representing the total loss for the batch.
 *
 * Host Assumptions:
 * - Launched with a single work-group, where global_size == local_size.
 * - Host must provide a __local memory buffer of size: local_size[0] * sizeof(float).
 */
__kernel void aggregate_partial_losses(
    // Local memory, dynamically allocated by the host
    __local float *l_reduction_mem,

    // Inputs
    __global const float *__restrict partial_loss_buf,
    // Output
    __global float *__restrict final_loss_buf,
    // Dimensions
    int num_partial_sums) {

    // --- Get IDs ---
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- Phase 1: Private Accumulation ---
    // Each thread in the work-group sums up a portion of the partial_loss_buf.
    float p_sum = 0.0f;
    for (int i = lid; i < num_partial_sums; i += lsize) {
        p_sum += partial_loss_buf[i];
    }

    // --- Phase 2: Parallel Reduction in Local Memory ---
    l_reduction_mem[lid] = p_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            l_reduction_mem[lid] += l_reduction_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Phase 3: Final Write ---
    // The leader of this single work-group writes the grand total to the final output buffer.
    if (lid == 0) {
        final_loss_buf[0] = l_reduction_mem[0];
    }
}
