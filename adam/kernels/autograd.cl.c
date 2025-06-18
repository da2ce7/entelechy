// autograd.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 8) Implements the first stage of a two-stage parallel reduction for the batch loss.
 *
 * Each work-group is assigned a slice of the batch. Threads within the group first compute
 * a partial sum of the cross-entropy loss in private registers. These private sums are then
 * combined using a single, efficient parallel reduction within the work-group's local memory.
 */
__kernel void calculate_partial_losses(
    // Local memory, dynamically allocated by the host
    __local float *l_loss_sums,

    // Inputs
    __global const SCALAR_TYPE *__restrict ensemble_probs_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const int *__restrict targets_buf,

    // Output
    __global float *__restrict partial_loss_buf,

    // Dimensions
    int padded_batch_size,
    int output_classes) {

    const uint gid      = get_global_id(0);
    const uint lid      = get_local_id(0);
    const uint lsize    = get_local_size(0);
    const uint group_id = get_group_id(0);

    float p_loss_sum = 0.0f;
    for (uint b = gid; b < padded_batch_size; b += get_global_size(0)) {
        if (targets_mask[b] < (SCALAR_TYPE)0.5f) {
            continue;
        }
        const int   true_class = targets_buf[b];
        const uint  prob_idx   = b * output_classes + true_class;
        SCALAR_TYPE prob       = ensemble_probs_buf[prob_idx];
        p_loss_sum += -log(fmax((float)prob, 1e-7f));
    }

    l_loss_sums[lid] = p_loss_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            l_loss_sums[lid] += l_loss_sums[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial_loss_buf[group_id] = l_loss_sums[0];
    }
}

/**
 * @brief (Node 9) Implements the second and final stage of the parallel loss reduction.
 *
 * This kernel is launched with a single work-group to sum the small number of partial results
 * from the previous stage. It uses the same efficient local memory reduction pattern to produce the
 * single, final floating-point value representing the total loss for the batch.
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

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    float p_sum = 0.0f;
    for (int i = lid; i < num_partial_sums; i += lsize) {
        p_sum += partial_loss_buf[i];
    }

    l_reduction_mem[lid] = p_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            l_reduction_mem[lid] += l_reduction_mem[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        final_loss_buf[0] = l_reduction_mem[0];
    }
}

/**
 * @brief (Node 10) Calculates gradients for all exit-specific parameters.
 *
 * This implementation is memory-efficient, tiling the calculation over the `output_classes` dimension.
 * For each tile, gradients are first accumulated over the batch into private registers. Then, a fast
 * parallel reduction is performed in local memory for each class within the tile. The `grad_hidden_contributions`
 * are calculated separately in a second, untiled pass.
 */
__kernel void calculate_exit_gradients(
    // Local Memory
    __local SCALAR_TYPE *local_grad_w,
    __local SCALAR_TYPE *local_grad_b,

    // Inputs
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict exit_probs_buf,
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf,
    __global const int *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict exit_weights_buf,

    // Outputs
    __global SCALAR_TYPE *__restrict grad_exit_weights,
    __global SCALAR_TYPE *__restrict grad_exit_biases,
    __global SCALAR_TYPE *__restrict grad_hidden_contributions_buf,

    // Dims
    int padded_batch_size,
    int hidden_dim,
    int output_classes,
    int num_exits) {

    const uint e_idx = get_group_id(0);
    const uint h_idx = get_group_id(1);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Tiled pass for parameter gradients
    for (int c_base = 0; c_base < output_classes; c_base += C_TILE_SIZE) {
        SCALAR_TYPE p_grad_w[C_TILE_SIZE] = {SCALAR_ZERO};
        SCALAR_TYPE p_grad_b[C_TILE_SIZE] = {SCALAR_ZERO};

        for (int b = lid; b < padded_batch_size; b += lsize) {
            if (targets_mask[b] < (SCALAR_TYPE)0.5f) {
                continue;
            }

            const uint        h_block                  = h_idx / SIMD_WIDTH;
            const uint        h_lane                   = h_idx % SIMD_WIDTH;
            const uint        padded_hidden_dim_blocks = hidden_dim / SIMD_WIDTH;
            const uint        physical_hidden_idx      = b * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
            const SCALAR_TYPE h_val                    = hidden_buf[physical_hidden_idx];

            const SCALAR_TYPE weight_be  = ensemble_weights_buf[b * num_exits + e_idx];
            const int         true_class = targets_buf[b];

#pragma unroll
            for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
                const int c_global = c_base + c_local;
                if (c_global >= output_classes) {
                    continue;
                }
                SCALAR_TYPE is_target      = select(SCALAR_ZERO, (SCALAR_TYPE)1.0f, c_global == true_class);
                SCALAR_TYPE prob           = exit_probs_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c_global];
                SCALAR_TYPE d_loss_d_logit = weight_be * (prob - is_target);

                p_grad_w[c_local] += d_loss_d_logit * h_val;
                if (h_idx == 0) {
                    p_grad_b[c_local] += d_loss_d_logit;
                }
            }
        }

        for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {
            const int c_global = c_base + c_local;
            if (c_global >= output_classes) {
                continue;
            }
            local_grad_w[lid] = p_grad_w[c_local];
            if (h_idx == 0) {
                local_grad_b[lid] = p_grad_b[c_local];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    local_grad_w[lid] += local_grad_w[lid + stride];
                    if (h_idx == 0) {
                        local_grad_b[lid] += local_grad_b[lid + stride];
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (lid == 0) {
                grad_exit_weights[e_idx * hidden_dim * output_classes + h_idx * output_classes + c_global] = local_grad_w[0];
                if (h_idx == 0) {
                    grad_exit_biases[e_idx * output_classes + c_global] = local_grad_b[0];
                }
            }
        }
    }

    // Separate pass for hidden layer contributions
    for (int b = lid; b < padded_batch_size; b += lsize) {
        if (targets_mask[b] < (SCALAR_TYPE)0.5f) {
            grad_hidden_contributions_buf[e_idx * padded_batch_size * hidden_dim + b * hidden_dim + h_idx] = SCALAR_ZERO;
            continue;
        }

        SCALAR_TYPE       grad_h_contribution = SCALAR_ZERO;
        const SCALAR_TYPE weight_be           = ensemble_weights_buf[b * num_exits + e_idx];
        const int         true_class          = targets_buf[b];

        for (int c = 0; c < output_classes; ++c) {
            SCALAR_TYPE is_target      = select(SCALAR_ZERO, (SCALAR_TYPE)1.0f, c == true_class);
            SCALAR_TYPE prob           = exit_probs_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c];
            SCALAR_TYPE d_loss_d_logit = weight_be * (prob - is_target);
            grad_h_contribution += d_loss_d_logit * exit_weights_buf[e_idx * hidden_dim * output_classes + h_idx * output_classes + c];
        }
        grad_hidden_contributions_buf[e_idx * padded_batch_size * hidden_dim + b * hidden_dim + h_idx] = grad_h_contribution;
    }
}

/**
 * @brief (Node 11) Implements an embarrassingly parallel "map" operation.
 *
 * Each work-item is independent and handles one element of the output buffer. It performs a
 * fast serial summation over the number of exits in private registers, then decodes the
 * physical SIMD-aware layout of the hidden buffer to apply the ReLU derivative.
 */
__kernel void aggregate_and_backprop_activation(
    __global const SCALAR_TYPE *__restrict hidden_buf,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict grad_hidden_contributions_buf,
    __global SCALAR_TYPE *__restrict grad_pre_activation_buf,
    int padded_batch_size,
    int hidden_dim,
    int num_exits) {
    const uint b_idx = get_global_id(0);
    const uint j_idx = get_global_id(1);

    if (b_idx >= padded_batch_size || j_idx >= hidden_dim)
        return;

    if (hidden_mask[b_idx] < 0.5f) {
        grad_pre_activation_buf[b_idx * hidden_dim + j_idx] = SCALAR_ZERO;
        return;
    }

    SCALAR_TYPE aggregated_grad_h = SCALAR_ZERO;
    for (int e = 0; e < num_exits; ++e) {
        aggregated_grad_h += grad_hidden_contributions_buf[e * padded_batch_size * hidden_dim + b_idx * hidden_dim + j_idx];
    }

    const uint        h_block                  = j_idx / SIMD_WIDTH;
    const uint        h_lane                   = j_idx % SIMD_WIDTH;
    const uint        padded_hidden_dim_blocks = hidden_dim / SIMD_WIDTH;
    const uint        physical_hidden_idx      = b_idx * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;
    const SCALAR_TYPE hidden_val               = hidden_buf[physical_hidden_idx];

    const SCALAR_TYPE d_activation                      = select(SCALAR_ZERO, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);
    grad_pre_activation_buf[b_idx * hidden_dim + j_idx] = aggregated_grad_h * d_activation;
}

/**
 * @brief (Node 12) Implements gradient calculation using a "work-group per gradient" strategy.
 *
 * Each work-group is responsible for computing a single element of the `grad_weights` and
 * `grad_biases` tensors. The threads within the group parallelize the summation over the batch
 * dimension, using a fast local memory reduction to produce the final value.
 */
__kernel void calculate_dense_layer_gradients(
    __local SCALAR_TYPE *local_grad_w,
    __local SCALAR_TYPE *local_grad_b,
    __global const SCALAR_TYPE *__restrict input_buf,
    __global const SCALAR_TYPE *__restrict input_mask,
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf,
    __global SCALAR_TYPE *__restrict grad_weights,
    __global SCALAR_TYPE *__restrict grad_biases,
    int padded_batch_size,
    int input_dim,
    int hidden_dim) {
    const uint i_idx = get_group_id(0);
    const uint j_idx = get_group_id(1);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    SCALAR_TYPE p_grad_w = SCALAR_ZERO;
    SCALAR_TYPE p_grad_b = SCALAR_ZERO;

    for (int b = lid; b < padded_batch_size; b += lsize) {
        if (input_mask[b] < 0.5f)
            continue;
        p_grad_w += grad_pre_activation_buf[b * hidden_dim + j_idx] * input_buf[b * input_dim + i_idx];
        if (i_idx == 0)
            p_grad_b += grad_pre_activation_buf[b * hidden_dim + j_idx];
    }

    local_grad_w[lid] = p_grad_w;
    if (i_idx == 0)
        local_grad_b[lid] = p_grad_b;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_grad_w[lid] += local_grad_w[lid + stride];
            if (i_idx == 0)
                local_grad_b[lid] += local_grad_b[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        grad_weights[i_idx * hidden_dim + j_idx] = local_grad_w[0];
        if (i_idx == 0)
            grad_biases[j_idx] = local_grad_b[0];
    }
}

/**
 * @brief (Node 13) Implements backpropagation to the layer's input via a dot product.
 *
 * This is an embarrassingly parallel kernel where each work-item is independent. Each item
 * calculates one element of the output `grad_input` buffer by performing a dot product
 * between a row of the `grad_pre_activation` matrix and a column of the transposed weight
 * matrix (which is a row of the original `weights` matrix).
 */
__kernel void backprop_input_gradient(
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf,
    __global const SCALAR_TYPE *__restrict weights_buf,
    __global SCALAR_TYPE *__restrict grad_input_buf,
    int padded_batch_size,
    int input_dim,
    int hidden_dim) {
    const uint b_idx = get_global_id(0);
    const uint i_idx = get_global_id(1);

    if (b_idx >= padded_batch_size || i_idx >= input_dim)
        return;

    SCALAR_TYPE sum = SCALAR_ZERO;
    for (int j = 0; j < hidden_dim; ++j) {
        sum += grad_pre_activation_buf[b_idx * hidden_dim + j] * weights_buf[i_idx * hidden_dim + j];
    }
    grad_input_buf[b_idx * input_dim + i_idx] = sum;
}

/**
 * @brief (Node 14) Calculates gradients for the temperature parameters.
 *
 * This implementation uses the "work-group per temperature" strategy. Each work-group is
 * responsible for a single temperature's gradient. The threads within the group parallelize
 * the summation over the batch dimension, using a fast local memory reduction to produce the final sum.
 */
__kernel void calculate_temp_gradients(
    __local SCALAR_TYPE *local_grad_sum,
    __global const SCALAR_TYPE *__restrict unscaled_logits_buf,
    __global const SCALAR_TYPE *__restrict exit_probs_buf,
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf,
    __global const SCALAR_TYPE *__restrict temps_buf,
    __global const int *__restrict targets_buf,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global SCALAR_TYPE *__restrict grad_temps,
    int padded_batch_size,
    int output_classes,
    int num_exits) {
    const uint e_idx = get_group_id(0);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    SCALAR_TYPE       p_grad_sum  = SCALAR_ZERO;
    const SCALAR_TYPE temp        = temps_buf[e_idx];
    const SCALAR_TYPE temp_sq_inv = -1.0f / (temp * temp);

    for (int b = lid; b < padded_batch_size; b += lsize) {
        if (targets_mask[b] < 0.5f)
            continue;
        const int         true_class        = targets_buf[b];
        const SCALAR_TYPE weight_be         = ensemble_weights_buf[b * num_exits + e_idx];
        SCALAR_TYPE       grad_contribution = SCALAR_ZERO;
        for (int c = 0; c < output_classes; ++c) {
            SCALAR_TYPE is_target      = select(SCALAR_ZERO, (SCALAR_TYPE)1.0f, c == true_class);
            SCALAR_TYPE prob           = exit_probs_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c];
            SCALAR_TYPE d_loss_d_logit = weight_be * (prob - is_target);
            SCALAR_TYPE unscaled_logit = unscaled_logits_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c];
            grad_contribution += d_loss_d_logit * (unscaled_logit * temp_sq_inv);
        }
        p_grad_sum += grad_contribution;
    }

    local_grad_sum[lid] = p_grad_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_grad_sum[lid] += local_grad_sum[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        grad_temps[e_idx] = local_grad_sum[0];
    }
}
