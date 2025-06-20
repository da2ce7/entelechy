// templates.cl.h
//
// This file contains the macro-based templates for generating kernel variants.
// It is designed to be included by the main C source files to instantiate
// the concrete kernel implementations for CCE and BCE.
//
// CONVENTION: This file does NOT have #ifndef include guards, as it is a
// source of macro definitions, not a standard declarative header.

#define COMPUTE_ALL_EXITS_TEMPLATE(KERNEL_NAME, TARGET_PTR_T, TARGET_T, IS_CCE_VERSION)                                                                                                                \
    /**                                                                                                                                                                                                \
     * @brief (Node 5) Generic "map" kernel for computing exit outputs. Instantiated from template.                                                                                                    \
     */                                                                                                                                                                                                \
    __kernel void KERNEL_NAME(                                                                                                                                                                         \
        __global const SCALAR_TYPE *__restrict hidden_buf, __global const SCALAR_TYPE *__restrict hidden_mask, __global const SCALAR_TYPE *__restrict exit_weights_buf,                                \
        __global const SCALAR_TYPE *__restrict exit_biases_buf, __global const SCALAR_TYPE *__restrict temps_buf, TARGET_PTR_T targets_buf, __global const SCALAR_TYPE *__restrict targets_mask,       \
        __global SCALAR_TYPE *__restrict unscaled_logits_buf, __global SCALAR_TYPE *__restrict exit_probs_buf, __global SCALAR_TYPE *__restrict per_exit_losses_buf, int padded_batch_size,            \
        int hidden_dim, int output_classes, int num_exits) {                                                                                                                                           \
        const uint exit_idx  = get_global_id(0);                                                                                                                                                       \
        const uint batch_idx = get_global_id(1);                                                                                                                                                       \
                                                                                                                                                                                                       \
        if (exit_idx >= num_exits || batch_idx >= padded_batch_size || hidden_mask[batch_idx] < 0.5f || targets_mask[batch_idx] < 0.5f) {                                                              \
            /* To ensure determinism, explicitly zero out outputs for invalid items. */                                                                                                                \
            if (exit_idx < num_exits && batch_idx < padded_batch_size) {                                                                                                                               \
                const uint out_base_idx = exit_idx * padded_batch_size * output_classes + batch_idx * output_classes;                                                                                  \
                for (int c = 0; c < output_classes; ++c) {                                                                                                                                             \
                    unscaled_logits_buf[out_base_idx + c] = SCALAR_ZERO;                                                                                                                               \
                    exit_probs_buf[out_base_idx + c]      = SCALAR_ZERO;                                                                                                                               \
                }                                                                                                                                                                                      \
                per_exit_losses_buf[exit_idx * padded_batch_size + batch_idx] = SCALAR_ZERO;                                                                                                           \
            }                                                                                                                                                                                          \
            return;                                                                                                                                                                                    \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        /* --- 1. Compute unscaled logits (W*h + b) and store them --- */                                                                                                                              \
        const uint  out_base_idx = exit_idx * padded_batch_size * output_classes + batch_idx * output_classes;                                                                                         \
        SCALAR_TYPE p_logits[C_TILE_SIZE]; /* Assumes output_classes <= C_TILE_SIZE */                                                                                                                 \
        for (int c = 0; c < output_classes; ++c) {                                                                                                                                                     \
            SCALAR_TYPE logit = exit_biases_buf[exit_idx * output_classes + c];                                                                                                                        \
            for (int h = 0; h < hidden_dim; ++h) {                                                                                                                                                     \
                const uint h_block                  = h / SIMD_WIDTH;                                                                                                                                  \
                const uint h_lane                   = h % SIMD_WIDTH;                                                                                                                                  \
                const uint padded_hidden_dim_blocks = (hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;                                                                                                      \
                const uint physical_hidden_idx      = batch_idx * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;                                                               \
                logit += hidden_buf[physical_hidden_idx] * exit_weights_buf[exit_idx * hidden_dim * output_classes + h * output_classes + c];                                                          \
            }                                                                                                                                                                                          \
            p_logits[c]                           = logit;                                                                                                                                             \
            unscaled_logits_buf[out_base_idx + c] = logit;                                                                                                                                             \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        /* --- 2. Apply activation and compute loss based on problem type --- */                                                                                                                       \
        SCALAR_TYPE       total_loss = SCALAR_ZERO;                                                                                                                                                    \
        const SCALAR_TYPE temp_inv   = 1.0f / temps_buf[exit_idx];                                                                                                                                     \
                                                                                                                                                                                                       \
        if (IS_CCE_VERSION) { /* --- CCE Path (Softmax) --- */                                                                                                                                         \
            SCALAR_TYPE max_logit = -INFINITY;                                                                                                                                                         \
            for (int c = 0; c < output_classes; ++c)                                                                                                                                                   \
                max_logit = fmax(max_logit, p_logits[c]);                                                                                                                                              \
                                                                                                                                                                                                       \
            SCALAR_TYPE sum_exp = SCALAR_ZERO;                                                                                                                                                         \
            for (int c = 0; c < output_classes; ++c) {                                                                                                                                                 \
                sum_exp += exp((p_logits[c] - max_logit) * temp_inv);                                                                                                                                  \
            }                                                                                                                                                                                          \
            sum_exp = fmax(sum_exp, (SCALAR_TYPE)1e-7f);                                                                                                                                               \
                                                                                                                                                                                                       \
            const int true_class = targets_buf[batch_idx];                                                                                                                                             \
            for (int c = 0; c < output_classes; ++c) {                                                                                                                                                 \
                SCALAR_TYPE scaled_logit         = (p_logits[c] - max_logit) * temp_inv;                                                                                                               \
                SCALAR_TYPE prob                 = exp(scaled_logit) / sum_exp;                                                                                                                        \
                exit_probs_buf[out_base_idx + c] = prob;                                                                                                                                               \
                if (c == true_class) {                                                                                                                                                                 \
                    total_loss = -log(fmax(prob, (SCALAR_TYPE)1e-7f));                                                                                                                                 \
                }                                                                                                                                                                                      \
            }                                                                                                                                                                                          \
        } else { /* --- BCE Path (Sigmoid) --- */                                                                                                                                                      \
            const uint target_base_idx = batch_idx * output_classes;                                                                                                                                   \
            for (int c = 0; c < output_classes; ++c) {                                                                                                                                                 \
                SCALAR_TYPE prob                 = 1.0f / (1.0f + exp(-p_logits[c] * temp_inv));                                                                                                       \
                exit_probs_buf[out_base_idx + c] = prob;                                                                                                                                               \
                                                                                                                                                                                                       \
                const TARGET_T target_val = targets_buf[target_base_idx + c];                                                                                                                          \
                total_loss -= (target_val * log(fmax(prob, (SCALAR_TYPE)1e-9f)) + (1.0f - target_val) * log(fmax(1.0f - prob, (SCALAR_TYPE)1e-9f)));                                                   \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
        per_exit_losses_buf[exit_idx * padded_batch_size + batch_idx] = total_loss;                                                                                                                    \
    }

#define CALCULATE_EXIT_GRADIENTS_TEMPLATE(KERNEL_NAME, TARGET_PTR_T, TARGET_T, IS_CCE_VERSION)                                                                                                         \
    /**                                                                                                                                                                                                \
     * @brief (Node 10) Generic kernel for exit parameter gradients. Instantiated from template.                                                                                                       \
     */                                                                                                                                                                                                \
    __kernel void KERNEL_NAME(                                                                                                                                                                         \
        __local SCALAR_TYPE *local_grad_w, __local SCALAR_TYPE *local_grad_b, __global const SCALAR_TYPE *__restrict hidden_buf, __global const SCALAR_TYPE *__restrict exit_probs_buf,                \
        __global const SCALAR_TYPE *__restrict ensemble_weights_buf, TARGET_PTR_T targets_buf, __global const SCALAR_TYPE *__restrict targets_mask,                                                    \
        __global const SCALAR_TYPE *__restrict exit_weights_buf, __global SCALAR_TYPE *__restrict grad_exit_weights, __global SCALAR_TYPE *__restrict grad_exit_biases,                                \
        __global SCALAR_TYPE *__restrict grad_hidden_contributions_buf, int padded_batch_size, int hidden_dim, int output_classes, int num_exits) {                                                    \
        const uint e_idx = get_group_id(0);                                                                                                                                                            \
        const uint h_idx = get_group_id(1);                                                                                                                                                            \
        const uint lid   = get_local_id(0);                                                                                                                                                            \
        const uint lsize = get_local_size(0);                                                                                                                                                          \
                                                                                                                                                                                                       \
        /* Tiled pass for parameter gradients */                                                                                                                                                       \
        for (int c_base = 0; c_base < output_classes; c_base += C_TILE_SIZE) {                                                                                                                         \
            SCALAR_TYPE p_grad_w[C_TILE_SIZE] = {SCALAR_ZERO};                                                                                                                                         \
            SCALAR_TYPE p_grad_b[C_TILE_SIZE] = {SCALAR_ZERO};                                                                                                                                         \
                                                                                                                                                                                                       \
            for (int b = lid; b < padded_batch_size; b += lsize) {                                                                                                                                     \
                if (targets_mask[b] < 0.5f)                                                                                                                                                            \
                    continue;                                                                                                                                                                          \
                                                                                                                                                                                                       \
                const uint        h_block                  = h_idx / SIMD_WIDTH;                                                                                                                       \
                const uint        h_lane                   = h_idx % SIMD_WIDTH;                                                                                                                       \
                const uint        padded_hidden_dim_blocks = (hidden_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;                                                                                               \
                const uint        physical_hidden_idx      = b * padded_hidden_dim_blocks * SIMD_WIDTH + h_block * SIMD_WIDTH + h_lane;                                                                \
                const SCALAR_TYPE h_val                    = hidden_buf[physical_hidden_idx];                                                                                                          \
                const SCALAR_TYPE weight_be                = ensemble_weights_buf[b * num_exits + e_idx];                                                                                              \
                                                                                                                                                                                                       \
                for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {                                                                                                                              \
                    const int c_global = c_base + c_local;                                                                                                                                             \
                    if (c_global >= output_classes)                                                                                                                                                    \
                        continue;                                                                                                                                                                      \
                                                                                                                                                                                                       \
                    SCALAR_TYPE prob = exit_probs_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c_global];                                                                     \
                    SCALAR_TYPE d_loss_d_logit_base;                                                                                                                                                   \
                    if (IS_CCE_VERSION) { /* CCE Gradient */                                                                                                                                           \
                        const int true_class = targets_buf[b];                                                                                                                                         \
                        d_loss_d_logit_base  = select(prob, prob - 1.0f, c_global == true_class);                                                                                                      \
                    } else { /* BCE Gradient */                                                                                                                                                        \
                        const TARGET_T target_val = targets_buf[b * output_classes + c_global];                                                                                                        \
                        d_loss_d_logit_base       = prob - target_val;                                                                                                                                 \
                    }                                                                                                                                                                                  \
                    SCALAR_TYPE d_loss_d_logit = weight_be * d_loss_d_logit_base;                                                                                                                      \
                    p_grad_w[c_local] += d_loss_d_logit * h_val;                                                                                                                                       \
                    if (h_idx == 0)                                                                                                                                                                    \
                        p_grad_b[c_local] += d_loss_d_logit;                                                                                                                                           \
                }                                                                                                                                                                                      \
            }                                                                                                                                                                                          \
            /* Local memory reduction for parameter gradients */                                                                                                                                       \
            for (int c_local = 0; c_local < C_TILE_SIZE; ++c_local) {                                                                                                                                  \
                const int c_global = c_base + c_local;                                                                                                                                                 \
                if (c_global >= output_classes)                                                                                                                                                        \
                    continue;                                                                                                                                                                          \
                local_grad_w[lid] = p_grad_w[c_local];                                                                                                                                                 \
                if (h_idx == 0)                                                                                                                                                                        \
                    local_grad_b[lid] = p_grad_b[c_local];                                                                                                                                             \
                barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                          \
                for (uint stride = lsize / 2; stride > 0; stride >>= 1) {                                                                                                                              \
                    if (lid < stride) {                                                                                                                                                                \
                        local_grad_w[lid] += local_grad_w[lid + stride];                                                                                                                               \
                        if (h_idx == 0)                                                                                                                                                                \
                            local_grad_b[lid] += local_grad_b[lid + stride];                                                                                                                           \
                    }                                                                                                                                                                                  \
                    barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                      \
                }                                                                                                                                                                                      \
                if (lid == 0) {                                                                                                                                                                        \
                    grad_exit_weights[e_idx * hidden_dim * output_classes + h_idx * output_classes + c_global] = local_grad_w[0];                                                                      \
                    if (h_idx == 0)                                                                                                                                                                    \
                        grad_exit_biases[e_idx * output_classes + c_global] = local_grad_b[0];                                                                                                         \
                }                                                                                                                                                                                      \
            }                                                                                                                                                                                          \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        /* Separate pass for hidden layer contributions */                                                                                                                                             \
        for (int b = lid; b < padded_batch_size; b += lsize) {                                                                                                                                         \
            const uint grad_h_out_idx = e_idx * padded_batch_size * hidden_dim + b * hidden_dim + h_idx;                                                                                               \
            if (targets_mask[b] < 0.5f) {                                                                                                                                                              \
                grad_hidden_contributions_buf[grad_h_out_idx] = SCALAR_ZERO;                                                                                                                           \
                continue;                                                                                                                                                                              \
            }                                                                                                                                                                                          \
            SCALAR_TYPE       grad_h_contribution = SCALAR_ZERO;                                                                                                                                       \
            const SCALAR_TYPE weight_be           = ensemble_weights_buf[b * num_exits + e_idx];                                                                                                       \
            for (int c = 0; c < output_classes; ++c) {                                                                                                                                                 \
                SCALAR_TYPE prob = exit_probs_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c];                                                                                \
                SCALAR_TYPE d_loss_d_logit_base;                                                                                                                                                       \
                if (IS_CCE_VERSION) { /* CCE Gradient */                                                                                                                                               \
                    const int true_class = targets_buf[b];                                                                                                                                             \
                    d_loss_d_logit_base  = select(prob, prob - 1.0f, c == true_class);                                                                                                                 \
                } else { /* BCE Gradient */                                                                                                                                                            \
                    const TARGET_T target_val = targets_buf[b * output_classes + c];                                                                                                                   \
                    d_loss_d_logit_base       = prob - target_val;                                                                                                                                     \
                }                                                                                                                                                                                      \
                grad_h_contribution += (weight_be * d_loss_d_logit_base) * exit_weights_buf[e_idx * hidden_dim * output_classes + h_idx * output_classes + c];                                         \
            }                                                                                                                                                                                          \
            grad_hidden_contributions_buf[grad_h_out_idx] = grad_h_contribution;                                                                                                                       \
        }                                                                                                                                                                                              \
    }

#define CALCULATE_TEMP_GRADIENTS_TEMPLATE(KERNEL_NAME, TARGET_PTR_T, TARGET_T, IS_CCE_VERSION)                                                                                                         \
    /**                                                                                                                                                                                                \
     * @brief (Node 14) Generic kernel for temperature gradients. Instantiated from template.                                                                                                          \
     */                                                                                                                                                                                                \
    __kernel void KERNEL_NAME(                                                                                                                                                                         \
        __local SCALAR_TYPE *local_grad_sum, __global const SCALAR_TYPE *__restrict unscaled_logits_buf, __global const SCALAR_TYPE *__restrict exit_probs_buf,                                        \
        __global const SCALAR_TYPE *__restrict ensemble_weights_buf, __global const SCALAR_TYPE *__restrict temps_buf, TARGET_PTR_T targets_buf, __global const SCALAR_TYPE *__restrict targets_mask,  \
        __global SCALAR_TYPE *__restrict grad_temps, int padded_batch_size, int output_classes, int num_exits) {                                                                                       \
        const uint e_idx = get_group_id(0);                                                                                                                                                            \
        const uint lid   = get_local_id(0);                                                                                                                                                            \
        const uint lsize = get_local_size(0);                                                                                                                                                          \
                                                                                                                                                                                                       \
        SCALAR_TYPE       p_grad_sum  = SCALAR_ZERO;                                                                                                                                                   \
        const SCALAR_TYPE temp        = temps_buf[e_idx];                                                                                                                                              \
        const SCALAR_TYPE temp_sq_inv = -1.0f / (temp * temp);                                                                                                                                         \
                                                                                                                                                                                                       \
        for (int b = lid; b < padded_batch_size; b += lsize) {                                                                                                                                         \
            if (targets_mask[b] < 0.5f)                                                                                                                                                                \
                continue;                                                                                                                                                                              \
                                                                                                                                                                                                       \
            SCALAR_TYPE       grad_contribution_for_sample = SCALAR_ZERO;                                                                                                                              \
            const SCALAR_TYPE weight_be                    = ensemble_weights_buf[b * num_exits + e_idx];                                                                                              \
                                                                                                                                                                                                       \
            for (int c = 0; c < output_classes; ++c) {                                                                                                                                                 \
                SCALAR_TYPE prob = exit_probs_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c];                                                                                \
                SCALAR_TYPE d_loss_d_logit_base;                                                                                                                                                       \
                if (IS_CCE_VERSION) { /* CCE Gradient */                                                                                                                                               \
                    const int true_class = targets_buf[b];                                                                                                                                             \
                    d_loss_d_logit_base  = select(prob, prob - 1.0f, c == true_class);                                                                                                                 \
                } else { /* BCE Gradient */                                                                                                                                                            \
                    const TARGET_T target_val = targets_buf[b * output_classes + c];                                                                                                                   \
                    d_loss_d_logit_base       = prob - target_val;                                                                                                                                     \
                }                                                                                                                                                                                      \
                SCALAR_TYPE unscaled_logit = unscaled_logits_buf[e_idx * padded_batch_size * output_classes + b * output_classes + c];                                                                 \
                grad_contribution_for_sample += d_loss_d_logit_base * unscaled_logit;                                                                                                                  \
            }                                                                                                                                                                                          \
            p_grad_sum += weight_be * grad_contribution_for_sample;                                                                                                                                    \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        p_grad_sum *= temp_sq_inv; /* dL/dT = dL/d(logits) * d(logits)/dT */                                                                                                                           \
        local_grad_sum[lid] = p_grad_sum;                                                                                                                                                              \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                                  \
                                                                                                                                                                                                       \
        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {                                                                                                                                      \
            if (lid < stride) {                                                                                                                                                                        \
                local_grad_sum[lid] += local_grad_sum[lid + stride];                                                                                                                                   \
            }                                                                                                                                                                                          \
            barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                                                              \
        }                                                                                                                                                                                              \
                                                                                                                                                                                                       \
        if (lid == 0) {                                                                                                                                                                                \
            grad_temps[e_idx] = local_grad_sum[0];                                                                                                                                                     \
        }                                                                                                                                                                                              \
    }
