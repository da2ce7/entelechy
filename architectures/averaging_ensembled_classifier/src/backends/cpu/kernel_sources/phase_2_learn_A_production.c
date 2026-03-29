/* phase_2_learn_A_production.c — CPU Learn-phase production kernels (Nodes 6, 7, 8).
 *
 * Algorithmic reference: kernels/phase_2_learn_A_production.cl.c
 * CCE: softmax + cross-entropy loss.
 * BCE: sigmoid + binary cross-entropy loss.
 * Module param grads: FLAG dispatch for CCE/BCE.
 */
#include "cpu_kernels.h"

/* ================================================================
 * task_cce_probs_loss (Node 6)
 *
 * Each task processes one tile: (module_local, batch) over the
 * assigned class chunk.  Computes softmax probabilities and
 * accumulates the cross-entropy loss contribution.
 *
 * task_index maps onto the (module_local, batch) pair:
 *   module_local = task_index / total_batch_count
 *   batch_idx    = task_index % total_batch_count
 * ================================================================ */
void task_cce_probs_loss(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    CceChunkArgs* a = (CceChunkArgs*)raw_args;

    /* Decompose task_index into (module_local, batch) */
    const uint module_local_idx = task_index / a->total_batch_count;
    const uint batch_idx        = task_index % a->total_batch_count;

    if (module_local_idx >= a->modules_per_chunk)
        return;

    /* Decode tile indices from flat_tile_index */
    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    /* Loss write address: monolithic (module_global, batch) layout (matches OpenCL scatter-write) */
    const size_t loss_out_idx = (size_t)module_global_idx * a->total_batch_count + batch_idx;

    /* Skip masked samples */
    if (a->sample_mask[batch_idx] < 0.5f) {
        /* Zero loss for this (module, batch) — scatter-write to monolithic buffer */
        a->final_loss[loss_out_idx] = 0.0f;
        /* Zero the prob output for this (module, batch, class_chunk) */
        const size_t prob_tile_base = (size_t)a->flat_tile_index
            * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;
        const size_t prob_base = prob_tile_base
            + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
            + (size_t)batch_idx * a->classes_per_chunk;
        for (uint c = 0; c < a->classes_per_chunk; c++)
            a->partial_probs[prob_base + c] = 0.0f;
        return;
    }

    /* --- Fused, Numerically Stable Softmax (Global Reduction) ---
     * Softmax requires a GLOBAL max and sum_exp across ALL classes, not just
     * this tile's chunk.  Matches OpenCL compute_probs_loss_cce_chunk. */
    const float temp     = a->temps[module_global_idx];
    const float inv_temp = 1.0f / temp;
    const size_t base_logits_idx = (size_t)module_global_idx * a->total_batch_count
        * a->padded_total_output_class_count
        + (size_t)batch_idx * a->padded_total_output_class_count;

    /* Pass 1: find max scaled logit across ALL classes for numerical stability */
    float max_scaled_logit = -3.402823466e+38f; /* -FLT_MAX */
    for (uint c = 0; c < a->total_output_class_count; c++) {
        float scaled = a->logits[base_logits_idx + c] * inv_temp;
        if (scaled > max_scaled_logit) max_scaled_logit = scaled;
    }
    if (max_scaled_logit == -3.402823466e+38f) max_scaled_logit = 0.0f;

    /* Pass 2: compute sum of exponentials across ALL classes */
    float sum_exp = 0.0f;
    for (uint c = 0; c < a->total_output_class_count; c++) {
        sum_exp += expf(a->logits[base_logits_idx + c] * inv_temp - max_scaled_logit);
    }
    const float inv_sum_exp = (sum_exp > NUMERICAL_STABILITY_EPSILON)
        ? (1.0f / sum_exp) : 0.0f;

    /* Pass 3: write this tile's partial probabilities using global softmax */
    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;
    const size_t prob_base = prob_tile_base
        + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
        + (size_t)batch_idx * a->classes_per_chunk;

    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global < a->total_output_class_count) {
            float prob = expf(a->logits[base_logits_idx + c_global] * inv_temp
                             - max_scaled_logit) * inv_sum_exp;
            a->partial_probs[prob_base + c_local] = prob;
        } else {
            a->partial_probs[prob_base + c_local] = 0.0f;
        }
    }

    /* Loss: re-compute true class probability from global softmax, then
     * scatter-write to monolithic (module, batch) buffer (matches OpenCL). */
    const int   true_class_idx = a->targets[batch_idx];
    const float logit_true     = a->logits[base_logits_idx + true_class_idx];
    const float prob_true      = expf(logit_true * inv_temp - max_scaled_logit) * inv_sum_exp;
    a->final_loss[loss_out_idx] = -logf(fmaxf(prob_true, NUMERICAL_STABILITY_EPSILON));
}

/* ================================================================
 * task_bce_probs_loss (Node 7)
 *
 * Each task processes one tile: (module_local, batch) over the
 * assigned class chunk.  Computes sigmoid probabilities and
 * binary cross-entropy loss per class.
 * ================================================================ */
void task_bce_probs_loss(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    BceChunkArgs* a = (BceChunkArgs*)raw_args;

    const uint module_local_idx = task_index / a->total_batch_count;
    const uint batch_idx        = task_index % a->total_batch_count;

    if (module_local_idx >= a->modules_per_chunk)
        return;

    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;
    const size_t prob_base = prob_tile_base
        + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
        + (size_t)batch_idx * a->classes_per_chunk;

    if (a->sample_mask[batch_idx] < 0.5f) {
        for (uint c = 0; c < a->classes_per_chunk; c++)
            a->partial_probs[prob_base + c] = 0.0f;
        if (a->partial_loss != NULL) {
            const size_t loss_idx = (size_t)a->flat_tile_index
                * a->modules_per_chunk * a->total_batch_count
                + (size_t)module_local_idx * a->total_batch_count + batch_idx;
            a->partial_loss[loss_idx] = 0.0f;
        }
        return;
    }

    const float temp = a->temps[module_global_idx];
    const float inv_temp = 1.0f / temp;
    float loss_sum = 0.0f;

    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global >= a->total_output_class_count) {
            a->partial_probs[prob_base + c_local] = 0.0f;
            continue;
        }

        const size_t logit_idx = (size_t)module_global_idx * a->total_batch_count
            * a->padded_total_output_class_count
            + (size_t)batch_idx * a->padded_total_output_class_count
            + c_global;
        float scaled_logit = a->logits[logit_idx] * inv_temp;

        /* Numerically stable sigmoid: handle positive/negative branches */
        float prob;
        if (scaled_logit >= 0.0f) {
            float e = expf(-scaled_logit);
            prob = 1.0f / (1.0f + e);
        } else {
            float e = expf(scaled_logit);
            prob = e / (1.0f + e);
        }
        a->partial_probs[prob_base + c_local] = prob;

        /* Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)] */
        const size_t target_idx = (size_t)batch_idx
            * a->padded_total_output_class_count + c_global;
        float y = a->targets[target_idx];
        float log_p   = logf(fmaxf(prob, NUMERICAL_STABILITY_EPSILON));
        float log_1mp = logf(fmaxf(1.0f - prob, NUMERICAL_STABILITY_EPSILON));
        loss_sum += -(y * log_p + (1.0f - y) * log_1mp);
    }

    if (a->partial_loss != NULL) {
        const size_t loss_idx = (size_t)a->flat_tile_index
            * a->modules_per_chunk * a->total_batch_count
            + (size_t)module_local_idx * a->total_batch_count + batch_idx;
        a->partial_loss[loss_idx] = loss_sum;
    }
}

/* ================================================================
 * task_module_param_grads (Node 8)
 *
 * Each task computes the gradients for one (module_local, h_idx)
 * coordinate pair, performing a batch reduction.
 *
 * task_index encodes: (module_local * hidden_count + h_idx)
 * ================================================================ */
void task_module_param_grads(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    ModuleParamGradsArgs* a = (ModuleParamGradsArgs*)raw_args;

    /* Decompose task_index into (module_local, h_idx) */
    const uint module_local_idx = task_index / a->hidden_count;
    const uint h_idx            = task_index % a->hidden_count;

    if (module_local_idx >= a->modules_per_chunk || h_idx >= a->hidden_count)
        return;

    /* Decode tile structure */
    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;

    /* For each class in the chunk, reduce over batch */
    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global >= a->total_output_class_count) break;

        float grad_w = 0.0f;
        float grad_b = 0.0f;

        for (uint b = 0; b < a->total_batch_count; b++) {
            if (a->sample_mask[b] < 0.5f) continue;

            /* Read pre-computed probability */
            const size_t prob_idx = prob_tile_base
                + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
                + (size_t)b * a->classes_per_chunk + c_local;
            const float prob = a->partial_probs[prob_idx];

            /* Compute d_loss_d_logit based on problem type */
            float d_loss_d_logit;
            if (a->problem_type == PROBLEM_TYPE_CCE) {
                const int* targets_cce = (const int*)a->targets;
                d_loss_d_logit = ((int)c_global == targets_cce[b])
                    ? (prob - 1.0f) : prob;
            } else { /* BCE */
                const float* targets_bce = (const float*)a->targets;
                d_loss_d_logit = prob - targets_bce[
                    (size_t)b * a->padded_total_output_class_count + c_global];
            }

            /* Accumulate weight gradient: dL/dW = dL/dLogit * h_val */
            const float h_val = a->hidden_activations[
                (size_t)b * a->padded_hidden_count + h_idx];
            grad_w += d_loss_d_logit * h_val;

            /* Bias gradient only computed once (h_idx == 0) */
            if (h_idx == 0) {
                grad_b += d_loss_d_logit;
            }
        }

        /* Write weight gradient */
        const size_t w_base = (size_t)a->flat_tile_index
            * a->modules_per_chunk * a->padded_hidden_count
            * a->padded_total_output_class_count;
        const size_t w_idx = w_base
            + (size_t)module_local_idx * a->padded_hidden_count
                * a->padded_total_output_class_count
            + (size_t)h_idx * a->padded_total_output_class_count
            + c_global;
        a->partial_grad_weights_module[w_idx] = grad_w;

        /* Write bias gradient (only from h_idx == 0 tasks) */
        if (h_idx == 0) {
            const size_t b_base = (size_t)a->flat_tile_index
                * a->modules_per_chunk * a->padded_total_output_class_count;
            const size_t b_idx = b_base
                + (size_t)module_local_idx * a->padded_total_output_class_count
                + c_global;
            a->partial_grad_biases_module[b_idx] = grad_b;
        }
    }
}
