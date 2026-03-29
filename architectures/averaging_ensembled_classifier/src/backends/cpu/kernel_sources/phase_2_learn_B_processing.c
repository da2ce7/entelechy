/* phase_2_learn_B_processing.c — CPU Learn-phase processing kernels (Nodes 9, 10, 11).
 *
 * Algorithmic reference: kernels/phase_2_learn_A_production.cl.c (Nodes 9, 10),
 *                        kernels/phase_2_learn_B_processing.cl.c (Node 11)
 * Node 9:  backprop error to hidden
 * Node 10: temperature gradients
 * Node 11: clip partial gradients (virtual-vector L2 norm + scale)
 */
#include "cpu_kernels.h"

/* ================================================================
 * task_backprop_to_hidden (Node 9)
 *
 * Each task computes one (module_local, batch, h_idx) element
 * of the partial grad_hidden_activations AoS buffer.
 *
 * task_index = module_local * (batch * padded_hidden)
 *            + batch * padded_hidden + h_idx
 * ================================================================ */
void task_backprop_to_hidden(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    BackpropToHiddenArgs* a = (BackpropToHiddenArgs*)raw_args;

    /* Decompose flat task_index into (module_local, batch, h_idx) */
    const uint bh = a->total_batch_count * a->padded_hidden_count;
    const uint module_local_idx = task_index / bh;
    const uint remainder        = task_index % bh;
    const uint batch_idx        = remainder / a->padded_hidden_count;
    const uint h_idx            = remainder % a->padded_hidden_count;

    if (module_local_idx >= a->modules_per_chunk ||
        batch_idx >= a->total_batch_count ||
        h_idx >= a->hidden_count)
        return;

    /* Calculate write address in AoS collection buffer */
    const size_t tile_size = (size_t)a->modules_per_chunk
        * a->total_batch_count * a->padded_hidden_count;
    const size_t tile_base = (size_t)a->flat_tile_index * tile_size;
    const size_t local_off = (size_t)module_local_idx * a->total_batch_count
        * a->padded_hidden_count
        + (size_t)batch_idx * a->padded_hidden_count + h_idx;
    const size_t out_idx = tile_base + local_off;

    /* Masked samples write zero */
    if (a->sample_mask[batch_idx] < 0.5f) {
        a->partial_grad_hidden_activations_aos[out_idx] = 0.0f;
        return;
    }

    /* Decode tile structure */
    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;

    /* Dot product over class chunk: sum (d_loss_d_logit * weight) */
    float grad_h_accum = 0.0f;
    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global >= a->total_output_class_count) break;

        /* Read probability */
        const size_t prob_idx = prob_tile_base
            + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
            + (size_t)batch_idx * a->classes_per_chunk + c_local;
        const float prob = a->partial_probs[prob_idx];

        /* Compute d_loss_d_logit */
        float d_loss_d_logit;
        if (a->problem_type == PROBLEM_TYPE_CCE) {
            const int* targets_cce = (const int*)a->targets;
            d_loss_d_logit = ((int)c_global == targets_cce[batch_idx])
                ? (prob - 1.0f) : prob;
        } else {
            const float* targets_bce = (const float*)a->targets;
            d_loss_d_logit = prob - targets_bce[
                (size_t)batch_idx * a->padded_total_output_class_count + c_global];
        }

        /* Read weight and accumulate */
        const size_t w_idx = (size_t)module_global_idx * a->padded_hidden_count
            * a->padded_total_output_class_count
            + (size_t)h_idx * a->padded_total_output_class_count
            + c_global;
        grad_h_accum += d_loss_d_logit * a->weights_module[w_idx];
    }

    a->partial_grad_hidden_activations_aos[out_idx] = grad_h_accum;
}

/* ================================================================
 * task_temp_gradients (Node 10)
 *
 * Each task computes one module_local's partial temperature
 * gradient, reducing over all (batch, class_chunk) pairs.
 *
 * task_index = module_local_idx
 * ================================================================ */
void task_temp_gradients(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    TempGradientsArgs* a = (TempGradientsArgs*)raw_args;

    const uint module_local_idx = task_index;
    if (module_local_idx >= a->modules_per_chunk)
        return;

    /* Decode tile structure */
    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;

    float p_grad_sum = 0.0f;

    /* Reduction over batch and class chunk */
    for (uint b = 0; b < a->total_batch_count; b++) {
        if (a->sample_mask[b] < 0.5f) continue;

        float grad_for_sample = 0.0f;
        for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
            const uint c_global = class_offset + c_local;
            if (c_global >= a->total_output_class_count) break;

            /* Read probability */
            const size_t prob_idx = prob_tile_base
                + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
                + (size_t)b * a->classes_per_chunk + c_local;
            const float prob = a->partial_probs[prob_idx];

            /* Compute d_loss_d_logit */
            float d_loss_d_logit;
            if (a->problem_type == PROBLEM_TYPE_CCE) {
                const int* targets_cce = (const int*)a->targets;
                d_loss_d_logit = ((int)c_global == targets_cce[b])
                    ? (prob - 1.0f) : prob;
            } else {
                const float* targets_bce = (const float*)a->targets;
                d_loss_d_logit = prob - targets_bce[
                    (size_t)b * a->padded_total_output_class_count + c_global];
            }

            /* Read logit for chain rule */
            const size_t logit_idx = (size_t)module_global_idx * a->total_batch_count
                * a->padded_total_output_class_count
                + (size_t)b * a->padded_total_output_class_count
                + c_global;
            grad_for_sample += d_loss_d_logit * a->logits[logit_idx];
        }
        p_grad_sum += grad_for_sample;
    }

    /* Apply final chain rule step: dL/dTemp = sum * (-1/T^2) */
    const float temp = a->temps[module_global_idx];
    const float final_grad = p_grad_sum * (-1.0f / (temp * temp));

    /* Write to unique slot */
    const size_t out_idx = (size_t)a->flat_tile_index * a->modules_per_chunk
        + module_local_idx;
    a->partial_grad_temps[out_idx] = final_grad;
}

/* ================================================================
 * task_clip_partial_grads (Node 11)
 *
 * Partial-Group-Wise clipping. Treats 4 gradient buffers as a
 * "virtual vector", computes L2 norm, conditionally scales.
 *
 * task_index is always 0 (one tile per dispatch).
 * ================================================================ */
void task_clip_partial_grads(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    ClipPartialsArgs* a = (ClipPartialsArgs*)raw_args;

    /* Calculate per-buffer element counts */
    const uint n_weights = a->modules_per_chunk * a->padded_hidden_count
        * a->padded_total_output_class_count;
    const uint n_biases  = a->modules_per_chunk * a->padded_total_output_class_count;
    const uint n_temps   = a->modules_per_chunk;
    const uint n_hidden  = a->modules_per_chunk * a->total_batch_count
        * a->padded_hidden_count;

    /* Base offsets for this tile in the collection buffers */
    const size_t w_base = (size_t)a->flat_tile_index * n_weights;
    const size_t b_base = (size_t)a->flat_tile_index * n_biases;
    const size_t t_base = (size_t)a->flat_tile_index * n_temps;
    const size_t h_base = (size_t)a->flat_tile_index * n_hidden;

    /* Pass 1: compute sum of squares across all 4 buffers */
    float sum_sq = 0.0f;

    for (uint i = 0; i < n_weights; i++) {
        float v = a->partial_grad_weights_module[w_base + i];
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_biases; i++) {
        float v = a->partial_grad_biases_module[b_base + i];
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_temps; i++) {
        float v = a->partial_grad_temps[t_base + i];
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_hidden; i++) {
        float v = a->partial_grad_hidden_activations_aos[h_base + i];
        sum_sq += v * v;
    }

    /* Determine threshold */
    float threshold;
    if (a->use_per_item_norm == 1) {
        threshold = a->clipping_threshold_per_item[a->flat_tile_index];
    } else {
        threshold = a->clipping_threshold_t_pre;
    }

    /* Compute scale factor */
    float norm = sqrtf(sum_sq);
    float scale = 1.0f;
    if (norm > threshold) {
        scale = threshold / (norm + a->epsilon);
    }

    /* Pass 2: scale and write to destination buffers */
    for (uint i = 0; i < n_weights; i++) {
        a->clipped_partial_grad_weights_module[w_base + i] =
            a->partial_grad_weights_module[w_base + i] * scale;
    }
    for (uint i = 0; i < n_biases; i++) {
        a->clipped_partial_grad_biases_module[b_base + i] =
            a->partial_grad_biases_module[b_base + i] * scale;
    }
    for (uint i = 0; i < n_temps; i++) {
        a->clipped_partial_grad_temps[t_base + i] =
            a->partial_grad_temps[t_base + i] * scale;
    }
    for (uint i = 0; i < n_hidden; i++) {
        a->clipped_partial_grad_hidden_activations_aos[h_base + i] =
            a->partial_grad_hidden_activations_aos[h_base + i] * scale;
    }
}
