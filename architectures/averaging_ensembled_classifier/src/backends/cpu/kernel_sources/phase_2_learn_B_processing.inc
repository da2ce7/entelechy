/* phase_2_learn_B_processing.inc — Precision-parameterized processing kernels (Nodes 9, 10, 11).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_A_production.cl.c (Nodes 9, 10),
 *                        kernels/phase_2_learn_B_processing.cl.c (Node 11)
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - partial_probs, sample_mask, partial_grad_*, clipped_*, logits, targets (BCE) → storage
 *   - weights_module, temps → state
 */

/* ================================================================
 * task_backprop_to_hidden (Node 9)
 * ================================================================ */
void PREC_FN(task_backprop_to_hidden)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(BackpropToHiddenArgs)* a = (PREC_TY(BackpropToHiddenArgs)*)raw_args;

    const uint bh = a->total_batch_count * a->padded_hidden_count;
    const uint module_local_idx = task_index / bh;
    const uint remainder        = task_index % bh;
    const uint batch_idx        = remainder / a->padded_hidden_count;
    const uint h_idx            = remainder % a->padded_hidden_count;

    if (module_local_idx >= a->modules_per_chunk ||
        batch_idx >= a->total_batch_count ||
        h_idx >= a->hidden_count)
        return;

    const size_t tile_size = (size_t)a->modules_per_chunk
        * a->total_batch_count * a->padded_hidden_count;
    const size_t tile_base = (size_t)a->flat_tile_index * tile_size;
    const size_t local_off = (size_t)module_local_idx * a->total_batch_count
        * a->padded_hidden_count
        + (size_t)batch_idx * a->padded_hidden_count + h_idx;
    const size_t out_idx = tile_base + local_off;

    if ((COMPUTE_T)scalar_load_storage(a->sample_mask, batch_idx) < (COMPUTE_T)0.5) {
        scalar_store_storage(a->partial_grad_hidden_activations_aos, out_idx, 0.0f);
        return;
    }

    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;

    COMPUTE_T grad_h_accum = (COMPUTE_T)0.0;
    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global >= a->total_output_class_count) break;

        const size_t prob_idx = prob_tile_base
            + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
            + (size_t)batch_idx * a->classes_per_chunk + c_local;
        const COMPUTE_T prob = (COMPUTE_T)scalar_load_storage(a->partial_probs, prob_idx);

        COMPUTE_T d_loss_d_logit;
        if (a->problem_type == PROBLEM_TYPE_CCE) {
            const int* targets_cce = (const int*)a->targets;
            d_loss_d_logit = ((int)c_global == targets_cce[batch_idx])
                ? (prob - (COMPUTE_T)1.0) : prob;
        } else {
            const STORAGE_T* targets_bce = (const STORAGE_T*)a->targets;
            d_loss_d_logit = prob - (COMPUTE_T)scalar_load_storage(targets_bce,
                (size_t)batch_idx * a->padded_total_output_class_count + c_global);
        }

        const size_t w_idx = (size_t)module_global_idx * a->padded_hidden_count
            * a->padded_total_output_class_count
            + (size_t)h_idx * a->padded_total_output_class_count
            + c_global;
        grad_h_accum += d_loss_d_logit * (COMPUTE_T)scalar_load_state(a->weights_module, w_idx);
    }

    scalar_store_storage(a->partial_grad_hidden_activations_aos, out_idx, grad_h_accum);
}

/* ================================================================
 * task_temp_gradients (Node 10)
 * ================================================================ */
void PREC_FN(task_temp_gradients)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(TempGradientsArgs)* a = (PREC_TY(TempGradientsArgs)*)raw_args;

    const uint module_local_idx = task_index;
    if (module_local_idx >= a->modules_per_chunk)
        return;

    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;

    COMPUTE_T p_grad_sum = (COMPUTE_T)0.0;

    for (uint b = 0; b < a->total_batch_count; b++) {
        if ((COMPUTE_T)scalar_load_storage(a->sample_mask, b) < (COMPUTE_T)0.5) continue;

        COMPUTE_T grad_for_sample = (COMPUTE_T)0.0;
        for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
            const uint c_global = class_offset + c_local;
            if (c_global >= a->total_output_class_count) break;

            const size_t prob_idx = prob_tile_base
                + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
                + (size_t)b * a->classes_per_chunk + c_local;
            const COMPUTE_T prob = (COMPUTE_T)scalar_load_storage(a->partial_probs, prob_idx);

            COMPUTE_T d_loss_d_logit;
            if (a->problem_type == PROBLEM_TYPE_CCE) {
                const int* targets_cce = (const int*)a->targets;
                d_loss_d_logit = ((int)c_global == targets_cce[b])
                    ? (prob - (COMPUTE_T)1.0) : prob;
            } else {
                const STORAGE_T* targets_bce = (const STORAGE_T*)a->targets;
                d_loss_d_logit = prob - (COMPUTE_T)scalar_load_storage(targets_bce,
                    (size_t)b * a->padded_total_output_class_count + c_global);
            }

            const size_t logit_idx = (size_t)module_global_idx * a->total_batch_count
                * a->padded_total_output_class_count
                + (size_t)b * a->padded_total_output_class_count
                + c_global;
            grad_for_sample += d_loss_d_logit * (COMPUTE_T)scalar_load_storage(a->logits, logit_idx);
        }
        p_grad_sum += grad_for_sample;
    }

    const COMPUTE_T temp = (COMPUTE_T)scalar_load_state(a->temps, module_global_idx);
    const COMPUTE_T final_grad = p_grad_sum * (-(COMPUTE_T)1.0 / (temp * temp));

    const size_t out_idx = (size_t)a->flat_tile_index * a->modules_per_chunk
        + module_local_idx;
    scalar_store_storage(a->partial_grad_temps, out_idx, final_grad);
}

/* ================================================================
 * task_clip_partial_grads (Node 11)
 * ================================================================ */
void PREC_FN(task_clip_partial_grads)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    PREC_TY(ClipPartialsArgs)* a = (PREC_TY(ClipPartialsArgs)*)raw_args;

    const uint n_weights = a->modules_per_chunk * a->padded_hidden_count
        * a->padded_total_output_class_count;
    const uint n_biases  = a->modules_per_chunk * a->padded_total_output_class_count;
    const uint n_temps   = a->modules_per_chunk;
    const uint n_hidden  = a->modules_per_chunk * a->total_batch_count
        * a->padded_hidden_count;

    const size_t w_base = (size_t)a->flat_tile_index * n_weights;
    const size_t b_base = (size_t)a->flat_tile_index * n_biases;
    const size_t t_base = (size_t)a->flat_tile_index * n_temps;
    const size_t h_base = (size_t)a->flat_tile_index * n_hidden;

    COMPUTE_T sum_sq = (COMPUTE_T)0.0;

    for (uint i = 0; i < n_weights; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_weights_module, w_base + i);
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_biases; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_biases_module, b_base + i);
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_temps; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_temps, t_base + i);
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_hidden; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_hidden_activations_aos, h_base + i);
        sum_sq += v * v;
    }

    COMPUTE_T threshold;
    if (a->use_per_item_norm == 1) {
        threshold = scalar_load_storage(a->clipping_threshold_per_item, a->flat_tile_index);
    } else {
        threshold = (COMPUTE_T)a->clipping_threshold_t_pre;
    }

    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    COMPUTE_T scale = (COMPUTE_T)1.0;
    /* Diagnostic bypass: negative threshold skips all clipping */
    if (threshold >= (COMPUTE_T)0.0 && norm > threshold) {
        scale = threshold / (norm + (COMPUTE_T)a->epsilon);
    }

    for (uint i = 0; i < n_weights; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_weights_module, w_base + i);
        scalar_store_storage(a->clipped_partial_grad_weights_module, w_base + i, v * scale);
    }
    for (uint i = 0; i < n_biases; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_biases_module, b_base + i);
        scalar_store_storage(a->clipped_partial_grad_biases_module, b_base + i, v * scale);
    }
    for (uint i = 0; i < n_temps; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_temps, t_base + i);
        scalar_store_storage(a->clipped_partial_grad_temps, t_base + i, v * scale);
    }
    for (uint i = 0; i < n_hidden; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_hidden_activations_aos, h_base + i);
        scalar_store_storage(a->clipped_partial_grad_hidden_activations_aos, h_base + i, v * scale);
    }
}
