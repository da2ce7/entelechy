/* phase_2_learn_A_production.inc — Precision-parameterized production kernels (Nodes 6, 7, 8).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_A_production.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - logits, sample_mask, partial_probs, partial_loss, hidden_activations,
 *     partial_grad_*, targets (BCE float) → storage
 *   - temps → state
 *   - final_loss (CCE) → compute (COMPUTE_T*)
 */

/* ================================================================
 * task_cce_probs_loss (Node 6)
 * ================================================================ */
void PREC_FN(task_cce_probs_loss)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(CceChunkArgs)* a = (PREC_TY(CceChunkArgs)*)raw_args;

    const uint module_local_idx = task_index / a->total_batch_count;
    const uint batch_idx        = task_index % a->total_batch_count;

    if (module_local_idx >= a->modules_per_chunk)
        return;

    const uint module_chunk_idx  = a->flat_tile_index / a->num_class_chunks;
    const uint module_global_idx = module_chunk_idx * a->modules_per_chunk + module_local_idx;
    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t loss_out_idx = (size_t)module_global_idx * a->total_batch_count + batch_idx;

    if ((COMPUTE_T)scalar_load_storage(a->sample_mask, batch_idx) < (COMPUTE_T)0.5) {
        a->final_loss[loss_out_idx] = (COMPUTE_T)0.0;  /* compute-role: direct write */
        const size_t prob_tile_base = (size_t)a->flat_tile_index
            * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;
        const size_t prob_base = prob_tile_base
            + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
            + (size_t)batch_idx * a->classes_per_chunk;
        for (uint c = 0; c < a->classes_per_chunk; c++)
            scalar_store_storage(a->partial_probs, prob_base + c, 0.0f);
        return;
    }

    const COMPUTE_T temp     = scalar_load_state(a->temps, module_global_idx);
    const COMPUTE_T inv_temp = (COMPUTE_T)1.0 / temp;
    const size_t base_logits_idx = (size_t)module_global_idx * a->total_batch_count
        * a->padded_total_output_class_count
        + (size_t)batch_idx * a->padded_total_output_class_count;

    COMPUTE_T max_scaled_logit = -(COMPUTE_T)3.402823466e+38;
    for (uint c = 0; c < a->total_output_class_count; c++) {
        COMPUTE_T scaled = (COMPUTE_T)scalar_load_storage(a->logits, base_logits_idx + c) * inv_temp;
        if (scaled > max_scaled_logit) max_scaled_logit = scaled;
    }
    if (max_scaled_logit == -(COMPUTE_T)3.402823466e+38) max_scaled_logit = (COMPUTE_T)0.0;

    COMPUTE_T sum_exp = (COMPUTE_T)0.0;
    for (uint c = 0; c < a->total_output_class_count; c++) {
        sum_exp += COMPUTE_EXP((COMPUTE_T)scalar_load_storage(a->logits, base_logits_idx + c) * inv_temp - max_scaled_logit);
    }
    const COMPUTE_T inv_sum_exp = (sum_exp > (COMPUTE_T)NUMERICAL_STABILITY_EPSILON)
        ? ((COMPUTE_T)1.0 / sum_exp) : (COMPUTE_T)0.0;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;
    const size_t prob_base = prob_tile_base
        + (size_t)module_local_idx * a->total_batch_count * a->classes_per_chunk
        + (size_t)batch_idx * a->classes_per_chunk;

    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global < a->total_output_class_count) {
            COMPUTE_T prob = COMPUTE_EXP((COMPUTE_T)scalar_load_storage(a->logits, base_logits_idx + c_global) * inv_temp
                             - max_scaled_logit) * inv_sum_exp;
            scalar_store_storage(a->partial_probs, prob_base + c_local, prob);
        } else {
            scalar_store_storage(a->partial_probs, prob_base + c_local, 0.0f);
        }
    }

    const int   true_class_idx = a->targets[batch_idx];
    const COMPUTE_T logit_true     = scalar_load_storage(a->logits, base_logits_idx + (uint)true_class_idx);
    const COMPUTE_T prob_true      = COMPUTE_EXP(logit_true * inv_temp - max_scaled_logit) * inv_sum_exp;

    /* Only the tile containing the target class writes the loss (scatter-write predicate).
     * This prevents data races when multiple class chunks process the same (module, batch) pair. */
    const bool target_in_this_chunk = ((uint)true_class_idx >= class_offset &&
                                       (uint)true_class_idx < class_offset + a->classes_per_chunk);
    if (target_in_this_chunk) {
        a->final_loss[loss_out_idx] = -COMPUTE_LOG(COMPUTE_FMAX(prob_true, (COMPUTE_T)NUMERICAL_STABILITY_EPSILON));  /* compute-role */
    }
}

/* ================================================================
 * task_bce_probs_loss (Node 7)
 * ================================================================ */
void PREC_FN(task_bce_probs_loss)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(BceChunkArgs)* a = (PREC_TY(BceChunkArgs)*)raw_args;

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

    if ((COMPUTE_T)scalar_load_storage(a->sample_mask, batch_idx) < (COMPUTE_T)0.5) {
        for (uint c = 0; c < a->classes_per_chunk; c++)
            scalar_store_storage(a->partial_probs, prob_base + c, 0.0f);
        if (a->partial_loss != NULL) {
            const size_t loss_idx = (size_t)a->flat_tile_index
                * a->modules_per_chunk * a->total_batch_count
                + (size_t)module_local_idx * a->total_batch_count + batch_idx;
            scalar_store_storage(a->partial_loss, loss_idx, 0.0f);
        }
        return;
    }

    const COMPUTE_T temp = (COMPUTE_T)scalar_load_state(a->temps, module_global_idx);
    const COMPUTE_T inv_temp = (COMPUTE_T)1.0 / temp;
    COMPUTE_T loss_sum = (COMPUTE_T)0.0;

    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global >= a->total_output_class_count) {
            scalar_store_storage(a->partial_probs, prob_base + c_local, 0.0f);
            continue;
        }

        const size_t logit_idx = (size_t)module_global_idx * a->total_batch_count
            * a->padded_total_output_class_count
            + (size_t)batch_idx * a->padded_total_output_class_count
            + c_global;
        COMPUTE_T scaled_logit = (COMPUTE_T)scalar_load_storage(a->logits, logit_idx) * inv_temp;

        COMPUTE_T prob;
        if (scaled_logit >= (COMPUTE_T)0.0) {
            COMPUTE_T e = COMPUTE_EXP(-scaled_logit);
            prob = (COMPUTE_T)1.0 / ((COMPUTE_T)1.0 + e);
        } else {
            COMPUTE_T e = COMPUTE_EXP(scaled_logit);
            prob = e / ((COMPUTE_T)1.0 + e);
        }
        scalar_store_storage(a->partial_probs, prob_base + c_local, prob);

        const size_t target_idx = (size_t)batch_idx
            * a->padded_total_output_class_count + c_global;
        COMPUTE_T y = scalar_load_storage(a->targets, target_idx);
        COMPUTE_T log_p   = COMPUTE_LOG(COMPUTE_FMAX(prob, (COMPUTE_T)NUMERICAL_STABILITY_EPSILON));
        COMPUTE_T log_1mp = COMPUTE_LOG(COMPUTE_FMAX((COMPUTE_T)1.0 - prob, (COMPUTE_T)NUMERICAL_STABILITY_EPSILON));
        loss_sum += -(y * log_p + ((COMPUTE_T)1.0 - y) * log_1mp);
    }

    if (a->partial_loss != NULL) {
        const size_t loss_idx = (size_t)a->flat_tile_index
            * a->modules_per_chunk * a->total_batch_count
            + (size_t)module_local_idx * a->total_batch_count + batch_idx;
        scalar_store_storage(a->partial_loss, loss_idx, loss_sum);
    }
}

/* ================================================================
 * task_module_param_grads (Node 8)
 * ================================================================ */
void PREC_FN(task_module_param_grads)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ModuleParamGradsArgs)* a = (PREC_TY(ModuleParamGradsArgs)*)raw_args;

    const uint module_local_idx = task_index / a->hidden_count;
    const uint h_idx            = task_index % a->hidden_count;

    if (module_local_idx >= a->modules_per_chunk || h_idx >= a->hidden_count)
        return;

    const uint class_chunk_idx   = a->flat_tile_index % a->num_class_chunks;
    const uint class_offset      = class_chunk_idx * a->classes_per_chunk;

    const size_t prob_tile_base = (size_t)a->flat_tile_index
        * a->modules_per_chunk * a->total_batch_count * a->classes_per_chunk;

    for (uint c_local = 0; c_local < a->classes_per_chunk; c_local++) {
        const uint c_global = class_offset + c_local;
        if (c_global >= a->total_output_class_count) break;

        COMPUTE_T grad_w = (COMPUTE_T)0.0;
        COMPUTE_T grad_b = (COMPUTE_T)0.0;

        for (uint b = 0; b < a->total_batch_count; b++) {
            if ((COMPUTE_T)scalar_load_storage(a->sample_mask, b) < (COMPUTE_T)0.5) continue;

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

            const COMPUTE_T h_val = (COMPUTE_T)scalar_load_storage(a->hidden_activations,
                (size_t)b * a->padded_hidden_count + h_idx);
            grad_w += d_loss_d_logit * h_val;

            if (h_idx == 0) {
                grad_b += d_loss_d_logit;
            }
        }

        const size_t w_base = (size_t)a->flat_tile_index
            * a->modules_per_chunk * a->padded_hidden_count
            * a->padded_total_output_class_count;
        const size_t w_idx = w_base
            + (size_t)module_local_idx * a->padded_hidden_count
                * a->padded_total_output_class_count
            + (size_t)h_idx * a->padded_total_output_class_count
            + c_global;
        scalar_store_storage(a->partial_grad_weights_module, w_idx, grad_w);

        if (h_idx == 0) {
            const size_t b_base = (size_t)a->flat_tile_index
                * a->modules_per_chunk * a->padded_total_output_class_count;
            const size_t b_idx = b_base
                + (size_t)module_local_idx * a->padded_total_output_class_count
                + c_global;
            scalar_store_storage(a->partial_grad_biases_module, b_idx, grad_b);
        }
    }
}
