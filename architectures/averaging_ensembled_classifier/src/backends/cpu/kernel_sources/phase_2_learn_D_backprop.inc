/* phase_2_learn_D_backprop.inc — Precision-parameterized backprop kernels (Nodes 17, 18, 19).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_D_backprop.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - sample_mask, hidden_activations, input, partial_grad_*, clipped_partial_grad_* → storage
 *   - summed_grad_hidden_activations → compute (COMPUTE_T*)
 */

/* ================================================================
 * task_backprop_shared_weights (Node 17)
 * ================================================================ */
void PREC_FN(task_backprop_shared_weights)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(BackpropSharedWeightsArgs)* a = (PREC_TY(BackpropSharedWeightsArgs)*)raw_args;

    const uint i_idx = task_index / a->padded_hidden_count;
    const uint j_idx = task_index % a->padded_hidden_count;

    if (i_idx >= a->padded_input_count || j_idx >= a->padded_hidden_count)
        return;

    COMPUTE_T grad_sw = (COMPUTE_T)0.0;
    for (uint b_local = 0; b_local < a->batch_chunk_count; b_local++) {
        const uint b_global = a->batch_chunk_offset + b_local;

        if (scalar_load_storage(a->sample_mask, b_global) < 0.5f)
            continue;

        const size_t h_off = (size_t)b_global * a->padded_hidden_count + j_idx;
        const COMPUTE_T grad_h    = a->summed_grad_hidden_activations[h_off];  /* COMPUTE_T* compute */
        const COMPUTE_T hidden_val = scalar_load_storage(a->hidden_activations, h_off);

        const COMPUTE_T d_act = (hidden_val > (COMPUTE_T)0.0) ? (COMPUTE_T)1.0 : (COMPUTE_T)0.0;
        const COMPUTE_T dL_dZ = grad_h * d_act;

        const COMPUTE_T input_val = scalar_load_storage(a->input,
            (size_t)b_global * a->padded_input_count + i_idx);
        grad_sw += dL_dZ * input_val;
    }

    const size_t chunk_base = (size_t)a->batch_chunk_index
        * a->padded_input_count * a->padded_hidden_count;
    const size_t out_idx = chunk_base
        + (size_t)j_idx * a->padded_input_count + i_idx;
    scalar_store_storage(a->partial_grad_weights_shared, out_idx, grad_sw);
}

/* ================================================================
 * task_backprop_shared_biases (Node 18)
 * ================================================================ */
void PREC_FN(task_backprop_shared_biases)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(BackpropSharedBiasesArgs)* a = (PREC_TY(BackpropSharedBiasesArgs)*)raw_args;

    const uint j_idx = task_index;
    if (j_idx >= a->padded_hidden_count)
        return;

    COMPUTE_T grad_sb = (COMPUTE_T)0.0;
    for (uint b_local = 0; b_local < a->batch_chunk_count; b_local++) {
        const uint b_global = a->batch_chunk_offset + b_local;

        if (scalar_load_storage(a->sample_mask, b_global) < 0.5f)
            continue;

        const size_t h_off = (size_t)b_global * a->padded_hidden_count + j_idx;
        const COMPUTE_T grad_h    = a->summed_grad_hidden_activations[h_off];  /* COMPUTE_T* compute */
        const COMPUTE_T hidden_val = scalar_load_storage(a->hidden_activations, h_off);

        const COMPUTE_T d_act = (hidden_val > (COMPUTE_T)0.0) ? (COMPUTE_T)1.0 : (COMPUTE_T)0.0;
        grad_sb += grad_h * d_act;
    }

    const size_t chunk_base = (size_t)a->batch_chunk_index * a->padded_hidden_count;
    scalar_store_storage(a->partial_grad_biases_shared, chunk_base + j_idx, grad_sb);
}

/* ================================================================
 * task_clip_shared_grads (Node 19)
 * ================================================================ */
void PREC_FN(task_clip_shared_grads)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    PREC_TY(ClipSharedGradsArgs)* a = (PREC_TY(ClipSharedGradsArgs)*)raw_args;

    const uint n_w = a->weights_parameter_count;
    const uint n_b = a->biases_parameter_count;

    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint i = 0; i < n_w; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_weights_shared, i);
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_b; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_biases_shared, i);
        sum_sq += v * v;
    }

    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    COMPUTE_T scale = (COMPUTE_T)1.0;
    if (norm > (COMPUTE_T)a->clipping_threshold_t_pre) {
        scale = (COMPUTE_T)a->clipping_threshold_t_pre / (norm + (COMPUTE_T)a->epsilon);
    }

    for (uint i = 0; i < n_w; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_weights_shared, i);
        scalar_store_storage(a->clipped_partial_grad_weights_shared,
            a->weights_write_offset_elements + i, v * scale);
    }
    for (uint i = 0; i < n_b; i++) {
        COMPUTE_T v = scalar_load_storage(a->partial_grad_biases_shared, i);
        scalar_store_storage(a->clipped_partial_grad_biases_shared,
            a->biases_write_offset_elements + i, v * scale);
    }
}
