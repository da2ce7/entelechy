/* phase_2_learn_D_backprop.inc — Precision-parameterized backprop kernels (Nodes 17, 18, 19).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_D_backprop.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - sample_mask, hidden_activations, input, partial_grad_*, clipped_partial_grad_* → storage
 *   - summed_grad_hidden_activations → compute (COMPUTE_T*)
 *
 * Argument naming convention (ADR-013):
 *   Struct field accesses match exactly the full-form @param names defined in
 *   kernels/kernels.cl.h.
 */

/* ================================================================
 * task_backprop_shared_weights (Node 17)
 * ================================================================ */
void PREC_FN(task_backprop_shared_weights)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(BackpropSharedWeightsArgs)* a = (PREC_TY(BackpropSharedWeightsArgs)*)raw_args;

    const uint i_idx = task_index / a->src_scalar_NATURAL_padded_hidden_count;
    const uint j_idx = task_index % a->src_scalar_NATURAL_padded_hidden_count;

    if (i_idx >= a->src_scalar_NATURAL_padded_input_count || j_idx >= a->src_scalar_NATURAL_padded_hidden_count)
        return;

    /* SIMD-major layout: [h_block][input][SIMD_WIDTH] */
    const uint hb   = j_idx / SIMD_WIDTH;
    const uint lane = j_idx % SIMD_WIDTH;
    const size_t out_idx = (size_t)hb * a->src_scalar_NATURAL_padded_input_count * SIMD_WIDTH
                         + (size_t)i_idx * SIMD_WIDTH + lane;

    /* Padding Zero-Establishment (CONTRACT.md): The kernel is the sole
     * guarantor of zeros at padding positions. */
    const bool is_padding = (i_idx >= a->src_scalar_NATURAL_input_count) || (j_idx >= a->src_scalar_NATURAL_hidden_count);
    if (is_padding) {
        scalar_store_storage(a->dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major, out_idx, (COMPUTE_T)0.0);
        return;
    }

    COMPUTE_T grad_sw = (COMPUTE_T)0.0;
    for (uint b_local = 0; b_local < a->src_scalar_NATURAL_batch_chunk_count; b_local++) {
        const uint b_global = a->src_scalar_NATURAL_batch_chunk_offset + b_local;

        if (!load_sample_mask(a->src_buffer_GLOBAL_sample_mask, b_global))
            continue;

        const size_t h_off = (size_t)b_global * a->src_scalar_NATURAL_padded_hidden_count + j_idx;
        const COMPUTE_T grad_h    = a->src_buffer_GLOBAL_summed_grad_hidden_activations[h_off];  /* COMPUTE_T* compute */
        const COMPUTE_T hidden_val = (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_hidden_activations, h_off);

        COMPUTE_T d_act;
        if (a->src_scalar_FLAG_use_explicit_hidden_mask) {
            d_act = (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_hidden_mask, h_off);
        } else {
            d_act = (hidden_val > (COMPUTE_T)0.0) ? (COMPUTE_T)1.0 : (COMPUTE_T)0.0;
        }
        const COMPUTE_T dL_dZ = grad_h * d_act;

        const COMPUTE_T input_val = (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_input,
            (size_t)b_global * a->src_scalar_NATURAL_padded_input_count + i_idx);
        grad_sw += dL_dZ * input_val;
    }

    scalar_store_storage(a->dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major, out_idx, grad_sw);
}

/* ================================================================
 * task_backprop_shared_biases (Node 18)
 * ================================================================ */
void PREC_FN(task_backprop_shared_biases)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(BackpropSharedBiasesArgs)* a = (PREC_TY(BackpropSharedBiasesArgs)*)raw_args;

    const uint j_idx = task_index;
    if (j_idx >= a->src_scalar_NATURAL_padded_hidden_count)
        return;

    /* Padding Zero-Establishment (CONTRACT.md): The kernel is the sole
     * guarantor of zeros at padding positions. */
    if (j_idx >= a->src_scalar_NATURAL_hidden_count) {
        scalar_store_storage(a->dest_buffer_GLOBAL_partial_grad_biases_shared, j_idx, (COMPUTE_T)0.0);
        return;
    }

    COMPUTE_T grad_sb = (COMPUTE_T)0.0;
    for (uint b_local = 0; b_local < a->src_scalar_NATURAL_batch_chunk_count; b_local++) {
        const uint b_global = a->src_scalar_NATURAL_batch_chunk_offset + b_local;

        if (!load_sample_mask(a->src_buffer_GLOBAL_sample_mask, b_global))
            continue;

        const size_t h_off = (size_t)b_global * a->src_scalar_NATURAL_padded_hidden_count + j_idx;
        const COMPUTE_T grad_h    = a->src_buffer_GLOBAL_summed_grad_hidden_activations[h_off];  /* COMPUTE_T* compute */
        const COMPUTE_T hidden_val = (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_hidden_activations, h_off);

        COMPUTE_T d_act;
        if (a->src_scalar_FLAG_use_explicit_hidden_mask) {
            d_act = (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_hidden_mask, h_off);
        } else {
            d_act = (hidden_val > (COMPUTE_T)0.0) ? (COMPUTE_T)1.0 : (COMPUTE_T)0.0;
        }
        grad_sb += grad_h * d_act;
    }

    scalar_store_storage(a->dest_buffer_GLOBAL_partial_grad_biases_shared, j_idx, grad_sb);
}

/* ================================================================
 * task_clip_shared_grads (Node 19)
 * ================================================================ */
void PREC_FN(task_clip_shared_grads)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    PREC_TY(ClipSharedGradsArgs)* a = (PREC_TY(ClipSharedGradsArgs)*)raw_args;

    const uint n_w = a->src_scalar_NATURAL_weights_parameter_count;
    const uint n_b = a->src_scalar_NATURAL_biases_parameter_count;

    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint i = 0; i < n_w; i++) {
        COMPUTE_T v = scalar_load_storage(a->src_buffer_GLOBAL_partial_grad_weights_shared_simd_major, i);
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_b; i++) {
        COMPUTE_T v = scalar_load_storage(a->src_buffer_GLOBAL_partial_grad_biases_shared, i);
        sum_sq += v * v;
    }

    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    COMPUTE_T scale = (COMPUTE_T)1.0;
    /* Diagnostic bypass: negative threshold skips all clipping */
    if (a->src_scalar_REAL_clipping_threshold_t_pre >= (COMPUTE_T)0.0 && norm > (COMPUTE_T)a->src_scalar_REAL_clipping_threshold_t_pre) {
        scale = (COMPUTE_T)a->src_scalar_REAL_clipping_threshold_t_pre / (norm + (COMPUTE_T)a->src_scalar_REAL_epsilon);
    }

    for (uint i = 0; i < n_w; i++) {
        COMPUTE_T v = scalar_load_storage(a->src_buffer_GLOBAL_partial_grad_weights_shared_simd_major, i);
        scalar_store_storage(a->dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major,
            a->out_scalar_NATURAL_weights_write_offset + i, v * scale);
    }
    for (uint i = 0; i < n_b; i++) {
        COMPUTE_T v = scalar_load_storage(a->src_buffer_GLOBAL_partial_grad_biases_shared, i);
        scalar_store_storage(a->dest_buffer_GLOBAL_clipped_partial_grad_biases_shared,
            a->out_scalar_NATURAL_biases_write_offset + i, v * scale);
    }
}
