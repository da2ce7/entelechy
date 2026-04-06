/* phase_1_act.inc — Precision-parameterized Act-phase kernels (Nodes 4, 5).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_1_act.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - input, sample_mask, hidden_activations, hidden_mask, logits → storage
 *   - weights_*, biases_* → state
 */

/* ================================================================
 * task_forward_pass (Node 4)
 *
 * Each task processes one sample (task_index = sample within chunk).
 * SIMD-major weight layout: weights[h_block][input_dim][SIMD_WIDTH]
 * ================================================================ */
void PREC_FN(task_forward_pass)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ForwardPassArgs)* a = (PREC_TY(ForwardPassArgs)*)raw_args;

    if (task_index >= a->batch_chunk_count) return;

    const uint sample = a->batch_chunk_offset + task_index;
    const float mask_val = scalar_load_storage(a->sample_mask, sample);

    STORAGE_T* hidden_row     = a->hidden_activations + (size_t)sample * a->padded_hidden_count;
    STORAGE_T* hidden_msk_row = a->hidden_mask + (size_t)sample * a->padded_hidden_count;

    /* Padded samples: zero out and return */
    if (mask_val < 0.5f) {
        memset(hidden_row, 0, a->padded_hidden_count * PREC_SIZEOF_STORAGE);
        memset(hidden_msk_row, 0, a->padded_hidden_count * PREC_SIZEOF_STORAGE);
        return;
    }

    const STORAGE_T* input_row = a->input + (size_t)sample * a->padded_input_count;
    const uint padded_input  = a->padded_input_count;
    const uint padded_hidden = a->padded_hidden_count;
    const uint num_h_blocks  = padded_hidden / SIMD_WIDTH;

    const simd_float v_zero = simd_zero();
    const simd_float v_mask = simd_set1(mask_val);

    for (uint hb = 0; hb < num_h_blocks; hb++) {
        const uint h_offset = hb * SIMD_WIDTH;

        /* Initialize with bias (state-role buffer) */
        simd_float accum = simd_load_state(&a->biases_shared[h_offset]);

        /* Weight pointer for this h_block (state-role buffer):
         * weights_simd_major[hb * padded_input * SIMD_WIDTH + i * SIMD_WIDTH] */
        const STATE_T* w_block = a->weights_shared_simd_major
                               + (size_t)hb * padded_input * SIMD_WIDTH;

        /* Dot product across all input dimensions */
        for (uint i = 0; i < padded_input; i++) {
            simd_float x_broadcast = simd_set1(scalar_load_storage(input_row, i));
            simd_float w_vec = simd_load_state(&w_block[(size_t)i * SIMD_WIDTH]);
            accum = simd_fmadd(w_vec, x_broadcast, accum);
        }

        /* ReLU */
        simd_float activated = simd_max(v_zero, accum);

        /* Apply sample mask */
        activated = simd_mul(activated, v_mask);

        /* Store hidden activations (storage-role buffer) */
        simd_store_storage(&hidden_row[h_offset], activated);

        /* Hidden mask: 1.0 where pre-ReLU > 0 AND sample is valid */
        simd_float relu_mask = simd_select(v_zero, v_mask, accum);
        simd_store_storage(&hidden_msk_row[h_offset], relu_mask);
    }
}

/* ================================================================
 * task_render_logits (Node 5)
 *
 * Each task processes one flat work-unit.  We decompose task_index
 * into (module_local, batch_local, class_local) using the chunk dims.
 * ================================================================ */
void PREC_FN(task_render_logits)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(RenderLogitsArgs)* a = (PREC_TY(RenderLogitsArgs)*)raw_args;

    /* Decompose flat task_index into 3D coordinates:
     * task_index = module_local * (batch_chunk_count * class_chunk_count)
     *            + batch_local * class_chunk_count
     *            + class_local */
    const uint mc_bc = a->batch_chunk_count * a->class_chunk_count;
    const uint module_local_idx = task_index / mc_bc;
    const uint remainder        = task_index % mc_bc;
    const uint batch_local_idx  = remainder / a->class_chunk_count;
    const uint class_local_idx  = remainder % a->class_chunk_count;

    if (module_local_idx >= a->module_chunk_count ||
        batch_local_idx  >= a->batch_chunk_count ||
        class_local_idx  >= a->class_chunk_count)
        return;

    const uint module_global = a->module_chunk_offset + module_local_idx;
    const uint batch_global  = a->batch_chunk_offset + batch_local_idx;
    const uint class_global  = a->class_chunk_offset + class_local_idx;

    /* Output address (storage-role buffer) */
    const size_t out_idx = (size_t)module_global * a->total_batch_count
                              * a->padded_total_output_class_count
                         + (size_t)batch_global * a->padded_total_output_class_count
                         + class_global;

    /* Initialize with bias (state-role buffer) */
    COMPUTE_T logit = scalar_load_state(a->biases_module,
                      (size_t)module_global * a->padded_total_output_class_count + class_global);

    /* Sparse dot product over hidden dimension */
    const size_t h_base = (size_t)batch_global * a->padded_hidden_count;
    for (uint h = 0; h < a->hidden_count; h++) {
        const COMPUTE_T h_mask = (COMPUTE_T)scalar_load_storage(a->hidden_mask, h_base + h);
        if (h_mask > (COMPUTE_T)0.5) {
            const COMPUTE_T h_val = (COMPUTE_T)scalar_load_storage(a->hidden_activations, h_base + h);
            const size_t w_idx = (size_t)module_global * a->padded_hidden_count
                                   * a->padded_total_output_class_count
                               + (size_t)h * a->padded_total_output_class_count
                               + class_global;
            logit += h_val * (COMPUTE_T)scalar_load_state(a->weights_module, w_idx);
        }
    }

    scalar_store_storage(a->logits, out_idx, logit);
}
