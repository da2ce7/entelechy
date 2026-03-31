/* phase_3_update.inc — Precision-parameterized Update-phase kernels (Nodes 21, 24, 25)
 *                      + get_struct_size_* verification exports.
 *
 * Included by phase_3_update.c with REAL_T / PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_3_update.cl.c
 */

/* ================================================================
 * task_normalize_gradients (Node 21)
 * ================================================================ */
void PREC_FN(task_normalize_gradients)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(NormalizeGradientsArgs)* a = (PREC_TY(NormalizeGradientsArgs)*)raw_args;

    if (task_index >= a->parameter_count)
        return;

    const float normalizer = 1.0f / (a->effective_batch_size + a->epsilon);
    a->final_grad[task_index] = a->summed_grad[task_index] * normalizer;
}

/* ================================================================
 * task_adam_update (Node 24)
 * ================================================================ */
void PREC_FN(task_adam_update)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(AdamUpdateArgs)* a = (PREC_TY(AdamUpdateArgs)*)raw_args;

    if (task_index >= a->parameter_count)
        return;

    const float g      = a->final_grad[task_index];
    const float m_prev = a->m1[task_index];
    const float v_prev = a->m2[task_index];

    const float m_new = a->beta1 * m_prev + (1.0f - a->beta1) * g;
    const float v_new = a->beta2 * v_prev + (1.0f - a->beta2) * (g * g);

    const float m_hat = m_new / (1.0f - a->beta1_pow_t);
    const float v_hat = v_new / (1.0f - a->beta2_pow_t);

    const float update = a->learning_rate * m_hat
        / (sqrtf(v_hat) + a->epsilon);

    a->parameters[task_index] -= update;
    a->m1[task_index] = m_new;
    a->m2[task_index] = v_new;
}

/* ================================================================
 * task_clamp_temperatures (Node 25)
 * ================================================================ */
void PREC_FN(task_clamp_temperatures)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ClampTemperaturesArgs)* a = (PREC_TY(ClampTemperaturesArgs)*)raw_args;

    if (task_index >= a->total_modules_count)
        return;

    float val = a->temperatures[task_index];
    if (val < a->min_value) val = a->min_value;
    if (val > a->max_value) val = a->max_value;
    a->temperatures[task_index] = val;
}

/* ================================================================
 * Layout Verification Exports (ADR-015)
 * ================================================================ */
size_t PREC_FN(get_struct_size_forward_pass_args)(void)          { return sizeof(PREC_TY(ForwardPassArgs)); }
size_t PREC_FN(get_struct_size_render_logits_args)(void)         { return sizeof(PREC_TY(RenderLogitsArgs)); }
size_t PREC_FN(get_struct_size_cce_chunk_args)(void)             { return sizeof(PREC_TY(CceChunkArgs)); }
size_t PREC_FN(get_struct_size_bce_chunk_args)(void)             { return sizeof(PREC_TY(BceChunkArgs)); }
size_t PREC_FN(get_struct_size_module_param_grads_args)(void)    { return sizeof(PREC_TY(ModuleParamGradsArgs)); }
size_t PREC_FN(get_struct_size_backprop_to_hidden_args)(void)    { return sizeof(PREC_TY(BackpropToHiddenArgs)); }
size_t PREC_FN(get_struct_size_temp_gradients_args)(void)        { return sizeof(PREC_TY(TempGradientsArgs)); }
size_t PREC_FN(get_struct_size_clip_partials_args)(void)         { return sizeof(PREC_TY(ClipPartialsArgs)); }
size_t PREC_FN(get_struct_size_gather_permute_args)(void)        { return sizeof(PREC_TY(GatherPermuteArgs)); }
size_t PREC_FN(get_struct_size_reduction_tree_plan)(void)        { return sizeof(PREC_TY(ReductionTreePlanC)); }
size_t PREC_FN(get_struct_size_stabilize_reduce_args)(void)      { return sizeof(PREC_TY(StabilizeReduceArgs)); }
size_t PREC_FN(get_struct_size_clip_intermediate_args)(void)     { return sizeof(PREC_TY(ClipIntermediateArgs)); }
size_t PREC_FN(get_struct_size_backprop_shared_weights_args)(void) { return sizeof(PREC_TY(BackpropSharedWeightsArgs)); }
size_t PREC_FN(get_struct_size_backprop_shared_biases_args)(void)  { return sizeof(PREC_TY(BackpropSharedBiasesArgs)); }
size_t PREC_FN(get_struct_size_clip_shared_grads_args)(void)     { return sizeof(PREC_TY(ClipSharedGradsArgs)); }
size_t PREC_FN(get_struct_size_normalize_gradients_args)(void)   { return sizeof(PREC_TY(NormalizeGradientsArgs)); }
size_t PREC_FN(get_struct_size_adam_update_args)(void)            { return sizeof(PREC_TY(AdamUpdateArgs)); }
size_t PREC_FN(get_struct_size_clamp_temperatures_args)(void)    { return sizeof(PREC_TY(ClampTemperaturesArgs)); }
