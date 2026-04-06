/* phase_3_update.inc — Precision-parameterized Update-phase kernels (Nodes 21, 24, 25)
 *                      + get_struct_size_* verification exports.
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_3_update.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - summed_grad, final_grad → compute (COMPUTE_T*)
 *   - parameters, m1, m2, temperatures → state
 */

/* ================================================================
 * task_normalize_gradients (Node 21)
 * ================================================================ */
void PREC_FN(task_normalize_gradients)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(NormalizeGradientsArgs)* a = (PREC_TY(NormalizeGradientsArgs)*)raw_args;

    if (task_index >= a->parameter_count)
        return;

    const COMPUTE_T normalizer = (COMPUTE_T)1.0 / (a->effective_batch_size + a->epsilon);
    a->final_grad[task_index] = a->summed_grad[task_index] * normalizer;
}

/* ================================================================
 * task_adam_update (Node 24)
 *
 * State-Precision Accumulation (ADR-027):
 *   - EMA updates and parameter subtraction in ACCUM_T = max(COMPUTE_T, STATE_T)
 *   - Bias correction and delta computation in COMPUTE_T
 *   - When ACCUM_T == COMPUTE_T, all casts are identity operations
 * ================================================================ */
void PREC_FN(task_adam_update)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(AdamUpdateArgs)* a = (PREC_TY(AdamUpdateArgs)*)raw_args;

    if (task_index >= a->parameter_count)
        return;

    /* Load gradient in COMPUTE_T (precision role: compute) */
    const COMPUTE_T g = a->final_grad[task_index];

    /* State-Precision Accumulation: load moments at full state precision */
    const ACCUM_T m_prev = scalar_load_state_for_accum(a->m1, task_index);
    const ACCUM_T v_prev = scalar_load_state_for_accum(a->m2, task_index);

    /* EMA updates in ACCUM_T (preserves state precision) */
    const ACCUM_T g_accum     = scalar_widen_to_accum(g);
    const ACCUM_T beta1_accum = scalar_widen_to_accum(a->beta1);
    const ACCUM_T beta2_accum = scalar_widen_to_accum(a->beta2);

    /* First moment: m_new = β₁ · m_prev + (1 - β₁) · g */
    const ACCUM_T m_new = beta1_accum * m_prev +
                          ((ACCUM_T)1.0 - beta1_accum) * g_accum;

    /* Second moment: v_new = β₂ · v_prev + (1 - β₂) · g² */
    const ACCUM_T v_new = beta2_accum * v_prev +
                          ((ACCUM_T)1.0 - beta2_accum) * (g_accum * g_accum);

    /* Store updated moments at state precision */
    scalar_store_state_from_accum(a->m1, task_index, m_new);
    scalar_store_state_from_accum(a->m2, task_index, v_new);

    /* Bias correction in COMPUTE_T (transformative) */
    const COMPUTE_T m_hat = scalar_narrow_from_accum(m_new) /
                            ((COMPUTE_T)1.0 - a->beta1_pow_t);
    const COMPUTE_T v_hat = scalar_narrow_from_accum(v_new) /
                            ((COMPUTE_T)1.0 - a->beta2_pow_t);

    /* Parameter update: delta in COMPUTE_T, subtraction in ACCUM_T (accumulative) */
    const COMPUTE_T update = a->learning_rate * m_hat
        / (COMPUTE_SQRT(v_hat) + a->epsilon);

    const ACCUM_T param_val = scalar_load_state_for_accum(a->parameters, task_index);
    scalar_store_state_from_accum(a->parameters, task_index,
                                  param_val - scalar_widen_to_accum(update));
}

/* ================================================================
 * task_clamp_temperatures (Node 25)
 * ================================================================ */
void PREC_FN(task_clamp_temperatures)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ClampTemperaturesArgs)* a = (PREC_TY(ClampTemperaturesArgs)*)raw_args;

    if (task_index >= a->total_modules_count)
        return;

    COMPUTE_T val = scalar_load_state(a->temperatures, task_index);
    if (val < a->min_value) val = a->min_value;
    if (val > a->max_value) val = a->max_value;
    scalar_store_state(a->temperatures, task_index, val);
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
size_t PREC_FN(get_struct_size_reduction_tree_plan_compute_entry)(void) { return sizeof(PREC_TY(ReductionTreePlanComputeEntry)); }
size_t PREC_FN(get_struct_size_stabilize_reduce_args)(void)      { return sizeof(PREC_TY(StabilizeReduceArgs)); }
size_t PREC_FN(get_struct_size_clip_intermediate_args)(void)     { return sizeof(PREC_TY(ClipIntermediateArgs)); }
size_t PREC_FN(get_struct_size_backprop_shared_weights_args)(void) { return sizeof(PREC_TY(BackpropSharedWeightsArgs)); }
size_t PREC_FN(get_struct_size_backprop_shared_biases_args)(void)  { return sizeof(PREC_TY(BackpropSharedBiasesArgs)); }
size_t PREC_FN(get_struct_size_clip_shared_grads_args)(void)     { return sizeof(PREC_TY(ClipSharedGradsArgs)); }
size_t PREC_FN(get_struct_size_normalize_gradients_args)(void)   { return sizeof(PREC_TY(NormalizeGradientsArgs)); }
size_t PREC_FN(get_struct_size_adam_update_args)(void)            { return sizeof(PREC_TY(AdamUpdateArgs)); }
size_t PREC_FN(get_struct_size_clamp_temperatures_args)(void)    { return sizeof(PREC_TY(ClampTemperaturesArgs)); }
