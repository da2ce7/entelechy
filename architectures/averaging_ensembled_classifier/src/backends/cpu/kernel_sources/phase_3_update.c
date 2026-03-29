/* phase_3_update.c — CPU Update-phase kernels (Nodes 21, 24, 25)
 *                     + get_struct_size_* verification exports
 *                     + get_simd_width query.
 *
 * Algorithmic reference: kernels/phase_3_update.cl.c
 */
#include "cpu_kernels.h"

/* ================================================================
 * task_normalize_gradients (Node 21)
 *
 * Element-wise gradient normalization by effective batch size.
 * Each task processes one element.
 *
 * task_index = element index
 * ================================================================ */
void task_normalize_gradients(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    NormalizeGradientsArgs* a = (NormalizeGradientsArgs*)raw_args;

    if (task_index >= a->parameter_count)
        return;

    const float normalizer = 1.0f / (a->effective_batch_size + a->epsilon);
    a->final_grad[task_index] = a->summed_grad[task_index] * normalizer;
}

/* ================================================================
 * task_adam_update (Node 24)
 *
 * Standard Adam optimizer step with host-precomputed beta powers.
 * Each task processes one parameter.
 *
 * task_index = parameter index
 * ================================================================ */
void task_adam_update(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    AdamUpdateArgs* a = (AdamUpdateArgs*)raw_args;

    if (task_index >= a->parameter_count)
        return;

    const float g      = a->final_grad[task_index];
    const float m_prev = a->m1[task_index];
    const float v_prev = a->m2[task_index];

    /* Update biased moment estimates */
    const float m_new = a->beta1 * m_prev + (1.0f - a->beta1) * g;
    const float v_new = a->beta2 * v_prev + (1.0f - a->beta2) * (g * g);

    /* Bias-corrected estimates (beta powers pre-computed by host) */
    const float m_hat = m_new / (1.0f - a->beta1_pow_t);
    const float v_hat = v_new / (1.0f - a->beta2_pow_t);

    /* Parameter update */
    const float update = a->learning_rate * m_hat
        / (sqrtf(v_hat) + a->epsilon);

    a->parameters[task_index] -= update;
    a->m1[task_index] = m_new;
    a->m2[task_index] = v_new;
}

/* ================================================================
 * task_clamp_temperatures (Node 25)
 *
 * In-place clamp of temperature parameters to [min, max].
 * Each task processes one temperature.
 *
 * task_index = module index
 * ================================================================ */
void task_clamp_temperatures(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    ClampTemperaturesArgs* a = (ClampTemperaturesArgs*)raw_args;

    if (task_index >= a->total_modules_count)
        return;

    float val = a->temperatures[task_index];
    if (val < a->min_value) val = a->min_value;
    if (val > a->max_value) val = a->max_value;
    a->temperatures[task_index] = val;
}

/* ================================================================
 * SIMD width query
 * ================================================================ */
uint get_simd_width(void) {
    return SIMD_WIDTH;
}

/* ================================================================
 * Layout Verification Exports (ADR-015)
 * ================================================================ */
size_t get_struct_size_forward_pass_args(void)          { return sizeof(ForwardPassArgs); }
size_t get_struct_size_render_logits_args(void)         { return sizeof(RenderLogitsArgs); }
size_t get_struct_size_cce_chunk_args(void)             { return sizeof(CceChunkArgs); }
size_t get_struct_size_bce_chunk_args(void)             { return sizeof(BceChunkArgs); }
size_t get_struct_size_module_param_grads_args(void)    { return sizeof(ModuleParamGradsArgs); }
size_t get_struct_size_backprop_to_hidden_args(void)    { return sizeof(BackpropToHiddenArgs); }
size_t get_struct_size_temp_gradients_args(void)        { return sizeof(TempGradientsArgs); }
size_t get_struct_size_clip_partials_args(void)         { return sizeof(ClipPartialsArgs); }
size_t get_struct_size_gather_permute_args(void)        { return sizeof(GatherPermuteArgs); }
size_t get_struct_size_reduction_tree_plan(void)        { return sizeof(ReductionTreePlanC); }
size_t get_struct_size_stabilize_reduce_args(void)      { return sizeof(StabilizeReduceArgs); }
size_t get_struct_size_clip_intermediate_args(void)     { return sizeof(ClipIntermediateArgs); }
size_t get_struct_size_backprop_shared_weights_args(void) { return sizeof(BackpropSharedWeightsArgs); }
size_t get_struct_size_backprop_shared_biases_args(void)  { return sizeof(BackpropSharedBiasesArgs); }
size_t get_struct_size_clip_shared_grads_args(void)     { return sizeof(ClipSharedGradsArgs); }
size_t get_struct_size_normalize_gradients_args(void)   { return sizeof(NormalizeGradientsArgs); }
size_t get_struct_size_adam_update_args(void)            { return sizeof(AdamUpdateArgs); }
size_t get_struct_size_clamp_temperatures_args(void)    { return sizeof(ClampTemperaturesArgs); }
