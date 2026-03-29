/* phase_2_learn_D_backprop.c — CPU streaming backprop kernels (Nodes 17, 18, 19).
 *
 * Algorithmic reference: kernels/phase_2_learn_D_backprop.cl.c
 * Node 17: backprop_shared_weights — outer product reduction
 * Node 18: backprop_shared_biases  — batch reduction
 * Node 19: clip_shared_grads       — virtual-vector clip with placement write
 */
#include "cpu_kernels.h"

/* ================================================================
 * task_backprop_shared_weights (Node 17)
 *
 * Each task computes one (i_idx, j_idx) gradient element,
 * reducing over the assigned batch chunk.
 *
 * task_index = i_idx * padded_hidden_count + j_idx
 * ================================================================ */
void task_backprop_shared_weights(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    BackpropSharedWeightsArgs* a = (BackpropSharedWeightsArgs*)raw_args;

    const uint i_idx = task_index / a->padded_hidden_count;
    const uint j_idx = task_index % a->padded_hidden_count;

    if (i_idx >= a->padded_input_count || j_idx >= a->padded_hidden_count)
        return;

    /* Reduce over batch chunk */
    float grad_sw = 0.0f;
    for (uint b_local = 0; b_local < a->batch_chunk_count; b_local++) {
        const uint b_global = a->batch_chunk_offset + b_local;

        if (a->sample_mask[b_global] < 0.5f)
            continue;

        /* Chain rule: dL/dW_ij = dL/dA_j * dA_j/dZ_j * dZ_j/dW_ij */
        const size_t h_off = (size_t)b_global * a->padded_hidden_count + j_idx;
        const float grad_h    = a->summed_grad_hidden_activations[h_off];
        const float hidden_val = a->hidden_activations[h_off];

        /* ReLU derivative */
        const float d_act = (hidden_val > 0.0f) ? 1.0f : 0.0f;
        const float dL_dZ = grad_h * d_act;

        /* dZ/dW_ij = input[i] */
        const float input_val = a->input[
            (size_t)b_global * a->padded_input_count + i_idx];
        grad_sw += dL_dZ * input_val;
    }

    /* Write to placement: hidden-major order (j * padded_input + i)
     * matching SIMD-major weight layout */
    const size_t chunk_base = (size_t)a->batch_chunk_index
        * a->padded_input_count * a->padded_hidden_count;
    const size_t out_idx = chunk_base
        + (size_t)j_idx * a->padded_input_count + i_idx;
    a->partial_grad_weights_shared[out_idx] = grad_sw;
}

/* ================================================================
 * task_backprop_shared_biases (Node 18)
 *
 * Each task computes one j_idx bias gradient element,
 * reducing over the assigned batch chunk.
 *
 * task_index = j_idx
 * ================================================================ */
void task_backprop_shared_biases(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    BackpropSharedBiasesArgs* a = (BackpropSharedBiasesArgs*)raw_args;

    const uint j_idx = task_index;
    if (j_idx >= a->padded_hidden_count)
        return;

    float grad_sb = 0.0f;
    for (uint b_local = 0; b_local < a->batch_chunk_count; b_local++) {
        const uint b_global = a->batch_chunk_offset + b_local;

        if (a->sample_mask[b_global] < 0.5f)
            continue;

        const size_t h_off = (size_t)b_global * a->padded_hidden_count + j_idx;
        const float grad_h    = a->summed_grad_hidden_activations[h_off];
        const float hidden_val = a->hidden_activations[h_off];

        /* ReLU derivative */
        const float d_act = (hidden_val > 0.0f) ? 1.0f : 0.0f;
        grad_sb += grad_h * d_act;
    }

    const size_t chunk_base = (size_t)a->batch_chunk_index * a->padded_hidden_count;
    a->partial_grad_biases_shared[chunk_base + j_idx] = grad_sb;
}

/* ================================================================
 * task_clip_shared_grads (Node 19)
 *
 * Virtual-vector clip of weight + bias gradient chunks with
 * placement write to host-specified offsets.
 *
 * task_index is always 0 (one dispatch per chunk).
 * ================================================================ */
void task_clip_shared_grads(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    ClipSharedGradsArgs* a = (ClipSharedGradsArgs*)raw_args;

    const uint n_w = a->weights_parameter_count;
    const uint n_b = a->biases_parameter_count;
    const uint total = n_w + n_b;

    /* Pass 1: sum of squares */
    float sum_sq = 0.0f;
    for (uint i = 0; i < n_w; i++) {
        float v = a->partial_grad_weights_shared[i];
        sum_sq += v * v;
    }
    for (uint i = 0; i < n_b; i++) {
        float v = a->partial_grad_biases_shared[i];
        sum_sq += v * v;
    }

    float norm = sqrtf(sum_sq);
    float scale = 1.0f;
    if (norm > a->clipping_threshold_t_pre) {
        scale = a->clipping_threshold_t_pre / (norm + a->epsilon);
    }

    /* Pass 2: scale and write to placement offsets */
    for (uint i = 0; i < n_w; i++) {
        a->clipped_partial_grad_weights_shared[
            a->weights_write_offset_elements + i] =
            a->partial_grad_weights_shared[i] * scale;
    }
    for (uint i = 0; i < n_b; i++) {
        a->clipped_partial_grad_biases_shared[
            a->biases_write_offset_elements + i] =
            a->partial_grad_biases_shared[i] * scale;
    }
}
