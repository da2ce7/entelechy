/* phase_2_learn_C_reduction.inc — Precision-parameterized reduction kernels (Nodes 13-16, 20).
 *
 * Included by phase_2_learn_C_reduction.c with REAL_T / PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_C_reduction.cl.c
 */

/* ================================================================
 * task_gather_permute_grad_h (Node 13)
 * ================================================================ */
void PREC_FN(task_gather_permute_grad_h)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(GatherPermuteArgs)* a = (PREC_TY(GatherPermuteArgs)*)raw_args;

    const uint bh_total = a->total_batch_count * a->padded_hidden_count;
    const uint bh_flat_idx       = task_index / a->total_modules_count;
    const uint module_global_idx = task_index % a->total_modules_count;

    if (bh_flat_idx >= bh_total || module_global_idx >= a->total_modules_count)
        return;

    const uint batch_idx = bh_flat_idx / a->padded_hidden_count;
    const uint h_idx     = bh_flat_idx % a->padded_hidden_count;

    const uint module_chunk_idx = module_global_idx / a->modules_per_chunk;
    const uint module_local_idx = module_global_idx % a->modules_per_chunk;

    float accum = 0.0f;
    for (uint cc = 0; cc < a->num_class_chunks; cc++) {
        const uint flat_tile_idx = module_chunk_idx * a->num_class_chunks + cc;
        if (flat_tile_idx >= a->total_tile_count) break;

        const size_t tile_size = (size_t)a->modules_per_chunk
            * a->total_batch_count * a->padded_hidden_count;
        const size_t tile_base = (size_t)flat_tile_idx * tile_size;
        const size_t local_off = (size_t)module_local_idx
            * a->total_batch_count * a->padded_hidden_count
            + (size_t)batch_idx * a->padded_hidden_count + h_idx;
        accum += a->clipped_partial_grad_hidden_activations_aos[tile_base + local_off];
    }

    const size_t write_idx = (size_t)bh_flat_idx * a->padded_total_modules_count
        + module_global_idx;
    a->clipped_grad_hidden_activations_permuted_soa[write_idx] = accum;
}

/* ================================================================
 * task_stabilize_reduce_grad_h (Node 16)
 * ================================================================ */
void PREC_FN(task_stabilize_reduce_grad_h)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(StabilizeReduceArgs)* a = (PREC_TY(StabilizeReduceArgs)*)raw_args;

    const uint row_idx = task_index;
    const uint total_rows = a->total_batch_count * a->padded_hidden_count;
    if (row_idx >= total_rows)
        return;

    const size_t row_offset = (size_t)row_idx * a->padded_total_modules_count;

    const uint M = a->total_modules_count;

    if (M <= 1) {
        float val = (M == 1)
            ? a->grad_hidden_activations_permuted_soa[row_offset]
            : 0.0f;
        a->summed_grad_hidden_activations[row_idx] = val;
        return;
    }

    uint K = a->policy_max_k;
    if (K < 2) K = 2;

    uint num_stages = 0;
    uint temp_items = M;
    while (temp_items > 1) {
        temp_items = (temp_items + K - 1) / K;
        num_stages++;
    }

    float stack_a[1024];
    float stack_b[1024];
    float* heap_a = NULL;
    float* heap_b = NULL;
    float* current;
    float* next;

    if (M <= 1024) {
        current = stack_a;
        next    = stack_b;
    } else {
        heap_a = (float*)malloc(M * sizeof(float));
        heap_b = (float*)malloc(M * sizeof(float));
        if (!heap_a || !heap_b) {
            free(heap_a);
            free(heap_b);
            a->summed_grad_hidden_activations[row_idx] = 0.0f;
            return;
        }
        current = heap_a;
        next    = heap_b;
    }

    /* Load initial values (REAL_T → float conversion) */
    for (uint i = 0; i < M; i++) {
        current[i] = a->grad_hidden_activations_permuted_soa[row_offset + i];
    }

    uint num_items = M;
    for (uint s = 0; s < num_stages; s++) {
        uint j = num_stages - 1 - s;
        float T_policy = a->policy_t_algorithmic + a->policy_lambda * (float)(j * j);
        uint K_actual = (K < num_items) ? K : num_items;
        float T_safety = (K_actual > 0)
            ? (a->fp_max / (float)K_actual) : a->fp_max;
        float threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint new_count = (num_items + K - 1) / K;
        for (uint g = 0; g < new_count; g++) {
            uint start = g * K;
            uint end = start + K;
            if (end > num_items) end = num_items;

            float sum = 0.0f;
            for (uint i = start; i < end; i++) {
                sum += current[i];
            }

            float norm = fabsf(sum);
            if (norm > threshold) {
                sum *= threshold / (norm + a->epsilon);
            }
            next[g] = sum;
        }

        float* tmp = current;
        current = next;
        next = tmp;
        num_items = new_count;
    }

    a->summed_grad_hidden_activations[row_idx] = current[0];
    free(heap_a);
    free(heap_b);
}

/* ================================================================
 * task_clip_intermediate (Nodes 15b, 20b)
 * ================================================================ */
void PREC_FN(task_clip_intermediate)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    PREC_TY(ClipIntermediateArgs)* a = (PREC_TY(ClipIntermediateArgs)*)raw_args;

    float sum_sq = 0.0f;
    for (uint i = 0; i < a->parameter_count; i++) {
        float v = a->intermediate_grad[i];
        sum_sq += v * v;
    }

    float norm = sqrtf(sum_sq);
    if (norm <= a->clipping_threshold_t_j)
        return;

    float scale = a->clipping_threshold_t_j / (norm + a->epsilon);

    for (uint i = 0; i < a->parameter_count; i++) {
        a->intermediate_grad[i] *= scale;
    }
}

/* ================================================================
 * Internal: per-precision ReduceNodeArgs and helper
 * ================================================================ */
typedef struct {
    const REAL_T* source;
    const uint*   offset_list;
    uint          fan_in;
    uint          partial_width;
    REAL_T*       dest;
    uint          dest_offset;
    float         threshold;
    float         epsilon;
} PREC_TY(ReduceNodeArgs);

static void PREC_FN(task_reduce_and_clip_node)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ReduceNodeArgs)* a = (PREC_TY(ReduceNodeArgs)*)raw_args;

    const uint pw = a->partial_width;
    REAL_T* dest = a->dest + (size_t)task_index * pw;

    for (uint e = 0; e < pw; e++) {
        dest[e] = 0.0f;
    }

    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        if (off == 0xFFFFFFFF) continue;
        const REAL_T* src = a->source + (size_t)off;
        for (uint e = 0; e < pw; e++) {
            dest[e] += src[e];
        }
    }

    float sum_sq = 0.0f;
    for (uint e = 0; e < pw; e++) {
        sum_sq += dest[e] * dest[e];
    }
    float norm = sqrtf(sum_sq);
    if (norm > a->threshold) {
        float scale = a->threshold / (norm + a->epsilon);
        for (uint e = 0; e < pw; e++) {
            dest[e] *= scale;
        }
    }
}

/* ================================================================
 * execute_reduction_tree
 * ================================================================ */
void PREC_FN(execute_reduction_tree)(ThreadPool* pool, PREC_TY(ReductionTreePlanC)* plan) {
    if (plan->num_stages == 0) return;

    const REAL_T* current_src = plan->partial_collection;
    REAL_T* staging[2];
    staging[0] = plan->staging_buffer_0;
    staging[1] = plan->staging_buffer_1;
    uint dst_idx = 0;

    for (uint s = 0; s < plan->num_stages; s++) {
        uint j = plan->num_stages - 1 - s;
        float T_policy = plan->t_algorithmic + plan->lambda * (float)(j * j);
        uint K = plan->stage_fan_in[s];
        float T_safety = (K > 0) ? (plan->fp_max / (float)K) : plan->fp_max;
        float threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint node_count = plan->stage_node_counts[s];

        REAL_T* dest;
        if (s == plan->num_stages - 1) {
            dest = plan->output;
        } else {
            dest = staging[dst_idx];
        }

        PREC_TY(ReduceNodeArgs) args;
        args.source = current_src;
        args.offset_list = plan->offset_lists_flat
            + plan->stage_offsets_into_list[s];
        args.fan_in = K;
        args.partial_width = plan->partial_width;
        args.dest = dest;
        args.dest_offset = 0;
        args.threshold = threshold;
        args.epsilon = plan->epsilon;

        pool_dispatch_and_wait(pool,
            (void (*)(void*, uint, uint))PREC_FN(task_reduce_and_clip_node),
            &args, node_count);

        if (s < plan->num_stages - 1) {
            current_src = dest;
            dst_idx = 1 - dst_idx;
        }
    }
}
