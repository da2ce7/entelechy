/* phase_2_learn_C_reduction.inc — Precision-parameterized reduction kernels (Nodes 13-16, 20).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_C_reduction.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - clipped_partial_grad_*, grad_*_permuted, partial_collection, staging → storage
 *   - summed_grad_*, output → compute (COMPUTE_T)
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

    COMPUTE_T accum = (COMPUTE_T)0.0;
    for (uint cc = 0; cc < a->num_class_chunks; cc++) {
        const uint flat_tile_idx = module_chunk_idx * a->num_class_chunks + cc;
        if (flat_tile_idx >= a->total_tile_count) break;

        const size_t tile_size = (size_t)a->modules_per_chunk
            * a->total_batch_count * a->padded_hidden_count;
        const size_t tile_base = (size_t)flat_tile_idx * tile_size;
        const size_t local_off = (size_t)module_local_idx
            * a->total_batch_count * a->padded_hidden_count
            + (size_t)batch_idx * a->padded_hidden_count + h_idx;
        accum += (COMPUTE_T)scalar_load_storage(a->clipped_partial_grad_hidden_activations_aos,
                                     tile_base + local_off);
    }

    const size_t write_idx = (size_t)bh_flat_idx * a->padded_total_modules_count
        + module_global_idx;
    scalar_store_storage(a->clipped_grad_hidden_activations_permuted_soa, write_idx, accum);
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
        COMPUTE_T val = (M == 1)
            ? (COMPUTE_T)scalar_load_storage(a->grad_hidden_activations_permuted_soa, row_offset)
            : (COMPUTE_T)0.0;
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

    COMPUTE_T stack_a[1024];
    COMPUTE_T stack_b[1024];
    COMPUTE_T* heap_a = NULL;
    COMPUTE_T* heap_b = NULL;
    COMPUTE_T* current;
    COMPUTE_T* next;

    if (M <= 1024) {
        current = stack_a;
        next    = stack_b;
    } else {
        heap_a = (COMPUTE_T*)malloc(M * sizeof(COMPUTE_T));
        heap_b = (COMPUTE_T*)malloc(M * sizeof(COMPUTE_T));
        if (!heap_a || !heap_b) {
            free(heap_a);
            free(heap_b);
            a->summed_grad_hidden_activations[row_idx] = (COMPUTE_T)0.0;
            return;
        }
        current = heap_a;
        next    = heap_b;
    }

    /* Load initial values (STORAGE_T → COMPUTE_T conversion) */
    for (uint i = 0; i < M; i++) {
        current[i] = scalar_load_storage(a->grad_hidden_activations_permuted_soa,
                                         row_offset + i);
    }

    uint num_items = M;
    for (uint s = 0; s < num_stages; s++) {
        uint j = num_stages - 1 - s;
        COMPUTE_T T_policy = a->policy_t_algorithmic + a->policy_lambda * (COMPUTE_T)(j * j);
        uint K_actual = (K < num_items) ? K : num_items;
        COMPUTE_T T_safety = (K_actual > 0)
            ? (a->fp_max / (COMPUTE_T)K_actual) : a->fp_max;
        COMPUTE_T threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint new_count = (num_items + K - 1) / K;
        for (uint g = 0; g < new_count; g++) {
            uint start = g * K;
            uint end = start + K;
            if (end > num_items) end = num_items;

            COMPUTE_T sum = (COMPUTE_T)0.0;
            for (uint i = start; i < end; i++) {
                sum += current[i];
            }

            COMPUTE_T norm = COMPUTE_FABS(sum);
            if (norm > threshold) {
                sum *= threshold / (norm + a->epsilon);
            }
            next[g] = sum;
        }

        COMPUTE_T* tmp = current;
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

    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint i = 0; i < a->parameter_count; i++) {
        COMPUTE_T v = a->intermediate_grad[i];
        sum_sq += v * v;
    }

    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    if (norm <= a->clipping_threshold_t_j)
        return;

    COMPUTE_T scale = a->clipping_threshold_t_j / (norm + a->epsilon);

    for (uint i = 0; i < a->parameter_count; i++) {
        a->intermediate_grad[i] *= scale;
    }
}

/* ================================================================
 * Internal: per-precision ReduceNodeArgs and helpers
 *
 * The reduction tree uses STORAGE_T* for intermediate staging buffers
 * but writes the final result to COMPUTE_T* output (compute role).
 * ================================================================ */
typedef struct {
    const STORAGE_T* source;
    const uint*      offset_list;
    uint             fan_in;
    uint             partial_width;
    STORAGE_T*       staging_dest;   /* Intermediate stages write here */
    COMPUTE_T        threshold;
    COMPUTE_T        epsilon;
} PREC_TY(ReduceNodeStagingArgs);

typedef struct {
    const STORAGE_T* source;
    const uint*      offset_list;
    uint             fan_in;
    uint             partial_width;
    COMPUTE_T*       compute_dest;   /* Final stage writes here */
    COMPUTE_T        threshold;
    COMPUTE_T        epsilon;
} PREC_TY(ReduceNodeFinalArgs);

static void PREC_FN(task_reduce_staging)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ReduceNodeStagingArgs)* a = (PREC_TY(ReduceNodeStagingArgs)*)raw_args;

    const uint pw = a->partial_width;
    const size_t dest_base = (size_t)task_index * pw;

    /* Zero destination (staging buffer) */
    for (uint e = 0; e < pw; e++) {
        scalar_store_storage(a->staging_dest, dest_base + e, 0.0f);
    }

    /* Accumulate from sources */
    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    COMPUTE_T accum[pw];
    for (uint e = 0; e < pw; e++) accum[e] = (COMPUTE_T)0.0;

    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        if (off == 0xFFFFFFFF) continue;
        const size_t src_base = (size_t)off;
        for (uint e = 0; e < pw; e++) {
            accum[e] += (COMPUTE_T)scalar_load_storage(a->source, src_base + e);
        }
    }

    /* Clip and store to staging */
    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint e = 0; e < pw; e++) {
        sum_sq += accum[e] * accum[e];
    }
    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    if (norm > a->threshold) {
        COMPUTE_T scale = a->threshold / (norm + a->epsilon);
        for (uint e = 0; e < pw; e++) {
            accum[e] *= scale;
        }
    }
    for (uint e = 0; e < pw; e++) {
        scalar_store_storage(a->staging_dest, dest_base + e, accum[e]);
    }
}

static void PREC_FN(task_reduce_final)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ReduceNodeFinalArgs)* a = (PREC_TY(ReduceNodeFinalArgs)*)raw_args;

    const uint pw = a->partial_width;
    COMPUTE_T* dest = a->compute_dest + (size_t)task_index * pw;

    /* Zero destination (COMPUTE_T output) */
    for (uint e = 0; e < pw; e++) {
        dest[e] = (COMPUTE_T)0.0;
    }

    /* Accumulate from sources */
    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        if (off == 0xFFFFFFFF) continue;
        const size_t src_base = (size_t)off;
        for (uint e = 0; e < pw; e++) {
            dest[e] += (COMPUTE_T)scalar_load_storage(a->source, src_base + e);
        }
    }

    /* Clip */
    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint e = 0; e < pw; e++) {
        sum_sq += dest[e] * dest[e];
    }
    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    if (norm > a->threshold) {
        COMPUTE_T scale = a->threshold / (norm + a->epsilon);
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

    const STORAGE_T* current_src = plan->partial_collection;
    STORAGE_T* staging[2];
    staging[0] = plan->staging_buffer_0;
    staging[1] = plan->staging_buffer_1;
    uint dst_idx = 0;

    for (uint s = 0; s < plan->num_stages; s++) {
        uint j = plan->num_stages - 1 - s;
        COMPUTE_T T_policy = plan->t_algorithmic + plan->lambda * (COMPUTE_T)(j * j);
        uint K = plan->stage_fan_in[s];
        COMPUTE_T T_safety = (K > 0) ? (plan->fp_max / (COMPUTE_T)K) : plan->fp_max;
        COMPUTE_T threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint node_count = plan->stage_node_counts[s];
        const uint* offsets = plan->offset_lists_flat + plan->stage_offsets_into_list[s];

        if (s == plan->num_stages - 1) {
            /* Final stage: write to COMPUTE_T* output */
            PREC_TY(ReduceNodeFinalArgs) args;
            args.source = current_src;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->partial_width;
            args.compute_dest = plan->output;
            args.threshold = threshold;
            args.epsilon = plan->epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_final),
                &args, node_count);
        } else {
            /* Intermediate stage: write to STORAGE_T* staging */
            PREC_TY(ReduceNodeStagingArgs) args;
            args.source = current_src;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->partial_width;
            args.staging_dest = staging[dst_idx];
            args.threshold = threshold;
            args.epsilon = plan->epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_staging),
                &args, node_count);

            current_src = staging[dst_idx];
            dst_idx = 1 - dst_idx;
        }
    }
}

/* ================================================================
 * ADR-026: Precision-Typed Reduction Kernel Variants (Compute-Entry)
 *
 * Used for:
 *   - Interior stages of multi-stage reduction trees
 *   - Leaf stages whose source is natively COMPUTE_T (e.g., BCE loss)
 * ================================================================ */

/* --- Compute-entry argument structs (read COMPUTE_T directly) --- */
typedef struct {
    const COMPUTE_T* source;
    const uint*      offset_list;
    uint             fan_in;
    uint             partial_width;
    COMPUTE_T*       staging_dest;   /* Interior stages write here (COMPUTE_T) */
    COMPUTE_T        threshold;
    COMPUTE_T        epsilon;
} PREC_TY(ReduceNodeStagingFromComputeArgs);

typedef struct {
    const COMPUTE_T* source;
    const uint*      offset_list;
    uint             fan_in;
    uint             partial_width;
    COMPUTE_T*       compute_dest;   /* Final stage writes here */
    COMPUTE_T        threshold;
    COMPUTE_T        epsilon;
} PREC_TY(ReduceNodeFinalFromComputeArgs);

/* task_reduce_staging_from_compute — compute-entry variant (ADR-026 §4) */
static void PREC_FN(task_reduce_staging_from_compute)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ReduceNodeStagingFromComputeArgs)* a = (PREC_TY(ReduceNodeStagingFromComputeArgs)*)raw_args;

    const uint pw = a->partial_width;
    const size_t dest_base = (size_t)task_index * pw;

    /* Zero destination */
    for (uint e = 0; e < pw; e++) {
        a->staging_dest[dest_base + e] = (COMPUTE_T)0.0;
    }

    /* Accumulate from COMPUTE_T sources (no precision conversion) */
    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    COMPUTE_T accum[pw];
    for (uint e = 0; e < pw; e++) accum[e] = (COMPUTE_T)0.0;

    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        if (off == 0xFFFFFFFF) continue;
        const size_t src_base = (size_t)off;
        for (uint e = 0; e < pw; e++) {
            /* Direct COMPUTE_T read (no scalar_load_storage) */
            accum[e] += a->source[src_base + e];
        }
    }

    /* Clip and store */
    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint e = 0; e < pw; e++) {
        sum_sq += accum[e] * accum[e];
    }
    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    if (norm > a->threshold) {
        COMPUTE_T scale = a->threshold / (norm + a->epsilon);
        for (uint e = 0; e < pw; e++) {
            accum[e] *= scale;
        }
    }
    for (uint e = 0; e < pw; e++) {
        a->staging_dest[dest_base + e] = accum[e];
    }
}

/* task_reduce_final_from_compute — compute-entry variant (ADR-026 §4) */
static void PREC_FN(task_reduce_final_from_compute)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ReduceNodeFinalFromComputeArgs)* a = (PREC_TY(ReduceNodeFinalFromComputeArgs)*)raw_args;

    const uint pw = a->partial_width;
    COMPUTE_T* dest = a->compute_dest + (size_t)task_index * pw;

    /* Zero destination */
    for (uint e = 0; e < pw; e++) {
        dest[e] = (COMPUTE_T)0.0;
    }

    /* Accumulate from COMPUTE_T sources (no precision conversion) */
    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        if (off == 0xFFFFFFFF) continue;
        const size_t src_base = (size_t)off;
        for (uint e = 0; e < pw; e++) {
            /* Direct COMPUTE_T read (no scalar_load_storage) */
            dest[e] += a->source[src_base + e];
        }
    }

    /* Clip */
    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint e = 0; e < pw; e++) {
        sum_sq += dest[e] * dest[e];
    }
    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);
    if (norm > a->threshold) {
        COMPUTE_T scale = a->threshold / (norm + a->epsilon);
        for (uint e = 0; e < pw; e++) {
            dest[e] *= scale;
        }
    }
}

/* ================================================================
 * execute_reduction_tree_from_compute (ADR-026 §3)
 *
 * Compute-entry variant: reads from COMPUTE_T* partial_collection.
 * Used when source buffer is natively compute-role (e.g., BCE loss,
 * prior stage output from storage-entry reduction).
 * ================================================================ */
void PREC_FN(execute_reduction_tree_from_compute)(ThreadPool* pool, PREC_TY(ReductionTreePlanComputeEntry)* plan) {
    if (plan->num_stages == 0) return;

    const COMPUTE_T* current_src = plan->partial_collection;
    COMPUTE_T* staging[2];
    staging[0] = plan->staging_buffer_0;
    staging[1] = plan->staging_buffer_1;
    uint dst_idx = 0;

    for (uint s = 0; s < plan->num_stages; s++) {
        uint j = plan->num_stages - 1 - s;
        COMPUTE_T T_policy = plan->t_algorithmic + plan->lambda * (COMPUTE_T)(j * j);
        uint K = plan->stage_fan_in[s];
        COMPUTE_T T_safety = (K > 0) ? (plan->fp_max / (COMPUTE_T)K) : plan->fp_max;
        COMPUTE_T threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint node_count = plan->stage_node_counts[s];
        const uint* offsets = plan->offset_lists_flat + plan->stage_offsets_into_list[s];

        if (s == plan->num_stages - 1) {
            /* Final stage: write to COMPUTE_T* output */
            PREC_TY(ReduceNodeFinalFromComputeArgs) args;
            args.source = current_src;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->partial_width;
            args.compute_dest = plan->output;
            args.threshold = threshold;
            args.epsilon = plan->epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_final_from_compute),
                &args, node_count);
        } else {
            /* Interior stage: write to COMPUTE_T* staging */
            PREC_TY(ReduceNodeStagingFromComputeArgs) args;
            args.source = current_src;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->partial_width;
            args.staging_dest = staging[dst_idx];
            args.threshold = threshold;
            args.epsilon = plan->epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_staging_from_compute),
                &args, node_count);

            current_src = staging[dst_idx];
            dst_idx = 1 - dst_idx;
        }
    }
}
