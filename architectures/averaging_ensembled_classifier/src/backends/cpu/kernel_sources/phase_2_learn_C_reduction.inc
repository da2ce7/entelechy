/* phase_2_learn_C_reduction.inc — Precision-parameterized reduction kernels (Nodes 13-16, 20).
 *
 * Included by cpu_kernels.c with STORAGE_T/COMPUTE_T/STATE_T/PRECISION_SUFFIX defined.
 * Algorithmic reference: kernels/phase_2_learn_C_reduction.cl.c
 *
 * Buffer role assignments (ADR-023 §2.5):
 *   - clipped_partial_grad_*, grad_*_permuted, partial_collection, staging → storage
 *   - summed_grad_*, output → compute (COMPUTE_T)
 *
 * Argument naming convention (ADR-013):
 *   Struct field accesses match exactly the full-form @param names defined in
 *   kernels/kernels.cl.h.
 */

/* ================================================================
 * task_gather_permute_grad_hidden_activations (Node 13)
 * ================================================================ */
void PREC_FN(task_gather_permute_grad_hidden_activations)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(GatherPermuteArgs)* a = (PREC_TY(GatherPermuteArgs)*)raw_args;

    const uint bh_total = a->src_scalar_NATURAL_total_batch_count * a->src_scalar_NATURAL_padded_hidden_count;
    const uint bh_flat_idx       = task_index / a->src_scalar_NATURAL_total_modules_count;
    const uint module_global_idx = task_index % a->src_scalar_NATURAL_total_modules_count;

    if (bh_flat_idx >= bh_total || module_global_idx >= a->src_scalar_NATURAL_total_modules_count)
        return;

    const uint batch_idx = bh_flat_idx / a->src_scalar_NATURAL_padded_hidden_count;
    const uint h_idx     = bh_flat_idx % a->src_scalar_NATURAL_padded_hidden_count;

    const uint module_chunk_idx = module_global_idx / a->src_scalar_NATURAL_modules_per_chunk;
    const uint module_local_idx = module_global_idx % a->src_scalar_NATURAL_modules_per_chunk;

    COMPUTE_T accum = (COMPUTE_T)0.0;
    for (uint cc = 0; cc < a->src_scalar_NATURAL_num_class_chunks; cc++) {
        const uint flat_tile_idx = module_chunk_idx * a->src_scalar_NATURAL_num_class_chunks + cc;
        if (flat_tile_idx >= a->src_scalar_NATURAL_total_tile_count) break;

        const size_t tile_size = (size_t)a->src_scalar_NATURAL_modules_per_chunk
            * a->src_scalar_NATURAL_total_batch_count * a->src_scalar_NATURAL_padded_hidden_count;
        const size_t tile_base = (size_t)flat_tile_idx * tile_size;
        const size_t local_off = (size_t)module_local_idx
            * a->src_scalar_NATURAL_total_batch_count * a->src_scalar_NATURAL_padded_hidden_count
            + (size_t)batch_idx * a->src_scalar_NATURAL_padded_hidden_count + h_idx;
        accum += (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
                                     tile_base + local_off);
    }

    const size_t write_idx = (size_t)bh_flat_idx * a->src_scalar_NATURAL_padded_total_modules_count
        + module_global_idx;
    scalar_store_storage(a->dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa, write_idx, accum);
}

/* ================================================================
 * task_stabilize_reduce_grad_hidden_activations (Node 16)
 * ================================================================ */
void PREC_FN(task_stabilize_reduce_grad_hidden_activations)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(StabilizeReduceArgs)* a = (PREC_TY(StabilizeReduceArgs)*)raw_args;

    const uint row_idx = task_index;
    const uint total_rows = a->src_scalar_NATURAL_total_batch_count * a->src_scalar_NATURAL_padded_hidden_count;
    if (row_idx >= total_rows)
        return;

    const size_t row_offset = (size_t)row_idx * a->src_scalar_NATURAL_padded_total_modules_count;

    const uint M = a->src_scalar_NATURAL_total_modules_count;

    if (M <= 1) {
        COMPUTE_T val = (M == 1)
            ? (COMPUTE_T)scalar_load_storage(a->src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa, row_offset)
            : (COMPUTE_T)0.0;
        a->dest_buffer_GLOBAL_summed_grad_hidden_activations[row_idx] = val;
        return;
    }

    /* CPU pre-accumulation: load all M elements (W = M, each "thread" is one task). */
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
            a->dest_buffer_GLOBAL_summed_grad_hidden_activations[row_idx] = (COMPUTE_T)0.0;
            return;
        }
        current = heap_a;
        next    = heap_b;
    }

    /* Load initial values (STORAGE_T -> COMPUTE_T conversion) */
    for (uint i = 0; i < M; i++) {
        current[i] = scalar_load_storage(a->src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,
                                         row_offset + i);
    }

    /* Pre-accumulation clip (host-prescribed threshold). */
    /* For CPU (W = M), host sets t_pre = fp_max (no-op when values are safe). */
    for (uint i = 0; i < M; i++) {
        COMPUTE_T mag = COMPUTE_FABS(current[i]);
        if (mag > a->src_scalar_REAL_clipping_threshold_t_pre) {
            current[i] *= a->src_scalar_REAL_clipping_threshold_t_pre / (mag + a->src_scalar_REAL_epsilon);
        }
    }

    /* Staged reduction using the host-prescribed per-stage thresholds.
     * Binary tree: at stage s, stride = 2^s, each pair separated by stride. */
    uint num_items = M;
    for (uint s = 0; s < a->src_scalar_NATURAL_num_reduction_stages; s++) {
        COMPUTE_T threshold = a->src_buffer_GLOBAL_CONST_clipping_threshold_per_stage[s];

        uint new_count = (num_items + 1) / 2;
        for (uint g = 0; g < new_count; g++) {
            uint left  = g * 2;
            uint right = left + 1;
            COMPUTE_T sum;
            if (right < num_items) {
                sum = current[left] + current[right];
            } else {
                sum = current[left];
            }

            COMPUTE_T norm = COMPUTE_FABS(sum);
            if (norm > threshold) {
                sum *= threshold / (norm + a->src_scalar_REAL_epsilon);
            }
            next[g] = sum;
        }

        COMPUTE_T* tmp = current;
        current = next;
        next = tmp;
        num_items = new_count;
    }

    a->dest_buffer_GLOBAL_summed_grad_hidden_activations[row_idx] = current[0];
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

    /* Diagnostic bypass: negative threshold skips all clipping (ADR-019) */
    if (a->src_scalar_REAL_clipping_threshold_t_j < (COMPUTE_T)0.0)
        return;

    COMPUTE_T sum_sq = (COMPUTE_T)0.0;
    for (uint i = 0; i < a->src_scalar_NATURAL_parameter_count; i++) {
        COMPUTE_T v = a->update_buffer_GLOBAL_intermediate_grad[i];
        sum_sq += v * v;
    }

    COMPUTE_T norm = COMPUTE_SQRT(sum_sq);

    if (norm <= a->src_scalar_REAL_clipping_threshold_t_j)
        return;

    COMPUTE_T scale = a->src_scalar_REAL_clipping_threshold_t_j / (norm + a->src_scalar_REAL_epsilon);

    for (uint i = 0; i < a->src_scalar_NATURAL_parameter_count; i++) {
        a->update_buffer_GLOBAL_intermediate_grad[i] *= scale;
    }
}

/* ================================================================
 * Internal: per-precision ReduceNodeArgs and helpers
 *
 * ADR-026 Compliance: Stage 0 reads STORAGE_T and writes COMPUTE_T.
 * All subsequent stages read/write COMPUTE_T exclusively.
 * ================================================================ */

/* Stage-0 arguments: storage→compute transition (ADR-026 §3) */
typedef struct {
    const STORAGE_T* source;         /* STORAGE_T input (leaf stage) */
    const uint*      offset_list;
    uint             fan_in;
    uint             partial_width;
    COMPUTE_T*       compute_dest;   /* COMPUTE_T output */
    COMPUTE_T        threshold;
    COMPUTE_T        epsilon;
} PREC_TY(ReduceNodeStage0Args);

/* task_reduce_stage0 — storage-entry leaf: reads STORAGE_T, writes COMPUTE_T */
static void PREC_FN(task_reduce_stage0)(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    PREC_TY(ReduceNodeStage0Args)* a = (PREC_TY(ReduceNodeStage0Args)*)raw_args;

    const uint pw = a->partial_width;
    COMPUTE_T* dest = a->compute_dest + (size_t)task_index * pw;

    /* Zero destination (COMPUTE_T) */
    for (uint e = 0; e < pw; e++) {
        dest[e] = (COMPUTE_T)0.0;
    }

    /* Accumulate from STORAGE_T sources (precision boundary conversion) */
    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        if (off == 0xFFFFFFFF) continue;
        const size_t src_base = (size_t)off;
        for (uint e = 0; e < pw; e++) {
            dest[e] += (COMPUTE_T)scalar_load_storage(a->source, src_base + e);
        }
    }

    /* Diagnostic bypass: negative threshold skips all clipping (ADR-019) */
    if (a->threshold >= (COMPUTE_T)0.0) {
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
}

/* Legacy structs kept for reference but no longer used in execute_reduction_tree */
typedef struct {
    const STORAGE_T* source;
    const uint*      offset_list;
    uint             fan_in;
    uint             partial_width;
    COMPUTE_T*       compute_dest;   /* Final stage writes here */
    COMPUTE_T        threshold;
    COMPUTE_T        epsilon;
} PREC_TY(ReduceNodeFinalArgs);

/* ================================================================
 * ADR-026: Precision-Typed Reduction Kernel Variants (Compute-Entry)
 *
 * Used for:
 *   - Interior stages of multi-stage reduction trees
 *   - Leaf stages whose source is natively COMPUTE_T (e.g., BCE loss)
 *
 * NOTE: These are defined here (before execute_reduction_tree) so that
 * execute_reduction_tree can reference them for stages >= 1.
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

    /* Diagnostic bypass: negative threshold skips all clipping (ADR-019) */
    if (a->threshold >= (COMPUTE_T)0.0) {
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

    /* Diagnostic bypass: negative threshold skips all clipping (ADR-019) */
    if (a->threshold >= (COMPUTE_T)0.0) {
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
}

/* NOTE: task_reduce_staging and task_reduce_final removed per ADR-026 compliance.
 * Stage 0 now uses task_reduce_stage0 (storage→compute).
 * Stages ≥1 use task_reduce_staging_from_compute / task_reduce_final_from_compute.
 */

/* ================================================================
 * execute_reduction_tree (ADR-026 Compliant)
 *
 * Stage 0: reads STORAGE_T via task_reduce_stage0, writes COMPUTE_T
 * Stages ≥1: reads/writes COMPUTE_T via _from_compute variants
 * ================================================================ */
void PREC_FN(execute_reduction_tree)(ThreadPool* pool, PREC_TY(ReductionTreePlanStorageEntry)* plan) {
    if (plan->src_scalar_NATURAL_num_stages == 0) return;

    COMPUTE_T* staging[2];
    staging[0] = plan->update_buffer_GLOBAL_intermediate_partial_0;
    staging[1] = plan->update_buffer_GLOBAL_intermediate_partial_1;
    uint dst_idx = 0;

    for (uint s = 0; s < plan->src_scalar_NATURAL_num_stages; s++) {
        uint j = plan->src_scalar_NATURAL_num_stages - 1 - s;
        COMPUTE_T T_policy = plan->src_scalar_REAL_t_algorithmic + plan->src_scalar_REAL_lambda * (COMPUTE_T)(j * j);
        uint K = plan->src_buffer_GLOBAL_CONST_stage_fan_in[s];
        COMPUTE_T T_safety = (K > 0) ? (plan->src_scalar_REAL_fp_max / (COMPUTE_T)K) : plan->src_scalar_REAL_fp_max;
        COMPUTE_T threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint node_count = plan->src_buffer_GLOBAL_CONST_stage_node_counts[s];
        const uint* offsets = plan->src_buffer_GLOBAL_CONST_offset_list_flat + plan->src_buffer_GLOBAL_CONST_stage_list_offset[s];

        if (s == 0) {
            /* Stage 0: storage-entry — reads STORAGE_T, writes COMPUTE_T */
            PREC_TY(ReduceNodeStage0Args) args;
            args.source = plan->src_buffer_GLOBAL_partial_collection;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->src_scalar_NATURAL_partial_width;
            args.compute_dest = (s == plan->src_scalar_NATURAL_num_stages - 1) ? plan->dest_buffer_GLOBAL_summed_partial : staging[dst_idx];
            args.threshold = threshold;
            args.epsilon = plan->src_scalar_REAL_epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_stage0),
                &args, node_count);
        } else if (s == plan->src_scalar_NATURAL_num_stages - 1) {
            /* Final stage (s >= 1): compute-entry, write to output */
            PREC_TY(ReduceNodeFinalFromComputeArgs) args;
            args.source = staging[1 - dst_idx];  /* previous stage's output */
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->src_scalar_NATURAL_partial_width;
            args.compute_dest = plan->dest_buffer_GLOBAL_summed_partial;
            args.threshold = threshold;
            args.epsilon = plan->src_scalar_REAL_epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_final_from_compute),
                &args, node_count);
        } else {
            /* Interior stage (1 <= s < num_stages-1): compute-entry, write to staging */
            PREC_TY(ReduceNodeStagingFromComputeArgs) args;
            args.source = staging[1 - dst_idx];  /* previous stage's output */
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->src_scalar_NATURAL_partial_width;
            args.staging_dest = staging[dst_idx];
            args.threshold = threshold;
            args.epsilon = plan->src_scalar_REAL_epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_staging_from_compute),
                &args, node_count);
        }

        dst_idx = 1 - dst_idx;
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
    if (plan->src_scalar_NATURAL_num_stages == 0) return;

    const COMPUTE_T* current_src = plan->src_buffer_GLOBAL_partial_collection;
    COMPUTE_T* staging[2];
    staging[0] = plan->update_buffer_GLOBAL_intermediate_partial_0;
    staging[1] = plan->update_buffer_GLOBAL_intermediate_partial_1;
    uint dst_idx = 0;

    for (uint s = 0; s < plan->src_scalar_NATURAL_num_stages; s++) {
        uint j = plan->src_scalar_NATURAL_num_stages - 1 - s;
        COMPUTE_T T_policy = plan->src_scalar_REAL_t_algorithmic + plan->src_scalar_REAL_lambda * (COMPUTE_T)(j * j);
        uint K = plan->src_buffer_GLOBAL_CONST_stage_fan_in[s];
        COMPUTE_T T_safety = (K > 0) ? (plan->src_scalar_REAL_fp_max / (COMPUTE_T)K) : plan->src_scalar_REAL_fp_max;
        COMPUTE_T threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint node_count = plan->src_buffer_GLOBAL_CONST_stage_node_counts[s];
        const uint* offsets = plan->src_buffer_GLOBAL_CONST_offset_list_flat + plan->src_buffer_GLOBAL_CONST_stage_list_offset[s];

        if (s == plan->src_scalar_NATURAL_num_stages - 1) {
            /* Final stage: write to COMPUTE_T* output */
            PREC_TY(ReduceNodeFinalFromComputeArgs) args;
            args.source = current_src;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->src_scalar_NATURAL_partial_width;
            args.compute_dest = plan->dest_buffer_GLOBAL_summed_partial;
            args.threshold = threshold;
            args.epsilon = plan->src_scalar_REAL_epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_final_from_compute),
                &args, node_count);
        } else {
            /* Interior stage: write to COMPUTE_T* staging */
            PREC_TY(ReduceNodeStagingFromComputeArgs) args;
            args.source = current_src;
            args.offset_list = offsets;
            args.fan_in = K;
            args.partial_width = plan->src_scalar_NATURAL_partial_width;
            args.staging_dest = staging[dst_idx];
            args.threshold = threshold;
            args.epsilon = plan->src_scalar_REAL_epsilon;

            pool_dispatch_and_wait(pool,
                (void (*)(void*, uint, uint))PREC_FN(task_reduce_staging_from_compute),
                &args, node_count);

            current_src = staging[dst_idx];
            dst_idx = 1 - dst_idx;
        }
    }
}
