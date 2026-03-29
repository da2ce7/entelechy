/* phase_2_learn_C_reduction.c — CPU Learn-phase reduction kernels (Nodes 13-16, 20).
 *
 * Algorithmic reference: kernels/phase_2_learn_C_reduction.cl.c
 * Node 13: gather_permute_grad_h (AoS → SoA + cross-chunk sum)
 * Node 14/15/20: reduction tree (sum + clip stages)
 * Node 16: stabilize_reduce_grad_h (inline multi-stage reduction)
 * clip_intermediate: in-place L2 norm clip
 */
#include "cpu_kernels.h"

/* ================================================================
 * task_gather_permute_grad_h (Node 13)
 *
 * Scatter-gather from tiled AoS partial layout to contiguous
 * SoA output.  Each task processes one (bh_flat, module_global)
 * element of the destination buffer.
 *
 * task_index encodes a flat (bh_flat_idx, module_global_idx) pair:
 *   bh_flat_idx       = task_index / total_modules_count
 *   module_global_idx = task_index % total_modules_count
 * ================================================================ */
void task_gather_permute_grad_h(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    GatherPermuteArgs* a = (GatherPermuteArgs*)raw_args;

    const uint bh_total = a->total_batch_count * a->padded_hidden_count;
    const uint bh_flat_idx       = task_index / a->total_modules_count;
    const uint module_global_idx = task_index % a->total_modules_count;

    if (bh_flat_idx >= bh_total || module_global_idx >= a->total_modules_count)
        return;

    /* Decompose bh_flat for source addressing */
    const uint batch_idx = bh_flat_idx / a->padded_hidden_count;
    const uint h_idx     = bh_flat_idx % a->padded_hidden_count;

    /* Find module chunk and local index */
    const uint module_chunk_idx = module_global_idx / a->modules_per_chunk;
    const uint module_local_idx = module_global_idx % a->modules_per_chunk;

    /* Sum partial results from all class chunks */
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

    /* Write to SoA output: [bh_flat * padded_total_modules + module] */
    const size_t write_idx = (size_t)bh_flat_idx * a->padded_total_modules_count
        + module_global_idx;
    a->clipped_grad_hidden_activations_permuted_soa[write_idx] = accum;
}

/* ================================================================
 * task_stabilize_reduce_grad_h (Node 16)
 *
 * Per-row reduction of SoA buffer with multi-stage stabilization.
 * Each task processes one row (one (batch, h) pair).
 *
 * task_index = row_idx = batch * padded_hidden + h
 * ================================================================ */
void task_stabilize_reduce_grad_h(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    StabilizeReduceArgs* a = (StabilizeReduceArgs*)raw_args;

    const uint row_idx = task_index;
    const uint total_rows = a->total_batch_count * a->padded_hidden_count;
    if (row_idx >= total_rows)
        return;

    const size_t row_offset = (size_t)row_idx * a->padded_total_modules_count;

    /* Phase 1: initial sum from all modules into scratch buffer */
    /* Allocate workspace for reduction (on stack, bounded by module count) */
    const uint M = a->total_modules_count;

    /* If only 1 module, no reduction needed */
    if (M <= 1) {
        float val = (M == 1)
            ? a->grad_hidden_activations_permuted_soa[row_offset]
            : 0.0f;
        a->summed_grad_hidden_activations[row_idx] = val;
        return;
    }

    /* Pre-compute reduction plan:  ceil(log_K(M)) stages where K = policy_max_k */
    uint K = a->policy_max_k;
    if (K < 2) K = 2; /* Safety: fan-in must be >= 2 */

    uint num_stages = 0;
    uint temp_items = M;
    while (temp_items > 1) {
        temp_items = (temp_items + K - 1) / K;
        num_stages++;
    }

    /* We need two scratch buffers for ping-pong reduction.
     * Use stack allocation for small M; fall back to heap for large M. */
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

    /* Load initial values */
    for (uint i = 0; i < M; i++) {
        current[i] = a->grad_hidden_activations_permuted_soa[row_offset + i];
    }

    /* Phase 2: staged sum-then-clip reduction */
    uint num_items = M;
    for (uint s = 0; s < num_stages; s++) {
        /* Synthesize stage threshold */
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

            /* Clip: for scalar, L2 norm = |sum| */
            float norm = fabsf(sum);
            if (norm > threshold) {
                sum *= threshold / (norm + a->epsilon);
            }
            next[g] = sum;
        }

        /* Swap buffers */
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
 *
 * In-place L2 norm clip of an intermediate gradient buffer.
 * task_index is always 0 (single dispatch per buffer).
 * ================================================================ */
void task_clip_intermediate(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    (void)task_index;
    ClipIntermediateArgs* a = (ClipIntermediateArgs*)raw_args;

    /* Pass 1: sum of squares */
    float sum_sq = 0.0f;
    for (uint i = 0; i < a->parameter_count; i++) {
        float v = a->intermediate_grad[i];
        sum_sq += v * v;
    }

    float norm = sqrtf(sum_sq);
    if (norm <= a->clipping_threshold_t_j)
        return; /* Early exit: no clipping needed */

    float scale = a->clipping_threshold_t_j / (norm + a->epsilon);

    /* Pass 2: in-place scale */
    for (uint i = 0; i < a->parameter_count; i++) {
        a->intermediate_grad[i] *= scale;
    }
}

/* ================================================================
 * Internal: task_reduce_one_node
 *
 * Sum K partials from an offset list into one output.
 * Used by execute_reduction_tree for Nodes 14/15a/20a.
 *
 * Args packed in ReductionTreePlanC context; task_index = node index.
 * ================================================================ */

/* Helper struct for per-stage dispatch within execute_reduction_tree */
typedef struct {
    const float* source;
    const uint*  offset_list;
    uint         fan_in;
    uint         partial_width;
    float*       dest;
    uint         dest_offset; /* base offset for this node's output */
    float        threshold;
    float        epsilon;
} ReduceNodeArgs;

static void task_reduce_and_clip_node(void* raw_args, uint task_index, uint thread_id) {
    (void)thread_id;
    ReduceNodeArgs* a = (ReduceNodeArgs*)raw_args;
    /* Each task processes one element of the partial_width vector
     * for node task_index / partial_width, element task_index % partial_width
     * --- but we dispatch one task per node, reducing all elements per node. */

    /* Sum fan_in partials for all elements */
    const uint pw = a->partial_width;
    float* dest = a->dest + (size_t)task_index * pw;

    /* Zero output */
    for (uint e = 0; e < pw; e++) {
        dest[e] = 0.0f;
    }

    /* Accumulate from each partial in the offset list */
    const uint* offsets = a->offset_list + (size_t)task_index * a->fan_in;
    for (uint k = 0; k < a->fan_in; k++) {
        uint off = offsets[k];
        /* Sentinel check: offset ~0 means "absent partial" */
        if (off == 0xFFFFFFFF) continue;
        const float* src = a->source + (size_t)off;
        for (uint e = 0; e < pw; e++) {
            dest[e] += src[e];
        }
    }

    /* Clip the reduced result */
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
 *
 * Orchestrates the full multi-stage reduction with per-stage
 * sum + clip. Uses ping-pong staging buffers.
 *
 * Plan fields:
 *   partial_collection   — all input partials
 *   offset_lists_flat    — flattened offset list for all stages
 *   stage_offsets_into_list — start index per stage
 *   stage_fan_in         — K per stage
 *   stage_node_counts    — number of reduction nodes per stage
 *   staging_buffer_0/1   — ping-pong intermediate buffers
 *   output               — final result buffer
 *   partial_width        — elements per partial
 *   num_stages           — total stages
 *   t_algorithmic, lambda, fp_max, epsilon — policy params
 * ================================================================ */
void execute_reduction_tree(ThreadPool* pool, ReductionTreePlanC* plan) {
    if (plan->num_stages == 0) return;

    const float* current_src = plan->partial_collection;
    float* staging[2];
    staging[0] = plan->staging_buffer_0;
    staging[1] = plan->staging_buffer_1;
    uint dst_idx = 0;

    for (uint s = 0; s < plan->num_stages; s++) {
        /* Compute threshold for this stage */
        uint j = plan->num_stages - 1 - s;
        float T_policy = plan->t_algorithmic + plan->lambda * (float)(j * j);
        uint K = plan->stage_fan_in[s];
        float T_safety = (K > 0) ? (plan->fp_max / (float)K) : plan->fp_max;
        float threshold = (T_policy < T_safety) ? T_policy : T_safety;

        uint node_count = plan->stage_node_counts[s];

        /* Determine destination: last stage -> output, else -> staging buffer */
        float* dest;
        if (s == plan->num_stages - 1) {
            dest = plan->output;
        } else {
            dest = staging[dst_idx];
        }

        /* Build args for dispatch */
        ReduceNodeArgs args;
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
            (void (*)(void*, uint, uint))task_reduce_and_clip_node,
            &args, node_count);

        /* Advance: current dst becomes next src */
        if (s < plan->num_stages - 1) {
            current_src = dest;
            dst_idx = 1 - dst_idx; /* swap */
        }
    }
}
