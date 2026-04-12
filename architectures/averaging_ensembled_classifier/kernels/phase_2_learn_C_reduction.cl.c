// phase_2_learn_C_reduction.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: aggregate_register_reduce (Node 14, 15a, 20a) ---
// Tier 1 Strategy: A lightweight "map" kernel designed for low-latency reductions where
// the number of partials (K) is small. Each work-item is responsible for one element of the
// final output tensor. It performs the entire reduction for that element serially in its
// own private registers, avoiding all local memory and synchronization overhead.
__kernel void aggregate_register_reduce(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    // --- 1. Work-Item to Output Element Mapping ---
    // Each thread is assigned to compute one final value in the destination buffer.
    const uint element_idx = get_global_id(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // --- 2. Register-Based Reduction via Indirection ---
    // The accumulator is a private register, the fastest memory available.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    // This loop iterates through the "map" provided by the host.
    for (uint i = 0; i < src_scalar_NATURAL_partial_offset_list_count; ++i) {
        // Read the memory offset for the i-th partial result.
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        // Gather the scattered data: use the offset to find the start of the i-th partial
        // buffer and add our specific element's value to the accumulator.
        accum += load_storage(src_buffer_GLOBAL_partial_collection, offset + element_idx);
    }

    // --- 3. Final Output Calculation ---
    // Optionally compute the average if requested by the host.
    if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE && src_scalar_NATURAL_partial_offset_list_count > 0) {
        accum /= (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
    }

    // Compute-role output: write directly in COMPUTE_TYPE (no narrowing).
    dest_buffer_GLOBAL_partial[element_idx] = accum;
}

// --- Implementation: aggregate_local_reduce (Node 14, 15a, 20a) ---
// Tier 2 Strategy: A "work-group per element" kernel for high-throughput reductions where
// the number of partials (K) is large. A full work-group collaborates to compute a single
// element of the output tensor. Threads first reduce a slice of the inputs into private
// registers, then perform a final, fast reduction using __local memory.
__kernel void aggregate_local_reduce(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    // --- 1. Work-Group to Output Element Mapping ---
    // The work-group ID determines which output element this entire work-group will compute.
    const uint element_idx = get_group_id(0);
    // The local thread ID determines which slice of the work this thread handles.
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // --- 2. Parallel Gather-and-Sum from Indirection List ---
    // Each thread accumulates a partial sum for the work-group's assigned element.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    // The work-group's threads collaborate by processing the offset list in a strided manner.
    for (uint i = lid; i < src_scalar_NATURAL_partial_offset_list_count; i += lsize) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        accum += load_storage(src_buffer_GLOBAL_partial_collection, offset + element_idx);
    }

    // --- 3. Intra-Work-Group Reduction using Local Memory ---
    // Store this thread's partial sum in fast local memory.
    update_buffer_LOCAL_reduction_tile[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE); // Ensure all threads have written their sums.

    // Standard parallel reduction tree to sum the values in local memory.
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. Final Output Calculation and Write ---
    // The leader thread (lid=0) now holds the final sum for this element.
    if (lid == 0) {
        COMPUTE_TYPE result = update_buffer_LOCAL_reduction_tile[0];
        if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE && src_scalar_NATURAL_partial_offset_list_count > 0) {
            result /= (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
        }
        // Compute-role output: write directly in COMPUTE_TYPE (no narrowing).
        dest_buffer_GLOBAL_partial[element_idx] = result;
    }
}

// --- Implementation: clip_intermediate_grad (Node 15b, 20b) ---
// Strategy: An efficient, in-place, two-pass clipping utility. Pass 1 calculates the
// L2 norm of the entire input buffer using a parallel reduction. Pass 2 conditionally
// scales the buffer in-place. The kernel is optimized with an early-exit path to
// bypass the expensive second pass if no clipping is required, making it a
// lightweight primitive ideal for repeated use within a reduction tree.
__kernel void clip_intermediate_grad(
    __local COMPUTE_TYPE  *update_buffer_LOCAL_reduction_tile,
    __global COMPUTE_TYPE *update_buffer_GLOBAL_intermediate_grad,
    COMPUTE_TYPE           src_scalar_REAL_clipping_threshold_t_j,
    COMPUTE_TYPE           src_scalar_REAL_epsilon,
    uint                   src_scalar_NATURAL_parameter_count) {

    // Diagnostic bypass: negative threshold skips all clipping (ADR-019).
    // This allows inspection of raw gradients without modification.
    if (src_scalar_REAL_clipping_threshold_t_j < COMPUTE_ZERO) {
        return;
    }

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);
    // This kernel assumes a 1D work-group dispatch.

    // --- 1. Pass 1: Parallel Reduction to find Sum of Squares ---
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;
    // Each thread in the work-group sums the squares from a strided slice of the input buffer.
    // Compute-role buffer: read directly (no widening conversion).
    for (uint i = lid; i < src_scalar_NATURAL_parameter_count; i += lsize) {
        COMPUTE_TYPE val = update_buffer_GLOBAL_intermediate_grad[i];
        local_sq_sum += val * val;
    }

    // Perform a standard parallel reduction on the partial sums using fast local memory.
    update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 2. Determine Scaling Factor and Broadcast via Local Memory ---
    // The leader thread (lid=0) computes the final norm and the required scaling factor.
    if (lid == 0) {
        const COMPUTE_TYPE total_sum_sq = update_buffer_LOCAL_reduction_tile[0];
        const COMPUTE_TYPE norm         = MATH_FN sqrt(total_sum_sq);

        COMPUTE_TYPE scale_factor = COMPUTE_ONE;
        // Only trigger scaling if the norm exceeds the threshold for this reduction stage.
        if (norm > src_scalar_REAL_clipping_threshold_t_j) {
            scale_factor = src_scalar_REAL_clipping_threshold_t_j / (norm + src_scalar_REAL_epsilon);
        }
        // Optimization: Broadcast the computed factor to all threads via a single local memory slot.
        update_buffer_LOCAL_reduction_tile[0] = scale_factor;
    }

    // Synchronize to ensure all threads see the single, authoritative scale_factor.
    barrier(CLK_LOCAL_MEM_FENCE);
    const COMPUTE_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

    // --- 3. Performance Optimization: Early Exit ---
    // If the scaling factor is 1.0, no clipping is needed. The entire work-group can
    // exit now, saving the cost of a full global memory read/write cycle.
    if (scale_factor >= COMPUTE_ONE) {
        return;
    }

    // --- 4. Pass 2: Conditional In-Place Scaling ---
    // This second pass only executes if clipping is required.
    // Compute-role buffer: read/write directly (no precision conversion).
    for (uint i = lid; i < src_scalar_NATURAL_parameter_count; i += lsize) {
        update_buffer_GLOBAL_intermediate_grad[i] *= scale_factor;
    }
}

// --- Implementation: stabilize_and_reduce_grad_hidden_activations (Node 16) ---
// Strategy: A specialized reduction engine using a "work-group per row" model.
// The kernel receives a host-prescribed threshold schedule and applies it.
// Phase 1: Pre-accumulation — each thread reduces its assigned elements and clips.
// Phase 2: Staged reduction — applies the per-stage threshold schedule.
__kernel void stabilize_and_reduce_grad_hidden_activations(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_summed_grad_hidden_activations,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_CONST_clipping_threshold_per_stage,
    uint                         src_scalar_NATURAL_num_reduction_stages,
    COMPUTE_TYPE                 src_scalar_REAL_clipping_threshold_t_pre,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_padded_total_modules_count) {

    // --- Phase 0: Setup ---
    const uint row_idx = get_group_id(0);
    const uint lid     = get_local_id(0);
    const uint lsize   = get_local_size(0);

    const uint total_rows = src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count;
    if (row_idx >= total_rows) {
        return;
    }

    // --- Phase 1: Pre-accumulation ---
    // Each thread accumulates ceil(total_modules_count / lsize) elements,
    // then clips to the host-prescribed pre-accumulation threshold.
    COMPUTE_TYPE thread_accumulator = COMPUTE_ZERO;
    const long   row_offset         = (long)row_idx * src_scalar_NATURAL_padded_total_modules_count;
    for (uint i = lid; i < src_scalar_NATURAL_total_modules_count; i += lsize) {
        thread_accumulator += load_storage(src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa, row_offset + i);
    }

    // Pre-accumulation clip (host-prescribed threshold).
    const COMPUTE_TYPE mag = fabs(thread_accumulator);
    if (mag > src_scalar_REAL_clipping_threshold_t_pre) {
        thread_accumulator *= src_scalar_REAL_clipping_threshold_t_pre / (mag + src_scalar_REAL_epsilon);
    }

    update_buffer_LOCAL_reduction_tile[lid] = thread_accumulator;

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 2: Staged Reduction ---
    // Binary tree reduction with per-stage host-prescribed thresholds.
    uint num_items_in_stage = lsize;
    for (uint s = 0; s < src_scalar_NATURAL_num_reduction_stages; ++s) {
        const COMPUTE_TYPE threshold = src_buffer_GLOBAL_CONST_clipping_threshold_per_stage[s];

        // Binary tree: each active thread sums itself and its partner.
        const uint stride = 1u << s;
        const uint pair   = 2u * stride;
        if (lid % pair == 0 && lid + stride < num_items_in_stage) {
            COMPUTE_TYPE sum_vec = update_buffer_LOCAL_reduction_tile[lid]
                                 + update_buffer_LOCAL_reduction_tile[lid + stride];

            COMPUTE_TYPE norm = fabs(sum_vec);
            if (norm > threshold) {
                sum_vec *= threshold / (norm + src_scalar_REAL_epsilon);
            }
            update_buffer_LOCAL_reduction_tile[lid] = sum_vec;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- Phase 3: Final Write ---
    if (lid == 0) {
        dest_buffer_GLOBAL_summed_grad_hidden_activations[row_idx] = update_buffer_LOCAL_reduction_tile[0];
    }
}

// --- Implementation: reduce_k_fan_in_and_clip (ADR-019) ---
// Strategy: One work-group per reduction node.  Threads within a work-group
// collaborate on the element-wise K-partial summation (Phase 1), then perform
// a local-memory parallel reduction for the L2 norm (Phase 2), and finally
// conditionally clip and write (Phase 3).  This fuses the reduce+clip into a
// single dispatch, eliminating intermediate global memory traffic.
__kernel void reduce_k_fan_in_and_clip(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_offset_list_flat,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_stage_partial,
    uint                         src_scalar_NATURAL_fan_in,
    uint                         src_scalar_NATURAL_node_count,
    uint                         src_scalar_NATURAL_partial_width,
    COMPUTE_TYPE                 src_scalar_REAL_clipping_threshold_t_j,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon) {

    // --- 0. Work-Group to Reduction Node Mapping ---
    const uint node_id = get_group_id(0);
    const uint lid     = get_local_id(0);
    const uint lsize   = get_local_size(0);

    if (node_id >= src_scalar_NATURAL_node_count) {
        return;
    }

    const uint offset_base = node_id * src_scalar_NATURAL_fan_in;
    const uint dest_base   = node_id * src_scalar_NATURAL_partial_width;

    // --- Phase 1: Gather and Accumulate (element-parallel with striding) ---
    // Each thread strides across the partial_width dimension, accumulating
    // element-wise sums across all K input partials for this node.
    // We accumulate the sum of squares simultaneously for the L2 norm.
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;

    for (uint elem = lid; elem < src_scalar_NATURAL_partial_width; elem += lsize) {
        COMPUTE_TYPE accum = COMPUTE_ZERO;

        for (uint k = 0; k < src_scalar_NATURAL_fan_in; ++k) {
            const uint offset = src_buffer_GLOBAL_CONST_offset_list_flat[offset_base + k];
            if (offset != SENTINEL_ABSENT_PARTIAL) {
                accum += load_storage(src_buffer_GLOBAL_partial_collection, offset + elem);
            }
        }

        // Store the accumulated sum in the destination (compute-role output),
        // to be potentially scaled in-place during Phase 3.
        dest_buffer_GLOBAL_stage_partial[dest_base + elem] = accum;

        // Accumulate the square for this thread's L2 norm contribution.
        local_sq_sum += accum * accum;
    }

    // --- Phase 2: Per-Node L2 Norm via Local Memory Parallel Reduction ---
    // Clipping is enabled for any non-negative threshold. Negative values
    // (e.g., -1.0) bypass clipping entirely (diagnostic mode). Zero means
    // "clip to zero norm", which zeroes all gradients — valid but destructive.
    if (src_scalar_REAL_clipping_threshold_t_j >= COMPUTE_ZERO) {
        update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // --- Compute scaling factor and broadcast ---
        if (lid == 0) {
            const COMPUTE_TYPE total_sum_sq = update_buffer_LOCAL_reduction_tile[0];
            const COMPUTE_TYPE norm         = MATH_FN sqrt(total_sum_sq);

            COMPUTE_TYPE scale_factor = (COMPUTE_TYPE)1.0f;
            if (norm > src_scalar_REAL_clipping_threshold_t_j) {
                scale_factor = src_scalar_REAL_clipping_threshold_t_j / (norm + src_scalar_REAL_epsilon);
            }
            update_buffer_LOCAL_reduction_tile[0] = scale_factor;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const COMPUTE_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

        // --- Phase 3: Conditional In-Place Scaling ---
        if (scale_factor < (COMPUTE_TYPE)1.0f) {
            for (uint elem = lid; elem < src_scalar_NATURAL_partial_width; elem += lsize) {
                // Direct COMPUTE_TYPE read/write (compute-role output).
                COMPUTE_TYPE val = dest_buffer_GLOBAL_stage_partial[dest_base + elem];
                dest_buffer_GLOBAL_stage_partial[dest_base + elem] = val * scale_factor;
            }
        }
    }
}

// =============================================================================
// ADR-026: Precision-Typed Reduction Kernel Variants (_from_compute)
// =============================================================================
// These variants read from COMPUTE_TYPE buffers directly (no load_storage
// conversion), used for:
//   - Interior stages of multi-stage reduction trees
//   - Leaf stages whose source collection is natively COMPUTE_TYPE (e.g., BCE loss)

// --- Implementation: aggregate_register_reduce_from_compute (ADR-026 §5) ---
// Compute-entry variant: identical algorithm, but reads COMPUTE_TYPE directly.
__kernel void aggregate_register_reduce_from_compute(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    const uint element_idx = get_global_id(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // Register-based reduction — no precision boundary conversion needed.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    for (uint i = 0; i < src_scalar_NATURAL_partial_offset_list_count; ++i) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        // Direct COMPUTE_TYPE read (no load_storage).
        accum += src_buffer_GLOBAL_partial_collection[offset + element_idx];
    }

    if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE && src_scalar_NATURAL_partial_offset_list_count > 0) {
        accum /= (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
    }

    // Direct COMPUTE_TYPE write (no store_storage).
    dest_buffer_GLOBAL_partial[element_idx] = accum;
}

// --- Implementation: aggregate_local_reduce_from_compute (ADR-026 §5) ---
// Compute-entry variant: identical algorithm, but reads COMPUTE_TYPE directly.
__kernel void aggregate_local_reduce_from_compute(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    const uint element_idx = get_group_id(0);
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // Parallel gather — no precision boundary conversion needed.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    for (uint i = lid; i < src_scalar_NATURAL_partial_offset_list_count; i += lsize) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        // Direct COMPUTE_TYPE read (no load_storage).
        accum += src_buffer_GLOBAL_partial_collection[offset + element_idx];
    }

    // Intra-work-group reduction.
    update_buffer_LOCAL_reduction_tile[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        COMPUTE_TYPE result = update_buffer_LOCAL_reduction_tile[0];
        if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE && src_scalar_NATURAL_partial_offset_list_count > 0) {
            result /= (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
        }
        // Direct COMPUTE_TYPE write (no store_storage).
        dest_buffer_GLOBAL_partial[element_idx] = result;
    }
}

// --- Implementation: reduce_k_fan_in_and_clip_from_compute (ADR-026 §4) ---
// Compute-entry variant: identical algorithm, but reads COMPUTE_TYPE directly.
// Used for interior stages of multi-stage trees or for COMPUTE_TYPE source
// collections (e.g., BCE loss partials).
__kernel void reduce_k_fan_in_and_clip_from_compute(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_offset_list_flat,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_stage_partial,
    uint                         src_scalar_NATURAL_fan_in,
    uint                         src_scalar_NATURAL_node_count,
    uint                         src_scalar_NATURAL_partial_width,
    COMPUTE_TYPE                 src_scalar_REAL_clipping_threshold_t_j,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon) {

    const uint node_id = get_group_id(0);
    const uint lid     = get_local_id(0);
    const uint lsize   = get_local_size(0);

    if (node_id >= src_scalar_NATURAL_node_count) {
        return;
    }

    const uint offset_base = node_id * src_scalar_NATURAL_fan_in;
    const uint dest_base   = node_id * src_scalar_NATURAL_partial_width;

    // Phase 1: Gather and accumulate — no precision boundary conversion.
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;

    for (uint elem = lid; elem < src_scalar_NATURAL_partial_width; elem += lsize) {
        COMPUTE_TYPE accum = COMPUTE_ZERO;

        for (uint k = 0; k < src_scalar_NATURAL_fan_in; ++k) {
            const uint offset = src_buffer_GLOBAL_CONST_offset_list_flat[offset_base + k];
            if (offset != SENTINEL_ABSENT_PARTIAL) {
                // Direct COMPUTE_TYPE read (no load_storage).
                accum += src_buffer_GLOBAL_partial_collection[offset + elem];
            }
        }

        // Direct COMPUTE_TYPE write.
        dest_buffer_GLOBAL_stage_partial[dest_base + elem] = accum;

        local_sq_sum += accum * accum;
    }

    // Phase 2: Per-node L2 norm via local memory parallel reduction.
    // Clipping is enabled for any non-negative threshold. Negative values
    // bypass clipping entirely (diagnostic mode).
    if (src_scalar_REAL_clipping_threshold_t_j >= COMPUTE_ZERO) {
        update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            const COMPUTE_TYPE total_sum_sq = update_buffer_LOCAL_reduction_tile[0];
            const COMPUTE_TYPE norm         = MATH_FN sqrt(total_sum_sq);

            COMPUTE_TYPE scale_factor = (COMPUTE_TYPE)1.0f;
            if (norm > src_scalar_REAL_clipping_threshold_t_j) {
                scale_factor = src_scalar_REAL_clipping_threshold_t_j / (norm + src_scalar_REAL_epsilon);
            }
            update_buffer_LOCAL_reduction_tile[0] = scale_factor;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const COMPUTE_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

        // Phase 3: Conditional in-place scaling.
        if (scale_factor < (COMPUTE_TYPE)1.0f) {
            for (uint elem = lid; elem < src_scalar_NATURAL_partial_width; elem += lsize) {
                // Direct COMPUTE_TYPE read/write.
                COMPUTE_TYPE val = dest_buffer_GLOBAL_stage_partial[dest_base + elem];
                dest_buffer_GLOBAL_stage_partial[dest_base + elem] = val * scale_factor;
            }
        }
    }
}
