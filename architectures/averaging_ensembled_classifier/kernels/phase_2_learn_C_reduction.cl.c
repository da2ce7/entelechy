// phase_2_learn_C_reduction.cl.c
//
// Learn-phase reduction and aggregation kernel implementations:
//   Nodes 14, 15, 16, 20 — and the ADR-019 K-fan-in primitives.
// Reference specification: kernels.cl.h (ADR-013 designation).

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// ===========================================================================
// Nodes 14, 15a, 20a — aggregate_register_reduce   (Tier 1, storage-entry)
// ===========================================================================
// Strategy: Lightweight "map" kernel for low-latency reductions where the
// number of partials (N) is small.  Each work-item computes one element of
// the output tensor by serially accumulating all N partials for that element
// in private registers — no local memory, no synchronization.
//
// Dispatch geometry:
//   global = (partial_width)
//   local  = backend-selected
//
// get_global_id(0) → output element index within the destination partial.

__kernel void aggregate_register_reduce(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    // --- 1. Work-Item → Output Element Mapping ----------------------------
    const uint element_idx = get_global_id(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // --- 2. Register-Based Reduction via Indirection ----------------------
    // The accumulator is a private register, the fastest memory available.
    // The loop iterates through the host-provided indirection map.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    for (uint i = 0; i < src_scalar_NATURAL_partial_offset_list_count; ++i) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        accum += load_storage(src_buffer_GLOBAL_partial_collection,
                              offset + element_idx);
    }

    // --- 3. Final Output --------------------------------------------------
    if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE &&
        src_scalar_NATURAL_partial_offset_list_count > 0) {
        accum /= (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
    }

    // Compute-role output: write directly in COMPUTE_TYPE (no narrowing).
    dest_buffer_GLOBAL_partial[element_idx] = accum;
}

// ===========================================================================
// Nodes 14, 15a, 20a — aggregate_local_reduce       (Tier 2, storage-entry)
// ===========================================================================
// Strategy: "Work-group per element" kernel for high-throughput reductions
// where the number of partials (N) is large.  A full work-group collaborates
// to reduce all N partials for a single output element.  Threads first
// accumulate strided slices into private registers, then perform a fast tree
// reduction in __local memory.
//
// Dispatch geometry:
//   global = (partial_width * work_group_size)
//   local  = (work_group_size)
//
// get_group_id(0)  → output element index.
// get_local_id(0)  → reduction lane within the work-group.
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

__kernel void aggregate_local_reduce(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    // --- 1. Work-Group → Output Element Mapping ---------------------------
    const uint element_idx = get_group_id(0);
    const uint lid         = get_local_id(0);
    const uint lsize       = get_local_size(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // --- 2. Parallel Gather-and-Sum via Indirection -----------------------
    // Each thread accumulates a partial sum for the work-group's assigned
    // element, striding across the offset list.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    for (uint i = lid; i < src_scalar_NATURAL_partial_offset_list_count;
         i += lsize) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        accum += load_storage(src_buffer_GLOBAL_partial_collection,
                              offset + element_idx);
    }

    // --- 3. Intra-Work-Group Tree Reduction -------------------------------
    update_buffer_LOCAL_reduction_tile[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. Final Output --------------------------------------------------
    if (lid == 0) {
        COMPUTE_TYPE result = update_buffer_LOCAL_reduction_tile[0];
        if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE &&
            src_scalar_NATURAL_partial_offset_list_count > 0) {
            result /=
                (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
        }
        // Compute-role output: write directly in COMPUTE_TYPE (no narrowing).
        dest_buffer_GLOBAL_partial[element_idx] = result;
    }
}

// ===========================================================================
// ADR-026: Precision-Typed Reduction Kernel Variants (Compute-Entry)
// ---------------------------------------------------------------------------
// These variants are algorithmically identical to their storage-entry
// counterparts but read COMPUTE_TYPE intermediates directly (no
// load_storage() conversion).  Used for:
//   - Interior stages of multi-stage reduction trees (prior stage outputs)
//   - Leaf stages whose source collection is natively COMPUTE_TYPE
//     (e.g., BCE loss partials from Node 7)
// When STORAGE_TYPE == COMPUTE_TYPE, both variants compile to identical
// machine code.
// ===========================================================================

// ===========================================================================
// Nodes 14, 15a, 20a — aggregate_register_reduce_from_compute    (Tier 1)
// ===========================================================================
// See aggregate_register_reduce for strategy and dispatch geometry.

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

    // Register-based reduction — no precision boundary conversion.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    for (uint i = 0; i < src_scalar_NATURAL_partial_offset_list_count; ++i) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        accum += src_buffer_GLOBAL_partial_collection[offset + element_idx];
    }

    if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE &&
        src_scalar_NATURAL_partial_offset_list_count > 0) {
        accum /= (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
    }

    dest_buffer_GLOBAL_partial[element_idx] = accum;
}

// ===========================================================================
// Nodes 14, 15a, 20a — aggregate_local_reduce_from_compute        (Tier 2)
// ===========================================================================
// See aggregate_local_reduce for strategy and dispatch geometry.

__kernel void aggregate_local_reduce_from_compute(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint         *src_buffer_GLOBAL_CONST_partial_offset_list,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial,
    uint                         src_scalar_NATURAL_partial_offset_list_count,
    uint                         src_scalar_NATURAL_partial_width,
    uint                         src_scalar_FLAG_operation_type) {

    const uint element_idx = get_group_id(0);
    const uint lid         = get_local_id(0);
    const uint lsize       = get_local_size(0);

    if (element_idx >= src_scalar_NATURAL_partial_width) {
        return;
    }

    // Parallel gather — no precision boundary conversion.
    COMPUTE_TYPE accum = COMPUTE_ZERO;
    for (uint i = lid; i < src_scalar_NATURAL_partial_offset_list_count;
         i += lsize) {
        const uint offset = src_buffer_GLOBAL_CONST_partial_offset_list[i];
        accum += src_buffer_GLOBAL_partial_collection[offset + element_idx];
    }

    // Intra-work-group tree reduction.
    update_buffer_LOCAL_reduction_tile[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        COMPUTE_TYPE result = update_buffer_LOCAL_reduction_tile[0];
        if (src_scalar_FLAG_operation_type == AGG_MODE_AVERAGE &&
            src_scalar_NATURAL_partial_offset_list_count > 0) {
            result /=
                (COMPUTE_TYPE)src_scalar_NATURAL_partial_offset_list_count;
        }
        dest_buffer_GLOBAL_partial[element_idx] = result;
    }
}

// ===========================================================================
// Nodes 15b, 20b — clip_intermediate_grad
// ===========================================================================
// Strategy: Two-pass in-place clipping utility.  A single work-group
// computes the L2 norm of a contiguous intermediate gradient buffer (the
// output of a preceding aggregate_* kernel) and conditionally scales it
// in-place.  An early-exit path bypasses the second global memory pass
// when the norm is within the threshold, saving bandwidth.
//
// Dispatch geometry:
//   global = (work_group_size)         [one work-group per buffer]
//   local  = (work_group_size)
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

__kernel void clip_intermediate_grad(
    __local COMPUTE_TYPE  *update_buffer_LOCAL_reduction_tile,
    __global COMPUTE_TYPE *update_buffer_GLOBAL_intermediate_grad,
    COMPUTE_TYPE           src_scalar_REAL_clipping_threshold_t_j,
    COMPUTE_TYPE           src_scalar_REAL_epsilon,
    uint                   src_scalar_NATURAL_parameter_count) {

    // Diagnostic bypass: negative threshold skips all clipping (ADR-019).
    // Uniform branch — all threads exit together before any barrier.
    if (src_scalar_REAL_clipping_threshold_t_j < COMPUTE_ZERO) {
        return;
    }

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- 1. Pass 1: Parallel Sum-of-Squares Reduction ---------------------
    // Each thread accumulates squared values from a strided slice.
    // Compute-role buffer: read directly (no widening conversion).
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;
    for (uint i = lid; i < src_scalar_NATURAL_parameter_count; i += lsize) {
        const COMPUTE_TYPE val = update_buffer_GLOBAL_intermediate_grad[i];
        local_sq_sum += val * val;
    }

    update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 2. Scale Factor Computation & Broadcast --------------------------
    // Thread 0 derives the scaling factor and broadcasts it via local
    // memory slot 0.  Reuse of slot 0 is safe: the final barrier of the
    // reduction loop guarantees all threads have completed their reads
    // before thread 0 overwrites the value.
    if (lid == 0) {
        const COMPUTE_TYPE norm =
            MATH_SQRT(update_buffer_LOCAL_reduction_tile[0]);

        COMPUTE_TYPE scale_factor = COMPUTE_ONE;
        if (norm > src_scalar_REAL_clipping_threshold_t_j) {
            scale_factor = src_scalar_REAL_clipping_threshold_t_j
                         / (norm + src_scalar_REAL_epsilon);
        }
        update_buffer_LOCAL_reduction_tile[0] = scale_factor;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const COMPUTE_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

    // --- 3. Early Exit (no clipping needed) -------------------------------
    // Uniform branch — all threads read the same broadcasted factor.
    // Skipping the second global memory pass saves significant bandwidth.
    if (scale_factor >= COMPUTE_ONE) {
        return;
    }

    // --- 4. Pass 2: Conditional In-Place Scaling --------------------------
    // Compute-role buffer: read/write directly (no precision conversion).
    for (uint i = lid; i < src_scalar_NATURAL_parameter_count; i += lsize) {
        update_buffer_GLOBAL_intermediate_grad[i] *= scale_factor;
    }
}

// ===========================================================================
// Node 16 — stabilize_and_reduce_grad_hidden_activations
// ===========================================================================
// Strategy: Specialized "work-group per row" reduction engine.  Each
// work-group reduces one row of the contiguous SoA buffer produced by the
// upstream Item Synchronization Point (Node 13), applying a host-prescribed
// stabilization schedule.
//
//   Phase 1 — Pre-accumulation: each thread serially accumulates
//     ceil(total_modules_count / lsize) elements from its assigned row,
//     then clips its scalar accumulator to the host-prescribed
//     pre-accumulation threshold.
//   Phase 2 — Staged reduction: upward-sweep binary tree in __local memory
//     with per-stage clip thresholds from the host-prescribed schedule.
//   Phase 3 — Final write: thread 0 writes the scalar result.
//
// Dispatch geometry:
//   global = (total_batch_count * padded_hidden_count * work_group_size)
//   local  = (work_group_size)
//
// get_group_id(0)  → row index (flattened batch × hidden).
// get_local_id(0)  → reduction lane within the work-group.
//
// The upward-sweep binary tree starts at stride 1 (adjacent pairs) and
// doubles each stage, leaving the final result in local[0].  This pattern
// matches the host-prescribed stage count: stage 0 is the leaf (most
// permissive threshold), stage num-1 is the root (tightest threshold).
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

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

    // --- 0. Work-Group → Row Mapping --------------------------------------
    const uint row_idx = get_group_id(0);
    const uint lid     = get_local_id(0);
    const uint lsize   = get_local_size(0);

    const uint total_rows =
        src_scalar_NATURAL_total_batch_count
        * src_scalar_NATURAL_padded_hidden_count;
    if (row_idx >= total_rows) {
        return;
    }

    // --- 1. Phase 1: Pre-accumulation ------------------------------------
    // Each thread accumulates ceil(total_modules_count / lsize) elements
    // from its assigned row, then clips to the host-prescribed threshold.
    // When total_modules_count <= lsize, each thread loads at most one
    // element and the clip is effectively a no-op (the host sets t_pre
    // high enough to avoid attenuation of single values).
    COMPUTE_TYPE thread_accum = COMPUTE_ZERO;
    const long   row_offset   =
        (long)row_idx * src_scalar_NATURAL_padded_total_modules_count;

    for (uint i = lid; i < src_scalar_NATURAL_total_modules_count;
         i += lsize) {
        thread_accum += load_storage(
            src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,
            row_offset + i);
    }

    // Pre-accumulation clip (host-prescribed threshold).
    // For a single scalar, the L2 norm equals the absolute value.
    const COMPUTE_TYPE mag = fabs(thread_accum);
    if (mag > src_scalar_REAL_clipping_threshold_t_pre) {
        thread_accum *= src_scalar_REAL_clipping_threshold_t_pre
                      / (mag + src_scalar_REAL_epsilon);
    }

    update_buffer_LOCAL_reduction_tile[lid] = thread_accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- 2. Phase 2: Staged Reduction with Interleaved Clipping ----------
    // Upward-sweep binary tree: stride doubles each stage, merging
    // adjacent pairs.  After each summation, the per-element result is
    // clipped to the host-prescribed threshold for that stage.
    //
    // Stage indexing (per contract):
    //   stage 0 → leaf (most permissive, farthest from root)
    //   stage num-1 → root (tightest, = T_algorithmic)
    //
    // When num_reduction_stages == 0 (total_modules_count <= 1), this
    // loop is bypassed entirely and thread 0 writes its pre-accumulation
    // result directly.
    for (uint s = 0; s < src_scalar_NATURAL_num_reduction_stages; ++s) {
        const COMPUTE_TYPE threshold =
            src_buffer_GLOBAL_CONST_clipping_threshold_per_stage[s];

        const uint stride = 1u << s;
        const uint pair   = stride << 1;

        // Bitwise test: (lid & (pair - 1)) == 0 selects the "owning"
        // thread of each pair.  Equivalent to lid % pair == 0 for
        // power-of-two pair, but avoids integer division.
        if ((lid & (pair - 1)) == 0 && lid + stride < lsize) {
            COMPUTE_TYPE sum_val =
                update_buffer_LOCAL_reduction_tile[lid]
                + update_buffer_LOCAL_reduction_tile[lid + stride];

            // Per-element scalar clip (L2 norm of a scalar = |scalar|).
            const COMPUTE_TYPE norm_val = fabs(sum_val);
            if (norm_val > threshold) {
                sum_val *= threshold / (norm_val + src_scalar_REAL_epsilon);
            }
            update_buffer_LOCAL_reduction_tile[lid] = sum_val;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 3. Phase 3: Final Write -----------------------------------------
    // Thread 0 holds the fully reduced and stabilized scalar result.
    if (lid == 0) {
        dest_buffer_GLOBAL_summed_grad_hidden_activations[row_idx] =
            update_buffer_LOCAL_reduction_tile[0];
    }
}

// ===========================================================================
// ADR-019: K-Fan-In Reduction Kernel Primitives
// ===========================================================================

// ===========================================================================
// Nodes 14, 15a, 20a — reduce_k_fan_in_and_clip     (storage-entry)
// ===========================================================================
// Strategy: "One work-group per reduction node" fused reduce-and-clip.
// Each work-group processes a single reduction node by:
//
//   Phase 1 — Element-parallel gather: threads stride across the
//     partial_width dimension, accumulating element-wise sums across K
//     input partials.  The sum-of-squares is tracked simultaneously for
//     the L2 norm.  Results are written to the destination buffer.
//   Phase 2 — L2 norm: local-memory tree reduction of per-thread
//     sum-of-squares, yielding a single per-node norm.
//   Phase 3 — Conditional scaling: if the norm exceeds the threshold,
//     all elements are uniformly scaled in the destination buffer.
//
// Fusing summation and clipping into a single kernel eliminates the
// intermediate global memory traffic that separate aggregate + clip
// dispatches would incur.
//
// Dispatch geometry:
//   global = (node_count * work_group_size)
//   local  = (work_group_size)
//
// get_group_id(0)  → reduction node index.
// get_local_id(0)  → element-parallel lane within the work-group.
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

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

    // --- 0. Work-Group → Reduction Node Mapping ---------------------------
    const uint node_id = get_group_id(0);
    const uint lid     = get_local_id(0);
    const uint lsize   = get_local_size(0);

    if (node_id >= src_scalar_NATURAL_node_count) {
        return;
    }

    const uint offset_base = node_id * src_scalar_NATURAL_fan_in;
    const uint dest_base   = node_id * src_scalar_NATURAL_partial_width;

    // --- 1. Phase 1: Gather and Accumulate --------------------------------
    // Each thread strides across partial_width, summing K partials per
    // element.  The sum-of-squares is tracked simultaneously so Phase 2
    // requires no re-read from the destination buffer.
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;

    for (uint elem = lid; elem < src_scalar_NATURAL_partial_width;
         elem += lsize) {
        COMPUTE_TYPE accum = COMPUTE_ZERO;

        for (uint k = 0; k < src_scalar_NATURAL_fan_in; ++k) {
            const uint offset =
                src_buffer_GLOBAL_CONST_offset_list_flat[offset_base + k];
            if (offset != SENTINEL_ABSENT_PARTIAL) {
                accum += load_storage(src_buffer_GLOBAL_partial_collection,
                                      offset + elem);
            }
        }

        // Write the accumulated sum (compute-role output).
        dest_buffer_GLOBAL_stage_partial[dest_base + elem] = accum;
        local_sq_sum += accum * accum;
    }

    // --- 2. Phase 2: Per-Node L2 Norm via Local Memory Reduction ----------
    // Clipping is enabled for non-negative thresholds.  Negative values
    // bypass clipping entirely (diagnostic mode).  Zero clips to zero
    // norm (zeroes all gradients — valid but destructive).
    if (src_scalar_REAL_clipping_threshold_t_j >= COMPUTE_ZERO) {
        update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                update_buffer_LOCAL_reduction_tile[lid] +=
                    update_buffer_LOCAL_reduction_tile[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Scale factor computation & broadcast via local memory slot 0.
        if (lid == 0) {
            const COMPUTE_TYPE norm =
                MATH_SQRT(update_buffer_LOCAL_reduction_tile[0]);

            COMPUTE_TYPE scale_factor = COMPUTE_ONE;
            if (norm > src_scalar_REAL_clipping_threshold_t_j) {
                scale_factor = src_scalar_REAL_clipping_threshold_t_j
                             / (norm + src_scalar_REAL_epsilon);
            }
            update_buffer_LOCAL_reduction_tile[0] = scale_factor;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const COMPUTE_TYPE scale_factor =
            update_buffer_LOCAL_reduction_tile[0];

        // --- 3. Phase 3: Conditional In-Place Scaling ---------------------
        // Each thread re-reads its strided slice from the destination
        // buffer (same addresses written by the same thread in Phase 1)
        // and applies the uniform scale factor.
        if (scale_factor < COMPUTE_ONE) {
            for (uint elem = lid; elem < src_scalar_NATURAL_partial_width;
                 elem += lsize) {
                dest_buffer_GLOBAL_stage_partial[dest_base + elem] *=
                    scale_factor;
            }
        }
    }
}

// ===========================================================================
// Nodes 14, 15a, 20a — reduce_k_fan_in_and_clip_from_compute
//                                         (ADR-026 compute-entry variant)
// ===========================================================================
// Algorithmically identical to reduce_k_fan_in_and_clip but reads
// COMPUTE_TYPE intermediates directly (no load_storage() conversion).
// Used for interior stages of multi-stage reduction trees (where the source
// is a prior stage's COMPUTE_TYPE output) and for leaf stages whose source
// collection is natively COMPUTE_TYPE (e.g., BCE loss partials from Node 7).
// See reduce_k_fan_in_and_clip for strategy, dispatch geometry, and phase
// structure.

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

    // --- 1. Phase 1: Gather and Accumulate (no precision conversion) ------
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;

    for (uint elem = lid; elem < src_scalar_NATURAL_partial_width;
         elem += lsize) {
        COMPUTE_TYPE accum = COMPUTE_ZERO;

        for (uint k = 0; k < src_scalar_NATURAL_fan_in; ++k) {
            const uint offset =
                src_buffer_GLOBAL_CONST_offset_list_flat[offset_base + k];
            if (offset != SENTINEL_ABSENT_PARTIAL) {
                accum +=
                    src_buffer_GLOBAL_partial_collection[offset + elem];
            }
        }

        dest_buffer_GLOBAL_stage_partial[dest_base + elem] = accum;
        local_sq_sum += accum * accum;
    }

    // --- 2. Phase 2: Per-Node L2 Norm via Local Memory Reduction ----------
    if (src_scalar_REAL_clipping_threshold_t_j >= COMPUTE_ZERO) {
        update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                update_buffer_LOCAL_reduction_tile[lid] +=
                    update_buffer_LOCAL_reduction_tile[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            const COMPUTE_TYPE norm =
                MATH_SQRT(update_buffer_LOCAL_reduction_tile[0]);

            COMPUTE_TYPE scale_factor = COMPUTE_ONE;
            if (norm > src_scalar_REAL_clipping_threshold_t_j) {
                scale_factor = src_scalar_REAL_clipping_threshold_t_j
                             / (norm + src_scalar_REAL_epsilon);
            }
            update_buffer_LOCAL_reduction_tile[0] = scale_factor;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const COMPUTE_TYPE scale_factor =
            update_buffer_LOCAL_reduction_tile[0];

        // --- 3. Phase 3: Conditional In-Place Scaling ---------------------
        if (scale_factor < COMPUTE_ONE) {
            for (uint elem = lid; elem < src_scalar_NATURAL_partial_width;
                 elem += lsize) {
                dest_buffer_GLOBAL_stage_partial[dest_base + elem] *=
                    scale_factor;
            }
        }
    }
}
