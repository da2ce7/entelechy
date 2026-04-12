// phase_2_learn_B_processing.cl.c
//
// Learn-phase gradient-processing kernel implementations: Nodes 11, 13.
// Reference specification: kernels.cl.h (ADR-013 designation).

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// ===========================================================================
// Node 11 — clip_partial_gradients
// ===========================================================================
// Strategy: Two-pass virtual-vector algorithm.  A single work-group processes
// one logical tile's complete gradient set — four physically disjoint buffers
// (weights, biases, temperatures, hidden activations) treated as one
// contiguous "virtual vector."
//
//   Pass 1: Parallel sum-of-squares reduction over the virtual vector,
//           yielding the joint L2 norm.
//   Pass 2: Conditional uniform scaling of all elements by a single factor
//           derived from the norm and the clipping threshold.
//
// The uniform-scaling design is the mechanism for Padding Zero-Preservation:
// zero inputs map to zero outputs for all finite scale factors (see contract
// Behavioral Invariants).  No per-element additive term exists.
//
// Dispatch geometry:
//   global = (work_group_size)         [one work-group per tile]
//   local  = (work_group_size)
//
// The host dispatches this kernel once per tile with a distinct
// `flat_tile_index` scalar.  The single work-group handles the entire tile.
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection
// (the KERNEL_ATTR hint targets SIMD_WIDTH, which is a hardware power
// of two on all supported backends).

__kernel void clip_partial_gradients(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_weights_module,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_biases_module,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_temps,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_hidden_activations_aos,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_CONST_clipping_threshold_per_item,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_weights_module,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_biases_module,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
    uint                         src_scalar_FLAG_use_per_item_norm,
    COMPUTE_TYPE                 src_scalar_REAL_clipping_threshold_t_pre,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs (total_tile_count derivation requires
    // num_class_chunks) and Validation Preconditions; the kernel addresses
    // buffers via flat_tile_index and the padded dimension parameters.
    (void)src_scalar_NATURAL_num_class_chunks;
    (void)src_scalar_NATURAL_classes_per_chunk;
    (void)src_scalar_NATURAL_total_tile_count;

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- 1. Per-Buffer Element Counts & Base Offsets ----------------------
    // The four gradient buffers form a "virtual vector" whose total element
    // count drives the strided iteration in both passes.  Weights and biases
    // use the full PADDED class dimension to match the parameter layout —
    // this ensures flat-index correspondence with the reduction pipeline
    // and optimizer update buffers.
    const uint n_weights =
        src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_padded_hidden_count
        * src_scalar_NATURAL_padded_total_output_class_count;
    const uint n_biases =
        src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_padded_total_output_class_count;
    const uint n_temps =
        src_scalar_NATURAL_modules_per_chunk;
    const uint n_hidden =
        src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_total_batch_count
        * src_scalar_NATURAL_padded_hidden_count;
    const uint total_elements = n_weights + n_biases + n_temps + n_hidden;

    // Buffer-relative base offsets for this tile within each collection.
    // Long casts guard against overflow for large tensor dimensions.
    const long weight_base = (long)src_scalar_NATURAL_flat_tile_index * n_weights;
    const long bias_base   = (long)src_scalar_NATURAL_flat_tile_index * n_biases;
    const long temp_base   = (long)src_scalar_NATURAL_flat_tile_index * n_temps;
    const long hidden_base = (long)src_scalar_NATURAL_flat_tile_index * n_hidden;

    // Segment boundaries for virtual-vector indexing.
    const uint seg_w_end = n_weights;
    const uint seg_b_end = n_weights + n_biases;
    const uint seg_t_end = n_weights + n_biases + n_temps;

    // --- 2. Pass 1: Sum-of-Squares (Joint L2 Norm) -----------------------
    // Each thread accumulates a partial sum-of-squares from a strided slice
    // of the virtual vector.  The conditional chain maps a linear index `i`
    // to the correct physical buffer segment.
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;

    for (uint i = lid; i < total_elements; i += lsize) {
        COMPUTE_TYPE val;
        if (i < seg_w_end) {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_weights_module,
                weight_base + i);
        } else if (i < seg_b_end) {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_biases_module,
                bias_base + (long)(i - seg_w_end));
        } else if (i < seg_t_end) {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_temps,
                temp_base + (long)(i - seg_b_end));
        } else {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_hidden_activations_aos,
                hidden_base + (long)(i - seg_t_end));
        }
        local_sq_sum += val * val;
    }

    // Work-group tree reduction to obtain the total sum-of-squares.
    update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 3. Scale Factor Computation & Broadcast --------------------------
    // Thread 0 derives the scaling factor from the joint norm and the
    // clipping threshold, then broadcasts it to all threads via local
    // memory.  The reuse of slot 0 is safe: the last barrier of the
    // reduction loop guarantees all threads have completed their reads
    // before thread 0 overwrites the value.
    if (lid == 0) {
        const COMPUTE_TYPE threshold =
            (src_scalar_FLAG_use_per_item_norm == 1)
                ? src_buffer_GLOBAL_CONST_clipping_threshold_per_item[
                      src_scalar_NATURAL_flat_tile_index]
                : src_scalar_REAL_clipping_threshold_t_pre;

        const COMPUTE_TYPE norm =
            MATH_SQRT(update_buffer_LOCAL_reduction_tile[0]);

        COMPUTE_TYPE scale_factor = COMPUTE_ONE;
        // Only scale when threshold is non-negative (diagnostic bypass:
        // negative threshold disables clipping) and norm exceeds it.
        if (threshold >= COMPUTE_ZERO && norm > threshold) {
            scale_factor = threshold / (norm + src_scalar_REAL_epsilon);
        }

        // Broadcast via local memory slot 0.
        update_buffer_LOCAL_reduction_tile[0] = scale_factor;
    }

    // Synchronize to ensure all threads read thread 0's computed factor.
    barrier(CLK_LOCAL_MEM_FENCE);
    const COMPUTE_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

    // --- 4. Pass 2: Conditional Uniform Scaling ---------------------------
    // Each thread re-reads its strided slice of the virtual vector, applies
    // the single broadcasted scale factor, and writes to the corresponding
    // destination buffer.  Uniform scaling maps zero inputs to zero outputs
    // (Padding Zero-Preservation invariant).
    for (uint i = lid; i < total_elements; i += lsize) {
        COMPUTE_TYPE val;
        if (i < seg_w_end) {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_weights_module,
                weight_base + i);
            store_storage(
                dest_buffer_GLOBAL_clipped_partial_grad_weights_module,
                weight_base + i, val * scale_factor);
        } else if (i < seg_b_end) {
            const long rel = (long)(i - seg_w_end);
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_biases_module,
                bias_base + rel);
            store_storage(
                dest_buffer_GLOBAL_clipped_partial_grad_biases_module,
                bias_base + rel, val * scale_factor);
        } else if (i < seg_t_end) {
            const long rel = (long)(i - seg_b_end);
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_temps,
                temp_base + rel);
            store_storage(
                dest_buffer_GLOBAL_clipped_partial_grad_temps,
                temp_base + rel, val * scale_factor);
        } else {
            const long rel = (long)(i - seg_t_end);
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_hidden_activations_aos,
                hidden_base + rel);
            store_storage(
                dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
                hidden_base + rel, val * scale_factor);
        }
    }
}

// ===========================================================================
// Node 13 — gather_and_permute_grad_hidden_activations
// ===========================================================================
// Strategy: Gather-sum-permute map kernel.  Each work-item is assigned a
// single scalar in the *destination* SoA buffer.  It gathers the
// corresponding partial results from all class-chunk tiles, sums them (the
// implicit reduction over the class-chunk dimension), and writes the final
// value.  This single-pass approach simultaneously solves the "Transpose
// Illusion" — transforming the chunked AoS tile collection into a dense,
// reduction-ready SoA layout — without an intermediate staging buffer.
//
// This kernel is the canonical Item Synchronization Point: it cannot
// execute until all upstream clipped partials (Node 11) are fully rendered.
//
// Dispatch geometry:
//   global = (total_batch_count * padded_hidden_count, total_modules_count)
//   local  = backend-selected
//
// get_global_id(0)  → flattened (batch, hidden) index into destination rows
// get_global_id(1)  → module index into destination columns
//
// Padding Zero-Preservation:
//   Module dimension: work-items with module_global_idx >= total_modules_count
//     are rejected by the boundary check; the ZERO_REQUIRED initialization
//     is the primary guarantor of zeros at padded module positions.
//   Hidden dimension: source data carries zero at h_idx >= hidden_count
//     (Node 9's Padding Zero-Establishment, preserved by Node 11).  The
//     class-chunk sum of zeros is zero, and the store preserves it.

__kernel void gather_and_permute_grad_hidden_activations(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_padded_total_modules_count,
    uint                         src_scalar_NATURAL_num_module_chunks,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel
    // derives module chunk membership from modules_per_chunk directly.
    (void)src_scalar_NATURAL_hidden_count;
    (void)src_scalar_NATURAL_num_module_chunks;

    // --- 1. Work-Item → Destination Coordinate ----------------------------
    // The 2D dispatch grid maps directly to the SoA output tensor's logical
    // coordinates: bh_flat_idx indexes the (batch × hidden) row dimension,
    // module_global_idx indexes the module column dimension.
    const uint bh_flat_idx       = get_global_id(0);
    const uint module_global_idx = get_global_id(1);

    const uint dest_rows =
        src_scalar_NATURAL_total_batch_count
        * src_scalar_NATURAL_padded_hidden_count;

    if (bh_flat_idx >= dest_rows ||
        module_global_idx >= src_scalar_NATURAL_total_modules_count) {
        return;
    }

    // --- 2. Source Coordinate Decomposition --------------------------------
    // De-flatten the destination row index to obtain the batch and hidden
    // indices needed for the AoS source address calculation.
    const uint batch_idx = bh_flat_idx / src_scalar_NATURAL_padded_hidden_count;
    const uint h_idx     = bh_flat_idx % src_scalar_NATURAL_padded_hidden_count;

    // Determine the module chunk and local offset for the tile lookup.
    const uint module_chunk_idx =
        module_global_idx / src_scalar_NATURAL_modules_per_chunk;
    const uint module_local_idx =
        module_global_idx % src_scalar_NATURAL_modules_per_chunk;

    // Source tile geometry (constant across the class-chunk loop).
    const long tile_size =
        (long)src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_total_batch_count
        * src_scalar_NATURAL_padded_hidden_count;

    // Local offset within a tile for this (module, batch, hidden) triple.
    // AoS layout: (tile, module_local, batch, padded_hidden).
    const long local_offset =
          (long)module_local_idx
              * src_scalar_NATURAL_total_batch_count
              * src_scalar_NATURAL_padded_hidden_count
        + (long)batch_idx * src_scalar_NATURAL_padded_hidden_count
        + h_idx;

    // --- 3. Implicit Reduction over Class Chunks --------------------------
    // All class-chunk tiles that share this module chunk contain a partial
    // contribution to Grad_H[batch, hidden, module].  Summing them here
    // completes the class-dimension reduction that was deferred when the
    // tiles were produced by Nodes 9/11.
    COMPUTE_TYPE accum = COMPUTE_ZERO;

    for (uint class_chunk_idx = 0;
         class_chunk_idx < src_scalar_NATURAL_num_class_chunks;
         ++class_chunk_idx) {

        const uint flat_tile_idx =
            module_chunk_idx * src_scalar_NATURAL_num_class_chunks
            + class_chunk_idx;

        // Defense-in-depth: guard against a malformed tile grid.  For a
        // well-constructed dispatch this check is always true.
        if (flat_tile_idx < src_scalar_NATURAL_total_tile_count) {
            const long read_idx =
                (long)flat_tile_idx * tile_size + local_offset;
            accum += load_storage(
                src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
                read_idx);
        }
    }

    // --- 4. Write to SoA Destination Buffer -------------------------------
    // The SoA layout places all modules for a given (batch, hidden) pair
    // contiguously, enabling coalesced reads by the downstream Node 16
    // reduction kernel.
    const long write_idx =
        (long)bh_flat_idx * src_scalar_NATURAL_padded_total_modules_count
        + module_global_idx;
    store_storage(
        dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,
        write_idx, accum);
}
