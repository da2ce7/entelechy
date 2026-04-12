// phase_2_learn_A_production.cl.c
//
// Learn-phase gradient-production kernel implementations: Nodes 8, 9, 10.
// Reference specification: kernels.cl.h (ADR-013 designation).

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// ===========================================================================
// Node 8 — calculate_module_param_grads_chunk
// ===========================================================================
// Strategy: "Work-group per gradient component" reduction.  Each work-group
// is assigned a single scalar in the output gradient tensor (e.g.,
// dL/dW_{m,h,c}).  Threads within the group collaborate via local memory to
// perform a tree-structured reduction (summation) over the batch chunk
// dimension.
//
// Dispatch geometry:
//   global = (modules_per_chunk * batch_reduction_wg_size,
//             padded_hidden_count,
//             classes_per_chunk)
//   local  = (batch_reduction_wg_size, 1, 1)
//
// get_group_id(0)  → module index within the chunk
// get_group_id(1)  → hidden neuron index (padded extent)
// get_group_id(2)  → class index within the class chunk
// get_local_id(0)  → batch reduction lane
//
// Fused weight/bias scheme:  Every work-group computes the weight gradient
// dL/dW (which depends on h_idx via dL/dLogit × h[h_idx]).  The bias
// gradient dL/dB = Σ dL/dLogit is h_idx-independent; only h_idx == 0
// work-groups compute it, avoiding redundant summation across all
// hidden-dimension groups.  Both reductions reuse the same local memory
// tile — safe because h_idx is a group-level coordinate (all threads
// within a group share the same h_idx) and Stage A's final barrier
// precedes Stage B's first write.  See step 6 for the barrier analysis.
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection
// (the KERNEL_ATTR hint targets SIMD_WIDTH, which is a hardware power
// of two on all supported backends).

__kernel void calculate_module_param_grads_chunk(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,
    __global const void         *src_buffer_GLOBAL_targets,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_weights_module,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_biases_module,
    uint                         src_scalar_FLAG_problem_type,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_batch_chunk_index,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_num_batch_chunks,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel
    // derives module_global_idx from the tile decomposition.
    (void)src_scalar_NATURAL_total_modules_count;
    (void)src_scalar_NATURAL_total_tile_count;

    // --- 1. Work-Group → Gradient Component Coordinate --------------------
    const uint module_local_idx = get_group_id(0);
    const uint h_idx            = get_group_id(1);
    const uint class_local_idx  = get_group_id(2);
    const uint lid              = get_local_id(0);
    const uint lsize            = get_local_size(0);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk ||
        h_idx >= src_scalar_NATURAL_padded_hidden_count ||
        class_local_idx >= src_scalar_NATURAL_classes_per_chunk) {
        return;
    }

    // --- 2. Tile Decomposition & Global Coordinates -----------------------
    const uint module_chunk_idx =
        src_scalar_NATURAL_flat_tile_index
        / src_scalar_NATURAL_num_class_chunks;
    const uint class_chunk_idx =
        src_scalar_NATURAL_flat_tile_index
        % src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx =
        module_chunk_idx * src_scalar_NATURAL_modules_per_chunk
        + module_local_idx;
    const uint class_global_idx =
        class_chunk_idx * src_scalar_NATURAL_classes_per_chunk
        + class_local_idx;

    // --- 3. Output Write Addresses ----------------------------------------
    // Weight: 5D layout
    //   (tile, batch_chunk, module, padded_hidden, padded_class).
    // Each tile writes only classes_per_chunk positions within the full
    // padded_total_output_class_count-wide innermost dimension; the
    // ZERO_REQUIRED initialization covers all unwritten positions
    // (class-chunk amplification pattern — see CONCEPT.md §11).
    // Long casts guard against overflow for large tensor dimensions.
    const long weight_slice_stride =
        (long)src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_padded_hidden_count
        * src_scalar_NATURAL_padded_total_output_class_count;

    const long weight_out_idx =
          (long)src_scalar_NATURAL_flat_tile_index
              * src_scalar_NATURAL_num_batch_chunks
              * weight_slice_stride
        + (long)src_scalar_NATURAL_batch_chunk_index
              * weight_slice_stride
        + (long)module_local_idx
              * src_scalar_NATURAL_padded_hidden_count
              * src_scalar_NATURAL_padded_total_output_class_count
        + (long)h_idx
              * src_scalar_NATURAL_padded_total_output_class_count
        + class_global_idx;

    // Bias: 4D layout (tile, batch_chunk, module, padded_class).
    const long bias_slice_stride =
        (long)src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_padded_total_output_class_count;

    const long bias_out_idx =
          (long)src_scalar_NATURAL_flat_tile_index
              * src_scalar_NATURAL_num_batch_chunks
              * bias_slice_stride
        + (long)src_scalar_NATURAL_batch_chunk_index
              * bias_slice_stride
        + (long)module_local_idx
              * src_scalar_NATURAL_padded_total_output_class_count
        + class_global_idx;

    // --- 4. Padding Zero-Preservation (early exit) ------------------------
    // Destination buffers have Initialization Contract: ZERO_REQUIRED — the
    // host guarantees all positions are zero before any dispatch.  At padding
    // positions (h >= hidden_count, or class beyond the logical extent), the
    // gradient is mathematically zero because upstream hidden activations at
    // padding indices are zero (Zero-Propagation Theorem).  The early exit
    // avoids the batch reduction loop; the zero writes by thread 0 are
    // redundant with ZERO_REQUIRED but serve as defense-in-depth.
    if (h_idx >= src_scalar_NATURAL_hidden_count ||
        class_global_idx >= src_scalar_NATURAL_total_output_class_count) {
        if (lid == 0) {
            store_storage(dest_buffer_GLOBAL_partial_grad_weights_module,
                          weight_out_idx, COMPUTE_ZERO);
            // Bias is h_idx-independent — only h_idx == 0 groups write it.
            // For h_idx >= hidden_count (>= 1), this branch is unreachable.
            if (h_idx == 0) {
                store_storage(dest_buffer_GLOBAL_partial_grad_biases_module,
                              bias_out_idx, COMPUTE_ZERO);
            }
        }
        return;
    }

    // --- 5. Parallel Reduction over Batch Chunk ---------------------------
    const COMPUTE_TYPE inv_temp =
        COMPUTE_ONE
        / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);

    // Probability tile layout: (tile, module, total_batch, classes_per_chunk).
    // Stride uses total_batch_count because the prob buffer covers the full
    // batch, even though this dispatch processes only one batch chunk.
    const long prob_tile_base =
        (long)src_scalar_NATURAL_flat_tile_index
            * src_scalar_NATURAL_modules_per_chunk
            * src_scalar_NATURAL_total_batch_count
            * src_scalar_NATURAL_classes_per_chunk;

    COMPUTE_TYPE p_grad_w = COMPUTE_ZERO;
    COMPUTE_TYPE p_grad_b = COMPUTE_ZERO;

    // Each thread sums a strided slice of the batch chunk.
    for (uint b_local = lid;
         b_local < src_scalar_NATURAL_batch_chunk_count;
         b_local += lsize) {

        const uint b = src_scalar_NATURAL_batch_chunk_offset + b_local;

        if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, b)) {
            continue;
        }

        // -- dL/dLogit (upstream gradient from loss) --
        const long prob_read_idx =
            prob_tile_base
            + (long)module_local_idx
                  * src_scalar_NATURAL_total_batch_count
                  * src_scalar_NATURAL_classes_per_chunk
            + (long)b * src_scalar_NATURAL_classes_per_chunk
            + class_local_idx;
        const COMPUTE_TYPE prob =
            load_storage(src_buffer_GLOBAL_partial_probs, prob_read_idx);

        COMPUTE_TYPE d_loss_d_logit;
        if (src_scalar_FLAG_problem_type == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce =
                (const __global int *)src_buffer_GLOBAL_targets;
            d_loss_d_logit =
                (class_global_idx == (uint)targets_cce[b])
                    ? (prob - COMPUTE_ONE)
                    : prob;
        } else { // PROBLEM_TYPE_BCE
            const __global STORAGE_TYPE *targets_bce =
                (const __global STORAGE_TYPE *)src_buffer_GLOBAL_targets;
            d_loss_d_logit =
                prob
                - load_storage(
                      targets_bce,
                      (long)b
                          * src_scalar_NATURAL_padded_total_output_class_count
                      + class_global_idx);
        }

        // Temperature chain rule: dL/dz = dL/dz_s × (1/τ)
        d_loss_d_logit *= inv_temp;

        // -- Accumulate weight gradient: dL/dW = dL/dLogit × h --
        const COMPUTE_TYPE h_val = load_storage(
            src_buffer_GLOBAL_hidden_activations,
            (long)b * src_scalar_NATURAL_padded_hidden_count + h_idx);
        p_grad_w += d_loss_d_logit * h_val;

        // -- Bias gradient (dL/dB = dL/dLogit) is h_idx-independent.
        //    Computed only at h_idx == 0 to avoid redundant work. --
        if (h_idx == 0) {
            p_grad_b += d_loss_d_logit;
        }
    }

    // --- 6. Intra-Workgroup Reduction & Write -----------------------------
    // Two-stage process fuses weight and bias gradient output into a single
    // kernel dispatch.  h_idx is a work-GROUP coordinate (get_group_id(1)),
    // so all threads within a group uniformly enter or skip the bias stage —
    // barriers are reached by the full work-group in both cases.
    //
    // Local memory reuse safety: Stage A writes bias partials, reduces, and
    // stores the final result to global memory.  Stage B then overwrites the
    // same local tile with weight partials.  The last barrier of Stage A's
    // reduction loop guarantees all threads have completed their local memory
    // reads before Stage B's first write — no intervening reads occur.

    // Stage A: Bias gradient — only h_idx == 0 work-groups participate.
    if (h_idx == 0) {
        update_buffer_LOCAL_reduction_tile[lid] = p_grad_b;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                update_buffer_LOCAL_reduction_tile[lid] +=
                    update_buffer_LOCAL_reduction_tile[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            store_storage(dest_buffer_GLOBAL_partial_grad_biases_module,
                          bias_out_idx,
                          update_buffer_LOCAL_reduction_tile[0]);
        }
    }

    // Stage B: Weight gradient — all non-padding work-groups participate.
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_w;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        store_storage(dest_buffer_GLOBAL_partial_grad_weights_module,
                      weight_out_idx,
                      update_buffer_LOCAL_reduction_tile[0]);
    }
}

// ===========================================================================
// Node 9 — backprop_error_to_hidden_chunk
// ===========================================================================
// Strategy: 3D map kernel.  Each work-item computes a single scalar in the
// partial upstream gradient tensor (Grad_H) by performing a serial dot
// product over its assigned class chunk.  This maps a matrix-vector multiply
// to embarrassingly parallel work-items.
//
// Dispatch geometry:
//   global = (modules_per_chunk, total_batch_count, padded_hidden_count)
//   local  = backend-selected
//
// get_global_id(0)  → module index within the chunk
// get_global_id(1)  → batch sample index (full batch — no chunking)
// get_global_id(2)  → hidden neuron index (padded extent)
//
// The contract mandates no batch-chunking: the downstream Item
// Synchronization Point (Node 13) requires a monolithic collection buffer.
// Padding Zero-Establishment writes zeros for h_idx >= hidden_count
// (Initialization Contract: NONE — the kernel is the sole guarantor).

__kernel void backprop_error_to_hidden_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,
    __global const void         *src_buffer_GLOBAL_targets,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_weights_module,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,
    uint                         src_scalar_FLAG_problem_type,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel
    // derives module_global_idx from the tile decomposition.
    (void)src_scalar_NATURAL_total_modules_count;
    (void)src_scalar_NATURAL_total_tile_count;

    // --- 1. Work-Item → Output Coordinate ---------------------------------
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);
    const uint h_idx            = get_global_id(2);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk ||
        batch_idx >= src_scalar_NATURAL_total_batch_count ||
        h_idx >= src_scalar_NATURAL_padded_hidden_count) {
        return;
    }

    // --- 2. Placement Contract: Output Write Address ----------------------
    // AoS layout: (tile, module, batch, padded_hidden).
    const long tile_size =
        (long)src_scalar_NATURAL_modules_per_chunk
        * src_scalar_NATURAL_total_batch_count
        * src_scalar_NATURAL_padded_hidden_count;
    const long out_idx =
          (long)src_scalar_NATURAL_flat_tile_index * tile_size
        + (long)module_local_idx
              * src_scalar_NATURAL_total_batch_count
              * src_scalar_NATURAL_padded_hidden_count
        + (long)batch_idx * src_scalar_NATURAL_padded_hidden_count
        + h_idx;

    // --- 3. Sample Mask Guard ---------------------------------------------
    if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, batch_idx)) {
        store_storage(
            dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,
            out_idx, COMPUTE_ZERO);
        return;
    }

    // --- 4. Padding Zero-Establishment ------------------------------------
    // Initialization Contract: NONE — the kernel is the sole guarantor that
    // padding positions carry zero.  Downstream L2 norms (Node 11) read the
    // full padded extent; incorrect padding would corrupt the norm.
    if (h_idx >= src_scalar_NATURAL_hidden_count) {
        store_storage(
            dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,
            out_idx, COMPUTE_ZERO);
        return;
    }

    // --- 5. Tile Decomposition & Global Coordinates -----------------------
    const uint module_chunk_idx =
        src_scalar_NATURAL_flat_tile_index
        / src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx =
        module_chunk_idx * src_scalar_NATURAL_modules_per_chunk
        + module_local_idx;
    const COMPUTE_TYPE inv_temp =
        COMPUTE_ONE
        / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);

    const uint class_chunk_idx =
        src_scalar_NATURAL_flat_tile_index
        % src_scalar_NATURAL_num_class_chunks;
    const uint class_offset =
        class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;

    // Probability tile layout: (tile, module, total_batch, classes_per_chunk).
    const long prob_tile_base =
        (long)src_scalar_NATURAL_flat_tile_index
            * src_scalar_NATURAL_modules_per_chunk
            * src_scalar_NATURAL_total_batch_count
            * src_scalar_NATURAL_classes_per_chunk;

    // --- 6. Dot Product over Class Chunk ----------------------------------
    // Computes dL/dh = Σ_c [ dL/dLogit_c × W_{m,h,c} ] for this tile's
    // class range.  Partial results from all class-chunk tiles are summed
    // by the downstream gather (Node 13).
    COMPUTE_TYPE grad_h_accum = COMPUTE_ZERO;

    for (uint c_local = 0;
         c_local < src_scalar_NATURAL_classes_per_chunk;
         ++c_local) {

        const uint c_global = class_offset + c_local;
        // Classes within a chunk are contiguous; once beyond the logical
        // extent, all remaining c_local values exceed it as well.
        if (c_global >= src_scalar_NATURAL_total_output_class_count) {
            break;
        }

        // -- Read probability from the tile-local slice --
        const long prob_read_idx =
            prob_tile_base
            + (long)module_local_idx
                  * src_scalar_NATURAL_total_batch_count
                  * src_scalar_NATURAL_classes_per_chunk
            + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk
            + c_local;
        const COMPUTE_TYPE prob =
            load_storage(src_buffer_GLOBAL_partial_probs, prob_read_idx);

        // -- dL/dLogit for this class --
        COMPUTE_TYPE d_loss_d_logit;
        if (src_scalar_FLAG_problem_type == PROBLEM_TYPE_CCE) {
            const __global int *targets_cce =
                (const __global int *)src_buffer_GLOBAL_targets;
            d_loss_d_logit =
                (c_global == (uint)targets_cce[batch_idx])
                    ? (prob - COMPUTE_ONE)
                    : prob;
        } else { // PROBLEM_TYPE_BCE
            const __global STORAGE_TYPE *targets_bce =
                (const __global STORAGE_TYPE *)src_buffer_GLOBAL_targets;
            d_loss_d_logit =
                prob
                - load_storage(
                      targets_bce,
                      (long)batch_idx
                          * src_scalar_NATURAL_padded_total_output_class_count
                      + c_global);
        }

        // Temperature chain rule: dL/dz = dL/dz_s × (1/τ)
        d_loss_d_logit *= inv_temp;

        // Accumulate: dL/dh += dL/dLogit × W[module, h, class]
        const long weight_idx =
              (long)module_global_idx
                  * src_scalar_NATURAL_padded_hidden_count
                  * src_scalar_NATURAL_padded_total_output_class_count
            + (long)h_idx
                  * src_scalar_NATURAL_padded_total_output_class_count
            + c_global;
        grad_h_accum +=
            d_loss_d_logit
            * load_state(src_buffer_GLOBAL_CONST_weights_module, weight_idx);
    }

    // --- 7. Write Result --------------------------------------------------
    store_storage(dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,
                  out_idx, grad_h_accum);
}

// ===========================================================================
// Node 10 — calculate_chunk_temp_gradients
// ===========================================================================
// Strategy: "Work-group per module" hierarchical reduction.  Each work-group
// computes the partial temperature gradient for a single module within its
// tile, performing a three-level reduction:
//
//   Level 1 (batch — strided across threads)
//     Level 2 (class chunk — serial per sample)
//       contribution = dL/dz_s × logit
//   Level 3 (work-group tree reduction in local memory)
//
// The result is one partial per (tile, module) pair.  The final chain rule
// factor (−1/τ²) is applied after the full reduction.
//
// Dispatch geometry:
//   global = (modules_per_chunk * batch_reduction_wg_size)
//   local  = (batch_reduction_wg_size, 1, 1)
//
// get_group_id(0)   → module index within the chunk
// get_local_id(0)   → batch reduction lane
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

__kernel void calculate_chunk_temp_gradients(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,
    __global const void         *src_buffer_GLOBAL_targets,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_temps,
    uint                         src_scalar_FLAG_problem_type,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel
    // derives module_global_idx from the tile decomposition.
    (void)src_scalar_NATURAL_total_modules_count;
    (void)src_scalar_NATURAL_total_tile_count;

    // --- 1. Work-Group → Module Coordinate --------------------------------
    const uint module_local_idx = get_group_id(0);
    const uint lid              = get_local_id(0);
    const uint lsize            = get_local_size(0);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk) {
        return;
    }

    // --- 2. Tile Decomposition & Global Coordinates -----------------------
    const uint module_chunk_idx =
        src_scalar_NATURAL_flat_tile_index
        / src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx =
        module_chunk_idx * src_scalar_NATURAL_modules_per_chunk
        + module_local_idx;
    const uint class_chunk_idx =
        src_scalar_NATURAL_flat_tile_index
        % src_scalar_NATURAL_num_class_chunks;
    const uint class_offset =
        class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;

    // Probability tile layout: (tile, module, total_batch, classes_per_chunk).
    const long prob_tile_base =
        (long)src_scalar_NATURAL_flat_tile_index
            * src_scalar_NATURAL_modules_per_chunk
            * src_scalar_NATURAL_total_batch_count
            * src_scalar_NATURAL_classes_per_chunk;

    // --- 3. Three-Level Hierarchical Reduction ----------------------------
    // The temperature gradient uses the chain rule:
    //   dL/dτ = Σ_{b,c} [ dL/dz_s_c × z_c ] × (−1/τ²)
    //
    // where dL/dz_s is the gradient w.r.t. the scaled logit z_s = z/τ,
    // and z_c is the unscaled logit.  The 1/τ factor from the Logit→z_s
    // chain rule is NOT folded into dL/dz_s here — unlike Nodes 8 and 9
    // which need dL/dz (w.r.t. the raw logit) — because the temperature
    // derivative dz_s/dτ = −z/τ² already accounts for it.

    // Level 1 (batch — strided across threads)
    COMPUTE_TYPE thread_partial = COMPUTE_ZERO;

    for (uint b = lid; b < src_scalar_NATURAL_total_batch_count;
         b += lsize) {

        if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, b)) {
            continue;
        }

        // Level 2 (class chunk — serial per sample)
        //   Accumulates: Σ_c [ dL/dz_s_c × logit_c ]
        COMPUTE_TYPE sample_accum = COMPUTE_ZERO;

        for (uint c_local = 0;
             c_local < src_scalar_NATURAL_classes_per_chunk;
             ++c_local) {

            const uint c_global = class_offset + c_local;
            // Classes within a chunk are contiguous; once beyond the logical
            // extent, all remaining c_local values exceed it as well.
            if (c_global >= src_scalar_NATURAL_total_output_class_count) {
                break;
            }

            // -- Read probability --
            const long prob_read_idx =
                prob_tile_base
                + (long)module_local_idx
                      * src_scalar_NATURAL_total_batch_count
                      * src_scalar_NATURAL_classes_per_chunk
                + (long)b * src_scalar_NATURAL_classes_per_chunk
                + c_local;
            const COMPUTE_TYPE prob =
                load_storage(src_buffer_GLOBAL_partial_probs, prob_read_idx);

            // -- dL/dz_s (gradient w.r.t. scaled logit) --
            COMPUTE_TYPE d_loss_d_scaled_logit;
            if (src_scalar_FLAG_problem_type == PROBLEM_TYPE_CCE) {
                const __global int *targets_cce =
                    (const __global int *)src_buffer_GLOBAL_targets;
                d_loss_d_scaled_logit =
                    (c_global == (uint)targets_cce[b])
                        ? (prob - COMPUTE_ONE)
                        : prob;
            } else { // PROBLEM_TYPE_BCE
                const __global STORAGE_TYPE *targets_bce =
                    (const __global STORAGE_TYPE *)src_buffer_GLOBAL_targets;
                d_loss_d_scaled_logit =
                    prob
                    - load_storage(
                          targets_bce,
                          (long)b
                              * src_scalar_NATURAL_padded_total_output_class_count
                          + c_global);
            }

            // -- Accumulate: dL/dz_s × logit (unscaled) --
            const long logit_idx =
                  (long)module_global_idx
                      * src_scalar_NATURAL_total_batch_count
                      * src_scalar_NATURAL_padded_total_output_class_count
                + (long)b
                      * src_scalar_NATURAL_padded_total_output_class_count
                + c_global;
            sample_accum +=
                d_loss_d_scaled_logit
                * load_storage(src_buffer_GLOBAL_logits, logit_idx);
        }

        thread_partial += sample_accum;
    }

    // --- 4. Level 3: Work-Group Tree Reduction ----------------------------
    update_buffer_LOCAL_reduction_tile[lid] = thread_partial;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 5. Chain Rule Finalization & Write --------------------------------
    // dL/dτ = Σ_{b,c} [ dL/dz_s × logit ] × (−1/τ²)
    //
    // The sum was accumulated in the three-level reduction above.  The
    // final multiplicative factor completes the derivative via the chain
    // rule:  dz_s/dτ = d(logit/τ)/dτ = −logit/τ².
    if (lid == 0) {
        const COMPUTE_TYPE temp =
            load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);
        const COMPUTE_TYPE final_partial_grad =
            update_buffer_LOCAL_reduction_tile[0]
            * (-COMPUTE_ONE / (temp * temp));

        const long out_idx =
            (long)src_scalar_NATURAL_flat_tile_index
                * src_scalar_NATURAL_modules_per_chunk
            + module_local_idx;
        store_storage(dest_buffer_GLOBAL_partial_grad_temps,
                      out_idx, final_partial_grad);
    }
}
