// phase_2_learn_D_backprop.cl.c
//
// Learn-phase streaming shared-layer backpropagation kernel implementations:
//   Nodes 17, 18, 19.
// Reference specification: kernels.cl.h (ADR-013 designation).

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// ===========================================================================
// Node 17 — backprop_shared_weights_chunk
// ===========================================================================
// Strategy: "Work-group per gradient component" reduction for the shared
// weight matrix.  Each work-group computes a single scalar dL/dW_{i,j}
// (the partial gradient for one shared weight) by collaboratively reducing
// contributions from all samples in the assigned batch chunk via local
// memory.
//
// The SIMD-major (SoA) write pattern matches the persistent shared weight
// layout consumed by Node 4's forward pass and updated by Node 24's Adam
// update.  This ensures flat-index correspondence between gradient and
// parameter buffers through the layout-agnostic reduction pipeline.
//
// Dispatch geometry:
//   global = (padded_input_count * batch_reduction_wg_size,
//             padded_hidden_count)
//   local  = (batch_reduction_wg_size, 1)
//
// get_group_id(0)  → input dimension index (i)
// get_group_id(1)  → hidden dimension index (j)
// get_local_id(0)  → batch reduction lane
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

__kernel void backprop_shared_weights_chunk(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,
    uint                         src_scalar_FLAG_use_explicit_hidden_mask,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_batch_chunk_index,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_num_batch_chunks,
    uint                         src_scalar_NATURAL_input_count,
    uint                         src_scalar_NATURAL_padded_input_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_final_grad_hidden_activations_total_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel
    // iterates via batch_chunk_offset/count and indexes via padded dimensions.
    (void)src_scalar_NATURAL_total_batch_count;
    (void)src_scalar_NATURAL_num_batch_chunks;
    (void)src_scalar_NATURAL_final_grad_hidden_activations_total_count;

    // --- 1. Work-Group → Gradient Component Coordinate --------------------
    const uint i_idx = get_group_id(0);   // input dimension index
    const uint j_idx = get_group_id(1);   // hidden dimension index
    const uint lid   = get_local_id(0);   // batch reduction lane
    const uint lsize = get_local_size(0);

    if (i_idx >= src_scalar_NATURAL_padded_input_count ||
        j_idx >= src_scalar_NATURAL_padded_hidden_count) {
        return;
    }

    // --- 2. SIMD-Major Write Address Computation --------------------------
    // The destination uses SoA layout: (batch_chunk, h_block, padded_input,
    // SIMD_lane), matching the persistent weight buffer's SIMD-major layout.
    // Long casts guard against overflow for large tensor dimensions.
    const uint hb   = j_idx / SIMD_WIDTH;
    const uint lane = j_idx % SIMD_WIDTH;
    const long chunk_base_offset =
        (long)src_scalar_NATURAL_batch_chunk_index
        * src_scalar_NATURAL_padded_input_count
        * src_scalar_NATURAL_padded_hidden_count;
    const long grad_w_out_idx =
          chunk_base_offset
        + (long)hb * src_scalar_NATURAL_padded_input_count * SIMD_WIDTH
        + (long)i_idx * SIMD_WIDTH
        + lane;

    // --- 3. Padding Zero-Establishment (early exit) -----------------------
    // Initialization Contract: NONE — the kernel is the sole guarantor that
    // padding positions carry zero.  Downstream L2 norms (Node 19) read
    // the full padded extent; incorrect padding would corrupt the norm.
    if (i_idx >= src_scalar_NATURAL_input_count ||
        j_idx >= src_scalar_NATURAL_hidden_count) {
        if (lid == 0) {
            store_storage(
                dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major,
                grad_w_out_idx, COMPUTE_ZERO);
        }
        return;
    }

    // --- 4. Parallel Reduction over Batch Chunk ---------------------------
    // Each thread accumulates dL/dW_{i,j} contributions from a strided
    // slice of the batch chunk.  The chain rule decomposes as:
    //   dL/dW_{i,j} = Σ_b [ dL/dA_j × dA_j/dZ_j × dZ_j/dW_{i,j} ]
    //               = Σ_b [ grad_h[j] × relu_mask[j] × input[i] ]
    COMPUTE_TYPE p_grad_sw = COMPUTE_ZERO;

    for (uint b_local = lid;
         b_local < src_scalar_NATURAL_batch_chunk_count;
         b_local += lsize) {

        const uint b_global = src_scalar_NATURAL_batch_chunk_offset + b_local;

        if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, b_global)) {
            continue;
        }

        const long hidden_offset =
            (long)b_global * src_scalar_NATURAL_padded_hidden_count + j_idx;

        // dL/dA_j — upstream gradient from the reduction engine (Node 16).
        // Compute-role buffer: read directly (no widening conversion).
        const COMPUTE_TYPE grad_h =
            src_buffer_GLOBAL_summed_grad_hidden_activations[hidden_offset];

        // dA_j/dZ_j — ReLU derivative.
        // Mask source: explicit buffer (FLAG=1) preserves compute-precision
        // derivative truth; derived (FLAG=0) recomputes from stored
        // activation.  When FLAG=1, the hidden_activations read is elided —
        // the mask provides the derivative directly, saving one global
        // memory load per sample.
        COMPUTE_TYPE d_activation;
        if (src_scalar_FLAG_use_explicit_hidden_mask) {
            d_activation = load_storage(src_buffer_GLOBAL_hidden_mask,
                                        hidden_offset);
        } else {
            const COMPUTE_TYPE hidden_val = load_storage(
                src_buffer_GLOBAL_hidden_activations, hidden_offset);
            d_activation =
                (hidden_val > COMPUTE_ZERO) ? COMPUTE_ONE : COMPUTE_ZERO;
        }

        // dZ_j/dW_{i,j} = input[i] for the affine transform Z = W*x + b.
        const COMPUTE_TYPE input_val = load_storage(
            src_buffer_GLOBAL_input,
            (long)b_global * src_scalar_NATURAL_padded_input_count + i_idx);

        p_grad_sw += grad_h * d_activation * input_val;
    }

    // --- 5. Intra-Workgroup Tree Reduction & Write ------------------------
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_sw;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        store_storage(
            dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major,
            grad_w_out_idx, update_buffer_LOCAL_reduction_tile[0]);
    }
}

// ===========================================================================
// Node 18 — backprop_shared_biases_chunk
// ===========================================================================
// Strategy: "Work-group per gradient component" reduction for the shared
// bias vector.  Each work-group computes a single scalar dL/dB_j by
// collaboratively reducing contributions from all samples in the assigned
// batch chunk via local memory.  The 1D dispatch is simpler and more
// efficient than a 2D model because the bias gradient does not depend on
// the input dimension.
//
// Dispatch geometry:
//   global = (padded_hidden_count * batch_reduction_wg_size)
//   local  = (batch_reduction_wg_size)
//
// get_group_id(0)  → hidden dimension index (j)
// get_local_id(0)  → batch reduction lane
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

__kernel void backprop_shared_biases_chunk(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,
    uint                         src_scalar_FLAG_use_explicit_hidden_mask,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_grad_biases_shared,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_batch_chunk_index,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_num_batch_chunks,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_final_grad_hidden_activations_total_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel
    // iterates via batch_chunk_offset/count and indexes via padded dimensions.
    (void)src_scalar_NATURAL_total_batch_count;
    (void)src_scalar_NATURAL_num_batch_chunks;
    (void)src_scalar_NATURAL_final_grad_hidden_activations_total_count;

    // --- 1. Work-Group → Gradient Component Coordinate --------------------
    const uint j_idx = get_group_id(0);   // hidden dimension index
    const uint lid   = get_local_id(0);   // batch reduction lane
    const uint lsize = get_local_size(0);

    if (j_idx >= src_scalar_NATURAL_padded_hidden_count) {
        return;
    }

    // --- 2. Write Address Computation -------------------------------------
    // Linear layout: (batch_chunk, padded_hidden).
    const long chunk_base_offset =
        (long)src_scalar_NATURAL_batch_chunk_index
        * src_scalar_NATURAL_padded_hidden_count;
    const long grad_b_out_idx = chunk_base_offset + j_idx;

    // --- 3. Padding Zero-Establishment (early exit) -----------------------
    // Initialization Contract: NONE — the kernel is the sole guarantor that
    // padding positions carry zero.  Downstream L2 norms (Node 19) read
    // the full padded extent; incorrect padding would corrupt the norm.
    if (j_idx >= src_scalar_NATURAL_hidden_count) {
        if (lid == 0) {
            store_storage(dest_buffer_GLOBAL_partial_grad_biases_shared,
                          grad_b_out_idx, COMPUTE_ZERO);
        }
        return;
    }

    // --- 4. Parallel Reduction over Batch Chunk ---------------------------
    // Each thread accumulates dL/dB_j contributions from a strided slice
    // of the batch chunk.  The chain rule decomposes as:
    //   dL/dB_j = Σ_b [ dL/dA_j × dA_j/dZ_j × dZ_j/dB_j ]
    //           = Σ_b [ grad_h[j] × relu_mask[j] × 1 ]
    // (dZ_j/dB_j = 1 for the affine transform Z = W*x + b.)
    COMPUTE_TYPE p_grad_sb = COMPUTE_ZERO;

    for (uint b_local = lid;
         b_local < src_scalar_NATURAL_batch_chunk_count;
         b_local += lsize) {

        const uint b_global = src_scalar_NATURAL_batch_chunk_offset + b_local;

        if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, b_global)) {
            continue;
        }

        const long hidden_offset =
            (long)b_global * src_scalar_NATURAL_padded_hidden_count + j_idx;

        // dL/dA_j — upstream gradient from the reduction engine (Node 16).
        // Compute-role buffer: read directly (no widening conversion).
        const COMPUTE_TYPE grad_h =
            src_buffer_GLOBAL_summed_grad_hidden_activations[hidden_offset];

        // dA_j/dZ_j — ReLU derivative.
        // Same elision strategy as Node 17: when FLAG=1, the explicit mask
        // buffer provides the derivative directly and the hidden_activations
        // read is skipped, saving one global memory load per sample.
        COMPUTE_TYPE d_activation;
        if (src_scalar_FLAG_use_explicit_hidden_mask) {
            d_activation = load_storage(src_buffer_GLOBAL_hidden_mask,
                                        hidden_offset);
        } else {
            const COMPUTE_TYPE hidden_val = load_storage(
                src_buffer_GLOBAL_hidden_activations, hidden_offset);
            d_activation =
                (hidden_val > COMPUTE_ZERO) ? COMPUTE_ONE : COMPUTE_ZERO;
        }

        p_grad_sb += grad_h * d_activation;
    }

    // --- 5. Intra-Workgroup Tree Reduction & Write ------------------------
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_sb;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] +=
                update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        store_storage(dest_buffer_GLOBAL_partial_grad_biases_shared,
                      grad_b_out_idx, update_buffer_LOCAL_reduction_tile[0]);
    }
}

// ===========================================================================
// Node 19 — clip_shared_gradients_chunk
// ===========================================================================
// Strategy: Two-pass virtual-vector algorithm for the streaming shared-layer
// path.  A single work-group processes one batch chunk's complete shared
// gradient set — weight and bias gradients treated as one contiguous
// "virtual vector."
//
//   Pass 1: Parallel sum-of-squares reduction over the virtual vector,
//           yielding the joint L2 norm.
//   Pass 2: Conditional uniform scaling and placement write to the
//           host-specified destination offsets in the collection buffers.
//
// This kernel fulfills the same stabilization role as Node 11 for the
// module path, but operates within the "True Streaming" backpropagation
// model where gradients are clipped immediately per batch chunk.
//
// Dispatch geometry:
//   global = (work_group_size)         [one work-group per chunk]
//   local  = (work_group_size)
//
// The host provides explicit write offsets, making this kernel a "dumb"
// numerical primitive that writes to host-specified memory locations.
//
// The tree reduction requires get_local_size(0) to be a power of two.
// This is guaranteed by the Orchestration tier's dispatch selection.

__kernel void clip_shared_gradients_chunk(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_weights_shared_simd_major,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_biases_shared,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_biases_shared,
    COMPUTE_TYPE                 src_scalar_REAL_clipping_threshold_t_pre,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_weights_parameter_count,
    uint                         src_scalar_NATURAL_biases_parameter_count,
    uint                         dest_scalar_NATURAL_weights_write_offset,
    uint                         dest_scalar_NATURAL_biases_write_offset,
    uint                         src_scalar_NATURAL_num_batch_chunks) {

    // Axiom 1.4 — interface completeness.  This parameter exists for the
    // host's Validation Preconditions (write offset bounds checking); the
    // kernel's iteration is driven by the parameter counts.
    (void)src_scalar_NATURAL_num_batch_chunks;

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // --- 1. Virtual-Vector Geometry ---------------------------------------
    // The two gradient buffers form a "virtual vector" whose total element
    // count drives the strided iteration in both passes.  The segment
    // boundary maps a linear index to the correct physical buffer.
    const uint total_elements =
        src_scalar_NATURAL_weights_parameter_count
        + src_scalar_NATURAL_biases_parameter_count;

    // --- 2. Pass 1: Sum-of-Squares (Joint L2 Norm) -----------------------
    // Each thread accumulates a partial sum-of-squares from a strided slice
    // of the virtual vector.  The conditional chain maps a linear index `i`
    // to the correct physical buffer segment.
    COMPUTE_TYPE local_sq_sum = COMPUTE_ZERO;

    for (uint i = lid; i < total_elements; i += lsize) {
        COMPUTE_TYPE val;
        if (i < src_scalar_NATURAL_weights_parameter_count) {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_weights_shared_simd_major, i);
        } else {
            val = load_storage(
                src_buffer_GLOBAL_partial_grad_biases_shared,
                i - src_scalar_NATURAL_weights_parameter_count);
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
        const COMPUTE_TYPE norm =
            MATH_SQRT(update_buffer_LOCAL_reduction_tile[0]);

        COMPUTE_TYPE scale_factor = COMPUTE_ONE;
        if (src_scalar_REAL_clipping_threshold_t_pre >= COMPUTE_ZERO &&
            norm > src_scalar_REAL_clipping_threshold_t_pre) {
            scale_factor = src_scalar_REAL_clipping_threshold_t_pre
                         / (norm + src_scalar_REAL_epsilon);
        }
        update_buffer_LOCAL_reduction_tile[0] = scale_factor;
    }

    // Synchronize to ensure all threads read thread 0's computed factor.
    barrier(CLK_LOCAL_MEM_FENCE);
    const COMPUTE_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

    // --- 4. Pass 2: Conditional Scaling & Placement Write -----------------
    // Each thread re-reads its strided slice of the virtual vector, applies
    // the single broadcasted scale factor, and writes to the host-specified
    // destination offset in the collection buffer.  The host provides the
    // exact base offset; the kernel adds the element's relative index.
    // Uniform scaling maps zero inputs to zero outputs (Padding
    // Zero-Preservation invariant).
    for (uint i = lid; i < total_elements; i += lsize) {
        if (i < src_scalar_NATURAL_weights_parameter_count) {
            const COMPUTE_TYPE val = load_storage(
                src_buffer_GLOBAL_partial_grad_weights_shared_simd_major, i);
            store_storage(
                dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major,
                dest_scalar_NATURAL_weights_write_offset + i,
                val * scale_factor);
        } else {
            const uint relative_idx =
                i - src_scalar_NATURAL_weights_parameter_count;
            const COMPUTE_TYPE val = load_storage(
                src_buffer_GLOBAL_partial_grad_biases_shared, relative_idx);
            store_storage(
                dest_buffer_GLOBAL_clipped_partial_grad_biases_shared,
                dest_scalar_NATURAL_biases_write_offset + relative_idx,
                val * scale_factor);
        }
    }
}
