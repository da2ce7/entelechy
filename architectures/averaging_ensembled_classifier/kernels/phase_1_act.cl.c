// phase_1_act.cl.c
//
// Act-phase kernel implementations: Nodes 4, 5, 6, 7.
// Reference specification: kernels.cl.h (ADR-013 designation).

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// ===========================================================================
// Node 4 — forward_pass
// ===========================================================================
// Strategy: Tiled matrix-vector multiplication.  Each work-group computes one
// SIMD-width tile of the output hidden activations for one sample.  Local
// memory broadcasts the input vector slice to all lanes, reducing global
// memory traffic by a factor of SIMD_WIDTH.
//
// Dispatch geometry:
//   global = (batch_chunk_count * SIMD_WIDTH, padded_hidden_count / SIMD_WIDTH)
//   local  = (SIMD_WIDTH, 1)
//
// Each work-group (dim 0) processes one sample; get_local_id(0) selects the
// hidden neuron lane within the SIMD tile.  get_global_id(1) selects the
// hidden-neuron tile index.

__kernel void forward_pass(
    __local COMPUTE_TYPE        *update_buffer_LOCAL_simd_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_weights_shared_simd_major,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_biases_shared,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_hidden_activations,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_hidden_mask,
    uint                         dest_scalar_FLAG_produce_hidden_mask,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_input_count,
    uint                         src_scalar_NATURAL_padded_input_count,
    uint                         src_scalar_NATURAL_padded_hidden_count) {

    // Axiom 1.4 — interface completeness.  These parameters exist for the
    // host's Calculability Proofs and Validation Preconditions; the kernel's
    // tiled loop is driven entirely by padded_input_count.
    (void)src_scalar_NATURAL_input_count;
    (void)src_scalar_NATURAL_total_batch_count;

    // --- 1. Work-Item → Logical Coordinate --------------------------------
    const uint bid     = get_group_id(0);   // sample within the batch chunk
    const uint h_block = get_global_id(1);  // hidden-neuron tile index
    const uint lid     = get_local_id(0);   // SIMD lane

    if (bid >= src_scalar_NATURAL_batch_chunk_count) {
        return;
    }

    const uint effective_bid = src_scalar_NATURAL_batch_chunk_offset + bid;
    const uint hidden_idx    = effective_bid * src_scalar_NATURAL_padded_hidden_count
                             + h_block * SIMD_WIDTH + lid;

    // --- 2. Sample Mask Guard ---------------------------------------------
    // All lanes in the work-group share the same effective_bid, so the mask
    // decision is uniform — no divergent subset can reach the barriers below.
    if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, effective_bid)) {
        store_storage(dest_buffer_GLOBAL_hidden_activations, hidden_idx,
                      COMPUTE_ZERO);
        if (dest_scalar_FLAG_produce_hidden_mask) {
            store_storage(dest_buffer_GLOBAL_hidden_mask, hidden_idx,
                          COMPUTE_ZERO);
        }
        return;
    }

    // --- 3. Tiled Matrix-Vector Multiply -----------------------------------
    const uint TILE_SIZE = SIMD_WIDTH;

    // Partition local memory into two contiguous sub-arrays:
    //   tile_input[TILE_SIZE]              — broadcast tile for the input slice
    //   tile_weights[TILE_SIZE][TILE_SIZE] — weight matrix tile, row-major
    __local COMPUTE_TYPE *tile_input   = update_buffer_LOCAL_simd_tile;
    __local COMPUTE_TYPE *tile_weights = &update_buffer_LOCAL_simd_tile[TILE_SIZE];

    // Seed the accumulator with the bias for this hidden neuron.
    COMPUTE_TYPE accum = load_state(src_buffer_GLOBAL_CONST_biases_shared,
                                    h_block * TILE_SIZE + lid);

    for (uint t = 0; t < src_scalar_NATURAL_padded_input_count;
         t += TILE_SIZE) {

        // -- Cooperative load: input vector tile --
        const uint input_idx =
            effective_bid * src_scalar_NATURAL_padded_input_count + t + lid;
        tile_input[lid] = (t + lid < src_scalar_NATURAL_padded_input_count)
            ? load_storage(src_buffer_GLOBAL_input, input_idx)
            : COMPUTE_ZERO;

        // -- Cooperative load: weight matrix tile --
        // SIMD-major layout: [h_block][t+i][lid] — coalesced across lanes.
        for (uint i = 0; i < TILE_SIZE; ++i) {
            const uint weight_idx =
                h_block * src_scalar_NATURAL_padded_input_count * TILE_SIZE
                + (t + i) * TILE_SIZE + lid;
            tile_weights[i * TILE_SIZE + lid] =
                (t + i < src_scalar_NATURAL_padded_input_count)
                    ? load_state(
                          src_buffer_GLOBAL_CONST_weights_shared_simd_major,
                          weight_idx)
                    : COMPUTE_ZERO;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // -- Dot product from fast local memory --
        // tile_input is loaded once per tile from global memory but read
        // TILE_SIZE times from local memory — the core bandwidth saving.
        for (uint k = 0; k < TILE_SIZE; ++k) {
            accum += tile_input[k] * tile_weights[k * TILE_SIZE + lid];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 4. ReLU Activation & Concurrent Mask Production -------------------
    const COMPUTE_TYPE activation = fmax(accum, COMPUTE_ZERO);
    store_storage(dest_buffer_GLOBAL_hidden_activations, hidden_idx,
                  activation);

    // When enabled, the compute-precision derivative truth is captured
    // before any storage narrowing (see CONCEPT.md §2, mask strategy).
    if (dest_scalar_FLAG_produce_hidden_mask) {
        const COMPUTE_TYPE mask_val =
            (activation > COMPUTE_ZERO) ? COMPUTE_ONE : COMPUTE_ZERO;
        store_storage(dest_buffer_GLOBAL_hidden_mask, hidden_idx, mask_val);
    }
}

// ===========================================================================
// Node 5 — render_logits_chunk
// ===========================================================================
// Strategy: Pure 3D map kernel.  Each work-item computes exactly one logit
// value — an embarrassingly parallel structure that scales linearly with the
// (module × batch × class) volume.
//
// Dispatch geometry:
//   global = (module_chunk_count, batch_chunk_count, class_chunk_count)
//   local  = backend-selected
//
// The sparsity-aware dot product exploits upstream ReLU zeros: for each
// hidden dimension where the mask indicates a zeroed unit, the weight read
// and multiply are elided entirely.

__kernel void render_logits_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,
    uint                         src_scalar_FLAG_use_explicit_hidden_mask,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_weights_module,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_biases_module,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_logits,
    uint                         src_scalar_NATURAL_batch_chunk_offset,
    uint                         src_scalar_NATURAL_batch_chunk_count,
    uint                         src_scalar_NATURAL_module_chunk_offset,
    uint                         src_scalar_NATURAL_module_chunk_count,
    uint                         src_scalar_NATURAL_class_chunk_offset,
    uint                         src_scalar_NATURAL_class_chunk_count,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_hidden_count,
    uint                         src_scalar_NATURAL_padded_hidden_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count) {

    // Axiom 1.4 — interface completeness.  Consumed by host-side
    // Calculability Proofs and Validation Preconditions only.
    (void)src_scalar_NATURAL_total_output_class_count;
    (void)src_scalar_NATURAL_total_modules_count;

    // --- 1. Work-Item → Logical Coordinate --------------------------------
    const uint module_local_idx = get_global_id(0);
    const uint batch_local_idx  = get_global_id(1);
    const uint class_local_idx  = get_global_id(2);

    if (module_local_idx >= src_scalar_NATURAL_module_chunk_count ||
        batch_local_idx  >= src_scalar_NATURAL_batch_chunk_count  ||
        class_local_idx  >= src_scalar_NATURAL_class_chunk_count) {
        return;
    }

    // --- 2. Global Index Calculation --------------------------------------
    const uint module_global_idx =
        src_scalar_NATURAL_module_chunk_offset + module_local_idx;
    const uint batch_global_idx =
        src_scalar_NATURAL_batch_chunk_offset + batch_local_idx;
    const uint class_global_idx =
        src_scalar_NATURAL_class_chunk_offset + class_local_idx;

    // Long casts guard against overflow for large tensor dimensions.
    const long out_idx =
          (long)module_global_idx
              * src_scalar_NATURAL_total_batch_count
              * src_scalar_NATURAL_padded_total_output_class_count
        + (long)batch_global_idx
              * src_scalar_NATURAL_padded_total_output_class_count
        + (long)class_global_idx;

    // --- 3. Sample Mask Guard ---------------------------------------------
    if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, batch_global_idx)) {
        store_storage(dest_buffer_GLOBAL_logits, out_idx, COMPUTE_ZERO);
        return;
    }

    // --- 4. Sparsity-Aware Dot Product ------------------------------------
    // Seed with the module bias.
    COMPUTE_TYPE logit = load_state(
        src_buffer_GLOBAL_CONST_biases_module,
        (long)module_global_idx
            * src_scalar_NATURAL_padded_total_output_class_count
        + class_global_idx);

    const long hidden_base_idx =
        (long)batch_global_idx * src_scalar_NATURAL_padded_hidden_count;

    for (uint h = 0; h < src_scalar_NATURAL_hidden_count; ++h) {
        const COMPUTE_TYPE h_val =
            load_storage(src_buffer_GLOBAL_hidden_activations,
                         hidden_base_idx + h);

        // Mask source: explicit buffer (FLAG=1) preserves compute-precision
        // derivative truth; derived (FLAG=0) recomputes from stored activation.
        const COMPUTE_TYPE h_mask = src_scalar_FLAG_use_explicit_hidden_mask
            ? load_storage(src_buffer_GLOBAL_hidden_mask,
                           hidden_base_idx + h)
            : ((h_val > COMPUTE_ZERO) ? COMPUTE_ONE : COMPUTE_ZERO);

        if (h_mask > (COMPUTE_TYPE)0.5f) {
            const long weight_idx =
                  (long)module_global_idx
                      * src_scalar_NATURAL_padded_hidden_count
                      * src_scalar_NATURAL_padded_total_output_class_count
                + (long)h * src_scalar_NATURAL_padded_total_output_class_count
                + (long)class_global_idx;

            logit += h_val
                   * load_state(src_buffer_GLOBAL_CONST_weights_module,
                                weight_idx);
        }
    }

    store_storage(dest_buffer_GLOBAL_logits, out_idx, logit);
}

// ===========================================================================
// Node 6 — compute_probs_loss_cce_chunk
// ===========================================================================
// Strategy: Fused Softmax/CCE kernel.  2D dispatch where each work-item is a
// self-contained reduction engine for one (module, sample) pair.  The
// work-item serially computes the numerically stable Softmax (max-subtract,
// sum-exp) over the full class dimension, then acts as a Partial Renderer
// for its probability chunk and performs a predicated scatter-write for the
// CCE loss value.
//
// Dispatch geometry:
//   global = (modules_per_chunk, total_batch_count)
//   local  = backend-selected
//
// Loss write predicate: only the tile whose class chunk contains the target
// class writes the loss.  All other tiles skip; the ZERO_REQUIRED
// initialization of dest_buffer_GLOBAL_final_loss guarantees unwritten
// positions are zero.

__kernel void compute_probs_loss_cce_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global const int          *src_buffer_GLOBAL_targets,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_probs,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_final_loss,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.
    (void)src_scalar_NATURAL_total_modules_count;
    (void)src_scalar_NATURAL_total_tile_count;

    // --- 1. Work-Item → Logical Coordinate --------------------------------
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk ||
        batch_idx >= src_scalar_NATURAL_total_batch_count) {
        return;
    }

    // --- 2. Tile Decomposition --------------------------------------------
    const uint module_chunk_idx =
        src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint class_chunk_idx =
        src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx =
        module_chunk_idx * src_scalar_NATURAL_modules_per_chunk
        + module_local_idx;
    const uint class_offset =
        class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;

    const long loss_out_idx =
        (long)module_global_idx * src_scalar_NATURAL_total_batch_count
        + batch_idx;

    const long prob_tile_base =
        (long)src_scalar_NATURAL_flat_tile_index
            * src_scalar_NATURAL_modules_per_chunk
            * src_scalar_NATURAL_total_batch_count
            * src_scalar_NATURAL_classes_per_chunk;

    // --- 3. Sample Mask Guard ---------------------------------------------
    if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, batch_idx)) {
        // Zero partial probabilities for this tile.  The loss buffer is
        // pre-zeroed (ZERO_REQUIRED); no write needed.
        for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk;
             ++c_local) {
            const long prob_write_idx =
                  prob_tile_base
                + (long)module_local_idx
                      * src_scalar_NATURAL_total_batch_count
                      * src_scalar_NATURAL_classes_per_chunk
                + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk
                + c_local;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx,
                          COMPUTE_ZERO);
        }
        return;
    }

    // --- 4. Numerically Stable Softmax (serial reduction) -----------------
    const COMPUTE_TYPE temp_inv =
        COMPUTE_ONE
        / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);

    const long base_logits_idx =
          (long)module_global_idx
              * src_scalar_NATURAL_total_batch_count
              * src_scalar_NATURAL_padded_total_output_class_count
        + (long)batch_idx * src_scalar_NATURAL_padded_total_output_class_count;

    // Pass 1: Find maximum scaled logit for log-sum-exp stability.
    COMPUTE_TYPE max_scaled_logit = -COMPUTE_FP_MAX;
    for (uint c = 0; c < src_scalar_NATURAL_total_output_class_count; ++c) {
        const COMPUTE_TYPE sl =
            load_storage(src_buffer_GLOBAL_logits, base_logits_idx + c)
            * temp_inv;
        max_scaled_logit = fmax(max_scaled_logit, sl);
    }
    // Defensive: collapse to zero if no classes exist (pathological case).
    if (max_scaled_logit == -COMPUTE_FP_MAX) {
        max_scaled_logit = COMPUTE_ZERO;
    }

    // Pass 2: Sum of shifted exponentials.
    COMPUTE_TYPE sum_exp = COMPUTE_ZERO;
    for (uint c = 0; c < src_scalar_NATURAL_total_output_class_count; ++c) {
        sum_exp += MATH_EXP(
            (load_storage(src_buffer_GLOBAL_logits, base_logits_idx + c)
             * temp_inv)
            - max_scaled_logit);
    }
    // sum_exp >= 1.0 always (the max-logit term contributes exp(0) = 1),
    // so this guard is purely defensive.
    const COMPUTE_TYPE inv_sum_exp =
        (sum_exp > (COMPUTE_TYPE)NUMERICAL_STABILITY_EPSILON)
            ? (COMPUTE_ONE / sum_exp)
            : COMPUTE_ZERO;

    // --- 5. Partial Probability Rendering ---------------------------------
    for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk;
         ++c_local) {
        const uint c_global = class_offset + c_local;
        const long prob_write_idx =
              prob_tile_base
            + (long)module_local_idx
                  * src_scalar_NATURAL_total_batch_count
                  * src_scalar_NATURAL_classes_per_chunk
            + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk
            + c_local;

        if (c_global < src_scalar_NATURAL_total_output_class_count) {
            const COMPUTE_TYPE logit =
                load_storage(src_buffer_GLOBAL_logits,
                             base_logits_idx + c_global);
            const COMPUTE_TYPE prob =
                MATH_EXP((logit * temp_inv) - max_scaled_logit) * inv_sum_exp;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx,
                          prob);
        } else {
            // Zero-fill trailing positions beyond total_output_class_count.
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx,
                          COMPUTE_ZERO);
        }
    }

    // --- 6. Loss Scatter-Write (predicated) -------------------------------
    // Only the tile whose class chunk contains the target class writes.
    // The (uint) cast ensures negative target indices (contract violation)
    // fail the range check gracefully rather than producing a wild write.
    const int  true_class_idx  = src_buffer_GLOBAL_targets[batch_idx];
    const uint class_chunk_end =
        class_offset + src_scalar_NATURAL_classes_per_chunk;

    if ((uint)true_class_idx >= class_offset &&
        (uint)true_class_idx <  class_chunk_end) {
        // Recompute the true-class probability at full compute precision,
        // avoiding the storage-narrowing round-trip via partial_probs.
        // The reduction parameters (max_scaled_logit, inv_sum_exp) are
        // already available — no additional global reads beyond the logit.
        const COMPUTE_TYPE logit_tc =
            load_storage(src_buffer_GLOBAL_logits,
                         base_logits_idx + true_class_idx);
        const COMPUTE_TYPE prob_tc =
            MATH_EXP((logit_tc * temp_inv) - max_scaled_logit) * inv_sum_exp;

        // Precision role: compute — direct write, no narrowing.
        dest_buffer_GLOBAL_final_loss[loss_out_idx] =
            -MATH_LOG(fmax(prob_tc,
                           (COMPUTE_TYPE)NUMERICAL_STABILITY_EPSILON));
    }
}

// ===========================================================================
// Node 7 — compute_probs_loss_bce_chunk
// ===========================================================================
// Strategy: 2D dispatch acting as a Dual Partial Renderer.  Each work-item
// handles one (module, sample) pair within its tile, computing a Sigmoid
// probability slice and accumulating the BCE loss over that same slice.
//
// Dispatch geometry:
//   global = (modules_per_chunk, total_batch_count)
//   local  = backend-selected
//
// The Sigmoid uses the numerically stable two-branch form (positive/negative
// logit split to prevent exp() overflow).  The BCE loss uses the logit-domain
// identity: L = max(x,0) - x·y + log(1+exp(-|x|)), which avoids log(0) and
// the epsilon-floor kink inherent in computing log(σ(x)) explicitly.

__kernel void compute_probs_loss_bce_chunk(
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,
    __global const STATE_TYPE   *src_buffer_GLOBAL_CONST_temps,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_targets,
    __global const uint         *src_buffer_GLOBAL_sample_mask,
    __global STORAGE_TYPE       *dest_buffer_GLOBAL_partial_probs,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_partial_loss,
    uint                         src_scalar_NATURAL_flat_tile_index,
    uint                         src_scalar_NATURAL_num_class_chunks,
    uint                         src_scalar_NATURAL_classes_per_chunk,
    uint                         src_scalar_NATURAL_modules_per_chunk,
    uint                         src_scalar_NATURAL_total_batch_count,
    uint                         src_scalar_NATURAL_total_output_class_count,
    uint                         src_scalar_NATURAL_padded_total_output_class_count,
    uint                         src_scalar_NATURAL_total_modules_count,
    uint                         src_scalar_NATURAL_total_tile_count) {

    // Axiom 1.4 — interface completeness.
    (void)src_scalar_NATURAL_total_modules_count;
    (void)src_scalar_NATURAL_total_tile_count;

    // --- 1. Work-Item → Logical Coordinate --------------------------------
    const uint module_local_idx = get_global_id(0);
    const uint batch_idx        = get_global_id(1);

    if (module_local_idx >= src_scalar_NATURAL_modules_per_chunk ||
        batch_idx >= src_scalar_NATURAL_total_batch_count) {
        return;
    }

    // --- 2. Tile Decomposition & Write Addresses --------------------------
    const uint module_chunk_idx =
        src_scalar_NATURAL_flat_tile_index / src_scalar_NATURAL_num_class_chunks;
    const uint class_chunk_idx =
        src_scalar_NATURAL_flat_tile_index % src_scalar_NATURAL_num_class_chunks;
    const uint module_global_idx =
        module_chunk_idx * src_scalar_NATURAL_modules_per_chunk
        + module_local_idx;
    const uint class_offset =
        class_chunk_idx * src_scalar_NATURAL_classes_per_chunk;

    const long loss_tile_base =
        (long)src_scalar_NATURAL_flat_tile_index
            * src_scalar_NATURAL_modules_per_chunk
            * src_scalar_NATURAL_total_batch_count;
    const long loss_write_idx =
        loss_tile_base
        + (long)module_local_idx * src_scalar_NATURAL_total_batch_count
        + batch_idx;

    const long prob_tile_base =
        (long)src_scalar_NATURAL_flat_tile_index
            * src_scalar_NATURAL_modules_per_chunk
            * src_scalar_NATURAL_total_batch_count
            * src_scalar_NATURAL_classes_per_chunk;

    // --- 3. Sample Mask Guard ---------------------------------------------
    if (!load_sample_mask(src_buffer_GLOBAL_sample_mask, batch_idx)) {
        // Precision role: compute — direct write, no narrowing.
        dest_buffer_GLOBAL_partial_loss[loss_write_idx] = COMPUTE_ZERO;

        for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk;
             ++c_local) {
            const long prob_write_idx =
                  prob_tile_base
                + (long)module_local_idx
                      * src_scalar_NATURAL_total_batch_count
                      * src_scalar_NATURAL_classes_per_chunk
                + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk
                + c_local;
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx,
                          COMPUTE_ZERO);
        }
        return;
    }

    // --- 4. Fused Probability & Loss Computation --------------------------
    COMPUTE_TYPE       partial_loss_accum = COMPUTE_ZERO;
    const COMPUTE_TYPE temp_inv =
        COMPUTE_ONE
        / load_state(src_buffer_GLOBAL_CONST_temps, module_global_idx);

    for (uint c_local = 0; c_local < src_scalar_NATURAL_classes_per_chunk;
         ++c_local) {
        const uint c_global = class_offset + c_local;
        const long prob_write_idx =
              prob_tile_base
            + (long)module_local_idx
                  * src_scalar_NATURAL_total_batch_count
                  * src_scalar_NATURAL_classes_per_chunk
            + (long)batch_idx * src_scalar_NATURAL_classes_per_chunk
            + c_local;

        if (c_global < src_scalar_NATURAL_total_output_class_count) {
            const long base_read_idx =
                  (long)module_global_idx
                      * src_scalar_NATURAL_total_batch_count
                      * src_scalar_NATURAL_padded_total_output_class_count
                + (long)batch_idx
                      * src_scalar_NATURAL_padded_total_output_class_count
                + c_global;

            const COMPUTE_TYPE logit =
                load_storage(src_buffer_GLOBAL_logits, base_read_idx);
            const COMPUTE_TYPE scaled_logit = logit * temp_inv;

            // -- Numerically stable Sigmoid (two-branch form) --
            // Positive branch avoids exp(+large); negative branch avoids
            // exp(+large) via the symmetric identity.
            COMPUTE_TYPE prob;
            if (scaled_logit >= COMPUTE_ZERO) {
                const COMPUTE_TYPE e = MATH_EXP(-scaled_logit);
                prob = COMPUTE_ONE / (COMPUTE_ONE + e);
            } else {
                const COMPUTE_TYPE e = MATH_EXP(scaled_logit);
                prob = e / (COMPUTE_ONE + e);
            }

            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx,
                          prob);

            // -- Logit-domain BCE loss --
            // L(x,y) = max(x,0) - x·y + log(1 + exp(-|x|))
            //
            // This identity is mathematically equivalent to
            //   -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
            // but avoids computing log(p) and log(1-p) explicitly, removing
            // the epsilon-floor discontinuity and improving precision for
            // extreme logit magnitudes.  The exp(-|x|) argument is always
            // ≤ 0, so the exponential is bounded in [0, 1] — no overflow.
            const COMPUTE_TYPE target_val = load_storage(
                src_buffer_GLOBAL_targets,
                (long)batch_idx
                    * src_scalar_NATURAL_padded_total_output_class_count
                + c_global);
            const COMPUTE_TYPE abs_sl = fabs(scaled_logit);
            partial_loss_accum += fmax(scaled_logit, COMPUTE_ZERO)
                                - scaled_logit * target_val
                                + MATH_LOG(COMPUTE_ONE + MATH_EXP(-abs_sl));
        } else {
            // Zero-fill trailing positions beyond total_output_class_count.
            store_storage(dest_buffer_GLOBAL_partial_probs, prob_write_idx,
                          COMPUTE_ZERO);
        }
    }

    // --- 5. Partial Loss Write --------------------------------------------
    // Contractually summed by the reduction engine (Node 14) to produce the
    // final per-(module, sample) BCE loss.
    // Precision role: compute — direct write, no narrowing.
    dest_buffer_GLOBAL_partial_loss[loss_write_idx] = partial_loss_accum;
}
