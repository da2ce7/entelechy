// network_operations.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

__kernel void forward_pass(
    __local SCALAR_TYPE        *local_mem,
    __global const SCALAR_TYPE *input,
    __global const SCALAR_TYPE *input_mask,
    __global const SCALAR_TYPE *weights,
    __global const SCALAR_TYPE *biases,
    __global SCALAR_TYPE       *hidden,
    __global SCALAR_TYPE       *hidden_mask,
    int                         padded_input_dim,
    int                         padded_hidden_dim) {

    const uint bid  = get_global_id(0); // Batch index
    const uint h    = get_global_id(1); // Hidden block index
    const uint lid  = get_local_id(0);  // SIMD lane in workgroup
    const uint SIMD = SIMD_WIDTH;

    const uint           tile_stride  = get_local_size(0);
    __local SCALAR_TYPE *tile_input   = local_mem;
    __local SCALAR_TYPE *tile_weights = local_mem + tile_stride;

    if (h * SIMD >= padded_hidden_dim) {
        return;
    }

    // Process valid samples using mask
    if (input_mask[bid] < SCALAR_TYPE(0.5)) {
        hidden_mask[bid] = SCALAR_ZERO;
        for (uint l = 0; l < SIMD; l++) {
            hidden[bid * padded_hidden_dim + h * SIMD + l] = SCALAR_ZERO; // Updated indexing
        }
        return;
    }
    hidden_mask[bid] = SCALAR_TYPE(1.0);

    SCALAR_TYPE accum = SCALAR_ZERO;

    // Process input in tiles using padded dimension
    for (uint t = 0; t < padded_input_dim; t += tile_stride) {
        const uint copy_len = min(tile_stride, padded_input_dim - t);

        // Load input tile from global to local memory
        if (lid < copy_len) {
            tile_input[lid] = input[bid * padded_input_dim + t + lid];
        }

        // Coalesced weight load with 3D indexing
        if (lid < copy_len) {
            uint weight_idx   = h * padded_input_dim * SIMD + (t + lid) * SIMD + lid;
            tile_weights[lid] = weights[weight_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

// Vectorized accumulation
#if SIMD_WIDTH >= 4
#pragma unroll
        for (uint l = 0; l < copy_len; l += 4) {
            SCALAR_TYPE4 in_vec = vload4(l, tile_input);
            SCALAR_TYPE4 wt_vec = vload4(l, tile_weights);
            accum += dot(in_vec, wt_vec);
        }
#else
        for (uint l = 0; l < copy_len; l++) {
            accum += tile_input[l] * tile_weights[l];
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result with ReLU
    const uint hidden_idx = bid * padded_hidden_dim + h * SIMD + lid; // padded_hidden_dim used
    accum += biases[h * SIMD + lid];
    hidden[hidden_idx] = scalar_relu(accum);
}

#define SAFE_LOG_MIN (SCALAR_TYPE)(1.0e-7)
#define EXIT_PROB_INDEX(exit, batch, class) ((exit) * padded_batch_size * padded_output_classes + (batch) * padded_output_classes + (class))

__kernel void compute_exit_probabilities(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict hidden,
    __global const SCALAR_TYPE *__restrict hidden_mask,
    __global const SCALAR_TYPE *__restrict exit_weights,
    __global const SCALAR_TYPE *__restrict exit_biases,
    __global SCALAR_TYPE *__restrict exit_probs,
    __global SCALAR_TYPE *__restrict exit_probs_mask,
    __global SCALAR_TYPE *__restrict losses,
    __global SCALAR_TYPE *__restrict losses_mask,
    __global const int *__restrict targets,
    __global const SCALAR_TYPE *__restrict targets_mask,
    __global const SCALAR_TYPE *__restrict temperatures,
    int padded_batch_size,
    int hidden_dim,
    int output_classes,
    int padded_hidden_dim,
    int padded_output_classes,
    int num_exits)
{
    // Precompute common indices
    const uint exit_idx = get_global_id(0);
    const uint batch_idx = get_global_id(1);
    const uint lid = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Early exit checks and initializations
    const SCALAR_TYPE mask_val = hidden_mask[batch_idx] * targets_mask[batch_idx];
    exit_probs_mask[batch_idx] = (mask_val > 0.5f) ? (SCALAR_TYPE)1.0 : SCALAR_ZERO;
    losses_mask[batch_idx] = exit_probs_mask[batch_idx];

    if(mask_val <= 0.5f || exit_idx >= num_exits) {
        for(uint c = lid; c < padded_output_classes; c += lsize) {
            exit_probs[EXIT_PROB_INDEX(exit_idx, batch_idx, c)] = SCALAR_ZERO;
        }
        return;
    }

    const int true_class = targets[batch_idx];
    const uint exit_offset = EXIT_PROB_INDEX(exit_idx, batch_idx, 0);
    const uint weight_base = exit_idx * padded_hidden_dim * padded_output_classes;

    // Shared memory buffers
    __local SCALAR_TYPE max_buffer[WG_SIZE];
    __local SCALAR_TYPE sum_buffer[WG_SIZE];
    __local SCALAR_TYPE loss_buffer[WG_SIZE];
    __local int wg_found;

    // Phase 1: Vectorized logit computation
    const uint k_stride = padded_hidden_dim * SIMD_WIDTH;

// Process vector chunks
#pragma unroll 2
    for (uint vc = lid; vc < vec_classes; vc += lsize) {
        const uint  c_base    = vc * vec_size;
        SCALAR_TYPE logits[4] = {0};

// Load initial biases
#if SIMD_WIDTH >= 4
        const SCALAR_TYPE4 bias_vec = vload4(vc, exit_biases + exit_idx * padded_output_classes);
        logits[0]                   = bias_vec.x;
        logits[1]                   = bias_vec.y;
        logits[2]                   = bias_vec.z;
        logits[3]                   = bias_vec.w;
#else
#pragma unroll
        for (uint i = 0; i < vec_size; i++) {
            logits[i] = exit_biases[exit_idx * padded_output_classes + c_base + i];
        }
#endif

// Accumulate weights
#pragma unroll 4
        for (uint k = 0; k < hidden_dim; k++) {
            const uint weight_idx = weight_base + (c_base / SIMD_WIDTH) * k_stride + k * SIMD_WIDTH + (batch_idx % SIMD_WIDTH);

#if SIMD_WIDTH >= 4
            const SCALAR_TYPE4 weight_vec = vload4(0, &exit_weights[weight_idx]);
            const SCALAR_TYPE  h          = hidden[batch_idx * padded_hidden_dim + k];

#pragma unroll
            for (uint i = 0; i < vec_size; i++) {
                logits[i] += h * weight_vec[i];
            }
#else
            const SCALAR_TYPE h = hidden[batch_idx * padded_hidden_dim + k];
#pragma unroll
            for (uint i = 0; i < vec_size; i++) {
                logits[i] += h * exit_weights[weight_idx + i];
            }
#endif
        }

// Store logits and track max
#pragma unroll
        for (uint i = 0; i < vec_size; i++) {
            const uint        c         = c_base + i;
            const SCALAR_TYPE logit     = logits[i] * native_recip(temperatures[exit_idx]);
            exit_probs[exit_offset + c] = logit;
            max_logit                   = fmax(max_logit, logit);
        }
    }

// Process remainder elements
#pragma unroll 1
    for (uint c = vec_classes * vec_size + lid; c < output_classes; c += lsize) {
        SCALAR_TYPE logit = exit_biases[exit_idx * padded_output_classes + c];

#pragma unroll 4
        for (uint k = 0; k < hidden_dim; k++) {
            const uint weight_idx = weight_base + (c / SIMD_WIDTH) * k_stride + k * SIMD_WIDTH + (batch_idx % SIMD_WIDTH);

            logit += hidden[batch_idx * padded_hidden_dim + k] * exit_weights[weight_idx];
        }

        logit *= native_recip(temperatures[exit_idx]);
        exit_probs[exit_offset + c] = logit;
        max_logit                   = fmax(max_logit, logit);
    }

    // Max reduction
    max_buffer[lid] = max_logit;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            max_buffer[lid] = fmax(max_buffer[lid], max_buffer[lid + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const SCALAR_TYPE global_max = max_buffer[0];

// Phase 2: Exp and sum
#pragma unroll 4
    for (uint c = lid; c < output_classes; c += lsize) {
        SCALAR_TYPE val             = exp(exit_probs[exit_offset + c] - global_max);
        exit_probs[exit_offset + c] = val;
        sum_exp += val;
    }

    // Sum reduction
    sum_buffer[lid] = sum_exp;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum_buffer[lid] += sum_buffer[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const SCALAR_TYPE sum_total = fmax(sum_buffer[0], SAFE_LOG_MIN);

    // Phase 3: Final probabilities and loss
  // Initialize tracking
  if(lid == 0) wg_found = 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  SCALAR_TYPE final_loss = SCALAR_ZERO;

  // Process classes in chunks aligned with WG size
  for(uint c_base = 0; c_base < output_classes; c_base += lsize) {
      const uint c = c_base + lid;

      // Exit early if class processed
      if(c >= output_classes || wg_found > 0) break;

      // Compute probability (mandatory)
      const SCALAR_TYPE prob = exit_probs[exit_offset + c] / sum_total;
      exit_probs[exit_offset + c] = prob;

      // Early exit: first thread to find target claims it
      if(c == (uint)true_class) {
          final_loss = -native_log(fmax(prob, SAFE_LOG_MIN));
          wg_found = 1; // Signal found using local mem

          // Broadcast final loss to buffer
          loss_buffer[lid] = final_loss;
      }
  }

  // Fast reduction if target found
  barrier(CLK_LOCAL_MEM_FENCE);
  if(wg_found) {
      for(uint stride = lsize/2; stride > 0; stride >>= 1) {
          if(lid < stride) loss_buffer[lid] += loss_buffer[lid + stride];
          barrier(CLK_LOCAL_MEM_FENCE);
      }
      if(lid == 0) losses[EXIT_PROB_INDEX(exit_idx, batch_idx, 0)] = loss_buffer[0];
  } else if(lid == 0) {
      losses[EXIT_PROB_INDEX(exit_idx, batch_idx, 0)] = SCALAR_ZERO;
  }

  // Zero-pad remaining classes
  for(uint c = output_classes + lid; c < padded_output_classes; c += lsize) {
      exit_probs[EXIT_PROB_INDEX(exit_idx, batch_idx, c)] = SCALAR_ZERO;
  }
}
