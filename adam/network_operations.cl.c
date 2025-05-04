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
