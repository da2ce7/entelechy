// aggregation_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: transpose_grad_h (Node 8) ---
// Strategy: A classic two-phase tiled matrix transpose.
// Phase 1: Threads in a work-group cooperate to read a tile of the source
// matrix into __local memory with coalesced accesses.
// Phase 2: After a barrier, threads write from the __local tile to the
// destination matrix, using transposed indices to perform the transpose.
__kernel void transpose_grad_h(__local SCALAR_TYPE *tile, __global const SCALAR_TYPE *__restrict grad_h_aos_buf, __global SCALAR_TYPE *__restrict grad_h_soa_buf, int num_chunks, int num_items) {

#define TILE_DIM C_TILE_SIZE
// Pad the tile dimension by 1 to prevent local memory bank conflicts on GPUs
// where local memory banks are equal to the warp/wavefront size. This is a
// critical performance optimization.
#define PADDED_TILE_DIM (TILE_DIM + 1)

    // Tile indices based on the work-group's position in the grid
    const int tile_x = get_group_id(0);
    const int tile_y = get_group_id(1);

    // Thread indices within the local work-group (the tile)
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    // --- Phase 1: Coalesced Read from Global (AoS) to Local Memory ---
    const int x_in = tile_x * TILE_DIM + local_x;
    const int y_in = tile_y * TILE_DIM + local_y;

    if (x_in < num_items && y_in < num_chunks) {
        const int in_idx = y_in * num_items + x_in;
        // The padded dimension avoids bank conflicts when we read back transposed.
        tile[local_y * PADDED_TILE_DIM + local_x] = grad_h_aos_buf[in_idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 2: Coalesced Write from Local to Global (SoA) Memory ---
    const int x_out = tile_y * TILE_DIM + local_x;
    const int y_out = tile_x * TILE_DIM + local_y;

    if (x_out < num_chunks && y_out < num_items) {
        const int out_idx       = y_out * num_chunks + x_out;
        grad_h_soa_buf[out_idx] = tile[local_x * PADDED_TILE_DIM + local_y];
    }
}

// --- Implementation: aggregate_identity (Tier 0) ---
// Strategy: A simple memory copy. Each work-item maps to one element of the
// tensor. This provides a near-zero-cost abstraction path for the N=1 case,
// fulfilling the unified dataflow architecture principle.
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_items_to_reduce,
    int item_stride,
    int reduction_mode_flag) {
    const int i = get_global_id(0);
    if (i >= item_stride) {
        return;
    }
    final_output_buf[i] = partial_input_buf[i];
}

// --- Implementation: aggregate_register_reduce (Tier 1) ---
// Strategy: A "map" kernel where each work-item computes a single element of
// the final tensor. The reduction loop is performed entirely in private registers
// (`accum`), making it highly efficient for a small number of items to reduce.
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_items_to_reduce,
    int item_stride,
    int reduction_mode_flag) {
    const int i = get_global_id(0);
    if (i >= item_stride) {
        return;
    }

    SCALAR_TYPE accum = SCALAR_ZERO;
    // For SoA data, iterating through the partial results for a single element `i`
    // requires striding by `item_stride`. This leads to coalesced reads across
    // work-items for the first few iterations.
    for (int j = 0; j < num_items_to_reduce; j++) {
        accum += partial_input_buf[j * item_stride + i];
    }

    if (reduction_mode_flag == AGG_MODE_AVERAGE && num_items_to_reduce > 0) {
        accum /= (SCALAR_TYPE)num_items_to_reduce;
    }

    final_output_buf[i] = accum;
}

// --- Implementation: aggregate_local_reduce (Tier 2/3) ---
// Strategy: A "work-group per element" reduction. Each work-group is responsible
// for computing one element of the final output tensor. Threads within the group
// collaboratively sum their assigned partial results, then perform a final, fast
// reduction using __local memory.
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_items_to_reduce,
    int item_stride,
    int reduction_mode_flag) {

    // A whole work-group computes a single element of the final output tensor.
    const int element_idx = get_group_id(0);
    const int lid         = get_local_id(0);
    const int lsize       = get_local_size(0);

    SCALAR_TYPE accum = SCALAR_ZERO;

    // The base pointer is the start of the data for the element this work-group is responsible for.
    const __global SCALAR_TYPE *base_input_ptr = partial_input_buf + element_idx;

    // Each work-item in the group sums a strided slice of the inputs.
    for (int i = lid; i < num_items_to_reduce; i += lsize) {
        // Accessing the i-th partial result for this specific element. The SoA layout
        // ensures that concurrent reads from different work-groups are coalesced.
        accum += base_input_ptr[i * item_stride];
    }

    local_mem[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Standard parallel reduction within the work-group.
    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_mem[lid] += local_mem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes the final result for this element.
    if (lid == 0) {
        SCALAR_TYPE result = local_mem[0];
        if (reduction_mode_flag == AGG_MODE_AVERAGE && num_items_to_reduce > 0) {
            result /= (SCALAR_TYPE)num_items_to_reduce;
        }
        final_output_buf[element_idx] = result;
    }
}
