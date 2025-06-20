// aggregation_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/**
 * @brief (Node 8) Transposes grad_h (High-Performance Tiled Version).
 *
 * This kernel implements a classic tiled matrix transpose to efficiently transform
 * the partial_grad_h layout from AoS (chunk-major) to SoA (item-major). This
 * ensures the data is contiguous for the aggregation kernels.
 *
 * WORK DISPATCH: 2D grid. The global size must be a multiple of the local size.
 *   - Global Size: (ceil(num_items / TILE_DIM) * TILE_DIM, ceil(num_chunks / TILE_DIM) * TILE_DIM)
 *   - Local Size:  (TILE_DIM, TILE_DIM)
 *   - The host must use C_TILE_SIZE for TILE_DIM.
 */
__kernel void transpose_grad_h(
    // NOTE: C_TILE_SIZE is the tile dimension. For best performance, add 1 to the
    // padded dimension to prevent local memory bank conflicts.
    __local SCALAR_TYPE *tile,                             // [MEMORY size: C_TILE_SIZE * (C_TILE_SIZE + 1)]
    __global const SCALAR_TYPE *__restrict grad_h_aos_buf, // [INPUT  shape: (num_chunks, num_items)]
    __global SCALAR_TYPE *__restrict grad_h_soa_buf,       // [OUTPUT shape: (num_items,  num_chunks)]
    int num_chunks,
    int num_items) {

#define TILE_DIM C_TILE_SIZE
#define PADDED_TILE_DIM (TILE_DIM + 1)

    // Tile indices based on the work-group's position in the grid
    const int tile_x = get_group_id(0);
    const int tile_y = get_group_id(1);

    // Thread indices within the local work-group (the tile)
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    // --- Phase 1: Coalesced Read from Global (AoS) to Local Memory ---
    // `x` corresponds to `item_idx`, `y` to `chunk_idx`.
    // We are reading from a matrix of size: (num_chunks, num_items)
    const int x_in = tile_x * TILE_DIM + local_x;
    const int y_in = tile_y * TILE_DIM + local_y;

    if (x_in < num_items && y_in < num_chunks) {
        const int in_idx = y_in * num_items + x_in;
        // The padded dimension avoids bank conflicts when we read back transposed.
        tile[local_y * PADDED_TILE_DIM + local_x] = grad_h_aos_buf[in_idx];
    }

    // Synchronize to ensure the entire tile is loaded before proceeding.
    barrier(CLK_LOCAL_MEM_FENCE);

    // --- Phase 2: Coalesced Write from Local to Global (SoA) Memory ---
    // `x` corresponds to `chunk_idx`, `y` to `item_idx`.
    // We are writing to a matrix of size: (num_items, num_chunks)
    // The source tile is read with swapped local indices to perform the transpose.
    const int x_out = tile_y * TILE_DIM + local_x;
    const int y_out = tile_x * TILE_DIM + local_y;

    if (x_out < num_chunks && y_out < num_items) {
        const int out_idx       = y_out * num_chunks + x_out;
        grad_h_soa_buf[out_idx] = tile[local_x * PADDED_TILE_DIM + local_y];
    }
}

// --- Phase 5 & 7: Generic Tiered Aggregation Engine ---

/**
 * @brief Generic, tiered aggregation engine interface.
 *
 * All aggregation kernels share an identical signature. for true "plug-and-play"
 * swapping by the host orchestrator. The host selects the appropriate kernel implementation
 * based on the number of items to reduce, but calls it through a single, consistent interface.
 *
 * @param local_mem             Local memory buffer, used by reduction-heavy kernels.
 * @param partial_input_buf     The buffer of partial results to be aggregated. The layout
 *                              is assumed to be SoA (Structure of Arrays).
 * @param final_output_buf      The destination buffer for the single, aggregated result.
 * @param num_items_to_reduce   The number of partial results to aggregate (e.g., num_chunks).
 * @param item_stride           The total size of a single output tensor (e.g., grad_h size).
 *                              For SoA data, this is the distance between consecutive elements
 *                              of the *same* partial result.
 * @param reduction_mode_flag   Specifies the operation (e.g., AGG_MODE_SUM).
 */

/**
 * @brief (Node 9 & 12, Tier 0) The "Identity" case.
 */
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_items_to_reduce,
    int item_stride,
    int reduction_mode_flag) {
    // local_mem, num_items_to_reduce, and reduction_mode_flag are unused but
    // required for a consistent host-side interface.
    const int i = get_global_id(0);
    if (i >= item_stride) {
        return;
    }
    final_output_buf[i] = partial_input_buf[i];
}

/**
 * @brief (Node 9 & 12, Tier 1) The "Register Reduce".
 */
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_items_to_reduce,
    int item_stride,
    int reduction_mode_flag) {
    // local_mem is unused but required for a consistent host-side interface.
    const int i = get_global_id(0);
    if (i >= item_stride) {
        return;
    }

    SCALAR_TYPE accum = SCALAR_ZERO;
    // For SoA data, the partial results for item 'i' are strided.
    for (int j = 0; j < num_items_to_reduce; j++) {
        accum += partial_input_buf[j * item_stride + i];
    }

    if (reduction_mode_flag == AGG_MODE_AVERAGE && num_items_to_reduce > 0) {
        accum /= (SCALAR_TYPE)num_items_to_reduce;
    }

    final_output_buf[i] = accum;
}

/**
 * @brief (Node 9 & 12, Tier 2 & 3) The "Local Reduce" workhorse.
 *
 * WORK DISPATCH: Global size is (item_stride * local_size, 1, 1).
 *                Local size is (local_size, 1, 1).
 * This launches `item_stride` work-groups, one for each element of the output vector.
 */
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_items_to_reduce,
    int item_stride,
    int reduction_mode_flag) {

    const int element_idx = get_group_id(0); // Which element of the final tensor we are computing (0 to item_stride-1).
    const int lid         = get_local_id(0);
    const int lsize       = get_local_size(0);

    SCALAR_TYPE accum = SCALAR_ZERO;

    // Each work-item in the group sums a strided slice of the inputs.
    // The base pointer is the start of the data for the element this work-group is responsible for.
    const __global SCALAR_TYPE *base_input_ptr = partial_input_buf + element_idx;

    for (int i = lid; i < num_items_to_reduce; i += lsize) {
        // Access the i-th partial result for this specific element.
        accum += base_input_ptr[i * item_stride];
    }

    local_mem[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction within the work-group (unchanged)
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
