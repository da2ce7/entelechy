// phase_2_learn_B_processing.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif


// --- Implementation: transpose_chunk (Node 11) ---
// Strategy: A fully generic, tiled matrix transpose utility. This kernel is a
// cornerstone of the latency-hiding strategy for the backpropagation path.
// By operating on abstract chunks with specified offsets and leading dimensions,
// it can be enqueued immediately after its preceding compute kernel on a per-chunk
// basis. This allows the GPU's out-of-order scheduler to overlap this memory-bound
// operation for chunk N with the compute-bound operations for chunk N+1,
// maximizing hardware utilization.
__kernel void transpose_chunk(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict in_buf,
    __global SCALAR_TYPE *__restrict out_buf,
    int in_offset_elements,
    int out_offset_elements,
    int num_rows_in_chunk,
    int num_cols_in_chunk,
    int in_leading_dim,
    int out_leading_dim) {

#define TILE_DIM C_TILE_SIZE
// Pad the tile dimension by 1 to prevent local memory bank conflicts when transposing.
#define PADDED_TILE_DIM (TILE_DIM + 1)

    // Phase 1: Coalesced Read from Global (Source) to Local Memory
    //
    // Work-item's logical (x, y) coordinates within the source chunk.
    const int x_in = get_group_id(0) * TILE_DIM + get_local_id(0);
    const int y_in = get_group_id(1) * TILE_DIM + get_local_id(1);

    // Boundary check to ensure we only read from within the source chunk's bounds.
    if (x_in < num_cols_in_chunk && y_in < num_rows_in_chunk) {
        // Calculate the physical 1D index into the global input buffer using the
        // provided offset and leading dimension (stride).
        const int in_idx                                               = in_offset_elements + (y_in * in_leading_dim) + x_in;
        local_mem[get_local_id(1) * PADDED_TILE_DIM + get_local_id(0)] = in_buf[in_idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Coalesced Write from Local to Global (Destination) Memory
    //
    // Transpose the local coordinates to find the logical (x, y) coordinates
    // for the destination chunk.
    const int x_out = get_group_id(1) * TILE_DIM + get_local_id(0);
    const int y_out = get_group_id(0) * TILE_DIM + get_local_id(1);

    // Boundary check against the transposed dimensions of the chunk.
    if (x_out < num_rows_in_chunk && y_out < num_cols_in_chunk) {
        // Calculate the physical 1D index into the global output buffer using its
        // distinct offset and leading dimension.
        const int out_idx = out_offset_elements + (y_out * out_leading_dim) + x_out;
        out_buf[out_idx]  = local_mem[get_local_id(0) * PADDED_TILE_DIM + get_local_id(1)];
    }
}