// aggregation_kernels.cl.c

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

// --- Implementation: aggregate_identity (Node 12, 15) (Tier 0: N=1) ---
// Strategy: A simple memory copy. Each work-item maps to one element of the
// tensor. This provides a near-zero-cost abstraction path for the N=1 case,
// fulfilling the unified dataflow architecture principle.
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_partials_to_reduce,
    int elements_per_partial,
    int reduction_mode_flag) {
    const int i = get_global_id(0);
    if (i >= elements_per_partial) {
        return;
    }
    final_output_buf[i] = partial_input_buf[i];
}

// --- Implementation: aggregate_register_reduce (Node 12, 15) (Tier 1: N is small) ---
// Strategy: A "map" kernel where each work-item computes a single element of
// the final tensor. The reduction loop is performed entirely in private registers
// (`accum`), making it highly efficient for a small number of partials to reduce.
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_partials_to_reduce,
    int elements_per_partial,
    int reduction_mode_flag) {
    const int i = get_global_id(0);
    if (i >= elements_per_partial) {
        return;
    }

    SCALAR_TYPE accum = SCALAR_ZERO;
    // For SoA data, iterating through the partial results for a single element `i`
    // requires striding by `elements_per_partial`.
    for (int j = 0; j < num_partials_to_reduce; j++) {
        accum += partial_input_buf[j * elements_per_partial + i];
    }

    if (reduction_mode_flag == AGG_MODE_AVERAGE && num_partials_to_reduce > 0) {
        accum /= (SCALAR_TYPE)num_partials_to_reduce;
    }

    final_output_buf[i] = accum;
}

// --- Implementation: aggregate_local_reduce (Node 12, 15) (Tier 2: N is large) ---
// Strategy: A "work-group per element" reduction. Each work-group is responsible
// for computing one element of the final output tensor. Threads within the group
// collaboratively sum their assigned partial results, then perform a final, fast
// reduction using __local memory.
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict partial_input_buf,
    __global SCALAR_TYPE *__restrict final_output_buf,
    int num_partials_to_reduce,
    int elements_per_partial,
    int reduction_mode_flag) {

    // A whole work-group computes a single element of the final output tensor.
    const int element_idx = get_group_id(0);
    const int lid         = get_local_id(0);
    const int lsize       = get_local_size(0);

    SCALAR_TYPE accum = SCALAR_ZERO;

    // The base pointer is the start of the data for the element this work-group is responsible for.
    const __global SCALAR_TYPE *base_input_ptr = partial_input_buf + element_idx;

    // Each work-item in the group sums a strided slice of the partial results.
    for (int i = lid; i < num_partials_to_reduce; i += lsize) {
        // Accessing the i-th partial result for this specific element. The SoA layout
        // ensures that concurrent reads from different work-groups are coalesced.
        accum += base_input_ptr[i * elements_per_partial];
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
        if (reduction_mode_flag == AGG_MODE_AVERAGE && num_partials_to_reduce > 0) {
            result /= (SCALAR_TYPE)num_partials_to_reduce;
        }
        final_output_buf[element_idx] = result;
    }
}
// --- Implementation: reduce_grad_h_over_modules (Node 13) ---
// Strategy: A padding-aware reduction kernel serving as the "join" for the
// multi-head fork. Each work-group computes one output element, reducing over
// the module dimension. This ensures coalesced global memory reads from the
// input buffer, even when its physical stride (`padded_total_modules`) differs
// from its logical dimension (`total_modules`) due to memory alignment.
__kernel void reduce_grad_h_over_modules(
    __local SCALAR_TYPE *local_mem,
    __global const SCALAR_TYPE *__restrict aggregated_grad_h_soa,
    __global SCALAR_TYPE *__restrict final_grad_h_buf,
    int total_elements,
    int total_modules,
    int padded_total_modules) {

    const int element_idx = get_group_id(0);
    const int lid         = get_local_id(0);
    const int lsize       = get_local_size(0);

    // Guard against excess work-groups.
    if (element_idx >= total_elements) {
        return;
    }

    SCALAR_TYPE accum = SCALAR_ZERO;

    // Get the start of the row using the physical stride to handle padding.
    const __global SCALAR_TYPE *row_start_ptr = aggregated_grad_h_soa + (long)element_idx * padded_total_modules;

    // Sum over the LOGICAL number of modules, skipping any padding elements.
    for (int i = lid; i < total_modules; i += lsize) {
        accum += row_start_ptr[i];
    }

    // Perform a standard parallel reduction within the work-group.
    local_mem[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_mem[lid] += local_mem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread writes the final result.
    if (lid == 0) {
        final_grad_h_buf[element_idx] = local_mem[0];
    }
}
