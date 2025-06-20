// aggregation_kernels.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

/*
 * =================================================================================================
 * == AGGREGATION ENGINE IMPLEMENTATIONS (Node 8 & 10)                                            ==
 * =================================================================================================
 *
 * This file contains the tiered implementations for the generic aggregation engine.
 * The host is responsible for selecting and launching the correct kernel based on the
 * number of items that need to be reduced (`num_items_to_reduce`).
 *
 * These kernels implement the logic for BOTH aggregation steps in the DAG:
 * - Node 8: Aggregates initial partial results (Probs, Loss, Gradients for Exits/Temps).
 * - Node 10: Aggregates partial gradients for the Shared Layer from the backprop stream.
 */

/**
 * @brief (Node 8 & 10, Tier 0) The "Identity" case for a single item to reduce.
 *
 * Simply copies data from the partial input buffer to the final output buffer.
 * This is effectively a `memcpy`, used when num_chunks is 1.
 */
__kernel void aggregate_identity(__global const SCALAR_TYPE *__restrict partial_input_buf, __global SCALAR_TYPE *__restrict final_output_buf, int item_stride) {
    const int i = get_global_id(0);
    if (i >= item_stride) {
        return;
    }
    final_output_buf[i] = partial_input_buf[i];
}

/**
 * @brief (Node 8 & 10, Tier 1) The "Register Reduce" for a small number of items.
 *
 * Each work-item is responsible for one element in the final output vector. It performs a fast,
 * serial summation in a private register over the few items to be reduced.
 * WORK DISPATCH: Global size should be (item_stride, 1, 1).
 */
__kernel void aggregate_register_reduce(
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
    for (int j = 0; j < num_items_to_reduce; j++) {
        accum += partial_input_buf[j * item_stride + i];
    }

    if (reduction_mode_flag == AGG_MODE_AVERAGE && num_items_to_reduce > 0) {
        accum /= (SCALAR_TYPE)num_items_to_reduce;
    }

    final_output_buf[i] = accum;
}

/**
 * @brief (Node 8 & 10, Tier 2 & 3) The "Local Reduce" workhorse for medium-to-large reductions.
 *
 * This kernel can be used as a standalone Tier 2 reduction or as a building block for the
 * multi-stage Tier 3 hierarchical reduction. A single work-group is responsible for reducing
 * all inputs corresponding to a single element of the output vector.
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

    const int i     = get_group_id(0);
    const int lid   = get_local_id(0);
    const int lsize = get_local_size(0);

    SCALAR_TYPE accum = SCALAR_ZERO;

    for (int j = lid; j < num_items_to_reduce; j += lsize) {
        accum += partial_input_buf[j * item_stride + i];
    }

    local_mem[lid] = accum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = lsize / 2; s > 0; s >>= 1) {
        if (lid < s) {
            local_mem[lid] += local_mem[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        SCALAR_TYPE result = local_mem[0];
        if (reduction_mode_flag == AGG_MODE_AVERAGE && num_items_to_reduce > 0) {
            result /= (SCALAR_TYPE)num_items_to_reduce;
        }
        final_output_buf[i] = result;
    }
}
