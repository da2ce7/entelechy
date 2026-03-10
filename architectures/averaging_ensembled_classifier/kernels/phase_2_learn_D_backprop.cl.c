// phase_2_learn_D_backprop.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: backprop_shared_weights_chunk (Node 17) ---
// Strategy: A "work-group per gradient" reduction kernel designed for the "True
// Streaming" backpropagation model. Each work-group is assigned to compute one
// scalar gradient value for a single shared weight (dL/dW_ij). Threads within
// the group collaborate to reduce (sum) the contributions from all samples
// in the assigned batch chunk.
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_input,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_partial_grad_weights_shared,
    uint                        src_scalar_NATURAL_batch_chunk_offset,
    uint                        src_scalar_NATURAL_batch_chunk_count,
    uint                        src_scalar_NATURAL_batch_chunk_index,
    uint                        src_scalar_NATURAL_total_batch_count,
    uint                        src_scalar_NATURAL_num_batch_chunks_count,
    uint                        src_scalar_NATURAL_padded_input_count,
    uint                        src_scalar_NATURAL_padded_hidden_count,
    uint                        src_scalar_NATURAL_final_grad_hidden_total_element_count) {

    // --- 1. Work-Group to Gradient Component Mapping ---
    // The 2D work-group ID maps to a coordinate in the shared weight matrix (input_dim, hidden_dim).
    const uint i_idx = get_group_id(0); // Index along the input dimension.
    const uint j_idx = get_group_id(1); // Index along the hidden dimension.
    // Threads within the group collaborate on the reduction over the batch chunk.
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Boundary check for the gradient component this work-group is assigned.
    if (i_idx >= src_scalar_NATURAL_padded_input_count || j_idx >= src_scalar_NATURAL_padded_hidden_count) {
        return;
    }

    // --- 2. Parallel Reduction over the Batch Chunk ---
    SCALAR_TYPE p_grad_sw = SCALAR_ZERO; // This thread's partial sum.

    // Each thread calculates the gradient contribution from a strided slice of samples in the chunk.
    for (uint b_local = lid; b_local < src_scalar_NATURAL_batch_chunk_count; b_local += lsize) {
        const uint b_global = src_scalar_NATURAL_batch_chunk_offset + b_local;

        // Skip computation for any padded samples within the chunk.
        if (src_buffer_GLOBAL_sample_mask[b_global] < (SCALAR_TYPE)0.5f) {
            continue;
        }

        // --- Apply the Chain Rule: dL/dW_ij = (dL/dA_j) * (dA_j/dZ_j) * (dZ_j/dW_ij) ---
        const long        hidden_offset = (long)b_global * src_scalar_NATURAL_padded_hidden_count + j_idx;
        const SCALAR_TYPE grad_h        = src_buffer_GLOBAL_summed_grad_hidden_activations[hidden_offset]; // (dL/dA_j)
        const SCALAR_TYPE hidden_val    = src_buffer_GLOBAL_hidden_activations[hidden_offset];

        // Derivative of ReLU activation: (dA_j/dZ_j)
        const SCALAR_TYPE d_activation = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);
        const SCALAR_TYPE dL_dZ_j      = grad_h * d_activation;

        // Final term: (dZ_j/dW_ij), which is simply the corresponding input value.
        const SCALAR_TYPE input_val = src_buffer_GLOBAL_input[(long)b_global * src_scalar_NATURAL_padded_input_count + i_idx];
        p_grad_sw += dL_dZ_j * input_val;
    }

    // --- 3. Intra-Workgroup Reduction & Final Write ---
    // Perform a standard parallel reduction on the partial sums using local memory.
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_sw;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // The leader thread writes the final, reduced partial gradient for this chunk
    // to its unique slot in the collection buffer, fulfilling the placement contract.
    // WHY: The write uses (hidden-major, input) order — j_idx * padded_input + i_idx —
    // matching the forward_pass kernel's SIMD-major weight layout (h * padded_input + i).
    // This ensures the flat gradient layout is element-wise compatible with the weight
    // buffer, so the Adam update applies each gradient to the correct weight.
    if (lid == 0) {
        const long chunk_base_offset                                   = (long)src_scalar_NATURAL_batch_chunk_index * src_scalar_NATURAL_padded_input_count * src_scalar_NATURAL_padded_hidden_count;
        const long grad_w_out_idx                                      = chunk_base_offset + (long)j_idx * src_scalar_NATURAL_padded_input_count + i_idx;
        dest_buffer_GLOBAL_partial_grad_weights_shared[grad_w_out_idx] = update_buffer_LOCAL_reduction_tile[0];
    }
}

// --- Implementation: backprop_shared_biases_chunk (Node 18) ---
// Strategy: A highly efficient "work-group per gradient" reduction kernel optimized
// for a 1D output. Each work-group is assigned to compute the partial gradient for a
// single shared bias term (dL/dB_j). This 1D dispatch is simpler and more efficient
// than a 2D model. Threads within the group collaborate to reduce the contributions
// from all samples in their assigned batch chunk.
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_partial_grad_biases_shared,
    uint                        src_scalar_NATURAL_batch_chunk_offset,
    uint                        src_scalar_NATURAL_batch_chunk_count,
    uint                        src_scalar_NATURAL_batch_chunk_index,
    uint                        src_scalar_NATURAL_total_batch_count,
    uint                        src_scalar_NATURAL_num_batch_chunks_count,
    uint                        src_scalar_NATURAL_padded_hidden_count,
    uint                        src_scalar_NATURAL_final_grad_hidden_total_element_count) {

    // --- 1. Work-Group to Gradient Component Mapping ---
    // The 1D work-group ID maps to an index in the shared bias vector.
    const uint j_idx = get_group_id(0);
    // Threads within the group collaborate on the reduction over the batch chunk.
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);

    // Boundary check for the gradient component this work-group is assigned.
    if (j_idx >= src_scalar_NATURAL_padded_hidden_count) {
        return;
    }

    // --- 2. Parallel Reduction over the Batch Chunk ---
    SCALAR_TYPE p_grad_sb = SCALAR_ZERO; // This thread's partial sum.

    // Each thread calculates the gradient contribution from a strided slice of samples in the chunk.
    for (uint b_local = lid; b_local < src_scalar_NATURAL_batch_chunk_count; b_local += lsize) {
        const uint b_global = src_scalar_NATURAL_batch_chunk_offset + b_local;

        // Skip computation for any padded samples within the chunk.
        if (src_buffer_GLOBAL_sample_mask[b_global] < (SCALAR_TYPE)0.5f) {
            continue;
        }

        // --- Apply the Chain Rule: dL/dB_j = (dL/dA_j) * (dA_j/dZ_j) * (dZ_j/dB_j) ---
        // For biases, dZ_j/dB_j = 1, so the gradient is simply dL/dZ_j.
        const long        hidden_offset = (long)b_global * src_scalar_NATURAL_padded_hidden_count + j_idx;
        const SCALAR_TYPE grad_h        = src_buffer_GLOBAL_summed_grad_hidden_activations[hidden_offset]; // (dL/dA_j)
        const SCALAR_TYPE hidden_val    = src_buffer_GLOBAL_hidden_activations[hidden_offset];

        // Derivative of ReLU activation: (dA_j/dZ_j)
        const SCALAR_TYPE d_activation = select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, hidden_val > SCALAR_ZERO);

        // Sum the contributions to the gradient, dL/dB_j. This calculation is simpler
        // than for weights as it does not require reading from the main input buffer.
        p_grad_sb += grad_h * d_activation;
    }

    // --- 3. Intra-Workgroup Reduction & Final Write ---
    // Perform a standard parallel reduction on the partial sums using local memory.
    update_buffer_LOCAL_reduction_tile[lid] = p_grad_sb;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // The leader thread writes the final, reduced partial gradient for this chunk
    // to its unique slot in the collection buffer, fulfilling the placement contract.
    if (lid == 0) {
        const long chunk_base_offset                                  = (long)src_scalar_NATURAL_batch_chunk_index * src_scalar_NATURAL_padded_hidden_count;
        const long grad_b_out_idx                                     = chunk_base_offset + j_idx;
        dest_buffer_GLOBAL_partial_grad_biases_shared[grad_b_out_idx] = update_buffer_LOCAL_reduction_tile[0];
    }
}

// --- Implementation: clip_shared_gradients_chunk (Node 19) ---
// Strategy: A streamable utility that acts as a stabilization gateway. It operates on
// the raw output chunks from the shared backpropagation kernels (Nodes 17 & 18).
// Using a "virtual vector" abstraction, it computes a single L2 norm across both
// weight and bias gradients and then writes the conditionally-scaled results to a
// destination address explicitly provided by the host.
__kernel void clip_shared_gradients_chunk(
    __local SCALAR_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_weights_shared,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_biases_shared,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_weights_shared,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_biases_shared,
    SCALAR_TYPE                 src_scalar_REAL_clipping_threshold_t_pre,
    SCALAR_TYPE                 src_scalar_REAL_epsilon,
    uint                        src_scalar_NATURAL_weights_parameter_count,
    uint                        src_scalar_NATURAL_biases_parameter_count,
    uint                        dest_scalar_NATURAL_weights_write_offset_elements,
    uint                        dest_scalar_NATURAL_biases_write_offset_elements,
    uint                        src_scalar_NATURAL_num_batch_chunks) {

    // --- 1. Setup ---
    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);
    // The total number of elements in the logically concatenated "virtual vector".
    const uint total_elements = src_scalar_NATURAL_weights_parameter_count + src_scalar_NATURAL_biases_parameter_count;

    // --- 2. Pass 1: Calculate Sum of Squares for L2 Norm ---
    SCALAR_TYPE local_sq_sum = SCALAR_ZERO;
    // Each thread calculates a partial sum of squares from a strided slice of the virtual vector.
    for (uint i = lid; i < total_elements; i += lsize) {
        SCALAR_TYPE val;
        // This conditional logic maps the linear index `i` to the correct physical buffer.
        if (i < src_scalar_NATURAL_weights_parameter_count) {
            val = src_buffer_GLOBAL_partial_grad_weights_shared[i];
        } else {
            val = src_buffer_GLOBAL_partial_grad_biases_shared[i - src_scalar_NATURAL_weights_parameter_count];
        }
        local_sq_sum += val * val;
    }

    // Perform a standard parallel reduction on the partial sums using local memory.
    update_buffer_LOCAL_reduction_tile[lid] = local_sq_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            update_buffer_LOCAL_reduction_tile[lid] += update_buffer_LOCAL_reduction_tile[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // --- 3. Determine Scaling Factor and Broadcast ---
    // The leader thread computes the final factor and broadcasts it via local memory.
    if (lid == 0) {
        const SCALAR_TYPE total_sum_sq = update_buffer_LOCAL_reduction_tile[0];
        const SCALAR_TYPE norm         = MATH_FN sqrt(total_sum_sq);

        SCALAR_TYPE scale_factor = 1.0f;
        if (norm > src_scalar_REAL_clipping_threshold_t_pre) {
            scale_factor = src_scalar_REAL_clipping_threshold_t_pre / (norm + src_scalar_REAL_epsilon);
        }
        update_buffer_LOCAL_reduction_tile[0] = scale_factor;
    }

    // Synchronize to ensure all threads see the computed scale_factor.
    barrier(CLK_LOCAL_MEM_FENCE);
    const SCALAR_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

    // --- 4. Pass 2: Conditional Scaling and Placement Write ---
    // Each thread applies the single, broadcasted scale_factor to its slice of the virtual vector,
    // writing the result to the host-specified destination offset.
    for (uint i = lid; i < total_elements; i += lsize) {
        if (i < src_scalar_NATURAL_weights_parameter_count) {
            const SCALAR_TYPE val = src_buffer_GLOBAL_partial_grad_weights_shared[i];
            // This write operation is the fulfillment of the placement contract. The host provides the
            // exact base offset, and this kernel simply adds the element's relative index.
            dest_buffer_GLOBAL_clipped_partial_grad_weights_shared[dest_scalar_NATURAL_weights_write_offset_elements + i] = val * scale_factor;
        } else {
            const uint        relative_idx                                                                                         = i - src_scalar_NATURAL_weights_parameter_count;
            const SCALAR_TYPE val                                                                                                  = src_buffer_GLOBAL_partial_grad_biases_shared[relative_idx];
            dest_buffer_GLOBAL_clipped_partial_grad_biases_shared[dest_scalar_NATURAL_biases_write_offset_elements + relative_idx] = val * scale_factor;
        }
    }
}
