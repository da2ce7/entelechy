// phase_2_learn_B_processing.cl.c

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// --- Implementation: clip_partial_gradients (Node 11) ---
// Strategy: A two-pass algorithm that treats four distinct physical gradient buffers as a
// single "virtual vector". Pass 1 calculates the total L2 norm of this virtual vector
// using a parallel reduction. Pass 2 uses the computed norm to conditionally scale all
// elements of the virtual vector, writing them to their corresponding destination buffers.
// The entire operation is handled by a single work-group per logical item (tile).
__kernel void clip_partial_gradients(
    __local SCALAR_TYPE        *update_buffer_LOCAL_reduction_tile,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_weights_module,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_biases_module,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_temps,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_hidden_activations_aos,
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_clipping_threshold_per_item,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_weights_module,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_biases_module,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_temps,
    __global SCALAR_TYPE       *dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
    uint                        src_scalar_FLAG_use_per_item_norm,
    SCALAR_TYPE                 src_scalar_REAL_clipping_threshold_t_pre,
    SCALAR_TYPE                 src_scalar_REAL_epsilon,
    uint                        src_scalar_NATURAL_flat_tile_index,
    uint                        src_scalar_NATURAL_num_class_chunks,
    uint                        src_scalar_NATURAL_classes_per_chunk,
    uint                        src_scalar_NATURAL_modules_per_chunk,
    uint                        src_scalar_NATURAL_total_batch_count,
    uint                        src_scalar_NATURAL_padded_hidden_count,
    uint                        src_scalar_NATURAL_total_tile_count) {

    const uint lid   = get_local_id(0);
    const uint lsize = get_local_size(0);
    // This kernel assumes a 1D work-group dispatch per logical item (tile).

    // --- 1. Calculate Per-Buffer Element Counts and Base Offsets for the Current Tile ---
    // This section defines the size of each segment of our "virtual vector".
    const uint n_weights      = src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_classes_per_chunk;
    const uint n_biases       = src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_classes_per_chunk;
    const uint n_temps        = src_scalar_NATURAL_modules_per_chunk;
    const uint n_hidden       = src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count;
    const uint total_elements = n_weights + n_biases + n_temps + n_hidden;

    // These offsets point to the start of this tile's data within the large collection buffers.
    const long weight_base_offset = (long)src_scalar_NATURAL_flat_tile_index * n_weights;
    const long bias_base_offset   = (long)src_scalar_NATURAL_flat_tile_index * n_biases;
    const long temp_base_offset   = (long)src_scalar_NATURAL_flat_tile_index * n_temps;
    const long hidden_base_offset = (long)src_scalar_NATURAL_flat_tile_index * n_hidden;

    // --- 2. Pass 1: Calculate Sum of Squares for L2 Norm ---
    // Each thread calculates a partial sum of squares from a strided slice of the virtual vector.
    SCALAR_TYPE local_sq_sum = SCALAR_ZERO;
    for (uint i = lid; i < total_elements; i += lsize) {
        SCALAR_TYPE val;
        // This conditional logic maps the linear index `i` to the correct physical buffer.
        if (i < n_weights) {
            val = src_buffer_GLOBAL_partial_grad_weights_module[weight_base_offset + i];
        } else if (i < n_weights + n_biases) {
            val = src_buffer_GLOBAL_partial_grad_biases_module[bias_base_offset + i - n_weights];
        } else if (i < n_weights + n_biases + n_temps) {
            val = src_buffer_GLOBAL_partial_grad_temps[temp_base_offset + i - n_weights - n_biases];
        } else {
            val = src_buffer_GLOBAL_partial_grad_hidden_activations_aos[hidden_base_offset + i - n_weights - n_biases - n_temps];
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

    // --- 3. Determine Scaling Factor and Broadcast via Local Memory ---
    // The leader thread (lid=0) computes the final norm and scaling factor.
    if (lid == 0) {
        // Select the clipping threshold based on the host-provided flag.
        SCALAR_TYPE threshold;
        if (src_scalar_FLAG_use_per_item_norm == 1) {
            threshold = src_buffer_GLOBAL_CONST_clipping_threshold_per_item[src_scalar_NATURAL_flat_tile_index];
        } else {
            threshold = src_scalar_REAL_clipping_threshold_t_pre;
        }

        const SCALAR_TYPE total_sum_sq = update_buffer_LOCAL_reduction_tile[0];
        const SCALAR_TYPE norm         = MATH_FN sqrt(total_sum_sq);

        SCALAR_TYPE scale_factor = 1.0f;
        // Only compute a new scale factor if the norm exceeds the threshold.
        if (norm > threshold) {
            // Add epsilon for numerical stability, preventing division by zero if norm is very close to threshold.
            scale_factor = threshold / (norm + src_scalar_REAL_epsilon);
        }
        // Optimization: Write the scale factor back to local memory to broadcast it to all threads.
        update_buffer_LOCAL_reduction_tile[0] = scale_factor;
    }

    // Synchronize to ensure all threads see the computed scale_factor.
    barrier(CLK_LOCAL_MEM_FENCE);
    const SCALAR_TYPE scale_factor = update_buffer_LOCAL_reduction_tile[0];

    // --- 4. Pass 2: Conditionally Scale and Write to Destination ---
    // Each thread applies the single, broadcasted scale_factor to its slice of the virtual vector.
    for (uint i = lid; i < total_elements; i += lsize) {
        SCALAR_TYPE val;
        // This second `if/else` chain reads the original values again and writes the scaled
        // result to the corresponding destination buffer.
        if (i < n_weights) {
            val                                                                            = src_buffer_GLOBAL_partial_grad_weights_module[weight_base_offset + i];
            dest_buffer_GLOBAL_clipped_partial_grad_weights_module[weight_base_offset + i] = val * scale_factor;
        } else if (i < n_weights + n_biases) {
            long relative_idx                                                                      = i - n_weights;
            val                                                                                    = src_buffer_GLOBAL_partial_grad_biases_module[bias_base_offset + relative_idx];
            dest_buffer_GLOBAL_clipped_partial_grad_biases_module[bias_base_offset + relative_idx] = val * scale_factor;
        } else if (i < n_weights + n_biases + n_temps) {
            long relative_idx                                                              = i - n_weights - n_biases;
            val                                                                            = src_buffer_GLOBAL_partial_grad_temps[temp_base_offset + relative_idx];
            dest_buffer_GLOBAL_clipped_partial_grad_temps[temp_base_offset + relative_idx] = val * scale_factor;
        } else {
            long relative_idx = i - n_weights - n_biases - n_temps;
            val               = src_buffer_GLOBAL_partial_grad_hidden_activations_aos[hidden_base_offset + relative_idx];
            dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos[hidden_base_offset + relative_idx] = val * scale_factor;
        }
    }
}

// --- Implementation: gather_and_permute_grad_hidden_activations (Node 13) ---
// Strategy: A specialized "gather-sum-permute" kernel that acts as a global barrier.
// Each work-item is assigned to compute one scalar value in the *destination* SoA buffer.
// It does this by traversing all relevant source tiles, gathering the scattered partial
// results, and summing them (an implicit reduction) into a single value. This mapping
// elegantly solves the "Transpose Illusion," transforming the chunked AoS input into a
// dense, reduction-ready SoA output in a single pass.
__kernel void gather_and_permute_grad_hidden_activations(
    __global const SCALAR_TYPE *src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,
    __global SCALAR_TYPE *dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_padded_total_modules_count,
    uint src_scalar_NATURAL_num_module_chunks_count,
    uint src_scalar_NATURAL_modules_per_chunk_count,
    uint src_scalar_NATURAL_num_class_chunks_count,
    uint src_scalar_NATURAL_total_tile_count) {

    // --- 1. Work-Item to Destination Coordinate Mapping ---
    // The 2D dispatch grid maps directly to the logical coordinates of the SoA output tensor.
    // 'bh_flat_idx' represents a flattened (batch, hidden) coordinate.
    const uint bh_flat_idx       = get_global_id(0);
    // 'module_global_idx' represents the module coordinate.
    const uint module_global_idx = get_global_id(1);

    // Boundary check against the logical dimensions of the destination tensor.
    if (bh_flat_idx >= (src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count) ||
        module_global_idx >= src_scalar_NATURAL_total_modules_count) {
        return;
    }

    // --- 2. Implicit Reduction over Class Chunks ---
    // This work-item is responsible for the final Grad_H[b,h,m]. It must sum the
    // partial results from all class chunks that contributed to this value.
    SCALAR_TYPE accum = SCALAR_ZERO;

    // De-flatten the work-item's assigned destination coordinates to find the
    // source coordinates needed for the gather operation.
    const uint batch_idx = bh_flat_idx / src_scalar_NATURAL_padded_hidden_count;
    const uint h_idx     = bh_flat_idx % src_scalar_NATURAL_padded_hidden_count;

    // From the global module index, find the specific chunk and local index within it.
    const uint module_chunk_idx = module_global_idx / src_scalar_NATURAL_modules_per_chunk_count;
    const uint module_local_idx = module_global_idx % src_scalar_NATURAL_modules_per_chunk_count;

    // Loop through all class chunks to gather and sum partial results.
    for (uint class_chunk_idx = 0; class_chunk_idx < src_scalar_NATURAL_num_class_chunks_count; ++class_chunk_idx) {
        // Reconstruct the flat_tile_index that contains the partial data we need.
        const uint flat_tile_idx = module_chunk_idx * src_scalar_NATURAL_num_class_chunks_count + class_chunk_idx;

        if (flat_tile_idx < src_scalar_NATURAL_total_tile_count) {
            // --- 3. Complex Source Address Calculation ---
            // This is the core of the gather logic: calculating the precise 1D address within the
            // monolithic source buffer for the desired partial result.
            const long tile_size        = (long)src_scalar_NATURAL_modules_per_chunk_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count;
            const long tile_base_offset = (long)flat_tile_idx * tile_size;

            const long local_offset = (long)module_local_idx * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count +
                                      (long)batch_idx * src_scalar_NATURAL_padded_hidden_count +
                                      h_idx;

            const long read_idx = tile_base_offset + local_offset;
            accum += src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos[read_idx];
        }
    }

    // --- 4. Final Write to SoA Buffer ---
    // The write address calculation is simple due to the SoA layout. All data for a given
    // (batch, hidden) pair is laid out contiguously across modules. This write operation,
    // performed by all threads in parallel, completes the permutation from AoS to SoA.
    const long write_idx = (long)bh_flat_idx * src_scalar_NATURAL_padded_total_modules_count + module_global_idx;
    dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa[write_idx] = accum;
}
