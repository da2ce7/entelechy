// kernels.cl.h

#ifndef KERNELS_CL_H
#define KERNELS_CL_H

// Check for OpenCL environment (target: OpenCL 1.2 without atomics)
#ifdef __OPENCL_VERSION__
// Require OpenCL 1.2 or later
#if __OPENCL_VERSION__ < 120
#error "OpenCL 1.2 or newer is required. Please use a compatible device/driver."
#endif

// Enable FP16 extension if using half precision
#if SCALAR_TYPE == half
#if !defined(cl_khr_fp16)
#error "FP16 extension (cl_khr_fp16) required for half precision but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define SCALAR_ZERO 0.0h
#else
#define SCALAR_ZERO 0.0f
#endif

// Define standard kernel attributes for OpenCL environment
#define KERNEL_ATTR __attribute__((work_group_size_hint(SIMD_WIDTH, 1, 1)))

// In OpenCL mode, require SCALAR_TYPE, SIMD_WIDTH and C_TILE_SIZE to be defined
#ifndef SCALAR_TYPE
#error "SCALAR_TYPE must be defined in OpenCL mode"
#endif
#ifndef SIMD_WIDTH
#error "SIMD_WIDTH must be defined in OpenCL mode"
#endif
#ifndef C_TILE_SIZE
#error "C_TILE_SIZE must be defined in OpenCL mode"
#endif

#else
// Host/C++ mode stub definitions
#include <float.h> // For FLT_MAX
#include <math.h>  // For fmax, sqrt, etc.
#include <stdio.h> // For printf in debug stubs

#ifndef __kernel
#define __kernel
#endif
#ifndef __local
#define __local
#endif
#ifndef __global
#define __global
#endif
#ifndef uint
#define uint int
#endif
#define KERNEL_ATTR
#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#define SCALAR_ZERO 0.0f
#endif
#ifndef SIMD_WIDTH
#define SIMD_WIDTH 1
#endif
#ifndef C_TILE_SIZE
#define C_TILE_SIZE 1
#endif
#define CLK_LOCAL_MEM_FENCE 0x01
#define CLK_GLOBAL_MEM_FENCE 0x02
#define min(a, b) (((a) < (b)) ? (a) : (b))
inline int         get_global_id(int dim) { return 0; }
inline int         get_local_id(int dim) { return 0; }
inline int         get_group_id(int dim) { return 0; }
inline int         get_local_size(int dim) { return 1; }
inline int         get_global_size(int dim) { return 1; }
inline int         get_num_groups(int dim) { return 1; }
inline void        barrier(int flags) { (void)flags; }
inline SCALAR_TYPE clamp(SCALAR_TYPE val, SCALAR_TYPE min_val, SCALAR_TYPE max_val) { return fmin(fmax(val, min_val), max_val); }
inline SCALAR_TYPE select(SCALAR_TYPE a, SCALAR_TYPE b, int c) { return (c) ? b : a; }
#endif // __OPENCL_VERSION__

// --- Host-configurable Flags and Enums ---

// Used by kernels to select loss/gradient math (e.g., Softmax vs. Sigmoid).
#define PROBLEM_TYPE_CCE 0
#define PROBLEM_TYPE_BCE 1

// Used by the generic `aggregate_kernel` to select the reduction operation.
#define AGG_MODE_SUM 0
#define AGG_MODE_AVERAGE 1 // Placeholder for more complex ops

// Common math configuration
#ifndef USE_FAST_MATH
#define USE_FAST_MATH 0
#endif
#if USE_FAST_MATH
#define MATH_FN native_
#else
#define MATH_FN
#endif

// =================================================================================================
// == KERNEL DECLARATIONS FOR THE UNIFIED STREAMING ARCHITECTURE                                  ==
// =================================================================================================

// --- Phase 3: Universal Chunk Processing Loop Kernels ---

/**
 * @brief (Node 4) Performs the feed-forward pass for the neural network's shared layer.
 *
 * Host Assumptions:
 * - This kernel's contract includes both computing hidden activations and propagating the validity mask.
 * - It can be used in two modes, controlled by the host's `hidden_lifecycle_mode`:
 *   1. PRECOMPUTE: Called once on the full batch if VRAM allows, caching the result.
 *   2. STREAM: Called repeatedly for each chunk of the batch if VRAM is constrained.
 * - `batch_offset` and `num_batch_samples` define the slice of the batch to process.
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                                // [MEMORY] Local memory for tiling.
    __global const SCALAR_TYPE *__restrict input_buf,              // [INPUT] The full batch's input feature data.
    __global const SCALAR_TYPE *__restrict input_mask,             // [INPUT] The full batch's mask for valid samples.
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf, // [INPUT] SIMD-major weights.
    __global const SCALAR_TYPE *__restrict biases_buf,             // [INPUT] Biases for the hidden layer.
    __global SCALAR_TYPE *__restrict hidden_out_buf,               // [OUTPUT] Resulting hidden layer activations for the processed slice.
    __global SCALAR_TYPE *__restrict hidden_mask_out,              // [OUTPUT] Propagated mask for the hidden layer.
    int batch_offset,                                              // [PARAM] The starting sample index for this chunk.
    int num_batch_samples,                                         // [PARAM] The number of samples to process in this chunk.
    int padded_input_dim,                                          // [PARAM] The padded dimension of the input layer.
    int padded_hidden_dim                                          // [PARAM] The padded dimension of the hidden layer.
);


/**
 * @brief (Node 5) Computes forward pass outputs for a single chunk of exits.
 *
 * Host Assumptions:
 * - The host provides buffer pointers and dimensions scoped to the *current chunk*.
 * - `problem_type_flag` (CCE/BCE) is uniform for all threads and dictates the math used.
 * - This kernel writes its results to slices of larger "partial" result buffers.
 */
__kernel void compute_chunk_outputs(
    __global const SCALAR_TYPE *__restrict hidden_buf,          // [INPUT] Hidden activations for the current batch slice.
    __global const void *__restrict targets_buf,                // [INPUT] Ground truth labels (cast internally based on problem type).
    __global const SCALAR_TYPE *__restrict hidden_mask,         // [INPUT] Mask for valid hidden activations (from Node 4).
    __global const SCALAR_TYPE *__restrict targets_mask,        // [INPUT] Mask for valid target labels.
    __global const SCALAR_TYPE *__restrict exit_weights_buf,    // [INPUT] Exit weights for this chunk.
    __global const SCALAR_TYPE *__restrict exit_biases_buf,     // [INPUT] Exit biases for this chunk.
    __global const SCALAR_TYPE *__restrict temps_buf,           // [INPUT] Temperature values for this chunk.
    __global SCALAR_TYPE *__restrict partial_logits_out,        // [OUTPUT] Slice for this chunk's unscaled logits.
    __global SCALAR_TYPE *__restrict partial_probs_out,         // [OUTPUT] Slice for this chunk's probabilities.
    __global SCALAR_TYPE *__restrict partial_loss_out,          // [OUTPUT] Slice for this chunk's per-sample loss.
    int problem_type_flag,                                      // [PARAM] PROBLEM_TYPE_CCE or PROBLEM_TYPE_BCE.
    int chunk_id,                                               // [PARAM] The logical ID of this chunk, used for output offseting.
    int param_offset,                                           // [PARAM] The starting parameter index for this chunk.
    int batch_size,                                             // [PARAM] The total number of samples in the batch.
    int hidden_dim,                                             // [PARAM] The dimension of the shared hidden layer.
    int output_classes,                                         // [PARAM] The number of output classes for each exit.
    int chunk_size                                              // [PARAM] The number of exits to process in this specific launch.
);


// in kernels.cl.h

/**
 * @brief (Node 6) Computes gradients for a single chunk of exits.
 *
 * Host Assumptions:
 * - The host must allocate and pass a __local memory buffer of sufficient size for the reduction
 *   (at least `work_group_size * sizeof(SCALAR_TYPE)`).
 * - This kernel has two types of gradient outputs:
 *   1. `grad_exit_*_out`: Final gradients for this chunk's parameters, ready for immediate/streaming Adam update.
 *   2. `partial_grad_h_out`: Partial gradient contribution for the shared layer, destined for the aggregation engine.
 */
__kernel void calculate_chunk_gradients(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY] Local memory for gradient reduction.
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [INPUT] Hidden activations for the current batch slice.
    __global const SCALAR_TYPE *__restrict probs_buf,        // [INPUT] Probabilities for this chunk (from Node 5).
    __global const void *__restrict targets_buf,             // [INPUT] Ground truth labels.
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [INPUT] Exit weights for this chunk.
    __global SCALAR_TYPE *__restrict partial_grad_h_out,     // [OUTPUT] Slice for this chunk's partial grad_H.
    __global SCALAR_TYPE *__restrict grad_exit_w_out,        // [OUTPUT] Final weight gradients for this chunk.
    __global SCALAR_TYPE *__restrict grad_exit_b_out,        // [OUTPUT] Final bias gradients for this chunk.
    int problem_type_flag,                                   // [PARAM] PROBLEM_TYPE_CCE or PROBLEM_TYPE_BCE.
    int chunk_id,                                            // [PARAM] The logical ID of this chunk, for indexing partial outputs.
    int param_offset,                                        // [PARAM] The starting parameter index for this chunk.
    int batch_size,                                          // [PARAM] The total number of samples in the batch.
    int hidden_dim,                                          // [PARAM] The dimension of the shared hidden layer.
    int output_classes,                                      // [PARAM] The number of output classes for each exit.
    int chunk_size                                           // [PARAM] The number of exits to process in this specific launch.
);


// in kernels.cl.h

/**
 * @brief (Node 7) Computes temperature gradients for a single chunk of exits.
 *
 * Host Assumptions:
 * - The host must allocate and pass a __local memory buffer of sufficient size for the reduction
 *   (at least `work_group_size * sizeof(SCALAR_TYPE)`).
 * - The host must provide the `targets_mask` to prevent gradient calculation on padded samples.
 * - Output is a partial gradient that must be aggregated by Node 8 before being used.
 */
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,                             // [MEMORY] Local memory for gradient reduction.
    __global const SCALAR_TYPE *__restrict unscaled_logits_buf, // [INPUT] Unscaled logits for this chunk (from Node 5).
    __global const SCALAR_TYPE *__restrict probs_buf,           // [INPUT] Probabilities for this chunk (from Node 5).
    __global const void *__restrict targets_buf,                // [INPUT] Ground truth labels.
    __global const SCALAR_TYPE *__restrict targets_mask,        // [INPUT] Mask to identify valid (non-padded) samples.
    __global const SCALAR_TYPE *__restrict temps_buf,           // [INPUT] Temperature values for this chunk.
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,    // [OUTPUT] Slice for this chunk's partial grad_temps.
    int problem_type_flag,                                      // [PARAM] PROBLEM_TYPE_CCE or PROBLEM_TYPE_BCE.
    int chunk_id,                                               // [PARAM] The logical ID of this chunk, for indexing.
    int param_offset,                                           // [PARAM] The starting parameter index for this chunk.
    int batch_size,                                             // [PARAM] The total number of samples in the batch.
    int output_classes,                                         // [PARAM] The number of output classes for each exit.
    int chunk_size                                              // [PARAM] The number of exits to process in this specific launch.
);


// --- Phase 4: Generic Aggregation Engine ---
//
// The following declarations represent the tiered implementations for the aggregation engine (Node 8).
// The host is responsible for choosing and launching the correct kernel based on `num_items_to_reduce`.
// The generic `aggregate_kernel` from the blueprint is now formalized into these concrete, callable kernels.
//

/**
 * @brief (Node 8, Tier 0: Identity) Kernel for the `num_items_to_reduce == 1` case. Performs a direct copy.
 */
__kernel void aggregate_identity(
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [INPUT] The full buffer of partial results (from one chunk).
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUTPUT] The final, single, "aggregated" result.
    int item_stride                                           // [PARAM] The number of elements per item (e.g., batch_size).
);

/**
 * @brief (Node 8, Tier 1: Register) High-performance reduction for a small number of items.
 */
__kernel void aggregate_register_reduce(
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [INPUT] The full buffer of partial results from all chunks.
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUTPUT] The final, single, aggregated result.
    int num_items_to_reduce,                                  // [PARAM] The number of chunks to aggregate.
    int item_stride,                                          // [PARAM] The number of elements per item (e.g., batch_size).
    int reduction_mode_flag                                   // [PARAM] AGG_MODE_SUM or AGG_MODE_AVERAGE.
);

/**
 * @brief (Node 8, Tier 2/3: Local) Workhorse reduction kernel for medium-to-large numbers of items.
 *
 * Host Assumptions:
 * - The host must allocate and pass a __local memory buffer of sufficient size for the reduction.
 */
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY] Local memory for the parallel reduction.
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [INPUT] The full buffer of partial results from all chunks.
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUTPUT] The final, single, aggregated result.
    int num_items_to_reduce,                                  // [PARAM] The number of chunks to aggregate.
    int item_stride,                                          // [PARAM] The number of elements per item (e.g., batch_size).
    int reduction_mode_flag                                   // [PARAM] AGG_MODE_SUM or AGG_MODE_AVERAGE.
);


// --- Phase 5: Final Global Backpropagation & Updates ---

/**
 * @brief (Node 9) Aggregates upstream gradients and backpropagates through the activation function.
 *
 * Host Assumptions:
 * - This kernel requires the final, aggregated `grad_H` from Node 8.
 * - It also requires the full, non-chunked `hidden_buf` from a pre-computation run of Node 4.
 */
__kernel void finalize_backprop_activation(
    __global const SCALAR_TYPE *__restrict final_grad_h_buf,  // [INPUT] Aggregated upstream gradients for the hidden layer.
    __global const SCALAR_TYPE *__restrict full_hidden_buf,   // [INPUT] Full hidden activations (for ReLU derivative).
    __global const SCALAR_TYPE *__restrict hidden_mask,       // [INPUT] Mask for valid hidden activations.
    __global SCALAR_TYPE *__restrict grad_pre_activation_out, // [OUTPUT] Gradient signal before the activation function.
    int padded_batch_size,                                    // [PARAM] Padded size of the batch dimension.
    int padded_hidden_dim                                     // [PARAM] Padded dimension of the hidden layer.
);

/**
 * @brief (Node 10) Calculates gradients for a dense layer's parameters (weights and biases).
 *
 * Host Assumptions:
 * - Operates on the final, batch-wide gradient from Node 9.
 * - `grad_weights` and `grad_biases` must be zeroed before calling.
 */
__kernel void calculate_dense_layer_gradients(
    __local SCALAR_TYPE *local_grad_w,                              // [MEMORY] Local memory for weight gradient reduction.
    __local SCALAR_TYPE *local_grad_b,                              // [MEMORY] Local memory for bias gradient reduction.
    __global const SCALAR_TYPE *__restrict input_buf,               // [INPUT] The original input to the layer.
    __global const SCALAR_TYPE *__restrict input_mask,              // [INPUT] Mask for valid samples.
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf, // [INPUT] Gradient signal from Node 9.
    __global SCALAR_TYPE *__restrict grad_weights,                  // [OUTPUT] Final gradient for the weight matrix.
    __global SCALAR_TYPE *__restrict grad_biases,                   // [OUTPUT] Final gradient for the bias vector.
    int padded_batch_size,                                          // [PARAM] Padded size of the batch dimension.
    int input_dim,                                                  // [PARAM] Padded dimension of the layer's input.
    int hidden_dim                                                  // [PARAM] Padded dimension of the layer's output.
);

/**
 * @brief (Node 11) Propagates the gradient back to the input of a layer (W.T * grad).
 *
 * Host Assumptions:
 * - Requires weights in a standard row-major layout for efficient matrix multiplication.
 */
__kernel void backprop_input_gradient(
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf, // [INPUT] Upstream gradient from Node 9.
    __global const SCALAR_TYPE *__restrict weights_standard_buf,    // [INPUT] Original layer weights in standard layout.
    __global SCALAR_TYPE *__restrict grad_input_buf,                // [OUTPUT] Downstream gradient for the previous layer.
    int padded_batch_size,                                          // [PARAM] Padded size of the batch dimension.
    int input_dim,                                                  // [PARAM] Padded dimension of the layer's input.
    int hidden_dim                                                  // [PARAM] Padded dimension of the layer's output.
);

/**
 * @brief (Node 12) Performs Adam optimizer update for a parameter buffer.
 *
 * Host Assumptions:
 * - This generic kernel is used for all parameter updates, both streaming (in-loop) and global (post-aggregation).
 * - For streaming updates, the host provides a `param_offset` to update a specific slice of the buffers.
 */
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad, // [INPUT] Gradient for the parameters.
    SCALAR_TYPE beta1,                           // [PARAM] Adam hyperparameter beta1.
    SCALAR_TYPE beta2,                           // [PARAM] Adam hyperparameter beta2.
    SCALAR_TYPE beta1_t,                         // [PARAM] Time-adjusted beta1 (beta1^t).
    SCALAR_TYPE beta2_t,                         // [PARAM] Time-adjusted beta2 (beta2^t).
    SCALAR_TYPE learning_rate,                   // [PARAM] Global learning rate.
    SCALAR_TYPE epsilon,                         // [PARAM] Adam epsilon for numerical stability.
    __global SCALAR_TYPE *__restrict param,      // [IN/OUT] Parameters to be updated.
    __global SCALAR_TYPE *__restrict m1,         // [IN/OUT] First moment vector (momentum).
    __global SCALAR_TYPE *__restrict m2,         // [IN/OUT] Second moment vector (RMSprop).
    int param_offset,                            // [PARAM] The starting element index for this update slice.
    int num_params_to_update                     // [PARAM] The number of elements to update in this slice.
);


/**
 * @brief (Node 13) Clamps temperature values within a specified range.
 *
 * Host Assumptions:
 * - A final utility kernel called after temperature parameters have been updated.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [IN/OUT] Temperature values to be clamped.
    SCALAR_TYPE min_temp,                       // [PARAM] Minimum allowed temperature value.
    SCALAR_TYPE max_temp,                       // [PARAM] Maximum allowed temperature value.
    int         num_exits                       // [PARAM] The total number of temperatures to clamp.
);

#endif // KERNELS_CL_H
