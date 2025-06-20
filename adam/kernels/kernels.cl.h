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

//--------------------------------------------------------------------------------
/// --- Common Parameter Definitions ---
///
/// @param local_mem            A pointer to __local memory. Its required size is specified
///                             in each kernel's signature as a [MEMORY size: ...] contract.
/// @param problem_type_flag    Selects math path (0 for PROBLEM_TYPE_CCE, 1 for PROBLEM_TYPE_BCE).
/// @param reduction_mode_flag  Selects aggregation op (0 for AGG_MODE_SUM, 1 for AGG_MODE_AVERAGE).
/// @param chunk_id             The logical identifier for a data chunk [>= 0].
/// @param param_offset         The starting index for a slice of a larger parameter buffer [>= 0].
/// @param chunk_size           The number of items (e.g., exits) processed by a chunk [> 0].
/// @param batch_offset         The starting sample index for a chunk of a larger batch [>= 0].
/// @param num_batch_samples    The number of samples to process in a given batch chunk [> 0].
/// @param full_batch_size      The total number of samples in the complete logical batch [> 0].
/// @param hidden_dim           The logical dimension of the hidden layer [> 0].
/// @param padded_hidden_dim    The physical memory dimension of the hidden layer [>= hidden_dim].
/// @param padded_input_dim     The physical memory dimension of the input layer (must be >= logical input dim).
/// @param output_classes       The number of output classes [> 0, must be <= C_TILE_SIZE in some kernels].
//--------------------------------------------------------------------------------

// --- Phase 3: Universal Chunk Processing (Forward Pass & Partial Grads) ---

/**
 * @brief (Node 4) Computes hidden activations for a slice of the input batch.
 * @contract The activation function is ReLU (Rectified Linear Unit).
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                                // [MEMORY size: SIMD_WIDTH * (1 + SIMD_WIDTH) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict input_buf,              // [IN]  Shape: (full_batch_size, padded_input_dim)
    __global const SCALAR_TYPE *__restrict input_mask,             // [IN]  Shape: (full_batch_size)
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf, // [IN]  Shape: (ceil(h_dim/SW), p_in_dim, SW)
    __global const SCALAR_TYPE *__restrict biases_buf,             // [IN]  Shape: (padded_hidden_dim)
    __global SCALAR_TYPE *__restrict hidden_out_buf,               // [OUT] Shape: (num_batch_samples, padded_hidden_dim)
    __global SCALAR_TYPE *__restrict hidden_mask_out,              // [OUT] Shape: (num_batch_samples)
    int batch_offset,                                              // [IN scalar: >= 0, The starting sample index for this chunk within input_buf]
    int num_batch_samples,                                         // [IN scalar: > 0, The number of samples to process from input_buf for this chunk]
    int padded_input_dim,                                          // [IN scalar: > 0, The physical (padded) dimension of the input layer]
    int padded_hidden_dim);                                        // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

/**
 * @brief (Node 5) Computes partial logits, probabilities, and loss for a chunk of exits.
 * @contract output_classes must be <= C_TILE_SIZE.
 */
__kernel void compute_chunk_outputs(
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (full_batch_size, padded_hidden_dim)
    __global const void *__restrict targets_buf,             // [IN]  Shape: (full_batch_size, ...)
    __global const SCALAR_TYPE *__restrict hidden_mask,      // [IN]  Shape: (full_batch_size)
    __global const SCALAR_TYPE *__restrict targets_mask,     // [IN]  Shape: (full_batch_size)
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [IN]  Shape: (total_num_exits, hidden_dim, out_classes)
    __global const SCALAR_TYPE *__restrict exit_biases_buf,  // [IN]  Shape: (total_num_exits, out_classes)
    __global const SCALAR_TYPE *__restrict temps_buf,        // [IN]  Shape: (total_num_exits)
    __global SCALAR_TYPE *__restrict partial_logits_out,     // [OUT] Shape: (total_num_chunks, ...)
    __global SCALAR_TYPE *__restrict partial_probs_out,      // [OUT] Shape: (total_num_chunks, ...)
    __global SCALAR_TYPE *__restrict partial_loss_out,       // [OUT] Shape: (total_num_chunks, ...)
    int problem_type_flag,                                   // [IN scalar: 0|1, Selects CCE or BCE math path]
    int chunk_id,                                            // [IN scalar: >= 0, The logical index of this chunk, used for writing outputs]
    int param_offset,                                        // [IN scalar: >= 0, The starting index of this chunk's parameters in global buffers]
    int full_batch_size,                                     // [IN scalar: > 0, The total number of samples in the complete logical batch]
    int hidden_dim,                                          // [IN scalar: > 0, The logical dimension of the hidden layer]
    int output_classes,                                      // [IN scalar: > 0 & <= C_TILE_SIZE, The number of output classes for this chunk's exits]
    int chunk_size,                                          // [IN scalar: > 0, The number of exits processed by this kernel invocation]
    int padded_hidden_dim);                                  // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

/**
 * @brief (Node 6) Computes partial gradients for exit parameters and hidden layer contributions.
 * @contract output_classes must be <= C_TILE_SIZE.
 */
__kernel void calculate_chunk_gradients(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict hidden_buf,        // [IN]  Shape: (full_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict partial_probs_buf, // [IN]  Shape: (total_num_chunks, ...)
    __global const void *__restrict targets_buf,              // [IN]  Shape: (full_batch_size, ...)
    __global const SCALAR_TYPE *__restrict exit_weights_buf,  // [IN]  Shape: (total_num_exits, ...)
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,  // [OUT] Shape: (num_chunks, batch_size, h_dim)
    __global SCALAR_TYPE *__restrict partial_grad_exit_w_out, // [OUT] Shape: (total_num_chunks, ...)
    __global SCALAR_TYPE *__restrict partial_grad_exit_b_out, // [OUT] Shape: (total_num_chunks, ...)
    int problem_type_flag,                                   // [IN scalar: 0|1, Selects CCE or BCE math path]
    int chunk_id,                                            // [IN scalar: >= 0, The logical index of this chunk, used for reading/writing outputs]
    int param_offset,                                        // [IN scalar: >= 0, The starting index of this chunk's parameters in global buffers]
    int full_batch_size,                                     // [IN scalar: > 0, The total number of samples in the complete logical batch]
    int hidden_dim,                                          // [IN scalar: > 0, The logical dimension of the hidden layer]
    int output_classes,                                      // [IN scalar: > 0 & <= C_TILE_SIZE, The number of output classes for this chunk's exits]
    int chunk_size,                                          // [IN scalar: > 0, The number of exits processed by this kernel invocation]
    int padded_hidden_dim);                                  // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

/**
 * @brief (Node 7) Computes partial temperature gradients for a chunk of exits.
 */
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,                            // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict partial_logits_buf, // [IN]  Shape: (total_num_chunks, ...)
    __global const SCALAR_TYPE *__restrict partial_probs_buf,  // [IN]  Shape: (total_num_chunks, ...)
    __global const void *__restrict targets_buf,               // [IN]  Shape: (full_batch_size, ...)
    __global const SCALAR_TYPE *__restrict targets_mask,       // [IN]  Shape: (full_batch_size)
    __global const SCALAR_TYPE *__restrict temps_buf,          // [IN]  Shape: (total_num_exits)
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,   // [OUT] Shape: (total_num_chunks, ...)
    int problem_type_flag,                                     // [IN scalar: 0|1, Selects CCE or BCE math path]
    int chunk_id,                                              // [IN scalar: >= 0, The logical index of this chunk, used for reading/writing outputs]
    int param_offset,                                          // [IN scalar: >= 0, The starting index of this chunk's parameters in global buffers]
    int full_batch_size,                                       // [IN scalar: > 0, The total number of samples in the complete logical batch]
    int output_classes,                                        // [IN scalar: > 0, The number of output classes for this chunk's exits]
    int chunk_size);                                           // [IN scalar: > 0, The number of exits processed by this kernel invocation]

// --- Phase 4: Data Layout Transformation ---

/**
 * @brief (Node 8) Transposes a matrix from chunk-major to item-major layout.
 * @contract Work dispatch must be a 2D grid with local size (C_TILE_SIZE, C_TILE_SIZE, 1).
 * @contract Global size must be a multiple of local size, covering the full matrix.
 */
__kernel void transpose_grad_h(
    __local SCALAR_TYPE *local_mem,                        // [MEMORY size: C_TILE_SIZE * (C_TILE_SIZE + 1) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict grad_h_aos_buf, // [IN]  Shape: (num_chunks, num_items)
    __global SCALAR_TYPE *__restrict grad_h_soa_buf,       // [OUT] Shape: (num_items, num_chunks)
    int num_chunks,                                        // [IN scalar: > 0, The number of chunks (input matrix height)]
    int num_items);                                        // [IN scalar: > 0, The number of items per chunk (input matrix width)]

// --- Phase 5 & 7: Generic Tiered Aggregation Engine ---

// Kernels for aggregating partial results, tiered by the number of items (`N`).
// They share a signature, but differ in their use of parameters and memory.
// The host orchestrator selects the appropriate kernel based on `N`.

/**
 * @brief (Tier 2/3: N is large) Reduces partial results using local memory.
 * This serves as the baseline contract for the aggregation engine.
 */
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (num_items_to_reduce, item_stride)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (item_stride)
    int num_items_to_reduce,                                  // [IN scalar: > 0, Number of partial results to reduce (N)]
    int item_stride,                                          // [IN scalar: > 0, Size of one item/tensor in elements]
    int reduction_mode_flag                                   // [IN scalar: 0|1, The aggregation operation (e.g., SUM or AVERAGE)]
);

/**
 * @brief (Tier 1: N is small) Reduces partial results using registers.
 * @contract Other parameters follow the baseline contract.
 */
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY (unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (num_items_to_reduce, item_stride)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (item_stride)
    int num_items_to_reduce,                                  // [IN scalar: > 0, Number of partial results to reduce (N)]
    int item_stride,                                          // [IN scalar: > 0, Size of one item/tensor in elements]
    int reduction_mode_flag);                                 // [IN scalar: 0|1, The aggregation operation (e.g., SUM or AVERAGE)]

/**
 * @brief (Tier 0: N=1) Identity pass-through for a single partial result.
 * @contract Input shape `partial_input_buf` is effectively (item_stride).
 */
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY (unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (item_stride)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (item_stride)
    int num_items_to_reduce,                                  // [IN scalar: unused, Present for signature compatibility]
    int item_stride,                                          // [IN scalar: > 0, The number of elements to copy]
    int reduction_mode_flag                                   // [IN scalar: unused, Present for signature compatibility]
);

// --- Phase 6: Streaming Shared Layer Backpropagation ---

/**
 * @brief (Node 10) Computes partial gradients for shared layer weights from a batch chunk.
 */
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict input_buf,        // [IN]  Shape: (full_batch_size, padded_input_dim)
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (full_batch_size, ...)
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [IN]  Shape: (full_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict input_mask,       // [IN]  Shape: (full_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,    // [OUT] Shape: (num_batch_chunks, p_in_dim, p_h_dim)
    int batch_offset,                                        // [IN scalar: >= 0, The starting sample index for this chunk within the batch]
    int num_batch_samples,                                   // [IN scalar: > 0, The number of samples to process in this chunk]
    int chunk_id,                                            // [IN scalar: >= 0, The logical index of this batch chunk for writing partial results]
    int padded_input_dim,                                    // [IN scalar: > 0, The physical (padded) dimension of the input layer]
    int padded_hidden_dim);                                  // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

/**
 * @brief (Node 11) Computes partial gradients for shared layer biases from a batch chunk.
 */
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (full_batch_size, ...)
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [IN]  Shape: (full_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict input_mask,       // [IN]  Shape: (full_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,    // [OUT] Shape: (num_batch_chunks, padded_hidden_dim)
    int batch_offset,                                        // [IN scalar: >= 0, The starting sample index for this chunk within the batch]
    int num_batch_samples,                                   // [IN scalar: > 0, The number of samples to process in this chunk]
    int chunk_id,                                            // [IN scalar: >= 0, The logical index of this batch chunk for writing partial results]
    int padded_hidden_dim);                                  // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

// --- Phase 8: Finalization & Dispatch ---

/**
 * @brief (Node 14) Applies Adam optimizer update to a slice of a parameter buffer.
 */
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad, // [IN]  Final, aggregated gradients for this parameter slice.
    SCALAR_TYPE beta1,                           // [IN scalar: (0,1), The exponential decay rate for the 1st moment estimates]
    SCALAR_TYPE beta2,                           // [IN scalar: (0,1), The exponential decay rate for the 2nd moment estimates]
    SCALAR_TYPE beta1_t,                         // [IN scalar: (0,1), beta1 to the power of the current timestep]
    SCALAR_TYPE beta2_t,                         // [IN scalar: (0,1), beta2 to the power of the current timestep]
    SCALAR_TYPE learning_rate,                   // [IN scalar: > 0, The learning rate (step size)]
    SCALAR_TYPE epsilon,                         // [IN scalar: > 0, A small value to prevent division by zero]
    __global SCALAR_TYPE *__restrict param,      // [IN/OUT] The parameters to be updated.
    __global SCALAR_TYPE *__restrict m1,         // [IN/OUT] The 1st moment vector (momentum).
    __global SCALAR_TYPE *__restrict m2,         // [IN/OUT] The 2nd moment vector (velocity).
    int param_offset,                            // [IN scalar: >= 0, The starting index to apply the update within the global buffers]
    int num_params_to_update);                   // [IN scalar: > 0, The number of parameters to update in this slice]

/**
 * @brief (Node 15) Clamps temperature parameters within a [min, max] range.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [IN/OUT] The temperature parameters to be clamped.
    SCALAR_TYPE min_temp,                       // [IN scalar: any, The minimum allowed temperature value]
    SCALAR_TYPE max_temp,                       // [IN scalar: any, The maximum allowed temperature value, must be >= min_temp]
    int         num_exits);                     // [IN scalar: > 0, The total number of temperature parameters to clamp]

#endif // KERNELS_CL_H
