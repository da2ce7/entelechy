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

// Computes the flattened 1D index for a hidden unit, assuming a row-major memory layout.
#define GET_PHYSICAL_HIDDEN_IDX(b, h, padded_h_dim) ((b) * (padded_h_dim) + (h))

// --- Host-configurable Flags and Enums ---

// Used by kernels to select loss/gradient math (e.g., Softmax vs. Sigmoid).
#define PROBLEM_TYPE_CCE 0
#define PROBLEM_TYPE_BCE 1

// Used by the generic `aggregate_kernel` to select the reduction operation.
#define AGG_MODE_SUM 0
#define AGG_MODE_AVERAGE 1

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
/// @param local_mem             A pointer to __local memory.
/// @param problem_type_flag     Selects math path (0 for PROBLEM_TYPE_CCE, 1 for PROBLEM_TYPE_BCE).
/// @param reduction_mode_flag   Selects aggregation op (0 for AGG_MODE_SUM, 1 for AGG_MODE_AVERAGE).
/// @param exit_chunk_id         The logical identifier for an exit chunk [>= 0].
/// @param exit_param_offset     The starting index for a slice of an exit-related parameter buffer [>= 0].
/// @param num_exits_in_chunk    The number of exits processed by an exit chunk [> 0].
/// @param batch_chunk_id        The logical identifier for a batch chunk [>= 0].
/// @param batch_offset          The starting sample index for a batch chunk [>= 0].
/// @param num_batch_samples     The number of samples to process in a given batch chunk [> 0].
/// @param class_offset          The starting class index for a class chunk [>= 0].
/// @param num_classes_in_chunk  The number of classes to process in a class chunk [> 0].
/// @param total_batch_size      The total number of samples in the complete logical batch [> 0].
/// @param hidden_dim            The logical dimension of the hidden layer [> 0].
/// @param padded_hidden_dim     The physical memory dimension (row pitch) of the hidden layer, pre-calculated on the host as a multiple of SIMD_WIDTH [>= hidden_dim].
/// @param total_exits           The total number of exits for the full problem [> 0].
/// @param total_output_classes  The total number of output classes for the full problem [> 0].
//--------------------------------------------------------------------------------

// --- Phase 4: Shared Layer Forward Pass ---

/**
 * @brief (Node 4) Computes hidden activations for a slice of the input batch.
 * @contract The activation function is ReLU (Rectified Linear Unit).
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                                // [MEMORY size: SIMD_WIDTH * (1 + SIMD_WIDTH) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict input_buf,              // [IN]  Shape: (total_batch_size, padded_input_dim)
    __global const SCALAR_TYPE *__restrict input_mask,             // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf, // [IN]  Shape: (ceil(h_dim/SW), p_in_dim, SW)
    __global const SCALAR_TYPE *__restrict biases_buf,             // [IN]  Shape: (padded_hidden_dim)
    __global SCALAR_TYPE *__restrict hidden_out_buf,               // [OUT] Shape: (num_batch_samples, padded_hidden_dim)
    __global SCALAR_TYPE *__restrict hidden_mask_out,              // [OUT] Shape: (num_batch_samples)
    int batch_offset,                                              // [IN scalar: >= 0, The starting sample index for this chunk within input_buf]
    int num_batch_samples,                                         // [IN scalar: > 0, The number of samples to process from input_buf for this chunk]
    int padded_input_dim,                                          // [IN scalar: > 0, The physical (padded) dimension of the input layer]
    int padded_hidden_dim);                                        // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

// --- Phase 5: Exit Layer Forward Pass & Gradient Computation ---

/**
 * @brief (Node 5) Computes a chunk of raw logits for a slice of exits and classes.
 * @contract First stage of the three-part output computation. This is fully streamable.
 *           `num_classes_in_chunk` should be <= a hardware-friendly tile size.
 */
__kernel void compute_logits_chunk(
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict hidden_mask,      // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [IN]  Shape: (total_exits, hidden_dim, total_output_classes)
    __global const SCALAR_TYPE *__restrict exit_biases_buf,  // [IN]  Shape: (total_exits, total_output_classes)
    __global SCALAR_TYPE *__restrict full_logits_out,        // [OUT] Shape: (total_exits, total_batch_size, total_output_classes)
    int exit_chunk_id,                                       // [IN scalar: >= 0, The logical index of the EXIT chunk]
    int exit_param_offset,                                   // [IN scalar: >= 0, The starting EXIT index for this chunk's parameters]
    int num_exits_in_chunk,                                  // [IN scalar: > 0, The number of exits processed by this kernel invocation]
    int class_offset,                                        // [IN scalar: >= 0, The starting CLASS index for this chunk]
    int num_classes_in_chunk,                                // [IN scalar: > 0, The number of classes processed by this kernel invocation]
    int total_batch_size,                                    // [IN scalar: > 0, The total number of samples in the logical batch]
    int hidden_dim,                                          // [IN scalar: > 0, The logical dimension of the hidden layer]
    int padded_hidden_dim,                                   // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]
    int total_output_classes);                               // [IN scalar: > 0, The total number of classes in the full problem]

/**
 * @brief (Node 6) Reduces the full logit buffer to find normalization terms for Softmax.
 * @contract Second stage of the three-part output computation. This kernel is a
 *           synchronization point and is NOT streamable over classes. It runs once
 *           after all `compute_logits_chunk` calls are complete.
 */
__kernel void reduce_logits_for_softmax(
    __global const SCALAR_TYPE *__restrict full_logits_buf, // [IN]  Shape: (total_exits, total_batch_size, total_output_classes)
    __global const SCALAR_TYPE *__restrict temps_buf,       // [IN]  Shape: (total_exits)
    __global SCALAR_TYPE *__restrict softmax_params_out,    // [OUT] Shape: (total_exits, total_batch_size, 2) -> [max_logit, sum_exp]
    int total_exits,                                        // [IN scalar: > 0, The total number of exits in the full problem]
    int total_batch_size,                                   // [IN scalar: > 0, The total number of samples in the logical batch]
    int total_output_classes);                              // [IN scalar: > 0, The total number of classes in the full problem]

/**
 * @brief (Node 7a - CCE Path) Computes probabilities for a class chunk and scatters CCE loss values.
 * @contract This kernel is only for PROBLEM_TYPE_CCE. Host must zero-initialize final_loss_out.
 */
__kernel void compute_probs_loss_cce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf,    // [IN]
    __global const SCALAR_TYPE *__restrict softmax_params_buf, // [IN]
    __global const SCALAR_TYPE *__restrict temps_buf,          // [IN]
    __global const int *__restrict targets_cce_buf,            // [IN] Specialized for int targets
    __global const SCALAR_TYPE *__restrict targets_mask,       // [IN]
    __global SCALAR_TYPE *__restrict partial_probs_out,        // [OUT] Shape: (total_exits, total_batch_size, total_output_classes)
    __global SCALAR_TYPE *__restrict final_loss_out,           // [OUT] Shape: (total_exits, total_batch_size)
    int exit_chunk_id,                                         // [IN scalar] EXIT chunk index
    int exit_param_offset,                                     // [IN scalar] Starting EXIT index
    int num_exits_in_chunk,                                    // [IN scalar] Num exits in this chunk
    int class_offset,                                          // [IN scalar] Starting CLASS index
    int num_classes_in_chunk,                                  // [IN scalar]
    int total_batch_size,                                      // [IN scalar]
    int total_output_classes);                                 // [IN scalar]

/**
 * @brief (Node 7b - BCE Path) Computes probabilities for a class chunk and a PARTIAL BCE loss.
 * @contract This kernel is only for PROBLEM_TYPE_BCE. The partial_loss_out must be aggregated.
 */
__kernel void compute_probs_loss_bce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf, // [IN]
    __global const SCALAR_TYPE *__restrict temps_buf,       // [IN]
    __global const SCALAR_TYPE *__restrict targets_bce_buf, // [IN] Specialized for SCALAR_TYPE targets
    __global const SCALAR_TYPE *__restrict targets_mask,    // [IN]
    __global SCALAR_TYPE *__restrict partial_probs_out,     // [OUT] Shape: (total_exits, total_batch_size, total_output_classes)
    __global SCALAR_TYPE *__restrict partial_loss_out,      // [OUT] Shape: (num_class_chunks, total_exits, total_batch_size)
    int exit_chunk_id,                                      // [IN scalar] EXIT chunk index
    int exit_param_offset,                                  // [IN scalar] Starting EXIT index
    int num_exits_in_chunk,                                 // [IN scalar] Num exits in this chunk
    int class_chunk_id,                                     // [IN scalar] CLASS chunk index
    int class_offset,                                       // [IN scalar] Starting CLASS index
    int num_classes_in_chunk,                               // [IN scalar]
    int total_batch_size,                                   // [IN scalar]
    int total_output_classes,                               // [IN scalar]
    int total_exits);                                       // [IN scalar]

// --- Phase 8-10: Parallel Gradient Computation ---

/**
 * @brief (Node 8) Computes partial exit param gradients (W, B) for a class chunk.
 * @contract This kernel is responsible ONLY for the local exit layer gradients.
 *           It computes the error signal `(prob - target)` on the fly.
 *           This kernel can run in parallel with Nodes 9 and 10.
 */
__kernel void calculate_exit_param_grads_chunk(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict hidden_buf,        // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict partial_probs_buf, // [IN]  Shape: (total_exits, total_batch_size, total_output_classes)
    __global const void *__restrict targets_buf,              // [IN]  Shape: (total_batch_size, ...) [Type depends on problem]
    __global SCALAR_TYPE *__restrict partial_grad_exit_w_out, // [OUT] Shape: (num_class_chunks, total_exits, h_dim, total_output_classes)
    __global SCALAR_TYPE *__restrict partial_grad_exit_b_out, // [OUT] Shape: (num_class_chunks, total_exits, total_output_classes)
    int problem_type_flag,                                    // [IN scalar: 0|1]
    int exit_chunk_id,                                        // [IN scalar: >= 0, EXIT chunk index]
    int exit_param_offset,                                    // [IN scalar: >= 0, Starting EXIT index for this chunk]
    int num_exits_in_chunk,                                   // [IN scalar: > 0, Num EXITS in this chunk]
    int class_chunk_id,                                       // [IN scalar: >= 0, CLASS chunk index, for indexing output buffer]
    int class_offset,                                         // [IN scalar: >= 0, Starting CLASS index for this chunk]
    int num_classes_in_chunk,                                 // [IN scalar: > 0]
    int total_batch_size,                                     // [IN scalar: > 0]
    int hidden_dim,                                           // [IN scalar: > 0]
    int padded_hidden_dim,                                    // [IN scalar: > 0]
    int total_output_classes,                                 // [IN scalar: > 0]
    int total_exits);                                         // [IN scalar: > 0]

/**
 * @brief (Node 9) Computes the partial upstream gradient for the hidden layer.
 * @contract This kernel is responsible ONLY for the `Grad_H` backpropagation.
 *           It computes the error signal `(prob - target)` on the fly.
 *           This kernel can run in parallel with Nodes 8 and 10.
 */
__kernel void backprop_error_to_hidden_chunk(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict partial_probs_buf, // [IN]  Shape: (total_exits, total_batch_size, total_output_classes)
    __global const void *__restrict targets_buf,              // [IN]  Shape: (total_batch_size, ...) [Type depends on problem]
    __global const SCALAR_TYPE *__restrict exit_weights_buf,  // [IN]  Shape: (total_exits, hidden_dim, total_output_classes)
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,  // [OUT] Shape: (num_class_chunks, total_exits, batch_size, h_dim)
    int problem_type_flag,                                    // [IN scalar: 0|1]
    int exit_chunk_id,                                        // [IN scalar: >= 0, EXIT chunk index]
    int exit_param_offset,                                    // [IN scalar: >= 0, Starting EXIT index for this chunk]
    int num_exits_in_chunk,                                   // [IN scalar: > 0, Num EXITS in this chunk]
    int class_chunk_id,                                       // [IN scalar: >= 0, CLASS chunk index, for indexing output buffer]
    int class_offset,                                         // [IN scalar: >= 0, Starting CLASS index for this chunk]
    int num_classes_in_chunk,                                 // [IN scalar: > 0]
    int total_batch_size,                                     // [IN scalar: > 0]
    int hidden_dim,                                           // [IN scalar: > 0]
    int total_output_classes,                                 // [IN scalar: > 0]
    int total_exits);                                         // [IN scalar: > 0]

/**
 * @brief (Node 10) Computes partial temperature gradients for a chunk of exits and classes.
 * @contract Produces a partial temperature gradient per class-chunk that must be aggregated.
 *           This kernel can run in parallel with Nodes 8 and 9.
 */
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict full_logits_buf,   // [IN]  Shape: (total_exits, total_batch_size, total_output_classes)
    __global const SCALAR_TYPE *__restrict partial_probs_buf, // [IN]  Shape: (total_exits, total_batch_size, total_output_classes)
    __global const void *__restrict targets_buf,              // [IN]  Shape: (total_batch_size, ...)
    __global const SCALAR_TYPE *__restrict targets_mask,      // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict temps_buf,         // [IN]  Shape: (total_exits)
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,  // [OUT] Shape: (num_class_chunks, total_exits)
    int problem_type_flag,                                    // [IN scalar: 0|1]
    int exit_chunk_id,                                        // [IN scalar: >= 0, EXIT chunk index]
    int exit_param_offset,                                    // [IN scalar: >= 0, Starting EXIT index]
    int num_exits_in_chunk,                                   // [IN scalar: > 0, Num exits in this chunk]
    int class_chunk_id,                                       // [IN scalar: >= 0, CLASS chunk index]
    int class_offset,                                         // [IN scalar: >= 0, Starting CLASS index]
    int num_classes_in_chunk,                                 // [IN scalar: > 0, The number of classes processed by this chunk]
    int total_batch_size,                                     // [IN scalar: > 0]
    int total_output_classes,                                 // [IN scalar: > 0]
    int total_exits);                                         // [IN scalar: > 0]

// --- Phase 11: Data Layout Transformation ---

/**
 * @brief (Node 11) Transposes a matrix from row-major to column-major for subsequent aggregation.
 * @contract Work dispatch must be a 2D grid with local size (C_TILE_SIZE, C_TILE_SIZE, 1).
 * @contract Global size must be a multiple of local size, covering the full matrix.
 */
__kernel void transpose_grad_h(
    __local SCALAR_TYPE *local_mem,                        // [MEMORY size: C_TILE_SIZE * (C_TILE_SIZE + 1) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict grad_h_aos_buf, // [IN]  Shape: (num_source_rows, num_source_cols)
    __global SCALAR_TYPE *__restrict grad_h_soa_buf,       // [OUT] Shape: (num_source_cols, num_source_rows)
    int num_source_rows,                                   // [IN scalar: > 0, The number of rows in the source matrix (e.g., class_chunks)]
    int num_source_cols);                                  // [IN scalar: > 0, The number of columns in the source matrix (e.g., items per chunk)]

// --- Phase 12 & 15: Generic Tiered Aggregation Engine ---

/**
 * @brief (Node 12, 15) (Tier 0: N=1) Identity pass-through for a single partial result.
 */
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY (unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (elements_per_partial)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (elements_per_partial)
    int num_partials_to_reduce,                               // [IN scalar: unused, Present for signature compatibility]
    int elements_per_partial,                                 // [IN scalar: > 0, The number of elements to copy]
    int reduction_mode_flag                                   // [IN scalar: unused, Present for signature compatibility]
);

/**
 * @brief (Node 12, 15) (Tier 1: N is small) Reduces partial results using registers.
 */
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY (unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (num_partials_to_reduce, elements_per_partial)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (elements_per_partial)
    int num_partials_to_reduce,                               // [IN scalar: > 0, Number of partial results to reduce (N)]
    int elements_per_partial,                                 // [IN scalar: > 0, Size of one item/tensor in elements]
    int reduction_mode_flag);                                 // [IN scalar: 0|1, The aggregation operation (e.g., SUM or AVERAGE)]

/**
 * @brief (Node 12, 15) (Tier 2: N is large) Reduces partial results using local memory.
 */
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (num_partials_to_reduce, elements_per_partial)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (elements_per_partial)
    int num_partials_to_reduce,                               // [IN scalar: > 0, Number of partial results to reduce (N)]
    int elements_per_partial,                                 // [IN scalar: > 0, Size of one item/tensor in elements]
    int reduction_mode_flag                                   // [IN scalar: 0|1, The aggregation operation (e.g., SUM or AVERAGE)]
);

// --- Phase 13-14: Streaming Shared Layer Backpropagation ---

/**
 * @brief (Node 13) Computes partial gradients for shared layer weights from a batch chunk.
 */
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict input_buf,        // [IN]  Shape: (total_batch_size, padded_input_dim)
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (total_batch_size, ...)
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict input_mask,       // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,    // [OUT] Shape: (num_batch_chunks, p_in_dim, p_h_dim)
    int batch_offset,                                        // [IN scalar: >= 0, The starting sample index for this chunk within the batch]
    int num_batch_samples,                                   // [IN scalar: > 0, The number of samples to process in this chunk]
    int batch_chunk_id,                                      // [IN scalar: >= 0, The logical index of this batch chunk for writing partial results]
    int padded_input_dim,                                    // [IN scalar: > 0, The physical (padded) dimension of the input layer]
    int padded_hidden_dim);                                  // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

/**
 * @brief (Node 14) Computes partial gradients for shared layer biases from a batch chunk.
 */
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (total_batch_size, ...)
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict input_mask,       // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,    // [OUT] Shape: (num_batch_chunks, padded_hidden_dim)
    int batch_offset,                                        // [IN scalar: >= 0, The starting sample index for this chunk within the batch]
    int num_batch_samples,                                   // [IN scalar: > 0, The number of samples to process in this chunk]
    int batch_chunk_id,                                      // [IN scalar: >= 0, The logical index of this batch chunk for writing partial results]
    int padded_hidden_dim);                                  // [IN scalar: > 0, The physical (padded) dimension of the hidden layer]

// --- Phase 16-18: Finalization & Dispatch ---

/**
 * @brief (Node 17) Applies Adam optimizer update to a slice of a parameter buffer.
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
 * @brief (Node 18) Clamps temperature parameters within a [min, max] range.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [IN/OUT] The temperature parameters to be clamped.
    SCALAR_TYPE min_temp,                       // [IN scalar: any, The minimum allowed temperature value]
    SCALAR_TYPE max_temp,                       // [IN scalar: any, The maximum allowed temperature value, must be >= min_temp]
    int         total_exits);                           // [IN scalar: > 0, The total number of temperature parameters to clamp]

#endif // KERNELS_CL_H
