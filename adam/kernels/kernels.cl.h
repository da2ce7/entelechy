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

// --- Phase 3: Universal Chunk Processing (Forward Pass & Partial Grads) ---

/** (Node 4) */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                                // [MEMORY size: Kernel-dependent for efficient tiling]
    __global const SCALAR_TYPE *__restrict input_buf,              // [INPUT  shape: (full_batch_size, padded_input_dim)]
    __global const SCALAR_TYPE *__restrict input_mask,             // [INPUT  shape: (full_batch_size)]
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf, // [INPUT  shape: (ceil(h_dim/SW), p_in_dim, SW)]
    __global const SCALAR_TYPE *__restrict biases_buf,             // [INPUT  shape: (padded_hidden_dim)]
    __global SCALAR_TYPE *__restrict hidden_out_buf,               // [OUTPUT shape: (num_batch_samples, padded_hidden_dim)]
    __global SCALAR_TYPE *__restrict hidden_mask_out,              // [OUTPUT shape: (num_batch_samples)]
    int batch_offset,                                              // [PARAM  scalar: >= 0, Starting sample index for this chunk]
    int num_batch_samples,                                         // [PARAM  scalar: > 0, Number of samples to process in this chunk]
    int padded_input_dim,                                          // [PARAM  scalar: >= input_dim]
    int padded_hidden_dim                                          // [PARAM  scalar: >= hidden_dim]
);

/** (Node 5) */
__kernel void compute_chunk_outputs(
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [INPUT  shape: (full_batch_size, padded_hidden_dim), in SIMD-block layout]
    __global const void *__restrict targets_buf,             // [INPUT  shape: (full_batch_size, ...), Cast internally based on problem type]
    __global const SCALAR_TYPE *__restrict hidden_mask,      // [INPUT  shape: (full_batch_size)]
    __global const SCALAR_TYPE *__restrict targets_mask,     // [INPUT  shape: (full_batch_size)]
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [INPUT  shape: (total_num_exits, hidden_dim, out_classes), This kernel reads slice at param_offset]
    __global const SCALAR_TYPE *__restrict exit_biases_buf,  // [INPUT  shape: (total_num_exits, out_classes), This kernel reads slice at param_offset]
    __global const SCALAR_TYPE *__restrict temps_buf,        // [INPUT  shape: (total_num_exits), This kernel reads slice at param_offset]
    __global SCALAR_TYPE *__restrict partial_logits_out,     // [OUTPUT shape: (total_num_chunks, ...), This kernel writes its result to slice chunk_id]
    __global SCALAR_TYPE *__restrict partial_probs_out,      // [OUTPUT shape: (total_num_chunks, ...), This kernel writes its result to slice chunk_id]
    __global SCALAR_TYPE *__restrict partial_loss_out,       // [OUTPUT shape: (total_num_chunks, ...), This kernel writes its result to slice chunk_id]
    int problem_type_flag,                                   // [PARAM  scalar: PROBLEM_TYPE_CCE or PROBLEM_TYPE_BCE]
    int chunk_id,                                            // [PARAM  scalar: >= 0, The logical ID of this chunk]
    int param_offset,                                        // [PARAM  scalar: >= 0, Starting parameter index for this chunk]
    int full_batch_size,                                     // [PARAM  scalar: > 0]
    int hidden_dim,                                          // [PARAM  scalar: > 0, The logical dimension of the hidden layer (used for loop bounds)]
    int output_classes,                                      // [PARAM  scalar: > 0 | **CONTRACT: Must be <= C_TILE_SIZE** due to internal stack allocation]
    int chunk_size,                                          // [PARAM  scalar: > 0, Number of exits in this chunk]
    int padded_hidden_dim                                    // [PARAM  scalar: >= hidden_dim, The physical dimension used for memory layout calculations]
);

/** (Node 6) */
__kernel void calculate_chunk_gradients(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: Kernel-dependent for parallel reduction]
    __global const SCALAR_TYPE *__restrict hidden_buf,        // [INPUT  shape: (full_batch_size, padded_hidden_dim), in SIMD-block layout]
    __global const SCALAR_TYPE *__restrict partial_probs_buf, // [INPUT  shape: (total_num_chunks, ...), Reads partial probabilities from slice chunk_id]
    __global const void *__restrict targets_buf,              // [INPUT  shape: (full_batch_size, ...), Cast internally]
    __global const SCALAR_TYPE *__restrict exit_weights_buf,  // [INPUT  shape: (total_num_exits, ...), Reads slice at param_offset]
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,  // [OUTPUT shape: (num_chunks, batch_size, h_dim), AoS layout]
    __global SCALAR_TYPE *__restrict partial_grad_exit_w_out, // [OUTPUT shape: (total_num_chunks, ...), Writes to slice chunk_id]
    __global SCALAR_TYPE *__restrict partial_grad_exit_b_out, // [OUTPUT shape: (total_num_chunks, ...), Writes to slice chunk_id]
    int problem_type_flag,                                    // [PARAM  scalar: PROBLEM_TYPE_CCE or PROBLEM_TYPE_BCE]
    int chunk_id,                                             // [PARAM  scalar: >= 0, The logical ID of this chunk]
    int param_offset,                                         // [PARAM  scalar: >= 0, Starting parameter index for this chunk]
    int full_batch_size,                                      // [PARAM  scalar: > 0]
    int hidden_dim,                                           // [PARAM  scalar: > 0, The logical dimension of the hidden layer (used for loop bounds)]
    int output_classes,                                       // [PARAM  scalar: > 0 | **CONTRACT: Must be <= C_TILE_SIZE** due to internal stack allocation]
    int chunk_size,                                           // [PARAM  scalar: > 0, Number of exits in this chunk]
    int padded_hidden_dim                                     // [PARAM  scalar: >= hidden_dim, The physical dimension used for memory layout calculations]
);

/** (Node 7) */
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,                            // [MEMORY size: Kernel-dependent for parallel reduction]
    __global const SCALAR_TYPE *__restrict partial_logits_buf, // [INPUT  shape: (total_num_chunks, ...), Reads partial logits from slice chunk_id]
    __global const SCALAR_TYPE *__restrict partial_probs_buf,  // [INPUT  shape: (total_num_chunks, ...), Reads partial probabilities from slice chunk_id]
    __global const void *__restrict targets_buf,               // [INPUT  shape: (full_batch_size, ...), Cast internally]
    __global const SCALAR_TYPE *__restrict targets_mask,       // [INPUT  shape: (full_batch_size)]
    __global const SCALAR_TYPE *__restrict temps_buf,          // [INPUT  shape: (total_num_exits), Reads slice at param_offset]
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,   // [OUTPUT shape: (total_num_chunks, ...), Writes to slice chunk_id]
    int problem_type_flag,                                     // [PARAM  scalar: PROBLEM_TYPE_CCE or PROBLEM_TYPE_BCE]
    int chunk_id,                                              // [PARAM  scalar: >= 0, The logical ID of this chunk]
    int param_offset,                                          // [PARAM  scalar: >= 0, Starting parameter index for this chunk]
    int full_batch_size,                                       // [PARAM  scalar: > 0]
    int output_classes,                                        // [PARAM  scalar: > 0]
    int chunk_size                                             // [PARAM  scalar: > 0, Number of exits in this chunk]
);

// --- Phase 4: Data Layout Transformation ---

/**
 * (Node 8)
 *
 * WORK DISPATCH: A 2D grid of work-groups must be used.
 *  - Local Size:  (C_TILE_SIZE, C_TILE_SIZE, 1)
 *  - Global Size: Must be a multiple of the local size, sufficient to cover the
 *                 entire matrix. Recommended calculation by host:
 *                 `global_x = ceil(num_items / C_TILE_SIZE) * C_TILE_SIZE`
 *                 `global_y = ceil(num_chunks / C_TILE_SIZE) * C_TILE_SIZE`
 */
__kernel void transpose_grad_h(
    __local SCALAR_TYPE *local_mem,                        // [MEMORY size: C_TILE_SIZE * (C_TILE_SIZE + 1) for optimal performance]
    __global const SCALAR_TYPE *__restrict grad_h_aos_buf, // [INPUT  shape: (num_chunks, num_items)]
    __global SCALAR_TYPE *__restrict grad_h_soa_buf,       // [OUTPUT shape: (num_items, num_chunks)]
    int num_chunks,                                        // [PARAM  scalar: > 0, The height of the input matrix to transpose]
    int num_items                                          // [PARAM  scalar: > 0, The width of the input matrix (batch_size * hidden_dim)]
);

// --- Phase 5 & 7: Generic Tiered Aggregation Engine ---

/** (Node 9 & 12, Tier 0) */
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY (Unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [INPUT  shape: (item_stride), The single partial result to pass through]
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUTPUT shape: (item_stride), The destination buffer]
    int num_items_to_reduce,                                  // [PARAM  (Unused), Implicitly 1]
    int item_stride,                                          // [PARAM  scalar: > 0, The size of one item/tensor in elements]
    int reduction_mode_flag                                   // [PARAM  (Unused)]
);

/** (Node 9 & 12, Tier 1) */
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY (Unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [INPUT  shape: (num_items_to_reduce, item_stride) with SoA layout]
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUTPUT shape: (item_stride), The single, reduced output item]
    int num_items_to_reduce,                                  // [PARAM  scalar: > 1, Number of partial results to reduce]
    int item_stride,                                          // [PARAM  scalar: > 0, The size of one item/tensor in elements]
    int reduction_mode_flag                                   // [PARAM  scalar: AGG_MODE_SUM or AGG_MODE_AVERAGE]
);

/** (Node 9 & 12, Tier 2-3) */
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: Workgroup-dependent for parallel reduction]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [INPUT  shape: (num_items_to_reduce, item_stride) with SoA layout]
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUTPUT shape: (item_stride), The single, reduced output item]
    int num_items_to_reduce,                                  // [PARAM  scalar: > 1, Number of partial results to reduce]
    int item_stride,                                          // [PARAM  scalar: > 0, The size of one item/tensor in elements]
    int reduction_mode_flag                                   // [PARAM  scalar: AGG_MODE_SUM or AGG_MODE_AVERAGE]
);

// --- Phase 6: Streaming Shared Layer Backpropagation ---

/** (Node 10) Calculates PARTIAL gradients for the shared layer WEIGHTS for a CHUNK of the BATCH. */
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: Kernel-dependent for parallel reduction]
    __global const SCALAR_TYPE *__restrict input_buf,        // [INPUT  shape: (full_batch_size, padded_input_dim)]
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [INPUT  shape: (full_batch_size, ...), SIMD-aware layout]
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [INPUT  shape: (full_batch_size, padded_hidden_dim), Row-major]
    __global const SCALAR_TYPE *__restrict input_mask,       // [INPUT  shape: (full_batch_size)]
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,    // [OUTPUT shape: (num_batch_chunks, padded_input_dim, padded_hidden_dim)]
    int batch_offset,                                        // [PARAM  scalar: >= 0, Starting sample index for this chunk]
    int num_batch_samples,                                   // [PARAM  scalar: > 0, Number of samples to process in this chunk]
    int chunk_id,                                            // [PARAM  scalar: >= 0, The logical ID for this batch chunk]
    int padded_input_dim,                                    // [PARAM  scalar: >= input_dim, The physical dimension of the input layer]
    int padded_hidden_dim                                    // [PARAM  scalar: >= hidden_dim, The physical dimension of the hidden layer]
);

/** (Node 11) Calculates PARTIAL gradients for the shared layer BIASES for a CHUNK of the BATCH. */
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: Kernel-dependent for parallel reduction]
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [INPUT  shape: (full_batch_size, ...), SIMD-aware layout]
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [INPUT  shape: (full_batch_size, padded_hidden_dim), Row-major]
    __global const SCALAR_TYPE *__restrict input_mask,       // [INPUT  shape: (full_batch_size)]
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,    // [OUTPUT shape: (num_batch_chunks, padded_hidden_dim)]
    int batch_offset,                                        // [PARAM  scalar: >= 0, Starting sample index for this chunk]
    int num_batch_samples,                                   // [PARAM  scalar: > 0, Number of samples to process in this chunk]
    int chunk_id,                                            // [PARAM  scalar: >= 0, The logical ID for this batch chunk]
    int padded_hidden_dim                                    // [PARAM  scalar: >= hidden_dim, The physical dimension of the hidden layer]
);

// --- Phase 8: Finalization & Dispatch ---

/** (Node 14) */
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad, // [INPUT  shape: (total_params), Reads gradient slice starting at param_offset]
    SCALAR_TYPE beta1,                           // [PARAM  scalar: Adam hyperparameter]
    SCALAR_TYPE beta2,                           // [PARAM  scalar: Adam hyperparameter]
    SCALAR_TYPE beta1_t,                         // [PARAM  scalar: Adam hyperparameter, beta1^t]
    SCALAR_TYPE beta2_t,                         // [PARAM  scalar: Adam hyperparameter, beta2^t]
    SCALAR_TYPE learning_rate,                   // [PARAM  scalar: Adam hyperparameter]
    SCALAR_TYPE epsilon,                         // [PARAM  scalar: Adam hyperparameter]
    __global SCALAR_TYPE *__restrict param,      // [IN/OUT shape: (total_params), Updates slice starting at param_offset]
    __global SCALAR_TYPE *__restrict m1,         // [IN/OUT shape: (total_params), Updates slice starting at param_offset]
    __global SCALAR_TYPE *__restrict m2,         // [IN/OUT shape: (total_params), Updates slice starting at param_offset]
    int param_offset,                            // [PARAM  scalar: >= 0, The starting index of parameters to update]
    int num_params_to_update                     // [PARAM  scalar: > 0, The number of parameters this call will update]
);

/** (Node 15) */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [IN/OUT shape: (num_exits)]
    SCALAR_TYPE min_temp,                       // [PARAM  scalar: The minimum allowed temperature]
    SCALAR_TYPE max_temp,                       // [PARAM  scalar: The maximum allowed temperature]
    int         num_exits                       // [PARAM  scalar: The total number of temperature parameters]
);

#endif // KERNELS_CL_H
