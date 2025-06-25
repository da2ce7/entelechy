// kernels.cl.h

#ifndef KERNELS_CL_H
#define KERNELS_CL_H

// --- Architectural Contract Note ---
// This header constitutes the sole and sufficient technical contract
// for interaction between host and device implementations:
//
// 1. **Device Specification:**
//    Kernels are defined as stateless computational units. Their behavior,
//    memory layouts, and interface constraints are fully specified here.
//    Device implementations require no external context beyond this document.
//
// 2. **Host Interface:**
//    Kernel invocation parameters, buffer semantics, and synchronization
//    requirements are exhaustively defined. Host code requires no knowledge
//    of device internals or optimization strategies beyond these specifications.
//
// Explicitly out of scope:
// - Host orchestration logic (e.g., task graphs, reduction strategies)
// - Device hardware optimizations (e.g., register allocation, vectorization)
//
// Adherence to this contract ensures strict separation of concerns and
// bidirectional implementation independence.

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
inline SCALAR_TYPE pown(SCALAR_TYPE base, int exp) { return pow(base, (SCALAR_TYPE)exp); }
#endif // __OPENCL_VERSION__

// Computes the flattened 1D index for a hidden unit, assuming a row-major memory layout.
#define GET_PHYSICAL_HIDDEN_IDX(b, h, padded_h_dim) ((b) * (padded_h_dim) + (h))

// --- Host-configurable Flags and Enums ---
#define PROBLEM_TYPE_CCE 0 // Selects Softmax/Cross-Entropy Loss math path
#define PROBLEM_TYPE_BCE 1 // Selects Sigmoid/Binary Cross-Entropy math path
#define AGG_MODE_SUM 0     // Selects summation for aggregation
#define AGG_MODE_AVERAGE 1 // Selects averaging for aggregation

// --- Common Math Configuration ---
#ifndef USE_FAST_MATH
#define USE_FAST_MATH 0
#endif
#if USE_FAST_MATH
#define MATH_FN native_
#else
#define MATH_FN
#endif

// --- Phase 4: Shared Layer Forward Pass ---

/**
 * @brief (Node 4) Computes hidden activations for a slice of the input batch.
 * @contract Applies a (Weights * Input + Bias) transform followed by a ReLU activation.
 * @usage (Host) Generic forward pass kernel.
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                                // [MEMORY size: SIMD_WIDTH * (1 + SIMD_WIDTH) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict input_buf,              // [IN]  Shape: (total_batch_size, padded_input_dim)
    __global const SCALAR_TYPE *__restrict sample_mask,            // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf, // [IN]  Shape: (padded_hidden_dim/SW, padded_input_dim, SW)
    __global const SCALAR_TYPE *__restrict biases_buf,             // [IN]  Shape: (padded_hidden_dim)
    __global SCALAR_TYPE *__restrict hidden_out_buf,               // [OUT] Shape: (total_batch_size, padded_hidden_dim)
    __global SCALAR_TYPE *__restrict hidden_mask_out,              // [OUT] Shape: (total_batch_size)
    int batch_offset,                                              // [IN scalar: >= 0, Start sample index for this chunk]
    int num_batch_samples,                                         // [IN scalar: > 0, Number of samples in this chunk]
    int padded_input_dim,                                          // [IN scalar: > 0, Padded input dimension]
    int padded_hidden_dim);                                        // [IN scalar: > 0, Padded hidden dimension]

// --- Phase 5-7: Module Layer Forward Pass & Loss ---

/**
 * @brief (Node 5) Computes a chunk of raw logits for a slice of modules and classes.
 * @contract Performs a (Weights * Input + Bias) transform. It is designed to be
 *           ROBUST TO MEMORY PADDING, using the physical stride `padded_total_output_classes`
 *           to correctly navigate module parameter buffers.
 * @usage (Host) Called in a loop over module/class chunks.
 */
__kernel void compute_logits_chunk(
    __global const SCALAR_TYPE *__restrict hidden_buf,         // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict hidden_mask,        // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict module_weights_buf, // [IN]  Shape: (total_modules, hidden_dim, padded_total_output_classes)
    __global const SCALAR_TYPE *__restrict module_biases_buf,  // [IN]  Shape: (total_modules, padded_total_output_classes)
    __global SCALAR_TYPE *__restrict full_logits_out,          // [OUT] Shape: (total_modules, total_batch_size, padded_total_output_classes)
    int module_batch_chunk_index,                              // [IN scalar: >= 0]
    int module_param_offset,                                   // [IN scalar: >= 0]
    int num_modules_in_chunk,                                  // [IN scalar: > 0]
    int class_offset,                                          // [IN scalar: >= 0]
    int num_classes_in_chunk,                                  // [IN scalar: > 0]
    int total_batch_size,                                      // [IN scalar: > 0]
    int hidden_dim,                                            // [IN scalar: > 0]
    int padded_hidden_dim,                                     // [IN scalar: > 0]
    int total_output_classes,                                  // [IN scalar: > 0, The logical number of classes]
    int padded_total_output_classes);                          // [IN scalar: > 0, The physical stride for class dimension]

/**
 * @brief (Node 6) Reduces logits to find normalization terms for numerically stable Softmax.
 * @contract Finds the max logit and computes the sum of exps. It is designed to be
 *           ROBUST TO MEMORY PADDING, using the physical stride `padded_total_output_classes`
 *           to correctly navigate the logit buffer.
 * @usage (Host) CCE Path Only. Acts as a synchronization point before kernel (7a).
 */
__kernel void reduce_logits_for_softmax(
    __global const SCALAR_TYPE *__restrict full_logits_buf, // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const SCALAR_TYPE *__restrict temps_buf,       // [IN]  Shape: (total_modules)
    __global SCALAR_TYPE *__restrict softmax_params_out,    // [OUT] Shape: (total_modules, total_batch_size, 2) -> [max_logit, sum_exp]
    int total_modules,                                      // [IN scalar: > 0]
    int total_batch_size,                                   // [IN scalar: > 0]
    int total_output_classes,                               // [IN scalar: > 0, The logical number of classes to reduce]
    int padded_total_output_classes);                       // [IN scalar: > 0, The physical stride for the class dimension]

/**
 * @brief (Node 7a - CCE Path) Computes probabilities and scatters final CCE loss values.
 * @contract Computes Softmax probabilities and final CCE loss. It is designed to be
 *           ROBUST TO MEMORY PADDING, using the physical stride `padded_total_output_classes`
 *           to correctly navigate all class-dimensioned buffers.
 * @usage (Host) CCE Path Only. Depends on kernel (6).
 */
__kernel void compute_probs_loss_cce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf,    // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const SCALAR_TYPE *__restrict softmax_params_buf, // [IN]  Shape: (total_modules, total_batch_size, 2)
    __global const SCALAR_TYPE *__restrict temps_buf,          // [IN]  Shape: (total_modules)
    __global const int *__restrict targets_cce_buf,            // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict sample_mask,        // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_probs_out,        // [OUT] Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global SCALAR_TYPE *__restrict final_loss_out,           // [OUT] Shape: (total_modules, total_batch_size)
    int module_batch_chunk_index,                              // [IN scalar: >= 0]
    int module_param_offset,                                   // [IN scalar: >= 0]
    int num_modules_in_chunk,                                  // [IN scalar: > 0]
    int class_offset,                                          // [IN scalar: >= 0]
    int num_classes_in_chunk,                                  // [IN scalar: > 0]
    int total_batch_size,                                      // [IN scalar: > 0]
    int total_output_classes,                                  // [IN scalar: > 0, The logical number of classes]
    int padded_total_output_classes);                          // [IN scalar: > 0, The physical stride for class dimension]

/**
 * @brief (Node 7b - BCE Path) Computes probabilities and a PARTIAL BCE loss for a tile.
 * @contract Computes Sigmoid probabilities and a partial BCE loss. This kernel is a
 *           Partial Renderer for BOTH of its outputs. It accepts a unique `flat_tile_index`
 *           from the host to calculate the write offset into the `partial_probs_out` and
 *           `partial_loss_out` collection buffers, preventing data races. The final
 *           results must be consolidated by an aggregation kernel.
 * @usage (Host) BCE Path Only. Called once per tile in the execution grid.
 */
__kernel void compute_probs_loss_bce_chunk(
    __global const SCALAR_TYPE *__restrict full_logits_buf, // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const SCALAR_TYPE *__restrict temps_buf,       // [IN]  Shape: (total_modules)
    __global const SCALAR_TYPE *__restrict targets_bce_buf, // [IN]  Shape: (total_batch_size, padded_total_output_classes)
    __global const SCALAR_TYPE *__restrict sample_mask,     // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_probs_out,     // [OUT] Shape: (total_tiles, num_modules_in_chunk, total_batch_size, num_classes_in_chunk)
    __global SCALAR_TYPE *__restrict partial_loss_out,      // [OUT] Shape: (total_tiles, num_modules_in_chunk, total_batch_size)
    int module_batch_chunk_index,                           // [IN scalar: >= 0, Logical MODULE chunk index for parameter selection]
    int module_param_offset,                                // [IN scalar: >= 0, Start module index for this chunk]
    int num_modules_in_chunk,                               // [IN scalar: > 0, Number of modules in this chunk]
    int class_batch_chunk_index,                            // [IN scalar: >= 0, Logical CLASS chunk index for parameter selection]
    int flat_tile_index,                                    // [IN scalar: >= 0, The unique flat index for this tile's output placement]
    int class_offset,                                       // [IN scalar: >= 0, Start class index for this chunk]
    int num_classes_in_chunk,                               // [IN scalar: > 0, Number of classes in this chunk]
    int total_batch_size,                                   // [IN scalar: > 0]
    int total_output_classes,                               // [IN scalar: > 0, The logical number of classes]
    int padded_total_output_classes,                        // [IN scalar: > 0, The physical stride for class dimension]
    int total_modules);                                     // [IN scalar: > 0]

// --- Phase 8-10: Parallel Gradient Computation (Corrected & Harmonized) ---

/**
 * @brief (Node 8) Computes partial module param gradients (Weights, Biases) for a class chunk.
 * @contract Computes a partial gradient via reduction over the batch dimension. It is
 *           designed to be ROBUST TO MEMORY PADDING, using the physical stride
 *           `padded_total_output_classes` to correctly navigate all class-dimensioned buffers.
 *           The kernel accepts a unique `flat_tile_index` from the host to
 *           calculate the write offset into the output collection buffers,
 *           ensuring that parallel invocations do not cause race conditions.
 *           If `problem_type_flag`=0 (CCE), `targets_buf` is `__global int*`.
 *           If `problem_type_flag`=1 (BCE), `targets_buf` is `__global SCALAR_TYPE*`.
 */
__kernel void calculate_module_param_grads_chunk(
    __local SCALAR_TYPE *local_mem,                             // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict hidden_buf,          // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict partial_probs_buf,   // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const void *__restrict targets_buf,                // [IN]  Shape: Generic, cast based on problem_type_flag
    __global const SCALAR_TYPE *__restrict sample_mask,         // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_module_w_out, // [OUT] Shape: (total_tiles, total_modules, h_dim, padded_total_output_classes)
    __global SCALAR_TYPE *__restrict partial_grad_module_b_out, // [OUT] Shape: (total_tiles, total_modules, padded_total_output_classes)
    int problem_type_flag,                                      // [IN scalar: 0|1, CCE or BCE]
    int module_batch_chunk_index,                               // [IN scalar: >= 0]
    int module_param_offset,                                    // [IN scalar: >= 0]
    int num_modules_in_chunk,                                   // [IN scalar: > 0]
    int class_batch_chunk_index,                                // [IN scalar: >= 0, Logical CLASS chunk index for parameter selection]
    int flat_tile_index,                                        // [IN scalar: >= 0, The unique flat index for this tile's output placement]
    int class_offset,                                           // [IN scalar: >= 0]
    int num_classes_in_chunk,                                   // [IN scalar: > 0]
    int total_batch_size,                                       // [IN scalar: > 0]
    int hidden_dim,                                             // [IN scalar: > 0]
    int padded_hidden_dim,                                      // [IN scalar: > 0]
    int total_output_classes,                                   // [IN scalar: > 0, The logical number of classes]
    int padded_total_output_classes,                            // [IN scalar: > 0, The physical stride for class dimension]
    int total_modules);                                         // [IN scalar: > 0]

/**
 * @brief (Node 9) Computes the partial upstream gradient for the hidden layer (Grad_H).
 * @contract Produces a partial upstream gradient via reduction over the class dimension.
 *           It is designed to be ROBUST TO MEMORY PADDING, using the physical stride
 *           `padded_total_output_classes` to correctly navigate all class-dimensioned buffers.
 *           The kernel accepts a unique `flat_tile_index` from the host to
 *           calculate the write offset into the `partial_grad_h_aos_out` collection buffer.
 *           If `problem_type_flag`=0 (CCE), `targets_buf` is `__global int*`.
 *           If `problem_type_flag`=1 (BCE), `targets_buf` is `__global SCALAR_TYPE*`.
 */
__kernel void backprop_error_to_hidden_chunk(
    __local SCALAR_TYPE *local_mem,                            // [MEMORY size: (unused)]
    __global const SCALAR_TYPE *__restrict partial_probs_buf,  // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const void *__restrict targets_buf,               // [IN]  Shape: Generic, cast based on problem_type_flag
    __global const SCALAR_TYPE *__restrict sample_mask,        // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict module_weights_buf, // [IN]  Shape: (total_modules, hidden_dim, padded_total_output_classes)
    __global SCALAR_TYPE *__restrict partial_grad_h_aos_out,   // [OUT] Shape: (total_tiles, total_modules, total_batch_size, hidden_dim)
    int problem_type_flag,                                     // [IN scalar: 0|1, CCE or BCE]
    int module_batch_chunk_index,                              // [IN scalar: >= 0]
    int module_param_offset,                                   // [IN scalar: >= 0]
    int num_modules_in_chunk,                                  // [IN scalar: > 0]
    int class_batch_chunk_index,                               // [IN scalar: >= 0, Logical CLASS chunk index for parameter selection]
    int flat_tile_index,                                       // [IN scalar: >= 0, The unique flat index for this tile's output placement]
    int class_offset,                                          // [IN scalar: >= 0]
    int num_classes_in_chunk,                                  // [IN scalar: > 0]
    int total_batch_size,                                      // [IN scalar: > 0]
    int hidden_dim,                                            // [IN scalar: > 0]
    int total_output_classes,                                  // [IN scalar: > 0, The logical number of classes]
    int padded_total_output_classes,                           // [IN scalar: > 0, The physical stride for class dimension]
    int total_modules);                                        // [IN scalar: > 0]

/**
 * @brief (Node 10) Computes partial temperature gradients for a chunk of classes.
 * @contract Computes a partial gradient via reduction over batch and classes. It is
 *           designed to be ROBUST TO MEMORY PADDING, using the physical stride
 *           `padded_total_output_classes` to correctly navigate all class-dimensioned buffers.
 *           The kernel accepts a unique `flat_tile_index` from the host to
 *           calculate the write offset into the `partial_grad_temps_out` collection buffer.
 *           If `problem_type_flag`=0 (CCE), `targets_buf` is `__global int*`.
 *           If `problem_type_flag`=1 (BCE), `targets_buf` is `__global SCALAR_TYPE*`.
 */
__kernel void calculate_chunk_temp_gradients(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict full_logits_buf,   // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const SCALAR_TYPE *__restrict partial_probs_buf, // [IN]  Shape: (total_modules, total_batch_size, padded_total_output_classes)
    __global const void *__restrict targets_buf,              // [IN]  Shape: Generic, cast based on problem_type_flag
    __global const SCALAR_TYPE *__restrict sample_mask,       // [IN]  Shape: (total_batch_size)
    __global const SCALAR_TYPE *__restrict temps_buf,         // [IN]  Shape: (total_modules)
    __global SCALAR_TYPE *__restrict partial_grad_temps_out,  // [OUT] Shape: (total_tiles, total_modules)
    int problem_type_flag,                                    // [IN scalar: 0|1, CCE or BCE]
    int module_batch_chunk_index,                             // [IN scalar: >= 0]
    int module_param_offset,                                  // [IN scalar: >= 0]
    int num_modules_in_chunk,                                 // [IN scalar: > 0]
    int class_batch_chunk_index,                              // [IN scalar: >= 0, Logical CLASS chunk index for parameter selection]
    int flat_tile_index,                                      // [IN scalar: >= 0, The unique flat index for this tile's output placement]
    int class_offset,                                         // [IN scalar: >= 0]
    int num_classes_in_chunk,                                 // [IN scalar: > 0]
    int total_batch_size,                                     // [IN scalar: > 0]
    int total_output_classes,                                 // [IN scalar: > 0, The logical number of classes]
    int padded_total_output_classes,                          // [IN scalar: > 0, The physical stride for class dimension]
    int total_modules);                                       // [IN scalar: > 0]

// --- Phase 11: Data Layout Transformation ---

/**
 * @brief (Node 11a) Transposes a rectangular slice (chunk) of a matrix.
 * @contract Reads a sub-matrix from `in_buf` and writes its transpose to a
 *           corresponding slice in `out_buf`. It uses element-based offsets
 *           and leading dimension arguments to correctly handle sub-regions
 *           within larger, potentially padded, parent buffers.
 * @usage (Host) Generic, streamable matrix transpose utility.
 */
__kernel void transpose_chunk(
    __local SCALAR_TYPE *local_mem,                // [MEMORY size: C_TILE_SIZE * (C_TILE_SIZE + 1) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict in_buf, // [IN]  Source buffer containing the slice to transpose
    __global SCALAR_TYPE *__restrict out_buf,      // [OUT] Destination buffer for the transposed slice
    int in_offset_elements,                        // [IN scalar: >= 0, Start element offset into in_buf for the slice]
    int out_offset_elements,                       // [IN scalar: >= 0, Start element offset into out_buf for the slice]
    int num_rows_in_chunk,                         // [IN scalar: > 0, The number of rows in the chunk to process]
    int num_cols_in_chunk,                         // [IN scalar: > 0, The number of columns in the chunk to process]
    int in_leading_dim,                            // [IN scalar: > 0, The leading dimension (stride) of the IN buffer]
    int out_leading_dim);                          // [IN scalar: > 0, The leading dimension (stride) of the OUT buffer]

/**
 * @brief (Node 11b) Gathers and permutes scattered partial Grad_H chunks into a single, reduction-ready buffer.
 * @contract This is a specialized permutation kernel. Each work-item is responsible for
 *           a single hidden activation (b, h) across all modules. It reads (gathers)
 *           the gradient contributions for this activation from multiple, non-contiguous
 *           source partials and writes them into a single, contiguous row in the destination buffer.
 * @usage (Host) Essential link between chunked backpropagation (Node 9) and specialized
 *           reduction (Node 13). The host MUST provide correct strides and offsets that map
 *           the scattered source layout to the dense destination layout.
 */
__kernel void gather_and_permute_grad_h(
    __global const SCALAR_TYPE *__restrict partial_grad_h_aos_buf, // [IN]  Shape: The full collection of partials from (9)
    __global SCALAR_TYPE *__restrict aggregated_grad_h_soa_out,  // [OUT] Shape: (total_batch_size * hidden_dim, padded_total_modules)
    int total_batch_size,                                        // [IN scalar: > 0]
    int hidden_dim,                                              // [IN scalar: > 0]
    int total_modules,                                           // [IN scalar: > 0, The logical number of modules]
    int padded_total_modules,                                    // [IN scalar: > 0, The physical stride of the OUT buffer]
    int num_module_chunks,                                       // [IN scalar: > 0, The number of chunks the module dim was split into]
    int modules_per_chunk,                                       // [IN scalar: > 0, The number of modules in each partial]
    int num_class_chunks);                                       // [IN scalar: > 0, The number of chunks the class dim was split into]


// --- Phase 12 & 16: Generic Tiered Aggregation Engine ---

/**
 * @brief (Node 12, 16) Tier 0 (N=1): Identity pass-through copy.
 * @contract Copies `elements_per_partial` elements from input to output.
 *           `num_partials_to_reduce` must be 1.
 * @usage (Host) A pass-through copy kernel. Used when only one partial input needs to be
 *           moved to the final output buffer, representing the terminal base case
 *           for any reduction operation.
 */
__kernel void aggregate_identity(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: (unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (elements_per_partial)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (elements_per_partial)
    int num_partials_to_reduce,                               // [IN scalar: unused]
    int elements_per_partial,                                 // [IN scalar: > 0, Number of elements to copy]
    int reduction_mode_flag);                                 // [IN scalar: unused]

/**
 * @brief (Node 12, 16) Tier 1 (N is small): Reduces partial results using registers.
 * @contract Reduces `num_partials_to_reduce` segments from the input buffer.
 *           Each work-item handles one element across all partials.
 * @usage (Host) A generic reduction kernel for consolidating a small number of partial
 *           results. It is optimized to perform the reduction summation primarily
 *           within registers, making it efficient for small `num_partials_to_reduce`.
 */
__kernel void aggregate_register_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: (unused)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (num_partials_to_reduce, elements_per_partial)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (elements_per_partial)
    int num_partials_to_reduce,                               // [IN scalar: > 0, Number of partials to reduce]
    int elements_per_partial,                                 // [IN scalar: > 0, Elements in one partial tensor]
    int reduction_mode_flag);                                 // [IN scalar: 0|1, SUM or AVERAGE]

/**
 * @brief (Node 12, 16) Tier 2 (N is large): Reduces partial results using local memory.
 * @contract Reduces `num_partials_to_reduce` segments from the input buffer.
 *           Each work-group handles one element across all partials using local memory.
 * @usage (Host) A generic, scalable reduction kernel for consolidating a large number
 *           of partial results. It uses local memory to perform an efficient
 *           parallel reduction within each work-group, making it the workhorse
 *           for any large-scale aggregation task.
 */
__kernel void aggregate_local_reduce(
    __local SCALAR_TYPE *local_mem,                           // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict partial_input_buf, // [IN]  Shape: (num_partials_to_reduce, elements_per_partial)
    __global SCALAR_TYPE *__restrict final_output_buf,        // [OUT] Shape: (elements_per_partial)
    int num_partials_to_reduce,                               // [IN scalar: > 0, Number of partials to reduce]
    int elements_per_partial,                                 // [IN scalar: > 0, Elements in one partial tensor]
    int reduction_mode_flag);                                 // [IN scalar: 0|1, SUM or AVERAGE]

// --- Phase 13: Specialized Grad_H Reduction ---

/**
 * @brief (Node 13) Reduces the aggregated, module-major Grad_H buffer to the final upstream gradient.
 * @contract Designed to be ROBUST TO MEMORY PADDING. It sums the contributions from all logical modules
 *           for each hidden activation by using the physical stride (`padded_total_modules`) to correctly
 *           navigate the potentially padded input buffer.
 *           Performs the reduction: (total_elements, padded_total_modules) -> (total_elements).
 * @usage (Host) Specialized "join" operation for the parallel multi-head fork. Replaces a
 *           less efficient transpose-and-aggregate sequence with a single purpose-built kernel. The host
 *           MUST provide the physical leading dimension of the input buffer.
 */
__kernel void reduce_grad_h_over_modules(
    __local SCALAR_TYPE *local_mem,                               // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict aggregated_grad_h_soa, // [IN]  Shape: (total_elements, padded_total_modules)
    __global SCALAR_TYPE *__restrict final_grad_h_buf,            // [OUT] Shape: (total_elements) -> Logically (batch, hidden)
    int total_elements,                                           // [IN scalar: > 0, The number of elements to reduce (B * H)]
    int total_modules,                                            // [IN scalar: > 0, The logical number of modules to sum over]
    int padded_total_modules);                                    // [IN scalar: > 0, The physical leading dimension (stride) of the input buffer]

// --- Phase 14-15: Streaming Shared Layer Backpropagation ---

/**
 * @brief (Node 14) Computes partial gradients for shared layer weights from a batch chunk.
 * @contract Produces a *partial* weight gradient by reducing over a chunk of the batch.
 *           Implicitly filters gradients using the ReLU derivative (`hidden_buf` > 0).
 * @usage (Host) Called in a loop over batch chunks. Can be launched in parallel with (15).
 */
__kernel void backprop_shared_weights_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict input_buf,        // [IN]  Shape: (total_batch_size, padded_input_dim)
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict sample_mask,      // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_sw_out,    // [OUT] Shape: (num_batch_chunks, padded_input_dim, padded_hidden_dim)
    int batch_offset,                                        // [IN scalar: >= 0, Start sample index]
    int num_batch_samples,                                   // [IN scalar: > 0, Number of samples in chunk]
    int batch_chunk_index,                                   // [IN scalar: >= 0, Logical BATCH chunk index]
    int padded_input_dim,                                    // [IN scalar: > 0]
    int padded_hidden_dim);                                  // [IN scalar: > 0]

/**
 * @brief (Node 15) Computes partial gradients for shared layer biases from a batch chunk.
 * @contract Produces a *partial* bias gradient by reducing over a chunk of the batch.
 *           Implicitly filters gradients using the ReLU derivative (`hidden_buf` > 0).
 * @usage (Host) Called in a loop over batch chunks. Can be launched in parallel with (14).
 */
__kernel void backprop_shared_biases_chunk(
    __local SCALAR_TYPE *local_mem,                          // [MEMORY size: get_local_size(0) * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict final_grad_h_buf, // [IN]  Shape: (total_batch_size, padded_hidden_dim)
    __global const SCALAR_TYPE *__restrict sample_mask,      // [IN]  Shape: (total_batch_size)
    __global SCALAR_TYPE *__restrict partial_grad_sb_out,    // [OUT] Shape: (num_batch_chunks, padded_hidden_dim)
    int batch_offset,                                        // [IN scalar: >= 0, Start sample index]
    int num_batch_samples,                                   // [IN scalar: > 0, Number of samples in chunk]
    int batch_chunk_index,                                   // [IN scalar: >= 0, Logical BATCH chunk index]
    int padded_hidden_dim);                                  // [IN scalar: > 0]

// --- Phase 17-19: Finalization & Dispatch ---

/**
 * @brief (Node 18) Applies Adam optimizer update to a slice of a parameter buffer.
 * @contract Performs the complete Adam update, including the bias correction term
 *           which is calculated INTERNALLY from the global step `t`.
 *           NOTE: The contract requires that the `param`, `grad`, `m1`, and `m2`
 *           buffers have IDENTICAL physical memory layouts and sizes.
 * @usage (Host) Generic optimizer called once per parameter group. The host is responsible
 *           for enforcing layout consistency, typically via harmonized padding rules
 *           during buffer creation. `num_params_to_update` MUST match the full
 *           physical size of these buffers.
 */
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad, // [IN]     Shape: (num_params_to_update)
    SCALAR_TYPE beta1,                           // [IN scalar: (0,1), Decay rate for 1st moment]
    SCALAR_TYPE beta2,                           // [IN scalar: (0,1), Decay rate for 2nd moment]
    SCALAR_TYPE learning_rate,                   // [IN scalar: > 0]
    SCALAR_TYPE epsilon,                         // [IN scalar: > 0]
    uint        t,                               // [IN scalar: > 0, The current global training step]
    __global SCALAR_TYPE *__restrict param,      // [IN/OUT] Shape: (num_params_to_update)
    __global SCALAR_TYPE *__restrict m1,         // [IN/OUT] Shape: (num_params_to_update), 1st moment vector
    __global SCALAR_TYPE *__restrict m2,         // [IN/OUT] Shape: (num_params_to_update), 2nd moment vector
    int param_offset,                            // [IN scalar: >= 0, Start index for this parameter slice]
    int num_params_to_update                     // [IN scalar: > 0, Number of params in this slice]
);

/**
 * @brief (Node 19) Clamps temperature parameters within a [min, max] range.
 * @contract Enforces `param = clamp(param, min_temp, max_temp)`.
 * @usage (Host) Called only on the temperature buffer, immediately after its Adam update.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [IN/OUT] Shape: (total_modules)
    SCALAR_TYPE min_temp,                       // [IN scalar: Minimum allowed temperature value]
    SCALAR_TYPE max_temp,                       // [IN scalar: Maximum allowed temperature value]
    int         total_modules);                         // [IN scalar: > 0, The number of temperatures to clamp]

#endif // KERNELS_CL_H
