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
 */
__kernel void forward_pass(
    /**
     * @param update_buffer_LOCAL_simd_tile Local memory for optimizing SIMD operations.
     *        - Local Memory Requirement: [SIMD_WIDTH * (1 + SIMD_WIDTH) * sizeof(SCALAR_TYPE)] bytes
     *        - Validation Preconditions: Host must allocate this amount via clSetKernelArg.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_simd_tile,

    /**
     * @param src_buffer_GLOBAL_input The primary data source for the computational unit.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: ROW, Formula: Post-pad cols to 128-byte alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_shared_simd_major The learnable shared weights in a SIMD-friendly layout.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count/SIMD_WIDTH, src_scalar_NATURAL_padded_input_count, SIMD_WIDTH)
     *        - Padding Contract: {Type: TILE, Formula: hidden_dim padded to SIMD_WIDTH; input_dim padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_weights_shared_simd_major,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_shared The learnable shared biases.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: ROW, Formula: Padded to SIMD_WIDTH}
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_biases_shared,

    /**
     * @param dest_buffer_GLOBAL_hidden_activations The output tensor of the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: ROW, Formula: Padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_hidden_activations,

    /**
     * @param dest_buffer_GLOBAL_hidden_mask Derived mask from ReLU operation (1 if activation > 0, else 0).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: ROW, Formula: Padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_hidden_mask,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_padded_hidden_count);

// --- Phase 5-7: Module Layer Forward Pass & Loss ---

/**
 * @brief (Node 5) Computes a chunk of raw logits for a slice of modules, classes, and batch samples.
 */
__kernel void compute_logits_chunk(
    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: ROW, Formula: Padded to alignment}
     *        - Calculability Proof: From Node 4 output contract.
     *        - Validation Preconditions: Must be the valid output of Node 4.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask The ReLU mask corresponding to the hidden activations.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: ROW, Formula: Padded to alignment}
     *        - Calculability Proof: From Node 4 output contract.
     *        - Validation Preconditions: Must be the valid output of Node 4.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: TILE, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules, src_scalar_NATURAL_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_module The learnable biases for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: ROW, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_biases_module,

    /**
     * @param dest_buffer_GLOBAL_logits The raw, pre-activation output tensor for all modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: TILE, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_logits,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_module_chunk_offset,
    uint src_scalar_NATURAL_module_chunk_count,
    uint src_scalar_NATURAL_class_chunk_offset,
    uint src_scalar_NATURAL_class_chunk_count,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules);

/**
 * @brief (Node 6) Fused kernel to compute probabilities and final CCE loss for a tile.
 * @contract Performs the complete, temperature-aware, numerically stable Softmax calculation internally.
 *           1. Scales logits by temperature: `scaled_logit = logit / T`.
 *           2. Finds max of scaled logits for stability: `max_sl = max(scaled_logit)`.
 *           3. Computes `exp(scaled_logit - max_sl)` and sums for the denominator.
 *           4. Normalizes to get final probabilities.
 *           5. Computes final CCE loss and scatters to its final destination (no aggregation needed).
 */
__kernel void compute_probs_loss_cce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: TILE, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: From Node 5 output contract.
     *        - Validation Preconditions: Must be the valid output of Node 5.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets_cce The ground truth labels for CCE.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Values must be in [0, total_output_class_count - 1].
     */
    __global const int *src_buffer_GLOBAL_targets_cce,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_final_loss_cce The monolithic buffer for final CCE loss values (scatter-write).
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Host shall zero-initialize this buffer.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_final_loss_cce,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules,
    uint src_scalar_NATURAL_num_total_tiles);

/**
 * @brief (Node 7) Computes probabilities and PARTIAL BCE loss for a tile.
 * @contract A "Partial Renderer" for both probabilities and loss. Uses `flat_tile_index`
 *           as the sole source of truth for work-item identity and output placement.
 */
__kernel void compute_probs_loss_bce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: TILE, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: From Node 5 output contract.
     *        - Validation Preconditions: Must be the valid output of Node 5.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets_bce The ground truth labels for BCE.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: ROW, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_targets_bce,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_partial_loss_bce The collection buffer for this tile's computed partial loss.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_loss_bce,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules,
    uint src_scalar_NATURAL_num_total_tiles);

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

// --- Phase 8-10: Parallel Gradient Computation ---

/**
 * @brief (Node 8) Computes partial module param gradients (Weights, Biases) for a tile.
 * @contract A "Partial Renderer". Uses `flat_tile_index` to derive its logical work location
 *           and physical output placement. Operates on chunks of the batch dimension.
 */
__kernel void calculate_module_param_grads_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Local Memory Requirement: [get_local_size(0) * sizeof(SCALAR_TYPE)] bytes
     *        - Validation Preconditions: Host must allocate this amount via clSetKernelArg.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: ROW, Formula: Padded to alignment}
     *        - Calculability Proof: From Node 4 output contract.
     *        - Validation Preconditions: Must be the valid output of Node 4.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: From Node 6/7 output contracts.
     *        - Validation Preconditions: Buffer must be valid output of Node 6/7.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets_generic The ground truth labels (type-punned).
     *        - Tensor Shape: Varies. If CCE, (total_batch_size) of int. If BCE, (total_batch_size, padded_total_output_class_count) of SCALAR_TYPE.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type.
     *        - Validation Preconditions: Host must provide the correct target buffer for the chosen problem type.
     */
    __global const void *src_buffer_GLOBAL_targets_generic,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_module The collection buffer for this tile's computed weight gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_hidden_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_module The collection buffer for this tile's computed bias gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_biases_module,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules,
    uint src_scalar_NATURAL_num_total_tiles);

/**
 * @brief (Node 9) Computes the partial upstream gradient for the hidden layer (Grad_H) for a tile.
 */
__kernel void backprop_error_to_hidden_chunk(
    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: From Node 6/7 output contracts.
     *        - Validation Preconditions: Buffer must be valid output of Node 6/7.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets_generic The ground truth labels (type-punned).
     *        - Tensor Shape: Varies. If CCE, (total_batch_size) of int. If BCE, (total_batch_size, padded_total_output_class_count) of SCALAR_TYPE.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type.
     *        - Validation Preconditions: Host must provide the correct target buffer for the chosen problem type.
     */
    __global const void *src_buffer_GLOBAL_targets_generic,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: TILE, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules, src_scalar_NATURAL_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_h_aos The collection buffer for this tile's computed upstream gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_batch_chunk_count, src_scalar_NATURAL_hidden_count)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated. This buffer is consumed by Node 12.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_h_aos,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules,
    uint src_scalar_NATURAL_num_total_tiles);

/**
 * @brief (Node 10) Computes partial temperature gradients for a tile.
 */
__kernel void calculate_chunk_temp_gradients(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Local Memory Requirement: [get_local_size(0) * sizeof(SCALAR_TYPE)] bytes
     *        - Validation Preconditions: Host must allocate this amount via clSetKernelArg.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: TILE, Formula: output_class_count padded to alignment}
     *        - Calculability Proof: From Node 5 output contract.
     *        - Validation Preconditions: Must be the valid output of Node 5.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: From Node 6/7 output contracts.
     *        - Validation Preconditions: Buffer must be valid output of Node 6/7.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets_generic The ground truth labels (type-punned).
     *        - Tensor Shape: Varies. If CCE, (total_batch_size) of int. If BCE, (total_batch_size, padded_total_output_class_count) of SCALAR_TYPE.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type.
     *        - Validation Preconditions: Host must provide the correct target buffer for the chosen problem type.
     */
    __global const void *src_buffer_GLOBAL_targets_generic,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_size]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules)
     *        - Padding Contract: None.
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_temps The collection buffer for this tile's computed temperature gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk)
     *        - Padding Contract: None.
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_temps,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules,
    uint src_scalar_NATURAL_num_total_tiles);

// --- Phase 11-12: Data Layout Transformation ---

/**
 * @brief (Node 11) Transposes a rectangular slice (chunk) of a matrix. General-purpose utility.
 * @contract Reads a sub-matrix from `src_buffer_GLOBAL_input` and writes its transpose to a
 *           corresponding slice in `dest_buffer_GLOBAL_output`. It uses element-based offsets
 *           and leading dimension arguments to correctly handle sub-regions
 *           within larger, potentially padded, parent buffers.
 */
__kernel void transpose_chunk(
    /**
     * @param update_buffer_LOCAL_transpose_tile Local memory for optimizing the transpose via tiling.
     *        - Local Memory Requirement: [C_TILE_SIZE * (C_TILE_SIZE + 1) * sizeof(SCALAR_TYPE)] bytes. The +1 is for bank conflict avoidance.
     *        - Validation Preconditions: Host must allocate this amount via clSetKernelArg. get_local_size(0) and get_local_size(1) should be C_TILE_SIZE.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_transpose_tile,

    /**
     * @param src_buffer_GLOBAL_input The source buffer containing the slice to transpose.
     *        - Tensor Shape: Host-defined.
     *        - Padding Contract: Host-defined, handled via `src_scalar_NATURAL_in_leading_dim`.
     *        - Calculability Proof: Provided by host at runtime.
     *        - Validation Preconditions: `src_scalar_NATURAL_in_offset_elements` and dimensions must be within buffer bounds.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param dest_buffer_GLOBAL_output The destination buffer for the transposed slice.
     *        - Tensor Shape: Host-defined.
     *        - Padding Contract: Host-defined, handled via `src_scalar_NATURAL_out_leading_dim`.
     *        - Calculability Proof: Provided by host at runtime.
     *        - Validation Preconditions: `src_scalar_NATURAL_out_offset_elements` and dimensions must be within buffer bounds.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_output,

    uint src_scalar_NATURAL_in_offset_elements,
    uint src_scalar_NATURAL_out_offset_elements,
    uint src_scalar_NATURAL_rows_count,
    uint src_scalar_NATURAL_cols_count,
    uint src_scalar_NATURAL_in_leading_dim,
    uint src_scalar_NATURAL_out_leading_dim);

/**
 * @brief (Node 12) Specialized Kernel: Gathers scattered partial Grad_H chunks into a single, reduction-ready buffer.
 * @contract This kernel is the solution to the "Transpose Illusion." It performs a global permutation,
 *           reading from many non-contiguous partial results buffers into one dense SoA buffer.
 *           NOTE: This kernel also implicitly performs a reduction over the class_chunk dimension during the gather operation.
 */
__kernel void gather_and_permute_grad_h(
    /**
     * @param src_buffer_GLOBAL_partial_grad_h_aos The full collection of partial upstream gradients from Node 9.
     *        - Tensor Shape: (src_scalar_NATURAL_num_total_tiles, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_size, src_scalar_NATURAL_hidden_count)
     *        - Padding Contract: None.
     *        - Calculability Proof: From Node 9 output contract.
     *        - Validation Preconditions: Buffer must be the valid output of Node 9.
     *        - ARCHITECTURAL NOTE: The `total_batch_size` dimension is monolithic. This kernel's global
     *          permutation requirement means its input-producing kernel (Node 9) cannot be streamed
     *          over the batch dimension. This is a deliberate architectural synchronization point.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_h_aos,

    /**
     * @param dest_buffer_GLOBAL_grad_h_permuted_soa The final, contiguous, SoA-layout buffer ready for reduction by Node 14.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_size * src_scalar_NATURAL_hidden_count, src_scalar_NATURAL_padded_total_modules)
     *        - Padding Contract: {Type: ROW, Formula: `total_modules` padded to alignment for the trailing dimension}
     *        - Calculability Proof: All dimensions are scalars provided by the Host Orchestrator.
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_grad_h_permuted_soa,

    uint src_scalar_NATURAL_total_batch_size,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_total_modules,
    uint src_scalar_NATURAL_padded_total_modules,
    uint src_scalar_NATURAL_num_module_chunks,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_num_total_tiles);

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
