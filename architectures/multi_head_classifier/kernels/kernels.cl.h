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

// --- Mandatory Architectural Constants (Article 5) ---

// The System Contract defines LOCAL_MEM_BANK_PADDING as a fixed
// architectural constant. The build system MUST provide this exact value.
#if !defined(LOCAL_MEM_BANK_PADDING) || (LOCAL_MEM_BANK_PADDING != 1)
#error "System Contract Violation: LOCAL_MEM_BANK_PADDING must be defined and have a value of exactly 1."
#endif

// --- Mandatory Build-Time Symbols (Article 6) ---

#ifndef SCALAR_TYPE
#error "System Contract Violation: SCALAR_TYPE must be defined by the host build system."
#endif
#ifndef SIMD_WIDTH
#error "System Contract Violation: SIMD_WIDTH must be defined by the host build system."
#endif
#ifndef C_TILE_SIZE
#error "System Contract Violation: C_TILE_SIZE must be defined by the host build system."
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
#ifndef LOCAL_MEM_BANK_PADDING
#define LOCAL_MEM_BANK_PADDING 1
#endif
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
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable"
 */
__kernel void forward_pass(
    /**
     * @param update_buffer_LOCAL_simd_tile A local memory resource for tiling to optimize SIMD operations.
     *        - Tensor Shape: (SIMD_WIDTH, SIMD_WIDTH + LOCAL_MEM_BANK_PADDING)
     *        - Padding Contract: {Type: BANK_CONFLICT_AVOIDANCE, Formula: "Pad row stride by LOCAL_MEM_BANK_PADDING"}
     *        - Calculability Proof: [Compile-time constant: SIMD_WIDTH, System Contract constant: LOCAL_MEM_BANK_PADDING]
     *        - Validation Preconditions: Host shall allocate size according to the formula derived from this contract.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_simd_tile,

    /**
     * @param src_buffer_GLOBAL_input The primary data source for the computational unit.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Pad row stride to 128-byte alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: The access slice defined by chunk parameters must be within the buffer's bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset +
     * src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_shared_simd_major The learnable shared weights in a SIMD-friendly layout.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count/SIMD_WIDTH, src_scalar_NATURAL_padded_input_count, SIMD_WIDTH)
     *        - Padding Contract: {Type: SIMD, Formula: "hidden_dim padded to SIMD_WIDTH; input_dim padded for alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_weights_shared_simd_major,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_shared The learnable shared biases.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: SIMD, Formula: "Padded to SIMD_WIDTH"}
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_biases_shared,

    /**
     * @param dest_buffer_GLOBAL_hidden_activations The output tensor of the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(SCALAR_TYPE)] bytes.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_hidden_activations,

    /**
     * @param dest_buffer_GLOBAL_hidden_mask Derived mask from ReLU operation (1 if activation > 0, else 0).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(SCALAR_TYPE)] bytes.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_hidden_mask,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_padded_hidden_count);

// --- Phase 5-7: Module Layer Forward Pass & Loss ---
/**
 * @brief (Node 5) Computes a chunk of raw logits for a slice of modules, classes, and batch samples.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable. Consumes chunked inputs to produce slice of monolithic output."
 */
__kernel void compute_logits_chunk(
    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: The batch slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask The ReLU mask corresponding to the hidden activations.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: The batch slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: SIMD, Formula: "output_class_count padded for SIMD/Cache alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: The module and class slices must be within bounds, as proven by: [(src_scalar_NATURAL_module_chunk_offset + src_scalar_NATURAL_module_chunk_count) <=
     * src_scalar_NATURAL_total_modules_count] AND [(src_scalar_NATURAL_class_chunk_offset + src_scalar_NATURAL_class_chunk_count) <= src_scalar_NATURAL_total_output_class_count].
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_module The learnable biases for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: SIMD, Formula: "output_class_count padded for SIMD alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: The module and class slices must be within bounds, as proven by: [(src_scalar_NATURAL_module_chunk_offset + src_scalar_NATURAL_module_chunk_count) <=
     * src_scalar_NATURAL_total_modules_count] AND [(src_scalar_NATURAL_class_chunk_offset + src_scalar_NATURAL_class_chunk_count) <= src_scalar_NATURAL_total_output_class_count].
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_biases_module,

    /**
     * @param dest_buffer_GLOBAL_logits The raw, pre-activation output tensor for all modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_total_output_class_count * sizeof(SCALAR_TYPE)] bytes.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_logits,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_module_chunk_offset,
    uint src_scalar_NATURAL_module_chunk_count,
    uint src_scalar_NATURAL_class_chunk_offset,
    uint src_scalar_NATURAL_class_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count);

/**
 * @brief (Node 6) Fused kernel to compute probabilities and final CCE loss for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "The implementation is a fused, indivisible unit for numerically stable Softmax calculation."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer for probabilities output."
 */
__kernel void compute_probs_loss_cce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (class indices).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Values must be in [0, total_output_class_count - 1].
     */
    __global const int *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host must provide a valid flat_tile_index and other necessary scalars for the grid_mod_cls placement strategy.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_final_loss The monolithic buffer for final loss values.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] Host shall zero-initialize this buffer. [2] The kernel populates this buffer via a direct scatter-write; no host-side aggregation is required for this
     * parameter.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_final_loss,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_total_tile_count);

/**
 * @brief (Node 7) Computes probabilities and PARTIAL BCE loss for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Dual Partial Renderer. Uses flat_tile_index for both probability and loss outputs."
 */
__kernel void compute_probs_loss_bce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (multi-hot encoded).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host must provide a valid flat_tile_index and other necessary scalars for the grid_mod_cls placement strategy.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_partial_loss The collection buffer for this tile's computed partial loss.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host must provide a valid flat_tile_index and other necessary scalars for the grid_mod_cls placement strategy.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_loss,

    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_total_tile_count);

// --- Phase 8-10: Parallel Gradient Computation ---

/**
 * @brief (Node 8) Computes partial module param gradients (Weights, Biases) for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Dual Partial Renderer. Uses flat_tile_index for both weight and bias gradient outputs."
 */
__kernel void calculate_module_param_grads_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(SCALAR_TYPE)`.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout and type correspond to the
     * value of `src_scalar_FLAG_problem_type`. [3] The kernel implementation will cast this pointer internally based on the flag.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_module The collection buffer for this tile's computed weight gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_module The collection buffer for this tile's computed bias gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_biases_module,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_total_tile_count);

/**
 * @brief (Node 9) Computes the partial upstream gradient for the hidden layer (Grad_H) for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer for a monolithic intermediate buffer."
 */
__kernel void backprop_error_to_hidden_chunk(
    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout and type correspond to the
     * value of `src_scalar_FLAG_problem_type`.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: The module targeted by the tile index must be within bounds. The logic for deriving the module index from flat_tile_index is complex, but the overarching tile
     * index itself must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_hidden_activations_aos The collection buffer for this tile's computed upstream gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] [ARCHITECTURAL
     * SYNCHRONIZATION POINT] The consumer of this buffer (Node 12) is a global permutation requiring a monolithic input. Therefore, the Host Orchestrator MUST NOT stream the batch dimension when
     * populating this buffer. It must be computed in full via the 'Accumulate via Recompute' strategy.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_total_tile_count);

/**
 * @brief (Node 10) Computes partial temperature gradients for a tile.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer for temperature gradients."
 */
__kernel void calculate_chunk_temp_gradients(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(SCALAR_TYPE)`.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Must be the valid output of Node 5.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: Buffer must be valid output of Node 6/7.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout and type correspond to the
     * value of `src_scalar_FLAG_problem_type`.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_temps The collection buffer for this tile's computed temperature gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host must provide valid `src_scalar_NATURAL_flat_tile_index` and `src_scalar_NATURAL_num_class_chunks` scalars.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_temps,

    uint src_scalar_FLAG_problem_type,
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_total_output_class_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_total_tile_count);

// --- Phase 11-12: Data Layout Transformation ---

/**
 * @brief (Node 11) Transposes a rectangular slice (chunk) of a matrix. General-purpose utility.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable Utility"
 */
__kernel void transpose_chunk(
    /**
     * @param update_buffer_LOCAL_transpose_tile Local memory for optimizing the transpose via tiling.
     *        - Tensor Shape: (C_TILE_SIZE, C_TILE_SIZE + LOCAL_MEM_BANK_PADDING)
     *        - Padding Contract: {Type: BANK_CONFLICT_AVOIDANCE, Formula: "Pad row stride by LOCAL_MEM_BANK_PADDING"}
     *        - Calculability Proof: [Compile-time constant: C_TILE_SIZE, System Contract constant: LOCAL_MEM_BANK_PADDING]
     *        - Validation Preconditions: [1] Host shall allocate size according to the formula derived from this contract. [2] Host must dispatch this kernel with a 2D local work-group size of
     * (C_TILE_SIZE, C_TILE_SIZE).
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_transpose_tile,

    /**
     * @param src_buffer_GLOBAL_input The source buffer containing the slice to transpose.
     *        - Tensor Shape: (src_scalar_NATURAL_in_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_in_total_element_count]
     *        - Validation Preconditions: [1] The sub-region defined by kernel arguments must be within the buffer's physical bounds, as proven by: (src_scalar_NATURAL_in_offset +
     * ((src_scalar_NATURAL_sub_region_dim1_count - 1) * src_scalar_NATURAL_in_leading_dim) + (src_scalar_NATURAL_sub_region_dim2_count - 1)) < src_scalar_NATURAL_in_total_element_count. [2] The
     * kernel is agnostic to this buffer's padding strategy; correct navigation is guaranteed by the `in_offset` and `in_leading_dim` scalars.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param dest_buffer_GLOBAL_output The destination buffer for the transposed slice.
     *        - Tensor Shape: (src_scalar_NATURAL_out_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_out_total_element_count]
     *        - Validation Preconditions: [1] The sub-region defined by kernel arguments must be within the buffer's physical bounds, as proven by: (src_scalar_NATURAL_out_offset +
     * ((src_scalar_NATURAL_sub_region_dim2_count - 1) * src_scalar_NATURAL_out_leading_dim) + (src_scalar_NATURAL_sub_region_dim1_count - 1)) < src_scalar_NATURAL_out_total_element_count. [2] The
     * kernel is agnostic to this buffer's padding strategy; correct navigation is guaranteed by the `out_offset` and `out_leading_dim` scalars.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_output,

    uint src_scalar_NATURAL_in_offset,
    uint src_scalar_NATURAL_out_offset,
    uint src_scalar_NATURAL_sub_region_dim1_count,
    uint src_scalar_NATURAL_sub_region_dim2_count,
    uint src_scalar_NATURAL_in_leading_dim,
    uint src_scalar_NATURAL_out_leading_dim,
    uint src_scalar_NATURAL_in_total_element_count,
    uint src_scalar_NATURAL_out_total_element_count);

/**
 * @brief (Node 12) Specialized Kernel: Gathers scattered partial gradients into a single, reduction-ready buffer.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a specialized architectural primitive designed to solve the 'Transpose Illusion' by gathering scattered partial results into a dense, reduction-ready
 * SoA layout."
 *        - Behavioral Invariants: "The gather operation performs an implicit reduction (summation) over the `class_chunk` dimension."
 *        - Synchronization Model: "Global Barrier. This kernel cannot execute until all its partial inputs from Node 9 are fully rendered."
 *        - Idempotency: "Associatively Non-Idempotent"
 */
__kernel void gather_and_permute_grad_hidden_activations(
    /**
     * @param src_buffer_GLOBAL_partial_grad_hidden_activations_aos The full collection of partial upstream gradients from Node 9.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [ARCHITECTURAL SYNCHRONIZATION POINT] The consumer of this buffer (this kernel, Node 12) is a global permutation requiring a monolithic input. Therefore, the
     * Host Orchestrator MUST NOT stream the batch dimension when dispatching the producer of this buffer (Node 9). The 'Accumulate via Recompute' strategy is mandatory for the upstream data path.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_grad_hidden_activations_aos,

    /**
     * @param dest_buffer_GLOBAL_grad_hidden_activations_permuted_soa The final, contiguous, SoA-layout buffer ready for reduction by Node 14.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Trailing dimension (`total_modules_count`) is Host-padded to `padded_total_modules_count` for alignment."}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count]
     *        - Validation Preconditions: Host shall ensure sufficient VRAM is allocated.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_grad_hidden_activations_permuted_soa,

    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_padded_total_modules_count,
    uint src_scalar_NATURAL_num_module_chunks_count,
    uint src_scalar_NATURAL_modules_per_chunk_count,
    uint src_scalar_NATURAL_num_class_chunks_count,
    uint src_scalar_NATURAL_total_tile_count);

// --- Phase 13 & 17: Recursive, Tiered Aggregation Engine ---

/**
 * @brief (Node 13, 17) Tier 0 (N=1): Identity pass-through copy. Base case for reduction.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel forms the base case of the reduction engine. The Host Orchestrator is contractually obligated to invoke this kernel if and only if the number of partials
 * to be reduced is exactly 1."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Utility / Base Case"
 */
__kernel void aggregate_identity(
    /**
     * @param src_buffer_GLOBAL_partial_input The single source partial buffer to be copied.
     *        - Tensor Shape: (src_scalar_NATURAL_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_element_count]
     *        - Validation Preconditions: Buffer must be non-NULL and readable.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_input,

    /**
     * @param dest_buffer_GLOBAL_final_output The destination buffer for the copied data.
     *        - Tensor Shape: (src_scalar_NATURAL_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_element_count]
     *        - Validation Preconditions: Buffer must be non-NULL and host must allocate at least [src_scalar_NATURAL_total_element_count * sizeof(SCALAR_TYPE)] bytes.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_final_output,

    uint src_scalar_NATURAL_total_element_count);

/**
 * @brief (Node 13, 17) Tier 1 (N is small): Reduces partial results using registers.
 * @kernel_contract
 *        - Holistic Constraints: "The Host Orchestrator invokes this tier of the reduction engine for a small number of partials (N > 1)."
 *        - Behavioral Invariants: "The reduction policy (e.g., SUM or AVERAGE) is controlled by the `operation_type` flag."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Tier 1"
 */
__kernel void aggregate_register_reduce(
    /**
     * @param src_buffer_GLOBAL_partial_input The collection of source partial buffers to be reduced.
     *        - Tensor Shape: (src_scalar_NATURAL_in_partials_count, src_scalar_NATURAL_partial_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_in_partials_count, src_scalar_NATURAL_partial_element_count]
     *        - Validation Preconditions: Buffer must be non-NULL and readable.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_input,

    /**
     * @param dest_buffer_GLOBAL_partial_output The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_element_count]
     *        - Validation Preconditions: Host must allocate at least [src_scalar_NATURAL_partial_element_count * sizeof(SCALAR_TYPE)] bytes.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_output,

    uint src_scalar_NATURAL_in_partials_count,
    uint src_scalar_NATURAL_partial_element_count,
    uint src_scalar_FLAG_operation_type);

/**
 * @brief (Node 13, 17) Tier 2 (N is large): Reduces partial results using local memory.
 * @kernel_contract
 *        - Holistic Constraints: "The Host Orchestrator invokes this tier of the reduction engine for a large number of partials."
 *        - Behavioral Invariants: "The reduction policy (e.g., SUM or AVERAGE) is controlled by the `operation_type` flag."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Tier 2 / Work-group Parallel"
 */
__kernel void aggregate_local_reduce(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for performing the intra-work-group reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(SCALAR_TYPE)`.
     *        - Performance Notes: For optimal performance, the work-group size for dimension 0 should be a power of 2 to ensure conflict-free parallel reduction.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_input The collection of source partial buffers to be reduced.
     *        - Tensor Shape: (src_scalar_NATURAL_in_partials_count, src_scalar_NATURAL_partial_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_in_partials_count, src_scalar_NATURAL_partial_element_count]
     *        - Validation Preconditions: Buffer must be non-NULL and readable.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_partial_input,

    /**
     * @param dest_buffer_GLOBAL_partial_output The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_element_count]
     *        - Validation Preconditions: Host must allocate at least [src_scalar_NATURAL_partial_element_count * sizeof(SCALAR_TYPE)] bytes.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_output,

    uint src_scalar_NATURAL_in_partials_count,
    uint src_scalar_NATURAL_partial_element_count,
    uint src_scalar_FLAG_operation_type);

// --- Phase 15-16: Streaming Shared Layer Backpropagation ---

/**
 * @brief (Node 15) Computes partial gradients for shared layer weights from a batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer. Designed for the 'True Streaming' backpropagation model."
 */
__kernel void backprop_shared_weights_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(SCALAR_TYPE)`.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_input The initial, untransformed input data for the batch.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer (Node 4).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_final_grad_hidden_activations The final, consolidated upstream gradient from Node 14.
     *        - Tensor Shape: (src_scalar_NATURAL_final_grad_hidden_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_final_grad_hidden_total_element_count]
     *        - Validation Preconditions: [1] The logical shape assumed by this kernel must match the physical size of the provided buffer, as proven by: (src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count) == src_scalar_NATURAL_final_grad_hidden_total_element_count. [2] Buffer must be the valid output of Node 14.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_final_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_shared The collection buffer for this chunk's computed weight gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_input_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_input_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: linear_batch(src_scalar_NATURAL_batch_chunk_index)
     *        - Validation Preconditions: Host must provide a valid `src_scalar_NATURAL_batch_chunk_index`.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_weights_shared,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_batch_chunk_index,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_num_batch_chunks_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_final_grad_hidden_total_element_count);

/**
 * @brief (Node 16) Computes partial gradients for shared layer biases from a batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer. Designed for the 'True Streaming' backpropagation model."
 */
__kernel void backprop_shared_biases_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(SCALAR_TYPE)`.
     */
    __local SCALAR_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer (Node 4).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_final_grad_hidden_activations The final, consolidated upstream gradient from Node 14.
     *        - Tensor Shape: (src_scalar_NATURAL_final_grad_hidden_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_final_grad_hidden_total_element_count]
     *        - Validation Preconditions: [1] The logical shape assumed by this kernel must match the physical size of the provided buffer, as proven by: (src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count) == src_scalar_NATURAL_final_grad_hidden_total_element_count. [2] Buffer must be the valid output of Node 14.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_final_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Buffer must be non-NULL.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_shared The collection buffer for this chunk's computed bias gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: linear_batch(src_scalar_NATURAL_batch_chunk_index)
     *        - Validation Preconditions: Host must provide a valid `src_scalar_NATURAL_batch_chunk_index`.
     */
    __global SCALAR_TYPE *dest_buffer_GLOBAL_partial_grad_biases_shared,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_batch_chunk_index,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_num_batch_chunks_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_final_grad_hidden_total_element_count);

// --- Phase 19-20: Finalization & Updates ---

/**
 * @brief (Node 19) Applies Adam optimizer update to an entire parameter group. Single dispatch.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "The implementation is strictly forbidden from using `pown` or any equivalent function. The host is solely responsible for providing pre-computed bias correction
 * terms (`beta1_pow_t`, `beta2_pow_t`) to ensure long-term numerical stability."
 *        - Idempotency: "Fundamentally Non-Idempotent (Stateful). Modifies multiple state buffers in-place."
 *        - Synchronization Model: "Stateful Optimizer Update"
 */
__kernel void adam_update(
    /**
     * @param src_buffer_GLOBAL_grad The buffer containing the final, aggregated gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Must be a non-NULL, valid gradient buffer.
     */
    __global const SCALAR_TYPE *src_buffer_GLOBAL_grad,

    /**
     * @param update_buffer_GLOBAL_parameters The parameter buffer to be updated in-place (e.g., weights, biases).
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: [1] Must be non-NULL. [2] The physical memory layout must be identical to `grad`, `m1`, and `m2` buffers.
     */
    __global SCALAR_TYPE *update_buffer_GLOBAL_parameters,

    /**
     * @param update_buffer_GLOBAL_m1 The first moment vector buffer to be updated in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: [1] Must be non-NULL. [2] The physical memory layout must be identical to other state buffers.
     */
    __global SCALAR_TYPE *update_buffer_GLOBAL_m1,

    /**
     * @param update_buffer_GLOBAL_m2 The second moment vector buffer to be updated in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: [1] Must be non-NULL. [2] The physical memory layout must be identical to other state buffers.
     */
    __global SCALAR_TYPE *update_buffer_GLOBAL_m2,

    SCALAR_TYPE src_scalar_REAL_learning_rate,
    SCALAR_TYPE src_scalar_REAL_beta1_pow_t,
    SCALAR_TYPE src_scalar_REAL_beta2_pow_t,
    SCALAR_TYPE src_scalar_REAL_beta1,
    SCALAR_TYPE src_scalar_REAL_beta2,
    SCALAR_TYPE src_scalar_REAL_epsilon,
    uint        src_scalar_NATURAL_parameter_count);

/**
 * @brief (Node 20) Clamps temperature parameters within a [min, max] range.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Enforces `temps = clamp(temps, min_value, max_value)` for each element."
 *        - Idempotency: "Fundamentally Non-Idempotent (Stateful). Modifies the temps buffer in-place."
 *        - Synchronization Model: "Finalizer Utility"
 */
__kernel void clamp_temperatures(
    /**
     * @param update_buffer_GLOBAL_temps The temperature parameter buffer to be clamped in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: Buffer must be the valid, updated output of Node 19's Adam optimizer.
     */
    __global SCALAR_TYPE *update_buffer_GLOBAL_temps,

    SCALAR_TYPE src_scalar_REAL_min_value,
    SCALAR_TYPE src_scalar_REAL_max_value,
    uint        src_scalar_NATURAL_total_modules_count);

#endif // KERNELS_CL_H
