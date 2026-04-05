// kernels.cl.h
//
// --- ADR-013 Designation ---
// This file is the algorithmic specification document for the
// averaging_ensembled_classifier architecture.  Its @kernel_contract
// blocks and @param annotations constitute the authoritative,
// language-neutral reference that all backend implementations
// (OpenCL, Vulkan SPIR-V, CPU SIMD) must implement against.
// See: adr/ADR-013-kernel-source-strategy.md

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

// --- Mandatory Build-Time Symbols (CONTRACT.md Article 6, amended by ADR-020 §3.6) ---

#ifndef STORAGE_TYPE
#error "System Contract Violation: STORAGE_TYPE must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE
#error "System Contract Violation: COMPUTE_TYPE must be defined by the host build system."
#endif
#ifndef STATE_TYPE
#error "System Contract Violation: STATE_TYPE must be defined by the host build system."
#endif
#ifndef STORAGE_TYPE_IS_HALF
#error "System Contract Violation: STORAGE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE_IS_HALF
#error "System Contract Violation: COMPUTE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE_IS_DOUBLE
#error "System Contract Violation: COMPUTE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif
#ifndef STATE_TYPE_IS_DOUBLE
#error "System Contract Violation: STATE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif
#ifndef SIMD_WIDTH
#error "System Contract Violation: SIMD_WIDTH must be defined by the host build system."
#endif
#ifndef C_TILE_SIZE
#error "System Contract Violation: C_TILE_SIZE must be defined by the host build system."
#endif
#ifndef NUMERICAL_STABILITY_EPSILON
#error "System Contract Violation: NUMERICAL_STABILITY_EPSILON must be defined by the host build system."
#endif

// Enable FP16 extension if using half precision in storage or compute roles.
#if STORAGE_TYPE_IS_HALF
#if !defined(cl_khr_fp16)
#error "FP16 extension (cl_khr_fp16) required for half precision storage but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#if COMPUTE_TYPE_IS_HALF && !STORAGE_TYPE_IS_HALF
#if !defined(cl_khr_fp16)
#error "FP16 extension (cl_khr_fp16) required for half precision compute but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Enable FP64 extension if using double precision in compute or state roles.
#if COMPUTE_TYPE_IS_DOUBLE || STATE_TYPE_IS_DOUBLE
#if !defined(cl_khr_fp64)
#error "FP64 extension (cl_khr_fp64) required for double precision but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// Compute-role zero literal
#if COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_ZERO 0.0
#elif COMPUTE_TYPE_IS_HALF
#define COMPUTE_ZERO ((COMPUTE_TYPE)0.0h)
#else
#define COMPUTE_ZERO ((COMPUTE_TYPE)0.0f)
#endif

// Define standard kernel attributes for OpenCL environment
#define KERNEL_ATTR __attribute__((work_group_size_hint(SIMD_WIDTH, 1, 1)))

// --- Mandatory Architectural Constants (Article 5) ---

// The System Contract defines LOCAL_MEM_BANK_PADDING as a fixed
// architectural constant. The build system MUST provide this exact value.
#if !defined(LOCAL_MEM_BANK_PADDING) || (LOCAL_MEM_BANK_PADDING != 1)
#error "System Contract Violation: LOCAL_MEM_BANK_PADDING must be defined and have a value of exactly 1."
#endif

// --- Precision Boundary Abstractions (ADR-020 §4.4, ADR-023 §1) ------
// These are the sole mechanism for crossing precision role boundaries.
// When STORAGE_TYPE == COMPUTE_TYPE, these compile to identity casts
// that any OpenCL compiler eliminates. No #ifdef on type equality is
// used anywhere in the kernel sources.

static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx)
{
#if STORAGE_TYPE_IS_HALF
    return (COMPUTE_TYPE)vload_half(idx, (__global const half *)buf);
#else
    return (COMPUTE_TYPE)buf[idx];
#endif
}

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
#if STORAGE_TYPE_IS_HALF
    vstore_half((half)val, idx, (__global half *)buf);
#else
    buf[idx] = (STORAGE_TYPE)val;
#endif
}

// FP64 Precision Boundary Notes (ADR-024 §3.2):
// When STATE_TYPE = double and COMPUTE_TYPE = float:
//   load_state: double → float narrowing (precision loss accepted;
//               the value is about to enter lower-precision arithmetic)
//   store_state_update: float → double widening (no precision loss;
//                       the narrower compute value preserves all its bits)
// When STATE_TYPE = double and COMPUTE_TYPE = double:
//   Both casts are identity operations, eliminated by the compiler.
//
// STATE_TYPE = half Note (ADR-024 invariant):
//   STATE_TYPE = half is architecturally valid only when STORAGE_TYPE is also
//   half (enforced by PrecisionConfig.__post_init__), guaranteeing cl_khr_fp16
//   is enabled. Plain array access on half* is valid when the extension is
//   active, so no vload_half/vstore_half path is required here.

static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx)
{
    return (COMPUTE_TYPE)buf[idx];
}

static inline void store_state(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}

// Semantically distinct from store_state(): marks an in-place optimizer
// state mutation. Currently identical; exists for future extensibility.
static inline void store_state_update(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}
// --- End Precision Boundary Abstractions ---------------------------------

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
#ifndef DEBUG_MODE
#define DEBUG_MODE 1
#endif
#ifndef uint
#define uint unsigned int
#endif
#define KERNEL_ATTR
#ifndef LOCAL_MEM_BANK_PADDING
#define LOCAL_MEM_BANK_PADDING 1
#endif
#ifndef STORAGE_TYPE
#define STORAGE_TYPE float
#endif
#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE float
#endif
#ifndef STATE_TYPE
#define STATE_TYPE float
#endif
#ifndef STORAGE_TYPE_IS_HALF
#define STORAGE_TYPE_IS_HALF 0
#endif
#ifndef COMPUTE_TYPE_IS_HALF
#define COMPUTE_TYPE_IS_HALF 0
#endif
#ifndef COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_TYPE_IS_DOUBLE 0
#endif
#ifndef STATE_TYPE_IS_DOUBLE
#define STATE_TYPE_IS_DOUBLE 0
#endif
#ifndef COMPUTE_ZERO
#define COMPUTE_ZERO 0.0f
#endif
#ifndef SIMD_WIDTH
#define SIMD_WIDTH 1
#endif
#ifndef C_TILE_SIZE
#define C_TILE_SIZE 1
#endif
#ifndef NUMERICAL_STABILITY_EPSILON
#define NUMERICAL_STABILITY_EPSILON 1
#endif
#define CLK_LOCAL_MEM_FENCE 0x01
#define CLK_GLOBAL_MEM_FENCE 0x02
#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif
inline int          get_global_id(int dim) { return 0; }
inline int          get_local_id(int dim) { return 0; }
inline int          get_group_id(int dim) { return 0; }
inline int          get_local_size(int dim) { return 1; }
inline int          get_global_size(int dim) { return 1; }
inline int          get_num_groups(int dim) { return 1; }
inline void         barrier(int flags) { (void)flags; }
inline COMPUTE_TYPE clamp(COMPUTE_TYPE val, COMPUTE_TYPE min_val, COMPUTE_TYPE max_val) { return fmin(fmax(val, min_val), max_val); }
inline COMPUTE_TYPE select(COMPUTE_TYPE a, COMPUTE_TYPE b, int c) { return (c) ? b : a; }
// OpenCL pown: use powf to avoid conflict with C23 pown declaration
#define pown(base, exp) ((COMPUTE_TYPE)powf((float)(base), (float)(exp)))
// Host-mode precision boundary stubs (identity operations)
inline COMPUTE_TYPE load_storage(const STORAGE_TYPE *buf, size_t idx) { return (COMPUTE_TYPE)buf[idx]; }
inline void store_storage(STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val) { buf[idx] = (STORAGE_TYPE)val; }
inline COMPUTE_TYPE load_state(const STATE_TYPE *buf, size_t idx) { return (COMPUTE_TYPE)buf[idx]; }
inline void store_state(STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) { buf[idx] = (STATE_TYPE)val; }
// Semantically distinct from store_state(): marks an in-place optimizer
// state mutation. Currently identical; exists for future extensibility.
inline void store_state_update(STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) { buf[idx] = (STATE_TYPE)val; }
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
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs loaded via load_storage(); state-role inputs loaded via load_state(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."
 */
__kernel void forward_pass(
    /**
     * @param update_buffer_LOCAL_simd_tile A local memory resource for tiling to optimize SIMD operations.
     *        - Tensor Shape: (SIMD_WIDTH, SIMD_WIDTH + LOCAL_MEM_BANK_PADDING)
     *        - Padding Contract: {Type: BANK_CONFLICT_AVOIDANCE, Formula: "Pad row stride by LOCAL_MEM_BANK_PADDING"}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Compile-time constant: SIMD_WIDTH, System Contract constant: LOCAL_MEM_BANK_PADDING]
     *        - Validation Preconditions: Host shall allocate size according to the formula derived from this contract.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_simd_tile,

    /**
     * @param src_buffer_GLOBAL_input The primary data source for the computational unit.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Pad row stride to 128-byte alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: [1] The access slice defined by chunk parameters must be within the buffer's bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset +
     * src_scalar_NATURAL_batch_chunk_count) <= src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_input_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_shared_simd_major The learnable shared weights in a SIMD-friendly layout.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count/SIMD_WIDTH, src_scalar_NATURAL_padded_input_count, SIMD_WIDTH)
     *        - Padding Contract: {Type: SIMD, Formula: "hidden_dim padded to SIMD_WIDTH; input_dim padded for alignment"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_input_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_weights_shared_simd_major,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_shared The learnable shared biases.
     *        - Tensor Shape: (src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: SIMD, Formula: "Padded to SIMD_WIDTH"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_padded_hidden_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_biases_shared,

    /**
     * @param dest_buffer_GLOBAL_hidden_activations The output tensor of the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The write slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_hidden_activations,

    /**
     * @param dest_buffer_GLOBAL_hidden_mask Derived mask from ReLU operation (1 if activation > 0, else 0).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The write slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_hidden_mask,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_padded_hidden_count);

// --- Phase 5-7: Module Layer Forward Pass & Loss ---

/**
 * @brief (Node 5) Renders a contiguous slice of the monolithic logits buffer.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Monolithic Slice Renderer. Consumes chunked inputs to render a final slice of a monolithic output buffer."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; logit output narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE. Sparsity-Aware Dot Product: The `hidden_mask` parameter is used as a branch predicate to elide dot-product terms corresponding to ReLU-zeroed hidden units. For each hidden dimension where `hidden_mask[b][h] < 0.5`, the activation and weight reads are skipped entirely. This is a performance optimization exploiting upstream ReLU sparsity; it does not affect mathematical correctness. This optimization is non-mandatory: an implementation that unconditionally evaluates all hidden dimensions produces identical results. The mask's presence in the interface enables but does not require the sparsity exploitation."
 */
__kernel void render_logits_chunk(
    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_hidden_mask The ReLU mask corresponding to the hidden activations.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: SIMD, Formula: "output_class_count padded for SIMD/Cache alignment"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The module and class slices must be within bounds, as proven by: [(src_scalar_NATURAL_module_chunk_offset + src_scalar_NATURAL_module_chunk_count) <=
     * src_scalar_NATURAL_total_modules_count] AND [(src_scalar_NATURAL_class_chunk_offset + src_scalar_NATURAL_class_chunk_count) <= src_scalar_NATURAL_total_output_class_count]. [2] Host shall
     * allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param src_buffer_GLOBAL_CONST_biases_module The learnable biases for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: SIMD, Formula: "output_class_count padded for SIMD alignment"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The module and class slices must be within bounds, as proven by: [(src_scalar_NATURAL_module_chunk_offset + src_scalar_NATURAL_module_chunk_count) <=
     * src_scalar_NATURAL_total_modules_count] AND [(src_scalar_NATURAL_class_chunk_offset + src_scalar_NATURAL_class_chunk_count) <= src_scalar_NATURAL_total_output_class_count]. [2] Host shall
     * allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_biases_module,

    /**
     * @param dest_buffer_GLOBAL_logits The raw, pre-activation output tensor for all modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_logits,

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
 *        - Behavioral Invariants: "The implementation is a fused, indivisible unit for numerically stable Softmax calculation. Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; partial_probs narrowed via store_storage(); final_loss written directly in COMPUTE_TYPE (no narrowing). All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer for probabilities output."
 *        - Kernel Bifurcation: "CONCEPT.md Principle 3(B) — separate kernel required due to incompatible type signatures,
 *          memory layouts, and DAG topology vs. Node 7 (BCE path). CONTRACT §7.0 Exception applies."
 */
__kernel void compute_probs_loss_cce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes for this buffer.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (class indices).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] Values must be in [0, total_output_class_count - 1]. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(int)] bytes for this
     * buffer.
     */
    __global const int *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes for this buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_final_loss The monolithic buffer for final loss values.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] Host shall zero-initialize this buffer. [2] The kernel populates this buffer via a direct scatter-write; no host-side aggregation is required. [3] Host
     * shall allocate exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_final_loss,

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
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; partial_probs narrowed via store_storage(); partial_loss written in COMPUTE_TYPE. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Dual Partial Renderer. Uses flat_tile_index for both probability and loss outputs."
 *        - Kernel Bifurcation: "CONCEPT.md Principle 3(B) — separate kernel required due to incompatible type signatures,
 *          memory layouts, and DAG topology vs. Node 6 (CCE path). CONTRACT §7.0 Exception applies."
 */
__kernel void compute_probs_loss_bce_chunk(
    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] The tile access must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate exactly
     * [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes for this buffer.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (multi-hot encoded).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes for this buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_probs The collection buffer for this tile's computed probabilities.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_probs,

    /**
     * @param dest_buffer_GLOBAL_partial_loss The collection buffer for this tile's computed partial loss.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial_loss,

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
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Dual Partial Renderer. Uses flat_tile_index for both weight and bias gradient outputs."
 */
__kernel void calculate_module_param_grads_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout, type, and total size
     * correspond to the value of `src_scalar_FLAG_problem_type`. [3] The kernel implementation will cast this pointer internally based on the flag.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_module The collection buffer for this tile's computed weight gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: NONE}
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     * [3] Each tile writes only `classes_per_chunk` positions within the `padded_total_output_class_count`-wide innermost dimension. The consumer (Node 11) reads the full padded extent for its L2 norm computation.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_module The collection buffer for this tile's computed bias gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: NONE}
     *        - Initialization Contract: {Type: ZERO_REQUIRED}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STORAGE_TYPE)] bytes.
     * [3] Each tile writes only `classes_per_chunk` positions within the `padded_total_output_class_count`-wide innermost dimension. The consumer (Node 11) reads the full padded extent for its L2 norm computation.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_biases_module,

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
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened upon load; storage-role output narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer for a monolithic intermediate buffer."
 */
__kernel void backprop_error_to_hidden_chunk(
    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout, type, and total size
     * correspond to the value of `src_scalar_FLAG_problem_type`.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes for this buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_weights_module The learnable weights for all classifier modules.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The overarching tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_padded_hidden_count * src_scalar_NATURAL_padded_total_output_class_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_weights_module,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_hidden_activations_aos The collection buffer for this tile's computed upstream gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     * [3] [ARCHITECTURAL SYNCHRONIZATION POINT] The consumer (Node 13) requires a monolithic input collection for its gather operation. Therefore, the Host Orchestrator MUST NOT stream the batch
     * dimension when populating this buffer.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_hidden_activations_aos,

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
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role and state-role inputs widened upon load; partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer for temperature gradients."
 */
__kernel void calculate_chunk_temp_gradients(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_logits The raw, pre-activation output from Node 5.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: CACHE, Formula: "output_class_count padded for alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_modules_count * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_total_output_class_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_logits,

    /**
     * @param src_buffer_GLOBAL_partial_probs The collection of partial probabilities from Node 6 or 7.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_classes_per_chunk]
     *        - Validation Preconditions: [1] The requested tile must be within the total number of tiles, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2]
     * Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_classes_per_chunk *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_probs,

    /**
     * @param src_buffer_GLOBAL_targets The ground truth labels (type-punned pointer).
     *        - Tensor Shape: Varies based on problem type flag.
     *        - Padding Contract: Varies.
     *        - Calculability Proof: Dependent on problem type flag.
     *        - Validation Preconditions: [1] This is a type-punned pointer (`void*`). [2] Host is contractually obligated to provide the correct target buffer whose layout, type, and total size
     * correspond to the value of `src_scalar_FLAG_problem_type`.
     */
    __global const void *src_buffer_GLOBAL_targets,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes for this buffer.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param src_buffer_GLOBAL_CONST_temps The learnable temperature parameters for logit scaling.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: [1] The tile access must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate exactly
     * [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes.
     */
    __global const STATE_TYPE *src_buffer_GLOBAL_CONST_temps,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_temps The collection buffer for this tile's computed temperature gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: [1] The write tile index must be valid, as proven by: src_scalar_NATURAL_flat_tile_index < src_scalar_NATURAL_total_tile_count. [2] Host shall allocate
     * exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_temps,

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

// --- Phase 11: Gradient Clipping ---

/**
 * @brief (Node 11) [Utility Kernel] Computes the total L2 Norm for a single item's partial gradients and conditionally scales them. Supports both a single batch-wide clipping norm and per-item norms.
 * @kernel_contract
 *        - Holistic Constraints: "The kernel processes the complete set of partial gradients for a single logical work item (`flat_tile_index`). The clipping threshold is determined by
 * `src_scalar_FLAG_use_per_item_norm`."
 *        - Behavioral Invariants: "[1] Implements a two-pass algorithm: Norm calculation followed by conditional scaling. [2] An epsilon term shall be used to prevent division by zero when
 * calculating the scaling factor. Precision Boundary Conversion: storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Utility / Stability Primitive. Acts as a barrier for a single item's partial results before reduction."
 */
__kernel void clip_partial_gradients(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reduction of the sum-of-squares.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_grad_weights_module Source buffer from Node 8.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_weights_module,

    /**
     * @param src_buffer_GLOBAL_partial_grad_biases_module Source buffer from Node 8.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_biases_module,

    /**
     * @param src_buffer_GLOBAL_partial_grad_temps Source buffer from Node 10.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_temps,

    /**
     * @param src_buffer_GLOBAL_partial_grad_hidden_activations_aos Source buffer from Node 9.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The `flat_tile_index` must be within bounds. [2] Host must allocate buffer with size consistent with the Calculability Proof.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_hidden_activations_aos,

    /**
     * @param src_buffer_GLOBAL_CONST_clipping_threshold_per_item [CONDITIONAL] A buffer containing a distinct clipping threshold for each item.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count]
     *        - Validation Preconditions: [1] This buffer is read from ONLY IF `src_scalar_FLAG_use_per_item_norm` == 1. [2] If the flag is set, the Host MUST provide a valid buffer of size
     * [src_scalar_NATURAL_total_tile_count * sizeof(STORAGE_TYPE)]. [3] If the flag is not set, the Host MAY pass a NULL pointer for this argument.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_CONST_clipping_threshold_per_item,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_weights_module Output for clipped weight gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_weights_module`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_weights_module,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_biases_module Output for clipped bias gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_padded_total_output_class_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_biases_module`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_biases_module,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_temps Output for clipped temperature gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_temps`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_temps,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos Output for clipped upstream gradients.
     *        - Tensor Shape: Identical to its `src_` counterpart.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: grid_mod_cls(src_scalar_NATURAL_flat_tile_index)
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_partial_grad_hidden_activations_aos`.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,

    /**
     * @param src_scalar_FLAG_use_per_item_norm A flag to select the clipping threshold source.
     *        - Validation Preconditions: Must be 0 or 1. If 0, `src_scalar_REAL_clipping_threshold_t_pre` is used. If 1, the value is sourced from
     * `src_buffer_GLOBAL_CONST_clipping_threshold_per_item`.
     */
    uint src_scalar_FLAG_use_per_item_norm,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_pre [CONDITIONAL] The maximum permissible L2 norm, applied to all items if the controlling flag is 0.
     *        - Validation Preconditions: Must be a positive real number. This value is IGNORED if `src_scalar_FLAG_use_per_item_norm` == 1.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_pre,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    // --- Dimension and Placement Parameters ---
    uint src_scalar_NATURAL_flat_tile_index,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_classes_per_chunk,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_padded_total_output_class_count,
    uint src_scalar_NATURAL_total_tile_count);

// --- Phase 13: Data Layout Transformation & Permutation ---

/**
 * @brief (Node 13) Specialized Kernel: Gathers scattered partial gradients into a single, reduction-ready buffer.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a specialized architectural primitive designed to solve the 'Transpose Illusion' by gathering scattered partial results into a dense, reduction-ready
 * SoA layout."
 *        - Behavioral Invariants: "The gather operation performs an implicit reduction (summation) over the `class_chunk` dimension. Precision Boundary Conversion: storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."
 *        - Synchronization Model: "Global Barrier. This kernel cannot execute until all its clipped partial inputs from Node 11 are fully rendered."
 *        - Idempotency: "Associatively Non-Idempotent"
 */
__kernel void gather_and_permute_grad_hidden_activations(
    /**
     * @param src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos The full collection of *clipped* partial upstream gradients, produced by Node 11.
     *        - Tensor Shape: (src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_tile_count, src_scalar_NATURAL_modules_per_chunk, src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_total_tile_count * src_scalar_NATURAL_modules_per_chunk * src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes. [2] [ARCHITECTURAL SYNCHRONIZATION POINT] The consumer (this kernel) requires a monolithic input fully populated by its
     * preceding dependency, Node (11).
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos,

    /**
     * @param dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa The final, contiguous, SoA-layout buffer ready for reduction by Node 16.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Trailing dimension (`total_modules_count`) is Host-padded to `padded_total_modules_count` for alignment."}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly [(src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count) * src_scalar_NATURAL_padded_total_modules_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,

    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_hidden_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_padded_total_modules_count,
    uint src_scalar_NATURAL_num_module_chunks,
    uint src_scalar_NATURAL_modules_per_chunk,
    uint src_scalar_NATURAL_num_class_chunks,
    uint src_scalar_NATURAL_total_tile_count);

// --- Phase 14, 15 & 20: Aggregation Engine ---

/**
 * @brief (Node 14, 15a & 20a) Tier 1 (N is small): Reduces scattered partial results using registers and an indirection list.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a collection buffer, located via an explicit offset list. This avoids host-side staging
 * copies."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 */
__kernel void aggregate_register_reduce(
    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing all partial results for this stage.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and
     * `src_scalar_NATURAL_partial_width`.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

/**
 * @brief (Node 14, 15a & 20a) Tier 2 (N is large): Reduces scattered partial results using local memory and an indirection list.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a collection buffer, located via an explicit offset list. This avoids host-side staging
 * copies."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage / Work-group Parallel"
 */
__kernel void aggregate_local_reduce(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for performing the intra-work-group reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing all partial results for this stage.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and
     * `src_scalar_NATURAL_partial_width`.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

// --- ADR-026: Precision-Typed Reduction Kernel Variants (Compute-Entry) ---

/**
 * @brief (Node 14, 15a & 20a) Compute-entry variant of aggregate_register_reduce.
 *        Identical algorithm reading COMPUTE_TYPE intermediates directly.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a COMPUTE_TYPE collection buffer, located via an explicit offset list."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. All buffers are compute-role; no precision boundary conversion is required. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 *        - Precision Variant: "Compute-entry variant of aggregate_register_reduce. Used for interior stages of multi-stage reduction trees (where the source is a prior stage's COMPUTE_TYPE output) and for leaf stages whose source collection is natively COMPUTE_TYPE."
 */
__kernel void aggregate_register_reduce_from_compute(
    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing COMPUTE_TYPE intermediate results.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and `src_scalar_NATURAL_partial_width`.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

/**
 * @brief (Node 14, 15a & 20a) Compute-entry variant of aggregate_local_reduce.
 *        Identical algorithm reading COMPUTE_TYPE intermediates directly.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel operates on scattered (non-contiguous) input partials from a COMPUTE_TYPE collection buffer, located via an explicit offset list."
 *        - Behavioral Invariants: "The reduction policy (SUM/AVERAGE) is controlled by the `operation_type` flag. All buffers are compute-role; no precision boundary conversion is required. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage / Work-group Parallel"
 *        - Precision Variant: "Compute-entry variant of aggregate_local_reduce. Used for interior stages of multi-stage reduction trees (where the source is a prior stage's COMPUTE_TYPE output) and for leaf stages whose source collection is natively COMPUTE_TYPE."
 */
__kernel void aggregate_local_reduce_from_compute(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for performing the intra-work-group reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing COMPUTE_TYPE intermediate results.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that encompasses all memory regions referenced by the combination of `src_buffer_GLOBAL_CONST_partial_offset_list` and `src_scalar_NATURAL_partial_width`.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_partial_offset_list The indirection table. Each element is an offset into `src_buffer_GLOBAL_partial_collection`.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_offset_list_count)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_partial_offset_list_count]
     *        - Validation Preconditions: Host must provide a buffer containing exactly `src_scalar_NATURAL_partial_offset_list_count` uints.
     */
    __global const uint *src_buffer_GLOBAL_CONST_partial_offset_list,

    /**
     * @param dest_buffer_GLOBAL_partial The destination buffer for the single, reduced partial result.
     *        - Tensor Shape: (src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly [src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_partial,

    uint src_scalar_NATURAL_partial_offset_list_count,
    uint src_scalar_NATURAL_partial_width,
    uint src_scalar_FLAG_operation_type);

/**
 * @brief (Node 15b, 20b) [Utility Kernel] Applies partial-group-wise clipping to a single, contiguous, intermediate gradient buffer.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a core component of the host-driven, recursive clip-aggregation engine. It atomically computes an L2 norm over its entire input buffer and
 * conditionally scales that buffer in-place."
 *        - Behavioral Invariants: "An epsilon term shall be used to prevent division by zero when calculating the scaling factor. The implementation must use local memory for the norm reduction to be
 * scalable. All buffers are compute-role; no precision boundary conversion is required."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage Clip Primitive"
 */
__kernel void clip_intermediate_grad(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for performing the intra-work-group reduction of the sum-of-squares for the L2 norm.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param update_buffer_GLOBAL_intermediate_grad The buffer to be clipped in-place. This is typically the output of a preceding `aggregate_*` kernel.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host must provide a valid buffer containing exactly `src_scalar_NATURAL_parameter_count` elements.
     */
    __global COMPUTE_TYPE *update_buffer_GLOBAL_intermediate_grad,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_j The clipping threshold for this specific reduction stage `j`.
     *        - Calculability Proof: [Host-side calculation based on the active stabilization policy (e.g., Quadratic Scaling Policy)]
     *        - Validation Preconditions: The value must be a positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_j,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent division by zero during norm calculation.
     *        - Validation Preconditions: Must be a small, positive real number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    /**
     * @param src_scalar_NATURAL_parameter_count The total number of elements in the `update_buffer_GLOBAL_intermediate_grad` buffer.
     *        - Calculability Proof: [Known by Host Orchestrator based on the parameter group being processed]
     *        - Validation Preconditions: Must match the element count of the `update_buffer_GLOBAL_intermediate_grad` buffer.
     */
    uint src_scalar_NATURAL_parameter_count);

// --- Phase 16: Specialized Grad_H Reduction ---
/**
 * @brief (Node 16) Specialized Kernel: Reduces the permuted Grad_H buffer using a self-contained, multi-stage, numerically-stable reduction algorithm.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel performs a complete, row-wise reduction on the monolithic, contiguous SoA buffer produced by the upstream Item Synchronization Point (Node 13)."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE. The kernel's behavior is contractually bound to the following internal logic:
 *
 *          0. **Data Ingress Safety Invariant:**
 *             If the implementation introduces a pre-reduction accumulation step that sums `F` input elements per accumulator (e.g., each thread in a work-group accumulates `ceil(total_modules_count / F_total)` elements), each accumulator MUST be clipped to `src_scalar_REAL_fp_max / F_total` before entering the staged reduction, where `F_total` is the number of accumulators that the first reduction stage will sum. This prevents overflow at the first stage regardless of the total module count or the implementation's parallelism geometry. Implementations with no pre-reduction accumulation (e.g., sequential processing of all elements) need not apply this clip, as the staged reduction's own per-stage safety ceiling is sufficient.
 *
 *          1. **Pre-computation Phase (Single, Initial Calculation):**
 *             a. Determine tactical fan-in: `K_plan = min(src_scalar_NATURAL_policy_max_k, get_local_size(0))`
 *             b. Determine true tree depth: `num_stages = ceil(log(total_modules_count) / log(K_plan))`
 *
 *          2. **Per-Stage Execution (`s` from 0 to `num_stages-1`):**
 *             a. Calculate stage index relative to root: `j = num_stages - 1 - s`
 *             b. Calculate algorithmic policy threshold: `T_policy = src_scalar_REAL_policy_t_algorithmic + (src_scalar_REAL_policy_lambda * j * j)`
 *             c. Calculate hardware safety ceiling for this stage's actual fan-in (`K_actual`): `T_safety = src_scalar_REAL_fp_max / K_actual`
 *             d. Synthesize final, authoritative threshold: `final_threshold_for_stage = min(T_policy, T_safety)`
 *
 *          This sequence is the sole valid method for stabilizing the reduction. Deviation is a contract violation."
 *        - Synchronization Model: "Specialized Reduction Kernel / Global Barrier"
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Architectural Justification: "This kernel's contract directly addresses a potential logical fallacy. A naive analysis might conclude that: (a) the kernel's internal planning violates
 * host/device jurisdictional separation, or (b) the threshold calculation is logically circular (`K` depends on `T` which depends on `J` which depends on `K`). This contract asserts that the design
 * is sound by mandating a strict two-phase execution model that resolves both issues.
 *
 *          The `Pre-computation Phase` firmly establishes the kernel's role as a 'Computational Agent,' not a 'Silent Monolith.' It synthesizes the Host's policy (`policy_max_k`) with its own runtime
 *          context (`get_local_size(0)`) to produce a fixed, non-negotiable reduction plan (`K_plan`, `num_stages`). This linearizes the problem.
 *
 *          The subsequent `Per-Stage Execution Phase` then executes this plan, with all dependencies resolved. This model confirms the Host retains sole control of stabilization policy, while the
 * Device retains sole control of its immediate execution geometry. The public formula in `Behavioral Invariants` makes this collaboration transparent and verifiable, not hidden."
 */
__kernel void stabilize_and_reduce_grad_hidden_activations(
    /**
     * @param update_buffer_LOCAL_reduction_tile A work-group exclusive memory resource for high-bandwidth parallel reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_grad_hidden_activations_permuted_soa The pre-gathered, contiguous input data in SoA layout, ensuring optimal memory access for row-wise reduction.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Trailing dimension (`total_modules_count`) is Host-padded to `padded_total_modules_count` for alignment."}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count, src_scalar_NATURAL_padded_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly [(src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count) * src_scalar_NATURAL_padded_total_modules_count *
     * sizeof(STORAGE_TYPE)] bytes. This buffer must be fully populated by Node 13 before dispatch.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_grad_hidden_activations_permuted_soa,

    /**
     * @param dest_buffer_GLOBAL_summed_grad_hidden_activations The destination for the single, final, summed hidden layer gradient vector.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly [(src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count) * sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_scalar_REAL_fp_max The absolute maximum finite value for the current scalar type. Used to calculate hardware safety ceilings.
     *        - Validation Preconditions: Must be a positive real number (e.g., 65504.0 for FP16).
     */
    COMPUTE_TYPE src_scalar_REAL_fp_max,

    /**
     * @param src_scalar_REAL_policy_t_algorithmic The user's target final gradient norm. Serves as the anchor for the stabilization policy.
     *        - Validation Preconditions: Must be a non-negative real number. A value of 0 indicates a safety-only policy.
     */
    COMPUTE_TYPE src_scalar_REAL_policy_t_algorithmic,

    /**
     * @param src_scalar_REAL_policy_lambda The quadratic scaling parameter that controls the curvature of the stabilization policy funnel.
     *        - Validation Preconditions: Must be a non-negative real number.
     */
    COMPUTE_TYPE src_scalar_REAL_policy_lambda,

    /**
     * @param src_scalar_NATURAL_policy_max_k Host-provided, pre-sanitized upper bound for the kernel's internal reduction fan-in (`K`).
     *        - Validation Preconditions: [1] This parameter is the Host's final, authoritative command on maximum fan-in; it is not a suggestion. [2] The Host is contractually obligated to compute
     * this value by synthesizing three distinct constraints and taking their minimum: a. The user's desired reduction policy (e.g., `K=2` for max reproducibility). b. The physical hardware limits of
     * the target device (e.g., `device.max_work_group_size`). c. The absolute mathematical safety limit required to prevent signal annihilation, derived from a system-defined `min_threshold` (e.g.,
     * `FP_FORMAT_MAX / min_threshold`).
     *        - Performance Notes: "This Host-side synthesis is mandatory because the kernel, by design, does not receive a `min_threshold` parameter. This architectural choice delegates the
     * responsibility for preventing signal annihilation to the Host, allowing the kernel to remain a more focused and efficient computational unit."
     */
    uint src_scalar_NATURAL_policy_max_k,

    COMPUTE_TYPE src_scalar_REAL_epsilon,
    uint        src_scalar_NATURAL_total_batch_count,
    uint        src_scalar_NATURAL_padded_hidden_count,
    uint        src_scalar_NATURAL_total_modules_count,
    uint        src_scalar_NATURAL_padded_total_modules_count);

// --- Phase 17-18: Streaming Shared Layer Backpropagation ---

/**
 * @brief (Node 17) Computes partial gradients for shared layer weights from a batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); compute-role gradient consumed directly; partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE. ReLU derivative is computed internally from `hidden_activations` (mask = activation > 0); no explicit `hidden_mask` input is required. This avoids introducing an additional buffer dependency in the streaming backpropagation path, where minimizing the parameter set of the StreamingLoopNode body reduces orchestration complexity. The sparsity-predicated approach used by Node 5 is architecturally valid here but is not applied."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer. Designed for the 'True Streaming' backpropagation model."
 */
__kernel void backprop_shared_weights_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_input The initial, untransformed input data for the batch.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_input_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_input_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_input,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer (Node 4).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_summed_grad_hidden_activations The final, consolidated upstream gradient from Node 16.
     *        - Tensor Shape: (src_scalar_NATURAL_final_grad_hidden_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_final_grad_hidden_total_element_count]
     *        - Validation Preconditions: The logical shape assumed by this kernel must match the physical size of the provided buffer, as proven by: (src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count) == src_scalar_NATURAL_final_grad_hidden_total_element_count.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_weights_shared The collection buffer for this chunk's computed weight gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_input_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_input_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: linear_batch(src_scalar_NATURAL_batch_chunk_index)
     *        - Validation Preconditions: [1] The write chunk index must be valid, as proven by: src_scalar_NATURAL_batch_chunk_index < src_scalar_NATURAL_num_batch_chunks_count. [2] Host shall
     * allocate exactly [src_scalar_NATURAL_num_batch_chunks_count * src_scalar_NATURAL_padded_input_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_weights_shared,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_batch_chunk_index,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_num_batch_chunks_count,
    uint src_scalar_NATURAL_padded_input_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_final_grad_hidden_total_element_count);

/**
 * @brief (Node 18) Computes partial gradients for shared layer biases from a batch chunk.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role inputs widened via load_storage(); compute-role gradient consumed directly; partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE. ReLU derivative is computed internally from `hidden_activations` (mask = activation > 0); no explicit `hidden_mask` input is required. This avoids introducing an additional buffer dependency in the streaming backpropagation path, where minimizing the parameter set of the StreamingLoopNode body reduces orchestration complexity. The sparsity-predicated approach used by Node 5 is architecturally valid here but is not applied."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Partial Renderer. Designed for the 'True Streaming' backpropagation model."
 */
__kernel void backprop_shared_biases_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reductions.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_hidden_activations The intermediate activations from the shared layer (Node 4).
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host must ensure this buffer was allocated to exactly [src_scalar_NATURAL_total_batch_count * src_scalar_NATURAL_padded_hidden_count *
     * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_summed_grad_hidden_activations The final, consolidated upstream gradient from Node 16.
     *        - Tensor Shape: (src_scalar_NATURAL_final_grad_hidden_total_element_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_final_grad_hidden_total_element_count]
     *        - Validation Preconditions: The logical shape assumed by this kernel must match the physical size of the provided buffer, as proven by: (src_scalar_NATURAL_total_batch_count *
     * src_scalar_NATURAL_padded_hidden_count) == src_scalar_NATURAL_final_grad_hidden_total_element_count.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_sample_mask A tensor defining the validity (1) or padding (0) status of samples.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
     *        - Validation Preconditions: [1] The batch access slice must be within bounds, as proven by: (src_scalar_NATURAL_batch_chunk_offset + src_scalar_NATURAL_batch_chunk_count) <=
     * src_scalar_NATURAL_total_batch_count. [2] Host shall allocate exactly [src_scalar_NATURAL_total_batch_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask,

    /**
     * @param dest_buffer_GLOBAL_partial_grad_biases_shared The collection buffer for this chunk's computed bias gradients.
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks_count, src_scalar_NATURAL_padded_hidden_count]
     *        - Placement Contract: linear_batch(src_scalar_NATURAL_batch_chunk_index)
     *        - Validation Preconditions: [1] The write chunk index must be valid, as proven by: src_scalar_NATURAL_batch_chunk_index < src_scalar_NATURAL_num_batch_chunks_count. [2] Host shall
     * allocate exactly [src_scalar_NATURAL_num_batch_chunks_count * src_scalar_NATURAL_padded_hidden_count * sizeof(STORAGE_TYPE)] bytes.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_partial_grad_biases_shared,

    uint src_scalar_NATURAL_batch_chunk_offset,
    uint src_scalar_NATURAL_batch_chunk_count,
    uint src_scalar_NATURAL_batch_chunk_index,
    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_num_batch_chunks_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_final_grad_hidden_total_element_count);

/**
 * @brief (Node 19) [Utility Kernel] Computes the L2 Norm for a single SHARED GRADIENT
 * chunk, conditionally scales it, and writes the result to a destination memory address
 * provided by the host.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "[1] Implements a two-pass algorithm: Norm calculation followed
 *          by conditional scaling. [2] The L2 norm is computed over the concatenated
 *          vector of both weight and bias gradients for the chunk. Precision Boundary Conversion: storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Streamable Utility / Stability Primitive. The responsibility
 *          for calculating the write offset is delegated entirely to the host, making this kernel
 *          a 'dumb' numerical primitive that writes to an explicitly provided memory location."
 */
__kernel void clip_shared_gradients_chunk(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for work-group reduction of the sum-of-squares.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to the work-group
     *          size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_grad_weights_shared The partial weight gradients for a single
     *        data chunk, produced by Node 17.
     *        - Tensor Shape: (src_scalar_NATURAL_weights_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_weights_parameter_count]
     *        - Validation Preconditions: Host shall ensure this buffer is a contiguous memory
     *          region containing the complete partial weight gradient for the chunk being processed.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_weights_shared,

    /**
     * @param src_buffer_GLOBAL_partial_grad_biases_shared The partial bias gradients for a single
     *        data chunk, produced by Node 18.
     *        - Tensor Shape: (src_scalar_NATURAL_biases_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_biases_parameter_count]
     *        - Validation Preconditions: Host shall ensure this buffer is a contiguous memory
     *          region containing the complete partial bias gradient for the chunk being processed.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_grad_biases_shared,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_weights_shared The COLLECTION buffer for all clipped
     *        partial weight gradients, ready for consumption by an aggregate_* kernel (Node 20).
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_weights_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_weights_parameter_count]
     *        - Validation Preconditions: The Host is responsible for providing a valid
     *          `dest_scalar_NATURAL_weights_write_offset_elements` such that the write operation
     *          remains within the bounds of this collection buffer.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_weights_shared,

    /**
     * @param dest_buffer_GLOBAL_clipped_partial_grad_biases_shared The COLLECTION buffer for all clipped
     *        partial bias gradients, ready for consumption by an aggregate_* kernel (Node 20).
     *        - Tensor Shape: (src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_biases_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_num_batch_chunks, src_scalar_NATURAL_biases_parameter_count]
     *        - Validation Preconditions: The Host is responsible for providing a valid
     *          `dest_scalar_NATURAL_biases_write_offset_elements` such that the write operation
     *          remains within the bounds of this collection buffer.
     */
    __global STORAGE_TYPE *dest_buffer_GLOBAL_clipped_partial_grad_biases_shared,

    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_pre,
    COMPUTE_TYPE src_scalar_REAL_epsilon,
    uint        src_scalar_NATURAL_weights_parameter_count,
    uint        src_scalar_NATURAL_biases_parameter_count,
    uint        dest_scalar_NATURAL_weights_write_offset_elements,
    uint        dest_scalar_NATURAL_biases_write_offset_elements,
    uint        src_scalar_NATURAL_num_batch_chunks);

// --- Phase 21-25: Finalization & Updates ---

/**
 * @brief (Node 21) [Utility Kernel] Normalizes a buffer of summed gradients by dividing each element by the effective batch size.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel is a generic, element-wise scaling utility designed to operate on any parameter group's summed gradient buffer."
 *        - Behavioral Invariants: "Performs element-wise division: `output[i] = input[i] / (effective_batch_size + epsilon)`. An epsilon term MUST be used to prevent division by zero if the
 * effective_batch_size is 0. All buffers are compute-role; no precision boundary conversion is required."
 *        - Idempotency: "Strictly Idempotent"
 *        - Synchronization Model: "Finalizer Utility / Batch-wide Normalizer. Executes after the reduction engine and before the optimizer update."
 */
__kernel void normalize_gradients(
    /**
     * @param src_buffer_GLOBAL_summed_grad The buffer of aggregated, batch-wide gradients from the reduction engine.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host shall ensure this buffer contains the complete, summed gradients for a parameter group before dispatch.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad,

    /**
     * @param dest_buffer_GLOBAL_final_grad The output buffer containing the normalized, average gradients ready for the optimizer.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host shall allocate a buffer with a size and layout identical to `src_buffer_GLOBAL_summed_grad`.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_final_grad,

    /**
     * @param src_scalar_REAL_effective_batch_size The normalization factor.
     *        - Calculability Proof: [Host-side calculation: `sum(src_buffer_GLOBAL_sample_mask)`]
     *        - Validation Preconditions: [1] The Host is contractually obligated to calculate this value by performing a reduction (sum) over the `sample_mask` buffer for the entire batch. [2] The
     * value must be >= 0.
     */
    COMPUTE_TYPE src_scalar_REAL_effective_batch_size,

    /**
     * @param src_scalar_REAL_epsilon A small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    /**
     * @param src_scalar_NATURAL_parameter_count The total number of elements in the gradient buffers.
     *        - Calculability Proof: [Dependent on the specific parameter group being processed]
     *        - Validation Preconditions: Must match the element count of the src/dest buffers.
     */
    uint src_scalar_NATURAL_parameter_count);

/**
 * @brief (Node 24) Applies Adam optimizer update to an entire parameter group. Single dispatch.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "The implementation is strictly forbidden from using `pown` or any equivalent function. The host is solely responsible for providing pre-computed bias correction
 * terms (`beta1_pow_t`, `beta2_pow_t`) to ensure long-term numerical stability. Precision Boundary Conversion: state-role moment and parameter buffers accessed via load_state()/store_state_update(); gradient consumed directly in COMPUTE_TYPE. EMA arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Fundamentally Non-Idempotent (Stateful). Modifies multiple state buffers in-place."
 *        - Synchronization Model: "Stateful Optimizer Update. Consumes final gradients after the Batch Synchronization Point."
 */
__kernel void adam_update(
    /**
     * @param src_buffer_GLOBAL_final_grad The buffer containing the final, normalized, batch-averaged gradients from Node 21.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_parameter_count * sizeof(COMPUTE_TYPE)] bytes for this buffer.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_final_grad,

    /**
     * @param update_buffer_GLOBAL_parameters The parameter buffer to be updated in-place (e.g., weights, biases).
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_parameter_count * sizeof(STATE_TYPE)] bytes. [2] The physical memory layout must be identical to
     * `final_grad`, `m1`, and `m2` buffers.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_parameters,

    /**
     * @param update_buffer_GLOBAL_m1 The first moment vector buffer to be updated in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_parameter_count * sizeof(STATE_TYPE)] bytes. [2] The physical memory layout must be identical to other
     * state buffers.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_m1,

    /**
     * @param update_buffer_GLOBAL_m2 The second moment vector buffer to be updated in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_parameter_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_parameter_count]
     *        - Validation Preconditions: [1] Host shall allocate exactly [src_scalar_NATURAL_parameter_count * sizeof(STATE_TYPE)] bytes. [2] The physical memory layout must be identical to other
     * state buffers.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_m2,

    COMPUTE_TYPE src_scalar_REAL_learning_rate,
    COMPUTE_TYPE src_scalar_REAL_beta1_pow_t,
    COMPUTE_TYPE src_scalar_REAL_beta2_pow_t,
    COMPUTE_TYPE src_scalar_REAL_beta1,
    COMPUTE_TYPE src_scalar_REAL_beta2,
    COMPUTE_TYPE src_scalar_REAL_epsilon,
    uint        src_scalar_NATURAL_parameter_count);

/**
 * @brief (Node 25) Clamps temperature parameters within a [min, max] range.
 * @kernel_contract
 *        - Holistic Constraints: "All constraints are defined by the parameter commentary blocks."
 *        - Behavioral Invariants: "Enforces `temps = clamp(temps, min_value, max_value)` for each element. Precision Boundary Conversion: state-role buffer accessed via load_state()/store_state_update(); clamp arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Fundamentally Non-Idempotent (Stateful). Modifies the temps buffer in-place."
 *        - Synchronization Model: "Finalizer Utility"
 */
__kernel void clamp_temperatures(
    /**
     * @param update_buffer_GLOBAL_temps The temperature parameter buffer to be clamped in-place.
     *        - Tensor Shape: (src_scalar_NATURAL_total_modules_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "state"
     *        - Calculability Proof: [src_scalar_NATURAL_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly [src_scalar_NATURAL_total_modules_count * sizeof(STATE_TYPE)] bytes for this buffer.
     */
    __global STATE_TYPE *update_buffer_GLOBAL_temps,

    COMPUTE_TYPE src_scalar_REAL_min_value,
    COMPUTE_TYPE src_scalar_REAL_max_value,
    uint        src_scalar_NATURAL_total_modules_count);

// --- ADR-019: K-Fan-In Reduction Kernel Primitive ---
// Sentinel value indicating an absent partial in the tail node of a
// K-fan-in reduction stage.  When num_partials is not divisible by K,
// the final node's offset list is padded with this sentinel.
#define SENTINEL_ABSENT_PARTIAL 0xFFFFFFFFu

/**
 * @brief (Node 14, 15a, 20a — multi-stage) Reduces groups of K scattered
 *        partials into independent output nodes with optional per-node L2 clip.
 * @kernel_contract
 *        - Holistic Constraints: "Each work-group processes one reduction node.
 *          The kernel reads K partials per node from the source buffer via an
 *          offset list, sums them, optionally clips the result per-node, and
 *          writes one output vector of partial_width elements. Supports absent
 *          partials via sentinel offset 0xFFFFFFFF for the tail node."
 *        - Behavioral Invariants: "When clipping_threshold > 0, per-node L2
 *          clip is applied: scale = threshold / (norm + epsilon). When
 *          clipping_threshold == 0, clip is bypassed (diagnostic mode).
 *          Epsilon prevents division by zero. Precision Boundary Conversion: storage-role partials widened via load_storage(); compute-role outputs written directly; LOCAL scratch uses COMPUTE_TYPE. All arithmetic exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 */
__kernel void reduce_k_fan_in_and_clip(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group
     *        parallel L2 norm reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to
     *          the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing all
     *        partial results referenced by the offset list.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "storage"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that
     *          encompasses all memory regions referenced by the combination of
     *          `src_buffer_GLOBAL_CONST_offset_list_flat` and
     *          `src_scalar_NATURAL_partial_width`.
     */
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_offset_list_flat Flat offset list with K
     *        consecutive entries per node. Sentinel SENTINEL_ABSENT_PARTIAL
     *        (0xFFFFFFFF) indicates an absent partial in the tail node.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in_K)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_fan_in_K]
     *        - Validation Preconditions: Host must provide a buffer containing exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in_K` uint entries.
     */
    __global const uint *src_buffer_GLOBAL_CONST_offset_list_flat,

    /**
     * @param dest_buffer_GLOBAL_stage_output Contiguous output buffer. Node n writes
     *        at `[n * partial_width, (n+1) * partial_width)`.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)` bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_stage_output,

    /**
     * @param src_scalar_NATURAL_fan_in_K Number of partials to reduce per node.
     *        - Validation Preconditions: Must be >= 2.
     */
    uint src_scalar_NATURAL_fan_in_K,

    /**
     * @param src_scalar_NATURAL_node_count Number of independent reduction nodes.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_node_count,

    /**
     * @param src_scalar_NATURAL_partial_width Number of elements per partial vector.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_partial_width,

    /**
     * @param src_scalar_REAL_clipping_threshold The clipping threshold for this stage.
     *        Value 0.0 disables clip (diagnostic mode).
     *        - Validation Preconditions: Must be >= 0.0.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold,

    /**
     * @param src_scalar_REAL_epsilon Small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon);

/**
 * @brief (Node 14, 15a, 20a — interior stages and compute-role leaf stages)
 *        Compute-entry variant of reduce_k_fan_in_and_clip. Identical algorithm,
 *        but reads COMPUTE_TYPE intermediates rather than STORAGE_TYPE partials.
 * @kernel_contract
 *        - Holistic Constraints: "Each work-group processes one reduction node.
 *          The kernel reads K partials per node from the source buffer via an
 *          offset list, sums them, optionally clips the result per-node, and
 *          writes one output vector of partial_width elements. Supports absent
 *          partials via sentinel offset 0xFFFFFFFF for the tail node."
 *        - Behavioral Invariants: "When clipping_threshold > 0, per-node L2
 *          clip is applied: scale = threshold / (norm + epsilon). When
 *          clipping_threshold == 0, clip is bypassed (diagnostic mode).
 *          Epsilon prevents division by zero. All buffers are compute-role;
 *          no precision boundary conversion is required. All arithmetic
 *          exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 *        - Precision Variant: "Compute-entry variant of reduce_k_fan_in_and_clip.
 *          Used for interior stages of multi-stage reduction trees (where the
 *          source is a prior stage's COMPUTE_TYPE output) and for leaf stages
 *          whose source collection is natively COMPUTE_TYPE (e.g., BCE loss
 *          partials from Node 7)."
 */
__kernel void reduce_k_fan_in_and_clip_from_compute(
    /**
     * @param update_buffer_LOCAL_reduction_tile Local memory for intra-work-group
     *        parallel L2 norm reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal to
     *          the work-group size in dimension 0 multiplied by `sizeof(COMPUTE_TYPE)`.
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_partial_collection The memory pool containing
     *        COMPUTE_TYPE intermediate results from a prior reduction stage or
     *        a natively compute-role partial collection.
     *        - Tensor Shape: Undefined.
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: N/A.
     *        - Validation Preconditions: Host must provide a valid buffer that
     *          encompasses all memory regions referenced by the combination of
     *          `src_buffer_GLOBAL_CONST_offset_list_flat` and
     *          `src_scalar_NATURAL_partial_width`.
     */
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,

    /**
     * @param src_buffer_GLOBAL_CONST_offset_list_flat Flat offset list with K
     *        consecutive entries per node. Sentinel SENTINEL_ABSENT_PARTIAL
     *        (0xFFFFFFFF) indicates an absent partial in the tail node.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in_K)
     *        - Padding Contract: {Type: NONE}
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_fan_in_K]
     *        - Validation Preconditions: Host must provide a buffer containing exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_fan_in_K` uint entries.
     */
    __global const uint *src_buffer_GLOBAL_CONST_offset_list_flat,

    /**
     * @param dest_buffer_GLOBAL_stage_output Contiguous output buffer. Node n writes
     *        at `[n * partial_width, (n+1) * partial_width)`.
     *        - Tensor Shape: (src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_node_count, src_scalar_NATURAL_partial_width]
     *        - Validation Preconditions: Host must allocate exactly
     *          `src_scalar_NATURAL_node_count * src_scalar_NATURAL_partial_width * sizeof(COMPUTE_TYPE)` bytes.
     */
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_stage_output,

    /**
     * @param src_scalar_NATURAL_fan_in_K Number of partials to reduce per node.
     *        - Validation Preconditions: Must be >= 2.
     */
    uint src_scalar_NATURAL_fan_in_K,

    /**
     * @param src_scalar_NATURAL_node_count Number of independent reduction nodes.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_node_count,

    /**
     * @param src_scalar_NATURAL_partial_width Number of elements per partial vector.
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_partial_width,

    /**
     * @param src_scalar_REAL_clipping_threshold The clipping threshold for this stage.
     *        Value 0.0 disables clip (diagnostic mode).
     *        - Validation Preconditions: Must be >= 0.0.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold,

    /**
     * @param src_scalar_REAL_epsilon Small constant to prevent division by zero.
     *        - Validation Preconditions: Must be a small, positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon);

#endif // KERNELS_CL_H
