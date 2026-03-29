/* cpu_kernels.h — Public ABI surface for libcpu_kernels (ADR-015).
 *
 * Declares:  per-kernel argument structs, task_<kernel_name> functions,
 *            execute_reduction_tree, pool lifecycle, get_struct_size_* verifiers.
 *
 * The Python FFI layer (Phase 3B) binds against this header.
 */
#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#include "cpu_simd.h"
#include "cpu_threads.h"
#include "cpu_export.h"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ================================================================
 * Build-Configuration Constants
 * ================================================================ */
#define LOCAL_MEM_BANK_PADDING      1
#define NUMERICAL_STABILITY_EPSILON 1e-7f
#define CACHE_LINE_BYTES            64

#ifndef C_TILE_SIZE
#define C_TILE_SIZE SIMD_WIDTH
#endif

#define PROBLEM_TYPE_CCE 0
#define PROBLEM_TYPE_BCE 1
#define AGG_MODE_SUM     0
#define AGG_MODE_AVERAGE 1

#ifndef uint
typedef uint32_t uint;
#endif

/* ================================================================
 * Argument Structures — one per kernel.
 * Field order: buffer pointers first, then scalars.
 * Order must exactly match the Python ctypes mirror in _ffi_types.py.
 * ================================================================ */

/* --- Act Phase --- */

typedef struct {
    const float* input;
    const float* sample_mask;
    const float* weights_shared_simd_major;
    const float* biases_shared;
    float*       hidden_activations;
    float*       hidden_mask;
    uint         batch_chunk_offset;
    uint         batch_chunk_count;
    uint         total_batch_count;
    uint         padded_input_count;
    uint         padded_hidden_count;
} ForwardPassArgs;

typedef struct {
    const float* hidden_activations;
    const float* hidden_mask;
    const float* weights_module;
    const float* biases_module;
    float*       logits;
    uint         batch_chunk_offset;
    uint         batch_chunk_count;
    uint         module_chunk_offset;
    uint         module_chunk_count;
    uint         class_chunk_offset;
    uint         class_chunk_count;
    uint         total_batch_count;
    uint         hidden_count;
    uint         padded_hidden_count;
    uint         total_output_class_count;
    uint         padded_total_output_class_count;
    uint         total_modules_count;
} RenderLogitsArgs;

/* --- Learn Phase A: Loss & Gradient Production --- */

typedef struct {
    const float* logits;
    const float* temps;
    const int*   targets;
    const float* sample_mask;
    float*       partial_probs;
    float*       final_loss;
    uint         flat_tile_index;
    uint         num_class_chunks;
    uint         classes_per_chunk;
    uint         modules_per_chunk;
    uint         total_batch_count;
    uint         total_output_class_count;
    uint         padded_total_output_class_count;
    uint         total_modules_count;
    uint         total_tile_count;
} CceChunkArgs;

typedef struct {
    const float* logits;
    const float* temps;
    const float* targets;
    const float* sample_mask;
    float*       partial_probs;
    float*       partial_loss;
    uint         flat_tile_index;
    uint         num_class_chunks;
    uint         classes_per_chunk;
    uint         modules_per_chunk;
    uint         total_batch_count;
    uint         total_output_class_count;
    uint         padded_total_output_class_count;
    uint         total_modules_count;
    uint         total_tile_count;
} BceChunkArgs;

typedef struct {
    const float* hidden_activations;
    const float* partial_probs;
    const void*  targets;
    const float* sample_mask;
    float*       partial_grad_weights_module;
    float*       partial_grad_biases_module;
    uint         problem_type;
    uint         flat_tile_index;
    uint         batch_chunk_offset;
    uint         batch_chunk_count;
    uint         num_class_chunks;
    uint         classes_per_chunk;
    uint         modules_per_chunk;
    uint         total_batch_count;
    uint         hidden_count;
    uint         padded_hidden_count;
    uint         total_output_class_count;
    uint         padded_total_output_class_count;
    uint         total_modules_count;
    uint         total_tile_count;
} ModuleParamGradsArgs;

/* --- Learn Phase B: Processing --- */

typedef struct {
    const float* partial_probs;
    const void*  targets;
    const float* sample_mask;
    const float* weights_module;
    float*       partial_grad_hidden_activations_aos;
    uint         problem_type;
    uint         flat_tile_index;
    uint         num_class_chunks;
    uint         classes_per_chunk;
    uint         modules_per_chunk;
    uint         total_batch_count;
    uint         hidden_count;
    uint         padded_hidden_count;
    uint         total_output_class_count;
    uint         padded_total_output_class_count;
    uint         total_modules_count;
    uint         total_tile_count;
} BackpropToHiddenArgs;

typedef struct {
    const float* logits;
    const float* partial_probs;
    const void*  targets;
    const float* sample_mask;
    const float* temps;
    float*       partial_grad_temps;
    uint         problem_type;
    uint         flat_tile_index;
    uint         num_class_chunks;
    uint         classes_per_chunk;
    uint         modules_per_chunk;
    uint         total_batch_count;
    uint         total_output_class_count;
    uint         padded_total_output_class_count;
    uint         total_modules_count;
    uint         total_tile_count;
} TempGradientsArgs;

typedef struct {
    const float* partial_grad_weights_module;
    const float* partial_grad_biases_module;
    const float* partial_grad_temps;
    const float* partial_grad_hidden_activations_aos;
    const float* clipping_threshold_per_item;
    float*       clipped_partial_grad_weights_module;
    float*       clipped_partial_grad_biases_module;
    float*       clipped_partial_grad_temps;
    float*       clipped_partial_grad_hidden_activations_aos;
    uint         use_per_item_norm;
    float        clipping_threshold_t_pre;
    float        epsilon;
    uint         flat_tile_index;
    uint         num_class_chunks;
    uint         classes_per_chunk;
    uint         modules_per_chunk;
    uint         total_batch_count;
    uint         padded_hidden_count;
    uint         padded_total_output_class_count;
    uint         total_tile_count;
} ClipPartialsArgs;

/* --- Learn Phase C: Reduction --- */

typedef struct {
    const float* clipped_partial_grad_hidden_activations_aos;
    float*       clipped_grad_hidden_activations_permuted_soa;
    uint         total_batch_count;
    uint         hidden_count;
    uint         padded_hidden_count;
    uint         total_modules_count;
    uint         padded_total_modules_count;
    uint         num_module_chunks;
    uint         modules_per_chunk;
    uint         num_class_chunks;
    uint         total_tile_count;
} GatherPermuteArgs;

typedef struct {
    const float* partial_collection;
    const uint*  offset_lists_flat;
    const uint*  stage_offsets_into_list;
    const uint*  stage_fan_in;
    const uint*  stage_node_counts;
    float*       staging_buffer_0;
    float*       staging_buffer_1;
    float*       output;
    uint         partial_width;
    uint         num_stages;
    float        t_algorithmic;
    float        lambda;
    float        fp_max;
    float        epsilon;
} ReductionTreePlanC;

typedef struct {
    const float* grad_hidden_activations_permuted_soa;
    float*       summed_grad_hidden_activations;
    float        fp_max;
    float        policy_t_algorithmic;
    float        policy_lambda;
    uint         policy_max_k;
    float        epsilon;
    uint         total_batch_count;
    uint         padded_hidden_count;
    uint         total_modules_count;
    uint         padded_total_modules_count;
} StabilizeReduceArgs;

typedef struct {
    float*  intermediate_grad;
    float   clipping_threshold_t_j;
    float   epsilon;
    uint    parameter_count;
} ClipIntermediateArgs;

/* --- Learn Phase D: Streaming Backprop --- */

typedef struct {
    const float* input;
    const float* hidden_activations;
    const float* summed_grad_hidden_activations;
    const float* sample_mask;
    float*       partial_grad_weights_shared;
    uint         batch_chunk_offset;
    uint         batch_chunk_count;
    uint         batch_chunk_index;
    uint         total_batch_count;
    uint         num_batch_chunks_count;
    uint         padded_input_count;
    uint         padded_hidden_count;
    uint         final_grad_hidden_total_element_count;
} BackpropSharedWeightsArgs;

typedef struct {
    const float* hidden_activations;
    const float* summed_grad_hidden_activations;
    const float* sample_mask;
    float*       partial_grad_biases_shared;
    uint         batch_chunk_offset;
    uint         batch_chunk_count;
    uint         batch_chunk_index;
    uint         total_batch_count;
    uint         num_batch_chunks_count;
    uint         padded_hidden_count;
    uint         final_grad_hidden_total_element_count;
} BackpropSharedBiasesArgs;

typedef struct {
    const float* partial_grad_weights_shared;
    const float* partial_grad_biases_shared;
    float*       clipped_partial_grad_weights_shared;
    float*       clipped_partial_grad_biases_shared;
    float        clipping_threshold_t_pre;
    float        epsilon;
    uint         weights_parameter_count;
    uint         biases_parameter_count;
    uint         weights_write_offset_elements;
    uint         biases_write_offset_elements;
    uint         num_batch_chunks;
} ClipSharedGradsArgs;

/* --- Update Phase --- */

typedef struct {
    const float* summed_grad;
    float*       final_grad;
    float        effective_batch_size;
    float        epsilon;
    uint         parameter_count;
} NormalizeGradientsArgs;

typedef struct {
    const float* final_grad;
    float*       parameters;
    float*       m1;
    float*       m2;
    float        learning_rate;
    float        beta1_pow_t;
    float        beta2_pow_t;
    float        beta1;
    float        beta2;
    float        epsilon;
    uint         parameter_count;
} AdamUpdateArgs;

typedef struct {
    float* temperatures;
    float  min_value;
    float  max_value;
    uint   total_modules_count;
} ClampTemperaturesArgs;

/* ================================================================
 * Task Function Declarations (uniform signature: void*, uint, uint)
 * ================================================================ */

/* Act Phase */
CPU_KERNELS_EXPORT void task_forward_pass(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_render_logits(void* args, uint task_index, uint thread_id);

/* Learn Phase A: Production */
CPU_KERNELS_EXPORT void task_cce_probs_loss(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_bce_probs_loss(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_module_param_grads(void* args, uint task_index, uint thread_id);

/* Learn Phase B: Processing */
CPU_KERNELS_EXPORT void task_backprop_to_hidden(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_temp_gradients(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_clip_partial_grads(void* args, uint task_index, uint thread_id);

/* Learn Phase C: Reduction */
CPU_KERNELS_EXPORT void task_gather_permute_grad_h(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_stabilize_reduce_grad_h(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_clip_intermediate(void* args, uint task_index, uint thread_id);

/* Learn Phase D: Streaming Backprop */
CPU_KERNELS_EXPORT void task_backprop_shared_weights(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_backprop_shared_biases(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_clip_shared_grads(void* args, uint task_index, uint thread_id);

/* Update Phase */
CPU_KERNELS_EXPORT void task_normalize_gradients(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_adam_update(void* args, uint task_index, uint thread_id);
CPU_KERNELS_EXPORT void task_clamp_temperatures(void* args, uint task_index, uint thread_id);

/* ================================================================
 * Reduction Engine
 * ================================================================ */
CPU_KERNELS_EXPORT void execute_reduction_tree(ThreadPool* pool,
                                                ReductionTreePlanC* plan);

/* ================================================================
 * Thread Pool Lifecycle (re-exported with visibility)
 * ================================================================ */
CPU_KERNELS_EXPORT ThreadPool* pool_create(uint num_threads);
CPU_KERNELS_EXPORT void        pool_destroy(ThreadPool* pool);
CPU_KERNELS_EXPORT void        pool_dispatch_and_wait(ThreadPool* pool,
                                                      void (*fn)(void*, uint, uint),
                                                      void* args, uint task_count);

/* ================================================================
 * SIMD width query (for Python-side discovery)
 * ================================================================ */
CPU_KERNELS_EXPORT uint get_simd_width(void);

/* ================================================================
 * Layout Verification Exports (ADR-015)
 * ================================================================ */
CPU_KERNELS_EXPORT size_t get_struct_size_forward_pass_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_render_logits_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_cce_chunk_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_bce_chunk_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_module_param_grads_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_to_hidden_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_temp_gradients_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_clip_partials_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_gather_permute_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan(void);
CPU_KERNELS_EXPORT size_t get_struct_size_stabilize_reduce_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_clip_intermediate_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_weights_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_biases_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_clip_shared_grads_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_normalize_gradients_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_adam_update_args(void);
CPU_KERNELS_EXPORT size_t get_struct_size_clamp_temperatures_args(void);

#endif /* CPU_KERNELS_H */
