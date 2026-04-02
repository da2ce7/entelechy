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
 * Multi-Precision Configuration (ADR-008, ADR-023 §2.3)
 *
 * Three precision variants are compiled: s32x32 (FP32 storage/FP32 state),
 * s16x16 (FP16 storage/FP16 state), and s16x32 (FP16 storage/FP32 state).
 * Each variant has suffixed struct types and function names.
 *
 * Computation always uses float (FP32) internally. STORAGE_T affects
 * buffer pointers for bandwidth-optimized transient data, STATE_T
 * affects persistent optimizer state (weights, biases, momentum).
 * ================================================================ */

/* COMPUTE_TYPE is invariant on the CPU backend: always float.
 * Per ADR-023 §2.1: CPU arithmetic always executes at FP32 precision. */
typedef float cpu_compute_t;

/* --- Macro: declare all structs for one precision variant --- */
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)                  \
                                                                               \
/* --- Act Phase --- */                                                        \
typedef struct {                                                               \
    const STORAGE_T* input;                                                    \
    const STORAGE_T* sample_mask;                                              \
    const STATE_T*   weights_shared_simd_major;                                \
    const STATE_T*   biases_shared;                                            \
    STORAGE_T*       hidden_activations;                                       \
    STORAGE_T*       hidden_mask;                                              \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             total_batch_count;                                        \
    uint             padded_input_count;                                       \
    uint             padded_hidden_count;                                      \
} ForwardPassArgs_##SUFFIX;                                                    \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* hidden_activations;                                       \
    const STORAGE_T* hidden_mask;                                              \
    const STATE_T*   weights_module;                                           \
    const STATE_T*   biases_module;                                            \
    STORAGE_T*       logits;                                                   \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             module_chunk_offset;                                      \
    uint             module_chunk_count;                                       \
    uint             class_chunk_offset;                                       \
    uint             class_chunk_count;                                        \
    uint             total_batch_count;                                        \
    uint             hidden_count;                                             \
    uint             padded_hidden_count;                                      \
    uint             total_output_class_count;                                 \
    uint             padded_total_output_class_count;                          \
    uint             total_modules_count;                                      \
} RenderLogitsArgs_##SUFFIX;                                                   \
                                                                               \
/* --- Learn Phase A: Loss & Gradient Production --- */                        \
typedef struct {                                                               \
    const STORAGE_T* logits;                                                   \
    const STATE_T*   temps;                                                    \
    const int*       targets;                                                  \
    const STORAGE_T* sample_mask;                                              \
    STORAGE_T*       partial_probs;                                            \
    float*           final_loss;                                               \
    uint             flat_tile_index;                                          \
    uint             num_class_chunks;                                         \
    uint             classes_per_chunk;                                        \
    uint             modules_per_chunk;                                        \
    uint             total_batch_count;                                        \
    uint             total_output_class_count;                                 \
    uint             padded_total_output_class_count;                          \
    uint             total_modules_count;                                      \
    uint             total_tile_count;                                         \
} CceChunkArgs_##SUFFIX;                                                       \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* logits;                                                   \
    const STATE_T*   temps;                                                    \
    const STORAGE_T* targets;                                                  \
    const STORAGE_T* sample_mask;                                              \
    STORAGE_T*       partial_probs;                                            \
    STORAGE_T*       partial_loss;                                             \
    uint             flat_tile_index;                                          \
    uint             num_class_chunks;                                         \
    uint             classes_per_chunk;                                        \
    uint             modules_per_chunk;                                        \
    uint             total_batch_count;                                        \
    uint             total_output_class_count;                                 \
    uint             padded_total_output_class_count;                          \
    uint             total_modules_count;                                      \
    uint             total_tile_count;                                         \
} BceChunkArgs_##SUFFIX;                                                       \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* hidden_activations;                                       \
    const STORAGE_T* partial_probs;                                            \
    const void*      targets;                                                  \
    const STORAGE_T* sample_mask;                                              \
    STORAGE_T*       partial_grad_weights_module;                              \
    STORAGE_T*       partial_grad_biases_module;                               \
    uint             problem_type;                                             \
    uint             flat_tile_index;                                          \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             num_class_chunks;                                         \
    uint             classes_per_chunk;                                        \
    uint             modules_per_chunk;                                        \
    uint             total_batch_count;                                        \
    uint             hidden_count;                                             \
    uint             padded_hidden_count;                                      \
    uint             total_output_class_count;                                 \
    uint             padded_total_output_class_count;                          \
    uint             total_modules_count;                                      \
    uint             total_tile_count;                                         \
} ModuleParamGradsArgs_##SUFFIX;                                               \
                                                                               \
/* --- Learn Phase B: Processing --- */                                        \
typedef struct {                                                               \
    const STORAGE_T* partial_probs;                                            \
    const void*      targets;                                                  \
    const STORAGE_T* sample_mask;                                              \
    const STATE_T*   weights_module;                                           \
    STORAGE_T*       partial_grad_hidden_activations_aos;                      \
    uint             problem_type;                                             \
    uint             flat_tile_index;                                          \
    uint             num_class_chunks;                                         \
    uint             classes_per_chunk;                                        \
    uint             modules_per_chunk;                                        \
    uint             total_batch_count;                                        \
    uint             hidden_count;                                             \
    uint             padded_hidden_count;                                      \
    uint             total_output_class_count;                                 \
    uint             padded_total_output_class_count;                          \
    uint             total_modules_count;                                      \
    uint             total_tile_count;                                         \
} BackpropToHiddenArgs_##SUFFIX;                                               \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* logits;                                                   \
    const STORAGE_T* partial_probs;                                            \
    const void*      targets;                                                  \
    const STORAGE_T* sample_mask;                                              \
    const STATE_T*   temps;                                                    \
    STORAGE_T*       partial_grad_temps;                                       \
    uint             problem_type;                                             \
    uint             flat_tile_index;                                          \
    uint             num_class_chunks;                                         \
    uint             classes_per_chunk;                                        \
    uint             modules_per_chunk;                                        \
    uint             total_batch_count;                                        \
    uint             total_output_class_count;                                 \
    uint             padded_total_output_class_count;                          \
    uint             total_modules_count;                                      \
    uint             total_tile_count;                                         \
} TempGradientsArgs_##SUFFIX;                                                  \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* partial_grad_weights_module;                              \
    const STORAGE_T* partial_grad_biases_module;                               \
    const STORAGE_T* partial_grad_temps;                                       \
    const STORAGE_T* partial_grad_hidden_activations_aos;                      \
    const STORAGE_T* clipping_threshold_per_item;                              \
    STORAGE_T*       clipped_partial_grad_weights_module;                      \
    STORAGE_T*       clipped_partial_grad_biases_module;                       \
    STORAGE_T*       clipped_partial_grad_temps;                               \
    STORAGE_T*       clipped_partial_grad_hidden_activations_aos;              \
    uint             use_per_item_norm;                                        \
    float            clipping_threshold_t_pre;                                 \
    float            epsilon;                                                  \
    uint             flat_tile_index;                                          \
    uint             num_class_chunks;                                         \
    uint             classes_per_chunk;                                        \
    uint             modules_per_chunk;                                        \
    uint             total_batch_count;                                        \
    uint             padded_hidden_count;                                      \
    uint             padded_total_output_class_count;                          \
    uint             total_tile_count;                                         \
} ClipPartialsArgs_##SUFFIX;                                                   \
                                                                               \
/* --- Learn Phase C: Reduction --- */                                         \
typedef struct {                                                               \
    const STORAGE_T* clipped_partial_grad_hidden_activations_aos;              \
    STORAGE_T*       clipped_grad_hidden_activations_permuted_soa;             \
    uint             total_batch_count;                                        \
    uint             hidden_count;                                             \
    uint             padded_hidden_count;                                      \
    uint             total_modules_count;                                      \
    uint             padded_total_modules_count;                               \
    uint             num_module_chunks;                                        \
    uint             modules_per_chunk;                                        \
    uint             num_class_chunks;                                         \
    uint             total_tile_count;                                         \
} GatherPermuteArgs_##SUFFIX;                                                  \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* partial_collection;                                       \
    const uint*      offset_lists_flat;                                        \
    const uint*      stage_offsets_into_list;                                  \
    const uint*      stage_fan_in;                                             \
    const uint*      stage_node_counts;                                        \
    STORAGE_T*       staging_buffer_0;                                         \
    STORAGE_T*       staging_buffer_1;                                         \
    float*           output;                                                   \
    uint             partial_width;                                            \
    uint             num_stages;                                               \
    float            t_algorithmic;                                            \
    float            lambda;                                                   \
    float            fp_max;                                                   \
    float            epsilon;                                                  \
} ReductionTreePlanC_##SUFFIX;                                                 \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* grad_hidden_activations_permuted_soa;                     \
    float*           summed_grad_hidden_activations;                           \
    float            fp_max;                                                   \
    float            policy_t_algorithmic;                                     \
    float            policy_lambda;                                            \
    uint             policy_max_k;                                             \
    float            epsilon;                                                  \
    uint             total_batch_count;                                        \
    uint             padded_hidden_count;                                      \
    uint             total_modules_count;                                      \
    uint             padded_total_modules_count;                               \
} StabilizeReduceArgs_##SUFFIX;                                                \
                                                                               \
typedef struct {                                                               \
    float*           intermediate_grad;                                        \
    float            clipping_threshold_t_j;                                   \
    float            epsilon;                                                  \
    uint             parameter_count;                                          \
} ClipIntermediateArgs_##SUFFIX;                                               \
                                                                               \
/* --- Learn Phase D: Streaming Backprop --- */                                \
typedef struct {                                                               \
    const STORAGE_T* input;                                                    \
    const STORAGE_T* hidden_activations;                                       \
    const float*     summed_grad_hidden_activations;                           \
    const STORAGE_T* sample_mask;                                              \
    STORAGE_T*       partial_grad_weights_shared;                              \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             batch_chunk_index;                                        \
    uint             total_batch_count;                                        \
    uint             num_batch_chunks_count;                                   \
    uint             padded_input_count;                                       \
    uint             padded_hidden_count;                                      \
    uint             final_grad_hidden_total_element_count;                    \
} BackpropSharedWeightsArgs_##SUFFIX;                                          \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* hidden_activations;                                       \
    const float*     summed_grad_hidden_activations;                           \
    const STORAGE_T* sample_mask;                                              \
    STORAGE_T*       partial_grad_biases_shared;                               \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             batch_chunk_index;                                        \
    uint             total_batch_count;                                        \
    uint             num_batch_chunks_count;                                   \
    uint             padded_hidden_count;                                      \
    uint             final_grad_hidden_total_element_count;                    \
} BackpropSharedBiasesArgs_##SUFFIX;                                           \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* partial_grad_weights_shared;                              \
    const STORAGE_T* partial_grad_biases_shared;                               \
    STORAGE_T*       clipped_partial_grad_weights_shared;                      \
    STORAGE_T*       clipped_partial_grad_biases_shared;                       \
    float            clipping_threshold_t_pre;                                 \
    float            epsilon;                                                  \
    uint             weights_parameter_count;                                  \
    uint             biases_parameter_count;                                   \
    uint             weights_write_offset_elements;                            \
    uint             biases_write_offset_elements;                             \
    uint             num_batch_chunks;                                         \
} ClipSharedGradsArgs_##SUFFIX;                                                \
                                                                               \
/* --- Update Phase --- */                                                     \
typedef struct {                                                               \
    const float*     summed_grad;                                              \
    float*           final_grad;                                               \
    float            effective_batch_size;                                     \
    float            epsilon;                                                  \
    uint             parameter_count;                                          \
} NormalizeGradientsArgs_##SUFFIX;                                             \
                                                                               \
typedef struct {                                                               \
    const float*     final_grad;                                               \
    STATE_T*         parameters;                                               \
    STATE_T*         m1;                                                       \
    STATE_T*         m2;                                                       \
    float            learning_rate;                                            \
    float            beta1_pow_t;                                              \
    float            beta2_pow_t;                                              \
    float            beta1;                                                    \
    float            beta2;                                                    \
    float            epsilon;                                                  \
    uint             parameter_count;                                          \
} AdamUpdateArgs_##SUFFIX;                                                     \
                                                                               \
typedef struct {                                                               \
    STATE_T*         temperatures;                                             \
    float            min_value;                                                \
    float            max_value;                                                \
    uint             total_modules_count;                                      \
} ClampTemperaturesArgs_##SUFFIX;

/* Instantiate structs for the three precision configurations (ADR-023 §2.4):
 *   s32x32: FP32 storage, FP32 state (uniform FP32)
 *   s16x16: FP16 storage, FP16 state (uniform FP16)
 *   s16x32: FP16 storage, FP32 state (mixed precision) */
DECLARE_PRECISION_STRUCTS(s16x16, _Float16, _Float16)
DECLARE_PRECISION_STRUCTS(s32x32, float, float)
DECLARE_PRECISION_STRUCTS(s16x32, _Float16, float)

/* ================================================================
 * Task Function Declarations — macro-generated per precision
 * Uniform signature: void*, uint, uint
 * ================================================================ */

#define DECLARE_PRECISION_FUNCTIONS(SUFFIX)                                     \
/* Act Phase */                                                                \
CPU_KERNELS_EXPORT void task_forward_pass_##SUFFIX(                            \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_render_logits_##SUFFIX(                           \
    void* args, uint task_index, uint thread_id);                              \
/* Learn Phase A: Production */                                                \
CPU_KERNELS_EXPORT void task_cce_probs_loss_##SUFFIX(                          \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_bce_probs_loss_##SUFFIX(                          \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_module_param_grads_##SUFFIX(                      \
    void* args, uint task_index, uint thread_id);                              \
/* Learn Phase B: Processing */                                                \
CPU_KERNELS_EXPORT void task_backprop_to_hidden_##SUFFIX(                      \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_temp_gradients_##SUFFIX(                          \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_clip_partial_grads_##SUFFIX(                      \
    void* args, uint task_index, uint thread_id);                              \
/* Learn Phase C: Reduction */                                                 \
CPU_KERNELS_EXPORT void task_gather_permute_grad_h_##SUFFIX(                   \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_stabilize_reduce_grad_h_##SUFFIX(                 \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_clip_intermediate_##SUFFIX(                       \
    void* args, uint task_index, uint thread_id);                              \
/* Learn Phase D: Streaming Backprop */                                        \
CPU_KERNELS_EXPORT void task_backprop_shared_weights_##SUFFIX(                 \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_backprop_shared_biases_##SUFFIX(                  \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_clip_shared_grads_##SUFFIX(                       \
    void* args, uint task_index, uint thread_id);                              \
/* Update Phase */                                                             \
CPU_KERNELS_EXPORT void task_normalize_gradients_##SUFFIX(                     \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_adam_update_##SUFFIX(                             \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_clamp_temperatures_##SUFFIX(                      \
    void* args, uint task_index, uint thread_id);                              \
/* Reduction Engine */                                                         \
CPU_KERNELS_EXPORT void execute_reduction_tree_##SUFFIX(                       \
    ThreadPool* pool, ReductionTreePlanC_##SUFFIX* plan);                      \
/* Layout Verification (ADR-015) */                                            \
CPU_KERNELS_EXPORT size_t get_struct_size_forward_pass_args_##SUFFIX(void);    \
CPU_KERNELS_EXPORT size_t get_struct_size_render_logits_args_##SUFFIX(void);   \
CPU_KERNELS_EXPORT size_t get_struct_size_cce_chunk_args_##SUFFIX(void);       \
CPU_KERNELS_EXPORT size_t get_struct_size_bce_chunk_args_##SUFFIX(void);       \
CPU_KERNELS_EXPORT size_t get_struct_size_module_param_grads_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_to_hidden_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_temp_gradients_args_##SUFFIX(void);  \
CPU_KERNELS_EXPORT size_t get_struct_size_clip_partials_args_##SUFFIX(void);   \
CPU_KERNELS_EXPORT size_t get_struct_size_gather_permute_args_##SUFFIX(void);  \
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan_##SUFFIX(void);  \
CPU_KERNELS_EXPORT size_t get_struct_size_stabilize_reduce_args_##SUFFIX(void);\
CPU_KERNELS_EXPORT size_t get_struct_size_clip_intermediate_args_##SUFFIX(void);\
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_weights_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_biases_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_clip_shared_grads_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_normalize_gradients_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_adam_update_args_##SUFFIX(void);     \
CPU_KERNELS_EXPORT size_t get_struct_size_clamp_temperatures_args_##SUFFIX(void);

/* Declare for all three precision configurations */
DECLARE_PRECISION_FUNCTIONS(s16x16)
DECLARE_PRECISION_FUNCTIONS(s32x32)
DECLARE_PRECISION_FUNCTIONS(s16x32)

/* Transitional function name aliases (removed when FFI layer is updated) */
#define task_forward_pass_fp32        task_forward_pass_s32x32
#define task_forward_pass_fp16        task_forward_pass_s16x16
#define task_render_logits_fp32       task_render_logits_s32x32
#define task_render_logits_fp16       task_render_logits_s16x16
#define task_cce_probs_loss_fp32      task_cce_probs_loss_s32x32
#define task_cce_probs_loss_fp16      task_cce_probs_loss_s16x16
#define task_bce_probs_loss_fp32      task_bce_probs_loss_s32x32
#define task_bce_probs_loss_fp16      task_bce_probs_loss_s16x16
#define task_module_param_grads_fp32  task_module_param_grads_s32x32
#define task_module_param_grads_fp16  task_module_param_grads_s16x16
#define task_backprop_to_hidden_fp32  task_backprop_to_hidden_s32x32
#define task_backprop_to_hidden_fp16  task_backprop_to_hidden_s16x16
#define task_temp_gradients_fp32      task_temp_gradients_s32x32
#define task_temp_gradients_fp16      task_temp_gradients_s16x16
#define task_clip_partial_grads_fp32  task_clip_partial_grads_s32x32
#define task_clip_partial_grads_fp16  task_clip_partial_grads_s16x16
#define task_gather_permute_grad_h_fp32  task_gather_permute_grad_h_s32x32
#define task_gather_permute_grad_h_fp16  task_gather_permute_grad_h_s16x16
#define task_stabilize_reduce_grad_h_fp32 task_stabilize_reduce_grad_h_s32x32
#define task_stabilize_reduce_grad_h_fp16 task_stabilize_reduce_grad_h_s16x16
#define task_clip_intermediate_fp32   task_clip_intermediate_s32x32
#define task_clip_intermediate_fp16   task_clip_intermediate_s16x16
#define task_backprop_shared_weights_fp32 task_backprop_shared_weights_s32x32
#define task_backprop_shared_weights_fp16 task_backprop_shared_weights_s16x16
#define task_backprop_shared_biases_fp32  task_backprop_shared_biases_s32x32
#define task_backprop_shared_biases_fp16  task_backprop_shared_biases_s16x16
#define task_clip_shared_grads_fp32   task_clip_shared_grads_s32x32
#define task_clip_shared_grads_fp16   task_clip_shared_grads_s16x16
#define task_normalize_gradients_fp32 task_normalize_gradients_s32x32
#define task_normalize_gradients_fp16 task_normalize_gradients_s16x16
#define task_adam_update_fp32         task_adam_update_s32x32
#define task_adam_update_fp16         task_adam_update_s16x16
#define task_clamp_temperatures_fp32  task_clamp_temperatures_s32x32
#define task_clamp_temperatures_fp16  task_clamp_temperatures_s16x16
#define execute_reduction_tree_fp32   execute_reduction_tree_s32x32
#define execute_reduction_tree_fp16   execute_reduction_tree_s16x16
#define get_struct_size_forward_pass_args_fp32  get_struct_size_forward_pass_args_s32x32
#define get_struct_size_forward_pass_args_fp16  get_struct_size_forward_pass_args_s16x16
#define get_struct_size_render_logits_args_fp32 get_struct_size_render_logits_args_s32x32
#define get_struct_size_render_logits_args_fp16 get_struct_size_render_logits_args_s16x16
#define get_struct_size_cce_chunk_args_fp32     get_struct_size_cce_chunk_args_s32x32
#define get_struct_size_cce_chunk_args_fp16     get_struct_size_cce_chunk_args_s16x16
#define get_struct_size_bce_chunk_args_fp32     get_struct_size_bce_chunk_args_s32x32
#define get_struct_size_bce_chunk_args_fp16     get_struct_size_bce_chunk_args_s16x16
#define get_struct_size_module_param_grads_args_fp32 get_struct_size_module_param_grads_args_s32x32
#define get_struct_size_module_param_grads_args_fp16 get_struct_size_module_param_grads_args_s16x16
#define get_struct_size_backprop_to_hidden_args_fp32 get_struct_size_backprop_to_hidden_args_s32x32
#define get_struct_size_backprop_to_hidden_args_fp16 get_struct_size_backprop_to_hidden_args_s16x16
#define get_struct_size_temp_gradients_args_fp32 get_struct_size_temp_gradients_args_s32x32
#define get_struct_size_temp_gradients_args_fp16 get_struct_size_temp_gradients_args_s16x16
#define get_struct_size_clip_partials_args_fp32 get_struct_size_clip_partials_args_s32x32
#define get_struct_size_clip_partials_args_fp16 get_struct_size_clip_partials_args_s16x16
#define get_struct_size_gather_permute_args_fp32 get_struct_size_gather_permute_args_s32x32
#define get_struct_size_gather_permute_args_fp16 get_struct_size_gather_permute_args_s16x16
#define get_struct_size_reduction_tree_plan_fp32 get_struct_size_reduction_tree_plan_s32x32
#define get_struct_size_reduction_tree_plan_fp16 get_struct_size_reduction_tree_plan_s16x16
#define get_struct_size_stabilize_reduce_args_fp32 get_struct_size_stabilize_reduce_args_s32x32
#define get_struct_size_stabilize_reduce_args_fp16 get_struct_size_stabilize_reduce_args_s16x16
#define get_struct_size_clip_intermediate_args_fp32 get_struct_size_clip_intermediate_args_s32x32
#define get_struct_size_clip_intermediate_args_fp16 get_struct_size_clip_intermediate_args_s16x16
#define get_struct_size_backprop_shared_weights_args_fp32 get_struct_size_backprop_shared_weights_args_s32x32
#define get_struct_size_backprop_shared_weights_args_fp16 get_struct_size_backprop_shared_weights_args_s16x16
#define get_struct_size_backprop_shared_biases_args_fp32 get_struct_size_backprop_shared_biases_args_s32x32
#define get_struct_size_backprop_shared_biases_args_fp16 get_struct_size_backprop_shared_biases_args_s16x16
#define get_struct_size_clip_shared_grads_args_fp32 get_struct_size_clip_shared_grads_args_s32x32
#define get_struct_size_clip_shared_grads_args_fp16 get_struct_size_clip_shared_grads_args_s16x16
#define get_struct_size_normalize_gradients_args_fp32 get_struct_size_normalize_gradients_args_s32x32
#define get_struct_size_normalize_gradients_args_fp16 get_struct_size_normalize_gradients_args_s16x16
#define get_struct_size_adam_update_args_fp32 get_struct_size_adam_update_args_s32x32
#define get_struct_size_adam_update_args_fp16 get_struct_size_adam_update_args_s16x16
#define get_struct_size_clamp_temperatures_args_fp32 get_struct_size_clamp_temperatures_args_s32x32
#define get_struct_size_clamp_temperatures_args_fp16 get_struct_size_clamp_temperatures_args_s16x16

/* ================================================================
 * Precision-agnostic exports
 * ================================================================ */
CPU_KERNELS_EXPORT ThreadPool* pool_create(uint num_threads);
CPU_KERNELS_EXPORT void        pool_destroy(ThreadPool* pool);
CPU_KERNELS_EXPORT void        pool_dispatch_and_wait(ThreadPool* pool,
                                                      void (*fn)(void*, uint, uint),
                                                      void* args, uint task_count);
CPU_KERNELS_EXPORT uint get_simd_width(void);

#endif /* CPU_KERNELS_H */
