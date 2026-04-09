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
#include "cpu_fp8.h"
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

/* ADR-031: Sample mask bitmask access — 32 samples per uint word, LSB-first */
static inline uint load_sample_mask(const uint *mask_words, uint sample_index) {
    return (mask_words[sample_index >> 5u] >> (sample_index & 31u)) & 1u;
}

/* ================================================================
 * Multi-Precision Configuration (ADR-008, ADR-023 §2.3, ADR-024 §4)
 *
 * Eleven precision variants are compiled using the three-axis scheme:
 * STORAGE_T × COMPUTE_T × STATE_T, with suffix s{s}c{c}x{x}.
 * Each variant has suffixed struct types and function names.
 *
 * COMPUTE_T is a per-instantiation parameter (ADR-024 §4.2).
 * The prior cpu_compute_t typedef is removed.
 *
 * STORAGE_T affects buffer pointers for bandwidth-optimized transient data,
 * COMPUTE_T affects arithmetic precision, STATE_T affects persistent optimizer
 * state (weights, biases, momentum).
 * ================================================================ */

/* --- Macro: declare all structs for one precision variant --- */
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, COMPUTE_T, STATE_T)       \
                                                                               \
/* --- Act Phase --- */                                                        \
typedef struct {                                                               \
    const STORAGE_T* input;                                                    \
    const uint*      sample_mask;                                              \
    const STATE_T*   weights_shared_simd_major;                                \
    const STATE_T*   biases_shared;                                            \
    STORAGE_T*       hidden_activations;                                       \
    STORAGE_T*       hidden_mask;                                              \
    uint             FLAG_produce_hidden_mask;                                 \
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
    uint             FLAG_use_explicit_hidden_mask;                            \
    const uint*      sample_mask;                                              \
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
    const uint*      sample_mask;                                              \
    STORAGE_T*       partial_probs;                                            \
    COMPUTE_T*       final_loss;                                               \
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
    const uint*      sample_mask;                                              \
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
    const uint*      sample_mask;                                              \
    const STATE_T*   temps;                                                    \
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
    const uint*      sample_mask;                                              \
    const STATE_T*   weights_module;                                           \
    const STATE_T*   temps;                                                    \
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
    const uint*      sample_mask;                                              \
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
    const COMPUTE_T* clipping_threshold_per_item;                              \
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
    COMPUTE_T*       output;                                                   \
    uint             partial_width;                                            \
    uint             num_stages;                                               \
    COMPUTE_T        t_algorithmic;                                             \
    COMPUTE_T        lambda;                                                   \
    COMPUTE_T        fp_max;                                                   \
    COMPUTE_T        epsilon;                                                  \
} ReductionTreePlanC_##SUFFIX;                                                 \
                                                                               \
/* ADR-026: Compute-entry variant — reads from COMPUTE_T partial_collection */ \
typedef struct {                                                               \
    const COMPUTE_T* partial_collection;                                       \
    const uint*      offset_lists_flat;                                        \
    const uint*      stage_offsets_into_list;                                  \
    const uint*      stage_fan_in;                                             \
    const uint*      stage_node_counts;                                        \
    COMPUTE_T*       staging_buffer_0;                                         \
    COMPUTE_T*       staging_buffer_1;                                         \
    COMPUTE_T*       output;                                                   \
    uint             partial_width;                                            \
    uint             num_stages;                                               \
    COMPUTE_T        t_algorithmic;                                            \
    COMPUTE_T        lambda;                                                   \
    COMPUTE_T        fp_max;                                                   \
    COMPUTE_T        epsilon;                                                  \
} ReductionTreePlanComputeEntry_##SUFFIX;                                      \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* grad_hidden_activations_permuted_soa;                     \
    COMPUTE_T*       summed_grad_hidden_activations;                           \
    COMPUTE_T        fp_max;                                                   \
    COMPUTE_T        policy_t_algorithmic;                                     \
    COMPUTE_T        policy_lambda;                                            \
    uint             policy_max_k;                                             \
    COMPUTE_T        epsilon;                                                  \
    uint             total_batch_count;                                        \
    uint             padded_hidden_count;                                      \
    uint             total_modules_count;                                      \
    uint             padded_total_modules_count;                               \
} StabilizeReduceArgs_##SUFFIX;                                                \
                                                                               \
typedef struct {                                                               \
    COMPUTE_T*       intermediate_grad;                                        \
    COMPUTE_T        clipping_threshold_t_j;                                   \
    COMPUTE_T        epsilon;                                                  \
    uint             parameter_count;                                          \
} ClipIntermediateArgs_##SUFFIX;                                               \
                                                                               \
/* --- Learn Phase D: Streaming Backprop --- */                                \
typedef struct {                                                               \
    const STORAGE_T* input;                                                    \
    const STORAGE_T* hidden_activations;                                       \
    const STORAGE_T* hidden_mask;                                              \
    uint             FLAG_use_explicit_hidden_mask;                            \
    const COMPUTE_T* summed_grad_hidden_activations;                           \
    const uint*      sample_mask;                                              \
    STORAGE_T*       partial_grad_weights_shared;                              \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             batch_chunk_index;                                        \
    uint             total_batch_count;                                        \
    uint             num_batch_chunks;                                   \
    uint             padded_input_count;                                       \
    uint             padded_hidden_count;                                      \
    uint             final_grad_hidden_total_element_count;                    \
} BackpropSharedWeightsArgs_##SUFFIX;                                          \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* hidden_activations;                                       \
    const STORAGE_T* hidden_mask;                                              \
    uint             FLAG_use_explicit_hidden_mask;                            \
    const COMPUTE_T* summed_grad_hidden_activations;                           \
    const uint*      sample_mask;                                              \
    STORAGE_T*       partial_grad_biases_shared;                               \
    uint             batch_chunk_offset;                                       \
    uint             batch_chunk_count;                                        \
    uint             batch_chunk_index;                                        \
    uint             total_batch_count;                                        \
    uint             num_batch_chunks;                                   \
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
    const COMPUTE_T* summed_grad;                                              \
    COMPUTE_T*       final_grad;                                               \
    COMPUTE_T        effective_batch_size;                                     \
    COMPUTE_T        epsilon;                                                  \
    uint             parameter_count;                                          \
} NormalizeGradientsArgs_##SUFFIX;                                             \
                                                                               \
typedef struct {                                                               \
    const COMPUTE_T* final_grad;                                               \
    STATE_T*         parameters;                                               \
    STATE_T*         m1;                                                       \
    STATE_T*         m2;                                                       \
    COMPUTE_T        learning_rate;                                            \
    COMPUTE_T        beta1_pow_t;                                              \
    COMPUTE_T        beta2_pow_t;                                              \
    COMPUTE_T        beta1;                                                    \
    COMPUTE_T        beta2;                                                    \
    COMPUTE_T        epsilon;                                                  \
    uint             parameter_offset;                                         \
    uint             parameter_count;                                          \
    uint             total_parameter_count;                                    \
} AdamUpdateArgs_##SUFFIX;                                                     \
                                                                               \
typedef struct {                                                               \
    STATE_T*         temperatures;                                             \
    COMPUTE_T        min_value;                                                \
    COMPUTE_T        max_value;                                                \
    uint             parameter_offset;                                         \
    uint             parameter_count;                                          \
    uint             total_parameter_count;                                    \
} ClampTemperaturesArgs_##SUFFIX;

/* Instantiate structs for all 14 three-axis precision combinations (ADR-024 §4.1):
 *   s{storage}c{compute}x{state}                                               */
DECLARE_PRECISION_STRUCTS(s16c16x16, _Float16, _Float16, _Float16)
DECLARE_PRECISION_STRUCTS(s16c16x32, _Float16, _Float16, float)
DECLARE_PRECISION_STRUCTS(s16c16x64, _Float16, _Float16, double)
DECLARE_PRECISION_STRUCTS(s16c32x16, _Float16, float,  _Float16)
DECLARE_PRECISION_STRUCTS(s16c32x32, _Float16, float,  float)
DECLARE_PRECISION_STRUCTS(s16c32x64, _Float16, float,  double)
DECLARE_PRECISION_STRUCTS(s16c64x16, _Float16, double, _Float16)
DECLARE_PRECISION_STRUCTS(s16c64x32, _Float16, double, float)
DECLARE_PRECISION_STRUCTS(s16c64x64, _Float16, double, double)
DECLARE_PRECISION_STRUCTS(s32c32x32, float,    float,  float)
DECLARE_PRECISION_STRUCTS(s32c32x64, float,    float,  double)
DECLARE_PRECISION_STRUCTS(s32c64x32, float,    double, float)
DECLARE_PRECISION_STRUCTS(s32c64x64, float,    double, double)
DECLARE_PRECISION_STRUCTS(s64c64x64, double,   double, double)

/* FP8 E4M3 storage variants with FP32/FP64 compute (ADR-025 §5.3) */
DECLARE_PRECISION_STRUCTS(s8e4c32x32, cpu_fp8_e4m3, float,  float)
DECLARE_PRECISION_STRUCTS(s8e4c32x64, cpu_fp8_e4m3, float,  double)
DECLARE_PRECISION_STRUCTS(s8e4c64x64, cpu_fp8_e4m3, double, double)

/* FP8 E5M2 storage variants with FP32/FP64 compute */
DECLARE_PRECISION_STRUCTS(s8e5c32x32, cpu_fp8_e5m2, float,  float)
DECLARE_PRECISION_STRUCTS(s8e5c32x64, cpu_fp8_e5m2, float,  double)
DECLARE_PRECISION_STRUCTS(s8e5c64x64, cpu_fp8_e5m2, double, double)

/* FP8 storage variants with FP16 compute — requires _Float16.
 * Guarded by HAS_FLOAT16 (emitted by Meson — see Step 9C.4).
 * The Meson build also skips c16 variant compilation when _Float16 is unavailable.
 */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
/* FP8 E4M3 storage variants with FP16 compute */
DECLARE_PRECISION_STRUCTS(s8e4c16x32, cpu_fp8_e4m3, _Float16, float)
DECLARE_PRECISION_STRUCTS(s8e4c16x64, cpu_fp8_e4m3, _Float16, double)
DECLARE_PRECISION_STRUCTS(s8e4c16x16, cpu_fp8_e4m3, _Float16, _Float16)

/* FP8 E5M2 storage variants with FP16 compute */
DECLARE_PRECISION_STRUCTS(s8e5c16x32, cpu_fp8_e5m2, _Float16, float)
DECLARE_PRECISION_STRUCTS(s8e5c16x64, cpu_fp8_e5m2, _Float16, double)
DECLARE_PRECISION_STRUCTS(s8e5c16x16, cpu_fp8_e5m2, _Float16, _Float16)

/* FP8 storage variants with FP16 state, FP32/FP64 compute */
DECLARE_PRECISION_STRUCTS(s8e4c32x16, cpu_fp8_e4m3, float,    _Float16)
DECLARE_PRECISION_STRUCTS(s8e4c64x16, cpu_fp8_e4m3, double,   _Float16)
DECLARE_PRECISION_STRUCTS(s8e5c32x16, cpu_fp8_e5m2, float,    _Float16)
DECLARE_PRECISION_STRUCTS(s8e5c64x16, cpu_fp8_e5m2, double,   _Float16)
#endif /* HAS_FLOAT16 */

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
CPU_KERNELS_EXPORT void execute_reduction_tree_from_compute_##SUFFIX(          \
    ThreadPool* pool, ReductionTreePlanComputeEntry_##SUFFIX* plan);           \
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
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan_compute_entry_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_stabilize_reduce_args_##SUFFIX(void);\
CPU_KERNELS_EXPORT size_t get_struct_size_clip_intermediate_args_##SUFFIX(void);\
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_weights_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_biases_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_clip_shared_grads_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_normalize_gradients_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_adam_update_args_##SUFFIX(void);     \
CPU_KERNELS_EXPORT size_t get_struct_size_clamp_temperatures_args_##SUFFIX(void);

/* Declare for all 14 three-axis precision configurations (ADR-024 §4.1) */
DECLARE_PRECISION_FUNCTIONS(s16c16x16)
DECLARE_PRECISION_FUNCTIONS(s16c16x32)
DECLARE_PRECISION_FUNCTIONS(s16c16x64)
DECLARE_PRECISION_FUNCTIONS(s16c32x16)
DECLARE_PRECISION_FUNCTIONS(s16c32x32)
DECLARE_PRECISION_FUNCTIONS(s16c32x64)
DECLARE_PRECISION_FUNCTIONS(s16c64x16)
DECLARE_PRECISION_FUNCTIONS(s16c64x32)
DECLARE_PRECISION_FUNCTIONS(s16c64x64)
DECLARE_PRECISION_FUNCTIONS(s32c32x32)
DECLARE_PRECISION_FUNCTIONS(s32c32x64)
DECLARE_PRECISION_FUNCTIONS(s32c64x32)
DECLARE_PRECISION_FUNCTIONS(s32c64x64)
DECLARE_PRECISION_FUNCTIONS(s64c64x64)

/* FP8 E4M3/E5M2 variants with FP32/FP64 compute (ADR-025 §5.3) */
DECLARE_PRECISION_FUNCTIONS(s8e4c32x32)
DECLARE_PRECISION_FUNCTIONS(s8e4c32x64)
DECLARE_PRECISION_FUNCTIONS(s8e4c64x64)
DECLARE_PRECISION_FUNCTIONS(s8e5c32x32)
DECLARE_PRECISION_FUNCTIONS(s8e5c32x64)
DECLARE_PRECISION_FUNCTIONS(s8e5c64x64)

/* FP8 variants with FP16 compute — requires _Float16 */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
DECLARE_PRECISION_FUNCTIONS(s8e4c16x32)
DECLARE_PRECISION_FUNCTIONS(s8e4c16x64)
DECLARE_PRECISION_FUNCTIONS(s8e4c16x16)
DECLARE_PRECISION_FUNCTIONS(s8e5c16x32)
DECLARE_PRECISION_FUNCTIONS(s8e5c16x64)
DECLARE_PRECISION_FUNCTIONS(s8e5c16x16)
/* FP8 storage with FP16 state, FP32/FP64 compute */
DECLARE_PRECISION_FUNCTIONS(s8e4c32x16)
DECLARE_PRECISION_FUNCTIONS(s8e4c64x16)
DECLARE_PRECISION_FUNCTIONS(s8e5c32x16)
DECLARE_PRECISION_FUNCTIONS(s8e5c64x16)
#endif /* HAS_FLOAT16 */

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
