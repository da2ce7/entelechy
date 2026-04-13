/* cpu_kernels.h — Public ABI surface for libcpu_kernels (ADR-015).
 *
 * Declares:  per-kernel argument structs, task_<kernel_name> functions,
 *            execute_reduction_tree, pool lifecycle, get_struct_size_* verifiers.
 *
 * The Python FFI layer (Phase 3B) binds against this header.
 *
 * Argument Naming Convention (ADR-013):
 *   Struct field names match exactly the full-form @param names defined in
 *   kernels/kernels.cl.h. This ensures a single authoritative naming scheme
 *   across all backend implementations (OpenCL, Vulkan, CPU).
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
#define CACHE_LINE_BYTES            64

/* ADR-031: Sentinel for absent partial in reduction tree offset lists (mirrors kernels.cl.h) */
#define SENTINEL_ABSENT_PARTIAL     0xFFFFFFFFu

/* Numerical stability epsilon for softmax/log operations.
 * The value 1e-7f is appropriate for the float range; it is cast to COMPUTE_T
 * at usage sites, so FP16/FP64 variants receive an approximation. This is
 * acceptable since structured per-kernel epsilon fields carry the authoritative
 * value (CONTRACT Article 6). */
#define NUMERICAL_STABILITY_EPSILON 1e-7f

#ifndef C_TILE_SIZE
#define C_TILE_SIZE SIMD_WIDTH
#endif

#define PROBLEM_TYPE_CCE 0
#define PROBLEM_TYPE_BCE 1
#define AGG_MODE_SUM     0
#define AGG_MODE_AVERAGE 1

#ifndef cpu_uint
typedef uint32_t cpu_uint;
#endif
#define uint cpu_uint

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
    const STORAGE_T* src_buffer_GLOBAL_input;                                  \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    const STATE_T*   src_buffer_GLOBAL_CONST_weights_shared_simd_major;        \
    const STATE_T*   src_buffer_GLOBAL_CONST_biases_shared;                    \
    STORAGE_T*       dest_buffer_GLOBAL_hidden_activations;                    \
    STORAGE_T*       dest_buffer_GLOBAL_hidden_mask;                           \
    uint             out_scalar_FLAG_produce_hidden_mask;                     \
    uint             src_scalar_NATURAL_batch_chunk_offset;                    \
    uint             src_scalar_NATURAL_batch_chunk_count;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_input_count;                           \
    uint             src_scalar_NATURAL_padded_input_count;                    \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
} ForwardPassArgs_##SUFFIX;                                                    \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_hidden_activations;                     \
    const STORAGE_T* src_buffer_GLOBAL_hidden_mask;                            \
    uint             src_scalar_FLAG_use_explicit_hidden_mask;                 \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    const STATE_T*   src_buffer_GLOBAL_CONST_weights_module;                   \
    const STATE_T*   src_buffer_GLOBAL_CONST_biases_module;                    \
    STORAGE_T*       dest_buffer_GLOBAL_logits;                                \
    uint             src_scalar_NATURAL_batch_chunk_offset;                    \
    uint             src_scalar_NATURAL_batch_chunk_count;                     \
    uint             src_scalar_NATURAL_module_chunk_offset;                   \
    uint             src_scalar_NATURAL_module_chunk_count;                    \
    uint             src_scalar_NATURAL_class_chunk_offset;                    \
    uint             src_scalar_NATURAL_class_chunk_count;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_hidden_count;                          \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_total_output_class_count;              \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_modules_count;                   \
} RenderLogitsArgs_##SUFFIX;                                                   \
                                                                               \
/* --- Learn Phase A: Loss & Gradient Production --- */                        \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_logits;                                 \
    const STATE_T*   src_buffer_GLOBAL_CONST_temps;                            \
    const int*       src_buffer_GLOBAL_targets;                                \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_probs;                         \
    COMPUTE_T*       dest_buffer_GLOBAL_final_loss;                            \
    uint             src_scalar_NATURAL_flat_tile_index;                       \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_classes_per_chunk;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_total_output_class_count;              \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} CceChunkArgs_##SUFFIX;                                                       \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_logits;                                 \
    const STATE_T*   src_buffer_GLOBAL_CONST_temps;                            \
    const STORAGE_T* src_buffer_GLOBAL_targets;                                \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_probs;                         \
    COMPUTE_T*       dest_buffer_GLOBAL_partial_loss;                          \
    uint             src_scalar_NATURAL_flat_tile_index;                       \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_classes_per_chunk;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_total_output_class_count;              \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} BceChunkArgs_##SUFFIX;                                                       \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_hidden_activations;                     \
    const STORAGE_T* src_buffer_GLOBAL_partial_probs;                          \
    const void*      src_buffer_GLOBAL_targets;                                \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    const STATE_T*   src_buffer_GLOBAL_CONST_temps;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_grad_weights_module;           \
    STORAGE_T*       dest_buffer_GLOBAL_partial_grad_biases_module;            \
    uint             src_scalar_FLAG_problem_type;                             \
    uint             src_scalar_NATURAL_flat_tile_index;                       \
    uint             src_scalar_NATURAL_batch_chunk_offset;                    \
    uint             src_scalar_NATURAL_batch_chunk_count;                     \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_classes_per_chunk;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_hidden_count;                          \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_total_output_class_count;              \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} ModuleParamGradsArgs_##SUFFIX;                                               \
                                                                               \
/* --- Learn Phase B: Processing --- */                                        \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_partial_probs;                          \
    const void*      src_buffer_GLOBAL_targets;                                \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    const STATE_T*   src_buffer_GLOBAL_CONST_weights_module;                   \
    const STATE_T*   src_buffer_GLOBAL_CONST_temps;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_grad_hidden_activations_aos;   \
    uint             src_scalar_FLAG_problem_type;                             \
    uint             src_scalar_NATURAL_flat_tile_index;                       \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_classes_per_chunk;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_hidden_count;                          \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_total_output_class_count;              \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} BackpropToHiddenArgs_##SUFFIX;                                               \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_logits;                                 \
    const STORAGE_T* src_buffer_GLOBAL_partial_probs;                          \
    const void*      src_buffer_GLOBAL_targets;                                \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    const STATE_T*   src_buffer_GLOBAL_CONST_temps;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_grad_temps;                    \
    uint             src_scalar_FLAG_problem_type;                             \
    uint             src_scalar_NATURAL_flat_tile_index;                       \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_classes_per_chunk;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_total_output_class_count;              \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} TempGradientsArgs_##SUFFIX;                                                  \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_partial_grad_weights_module;            \
    const STORAGE_T* src_buffer_GLOBAL_partial_grad_biases_module;             \
    const STORAGE_T* src_buffer_GLOBAL_partial_grad_temps;                     \
    const STORAGE_T* src_buffer_GLOBAL_partial_grad_hidden_activations_aos;    \
    const COMPUTE_T* src_buffer_GLOBAL_CONST_clipping_threshold_per_item;      \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_partial_grad_weights_module;   \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_partial_grad_biases_module;    \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_partial_grad_temps;            \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos; \
    uint             src_scalar_FLAG_use_per_item_norm;                        \
    COMPUTE_T        src_scalar_REAL_clipping_threshold_t_pre;                 \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
    uint             src_scalar_NATURAL_flat_tile_index;                       \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_classes_per_chunk;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_padded_total_output_class_count;       \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} ClipPartialsArgs_##SUFFIX;                                                   \
                                                                               \
/* --- Learn Phase C: Reduction --- */                                         \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos; \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa; \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_hidden_count;                          \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_padded_total_modules_count;            \
    uint             src_scalar_NATURAL_num_module_chunks;                     \
    uint             src_scalar_NATURAL_modules_per_chunk;                     \
    uint             src_scalar_NATURAL_num_class_chunks;                      \
    uint             src_scalar_NATURAL_total_tile_count;                      \
} GatherPermuteArgs_##SUFFIX;                                                  \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_partial_collection;                     \
    const uint*      src_buffer_GLOBAL_CONST_offset_list_flat;                 \
    const uint*      src_buffer_GLOBAL_CONST_stage_list_offset;                \
    const uint*      src_buffer_GLOBAL_CONST_stage_fan_in;                     \
    const uint*      src_buffer_GLOBAL_CONST_stage_node_counts;                \
    COMPUTE_T*       update_buffer_GLOBAL_intermediate_partial_0;  /* ADR-026: COMPUTE_T for all stages */ \
    COMPUTE_T*       update_buffer_GLOBAL_intermediate_partial_1;              \
    COMPUTE_T*       dest_buffer_GLOBAL_summed_partial;                        \
    uint             src_scalar_NATURAL_partial_width;                         \
    uint             src_scalar_NATURAL_num_stages;                            \
    COMPUTE_T        src_scalar_REAL_t_algorithmic;                            \
    COMPUTE_T        src_scalar_REAL_lambda;                                   \
    COMPUTE_T        src_scalar_REAL_fp_max;                                   \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
} ReductionTreePlanStorageEntry_##SUFFIX;                                     \
                                                                               \
/* ADR-026: Compute-entry variant — reads from COMPUTE_T partial_collection */ \
typedef struct {                                                               \
    const COMPUTE_T* src_buffer_GLOBAL_partial_collection;                     \
    const uint*      src_buffer_GLOBAL_CONST_offset_list_flat;                 \
    const uint*      src_buffer_GLOBAL_CONST_stage_list_offset;                \
    const uint*      src_buffer_GLOBAL_CONST_stage_fan_in;                     \
    const uint*      src_buffer_GLOBAL_CONST_stage_node_counts;                \
    COMPUTE_T*       update_buffer_GLOBAL_intermediate_partial_0;              \
    COMPUTE_T*       update_buffer_GLOBAL_intermediate_partial_1;              \
    COMPUTE_T*       dest_buffer_GLOBAL_summed_partial;                        \
    uint             src_scalar_NATURAL_partial_width;                         \
    uint             src_scalar_NATURAL_num_stages;                            \
    COMPUTE_T        src_scalar_REAL_t_algorithmic;                            \
    COMPUTE_T        src_scalar_REAL_lambda;                                   \
    COMPUTE_T        src_scalar_REAL_fp_max;                                   \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
} ReductionTreePlanComputeEntry_##SUFFIX;                                      \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa; \
    COMPUTE_T*       dest_buffer_GLOBAL_summed_grad_hidden_activations;        \
    const COMPUTE_T* src_buffer_GLOBAL_CONST_clipping_threshold_per_stage;     \
    uint             src_scalar_NATURAL_num_reduction_stages;                  \
    COMPUTE_T        src_scalar_REAL_clipping_threshold_t_pre;                 \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_total_modules_count;                   \
    uint             src_scalar_NATURAL_padded_total_modules_count;            \
} StabilizeReduceArgs_##SUFFIX;                                                \
                                                                               \
typedef struct {                                                               \
    COMPUTE_T*       update_buffer_GLOBAL_intermediate_grad;                   \
    COMPUTE_T        src_scalar_REAL_clipping_threshold_t_j;                   \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
    uint             src_scalar_NATURAL_parameter_count;                       \
} ClipIntermediateArgs_##SUFFIX;                                               \
                                                                               \
/* --- Learn Phase D: Streaming Backprop --- */                                \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_input;                                  \
    const STORAGE_T* src_buffer_GLOBAL_hidden_activations;                     \
    const STORAGE_T* src_buffer_GLOBAL_hidden_mask;                            \
    uint             src_scalar_FLAG_use_explicit_hidden_mask;                 \
    const COMPUTE_T* src_buffer_GLOBAL_summed_grad_hidden_activations;         \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major; \
    uint             src_scalar_NATURAL_batch_chunk_offset;                    \
    uint             src_scalar_NATURAL_batch_chunk_count;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_input_count;                           \
    uint             src_scalar_NATURAL_padded_input_count;                    \
    uint             src_scalar_NATURAL_hidden_count;                          \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_final_grad_hidden_activations_total_count; \
} BackpropSharedWeightsArgs_##SUFFIX;                                          \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_hidden_activations;                     \
    const STORAGE_T* src_buffer_GLOBAL_hidden_mask;                            \
    uint             src_scalar_FLAG_use_explicit_hidden_mask;                 \
    const COMPUTE_T* src_buffer_GLOBAL_summed_grad_hidden_activations;         \
    const uint*      src_buffer_GLOBAL_sample_mask;                            \
    STORAGE_T*       dest_buffer_GLOBAL_partial_grad_biases_shared;            \
    uint             src_scalar_NATURAL_batch_chunk_offset;                    \
    uint             src_scalar_NATURAL_batch_chunk_count;                     \
    uint             src_scalar_NATURAL_total_batch_count;                     \
    uint             src_scalar_NATURAL_hidden_count;                          \
    uint             src_scalar_NATURAL_padded_hidden_count;                   \
    uint             src_scalar_NATURAL_final_grad_hidden_activations_total_count; \
} BackpropSharedBiasesArgs_##SUFFIX;                                           \
                                                                               \
typedef struct {                                                               \
    const STORAGE_T* src_buffer_GLOBAL_partial_grad_weights_shared_simd_major; \
    const STORAGE_T* src_buffer_GLOBAL_partial_grad_biases_shared;             \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major; \
    STORAGE_T*       dest_buffer_GLOBAL_clipped_partial_grad_biases_shared;    \
    COMPUTE_T        src_scalar_REAL_clipping_threshold_t_pre;                 \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
    uint             src_scalar_NATURAL_weights_parameter_count;               \
    uint             src_scalar_NATURAL_biases_parameter_count;                \
    uint             out_scalar_NATURAL_weights_write_offset;                 \
    uint             out_scalar_NATURAL_biases_write_offset;                  \
    uint             src_scalar_NATURAL_num_batch_chunks;                      \
} ClipSharedGradsArgs_##SUFFIX;                                                \
                                                                               \
/* --- Update Phase --- */                                                     \
typedef struct {                                                               \
    const COMPUTE_T* src_buffer_GLOBAL_summed_grad;                            \
    COMPUTE_T*       dest_buffer_GLOBAL_final_grad;                            \
    COMPUTE_T        src_scalar_REAL_effective_batch_size;                     \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
    uint             src_scalar_NATURAL_parameter_count;                       \
} NormalizeGradientsArgs_##SUFFIX;                                             \
                                                                               \
typedef struct {                                                               \
    const COMPUTE_T* src_buffer_GLOBAL_final_grad;                             \
    STATE_T*         update_buffer_GLOBAL_parameters;                          \
    STATE_T*         update_buffer_GLOBAL_m1;                                  \
    STATE_T*         update_buffer_GLOBAL_m2;                                  \
    COMPUTE_T        src_scalar_REAL_learning_rate;                            \
    COMPUTE_T        src_scalar_REAL_beta1_pow_t;                              \
    COMPUTE_T        src_scalar_REAL_beta2_pow_t;                              \
    COMPUTE_T        src_scalar_REAL_beta1;                                    \
    COMPUTE_T        src_scalar_REAL_beta2;                                    \
    COMPUTE_T        src_scalar_REAL_epsilon;                                  \
    uint             src_scalar_NATURAL_parameter_offset;                      \
    uint             src_scalar_NATURAL_parameter_count;                       \
    uint             src_scalar_NATURAL_total_parameter_count;                 \
} AdamUpdateArgs_##SUFFIX;                                                     \
                                                                               \
typedef struct {                                                               \
    STATE_T*         update_buffer_GLOBAL_temps;                               \
    COMPUTE_T        src_scalar_REAL_min_value;                                \
    COMPUTE_T        src_scalar_REAL_max_value;                                \
    uint             src_scalar_NATURAL_parameter_offset;                      \
    uint             src_scalar_NATURAL_parameter_count;                       \
    uint             src_scalar_NATURAL_total_parameter_count;                 \
} ClampTemperaturesArgs_##SUFFIX;

/* Variant Matrix Summary (ADR-024 §4.1, ADR-025 §5.3):
 *   Non-FP8:  s16* (9, requires HAS_FLOAT16)
 *             s32* (4)
 *             s64* (1)
 *   FP8 E4M3: s8e4* (9, FP16 compute/state variants require HAS_FLOAT16)
 *   FP8 E5M2: s8e5* (9, FP16 compute/state variants require HAS_FLOAT16)
 *   Total: 32 configurations
 */

/* Instantiate structs for all 14 non-FP8 three-axis precision combinations (ADR-024 §4.1).
 * FP8 variants (18 additional configurations, ADR-025 §5.3) follow below.
 * Format: s{storage}c{compute}x{state}                                         */

/* FP16 storage variants — requires _Float16 (9 configurations) */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
DECLARE_PRECISION_STRUCTS(s16c16x16, _Float16, _Float16, _Float16)
DECLARE_PRECISION_STRUCTS(s16c16x32, _Float16, _Float16, float)
DECLARE_PRECISION_STRUCTS(s16c16x64, _Float16, _Float16, double)
DECLARE_PRECISION_STRUCTS(s16c32x16, _Float16, float,  _Float16)
DECLARE_PRECISION_STRUCTS(s16c32x32, _Float16, float,  float)
DECLARE_PRECISION_STRUCTS(s16c32x64, _Float16, float,  double)
DECLARE_PRECISION_STRUCTS(s16c64x16, _Float16, double, _Float16)
DECLARE_PRECISION_STRUCTS(s16c64x32, _Float16, double, float)
DECLARE_PRECISION_STRUCTS(s16c64x64, _Float16, double, double)
#endif /* HAS_FLOAT16 — s16* structs */

/* FP32/FP64 storage variants (5 configurations) */
DECLARE_PRECISION_STRUCTS(s32c32x32, float,    float,  float)
DECLARE_PRECISION_STRUCTS(s32c32x64, float,    float,  double)
DECLARE_PRECISION_STRUCTS(s32c64x32, float,    double, float)
DECLARE_PRECISION_STRUCTS(s32c64x64, float,    double, double)
DECLARE_PRECISION_STRUCTS(s64c64x64, double,   double, double)

/* FP8 E4M3 storage variants with FP32/FP64 compute (ADR-025 §5.3) */
DECLARE_PRECISION_STRUCTS(s8e4c32x32, cpu_fp8_e4m3, float,  float)
DECLARE_PRECISION_STRUCTS(s8e4c32x64, cpu_fp8_e4m3, float,  double)
DECLARE_PRECISION_STRUCTS(s8e4c64x32, cpu_fp8_e4m3, double, float)
DECLARE_PRECISION_STRUCTS(s8e4c64x64, cpu_fp8_e4m3, double, double)

/* FP8 E5M2 storage variants with FP32/FP64 compute */
DECLARE_PRECISION_STRUCTS(s8e5c32x32, cpu_fp8_e5m2, float,  float)
DECLARE_PRECISION_STRUCTS(s8e5c32x64, cpu_fp8_e5m2, float,  double)
DECLARE_PRECISION_STRUCTS(s8e5c64x32, cpu_fp8_e5m2, double, float)
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
CPU_KERNELS_EXPORT void task_gather_permute_grad_hidden_activations_##SUFFIX(  \
    void* args, uint task_index, uint thread_id);                              \
CPU_KERNELS_EXPORT void task_stabilize_reduce_grad_hidden_activations_##SUFFIX(\
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
    ThreadPool* pool, ReductionTreePlanStorageEntry_##SUFFIX* plan);           \
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
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan_storage_entry_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_reduction_tree_plan_compute_entry_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_stabilize_reduce_args_##SUFFIX(void);\
CPU_KERNELS_EXPORT size_t get_struct_size_clip_intermediate_args_##SUFFIX(void);\
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_weights_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_backprop_shared_biases_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_clip_shared_grads_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_normalize_gradients_args_##SUFFIX(void); \
CPU_KERNELS_EXPORT size_t get_struct_size_adam_update_args_##SUFFIX(void);     \
CPU_KERNELS_EXPORT size_t get_struct_size_clamp_temperatures_args_##SUFFIX(void);

/* Declare functions for all 32 precision configurations (see Variant Matrix Summary above) */

/* FP16 storage variants — requires _Float16 (9 configurations) */
#if defined(HAS_FLOAT16) && HAS_FLOAT16
DECLARE_PRECISION_FUNCTIONS(s16c16x16)
DECLARE_PRECISION_FUNCTIONS(s16c16x32)
DECLARE_PRECISION_FUNCTIONS(s16c16x64)
DECLARE_PRECISION_FUNCTIONS(s16c32x16)
DECLARE_PRECISION_FUNCTIONS(s16c32x32)
DECLARE_PRECISION_FUNCTIONS(s16c32x64)
DECLARE_PRECISION_FUNCTIONS(s16c64x16)
DECLARE_PRECISION_FUNCTIONS(s16c64x32)
DECLARE_PRECISION_FUNCTIONS(s16c64x64)
#endif /* HAS_FLOAT16 — s16* functions */

/* FP32/FP64 storage variants (5 configurations) */
DECLARE_PRECISION_FUNCTIONS(s32c32x32)
DECLARE_PRECISION_FUNCTIONS(s32c32x64)
DECLARE_PRECISION_FUNCTIONS(s32c64x32)
DECLARE_PRECISION_FUNCTIONS(s32c64x64)
DECLARE_PRECISION_FUNCTIONS(s64c64x64)

/* FP8 E4M3/E5M2 variants with FP32/FP64 compute (ADR-025 §5.3) */
DECLARE_PRECISION_FUNCTIONS(s8e4c32x32)
DECLARE_PRECISION_FUNCTIONS(s8e4c32x64)
DECLARE_PRECISION_FUNCTIONS(s8e4c64x32)
DECLARE_PRECISION_FUNCTIONS(s8e4c64x64)
DECLARE_PRECISION_FUNCTIONS(s8e5c32x32)
DECLARE_PRECISION_FUNCTIONS(s8e5c32x64)
DECLARE_PRECISION_FUNCTIONS(s8e5c64x32)
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

/* Enum constant getters — eliminates Python-side constant drift (ADR-015 §4.2) */
CPU_KERNELS_EXPORT uint get_problem_type_cce(void);
CPU_KERNELS_EXPORT uint get_problem_type_bce(void);
CPU_KERNELS_EXPORT uint get_agg_mode_sum(void);
CPU_KERNELS_EXPORT uint get_agg_mode_average(void);
CPU_KERNELS_EXPORT uint get_sentinel_absent_partial(void);

#endif /* CPU_KERNELS_H */
