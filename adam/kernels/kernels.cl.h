// kernels.cl.h

#ifndef KERNELS_CL_H
#define KERNELS_CL_H

// Check for OpenCL environment
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
#ifndef MAX_EXITS_ENSEMBLE
#error "MAX_EXITS_ENSEMBLE must be defined in OpenCL mode"
#endif

#else
#include <float.h> // For FLT_MAX
#include <math.h>  // For fmax, sqrt, etc.
#include <stdio.h> // For printf in debug stubs

// Host/C++ mode definitions
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

// Default to float types if not defined
#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#define SCALAR_ZERO 0.0f
#endif

// Default SIMD width for host-side code analysis
#ifndef SIMD_WIDTH
#define SIMD_WIDTH 1
#endif

// Default Tile Size for host-side code analysis
#ifndef C_TILE_SIZE
#define C_TILE_SIZE 1
#endif

// Max Exits for an Ensemble for host-side code analysis
#ifndef MAX_EXITS_ENSEMBLE
#define MAX_EXITS_ENSEMBLE 1
#endif

#define CLK_LOCAL_MEM_FENCE 0x01
#define CLK_GLOBAL_MEM_FENCE 0x02

// Provides a C-compatible 'min' macro
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Provide dummy implementations for OpenCL built-in functions
inline int  get_global_id(int dim) { return 0; }
inline int  get_local_id(int dim) { return 0; }
inline int  get_group_id(int dim) { return 0; }
inline int  get_local_size(int dim) { return 1; }
inline int  get_global_size(int dim) { return 1; }
inline int  get_num_groups(int dim) { return 1; }
inline void barrier(int flags) { (void)flags; /* no-op */ }

/**
 * @brief Host-side stub for OpenCL's clamp function.
 */
inline SCALAR_TYPE clamp(SCALAR_TYPE val, SCALAR_TYPE min_val, SCALAR_TYPE max_val) { return fmin(fmax(val, min_val), max_val); }

/**
 * @brief Host-side stub for OpenCL's select function.
 * Returns 'b' if 'c' is non-zero, otherwise returns 'a'.
 * Note: OpenCL's select has specific behavior for vector types with boolean vector conditions,
 * but this scalar version is sufficient for the usage in your kernels.
 */
inline SCALAR_TYPE select(SCALAR_TYPE a, SCALAR_TYPE b, int c) { return (c) ? b : a; }

#endif // __OPENCL_VERSION__

typedef __global const int *TargetPtrCCE;
typedef int                 TargetTypeCCE;

typedef __global const SCALAR_TYPE *TargetPtrBCE;
typedef SCALAR_TYPE                 TargetTypeBCE;

#ifndef USE_FAST_MATH
#define USE_FAST_MATH 0
#endif

// Common math configuration
#if USE_FAST_MATH
#define MATH_FN native_
#else
#define MATH_FN
#endif

// Kernel function declarations

/**
 * @brief (Node 4) Performs the feed-forward pass for the neural network's shared layer.
 *
 * Host Assumptions:
 * - This kernel requires weights in a special, pre-transposed "SIMD-major" layout for performance.
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                                // [MEMORY size: 2 * local_size[0] * sizeof(SCALAR_TYPE)] Local memory for tiling.
    __global const SCALAR_TYPE *__restrict input_buf,              // [INPUT shape: (padded_batch_size, padded_input_dim)] The batch's input feature data.
    __global const SCALAR_TYPE *__restrict input_mask,             // [INPUT shape: (padded_batch_size)] Mask for valid samples in the input.
    __global const SCALAR_TYPE *__restrict weights_simd_major_buf, // [INPUT physical_shape: (ceil(HIDDEN_DIM/SIMD_WIDTH), IN_DIM, SIMD_WIDTH)] SIMD-major weights.
    __global const SCALAR_TYPE *__restrict biases_buf,             // [INPUT shape: (padded_hidden_dim)] Biases for the hidden layer.
    __global SCALAR_TYPE *__restrict hidden_buf,                   // [OUTPUT physical_shape: (padded_batch_size, hidden_dim/SIMD_WIDTH, SIMD_WIDTH)] Resulting hidden layer activations.
    __global SCALAR_TYPE *__restrict hidden_mask,                  // [OUTPUT shape: (padded_batch_size)] Calculated mask for valid hidden activations.
    int padded_input_dim,                                          // [INPUT scalar: (assumes padded value >= INPUT_DIM)] The padded dimension of the input layer.
    int padded_hidden_dim                                          // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the hidden layer.
);

/**
 * @brief (Node 5, CCE) Computes softmax probabilities and Categorical Cross-Entropy loss.
 *
 * Host Assumptions:
 * - `hidden_buf` has a physical layout of (batch, hidden_dim/SIMD_WIDTH, SIMD_WIDTH) and must be read accordingly.
 */
__kernel void compute_all_exits_cce(
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [INPUT physical_shape: (padded_batch_size, hidden_dim/SIMD_WIDTH, SIMD_WIDTH)] Activations from the shared layer.
    __global const SCALAR_TYPE *__restrict hidden_mask,      // [INPUT shape: (padded_batch_size)] Mask for valid hidden activations.
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [INPUT shape: (NUM_EXITS, hidden_dim, output_classes)] Weights for all exits.
    __global const SCALAR_TYPE *__restrict exit_biases_buf,  // [INPUT shape: (NUM_EXITS, output_classes)] Biases for all exits.
    __global const SCALAR_TYPE *__restrict temps_buf,        // [INPUT shape: (NUM_EXITS)] Temperature values for each exit.
    TargetPtrCCE __restrict targets_buf,                     // [INPUT shape: (padded_batch_size)] [CCE-specific] Ground truth labels as integer class indices.
    __global const SCALAR_TYPE *__restrict targets_mask,     // [INPUT shape: (padded_batch_size)] Mask for valid labels.
    __global SCALAR_TYPE *__restrict unscaled_logits_buf,    // [OUTPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits before temperature scaling.
    __global SCALAR_TYPE *__restrict exit_probs_buf,         // [OUTPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities after softmax and temp scaling.
    __global SCALAR_TYPE *__restrict per_exit_losses_buf,    // [OUTPUT shape: (NUM_EXITS, padded_batch_size)] Per-sample, per-exit categorical cross-entropy loss.
    int padded_batch_size,                                   // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int hidden_dim,                                          // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the hidden layer.
    int output_classes,                                      // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                            // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 5, BCE) Computes sigmoid probabilities and summed Binary Cross-Entropy loss.
 *
 * Host Assumptions:
 * - `hidden_buf` has a physical layout of (batch, hidden_dim/SIMD_WIDTH, SIMD_WIDTH) and must be read accordingly.
 */
__kernel void compute_all_exits_bce(
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [INPUT physical_shape: (padded_batch_size, hidden_dim/SIMD_WIDTH, SIMD_WIDTH)] Activations from the shared layer.
    __global const SCALAR_TYPE *__restrict hidden_mask,      // [INPUT shape: (padded_batch_size)] Mask for valid hidden activations.
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [INPUT shape: (NUM_EXITS, hidden_dim, output_classes)] Weights for all exits.
    __global const SCALAR_TYPE *__restrict exit_biases_buf,  // [INPUT shape: (NUM_EXITS, output_classes)] Biases for all exits.
    __global const SCALAR_TYPE *__restrict temps_buf,        // [INPUT shape: (NUM_EXITS)] Temperature values for each exit.
    TargetPtrBCE __restrict targets_buf,                     // [INPUT shape: (padded_batch_size, output_classes)] [BCE-specific] Ground truth labels as one-hot encoded floats/halfs.
    __global const SCALAR_TYPE *__restrict targets_mask,     // [INPUT shape: (padded_batch_size)] Mask for valid labels.
    __global SCALAR_TYPE *__restrict unscaled_logits_buf,    // [OUTPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits before temperature scaling.
    __global SCALAR_TYPE *__restrict exit_probs_buf,         // [OUTPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities after sigmoid and temp scaling.
    __global SCALAR_TYPE *__restrict per_exit_losses_buf,    // [OUTPUT shape: (NUM_EXITS, padded_batch_size)] Per-sample, per-exit summed binary cross-entropy loss.
    int padded_batch_size,                                   // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int hidden_dim,                                          // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the hidden layer.
    int output_classes,                                      // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                            // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 6, Tier 1) Computes ensemble weights via a fast, in-register serial reduction.
 */
__kernel void ensemble_weights_reg_reduce(
    __global const SCALAR_TYPE *__restrict exit_probs,   // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from all exits.
    __global const SCALAR_TYPE *__restrict targets_mask, // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global SCALAR_TYPE *__restrict ensemble_weights,   // [OUTPUT shape: (padded_batch_size, num_exits)] Calculated ensemble weights.
    int padded_batch_size,                               // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes,                                  // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                        // [INPUT scalar: (assumes value <= MAX_EXITS_ENSEMBLE)] The total number of network exits.
);

/**
 * @brief (Node 6, Tier 2) Computes ensemble weights for one sample using a single work-group.
 */
__kernel void ensemble_weights_local_reduce(
    __local SCALAR_TYPE *l_reduction_mem,              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for reduction.
    __global const SCALAR_TYPE *__restrict exit_probs, // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from all exits.
    __global SCALAR_TYPE *__restrict ensemble_weights, // [OUTPUT shape: (padded_batch_size, num_exits)] Calculated ensemble weights.
    int num_exits,                                     // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
    int output_classes,                                // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int padded_batch_size                              // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
);
/**
 * @brief (Node 6, Hierarchical Tier, Stage 1) Processes chunks of exits to find the max logit and a stable sum-of-exponentials.
 */
__kernel void compute_exit_chunk(
    __global const SCALAR_TYPE *__restrict unscaled_logits, // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits from all exits.
    __global SCALAR_TYPE *__restrict chunk_max_out,         // [OUTPUT shape: (num_chunks, padded_batch_size)] Per-chunk maximum logit value.
    __global SCALAR_TYPE *__restrict chunk_sum_out,         // [OUTPUT shape: (num_chunks, padded_batch_size)] Per-chunk sum of exponentials relative to its max.
    int chunk_id,                                           // [INPUT scalar: (assumes value >= 0)] The ID of the chunk to process.
    int chunk_size,                                         // [INPUT scalar: (assumes value > 0)] The number of exits per chunk.
    int total_exits,                                        // [INPUT scalar: (assumes value == NUM_EXITS)] Total number of network exits.
    int padded_batch_size,                                  // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes                                      // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
);

/**
 * @brief (Node 6, Hierarchical Tier, Stage 2) Reduces a group of intermediate chunk results into a single result.
 */
__kernel void reduce_chunk_pair(
    __global const SCALAR_TYPE *__restrict input_max, // [INPUT shape: (num_input_chunks, padded_batch_size)] Max values from the previous reduction level.
    __global const SCALAR_TYPE *__restrict input_sum, // [INPUT shape: (num_input_chunks, padded_batch_size)] Sum values from the previous reduction level.
    __global SCALAR_TYPE *__restrict output_max,      // [OUTPUT shape: (num_output_chunks, padded_batch_size)] Destination for the new reduced max values.
    __global SCALAR_TYPE *__restrict output_sum,      // [OUTPUT shape: (num_output_chunks, padded_batch_size)] Destination for the new reduced sum values.
    int start_chunk_idx,                              // [INPUT scalar: (assumes value >= 0)] The starting index in the input buffers for this reduction group.
    int num_chunks_to_reduce,                         // [INPUT scalar: (assumes value > 0)] The number of input chunks to combine (the reduction factor).
    int total_input_chunks,                           // [INPUT scalar: (assumes value > 0)] The total number of chunks at this input level.
    int padded_batch_size                             // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
);

/**
 * @brief (Node 6, Hierarchical Tier, Stage 3) Computes the final normalized ensemble weights.
 */
__kernel void normalize_weights(
    __global const SCALAR_TYPE *__restrict unscaled_logits, // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits from all exits.
    __global const SCALAR_TYPE *__restrict global_max,      // [INPUT shape: (padded_batch_size)] Final maximum logit value across all exits.
    __global const SCALAR_TYPE *__restrict global_sum,      // [INPUT shape: (padded_batch_size)] Final sum-of-exponentials relative to the global max.
    __global SCALAR_TYPE *__restrict ensemble_weights,      // [OUTPUT shape: (padded_batch_size, num_exits)] The final calculated ensemble weights.
    int total_exits,                                        // [INPUT scalar: (assumes value == NUM_EXITS)] Total number of network exits.
    int padded_batch_size,                                  // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes                                      // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
);

/**
 * @brief (Node 7) Blends exit probabilities using pre-calculated weights to get the final distribution.
 */
__kernel void blend_ensemble_probabilities(
    __global const SCALAR_TYPE *__restrict exit_probs,       // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from all exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights, // [INPUT shape: (padded_batch_size, num_exits)] Calculated ensemble weights.
    __global const SCALAR_TYPE *__restrict targets_mask,     // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global SCALAR_TYPE *__restrict ensemble_probs,         // [OUTPUT shape: (padded_batch_size, output_classes)] Final blended probabilities.
    int padded_batch_size,                                   // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes,                                      // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                            // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 8) Computes partial cross-entropy losses for the batch.
 */
__kernel void calculate_partial_losses(
    __local float *l_loss_sums,                                // [MEMORY size: local_size[0] * sizeof(float)] Local memory for reduction.
    __global const SCALAR_TYPE *__restrict ensemble_probs_buf, // [INPUT shape: (padded_batch_size, output_classes)] Final blended probabilities.
    __global const SCALAR_TYPE *__restrict targets_mask,       // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global const int *__restrict targets_buf,                // [INPUT shape: (padded_batch_size)] Ground truth labels.
    __global float *__restrict partial_loss_buf,               // [OUTPUT shape: (num_work_groups)] Partial loss sums per work-group.
    int padded_batch_size,                                     // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes                                         // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
);

/**
 * @brief (Node 9) Aggregates partial loss sums into a final total loss.
 */
__kernel void aggregate_partial_losses(
    __local float *l_reduction_mem,                    // [MEMORY size: local_size[0] * sizeof(float)] Local memory for reduction.
    __global const float *__restrict partial_loss_buf, // [INPUT shape: (num_partial_sums)] Partial loss sums from previous kernel.
    __global float *__restrict final_loss_buf,         // [OUTPUT shape: (1)] Final total batch loss.
    int num_partial_sums                               // [INPUT scalar: (assumes value > 0)] The number of partial sums to aggregate.
);

/**
 * @brief (Node 10, CCE) Computes gradients for exit parameters based on CCE loss.
 *
 * Host Assumptions:
 * - `grad_exit_weights` and `grad_exit_biases` must be zeroed before calling.
 * - `hidden_buf` has a physical layout of (batch, hidden_dim/SIMD_WIDTH, SIMD_WIDTH) and must be read accordingly.
 */
__kernel void calculate_exit_gradients_cce(
    __local SCALAR_TYPE *local_grad_w,                              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for weight gradient reduction.
    __local SCALAR_TYPE *local_grad_b,                              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for bias gradient reduction.
    __global const SCALAR_TYPE *__restrict hidden_buf,              // [INPUT physical_shape: (padded_batch_size, hidden_dim/SIMD_WIDTH, SIMD_WIDTH)] Activations from the shared layer.
    __global const SCALAR_TYPE *__restrict exit_probs_buf,          // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from all exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf,    // [INPUT shape: (padded_batch_size, num_exits)] Calculated ensemble weights.
    TargetPtrCCE __restrict targets_buf,                            // [INPUT shape: (padded_batch_size)] [CCE-specific] Ground truth labels as integer class indices.
    __global const SCALAR_TYPE *__restrict targets_mask,            // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global const SCALAR_TYPE *__restrict exit_weights_buf,        // [INPUT shape: (NUM_EXITS, hidden_dim, output_classes)] Original weights for all exits.
    __global SCALAR_TYPE *__restrict grad_exit_weights,             // [OUTPUT shape: (NUM_EXITS, hidden_dim, output_classes)] Gradients for exit weights.
    __global SCALAR_TYPE *__restrict grad_exit_biases,              // [OUTPUT shape: (NUM_EXITS, output_classes)] Gradients for exit biases.
    __global SCALAR_TYPE *__restrict grad_hidden_contributions_buf, // [OUTPUT shape: (NUM_EXITS, padded_batch_size, hidden_dim)] Upstream gradient signals for the shared layer.
    int padded_batch_size,                                          // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int hidden_dim,                                                 // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the hidden layer.
    int output_classes,                                             // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                                   // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 10, BCE) Computes gradients for exit parameters based on BCE loss.
 *
 * Host Assumptions:
 * - `grad_exit_weights` and `grad_exit_biases` must be zeroed before calling.
 * - `hidden_buf` has a physical layout of (batch, hidden_dim/SIMD_WIDTH, SIMD_WIDTH) and must be read accordingly.
 */
__kernel void calculate_exit_gradients_bce(
    __local SCALAR_TYPE *local_grad_w,                              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for weight gradient reduction.
    __local SCALAR_TYPE *local_grad_b,                              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for bias gradient reduction.
    __global const SCALAR_TYPE *__restrict hidden_buf,              // [INPUT physical_shape: (padded_batch_size, hidden_dim/SIMD_WIDTH, SIMD_WIDTH)] Activations from the shared layer.
    __global const SCALAR_TYPE *__restrict exit_probs_buf,          // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from all exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf,    // [INPUT shape: (padded_batch_size, num_exits)] Calculated ensemble weights.
    TargetPtrBCE __restrict targets_buf,                            // [INPUT shape: (padded_batch_size, output_classes)] [BCE-specific] Ground truth labels as one-hot encoded floats/halfs.
    __global const SCALAR_TYPE *__restrict targets_mask,            // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global const SCALAR_TYPE *__restrict exit_weights_buf,        // [INPUT shape: (NUM_EXITS, hidden_dim, output_classes)] Original weights for all exits.
    __global SCALAR_TYPE *__restrict grad_exit_weights,             // [OUTPUT shape: (NUM_EXITS, hidden_dim, output_classes)] Gradients for exit weights.
    __global SCALAR_TYPE *__restrict grad_exit_biases,              // [OUTPUT shape: (NUM_EXITS, output_classes)] Gradients for exit biases.
    __global SCALAR_TYPE *__restrict grad_hidden_contributions_buf, // [OUTPUT shape: (NUM_EXITS, padded_batch_size, hidden_dim)] Upstream gradient signals for the shared layer.
    int padded_batch_size,                                          // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int hidden_dim,                                                 // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the hidden layer.
    int output_classes,                                             // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                                   // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 11) Aggregates upstream gradients and backpropagates through the activation function.
 *
 * Host Assumptions:
 * - `hidden_buf` has a physical layout of (batch, hidden_dim/SIMD_WIDTH, SIMD_WIDTH) and must be read accordingly.
 */
__kernel void aggregate_and_backprop_activation(
    __global const SCALAR_TYPE *__restrict hidden_buf,                    // [INPUT physical_shape: (padded_batch_size, hidden_dim/SIMD_WIDTH, SIMD_WIDTH)] Layer activations, for ReLU derivative.
    __global const SCALAR_TYPE *__restrict hidden_mask,                   // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global const SCALAR_TYPE *__restrict grad_hidden_contributions_buf, // [INPUT shape: (NUM_EXITS, padded_batch_size, hidden_dim)] Upstream gradient signals from all exits.
    __global SCALAR_TYPE *__restrict grad_pre_activation_buf,             // [OUTPUT shape: (padded_batch_size, hidden_dim)] Gradient signal before the activation function.
    int padded_batch_size,                                                // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int hidden_dim,                                                       // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the hidden layer.
    int num_exits                                                         // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 12) Calculates gradients for a dense layer's parameters (weights and biases).
 *
 * Host Assumptions:
 * - `grad_weights` and `grad_biases` must be zeroed before calling.
 */
__kernel void calculate_dense_layer_gradients(
    __local SCALAR_TYPE *local_grad_w,                              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for weight gradient reduction.
    __local SCALAR_TYPE *local_grad_b,                              // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for bias gradient reduction.
    __global const SCALAR_TYPE *__restrict input_buf,               // [INPUT shape: (padded_batch_size, input_dim)] The original input to the layer.
    __global const SCALAR_TYPE *__restrict input_mask,              // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf, // [INPUT shape: (padded_batch_size, hidden_dim)] Gradient signal from Node 11.
    __global SCALAR_TYPE *__restrict grad_weights,                  // [OUTPUT shape: (input_dim, hidden_dim)] Final gradient for the weight matrix.
    __global SCALAR_TYPE *__restrict grad_biases,                   // [OUTPUT shape: (hidden_dim)] Final gradient for the bias vector.
    int padded_batch_size,                                          // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int input_dim,                                                  // [INPUT scalar: (assumes padded value >= INPUT_DIM)] The padded dimension of the layer's input.
    int hidden_dim                                                  // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the layer's output.
);

/**
 * @brief (Node 13) Propagates the gradient back to the input of a layer (W.T * grad).
 *
 * Host Assumptions:
 * - This kernel requires weights in a standard row-major layout for efficient matrix multiplication.
 */
__kernel void backprop_input_gradient(
    __global const SCALAR_TYPE *__restrict grad_pre_activation_buf, // [INPUT shape: (padded_batch_size, hidden_dim)] Upstream gradient from Node 11.
    __global const SCALAR_TYPE *__restrict weights_standard_buf,    // [INPUT shape: (input_dim, hidden_dim)] Original layer weights in standard layout.
    __global SCALAR_TYPE *__restrict grad_input_buf,                // [OUTPUT shape: (padded_batch_size, input_dim)] Downstream gradient for the previous layer.
    int padded_batch_size,                                          // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int input_dim,                                                  // [INPUT scalar: (assumes padded value >= INPUT_DIM)] The padded dimension of the layer's input.
    int hidden_dim                                                  // [INPUT scalar: (assumes padded value >= HIDDEN_DIM)] The padded dimension of the layer's output.
);

/**
 * @brief (Node 14, CCE) Computes temperature gradients based on CCE loss.
 *
 * Host Assumptions:
 * - `grad_temps` must be zeroed before calling.
 */
__kernel void calculate_temp_gradients_cce(
    __local SCALAR_TYPE *local_grad_sum,                         // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for gradient summation.
    __global const SCALAR_TYPE *__restrict unscaled_logits_buf,  // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits from exits.
    __global const SCALAR_TYPE *__restrict exit_probs_buf,       // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf, // [INPUT shape: (padded_batch_size, num_exits)] Ensemble weights.
    __global const SCALAR_TYPE *__restrict temps_buf,            // [INPUT shape: (NUM_EXITS)] Current temperature values.
    TargetPtrCCE __restrict targets_buf,                         // [INPUT shape: (padded_batch_size)] [CCE-specific] Ground truth labels as integer class indices.
    __global const SCALAR_TYPE *__restrict targets_mask,         // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global SCALAR_TYPE *__restrict grad_temps,                 // [OUTPUT shape: (NUM_EXITS)] Final gradient for temperatures.
    int padded_batch_size,                                       // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes,                                          // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                                // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 14, BCE) Computes temperature gradients based on BCE loss.
 *
 * Host Assumptions:
 * - `grad_temps` must be zeroed before calling.
 */
__kernel void calculate_temp_gradients_bce(
    __local SCALAR_TYPE *local_grad_sum,                         // [MEMORY size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for gradient summation.
    __global const SCALAR_TYPE *__restrict unscaled_logits_buf,  // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits from exits.
    __global const SCALAR_TYPE *__restrict exit_probs_buf,       // [INPUT shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf, // [INPUT shape: (padded_batch_size, num_exits)] Ensemble weights.
    __global const SCALAR_TYPE *__restrict temps_buf,            // [INPUT shape: (NUM_EXITS)] Current temperature values.
    TargetPtrBCE __restrict targets_buf,                         // [INPUT shape: (padded_batch_size, output_classes)] [BCE-specific] Ground truth labels as one-hot encoded floats/halfs.
    __global const SCALAR_TYPE *__restrict targets_mask,         // [INPUT shape: (padded_batch_size)] Mask for valid samples.
    __global SCALAR_TYPE *__restrict grad_temps,                 // [OUTPUT shape: (NUM_EXITS)] Final gradient for temperatures.
    int padded_batch_size,                                       // [INPUT scalar: (assumes padded value >= BATCH_SIZE)] The padded size of the batch dimension.
    int output_classes,                                          // [INPUT scalar: (assumes padded value >= OUTPUT_CLASSES)] The padded dimension of the output classes.
    int num_exits                                                // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

/**
 * @brief (Node 15) Performs Adam optimizer update for a parameter buffer.
 */
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad, // [INPUT shape: (total_params)] Gradient for the parameters.
    SCALAR_TYPE beta1,                           // [INPUT scalar: (assumes 0 < value < 1)] Adam hyperparameter beta1.
    SCALAR_TYPE beta2,                           // [INPUT scalar: (assumes 0 < value < 1)] Adam hyperparameter beta2.
    SCALAR_TYPE beta1_t,                         // [INPUT scalar: (assumes 0 < value < 1)] Time-adjusted beta1 (beta1^t).
    SCALAR_TYPE beta2_t,                         // [INPUT scalar: (assumes 0 < value < 1)] Time-adjusted beta2 (beta2^t).
    SCALAR_TYPE learning_rate,                   // [INPUT scalar: (assumes value > 0)] Global learning rate.
    SCALAR_TYPE epsilon,                         // [INPUT scalar: (assumes value > 0)] Adam epsilon for numerical stability.
    __global SCALAR_TYPE *__restrict param,      // [IN/OUT shape: (total_params)] Parameters to be updated.
    __global SCALAR_TYPE *__restrict m1,         // [IN/OUT shape: (total_params)] First moment vector (momentum).
    __global SCALAR_TYPE *__restrict m2,         // [IN/OUT shape: (total_params)] Second moment vector (RMSprop).
    int total_params                             // [INPUT scalar: (assumes value > 0)] Total number of elements in the buffers.
);

/**
 * @brief (Node 16) Clamps temperature values within a specified range.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [IN/OUT shape: (NUM_EXITS)] Temperature values to be clamped.
    SCALAR_TYPE min_temp,                       // [INPUT scalar: (assumes value > 0)] Minimum allowed temperature value.
    SCALAR_TYPE max_temp,                       // [INPUT scalar: (assumes value > min_temp)] Maximum allowed temperature value.
    int         num_exits                       // [INPUT scalar: (assumes value == NUM_EXITS)] The total number of network exits.
);

#endif // KERNELS_CL_H
