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
 * This kernel computes the hidden layer activations using the input data, weights, and biases.
 * It uses a 3D NDRange for parallelization across the batch and hidden dimensions.
 *
 * Host Assumptions:
 * - The NDRange is set with global sizes (padded_batch_size, ceil(HIDDEN_DIM / SIMD_WIDTH), 1).
 * - Weights are pre-transposed to a "SIMD-major" layout for efficient memory access.
 */
__kernel void forward_pass(
    __local SCALAR_TYPE *local_mem,                     // [size: 2 * local_size[0] * sizeof(SCALAR_TYPE)] Local memory for tiling input and weights.
    __global const SCALAR_TYPE *__restrict input_buf,   // [shape: (padded_batch_size, padded_input_dim)] Input data buffer.
    __global const SCALAR_TYPE *__restrict input_mask,  // [shape: (padded_batch_size)] Mask for input data, indicates valid samples.
    __global const SCALAR_TYPE *__restrict weights_buf, // [shape: (ceil(HIDDEN_DIM / SIMD_WIDTH), INPUT_DIM, SIMD_WIDTH)] Pre-transposed weights buffer.
    __global const SCALAR_TYPE *__restrict biases_buf,  // [shape: (padded_hidden_dim)] Biases buffer.
    __global SCALAR_TYPE *__restrict hidden_buf,        // [shape: (padded_batch_size, ceil(HIDDEN_DIM / SIMD_WIDTH), SIMD_WIDTH)] Hidden layer output buffer.
    __global SCALAR_TYPE *__restrict hidden_mask,       // [shape: (padded_batch_size)] Mask for hidden layer, to be computed.
    int padded_input_dim,                               // Padded input dimension.
    int padded_hidden_dim                               // Padded hidden dimension.
);

/**
 * @brief (Node 5) Computes outputs for each network exit from the shared hidden layer.
 *
 * This kernel calculates the logits, probabilities, and per-exit losses for each exit using the hidden layer activations.
 * It uses a 2D NDRange for parallelization across exits and batch samples.
 *
 * Host Assumptions:
 * - The NDRange is set with global sizes (NUM_EXITS, PADDED_BATCH_SIZE).
 * - Local work size should be optimized by the host (1D or 2D).
 */
__kernel void compute_all_exits(
    __global const SCALAR_TYPE *__restrict hidden_buf,       // [shape: (padded_batch_size, hidden_dim)] Hidden layer output.
    __global const SCALAR_TYPE *__restrict hidden_mask,      // [shape: (padded_batch_size)] Mask for hidden layer.
    __global const SCALAR_TYPE *__restrict exit_weights_buf, // [shape: (NUM_EXITS, hidden_dim, output_classes)] Weights for exits.
    __global const SCALAR_TYPE *__restrict exit_biases_buf,  // [shape: (NUM_EXITS, output_classes)] Biases for exits.
    __global const SCALAR_TYPE *__restrict temps_buf,        // [shape: (NUM_EXITS)] Temperature values for each exit.
    __global const int *__restrict targets_buf,              // [shape: (padded_batch_size)] Target labels.
    __global const SCALAR_TYPE *__restrict targets_mask,     // [shape: (padded_batch_size)] Mask for targets.
    __global SCALAR_TYPE *__restrict unscaled_logits_buf,    // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Buffer for raw logits.
    __global SCALAR_TYPE *__restrict exit_probs_buf,         // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Buffer for probabilities after softmax.
    __global SCALAR_TYPE *__restrict per_exit_losses_buf,    // [shape: (NUM_EXITS, padded_batch_size)] Buffer for per-exit losses.
    int padded_batch_size,                                   // Padded batch size.
    int hidden_dim,                                          // Hidden dimension.
    int output_classes,                                      // Number of output classes.
    int num_exits                                            // Number of exits.
);

/**
 * @brief (Node 6, Tier 1) Computes ensemble weights via a fast, in-register serial reduction.
 *
 * This kernel is optimized for small numbers of exits (N <= 64) and performs the entire calculation
 * serially in private registers for each sample, with no synchronization overhead.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange where global_size = padded_batch_size.
 * - The number of exits (num_exits) must be <= MAX_EXITS_ENSEMBLE.
 */
__kernel void ensemble_weights_reg_reduce(
    __global const SCALAR_TYPE *__restrict exit_probs,   // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global const SCALAR_TYPE *__restrict targets_mask, // [shape: (padded_batch_size)] Mask for targets.
    __global SCALAR_TYPE *__restrict ensemble_weights,   // [shape: (padded_batch_size, num_exits)] Output ensemble weights.
    int padded_batch_size,                               // Padded batch size.
    int output_classes,                                  // Number of output classes.
    int num_exits                                        // Number of exits.
);

/**
 * @brief (Node 6, Tier 2 - CORRECTED) Computes ensemble weights for one sample using a single work-group.
 *
 * This kernel handles medium numbers of exits (64 < N <= 512) by assigning one full work-group to each sample.
 * It performs a parallel reduction within local memory to calculate the denominator before computing the weights.
 *
 * Host Assumptions:
 * - Launched such that each work-group handles one sample.
 * - Global size = (padded_batch_size * local_size), with local_size optimized by the host (e.g., 256).
 */
__kernel void ensemble_weights_local_reduce(
    __local SCALAR_TYPE *l_reduction_mem,              // [size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for reduction.
    __global const SCALAR_TYPE *__restrict exit_probs, // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global SCALAR_TYPE *__restrict ensemble_weights, // [shape: (padded_batch_size, num_exits)] Output ensemble weights.
    int num_exits,                                     // Number of exits.
    int output_classes,                                // Number of output classes.
    int padded_batch_size                              // The padded size of the batch dimension for correct indexing.
);

/**
 * @brief (Node 6, Tier 3, Map Stage) Calculates exp(confidence) for every exit.
 *
 * This kernel is part of the scalable map-reduce chain for large numbers of exits (N > 512).
 * Each work-item computes exp(confidence) for a single exit and sample.
 *
 * Host Assumptions:
 * - Launched with a 2D NDRange where global_size = (padded_batch_size, num_exits).
 */
__kernel void ensemble_weights_map_exp_conf(
    __global const SCALAR_TYPE *__restrict exit_probs,   // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global const SCALAR_TYPE *__restrict targets_mask, // [shape: (padded_batch_size)] Mask for targets.
    __global SCALAR_TYPE *__restrict temp_exp_conf_buf,  // [shape: (padded_batch_size, num_exits)] Temporary buffer for exp(confidence) values.
    int padded_batch_size,                               // Padded batch size.
    int num_exits,                                       // Number of exits.
    int output_classes                                   // Number of output classes.
);

/**
 * @brief (Node 6, Tier 3, Reduce Stage) Reduces partial exponentiated confidence values.
 *
 * This kernel reduces the large temporary buffer of exp(confidence) values into a smaller buffer of partial sums.
 * Each work-group handles one chunk of the reduction for a sample.
 *
 * Host Assumptions:
 * - Launched with a 2D grid, e.g., global_size = (padded_batch_size, num_chunks * workgroup_size), local_size = (1, workgroup_size).
 */
__kernel void reduce_partial_sums(
    __local SCALAR_TYPE *l_reduction_mem,                     // [size: local_size[1] * sizeof(SCALAR_TYPE)] Local memory for reduction.
    __global const SCALAR_TYPE *__restrict temp_exp_conf_buf, // [shape: (padded_batch_size, num_exits)] Input exp(confidence) values.
    __global SCALAR_TYPE *__restrict temp_partial_sums_buf,   // [shape: (padded_batch_size, num_chunks)] Output partial sums.
    int num_exits                                             // Number of exits.
);

/**
 * @brief (Node 6, Tier 3, Finalize Stage) Aggregates partial sums and computes final weights.
 *
 * This kernel performs a small, serial reduction on the partial sums to calculate the final denominator,
 * then computes and writes the final ensemble weights for each sample.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange where global_size = padded_batch_size.
 */
__kernel void ensemble_weights_finalize(
    __global const SCALAR_TYPE *__restrict temp_exp_conf_buf,     // [shape: (padded_batch_size, num_exits)] Input exp(confidence) values.
    __global const SCALAR_TYPE *__restrict temp_partial_sums_buf, // [shape: (padded_batch_size, num_chunks)] Input partial sums.
    __global const SCALAR_TYPE *__restrict targets_mask,          // [shape: (padded_batch_size)] Mask for targets.
    __global SCALAR_TYPE *__restrict ensemble_weights,            // [shape: (padded_batch_size, num_exits)] Final output ensemble weights.
    int padded_batch_size,                                        // Padded batch size.
    int num_exits,                                                // Number of exits.
    int num_chunks                                                // Number of chunks from the reduce stage.
);

/**
 * @brief (Node 7) Blends exit probabilities using pre-calculated weights to get the final distribution.
 *
 * This kernel computes the weighted average of all exit probabilities for each sample to produce the final blended probability distribution.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange where global_size = padded_batch_size.
 */
__kernel void blend_ensemble_probabilities(
    __global const SCALAR_TYPE *__restrict exit_probs,       // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from all exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights, // [shape: (padded_batch_size, num_exits)] Pre-calculated ensemble weights.
    __global const SCALAR_TYPE *__restrict targets_mask,     // [shape: (padded_batch_size)] Mask for targets.
    __global SCALAR_TYPE *__restrict ensemble_probs,         // [shape: (padded_batch_size, output_classes)] Final blended probabilities.
    int padded_batch_size,                                   // Padded batch size.
    int output_classes,                                      // Number of output classes.
    int num_exits                                            // Number of exits.
);

/**
 * @brief (Node 8) Computes partial cross-entropy losses for the batch.
 *
 * This kernel calculates partial sums of the cross-entropy loss for subsets of the batch, to be aggregated later.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange covering the batch, with local size <= 256.
 */
__kernel void calculate_partial_losses(
    __global const SCALAR_TYPE *__restrict ensemble_probs_buf, // [shape: (padded_batch_size, output_classes)] Ensemble probabilities.
    __global const SCALAR_TYPE *__restrict targets_mask,       // [shape: (padded_batch_size)] Mask for targets.
    __global const int *__restrict targets_buf,                // [shape: (padded_batch_size)] Target labels.
    __global float *__restrict partial_loss_buf,               // [shape: (num_work_groups)] Partial loss sums per work-group.
    int padded_batch_size,                                     // Padded batch size.
    int output_classes                                         // Number of output classes.
);

/**
 * @brief (Node 9) Aggregates partial loss sums into a final total loss.
 *
 * This kernel sums up the partial loss values computed by `calculate_partial_losses` to obtain the total loss.
 *
 * Host Assumptions:
 * - Launched with a single work-group.
 */
__kernel void aggregate_partial_losses(
    __global const float *__restrict partial_loss_buf, // [shape: (num_partial_sums)] Partial loss sums.
    __global float *__restrict final_loss_buf,         // [shape: (1)] Final total loss.
    int num_partial_sums                               // Number of partial sums.
);

/**
 * @brief (Node 10) Computes gradients for exit-specific parameters and contributions to hidden layer gradients.
 *
 * This kernel calculates gradients for the exit weights and biases, as well as the contributions to the hidden layer gradients.
 *
 * Host Assumptions:
 * - Launched with a 2D NDRange where global sizes = (NUM_EXITS, HIDDEN_DIM).
 * - grad_exit_weights and grad_exit_biases must be zeroed before calling.
 */
__kernel void calculate_exit_gradients(
    __local SCALAR_TYPE *local_grad_w,                              // [size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for weight gradients.
    __local SCALAR_TYPE *local_grad_b,                              // [size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for bias gradients.
    __global const SCALAR_TYPE *__restrict hidden_buf,              // [shape: (padded_batch_size, hidden_dim)] Hidden layer output.
    __global const SCALAR_TYPE *__restrict exit_probs_buf,          // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf,    // [shape: (padded_batch_size, num_exits)] Ensemble weights.
    __global const int *__restrict targets_buf,                     // [shape: (padded_batch_size)] Target labels.
    __global const SCALAR_TYPE *__restrict targets_mask,            // [shape: (padded_batch_size)] Mask for targets.
    __global const SCALAR_TYPE *__restrict exit_weights_buf,        // [shape: (NUM_EXITS, hidden_dim, output_classes)] Weights for exits.
    __global SCALAR_TYPE *__restrict grad_exit_weights,             // [shape: (NUM_EXITS, hidden_dim, output_classes)] Gradients for exit weights.
    __global SCALAR_TYPE *__restrict grad_exit_biases,              // [shape: (NUM_EXITS, output_classes)] Gradients for exit biases.
    __global SCALAR_TYPE *__restrict grad_hidden_contributions_buf, // [shape: (NUM_EXITS, padded_batch_size, hidden_dim)] Contributions to hidden gradients.
    int padded_batch_size,                                          // Padded batch size.
    int hidden_dim,                                                 // Hidden dimension.
    int output_classes,                                             // Number of output classes.
    int num_exits                                                   // Number of exits.
);

/**
 * @brief (Node 11) Computes gradients for shared weights and biases using aggregated hidden contributions.
 *
 * This kernel calculates gradients for the shared weights and biases of the neural network.
 *
 * Host Assumptions:
 * - Launched with a 2D NDRange where global sizes = (INPUT_DIM, HIDDEN_DIM).
 * - Must run after `calculate_exit_gradients`.
 * - grad_weights and grad_biases must be zeroed before calling.
 */
__kernel void calculate_shared_gradients(
    __local SCALAR_TYPE *local_grad_w,                                    // [size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for weight gradients.
    __local SCALAR_TYPE *local_grad_b,                                    // [size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for bias gradients.
    __global const SCALAR_TYPE *__restrict input_buf,                     // [shape: (padded_batch_size, input_dim)] Input data buffer.
    __global const SCALAR_TYPE *__restrict input_mask,                    // [shape: (padded_batch_size)] Mask for input data.
    __global const SCALAR_TYPE *__restrict hidden_buf,                    // [shape: (padded_batch_size, hidden_dim)] Hidden layer output.
    __global const SCALAR_TYPE *__restrict grad_hidden_contributions_buf, // [shape: (NUM_EXITS, padded_batch_size, hidden_dim)] Contributions from exits.
    __global SCALAR_TYPE *__restrict grad_weights,                        // [shape: (input_dim, hidden_dim)] Gradients for shared weights.
    __global SCALAR_TYPE *__restrict grad_biases,                         // [shape: (hidden_dim)] Gradients for shared biases.
    int padded_batch_size,                                                // Padded batch size.
    int input_dim,                                                        // Input dimension.
    int hidden_dim,                                                       // Hidden dimension.
    int num_exits                                                         // Number of exits.
);

/**
 * @brief (Node 12) Computes gradients for temperature parameters.
 *
 * This kernel calculates the gradients for the temperature parameters used in the exits.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange where global_size = NUM_EXITS.
 * - Requires `unscaled_logits_buf` from `compute_all_exits`.
 * - grad_temps must be zeroed before calling.
 */
__kernel void calculate_temp_gradients(
    __local SCALAR_TYPE *local_grad_sum,                         // [size: local_size[0] * sizeof(SCALAR_TYPE)] Local memory for gradient summation.
    __global const SCALAR_TYPE *__restrict unscaled_logits_buf,  // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Raw logits from exits.
    __global const SCALAR_TYPE *__restrict exit_probs_buf,       // [shape: (NUM_EXITS, padded_batch_size, output_classes)] Probabilities from exits.
    __global const SCALAR_TYPE *__restrict ensemble_weights_buf, // [shape: (padded_batch_size, num_exits)] Ensemble weights.
    __global const SCALAR_TYPE *__restrict temps_buf,            // [shape: (NUM_EXITS)] Temperature values.
    __global const int *__restrict targets_buf,                  // [shape: (padded_batch_size)] Target labels.
    __global const SCALAR_TYPE *__restrict targets_mask,         // [shape: (padded_batch_size)] Mask for targets.
    __global SCALAR_TYPE *__restrict grad_temps,                 // [shape: (NUM_EXITS)] Gradients for temperatures.
    int padded_batch_size,                                       // Padded batch size.
    int output_classes,                                          // Number of output classes.
    int num_exits                                                // Number of exits.
);

/**
 * @brief Performs Adam optimizer update for a parameter buffer.
 *
 * This kernel applies the Adam optimization algorithm to update the parameters based on the computed gradients.
 * It is used in the parameter update phase for shared weights, exit parameters, and temperatures.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange where global_size = total_params.
 * - Must run after gradient computation.
 * - Host provides Adam hyperparameters and pre-calculated bias corrections.
 */
__kernel void adam_update(
    __global const SCALAR_TYPE *__restrict grad, // [shape: (total_params)] Gradient buffer.
    SCALAR_TYPE beta1,                           // Adam hyperparameter beta1.
    SCALAR_TYPE beta2,                           // Adam hyperparameter beta2.
    SCALAR_TYPE beta1_t,                         // Time-adjusted beta1 (beta1^t).
    SCALAR_TYPE beta2_t,                         // Time-adjusted beta2 (beta2^t).
    SCALAR_TYPE learning_rate,                   // Global learning rate.
    SCALAR_TYPE epsilon,                         // Adam epsilon for numerical stability.
    __global SCALAR_TYPE *__restrict param,      // [shape: (total_params)] Parameter buffer, updated in-place.
    __global SCALAR_TYPE *__restrict m1,         // [shape: (total_params)] First moment vector, updated in-place.
    __global SCALAR_TYPE *__restrict m2,         // [shape: (total_params)] Second moment vector, updated in-place.
    int total_params                             // Total number of elements.
);

/**
 * @brief (Node 14) Clamps temperature values within a specified range.
 *
 * This kernel ensures that the temperature parameters stay within the defined minimum and maximum values.
 *
 * Host Assumptions:
 * - Launched with a 1D NDRange where global_size = NUM_EXITS.
 * - Assumes NUM_EXITS <= device's maximum work group size.
 */
__kernel void clamp_temperatures(
    __global SCALAR_TYPE *__restrict temps_buf, // [shape: (NUM_EXITS)] Temperature buffer, updated in-place.
    SCALAR_TYPE min_temp,                       // Minimum temperature value.
    SCALAR_TYPE max_temp,                       // Maximum temperature value.
    int         num_exits                       // Number of exits.
);

#endif // KERNELS_CL_H
