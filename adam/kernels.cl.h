#ifndef KERNELS_CL_H
#define KERNELS_CL_H

// Check for OpenCL environment
#ifdef __OPENCL_VERSION__
// In OpenCL mode, require SCALAR_TYPE and SIMD_WIDTH to be defined
#ifndef SCALAR_TYPE
#error "SCALAR_TYPE must be defined in OpenCL mode"
#endif
#ifndef SIMD_WIDTH
#error "SIMD_WIDTH must be defined in OpenCL mode"
#endif
#else
// Not in OpenCL mode, set default values
#ifndef __kernel
#define __kernel
#endif
#ifndef __local
#define __local
#endif
#ifndef __global
#define __global
#endif
#ifndef SCALAR_TYPE
#define SCALAR_TYPE float
#endif
#ifndef SIMD_WIDTH
#define SIMD_WIDTH 1
#endif
#endif

// Set default for USE_FAST_MATH if not defined
#ifndef USE_FAST_MATH
#define USE_FAST_MATH 0
#endif

// Kernel function declarations for neural network operations

/* ======== Feed Forward Pass ========
Host Assumptions:
- The kernel relies on the input_mask to process only valid elements, making no assumptions about the padding strategy.
- Local memory size is workgroup_x * sizeof(SCALAR_TYPE), where workgroup_x is the local work size in the x-dimension.
- Workgroup dimensions are optimized by the host, typically multiples of SIMD_WIDTH.
- Input buffer dimensions are (padded_batch_size, padded_input_dim), with padded sizes provided by the host.
- Weight matrix is reshaped for vectorized operations, but the kernel uses masks to handle valid elements.
- Hidden buffer is allocated with padding, but the kernel uses hidden_mask to process only valid elements.
*/
__kernel void forward_pass(
    __local SCALAR_TYPE* local_mem,     // [req_size: local_sizes[0] * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE* __restrict input,           // [shape: (padded_batch_size, padded_input_dim)]
    __global const SCALAR_TYPE* __restrict input_mask,      // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict weights,         // [shape: (ceil(HIDDEN_DIM/SIMD_WIDTH), INPUT_DIM, SIMD_WIDTH)]
    __global const SCALAR_TYPE* __restrict biases,          // [shape: padded_hidden_dim]
    __global SCALAR_TYPE* __restrict hidden,                // [shape: (padded_batch_size, padded_hidden_dim)]
    __global SCALAR_TYPE* __restrict hidden_mask            // [shape: padded_batch_size]
);

/* ======== Exit Probability Computation ========
Host Assumptions:
- The kernel relies on masks (e.g., hidden_mask, targets_mask) to process only valid elements, making no assumptions about the padding strategy.
- Local memory size is workgroup_x * sizeof(SCALAR_TYPE), where workgroup_x is the local work size in the x-dimension.
- Workgroup dimensions are optimized by the host, typically multiples of SIMD_WIDTH.
- The kernel processes each exit based on the provided exit_idx.
- Buffer dimensions (e.g., padded_batch_size, padded_hidden_dim) are provided by the host, and masks are used to handle valid elements.
*/
__kernel void compute_exit_probabilities(
    __local SCALAR_TYPE* local_mem,     // [req_size: local_sizes[0] * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE* __restrict hidden,          // [shape: (padded_batch_size, padded_hidden_dim)]
    __global const SCALAR_TYPE* __restrict hidden_mask,     // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict exit_weights,    // [shape: (NUM_EXITS, SIMD_WIDTH, ceil(HIDDEN_DIM)/SIMD_WIDTH, padded_output_classes)]
    __global const SCALAR_TYPE* __restrict exit_biases,     // [shape: (NUM_EXITS, padded_output_classes)]
    __global SCALAR_TYPE* __restrict exit_probs,            // [shape: (NUM_EXITS, padded_batch_size, padded_output_classes)]
    __global SCALAR_TYPE* __restrict exit_probs_mask,       // [shape: padded_batch_size]
    __global SCALAR_TYPE* __restrict losses,                // [shape: (NUM_EXITS, padded_batch_size)]
    __global SCALAR_TYPE* __restrict losses_mask,           // [shape: padded_batch_size]
    __global const int* __restrict targets,                 // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict targets_mask,    // [shape: padded_batch_size]
    int exit_idx,                         // Exit index between 0 and NUM_EXITS-1
    __global const SCALAR_TYPE* __restrict temperatures,    // [shape: NUM_EXITS]
    int padded_batch_size,                // Padded batch size provided by the host
    int hidden_dim,                       // True HIDDEN_DIM
    int output_classes,                   // True OUTPUT_CLASSES
    int padded_hidden_dim,                // Padded hidden dimension provided by the host
    int padded_output_classes,            // Padded output classes provided by the host
    int num_exits                         // HOST parameter NUM_EXITS
);

/* ======== Gradient Computation ========
Host Assumptions:
- The kernel relies on masks (e.g., input_mask, hidden_mask, targets_mask) to process only valid elements, making no assumptions about the padding strategy.
- Workgroup dimensions are optimized by the host, typically with local_sizes[0] being a multiple of SIMD_WIDTH.
- Local memory size is HIDDEN_DIM * sizeof(SCALAR_TYPE).
- Buffer dimensions (e.g., padded_batch_size, padded_input_dim) are provided by the host, and masks are used to handle valid elements.
*/
__kernel void compute_gradients(
    __local SCALAR_TYPE* local_mem,     // [req_size: HIDDEN_DIM * sizeof(SCALAR_TYPE)]
    __global const SCALAR_TYPE* __restrict input,           // [shape: (padded_batch_size, padded_input_dim)]
    __global const SCALAR_TYPE* __restrict input_mask,      // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict hidden,          // [shape: (padded_batch_size, padded_hidden_dim)]
    __global const SCALAR_TYPE* __restrict hidden_mask,     // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict exit_probs,      // [shape: (NUM_EXITS, padded_batch_size, padded_output_classes)]
    __global const SCALAR_TYPE* __restrict exit_probs_mask, // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict exit_weights,    // [shape: (NUM_EXITS, SIMD_WIDTH, ceil(HIDDEN_DIM)/SIMD_WIDTH, padded_output_classes)]
    __global SCALAR_TYPE* __restrict grad_weights,          // [shape: (INPUT_DIM, ceil(HIDDEN_DIM/SIMD_WIDTH), SIMD_WIDTH)]
    __global SCALAR_TYPE* __restrict grad_biases,           // [shape: padded_hidden_dim]
    __global SCALAR_TYPE* __restrict grad_exit_weights,     // [shape: (NUM_EXITS, SIMD_WIDTH, ceil(HIDDEN_DIM)/SIMD_WIDTH, padded_output_classes)]
    __global SCALAR_TYPE* __restrict grad_exit_biases,      // [shape: (NUM_EXITS, padded_output_classes)]
    __global const int* __restrict targets,                 // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict targets_mask,    // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict temperatures,    // [shape: NUM_EXITS]
    int input_dim,                        // Parameter INPUT_DIM
    int hidden_dim,                       // Parameter HIDDEN_DIM
    int output_classes,                   // Parameter OUTPUT_CLASSES
    int padded_input_dim,                 // Padded input dimension provided by the host
    int padded_hidden_dim,                // Padded hidden dimension provided by the host
    int padded_output_classes,            // Padded output classes provided by the host
    int padded_batch_size                 // Padded batch size provided by the host
);

/* ======== Temperature Gradient ========
Host Assumptions:
- The kernel relies on masks (e.g., targets_mask) to process only valid elements, making no assumptions about the padding strategy.
- Global work size is (NUM_EXITS,), with one work item per exit.
- Buffer dimensions (e.g., padded_batch_size, padded_output_classes) are provided by the host, and masks are used to handle valid elements.
*/
__kernel void compute_temp_gradients(
    __global const SCALAR_TYPE* __restrict exit_probs,      // [shape: (NUM_EXITS, padded_batch_size, padded_output_classes)]
    __global const SCALAR_TYPE* __restrict exit_probs_mask, // [shape: padded_batch_size]
    __global const int* __restrict targets,                 // [shape: padded_batch_size]
    __global const SCALAR_TYPE* __restrict targets_mask,    // [shape: padded_batch_size]
    __global SCALAR_TYPE* __restrict grad_temps,            // [shape: NUM_EXITS]
    __global const SCALAR_TYPE* __restrict temperatures,    // [shape: NUM_EXITS]
    int output_classes,                   // Actual class count
    int padded_output_classes,            // Padded output classes provided by the host
    int padded_batch_size,                // Padded batch size provided by the host
    int num_exits                         // HOST parameter NUM_EXITS
);

/* ======== Adam Update ========
Host Assumptions:
- The kernel updates the entire parameter buffer, relying on the host to manage any padding.
- total_params is the total number of elements in the parameter buffer, which may include padded elements.
*/
__kernel void adam_update(
    __global const SCALAR_TYPE* __restrict grad,            // [shape: total_params]
    __global SCALAR_TYPE* __restrict param,                 // [shape: total_params]
    __global SCALAR_TYPE* __restrict m1,                    // [shape: total_params]
    __global SCALAR_TYPE* __restrict m2,                    // [shape: total_params]
    SCALAR_TYPE beta1,                      // Adam parameter beta1
    SCALAR_TYPE beta2,                      // Adam parameter beta2
    SCALAR_TYPE beta1_t,                    // Time-adjusted beta1
    SCALAR_TYPE beta2_t,                    // Time-adjusted beta2
    SCALAR_TYPE learning_rate,              // Learning rate
    SCALAR_TYPE epsilon,                    // Adam parameter epsilon
    int total_params                        // Total number of elements in the parameter buffer
);

/* ======== Temperature Clamping ========
Host Assumptions:
- The kernel clamps all temperature values, including any padded elements if present.
- Global work size is (NUM_EXITS,), with one work item per temperature value.
- Assumes NUM_EXITS is within the device's maximum work group size.
*/
__kernel void clamp_temperatures(
    __global SCALAR_TYPE* __restrict temps, // [shape: NUM_EXITS]
    SCALAR_TYPE min_temp,                   // Minimum temperature value
    SCALAR_TYPE max_temp,                   // Maximum temperature value
    int num_exits                           // HOST parameter NUM_EXITS
);

#endif // KERNELS_CL_H