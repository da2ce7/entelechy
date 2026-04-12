// phase_3_update.cl.c
//
// Learn-phase finalization and parameter update kernel implementations:
//   Nodes 21, 24, 25.
// Reference specification: kernels.cl.h (ADR-013 designation).

#ifdef __OPENCL_VERSION__
#else
#include "kernels.cl.h"
#endif

// ===========================================================================
// Node 21 — normalize_gradients
// ===========================================================================
// Strategy: Embarrassingly parallel "map" kernel.  Each work-item normalizes
// exactly one element of the summed gradient buffer by dividing by the
// effective batch size.  This converts the batch-wide gradient sum from the
// reduction engine into a true average gradient, ensuring that learning
// dynamics are independent of batch size — a fundamental requirement for
// stable and reproducible training.
//
// Dispatch geometry:
//   global = (parameter_count)
//   local  = backend-selected
//
// get_global_id(0) → gradient element index.

__kernel void normalize_gradients(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_summed_grad,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_final_grad,
    COMPUTE_TYPE                 src_scalar_REAL_effective_batch_size,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_parameter_count) {

    // --- 1. Work-Item → Element Mapping -----------------------------------
    const uint i = get_global_id(0);

    if (i >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // --- 2. Normalization -------------------------------------------------
    // Pre-compute the reciprocal: multiplication is faster than per-element
    // division on GPU hardware.  The epsilon term prevents division by zero
    // when the effective batch size is 0 (e.g., all samples were masked).
    const COMPUTE_TYPE normalizer =
        COMPUTE_ONE
        / (src_scalar_REAL_effective_batch_size + src_scalar_REAL_epsilon);

    // Compute-role buffers: read/write directly (no precision conversion).
    dest_buffer_GLOBAL_final_grad[i] =
        src_buffer_GLOBAL_summed_grad[i] * normalizer;
}

// ===========================================================================
// Node 24 — adam_update
// ===========================================================================
// Strategy: Stateful, embarrassingly parallel "map" kernel.  Each work-item
// updates a single parameter and its corresponding first and second moment
// vectors according to the Adam optimizer algorithm.
//
// The implementation enforces the State-Precision Accumulation invariant:
//   - EMA updates (m, v) and parameter subtraction are performed in
//     ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE), preserving state fidelity
//     across unbounded training steps.
//   - Bias correction and parameter delta computation are transformative
//     operations performed in COMPUTE_TYPE.
//   - When ACCUM_TYPE == COMPUTE_TYPE (the common case), all widen/narrow
//     casts are identity operations eliminated by the compiler.
//
// The host provides pre-computed bias correction terms (beta1^t, beta2^t)
// in FP64, narrowed to COMPUTE_TYPE at the interface boundary.  This avoids
// on-device precision loss for these geometrically-decaying terms.
//
// Dispatch geometry:
//   global = (parameter_count)
//   local  = backend-selected
//
// get_global_id(0) → parameter index within the dispatch slice.
//
// ADR-030: State-role buffers are indexed via [parameter_offset + i].
// The host guarantees (parameter_offset + parameter_count) <=
// total_parameter_count.

__kernel void adam_update(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_final_grad,
    __global STATE_TYPE         *update_buffer_GLOBAL_parameters,
    __global STATE_TYPE         *update_buffer_GLOBAL_m1,
    __global STATE_TYPE         *update_buffer_GLOBAL_m2,
    COMPUTE_TYPE                 src_scalar_REAL_learning_rate,
    COMPUTE_TYPE                 src_scalar_REAL_beta1_pow_t,
    COMPUTE_TYPE                 src_scalar_REAL_beta2_pow_t,
    COMPUTE_TYPE                 src_scalar_REAL_beta1,
    COMPUTE_TYPE                 src_scalar_REAL_beta2,
    COMPUTE_TYPE                 src_scalar_REAL_epsilon,
    uint                         src_scalar_NATURAL_parameter_offset,
    uint                         src_scalar_NATURAL_parameter_count,
    uint                         src_scalar_NATURAL_total_parameter_count) {

    // Axiom 1.4 — interface completeness.  This parameter exists for the
    // host's Validation Preconditions (slice bounds checking); the kernel
    // indexes via parameter_offset directly.
    (void)src_scalar_NATURAL_total_parameter_count;

    // --- 1. Work-Item → Parameter Mapping ---------------------------------
    const uint i = get_global_id(0);

    if (i >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // ADR-030: State-role buffers use offset-based indexing.
    const uint state_idx = src_scalar_NATURAL_parameter_offset + i;

    // --- 2. Load Inputs ---------------------------------------------------
    // Gradient (compute-role): zero-indexed per-dispatch buffer.
    const COMPUTE_TYPE g = src_buffer_GLOBAL_final_grad[i];

    // Moments (state-role): loaded at full accumulation precision to
    // preserve FP64 fidelity when STATE_TYPE > COMPUTE_TYPE.
    const ACCUM_TYPE m_prev =
        load_state_for_accum(update_buffer_GLOBAL_m1, state_idx);
    const ACCUM_TYPE v_prev =
        load_state_for_accum(update_buffer_GLOBAL_m2, state_idx);

    // --- 3. EMA Updates in ACCUM_TYPE (State-Precision Accumulation) ------
    // Widen gradient and hyperparameters to accumulation precision.
    // When ACCUM_TYPE == COMPUTE_TYPE, these are identity casts.
    const ACCUM_TYPE g_accum     = widen_to_accum(g);
    const ACCUM_TYPE beta1_accum = widen_to_accum(src_scalar_REAL_beta1);
    const ACCUM_TYPE beta2_accum = widen_to_accum(src_scalar_REAL_beta2);

    // First moment: m_new = β₁ · m_prev + (1 − β₁) · g
    const ACCUM_TYPE m_new =
        beta1_accum * m_prev + (ACCUM_ONE - beta1_accum) * g_accum;

    // Second moment: v_new = β₂ · v_prev + (1 − β₂) · g²
    const ACCUM_TYPE v_new =
        beta2_accum * v_prev
        + (ACCUM_ONE - beta2_accum) * (g_accum * g_accum);

    // Store updated moments at state precision.
    store_state_from_accum(update_buffer_GLOBAL_m1, state_idx, m_new);
    store_state_from_accum(update_buffer_GLOBAL_m2, state_idx, v_new);

    // --- 4. Bias Correction in COMPUTE_TYPE (Transformative) --------------
    // These are bounded, per-step operations — COMPUTE_TYPE suffices.
    const COMPUTE_TYPE m_hat =
        narrow_from_accum(m_new)
        / (COMPUTE_ONE - src_scalar_REAL_beta1_pow_t);
    const COMPUTE_TYPE v_hat =
        narrow_from_accum(v_new)
        / (COMPUTE_ONE - src_scalar_REAL_beta2_pow_t);

    // --- 5. Parameter Update in ACCUM_TYPE (Accumulative) -----------------
    // The delta is transformative (computed fresh each step), but the
    // subtraction p -= delta is accumulative — p refines over unbounded
    // training steps.  Performing the subtraction in ACCUM_TYPE preserves
    // parameter precision when STATE_TYPE > COMPUTE_TYPE.
    const COMPUTE_TYPE param_delta =
        src_scalar_REAL_learning_rate * m_hat
        / (MATH_SQRT(v_hat) + src_scalar_REAL_epsilon);

    const ACCUM_TYPE current_param =
        load_state_for_accum(update_buffer_GLOBAL_parameters, state_idx);
    store_state_from_accum(
        update_buffer_GLOBAL_parameters, state_idx,
        current_param - widen_to_accum(param_delta));
}

// ===========================================================================
// Node 25 — clamp_temperatures
// ===========================================================================
// Strategy: Embarrassingly parallel "map" kernel.  Each work-item operates
// on a single temperature parameter independently, enforcing domain-specific
// [min, max] constraints.  This "parameter governor" ensures learnable
// temperatures remain in a stable and meaningful range, preventing numerical
// instability in downstream Softmax/Sigmoid computations.
//
// Dispatch geometry:
//   global = (parameter_count)
//   local  = backend-selected
//
// get_global_id(0) → parameter index within the dispatch slice.
//
// ADR-030: Buffer is indexed via [parameter_offset + i].

__kernel void clamp_temperatures(
    __global STATE_TYPE *update_buffer_GLOBAL_temps,
    COMPUTE_TYPE         src_scalar_REAL_min_value,
    COMPUTE_TYPE         src_scalar_REAL_max_value,
    uint                 src_scalar_NATURAL_parameter_offset,
    uint                 src_scalar_NATURAL_parameter_count,
    uint                 src_scalar_NATURAL_total_parameter_count) {

    // Axiom 1.4 — interface completeness.  This parameter exists for the
    // host's Validation Preconditions (slice bounds checking); the kernel
    // indexes via parameter_offset directly.
    (void)src_scalar_NATURAL_total_parameter_count;

    // --- 1. Work-Item → Parameter Mapping ---------------------------------
    const uint idx = get_global_id(0);

    if (idx >= src_scalar_NATURAL_parameter_count) {
        return;
    }

    // ADR-030: State-role buffers use offset-based indexing.
    const uint state_idx = src_scalar_NATURAL_parameter_offset + idx;

    // --- 2. In-Place Clamping ---------------------------------------------
    // Precision Boundary Conversion: load at state precision, clamp in
    // compute precision, store back to state precision.  The `clamp`
    // intrinsic is a highly optimized OpenCL built-in.
    const COMPUTE_TYPE current_val =
        load_state(update_buffer_GLOBAL_temps, state_idx);
    store_state_update(
        update_buffer_GLOBAL_temps, state_idx,
        clamp(current_val, src_scalar_REAL_min_value,
              src_scalar_REAL_max_value));
}
