# tests/tolerance_config.py
"""Per-kernel tolerance tables for Tier 2 and Tier 3 numerical correctness tests (ADR-008)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TolerancePair:
    atol: float
    rtol: float


FP32_DEFAULT = TolerancePair(atol=1e-5, rtol=1e-5)
FP16_DEFAULT = TolerancePair(atol=1e-2, rtol=1e-2)
MIXED_DEFAULT = TolerancePair(atol=1e-5, rtol=1e-5)  # compute is FP32

KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
        "fp16": TolerancePair(atol=5e-2, rtol=5e-2),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
        "fp16": TolerancePair(atol=5e-2, rtol=5e-2),
    },
    "stabilize_and_reduce_grad_hidden_activations": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "adam_update": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "aggregate_local_reduce": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "aggregate_register_reduce": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
}

# CPU-specific tolerance overrides — tighter than OpenCL due to IEEE 754
# compliance and deterministic reduction order. Used via get_cpu_tolerance().
CPU_KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "forward_pass": {
        "fp32": TolerancePair(atol=1e-6, rtol=1e-5),
    },
    "render_logits_chunk": {
        "fp32": TolerancePair(atol=1e-6, rtol=1e-5),
    },
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "calculate_module_param_grads_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "backprop_error_to_hidden_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "clip_partial_gradients": {
        "fp32": TolerancePair(atol=1e-6, rtol=1e-5),
    },
    "gather_and_permute_grad_hidden_activations": {
        "fp32": TolerancePair(atol=1e-6, rtol=1e-5),
    },
    "stabilize_and_reduce_grad_hidden_activations": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "backprop_shared_weights_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "backprop_shared_biases_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "clip_shared_gradients_chunk": {
        "fp32": TolerancePair(atol=1e-6, rtol=1e-5),
    },
    "normalize_gradients": {
        "fp32": TolerancePair(atol=1e-7, rtol=1e-6),
    },
    "adam_update": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "clamp_temperatures": {
        "fp32": TolerancePair(atol=0.0, rtol=0.0),
    },
    "act_plan_e2e": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-4),
    },
    "learn_plan_e2e": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-3),
    },
}


def get_tolerance(kernel_name: str, precision_label: str = "fp32") -> TolerancePair:
    """Look up tolerance for a kernel, falling back to defaults."""
    if kernel_name in KERNEL_TOLERANCES:
        overrides = KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
        # mixed_f16_f32 computes in FP32, so fall back to fp32 overrides
        if precision_label == "mixed" and "fp32" in overrides:
            return overrides["fp32"]
    if precision_label == "fp16":
        return FP16_DEFAULT
    if precision_label == "mixed":
        return MIXED_DEFAULT
    return FP32_DEFAULT


def get_cpu_tolerance(kernel_name: str, precision_label: str = "fp32") -> TolerancePair:
    """Look up CPU-specific tolerance, falling back to shared defaults."""
    if kernel_name in CPU_KERNEL_TOLERANCES:
        overrides = CPU_KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
    return get_tolerance(kernel_name, precision_label)


# ---------------------------------------------------------------------------
# Vulkan-specific tolerance overrides
# ---------------------------------------------------------------------------
# Slightly wider than CPU due to non-IEEE fused multiply-add, subgroup
# reduction order, and device-specific rounding.  Reduction-bearing kernels
# get 1e-4; element-wise kernels stay at 1e-5.

VULKAN_KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "forward_pass": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-5),
    },
    "render_logits_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-5),
    },
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "calculate_module_param_grads_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "backprop_error_to_hidden_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "clip_partial_gradients": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-5),
    },
    "gather_and_permute_grad_hidden_activations": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-5),
    },
    "stabilize_and_reduce_grad_hidden_activations": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "aggregate_local_reduce": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "aggregate_register_reduce": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "backprop_shared_weights_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "backprop_shared_biases_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "clip_shared_gradients_chunk": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-5),
    },
    "normalize_gradients": {
        "fp32": TolerancePair(atol=1e-5, rtol=1e-5),
    },
    "adam_update": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
    },
    "clamp_temperatures": {
        "fp32": TolerancePair(atol=0.0, rtol=0.0),
    },
    "act_plan_e2e": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "learn_plan_e2e": {
        "fp32": TolerancePair(atol=1e-3, rtol=1e-3),
    },
}


def get_vulkan_tolerance(kernel_name: str, precision_label: str = "fp32") -> TolerancePair:
    """Look up Vulkan-specific tolerance, falling back to shared defaults."""
    if kernel_name in VULKAN_KERNEL_TOLERANCES:
        overrides = VULKAN_KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
    return get_tolerance(kernel_name, precision_label)


# ---------------------------------------------------------------------------
# Tier 3: Cross-backend (parity) tolerance overrides
# ---------------------------------------------------------------------------
# Wider than Tier 2 because both backends independently diverge from the
# mathematical reference in different directions.  Accumulation order,
# fused multiply-add availability, and denormal handling all contribute.

TIER3_FP32_DEFAULT = TolerancePair(atol=1e-4, rtol=1e-4)
TIER3_FP16_DEFAULT = TolerancePair(atol=5e-2, rtol=5e-2)

TIER3_KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "stabilize_and_reduce_grad_hidden_activations": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "adam_update": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "aggregate_local_reduce": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
    "aggregate_register_reduce": {
        "fp32": TolerancePair(atol=5e-4, rtol=5e-4),
    },
}


def get_tier3_tolerance(kernel_name: str, precision_label: str = "fp32") -> TolerancePair:
    """Look up Tier 3 cross-backend tolerance, falling back to Tier 3 defaults."""
    if kernel_name in TIER3_KERNEL_TOLERANCES:
        overrides = TIER3_KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
    return TIER3_FP32_DEFAULT if precision_label == "fp32" else TIER3_FP16_DEFAULT


import math


def get_tier3_convergence_tolerance(
    step: int,
    precision_label: str = "fp32",
    *,
    alpha: float = 1.0,
) -> TolerancePair:
    """Cumulative tolerance for multi-step convergence parity (ADR-034).

    Tolerance grows as atol_base * (1 + alpha * sqrt(step)) to model
    sub-linear error accumulation from per-step rounding differences
    in reduction order, transcendental approximations, and FMA.

    The base tolerances come from the single-step Tier 3 learn_plan_e2e
    entry (loss comparison uses the softmax/loss kernel tolerance).
    """
    base = get_tier3_tolerance("learn_plan_e2e", precision_label)
    growth = 1.0 + alpha * math.sqrt(step)
    return TolerancePair(atol=base.atol * growth, rtol=base.rtol * growth)


def precision_label_from_config(precision) -> str:
    """Derive the tolerance label from a PrecisionConfig instance.

    Returns "fp32", "fp16", or "mixed" based on the three-role dtype
    configuration.  Mixed = FP16 storage with FP32 compute/state.
    """
    if precision.storage_dtype == np.dtype(np.float16):
        if precision.compute_dtype == np.dtype(np.float32):
            return "mixed"
        return "fp16"
    return "fp32"
