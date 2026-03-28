# tests/tolerance_config.py
"""Per-kernel tolerance tables for Tier 2 numerical correctness tests (ADR-008)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TolerancePair:
    atol: float
    rtol: float


FP32_DEFAULT = TolerancePair(atol=1e-5, rtol=1e-5)
FP16_DEFAULT = TolerancePair(atol=1e-2, rtol=1e-2)

KERNEL_TOLERANCES: dict[str, dict[str, TolerancePair]] = {
    "compute_probs_loss_cce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
        "fp16": TolerancePair(atol=5e-2, rtol=5e-2),
    },
    "compute_probs_loss_bce_chunk": {
        "fp32": TolerancePair(atol=1e-4, rtol=1e-4),
        "fp16": TolerancePair(atol=5e-2, rtol=5e-2),
    },
    "stabilize_reduce_grad_h": {
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


def get_tolerance(kernel_name: str, precision_label: str = "fp32") -> TolerancePair:
    """Look up tolerance for a kernel, falling back to defaults."""
    if kernel_name in KERNEL_TOLERANCES:
        overrides = KERNEL_TOLERANCES[kernel_name]
        if precision_label in overrides:
            return overrides[precision_label]
    return FP32_DEFAULT if precision_label == "fp32" else FP16_DEFAULT
