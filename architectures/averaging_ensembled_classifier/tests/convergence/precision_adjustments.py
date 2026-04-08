# tests/convergence/precision_adjustments.py
"""Precision-specific convergence criteria adjustments (ADR-028)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.precision_config import PrecisionConfig

from .criteria import ConvergenceCriteria


@dataclass(frozen=True)
class PrecisionAdjustment:
    accuracy_delta: float
    loss_delta: float
    epoch_multiplier: float


PRECISION_ADJUSTMENTS: dict[str, PrecisionAdjustment] = {
    "float32": PrecisionAdjustment(accuracy_delta=0.0, loss_delta=0.0, epoch_multiplier=1.0),
    "float16": PrecisionAdjustment(accuracy_delta=-0.02, loss_delta=0.05, epoch_multiplier=1.2),
    "float64": PrecisionAdjustment(accuracy_delta=0.0, loss_delta=-0.01, epoch_multiplier=0.9),
    "float8_e4m3fn": PrecisionAdjustment(accuracy_delta=-0.05, loss_delta=0.1, epoch_multiplier=1.5),
}


def adjust_criteria_for_precision(
    base: ConvergenceCriteria,
    precision: PrecisionConfig,
) -> ConvergenceCriteria:
    """Return a new ConvergenceCriteria with precision-specific adjustments applied."""
    key = precision.storage_dtype.name
    adjustments = PRECISION_ADJUSTMENTS.get(key)
    if adjustments is None:
        return base
    return ConvergenceCriteria(
        problem_id=base.problem_id,
        mode=base.mode,
        accuracy_threshold=base.accuracy_threshold + adjustments.accuracy_delta,
        loss_threshold=base.loss_threshold + adjustments.loss_delta,
        epoch_budget=int(base.epoch_budget * adjustments.epoch_multiplier),
        warmup_epochs=base.warmup_epochs,
        monotonicity_tolerance=base.monotonicity_tolerance,
        nan_inf_allowed=base.nan_inf_allowed,
    )
