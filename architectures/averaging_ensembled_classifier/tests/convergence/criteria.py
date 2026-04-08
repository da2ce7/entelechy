# tests/convergence/criteria.py
"""Multi-criteria convergence validation (ADR-028 Option F)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BaselineHyperparameters:
    """Hyperparameter configuration for a problem (ADR-028).

    ``learning_rate``, ``beta1``, ``beta2``, and ``epsilon`` are
    exercised by convergence tests through ``make_engine()`` via
    ``OptimizerConfig`` (ADR-029, Phase 12A).  ``weight_decay`` is
    retained for documentation only — ``OptimizerConfig`` does not
    include weight decay.
    """
    hidden_size: int
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    gradient_clip_threshold: float
    batch_size: int


@dataclass(frozen=True)
class ConvergenceCriteria:
    """Multi-criteria convergence specification for a problem (ADR-028 Option F)."""
    problem_id: str
    mode: Literal["CCE", "BCE"]
    accuracy_threshold: float
    loss_threshold: float
    epoch_budget: int
    warmup_epochs: int
    monotonicity_tolerance: float | None
    nan_inf_allowed: bool = False


@dataclass
class TrainingHistory:
    """Per-epoch training metrics for convergence validation."""
    loss_curve: list[float]
    accuracy_curve: list[float]
    has_nan_inf: bool

    @property
    def final_loss(self) -> float:
        return self.loss_curve[-1]

    @property
    def final_accuracy(self) -> float:
        return self.accuracy_curve[-1]

    def epochs_to_threshold(self, accuracy_threshold: float) -> int:
        """Return the first epoch (1-indexed) where accuracy >= threshold, or -1."""
        for i, acc in enumerate(self.accuracy_curve):
            if acc >= accuracy_threshold:
                return i + 1
        return -1


def assert_convergence(history: TrainingHistory, criteria: ConvergenceCriteria) -> None:
    """Assert that training history meets all convergence criteria.

    Checks (in order):
    1. No NaN/Inf in any intermediate value.
    2. Final accuracy >= accuracy_threshold.
    3. Final loss <= loss_threshold.
    4. Convergence achieved within epoch_budget.
    5. Loss monotonicity after warmup (if monotonicity_tolerance is not None).
    """
    # 1. NaN/Inf absence
    if not criteria.nan_inf_allowed:
        assert not history.has_nan_inf, "Training produced NaN/Inf values"

    # 2. Accuracy threshold
    assert history.final_accuracy >= criteria.accuracy_threshold, (
        f"Final accuracy {history.final_accuracy:.4f} < threshold {criteria.accuracy_threshold}"
    )

    # 3. Loss threshold
    assert history.final_loss <= criteria.loss_threshold, (
        f"Final loss {history.final_loss:.4f} > threshold {criteria.loss_threshold}"
    )

    # 4. Epoch budget
    epochs_needed = history.epochs_to_threshold(criteria.accuracy_threshold)
    assert epochs_needed != -1 and epochs_needed <= criteria.epoch_budget, (
        f"Required {epochs_needed} epochs to reach {criteria.accuracy_threshold:.0%}; "
        f"budget was {criteria.epoch_budget}"
    )

    # 5. Loss monotonicity after warmup
    if criteria.monotonicity_tolerance is not None:
        for i in range(criteria.warmup_epochs, len(history.loss_curve) - 1):
            prev_loss = history.loss_curve[i]
            next_loss = history.loss_curve[i + 1]
            if prev_loss > 0:
                allowed = prev_loss * (1 + criteria.monotonicity_tolerance)
                assert next_loss <= allowed, (
                    f"Non-monotonic loss at epoch {i + 2}: "
                    f"{prev_loss:.4f} -> {next_loss:.4f} "
                    f"(+{(next_loss - prev_loss) / prev_loss:.1%}, "
                    f"tolerance={criteria.monotonicity_tolerance:.1%})"
                )
