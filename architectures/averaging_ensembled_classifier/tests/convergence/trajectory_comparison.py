# tests/convergence/trajectory_comparison.py
"""Cross-backend trajectory comparison (ADR-028 Option H)."""
from __future__ import annotations

import warnings

from .criteria import TrainingHistory

TRAJECTORY_DIVERGENCE_THRESHOLD = 0.10  # 10% relative accuracy difference


def compare_trajectories(
    oracle_history: TrainingHistory,
    comparison_history: TrainingHistory,
    *,
    oracle_name: str = "oracle",
    comparison_name: str = "comparison",
    divergence_threshold: float = TRAJECTORY_DIVERGENCE_THRESHOLD,
) -> None:
    """Warn (do not fail) if training trajectories diverge between backends.

    Emits UserWarning for each epoch where relative accuracy difference
    exceeds the threshold.
    """
    n_epochs = min(
        len(oracle_history.accuracy_curve),
        len(comparison_history.accuracy_curve),
    )
    for epoch in range(n_epochs):
        oracle_acc = oracle_history.accuracy_curve[epoch]
        comp_acc = comparison_history.accuracy_curve[epoch]
        if oracle_acc > 0:
            rel_diff = abs(oracle_acc - comp_acc) / oracle_acc
            if rel_diff > divergence_threshold:
                warnings.warn(
                    f"Trajectory divergence at epoch {epoch + 1}: "
                    f"{oracle_name}={oracle_acc:.2%}, {comparison_name}={comp_acc:.2%}, "
                    f"rel_diff={rel_diff:.1%}",
                    stacklevel=2,
                )
