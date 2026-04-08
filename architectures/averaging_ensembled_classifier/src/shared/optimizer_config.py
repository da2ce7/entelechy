# src/shared/optimizer_config.py
"""Optimizer hyperparameter configuration (ADR-029)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .precision_config import PrecisionConfig


@dataclass(frozen=True)
class OptimizerConfig:
    """Adam optimizer hyperparameters for plan construction.

    All fields have defaults matching the previously hardcoded values in
    ``build_learn_plan()``, ensuring backward compatibility for code that
    does not specify optimizer configuration.

    When ``epsilon`` is ``None``, the precision-aware default
    (``PrecisionConfig.compute_epsilon``) is used — preserving the
    existing behavior where epsilon depends on the compute dtype.
    """

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not (0 <= self.beta1 < 1):
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not (0 <= self.beta2 < 1):
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.epsilon is not None and self.epsilon <= 0:
            raise ValueError(
                f"epsilon must be positive (or None for precision-aware default), got {self.epsilon}"
            )

    def resolve_epsilon(self, precision: PrecisionConfig) -> float:
        """Return epsilon, falling back to the precision-aware default.

        When epsilon is None, returns ``precision.compute_epsilon`` —
        the smallest representable value for the compute dtype that
        prevents division by zero in the Adam update denominator.
        """
        if self.epsilon is not None:
            return self.epsilon
        return precision.compute_epsilon
