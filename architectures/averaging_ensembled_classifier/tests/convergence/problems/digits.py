# tests/convergence/problems/digits.py
"""Digits (8x8) convergence problem (ADR-028)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_digits

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "digits"
MODE: Literal["CCE", "BCE"] = "CCE"

BASELINE = BaselineHyperparameters(
    hidden_size=128,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-4,
    gradient_clip_threshold=1.0,
    batch_size=64,
)

FAST_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.85,
    loss_threshold=0.8,
    epoch_budget=30,
    warmup_epochs=3,
    monotonicity_tolerance=None,
)

FULL_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.95,
    loss_threshold=0.25,
    epoch_budget=150,
    warmup_epochs=15,
    monotonicity_tolerance=0.05,
)


def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load the Digits dataset, scaled to [0, 1]."""
    data = load_digits(return_X_y=False)
    X = data.data.astype(np.float32)  # type: ignore[union-attr]
    X = X / 16.0  # pixels in [0, 16]
    y = data.target.astype(np.int64)  # type: ignore[union-attr]
    return X, y
