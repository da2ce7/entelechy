# tests/convergence/problems/iris.py
"""Iris convergence problem (ADR-028)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "iris"
MODE: Literal["CCE", "BCE"] = "CCE"

BASELINE = BaselineHyperparameters(
    hidden_size=16,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.0,
    gradient_clip_threshold=1.0,
    batch_size=32,
)

FAST_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.90,
    loss_threshold=0.5,
    epoch_budget=30,
    warmup_epochs=3,
    monotonicity_tolerance=None,
)

FULL_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.97,
    loss_threshold=0.15,
    epoch_budget=100,
    warmup_epochs=10,
    monotonicity_tolerance=0.05,
)


def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load the Iris dataset, scaled to [0, 1]."""
    data = load_iris(return_X_y=False)
    X = data.data.astype(np.float32)  # type: ignore[union-attr]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    y = data.target.astype(np.int64)  # type: ignore[union-attr]
    return X, y
