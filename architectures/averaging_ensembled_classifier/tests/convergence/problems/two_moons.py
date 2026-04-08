# tests/convergence/problems/two_moons.py
"""Two Moons convergence problem (ADR-028)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "two_moons"
MODE: Literal["CCE", "BCE"] = "BCE"

BASELINE = BaselineHyperparameters(
    hidden_size=16,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.0,
    gradient_clip_threshold=1.0,
    batch_size=64,
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
    accuracy_threshold=0.98,
    loss_threshold=0.10,
    epoch_budget=150,
    warmup_epochs=10,
    monotonicity_tolerance=0.05,
)


def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load Two Moons dataset, standardized to zero mean and unit variance."""
    X_raw, y_raw = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X = np.asarray(X_raw, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    y = np.asarray(y_raw, dtype=np.int64)
    return X, y
