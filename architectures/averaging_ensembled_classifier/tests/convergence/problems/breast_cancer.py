# tests/convergence/problems/breast_cancer.py
"""Breast Cancer convergence problem (ADR-028)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "breast_cancer"
MODE: Literal["CCE", "BCE"] = "BCE"

BASELINE = BaselineHyperparameters(
    hidden_size=32,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=1e-4,
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
    accuracy_threshold=0.96,
    loss_threshold=0.15,
    epoch_budget=100,
    warmup_epochs=10,
    monotonicity_tolerance=0.05,
)


def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load Breast Cancer dataset, standardized to zero mean and unit variance."""
    data = load_breast_cancer(return_X_y=False)
    X = data.data.astype(np.float32)  # type: ignore[union-attr]
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std
    y = data.target.astype(np.int64)  # type: ignore[union-attr]
    return X, y
