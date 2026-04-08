# tests/convergence/problems/xor.py
"""XOR convergence problem (ADR-028)."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "xor"
MODE: Literal["CCE", "BCE"] = "BCE"

BASELINE = BaselineHyperparameters(
    hidden_size=16,
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    weight_decay=0.0,
    gradient_clip_threshold=5.0,
    batch_size=32,
)

FAST_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.95,
    loss_threshold=0.3,
    epoch_budget=100,
    warmup_epochs=10,
    monotonicity_tolerance=None,
)

FULL_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.99,
    loss_threshold=0.05,
    epoch_budget=200,
    warmup_epochs=20,
    monotonicity_tolerance=0.05,
)


def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Generate XOR classification problem (seed=42, 200 samples)."""
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.int64)
    return X, y
