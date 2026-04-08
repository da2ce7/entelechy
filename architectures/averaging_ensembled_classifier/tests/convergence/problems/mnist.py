# tests/convergence/problems/mnist.py
"""MNIST-1k convergence problem (ADR-028). Requires torchvision."""
from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..criteria import BaselineHyperparameters, ConvergenceCriteria

PROBLEM_ID: str = "mnist_1k"
MODE: Literal["CCE", "BCE"] = "CCE"

BASELINE = BaselineHyperparameters(
    hidden_size=256,
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
    accuracy_threshold=0.80,
    loss_threshold=1.0,
    epoch_budget=20,
    warmup_epochs=3,
    monotonicity_tolerance=None,
)

FULL_CRITERIA = ConvergenceCriteria(
    problem_id=PROBLEM_ID,
    mode=MODE,
    accuracy_threshold=0.92,
    loss_threshold=0.35,
    epoch_budget=100,
    warmup_epochs=10,
    monotonicity_tolerance=0.10,
)


def load() -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Load MNIST 1k subset; skip test if torchvision unavailable."""
    import pytest
    torchvision = pytest.importorskip("torchvision", reason="MNIST requires torchvision")

    dataset = torchvision.datasets.MNIST(
        root="/tmp/convergence_data", train=True, download=True
    )
    X_full = dataset.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_full = dataset.targets.numpy().astype(np.int64)

    # Deterministic 1k subset
    rng = np.random.default_rng(42)
    indices = rng.choice(len(X_full), size=1000, replace=False)
    return X_full[indices], y_full[indices]
