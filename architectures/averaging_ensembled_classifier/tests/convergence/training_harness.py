# tests/convergence/training_harness.py
"""Reusable multi-epoch training driver (ADR-028)."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.shared.engine import Engine

from .criteria import TrainingHistory


def _compute_cce_loss(predictions: NDArray, labels: NDArray, n_classes: int) -> float:
    """Compute CCE loss using numpy (FP64 accumulation)."""
    eps = 1e-12
    one_hot = np.zeros((len(labels), n_classes), dtype=np.float64)
    one_hot[np.arange(len(labels)), labels] = 1.0
    pred_clipped = np.clip(predictions.astype(np.float64), eps, 1 - eps)
    return float(-np.sum(one_hot * np.log(pred_clipped)) / len(labels))


def _compute_bce_loss(predictions: NDArray, labels: NDArray) -> float:
    """Compute BCE loss using numpy (FP64 accumulation)."""
    eps = 1e-12
    pred_clipped = np.clip(predictions.astype(np.float64).squeeze(), eps, 1 - eps)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(pred_clipped) + (1 - y) * np.log(1 - pred_clipped)))


def run_training_loop(
    engine: Engine,
    X: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    epochs: int,
    batch_size: int,
    mode: str = "CCE",
    shuffle_seed: int = 42,
) -> TrainingHistory:
    """Execute multi-epoch training using Engine.train_batch().

    For each epoch:
    1. Shuffle training data (deterministic, seeded per epoch).
    2. Iterate over mini-batches, calling engine.train_batch(X_batch, y_batch).
    3. After all batches, compute epoch-level metrics.
    4. Check for NaN/Inf in predictions.

    Returns a TrainingHistory recording per-epoch loss and accuracy curves.
    """
    n_samples = X.shape[0]
    n_classes = int(y.max()) + 1 if mode == "CCE" else 1

    loss_curve: list[float] = []
    accuracy_curve: list[float] = []
    has_nan_inf = False

    for epoch in range(epochs):
        # Deterministic per-epoch shuffle
        rng = np.random.default_rng(shuffle_seed + epoch)
        perm = rng.permutation(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_predictions: list[NDArray] = []
        epoch_labels: list[NDArray] = []

        # Mini-batch iteration
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            prediction = engine.train_batch(X_batch, y_batch)

            # NaN/Inf detection
            if np.any(~np.isfinite(prediction)):
                has_nan_inf = True
                # Record partial results and terminate early
                loss_curve.append(float("inf"))
                accuracy_curve.append(0.0)
                return TrainingHistory(
                    loss_curve=loss_curve,
                    accuracy_curve=accuracy_curve,
                    has_nan_inf=True,
                )

            epoch_predictions.append(prediction)
            epoch_labels.append(y_batch)

        # Aggregate epoch metrics
        all_preds = np.concatenate(epoch_predictions, axis=0)
        all_labels = np.concatenate(epoch_labels, axis=0)

        # Compute loss
        if mode == "CCE":
            epoch_loss = _compute_cce_loss(all_preds, all_labels, n_classes)
        else:
            epoch_loss = _compute_bce_loss(all_preds, all_labels)

        # Compute accuracy
        if mode == "CCE":
            predicted_classes = np.argmax(all_preds, axis=1)
            epoch_acc = float((predicted_classes == all_labels).mean())
        else:
            predicted_classes = (all_preds.squeeze() > 0.5).astype(np.int64)
            epoch_acc = float((predicted_classes == all_labels).mean())

        loss_curve.append(epoch_loss)
        accuracy_curve.append(epoch_acc)

    return TrainingHistory(
        loss_curve=loss_curve,
        accuracy_curve=accuracy_curve,
        has_nan_inf=has_nan_inf,
    )
