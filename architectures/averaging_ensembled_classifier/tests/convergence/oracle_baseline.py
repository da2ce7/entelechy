# tests/convergence/oracle_baseline.py
"""Oracle D (NumPy Precision Oracle) baseline for convergence tests (ADR-033).

Provides an Oracle D baseline runner that produces the precision-limited
convergence trajectory for a given problem.  This serves as the ground
truth for what the engine should achieve: when the engine's trajectory
diverges from Oracle D's, it isolates whether the cause is a precision
effect (expected — Oracle D agrees) or a decomposition bug (unexpected —
Oracle D diverges from the engine but not from Oracle C).

Usage in convergence tests::

    from .oracle_baseline import run_oracle_d_baseline

    oracle_history = run_oracle_d_baseline(
        input_dim=4, hidden_dim=16, output_classes=3, num_modules=8,
        mode="CCE", X=X, y=y, epochs=100, batch_size=32,
        precision=NumpyPrecisionSpec.float32(),
        learning_rate=0.01,
    )

The returned ``OracleDHistory`` is compatible with the convergence
``trajectory_comparison`` module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tests.oracle.precision_oracle import (
    NumpyPrecisionSpec,
    NumPyPrecisionOracle,
    OracleDConfig,
)

from .criteria import TrainingHistory


@dataclass(frozen=True)
class OracleDBaselineConfig:
    """Problem configuration for Oracle D baselines.

    Maps the convergence problem's hyperparameters to OracleDConfig
    plus precision specification.
    """

    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    mode: str
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    precision: NumpyPrecisionSpec = NumpyPrecisionSpec.float32()
    mask_strategy: str = "explicit"
    model_gradient_storage: bool = True
    seed: int = 42

    def to_oracle_config(self) -> OracleDConfig:
        return OracleDConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_classes=self.output_classes,
            num_modules=self.num_modules,
            mode=self.mode,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
        )


def run_oracle_d_baseline(
    *,
    input_dim: int,
    hidden_dim: int,
    output_classes: int,
    num_modules: int,
    mode: str,
    X: NDArray[np.floating],
    y: NDArray[np.integer] | NDArray[np.floating],
    epochs: int,
    batch_size: int,
    precision: NumpyPrecisionSpec | None = None,
    learning_rate: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    mask_strategy: str = "explicit",
    model_gradient_storage: bool = True,
    seed: int = 42,
    shuffle_seed: int = 42,
) -> TrainingHistory:
    """Run Oracle D on a problem, returning a TrainingHistory.

    Mirrors the convergence ``run_training_loop`` interface but uses
    Oracle D instead of the engine.  The returned ``TrainingHistory``
    is directly comparable via ``trajectory_comparison.compare_trajectories``.

    Parameters
    ----------
    input_dim, hidden_dim, output_classes, num_modules, mode
        Model architecture parameters.
    X : NDArray, shape (N, input_dim)
        Input data (float).
    y : NDArray
        Labels.  CCE: ``(N,)`` int.  BCE: ``(N, output_classes)`` float.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size for epoch iteration.
    precision : NumpyPrecisionSpec or None
        Precision configuration.  Defaults to FP32.
    learning_rate, beta1, beta2, epsilon
        Optimizer hyperparameters.
    mask_strategy : str
        Hidden mask strategy (``"explicit"`` or ``"recompute"``).
    model_gradient_storage : bool
        Whether to model gradient storage round-trips.
    seed : int
        Weight initialization seed.
    shuffle_seed : int
        Per-epoch data shuffle seed base.

    Returns
    -------
    TrainingHistory
        Per-epoch loss and accuracy curves, compatible with engine
        training history for trajectory comparison.
    """
    if precision is None:
        precision = NumpyPrecisionSpec.float32()

    d_cfg = OracleDConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_classes=output_classes,
        num_modules=num_modules,
        mode=mode,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
    )

    oracle = NumPyPrecisionOracle.with_xavier_init(
        d_cfg,
        precision,
        seed=seed,
        mask_strategy=mask_strategy,
        model_gradient_storage=model_gradient_storage,
    )

    X_f = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)
    # BCE targets must be 2-D (N, output_classes) for Oracle D.
    if mode == "BCE" and y_arr.ndim == 1:
        y_arr = y_arr[:, np.newaxis].astype(np.float64)
    n_samples = X_f.shape[0]

    loss_curve: list[float] = []
    accuracy_curve: list[float] = []
    has_nan_inf = False

    for epoch in range(epochs):
        # Deterministic per-epoch shuffle (matches training_harness)
        rng = np.random.default_rng(shuffle_seed + epoch)
        perm = rng.permutation(n_samples)
        X_shuffled = X_f[perm]
        y_shuffled = y_arr[perm]

        # Mini-batch iteration
        epoch_predictions: list[np.ndarray] = []
        epoch_labels: list[np.ndarray] = []

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            probs = oracle.step(X_batch, y_batch)

            # Average across modules → (batch, output_classes)
            avg_probs = probs.mean(axis=0)

            if np.any(~np.isfinite(avg_probs)):
                has_nan_inf = True
                loss_curve.append(float("inf"))
                accuracy_curve.append(0.0)
                return TrainingHistory(
                    loss_curve=loss_curve,
                    accuracy_curve=accuracy_curve,
                    has_nan_inf=True,
                )

            epoch_predictions.append(avg_probs)
            epoch_labels.append(y_batch)

        all_preds = np.concatenate(epoch_predictions, axis=0)
        all_labels = np.concatenate(epoch_labels, axis=0)

        # Compute loss
        eps_loss = 1e-12
        if mode == "CCE":
            n_classes = int(all_labels.max()) + 1
            one_hot = np.zeros((len(all_labels), n_classes), dtype=np.float64)
            one_hot[np.arange(len(all_labels)), all_labels.astype(int)] = 1.0
            pred_clipped = np.clip(all_preds.astype(np.float64), eps_loss, 1 - eps_loss)
            epoch_loss = float(-np.sum(one_hot * np.log(pred_clipped)) / len(all_labels))
        else:
            pred_clipped = np.clip(
                all_preds.astype(np.float64).squeeze(), eps_loss, 1 - eps_loss,
            )
            y_f = all_labels.astype(np.float64).squeeze()
            epoch_loss = float(
                -np.mean(y_f * np.log(pred_clipped) + (1 - y_f) * np.log(1 - pred_clipped))
            )

        # Compute accuracy
        if mode == "CCE":
            predicted_classes = np.argmax(all_preds, axis=1)
            epoch_acc = float((predicted_classes == all_labels).mean())
        else:
            predicted_classes = (all_preds.squeeze() > 0.5).astype(np.int64)
            epoch_acc = float((predicted_classes == all_labels.squeeze()).mean())

        if not math.isfinite(epoch_loss):
            has_nan_inf = True

        loss_curve.append(epoch_loss)
        accuracy_curve.append(epoch_acc)

    return TrainingHistory(
        loss_curve=loss_curve,
        accuracy_curve=accuracy_curve,
        has_nan_inf=has_nan_inf,
    )
