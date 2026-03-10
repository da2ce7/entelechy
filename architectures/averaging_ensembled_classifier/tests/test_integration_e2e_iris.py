# tests/test_integration_e2e_iris.py
from __future__ import annotations

"""
End-to-End Integration Test: Train on Iris and Verify Classification.

This test exercises the complete system — from data loading through OpenCL
kernel compilation, buffer allocation, multi-epoch training, and finally
prediction evaluation — on the canonical Iris dataset.

It answers the most fundamental question about any classifier:
does it actually learn to predict?

An OpenCL device IS required.
"""

import os
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional OpenCL import — skip gracefully when absent.
# ---------------------------------------------------------------------------
try:
    import pyopencl as cl

    _has_opencl = True
    try:
        _ctx = cl.create_some_context(interactive=False)
        _has_opencl_device = len(_ctx.devices) > 0
        del _ctx
    except Exception:
        _has_opencl_device = False
except ImportError:
    _has_opencl = False
    _has_opencl_device = False

pytestmark = pytest.mark.skipif(
    not (_has_opencl and _has_opencl_device),
    reason="No OpenCL device available",
)

if TYPE_CHECKING:
    from sklearn.utils import Bunch

# ---------------------------------------------------------------------------
# Imports — guarded by the skip marker above.
# ---------------------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.utils import Bunch as _Bunch

from src.cl_context_manager import (
    OpenCLContextManager,
    Float32ComputeEnvironment,
    Float16ComputeEnvironment,
)
from src.arch_primitives import Float32Context
from src.model_spec import Float32ModelSpec
from src.parameter_space import ParameterSpace
from src.main_orchestrator import (
    TrainingOrchestrator,
    TrainingHyperparams,
    StabilizationConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "kernels",
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compute_env() -> Float32ComputeEnvironment:
    manager = OpenCLContextManager(kernel_source_dir=_KERNEL_DIR)
    return manager.build_and_discover(Float32Context)


@pytest.fixture(scope="module")
def iris_data(compute_env: Float32ComputeEnvironment):
    """Load Iris, cast to the environment's scalar type."""
    iris = cast(_Bunch, load_iris())
    X = iris.data.astype(compute_env.SCALAR_NP_TYPE)
    y = iris.target.astype(np.int32)
    return X, y


# =========================================================================
# The Test
# =========================================================================


class TestEndToEndIrisClassification:
    """Verify that the system learns to classify Iris flowers."""

    def _build_orchestrator(
        self,
        compute_env: Float32ComputeEnvironment,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int,
        learning_rate: float = 0.01,
    ) -> TrainingOrchestrator:
        spec = Float32ModelSpec(
            input_dim=X.shape[1],
            output_classes=len(np.unique(y)),
            hidden_dim=32,
            num_modules=8,
            simd_width=compute_env.arch_consts.simd_width,
            cache_line_bytes=compute_env.arch_consts.global_mem_cacheline_size,
        )
        param_space = ParameterSpace(spec=spec)
        hyperparams = TrainingHyperparams(
            epochs=epochs,
            learning_rate=learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-7,
            temp_min=0.1,
            temp_max=10.0,
            stabilization=StabilizationConfig(max_grad_norm=1.0, lambda_=1.0),
            reduction_k_grad_h=16,
        )
        return TrainingOrchestrator(
            model_spec=spec,
            param_space=param_space,
            compute_env=compute_env,
            hyperparams=hyperparams,
            adaptation_strategy="CACHE",
            problem_type_name="CCE",
            clipping_strategy_name="GLOBAL",
            batch_size=X.shape[0],
        )

    # -----------------------------------------------------------------
    # Core: does training for N epochs actually improve accuracy?
    # -----------------------------------------------------------------

    @staticmethod
    def _ensemble_predict(probs: np.ndarray, n_samples: int, n_classes: int) -> tuple:
        """Average per-module probabilities to get ensemble predictions.

        The `final_probs` buffer has shape (num_modules, batch_size, num_classes).
        The averaging-ensembled architecture defines the final prediction as the
        mean across the module axis.

        Returns (ensemble_probs, predictions) both clipped to real dimensions.
        """
        # Average over the module axis (axis=0).
        ensemble_probs = probs[:, :n_samples, :n_classes].mean(axis=0)
        predictions = np.argmax(ensemble_probs, axis=1)
        return ensemble_probs, predictions

    def test_iris_accuracy_above_chance(
        self,
        compute_env: Float32ComputeEnvironment,
        iris_data,
    ) -> None:
        """After training, accuracy must exceed random chance (33 %) by a wide margin."""
        X, y = iris_data
        orchestrator = self._build_orchestrator(compute_env, X, y, epochs=50)

        raw_probs = orchestrator.train(X, y)

        _, predictions = self._ensemble_predict(raw_probs, X.shape[0], len(np.unique(y)))
        accuracy = np.mean(predictions == y)
        print(f"\n  Iris accuracy after 50 epochs: {accuracy:.2%}")

        # Random chance for 3-class balanced Iris is ~33 %.
        # Any functioning classifier should comfortably exceed 60 % after
        # 50 gradient steps on such a simple dataset.  We intentionally
        # keep the bar modest — this test verifies learning, not SOTA.
        assert accuracy > 0.60, (
            f"Accuracy {accuracy:.2%} is not convincingly above chance. "
            f"The model may not be learning."
        )

    def test_iris_accuracy_converges_higher(
        self,
        compute_env: Float32ComputeEnvironment,
        iris_data,
    ) -> None:
        """With more epochs, the model should reach high accuracy on this trivially separable dataset."""
        X, y = iris_data
        orchestrator = self._build_orchestrator(compute_env, X, y, epochs=200)

        raw_probs = orchestrator.train(X, y)

        _, predictions = self._ensemble_predict(raw_probs, X.shape[0], len(np.unique(y)))
        accuracy = np.mean(predictions == y)
        print(f"\n  Iris accuracy after 200 epochs: {accuracy:.2%}")

        # Iris is linearly separable for 2 of 3 classes and nearly so for
        # the third.  A well-functioning ensemble classifier ought to reach
        # ≥ 85 % training accuracy after 200 full-batch gradient steps.
        assert accuracy > 0.85, (
            f"Accuracy {accuracy:.2%} after 200 epochs is unexpectedly low. "
            f"The model may not be converging properly."
        )

    def test_loss_decreases_over_training(
        self,
        compute_env: Float32ComputeEnvironment,
        iris_data,
    ) -> None:
        """Verify that the cross-entropy loss decreases between early and late training."""
        X, y = iris_data
        n_classes = len(np.unique(y))

        # --- First: train for just 5 epochs and read loss ---
        orchestrator_early = self._build_orchestrator(
            compute_env, X, y, epochs=5,
        )
        raw_probs_early = orchestrator_early.train(X, y)
        probs_early, _ = self._ensemble_predict(raw_probs_early, X.shape[0], n_classes)
        # Clip for numerical safety in log
        probs_early_clipped = np.clip(probs_early, 1e-12, 1.0)
        loss_early = -np.mean(np.log(probs_early_clipped[np.arange(len(y)), y]))

        # --- Then: train for 100 epochs and read loss ---
        orchestrator_late = self._build_orchestrator(
            compute_env, X, y, epochs=100,
        )
        raw_probs_late = orchestrator_late.train(X, y)
        probs_late, _ = self._ensemble_predict(raw_probs_late, X.shape[0], n_classes)
        probs_late_clipped = np.clip(probs_late, 1e-12, 1.0)
        loss_late = -np.mean(np.log(probs_late_clipped[np.arange(len(y)), y]))

        print(f"\n  Loss after  5 epochs: {loss_early:.4f}")
        print(f"  Loss after 100 epochs: {loss_late:.4f}")

        assert loss_late < loss_early, (
            f"Loss did not decrease: {loss_early:.4f} (5 epochs) -> {loss_late:.4f} (100 epochs). "
            f"The training loop may not be applying gradients correctly."
        )

    def test_probabilities_are_valid_distributions(
        self,
        compute_env: Float32ComputeEnvironment,
        iris_data,
    ) -> None:
        """Output probabilities must be non-negative and sum to ~1 per sample."""
        X, y = iris_data
        n_classes = len(np.unique(y))
        orchestrator = self._build_orchestrator(compute_env, X, y, epochs=10)

        raw_probs = orchestrator.train(X, y)
        probs, _ = self._ensemble_predict(raw_probs, X.shape[0], n_classes)

        assert np.all(probs >= 0), "Some output probabilities are negative."
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-4,
            err_msg="Probability rows do not sum to 1.0 — softmax may be broken.",
        )
