# tests/convergence/test_convergence_oracle_d.py
"""Oracle D precision-baseline convergence tests (ADR-033).

These tests use Oracle D (NumPy Precision Oracle) as a precision-aware
reference baseline and compare engine trajectories against it.

Two test categories:

1. **Oracle D self-convergence** — Oracle D alone must converge on each
   standard problem at FP32 precision (no engine dependency).  These
   validate that the mathematical model with faithful precision
   simulation converges before using it as a baseline.

2. **Engine vs Oracle D trajectory comparison** — The engine's
   convergence trajectory is compared against Oracle D's.  Divergence
   is logged as a warning (not a hard failure), since the engine's
   tiled decomposition, clipping, and reduction introduce expected
   differences.  Large divergence signals a decomposition bug.

Reference: ADR-033 (Precision Oracle Integration)
"""
from __future__ import annotations

import pytest

from tests.oracle.precision_oracle import NumpyPrecisionSpec

from .oracle_baseline import run_oracle_d_baseline
from .problems import iris, breast_cancer, xor
from .trajectory_comparison import compare_trajectories


# =====================================================================
# 1. Oracle D self-convergence on standard problems
# =====================================================================


class TestOracleDSelfConvergence:
    """Oracle D must converge on each standard problem at FP32.

    These are preconditions: if Oracle D fails to converge, it cannot
    serve as a baseline for engine comparison.
    """

    @pytest.mark.convergence
    @pytest.mark.convergence_fast
    def test_iris_cce_fp32(self) -> None:
        X, y = iris.load()
        history = run_oracle_d_baseline(
            input_dim=4,
            hidden_dim=iris.BASELINE.hidden_size,
            output_classes=3,
            num_modules=8,
            mode="CCE",
            X=X, y=y,
            epochs=iris.FAST_CRITERIA.epoch_budget,
            batch_size=iris.BASELINE.batch_size,
            learning_rate=iris.BASELINE.learning_rate,
            beta1=iris.BASELINE.beta1,
            beta2=iris.BASELINE.beta2,
            epsilon=iris.BASELINE.epsilon,
        )
        assert not history.has_nan_inf, "Oracle D produced NaN/Inf on Iris"
        assert history.final_loss < history.loss_curve[0], (
            f"Oracle D did not reduce loss on Iris: "
            f"initial={history.loss_curve[0]:.4f}, final={history.final_loss:.4f}"
        )

    @pytest.mark.convergence
    @pytest.mark.convergence_fast
    def test_xor_bce_fp32(self) -> None:
        X, y = xor.load()
        history = run_oracle_d_baseline(
            input_dim=2,
            hidden_dim=xor.BASELINE.hidden_size,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            X=X, y=y,
            epochs=xor.FAST_CRITERIA.epoch_budget,
            batch_size=xor.BASELINE.batch_size,
            learning_rate=xor.BASELINE.learning_rate,
            beta1=xor.BASELINE.beta1,
            beta2=xor.BASELINE.beta2,
            epsilon=xor.BASELINE.epsilon,
        )
        assert not history.has_nan_inf, "Oracle D produced NaN/Inf on XOR"
        assert history.final_loss < history.loss_curve[0], (
            f"Oracle D did not reduce loss on XOR: "
            f"initial={history.loss_curve[0]:.4f}, final={history.final_loss:.4f}"
        )

    @pytest.mark.convergence
    @pytest.mark.convergence_fast
    def test_breast_cancer_bce_fp32(self) -> None:
        X, y = breast_cancer.load()
        history = run_oracle_d_baseline(
            input_dim=30,
            hidden_dim=breast_cancer.BASELINE.hidden_size,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            X=X, y=y,
            epochs=breast_cancer.FAST_CRITERIA.epoch_budget,
            batch_size=breast_cancer.BASELINE.batch_size,
            learning_rate=breast_cancer.BASELINE.learning_rate,
            beta1=breast_cancer.BASELINE.beta1,
            beta2=breast_cancer.BASELINE.beta2,
            epsilon=breast_cancer.BASELINE.epsilon,
        )
        assert not history.has_nan_inf, "Oracle D produced NaN/Inf on Breast Cancer"
        assert history.final_loss < history.loss_curve[0], (
            f"Oracle D did not reduce loss on Breast Cancer: "
            f"initial={history.loss_curve[0]:.4f}, final={history.final_loss:.4f}"
        )


# =====================================================================
# 2. Oracle D multi-precision self-convergence
# =====================================================================

PRECISION_CONFIGS = [
    pytest.param(NumpyPrecisionSpec.float32(), id="fp32"),
    pytest.param(NumpyPrecisionSpec.mixed_f16_f32(), id="f16_f32"),
    pytest.param(NumpyPrecisionSpec.float64(), id="fp64"),
    pytest.param(NumpyPrecisionSpec.mixed_f32_f64_state(), id="f32_f64s"),
]


class TestOracleDPrecisionConvergence:
    """Oracle D converges on standard problems across precision configs.

    Validates that precision narrowing does not prevent convergence —
    the system should still learn, just potentially slower or to a
    different final accuracy.
    """

    @pytest.mark.convergence
    @pytest.mark.convergence_full
    @pytest.mark.parametrize("prec", PRECISION_CONFIGS)
    def test_iris_cce_precision(self, prec: NumpyPrecisionSpec) -> None:
        X, y = iris.load()
        history = run_oracle_d_baseline(
            input_dim=4,
            hidden_dim=iris.BASELINE.hidden_size,
            output_classes=3,
            num_modules=8,
            mode="CCE",
            X=X, y=y,
            epochs=iris.FULL_CRITERIA.epoch_budget,
            batch_size=iris.BASELINE.batch_size,
            precision=prec,
            learning_rate=iris.BASELINE.learning_rate,
            beta1=iris.BASELINE.beta1,
            beta2=iris.BASELINE.beta2,
            epsilon=iris.BASELINE.epsilon,
        )
        assert not history.has_nan_inf, f"Oracle D NaN/Inf on Iris/{prec}"
        assert history.final_loss < history.loss_curve[0], (
            f"Oracle D no improvement on Iris/{prec}: "
            f"{history.loss_curve[0]:.4f} → {history.final_loss:.4f}"
        )

    @pytest.mark.convergence
    @pytest.mark.convergence_full
    @pytest.mark.parametrize("prec", PRECISION_CONFIGS)
    def test_breast_cancer_bce_precision(self, prec: NumpyPrecisionSpec) -> None:
        X, y = breast_cancer.load()
        history = run_oracle_d_baseline(
            input_dim=30,
            hidden_dim=breast_cancer.BASELINE.hidden_size,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            X=X, y=y,
            epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
            batch_size=breast_cancer.BASELINE.batch_size,
            precision=prec,
            learning_rate=breast_cancer.BASELINE.learning_rate,
            beta1=breast_cancer.BASELINE.beta1,
            beta2=breast_cancer.BASELINE.beta2,
            epsilon=breast_cancer.BASELINE.epsilon,
        )
        assert not history.has_nan_inf, f"Oracle D NaN/Inf on Breast Cancer/{prec}"
        assert history.final_loss < history.loss_curve[0], (
            f"Oracle D no improvement on Breast Cancer/{prec}: "
            f"{history.loss_curve[0]:.4f} → {history.final_loss:.4f}"
        )


# =====================================================================
# 3. Engine vs Oracle D trajectory comparison
# =====================================================================


class TestEngineVsOracleD:
    """Compare engine convergence trajectory against Oracle D baseline.

    Trajectory divergence is *warned*, not asserted, because the engine
    applies tiled decomposition and staged clipping that Oracle D does
    not model.  Large divergence warrants investigation.
    """

    @pytest.mark.convergence
    @pytest.mark.convergence_full
    @pytest.mark.cpu
    def test_iris_engine_vs_oracle_d(self) -> None:
        from src.shared.optimizer_config import OptimizerConfig

        from .conftest import make_engine
        from .training_harness import run_training_loop

        X, y = iris.load()

        # Engine trajectory
        engine = make_engine(
            input_dim=4,
            hidden_dim=iris.BASELINE.hidden_size,
            output_classes=3,
            num_modules=8,
            mode="CCE",
            backend="cpu",
            gradient_clip_threshold=iris.BASELINE.gradient_clip_threshold,
            optimizer=OptimizerConfig(
                learning_rate=iris.BASELINE.learning_rate,
                beta1=iris.BASELINE.beta1,
                beta2=iris.BASELINE.beta2,
                epsilon=iris.BASELINE.epsilon,
            ),
        )
        engine_history = run_training_loop(
            engine, X, y,
            epochs=iris.FULL_CRITERIA.epoch_budget,
            batch_size=iris.BASELINE.batch_size,
            mode="CCE",
        )

        # Oracle D baseline
        oracle_history = run_oracle_d_baseline(
            input_dim=4,
            hidden_dim=iris.BASELINE.hidden_size,
            output_classes=3,
            num_modules=8,
            mode="CCE",
            X=X, y=y,
            epochs=iris.FULL_CRITERIA.epoch_budget,
            batch_size=iris.BASELINE.batch_size,
            learning_rate=iris.BASELINE.learning_rate,
            beta1=iris.BASELINE.beta1,
            beta2=iris.BASELINE.beta2,
            epsilon=iris.BASELINE.epsilon,
        )

        # Warn on divergence (do not fail — engine has clipping)
        compare_trajectories(
            oracle_history, engine_history,
            oracle_name="oracle_d",
            comparison_name="engine_cpu",
        )

    @pytest.mark.convergence
    @pytest.mark.convergence_full
    @pytest.mark.cpu
    def test_breast_cancer_engine_vs_oracle_d(self) -> None:
        from src.shared.optimizer_config import OptimizerConfig

        from .conftest import make_engine
        from .training_harness import run_training_loop

        X, y = breast_cancer.load()

        engine = make_engine(
            input_dim=30,
            hidden_dim=breast_cancer.BASELINE.hidden_size,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            backend="cpu",
            gradient_clip_threshold=breast_cancer.BASELINE.gradient_clip_threshold,
            optimizer=OptimizerConfig(
                learning_rate=breast_cancer.BASELINE.learning_rate,
                beta1=breast_cancer.BASELINE.beta1,
                beta2=breast_cancer.BASELINE.beta2,
                epsilon=breast_cancer.BASELINE.epsilon,
            ),
        )
        engine_history = run_training_loop(
            engine, X, y,
            epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
            batch_size=breast_cancer.BASELINE.batch_size,
            mode="BCE",
        )

        oracle_history = run_oracle_d_baseline(
            input_dim=30,
            hidden_dim=breast_cancer.BASELINE.hidden_size,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            X=X, y=y,
            epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
            batch_size=breast_cancer.BASELINE.batch_size,
            learning_rate=breast_cancer.BASELINE.learning_rate,
            beta1=breast_cancer.BASELINE.beta1,
            beta2=breast_cancer.BASELINE.beta2,
            epsilon=breast_cancer.BASELINE.epsilon,
        )

        compare_trajectories(
            oracle_history, engine_history,
            oracle_name="oracle_d",
            comparison_name="engine_cpu",
        )
