# tests/convergence/test_convergence_precision.py
"""Precision-parameterized convergence tests (ADR-028)."""
from __future__ import annotations

import pytest

from src.shared.optimizer_config import OptimizerConfig
from src.shared.precision_config import PrecisionConfig

from .conftest import make_engine
from .criteria import assert_convergence
from .precision_adjustments import adjust_criteria_for_precision
from .training_harness import run_training_loop
from .problems import iris, breast_cancer


PRECISION_CONFIGS = [
    pytest.param(PrecisionConfig.float32(), id="fp32"),
    # pytest.param(PrecisionConfig.mixed_f16_f32(), id="fp16"),  # enable when FP16 convergence verified
    # pytest.param(PrecisionConfig.float64(), id="fp64"),        # enable when double-precision lands
    # pytest.param(PrecisionConfig.float8_e4m3(), id="fp8"),     # enable when FP8 lands
]


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_iris_convergence_precision(precision):
    """Iris CCE convergence across precision configurations."""
    X, y = iris.load()
    criteria = adjust_criteria_for_precision(iris.FULL_CRITERIA, precision)
    engine = make_engine(
        input_dim=4,
        hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3,
        num_modules=8,
        mode="CCE",
        backend="cpu",
        precision=precision,
        gradient_clip_threshold=iris.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=iris.BASELINE.learning_rate,
            beta1=iris.BASELINE.beta1,
            beta2=iris.BASELINE.beta2,
            epsilon=iris.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=criteria.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, criteria)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
@pytest.mark.parametrize("precision", PRECISION_CONFIGS)
def test_breast_cancer_convergence_precision(precision):
    """Breast Cancer BCE convergence across precision configurations."""
    X, y = breast_cancer.load()
    criteria = adjust_criteria_for_precision(breast_cancer.FULL_CRITERIA, precision)
    engine = make_engine(
        input_dim=30,
        hidden_dim=breast_cancer.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
        precision=precision,
        gradient_clip_threshold=breast_cancer.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=breast_cancer.BASELINE.learning_rate,
            beta1=breast_cancer.BASELINE.beta1,
            beta2=breast_cancer.BASELINE.beta2,
            epsilon=breast_cancer.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=criteria.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, criteria)
