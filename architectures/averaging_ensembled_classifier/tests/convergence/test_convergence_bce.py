# tests/convergence/test_convergence_bce.py
"""BCE-mode convergence tests (ADR-028)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from src.shared.optimizer_config import OptimizerConfig

from .conftest import AVAILABLE_BACKENDS, make_engine
from .criteria import assert_convergence
from .training_harness import run_training_loop
from .trajectory_comparison import compare_trajectories
from .problems import xor, two_moons, breast_cancer


# -------------------------------------------------------------------------
# Fast tests — every commit
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_xor_convergence_fast():
    """XOR BCE smoke test: basic non-linear convergence on CPU/FP32."""
    X, y = xor.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=xor.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
        gradient_clip_threshold=xor.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=xor.BASELINE.learning_rate,
            beta1=xor.BASELINE.beta1,
            beta2=xor.BASELINE.beta2,
            epsilon=xor.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=xor.FAST_CRITERIA.epoch_budget,
        batch_size=xor.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, xor.FAST_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_breast_cancer_convergence_fast():
    """Breast Cancer BCE smoke test: moderate-dim convergence on CPU/FP32."""
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
    history = run_training_loop(
        engine, X, y,
        epochs=breast_cancer.FAST_CRITERIA.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, breast_cancer.FAST_CRITERIA)


# -------------------------------------------------------------------------
# Full tests — nightly/release
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_xor_convergence_full():
    """XOR BCE full: 99% accuracy within 200 epochs on CPU/FP32."""
    X, y = xor.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=xor.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
        gradient_clip_threshold=xor.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=xor.BASELINE.learning_rate,
            beta1=xor.BASELINE.beta1,
            beta2=xor.BASELINE.beta2,
            epsilon=xor.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=xor.FULL_CRITERIA.epoch_budget,
        batch_size=xor.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, xor.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_two_moons_convergence_full():
    """Two Moons BCE full: 98% accuracy within 150 epochs on CPU/FP32."""
    X, y = two_moons.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=two_moons.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend="cpu",
        gradient_clip_threshold=two_moons.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=two_moons.BASELINE.learning_rate,
            beta1=two_moons.BASELINE.beta1,
            beta2=two_moons.BASELINE.beta2,
            epsilon=two_moons.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=two_moons.FULL_CRITERIA.epoch_budget,
        batch_size=two_moons.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, two_moons.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_breast_cancer_convergence_full():
    """Breast Cancer BCE full: 96% accuracy within 100 epochs on CPU/FP32."""
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
    history = run_training_loop(
        engine, X, y,
        epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, breast_cancer.FULL_CRITERIA)


# -------------------------------------------------------------------------
# Cross-backend tests
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
def test_xor_convergence_cross_backend(backend_name):
    """XOR BCE convergence on each available backend, with trajectory comparison."""
    X, y = xor.load()
    engine = make_engine(
        input_dim=2,
        hidden_dim=xor.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend=backend_name,
        gradient_clip_threshold=xor.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=xor.BASELINE.learning_rate,
            beta1=xor.BASELINE.beta1,
            beta2=xor.BASELINE.beta2,
            epsilon=xor.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=xor.FULL_CRITERIA.epoch_budget,
        batch_size=xor.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, xor.FULL_CRITERIA)

    if backend_name != "cpu" and "cpu" in AVAILABLE_BACKENDS:
        oracle_engine = make_engine(
            input_dim=2,
            hidden_dim=xor.BASELINE.hidden_size,
            output_classes=1,
            num_modules=8,
            mode="BCE",
            backend="cpu",
            gradient_clip_threshold=xor.BASELINE.gradient_clip_threshold,
            optimizer=OptimizerConfig(
                learning_rate=xor.BASELINE.learning_rate,
                beta1=xor.BASELINE.beta1,
                beta2=xor.BASELINE.beta2,
                epsilon=xor.BASELINE.epsilon,
            ),
        )
        oracle_history = run_training_loop(
            oracle_engine, X, y,
            epochs=xor.FULL_CRITERIA.epoch_budget,
            batch_size=xor.BASELINE.batch_size,
            mode="BCE",
        )
        compare_trajectories(
            oracle_history, history,
            oracle_name="cpu", comparison_name=backend_name,
        )


@pytest.mark.convergence
@pytest.mark.convergence_full
def test_breast_cancer_convergence_cross_backend(backend_name):
    """Breast Cancer BCE convergence on each available backend."""
    X, y = breast_cancer.load()
    engine = make_engine(
        input_dim=30,
        hidden_dim=breast_cancer.BASELINE.hidden_size,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        backend=backend_name,
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
        epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
        batch_size=breast_cancer.BASELINE.batch_size,
        mode="BCE",
    )
    assert_convergence(history, breast_cancer.FULL_CRITERIA)

    if backend_name != "cpu" and "cpu" in AVAILABLE_BACKENDS:
        oracle_engine = make_engine(
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
        oracle_history = run_training_loop(
            oracle_engine, X, y,
            epochs=breast_cancer.FULL_CRITERIA.epoch_budget,
            batch_size=breast_cancer.BASELINE.batch_size,
            mode="BCE",
        )
        compare_trajectories(
            oracle_history, history,
            oracle_name="cpu", comparison_name=backend_name,
        )
