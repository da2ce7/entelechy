# tests/convergence/test_convergence_cce.py
"""CCE-mode convergence tests (ADR-028)."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

from src.shared.optimizer_config import OptimizerConfig

from .conftest import AVAILABLE_BACKENDS, make_engine
from .criteria import assert_convergence
from .training_harness import run_training_loop
from .trajectory_comparison import compare_trajectories
from .problems import iris, digits


# -------------------------------------------------------------------------
# Fast tests — every commit
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_iris_convergence_fast():
    """Iris CCE smoke test: basic training progress on CPU/FP32."""
    X, y = iris.load()
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
    history = run_training_loop(
        engine, X, y,
        epochs=iris.FAST_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, iris.FAST_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_fast
@pytest.mark.cpu
def test_digits_convergence_fast():
    """Digits CCE smoke test: mid-scale training progress on CPU/FP32."""
    X, y = digits.load()
    engine = make_engine(
        input_dim=64,
        hidden_dim=digits.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
        gradient_clip_threshold=digits.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=digits.BASELINE.learning_rate,
            beta1=digits.BASELINE.beta1,
            beta2=digits.BASELINE.beta2,
            epsilon=digits.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=digits.FAST_CRITERIA.epoch_budget,
        batch_size=digits.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, digits.FAST_CRITERIA)


# -------------------------------------------------------------------------
# Full tests — nightly/release
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_iris_convergence_full():
    """Iris CCE full convergence: 97% accuracy within 100 epochs on CPU/FP32."""
    X, y = iris.load()
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
    history = run_training_loop(
        engine, X, y,
        epochs=iris.FULL_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, iris.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_digits_convergence_full():
    """Digits CCE full convergence: 95% accuracy within 150 epochs on CPU/FP32."""
    X, y = digits.load()
    engine = make_engine(
        input_dim=64,
        hidden_dim=digits.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
        gradient_clip_threshold=digits.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=digits.BASELINE.learning_rate,
            beta1=digits.BASELINE.beta1,
            beta2=digits.BASELINE.beta2,
            epsilon=digits.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=digits.FULL_CRITERIA.epoch_budget,
        batch_size=digits.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, digits.FULL_CRITERIA)


# -------------------------------------------------------------------------
# Optional: MNIST and Fashion-MNIST (torchvision required)
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_mnist_convergence_full():
    """MNIST-1k CCE: 92% accuracy within 100 epochs on CPU/FP32."""
    from .problems import mnist

    X, y = mnist.load()  # skips if torchvision unavailable
    engine = make_engine(
        input_dim=784,
        hidden_dim=mnist.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
        gradient_clip_threshold=mnist.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=mnist.BASELINE.learning_rate,
            beta1=mnist.BASELINE.beta1,
            beta2=mnist.BASELINE.beta2,
            epsilon=mnist.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=mnist.FULL_CRITERIA.epoch_budget,
        batch_size=mnist.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, mnist.FULL_CRITERIA)


@pytest.mark.convergence
@pytest.mark.convergence_full
@pytest.mark.cpu
def test_fashion_mnist_convergence_full():
    """Fashion-MNIST-1k CCE: 85% accuracy within 150 epochs on CPU/FP32."""
    from .problems import fashion_mnist

    X, y = fashion_mnist.load()  # skips if torchvision unavailable
    engine = make_engine(
        input_dim=784,
        hidden_dim=fashion_mnist.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend="cpu",
        gradient_clip_threshold=fashion_mnist.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=fashion_mnist.BASELINE.learning_rate,
            beta1=fashion_mnist.BASELINE.beta1,
            beta2=fashion_mnist.BASELINE.beta2,
            epsilon=fashion_mnist.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=fashion_mnist.FULL_CRITERIA.epoch_budget,
        batch_size=fashion_mnist.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, fashion_mnist.FULL_CRITERIA)


# -------------------------------------------------------------------------
# Cross-backend tests
# -------------------------------------------------------------------------

@pytest.mark.convergence
@pytest.mark.convergence_full
def test_iris_convergence_cross_backend(backend_name):
    """Iris CCE convergence on each available backend, with trajectory comparison."""
    X, y = iris.load()
    engine = make_engine(
        input_dim=4,
        hidden_dim=iris.BASELINE.hidden_size,
        output_classes=3,
        num_modules=8,
        mode="CCE",
        backend=backend_name,
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
        epochs=iris.FULL_CRITERIA.epoch_budget,
        batch_size=iris.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, iris.FULL_CRITERIA)

    # Trajectory comparison against CPU oracle (if not already CPU)
    if backend_name != "cpu" and "cpu" in AVAILABLE_BACKENDS:
        oracle_engine = make_engine(
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
        oracle_history = run_training_loop(
            oracle_engine, X, y,
            epochs=iris.FULL_CRITERIA.epoch_budget,
            batch_size=iris.BASELINE.batch_size,
            mode="CCE",
        )
        compare_trajectories(
            oracle_history, history,
            oracle_name="cpu", comparison_name=backend_name,
        )


@pytest.mark.convergence
@pytest.mark.convergence_full
def test_digits_convergence_cross_backend(backend_name):
    """Digits CCE convergence on each available backend, with trajectory comparison."""
    X, y = digits.load()
    engine = make_engine(
        input_dim=64,
        hidden_dim=digits.BASELINE.hidden_size,
        output_classes=10,
        num_modules=8,
        mode="CCE",
        backend=backend_name,
        gradient_clip_threshold=digits.BASELINE.gradient_clip_threshold,
        optimizer=OptimizerConfig(
            learning_rate=digits.BASELINE.learning_rate,
            beta1=digits.BASELINE.beta1,
            beta2=digits.BASELINE.beta2,
            epsilon=digits.BASELINE.epsilon,
        ),
    )
    history = run_training_loop(
        engine, X, y,
        epochs=digits.FULL_CRITERIA.epoch_budget,
        batch_size=digits.BASELINE.batch_size,
        mode="CCE",
    )
    assert_convergence(history, digits.FULL_CRITERIA)

    if backend_name != "cpu" and "cpu" in AVAILABLE_BACKENDS:
        oracle_engine = make_engine(
            input_dim=64,
            hidden_dim=digits.BASELINE.hidden_size,
            output_classes=10,
            num_modules=8,
            mode="CCE",
            backend="cpu",
            gradient_clip_threshold=digits.BASELINE.gradient_clip_threshold,
            optimizer=OptimizerConfig(
                learning_rate=digits.BASELINE.learning_rate,
                beta1=digits.BASELINE.beta1,
                beta2=digits.BASELINE.beta2,
                epsilon=digits.BASELINE.epsilon,
            ),
        )
        oracle_history = run_training_loop(
            oracle_engine, X, y,
            epochs=digits.FULL_CRITERIA.epoch_budget,
            batch_size=digits.BASELINE.batch_size,
            mode="CCE",
        )
        compare_trajectories(
            oracle_history, history,
            oracle_name="cpu", comparison_name=backend_name,
        )
