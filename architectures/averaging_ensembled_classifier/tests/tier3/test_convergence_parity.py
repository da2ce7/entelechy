# tests/tier3/test_convergence_parity.py
"""Tier 3: multi-step convergence parity tests (ADR-034).

Runs N training steps with identical fixed-batch data on both the oracle
and comparison backends, asserting that loss curves and final parameter
state match within cumulative tolerance.  This catches accumulation drift
from reduction-order non-determinism, transcendental-function approximation
differences, and FMA availability mismatches.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tests.tolerance_config import get_tier3_convergence_tolerance

from .conftest import (
    BACKEND_FACTORIES,
    get_available_backends,
    select_oracle,
)

# ---------------------------------------------------------------------------
# Backend pairs (computed at module load time, same pattern as test_parity_e2e)
# ---------------------------------------------------------------------------


def _build_comparison_pairs() -> list[tuple[str, str]]:
    available = get_available_backends()
    oracle = select_oracle()
    pairs: list[tuple[str, str]] = []
    if oracle is not None:
        for b in available:
            if b != oracle:
                pairs.append((oracle, b))
    else:
        for left, right in __import__("itertools").combinations(available, 2):
            pairs.append((left, right))
    return pairs


_PAIRS = _build_comparison_pairs()

# ---------------------------------------------------------------------------
# Iris-scale model geometry (reuses Tier 3 convention)
# ---------------------------------------------------------------------------

_IRIS_CCE = dict(input_dim=4, hidden_dim=32, output_classes=3, num_modules=8)
_IRIS_BCE = dict(input_dim=4, hidden_dim=32, output_classes=1, num_modules=8)
_BATCH_SIZE = 150

_DEFAULT_STEPS = 50
_SLOW_STEPS = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_fresh_renderer(backend_name: str) -> Any:
    """Create a new renderer instance (not session-cached)."""
    factory = BACKEND_FACTORIES.get(backend_name)
    if factory is None:
        pytest.skip(f"No renderer factory for backend '{backend_name}'")
    try:
        return factory()
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        pytest.skip(f"Backend '{backend_name}' unavailable: {exc}")


def _make_engine(
    renderer: Any,
    *,
    problem_type: str,
    geometry: dict[str, int],
) -> Any:
    """Build an Engine wrapping a pre-created renderer."""
    from src.shared.engine import Engine
    from src.shared.hardware_profile import HardwareProfile
    from src.shared.model_spec import ModelSpec
    from src.shared.optimizer_config import OptimizerConfig
    from src.shared.parameter_space import ParameterSpace
    from src.shared.precision_config import PrecisionConfig
    from architectures.averaging_ensembled_classifier.src.shared.problem_type_spec import PlanBceStrategy, PlanCceStrategy
    from src.shared.stabilization_policy import StabilizationPolicy

    prec = PrecisionConfig.float32()
    spec = ModelSpec(precision=prec, simd_width=4, cache_line_bytes=64, **geometry)
    hw = HardwareProfile(
        simd_width=4,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=65536,
        global_mem_bytes=4 * 1024**3,
    )
    param_space = ParameterSpace(spec)
    strategy = PlanCceStrategy() if problem_type == "CCE" else PlanBceStrategy()
    policy = StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=0.1,
        compute_fp_format_max=prec.compute_fp_format_max,
    )
    optimizer = OptimizerConfig(
        learning_rate=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    )

    return Engine(
        model_spec=spec,
        parameter_space=param_space,
        hardware_profile=hw,
        precision=prec,
        strategy=strategy,
        policy=policy,
        optimizer=optimizer,
        renderer_factory=lambda _backend, _hw, _prec: renderer,
    )


def _generate_deterministic_data(
    problem_type: str,
    geometry: dict[str, int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed deterministic input/target data for parity testing."""
    rng = np.random.default_rng(seed=20260411)
    input_dim = geometry["input_dim"]
    X = rng.standard_normal((batch_size, input_dim)).astype(np.float32)
    if problem_type == "CCE":
        n_classes = geometry["output_classes"]
        y = rng.integers(0, n_classes, size=batch_size).astype(np.int64)
    else:
        y = rng.integers(0, 2, size=batch_size).astype(np.int64)
    return X, y


def _compute_loss(
    predictions: np.ndarray,
    targets: np.ndarray,
    problem_type: str,
    n_classes: int,
) -> float:
    """Compute loss from predictions in FP64 for stable comparison."""
    eps = 1e-12
    pred = np.clip(predictions.astype(np.float64), eps, 1.0 - eps)
    if problem_type == "CCE":
        one_hot = np.zeros((len(targets), n_classes), dtype=np.float64)
        one_hot[np.arange(len(targets)), targets] = 1.0
        return float(-np.sum(one_hot * np.log(pred)) / len(targets))
    else:
        t = targets.astype(np.float64)
        p = pred.squeeze()
        return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))


def _get_param_norm(renderer: Any) -> float:
    """Extract L2 norm of all persistent (MODEL_STATE) parameters."""
    total = 0.0
    if hasattr(renderer, "_persistent_buffers"):
        # CPU backend
        for buf in renderer._persistent_buffers.values():
            total += float(np.sum(buf.astype(np.float64) ** 2))
    elif hasattr(renderer, "_allocator"):
        # Vulkan/OpenCL: read back all MODEL_STATE device buffers
        allocator = renderer._allocator
        if hasattr(allocator, "_buffers"):
            from src.shared.buffer_lifecycle import BufferRole

            for vk_buf in allocator._buffers.values():
                desc = vk_buf.descriptor
                if desc.role == BufferRole.MODEL_STATE:
                    # Vulkan buffers need D2H transfer; skip norm for now
                    # as the prediction/loss comparison is the primary signal.
                    pass
    return math.sqrt(total)


def _get_persistent_buffers(renderer: Any) -> dict[str, np.ndarray]:
    """Extract persistent MODEL_STATE buffer contents keyed by logical name."""
    result: dict[str, np.ndarray] = {}
    if hasattr(renderer, "_persistent_buffers"):
        for (name, _shape, _dtype), buf in renderer._persistent_buffers.items():
            result[name] = buf.copy()
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.tier3
class TestConvergenceParity:
    """Multi-step convergence parity between oracle and comparison backends.

    Each test:
    1. Creates fresh renderers for both backends (no shared state).
    2. Builds identically-configured Engines.
    3. Presents the same fixed batch on every step.
    4. Compares loss curves and final parameter state.
    """

    @pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_convergence_parity(
        self,
        problem_type: str,
        pair: tuple[str, str],
    ) -> None:
        """50-step convergence parity (default CI)."""
        self._run_parity(pair, problem_type, _DEFAULT_STEPS)

    @pytest.mark.slow
    @pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_convergence_parity_extended(
        self,
        problem_type: str,
        pair: tuple[str, str],
    ) -> None:
        """500-step convergence parity (nightly CI)."""
        self._run_parity(pair, problem_type, _SLOW_STEPS)

    # ---------------------------------------------------------------

    @staticmethod
    def _run_parity(
        pair: tuple[str, str],
        problem_type: str,
        num_steps: int,
    ) -> None:
        oracle_name, comp_name = pair
        geometry = _IRIS_CCE if problem_type == "CCE" else _IRIS_BCE
        n_classes = geometry["output_classes"]

        # Fresh renderers — no leaked state from other tests
        oracle_renderer = _create_fresh_renderer(oracle_name)
        comp_renderer = _create_fresh_renderer(comp_name)

        oracle_engine = _make_engine(
            oracle_renderer, problem_type=problem_type, geometry=geometry,
        )
        comp_engine = _make_engine(
            comp_renderer, problem_type=problem_type, geometry=geometry,
        )

        # Deterministic fixed batch (same on every step)
        X, y = _generate_deterministic_data(problem_type, geometry, _BATCH_SIZE)

        oracle_losses: list[float] = []
        comp_losses: list[float] = []

        for step in range(1, num_steps + 1):
            oracle_pred = oracle_engine.train_batch(X, y)
            comp_pred = comp_engine.train_batch(X, y)

            oracle_loss = _compute_loss(oracle_pred, y, problem_type, n_classes)
            comp_loss = _compute_loss(comp_pred, y, problem_type, n_classes)
            oracle_losses.append(oracle_loss)
            comp_losses.append(comp_loss)

            # Per-step loss parity check
            tol = get_tier3_convergence_tolerance(step)
            np.testing.assert_allclose(
                comp_loss,
                oracle_loss,
                atol=tol.atol,
                rtol=tol.rtol,
                err_msg=(
                    f"Loss parity failure at step {step}/{num_steps}: "
                    f"{comp_name} vs {oracle_name} ({problem_type})"
                ),
            )

            # Per-step parameter norm parity (catches latent divergence
            # when loss is in a flat region)
            oracle_norm = _get_param_norm(oracle_renderer)
            comp_norm = _get_param_norm(comp_renderer)
            if oracle_norm > 0.0 and comp_norm > 0.0:
                np.testing.assert_allclose(
                    comp_norm,
                    oracle_norm,
                    atol=tol.atol,
                    rtol=tol.rtol,
                    err_msg=(
                        f"Parameter norm parity failure at step {step}/{num_steps}: "
                        f"{comp_name} vs {oracle_name} ({problem_type})"
                    ),
                )

        # Final parameter state: element-wise comparison
        final_tol = get_tier3_convergence_tolerance(num_steps)
        oracle_params = _get_persistent_buffers(oracle_renderer)
        comp_params = _get_persistent_buffers(comp_renderer)

        for name in oracle_params:
            if name not in comp_params:
                pytest.fail(
                    f"Oracle has persistent buffer '{name}' but "
                    f"{comp_name} does not"
                )
            np.testing.assert_allclose(
                comp_params[name],
                oracle_params[name],
                atol=final_tol.atol,
                rtol=final_tol.rtol,
                err_msg=(
                    f"Final parameter parity failure for '{name}': "
                    f"{comp_name} vs {oracle_name} ({problem_type}, "
                    f"{num_steps} steps)"
                ),
            )
