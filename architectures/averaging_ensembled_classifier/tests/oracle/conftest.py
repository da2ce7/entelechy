# tests/oracle/conftest.py
"""Shared fixtures for oracle tests.

All tests in this package require PyTorch; the entire directory is
auto-skipped when torch is not installed.

Oracle D (NumPyPrecisionOracle) is always available — it depends only
on numpy, not torch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch", reason="Oracle tests require PyTorch")

from .autograd_oracle import AutogradOracle
from .convergence_oracle import ConvergenceOracle
from .faithful_oracle import FaithfulOracle
from .oracle_config import OracleConfig
from .precision_oracle import (
    NumPyPrecisionOracle,
    OracleDConfig,
)


# ── Tolerance helper ─────────────────────────────────────────────────


def _loss_close(
    a: float, b: float, atol: float = 1e-9, rtol: float = 1e-9,
) -> bool:
    """Combined absolute+relative tolerance check (torch.allclose semantics).

    Bare ``abs(a - b) < atol`` breaks on large-magnitude losses (e.g.
    XOR with 200 samples) where the absolute diff is tiny *relative*
    to the values but exceeds a tight absolute threshold.
    """
    return abs(a - b) <= atol + rtol * abs(b)


# ── Small problem configs for fast unit tests ────────────────────────


@pytest.fixture()
def small_cce_config() -> OracleConfig:
    """Minimal CCE problem: 3 modules, 4 classes, 5 inputs, 8 hidden."""
    return OracleConfig(
        input_dim=5,
        hidden_dim=8,
        output_classes=4,
        num_modules=3,
        mode="CCE",
    )


@pytest.fixture()
def small_bce_config() -> OracleConfig:
    """Minimal BCE problem: 3 modules, 4 classes, 5 inputs, 8 hidden."""
    return OracleConfig(
        input_dim=5,
        hidden_dim=8,
        output_classes=4,
        num_modules=3,
        mode="BCE",
    )


@pytest.fixture()
def medium_cce_config() -> OracleConfig:
    """Medium CCE problem that triggers tiling: 20 modules, 20 classes."""
    return OracleConfig(
        input_dim=10,
        hidden_dim=16,
        output_classes=20,
        num_modules=20,
        mode="CCE",
    )


# ── Data generators ─────────────────────────────────────────────────


@pytest.fixture()
def small_cce_data(small_cce_config: OracleConfig) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(42)
    X = torch.randn(8, small_cce_config.input_dim, dtype=torch.float64, generator=gen)
    targets = torch.randint(0, small_cce_config.output_classes, (8,), generator=gen)
    return X, targets


@pytest.fixture()
def small_bce_data(small_bce_config: OracleConfig) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(42)
    X = torch.randn(8, small_bce_config.input_dim, dtype=torch.float64, generator=gen)
    targets = torch.randint(0, 2, (8, small_bce_config.output_classes), generator=gen).to(
        torch.float64
    )
    return X, targets


@pytest.fixture()
def medium_cce_data(medium_cce_config: OracleConfig) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(42)
    X = torch.randn(16, medium_cce_config.input_dim, dtype=torch.float64, generator=gen)
    targets = torch.randint(0, medium_cce_config.output_classes, (16,), generator=gen)
    return X, targets


# ── Oracle factory helpers ───────────────────────────────────────────


def sync_oracles(*oracles: FaithfulOracle | AutogradOracle | ConvergenceOracle) -> None:
    """Set all oracles to identical initial state (from the first one)."""
    state = oracles[0].export_state()
    for o in oracles[1:]:
        o.load_state(state)


@torch.no_grad()
def init_with_seed(
    oracle: FaithfulOracle | AutogradOracle | ConvergenceOracle,
    seed: int = 123,
) -> None:
    """Initialize oracle parameters with small deterministic values."""
    gen = torch.Generator().manual_seed(seed)
    for _, param in oracle.named_params():
        data = param.data if hasattr(param, 'data') else param
        # Small Xavier-like init
        fan_in = data.shape[-1] if data.dim() >= 2 else data.shape[0]
        std = 1.0 / fan_in**0.5
        data.copy_(torch.randn(data.shape, dtype=torch.float64, generator=gen) * std)


def make_unclipped_config(base: OracleConfig) -> OracleConfig:
    """Return config with effectively-infinite clip thresholds.

    Used by cross-oracle tests that need clipping disabled so that
    A/B/C/D produce mathematically identical gradients.
    """
    return OracleConfig(
        input_dim=base.input_dim,
        hidden_dim=base.hidden_dim,
        output_classes=base.output_classes,
        num_modules=base.num_modules,
        mode=base.mode,
        learning_rate=base.learning_rate,
        beta1=base.beta1,
        beta2=base.beta2,
        epsilon=base.epsilon,
        t_algorithmic=float("inf"),
        lambda_=base.lambda_,
        compute_fp_format_max=float("inf"),
        temp_min=base.temp_min,
        temp_max=base.temp_max,
    )


# ── Oracle D (Precision Oracle) helpers ──────────────────────────────


def oracle_config_to_d_config(cfg: OracleConfig) -> OracleDConfig:
    """Convert OracleConfig (torch-based oracles) to OracleDConfig (Oracle D).

    Maps the shared hyperparameters.  Oracle D uses its own
    ``NumpyPrecisionSpec`` for precision roles rather than the
    engine's ``PrecisionConfig``.
    """
    return OracleDConfig(
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_classes=cfg.output_classes,
        num_modules=cfg.num_modules,
        mode=cfg.mode,
        learning_rate=cfg.learning_rate,
        beta1=cfg.beta1,
        beta2=cfg.beta2,
        epsilon=cfg.epsilon,
        temp_min=cfg.temp_min,
        temp_max=cfg.temp_max,
        normalize_epsilon=cfg.epsilon,
    )


def sync_oracle_d_from_torch(
    oracle_d: NumPyPrecisionOracle,
    torch_state: dict[str, torch.Tensor],
) -> None:
    """Load Oracle D state from a torch-based oracle's export_state().

    Converts torch tensors to numpy arrays at Oracle D's state precision.
    """
    np_state: dict[str, np.ndarray] = {}
    for key, tensor in torch_state.items():
        np_state[key] = tensor.detach().cpu().numpy()
    oracle_d.load_state(np_state)


def torch_state_from_oracle_d(
    oracle_d: NumPyPrecisionOracle,
) -> dict[str, torch.Tensor]:
    """Export Oracle D state as torch tensors for cross-oracle comparison."""
    np_state = oracle_d.export_state()
    return {k: torch.from_numpy(v.copy()) for k, v in np_state.items()}  # pyright: ignore[reportUnknownMemberType]


# ── Oracle D fixtures ────────────────────────────────────────────────


@pytest.fixture()
def small_cce_d_config(small_cce_config: OracleConfig) -> OracleDConfig:
    """OracleDConfig corresponding to the small CCE OracleConfig."""
    return oracle_config_to_d_config(small_cce_config)


@pytest.fixture()
def small_bce_d_config(small_bce_config: OracleConfig) -> OracleDConfig:
    """OracleDConfig corresponding to the small BCE OracleConfig."""
    return oracle_config_to_d_config(small_bce_config)


@pytest.fixture()
def small_cce_np_data(
    small_cce_config: OracleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy data for standalone Oracle D tests.

    WARNING: uses ``np.random.default_rng(42)`` — NOT numerically
    identical to ``small_cce_data`` (which uses ``torch.Generator``
    with the same seed but a different RNG algorithm).  For
    cross-oracle D-vs-C tests, convert the torch fixture via
    ``_torch_to_numpy()`` instead of using this fixture.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((8, small_cce_config.input_dim))
    targets = rng.integers(0, small_cce_config.output_classes, size=(8,))
    return X, targets


@pytest.fixture()
def small_bce_np_data(
    small_bce_config: OracleConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy data for standalone Oracle D tests.

    WARNING: uses ``np.random.default_rng(42)`` — NOT numerically
    identical to ``small_bce_data`` (which uses ``torch.Generator``
    with the same seed but a different RNG algorithm).  For
    cross-oracle D-vs-C tests, convert the torch fixture via
    ``_torch_to_numpy()`` instead of using this fixture.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((8, small_bce_config.input_dim))
    targets = rng.integers(0, 2, size=(8, small_bce_config.output_classes)).astype(
        np.float64,
    )
    return X, targets


# ── XOR problem (matching convergence/problems/xor.py) ──────────────


@pytest.fixture()
def xor_config() -> OracleConfig:
    """XOR problem config: 8 modules, 1 output class, 2 inputs, 16 hidden."""
    return OracleConfig(
        input_dim=2,
        hidden_dim=16,
        output_classes=1,
        num_modules=8,
        mode="BCE",
        learning_rate=0.01,
    )


def _generate_xor_data(
    n: int = 200, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate XOR classification data (same seed as convergence/problems/xor.py)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float64)[:, np.newaxis]  # (N, 1)
    return X, y


@pytest.fixture()
def xor_data(xor_config: OracleConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """XOR data as torch tensors for cross-oracle tests."""
    X_np, y_np = _generate_xor_data()
    return (
        torch.from_numpy(X_np).to(torch.float64),
        torch.from_numpy(y_np).to(torch.float64),
    )


@pytest.fixture()
def xor_np_data() -> tuple[np.ndarray, np.ndarray]:
    """XOR data as numpy arrays for Oracle D tests."""
    return _generate_xor_data()


@pytest.fixture()
def xor_d_config(xor_config: OracleConfig) -> OracleDConfig:
    """OracleDConfig corresponding to the XOR OracleConfig."""
    return oracle_config_to_d_config(xor_config)
