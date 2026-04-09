# tests/oracle/conftest.py
"""Shared fixtures for oracle tests.

All tests in this package require PyTorch; the entire directory is
auto-skipped when torch is not installed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="Oracle tests require PyTorch")

from .autograd_oracle import AutogradOracle
from .convergence_oracle import ConvergenceOracle
from .faithful_oracle import FaithfulOracle
from .oracle_config import OracleConfig


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
    for name, param in oracle.named_params():
        data = param if isinstance(param, torch.Tensor) else param.data
        # Small Xavier-like init
        fan_in = data.shape[-1] if data.dim() >= 2 else data.shape[0]
        std = 1.0 / fan_in**0.5
        data.copy_(torch.randn(data.shape, dtype=torch.float64, generator=gen) * std)
