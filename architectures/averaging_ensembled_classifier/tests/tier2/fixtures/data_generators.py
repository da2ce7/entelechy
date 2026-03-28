# tests/tier2/fixtures/data_generators.py
"""Deterministic input data generators for Tier 2 tests.

All generators accept a numpy.random.Generator for reproducibility.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def make_rng(seed: int = 42) -> np.random.Generator:
    """Create a deterministic RNG with a fixed seed."""
    return np.random.default_rng(np.random.PCG64(seed))


def make_input_data(
    rng: np.random.Generator,
    batch_size: int,
    input_features: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> NDArray[np.floating[Any]]:
    """Generate random input data in [-1, 1]."""
    return rng.standard_normal((batch_size, input_features)).astype(dtype)


def make_weights(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
    scale: float = 0.1,
) -> NDArray[np.floating[Any]]:
    """Generate random weight matrix."""
    return (rng.standard_normal((rows, cols)) * scale).astype(dtype)


def make_biases(
    rng: np.random.Generator,
    size: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> NDArray[np.floating[Any]]:
    """Generate random bias vector initialized near zero."""
    return (rng.standard_normal(size) * 0.01).astype(dtype)


def make_sample_mask(
    rng: np.random.Generator,
    batch_size: int,
    active_fraction: float = 0.8,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> NDArray[np.floating[Any]]:
    """Generate a binary sample mask (1.0 = active, 0.0 = masked)."""
    mask = (rng.random(batch_size) < active_fraction).astype(dtype)
    return mask


def make_targets_cce(
    rng: np.random.Generator,
    batch_size: int,
    num_classes: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> NDArray[np.floating[Any]]:
    """Generate one-hot CCE targets."""
    indices = rng.integers(0, num_classes, size=batch_size)
    targets = np.zeros((batch_size, num_classes), dtype=dtype)
    targets[np.arange(batch_size), indices] = 1.0
    return targets


def make_targets_bce(
    rng: np.random.Generator,
    batch_size: int,
    num_classes: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
    positive_rate: float = 0.3,
) -> NDArray[np.floating[Any]]:
    """Generate multi-label BCE targets."""
    return (rng.random((batch_size, num_classes)) < positive_rate).astype(dtype)


def make_temperatures(
    rng: np.random.Generator,
    num_modules: int,
    dtype: np.dtype[Any] = np.dtype(np.float32),
    low: float = 0.5,
    high: float = 2.0,
) -> NDArray[np.floating[Any]]:
    """Generate temperature scalars for each module."""
    return rng.uniform(low, high, size=num_modules).astype(dtype)


def make_adam_state(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] = np.dtype(np.float32),
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Generate random Adam optimizer m1 and m2 state arrays."""
    m1 = (rng.standard_normal(shape) * 0.01).astype(dtype)
    m2 = np.abs(rng.standard_normal(shape) * 0.01).astype(dtype)
    return m1, m2
