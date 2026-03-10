# src/tests/conftest.py
"""Shared fixtures for src-level unit tests (no OpenCL required)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the src directory is importable with the same namespace-only trick
# used by the integration tests.
_arch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _arch_root not in sys.path:
    sys.path.insert(0, _arch_root)

if "src" not in sys.modules:
    import types as _types
    _pkg = _types.ModuleType("src")
    _pkg.__path__ = [os.path.join(_arch_root, "src")]
    _pkg.__package__ = "src"
    sys.modules["src"] = _pkg

from src.model_spec import Float32ModelSpec, Float16ModelSpec
from src.parameter_space import ParameterSpace
from src.workload_primitives import TilingScheme
from src.stabilization_policy import StabilizationPolicy


# ── Canonical small model (Iris-like) ──

IRIS = dict(
    input_dim=4,
    hidden_dim=32,
    output_classes=3,
    num_modules=8,
    simd_width=4,
    cache_line_bytes=64,
)


@pytest.fixture
def iris_spec() -> Float32ModelSpec:
    return Float32ModelSpec(**IRIS)


@pytest.fixture
def iris_param_space(iris_spec: Float32ModelSpec) -> ParameterSpace:
    return ParameterSpace(spec=iris_spec)


@pytest.fixture
def iris_tiling(iris_spec: Float32ModelSpec) -> TilingScheme:
    s = iris_spec
    return TilingScheme(
        num_module_chunks=(s.num_modules + 15) // 16,
        num_class_chunks=(s.output_classes + 15) // 16,
        total_modules=s.num_modules,
        total_classes=s.output_classes,
    )


# ── Stress model (Hydra) ──

HYDRA = dict(
    input_dim=16,
    hidden_dim=64,
    output_classes=10,
    num_modules=256,
    simd_width=4,
    cache_line_bytes=64,
)


@pytest.fixture
def hydra_spec() -> Float32ModelSpec:
    return Float32ModelSpec(**HYDRA)


# ── Stress model (Lexicon — huge output_classes) ──

LEXICON = dict(
    input_dim=8,
    hidden_dim=32,
    output_classes=10_000,
    num_modules=4,
    simd_width=4,
    cache_line_bytes=64,
)


@pytest.fixture
def lexicon_spec() -> Float32ModelSpec:
    return Float32ModelSpec(**LEXICON)
