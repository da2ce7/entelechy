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

from src.shared.model_spec import ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2
from src.shared.workload_primitives import TilingScheme
from src.shared.stabilization_policy import StabilizationPolicy


# ── Multi-precision configuration (ADR-020 §4.6, Phase 7C §15, ADR-024 §8) ──

PRECISION_CONFIGS = [
    pytest.param(PrecisionConfig.float32, id="fp32"),
    pytest.param(PrecisionConfig.mixed_f16_f32, id="mixed_f16_f32"),
]

FP64_PRECISION_CONFIGS = [
    pytest.param(PrecisionConfig.float64, id="float64"),
    pytest.param(PrecisionConfig.mixed_f32_f64_state, id="mixed_f32_f64_state"),
    pytest.param(PrecisionConfig.mixed_f16_f64_state, id="mixed_f16_f64_state"),
    pytest.param(PrecisionConfig.mixed_f32_f64, id="mixed_f32_f64"),
]

FP8_PRECISION_CONFIGS = [
    pytest.param(PrecisionConfig.fp8_e4m3, id="fp8_e4m3"),
    pytest.param(PrecisionConfig.fp8_e4m3_f16, id="fp8_e4m3_f16"),
    pytest.param(PrecisionConfig.fp8_e4m3_f64, id="fp8_e4m3_f64"),
    pytest.param(PrecisionConfig.fp8_e5m2, id="fp8_e5m2"),
    pytest.param(PrecisionConfig.fp8_e5m2_f16, id="fp8_e5m2_f16"),
    pytest.param(PrecisionConfig.fp8_e5m2_f64, id="fp8_e5m2_f64"),
]

# Direct-construction FP8 configs for mixed compute/state combinations
# not covered by factory classmethods (ADR-025 §2.3).
_f16_info = np.finfo(np.float16)
_f32_info = np.finfo(np.float32)


def _fp8_e4m3_f16_f64() -> PrecisionConfig:
    """E4M3 storage, FP16 compute, FP64 state (direct construction)."""
    return PrecisionConfig(
        storage_dtype=FP8_E4M3,
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,
        storage_mantissa_bits=3,
        compute_fp_format_max=float(_f16_info.max),
        compute_epsilon=float(_f16_info.eps),
    )


def _fp8_e4m3_f32_f64() -> PrecisionConfig:
    """E4M3 storage, FP32 compute, FP64 state (direct construction)."""
    return PrecisionConfig(
        storage_dtype=FP8_E4M3,
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,
        storage_mantissa_bits=3,
        compute_fp_format_max=float(_f32_info.max),
        compute_epsilon=float(_f32_info.eps),
    )


FP8_DIRECT_CONSTRUCTION_CONFIGS = [
    pytest.param(_fp8_e4m3_f16_f64, id="fp8_e4m3_f16_f64"),
    pytest.param(_fp8_e4m3_f32_f64, id="fp8_e4m3_f32_f64"),
]

ALL_PRECISION_CONFIGS = PRECISION_CONFIGS + FP64_PRECISION_CONFIGS + FP8_PRECISION_CONFIGS + FP8_DIRECT_CONSTRUCTION_CONFIGS

MODEL_SPEC_FACTORIES = [
    pytest.param(ModelSpec.float32, id="fp32"),
    pytest.param(ModelSpec.mixed_f16_f32, id="mixed_f16_f32"),
]

FP64_MODEL_SPEC_FACTORIES = [
    pytest.param(ModelSpec.float64, id="float64"),
    pytest.param(ModelSpec.mixed_f32_f64_state, id="mixed_f32_f64_state"),
    pytest.param(ModelSpec.mixed_f16_f64_state, id="mixed_f16_f64_state"),
    pytest.param(ModelSpec.mixed_f32_f64, id="mixed_f32_f64"),
]

ALL_MODEL_SPEC_FACTORIES = MODEL_SPEC_FACTORIES + FP64_MODEL_SPEC_FACTORIES


@pytest.fixture(params=PRECISION_CONFIGS)
def precision_config(request) -> PrecisionConfig:
    """Parameterized fixture yielding all three PrecisionConfig factories."""
    return request.param()


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
def iris_spec() -> ModelSpec:
    return ModelSpec.float32(**IRIS)


@pytest.fixture(params=MODEL_SPEC_FACTORIES)
def iris_spec_all_precisions(request) -> ModelSpec:
    """Iris model parameterized over all three precision configs."""
    return request.param(**IRIS)


@pytest.fixture
def iris_param_space(iris_spec: ModelSpec) -> ParameterSpace:
    return ParameterSpace(spec=iris_spec)


@pytest.fixture
def iris_tiling(iris_spec: ModelSpec) -> TilingScheme:
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
def hydra_spec() -> ModelSpec:
    return ModelSpec.float32(**HYDRA)


@pytest.fixture(params=MODEL_SPEC_FACTORIES)
def hydra_spec_all_precisions(request) -> ModelSpec:
    """Hydra model parameterized over all three precision configs."""
    return request.param(**HYDRA)


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
def lexicon_spec() -> ModelSpec:
    return ModelSpec.float32(**LEXICON)


@pytest.fixture(params=MODEL_SPEC_FACTORIES)
def lexicon_spec_all_precisions(request) -> ModelSpec:
    """Lexicon model parameterized over all three precision configs."""
    return request.param(**LEXICON)
