# tests/tier1/conftest.py
"""Tier 1 test fixtures — pure Python, no backend required."""
import pytest

from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.precision_config import PrecisionConfig
from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy
from src.shared.stabilization_policy import StabilizationPolicy
from src.shared.workload_primitives import TilingScheme


@pytest.fixture
def precision_fp32():
    return PrecisionConfig.float32()


@pytest.fixture
def precision_fp16():
    return PrecisionConfig.float16()


@pytest.fixture
def hardware():
    return HardwareProfile(
        simd_width=16,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=65536,
        global_mem_bytes=4 * 1024**3,
    )


@pytest.fixture
def model_spec(precision_fp32: PrecisionConfig) -> ModelSpec:
    return ModelSpec(
        precision=precision_fp32,
        input_dim=128,
        hidden_dim=64,
        output_classes=10,
        num_modules=4,
        simd_width=16,
        cache_line_bytes=64,
    )


@pytest.fixture
def small_spec(precision_fp32: PrecisionConfig) -> ModelSpec:
    """Minimal spec for simple structural tests."""
    return ModelSpec(
        precision=precision_fp32,
        input_dim=4,
        hidden_dim=16,
        output_classes=3,
        num_modules=1,
        simd_width=4,
        cache_line_bytes=64,
    )


@pytest.fixture
def policy(precision_fp32: PrecisionConfig) -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=0.1,
        fp_format_max=precision_fp32.fp_format_max,
    )


@pytest.fixture
def cce_strategy():
    return PlanCceStrategy()


@pytest.fixture
def bce_strategy():
    return PlanBceStrategy()


@pytest.fixture
def tiling(model_spec: ModelSpec) -> TilingScheme:
    chunk = 16
    return TilingScheme(
        num_module_chunks=max(1, (model_spec.num_modules + chunk - 1) // chunk),
        num_class_chunks=max(1, (model_spec.output_classes + chunk - 1) // chunk),
        total_modules=model_spec.num_modules,
        total_classes=model_spec.output_classes,
    )
