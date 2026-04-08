# tests/convergence/conftest.py
"""Convergence test infrastructure (ADR-028)."""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.engine import Engine
from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.optimizer_config import OptimizerConfig
from src.shared.parameter_space import ParameterSpace
from src.shared.precision_config import PrecisionConfig
from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy
from src.shared.stabilization_policy import StabilizationPolicy


# ---------------------------------------------------------------------------
# Backend availability (mirrors root conftest pattern — ADR-014, ADR-016)
# ---------------------------------------------------------------------------
def _load_build_config() -> dict[str, bool]:
    """Load the build manifest; return a dict of backend availability."""
    try:
        from src._build_config import BACKEND_CPU, BACKEND_OPENCL, BACKEND_VULKAN  # type: ignore[import-not-found]
        return {
            "cpu": BACKEND_CPU,
            "opencl": BACKEND_OPENCL,
            "vulkan": BACKEND_VULKAN,
        }
    except ImportError:
        raise RuntimeError(
            "_build_config.py not found. Run 'meson setup builddir' before testing."
        )


BUILD_CONFIG = _load_build_config()

AVAILABLE_BACKENDS: list[str] = [
    name for name, available in BUILD_CONFIG.items() if available
]


def pytest_collection_modifyitems(config, items):
    """Skip convergence tests based on backend availability and marker selection."""
    # 1. Skip all convergence tests if no backend is available
    if not AVAILABLE_BACKENDS:
        skip = pytest.mark.skip(reason="No backends available for convergence testing")
        for item in items:
            if "convergence" in item.keywords:
                item.add_marker(skip)
        return

    # 2. Skip convergence_full unless explicitly selected via -m
    markexpr = config.getoption("-m", default="")
    if "convergence_full" not in markexpr and markexpr.strip() != "convergence":
        skip_full = pytest.mark.skip(
            reason="convergence_full excluded from default runs; use -m convergence_full"
        )
        for item in items:
            if "convergence_full" in item.keywords and "convergence_fast" not in item.keywords:
                item.add_marker(skip_full)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reproducibility_seed():
    """Set deterministic random state for data shuffling/synthetic generation."""
    np.random.seed(42)
    yield


@pytest.fixture(params=AVAILABLE_BACKENDS)
def backend_name(request) -> str:
    """Yield each available backend name for cross-backend tests."""
    return request.param


def make_engine(
    *,
    input_dim: int,
    hidden_dim: int,
    output_classes: int,
    num_modules: int,
    mode: str,
    backend: str = "cpu",
    precision: PrecisionConfig | None = None,
    gradient_clip_threshold: float = 1.0,
    optimizer: OptimizerConfig | None = None,
) -> Engine:
    """Construct an Engine for convergence testing.

    Not a fixture — called directly by test functions to allow
    per-problem parameterization.
    """
    hw_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_classes=output_classes,
        num_modules=num_modules,
        simd_width=4,
        cache_line_bytes=64,
    )
    if precision is None or precision.storage_dtype == np.dtype("float32"):
        spec = ModelSpec.float32(**hw_kwargs)
    elif precision.storage_dtype == np.dtype("float16"):
        spec = ModelSpec.mixed_f16_f32(**hw_kwargs)
    else:
        spec = ModelSpec(precision=precision, **hw_kwargs)

    hardware = HardwareProfile(
        simd_width=4,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=None,
        global_mem_bytes=4 * 1024**3,
    )
    param_space = ParameterSpace(spec)

    if mode == "CCE":
        strategy = PlanCceStrategy()
    elif mode == "BCE":
        strategy = PlanBceStrategy()
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    policy = StabilizationPolicy(
        t_algorithmic=gradient_clip_threshold,
        lambda_=0.1,
        compute_fp_format_max=spec.precision.compute_fp_format_max,
    )

    return Engine(
        model_spec=spec,
        parameter_space=param_space,
        hardware_profile=hardware,
        precision=precision,
        backend=backend,
        strategy=strategy,
        policy=policy,
        optimizer=optimizer,
    )
