# tests/conftest.py
from __future__ import annotations

"""
Shared Fixtures and Helpers for the Test Suite.

Provides reusable pytest fixtures for model configurations. Backend
availability is driven exclusively by _build_config (ADR-014, ADR-016).
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure src submodules are importable
# ---------------------------------------------------------------------------
_arch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _arch_root not in sys.path:
    sys.path.insert(0, _arch_root)

# Pre-register the ``src`` package as a *namespace-only* package so that
# submodule imports never trigger ``src/__init__.py``.
if "src" not in sys.modules:
    import types as _types

    _pkg = _types.ModuleType("src")
    _pkg.__path__ = [os.path.join(_arch_root, "src")]
    _pkg.__package__ = "src"
    sys.modules["src"] = _pkg

# Make Meson-generated files (e.g. _build_config.py) discoverable as src.*
_builddir = os.path.join(_arch_root, "builddir")
if os.path.isdir(_builddir) and _builddir not in sys.modules["src"].__path__:
    sys.modules["src"].__path__.append(_builddir)

# ---------------------------------------------------------------------------
# Architecture imports (host-side only – no device needed)
# ---------------------------------------------------------------------------
from src.shared.model_spec import ModelSpec  # noqa: E402
from src.shared.parameter_space import ParameterSpace  # noqa: E402
from src.shared.precision_config import FP8_DTYPES  # noqa: E402
from src.shared.stabilization_policy import StabilizationPolicy  # noqa: E402
from src.shared.workload_primitives import TilingScheme  # noqa: E402


# ---------------------------------------------------------------------------
# _build_config-driven backend availability (ADR-014, ADR-016)
# ---------------------------------------------------------------------------
def _load_build_config() -> dict[str, bool]:
    """Load the build manifest; return a dict of backend availability."""
    try:
        from src._build_config import BACKEND_OPENCL, BACKEND_VULKAN, BACKEND_CPU  # type: ignore[import-not-found]
        return {
            "opencl": BACKEND_OPENCL,
            "vulkan": BACKEND_VULKAN,
            "cpu": BACKEND_CPU,
        }
    except ImportError:
        raise RuntimeError(
            "_build_config.py not found. Run 'meson setup builddir' before testing."
        )


BUILD_CONFIG = _load_build_config()


# ---------------------------------------------------------------------------
# pytest hooks — collection modifier and CLI options (ADR-016)
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--all-pairs",
        action="store_true",
        default=False,
        help="Run Tier 3 parity tests with all-pairs comparison (GPU-vs-GPU in addition to oracle)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests whose backend requirements are not met (ADR-016)."""
    for item in items:
        # Tier 2 / per-backend: skip if backend unavailable
        for backend in ("opencl", "vulkan", "cpu"):
            if backend in item.keywords and not BUILD_CONFIG.get(backend, False):
                item.add_marker(pytest.mark.skip(
                    reason=f"Backend '{backend}' not available (_build_config)",
                ))

        # Tier 3: skip parity tests if < 2 backends available
        if "tier3" in item.keywords:
            available = sum(BUILD_CONFIG.values())
            if available < 2:
                item.add_marker(pytest.mark.skip(
                    reason=f"Tier 3 requires >= 2 backends ({available} available)",
                ))

        # FP8: skip backend tests until FP8 kernel support lands (Phase 9B/9C/9D)
        if hasattr(item, "callspec"):
            params = item.callspec.params
            precision = params.get("precision") or params.get("precision_config")
            if precision is not None and callable(precision):
                try:
                    precision = precision()
                except Exception:
                    continue
            if (
                precision is not None
                and getattr(precision, "storage_dtype", None) in FP8_DTYPES
            ):
                backend_name = params.get("backend") or params.get("backend_name")
                if backend_name is not None or any(
                    b in item.keywords for b in ("opencl", "vulkan", "cpu")
                ):
                    item.add_marker(pytest.mark.skip(
                        reason="FP8 backend support not yet implemented (Phase 9B/9C/9D)",
                    ))


# =========================================================================
# Canonical "Iris-like" Small Model Configuration
# =========================================================================

IRIS_INPUT_DIM = 4
IRIS_HIDDEN_DIM = 32
IRIS_OUTPUT_CLASSES = 3
IRIS_NUM_MODULES = 8
IRIS_BATCH_SIZE = 150
IRIS_SIMD_WIDTH = 4
IRIS_CACHE_LINE_BYTES = 64


@pytest.fixture
def fp32_iris_spec() -> ModelSpec:
    """A small FP32 model spec modeled after the Iris validation scenario."""
    return ModelSpec.float32(
        input_dim=IRIS_INPUT_DIM,
        hidden_dim=IRIS_HIDDEN_DIM,
        output_classes=IRIS_OUTPUT_CLASSES,
        num_modules=IRIS_NUM_MODULES,
        simd_width=IRIS_SIMD_WIDTH,
        cache_line_bytes=IRIS_CACHE_LINE_BYTES,
    )


@pytest.fixture
def fp16_iris_spec() -> ModelSpec:
    """A small FP16-storage / FP32-compute model spec (Iris scenario)."""
    return ModelSpec.mixed_f16_f32(
        input_dim=IRIS_INPUT_DIM,
        hidden_dim=IRIS_HIDDEN_DIM,
        output_classes=IRIS_OUTPUT_CLASSES,
        num_modules=IRIS_NUM_MODULES,
        simd_width=IRIS_SIMD_WIDTH,
        cache_line_bytes=IRIS_CACHE_LINE_BYTES,
    )


@pytest.fixture
def mixed_iris_spec() -> ModelSpec:
    """A small mixed FP16-storage / FP32-compute model spec (Iris scenario)."""
    return ModelSpec.mixed_f16_f32(
        input_dim=IRIS_INPUT_DIM,
        hidden_dim=IRIS_HIDDEN_DIM,
        output_classes=IRIS_OUTPUT_CLASSES,
        num_modules=IRIS_NUM_MODULES,
        simd_width=IRIS_SIMD_WIDTH,
        cache_line_bytes=IRIS_CACHE_LINE_BYTES,
    )


# =========================================================================
# Stress-Test "Hydra" Model Configuration (Massive num_heads)
# =========================================================================

HYDRA_INPUT_DIM = 16
HYDRA_HIDDEN_DIM = 64
HYDRA_OUTPUT_CLASSES = 10
HYDRA_NUM_MODULES = 256
HYDRA_BATCH_SIZE = 32


@pytest.fixture
def fp32_hydra_spec() -> ModelSpec:
    """A large FP32 model spec modeled after the Hydra validation scenario."""
    return ModelSpec.float32(
        input_dim=HYDRA_INPUT_DIM,
        hidden_dim=HYDRA_HIDDEN_DIM,
        output_classes=HYDRA_OUTPUT_CLASSES,
        num_modules=HYDRA_NUM_MODULES,
        simd_width=IRIS_SIMD_WIDTH,
        cache_line_bytes=IRIS_CACHE_LINE_BYTES,
    )


# =========================================================================
# Stress-Test "Lexicon" Model Configuration (Massive output_classes)
# =========================================================================

LEXICON_INPUT_DIM = 8
LEXICON_HIDDEN_DIM = 32
LEXICON_OUTPUT_CLASSES = 10_000
LEXICON_NUM_MODULES = 4


@pytest.fixture
def fp32_lexicon_spec() -> ModelSpec:
    """A model spec with massive output classes (Lexicon scenario)."""
    return ModelSpec.float32(
        input_dim=LEXICON_INPUT_DIM,
        hidden_dim=LEXICON_HIDDEN_DIM,
        output_classes=LEXICON_OUTPUT_CLASSES,
        num_modules=LEXICON_NUM_MODULES,
        simd_width=IRIS_SIMD_WIDTH,
        cache_line_bytes=IRIS_CACHE_LINE_BYTES,
    )


# =========================================================================
# Derived Higher-Level Fixtures
# =========================================================================


@pytest.fixture
def iris_param_space(fp32_iris_spec: ModelSpec) -> ParameterSpace:
    return ParameterSpace(spec=fp32_iris_spec)


@pytest.fixture
def iris_tiling(fp32_iris_spec: ModelSpec) -> TilingScheme:
    spec = fp32_iris_spec
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


@pytest.fixture
def default_stabilization_policy() -> StabilizationPolicy:
    """The default policy with a moderate algorithmic threshold."""
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        compute_fp_format_max=float(np.finfo(np.float32).max),
    )


@pytest.fixture
def fp16_stabilization_policy() -> StabilizationPolicy:
    """A policy configured for FP16's limited dynamic range (Rodeo scenario)."""
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        compute_fp_format_max=float(np.finfo(np.float16).max),
    )
