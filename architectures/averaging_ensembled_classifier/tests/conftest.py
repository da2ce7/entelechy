# tests/conftest.py
from __future__ import annotations

"""
Shared Fixtures and Helpers for the Integration Test Suite.

This module provides reusable pytest fixtures that compose the architecture's
foundational primitives into ready-to-use test configurations. Fixtures are
layered to mirror the system's own dependency hierarchy:

  PrecisionContext -> ModelSpec -> ParameterSpace -> (TilingScheme, StabilizationPolicy, ...)

Device-dependent fixtures (requiring a live OpenCL context) are guarded by
a `requires_opencl` marker so that the host-side integration tests can run
anywhere, while full-pipeline tests are skipped gracefully when no GPU is
available.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Safe, conditional import of pyopencl
# ---------------------------------------------------------------------------
try:
    import pyopencl as cl

    _has_opencl = True
    try:
        _ctx = cl.create_some_context(interactive=False)
        _has_opencl_device = len(_ctx.devices) > 0
        del _ctx
    except Exception:
        _has_opencl_device = False
except ImportError:
    _has_opencl = False
    _has_opencl_device = False

import os
import sys

# Ensure the src *directory* (not the package __init__) is importable.
# We add the architecture root so that ``import src.arch_primitives`` resolves,
# but we must prevent the ``src/__init__.py`` from firing its heavy
# ``pyopencl``-dependent re-exports.
_arch_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _arch_root not in sys.path:
    sys.path.insert(0, _arch_root)

# Pre-register the ``src`` package as a *namespace-only* package so that
# submodule imports never trigger ``src/__init__.py`` (which drags in pyopencl).
if "src" not in sys.modules:
    import types as _types

    _pkg = _types.ModuleType("src")
    _pkg.__path__ = [os.path.join(_arch_root, "src")]
    _pkg.__package__ = "src"
    sys.modules["src"] = _pkg

# ---------------------------------------------------------------------------
# Architecture imports (host-side only – no device needed)
# ---------------------------------------------------------------------------
from src.model_spec import Float16ModelSpec, Float32ModelSpec  # noqa: E402
from src.parameter_space import ParameterSpace  # noqa: E402
from src.stabilization_policy import StabilizationPolicy  # noqa: E402
from src.workload_primitives import TilingScheme  # noqa: E402

# ---------------------------------------------------------------------------
# Pytest markers
# ---------------------------------------------------------------------------
requires_opencl = pytest.mark.skipif(
    not (_has_opencl and _has_opencl_device),
    reason="No OpenCL device available",
)


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
def fp32_iris_spec() -> Float32ModelSpec:
    """A small FP32 model spec modeled after the Iris validation scenario."""
    return Float32ModelSpec(
        input_dim=IRIS_INPUT_DIM,
        hidden_dim=IRIS_HIDDEN_DIM,
        output_classes=IRIS_OUTPUT_CLASSES,
        num_modules=IRIS_NUM_MODULES,
        simd_width=IRIS_SIMD_WIDTH,
        cache_line_bytes=IRIS_CACHE_LINE_BYTES,
    )


@pytest.fixture
def fp16_iris_spec() -> Float16ModelSpec:
    """A small FP16 model spec modeled after the Iris validation scenario."""
    return Float16ModelSpec(
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
def fp32_hydra_spec() -> Float32ModelSpec:
    """A large FP32 model spec modeled after the Hydra validation scenario."""
    return Float32ModelSpec(
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
def fp32_lexicon_spec() -> Float32ModelSpec:
    """A model spec with massive output classes (Lexicon scenario)."""
    return Float32ModelSpec(
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
def iris_param_space(fp32_iris_spec: Float32ModelSpec) -> ParameterSpace:
    return ParameterSpace(spec=fp32_iris_spec)


@pytest.fixture
def iris_tiling(fp32_iris_spec: Float32ModelSpec) -> TilingScheme:
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
        fp_format_max=float(np.finfo(np.float32).max),
    )


@pytest.fixture
def fp16_stabilization_policy() -> StabilizationPolicy:
    """A policy configured for FP16's limited dynamic range (Rodeo scenario)."""
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        fp_format_max=float(np.finfo(np.float16).max),
    )
