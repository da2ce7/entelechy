# tests/tier3/conftest.py
"""Tier 3 fixtures — oracle selection and multi-backend parity (ADR-016).

Tier 3 tests compare outputs across backends for the same input. The CPU
backend is the preferred oracle; when absent, all-pairs fallback is used.
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Any

import pytest

from tests.conftest import BUILD_CONFIG


# ---------------------------------------------------------------------------
# Oracle selection (ADR-016 Option C)
# ---------------------------------------------------------------------------

def _select_oracle() -> str | None:
    """Select the Tier 3 oracle backend.

    Returns the oracle backend name, or None if no oracle is available
    (triggering all-pairs fallback).
    """
    if BUILD_CONFIG.get("cpu", False):
        return "cpu"
    return None


def _get_available_backends() -> list[str]:
    """Return names of all available backends."""
    return [b for b in ("cpu", "opencl", "vulkan") if BUILD_CONFIG.get(b, False)]


def _get_tier3_pairs(config: Any) -> list[tuple[str, str]]:
    """Return (oracle_or_left, comparison) pairs for Tier 3 tests.

    Default: CPU-oracle vs each GPU backend.
    --all-pairs: adds GPU-vs-GPU pairs.
    """
    available = _get_available_backends()
    oracle = _select_oracle()

    pairs: list[tuple[str, str]] = []
    if oracle is not None:
        for b in available:
            if b != oracle:
                pairs.append((oracle, b))

    if config.getoption("--all-pairs", default=False) or oracle is None:
        for left, right in combinations(available, 2):
            if (left, right) not in pairs and (right, left) not in pairs:
                pairs.append((left, right))

    return pairs


# ---------------------------------------------------------------------------
# Renderer factory — per-backend initialization
# ---------------------------------------------------------------------------

_ARCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_KERNEL_DIR = os.path.join(_ARCH_ROOT, "kernels")


def _create_cpu_renderer() -> Any:
    """Create a CPUPlanRenderer."""
    from src.backends.cpu.renderer import CPUPlanRenderer
    return CPUPlanRenderer()


def _create_opencl_renderer() -> Any:
    """Create a fully initialized OpenCLPlanRenderer."""
    import pyopencl as cl

    from src.backends.opencl.discovery import discover_hardware
    from src.backends.opencl.kernel_bindings.binding_phase_2_learn_C import (
        AggregateLocalReduceBinding,
        AggregateRegisterReduceBinding,
        ClipIntermediateGradBinding,
        ReduceKFanInAndClipBinding,
    )
    from src.backends.opencl.kernel_bindings.dispatch_table import build_dispatch_table
    from src.backends.opencl.kernel_compilation import load_and_compile_kernels_from_path
    from src.backends.opencl.renderer import OpenCLPlanRenderer
    from src.backends.opencl.type_mapping import build_compiler_flags
    from src.shared.hardware_profile import HardwareProfile
    from src.shared.precision_config import PrecisionConfig

    ctx = cl.create_some_context(interactive=False)
    device = ctx.devices[0]
    queue = cl.CommandQueue(ctx, device)
    hw = discover_hardware(device)
    prec = PrecisionConfig.float32()
    flags = build_compiler_flags(prec, hw, c_tile_size=16)
    program = load_and_compile_kernels_from_path(ctx, device, flags, _KERNEL_DIR)
    dispatch_table = build_dispatch_table()

    renderer = OpenCLPlanRenderer(
        context=ctx, queue=queue, program=program,
        kernel_bindings=dispatch_table, hardware=hw,
    )
    renderer.set_reduction_bindings(
        register_reduce=AggregateRegisterReduceBinding(),
        local_reduce=AggregateLocalReduceBinding(),
        clip_intermediate=ClipIntermediateGradBinding(),
        k_fan_in=ReduceKFanInAndClipBinding(),
    )
    return renderer


def _create_vulkan_renderer() -> Any:
    """Create a fully initialized VulkanPlanRenderer."""
    from src.backends.vulkan.context import VulkanContext
    from src.backends.vulkan.renderer import VulkanPlanRenderer

    ctx = VulkanContext(enable_validation=False)
    return VulkanPlanRenderer(context=ctx)


_BACKEND_FACTORIES: dict[str, Any] = {
    "cpu": _create_cpu_renderer,
    "opencl": _create_opencl_renderer,
    "vulkan": _create_vulkan_renderer,
}


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def oracle_backend() -> str | None:
    """The name of the oracle backend, or None."""
    return _select_oracle()


@pytest.fixture(scope="session")
def available_backends() -> list[str]:
    """List of available backend names."""
    return _get_available_backends()


@pytest.fixture(scope="session")
def renderer_factory():
    """Factory that creates a renderer for a named backend.

    Created renderers are cached for the session to amortize initialization.
    Backends whose renderer cannot be created (e.g. missing module) raise
    pytest.skip at invocation time.
    """
    cache: dict[str, Any] = {}
    failed: dict[str, str] = {}

    def _get(backend_name: str) -> Any:
        if backend_name in failed:
            pytest.skip(f"Backend '{backend_name}' unavailable: {failed[backend_name]}")
        if backend_name not in cache:
            factory = _BACKEND_FACTORIES.get(backend_name)
            if factory is None:
                failed[backend_name] = "no renderer factory registered"
                pytest.skip(f"Backend '{backend_name}' unavailable: no renderer factory")
            try:
                cache[backend_name] = factory()
            except (ImportError, ModuleNotFoundError, OSError) as exc:
                failed[backend_name] = str(exc)
                pytest.skip(f"Backend '{backend_name}' unavailable: {exc}")
        return cache[backend_name]

    return _get


@pytest.fixture(scope="session")
def tier3_pairs(request) -> list[tuple[str, str]]:
    """Backend pairs for Tier 3 comparison."""
    return _get_tier3_pairs(request.config)
