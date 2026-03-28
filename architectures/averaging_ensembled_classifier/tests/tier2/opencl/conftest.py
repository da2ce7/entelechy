# tests/tier2/opencl/conftest.py
"""Session-scoped OpenCL fixtures for Tier 2 kernel correctness tests."""
from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("pyopencl")
import pyopencl as cl  # noqa: E402

# Device availability probe — must succeed for any test in this directory.
_has_device = False
try:
    _ctx = cl.create_some_context(interactive=False)
    _has_device = len(_ctx.devices) > 0
    del _ctx
except Exception:
    pass

if not _has_device:
    pytest.skip("No OpenCL device available", allow_module_level=True)

from src.backends.opencl.discovery import discover_hardware
from src.backends.opencl.kernel_bindings.binding_phase_2_learn_C import (
    AggregateLocalReduceBinding,
    AggregateRegisterReduceBinding,
    ClipIntermediateGradBinding,
)
from src.backends.opencl.kernel_bindings.dispatch_table import build_dispatch_table
from src.backends.opencl.kernel_compilation import load_and_compile_kernels_from_path
from src.backends.opencl.renderer import OpenCLPlanRenderer
from src.backends.opencl.type_mapping import build_compiler_flags
from src.shared.hardware_profile import HardwareProfile
from src.shared.precision_config import PrecisionConfig

# Locate kernels directory relative to test file
_ARCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
_KERNEL_DIR = os.path.join(_ARCH_ROOT, "kernels")


@pytest.fixture(scope="session")
def cl_context() -> cl.Context:
    """Session-scoped OpenCL context."""
    ctx = cl.create_some_context(interactive=False)
    return ctx


@pytest.fixture(scope="session")
def cl_device(cl_context: cl.Context) -> cl.Device:
    """Session-scoped OpenCL device."""
    return cl_context.devices[0]


@pytest.fixture(scope="session")
def cl_queue(cl_context: cl.Context, cl_device: cl.Device) -> cl.CommandQueue:
    """Session-scoped OpenCL command queue."""
    return cl.CommandQueue(cl_context, cl_device)


@pytest.fixture(scope="session")
def hardware_profile(cl_device: cl.Device) -> HardwareProfile:
    """Session-scoped hardware profile from device queries."""
    return discover_hardware(cl_device)


@pytest.fixture(scope="session")
def precision_fp32() -> PrecisionConfig:
    return PrecisionConfig.float32()


@pytest.fixture(scope="session")
def compiled_program_fp32(
    cl_context: cl.Context,
    cl_device: cl.Device,
    precision_fp32: PrecisionConfig,
    hardware_profile: HardwareProfile,
) -> cl.Program:
    """Session-scoped compiled OpenCL program (FP32)."""
    flags = build_compiler_flags(precision_fp32, hardware_profile, c_tile_size=16)
    return load_and_compile_kernels_from_path(
        cl_context, cl_device, flags, _KERNEL_DIR,
    )


@pytest.fixture(scope="session")
def dispatch_table() -> dict[str, Any]:
    """Session-scoped kernel binding dispatch table."""
    return build_dispatch_table()


@pytest.fixture(scope="session")
def renderer_fp32(
    cl_context: cl.Context,
    cl_queue: cl.CommandQueue,
    compiled_program_fp32: cl.Program,
    dispatch_table: dict[str, Any],
    hardware_profile: HardwareProfile,
) -> OpenCLPlanRenderer:
    """Session-scoped OpenCL plan renderer for FP32 tests."""
    renderer = OpenCLPlanRenderer(
        context=cl_context,
        queue=cl_queue,
        program=compiled_program_fp32,
        kernel_bindings=dispatch_table,
        hardware=hardware_profile,
    )
    renderer.set_reduction_bindings(
        register_reduce=AggregateRegisterReduceBinding(),
        local_reduce=AggregateLocalReduceBinding(),
        clip_intermediate=ClipIntermediateGradBinding(),
    )
    return renderer
