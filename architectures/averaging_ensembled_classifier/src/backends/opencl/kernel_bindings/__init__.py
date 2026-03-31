# src/backends/opencl/kernel_bindings/__init__.py
"""OpenCL KernelBinding dispatch adapters (ADR-007).

This package provides KernelBinding implementations that translate
plan-level parameters into concrete OpenCL kernel arguments and
dispatch dimensions. Each binding corresponds to a kernel declared
in kernels.cl.h.
"""

from .base import KernelBinding
from .dispatch_table import build_dispatch_table

__all__ = [
    "KernelBinding",
    "build_dispatch_table",
]
