# src/backends/opencl/__init__.py
"""OpenCL backend implementation."""

from .renderer import OpenCLPlanRenderer as OpenCLPlanRenderer
from .discovery import discover_hardware as discover_hardware
from .buffer_allocator import OpenCLBufferAllocator as OpenCLBufferAllocator
from .retrieval import OpenCLRetrievalFuture as OpenCLRetrievalFuture
from .type_mapping import build_compiler_flags as build_compiler_flags
from .context import OpenCLContext as OpenCLContext, load_and_compile_kernels as load_and_compile_kernels, load_and_compile_kernels_from_path as load_and_compile_kernels_from_path
from .kernel_bindings.base import KernelBinding as KernelBinding
from .kernel_bindings.dispatch_table import build_dispatch_table as build_dispatch_table
