# src/backends/opencl/__init__.py
"""OpenCL backend implementation."""

from .renderer import OpenCLPlanRenderer
from .discovery import discover_hardware
from .buffer_allocator import OpenCLBufferAllocator
from .retrieval import OpenCLRetrievalFuture
from .type_mapping import build_compiler_flags, numpy_dtype_to_cl_type_name
from .context import OpenCLContext
from .kernel_compilation import load_and_compile_kernels, load_and_compile_kernels_from_path
from .kernel_bindings.base import KernelBinding
from .kernel_bindings.dispatch_table import build_dispatch_table
