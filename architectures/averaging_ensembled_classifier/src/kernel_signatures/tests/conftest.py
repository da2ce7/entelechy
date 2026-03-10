# kernel_signatures/tests/conftest.py

"""
Shared fixtures and mock infrastructure for kernel signature contract tests.

This module provides:
  - Device-free mocks for ``BufferManager`` and ``DiscoveredArchConstants``.
  - A session-scoped C header parser that extracts kernel definitions from
    ``kernels.cl.h`` into structured metadata.
  - A Python argument classifier for verifying type-category conformance.
"""

from __future__ import annotations

import os
import re
import sys
from enum import Enum, auto
from typing import Dict, List, Tuple, Type

import numpy as np
import pytest

# --- Path bootstrapping (same pattern as src/tests/conftest.py) ---
_arch_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
if _arch_root not in sys.path:
    sys.path.insert(0, _arch_root)

if "src" not in sys.modules:
    import types as _types

    _pkg = _types.ModuleType("src")
    _pkg.__path__ = [os.path.join(_arch_root, "src")]
    _pkg.__package__ = "src"
    sys.modules["src"] = _pkg

from src.launcher_infra import BufferHandle


# =========================================================================
# Parameter Category Taxonomy
# =========================================================================


class ParamCategory(Enum):
    """The four fundamental type categories shared by C and Python arguments."""

    LOCAL_MEM = auto()
    GLOBAL_BUFFER = auto()
    UINT_SCALAR = auto()
    FLOAT_SCALAR = auto()


# =========================================================================
# C Header Parser
# =========================================================================

HEADER_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        "kernels",
        "kernels.cl.h",
    )
)


def _classify_c_param(decl: str) -> ParamCategory:
    """Classify a single C parameter declaration into its type category."""
    decl = " ".join(decl.split())  # normalize whitespace
    if "__local" in decl and "*" in decl:
        return ParamCategory.LOCAL_MEM
    if "__global" in decl and "*" in decl:
        return ParamCategory.GLOBAL_BUFFER
    if re.match(r"^\s*uint\b", decl):
        return ParamCategory.UINT_SCALAR
    if "SCALAR_TYPE" in decl and "*" not in decl:
        return ParamCategory.FLOAT_SCALAR
    if re.match(r"^\s*int\b", decl) and "*" not in decl:
        return ParamCategory.UINT_SCALAR
    raise ValueError(f"Cannot classify C parameter: '{decl}'")


def parse_kernel_header(path: str) -> Dict[str, List[ParamCategory]]:
    """
    Parse ``kernels.cl.h`` and return ``{kernel_name: [ParamCategory, ...]}``.

    The parser finds all ``__kernel void name(`` declarations, extracts the
    full parameter block (handling nested parens), strips doxygen comments,
    and classifies each parameter by its type category.
    """
    with open(path) as f:
        text = f.read()

    kernel_re = re.compile(r"__kernel\s+(?:KERNEL_ATTR\s+)?void\s+(\w+)\s*\(")
    kernels: Dict[str, List[ParamCategory]] = {}

    for match in kernel_re.finditer(text):
        name = match.group(1)
        start = match.end()  # position right after the opening '('

        # Find the matching closing ')'
        depth = 1
        pos = start
        while depth > 0 and pos < len(text):
            if text[pos] == "(":
                depth += 1
            elif text[pos] == ")":
                depth -= 1
            pos += 1

        param_text = text[start : pos - 1]

        # Strip block comments (/** ... */ and /* ... */)
        param_text = re.sub(r"/\*.*?\*/", "", param_text, flags=re.DOTALL)
        # Strip line comments (// ...)
        param_text = re.sub(r"//[^\n]*", "", param_text)

        # Split by comma and classify
        raw_params = [p.strip() for p in param_text.split(",")]
        raw_params = [p for p in raw_params if p]

        kernels[name] = [_classify_c_param(p) for p in raw_params]

    return kernels


@pytest.fixture(scope="session")
def parsed_header() -> Dict[str, List[ParamCategory]]:
    """Session-scoped fixture: parses kernels.cl.h once and caches the result."""
    return parse_kernel_header(HEADER_PATH)


# =========================================================================
# Mock Infrastructure
# =========================================================================


class _SentinelBuffer:
    """Stands in for a ``cl.Buffer`` in device-free tests."""

    def __init__(self, handle_id: int):
        self.handle_id = handle_id

    def __repr__(self) -> str:
        return f"<SentinelBuffer:{self.handle_id}>"


class MockBufferManager:
    """
    A device-free stand-in for ``BufferManager``.

    Supports the two methods that signature ``__post_init__`` and ``get_args``
    rely on: ``get_spec()`` and ``get_cl_buffer()``.
    """

    def __init__(self) -> None:
        self._specs: Dict[BufferHandle, Tuple[Tuple[int, ...], Type[np.floating]]] = {}
        self._buffers: Dict[BufferHandle, _SentinelBuffer] = {}
        self._next_id: int = 0

    def make_handle(
        self,
        shape: Tuple[int, ...],
        dtype: Type[np.floating] = np.float32,
    ) -> BufferHandle:
        """Create a ``BufferHandle``, register its spec, and assign a sentinel buffer."""
        handle = BufferHandle(id=self._next_id)
        self._next_id += 1
        self._specs[handle] = (shape, dtype)
        self._buffers[handle] = _SentinelBuffer(handle.id)
        return handle

    def get_spec(self, ref: BufferHandle) -> Tuple[Tuple[int, ...], Type[np.floating]]:
        return self._specs[ref]

    def get_cl_buffer(self, ref: BufferHandle) -> _SentinelBuffer:
        return self._buffers[ref]


class MockArchConsts:
    """
    Provides the minimum ``DiscoveredArchConstants`` interface required by all
    signature classes.  No OpenCL device is needed.
    """

    def __init__(
        self,
        simd_width: int = 4,
        scalar_np_type: Type[np.floating] = np.float32,
        optimal_workgroup_size_1d_reduction: int = 64,
        optimal_rectangular_tile_dim1: int = 8,
    ) -> None:
        self.simd_width = simd_width
        self._scalar_np_type = scalar_np_type
        self.optimal_workgroup_size_1d_reduction = optimal_workgroup_size_1d_reduction
        self.optimal_rectangular_tile_dim1 = optimal_rectangular_tile_dim1
        # Additional constants that may be queried
        self.global_mem_cacheline_size = 64
        self.local_mem_size_bytes = 32768
        self.optimal_square_tile_dim = 8

    @property
    def SCALAR_NP_TYPE(self) -> Type[np.floating]:
        return self._scalar_np_type

    @property
    def SCALAR_C_TYPE_NAME(self) -> str:
        return "float" if self._scalar_np_type == np.float32 else "half"


@pytest.fixture
def bm() -> MockBufferManager:
    return MockBufferManager()


@pytest.fixture
def ac() -> MockArchConsts:
    return MockArchConsts()


# =========================================================================
# Standard Test Dimensions
# =========================================================================

# All values chosen as multiples of simd_width=4 to avoid padding-related
# validation failures in signature __post_init__ methods.
BATCH = 8
INPUT_PADDED = 4
HIDDEN_PADDED = 8
CLASSES_PADDED = 4
CLASSES_NATURAL = 3
HIDDEN_NATURAL = 7
MODULES = 2
MODULES_PER_CHUNK = 2
CLASSES_PER_CHUNK = 4
NUM_MODULE_CHUNKS = 1
NUM_CLASS_CHUNKS = 1
TILES = NUM_MODULE_CHUNKS * NUM_CLASS_CHUNKS  # 1
BATCH_CHUNKS = 2
PADDED_MODULES = 4
WG_SIZE_REDUCTION = 64


# =========================================================================
# Python Argument Classifier
# =========================================================================


def classify_python_arg(arg: object) -> ParamCategory:
    """
    Classify a Python argument returned by ``get_args()`` into a
    ``ParamCategory``, mirroring the C-side classification.
    """
    import pyopencl as cl

    if isinstance(arg, cl.LocalMemory):
        return ParamCategory.LOCAL_MEM
    if isinstance(arg, _SentinelBuffer):
        return ParamCategory.GLOBAL_BUFFER
    if arg is None:
        # None is used for NULL buffer pointers (e.g., unused per-item norm buffer)
        return ParamCategory.GLOBAL_BUFFER
    if isinstance(arg, np.uint32):
        return ParamCategory.UINT_SCALAR
    if isinstance(arg, (np.float32, np.float16)):
        return ParamCategory.FLOAT_SCALAR
    raise ValueError(
        f"Cannot classify Python arg: {arg!r} (type: {type(arg).__name__})"
    )


# =========================================================================
# Standard WorkTile Factory
# =========================================================================

from src.workload_primitives import WorkTile


def make_tile(
    flat_tile_index: int = 0,
    module_chunk_index: int = 0,
    class_chunk_index: int = 0,
    num_class_chunks: int = NUM_CLASS_CHUNKS,
    module_chunk_offset: int = 0,
    modules_per_chunk: int = MODULES_PER_CHUNK,
    class_chunk_offset: int = 0,
    classes_per_chunk: int = CLASSES_PER_CHUNK,
) -> WorkTile:
    return WorkTile(
        module_chunk_index=module_chunk_index,
        class_chunk_index=class_chunk_index,
        flat_tile_index=flat_tile_index,
        num_class_chunks=num_class_chunks,
        module_chunk_offset=module_chunk_offset,
        modules_per_chunk=modules_per_chunk,
        class_chunk_offset=class_chunk_offset,
        classes_per_chunk=classes_per_chunk,
    )
