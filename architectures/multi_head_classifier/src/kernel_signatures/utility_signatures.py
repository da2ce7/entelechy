# kernel_signatures/utility_signatures.py

"""
Concrete KernelSignature Implementations for General-Purpose Utility Kernels.

This file is the canonical location for any `KernelSignature` that provides a
generic, reusable data manipulation primitive and is not logically bound to a
specific phase of the core learning algorithm. Examples include generic memory
copies, transpositions, and other common linear algebra operations.

Placing these signatures here ensures the `kernel_signatures` package remains
the complete and exclusive source for all host-to-device interface contracts,
upholding the principle of logical cohesion.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..launcher_infra import BufferHandle, KernelSignature
from ..memory_layout import _pad_to_multiple


@dataclass(frozen=True)
class IdentityCopySignature(KernelSignature):
    """
    Signature for the `identity_copy` kernel, a generic element-wise copy utility.

    This serves as the N=1 base case for the reduction engine, preventing the
    unnecessary launch of more complex aggregation kernels when a simple copy
    will suffice.
    """

    in_ref: BufferHandle
    out_ref: BufferHandle

    # --- Derived Scalar Fields ---
    width: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives element count and validates buffer consistency."""
        super().__post_init__()
        in_shape, _ = self._buffer_mgr.get_spec(self.in_ref)
        out_shape, _ = self._buffer_mgr.get_spec(self.out_ref)

        in_count = np.uint32(np.prod(in_shape))
        out_count = np.uint32(np.prod(out_shape))

        assert in_count == out_count, (
            f"Buffer spec mismatch for IdentityCopy: Input buffer has {in_count} elements, "
            f"but output buffer has {out_count} elements."
        )

        object.__setattr__(self, "width", out_count)

    @property
    def kernel_name(self) -> str:
        return "identity_copy"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Dispatches one work-item per element to be copied."""
        return (int(self.width),), None

    def get_args(self) -> List:
        """Returns all 3 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.in_ref),
            self._buffer_mgr.get_cl_buffer(self.out_ref),
            self.width,
        ]


@dataclass(frozen=True)
class TransposeChunkSignature(KernelSignature):
    """(Node 12) Signature for the general-purpose, tiled `transpose_chunk` kernel."""

    c_tile_size: int
    local_mem_bank_padding: int
    scalar_size_bytes: int
    in_ref: BufferHandle
    out_ref: BufferHandle
    in_offset: np.uint32
    out_offset: np.uint32
    height: np.uint32
    width: np.uint32
    in_stride: np.uint32
    out_stride: np.uint32

    # --- Derived Scalar Fields ---
    in_total_element_count: np.uint32 = field(init=False)
    out_total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives total element counts from buffer specs for kernel-side validation."""
        super().__post_init__()
        in_shape, _ = self._buffer_mgr.get_spec(self.in_ref)
        out_shape, _ = self._buffer_mgr.get_spec(self.out_ref)
        object.__setattr__(self, "in_total_element_count", np.uint32(np.prod(in_shape)))
        object.__setattr__(self, "out_total_element_count", np.uint32(np.prod(out_shape)))

    @property
    def kernel_name(self) -> str:
        return "transpose_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Calculates global and local work sizes for a tiled algorithm."""
        global_size = (_pad_to_multiple(self.width, self.c_tile_size), _pad_to_multiple(self.height, self.c_tile_size))
        local_size = (self.c_tile_size, self.c_tile_size)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns all 11 arguments in exact contractual order."""
        local_mem_size = self.c_tile_size * (self.c_tile_size + self.local_mem_bank_padding) * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.in_ref),
            self._buffer_mgr.get_cl_buffer(self.out_ref),
            self.in_offset,
            self.out_offset,
            self.height,
            self.width,
            self.in_stride,
            self.out_stride,
            self.in_total_element_count,
            self.out_total_element_count,
        ]
