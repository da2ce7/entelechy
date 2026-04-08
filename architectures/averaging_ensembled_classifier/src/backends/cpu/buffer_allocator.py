"""SIMD-aligned numpy buffer allocator for CPU backend (ADR-009)."""
from __future__ import annotations

import ctypes
from ctypes import POINTER, c_float
from typing import Literal

import numpy as np

from ...shared.buffer_lifecycle import BufferDescriptor, BufferHandle

c_float_p = POINTER(c_float)


class CPUBufferAllocator:
    """Allocates SIMD-aligned numpy arrays from BufferDescriptors.

    On CPU, device memory IS host memory. The critical requirement is
    SIMD alignment so simd_load/simd_store use aligned operations.

    Each buffer is allocated with the dtype matching its precision_role
    (ADR-023 §2.1):
      - "storage" → storage_dtype  (bandwidth lever)
      - "state"   → state_dtype    (optimizer stability lever)
      - "compute" → always float32 (CPU arithmetic invariant)
    """

    def __init__(
        self,
        simd_alignment: int,
        role_dtypes: dict[Literal["storage", "compute", "state"], np.dtype],
    ) -> None:
        self._alignment = simd_alignment
        self._role_dtypes = role_dtypes
        self._buffers: dict[BufferHandle, np.ndarray] = {}

    def allocate(self, descriptor: BufferDescriptor) -> None:
        """Allocate a SIMD-aligned numpy array for the given descriptor."""
        total_elements = 1
        for dim in descriptor.padded_shape:
            total_elements *= dim

        dtype = self._role_dtypes[descriptor.precision_role]
        buf = self._allocate_aligned(total_elements, dtype)
        self._buffers[descriptor.handle] = buf

    def get_buffer(self, handle: BufferHandle) -> np.ndarray:
        """Retrieve the numpy array for a given buffer handle."""
        return self._buffers[handle]

    def set_buffer(self, handle: BufferHandle, buf: np.ndarray) -> None:
        """Associate an externally-provided buffer with a handle."""
        self._buffers[handle] = buf

    def get_data_pointer(self, handle: BufferHandle) -> ctypes.c_void_p:
        """Return a ctypes pointer to the buffer data for FFI dispatch."""
        buf = self._buffers[handle]
        return ctypes.c_void_p(buf.ctypes.data)

    def release(self, handle: BufferHandle) -> None:
        """Release a buffer (allows garbage collection)."""
        self._buffers.pop(handle, None)

    def release_all(self) -> None:
        """Release all buffers."""
        self._buffers.clear()

    def _allocate_aligned(self, num_elements: int, dtype: np.dtype) -> np.ndarray:
        """Allocate a numpy array with guaranteed SIMD alignment.

        Modern numpy (≥1.20) typically aligns to 64 bytes by default.
        Verify and fall back to over-allocation + slicing if needed.
        """
        arr = np.zeros(num_elements, dtype=dtype)

        if self._alignment > 0 and arr.ctypes.data % self._alignment != 0:
            extra = self._alignment // arr.itemsize
            padded = np.zeros(num_elements + extra, dtype=dtype)
            offset = (
                (self._alignment - padded.ctypes.data % self._alignment)
                // arr.itemsize
            )
            arr = padded[offset:offset + num_elements]

        return arr
