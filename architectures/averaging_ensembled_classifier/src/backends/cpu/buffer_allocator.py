"""SIMD-aligned numpy buffer allocator for CPU backend (ADR-009)."""
from __future__ import annotations

import ctypes
from ctypes import POINTER, c_float

import numpy as np

from ...shared.buffer_lifecycle import BufferDescriptor, BufferHandle

c_float_p = POINTER(c_float)


class CPUBufferAllocator:
    """Allocates SIMD-aligned numpy arrays from BufferDescriptors.

    On CPU, device memory IS host memory. The critical requirement is
    SIMD alignment so simd_load/simd_store use aligned operations.
    """

    def __init__(self, simd_alignment: int, dtype: np.dtype) -> None:
        self._alignment = simd_alignment
        self._dtype = dtype
        self._buffers: dict[BufferHandle, np.ndarray] = {}

    def allocate(self, descriptor: BufferDescriptor) -> None:
        """Allocate a SIMD-aligned numpy array for the given descriptor."""
        total_elements = 1
        for dim in descriptor.padded_shape:
            total_elements *= dim

        buf = self._allocate_aligned(total_elements)
        self._buffers[descriptor.handle] = buf

    def get_buffer(self, handle: BufferHandle) -> np.ndarray:
        """Retrieve the numpy array for a given buffer handle."""
        return self._buffers[handle]

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

    def _allocate_aligned(self, num_elements: int) -> np.ndarray:
        """Allocate a numpy array with guaranteed SIMD alignment.

        Modern numpy (≥1.20) typically aligns to 64 bytes by default.
        Verify and fall back to over-allocation + slicing if needed.
        """
        arr = np.zeros(num_elements, dtype=self._dtype)

        if self._alignment > 0 and arr.ctypes.data % self._alignment != 0:
            extra = self._alignment // arr.itemsize
            padded = np.zeros(num_elements + extra, dtype=self._dtype)
            offset = (
                (self._alignment - padded.ctypes.data % self._alignment)
                // arr.itemsize
            )
            arr = padded[offset:offset + num_elements]

        return arr
