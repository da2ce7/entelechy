"""VulkanRetrievalFuture — fence-gated staging buffer readback (ADR-010).

Implements the RetrievalFuture Protocol for the Vulkan backend.
.wait() blocks on vkWaitForFences; .result() reads from the persistently
mapped staging buffer and strips padding to return the logical shape.
"""
from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from .buffer_allocator import VulkanStagingBuffer
from .context import VulkanContext

try:
    import vulkan as vk  # type: ignore[import-untyped]
except ImportError:
    pass

_UINT64_MAX = 2**64 - 1


class VulkanRetrievalFuture:
    """Fence-gated retrieval future for the Vulkan backend.

    After GPU work completes (fence signaled), reads from the
    persistently-mapped staging buffer, strips SIMD padding,
    and returns an unpadded numpy array.
    """

    def __init__(
        self,
        context: VulkanContext,
        fence: object,
        staging: VulkanStagingBuffer,
        node_id: str,
        logical_shape: tuple[int, ...],
        padded_size_bytes: int,
        dtype: np.dtype,
    ) -> None:
        self._ctx = context
        self._fence = fence
        self._staging = staging
        self._node_id = node_id
        self._logical_shape = logical_shape
        self._padded_size_bytes = padded_size_bytes
        self._dtype = dtype
        self._waited = False
        self._released = False

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        """Block until the GPU signals the fence. Idempotent."""
        if self._waited or self._released:
            return
        vk.vkWaitForFences(
            self._ctx.device, 1, [self._fence], vk.VK_TRUE, _UINT64_MAX
        )
        self._waited = True

    def result(self) -> NDArray[np.floating]:
        """Read from staging buffer, strip padding, return unpadded numpy array."""
        if self._released:
            raise RuntimeError(
                f"RetrievalFuture for '{self._node_id}' has been released"
            )
        self.wait()

        # Read padded data from the mapped staging memory
        padded_elements = self._padded_size_bytes // self._dtype.itemsize
        padded_array = np.empty(padded_elements, dtype=self._dtype)
        from vulkan._vulkan import ffi as _ffi  # noqa: PLC0415
        _ffi.memmove(
            _ffi.from_buffer(padded_array.data),
            self._staging.mapped_buf,
            self._padded_size_bytes,
        )

        # Strip padding to logical shape
        total_logical = 1
        for d in self._logical_shape:
            total_logical *= d
        flat = padded_array[:total_logical]
        return flat.reshape(self._logical_shape)

    def release(self) -> None:
        """Reset the fence and mark this future as consumed."""
        if self._released:
            return
        if self._waited and self._fence is not None:
            vk.vkResetFences(self._ctx.device, 1, [self._fence])
        self._released = True
