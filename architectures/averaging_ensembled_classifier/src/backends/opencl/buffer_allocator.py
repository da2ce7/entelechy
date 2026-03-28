# src/backends/opencl/buffer_allocator.py
"""Plan-level buffer allocation for OpenCL (ADR-009)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pyopencl as cl

from ...shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole


class OpenCLBufferAllocator:
    """Allocates and manages physical cl.Buffer objects for a single plan.

    Consumes BufferDescriptors from the plan and creates cl.Buffer objects
    with appropriate memory flags. Manages the handle -> cl.Buffer mapping.

    Lifetime: one allocator instance per renderer. Per-batch buffers are
    freed after each render(); MODEL_STATE buffers persist across batches.
    """

    def __init__(self, context: cl.Context, queue: cl.CommandQueue) -> None:
        self._context = context
        self._queue = queue
        self._buffers: dict[BufferHandle, cl.Buffer] = {}
        self._descriptors: dict[BufferHandle, BufferDescriptor] = {}
        # Renderer-internal buffers (reduction intermediates, scratch, etc.)
        self._internal_buffers: list[cl.Buffer] = []

    def allocate_plan_buffers(
        self, descriptors: dict[BufferHandle, BufferDescriptor]
    ) -> None:
        """Allocate cl.Buffers for all descriptors in the plan.

        MODEL_STATE buffers that already exist are kept (persisted across
        batches). All other roles are freshly allocated.
        """
        for handle, desc in descriptors.items():
            if handle in self._buffers:
                # MODEL_STATE buffer already allocated from a previous batch
                continue
            flags = self._flags_for_role(desc.role)
            buf = cl.Buffer(self._context, flags, size=desc.size_bytes)
            self._buffers[handle] = buf
            self._descriptors[handle] = desc

    def get_buffer(self, handle: BufferHandle) -> cl.Buffer:
        """Retrieve the physical cl.Buffer for a plan-level handle."""
        return self._buffers[handle]

    def upload(
        self,
        handle: BufferHandle,
        data: np.ndarray[Any, Any],
        wait_for: list[cl.Event] | None = None,
    ) -> cl.Event:
        """Enqueue an async host->device transfer."""
        buf = self._buffers[handle]
        event = cl.enqueue_copy(
            self._queue, buf, data,
            wait_for=wait_for, is_blocking=False,
        )
        return event

    def enqueue_read(
        self,
        handle: BufferHandle,
        host_buffer: np.ndarray[Any, Any],
        wait_for: list[cl.Event] | None = None,
    ) -> cl.Event:
        """Enqueue an async device->host transfer into a pre-allocated host buffer."""
        buf = self._buffers[handle]
        event = cl.enqueue_copy(
            self._queue, host_buffer, buf,
            wait_for=wait_for, is_blocking=False,
        )
        return event

    def allocate_internal(self, size_bytes: int) -> cl.Buffer:
        """Allocate a renderer-internal buffer (reduction intermediates, scratch)."""
        buf = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=size_bytes)
        self._internal_buffers.append(buf)
        return buf

    def release_non_persistent(self) -> None:
        """Release all buffers except MODEL_STATE (which persist across batches)."""
        handles_to_remove = [
            h for h, desc in self._descriptors.items()
            if desc.role != BufferRole.MODEL_STATE
        ]
        for h in handles_to_remove:
            buf = self._buffers.pop(h)
            buf.release()
            del self._descriptors[h]
        # Release all internal buffers
        for buf in self._internal_buffers:
            buf.release()
        self._internal_buffers.clear()

    @staticmethod
    def _flags_for_role(role: BufferRole) -> int:
        """Map BufferRole to cl.mem_flags."""
        if role == BufferRole.BATCH_INPUT:
            return cl.mem_flags.READ_ONLY
        return cl.mem_flags.READ_WRITE
