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
    with appropriate memory flags.  Manages the handle → cl.Buffer mapping.

    Lifetime: one allocator instance per renderer.  Per-batch buffers are
    freed after each render(); MODEL_STATE buffers persist across batches
    via the renderer's ``extract_buffers_by_role`` / ``inject_buffer`` flow.
    """

    def __init__(self, context: cl.Context, queue: cl.CommandQueue) -> None:
        self._context = context
        self._queue = queue
        self._buffers: dict[BufferHandle, cl.Buffer] = {}
        self._descriptors: dict[BufferHandle, BufferDescriptor] = {}
        self._internal_buffers: list[cl.Buffer] = []

    # ------------------------------------------------------------------
    # Plan-level allocation
    # ------------------------------------------------------------------

    def allocate_plan_buffers(
        self, descriptors: dict[BufferHandle, BufferDescriptor],
    ) -> None:
        """Allocate cl.Buffers for all descriptors not already registered.

        Buffers that were pre-registered via ``inject_buffer`` (e.g.
        persistent MODEL_STATE) are skipped.
        """
        for handle, desc in descriptors.items():
            if handle in self._buffers:
                continue
            flags = self._flags_for_role(desc.role)
            buf = cl.Buffer(self._context, flags, size=desc.size_bytes)
            # Zero-fill: GPU memory may contain residue from prior allocations.
            cl.enqueue_fill_buffer(
                self._queue, buf, np.zeros(1, dtype=np.uint8), 0, desc.size_bytes,
            )
            self._buffers[handle] = buf
            self._descriptors[handle] = desc

    # ------------------------------------------------------------------
    # Handle resolution
    # ------------------------------------------------------------------

    def get_buffer(self, handle: BufferHandle) -> cl.Buffer:
        """Retrieve the physical cl.Buffer for a plan-level handle."""
        try:
            return self._buffers[handle]
        except KeyError:
            raise KeyError(
                f"No cl.Buffer registered for handle {handle!r}.  "
                f"Known handles: {sorted(self._buffers.keys())}"
            ) from None

    def has_buffer(self, handle: BufferHandle) -> bool:
        """Check whether a handle is registered."""
        return handle in self._buffers

    # ------------------------------------------------------------------
    # Persistent-state management (called by the renderer)
    # ------------------------------------------------------------------

    def inject_buffer(
        self,
        handle: BufferHandle,
        buf: cl.Buffer,
        desc: BufferDescriptor,
    ) -> None:
        """Register a pre-existing cl.Buffer under a new plan handle.

        Used by the renderer to re-attach persistent MODEL_STATE buffers
        when a new plan's handles differ from the previous plan's.
        """
        self._buffers[handle] = buf
        self._descriptors[handle] = desc

    def extract_buffers_by_role(
        self, role: BufferRole,
    ) -> list[tuple[BufferHandle, cl.Buffer, BufferDescriptor]]:
        """Remove and return all buffers matching *role* without releasing them.

        The returned cl.Buffers are detached from this allocator — subsequent
        ``release_all`` will not destroy them.
        """
        extracted: list[tuple[BufferHandle, cl.Buffer, BufferDescriptor]] = []
        for h in list(self._buffers):
            desc = self._descriptors.get(h)
            if desc is not None and desc.role == role:
                extracted.append((h, self._buffers.pop(h), self._descriptors.pop(h)))
        return extracted

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    def upload(
        self,
        handle: BufferHandle,
        data: np.ndarray[Any, Any],
        wait_for: list[cl.Event] | None = None,
    ) -> cl.Event:
        """Enqueue an async host→device transfer."""
        return cl.enqueue_copy(
            self._queue, self._buffers[handle], data,
            wait_for=wait_for, is_blocking=False,
        )

    def enqueue_read(
        self,
        handle: BufferHandle,
        host_buffer: np.ndarray[Any, Any],
        wait_for: list[cl.Event] | None = None,
    ) -> cl.Event:
        """Enqueue an async device→host transfer into a pre-allocated array."""
        return cl.enqueue_copy(
            self._queue, host_buffer, self._buffers[handle],
            wait_for=wait_for, is_blocking=False,
        )

    # ------------------------------------------------------------------
    # Internal (renderer-owned) scratch buffers
    # ------------------------------------------------------------------

    def allocate_internal(self, size_bytes: int) -> cl.Buffer:
        """Allocate a renderer-internal buffer (reduction intermediates, etc.)."""
        buf = cl.Buffer(self._context, cl.mem_flags.READ_WRITE, size=size_bytes)
        self._internal_buffers.append(buf)
        return buf

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def release_non_persistent(self) -> None:
        """Release all buffers except MODEL_STATE."""
        for h in list(self._buffers):
            desc = self._descriptors.get(h)
            if desc is not None and desc.role == BufferRole.MODEL_STATE:
                continue
            self._buffers.pop(h).release()
            self._descriptors.pop(h, None)
        for buf in self._internal_buffers:
            buf.release()
        self._internal_buffers.clear()

    def release_all(self) -> None:
        """Release every buffer (including MODEL_STATE) and reset state."""
        for buf in self._buffers.values():
            buf.release()
        self._buffers.clear()
        self._descriptors.clear()
        for buf in self._internal_buffers:
            buf.release()
        self._internal_buffers.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flags_for_role(role: BufferRole) -> int:
        if role == BufferRole.BATCH_INPUT:
            return cl.mem_flags.READ_ONLY
        if role == BufferRole.BATCH_OUTPUT:
            return cl.mem_flags.WRITE_ONLY
        return cl.mem_flags.READ_WRITE
