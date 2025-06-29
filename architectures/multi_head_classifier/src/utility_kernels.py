# utility_kernels.py

"""
A Toolbox of General-Purpose Data Manipulation Primitives.

This module provides implementations for non-domain-specific utility operations
that are used by higher-level orchestration components. It serves as a clear
example of the architectural distinction between:

1. KernelSignatures: Pure, 1:1 representations of custom OpenCL kernel launches
   (e.g., for a complex tiled transpose).
2. Orchestrated Primitives: Host-side functions that achieve a goal by dispatching
   a sequence of standard, built-in driver commands (e.g., using a loop of
   `cl.enqueue_copy_buffer` to implement a "gather" operation).

By isolating these utilities, we keep the main `phase_*` signature files clean
and focused on their specific part of the training algorithm.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from .launcher_infra import BufferHandle, KernelSignature, BufferManager
from .memory_layout import _pad_to_multiple


# === Tiled Matrix Transpose (Utility Kernel) ===


@dataclass(frozen=True)
class TransposeChunkSignature(KernelSignature):
    """Signature for the general-purpose, tiled `transpose_chunk` kernel."""

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
    # CORRECTED: These are now derived, not passed in, per Edict #2.
    in_total_element_count: np.uint32 = field(init=False)
    out_total_element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives total element counts from buffer specs for kernel-side validation."""
        super().__post_init__()  # Call parent __init__ if it exists
        in_shape, _ = self._buffer_mgr.get_spec(self.in_ref)
        out_shape, _ = self._buffer_mgr.get_spec(self.out_ref)
        object.__setattr__(self, "in_total_element_count", np.uint32(np.prod(in_shape)))
        object.__setattr__(self, "out_total_element_count", np.uint32(np.prod(out_shape)))

    @property
    def kernel_name(self) -> str:
        return "transpose_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
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


# === Gather/Scatter Primitives (Host-Orchestrated) ===


def execute_gather(
    queue: cl.CommandQueue,
    buffer_mgr: BufferManager,
    source_handles: List[BufferHandle],
    dest_handle: BufferHandle,
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """
    Executes a "many-to-one" gather operation.

    Copies data from a list of many separate source buffers into one single,
    dense destination buffer. This is achieved by orchestrating a sequence of
    standard driver-level memory copies.
    """
    if not source_handles:
        return cl.UserEvent(queue.context)  # Return a completed event if no work

    events = []
    dest_buffer = buffer_mgr.get_cl_buffer(dest_handle)
    dest_offset_bytes = 0

    first_spec, first_dtype = buffer_mgr.get_spec(source_handles[0])
    partial_byte_size = int(np.prod(first_spec) * first_dtype().itemsize)

    for src_handle in source_handles:
        src_buffer = buffer_mgr.get_cl_buffer(src_handle)
        copy_event = cl.enqueue_copy_buffer(
            queue,
            src=src_buffer,
            dst=dest_buffer,
            byte_count=partial_byte_size,
            src_offset=0,
            dst_offset=dest_offset_bytes,
            wait_for=wait_for,
        )
        events.append(copy_event)
        dest_offset_bytes += partial_byte_size

    return cl.WaitForEvents(events)


def execute_scatter(
    queue: cl.CommandQueue,
    buffer_mgr: BufferManager,
    source_handle: BufferHandle,
    dest_handles: List[BufferHandle],
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """
    Executes a "one-to-many" scatter operation.

    Copies contiguous chunks from a single dense source buffer into a list of
    many separate destination buffers.
    """
    if not dest_handles:
        return cl.UserEvent(queue.context)  # Return a completed event if no work

    events = []
    source_buffer = buffer_mgr.get_cl_buffer(source_handle)
    source_offset_bytes = 0

    for dest_handle in dest_handles:
        dest_buffer = buffer_mgr.get_cl_buffer(dest_handle)
        dest_spec, dest_dtype = buffer_mgr.get_spec(dest_handle)
        byte_count = int(np.prod(dest_spec) * dest_dtype().itemsize)

        copy_event = cl.enqueue_copy_buffer(
            queue,
            src=source_buffer,
            dst=dest_buffer,
            byte_count=byte_count,
            src_offset=source_offset_bytes,
            dst_offset=0,
            wait_for=wait_for,
        )
        events.append(copy_event)
        source_offset_bytes += byte_count

    return cl.WaitForEvents(events)
