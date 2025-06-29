# utility_kernels.py

"""
A Toolbox of Host-Orchestrated, Driver-Level Utility Patterns.

This module provides implementations for non-domain-specific utility operations
that are composed of standard, built-in driver commands (e.g., a sequence of
`cl.enqueue_copy_buffer` calls).

This module is distinct from `kernel_signatures/utility_signatures.py`. That
file defines the contracts for custom OpenCL kernels, whereas this module
provides higher-level Python functions that *orchestrate* lower-level
primitives without necessarily launching custom C code.

By isolating these host-side patterns, we keep the main orchestrator logic
clean and focused on the primary training algorithm.
"""
from typing import List, Optional

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from .launcher_infra import BufferHandle, BufferManager


def execute_gather(
    queue: cl.CommandQueue,
    buffer_mgr: BufferManager,
    source_handles: List[BufferHandle],
    dest_handle: BufferHandle,
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """
    Executes a "many-to-one" gather operation via driver-level copies.

    DEPRECATION WARNING: This host-side gather is architecturally inferior to
    the device-side gather implemented by the indirection-based `aggregate_*`
    kernels. It is retained as a reference implementation or for edge cases
    where device-side gather is not applicable. For performance-critical
    reductions, the `ReductionTreeExecutor` should always be preferred.

    Copies data from a list of many separate source buffers into one single,
    dense destination buffer.
    """
    if not source_handles:
        user_event = cl.UserEvent(queue.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

    events = []
    dest_buffer = buffer_mgr.get_cl_buffer(dest_handle)
    dest_offset_bytes = 0

    # Assume all partials are the same size, based on the first one.
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
    Executes a "one-to-many" scatter operation via driver-level copies.

    Copies contiguous chunks from a single dense source buffer into a list of
    many separate destination buffers.
    """
    if not dest_handles:
        user_event = cl.UserEvent(queue.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

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
