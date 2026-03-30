# src/backends/opencl/discovery.py
"""OpenCL device discovery and HardwareProfile population (ADR-006)."""
from __future__ import annotations

from typing import Any, cast

import pyopencl as cl

from ...shared.hardware_profile import HardwareProfile


def discover_hardware(device: cl.Device) -> HardwareProfile:
    """Populate a HardwareProfile from an OpenCL device's capabilities.

    Computes max_reduce_fan_in from the device's work-group and local memory
    constraints — this is the Orchestration-tier derivation that the Policy
    tier consumes as an abstract budget.
    """
    simd_width: int = int(cast(Any, device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)))

    cache_line_bytes: int = int(cast(Any, device.get_info(cl.device_info.GLOBAL_MEM_CACHELINE_SIZE)))
    if cache_line_bytes == 0:
        cache_line_bytes = 64  # Conservative fallback

    max_work_group_size: int = int(cast(Any, device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)))
    local_mem_size: int = int(cast(Any, device.get_info(cl.device_info.LOCAL_MEM_SIZE)))
    global_mem_bytes: int = int(cast(Any, device.get_info(cl.device_info.GLOBAL_MEM_SIZE)))

    # max_reduce_fan_in: the maximum number of partials a single workgroup
    # can reduce using local memory ping-pong. Constrained by both work-group
    # size and local memory (2 * K * element_size for ping-pong).
    # Use float32 (4 bytes) as the conservative element size.
    element_size = 4
    max_fan_in_from_local = local_mem_size // (2 * element_size)
    max_reduce_fan_in = min(max_work_group_size, max_fan_in_from_local)

    return HardwareProfile(
        simd_width=simd_width,
        cache_line_bytes=cache_line_bytes,
        max_reduce_fan_in=max_reduce_fan_in,
        max_local_mem_bytes=local_mem_size,
        global_mem_bytes=global_mem_bytes,
    )
