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
    # PREFERRED_WORK_GROUP_SIZE_MULTIPLE is per-kernel in OpenCL <3.0, but
    # PyOpenCL may expose it at device level on some drivers. Guard against
    # LogicError on drivers that only support the per-kernel query.
    try:
        simd_width = int(
            cast(Any, device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE))
        )
    except (cl.LogicError, AttributeError):
        # Fallback: conservative default based on vendor heuristics.
        # 32 is safe for NVIDIA/AMD; Intel iGPU may prefer 16. A more
        # robust approach would query via a trivial compiled kernel.
        simd_width = 32

    cache_line_bytes: int = int(
        cast(Any, device.get_info(cl.device_info.GLOBAL_MEM_CACHELINE_SIZE))
    )
    if cache_line_bytes == 0:
        cache_line_bytes = 64  # Conservative fallback

    max_work_group_size: int = int(
        cast(Any, device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
    )
    local_mem_size: int = int(
        cast(Any, device.get_info(cl.device_info.LOCAL_MEM_SIZE))
    )
    global_mem_bytes: int = int(
        cast(Any, device.get_info(cl.device_info.GLOBAL_MEM_SIZE))
    )

    # max_reduce_fan_in: the maximum number of partials a single work-group
    # can reduce.  Reduction kernels allocate a single flat local tile of
    # get_local_size(0) × sizeof(element) — NOT a ping-pong pair.
    # Use float32 (4 bytes) as the conservative element size.
    #
    # NOTE: For FP64 compute (COMPUTE_TYPE = double), local-reduce kernels
    # allocate get_local_size(0) × 8 bytes, meaning the effective fan-in is
    # halved.  Since HardwareProfile is created before precision config is
    # known, the Policy tier MUST down-adjust when compute_dtype.itemsize > 4:
    #
    #   effective_max_fan_in = min(
    #       hardware.max_reduce_fan_in,
    #       hardware.max_local_mem_bytes // precision.compute_dtype.itemsize,
    #   )
    element_size = 4
    max_fan_in_from_local = local_mem_size // element_size
    max_reduce_fan_in = min(max_work_group_size, max_fan_in_from_local)

    return HardwareProfile(
        simd_width=simd_width,
        cache_line_bytes=cache_line_bytes,
        max_reduce_fan_in=max_reduce_fan_in,
        max_local_mem_bytes=local_mem_size,
        global_mem_bytes=global_mem_bytes,
    )
