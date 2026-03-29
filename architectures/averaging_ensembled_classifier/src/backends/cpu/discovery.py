"""CPU hardware discovery — HardwareProfile population (ADR-006)."""
from __future__ import annotations

import os
import platform
import subprocess
from ctypes import CDLL

from ...shared.hardware_profile import HardwareProfile


def discover_hardware(lib: CDLL | None = None) -> HardwareProfile:
    """Populate HardwareProfile from CPU hardware characteristics.

    If lib is provided (loaded CDLL), queries SIMD_WIDTH from the library.
    Otherwise falls back to platform-based detection.
    """
    return HardwareProfile(
        simd_width=_detect_simd_width(lib),
        cache_line_bytes=_detect_cache_line_bytes(),
        max_reduce_fan_in=256,
        max_local_mem_bytes=None,  # CPU has no local memory concept
        global_mem_bytes=_detect_system_memory(),
    )


def detect_thread_count() -> int:
    """Determine the optimal thread pool size.

    Prefers sched_getaffinity (respects cgroup/taskset) over cpu_count.
    """
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _detect_simd_width(lib: CDLL | None = None) -> int:
    """Query SIMD width from the compiled library or fall back to 1."""
    if lib is not None:
        try:
            return int(lib.get_simd_width())
        except (AttributeError, OSError):
            pass
    return 1


def _detect_cache_line_bytes() -> int:
    """Detect cache line size from OS. Default 64 for x86-64."""
    system = platform.system()
    if system == "Linux":
        try:
            path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
            with open(path) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            pass
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.cachelinesize"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return 64  # Safe default for x86-64


def _detect_system_memory() -> int:
    """Detect total physical memory in bytes."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, ValueError, OSError):
        pass
    return 4 * 1024 * 1024 * 1024  # 4 GiB fallback
