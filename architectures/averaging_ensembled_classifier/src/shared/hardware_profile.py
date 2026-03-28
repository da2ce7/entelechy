# src/shared/hardware_profile.py
"""Backend-neutral hardware capability profile (ADR-006)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of hardware capabilities, consumed by the plan builder.

    Each backend populates this from its native device queries.
    Field names describe Policy-tier consumption, not hardware origin.
    """
    cache_line_bytes: int
    max_local_memory_bytes: int
    max_work_group_size: int
    simd_width: int
    max_alloc_bytes: int
    global_memory_bytes: int
