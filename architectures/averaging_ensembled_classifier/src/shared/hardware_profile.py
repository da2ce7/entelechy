# src/shared/hardware_profile.py
"""Backend-neutral hardware capability profile (ADR-006, DESIGN.md §3.4)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of hardware capabilities (ADR-006, DESIGN.md §3.4).

    Each backend populates this from its native device queries.
    Field names describe Policy-tier consumption role, not hardware origin.
    """
    simd_width: int  # SIMD lane count in elements of COMPUTE_TYPE.
                     # When COMPUTE_TYPE = double, the effective SIMD width
                     # halves on the same hardware (e.g., AVX2: 8 for float,
                     # 4 for double). Backends report the width appropriate
                     # to the active PrecisionConfig.compute_dtype.
    cache_line_bytes: int
    max_reduce_fan_in: int
    max_local_mem_bytes: int | None
    global_mem_bytes: int
