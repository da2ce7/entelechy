# src/shared/reduction_tree_plan.py
"""Parametric reduction tree descriptor (ADR-003)."""
from dataclasses import dataclass
from typing import Literal

from .buffer_lifecycle import BufferHandle


@dataclass(frozen=True)
class ReductionTreePlan:
    """Parametric descriptor for a multi-stage log_K(N) reduction tree (ADR-003).

    Pre-computed by the Policy tier. Consumed atomically by the backend
    renderer, which selects kernel tiers (register-reduce vs. local-reduce)
    per stage and manages intermediate buffers.
    """
    num_partials: int
    fan_in_K: int
    num_stages: int
    elements_per_partial: int
    initial_offset_list: tuple[int, ...]
    tree_variant: Literal["sum", "sum_and_clip"]
    threshold_schedule: tuple[float | None, ...]
    partial_width: int
    source_buffer: BufferHandle
    destination_buffer: BufferHandle
