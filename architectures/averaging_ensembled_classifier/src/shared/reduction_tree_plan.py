"""Parametric reduction-tree descriptor (ADR-003).

Bridges the Policy tier (which plans the reduction topology) and the
Orchestration tier (which renders it into backend-native dispatches).

Three rendering regimes
───────────────────────
**Identity tier** (``num_stages == 0``, ``num_partials == 1``):
    No reduction dispatch.  The backend references the single partial
    directly, inserting a precision bridge if the source and destination
    buffer roles differ (CONCEPT.md §2).

**Single-stage** (``num_stages == 1``):
    ``aggregate_*_reduce`` (storage- or compute-entry variant per
    ADR-026) followed by ``clip_intermediate_grad`` when
    ``tree_variant == "sum_and_clip"``.

**Multi-stage** (``num_stages > 1``):
    ``reduce_k_fan_in_and_clip`` at the leaf stage (storage- or
    compute-entry), ``reduce_k_fan_in_and_clip_from_compute`` at all
    interior stages.  Clipping thresholds are drawn from
    ``threshold_schedule`` per stage.

Precision-variant selection
───────────────────────────
The leaf stage's kernel variant (storage-entry vs. compute-entry) is
determined by ``source_buffer``'s ``precision_role`` in the plan's
buffer namespace — an Orchestration-tier concern (ADR-026).  All
interior stages use compute-entry variants because prior stages
produce ``COMPUTE_TYPE`` output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .buffer_lifecycle import BufferHandle

__all__ = ["ReductionTreePlan"]


@dataclass(frozen=True)
class ReductionTreePlan:
    """Immutable descriptor for a ``log_K(N)`` reduction tree.

    Pre-computed by the Policy tier; consumed atomically by the
    Orchestration tier, which selects kernel tiers per stage and
    manages intermediate buffers.

    Parameters
    ----------
    num_partials:
        Total input partial count (≥ 1).
    fan_in:
        Reduction fan-in *K* — partials consumed per output node
        per stage.  1 when ``num_stages == 0`` (Identity tier).
    num_stages:
        Number of reduction stages (``⌈log_K(N)⌉``).  0 signals
        the Identity tier bypass (single partial, no dispatch).
    partial_width:
        Scalar element count per partial vector.  Governs both the
        read width at each stage and the destination buffer size.
    initial_offset_list:
        Element-offset array (not byte offsets) into the source
        collection buffer for stage-0 inputs.  Length must equal
        ``num_partials``.  Fulfils the Indirection Contract
        (CONCEPT.md §2).
    tree_variant:
        ``"sum"`` — pure summation (diagnostic reduction, Node 14).
        Threshold schedule is empty.
        ``"sum_and_clip"`` — summation with per-stage L2 clipping
        (gradient reduction, Nodes 15/20).  Threshold schedule has
        ``num_stages`` entries.
    threshold_schedule:
        Per-stage clipping thresholds.  Index 0 is the leaf stage;
        index ``num_stages - 1`` is the root.  Empty when
        ``tree_variant == "sum"`` (no clipping).  All entries are
        ``float`` when ``tree_variant == "sum_and_clip"``.
    source_buffer:
        Handle to the partial-collection input buffer.  Its
        ``precision_role`` (looked up from the plan's buffer
        namespace) determines leaf-stage kernel variant selection
        (ADR-026).
    destination_buffer:
        Handle to the fully-reduced output buffer.
    """

    num_partials: int
    fan_in: int
    num_stages: int
    partial_width: int
    initial_offset_list: tuple[int, ...]
    tree_variant: Literal["sum", "sum_and_clip"]
    threshold_schedule: tuple[float, ...]
    source_buffer: BufferHandle
    destination_buffer: BufferHandle

    def __post_init__(self) -> None:
        if self.num_partials < 1:
            raise ValueError(
                f"num_partials must be ≥ 1, got {self.num_partials}"
            )
        if self.fan_in < 1:
            raise ValueError(f"fan_in must be ≥ 1, got {self.fan_in}")
        if self.num_stages < 0:
            raise ValueError(
                f"num_stages must be ≥ 0, got {self.num_stages}"
            )
        if self.partial_width < 1:
            raise ValueError(
                f"partial_width must be ≥ 1, got {self.partial_width}"
            )
        if len(self.initial_offset_list) != self.num_partials:
            raise ValueError(
                f"initial_offset_list length "
                f"({len(self.initial_offset_list)}) "
                f"!= num_partials ({self.num_partials})"
            )
        # Identity tier: single partial → no reduction stages.
        if self.num_stages == 0 and self.num_partials != 1:
            raise ValueError(
                f"Identity tier (num_stages=0) requires "
                f"num_partials=1, got {self.num_partials}"
            )
        if self.num_stages == 0 and self.fan_in != 1:
            raise ValueError(
                f"Identity tier (num_stages=0) requires "
                f"fan_in=1, got {self.fan_in}"
            )
        # Variant ↔ schedule consistency.
        if self.tree_variant == "sum_and_clip":
            if len(self.threshold_schedule) != self.num_stages:
                raise ValueError(
                    f"sum_and_clip: threshold_schedule length "
                    f"({len(self.threshold_schedule)}) "
                    f"!= num_stages ({self.num_stages})"
                )
        elif self.tree_variant == "sum":
            if self.threshold_schedule:
                raise ValueError(
                    f"sum variant: threshold_schedule must be "
                    f"empty, got {len(self.threshold_schedule)} "
                    f"entries"
                )
