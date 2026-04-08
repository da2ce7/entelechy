# src/shared/plan_renderer.py
"""Backend-neutral plan rendering protocol (ADR-001)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .plan_types import ExecutionPlan
from .retrieval_future import RetrievalFuture


@runtime_checkable
class PlanRenderer(Protocol):
    """Protocol for backend-specific plan rendering (ADR-001).

    Each backend implements this protocol to translate an immutable
    ExecutionPlan into native dispatch calls. The renderer is the
    Orchestration-tier entry point.
    """

    def render(
        self,
        plan: ExecutionPlan,
        data_injections: dict[str, NDArray] | None = None,
    ) -> dict[str, RetrievalFuture]:
        """Render an execution plan using the backend's native dispatch model.

        Traverses the plan's topological order, dispatching each node
        according to its type. Returns a mapping of event_name -> RetrievalFuture
        for each RetrievalNode in the plan.

        Args:
            plan: The execution plan to render.
            data_injections: Optional mapping of buffer logical_name -> host data.
                Each entry is copied into the allocated plan buffer with the
                matching logical_name (e.g., "input_data", "targets_cce").
                The host array is padded/truncated to match the buffer's shape.
        """
        ...
