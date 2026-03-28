# src/shared/plan_renderer.py
"""Backend-neutral plan rendering protocol (ADR-001)."""
from typing import Protocol, runtime_checkable

from .plan_types import ExecutionPlan
from .retrieval_future import RetrievalFuture


@runtime_checkable
class PlanRenderer(Protocol):
    """Protocol for backend-specific plan rendering (ADR-001).

    Each backend implements this protocol to translate an immutable
    ExecutionPlan into native dispatch calls. The renderer is the
    Orchestration-tier entry point.
    """

    def render(self, plan: ExecutionPlan) -> dict[str, RetrievalFuture]:
        """Render an execution plan using the backend's native dispatch model.

        Traverses the plan's topological order, dispatching each node
        according to its type. Returns a mapping of event_name -> RetrievalFuture
        for each RetrievalNode in the plan.
        """
        ...
