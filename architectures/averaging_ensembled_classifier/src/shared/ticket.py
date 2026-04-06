# src/shared/ticket.py
"""WorkTicket and LearnHandle abstractions for user-facing API (ADR-018).

This module implements the stateful ticket pattern (ADR-018 Choice 1A) that
represents a unit of user intent. The ticket lifecycle mirrors the Act/Learn
temporal split, enabling Event-Triggered execution mode where inference and
learning are decoupled in time.

State machine:
    PENDING → ACT_COMPLETE → RESOLVED → CONSUMED

Shared-layer purity:
    This module imports only from Python stdlib, numpy, and other src/shared/
    modules. No src/backends/ imports are permitted.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .retrieval_future import RetrievalFuture
    from .engine import Engine


class InvalidTicketStateError(Exception):
    """Raised when a ticket operation is invalid for the current state.

    Examples:
        - resolve() on a CONSUMED ticket
        - resolve() called twice on the same ticket
    """


class TicketState(Enum):
    """WorkTicket lifecycle states (ADR-018 Choice 1A).

    The four states form a linear progression:
        PENDING → ACT_COMPLETE → RESOLVED → CONSUMED

    Each transition is triggered by a specific user action or internal
    completion event.
    """

    PENDING = auto()       # Act plan dispatched, inference not yet complete
    ACT_COMPLETE = auto()  # Inference complete, prediction cached
    RESOLVED = auto()      # Learn plan dispatched
    CONSUMED = auto()      # Learn complete, ticket lifecycle terminated


class LearnHandle:
    """Handle for observing Learn phase completion (ADR-018).

    The LearnHandle wraps the Learn plan's RetrievalFuture and provides
    a simplified interface for waiting on learning completion. It also
    manages the ticket state transition to CONSUMED.

    Future: diagnostics properties (Choice 6) would attach here when
    the observability decision is resolved.
    """

    __slots__ = ("_future", "_ticket")

    def __init__(self, future: "RetrievalFuture", ticket: "WorkTicket") -> None:
        """Initialize the LearnHandle.

        Args:
            future: The RetrievalFuture from the Learn plan's final retrieval.
            ticket: The associated WorkTicket for state transition management.
        """
        self._future = future
        self._ticket = ticket

    def wait(self) -> None:
        """Block until Learn phase completes.

        Transitions the associated ticket to CONSUMED state and releases
        device buffers. Idempotent — calling wait() multiple times is safe.
        """
        if self._ticket._state == TicketState.CONSUMED:
            # Already waited; no-op for idempotency
            return
        self._future.wait()
        self._future.release()
        self._ticket._state = TicketState.CONSUMED


class WorkTicket:
    """Stateful ticket representing a unit of user intent (ADR-018 Choice 1A).

    Lifecycle: PENDING → ACT_COMPLETE → RESOLVED → CONSUMED

    The ticket captures input data at submit time, ensuring system-enforced
    Act→Learn association even if the original array is mutated by the caller.

    Key design properties:
        - Data capture: Input is copied at submission to preserve Act→Learn link
        - Future-based: Operations return immediately; blocking on demand
        - Buffer lifecycle: Device buffers released according to Choice 4A (recompute)
        - Cached prediction: get_prediction() caches result for subsequent calls

    Shared-layer purity:
        WorkTicket contains no backend-specific imports. The Engine reference
        is used solely to dispatch the Learn plan via the renderer.
    """

    __slots__ = ("_x_data", "_act_future", "_engine", "_state", "_cached_prediction")

    def __init__(
        self,
        x_data: NDArray[np.floating],
        act_future: "RetrievalFuture",
        engine: "Engine",
    ) -> None:
        """Initialize a WorkTicket in PENDING state.

        Args:
            x_data: Input data for this batch. A copy is stored to preserve
                the Act→Learn association even if the caller mutates the original.
            act_future: The RetrievalFuture from the Act plan's inference retrieval.
            engine: The parent Engine for dispatching the Learn plan.
        """
        # Data capture: store a copy to ensure immutability (ADR-018)
        self._x_data: NDArray[np.floating] = np.copy(x_data)
        self._act_future = act_future
        self._engine = engine
        self._state = TicketState.PENDING
        self._cached_prediction: NDArray[np.floating] | None = None

    @property
    def state(self) -> TicketState:
        """Return the current lifecycle state of the ticket."""
        return self._state

    def get_prediction(self) -> NDArray[np.floating]:
        """Block until Act phase completes and return prediction probabilities.

        Transitions: PENDING → ACT_COMPLETE (first call)
        Subsequent calls return the cached prediction without state transition.

        Device buffers are released upon first call (Choice 4A: recompute).
        The Act plan forward-pass results are not preserved for the Learn phase;
        the Learn plan will recompute them.

        Returns:
            Prediction probabilities with shape (N, C) where N is batch size
            and C is the number of output classes. Values sum to 1 along axis 1.
        """
        if self._cached_prediction is not None:
            # Already have the prediction cached
            return self._cached_prediction

        # Block on the Act future and retrieve the result
        prediction = self._act_future.result()
        self._act_future.release()

        # Cache the prediction and transition state
        self._cached_prediction = prediction
        if self._state == TicketState.PENDING:
            self._state = TicketState.ACT_COMPLETE

        return self._cached_prediction

    def resolve(self, y_data: NDArray[np.integer]) -> LearnHandle:
        """Submit ground truth and dispatch Learn plan.

        Transitions: ACT_COMPLETE → RESOLVED
        (If called in PENDING state, implicitly waits for Act completion first.)

        Args:
            y_data: Ground truth labels with shape (N,) where N matches the
                batch size of x_data. Values are class indices in [0, C).

        Returns:
            A LearnHandle for observing Learn phase completion.

        Raises:
            InvalidTicketStateError: If called on a RESOLVED or CONSUMED ticket
                (i.e., resolve() has already been called).
        """
        # Validate state transition is legal
        if self._state in (TicketState.RESOLVED, TicketState.CONSUMED):
            raise InvalidTicketStateError(
                f"Cannot resolve ticket in {self._state.name} state. "
                "resolve() may only be called once per ticket."
            )

        # If still PENDING, implicitly complete the Act phase first
        if self._state == TicketState.PENDING:
            self.get_prediction()

        # Dispatch the Learn plan via the Engine
        learn_future = self._engine._dispatch_learn_plan(self._x_data, y_data)

        # Transition to RESOLVED and return the LearnHandle
        self._state = TicketState.RESOLVED
        return LearnHandle(learn_future, self)
