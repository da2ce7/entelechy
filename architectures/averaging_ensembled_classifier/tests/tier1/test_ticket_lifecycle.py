# tests/tier1/test_ticket_lifecycle.py
"""Tier 1 tests for WorkTicket state machine (ADR-018).

These tests validate the ticket lifecycle transitions, error cases, and
data capture semantics using mock RetrievalFutures. No backend is required.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.ticket import (
    InvalidTicketStateError,
    LearnHandle,
    TicketState,
    WorkTicket,
)

if TYPE_CHECKING:
    from src.shared.retrieval_future import RetrievalFuture


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


class MockRetrievalFuture:
    """Mock RetrievalFuture for testing without a real backend."""

    def __init__(
        self,
        node_id: str,
        result_data: np.ndarray,
    ) -> None:
        self._node_id = node_id
        self._result_data = result_data
        self._waited = False
        self._released = False

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        self._waited = True

    def result(self) -> np.ndarray:
        self.wait()
        return self._result_data

    def release(self) -> None:
        self._released = True


@pytest.fixture
def mock_engine() -> MagicMock:
    """Create a mock Engine for ticket tests."""
    engine = MagicMock()
    # Configure _dispatch_learn_plan to return a mock future
    learn_future = MockRetrievalFuture(
        node_id="final_batch_retrieval",
        result_data=np.array([[0.0]], dtype=np.float32),
    )
    engine._dispatch_learn_plan.return_value = learn_future
    return engine


@pytest.fixture
def prediction_data() -> np.ndarray:
    """Sample prediction probabilities."""
    # 4 samples, 3 classes, valid probability distribution
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
        [0.5, 0.3, 0.2],
    ], dtype=np.float32)
    return probs


@pytest.fixture
def input_data() -> np.ndarray:
    """Sample input data."""
    return np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ], dtype=np.float32)


@pytest.fixture
def target_data() -> np.ndarray:
    """Sample ground truth labels."""
    return np.array([0, 1, 2, 0], dtype=np.int32)


@pytest.fixture
def act_future(prediction_data: np.ndarray) -> MockRetrievalFuture:
    """Mock Act-phase retrieval future."""
    return MockRetrievalFuture(
        node_id="inference_retrieval",
        result_data=prediction_data,
    )


@pytest.fixture
def ticket(
    input_data: np.ndarray,
    act_future: MockRetrievalFuture,
    mock_engine: MagicMock,
) -> WorkTicket:
    """Create a fresh WorkTicket in PENDING state."""
    return WorkTicket(input_data, act_future, mock_engine)


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestTicketInitialState:
    """Tests for initial ticket state."""

    def test_ticket_initial_state_is_pending(self, ticket: WorkTicket) -> None:
        """Freshly created ticket is in PENDING state."""
        assert ticket.state == TicketState.PENDING

    def test_ticket_stores_data_copy(
        self,
        input_data: np.ndarray,
        act_future: MockRetrievalFuture,
        mock_engine: MagicMock,
    ) -> None:
        """Ticket stores a copy of input data, not a reference."""
        original = input_data.copy()
        ticket = WorkTicket(input_data, act_future, mock_engine)

        # Mutate the original array
        input_data[0, 0] = -999.0

        # Ticket's internal copy should be unchanged
        np.testing.assert_array_equal(ticket._x_data, original)


class TestGetPrediction:
    """Tests for get_prediction() method."""

    def test_get_prediction_transitions_to_act_complete(
        self,
        ticket: WorkTicket,
    ) -> None:
        """get_prediction() on PENDING ticket transitions to ACT_COMPLETE."""
        assert ticket.state == TicketState.PENDING
        _ = ticket.get_prediction()
        assert ticket.state == TicketState.ACT_COMPLETE

    def test_get_prediction_returns_prediction_data(
        self,
        ticket: WorkTicket,
        prediction_data: np.ndarray,
    ) -> None:
        """get_prediction() returns the inference result."""
        result = ticket.get_prediction()
        np.testing.assert_array_equal(result, prediction_data)

    def test_get_prediction_is_idempotent(
        self,
        ticket: WorkTicket,
        prediction_data: np.ndarray,
    ) -> None:
        """Multiple get_prediction() calls return same cached value."""
        result1 = ticket.get_prediction()
        result2 = ticket.get_prediction()
        result3 = ticket.get_prediction()

        # All results should be identical (same object due to caching)
        assert result1 is result2
        assert result2 is result3
        np.testing.assert_array_equal(result1, prediction_data)

    def test_get_prediction_releases_buffers(
        self,
        ticket: WorkTicket,
        act_future: MockRetrievalFuture,
    ) -> None:
        """get_prediction() releases the Act future's buffers."""
        assert not act_future._released
        _ = ticket.get_prediction()
        assert act_future._released

    def test_get_prediction_on_resolved_returns_cached(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
        prediction_data: np.ndarray,
    ) -> None:
        """get_prediction() on RESOLVED ticket returns cached prediction."""
        _ = ticket.get_prediction()
        ticket.resolve(target_data)
        assert ticket.state == TicketState.RESOLVED

        result = ticket.get_prediction()
        np.testing.assert_array_equal(result, prediction_data)
        # State should not change
        assert ticket.state == TicketState.RESOLVED

    def test_get_prediction_on_consumed_returns_cached(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
        prediction_data: np.ndarray,
    ) -> None:
        """get_prediction() on CONSUMED ticket returns cached prediction."""
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)
        handle.wait()
        assert ticket.state == TicketState.CONSUMED

        result = ticket.get_prediction()
        np.testing.assert_array_equal(result, prediction_data)
        # State should not change
        assert ticket.state == TicketState.CONSUMED


class TestResolve:
    """Tests for resolve() method."""

    def test_resolve_transitions_to_resolved(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
    ) -> None:
        """resolve() on ACT_COMPLETE ticket transitions to RESOLVED."""
        _ = ticket.get_prediction()
        assert ticket.state == TicketState.ACT_COMPLETE

        _ = ticket.resolve(target_data)
        assert ticket.state == TicketState.RESOLVED

    def test_resolve_on_pending_waits_for_act(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
        act_future: MockRetrievalFuture,
    ) -> None:
        """resolve() on PENDING ticket implicitly transitions through ACT_COMPLETE."""
        assert ticket.state == TicketState.PENDING
        assert not act_future._waited

        _ = ticket.resolve(target_data)

        # Should have waited on Act future
        assert act_future._waited
        # Should be in RESOLVED state (not ACT_COMPLETE)
        assert ticket.state == TicketState.RESOLVED

    def test_resolve_returns_learn_handle(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
    ) -> None:
        """resolve() returns a LearnHandle."""
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)
        assert isinstance(handle, LearnHandle)

    def test_resolve_dispatches_learn_plan(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
        mock_engine: MagicMock,
    ) -> None:
        """resolve() dispatches Learn plan via Engine."""
        _ = ticket.get_prediction()
        _ = ticket.resolve(target_data)

        mock_engine._dispatch_learn_plan.assert_called_once()
        call_args = mock_engine._dispatch_learn_plan.call_args
        # Check that x_data and y_data were passed
        np.testing.assert_array_equal(call_args[0][0], ticket._x_data)
        np.testing.assert_array_equal(call_args[0][1], target_data)

    def test_resolve_on_consumed_raises(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
    ) -> None:
        """resolve() on CONSUMED ticket raises InvalidTicketStateError."""
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)
        handle.wait()
        assert ticket.state == TicketState.CONSUMED

        with pytest.raises(InvalidTicketStateError) as exc_info:
            ticket.resolve(target_data)

        assert "CONSUMED" in str(exc_info.value)

    def test_double_resolve_raises(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
    ) -> None:
        """Second resolve() call raises InvalidTicketStateError."""
        _ = ticket.get_prediction()
        _ = ticket.resolve(target_data)
        assert ticket.state == TicketState.RESOLVED

        with pytest.raises(InvalidTicketStateError) as exc_info:
            ticket.resolve(target_data)

        assert "RESOLVED" in str(exc_info.value)


class TestLearnHandle:
    """Tests for LearnHandle class."""

    def test_learn_handle_wait_transitions_to_consumed(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
    ) -> None:
        """learn_handle.wait() transitions ticket to CONSUMED."""
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)
        assert ticket.state == TicketState.RESOLVED

        handle.wait()
        assert ticket.state == TicketState.CONSUMED

    def test_learn_handle_wait_is_idempotent(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
    ) -> None:
        """Multiple wait() calls are safe (idempotent)."""
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)

        handle.wait()
        handle.wait()
        handle.wait()

        assert ticket.state == TicketState.CONSUMED

    def test_learn_handle_wait_releases_buffers(
        self,
        ticket: WorkTicket,
        target_data: np.ndarray,
        mock_engine: MagicMock,
    ) -> None:
        """learn_handle.wait() releases Learn future's buffers."""
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)

        # Get the mock Learn future from the engine
        learn_future = mock_engine._dispatch_learn_plan.return_value
        assert not learn_future._released

        handle.wait()
        assert learn_future._released


class TestDataCapture:
    """Tests for data capture immutability."""

    def test_data_capture_is_copy(
        self,
        act_future: MockRetrievalFuture,
        mock_engine: MagicMock,
    ) -> None:
        """Mutating original array does not affect ticket's stored data."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        expected = original.copy()

        ticket = WorkTicket(original, act_future, mock_engine)

        # Mutate the original extensively
        original[0, 0] = -999.0
        original[1, 1] = 999.0
        original *= 100

        # Ticket's internal copy should be unchanged
        np.testing.assert_array_equal(ticket._x_data, expected)

    def test_data_capture_preserves_dtype(
        self,
        act_future: MockRetrievalFuture,
        mock_engine: MagicMock,
    ) -> None:
        """Data capture preserves the input array's dtype."""
        input_f16 = np.array([[1.0, 2.0]], dtype=np.float16)
        input_f64 = np.array([[1.0, 2.0]], dtype=np.float64)

        ticket_f16 = WorkTicket(input_f16, act_future, mock_engine)
        ticket_f64 = WorkTicket(input_f64, act_future, mock_engine)

        assert ticket_f16._x_data.dtype == np.float16
        assert ticket_f64._x_data.dtype == np.float64
