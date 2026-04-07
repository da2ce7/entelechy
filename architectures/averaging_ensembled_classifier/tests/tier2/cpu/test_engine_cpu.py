# tests/tier2/cpu/test_engine_cpu.py
"""Tier 2 CPU integration tests for Engine and WorkTicket (ADR-018).

These tests verify that the Engine and WorkTicket lifecycle work correctly
with the real CPUPlanRenderer. They test structural correctness (state
transitions, shape validation) rather than value correctness.

Value correctness depends on the renderer properly injecting input data
into allocated buffers, which is tested separately in kernel-level tests.
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if feature flag is disabled or CPU backend unavailable
try:
    from src._build_config import BACKEND_CPU, FEATURE_TICKET_API  # type: ignore[import-not-found]
except ImportError:
    BACKEND_CPU = False
    FEATURE_TICKET_API = False

pytestmark = [
    pytest.mark.skipif(not FEATURE_TICKET_API, reason="FEATURE_TICKET_API not enabled"),
    pytest.mark.skipif(not BACKEND_CPU, reason="CPU backend not available"),
    pytest.mark.cpu,
]


@pytest.fixture
def model_spec():
    """Small model spec for integration tests."""
    from src.shared.model_spec import ModelSpec
    from src.shared.precision_config import PrecisionConfig

    return ModelSpec(
        precision=PrecisionConfig.float32(),
        input_dim=8,
        hidden_dim=16,
        output_classes=4,
        num_modules=2,
        simd_width=8,
        cache_line_bytes=64,
    )


@pytest.fixture
def hardware_profile():
    """Hardware profile from CPU discovery."""
    from src.backends.cpu.discovery import discover_hardware
    return discover_hardware()


@pytest.fixture
def parameter_space(model_spec):
    """Parameter space for the model."""
    from src.shared.parameter_space import ParameterSpace
    return ParameterSpace(model_spec)


@pytest.fixture
def engine(model_spec, parameter_space, hardware_profile):
    """Engine with CPU backend."""
    from src.shared.engine import Engine
    return Engine(
        model_spec,
        parameter_space,
        hardware_profile,
        backend="cpu",
    )


@pytest.fixture
def input_data(model_spec):
    """Sample input data matching model's input_dim."""
    batch_size = 4
    return np.random.randn(batch_size, model_spec.input_dim).astype(np.float32)


@pytest.fixture
def target_data(model_spec):
    """Sample ground truth labels."""
    batch_size = 4
    return np.random.randint(0, model_spec.output_classes, size=(batch_size,)).astype(np.int32)


class TestEngineConstruction:
    """Tests for Engine construction with CPU backend."""

    def test_engine_constructs_with_cpu_backend(
        self, model_spec, parameter_space, hardware_profile
    ) -> None:
        """Engine successfully initializes with CPU backend."""
        from src.shared.engine import Engine

        engine = Engine(
            model_spec,
            parameter_space,
            hardware_profile,
            backend="cpu",
        )

        assert engine._renderer is not None

    def test_engine_auto_selects_cpu(
        self, model_spec, parameter_space, hardware_profile
    ) -> None:
        """Engine with backend='auto' selects CPU (CPU backend is enabled)."""
        from src.shared.engine import Engine

        engine = Engine(
            model_spec,
            parameter_space,
            hardware_profile,
            backend="auto",
        )

        # Should have created a renderer
        assert engine._renderer is not None


class TestSubmitReturnsTicket:
    """Tests for Engine.submit() returning WorkTicket."""

    def test_submit_returns_pending_ticket(self, engine, input_data) -> None:
        """engine.submit() returns a WorkTicket in PENDING state."""
        from src.shared.ticket import TicketState

        ticket = engine.submit(input_data)

        assert ticket.state == TicketState.PENDING

    def test_submit_stores_input_copy(self, engine, input_data) -> None:
        """Ticket stores a copy of input data."""
        ticket = engine.submit(input_data)

        # Mutate original - ticket's copy should be unchanged
        original = input_data.copy()
        input_data[0, 0] = -999.0

        np.testing.assert_array_equal(ticket._x_data, original)


class TestTicketLifecycleWithCPU:
    """Tests for full ticket lifecycle using actual CPU renderer."""

    def test_get_prediction_returns_array(self, engine, input_data, model_spec) -> None:
        """get_prediction() returns a numpy array with correct shape."""
        ticket = engine.submit(input_data)
        prediction = ticket.get_prediction()

        # Should have shape (batch_size, output_classes)
        expected_shape = (input_data.shape[0], model_spec.output_classes)
        assert prediction.shape == expected_shape
        assert prediction.dtype == np.float32

    def test_get_prediction_transitions_state(self, engine, input_data) -> None:
        """get_prediction() transitions ticket to ACT_COMPLETE."""
        from src.shared.ticket import TicketState

        ticket = engine.submit(input_data)
        assert ticket.state == TicketState.PENDING

        _ = ticket.get_prediction()
        assert ticket.state == TicketState.ACT_COMPLETE

    def test_resolve_transitions_state(self, engine, input_data, target_data) -> None:
        """resolve() transitions ticket to RESOLVED."""
        from src.shared.ticket import TicketState

        ticket = engine.submit(input_data)
        _ = ticket.get_prediction()
        assert ticket.state == TicketState.ACT_COMPLETE

        handle = ticket.resolve(target_data)
        assert ticket.state == TicketState.RESOLVED
        assert handle is not None

    def test_learn_handle_wait_transitions_to_consumed(
        self, engine, input_data, target_data
    ) -> None:
        """learn_handle.wait() transitions ticket to CONSUMED."""
        from src.shared.ticket import TicketState

        ticket = engine.submit(input_data)
        _ = ticket.get_prediction()
        handle = ticket.resolve(target_data)

        handle.wait()
        assert ticket.state == TicketState.CONSUMED

    def test_full_ticket_lifecycle(self, engine, input_data, target_data) -> None:
        """Full lifecycle: PENDING → ACT_COMPLETE → RESOLVED → CONSUMED."""
        from src.shared.ticket import TicketState

        # 1. Submit
        ticket = engine.submit(input_data)
        assert ticket.state == TicketState.PENDING

        # 2. Get prediction
        prediction = ticket.get_prediction()
        assert ticket.state == TicketState.ACT_COMPLETE
        assert prediction is not None

        # 3. Resolve with ground truth
        handle = ticket.resolve(target_data)
        assert ticket.state == TicketState.RESOLVED

        # 4. Wait for learning to complete
        handle.wait()
        assert ticket.state == TicketState.CONSUMED


class TestTrainBatchConvenience:
    """Tests for engine.train_batch() convenience method."""

    def test_train_batch_returns_prediction(
        self, engine, input_data, target_data, model_spec
    ) -> None:
        """train_batch() returns prediction with correct shape."""
        prediction = engine.train_batch(input_data, target_data)

        expected_shape = (input_data.shape[0], model_spec.output_classes)
        assert prediction.shape == expected_shape

    def test_train_batch_completes_without_error(
        self, engine, input_data, target_data
    ) -> None:
        """train_batch() completes the full lifecycle without error."""
        # Should not raise
        _ = engine.train_batch(input_data, target_data)


class TestSequentialTickets:
    """Tests for sequential ticket processing."""

    def test_sequential_tickets(self, engine, model_spec) -> None:
        """Multiple tickets can be created and resolved sequentially."""
        from src.shared.ticket import TicketState

        rng = np.random.default_rng(42)

        for i in range(3):
            x_data = rng.standard_normal((4, model_spec.input_dim)).astype(np.float32)
            y_data = rng.integers(0, model_spec.output_classes, size=(4,)).astype(np.int32)

            ticket = engine.submit(x_data)
            _ = ticket.get_prediction()
            ticket.resolve(y_data).wait()

            assert ticket.state == TicketState.CONSUMED
