# tests/tier1/test_engine_construction.py
"""Tier 1 tests for Engine construction and configuration (ADR-018).

These tests validate Engine initialization, backend selection logic, and
configuration handling using mock renderer factories. No actual backend
is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.precision_config import PrecisionConfig
from architectures.averaging_ensembled_classifier.src.shared.problem_type_spec import PlanCceStrategy, PlanBceStrategy
from src.shared.stabilization_policy import StabilizationPolicy


# Skip entire module if feature flag is disabled
try:
    from src._build_config import FEATURE_TICKET_API  # type: ignore[import-not-found]
except ImportError:
    FEATURE_TICKET_API = False

pytestmark = pytest.mark.skipif(
    not FEATURE_TICKET_API,
    reason="FEATURE_TICKET_API not enabled"
)


@pytest.fixture
def precision() -> PrecisionConfig:
    return PrecisionConfig.float32()


@pytest.fixture
def hardware() -> HardwareProfile:
    return HardwareProfile(
        simd_width=16,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=65536,
        global_mem_bytes=4 * 1024**3,
    )


@pytest.fixture
def model_spec(precision: PrecisionConfig) -> ModelSpec:
    return ModelSpec(
        precision=precision,
        input_dim=128,
        hidden_dim=64,
        output_classes=10,
        num_modules=4,
        simd_width=16,
        cache_line_bytes=64,
    )


@pytest.fixture
def parameter_space(model_spec: ModelSpec) -> ParameterSpace:
    return ParameterSpace(model_spec)


@pytest.fixture
def mock_renderer() -> MagicMock:
    """Create a mock PlanRenderer."""
    renderer = MagicMock()
    # Configure render() to return a dict with expected futures
    mock_future = MagicMock()
    mock_future.result.return_value = [[0.5, 0.5]]
    renderer.render.return_value = {
        "inference_retrieval": mock_future,
        "final_batch_retrieval": mock_future,
    }
    return renderer


@pytest.fixture
def mock_renderer_factory(mock_renderer: MagicMock):
    """Create a mock renderer factory."""
    def factory(backend: str, hardware: HardwareProfile, precision: PrecisionConfig):
        return mock_renderer
    return factory


class TestEngineConstruction:
    """Tests for Engine initialization."""

    def test_engine_accepts_model_spec(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine correctly stores model_spec."""
        from src.shared.engine import Engine
        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._model_spec is model_spec

    def test_engine_accepts_parameter_space(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine correctly stores parameter_space."""
        from src.shared.engine import Engine
        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._parameter_space is parameter_space

    def test_engine_accepts_hardware_profile(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine correctly stores hardware_profile."""
        from src.shared.engine import Engine
        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._hardware_profile is hardware

    def test_engine_default_precision_from_model_spec(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine uses model_spec.precision when precision not specified."""
        from src.shared.engine import Engine
        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._precision is model_spec.precision

    def test_engine_explicit_precision_override(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine uses explicit precision when provided."""
        from src.shared.engine import Engine
        custom_precision = PrecisionConfig.mixed_f16_f32()
        engine = Engine(
            model_spec, parameter_space, hardware,
            precision=custom_precision,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._precision is custom_precision
        assert engine._precision is not model_spec.precision

    def test_engine_default_strategy_is_cce(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine defaults to CCE strategy when not specified."""
        from src.shared.engine import Engine
        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )
        assert isinstance(engine._strategy, PlanCceStrategy)

    def test_engine_explicit_strategy_override(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine uses explicit strategy when provided."""
        from src.shared.engine import Engine
        bce_strategy = PlanBceStrategy()
        engine = Engine(
            model_spec, parameter_space, hardware,
            strategy=bce_strategy,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._strategy is bce_strategy

    def test_engine_auto_configures_policy(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
    ) -> None:
        """Engine auto-configures stabilization policy when not specified."""
        from src.shared.engine import Engine
        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )
        assert isinstance(engine._policy, StabilizationPolicy)
        # Auto-config uses model precision's compute_fp_format_max
        assert engine._policy.compute_fp_format_max == model_spec.precision.compute_fp_format_max

    def test_engine_explicit_policy_override(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer_factory,
        precision: PrecisionConfig,
    ) -> None:
        """Engine uses explicit policy when provided."""
        from src.shared.engine import Engine
        custom_policy = StabilizationPolicy(
            t_algorithmic=2.0,
            lambda_=0.5,
            compute_fp_format_max=precision.compute_fp_format_max,
        )
        engine = Engine(
            model_spec, parameter_space, hardware,
            policy=custom_policy,
            renderer_factory=mock_renderer_factory,
        )
        assert engine._policy is custom_policy


class TestRendererFactory:
    """Tests for renderer factory invocation."""

    def test_custom_factory_is_called(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer: MagicMock,
    ) -> None:
        """Custom renderer factory is called with correct arguments."""
        from src.shared.engine import Engine

        factory_called = []

        def tracking_factory(backend, hw, prec):
            factory_called.append((backend, hw, prec))
            return mock_renderer

        engine = Engine(
            model_spec, parameter_space, hardware,
            backend="cpu",
            renderer_factory=tracking_factory,
        )

        assert len(factory_called) == 1
        backend, hw, prec = factory_called[0]
        assert backend == "cpu"
        assert hw is hardware
        assert prec is model_spec.precision

    def test_factory_receives_explicit_precision(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer: MagicMock,
    ) -> None:
        """Factory receives explicit precision when provided."""
        from src.shared.engine import Engine

        custom_precision = PrecisionConfig.mixed_f16_f32()
        received_precision = []

        def tracking_factory(backend, hw, prec):
            received_precision.append(prec)
            return mock_renderer

        engine = Engine(
            model_spec, parameter_space, hardware,
            precision=custom_precision,
            renderer_factory=tracking_factory,
        )

        assert len(received_precision) == 1
        assert received_precision[0] is custom_precision

    def test_renderer_stored_from_factory(
        self,
        model_spec: ModelSpec,
        parameter_space: ParameterSpace,
        hardware: HardwareProfile,
        mock_renderer: MagicMock,
        mock_renderer_factory,
    ) -> None:
        """Engine stores the renderer returned by the factory."""
        from src.shared.engine import Engine

        engine = Engine(
            model_spec, parameter_space, hardware,
            renderer_factory=mock_renderer_factory,
        )

        assert engine._renderer is mock_renderer
