# tests/test_integration_execution_plan.py

"""
Integration Tests: ExecutionPlan Assembly & Strategy Composition.

These tests verify that the host-side orchestration layer correctly composes
its primitives into a coherent, verifiable ExecutionPlan.  They exercise:

  - TrainingOrchestrator._create_execution_plan assembly (mocked device)
  - Strategy pattern (CCE vs BCE) selection
  - Lifecycle policies (Cache vs Recompute)
  - Parameter flow iteration and bookkeeping

Target CONCEPT.md Validation Scenarios:
  - The Iris Case: basic end-to-end plan validity
  - Unified Execution Model: CCE and BCE both produce valid plans

No OpenCL device is required.  Where the full orchestrator requires OpenCL,
we test the constituent pieces directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.model_spec import Float32ModelSpec as Float32ModelSpec  # noqa: F401 (used by conftest fixtures)
from src.shared.parameter_space import ParameterSpace  # noqa: F401 (used by conftest fixtures)
from src.shared.stabilization_policy import StabilizationPolicy
from src.shared.workload_primitives import TilingScheme

# These modules require pyopencl at import time — gate them.
# The TYPE_CHECKING block provides Pylance with the real types for static
# analysis, while the runtime try/except handles the case where pyopencl
# is absent (the entire module is skipped via pytestmark below).
if TYPE_CHECKING:
    from src.backends.opencl.compute_patterns import ReductionPlan
    from src.backends.opencl.execution_plan import (
        BceStrategy,
        CacheProvider,
        CceStrategy,
        DataLifecyclePolicy,
        DependencyProvider,
        ExecutionPlan,
        ProblemTypeStrategy,
        StagedComputationProvider,
    )
    from src.backends.opencl.launcher_infra import BufferHandle

try:
    from src.backends.opencl.compute_patterns import ReductionPlan  # noqa: F811
    from src.backends.opencl.execution_plan import (  # noqa: F811
        BceStrategy,
        CacheProvider,
        CceStrategy,
        DataLifecyclePolicy,
        DependencyProvider,
        ExecutionPlan,
        ProblemTypeStrategy,
        StagedComputationProvider,
    )
    from src.backends.opencl.launcher_infra import BufferHandle  # noqa: F811

    _has_cl = True
except ImportError:
    _has_cl = False

pytestmark = pytest.mark.skipif(not _has_cl, reason="pyopencl not installed")


# =========================================================================
# Helpers
# =========================================================================


def _dummy_handle(id_: int = 0) -> BufferHandle:
    return BufferHandle(id=id_)


def _dummy_event() -> MagicMock:
    """Return a mock that quacks like cl.Event."""
    evt = MagicMock()
    evt.wait = MagicMock()
    return evt


def _build_minimal_plan(
    *,
    problem_type_name: str = "CCE",
    adaptation_strategy: str = "CACHE",
    clipping_strategy: str = "GLOBAL",
    batch_size: int = 150,
    num_modules: int = 8,
    output_classes: int = 3,
    stream_chunks: int = 4,
) -> ExecutionPlan:
    """Build a minimal, valid ExecutionPlan for host-side testing."""
    grid = TilingScheme(
        num_module_chunks=(num_modules + 15) // 16,
        num_class_chunks=(output_classes + 15) // 16,
        total_modules=num_modules,
        total_classes=output_classes,
    )
    reduction_plan = ReductionPlan(k=64)
    fp_max = float(np.finfo(np.float32).max)
    stab_policy = StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        fp_format_max=fp_max,
    )

    # Polymorphic strategy selection
    problem_strategy: ProblemTypeStrategy
    if problem_type_name == "CCE":
        problem_strategy = CceStrategy(targets_cce_ref=_dummy_handle(99))
    else:
        problem_strategy = BceStrategy(targets_bce_ref=_dummy_handle(100))

    # Minimal lifecycle policy (no actual device providers)
    providers: dict[str, DependencyProvider] = {}
    if adaptation_strategy == "CACHE":
        providers["hidden_activations"] = CacheProvider(
            handle=_dummy_handle(1),
            ready_event=_dummy_event(),
        )

    # Mock the hyperparams as a simple namespace
    hyperparams = MagicMock()
    hyperparams.adam_epsilon = 1e-7
    hyperparams.learning_rate = 0.001
    hyperparams.adam_beta1 = 0.9
    hyperparams.adam_beta2 = 0.999
    hyperparams.stabilization.max_grad_norm = 1.0
    hyperparams.stabilization.lambda_ = 1.0
    hyperparams.temp_min = 0.1
    hyperparams.temp_max = 10.0
    hyperparams.reduction_k_grad_h = 16

    return ExecutionPlan(
        grid=grid,
        reduction_plan=reduction_plan,
        lifecycle_policy=DataLifecyclePolicy(providers=providers),
        effective_batch_size=batch_size,
        problem_type=problem_strategy,
        clipping_strategy=clipping_strategy,
        stabilization_policy=stab_policy,
        hyperparams=hyperparams,
        shared_backprop_stream_chunks=stream_chunks,
        adaptation_strategy=adaptation_strategy,
    )


# =========================================================================
# 1. ExecutionPlan Assembly
# =========================================================================


class TestExecutionPlanAssembly:
    """Verify that a complete ExecutionPlan can be constructed from its parts."""

    def test_plan_creation_cce(self):
        plan = _build_minimal_plan(problem_type_name="CCE")
        assert isinstance(plan.problem_type, CceStrategy)
        assert plan.problem_type.required_targets_buffer_name == "targets_cce"

    def test_plan_creation_bce(self):
        plan = _build_minimal_plan(problem_type_name="BCE")
        assert isinstance(plan.problem_type, BceStrategy)
        assert plan.problem_type.required_targets_buffer_name == "targets_bce"

    def test_plan_is_frozen(self):
        """ExecutionPlan is a frozen dataclass; mutation must be impossible."""
        plan = _build_minimal_plan()
        with pytest.raises(AttributeError):
            plan.effective_batch_size = 999  # type: ignore[misc]

    def test_plan_carries_grid(self):
        plan = _build_minimal_plan(num_modules=8, output_classes=3)
        assert plan.grid.total_modules == 8
        assert plan.grid.total_classes == 3

    def test_plan_carries_stabilization_policy(self):
        plan = _build_minimal_plan()
        assert isinstance(plan.stabilization_policy, StabilizationPolicy)
        assert plan.stabilization_policy.t_algorithmic == 1.0


# =========================================================================
# 2. Strategy Pattern (CCE / BCE)
# =========================================================================


class TestProblemTypeStrategy:
    """Verify the polymorphic problem_type strategy."""

    def test_cce_strategy_provides_correct_target_buffer(self):
        strategy = CceStrategy(targets_cce_ref=_dummy_handle(42))
        assert strategy.required_targets_buffer_name == "targets_cce"

    def test_bce_strategy_provides_correct_target_buffer(self):
        strategy = BceStrategy(targets_bce_ref=_dummy_handle(43))
        assert strategy.required_targets_buffer_name == "targets_bce"

    def test_cce_and_bce_plans_differ_only_in_strategy(self):
        plan_cce = _build_minimal_plan(problem_type_name="CCE")
        plan_bce = _build_minimal_plan(problem_type_name="BCE")
        # Same grid, batch size, etc.
        assert plan_cce.effective_batch_size == plan_bce.effective_batch_size
        assert plan_cce.grid.total_tiles == plan_bce.grid.total_tiles
        # Different strategy types
        assert not isinstance(plan_cce.problem_type, type(plan_bce.problem_type))


# =========================================================================
# 3. Dependency Lifecycle Policies
# =========================================================================


class TestLifecyclePolicies:
    """Verify the DependencyProvider abstraction and DataLifecyclePolicy."""

    def test_cache_provider_resolve(self):
        """CacheProvider should return its pre-existing handle and event."""
        handle = _dummy_handle(10)
        event = _dummy_event()
        provider = CacheProvider(handle=handle, ready_event=event)
        resolved_h, resolved_e = provider.resolve(
            queue=MagicMock(),
            ex=MagicMock(),
            wait_for=[],
        )
        assert resolved_h == handle
        assert resolved_e is event

    def test_lifecycle_policy_key_lookup(self):
        providers: dict[str, DependencyProvider] = {
            "hidden_activations": CacheProvider(
                handle=_dummy_handle(1),
                ready_event=_dummy_event(),
            ),
        }
        policy = DataLifecyclePolicy(providers=providers)
        assert isinstance(policy.get_provider("hidden_activations"), CacheProvider)

    def test_lifecycle_policy_missing_key_raises(self):
        policy = DataLifecyclePolicy(providers={})
        with pytest.raises(KeyError, match="No lifecycle policy"):
            policy.get_provider("nonexistent_buffer")

    def test_staged_computation_provider_calls_fn(self):
        """StagedComputationProvider should delegate to its injected function."""
        from functools import partial

        def _mock_fn(
            queue: object, ex: object, wait_for: list[object], *, data: str = "hello"
        ) -> tuple[BufferHandle, MagicMock]:
            return _dummy_handle(77), _dummy_event()

        provider = StagedComputationProvider(computation_fn=partial(_mock_fn))
        h, _e = provider.resolve(queue=MagicMock(), ex=MagicMock(), wait_for=[])
        assert h.id == 77


# =========================================================================
# 4. ParameterSpace Flow Iteration
# =========================================================================


class TestParameterFlowIteration:
    """Verify that ParameterSpace yields consistent flow configurations."""

    def test_flow_names_are_unique(self, iris_param_space: ParameterSpace) -> None:
        names = [flow.name for flow in iris_param_space]
        assert len(names) == len(set(names))

    def test_specialized_flows_lack_optimizer_buffers(self, iris_param_space: ParameterSpace) -> None:
        for flow in iris_param_space:
            if flow.specialized_reduction:
                assert flow.m1_buffer_name == ""
                assert flow.m2_buffer_name == ""

    def test_non_specialized_flows_have_all_buffer_names(self, iris_param_space: ParameterSpace) -> None:
        for flow in iris_param_space:
            if not flow.specialized_reduction:
                assert flow.param_buffer_name
                assert flow.partial_grad_buffer_name
                assert flow.clipped_partial_grad_buffer_name
                assert flow.summed_grad_buffer_name
                assert flow.final_grad_buffer_name
                assert flow.m1_buffer_name
                assert flow.m2_buffer_name

    def test_expected_flow_count(self, iris_param_space: ParameterSpace) -> None:
        """Iris model should have 6 parameter flows."""
        flows = list(iris_param_space)
        assert len(flows) == 6

    def test_hidden_activations_flow_is_specialized(self, iris_param_space: ParameterSpace) -> None:
        ha_flows = [f for f in iris_param_space if f.name == "hidden_activations"]
        assert len(ha_flows) == 1
        assert ha_flows[0].specialized_reduction is True


# =========================================================================
# 5. Clipping Strategy Selection
# =========================================================================


class TestClippingStrategySelection:
    """Verify that clipping strategy is properly encoded in the plan."""

    @pytest.mark.parametrize("strategy_name", ["GLOBAL", "PER_ITEM"])
    def test_clipping_strategy_preserved(self, strategy_name: str) -> None:
        plan = _build_minimal_plan(clipping_strategy=strategy_name)
        assert plan.clipping_strategy == strategy_name


# =========================================================================
# 6. Adaptation Strategy Encoding
# =========================================================================


class TestAdaptationStrategy:
    """Verify that CACHE and RECOMPUTE strategies are correctly encoded."""

    def test_cache_strategy_has_hidden_provider(self):
        plan = _build_minimal_plan(adaptation_strategy="CACHE")
        assert plan.adaptation_strategy == "CACHE"
        provider = plan.lifecycle_policy.get_provider("hidden_activations")
        assert isinstance(provider, CacheProvider)

    def test_recompute_strategy_has_no_hidden_provider(self):
        plan = _build_minimal_plan(adaptation_strategy="RECOMPUTE_GRAD_H")
        assert plan.adaptation_strategy == "RECOMPUTE_GRAD_H"
        with pytest.raises(KeyError):
            plan.lifecycle_policy.get_provider("hidden_activations")
