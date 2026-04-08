# tests/tier1/test_optimizer_plumbing.py
"""Integration tests for optimizer hyperparameter plumbing (ADR-029)."""
import pytest

from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.optimizer_config import OptimizerConfig
from src.shared.plan_builder import build_learn_plan
from src.shared.problem_type_strategy import PlanCceStrategy
from src.shared.stabilization_policy import StabilizationPolicy


_HW = HardwareProfile(
    simd_width=4,
    cache_line_bytes=64,
    max_reduce_fan_in=256,
    max_local_mem_bytes=None,
    global_mem_bytes=4 * 1024**3,
)

_ADAM_NODE_IDS = ("adam_update_shared", "adam_update_module", "adam_update_temps")


def _make_spec() -> ModelSpec:
    return ModelSpec.float32(
        input_dim=4, hidden_dim=8, output_classes=3,
        num_modules=4, simd_width=4, cache_line_bytes=64,
    )


def _make_policy(spec: ModelSpec) -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=0.1,
        compute_fp_format_max=spec.precision.compute_fp_format_max,
    )


def _get_adam_scalars(plan, node_id: str) -> dict:
    """Extract scalar parameters from an adam_update plan node."""
    return plan.nodes[node_id].scalar_params


@pytest.mark.tier1
class TestBuildLearnPlanDefaults:
    """Verify build_learn_plan() without optimizer produces legacy values."""

    def test_default_scalars_match_hardcoded(self):
        """Default OptimizerConfig produces identical scalars to legacy."""
        spec = _make_spec()
        policy = _make_policy(spec)

        plan = build_learn_plan(spec, _HW, PlanCceStrategy(), 16, policy)

        for node_id in _ADAM_NODE_IDS:
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["learning_rate"] == 0.001
            assert scalars["beta1"] == 0.9
            assert scalars["beta2"] == 0.999
            assert scalars["beta1_pow_t"] == 0.9
            assert scalars["beta2_pow_t"] == 0.999
            assert scalars["epsilon"] == spec.precision.compute_epsilon

    def test_explicit_none_matches_omitted(self):
        """optimizer=None produces same plan as omitting the argument."""
        spec = _make_spec()
        policy = _make_policy(spec)

        plan_omitted = build_learn_plan(spec, _HW, PlanCceStrategy(), 16, policy)
        plan_none = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=None,
        )

        for node_id in _ADAM_NODE_IDS:
            if node_id not in plan_omitted.nodes:
                continue
            assert (
                _get_adam_scalars(plan_omitted, node_id)
                == _get_adam_scalars(plan_none, node_id)
            )


@pytest.mark.tier1
class TestBuildLearnPlanCustomOptimizer:
    """Verify custom OptimizerConfig values appear in plan nodes."""

    def test_custom_learning_rate(self):
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(learning_rate=0.01)

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in _ADAM_NODE_IDS:
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["learning_rate"] == 0.01
            # beta values remain default
            assert scalars["beta1"] == 0.9
            assert scalars["beta2"] == 0.999

    def test_custom_betas(self):
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(beta1=0.95, beta2=0.9999)

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in _ADAM_NODE_IDS:
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["beta1"] == 0.95
            assert scalars["beta2"] == 0.9999
            assert scalars["beta1_pow_t"] == 0.95   # beta1 ** 1
            assert scalars["beta2_pow_t"] == 0.9999  # beta2 ** 1

    def test_custom_epsilon(self):
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(epsilon=1e-7)

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in _ADAM_NODE_IDS:
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["epsilon"] == 1e-7

    def test_all_custom_values(self):
        """All four optimizer parameters overridden simultaneously."""
        spec = _make_spec()
        policy = _make_policy(spec)
        opt = OptimizerConfig(
            learning_rate=0.1, beta1=0.85, beta2=0.9995, epsilon=1e-6,
        )

        plan = build_learn_plan(
            spec, _HW, PlanCceStrategy(), 16, policy, optimizer=opt,
        )

        for node_id in _ADAM_NODE_IDS:
            if node_id not in plan.nodes:
                continue
            scalars = _get_adam_scalars(plan, node_id)
            assert scalars["learning_rate"] == 0.1
            assert scalars["beta1"] == 0.85
            assert scalars["beta2"] == 0.9995
            assert scalars["beta1_pow_t"] == 0.85
            assert scalars["beta2_pow_t"] == 0.9995
            assert scalars["epsilon"] == 1e-6
