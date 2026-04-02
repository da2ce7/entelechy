# tests/tier3/test_parity_e2e.py
"""Tier 3: end-to-end cross-backend parity tests (ADR-016).

Executes complete Act and Learn ExecutionPlans on both the oracle and
comparison backends, comparing plan-level observable outputs within
Tier 3 tolerances.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.tolerance_config import get_tier3_tolerance

from .conftest import _get_available_backends, _select_oracle


def _build_comparison_pairs() -> list[tuple[str, str]]:
    """Build comparison pairs for parametrize at module load time."""
    available = _get_available_backends()
    oracle = _select_oracle()
    pairs: list[tuple[str, str]] = []
    if oracle is not None:
        for b in available:
            if b != oracle:
                pairs.append((oracle, b))
    else:
        for left, right in __import__("itertools").combinations(available, 2):
            pairs.append((left, right))
    return pairs


_PAIRS = _build_comparison_pairs()

# Iris-scale FP32 model geometry for end-to-end tests.
_IRIS = dict(input_dim=4, hidden_dim=32, output_classes=3, num_modules=8)
_BATCH_SIZE = 150


def _build_act_plan(problem_type: str) -> Any:
    """Build an Iris-scale Act-phase ExecutionPlan."""
    from src.shared.hardware_profile import HardwareProfile
    from src.shared.model_spec import ModelSpec
    from src.shared.plan_builder import build_act_plan
    from src.shared.precision_config import PrecisionConfig
    from src.shared.problem_type_strategy import PlanBceStrategy, PlanCceStrategy

    prec = PrecisionConfig.float32()
    spec = ModelSpec(precision=prec, simd_width=4, cache_line_bytes=64, **_IRIS)
    hw = HardwareProfile(
        simd_width=4, cache_line_bytes=64, max_reduce_fan_in=256,
        max_local_mem_bytes=65536, global_mem_bytes=4 * 1024**3,
    )
    strategy = PlanCceStrategy() if problem_type == "CCE" else PlanBceStrategy()
    return build_act_plan(spec, hw, strategy, _BATCH_SIZE)


def _build_learn_plan(problem_type: str) -> Any:
    """Build an Iris-scale Learn-phase ExecutionPlan."""
    from src.shared.hardware_profile import HardwareProfile
    from src.shared.model_spec import ModelSpec
    from src.shared.plan_builder import build_learn_plan
    from src.shared.precision_config import PrecisionConfig
    from src.shared.problem_type_strategy import PlanBceStrategy, PlanCceStrategy
    from src.shared.stabilization_policy import StabilizationPolicy

    prec = PrecisionConfig.float32()
    spec = ModelSpec(precision=prec, simd_width=4, cache_line_bytes=64, **_IRIS)
    hw = HardwareProfile(
        simd_width=4, cache_line_bytes=64, max_reduce_fan_in=256,
        max_local_mem_bytes=65536, global_mem_bytes=4 * 1024**3,
    )
    strategy = PlanCceStrategy() if problem_type == "CCE" else PlanBceStrategy()
    policy = StabilizationPolicy(
        t_algorithmic=1.0, lambda_=1.0,
        compute_fp_format_max=float(np.finfo(np.float32).max),
    )
    return build_learn_plan(spec, hw, strategy, _BATCH_SIZE, policy)


# ---------------------------------------------------------------------------
# End-to-end parity tests
# ---------------------------------------------------------------------------


@pytest.mark.tier3
class TestParityE2E:
    """End-to-end cross-backend parity: complete Act and Learn cycles.

    Both backends render the same immutable ExecutionPlan starting from
    identical (zero-initialized) buffer state. The test compares
    plan-level observable outputs via RetrievalFuture.result().

    Full-fidelity parity with meaningful initial data (weights, inputs)
    requires a buffer injection API on the renderer. The zero-initialized
    comparison still validates:
    - Identical kernel dispatch ordering
    - Consistent buffer lifecycle handling
    - Matching reduction and streaming loop implementations
    - Softmax/loss computation on uniform inputs
    """

    @pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_act_plan_parity(
        self,
        problem_type: str,
        pair: tuple[str, str],
        renderer_factory: Any,
    ) -> None:
        """Act-phase parity: final probabilities match across backends."""
        oracle_name, comp_name = pair
        plan = _build_act_plan(problem_type)

        oracle_renderer = renderer_factory(oracle_name)
        comp_renderer = renderer_factory(comp_name)

        oracle_futures = oracle_renderer.render(plan)
        comp_futures = comp_renderer.render(plan)

        # Compare all retrieval futures that both backends produce.
        tol = get_tier3_tolerance("act_plan_e2e")
        for event_name in oracle_futures:
            if event_name not in comp_futures:
                pytest.fail(
                    f"Oracle produced event '{event_name}' but comparison "
                    f"backend '{comp_name}' did not"
                )
            oracle_val = oracle_futures[event_name].result()
            comp_val = comp_futures[event_name].result()

            np.testing.assert_allclose(
                comp_val, oracle_val,
                atol=tol.atol, rtol=tol.rtol,
                err_msg=(
                    f"Act-phase parity failure: {comp_name} vs {oracle_name} "
                    f"for event '{event_name}' ({problem_type})"
                ),
            )

    @pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_learn_plan_parity(
        self,
        problem_type: str,
        pair: tuple[str, str],
        renderer_factory: Any,
    ) -> None:
        """Learn-phase parity: updated parameters match across backends."""
        oracle_name, comp_name = pair
        plan = _build_learn_plan(problem_type)

        oracle_renderer = renderer_factory(oracle_name)
        comp_renderer = renderer_factory(comp_name)

        oracle_futures = oracle_renderer.render(plan)
        comp_futures = comp_renderer.render(plan)

        tol = get_tier3_tolerance("learn_plan_e2e")
        for event_name in oracle_futures:
            if event_name not in comp_futures:
                pytest.fail(
                    f"Oracle produced event '{event_name}' but comparison "
                    f"backend '{comp_name}' did not"
                )
            oracle_val = oracle_futures[event_name].result()
            comp_val = comp_futures[event_name].result()

            np.testing.assert_allclose(
                comp_val, oracle_val,
                atol=tol.atol, rtol=tol.rtol,
                err_msg=(
                    f"Learn-phase parity failure: {comp_name} vs {oracle_name} "
                    f"for event '{event_name}' ({problem_type})"
                ),
            )

    @pytest.mark.slow
    @pytest.mark.parametrize("problem_type", ["CCE", "BCE"])
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_stress_act_plan_parity(
        self,
        problem_type: str,
        pair: tuple[str, str],
        renderer_factory: Any,
    ) -> None:
        """Stress-scale Act parity: larger model exercises deep reduction trees."""
        from src.shared.hardware_profile import HardwareProfile
        from src.shared.model_spec import ModelSpec
        from src.shared.plan_builder import build_act_plan
        from src.shared.precision_config import PrecisionConfig
        from src.shared.problem_type_strategy import PlanBceStrategy, PlanCceStrategy

        oracle_name, comp_name = pair
        prec = PrecisionConfig.float32()
        spec = ModelSpec(
            precision=prec, input_dim=128, hidden_dim=512,
            output_classes=100, num_modules=32,
            simd_width=16, cache_line_bytes=64,
        )
        hw = HardwareProfile(
            simd_width=16, cache_line_bytes=64, max_reduce_fan_in=256,
            max_local_mem_bytes=65536, global_mem_bytes=4 * 1024**3,
        )
        strategy = PlanCceStrategy() if problem_type == "CCE" else PlanBceStrategy()
        plan = build_act_plan(spec, hw, strategy, batch_size=1024)

        oracle_renderer = renderer_factory(oracle_name)
        comp_renderer = renderer_factory(comp_name)

        oracle_futures = oracle_renderer.render(plan)
        comp_futures = comp_renderer.render(plan)

        tol = get_tier3_tolerance("act_plan_e2e")
        for event_name in oracle_futures:
            if event_name not in comp_futures:
                pytest.fail(
                    f"Stress Act parity: oracle event '{event_name}' "
                    f"missing from {comp_name}"
                )
            oracle_val = oracle_futures[event_name].result()
            comp_val = comp_futures[event_name].result()

            np.testing.assert_allclose(
                comp_val, oracle_val,
                atol=tol.atol, rtol=tol.rtol,
                err_msg=(
                    f"Stress Act-phase parity failure: {comp_name} vs "
                    f"{oracle_name} for '{event_name}' ({problem_type})"
                ),
            )
