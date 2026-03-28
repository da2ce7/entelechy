# src/tests/test_stabilization_policy.py
"""
Unit tests for StabilizationPolicy: threshold calculations, reduction tree
planning, and edge cases.

Bug-hunting focus:
* get_threshold_for_generic_stage must produce finite, positive thresholds
* plan_uniform_reduction_tree must produce safe_k ≥ 2 and valid num_stages
* get_leaf_safety_threshold must be ≥ min_threshold
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.stabilization_policy import StabilizationPolicy


@pytest.fixture
def default_policy() -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        fp_format_max=float(np.finfo(np.float32).max),
    )


@pytest.fixture
def fp16_policy() -> StabilizationPolicy:
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        fp_format_max=float(np.finfo(np.float16).max),
    )


class TestLeafSafetyThreshold:

    def test_positive_and_finite(self, default_policy: StabilizationPolicy) -> None:
        t = default_policy.get_leaf_safety_threshold()
        assert t > 0
        assert np.isfinite(t)

    def test_at_least_min_threshold(self, default_policy: StabilizationPolicy) -> None:
        t = default_policy.get_leaf_safety_threshold()
        assert t >= default_policy.min_threshold


class TestThresholdForGenericStage:

    @pytest.mark.parametrize("stage_j", [0, 1, 3, 10])
    def test_positive_and_finite(self, default_policy: StabilizationPolicy, stage_j: int) -> None:
        t = default_policy.get_threshold_for_generic_stage(stage_j=stage_j, runtime_fan_in_k=8)
        assert t > 0, f"Threshold for stage_j={stage_j} is {t}"
        assert np.isfinite(t), f"Threshold for stage_j={stage_j} is {t}"

    def test_root_stage_is_t_algorithmic(self, default_policy: StabilizationPolicy) -> None:
        """At stage j=0, the threshold should be close to t_algorithmic."""
        t = default_policy.get_threshold_for_generic_stage(stage_j=0, runtime_fan_in_k=8)
        # Should be at least t_algorithmic (possibly clamped to min_threshold)
        assert t >= default_policy.t_algorithmic or t >= default_policy.min_threshold

    def test_leaf_stages_have_larger_threshold(self, default_policy: StabilizationPolicy) -> None:
        """Higher j (closer to leaves) should produce a larger or equal threshold."""
        t0 = default_policy.get_threshold_for_generic_stage(stage_j=0, runtime_fan_in_k=8)
        t5 = default_policy.get_threshold_for_generic_stage(stage_j=5, runtime_fan_in_k=8)
        assert t5 >= t0, f"Expected leaf threshold ({t5}) >= root ({t0})"


class TestPlanUniformReductionTree:

    def test_single_partial(self, default_policy: StabilizationPolicy) -> None:
        safe_k, stages = default_policy.plan_uniform_reduction_tree(num_partials=1, hardware_max_fan_in=128)
        assert stages == 0

    def test_safe_k_at_least_2(self, default_policy: StabilizationPolicy) -> None:
        safe_k, stages = default_policy.plan_uniform_reduction_tree(num_partials=100, hardware_max_fan_in=128)
        assert safe_k >= 2

    def test_stages_sufficient_to_reduce(self, default_policy: StabilizationPolicy) -> None:
        """safe_k^num_stages must be ≥ num_partials."""
        for n in [8, 256, 10_000]:
            safe_k, stages = default_policy.plan_uniform_reduction_tree(num_partials=n, hardware_max_fan_in=128)
            assert safe_k ** stages >= n, (
                f"n={n}: safe_k={safe_k}, stages={stages}, safe_k^stages={safe_k**stages}"
            )

    def test_fp16_is_more_conservative(self, default_policy: StabilizationPolicy, fp16_policy: StabilizationPolicy) -> None:
        """FP16's smaller fp_format_max should produce a smaller or equal safe_k."""
        k32, _ = default_policy.plan_uniform_reduction_tree(num_partials=256, hardware_max_fan_in=128)
        k16, _ = fp16_policy.plan_uniform_reduction_tree(num_partials=256, hardware_max_fan_in=128)
        assert k16 <= k32


class TestSpecializedReductionPolicyK:

    def test_respects_hardware_limit(self, default_policy: StabilizationPolicy) -> None:
        k = default_policy.get_specialized_reduction_policy_k(user_policy_k=256, hardware_max_fan_in=64)
        assert k <= 64

    def test_respects_user_intent(self, default_policy: StabilizationPolicy) -> None:
        k = default_policy.get_specialized_reduction_policy_k(user_policy_k=16, hardware_max_fan_in=128)
        assert k <= 16

    def test_minimum_is_two(self, default_policy: StabilizationPolicy) -> None:
        k = default_policy.get_specialized_reduction_policy_k(user_policy_k=1, hardware_max_fan_in=1)
        assert k >= 2
