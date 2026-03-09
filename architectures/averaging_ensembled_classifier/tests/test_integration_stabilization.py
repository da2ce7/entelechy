# tests/test_integration_stabilization.py
from __future__ import annotations

"""
Integration Tests: StabilizationPolicy + Reduction Planning.

These tests verify the critical interplay between the StabilizationPolicy
and the reduction-tree planning logic that underpins the system's gradient
stabilization strategy (Nodes 15, 16, 20 as described in CONCEPT.md).

Target CONCEPT.md Validation Scenarios:
  - The Rodeo (Extreme Instability Resilience): FP16 safety thresholds
  - The Hydra (Massive num_heads): large reduction trees
  - The Marathon (Massive epochs): long-term numerical stability
  - The Data Tsunami (Massive batch_size): deep reduction trees

No OpenCL device is required.
"""

import math

import numpy as np
import pytest

from src.stabilization_policy import StabilizationPolicy

# =========================================================================
# 1. Quadratic Scaling Policy – Core Correctness
# =========================================================================


class TestQuadraticScalingPolicy:
    """Verify the T_j = T_algorithmic + λ·j² formula and safety clamping."""

    def test_root_stage_equals_t_algorithmic(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """At j=0 (root), the policy threshold should equal T_algorithmic."""
        policy = default_stabilization_policy
        # For a large-enough K the safety ceiling won't clamp
        threshold = policy.get_threshold_for_generic_stage(stage_j=0, runtime_fan_in_k=4)
        assert threshold == pytest.approx(policy.t_algorithmic, rel=1e-6)

    def test_threshold_increases_with_stage_depth(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """Deeper stages (j > 0) should have monotonically higher thresholds."""
        policy = default_stabilization_policy
        thresholds = [policy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=4) for j in range(5)]
        for i in range(1, len(thresholds)):
            assert (
                thresholds[i] >= thresholds[i - 1]
            ), f"Threshold at j={i} ({thresholds[i]}) < j={i-1} ({thresholds[i-1]})"

    def test_quadratic_formula_exact(self):
        """Verify exact T_j = T_algo + λ·j² for a known configuration."""
        policy = StabilizationPolicy(
            t_algorithmic=2.0,
            lambda_=0.5,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        for j in range(6):
            expected = 2.0 + 0.5 * (j**2)
            actual = policy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=4)
            assert actual == pytest.approx(expected, rel=1e-6), f"Mismatch at j={j}"

    def test_lambda_zero_gives_fixed_ceiling(self):
        """λ=0 → Fixed Ceiling strategy: all stages get T_algorithmic."""
        policy = StabilizationPolicy(
            t_algorithmic=5.0,
            lambda_=0.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        for j in range(10):
            threshold = policy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=4)
            assert threshold == pytest.approx(5.0, rel=1e-6)


# =========================================================================
# 2. Safety Ceiling (T_safety = FP_FORMAT_MAX / K)
# =========================================================================


class TestSafetyCeiling:
    """Verify that the hardware-aware safety ceiling is correctly applied."""

    def test_safety_ceiling_clamps_large_policy(self):
        """When the raw policy far exceeds fp_format_max, the final threshold
        is clamped below it.  The internal K-validation may adjust the
        effective fan-in, but the universal invariant holds:
        threshold <= fp_format_max / 2  (since validated K >= 2 always).
        """
        fp16_max = float(np.finfo(np.float16).max)  # ~65504
        policy = StabilizationPolicy(
            t_algorithmic=60000.0,  # Near FP16 max
            lambda_=10000.0,
            fp_format_max=fp16_max,
        )
        raw_policy = 60000.0 + 10000.0 * (3**2)  # 150000
        threshold = policy.get_threshold_for_generic_stage(stage_j=3, runtime_fan_in_k=4)
        # Safety mechanism must clamp well below the unchecked policy value
        assert threshold < raw_policy, "Safety should clamp excessive policy"
        # Universal invariant: threshold never exceeds fp_max / 2
        assert threshold <= fp16_max / 2 + 1.0

    def test_safety_threshold_bounded_by_half_fp_max(self, fp16_stabilization_policy: StabilizationPolicy) -> None:
        """Threshold never exceeds fp_format_max / 2 regardless of K,
        because the internal K-validation enforces K >= 2."""
        policy = fp16_stabilization_policy
        fp_max = policy.fp_format_max
        for j in [0, 1, 10, 50, 100]:
            for k in [2, 4, 8, 16, 64]:
                t = policy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=k)
                assert t <= fp_max / 2 + 1.0, f"j={j}, K={k}: threshold {t} > fp_max/2 ({fp_max / 2})"


# =========================================================================
# 3. Leaf-Level Safety Gate (Nodes 11 & 19)
# =========================================================================


class TestLeafSafetyGate:
    """Verify the leaf-level pre-reduction safety threshold."""

    def test_leaf_threshold_below_fp_max(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """The 10% safety margin must be respected."""
        policy = default_stabilization_policy
        leaf_t = policy.get_leaf_safety_threshold()
        assert leaf_t < policy.fp_format_max
        assert leaf_t == pytest.approx(policy.fp_format_max * 0.9, rel=1e-6)

    def test_fp16_leaf_threshold_within_range(self, fp16_stabilization_policy: StabilizationPolicy) -> None:
        """FP16 leaf threshold must be representable in FP16."""
        leaf_t = fp16_stabilization_policy.get_leaf_safety_threshold()
        fp16_max = float(np.finfo(np.float16).max)
        assert leaf_t < fp16_max
        assert leaf_t > 0


# =========================================================================
# 4. Reduction Tree Planning
# =========================================================================


class TestReductionTreePlanning:
    """Verify the strategic reduction planner that computes (K, num_stages)."""

    def test_trivial_case_single_partial(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """A single partial requires no reduction."""
        _k, stages = default_stabilization_policy.plan_uniform_reduction_tree(
            num_partials=1,
            hardware_max_fan_in=256,
        )
        assert stages == 0

    def test_k_respects_hardware_limit(self, default_stabilization_policy: StabilizationPolicy) -> None:
        hw_max = 16
        k, _stages = default_stabilization_policy.plan_uniform_reduction_tree(
            num_partials=1000,
            hardware_max_fan_in=hw_max,
        )
        assert k <= hw_max

    def test_k_is_at_least_2(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """K must be ≥ 2 for a reduction to make sense."""
        k, _stages = default_stabilization_policy.plan_uniform_reduction_tree(
            num_partials=100,
            hardware_max_fan_in=256,
        )
        assert k >= 2

    def test_stages_are_log_k_n(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """Number of stages should approximate ceil(log_K(N))."""
        N = 1024
        k, stages = default_stabilization_policy.plan_uniform_reduction_tree(
            num_partials=N,
            hardware_max_fan_in=256,
        )
        if k > 1:
            expected_stages = math.ceil(math.log(N) / math.log(k))
            assert stages == expected_stages

    def test_data_tsunami_large_batch(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """The Data Tsunami: 10,000 partials should produce a valid tree."""
        k, stages = default_stabilization_policy.plan_uniform_reduction_tree(
            num_partials=10_000,
            hardware_max_fan_in=256,
        )
        assert k >= 2
        assert stages >= 1
        # The tree should be able to reduce all partials
        assert k**stages >= 10_000

    def test_fp16_tree_respects_mathematical_safety(self):
        """FP16's limited range constrains K (Rodeo scenario)."""
        fp16_max = float(np.finfo(np.float16).max)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=fp16_max,
        )
        k, _stages = policy.plan_uniform_reduction_tree(
            num_partials=10_000,
            hardware_max_fan_in=256,
        )
        # K must be such that fp16_max / K >= min_threshold (1.0)
        assert k <= fp16_max / policy.min_threshold


# =========================================================================
# 5. Specialized Reduction Policy K (Node 16)
# =========================================================================


class TestSpecializedReductionK:
    """Verify the pre-flight contract synthesizer for the Grad_H reduction."""

    def test_respects_user_intent(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """The result should not exceed the user's requested K."""
        user_k = 8
        final_k = default_stabilization_policy.get_specialized_reduction_policy_k(
            user_policy_k=user_k,
            hardware_max_fan_in=256,
        )
        assert final_k <= user_k

    def test_respects_hardware_limit(self, default_stabilization_policy: StabilizationPolicy) -> None:
        """The result should not exceed the hardware maximum."""
        hw_limit = 32
        final_k = default_stabilization_policy.get_specialized_reduction_policy_k(
            user_policy_k=1000,
            hardware_max_fan_in=hw_limit,
        )
        assert final_k <= hw_limit

    def test_minimum_k_is_2(self, default_stabilization_policy: StabilizationPolicy) -> None:
        final_k = default_stabilization_policy.get_specialized_reduction_policy_k(
            user_policy_k=1,
            hardware_max_fan_in=256,
        )
        assert final_k >= 2

    def test_fp16_mathematical_safety_constrains_k(self):
        """FP16 should produce a smaller K than FP32 for the same inputs."""
        fp16_policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float16).max),
        )
        fp32_policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        fp16_k = fp16_policy.get_specialized_reduction_policy_k(
            user_policy_k=10000,
            hardware_max_fan_in=10000,
        )
        fp32_k = fp32_policy.get_specialized_reduction_policy_k(
            user_policy_k=10000,
            hardware_max_fan_in=10000,
        )
        assert fp16_k <= fp32_k


# =========================================================================
# 6. Orthogonal Separation of Concerns
# =========================================================================


class TestOrthogonalSeparation:
    """
    Verify that Algorithmic Regulation and Numerical Safety operate as
    orthogonal concerns (Section 3.4 of CONCEPT.md).
    """

    def test_policy_reduces_to_safety_only_when_t_algo_is_zero(self):
        """When the user sets no algorithmic target, only hardware safety applies."""
        fp32_max = float(np.finfo(np.float32).max)
        policy = StabilizationPolicy(
            t_algorithmic=0.0,
            lambda_=0.0,
            fp_format_max=fp32_max,
        )
        threshold = policy.get_threshold_for_generic_stage(stage_j=0, runtime_fan_in_k=4)
        expected_safety = fp32_max / 4
        # Should be the safety ceiling (or min_threshold)
        assert threshold <= expected_safety

    def test_large_lambda_creates_permissive_funnel(self):
        """A large λ should yield much higher early-stage thresholds."""
        narrow = StabilizationPolicy(t_algorithmic=1.0, lambda_=0.1, fp_format_max=1e38)
        wide = StabilizationPolicy(t_algorithmic=1.0, lambda_=100.0, fp_format_max=1e38)

        t_narrow_j5 = narrow.get_threshold_for_generic_stage(stage_j=5, runtime_fan_in_k=4)
        t_wide_j5 = wide.get_threshold_for_generic_stage(stage_j=5, runtime_fan_in_k=4)
        assert t_wide_j5 > t_narrow_j5
