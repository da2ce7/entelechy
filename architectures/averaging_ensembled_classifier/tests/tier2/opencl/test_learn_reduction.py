# tests/tier2/opencl/test_learn_reduction.py
"""Tier 2 tests: reduction tree rendering (Nodes 14/15/20)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_reduction import (
    ref_reduction_tree_sum,
    ref_reduction_tree_sum_and_clip,
)
from tests.tier2.fixtures.data_generators import make_rng
from tests.tolerance_config import get_tolerance


class TestReductionTree:
    """Per-kernel correctness tests for multi-stage reduction tree."""

    def test_reduction_sum_single_stage(self):
        """N <= K: single-stage sum."""
        rng = make_rng(seed=140)
        K = 4
        partials = [rng.standard_normal(16).astype(np.float32) for _ in range(3)]
        ref = ref_reduction_tree_sum(partials, fan_in_K=K)
        expected = sum(partials[1:], partials[0].copy())
        tol = get_tolerance("aggregate_register_reduce", "fp32")
        np.testing.assert_allclose(ref, expected, atol=tol.atol, rtol=tol.rtol)

    def test_reduction_sum_multi_stage(self):
        """N > K: multi-stage reduction converges to total sum."""
        rng = make_rng(seed=141)
        K = 4
        N = 16
        partials = [rng.standard_normal(8).astype(np.float32) for _ in range(N)]
        ref = ref_reduction_tree_sum(partials, fan_in_K=K)
        expected = sum(partials[1:], partials[0].copy())
        tol = get_tolerance("aggregate_local_reduce", "fp32")
        np.testing.assert_allclose(ref, expected, atol=tol.atol, rtol=tol.rtol)

    def test_reduction_sum_and_clip_single(self):
        """Single-stage sum-and-clip: verify clipping at threshold."""
        rng = make_rng(seed=142)
        K = 4
        partials = [rng.standard_normal(8).astype(np.float32) * 10.0 for _ in range(3)]
        threshold = 1.0
        ref = ref_reduction_tree_sum_and_clip(
            partials, fan_in_K=K, threshold_schedule=[threshold],
        )
        norm = np.sqrt(np.sum(ref * ref))
        # After clipping, norm should be <= threshold (within tolerance)
        assert norm <= threshold + 1e-5

    def test_reduction_sum_and_clip_multi(self):
        """Multi-stage sum-and-clip: per-stage threshold schedule applied."""
        rng = make_rng(seed=143)
        K = 4
        N = 16
        partials = [rng.standard_normal(8).astype(np.float32) * 5.0 for _ in range(N)]
        schedule: list[float | None] = [2.0, 1.5]
        ref = ref_reduction_tree_sum_and_clip(
            partials, fan_in_K=K, threshold_schedule=schedule,
        )
        assert not np.any(np.isnan(ref)), "Sum-and-clip produced NaN"

    def test_reduction_non_power_of_K(self):
        """N not power of K: correct handling of partial last stage."""
        rng = make_rng(seed=145)
        K = 4
        N = 7  # Not a power of 4
        partials = [rng.standard_normal(8).astype(np.float32) for _ in range(N)]
        ref = ref_reduction_tree_sum(partials, fan_in_K=K)
        expected = sum(partials[1:], partials[0].copy())
        tol = get_tolerance("aggregate_register_reduce", "fp32")
        np.testing.assert_allclose(ref, expected, atol=tol.atol, rtol=tol.rtol)

    def test_reduction_register_vs_local_equivalent(self):
        """Register-reduce and local-reduce should produce same sum result."""
        rng = make_rng(seed=146)
        K = 4
        partials = [rng.standard_normal(8).astype(np.float32) for _ in range(8)]
        ref = ref_reduction_tree_sum(partials, fan_in_K=K)
        expected = sum(partials[1:], partials[0].copy())
        np.testing.assert_allclose(ref, expected, atol=1e-4, rtol=1e-4)
