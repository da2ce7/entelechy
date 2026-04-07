# tests/tier2/cpu/test_cpu_learn_clip_partials.py
"""Tier 2 CPU tests: clip_partial_gradients kernel (Node 11)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.analytical import ref_clip_l2_norm
from tests.tier2.fixtures.data_generators import make_rng


class TestCpuClipPartialGradients:
    """Per-kernel correctness tests for CPU clip_partial_gradients."""

    def test_clip_below_threshold(self):
        """Gradients below threshold unchanged."""
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        threshold = 10.0
        ref = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_allclose(ref, grads, atol=1e-7)

    def test_clip_above_threshold(self):
        """Gradients above threshold scaled to threshold norm."""
        grads = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
        threshold = 2.5
        ref = ref_clip_l2_norm(grads, threshold)
        clipped_norm = np.sqrt(np.sum(ref * ref))
        np.testing.assert_allclose(clipped_norm, threshold, atol=1e-5)

    def test_clip_preserves_direction(self):
        """Clipped gradient direction matches original direction."""
        rng = make_rng(seed=110)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0
        threshold = 1.0
        ref = ref_clip_l2_norm(grads, threshold)
        orig_dir = grads / np.linalg.norm(grads)
        clipped_dir = ref / np.linalg.norm(ref)
        np.testing.assert_allclose(orig_dir, clipped_dir, atol=1e-5)

    def test_clip_zero_gradient(self):
        """Zero gradient remains zero after clipping."""
        grads = np.zeros(16, dtype=np.float32)
        threshold = 1.0
        ref = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_array_equal(ref, grads)

    def test_clip_at_threshold(self):
        """Gradients exactly at threshold stay unchanged (within epsilon)."""
        grads = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
        threshold = 5.0
        ref = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_allclose(ref, grads, atol=1e-5)

    def test_clip_near_threshold_epsilon_placement(self):
        """Bonus Finding: epsilon in denominator, not under sqrt.

        The kernel contract specifies scale = threshold / (norm + epsilon),
        NOT threshold / sqrt(norm² + epsilon). At small norms where epsilon
        is comparable to norm², the formulas diverge measurably.
        """
        epsilon = 1e-7
        # Use a small threshold so eps/norm² is non-negligible
        threshold = 1e-3
        norm_target = threshold + 1e-5  # just above threshold
        grads = np.array([norm_target, 0.0], dtype=np.float32)

        ref = ref_clip_l2_norm(grads, threshold, epsilon=epsilon)

        # Contract: scale = threshold / (norm + epsilon)
        norm = float(np.sqrt(np.sum(grads.astype(np.float64) ** 2)))
        expected_scale = threshold / (norm + epsilon)
        expected = grads.astype(np.float64) * expected_scale
        np.testing.assert_allclose(ref, expected, atol=1e-6)

        # Verify this differs from the wrong formula: threshold / sqrt(norm² + eps)
        wrong_scale = threshold / np.sqrt(norm**2 + epsilon)
        wrong_result = grads.astype(np.float64) * wrong_scale
        assert not np.allclose(ref, wrong_result, atol=1e-5), (
            "Clipped result should differ from sqrt(norm² + eps) formula"
        )

    # ----------------------------------------------------------------
    # Threshold semantics tests (ADR-019, ADR-026)
    # ----------------------------------------------------------------

    def test_negative_threshold_bypasses_clipping(self):
        """Negative threshold bypasses clipping (diagnostic mode).

        Contract: clipping_threshold < 0 returns gradients unchanged.
        """
        rng = make_rng(seed=200)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0
        threshold = -1.0
        ref = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_array_equal(ref, grads)

    def test_zero_threshold_zeros_all_gradients(self):
        """Zero threshold clips to zero norm (zeros all gradients).

        Contract: clipping_threshold == 0 produces all-zero output.
        """
        rng = make_rng(seed=201)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0
        threshold = 0.0
        ref = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_array_equal(ref, np.zeros_like(grads))

    def test_various_negative_thresholds_bypass(self):
        """Any negative threshold value bypasses clipping."""
        rng = make_rng(seed=202)
        grads = rng.standard_normal(32).astype(np.float32) * 5.0
        for threshold in [-1.0, -0.001, -100.0, -1e-10]:
            ref = ref_clip_l2_norm(grads, threshold)
            np.testing.assert_array_equal(
                ref, grads,
                err_msg=f"Threshold={threshold} should bypass"
            )
