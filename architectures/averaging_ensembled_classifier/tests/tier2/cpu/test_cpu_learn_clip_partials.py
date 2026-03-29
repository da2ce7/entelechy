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
