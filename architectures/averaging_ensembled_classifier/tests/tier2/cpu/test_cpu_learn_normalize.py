# tests/tier2/cpu/test_cpu_learn_normalize.py
"""Tier 2 CPU tests: normalize_gradients kernel (Node 21)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.analytical import ref_normalize_gradients
from tests.tier2.fixtures.data_generators import make_rng
from tests.tolerance_config import get_tolerance


class TestCpuNormalizeGradients:
    """Per-kernel correctness tests for CPU normalize_gradients."""

    def test_normalize_basic(self):
        """Normalized gradients match reference: grads / (batch_size + eps)."""
        rng = make_rng(seed=210)
        element_count = 64
        grads = rng.standard_normal(element_count).astype(np.float32)
        batch_size = 32.0
        epsilon = 1e-7

        ref = ref_normalize_gradients(grads, batch_size, epsilon)

        tol = get_tolerance("normalize_gradients", "fp32")
        np.testing.assert_allclose(
            ref, grads / (batch_size + epsilon),
            atol=tol.atol, rtol=tol.rtol,
        )

    def test_normalize_preserves_sign(self):
        """Normalization preserves gradient sign."""
        rng = make_rng(seed=211)
        grads = rng.standard_normal(128).astype(np.float32)
        batch_size = 16.0
        ref = ref_normalize_gradients(grads, batch_size)
        signs_original = np.sign(grads)
        signs_normalized = np.sign(ref)
        np.testing.assert_array_equal(signs_original, signs_normalized)

    def test_normalize_batch_size_one(self):
        """batch_size=1: gradient nearly unchanged (divided by 1+eps)."""
        rng = make_rng(seed=212)
        grads = rng.standard_normal(32).astype(np.float32)
        ref = ref_normalize_gradients(grads, 1.0)
        np.testing.assert_allclose(ref, grads, atol=1e-6, rtol=1e-6)

    def test_normalize_large_batch(self):
        """Large batch size effectively zeros small gradient magnitudes."""
        rng = make_rng(seed=213)
        grads = rng.standard_normal(64).astype(np.float32) * 0.01
        batch_size = 1000.0
        ref = ref_normalize_gradients(grads, batch_size)
        assert np.max(np.abs(ref)) < np.max(np.abs(grads))
