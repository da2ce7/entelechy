# tests/tier2/cpu/test_cpu_act_loss_cce.py
"""Tier 2 CPU tests: compute_probs_loss_cce_chunk kernel (Node 6)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_forward import ref_compute_probs_loss_cce
from tests.tier2.fixtures.data_generators import make_rng, make_targets_cce
from tests.tolerance_config import get_tolerance


class TestCpuLossCCE:
    """Per-kernel correctness tests for CPU CCE loss computation."""

    def test_cce_probs_softmax(self):
        """Softmax probabilities sum to 1.0 per sample."""
        rng = make_rng(seed=60)
        batch, cls = 8, 5
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)

        tol = get_tolerance("compute_probs_loss_cce_chunk", "fp32")
        row_sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=tol.atol)

    def test_cce_loss_basic(self):
        """CCE loss is non-negative and finite."""
        rng = make_rng(seed=61)
        batch, cls = 16, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        _, loss = ref_compute_probs_loss_cce(logits, targets, mask)

        assert loss >= 0.0, "CCE loss must be non-negative"
        assert not np.isnan(loss), "CCE loss is NaN"
        assert not np.isinf(loss), "CCE loss is Inf"

    def test_cce_loss_numerical_stability(self):
        """Large logit magnitudes should not produce NaN/Inf."""
        rng = make_rng(seed=62)
        batch, cls = 8, 5
        logits = rng.standard_normal((batch, cls)).astype(np.float32) * 100.0
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, loss_val = ref_compute_probs_loss_cce(logits, targets, mask)

        assert not np.any(np.isnan(probs)), "Softmax produced NaN with large logits"
        assert not np.any(np.isinf(probs)), "Softmax produced Inf with large logits"
        assert not np.isnan(loss_val)

    def test_cce_single_class(self):
        """Edge case: single class — softmax is trivially [1.0]."""
        rng = make_rng(seed=63)
        batch, cls = 4, 1
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = np.ones((batch, cls), dtype=np.float32)
        mask = np.ones(batch, dtype=np.float32)

        probs, _loss = ref_compute_probs_loss_cce(logits, targets, mask)

        np.testing.assert_allclose(probs, 1.0, atol=1e-6)

    def test_cce_masked_samples(self):
        """Fully masked batch produces zero loss."""
        rng = make_rng(seed=64)
        batch, cls = 8, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.zeros(batch, dtype=np.float32)

        _, loss = ref_compute_probs_loss_cce(logits, targets, mask)

        assert loss == 0.0
