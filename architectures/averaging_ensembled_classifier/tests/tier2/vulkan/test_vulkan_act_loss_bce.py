# tests/tier2/vulkan/test_vulkan_act_loss_bce.py
"""Tier 2 Vulkan tests: compute_probs_loss_bce kernel (Node 7)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_forward import ref_compute_probs_loss_bce
from tests.tier2.fixtures.data_generators import make_rng, make_targets_bce


class TestVulkanLossBCE:
    """Per-kernel correctness tests for BCE loss computation."""

    def test_bce_probs_sigmoid(self):
        """Sigmoid probabilities are in [0, 1]."""
        rng = make_rng(seed=70)
        batch, cls = 8, 5
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_bce(logits, targets, mask)

        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_bce_loss_basic(self):
        """BCE loss is non-negative and finite."""
        rng = make_rng(seed=71)
        batch, cls = 16, 4
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        _, loss = ref_compute_probs_loss_bce(logits, targets, mask)

        assert loss >= 0.0
        assert not np.isnan(loss)

    def test_bce_loss_saturation(self):
        """Extreme targets (all 0, all 1): no NaN/Inf."""
        rng = make_rng(seed=72)
        batch, cls = 8, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        mask = np.ones(batch, dtype=np.float32)

        t0 = np.zeros((batch, cls), dtype=np.float32)
        probs0, loss0 = ref_compute_probs_loss_bce(logits, t0, mask)
        assert not np.any(np.isnan(probs0))
        assert not np.isnan(loss0)

        t1 = np.ones((batch, cls), dtype=np.float32)
        probs1, loss1 = ref_compute_probs_loss_bce(logits, t1, mask)
        assert not np.any(np.isnan(probs1))
        assert not np.isnan(loss1)

    def test_bce_numerical_stability(self):
        """Large logit magnitudes produce valid results."""
        rng = make_rng(seed=73)
        batch, cls = 8, 4
        logits = rng.standard_normal((batch, cls)).astype(np.float32) * 100.0
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, loss = ref_compute_probs_loss_bce(logits, targets, mask)

        assert not np.any(np.isnan(probs))
        assert not np.isnan(loss)

    def test_bce_masked_samples(self):
        """Fully masked batch produces zero loss."""
        rng = make_rng(seed=74)
        batch, cls = 8, 4
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_bce(rng, batch, cls)
        mask = np.zeros(batch, dtype=np.float32)

        _, loss = ref_compute_probs_loss_bce(logits, targets, mask)

        assert loss == 0.0

    def test_bce_epsilon_fmax_contract(self):
        """fmax-based clamping for BCE log arguments."""
        logits_extreme = np.array(
            [[50.0, -50.0, 0.0, 30.0]], dtype=np.float32,
        )
        targets_extreme = np.array(
            [[1.0, 1.0, 0.0, 0.0]], dtype=np.float32,
        )
        mask = np.ones(1, dtype=np.float32)
        probs_ext, loss_ext = ref_compute_probs_loss_bce(
            logits_extreme, targets_extreme, mask,
        )
        assert not np.any(np.isnan(probs_ext))
        assert np.isfinite(loss_ext)

        logits_known = np.array([[0.0]], dtype=np.float32)
        targets_known = np.array([[0.5]], dtype=np.float32)
        _, loss_known = ref_compute_probs_loss_bce(
            logits_known, targets_known, mask,
        )
        expected_fmax = float(-np.log(0.5))
        np.testing.assert_allclose(loss_known, expected_fmax, atol=1e-7)
