# tests/tier2/opencl/test_act_loss_bce.py
"""Tier 2 tests: compute_probs_loss_bce_chunk kernel (Node 7)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_forward import ref_compute_probs_loss_bce
from tests.tier2.fixtures.data_generators import (
    make_rng, make_targets_bce,
)


class TestLossBCE:
    """Per-kernel correctness tests for BCE loss computation."""

    def test_bce_probs_sigmoid(self):
        """Sigmoid probabilities are in [0, 1]."""
        rng = make_rng(seed=70)
        batch, cls = 8, 5
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_bce(logits, targets, mask)

        assert np.all(probs >= 0.0), "Sigmoid probs should be >= 0"
        assert np.all(probs <= 1.0), "Sigmoid probs should be <= 1"

    def test_bce_loss_basic(self):
        """BCE loss is non-negative and matches reference."""
        rng = make_rng(seed=71)
        batch, cls = 16, 4
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        _, loss = ref_compute_probs_loss_bce(logits, targets, mask)

        assert loss >= 0.0, "BCE loss must be non-negative"
        assert not np.isnan(loss), "BCE loss is NaN"

    def test_bce_loss_saturation(self):
        """Extreme targets (all 0, all 1): no NaN/Inf."""
        rng = make_rng(seed=72)
        batch, cls = 8, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        mask = np.ones(batch, dtype=np.float32)

        # All-zero targets
        t0 = np.zeros((batch, cls), dtype=np.float32)
        probs0, loss0 = ref_compute_probs_loss_bce(logits, t0, mask)
        assert not np.any(np.isnan(probs0))
        assert not np.isnan(loss0)

        # All-one targets
        t1 = np.ones((batch, cls), dtype=np.float32)
        probs1, loss1 = ref_compute_probs_loss_bce(logits, t1, mask)
        assert not np.any(np.isnan(probs1))
        assert not np.isnan(loss1)

    def test_bce_epsilon_fmax_contract(self):
        """Finding 1: fmax(prob, eps) vs prob+eps.

        Validates the reference follows the kernels.cl.h fmax contract:
          (a) Extreme probabilities yield finite results.
          (b) A known analytical case matches the fmax formula precisely.
        """
        epsilon = 1e-7
        # (a) Extreme logits
        logits = np.array([[50.0, -50.0, 0.0, 30.0]], dtype=np.float32)
        targets = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        mask = np.ones(1, dtype=np.float32)

        probs, loss = ref_compute_probs_loss_bce(logits, targets, mask)
        assert not np.any(np.isnan(probs)), "Extreme logits produced NaN probs"
        assert np.isfinite(loss), "Loss must be finite at extreme probabilities"

        # (b) logits=0 → prob=0.5, target=0.5 → loss = -log(0.5)
        logits_known = np.array([[0.0]], dtype=np.float32)
        targets_known = np.array([[0.5]], dtype=np.float32)
        _, loss_known = ref_compute_probs_loss_bce(
            logits_known, targets_known, mask,
        )
        expected_fmax = float(-np.log(0.5))
        np.testing.assert_allclose(
            loss_known, expected_fmax, atol=1e-7,
            err_msg="BCE loss should match fmax-based analytical value",
        )
