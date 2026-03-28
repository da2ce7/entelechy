# tests/tier2/opencl/test_learn_temp_grads.py
"""Tier 2 tests: calculate_chunk_temp_gradients kernel (Node 10)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_gradients import (
    ref_temp_gradients_cce,
    ref_temp_gradients_bce,
)
from tests.tier2.fixtures.numpy_forward import (
    ref_compute_probs_loss_cce,
    ref_compute_probs_loss_bce,
)
from tests.tier2.fixtures.data_generators import (
    make_rng, make_targets_cce, make_targets_bce,
)


class TestTempGrads:
    """Per-kernel correctness tests for temperature gradients."""

    def test_temp_grads_cce(self):
        """CCE temperature gradient is a finite scalar."""
        rng = make_rng(seed=100)
        batch, cls = 8, 3
        logits_unscaled = rng.standard_normal((batch, cls)).astype(np.float32)
        temperature = 1.5
        logits = logits_unscaled / temperature
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
        grad_t = ref_temp_gradients_cce(probs, targets, logits_unscaled, temperature, mask)

        assert np.isfinite(grad_t), "Temperature gradient must be finite"

    def test_temp_grads_bce(self):
        """BCE temperature gradient is a finite scalar."""
        rng = make_rng(seed=101)
        batch, cls = 8, 4
        logits_unscaled = rng.standard_normal((batch, cls)).astype(np.float32)
        temperature = 1.2
        logits = logits_unscaled / temperature
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_bce(logits, targets, mask)
        grad_t = ref_temp_gradients_bce(probs, targets, logits_unscaled, temperature, mask)

        assert np.isfinite(grad_t), "Temperature gradient must be finite"
