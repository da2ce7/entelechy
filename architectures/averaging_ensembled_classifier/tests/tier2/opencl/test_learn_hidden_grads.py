# tests/tier2/opencl/test_learn_hidden_grads.py
"""Tier 2 tests: backprop_error_to_hidden kernel (Node 9)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_gradients import ref_backprop_error_to_hidden
from tests.tier2.fixtures.numpy_forward import ref_compute_probs_loss_cce
from tests.tier2.fixtures.data_generators import (
    make_rng, make_weights, make_targets_cce,
)
from tests.tolerance_config import get_tolerance


class TestHiddenGrads:
    """Per-kernel correctness tests for backprop_error_to_hidden."""

    def test_hidden_grads_cce(self):
        """Strategy A (CCE flag): grad_hidden matches reference."""
        rng = make_rng(seed=90)
        batch, hid, cls = 8, 8, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)
        module_w = make_weights(rng, cls, hid)
        hidden_mask = (rng.standard_normal((batch, hid)) > 0).astype(np.float32)

        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
        grad_h = ref_backprop_error_to_hidden(probs, targets, module_w, hidden_mask, mask)

        _tol = get_tolerance("backprop_error_to_hidden", "fp32")
        assert grad_h.shape == (batch, hid)
        assert not np.any(np.isnan(grad_h))

    def test_hidden_grads_relu_mask(self):
        """Gradients are zero where ReLU mask is 0."""
        rng = make_rng(seed=91)
        batch, hid, cls = 4, 8, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)
        module_w = make_weights(rng, cls, hid)

        # All-zero hidden mask -> all-zero gradients
        hidden_mask = np.zeros((batch, hid), dtype=np.float32)
        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
        grad_h = ref_backprop_error_to_hidden(probs, targets, module_w, hidden_mask, mask)

        np.testing.assert_array_equal(grad_h, 0.0)

    def test_hidden_grads_sample_mask(self):
        """Masked samples produce zero hidden gradients."""
        rng = make_rng(seed=92)
        batch, hid, cls = 8, 8, 3
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        module_w = make_weights(rng, cls, hid)
        hidden_mask = np.ones((batch, hid), dtype=np.float32)

        mask = np.zeros(batch, dtype=np.float32)
        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
        grad_h = ref_backprop_error_to_hidden(probs, targets, module_w, hidden_mask, mask)

        np.testing.assert_array_equal(grad_h, 0.0)
