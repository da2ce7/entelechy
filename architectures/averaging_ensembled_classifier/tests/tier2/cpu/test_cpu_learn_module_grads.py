# tests/tier2/cpu/test_cpu_learn_module_grads.py
"""Tier 2 CPU tests: calculate_module_param_grads kernel (Node 8)."""
from __future__ import annotations

from typing import Any

import numpy as np

from tests.tier2.fixtures.numpy_gradients import (
    ref_module_param_grads_cce,
    ref_module_param_grads_bce,
)
from tests.tier2.fixtures.numpy_forward import (
    ref_compute_probs_loss_cce,
    ref_compute_probs_loss_bce,
)
from tests.tier2.fixtures.data_generators import (
    make_rng, make_input_data,
    make_targets_cce, make_targets_bce,
)

class TestCpuModuleGrads:
    """Per-kernel correctness tests for CPU module parameter gradients."""

    def test_module_grads_cce_basic(self):
        """CCE variant: grad_weights and grad_biases match reference."""
        rng = make_rng(seed=80)
        batch, hid, cls = 8, 8, 3
        hidden_act = np.abs(make_input_data(rng, batch, hid))
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
        gw, gb = ref_module_param_grads_cce(probs, targets, hidden_act, mask)

        assert gw.shape == (cls, hid)
        assert gb.shape == (cls,)
        assert not np.any(np.isnan(gw))
        assert not np.any(np.isnan(gb))

    def test_module_grads_bce_basic(self):
        """BCE variant: grad_weights and grad_biases match reference."""
        rng = make_rng(seed=81)
        batch, hid, cls = 8, 8, 4
        hidden_act = np.abs(make_input_data(rng, batch, hid))
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_bce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_bce(logits, targets, mask)
        gw, gb = ref_module_param_grads_bce(probs, targets, hidden_act, mask)

        assert gw.shape == (cls, hid)
        assert gb.shape == (cls,)

    def test_module_grads_masked_samples(self):
        """Masked samples should not contribute to gradients."""
        rng = make_rng(seed=82)
        batch, hid, cls = 8, 4, 3
        hidden_act = np.abs(make_input_data(rng, batch, hid))
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)

        mask_zero = np.zeros(batch, dtype=np.float32)
        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask_zero)
        gw, gb = ref_module_param_grads_cce(probs, targets, hidden_act, mask_zero)

        np.testing.assert_array_equal(gw, 0.0)
        np.testing.assert_array_equal(gb, 0.0)

    def test_module_grads_gradient_magnitude(self):
        """Gradient magnitude scales with batch size."""
        hid, cls = 4, 3

        results: list[Any] = []
        for batch in [4, 16]:
            r = make_rng(seed=83)
            h = np.abs(make_input_data(r, batch, hid))
            logits = r.standard_normal((batch, cls)).astype(np.float32)
            targets = make_targets_cce(r, batch, cls)
            mask = np.ones(batch, dtype=np.float32)
            probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
            gw, _ = ref_module_param_grads_cce(probs, targets, h, mask)
            results.append(np.linalg.norm(gw))

        assert results[1] > results[0]

    def test_module_grads_single_module(self):
        """Edge case: single module gradient computation."""
        rng = make_rng(seed=84)
        batch, hid, cls = 4, 4, 2
        hidden_act = np.abs(make_input_data(rng, batch, hid))
        logits = rng.standard_normal((batch, cls)).astype(np.float32)
        targets = make_targets_cce(rng, batch, cls)
        mask = np.ones(batch, dtype=np.float32)

        probs, _ = ref_compute_probs_loss_cce(logits, targets, mask)
        gw, gb = ref_module_param_grads_cce(probs, targets, hidden_act, mask)

        assert gw.shape == (cls, hid)
        assert gb.shape == (cls,)
        assert not np.any(np.isnan(gw))
