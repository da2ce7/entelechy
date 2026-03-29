# tests/tier2/opencl/test_act_forward_pass.py
"""Tier 2 tests: forward_pass kernel (Node 4)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_forward import ref_forward_pass
from tests.tier2.fixtures.data_generators import (
    make_rng, make_input_data, make_weights, make_biases,
)
from tests.tolerance_config import get_tolerance


class TestForwardPass:
    """Per-kernel correctness tests for forward_pass."""

    def test_forward_pass_basic(self):
        """Small model: hidden activations match reference."""
        rng = make_rng(seed=40)
        batch, inp, hid = 8, 4, 8
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.ones(batch, dtype=np.float32)

        ref_act, ref_mask = ref_forward_pass(x, w, b, mask)

        _tol = get_tolerance("forward_pass", "fp32")
        assert ref_act.shape == (batch, hid)
        assert ref_mask.shape == (batch, hid)
        assert not np.any(np.isnan(ref_act))
        assert not np.any(np.isinf(ref_act))

    def test_forward_pass_relu_activation(self):
        """ReLU gating: negative pre-activations -> zero."""
        rng = make_rng(seed=41)
        batch, inp, hid = 4, 4, 8
        x = make_input_data(rng, batch, inp)
        # Weights that produce known-negative pre-activations
        w = -np.abs(make_weights(rng, hid, inp))
        b = np.full(hid, -10.0, dtype=np.float32)
        mask = np.ones(batch, dtype=np.float32)

        ref_act, ref_mask = ref_forward_pass(x, w, b, mask)

        # With large negative bias, all pre-activations should be negative
        assert np.all(ref_mask == 0.0), "All mask should be 0 for negative pre-act"
        assert np.all(ref_act == 0.0), "All activations should be 0 after ReLU"

    def test_forward_pass_sample_mask(self):
        """Masked samples produce zero activations."""
        rng = make_rng(seed=42)
        batch, inp, hid = 8, 4, 8
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        # Mask out first 4 samples
        mask = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

        ref_act, _ = ref_forward_pass(x, w, b, mask)

        assert np.all(ref_act[:4] == 0.0), "Masked samples should have zero activation"

    def test_forward_pass_multi_tile(self):
        """Multi-tile forward pass: all tiles produce correct partial results."""
        rng = make_rng(seed=43)
        batch, inp, hid = 32, 4, 16
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.ones(batch, dtype=np.float32)

        ref_act, _ = ref_forward_pass(x, w, b, mask)
        _tol = get_tolerance("forward_pass", "fp32")

        # Verify full result
        assert ref_act.shape == (batch, hid)
        assert not np.any(np.isnan(ref_act))

    def test_forward_pass_all_positive(self):
        """When all pre-activations are positive, mask is all 1.0."""
        rng = make_rng(seed=44)
        batch, inp, hid = 4, 4, 4
        x = np.abs(make_input_data(rng, batch, inp)) + 0.1
        w = np.abs(make_weights(rng, hid, inp)) + 0.1
        b = np.abs(make_biases(rng, hid)) + 1.0
        mask = np.ones(batch, dtype=np.float32)

        ref_act, ref_mask = ref_forward_pass(x, w, b, mask)

        assert np.all(ref_mask == 1.0), "All-positive inputs should yield all-1 mask"
        assert np.all(ref_act > 0.0), "All activations should be positive"

    def test_forward_pass_hidden_mask_binary(self):
        """Finding 3: Hidden mask values are exactly 0.0 or 1.0.

        The kernels.cl.h contract specifies the hidden mask is a derived
        ReLU mask: 1 if activation > 0, else 0.
        """
        rng = make_rng(seed=47)
        batch, inp, hid = 16, 8, 32
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.ones(batch, dtype=np.float32)

        _, ref_mask = ref_forward_pass(x, w, b, mask)

        unique_vals = set(np.unique(ref_mask))
        assert unique_vals <= {0.0, 1.0}, (
            f"Hidden mask must contain only {{0.0, 1.0}}, got {unique_vals}"
        )
        assert 0.0 in unique_vals and 1.0 in unique_vals, (
            "Test data should produce a mix of active and inactive pre-activations"
        )
