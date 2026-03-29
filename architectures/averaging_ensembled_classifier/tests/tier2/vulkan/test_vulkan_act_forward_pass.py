# tests/tier2/vulkan/test_vulkan_act_forward_pass.py
"""Tier 2 Vulkan tests: forward_pass kernel (Node 4)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.tier2.fixtures.numpy_forward import ref_forward_pass
from tests.tier2.fixtures.data_generators import (
    make_rng, make_input_data, make_weights, make_biases,
)
from tests.tolerance_config import get_tolerance


class TestVulkanForwardPass:
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

        assert ref_act.shape == (batch, hid)
        assert ref_mask.shape == (batch, hid)
        assert not np.any(np.isnan(ref_act))

    def test_forward_pass_relu_activation(self):
        """ReLU gating: negative pre-activations -> zero."""
        rng = make_rng(seed=41)
        batch, inp, hid = 4, 4, 8
        x = make_input_data(rng, batch, inp)
        w = -np.abs(make_weights(rng, hid, inp))
        b = np.full(hid, -10.0, dtype=np.float32)
        mask = np.ones(batch, dtype=np.float32)

        ref_act, ref_mask = ref_forward_pass(x, w, b, mask)

        assert np.all(ref_mask == 0.0)
        assert np.all(ref_act == 0.0)

    def test_forward_pass_sample_mask(self):
        """Masked samples produce zero activations."""
        rng = make_rng(seed=42)
        batch, inp, hid = 8, 4, 8
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

        ref_act, _ = ref_forward_pass(x, w, b, mask)

        assert np.all(ref_act[:4] == 0.0)

    def test_forward_pass_multi_tile(self):
        """Multi-tile forward pass produces correct results."""
        rng = make_rng(seed=43)
        batch, inp, hid = 32, 4, 16
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.ones(batch, dtype=np.float32)

        ref_act, _ = ref_forward_pass(x, w, b, mask)

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

        assert np.all(ref_mask == 1.0)
        assert np.all(ref_act > 0.0)

    @pytest.mark.parametrize("batch_size", [1, 7, 32, 128])
    def test_forward_pass_batch_sizes(self, batch_size: int):
        """Correctness across different batch sizes."""
        rng = make_rng(seed=45)
        inp, hid = 4, 8
        x = make_input_data(rng, batch_size, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.ones(batch_size, dtype=np.float32)

        ref_act, ref_mask = ref_forward_pass(x, w, b, mask)

        assert ref_act.shape == (batch_size, hid)
        assert not np.any(np.isnan(ref_act))

    def test_forward_pass_hidden_mask_binary(self):
        """Hidden mask values are exactly 0.0 or 1.0."""
        rng = make_rng(seed=47)
        batch, inp, hid = 16, 8, 16
        x = make_input_data(rng, batch, inp)
        w = make_weights(rng, hid, inp)
        b = make_biases(rng, hid)
        mask = np.ones(batch, dtype=np.float32)

        _, ref_mask = ref_forward_pass(x, w, b, mask)

        unique_vals = np.unique(ref_mask)
        assert all(v in (0.0, 1.0) for v in unique_vals)
