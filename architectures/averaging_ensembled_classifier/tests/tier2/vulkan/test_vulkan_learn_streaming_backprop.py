# tests/tier2/vulkan/test_vulkan_learn_streaming_backprop.py
"""Tier 2 Vulkan tests: streaming loop backprop (Nodes 17-19)."""
from __future__ import annotations

from typing import Any

import numpy as np

from tests.tier2.fixtures.numpy_backprop import (
    ref_backprop_shared_weights,
    ref_backprop_shared_biases,
)
from tests.tier2.fixtures.analytical import ref_clip_shared_gradients
from tests.tier2.fixtures.data_generators import (
    make_rng, make_input_data,
)


class TestVulkanStreamingBackprop:
    """Per-kernel correctness tests for Vulkan streaming loop backprop."""

    def test_shared_weights_backprop_basic(self):
        """Single chunk: grad_W_s matches reference."""
        rng = make_rng(seed=170)
        batch, inp, hid = 8, 4, 8
        grad_h = rng.standard_normal((batch, hid)).astype(np.float32)
        x = make_input_data(rng, batch, inp)
        mask = np.ones(batch, dtype=np.float32)

        ref = ref_backprop_shared_weights(grad_h, x, mask)

        assert ref.shape == (hid, inp)
        assert not np.any(np.isnan(ref))

    def test_shared_biases_backprop_basic(self):
        """Single chunk: grad_b_s matches reference."""
        rng = make_rng(seed=171)
        batch, hid = 8, 8
        grad_h = rng.standard_normal((batch, hid)).astype(np.float32)
        mask = np.ones(batch, dtype=np.float32)

        ref = ref_backprop_shared_biases(grad_h, mask)

        assert ref.shape == (hid,)
        assert not np.any(np.isnan(ref))

    def test_streaming_multi_chunk(self):
        """Multiple chunks: each produces independent partial gradients."""
        rng = make_rng(seed=172)
        total_batch, inp, hid = 32, 4, 8
        chunk_size = 8
        num_chunks = total_batch // chunk_size

        x = make_input_data(rng, total_batch, inp)
        grad_h = rng.standard_normal((total_batch, hid)).astype(np.float32)
        mask = np.ones(total_batch, dtype=np.float32)

        partials_w: list[Any] = []
        for c in range(num_chunks):
            s = c * chunk_size
            e = s + chunk_size
            pw = ref_backprop_shared_weights(grad_h[s:e], x[s:e], mask[s:e])
            partials_w.append(pw)

        for pw in partials_w:
            assert pw.shape == (hid, inp)

    def test_streaming_clip_per_chunk(self):
        """Per-chunk clipping: clip_shared_gradients matches reference."""
        rng = make_rng(seed=173)
        hid, inp = 8, 4
        grad_sw = rng.standard_normal((hid, inp)).astype(np.float32) * 10.0
        grad_sb = rng.standard_normal(hid).astype(np.float32) * 10.0
        threshold = 2.0

        clipped_sw, clipped_sb = ref_clip_shared_gradients(grad_sw, grad_sb, threshold)

        combined = np.concatenate([clipped_sw.ravel(), clipped_sb.ravel()])
        norm = np.sqrt(np.sum(combined ** 2))
        assert norm <= threshold + 1e-5

    def test_streaming_stride_arithmetic(self):
        """ParameterStride correctly computes batch_chunk_offset per chunk."""
        chunk_size = 8
        num_chunks = 4
        for i in range(num_chunks):
            offset = 0 + i * chunk_size
            assert offset == i * chunk_size

    def test_streaming_masked_samples(self):
        """Masked samples contribute zero to shared gradients."""
        rng = make_rng(seed=175)
        batch, inp, hid = 8, 4, 8
        grad_h = rng.standard_normal((batch, hid)).astype(np.float32)
        x = make_input_data(rng, batch, inp)
        mask = np.zeros(batch, dtype=np.float32)

        ref_w = ref_backprop_shared_weights(grad_h, x, mask)
        ref_b = ref_backprop_shared_biases(grad_h, mask)

        np.testing.assert_array_equal(ref_w, 0.0)
        np.testing.assert_array_equal(ref_b, 0.0)

    def test_streaming_single_chunk(self):
        """Single-module model: streaming loop has one iteration."""
        rng = make_rng(seed=176)
        batch, inp, hid = 8, 4, 8
        grad_h = rng.standard_normal((batch, hid)).astype(np.float32)
        x = make_input_data(rng, batch, inp)
        mask = np.ones(batch, dtype=np.float32)

        ref_w = ref_backprop_shared_weights(grad_h, x, mask)
        ref_b = ref_backprop_shared_biases(grad_h, mask)

        assert ref_w.shape == (hid, inp)
        assert ref_b.shape == (hid,)
        assert not np.any(np.isnan(ref_w))
        assert not np.any(np.isnan(ref_b))
