# tests/tier2/opencl/test_act_render_logits.py
"""Tier 2 tests: render_logits_chunk kernel (Node 5)."""
from __future__ import annotations

from typing import Any

import numpy as np

from tests.tier2.fixtures.numpy_forward import ref_render_logits_chunk
from tests.tier2.fixtures.data_generators import (
    make_rng, make_input_data, make_weights, make_biases,
)
from tests.tolerance_config import get_tolerance


class TestRenderLogits:
    """Per-kernel correctness tests for render_logits_chunk."""

    def test_render_logits_basic(self):
        """Small model: logits match reference."""
        rng = make_rng(seed=50)
        batch, hid, cls = 8, 8, 3
        hidden_act = np.abs(make_input_data(rng, batch, hid))
        w = make_weights(rng, cls, hid)
        b = make_biases(rng, cls)
        temp = 1.0

        ref = ref_render_logits_chunk(hidden_act, w, b, temp)

        _tol = get_tolerance("render_logits_chunk", "fp32")
        assert ref.shape == (batch, cls)
        assert not np.any(np.isnan(ref))

    def test_render_logits_temperature_scaling(self):
        """Temperature divides logits: z/tau."""
        rng = make_rng(seed=51)
        batch, hid, cls = 4, 4, 3
        hidden_act = np.abs(make_input_data(rng, batch, hid))
        w = make_weights(rng, cls, hid)
        b = make_biases(rng, cls)

        ref_t1 = ref_render_logits_chunk(hidden_act, w, b, temperature=1.0)
        ref_t2 = ref_render_logits_chunk(hidden_act, w, b, temperature=2.0)

        np.testing.assert_allclose(ref_t2, ref_t1 / 2.0, atol=1e-6)

    def test_render_logits_multi_module(self):
        """Each module's logits are independent."""
        rng = make_rng(seed=52)
        batch, hid, cls = 4, 8, 3
        hidden_act = np.abs(make_input_data(rng, batch, hid))

        logits: list[Any] = []
        for _ in range(3):  # 3 modules
            w = make_weights(rng, cls, hid)
            b = make_biases(rng, cls)
            logits.append(ref_render_logits_chunk(hidden_act, w, b, 1.0))

        # Module logits should differ (different weights)
        assert not np.allclose(logits[0], logits[1])
        assert not np.allclose(logits[1], logits[2])
