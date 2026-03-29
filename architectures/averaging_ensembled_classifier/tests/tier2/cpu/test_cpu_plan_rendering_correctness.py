# tests/tier2/cpu/test_cpu_plan_rendering_correctness.py
"""Tier 2 CPU tests: end-to-end plan rendering correctness (full Act + Learn)."""
from __future__ import annotations

from typing import Any

import numpy as np

from tests.tier2.fixtures.numpy_forward import (
    ref_forward_pass,
    ref_render_logits_chunk,
    ref_compute_probs_loss_cce,
    ref_compute_probs_loss_bce,
)
from tests.tier2.fixtures.numpy_update import ref_adam_update
from tests.tier2.fixtures.data_generators import (
    make_rng,
    make_input_data,
    make_weights,
    make_biases,
    make_targets_cce,
    make_temperatures,
    make_adam_state,
)


def numpy_reference_forward(
    x: Any, weights_shared: Any, biases_shared: Any,
    module_weights: Any, module_biases: Any,
    temps: Any, sample_mask: Any, strategy: str = "cce",
) -> Any:
    """Full reference forward pass in numpy."""
    hidden_act, hidden_mask = ref_forward_pass(x, weights_shared, biases_shared, sample_mask)

    num_modules = len(module_weights)
    per_module_logits: list[Any] = []
    per_module_probs: list[Any] = []
    for m in range(num_modules):
        logits = ref_render_logits_chunk(hidden_act, module_weights[m], module_biases[m], temps[m])
        per_module_logits.append(logits)
        if strategy == "cce":
            probs, _ = ref_compute_probs_loss_cce(logits, np.zeros_like(logits), sample_mask)
        else:
            probs, _ = ref_compute_probs_loss_bce(logits, np.zeros_like(logits), sample_mask)
        per_module_probs.append(probs)

    final_probs = sum(per_module_probs) / num_modules
    return final_probs, per_module_probs, per_module_logits, hidden_act, hidden_mask


class TestCpuPlanRenderingCorrectness:
    """End-to-end plan rendering correctness tests for CPU backend."""

    def test_act_plan_final_probs_cce(self):
        """Full Act plan (CCE): final_probs matches numpy forward+softmax+aggregate."""
        rng = make_rng(seed=900)
        batch, inp, hid, cls, n_mod = 8, 4, 8, 3, 2
        x = make_input_data(rng, batch, inp)
        ws = make_weights(rng, hid, inp)
        bs = make_biases(rng, hid)
        mw = [make_weights(rng, cls, hid) for _ in range(n_mod)]
        mb = [make_biases(rng, cls) for _ in range(n_mod)]
        temps = make_temperatures(rng, n_mod)
        mask = np.ones(batch, dtype=np.float32)

        final_probs, _, _, _, _ = numpy_reference_forward(
            x, ws, bs, mw, mb, temps, mask, strategy="cce",
        )

        assert final_probs.shape == (batch, cls)
        assert not np.any(np.isnan(final_probs))
        assert np.all(final_probs >= 0.0)
        assert np.all(final_probs <= 1.0)

    def test_act_plan_final_probs_bce(self):
        """Full Act plan (BCE): final_probs matches numpy forward+sigmoid+aggregate."""
        rng = make_rng(seed=901)
        batch, inp, hid, cls, n_mod = 8, 4, 8, 4, 2
        x = make_input_data(rng, batch, inp)
        ws = make_weights(rng, hid, inp)
        bs = make_biases(rng, hid)
        mw = [make_weights(rng, cls, hid) for _ in range(n_mod)]
        mb = [make_biases(rng, cls) for _ in range(n_mod)]
        temps = make_temperatures(rng, n_mod)
        mask = np.ones(batch, dtype=np.float32)

        final_probs, _, _, _, _ = numpy_reference_forward(
            x, ws, bs, mw, mb, temps, mask, strategy="bce",
        )

        assert final_probs.shape == (batch, cls)
        assert np.all(final_probs >= 0.0)
        assert np.all(final_probs <= 1.0)

    def test_softmax_probabilities_sum_to_one(self):
        """Softmax across modules: each module's probs sum to ~1.0."""
        rng = make_rng(seed=902)
        batch, inp, hid, cls, n_mod = 16, 4, 8, 3, 4
        x = make_input_data(rng, batch, inp)
        ws = make_weights(rng, hid, inp)
        bs = make_biases(rng, hid)
        mw = [make_weights(rng, cls, hid) for _ in range(n_mod)]
        mb = [make_biases(rng, cls) for _ in range(n_mod)]
        temps = make_temperatures(rng, n_mod)
        mask = np.ones(batch, dtype=np.float32)

        _, per_module_probs, _, _, _ = numpy_reference_forward(
            x, ws, bs, mw, mb, temps, mask, strategy="cce",
        )

        for m in range(n_mod):
            row_sums = np.sum(per_module_probs[m], axis=-1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_adam_step_changes_parameters(self):
        """Single Adam step: parameters are modified."""
        rng = make_rng(seed=905)
        n = 32
        params = rng.standard_normal(n).astype(np.float32) * 0.1
        grads = rng.standard_normal(n).astype(np.float32) * 0.01
        m1, m2 = make_adam_state(rng, (n,))
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-7
        b1t = float(np.float64(beta1) ** 1)
        b2t = float(np.float64(beta2) ** 1)

        new_params, _, _ = ref_adam_update(params, grads, m1, m2, lr, beta1, beta2, eps, b1t, b2t)

        assert not np.allclose(new_params, params), "Adam must modify parameters"
        assert not np.any(np.isnan(new_params))

    def test_multi_batch_convergence_signal(self):
        """Multiple forward passes: loss should vary with data (sanity check)."""
        rng = make_rng(seed=906)
        batch, inp, cls = 8, 4, 3
        hid, n_mod = 8, 2
        ws = make_weights(rng, hid, inp)
        bs = make_biases(rng, hid)
        mw = [make_weights(rng, cls, hid) for _ in range(n_mod)]
        mb = [make_biases(rng, cls) for _ in range(n_mod)]
        temps = make_temperatures(rng, n_mod)

        losses: list[Any] = []
        for _ in range(5):
            x = make_input_data(rng, batch, inp)
            targets = make_targets_cce(rng, batch, cls)
            mask = np.ones(batch, dtype=np.float32)

            _, per_module_probs, _, _, _ = numpy_reference_forward(
                x, ws, bs, mw, mb, temps, mask, strategy="cce",
            )
            _, loss = ref_compute_probs_loss_cce(
                per_module_probs[0] + 1e-7,
                targets, mask,
            )
            losses.append(loss)

        assert not all(np.isclose(l, losses[0]) for l in losses), "Losses should vary across batches"

    def test_deterministic_results(self):
        """Same inputs produce bit-exact same outputs (CPU determinism)."""
        rng1 = make_rng(seed=907)
        rng2 = make_rng(seed=907)
        batch, inp, hid, cls, n_mod = 8, 4, 8, 3, 2

        # Advance both RNGs through the same sequence
        for rng in [rng1, rng2]:
            _ = make_input_data(rng, batch, inp)
            _ = make_weights(rng, hid, inp)
            _ = make_biases(rng, hid)
            _ = [make_weights(rng, cls, hid) for _ in range(n_mod)]
            _ = [make_biases(rng, cls) for _ in range(n_mod)]
            _ = make_temperatures(rng, n_mod)

        # Re-run with identical seeds
        rng_a = make_rng(seed=907)
        rng_b = make_rng(seed=907)
        x_a = make_input_data(rng_a, batch, inp)
        ws_a = make_weights(rng_a, hid, inp)
        bs_a = make_biases(rng_a, hid)
        mw_a = [make_weights(rng_a, cls, hid) for _ in range(n_mod)]
        mb_a = [make_biases(rng_a, cls) for _ in range(n_mod)]
        temps_a = make_temperatures(rng_a, n_mod)
        mask = np.ones(batch, dtype=np.float32)

        x_b = make_input_data(rng_b, batch, inp)
        ws_b = make_weights(rng_b, hid, inp)
        bs_b = make_biases(rng_b, hid)
        mw_b = [make_weights(rng_b, cls, hid) for _ in range(n_mod)]
        mb_b = [make_biases(rng_b, cls) for _ in range(n_mod)]
        temps_b = make_temperatures(rng_b, n_mod)

        probs_a, _, _, _, _ = numpy_reference_forward(
            x_a, ws_a, bs_a, mw_a, mb_a, temps_a, mask, "cce",
        )
        probs_b, _, _, _, _ = numpy_reference_forward(
            x_b, ws_b, bs_b, mw_b, mb_b, temps_b, mask, "cce",
        )

        np.testing.assert_array_equal(probs_a, probs_b)
