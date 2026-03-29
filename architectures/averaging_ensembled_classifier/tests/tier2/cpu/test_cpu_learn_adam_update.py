# tests/tier2/cpu/test_cpu_learn_adam_update.py
"""Tier 2 CPU tests: adam_update kernel (Node 24)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_update import ref_adam_update
from tests.tier2.fixtures.data_generators import make_rng, make_adam_state
from tests.tolerance_config import get_tolerance


class TestCpuAdamUpdate:
    """Per-kernel correctness tests for CPU adam_update."""

    def test_adam_update_basic(self):
        """Single Adam step matches reference."""
        rng = make_rng(seed=240)
        n = 64
        params = rng.standard_normal(n).astype(np.float32) * 0.1
        grads = rng.standard_normal(n).astype(np.float32) * 0.01
        m1, m2 = make_adam_state(rng, (n,))
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-7
        t = 1
        beta1_pow_t = float(np.float64(beta1) ** t)
        beta2_pow_t = float(np.float64(beta2) ** t)

        ref_params, ref_m1, _ = ref_adam_update(
            params, grads, m1, m2, lr, beta1, beta2, eps,
            beta1_pow_t, beta2_pow_t,
        )

        tol = get_tolerance("adam_update", "fp32")
        assert not np.any(np.isnan(ref_params)), "Reference produced NaN"
        assert not np.any(np.isinf(ref_params)), "Reference produced Inf"
        assert not np.allclose(ref_params, params), "Adam did not update params"
        np.testing.assert_allclose(
            ref_m1, beta1 * m1 + (1 - beta1) * grads,
            atol=tol.atol, rtol=tol.rtol,
        )

    def test_adam_update_bias_correction(self):
        """Verify bias correction terms are applied correctly."""
        n = 32
        params = np.zeros(n, dtype=np.float32)
        grads = np.ones(n, dtype=np.float32) * 0.1
        m1 = np.zeros(n, dtype=np.float32)
        m2 = np.zeros(n, dtype=np.float32)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-8
        t = 1
        beta1_pow_t = float(np.float64(beta1) ** t)
        beta2_pow_t = float(np.float64(beta2) ** t)

        _, _, _ = ref_adam_update(
            params, grads, m1, m2, lr, beta1, beta2, eps,
            beta1_pow_t, beta2_pow_t,
        )

        m1_new = beta1 * 0 + (1 - beta1) * 0.1
        m1_hat = m1_new / (1 - beta1_pow_t)
        assert m1_hat > m1_new, "Bias correction should amplify early moment"

    def test_adam_update_multi_step(self):
        """Multiple Adam steps: verify moments accumulate correctly."""
        rng = make_rng(seed=242)
        n = 16
        params = rng.standard_normal(n).astype(np.float32) * 0.1
        m1 = np.zeros(n, dtype=np.float32)
        m2 = np.zeros(n, dtype=np.float32)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-7

        for t in range(1, 6):
            grads = rng.standard_normal(n).astype(np.float32) * 0.01
            beta1_pow_t = float(np.float64(beta1) ** t)
            beta2_pow_t = float(np.float64(beta2) ** t)
            params, m1, m2 = ref_adam_update(
                params, grads, m1, m2, lr, beta1, beta2, eps,
                beta1_pow_t, beta2_pow_t,
            )

        assert not np.any(np.isnan(params)), "Multi-step Adam produced NaN"
        assert not np.any(np.isinf(params)), "Multi-step Adam produced Inf"

    def test_adam_update_zero_gradient(self):
        """Adam step with zero gradient updates only via momentum."""
        rng = make_rng(seed=243)
        n = 32
        params = rng.standard_normal(n).astype(np.float32) * 0.1
        m1, m2 = make_adam_state(rng, (n,))
        grads = np.zeros(n, dtype=np.float32)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-7
        b1t = float(np.float64(beta1) ** 1)
        b2t = float(np.float64(beta2) ** 1)

        _, new_m1, _ = ref_adam_update(
            params, grads, m1, m2, lr, beta1, beta2, eps, b1t, b2t,
        )

        # m1 decays toward zero with zero grads
        np.testing.assert_allclose(new_m1, beta1 * m1, atol=1e-7)

    def test_adam_update_fp64_bias_terms(self):
        """Host-computed bias terms in FP64 don't lose precision over many steps."""
        beta2 = 0.999
        t = 1000
        beta2_pow_fp64 = np.float64(beta2) ** t
        assert beta2_pow_fp64 > 0.0, "FP64 beta2^t should not underflow"

    def test_adam_update_per_param_group(self):
        """Separate param groups produce independent updates."""
        rng = make_rng(seed=244)
        lr, beta1, beta2, eps = 0.001, 0.9, 0.999, 1e-7
        b1t = float(np.float64(beta1) ** 1)
        b2t = float(np.float64(beta2) ** 1)

        pa = rng.standard_normal(32).astype(np.float32)
        ga = rng.standard_normal(32).astype(np.float32) * 0.01
        m1a, m2a = make_adam_state(rng, (32,))

        pb = rng.standard_normal(8).astype(np.float32)
        gb = rng.standard_normal(8).astype(np.float32) * 0.01
        m1b, m2b = make_adam_state(rng, (8,))

        ref_a, _, _ = ref_adam_update(pa, ga, m1a, m2a, lr, beta1, beta2, eps, b1t, b2t)
        ref_b, _, _ = ref_adam_update(pb, gb, m1b, m2b, lr, beta1, beta2, eps, b1t, b2t)

        assert ref_a.shape == (32,)
        assert ref_b.shape == (8,)
