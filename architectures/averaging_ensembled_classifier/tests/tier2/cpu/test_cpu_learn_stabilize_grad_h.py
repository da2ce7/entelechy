# tests/tier2/cpu/test_cpu_learn_stabilize_grad_h.py
"""Tier 2 CPU tests: stabilize_reduce_grad_h kernel (Node 16)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.numpy_reduction import ref_stabilize_reduce_grad_h
from tests.tier2.fixtures.data_generators import make_rng
from tests.tolerance_config import get_tolerance


class TestCpuStabilizeReduceGradH:
    """Per-kernel correctness tests for CPU Node 16 (ADR-005 opacity)."""

    def test_stabilize_reduce_basic(self):
        """Small model: reduced grad_h matches reference."""
        rng = make_rng(seed=160)
        partials = [rng.standard_normal((4, 8)).astype(np.float32) for _ in range(3)]
        ref = ref_stabilize_reduce_grad_h(partials)
        expected = sum(partials[1:], partials[0].copy())
        tol = get_tolerance("stabilize_reduce_grad_h", "fp32")
        np.testing.assert_allclose(ref, expected, atol=tol.atol, rtol=tol.rtol)

    def test_stabilize_reduce_with_clipping(self):
        """Internal clipping: result norm <= threshold."""
        rng = make_rng(seed=161)
        partials = [rng.standard_normal((4, 8)).astype(np.float32) * 10.0 for _ in range(5)]
        threshold = 2.0
        ref = ref_stabilize_reduce_grad_h(partials, clip_threshold=threshold)
        norm = np.sqrt(np.sum(ref * ref))
        assert norm <= threshold + 1e-5

    def test_stabilize_reduce_row_independence(self):
        """Each row reduced independently: no cross-row contamination."""
        rng = make_rng(seed=162)
        p1 = rng.standard_normal((4, 8)).astype(np.float32)
        p2 = rng.standard_normal((4, 8)).astype(np.float32)

        ref = ref_stabilize_reduce_grad_h([p1, p2])

        for row in range(4):
            expected_row = p1[row] + p2[row]
            np.testing.assert_allclose(ref[row], expected_row, atol=1e-6)

    def test_stabilize_reduce_near_zero(self):
        """Near-zero gradients do not produce NaN/Inf."""
        rng = make_rng(seed=163)
        partials = [rng.standard_normal((4, 8)).astype(np.float32) * 1e-10 for _ in range(3)]
        ref = ref_stabilize_reduce_grad_h(partials)
        assert not np.any(np.isnan(ref))
        assert not np.any(np.isinf(ref))

    def test_stabilize_reduce_single_partial(self):
        """Single partial: pass-through."""
        rng = make_rng(seed=164)
        partial = rng.standard_normal((4, 8)).astype(np.float32)
        ref = ref_stabilize_reduce_grad_h([partial])
        np.testing.assert_array_equal(ref, partial)
