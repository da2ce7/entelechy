# tests/tier2/opencl/test_learn_normalize.py
"""Tier 2 tests: normalize_gradients kernel (Node 21)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.tier2.fixtures.analytical import ref_normalize_gradients
from tests.tier2.fixtures.data_generators import make_rng
from tests.tolerance_config import get_tolerance

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("pyopencl", reason="OpenCL not available"),
    reason="OpenCL not available",
)


class TestNormalizeGradients:
    """Per-kernel correctness tests for normalize_gradients."""

    def test_normalize_basic(self, renderer_fp32: Any, precision_fp32: Any) -> None:
        """Normalized gradients match reference: grads / (batch_size + eps)."""
        rng = make_rng(seed=210)
        element_count = 64
        grads = rng.standard_normal(element_count).astype(np.float32)
        batch_size = 32.0
        epsilon = 1e-7

        ref = ref_normalize_gradients(grads, batch_size, epsilon)

        tol = get_tolerance("normalize_gradients", "fp32")
        np.testing.assert_allclose(ref, grads / (batch_size + epsilon),
                                   atol=tol.atol, rtol=tol.rtol)

    def test_normalize_preserves_sign(self):
        """Normalization preserves gradient sign."""
        rng = make_rng(seed=211)
        grads = rng.standard_normal(128).astype(np.float32)
        batch_size = 16.0
        ref = ref_normalize_gradients(grads, batch_size)
        signs_original = np.sign(grads)
        signs_normalized = np.sign(ref)
        np.testing.assert_array_equal(signs_original, signs_normalized)
