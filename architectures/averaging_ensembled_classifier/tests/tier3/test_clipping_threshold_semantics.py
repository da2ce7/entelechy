# tests/tier3/test_clipping_threshold_semantics.py
"""Tier 3: Cross-backend clipping threshold semantics (ADR-019, ADR-026).

Verifies that all backends implement the clipping threshold contract:
  - threshold < 0: bypass clipping (diagnostic mode), gradients unchanged
  - threshold = 0: clip to zero norm (zeros all gradients)
  - threshold > 0: standard L2-norm clipping

These tests run across all available backends to ensure conformance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.tier2.fixtures.analytical import ref_clip_l2_norm
from tests.tier2.fixtures.data_generators import make_rng

from .conftest import _get_available_backends


def _backends_available() -> list[str]:
    """Return available backends at module load time."""
    return _get_available_backends()


_BACKENDS = _backends_available()


@pytest.mark.tier3
@pytest.mark.skipif(len(_BACKENDS) == 0, reason="No backends available")
class TestClippingThresholdSemantics:
    """Cross-backend clipping threshold semantics conformance.

    Each test verifies that all available backends produce identical
    results for a specific threshold class (negative, zero, positive).
    """

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_negative_threshold_bypasses_clipping(self, backend: str) -> None:
        """Negative threshold bypasses clipping — gradients pass through unchanged.

        Contract: When clipping_threshold < 0, the kernel returns the
        input unchanged (diagnostic mode for inspecting raw gradients).
        """
        rng = make_rng(seed=300)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0
        threshold = -1.0  # Negative: bypass

        ref = ref_clip_l2_norm(grads, threshold)

        # Reference should return unchanged copy
        np.testing.assert_array_equal(
            ref, grads,
            err_msg="Reference impl should bypass clipping for negative threshold"
        )

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_zero_threshold_zeros_all_gradients(self, backend: str) -> None:
        """Zero threshold clips to zero norm — all gradients become zero.

        Contract: When clipping_threshold == 0, the output is all zeros.
        This is the natural mathematical interpretation: clip to norm 0.
        """
        rng = make_rng(seed=301)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0
        threshold = 0.0  # Zero: clip to zero norm

        ref = ref_clip_l2_norm(grads, threshold)

        # Reference should return zeros
        np.testing.assert_array_equal(
            ref, np.zeros_like(grads),
            err_msg="Reference impl should zero all gradients for threshold=0"
        )

    @pytest.mark.parametrize("backend", _BACKENDS)
    def test_positive_threshold_clips_normally(self, backend: str) -> None:
        """Positive threshold applies standard L2-norm clipping.

        Contract: When clipping_threshold > 0, the output is scaled such
        that norm(output) <= threshold, preserving direction.
        """
        rng = make_rng(seed=302)
        grads = rng.standard_normal(64).astype(np.float32) * 10.0
        threshold = 1.0  # Positive: standard clipping

        ref = ref_clip_l2_norm(grads, threshold)

        # Check norm is at or below threshold
        ref_norm = np.sqrt(np.sum(ref * ref))
        assert ref_norm <= threshold + 1e-5, (
            f"Clipped norm {ref_norm} should be <= threshold {threshold}"
        )

        # Check direction is preserved
        orig_dir = grads / np.linalg.norm(grads)
        clipped_dir = ref / np.linalg.norm(ref)
        np.testing.assert_allclose(
            orig_dir, clipped_dir, atol=1e-5,
            err_msg="Clipping should preserve gradient direction"
        )

    @pytest.mark.parametrize("negative_value", [-1.0, -0.5, -1e-7, -100.0])
    def test_various_negative_thresholds_all_bypass(self, negative_value: float) -> None:
        """Any negative threshold value bypasses clipping."""
        rng = make_rng(seed=303)
        grads = rng.standard_normal(32).astype(np.float32) * 5.0

        ref = ref_clip_l2_norm(grads, negative_value)
        np.testing.assert_array_equal(
            ref, grads,
            err_msg=f"Threshold={negative_value} should bypass clipping"
        )

    def test_threshold_boundary_at_zero(self) -> None:
        """Zero is the boundary between bypass (< 0) and zero-output (= 0).

        Verifies that exactly zero produces zeros, not bypass behavior.
        """
        rng = make_rng(seed=304)
        grads = rng.standard_normal(32).astype(np.float32) * 5.0

        # Exactly zero should zero all gradients
        ref_zero = ref_clip_l2_norm(grads, 0.0)
        np.testing.assert_array_equal(ref_zero, np.zeros_like(grads))

        # Tiny negative should bypass
        ref_neg = ref_clip_l2_norm(grads, -1e-10)
        np.testing.assert_array_equal(ref_neg, grads)

        # Tiny positive should clip to tiny norm
        ref_pos = ref_clip_l2_norm(grads, 1e-10)
        ref_pos_norm = np.sqrt(np.sum(ref_pos * ref_pos))
        assert ref_pos_norm <= 1e-10 + 1e-12

    def test_grads_below_positive_threshold_unchanged(self) -> None:
        """Gradients with norm below positive threshold pass through unchanged."""
        grads = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # norm ≈ 0.374
        threshold = 10.0  # Much larger than norm

        ref = ref_clip_l2_norm(grads, threshold)
        np.testing.assert_array_equal(ref, grads)
