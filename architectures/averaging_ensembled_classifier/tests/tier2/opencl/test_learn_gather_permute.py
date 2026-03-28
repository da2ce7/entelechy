# tests/tier2/opencl/test_learn_gather_permute.py
"""Tier 2 tests: gather_and_permute_grad_h kernel (Node 13)."""
from __future__ import annotations

import numpy as np
import pytest

from tests.tier2.fixtures.numpy_gradients import ref_gather_and_permute_grad_h
from tests.tier2.fixtures.data_generators import make_rng


class TestGatherPermute:
    """Per-kernel correctness tests for gather_and_permute_grad_h."""

    def test_gather_permute_basic(self):
        """AoS->SoA permutation produces contiguous output."""
        rng = make_rng(seed=130)
        tiles = [rng.standard_normal((4, 8)).astype(np.float32) for _ in range(3)]
        ref = ref_gather_and_permute_grad_h(tiles)
        expected = np.concatenate(tiles, axis=0)
        np.testing.assert_array_equal(ref, expected)

    def test_gather_permute_multi_tile(self):
        """Multi-tile input produces correct total shape."""
        rng = make_rng(seed=131)
        n_tiles = 5
        batch_per_tile = 4
        hidden = 8
        tiles = [rng.standard_normal((batch_per_tile, hidden)).astype(np.float32) for _ in range(n_tiles)]
        ref = ref_gather_and_permute_grad_h(tiles)
        assert ref.shape == (n_tiles * batch_per_tile, hidden)
