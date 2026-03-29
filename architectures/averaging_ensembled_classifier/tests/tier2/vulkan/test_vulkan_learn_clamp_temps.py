# tests/tier2/vulkan/test_vulkan_learn_clamp_temps.py
"""Tier 2 Vulkan tests: clamp_temperatures kernel (Node 25)."""
from __future__ import annotations

import numpy as np

from tests.tier2.fixtures.analytical import ref_clamp_temperatures


class TestVulkanClampTemperatures:
    """Per-kernel correctness tests for Vulkan clamp_temperatures."""

    def test_clamp_basic(self):
        """Temperatures within range unchanged."""
        temps = np.array([0.5, 1.0, 1.5], dtype=np.float32)
        ref = ref_clamp_temperatures(temps, t_min=0.1, t_max=2.0)
        np.testing.assert_array_equal(ref, temps)

    def test_clamp_below_min(self):
        """Below-minimum temperatures raised to T_min."""
        temps = np.array([0.01, 0.05, 0.5], dtype=np.float32)
        ref = ref_clamp_temperatures(temps, t_min=0.1, t_max=2.0)
        expected = np.array([0.1, 0.1, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(ref, expected)

    def test_clamp_above_max(self):
        """Above-maximum temperatures lowered to T_max."""
        temps = np.array([1.5, 2.5, 3.0], dtype=np.float32)
        ref = ref_clamp_temperatures(temps, t_min=0.1, t_max=2.0)
        expected = np.array([1.5, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_array_equal(ref, expected)

    def test_clamp_boundary_values(self):
        """Temperatures exactly at boundaries remain unchanged."""
        temps = np.array([0.1, 2.0], dtype=np.float32)
        ref = ref_clamp_temperatures(temps, t_min=0.1, t_max=2.0)
        np.testing.assert_array_equal(ref, temps)
