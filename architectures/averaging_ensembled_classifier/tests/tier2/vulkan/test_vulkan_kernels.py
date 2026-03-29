# tests/tier2/vulkan/test_vulkan_kernels.py
"""Tier 2 Vulkan kernel correctness test stubs.

Stub — becomes functional when Phase 5 delivers the Vulkan renderer.
Test structure mirrors CPU and OpenCL Tier 2 tests:
  - Parameterized over the kernel inventory (ADR-013)
  - Each test dispatches a single kernel, compares against numpy fixtures
  - Tolerances from tests/tolerance_config.py
"""
from __future__ import annotations

import pytest

# The kernel inventory matches the full set from ADR-013.
# Each kernel will get its own test class or parameterized test
# when Phase 5 populates this file.

VULKAN_KERNEL_INVENTORY = [
    "forward_pass",
    "render_logits_chunk",
    "compute_probs_loss_cce_chunk",
    "compute_probs_loss_bce_chunk",
    "calculate_module_param_grads",
    "backprop_error_to_hidden",
    "calculate_temp_gradients",
    "clip_partial_gradients",
    "gather_and_permute_grad_h",
    "aggregate_register_reduce",
    "aggregate_local_reduce",
    "clip_intermediate_grad",
    "stabilize_reduce_grad_h",
    "backprop_shared_weights",
    "backprop_shared_biases",
    "clip_shared_gradients",
    "normalize_gradients",
    "adam_update",
    "clamp_temperatures",
]


@pytest.mark.tier2
@pytest.mark.vulkan
class TestVulkanKernels:
    """Vulkan Tier 2 kernel correctness tests (stub).

    Phase 5 will populate test methods following the pattern established
    by tests/tier2/cpu/ and tests/tier2/opencl/.
    """

    @pytest.mark.parametrize("kernel_name", VULKAN_KERNEL_INVENTORY)
    def test_kernel_stub(self, kernel_name: str) -> None:
        """Placeholder — Phase 5 replaces with per-kernel correctness tests."""
        pytest.skip(f"Vulkan kernel test for '{kernel_name}' not yet implemented (Phase 5)")
