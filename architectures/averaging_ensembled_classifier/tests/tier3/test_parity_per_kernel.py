# tests/tier3/test_parity_per_kernel.py
"""Tier 3: per-kernel cross-backend parity tests (ADR-016).

Each test dispatches a single kernel on both backends (oracle + comparison)
using identical inputs and compares outputs within Tier 3 tolerances.

NOTE: Full per-kernel isolation requires a single-kernel plan factory that
builds minimal ExecutionPlans containing a single KernelDispatchNode. Until
that factory exists, these tests validate the framework wiring (tolerance
lookup, renderer instantiation, backend pair generation) and will activate
fully when the plan builder supports isolated kernel dispatch.
"""
from __future__ import annotations

from typing import Any

import pytest

from tests.tolerance_config import get_tier3_tolerance

from .conftest import _get_available_backends, _select_oracle

# ---------------------------------------------------------------------------
# Kernel inventory (ADR-013)
# ---------------------------------------------------------------------------

KERNEL_INVENTORY = [
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

# Strategy A kernels — additionally parameterized over problem type.
STRATEGY_A_KERNELS = {
    "calculate_module_param_grads",
    "backprop_error_to_hidden",
    "calculate_temp_gradients",
}


def _build_comparison_pairs() -> list[tuple[str, str]]:
    """Build comparison pairs for parametrize at module load time."""
    available = _get_available_backends()
    oracle = _select_oracle()
    pairs: list[tuple[str, str]] = []
    if oracle is not None:
        for b in available:
            if b != oracle:
                pairs.append((oracle, b))
    else:
        for left, right in __import__("itertools").combinations(available, 2):
            pairs.append((left, right))
    return pairs


_PAIRS = _build_comparison_pairs()


# ---------------------------------------------------------------------------
# Per-kernel parity tests
# ---------------------------------------------------------------------------


@pytest.mark.tier3
class TestParityPerKernel:
    """Cross-backend per-kernel numerical parity.

    Tests are parameterized over the kernel inventory and available
    backend pairs. Each test validates the tolerance and renderer
    framework for the target kernel.

    Full dispatch-and-compare logic activates when a single-kernel
    plan factory is available (the factory must produce a minimal
    ExecutionPlan containing only the target KernelDispatchNode with
    deterministic input buffers).
    """

    @pytest.mark.parametrize("kernel_name", KERNEL_INVENTORY)
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_parity(
        self,
        kernel_name: str,
        pair: tuple[str, str],
        renderer_factory: Any,
    ) -> None:
        """Verify numerical parity for a single kernel across two backends."""
        oracle_name, comp_name = pair

        tol = get_tier3_tolerance(kernel_name)
        assert tol.atol >= 0
        assert tol.rtol >= 0

        # Verify both renderers can be instantiated.
        oracle_renderer = renderer_factory(oracle_name)
        comp_renderer = renderer_factory(comp_name)
        assert oracle_renderer is not None
        assert comp_renderer is not None

        # TODO: Once single-kernel plan factory is available, build a
        # minimal plan, render on both backends, and assert_allclose.
        # See Phase 4 plan §Step 4.5 for the target test pattern.
