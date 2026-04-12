# tests/tier3/test_parity_per_kernel.py
"""Tier 3: per-kernel cross-backend parity tests (ADR-016).

Each test dispatches a single kernel on both backends (oracle + comparison)
using identical inputs (zero-initialized buffers) and compares outputs
within Tier 3 tolerances.

The test builds a minimal ExecutionPlan containing a single kernel dispatch
node plus a RetrievalNode. Both backends render this plan from identical
(zero-initialized) buffer state, producing results that are compared
element-wise.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.tolerance_config import get_tier3_tolerance

from .conftest import get_available_backends, select_oracle

# ---------------------------------------------------------------------------
# Kernels whose zero-initialized-buffer parity is undefined.
#
# calculate_temp_gradients divides logits by temperatures.  With
# zero-initialized buffers the operation is 0/0: CPU (IEEE 754) returns
# NaN while GPU transcendental approximations may return 0.  Neither
# result is "correct" — the kernel is only meaningful with non-zero
# temperatures.  Mark these as expected failures so the rollback gate
# is not blocked by a pathological edge case.
# ---------------------------------------------------------------------------
_ZERO_INIT_XFAIL = {
    "calculate_chunk_temp_gradients",
}

# ---------------------------------------------------------------------------
# Kernel inventory (ADR-013)
#
# Abbreviated names matching tolerance_config keys.
# The single_kernel_plan_factory maps these to actual plan node IDs.
# ---------------------------------------------------------------------------

KERNEL_INVENTORY = [
    "forward_pass",
    "render_logits_chunk",
    "compute_probs_loss_cce_chunk",
    "compute_probs_loss_bce_chunk",
    "calculate_module_param_grads_chunk",
    "backprop_error_to_hidden_chunk",
    "calculate_chunk_temp_gradients",
    "clip_partial_gradients",
    "gather_and_permute_grad_hidden_activations",
    "aggregate_register_reduce",
    "aggregate_local_reduce",
    "clip_intermediate_grad",
    "stabilize_and_reduce_grad_hidden_activations",
    "backprop_shared_weights_chunk",
    "backprop_shared_biases_chunk",
    "clip_shared_gradients_chunk",
    "normalize_gradients",
    "adam_update",
    "clamp_temperatures",
]

# Strategy A kernels — additionally parameterized over problem type.
STRATEGY_A_KERNELS = {
    "calculate_module_param_grads_chunk",
    "backprop_error_to_hidden_chunk",
    "calculate_chunk_temp_gradients",
}


def _build_comparison_pairs() -> list[tuple[str, str]]:
    """Build comparison pairs for parametrize at module load time."""
    available = get_available_backends()
    oracle = select_oracle()
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


def _execute_and_extract(renderer: Any, plan: Any) -> np.ndarray:
    """Render a mini-plan and return the 'output' retrieval as a flat array."""
    futures = renderer.render(plan)
    return futures["output"].result().ravel()


# ---------------------------------------------------------------------------
# Per-kernel parity tests
# ---------------------------------------------------------------------------


@pytest.mark.tier3
class TestParityPerKernel:
    """Cross-backend per-kernel numerical parity.

    Tests are parameterized over the kernel inventory and available
    backend pairs. Each test builds a minimal single-kernel plan,
    renders it on both backends, and compares the output within
    Tier 3 tolerances.
    """

    @pytest.mark.parametrize("kernel_name", KERNEL_INVENTORY)
    @pytest.mark.parametrize("pair", _PAIRS, ids=[f"{a}_vs_{b}" for a, b in _PAIRS])
    def test_parity(
        self,
        kernel_name: str,
        pair: tuple[str, str],
        renderer_factory: Any,
        single_kernel_plan_factory: Any,
    ) -> None:
        """Verify numerical parity for a single kernel across two backends."""
        if kernel_name in _ZERO_INIT_XFAIL:
            pytest.xfail(
                f"'{kernel_name}' produces undefined results from "
                f"zero-initialized buffers (0/0 divergence)"
            )
        oracle_name, comp_name = pair

        tol = get_tier3_tolerance(kernel_name)
        assert tol.atol >= 0
        assert tol.rtol >= 0

        # Build the minimal plan for this kernel.
        plan = single_kernel_plan_factory.build(kernel_name)

        # Render on both backends.
        oracle_renderer = renderer_factory(oracle_name)
        comp_renderer = renderer_factory(comp_name)

        oracle_output = _execute_and_extract(oracle_renderer, plan)
        comp_output = _execute_and_extract(comp_renderer, plan)

        np.testing.assert_allclose(
            comp_output,
            oracle_output,
            atol=tol.atol,
            rtol=tol.rtol,
            err_msg=(
                f"Per-kernel parity failure: {comp_name} vs {oracle_name} "
                f"for kernel '{kernel_name}'"
            ),
        )
