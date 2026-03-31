# tests/tier3/conftest.py
"""Tier 3 fixtures — oracle selection and multi-backend parity (ADR-016).

Tier 3 tests compare outputs across backends for the same input. The CPU
backend is the preferred oracle; when absent, all-pairs fallback is used.
"""
from __future__ import annotations

import os
from itertools import combinations
from typing import Any

import pytest

from tests.conftest import BUILD_CONFIG


# ---------------------------------------------------------------------------
# Oracle selection (ADR-016 Option C)
# ---------------------------------------------------------------------------

def _select_oracle() -> str | None:
    """Select the Tier 3 oracle backend.

    Returns the oracle backend name, or None if no oracle is available
    (triggering all-pairs fallback).
    """
    if BUILD_CONFIG.get("cpu", False):
        return "cpu"
    return None


def _get_available_backends() -> list[str]:
    """Return names of all available backends."""
    return [b for b in ("cpu", "opencl", "vulkan") if BUILD_CONFIG.get(b, False)]


def _get_tier3_pairs(config: Any) -> list[tuple[str, str]]:
    """Return (oracle_or_left, comparison) pairs for Tier 3 tests.

    Default: CPU-oracle vs each GPU backend.
    --all-pairs: adds GPU-vs-GPU pairs.
    """
    available = _get_available_backends()
    oracle = _select_oracle()

    pairs: list[tuple[str, str]] = []
    if oracle is not None:
        for b in available:
            if b != oracle:
                pairs.append((oracle, b))

    if config.getoption("--all-pairs", default=False) or oracle is None:
        for left, right in combinations(available, 2):
            if (left, right) not in pairs and (right, left) not in pairs:
                pairs.append((left, right))

    return pairs


# ---------------------------------------------------------------------------
# Renderer factory — per-backend initialization
# ---------------------------------------------------------------------------

_ARCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_KERNEL_DIR = os.path.join(_ARCH_ROOT, "kernels")


def _create_cpu_renderer() -> Any:
    """Create a CPUPlanRenderer."""
    from src.backends.cpu.renderer import CPUPlanRenderer
    return CPUPlanRenderer()


def _create_opencl_renderer() -> Any:
    """Create a fully initialized OpenCLPlanRenderer."""
    import pyopencl as cl

    from src.backends.opencl.discovery import discover_hardware
    from src.backends.opencl.kernel_bindings.binding_phase_2_learn_C import (
        AggregateLocalReduceBinding,
        AggregateRegisterReduceBinding,
        ClipIntermediateGradBinding,
        ReduceKFanInAndClipBinding,
    )
    from src.backends.opencl.kernel_bindings.dispatch_table import build_dispatch_table
    from src.backends.opencl.context import load_and_compile_kernels_from_path
    from src.backends.opencl.renderer import OpenCLPlanRenderer
    from src.backends.opencl.type_mapping import build_compiler_flags
    from src.shared.hardware_profile import HardwareProfile
    from src.shared.precision_config import PrecisionConfig

    ctx = cl.create_some_context(interactive=False)
    device = ctx.devices[0]
    queue = cl.CommandQueue(ctx, device)
    hw = discover_hardware(device)
    prec = PrecisionConfig.float32()
    flags = build_compiler_flags(prec, hw, c_tile_size=16)
    program = load_and_compile_kernels_from_path(ctx, device, flags, _KERNEL_DIR)
    dispatch_table = build_dispatch_table()

    renderer = OpenCLPlanRenderer(
        context=ctx, queue=queue, program=program,
        kernel_bindings=dispatch_table, hardware=hw,
    )
    renderer.set_reduction_bindings(
        register_reduce=AggregateRegisterReduceBinding(),
        local_reduce=AggregateLocalReduceBinding(),
        clip_intermediate=ClipIntermediateGradBinding(),
        k_fan_in=ReduceKFanInAndClipBinding(),
    )
    return renderer


def _create_vulkan_renderer() -> Any:
    """Create a fully initialized VulkanPlanRenderer."""
    from src.backends.vulkan.context import VulkanContext
    from src.backends.vulkan.renderer import VulkanPlanRenderer

    ctx = VulkanContext(enable_validation=False)
    return VulkanPlanRenderer(context=ctx)


_BACKEND_FACTORIES: dict[str, Any] = {
    "cpu": _create_cpu_renderer,
    "opencl": _create_opencl_renderer,
    "vulkan": _create_vulkan_renderer,
}


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def oracle_backend() -> str | None:
    """The name of the oracle backend, or None."""
    return _select_oracle()


@pytest.fixture(scope="session")
def available_backends() -> list[str]:
    """List of available backend names."""
    return _get_available_backends()


@pytest.fixture(scope="session")
def renderer_factory():
    """Factory that creates a renderer for a named backend.

    Created renderers are cached for the session to amortize initialization.
    Backends whose renderer cannot be created (e.g. missing module) raise
    pytest.skip at invocation time.
    """
    cache: dict[str, Any] = {}
    failed: dict[str, str] = {}

    def _get(backend_name: str) -> Any:
        if backend_name in failed:
            pytest.skip(f"Backend '{backend_name}' unavailable: {failed[backend_name]}")
        if backend_name not in cache:
            factory = _BACKEND_FACTORIES.get(backend_name)
            if factory is None:
                failed[backend_name] = "no renderer factory registered"
                pytest.skip(f"Backend '{backend_name}' unavailable: no renderer factory")
            try:
                cache[backend_name] = factory()
            except (ImportError, ModuleNotFoundError, OSError) as exc:
                failed[backend_name] = str(exc)
                pytest.skip(f"Backend '{backend_name}' unavailable: {exc}")
        return cache[backend_name]

    return _get


@pytest.fixture(scope="session")
def tier3_pairs(request) -> list[tuple[str, str]]:
    """Backend pairs for Tier 3 comparison."""
    return _get_tier3_pairs(request.config)


# ---------------------------------------------------------------------------
# Single-kernel plan extraction (Step 4.5)
# ---------------------------------------------------------------------------

def _extract_node_plan(
    source_plan: Any,
    node_id: str,
    output_binding_name: str,
) -> Any:
    """Extract a single node from a plan into a minimal ExecutionPlan.

    Creates a two-node plan: the target kernel (with dependencies cleared)
    followed by a RetrievalNode on the specified output buffer. Both
    backends execute this identical plan from zero-initialized buffers,
    making the comparison meaningful for cross-backend parity.
    """
    from dataclasses import replace

    from src.shared.plan_types import (
        ExecutionPlan,
        KernelDispatchNode,
        ReductionTreeNode,
        RetrievalNode,
    )

    node = source_plan.nodes[node_id]

    if isinstance(node, KernelDispatchNode):
        isolated = replace(node, depends_on=frozenset())
        needed_handles = set(node.buffer_bindings.values())
        output_handle = node.buffer_bindings[output_binding_name]
    elif isinstance(node, ReductionTreeNode):
        isolated = replace(node, depends_on=frozenset())
        rp = node.reduction_plan
        needed_handles = {rp.source_buffer, rp.destination_buffer}
        output_handle = rp.destination_buffer
    else:
        raise ValueError(f"Cannot extract node type {type(node)}")

    output_desc = source_plan.buffers[output_handle]

    retrieval = RetrievalNode(
        node_id="retrieval",
        depends_on=frozenset({node_id}),
        source_buffer=output_handle,
        logical_shape=output_desc.padded_shape,
        event_name="output",
    )

    needed_handles.add(output_handle)
    buffers = {h: source_plan.buffers[h] for h in needed_handles}
    nodes_dict = {node_id: isolated, "retrieval": retrieval}
    topo = (node_id, "retrieval")

    return ExecutionPlan(
        nodes=nodes_dict,
        buffers=buffers,
        topological_order=topo,
        precision=source_plan.precision,
        hardware=source_plan.hardware,
    )


class _SingleKernelPlanBuilder:
    """Builds minimal single-kernel plans by extracting from full plans.

    Constructs full Act and Learn plans on init, then exposes a build()
    method that extracts any named kernel into a minimal two-node plan
    (kernel + retrieval). Both backends execute the same plan from
    zero-initialized buffers for parity comparison.
    """

    # Mapping: abbreviated kernel name →
    #   (plan_attr, node_id, output_binding_name_or_None_for_reduction)
    _KERNEL_MAP: dict[str, tuple[str, str, str | None]] = {
        # Act phase
        "forward_pass": ("_act_cce", "forward_pass", "hidden_activations"),
        "render_logits_chunk": ("_act_cce", "render_logits", "logits"),
        "compute_probs_loss_cce_chunk": (
            "_act_cce", "loss_computation", "partial_probs",
        ),
        "compute_probs_loss_bce_chunk": (
            "_act_bce", "loss_computation", "partial_probs",
        ),
        # Learn Phase I — gradient production
        "calculate_module_param_grads_chunk": (
            "_learn_cce", "calc_module_grads",
            "partial_grad_weights_module",
        ),
        "backprop_error_to_hidden_chunk": (
            "_learn_cce", "backprop_error_hidden",
            "partial_grad_hidden_activations_aos",
        ),
        "calculate_chunk_temp_gradients": (
            "_learn_cce", "calc_temp_grads", "partial_grad_temps",
        ),
        "clip_partial_gradients": (
            "_learn_cce", "clip_partial_grads",
            "clipped_partial_grad_weights_module",
        ),
        # Learn Phase II — aggregation
        "gather_and_permute_grad_hidden_activations": (
            "_learn_cce", "gather_permute_grad_h",
            "clipped_grad_hidden_activations_permuted_soa",
        ),
        "stabilize_and_reduce_grad_hidden_activations": (
            "_learn_cce", "stabilize_reduce_grad_h",
            "summed_grad_hidden_activations",
        ),
        "aggregate_register_reduce": (
            "_learn_cce", "reduce_mod_grads", None,
        ),
        "aggregate_local_reduce": (
            "_learn_cce", "reduce_temp_grads", None,
        ),
        "clip_intermediate_grad": (
            "_learn_cce", "reduce_mod_grads", None,
        ),
        # Learn Phase III — streaming backprop
        "backprop_shared_weights_chunk": (
            "_learn_cce", "backprop_shared_weights",
            "partial_grad_weights_shared",
        ),
        "backprop_shared_biases_chunk": (
            "_learn_cce", "backprop_shared_biases",
            "partial_grad_biases_shared",
        ),
        "clip_shared_gradients_chunk": (
            "_learn_cce", "clip_shared_grads",
            "clipped_partial_grad_weights_shared",
        ),
        # Learn Phase IV — normalization
        "normalize_gradients": (
            "_learn_cce", "normalize_gradients_module", "final_grad",
        ),
        # Learn Phase V — parameter update
        "adam_update": (
            "_learn_cce", "adam_update_shared", "parameters",
        ),
        "clamp_temperatures": (
            "_learn_cce", "clamp_temperatures", "temperatures",
        ),
    }

    def __init__(self) -> None:
        import numpy as np_

        from src.shared.hardware_profile import HardwareProfile
        from src.shared.model_spec import ModelSpec
        from src.shared.plan_builder import build_act_plan, build_learn_plan
        from src.shared.precision_config import PrecisionConfig
        from src.shared.problem_type_strategy import PlanBceStrategy, PlanCceStrategy
        from src.shared.stabilization_policy import StabilizationPolicy

        iris = dict(input_dim=4, hidden_dim=32, output_classes=3, num_modules=8)
        batch = 150

        prec = PrecisionConfig.float32()
        spec = ModelSpec(
            precision=prec, simd_width=4, cache_line_bytes=64, **iris,
        )
        hw = HardwareProfile(
            simd_width=4, cache_line_bytes=64, max_reduce_fan_in=256,
            max_local_mem_bytes=65536, global_mem_bytes=4 * 1024**3,
        )
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            fp_format_max=float(np_.finfo(np_.float32).max),
        )

        cce = PlanCceStrategy()
        bce = PlanBceStrategy()

        self._act_cce = build_act_plan(spec, hw, cce, batch)
        self._act_bce = build_act_plan(spec, hw, bce, batch)
        self._learn_cce = build_learn_plan(spec, hw, cce, batch, policy)
        self._learn_bce = build_learn_plan(spec, hw, bce, batch, policy)

    def build(self, kernel_name: str) -> Any:
        """Build a minimal ExecutionPlan for the given kernel.

        Returns an ExecutionPlan containing a single kernel dispatch node
        (or reduction tree node) plus a retrieval node. The retrieval
        event is always named "output".
        """
        if kernel_name not in self._KERNEL_MAP:
            raise ValueError(
                f"Unknown kernel '{kernel_name}'. "
                f"Known: {sorted(self._KERNEL_MAP)}"
            )

        plan_attr, node_id, output_binding = self._KERNEL_MAP[kernel_name]
        source_plan = getattr(self, plan_attr)

        if output_binding is None:
            # Reduction node — output_binding not used
            return _extract_node_plan(source_plan, node_id, "")
        return _extract_node_plan(source_plan, node_id, output_binding)


@pytest.fixture(scope="session")
def single_kernel_plan_factory():
    """Session-scoped factory for building minimal single-kernel plans.

    Usage: ``plan = single_kernel_plan_factory.build("forward_pass")``

    The factory builds full Act/Learn plans once, then extracts
    individual kernels into minimal two-node plans (kernel + retrieval).
    """
    return _SingleKernelPlanBuilder()
