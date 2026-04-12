# src/backends/opencl/kernel_bindings/dispatch_table.py
"""Kernel dispatch table: kernel_name → KernelBinding mapping (ADR-007).

This table is the single authoritative registry of all OpenCL kernel
bindings.  Keys are kernel names exactly as they appear in kernels.cl.h
(the ``__kernel void`` identifier), ensuring a 1:1 correspondence between
the algorithmic specification and the dispatch infrastructure.

The Orchestration tier's plan renderer looks up bindings by kernel name
from ``KernelDispatchNode.kernel_name``.  The reduction-tree renderer
may access reduction-engine bindings from this same table or receive
them via ``set_reduction_bindings()`` — both paths draw from the same
binding instances.
"""
from __future__ import annotations

from .base import KernelBinding

# --- Phase 1: Act (Forward Pass & Loss) ---
from .binding_phase_1_act import (
    ComputeProbsLossBceChunkBinding,
    ComputeProbsLossCceChunkBinding,
    ForwardPassBinding,
    RenderLogitsChunkBinding,
)

# --- Phase 2A: Learn — Gradient Production (Nodes 8, 9, 10) ---
from .binding_phase_2_learn_A import (
    BackpropErrorToHiddenChunkBinding,
    CalculateChunkTempGradientsBinding,
    CalculateModuleParamGradsChunkBinding,
)

# --- Phase 2B: Learn — Gradient Processing (Nodes 11, 13) ---
from .binding_phase_2_learn_B import (
    ClipPartialGradientsBinding,
    GatherAndPermuteGradHBinding,
)

# --- Phase 2C: Learn — Reduction & Aggregation Engine ---
from .binding_phase_2_learn_C import (
    AggregateLocalReduceBinding,
    AggregateLocalReduceFromComputeBinding,
    AggregateRegisterReduceBinding,
    AggregateRegisterReduceFromComputeBinding,
    ClipIntermediateGradBinding,
    ReduceKFanInAndClipBinding,
    ReduceKFanInAndClipFromComputeBinding,
    StabilizeReduceGradHBinding,
)

# --- Phase 2D: Learn — Streaming Shared-Layer Backprop (Nodes 17, 18, 19) ---
from .binding_phase_2_learn_D import (
    BackpropSharedBiasesChunkBinding,
    BackpropSharedWeightsChunkBinding,
    ClipSharedGradientsChunkBinding,
)

# --- Phase 3: Finalization & Parameter Updates (Nodes 21, 24, 25) ---
from .binding_phase_3_update import (
    AdamUpdateBinding,
    ClampTemperaturesBinding,
    NormalizeGradientsBinding,
)


def build_dispatch_table() -> dict[str, KernelBinding]:
    """Construct the kernel_name → KernelBinding dispatch table.

    Injected into ``OpenCLPlanRenderer`` at initialization.  Every entry
    maps a kernel name (matching the ``__kernel void`` identifier in
    ``kernels.cl.h``) to the ``KernelBinding`` instance that renders its
    dispatch arguments and grid geometry.

    Reduction-engine bindings (aggregate_*, reduce_k_fan_in_and_clip*,
    clip_intermediate_grad) are included for registry completeness.  The
    reduction-tree renderer accesses them via their ``marshal_args_direct``
    / ``compute_grid_direct`` convenience methods — the same instances
    serve both plan-driven and programmatic dispatch paths.
    """
    return {
        # -- Act phase (Nodes 4–7) -----------------------------------------
        "forward_pass": ForwardPassBinding(),
        "render_logits_chunk": RenderLogitsChunkBinding(),
        "compute_probs_loss_cce_chunk": ComputeProbsLossCceChunkBinding(),
        "compute_probs_loss_bce_chunk": ComputeProbsLossBceChunkBinding(),
        # -- Gradient production (Nodes 8–10) ------------------------------
        "calculate_module_param_grads_chunk": CalculateModuleParamGradsChunkBinding(),
        "backprop_error_to_hidden_chunk": BackpropErrorToHiddenChunkBinding(),
        "calculate_chunk_temp_gradients": CalculateChunkTempGradientsBinding(),
        # -- Gradient processing (Nodes 11, 13) ----------------------------
        "clip_partial_gradients": ClipPartialGradientsBinding(),
        "gather_and_permute_grad_hidden_activations": GatherAndPermuteGradHBinding(),
        # -- Reduction engine (Nodes 14, 15, 16, 20; ADR-019/026) ---------
        "aggregate_register_reduce": AggregateRegisterReduceBinding(),
        "aggregate_local_reduce": AggregateLocalReduceBinding(),
        "aggregate_register_reduce_from_compute": AggregateRegisterReduceFromComputeBinding(),
        "aggregate_local_reduce_from_compute": AggregateLocalReduceFromComputeBinding(),
        "clip_intermediate_grad": ClipIntermediateGradBinding(),
        "reduce_k_fan_in_and_clip": ReduceKFanInAndClipBinding(),
        "reduce_k_fan_in_and_clip_from_compute": ReduceKFanInAndClipFromComputeBinding(),
        "stabilize_and_reduce_grad_hidden_activations": StabilizeReduceGradHBinding(),
        # -- Streaming shared-layer backprop (Nodes 17–19) -----------------
        "backprop_shared_weights_chunk": BackpropSharedWeightsChunkBinding(),
        "backprop_shared_biases_chunk": BackpropSharedBiasesChunkBinding(),
        "clip_shared_gradients_chunk": ClipSharedGradientsChunkBinding(),
        # -- Finalization & updates (Nodes 21, 24, 25) ---------------------
        "normalize_gradients": NormalizeGradientsBinding(),
        "adam_update": AdamUpdateBinding(),
        "clamp_temperatures": ClampTemperaturesBinding(),
    }
