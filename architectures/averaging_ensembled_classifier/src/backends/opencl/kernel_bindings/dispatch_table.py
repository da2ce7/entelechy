# src/backends/opencl/kernel_bindings/dispatch_table.py
"""Kernel dispatch table: kernel_name -> KernelBinding mapping (ADR-007)."""
from __future__ import annotations

from .base import KernelBinding
from .binding_phase_1_act import (
    ForwardPassBinding,
    RenderLogitsChunkBinding,
    ComputeProbsLossCceBinding,
    ComputeProbsLossBceBinding,
)
from .binding_phase_2_learn_A import (
    CalculateModuleParamGradsCceBinding,
    CalculateModuleParamGradsBceBinding,
    BackpropErrorToHiddenBinding,
    CalculateChunkTempGradientsBinding,
)
from .binding_phase_2_learn_B import (
    ClipPartialGradientsBinding,
    GatherAndPermuteGradHBinding,
)
from .binding_phase_2_learn_C import (
    StabilizeReduceGradHBinding,
)
from .binding_phase_2_learn_D import (
    BackpropSharedWeightsBinding,
    BackpropSharedBiasesBinding,
    ClipSharedGradientsBinding,
)
from .binding_phase_3_update import (
    NormalizeGradientsBinding,
    AdamUpdateBinding,
    ClampTemperaturesBinding,
)


def build_dispatch_table() -> dict[str, KernelBinding]:
    """Construct the kernel_name -> KernelBinding dispatch table.

    This table is injected into OpenCLPlanRenderer at initialization.
    Each entry maps a kernel_name (as it appears in KernelDispatchNode.kernel_name
    and in kernels.cl.h) to the KernelBinding instance that can dispatch it.

    NOTE: Reduction engine bindings (aggregate_register_reduce,
    aggregate_local_reduce, clip_intermediate_grad) are NOT in this table.
    They are consumed directly by _render_reduction_tree() via
    set_reduction_bindings().
    """
    return {
        "forward_pass": ForwardPassBinding(),
        "render_logits_chunk": RenderLogitsChunkBinding(),
        "compute_probs_loss_cce_chunk": ComputeProbsLossCceBinding(),
        "compute_probs_loss_bce_chunk": ComputeProbsLossBceBinding(),
        "calculate_module_param_grads_cce": CalculateModuleParamGradsCceBinding(),
        "calculate_module_param_grads_bce": CalculateModuleParamGradsBceBinding(),
        "backprop_error_to_hidden": BackpropErrorToHiddenBinding(),
        "calculate_chunk_temp_gradients": CalculateChunkTempGradientsBinding(),
        "clip_partial_gradients": ClipPartialGradientsBinding(),
        "gather_and_permute_grad_h": GatherAndPermuteGradHBinding(),
        "stabilize_reduce_grad_h": StabilizeReduceGradHBinding(),
        "backprop_shared_weights_chunk": BackpropSharedWeightsBinding(),
        "backprop_shared_biases_chunk": BackpropSharedBiasesBinding(),
        "clip_shared_gradients_chunk": ClipSharedGradientsBinding(),
        "normalize_gradients": NormalizeGradientsBinding(),
        "adam_update": AdamUpdateBinding(),
        "clamp_temperatures": ClampTemperaturesBinding(),
    }
