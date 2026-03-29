"""Dispatch table: kernel_name → (task_fn_ptr, args_struct_class) (ADR-015).

Maps plan-model kernel names from KernelDispatchNode.kernel_name to
C task function pointers and their argument struct classes.
"""
from __future__ import annotations

import ctypes
from typing import Any

from . import _ffi_types as ffi


def build_dispatch_table(
    lib: ctypes.CDLL,
) -> dict[str, tuple[Any, type[ctypes.Structure]]]:
    """Build the kernel_name → (task_fn_ptr, args_struct_class) mapping.

    Strategy A kernels (Nodes 8, 9, 10): CCE and BCE variants map to
    the SAME C task function — the problem_type flag is in the struct.
    Strategy B kernels (Nodes 6, 7): CCE and BCE map to DIFFERENT functions.
    """
    return {
        # Act phase
        "forward_pass": (
            _fn_ptr(lib, "task_forward_pass"),
            ffi.ForwardPassArgs,
        ),
        "render_logits_chunk": (
            _fn_ptr(lib, "task_render_logits"),
            ffi.RenderLogitsArgs,
        ),
        # Learn phase A — production (Strategy B)
        "compute_probs_loss_cce_chunk": (
            _fn_ptr(lib, "task_cce_probs_loss"),
            ffi.CceChunkArgs,
        ),
        "compute_probs_loss_bce_chunk": (
            _fn_ptr(lib, "task_bce_probs_loss"),
            ffi.BceChunkArgs,
        ),
        # Learn phase A — production (Strategy A)
        "calculate_module_param_grads_chunk": (
            _fn_ptr(lib, "task_module_param_grads"),
            ffi.ModuleParamGradsArgs,
        ),
        # Learn phase B — processing (Strategy A)
        "backprop_error_to_hidden_chunk": (
            _fn_ptr(lib, "task_backprop_to_hidden"),
            ffi.BackpropToHiddenArgs,
        ),
        "calculate_chunk_temp_gradients": (
            _fn_ptr(lib, "task_temp_gradients"),
            ffi.TempGradientsArgs,
        ),
        "clip_partial_gradients": (
            _fn_ptr(lib, "task_clip_partial_grads"),
            ffi.ClipPartialsArgs,
        ),
        # Learn phase C — reduction
        "gather_and_permute_grad_hidden_activations": (
            _fn_ptr(lib, "task_gather_permute_grad_h"),
            ffi.GatherPermuteArgs,
        ),
        "stabilize_and_reduce_grad_hidden_activations": (
            _fn_ptr(lib, "task_stabilize_reduce_grad_h"),
            ffi.StabilizeReduceArgs,
        ),
        "clip_intermediate_grad": (
            _fn_ptr(lib, "task_clip_intermediate"),
            ffi.ClipIntermediateArgs,
        ),
        # Learn phase D — streaming backprop
        "backprop_shared_weights_chunk": (
            _fn_ptr(lib, "task_backprop_shared_weights"),
            ffi.BackpropSharedWeightsArgs,
        ),
        "backprop_shared_biases_chunk": (
            _fn_ptr(lib, "task_backprop_shared_biases"),
            ffi.BackpropSharedBiasesArgs,
        ),
        "clip_shared_gradients_chunk": (
            _fn_ptr(lib, "task_clip_shared_grads"),
            ffi.ClipSharedGradsArgs,
        ),
        # Update phase
        "normalize_gradients": (
            _fn_ptr(lib, "task_normalize_gradients"),
            ffi.NormalizeGradientsArgs,
        ),
        "adam_update": (
            _fn_ptr(lib, "task_adam_update"),
            ffi.AdamUpdateArgs,
        ),
        "clamp_temperatures": (
            _fn_ptr(lib, "task_clamp_temperatures"),
            ffi.ClampTemperaturesArgs,
        ),
    }


def _fn_ptr(lib: ctypes.CDLL, symbol_name: str) -> Any:
    """Resolve a task function's address from the loaded library."""
    return getattr(lib, symbol_name)
