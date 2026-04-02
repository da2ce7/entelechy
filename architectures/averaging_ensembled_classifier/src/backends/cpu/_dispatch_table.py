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
    suffix: str,
) -> dict[str, tuple[Any, type[ctypes.Structure]]]:
    """Build the kernel_name → (task_fn_ptr, args_struct_class) mapping.

    *suffix* selects the precision variant using three-axis suffixes
    (e.g. ``"s32c32x32"``, ``"s16c32x16"``, ``"s64c64x64"``).
    C function names and struct classes are resolved accordingly.

    Strategy A kernels (Nodes 8, 9, 10): CCE and BCE variants map to
    the SAME C task function — the problem_type flag is in the struct.
    Strategy B kernels (Nodes 6, 7): CCE and BCE map to DIFFERENT functions.
    """
    s = ffi.PRECISION_STRUCTS[suffix]

    return {
        # Act phase
        "forward_pass": (
            _fn_ptr(lib, f"task_forward_pass_{suffix}"),
            s["ForwardPassArgs"],
        ),
        "render_logits_chunk": (
            _fn_ptr(lib, f"task_render_logits_{suffix}"),
            s["RenderLogitsArgs"],
        ),
        # Learn phase A — production (Strategy B)
        "compute_probs_loss_cce_chunk": (
            _fn_ptr(lib, f"task_cce_probs_loss_{suffix}"),
            s["CceChunkArgs"],
        ),
        "compute_probs_loss_bce_chunk": (
            _fn_ptr(lib, f"task_bce_probs_loss_{suffix}"),
            s["BceChunkArgs"],
        ),
        # Learn phase A — production (Strategy A)
        "calculate_module_param_grads_chunk": (
            _fn_ptr(lib, f"task_module_param_grads_{suffix}"),
            s["ModuleParamGradsArgs"],
        ),
        # Learn phase B — processing (Strategy A)
        "backprop_error_to_hidden_chunk": (
            _fn_ptr(lib, f"task_backprop_to_hidden_{suffix}"),
            s["BackpropToHiddenArgs"],
        ),
        "calculate_chunk_temp_gradients": (
            _fn_ptr(lib, f"task_temp_gradients_{suffix}"),
            s["TempGradientsArgs"],
        ),
        "clip_partial_gradients": (
            _fn_ptr(lib, f"task_clip_partial_grads_{suffix}"),
            s["ClipPartialsArgs"],
        ),
        # Learn phase C — reduction
        "gather_and_permute_grad_hidden_activations": (
            _fn_ptr(lib, f"task_gather_permute_grad_h_{suffix}"),
            s["GatherPermuteArgs"],
        ),
        "stabilize_and_reduce_grad_hidden_activations": (
            _fn_ptr(lib, f"task_stabilize_reduce_grad_h_{suffix}"),
            s["StabilizeReduceArgs"],
        ),
        "clip_intermediate_grad": (
            _fn_ptr(lib, f"task_clip_intermediate_{suffix}"),
            s["ClipIntermediateArgs"],
        ),
        # Learn phase D — streaming backprop
        "backprop_shared_weights_chunk": (
            _fn_ptr(lib, f"task_backprop_shared_weights_{suffix}"),
            s["BackpropSharedWeightsArgs"],
        ),
        "backprop_shared_biases_chunk": (
            _fn_ptr(lib, f"task_backprop_shared_biases_{suffix}"),
            s["BackpropSharedBiasesArgs"],
        ),
        "clip_shared_gradients_chunk": (
            _fn_ptr(lib, f"task_clip_shared_grads_{suffix}"),
            s["ClipSharedGradsArgs"],
        ),
        # Update phase
        "normalize_gradients": (
            _fn_ptr(lib, f"task_normalize_gradients_{suffix}"),
            s["NormalizeGradientsArgs"],
        ),
        "adam_update": (
            _fn_ptr(lib, f"task_adam_update_{suffix}"),
            s["AdamUpdateArgs"],
        ),
        "clamp_temperatures": (
            _fn_ptr(lib, f"task_clamp_temperatures_{suffix}"),
            s["ClampTemperaturesArgs"],
        ),
    }


def _fn_ptr(lib: ctypes.CDLL, symbol_name: str) -> ctypes.c_void_p:
    """Resolve a task function's raw C address from the loaded library.

    Returns c_void_p to avoid creating a Python ffi closure when the
    address is passed to pool_dispatch_and_wait.  Worker threads call
    the function without the GIL, so the pointer must go straight to C.
    """
    return ctypes.cast(getattr(lib, symbol_name), ctypes.c_void_p)
