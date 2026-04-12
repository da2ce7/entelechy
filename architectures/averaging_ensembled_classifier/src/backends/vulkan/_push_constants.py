"""Push constant struct definitions and marshalling (VULKAN_BACKEND.md §Push Constants).

Each ctypes.Structure mirrors the GLSL push_constant layout for a shader.
The marshal function converts plan node scalar_params into packed bytes.

Precision-Aware Push Constants (ADR-024):
Some kernels (Node 11, Node 19) have COMPUTE_TYPE scalar parameters per the
kernel spec. When COMPUTE_TYPE = double, the GLSL layout changes due to FP64
alignment requirements. This module provides FP64 struct variants and a
lookup mechanism that selects the correct variant based on compute_dtype.
"""
from __future__ import annotations

from collections.abc import Mapping

import ctypes

import numpy as np


# ── Push Constant Structures ──
# Field order MUST match the GLSL layout(push_constant) declarations.

class ForwardPassPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_input_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("FLAG_produce_hidden_mask", ctypes.c_uint32),
    ]


class RenderLogitsPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("module_chunk_offset", ctypes.c_uint32),
        ("module_chunk_count", ctypes.c_uint32),
        ("class_chunk_offset", ctypes.c_uint32),
        ("class_chunk_count", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("hidden_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_output_class_count", ctypes.c_uint32),
        ("padded_total_output_class_count", ctypes.c_uint32),
        ("total_modules_count", ctypes.c_uint32),
        ("FLAG_use_explicit_hidden_mask", ctypes.c_uint32),
    ]


class ProbsLossPush(ctypes.Structure):
    _fields_ = [
        ("num_class_chunks", ctypes.c_uint32),
        ("classes_per_chunk", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("total_output_class_count", ctypes.c_uint32),
        ("padded_total_output_class_count", ctypes.c_uint32),
        ("total_modules_count", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
    ]


class GradientTilePush(ctypes.Structure):
    _fields_ = [
        ("num_class_chunks", ctypes.c_uint32),
        ("classes_per_chunk", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("total_output_class_count", ctypes.c_uint32),
        ("padded_total_output_class_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_modules_count", ctypes.c_uint32),
        ("padded_total_modules_count", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
    ]


class ClipPartialsPush(ctypes.Structure):
    """ClipPartialsPush for COMPUTE_TYPE = float (FP32)."""
    _fields_ = [
        ("use_per_item_norm", ctypes.c_uint32),
        ("clipping_threshold", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("num_class_chunks", ctypes.c_uint32),
        ("classes_per_chunk", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
        ("partial_stride", ctypes.c_uint32),
    ]


class GatherPermutePush(ctypes.Structure):
    _fields_ = [
        ("total_batch_count", ctypes.c_uint32),
        ("hidden_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_modules_count", ctypes.c_uint32),
        ("padded_total_modules_count", ctypes.c_uint32),
        ("num_module_chunks", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("num_class_chunks", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
    ]


class AggregatePush(ctypes.Structure):
    _fields_ = [
        ("partial_offset_list_count", ctypes.c_uint32),
        ("partial_width", ctypes.c_uint32),
        ("operation_type", ctypes.c_uint32),
        ("use_local_reduce", ctypes.c_uint32),
    ]


class ClipIntermediatePush(ctypes.Structure):
    _fields_ = [
        ("clipping_threshold", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("parameter_count", ctypes.c_uint32),
    ]


class ReduceKFanInPush(ctypes.Structure):
    """Push constants for reduce_k_fan_in_and_clip (ADR-019) COMPUTE_TYPE = float."""
    _fields_ = [
        ("fan_in", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
        ("partial_width", ctypes.c_uint32),
        ("clipping_threshold", ctypes.c_float),
        ("epsilon", ctypes.c_float),
    ]


class StabilizeReducePush(ctypes.Structure):
    _fields_ = [
        ("num_reduction_stages", ctypes.c_uint32),
        ("clipping_threshold_t_pre", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_modules_count", ctypes.c_uint32),
        ("padded_total_modules_count", ctypes.c_uint32),
    ]


class SharedBackpropWeightsPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("batch_chunk_index", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("num_batch_chunks", ctypes.c_uint32),
        ("input_count", ctypes.c_uint32),
        ("padded_input_count", ctypes.c_uint32),
        ("hidden_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("final_grad_hidden_activations_total_count", ctypes.c_uint32),
        ("FLAG_use_explicit_hidden_mask", ctypes.c_uint32),
    ]


class SharedBackpropBiasesPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("batch_chunk_index", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("num_batch_chunks", ctypes.c_uint32),
        ("hidden_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("final_grad_hidden_activations_total_count", ctypes.c_uint32),
        ("FLAG_use_explicit_hidden_mask", ctypes.c_uint32),
    ]


class ClipSharedPush(ctypes.Structure):
    """ClipSharedPush for COMPUTE_TYPE = float (FP32)."""
    _fields_ = [
        ("clipping_threshold", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("weights_parameter_count", ctypes.c_uint32),
        ("biases_parameter_count", ctypes.c_uint32),
        ("weights_write_offset", ctypes.c_uint32),
        ("biases_write_offset", ctypes.c_uint32),
        ("num_batch_chunks", ctypes.c_uint32),
    ]


# ── FP64 (double) Push Constant Variants (ADR-024) ──
# These match GLSL std430 layout when COMPUTE_TYPE = double.
# Padding fields account for double alignment requirements.


class ClipPartialsPushFP64(ctypes.Structure):
    """ClipPartialsPush for COMPUTE_TYPE = double (FP64).

    std430 layout: uint32 followed by double requires 4 bytes padding.
    """
    _pack_ = 1  # Disable automatic padding; we add explicit padding
    _fields_ = [
        ("use_per_item_norm", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),  # Align to 8 bytes for double
        ("clipping_threshold", ctypes.c_double),
        ("epsilon", ctypes.c_double),
        ("num_class_chunks", ctypes.c_uint32),
        ("classes_per_chunk", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
        ("partial_stride", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),  # Pad to 8-byte boundary at end
    ]


class ClipSharedPushFP64(ctypes.Structure):
    """ClipSharedPush for COMPUTE_TYPE = double (FP64).

    No leading padding since first field is already double.
    """
    _pack_ = 1
    _fields_ = [
        ("clipping_threshold", ctypes.c_double),
        ("epsilon", ctypes.c_double),
        ("weights_parameter_count", ctypes.c_uint32),
        ("biases_parameter_count", ctypes.c_uint32),
        ("weights_write_offset", ctypes.c_uint32),
        ("biases_write_offset", ctypes.c_uint32),
        ("num_batch_chunks", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),  # Pad to 8-byte boundary at end
    ]


class ClipIntermediatePushFP64(ctypes.Structure):
    """ClipIntermediatePush for COMPUTE_TYPE = double (FP64).

    No leading padding since first field is already double.
    """
    _pack_ = 1
    _fields_ = [
        ("clipping_threshold", ctypes.c_double),
        ("epsilon", ctypes.c_double),
        ("parameter_count", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),  # Pad to 8-byte boundary at end
    ]


class ReduceKFanInPushFP64(ctypes.Structure):
    """ReduceKFanInPush for COMPUTE_TYPE = double (FP64, ADR-019).

    std430 layout: 3 uints (12 bytes) + 4 bytes padding = 16 bytes align for doubles.
    """
    _pack_ = 1
    _fields_ = [
        ("fan_in", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
        ("partial_width", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),  # Align to 8 bytes for double
        ("clipping_threshold", ctypes.c_double),
        ("epsilon", ctypes.c_double),
    ]


# ── FP16 (half) Push Constant Variants (ADR-024 compliance) ──
# These match GLSL std430 layout when COMPUTE_TYPE = float16_t.
# FP16 fields use c_uint16 to hold the IEEE 754 half-precision bit pattern;
# marshal_push_constants converts Python floats to float16 bits.


class ClipPartialsPushFP16(ctypes.Structure):
    """ClipPartialsPush for COMPUTE_TYPE = float16_t (FP16).

    std430 layout: uint followed by two float16 (each 2 bytes) is tightly packed.
    """
    _pack_ = 1
    _fields_ = [
        ("use_per_item_norm", ctypes.c_uint32),
        ("clipping_threshold", ctypes.c_uint16),  # float16 as IEEE 754 bits
        ("epsilon", ctypes.c_uint16),              # float16 as IEEE 754 bits
        ("num_class_chunks", ctypes.c_uint32),
        ("classes_per_chunk", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
        ("partial_stride", ctypes.c_uint32),
    ]


class ClipSharedPushFP16(ctypes.Structure):
    """ClipSharedPush for COMPUTE_TYPE = float16_t (FP16).

    std430 layout: two float16 (4 bytes total) followed by uints.
    """
    _pack_ = 1
    _fields_ = [
        ("clipping_threshold", ctypes.c_uint16),  # float16 as IEEE 754 bits
        ("epsilon", ctypes.c_uint16),              # float16 as IEEE 754 bits
        ("weights_parameter_count", ctypes.c_uint32),
        ("biases_parameter_count", ctypes.c_uint32),
        ("weights_write_offset", ctypes.c_uint32),
        ("biases_write_offset", ctypes.c_uint32),
        ("num_batch_chunks", ctypes.c_uint32),
    ]


class ClipIntermediatePushFP16(ctypes.Structure):
    """ClipIntermediatePush for COMPUTE_TYPE = float16_t (FP16).

    std430 layout: two float16 (4 bytes total) followed by uint.
    """
    _pack_ = 1
    _fields_ = [
        ("clipping_threshold", ctypes.c_uint16),  # float16 as IEEE 754 bits
        ("epsilon", ctypes.c_uint16),              # float16 as IEEE 754 bits
        ("parameter_count", ctypes.c_uint32),
    ]


class ReduceKFanInPushFP16(ctypes.Structure):
    """ReduceKFanInPush for COMPUTE_TYPE = float16_t (FP16, ADR-019).

    std430 layout: 3 uints (12 bytes) + 2 float16 (4 bytes, packed) = 16 bytes.
    """
    _pack_ = 1
    _fields_ = [
        ("fan_in", ctypes.c_uint32),
        ("node_count", ctypes.c_uint32),
        ("partial_width", ctypes.c_uint32),
        ("clipping_threshold", ctypes.c_uint16),  # float16 as IEEE 754 bits
        ("epsilon", ctypes.c_uint16),              # float16 as IEEE 754 bits
    ]


class NormalizePush(ctypes.Structure):
    _fields_ = [
        ("effective_batch_size", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("parameter_count", ctypes.c_uint32),
    ]


class NormalizePushFP64(ctypes.Structure):
    """NormalizePush for COMPUTE_TYPE = double (FP64).

    std430 layout: two doubles (16 bytes) followed by uint with padding.
    """
    _fields_ = [
        ("effective_batch_size", ctypes.c_double),
        ("epsilon", ctypes.c_double),
        ("parameter_count", ctypes.c_uint32),
        ("_pad0", ctypes.c_uint32),  # Align to 8-byte boundary
    ]


class NormalizePushFP16(ctypes.Structure):
    """NormalizePush for COMPUTE_TYPE = float16_t (FP16).

    std430 layout: two float16 (4 bytes total) followed by uint.
    """
    _pack_ = 1
    _fields_ = [
        ("effective_batch_size", ctypes.c_uint16),  # float16 as IEEE 754 bits
        ("epsilon", ctypes.c_uint16),                # float16 as IEEE 754 bits
        ("parameter_count", ctypes.c_uint32),
    ]


class AdamUpdatePush(ctypes.Structure):
    _fields_ = [
        ("learning_rate", ctypes.c_float),
        ("beta1_pow_t", ctypes.c_float),
        ("beta2_pow_t", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("parameter_offset", ctypes.c_uint32),
        ("parameter_count", ctypes.c_uint32),
        ("total_parameter_count", ctypes.c_uint32),
    ]


class ClampTempsPush(ctypes.Structure):
    _fields_ = [
        ("min_value", ctypes.c_float),
        ("max_value", ctypes.c_float),
        ("parameter_offset", ctypes.c_uint32),
        ("parameter_count", ctypes.c_uint32),
        ("total_parameter_count", ctypes.c_uint32),
    ]


# ── Kernel name → struct class mapping ──

PUSH_CONSTANT_STRUCTS: dict[str, type[ctypes.Structure]] = {
    "forward_pass": ForwardPassPush,
    "render_logits_chunk": RenderLogitsPush,
    "compute_probs_loss_cce_chunk": ProbsLossPush,
    "compute_probs_loss_bce_chunk": ProbsLossPush,
    "calculate_module_param_grads_chunk": GradientTilePush,
    "backprop_error_to_hidden_chunk": GradientTilePush,
    "calculate_chunk_temp_gradients": GradientTilePush,
    "clip_partial_gradients": ClipPartialsPush,
    "gather_and_permute_grad_hidden_activations": GatherPermutePush,
    "aggregate_partials": AggregatePush,
    "aggregate_partials_from_compute": AggregatePush,  # ADR-026: shares push layout
    "clip_intermediate_grad": ClipIntermediatePush,
    "reduce_k_fan_in_and_clip": ReduceKFanInPush,  # ADR-019: K-fan-in with per-node clip
    "reduce_k_fan_in_and_clip_from_compute": ReduceKFanInPush,  # ADR-019/ADR-026: compute-entry
    "stabilize_and_reduce_grad_hidden_activations": StabilizeReducePush,
    "backprop_shared_weights_chunk": SharedBackpropWeightsPush,
    "backprop_shared_biases_chunk": SharedBackpropBiasesPush,
    "clip_shared_gradients_chunk": ClipSharedPush,
    "normalize_gradients": NormalizePush,
    "adam_update": AdamUpdatePush,
    "clamp_temperatures": ClampTempsPush,
}

# ── FP64 variant overrides (ADR-024) ──
# Kernels with COMPUTE_TYPE scalar parameters require different struct layouts
# when COMPUTE_TYPE = double due to FP64 alignment requirements.
PUSH_CONSTANT_STRUCTS_FP64: dict[str, type[ctypes.Structure]] = {
    "clip_partial_gradients": ClipPartialsPushFP64,
    "clip_shared_gradients_chunk": ClipSharedPushFP64,
    "clip_intermediate_grad": ClipIntermediatePushFP64,
    "reduce_k_fan_in_and_clip": ReduceKFanInPushFP64,  # ADR-019
    "reduce_k_fan_in_and_clip_from_compute": ReduceKFanInPushFP64,  # ADR-019/ADR-026
    "normalize_gradients": NormalizePushFP64,
}

# ── FP16 variant overrides (ADR-024 compliance) ──
# Kernels with COMPUTE_TYPE scalar parameters require different struct layouts
# when COMPUTE_TYPE = float16_t due to 2-byte field sizes.
PUSH_CONSTANT_STRUCTS_FP16: dict[str, type[ctypes.Structure]] = {
    "clip_partial_gradients": ClipPartialsPushFP16,
    "clip_shared_gradients_chunk": ClipSharedPushFP16,
    "clip_intermediate_grad": ClipIntermediatePushFP16,
    "reduce_k_fan_in_and_clip": ReduceKFanInPushFP16,  # ADR-019
    "reduce_k_fan_in_and_clip_from_compute": ReduceKFanInPushFP16,  # ADR-019/ADR-026
    "normalize_gradients": NormalizePushFP16,
}


# ── Descriptor buffer binding order ──
#
# Maps kernel_name → ordered tuple of plan buffer-key names.
# The i-th entry corresponds to ``layout(set=0, binding=i)`` in the shader.
# ``None`` marks a binding slot that has no plan-level key (e.g. the
# alternate-format targets buffer in Strategy-A shaders); the renderer
# leaves such slots uninitialised because dead-code elimination in the
# specialised pipeline guarantees they are never accessed.

BUFFER_BINDING_ORDER: dict[str, tuple[str | None, ...]] = {
    # Node 4
    "forward_pass": (
        "input", "sample_mask", "weights_shared_simd_major",
        "biases_shared", "hidden_activations", "hidden_mask",
    ),
    # Node 5
    "render_logits_chunk": (
        "hidden_activations", "hidden_mask", "sample_mask",
        "weights_module", "biases_module", "logits",
    ),
    # Node 6 (CCE)
    "compute_probs_loss_cce_chunk": (
        "logits", "temps", "targets", "sample_mask",
        "partial_probs", "final_loss",
    ),
    # Node 7 (BCE)
    "compute_probs_loss_bce_chunk": (
        "logits", "temps", "targets", "sample_mask",
        "partial_probs", "partial_loss",
    ),
    # Node 8 — Strategy A (dual targets: binding 3 unused for CCE)
    "calculate_module_param_grads_chunk": (
        "hidden_activations", "partial_probs", "targets", None,
        "sample_mask", "partial_grad_weights_module",
        "partial_grad_biases_module", "temps",
    ),
    # Node 9 — Strategy A (dual targets: binding 2 unused for CCE)
    "backprop_error_to_hidden_chunk": (
        "partial_probs", "targets", None, "sample_mask",
        "weights_module", "partial_grad_hidden_activations_aos", "temps",
    ),
    # Node 10 — Strategy A (dual targets: binding 3 unused for CCE)
    "calculate_chunk_temp_gradients": (
        "logits", "partial_probs", "targets", None,
        "sample_mask", "temps", "partial_grad_temps",
    ),
    # Node 11 (binding 4 = per-item threshold, unused when use_per_item_norm=0)
    "clip_partial_gradients": (
        "partial_grad_weights_module", "partial_grad_biases_module",
        "partial_grad_temps", "partial_grad_hidden_activations_aos", None,
        "clipped_partial_grad_weights_module", "clipped_partial_grad_biases_module",
        "clipped_partial_grad_temps", "clipped_partial_grad_hidden_activations_aos",
    ),
    # Node 13
    "gather_and_permute_grad_hidden_activations": (
        "clipped_partial_grad_hidden_activations_aos",
        "clipped_grad_hidden_activations_permuted_soa",
    ),
    # Node 16
    "stabilize_and_reduce_grad_hidden_activations": (
        "clipped_grad_hidden_activations_permuted_soa",
        "summed_grad_hidden_activations",
        "clipping_threshold_per_stage",
    ),
    # Node 17
    "backprop_shared_weights_chunk": (
        "input", "hidden_activations", "summed_grad_hidden_activations",
        "sample_mask", "hidden_mask", "partial_grad_weights_shared",
    ),
    # Node 18
    "backprop_shared_biases_chunk": (
        "hidden_activations", "summed_grad_hidden_activations",
        "sample_mask", "hidden_mask", "partial_grad_biases_shared",
    ),
    # Node 19
    "clip_shared_gradients_chunk": (
        "partial_grad_weights_shared", "partial_grad_biases_shared",
        "clipped_partial_grad_weights_shared", "clipped_partial_grad_biases_shared",
    ),
    # Node 21 (all instances share this order)
    "normalize_gradients": (
        "summed_grad", "final_grad",
    ),
    # Node 24 (all instances share this order)
    "adam_update": (
        "final_grad", "parameters", "m1", "m2",
    ),
    # Node 25
    "clamp_temperatures": (
        "temperatures",
    ),
}


# ── Descriptor binding counts per shader (Phase 5A §7) ──

DESCRIPTOR_BINDING_COUNTS: dict[str, int] = {
    "forward_pass": 6,
    "render_logits_chunk": 6,
    "compute_probs_loss_cce_chunk": 6,
    "compute_probs_loss_bce_chunk": 6,
    "calculate_module_param_grads_chunk": 8,
    "backprop_error_to_hidden_chunk": 7,
    "calculate_chunk_temp_gradients": 7,
    "clip_partial_gradients": 9,
    "gather_and_permute_grad_hidden_activations": 2,
    "aggregate_partials": 3,
    "aggregate_partials_from_compute": 3,  # ADR-026: same layout as storage-entry
    "clip_intermediate_grad": 1,
    "reduce_k_fan_in_and_clip": 3,  # ADR-019: [src_collection, offset_list, dest]
    "reduce_k_fan_in_and_clip_from_compute": 3,  # ADR-019/ADR-026: same layout
    "stabilize_and_reduce_grad_hidden_activations": 3,
    "backprop_shared_weights_chunk": 6,
    "backprop_shared_biases_chunk": 5,
    "clip_shared_gradients_chunk": 4,
    "normalize_gradients": 2,
    "adam_update": 4,
    "clamp_temperatures": 1,
}

# ── Scalar param name → struct field name mapping ──
# CONTRACT.md naming: src_scalar_NATURAL_<name>, src_scalar_REAL_<name>,
# src_scalar_FLAG_<name>, dest_scalar_NATURAL_<name> → strip prefix.

_PARAM_PREFIXES = (
    "src_scalar_NATURAL_",
    "src_scalar_REAL_",
    "src_scalar_FLAG_",
    "dest_scalar_NATURAL_",
)


def _strip_param_prefix(param_name: str) -> str:
    """Strip CONTRACT.md parameter naming prefix."""
    for prefix in _PARAM_PREFIXES:
        if param_name.startswith(prefix):
            return param_name[len(prefix):]
    return param_name


def marshal_push_constants(
    kernel_name: str,
    scalar_params: Mapping[str, int | float],
    compute_dtype: np.dtype | None = None,
) -> bytes:
    """Convert plan scalar_params into packed push constant bytes.

    Maps abstract parameter names (CONTRACT.md naming convention)
    to the corresponding ctypes push constant struct fields.

    Args:
        kernel_name: Name of the kernel.
        scalar_params: Scalar parameters from the plan node.
        compute_dtype: Optional compute dtype. When np.float64, selects FP64
            struct variants; when np.float16, selects FP16 struct variants
            for kernels with COMPUTE_TYPE scalar parameters.
            Defaults to None (FP32 structs).
    """
    # Select precision-specific struct variant if available
    if compute_dtype is not None and compute_dtype == np.float64:
        struct_cls = PUSH_CONSTANT_STRUCTS_FP64.get(
            kernel_name, PUSH_CONSTANT_STRUCTS[kernel_name]
        )
    elif compute_dtype is not None and compute_dtype == np.float16:
        struct_cls = PUSH_CONSTANT_STRUCTS_FP16.get(
            kernel_name, PUSH_CONSTANT_STRUCTS[kernel_name]
        )
    else:
        struct_cls = PUSH_CONSTANT_STRUCTS[kernel_name]

    struct = struct_cls()

    # Build field type lookup (skip padding fields)
    field_types: dict[str, type] = {}
    for field_info in struct_cls._fields_:
        fname, ftype = field_info[0], field_info[1]
        if not fname.startswith("_pad"):  # Skip padding fields
            field_types[fname] = ftype

    for param_name, value in scalar_params.items():
        field_name = _strip_param_prefix(param_name)
        if field_name in field_types:
            ftype = field_types[field_name]
            # Handle FP16 fields stored as uint16 bit patterns
            if ftype == ctypes.c_uint16 and isinstance(value, float):
                # Convert float to float16 IEEE 754 bits
                bits = np.float16(value).view(np.uint16).item()
                setattr(struct, field_name, bits)
            else:
                setattr(struct, field_name, ftype(value))

    return bytes(struct)
