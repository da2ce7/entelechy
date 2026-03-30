"""Push constant struct definitions and marshalling (VULKAN_BACKEND.md §Push Constants).

Each ctypes.Structure mirrors the GLSL push_constant layout for a shader.
The marshal function converts plan node scalar_params into packed bytes.
"""
from __future__ import annotations

from collections.abc import Mapping

import ctypes
from typing import ClassVar


# ── Push Constant Structures ──
# Field order MUST match the GLSL layout(push_constant) declarations.

class ForwardPassPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("padded_input_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
    ]


class RenderLogitsPush(ctypes.Structure):
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
        ("module_offset", ctypes.c_uint32),
        ("class_offset", ctypes.c_uint32),
        ("total_tile_count", ctypes.c_uint32),
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
        ("padded_hidden_count", ctypes.c_uint32),
        ("total_modules_count", ctypes.c_uint32),
        ("padded_total_modules_count", ctypes.c_uint32),
        ("num_class_chunks", ctypes.c_uint32),
        ("classes_per_chunk", ctypes.c_uint32),
        ("modules_per_chunk", ctypes.c_uint32),
        ("partial_stride", ctypes.c_uint32),
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


class StabilizeReducePush(ctypes.Structure):
    _fields_ = [
        ("fp_max", ctypes.c_float),
        ("policy_t_algorithmic", ctypes.c_float),
        ("policy_lambda", ctypes.c_float),
        ("policy_max_k", ctypes.c_uint32),
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
        ("num_batch_chunks_count", ctypes.c_uint32),
        ("padded_input_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("final_grad_hidden_total_element_count", ctypes.c_uint32),
    ]


class SharedBackpropBiasesPush(ctypes.Structure):
    _fields_ = [
        ("batch_chunk_offset", ctypes.c_uint32),
        ("batch_chunk_count", ctypes.c_uint32),
        ("batch_chunk_index", ctypes.c_uint32),
        ("total_batch_count", ctypes.c_uint32),
        ("num_batch_chunks_count", ctypes.c_uint32),
        ("padded_hidden_count", ctypes.c_uint32),
        ("final_grad_hidden_total_element_count", ctypes.c_uint32),
    ]


class ClipSharedPush(ctypes.Structure):
    _fields_ = [
        ("clipping_threshold", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("weights_parameter_count", ctypes.c_uint32),
        ("biases_parameter_count", ctypes.c_uint32),
        ("weights_write_offset", ctypes.c_uint32),
        ("biases_write_offset", ctypes.c_uint32),
        ("num_batch_chunks", ctypes.c_uint32),
    ]


class NormalizePush(ctypes.Structure):
    _fields_ = [
        ("effective_batch_size", ctypes.c_float),
        ("epsilon", ctypes.c_float),
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
        ("parameter_count", ctypes.c_uint32),
    ]


class ClampTempsPush(ctypes.Structure):
    _fields_ = [
        ("min_value", ctypes.c_float),
        ("max_value", ctypes.c_float),
        ("total_modules_count", ctypes.c_uint32),
    ]


# ── Kernel name → struct class mapping ──

PUSH_CONSTANT_STRUCTS: dict[str, type[ctypes.Structure]] = {
    "forward_pass": ForwardPassPush,
    "render_logits": RenderLogitsPush,
    "compute_probs_loss_cce": ProbsLossPush,
    "compute_probs_loss_bce": ProbsLossPush,
    "calculate_module_param_grads": GradientTilePush,
    "backprop_error_to_hidden": GradientTilePush,
    "calculate_temp_gradients": GradientTilePush,
    "clip_partial_gradients": ClipPartialsPush,
    "gather_and_permute": GatherPermutePush,
    "aggregate_partials": AggregatePush,
    "clip_intermediate_grad": ClipIntermediatePush,
    "stabilize_reduce_grad_h": StabilizeReducePush,
    "backprop_shared_weights": SharedBackpropWeightsPush,
    "backprop_shared_biases": SharedBackpropBiasesPush,
    "clip_shared_gradients": ClipSharedPush,
    "normalize_gradients": NormalizePush,
    "adam_update": AdamUpdatePush,
    "clamp_temperatures": ClampTempsPush,
}


# ── Descriptor binding counts per shader (Phase 5A §7) ──

DESCRIPTOR_BINDING_COUNTS: dict[str, int] = {
    "forward_pass": 6,
    "render_logits": 5,
    "compute_probs_loss_cce": 6,
    "compute_probs_loss_bce": 6,
    "calculate_module_param_grads": 7,
    "backprop_error_to_hidden": 6,
    "calculate_temp_gradients": 7,
    "clip_partial_gradients": 9,
    "gather_and_permute": 2,
    "aggregate_partials": 3,
    "clip_intermediate_grad": 1,
    "stabilize_reduce_grad_h": 2,
    "backprop_shared_weights": 5,
    "backprop_shared_biases": 4,
    "clip_shared_gradients": 4,
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
) -> bytes:
    """Convert plan scalar_params into packed push constant bytes.

    Maps abstract parameter names (CONTRACT.md naming convention)
    to the corresponding ctypes push constant struct fields.
    """
    struct_cls = PUSH_CONSTANT_STRUCTS[kernel_name]
    struct = struct_cls()

    # Build field type lookup
    field_types: dict[str, type] = {}
    for fname, ftype in struct_cls._fields_:  # type: ignore[reportAssignmentType]
        field_types[fname] = ftype

    for param_name, value in scalar_params.items():
        field_name = _strip_param_prefix(param_name)
        if field_name in field_types:
            setattr(struct, field_name, field_types[field_name](value))

    return bytes(struct)
