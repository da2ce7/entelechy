"""ctypes Structure definitions mirroring cpu_kernels.h argument structs (ADR-015).

Field order MUST exactly match the C struct field order in cpu_kernels.h.
Layout verification via _verify_layouts() catches size mismatches at load time.

Multi-precision support (ADR-008): the CPU library exports fp16, fp32, and
fp64 variants of every kernel and struct.  ``make_precision_types()`` creates
the ctypes Structure classes for a given precision suffix.  Module-level
names (``ForwardPassArgs``, ``c_real_p``, etc.) alias the fp32 variants for
backward compatibility.
"""
from __future__ import annotations

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_double,
    c_float,
    c_int32,
    c_uint16,
    c_uint32,
    c_void_p,
)

c_float_p = POINTER(c_float)
c_uint_p = POINTER(c_uint32)
c_int_p = POINTER(c_int32)

PRECISION_SUFFIXES = ("fp16", "fp32", "fp64")

# (c_real, c_real_p, size_bytes) per precision.  FP16 uses c_uint16 because
# ctypes has no _Float16; both are 2-byte types with identical alignment.
PRECISION_C_TYPES: dict[str, tuple[type, type, int]] = {
    "fp16": (c_uint16, POINTER(c_uint16), 2),
    "fp32": (c_float, POINTER(c_float), 4),
    "fp64": (c_double, POINTER(c_double), 8),
}

# Base names for struct-size getter exports (suffix appended per precision).
_LAYOUT_CHECK_BASE_NAMES: list[tuple[str, str]] = [
    ("get_struct_size_forward_pass_args", "ForwardPassArgs"),
    ("get_struct_size_render_logits_args", "RenderLogitsArgs"),
    ("get_struct_size_cce_chunk_args", "CceChunkArgs"),
    ("get_struct_size_bce_chunk_args", "BceChunkArgs"),
    ("get_struct_size_module_param_grads_args", "ModuleParamGradsArgs"),
    ("get_struct_size_backprop_to_hidden_args", "BackpropToHiddenArgs"),
    ("get_struct_size_temp_gradients_args", "TempGradientsArgs"),
    ("get_struct_size_clip_partials_args", "ClipPartialsArgs"),
    ("get_struct_size_gather_permute_args", "GatherPermuteArgs"),
    ("get_struct_size_reduction_tree_plan", "ReductionTreePlanFFI"),
    ("get_struct_size_stabilize_reduce_args", "StabilizeReduceArgs"),
    ("get_struct_size_clip_intermediate_args", "ClipIntermediateArgs"),
    ("get_struct_size_backprop_shared_weights_args", "BackpropSharedWeightsArgs"),
    ("get_struct_size_backprop_shared_biases_args", "BackpropSharedBiasesArgs"),
    ("get_struct_size_clip_shared_grads_args", "ClipSharedGradsArgs"),
    ("get_struct_size_normalize_gradients_args", "NormalizeGradientsArgs"),
    ("get_struct_size_adam_update_args", "AdamUpdateArgs"),
    ("get_struct_size_clamp_temperatures_args", "ClampTemperaturesArgs"),
]


def make_precision_types(
    suffix: str,
) -> tuple[dict[str, type], list[tuple[str, type]]]:
    """Create all ctypes Structure classes for a single precision.

    Returns ``(structs_dict, layout_checks)`` where:
      - *structs_dict*: ``{"ForwardPassArgs": <class>, ...}``
      - *layout_checks*: ``[("get_struct_size_forward_pass_args_fp32", <class>), ...]``
    """
    c_real, c_real_p, _ = PRECISION_C_TYPES[suffix]

    def _s(name: str, fields: list) -> type:
        """Create a named Structure subclass."""
        return type(f"{name}_{suffix}", (Structure,), {"_fields_": fields})

    structs: dict[str, type] = {}

    # --- Act Phase ---

    structs["ForwardPassArgs"] = _s("ForwardPassArgs", [
        ("input", c_real_p),
        ("sample_mask", c_real_p),
        ("weights_shared_simd_major", c_real_p),
        ("biases_shared", c_real_p),
        ("hidden_activations", c_real_p),
        ("hidden_mask", c_real_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("total_batch_count", c_uint32),
        ("padded_input_count", c_uint32),
        ("padded_hidden_count", c_uint32),
    ])

    structs["RenderLogitsArgs"] = _s("RenderLogitsArgs", [
        ("hidden_activations", c_real_p),
        ("hidden_mask", c_real_p),
        ("weights_module", c_real_p),
        ("biases_module", c_real_p),
        ("logits", c_real_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("module_chunk_offset", c_uint32),
        ("module_chunk_count", c_uint32),
        ("class_chunk_offset", c_uint32),
        ("class_chunk_count", c_uint32),
        ("total_batch_count", c_uint32),
        ("hidden_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
    ])

    # --- Learn Phase A: Loss & Gradient Production ---

    structs["CceChunkArgs"] = _s("CceChunkArgs", [
        ("logits", c_real_p),
        ("temps", c_real_p),
        ("targets", c_int_p),
        ("sample_mask", c_real_p),
        ("partial_probs", c_real_p),
        ("final_loss", c_real_p),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    structs["BceChunkArgs"] = _s("BceChunkArgs", [
        ("logits", c_real_p),
        ("temps", c_real_p),
        ("targets", c_real_p),
        ("sample_mask", c_real_p),
        ("partial_probs", c_real_p),
        ("partial_loss", c_real_p),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    structs["ModuleParamGradsArgs"] = _s("ModuleParamGradsArgs", [
        ("hidden_activations", c_real_p),
        ("partial_probs", c_real_p),
        ("targets", c_void_p),
        ("sample_mask", c_real_p),
        ("partial_grad_weights_module", c_real_p),
        ("partial_grad_biases_module", c_real_p),
        ("problem_type", c_uint32),
        ("flat_tile_index", c_uint32),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("hidden_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    # --- Learn Phase B: Processing ---

    structs["BackpropToHiddenArgs"] = _s("BackpropToHiddenArgs", [
        ("partial_probs", c_real_p),
        ("targets", c_void_p),
        ("sample_mask", c_real_p),
        ("weights_module", c_real_p),
        ("partial_grad_hidden_activations_aos", c_real_p),
        ("problem_type", c_uint32),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("hidden_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    structs["TempGradientsArgs"] = _s("TempGradientsArgs", [
        ("logits", c_real_p),
        ("partial_probs", c_real_p),
        ("targets", c_void_p),
        ("sample_mask", c_real_p),
        ("temps", c_real_p),
        ("partial_grad_temps", c_real_p),
        ("problem_type", c_uint32),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    structs["ClipPartialsArgs"] = _s("ClipPartialsArgs", [
        ("partial_grad_weights_module", c_real_p),
        ("partial_grad_biases_module", c_real_p),
        ("partial_grad_temps", c_real_p),
        ("partial_grad_hidden_activations_aos", c_real_p),
        ("clipping_threshold_per_item", c_real_p),
        ("clipped_partial_grad_weights_module", c_real_p),
        ("clipped_partial_grad_biases_module", c_real_p),
        ("clipped_partial_grad_temps", c_real_p),
        ("clipped_partial_grad_hidden_activations_aos", c_real_p),
        ("use_per_item_norm", c_uint32),
        ("clipping_threshold_t_pre", c_real),
        ("epsilon", c_real),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    # --- Learn Phase C: Reduction ---

    structs["GatherPermuteArgs"] = _s("GatherPermuteArgs", [
        ("clipped_partial_grad_hidden_activations_aos", c_real_p),
        ("clipped_grad_hidden_activations_permuted_soa", c_real_p),
        ("total_batch_count", c_uint32),
        ("hidden_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("padded_total_modules_count", c_uint32),
        ("num_module_chunks", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("num_class_chunks", c_uint32),
        ("total_tile_count", c_uint32),
    ])

    structs["ReductionTreePlanFFI"] = _s("ReductionTreePlanFFI", [
        ("partial_collection", c_real_p),
        ("offset_lists_flat", c_uint_p),
        ("stage_offsets_into_list", c_uint_p),
        ("stage_fan_in", c_uint_p),
        ("stage_node_counts", c_uint_p),
        ("staging_buffer_0", c_real_p),
        ("staging_buffer_1", c_real_p),
        ("output", c_real_p),
        ("partial_width", c_uint32),
        ("num_stages", c_uint32),
        ("t_algorithmic", c_real),
        ("lambda_", c_real),
        ("fp_max", c_real),
        ("epsilon", c_real),
    ])

    structs["StabilizeReduceArgs"] = _s("StabilizeReduceArgs", [
        ("grad_hidden_activations_permuted_soa", c_real_p),
        ("summed_grad_hidden_activations", c_real_p),
        ("fp_max", c_real),
        ("policy_t_algorithmic", c_real),
        ("policy_lambda", c_real),
        ("policy_max_k", c_uint32),
        ("epsilon", c_real),
        ("total_batch_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("padded_total_modules_count", c_uint32),
    ])

    structs["ClipIntermediateArgs"] = _s("ClipIntermediateArgs", [
        ("intermediate_grad", c_real_p),
        ("clipping_threshold_t_j", c_real),
        ("epsilon", c_real),
        ("parameter_count", c_uint32),
    ])

    # --- Learn Phase D: Streaming Backprop ---

    structs["BackpropSharedWeightsArgs"] = _s("BackpropSharedWeightsArgs", [
        ("input", c_real_p),
        ("hidden_activations", c_real_p),
        ("summed_grad_hidden_activations", c_real_p),
        ("sample_mask", c_real_p),
        ("partial_grad_weights_shared", c_real_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("batch_chunk_index", c_uint32),
        ("total_batch_count", c_uint32),
        ("num_batch_chunks_count", c_uint32),
        ("padded_input_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("final_grad_hidden_total_element_count", c_uint32),
    ])

    structs["BackpropSharedBiasesArgs"] = _s("BackpropSharedBiasesArgs", [
        ("hidden_activations", c_real_p),
        ("summed_grad_hidden_activations", c_real_p),
        ("sample_mask", c_real_p),
        ("partial_grad_biases_shared", c_real_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("batch_chunk_index", c_uint32),
        ("total_batch_count", c_uint32),
        ("num_batch_chunks_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("final_grad_hidden_total_element_count", c_uint32),
    ])

    structs["ClipSharedGradsArgs"] = _s("ClipSharedGradsArgs", [
        ("partial_grad_weights_shared", c_real_p),
        ("partial_grad_biases_shared", c_real_p),
        ("clipped_partial_grad_weights_shared", c_real_p),
        ("clipped_partial_grad_biases_shared", c_real_p),
        ("clipping_threshold_t_pre", c_real),
        ("epsilon", c_real),
        ("weights_parameter_count", c_uint32),
        ("biases_parameter_count", c_uint32),
        ("weights_write_offset_elements", c_uint32),
        ("biases_write_offset_elements", c_uint32),
        ("num_batch_chunks", c_uint32),
    ])

    # --- Update Phase ---

    structs["NormalizeGradientsArgs"] = _s("NormalizeGradientsArgs", [
        ("summed_grad", c_real_p),
        ("final_grad", c_real_p),
        ("effective_batch_size", c_real),
        ("epsilon", c_real),
        ("parameter_count", c_uint32),
    ])

    structs["AdamUpdateArgs"] = _s("AdamUpdateArgs", [
        ("final_grad", c_real_p),
        ("parameters", c_real_p),
        ("m1", c_real_p),
        ("m2", c_real_p),
        ("learning_rate", c_real),
        ("beta1_pow_t", c_real),
        ("beta2_pow_t", c_real),
        ("beta1", c_real),
        ("beta2", c_real),
        ("epsilon", c_real),
        ("parameter_count", c_uint32),
    ])

    structs["ClampTemperaturesArgs"] = _s("ClampTemperaturesArgs", [
        ("temperatures", c_real_p),
        ("min_value", c_real),
        ("max_value", c_real),
        ("total_modules_count", c_uint32),
    ])

    # Build layout checks with suffixed getter names
    layout_checks = [
        (f"{base_name}_{suffix}", structs[struct_key])
        for base_name, struct_key in _LAYOUT_CHECK_BASE_NAMES
    ]

    return structs, layout_checks


# --- Pre-built precision maps ---

PRECISION_STRUCTS: dict[str, dict[str, type]] = {}
PRECISION_LAYOUT_CHECKS: dict[str, list[tuple[str, type]]] = {}

_sfx, _structs, _checks = "", PRECISION_STRUCTS, PRECISION_LAYOUT_CHECKS
for _sfx in PRECISION_SUFFIXES:
    _structs, _checks = make_precision_types(_sfx)
    PRECISION_STRUCTS[_sfx] = _structs
    PRECISION_LAYOUT_CHECKS[_sfx] = _checks

del _sfx, _structs, _checks

# --- Backward-compatible module-level aliases (fp32) ---

c_real: type = c_float
c_real_p: type = POINTER(c_float)
REAL_SIZE_BYTES: int = ctypes.sizeof(c_float)

ForwardPassArgs = PRECISION_STRUCTS["fp32"]["ForwardPassArgs"]
RenderLogitsArgs = PRECISION_STRUCTS["fp32"]["RenderLogitsArgs"]
CceChunkArgs = PRECISION_STRUCTS["fp32"]["CceChunkArgs"]
BceChunkArgs = PRECISION_STRUCTS["fp32"]["BceChunkArgs"]
ModuleParamGradsArgs = PRECISION_STRUCTS["fp32"]["ModuleParamGradsArgs"]
BackpropToHiddenArgs = PRECISION_STRUCTS["fp32"]["BackpropToHiddenArgs"]
TempGradientsArgs = PRECISION_STRUCTS["fp32"]["TempGradientsArgs"]
ClipPartialsArgs = PRECISION_STRUCTS["fp32"]["ClipPartialsArgs"]
GatherPermuteArgs = PRECISION_STRUCTS["fp32"]["GatherPermuteArgs"]
ReductionTreePlanFFI = PRECISION_STRUCTS["fp32"]["ReductionTreePlanFFI"]
StabilizeReduceArgs = PRECISION_STRUCTS["fp32"]["StabilizeReduceArgs"]
ClipIntermediateArgs = PRECISION_STRUCTS["fp32"]["ClipIntermediateArgs"]
BackpropSharedWeightsArgs = PRECISION_STRUCTS["fp32"]["BackpropSharedWeightsArgs"]
BackpropSharedBiasesArgs = PRECISION_STRUCTS["fp32"]["BackpropSharedBiasesArgs"]
ClipSharedGradsArgs = PRECISION_STRUCTS["fp32"]["ClipSharedGradsArgs"]
NormalizeGradientsArgs = PRECISION_STRUCTS["fp32"]["NormalizeGradientsArgs"]
AdamUpdateArgs = PRECISION_STRUCTS["fp32"]["AdamUpdateArgs"]
ClampTemperaturesArgs = PRECISION_STRUCTS["fp32"]["ClampTemperaturesArgs"]

# Backward-compatible fp32 layout checks
LAYOUT_CHECKS: list[tuple[str, type]] = PRECISION_LAYOUT_CHECKS["fp32"]
