"""ctypes Structure definitions mirroring cpu_kernels.h argument structs (ADR-015).

Field order MUST exactly match the C struct field order in cpu_kernels.h.
Layout verification via _verify_layouts() catches size mismatches at load time.
"""
from __future__ import annotations

from ctypes import POINTER, Structure, c_float, c_int32, c_uint32, c_void_p

c_float_p = POINTER(c_float)
c_uint_p = POINTER(c_uint32)
c_int_p = POINTER(c_int32)


# --- Act Phase ---

class ForwardPassArgs(Structure):
    _fields_ = [
        ("input", c_float_p),
        ("sample_mask", c_float_p),
        ("weights_shared_simd_major", c_float_p),
        ("biases_shared", c_float_p),
        ("hidden_activations", c_float_p),
        ("hidden_mask", c_float_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("total_batch_count", c_uint32),
        ("padded_input_count", c_uint32),
        ("padded_hidden_count", c_uint32),
    ]


class RenderLogitsArgs(Structure):
    _fields_ = [
        ("hidden_activations", c_float_p),
        ("hidden_mask", c_float_p),
        ("weights_module", c_float_p),
        ("biases_module", c_float_p),
        ("logits", c_float_p),
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
    ]


# --- Learn Phase A: Loss & Gradient Production ---

class CceChunkArgs(Structure):
    _fields_ = [
        ("logits", c_float_p),
        ("temps", c_float_p),
        ("targets", c_int_p),
        ("sample_mask", c_float_p),
        ("partial_probs", c_float_p),
        ("final_loss", c_float_p),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ]


class BceChunkArgs(Structure):
    _fields_ = [
        ("logits", c_float_p),
        ("temps", c_float_p),
        ("targets", c_float_p),
        ("sample_mask", c_float_p),
        ("partial_probs", c_float_p),
        ("partial_loss", c_float_p),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("total_output_class_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("total_tile_count", c_uint32),
    ]


class ModuleParamGradsArgs(Structure):
    _fields_ = [
        ("hidden_activations", c_float_p),
        ("partial_probs", c_float_p),
        ("targets", c_void_p),
        ("sample_mask", c_float_p),
        ("partial_grad_weights_module", c_float_p),
        ("partial_grad_biases_module", c_float_p),
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
    ]


# --- Learn Phase B: Processing ---

class BackpropToHiddenArgs(Structure):
    _fields_ = [
        ("partial_probs", c_float_p),
        ("targets", c_void_p),
        ("sample_mask", c_float_p),
        ("weights_module", c_float_p),
        ("partial_grad_hidden_activations_aos", c_float_p),
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
    ]


class TempGradientsArgs(Structure):
    _fields_ = [
        ("logits", c_float_p),
        ("partial_probs", c_float_p),
        ("targets", c_void_p),
        ("sample_mask", c_float_p),
        ("temps", c_float_p),
        ("partial_grad_temps", c_float_p),
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
    ]


class ClipPartialsArgs(Structure):
    _fields_ = [
        ("partial_grad_weights_module", c_float_p),
        ("partial_grad_biases_module", c_float_p),
        ("partial_grad_temps", c_float_p),
        ("partial_grad_hidden_activations_aos", c_float_p),
        ("clipping_threshold_per_item", c_float_p),
        ("clipped_partial_grad_weights_module", c_float_p),
        ("clipped_partial_grad_biases_module", c_float_p),
        ("clipped_partial_grad_temps", c_float_p),
        ("clipped_partial_grad_hidden_activations_aos", c_float_p),
        ("use_per_item_norm", c_uint32),
        ("clipping_threshold_t_pre", c_float),
        ("epsilon", c_float),
        ("flat_tile_index", c_uint32),
        ("num_class_chunks", c_uint32),
        ("classes_per_chunk", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("total_batch_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("padded_total_output_class_count", c_uint32),
        ("total_tile_count", c_uint32),
    ]


# --- Learn Phase C: Reduction ---

class GatherPermuteArgs(Structure):
    _fields_ = [
        ("clipped_partial_grad_hidden_activations_aos", c_float_p),
        ("clipped_grad_hidden_activations_permuted_soa", c_float_p),
        ("total_batch_count", c_uint32),
        ("hidden_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("padded_total_modules_count", c_uint32),
        ("num_module_chunks", c_uint32),
        ("modules_per_chunk", c_uint32),
        ("num_class_chunks", c_uint32),
        ("total_tile_count", c_uint32),
    ]


class ReductionTreePlanFFI(Structure):
    _fields_ = [
        ("partial_collection", c_float_p),
        ("offset_lists_flat", c_uint_p),
        ("stage_offsets_into_list", c_uint_p),
        ("stage_fan_in", c_uint_p),
        ("stage_node_counts", c_uint_p),
        ("staging_buffer_0", c_float_p),
        ("staging_buffer_1", c_float_p),
        ("output", c_float_p),
        ("partial_width", c_uint32),
        ("num_stages", c_uint32),
        ("t_algorithmic", c_float),
        ("lambda_", c_float),
        ("fp_max", c_float),
        ("epsilon", c_float),
    ]


class StabilizeReduceArgs(Structure):
    _fields_ = [
        ("grad_hidden_activations_permuted_soa", c_float_p),
        ("summed_grad_hidden_activations", c_float_p),
        ("fp_max", c_float),
        ("policy_t_algorithmic", c_float),
        ("policy_lambda", c_float),
        ("policy_max_k", c_uint32),
        ("epsilon", c_float),
        ("total_batch_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("total_modules_count", c_uint32),
        ("padded_total_modules_count", c_uint32),
    ]


class ClipIntermediateArgs(Structure):
    _fields_ = [
        ("intermediate_grad", c_float_p),
        ("clipping_threshold_t_j", c_float),
        ("epsilon", c_float),
        ("parameter_count", c_uint32),
    ]


# --- Learn Phase D: Streaming Backprop ---

class BackpropSharedWeightsArgs(Structure):
    _fields_ = [
        ("input", c_float_p),
        ("hidden_activations", c_float_p),
        ("summed_grad_hidden_activations", c_float_p),
        ("sample_mask", c_float_p),
        ("partial_grad_weights_shared", c_float_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("batch_chunk_index", c_uint32),
        ("total_batch_count", c_uint32),
        ("num_batch_chunks_count", c_uint32),
        ("padded_input_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("final_grad_hidden_total_element_count", c_uint32),
    ]


class BackpropSharedBiasesArgs(Structure):
    _fields_ = [
        ("hidden_activations", c_float_p),
        ("summed_grad_hidden_activations", c_float_p),
        ("sample_mask", c_float_p),
        ("partial_grad_biases_shared", c_float_p),
        ("batch_chunk_offset", c_uint32),
        ("batch_chunk_count", c_uint32),
        ("batch_chunk_index", c_uint32),
        ("total_batch_count", c_uint32),
        ("num_batch_chunks_count", c_uint32),
        ("padded_hidden_count", c_uint32),
        ("final_grad_hidden_total_element_count", c_uint32),
    ]


class ClipSharedGradsArgs(Structure):
    _fields_ = [
        ("partial_grad_weights_shared", c_float_p),
        ("partial_grad_biases_shared", c_float_p),
        ("clipped_partial_grad_weights_shared", c_float_p),
        ("clipped_partial_grad_biases_shared", c_float_p),
        ("clipping_threshold_t_pre", c_float),
        ("epsilon", c_float),
        ("weights_parameter_count", c_uint32),
        ("biases_parameter_count", c_uint32),
        ("weights_write_offset_elements", c_uint32),
        ("biases_write_offset_elements", c_uint32),
        ("num_batch_chunks", c_uint32),
    ]


# --- Update Phase ---

class NormalizeGradientsArgs(Structure):
    _fields_ = [
        ("summed_grad", c_float_p),
        ("final_grad", c_float_p),
        ("effective_batch_size", c_float),
        ("epsilon", c_float),
        ("parameter_count", c_uint32),
    ]


class AdamUpdateArgs(Structure):
    _fields_ = [
        ("final_grad", c_float_p),
        ("parameters", c_float_p),
        ("m1", c_float_p),
        ("m2", c_float_p),
        ("learning_rate", c_float),
        ("beta1_pow_t", c_float),
        ("beta2_pow_t", c_float),
        ("beta1", c_float),
        ("beta2", c_float),
        ("epsilon", c_float),
        ("parameter_count", c_uint32),
    ]


class ClampTemperaturesArgs(Structure):
    _fields_ = [
        ("temperatures", c_float_p),
        ("min_value", c_float),
        ("max_value", c_float),
        ("total_modules_count", c_uint32),
    ]


# Mapping for layout verification: (C getter name, Python struct class)
LAYOUT_CHECKS: list[tuple[str, type]] = [
    ("get_struct_size_forward_pass_args", ForwardPassArgs),
    ("get_struct_size_render_logits_args", RenderLogitsArgs),
    ("get_struct_size_cce_chunk_args", CceChunkArgs),
    ("get_struct_size_bce_chunk_args", BceChunkArgs),
    ("get_struct_size_module_param_grads_args", ModuleParamGradsArgs),
    ("get_struct_size_backprop_to_hidden_args", BackpropToHiddenArgs),
    ("get_struct_size_temp_gradients_args", TempGradientsArgs),
    ("get_struct_size_clip_partials_args", ClipPartialsArgs),
    ("get_struct_size_gather_permute_args", GatherPermuteArgs),
    ("get_struct_size_reduction_tree_plan", ReductionTreePlanFFI),
    ("get_struct_size_stabilize_reduce_args", StabilizeReduceArgs),
    ("get_struct_size_clip_intermediate_args", ClipIntermediateArgs),
    ("get_struct_size_backprop_shared_weights_args", BackpropSharedWeightsArgs),
    ("get_struct_size_backprop_shared_biases_args", BackpropSharedBiasesArgs),
    ("get_struct_size_clip_shared_grads_args", ClipSharedGradsArgs),
    ("get_struct_size_normalize_gradients_args", NormalizeGradientsArgs),
    ("get_struct_size_adam_update_args", AdamUpdateArgs),
    ("get_struct_size_clamp_temperatures_args", ClampTemperaturesArgs),
]
