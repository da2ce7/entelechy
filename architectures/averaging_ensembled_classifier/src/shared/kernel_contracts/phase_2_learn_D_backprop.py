# src/shared/kernel_contracts/phase_2_learn_D_backprop.py
"""Phase 2D (Shared Backprop) kernel contracts — populated from kernels.cl.h."""
from . import (
    BufferParamSpec, KernelContract, KernelContractBlock, LocalMemorySpec,
    PaddingContract, PlacementContract, ScalarParamSpec,
)

backprop_shared_weights_contract = KernelContract(
    kernel_name="backprop_shared_weights_chunk",
    contract_block=KernelContractBlock(
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Partial Renderer. True Streaming backpropagation model.",
        behavioral_invariants=None,
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_input", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count", "padded_input_count"),
            padding_contract=PaddingContract("CACHE", "Padded to alignment"),
            calculability_proof=("total_batch_count", "padded_input_count"),
            validation_preconditions=("batch slice within bounds",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_hidden_activations", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count", "padded_hidden_count"),
            padding_contract=PaddingContract("CACHE", "Padded to alignment"),
            calculability_proof=("total_batch_count", "padded_hidden_count"),
            validation_preconditions=("batch slice within bounds",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_hidden_mask", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count", "padded_hidden_count"),
            padding_contract=PaddingContract("CACHE", "Padded to alignment"),
            calculability_proof=("total_batch_count", "padded_hidden_count"),
            validation_preconditions=("batch slice within bounds",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_summed_grad_hidden_activations", flow="src", memory_scope="GLOBAL",
            tensor_shape=("final_grad_hidden_activations_total_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("final_grad_hidden_activations_total_count",),
            validation_preconditions=("logical shape matches physical size",),
            precision_role="compute",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_sample_mask", flow="src", memory_scope="GLOBAL",
            tensor_shape=("(total_batch_count + 31) / 32",),
            padding_contract=PaddingContract("NONE", "Bits beyond total_batch_count zero-padded by host"),
            calculability_proof=("total_batch_count",),
            validation_preconditions=("batch slice within bounds",),
            precision_role=None,  # integer-typed bitmask (ADR-031)
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("num_batch_chunks", "padded_input_count", "padded_hidden_count"),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("num_batch_chunks", "padded_input_count", "padded_hidden_count"),
            validation_preconditions=("chunk write index valid",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("FLAG_use_explicit_hidden_mask", "src", "FLAG"),
        ScalarParamSpec("batch_chunk_offset", "src", "NATURAL"),
        ScalarParamSpec("batch_chunk_count", "src", "NATURAL"),
        ScalarParamSpec("batch_chunk_index", "src", "NATURAL"),
        ScalarParamSpec("total_batch_count", "src", "NATURAL"),
        ScalarParamSpec("num_batch_chunks", "src", "NATURAL"),
        ScalarParamSpec("input_count", "src", "NATURAL"),
        ScalarParamSpec("padded_input_count", "src", "NATURAL"),
        ScalarParamSpec("hidden_count", "src", "NATURAL"),
        ScalarParamSpec("padded_hidden_count", "src", "NATURAL"),
        ScalarParamSpec("final_grad_hidden_activations_total_count", "src", "NATURAL"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=PlacementContract(strategy="linear_batch", key_domain=None, context_params={}),
)

backprop_shared_biases_contract = KernelContract(
    kernel_name="backprop_shared_biases_chunk",
    contract_block=KernelContractBlock(
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Partial Renderer. True Streaming backpropagation model.",
        behavioral_invariants=None,
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_hidden_activations", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count", "padded_hidden_count"),
            padding_contract=PaddingContract("CACHE", "Padded to alignment"),
            calculability_proof=("total_batch_count", "padded_hidden_count"),
            validation_preconditions=("batch slice within bounds",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_hidden_mask", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count", "padded_hidden_count"),
            padding_contract=PaddingContract("CACHE", "Padded to alignment"),
            calculability_proof=("total_batch_count", "padded_hidden_count"),
            validation_preconditions=("batch slice within bounds",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_summed_grad_hidden_activations", flow="src", memory_scope="GLOBAL",
            tensor_shape=("final_grad_hidden_activations_total_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("final_grad_hidden_activations_total_count",),
            validation_preconditions=("logical shape matches physical size",),
            precision_role="compute",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_sample_mask", flow="src", memory_scope="GLOBAL",
            tensor_shape=("(total_batch_count + 31) / 32",),
            padding_contract=PaddingContract("NONE", "Bits beyond total_batch_count zero-padded by host"),
            calculability_proof=("total_batch_count",),
            validation_preconditions=("batch slice within bounds",),
            precision_role=None,  # integer-typed bitmask (ADR-031)
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_partial_grad_biases_shared", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("num_batch_chunks", "padded_hidden_count"),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("num_batch_chunks", "padded_hidden_count"),
            validation_preconditions=("chunk write index valid",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("FLAG_use_explicit_hidden_mask", "src", "FLAG"),
        ScalarParamSpec("batch_chunk_offset", "src", "NATURAL"),
        ScalarParamSpec("batch_chunk_count", "src", "NATURAL"),
        ScalarParamSpec("batch_chunk_index", "src", "NATURAL"),
        ScalarParamSpec("total_batch_count", "src", "NATURAL"),
        ScalarParamSpec("num_batch_chunks", "src", "NATURAL"),
        ScalarParamSpec("hidden_count", "src", "NATURAL"),
        ScalarParamSpec("padded_hidden_count", "src", "NATURAL"),
        ScalarParamSpec("final_grad_hidden_activations_total_count", "src", "NATURAL"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=PlacementContract(strategy="linear_batch", key_domain=None, context_params={}),
)

clip_shared_gradients_contract = KernelContract(
    kernel_name="clip_shared_gradients_chunk",
    contract_block=KernelContractBlock(
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        idempotency="Strictly Idempotent",
        synchronization_model="Streamable Utility / Stability Primitive.",
        behavioral_invariants=(
            "Two-pass algorithm: Norm calculation followed by conditional scaling.",
            "L2 norm computed over concatenated weight and bias gradients.",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_partial_grad_weights_shared_simd_major", flow="src", memory_scope="GLOBAL",
            tensor_shape=("weights_parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("weights_parameter_count",),
            validation_preconditions=("contiguous chunk region",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_partial_grad_biases_shared", flow="src", memory_scope="GLOBAL",
            tensor_shape=("biases_parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("biases_parameter_count",),
            validation_preconditions=("contiguous chunk region",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("num_batch_chunks", "weights_parameter_count"),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("num_batch_chunks", "weights_parameter_count"),
            validation_preconditions=("write offset within bounds",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_clipped_partial_grad_biases_shared", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("num_batch_chunks", "biases_parameter_count"),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("num_batch_chunks", "biases_parameter_count"),
            validation_preconditions=("write offset within bounds",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("clipping_threshold_t_pre", "src", "REAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
        ScalarParamSpec("weights_parameter_count", "src", "NATURAL"),
        ScalarParamSpec("biases_parameter_count", "src", "NATURAL"),
        ScalarParamSpec("weights_write_offset", "dest", "NATURAL"),
        ScalarParamSpec("biases_write_offset", "dest", "NATURAL"),
        ScalarParamSpec("num_batch_chunks", "src", "NATURAL"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=PlacementContract(strategy="linear_batch", key_domain=None, context_params={}),
)
