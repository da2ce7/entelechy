# src/shared/kernel_contracts/phase_2_learn_C_reduction.py
"""Phase 2C (Reduction) kernel contracts — populated from kernels.cl.h."""
from . import (
    BufferParamSpec, KernelContract, KernelContractBlock, LocalMemorySpec,
    PaddingContract, ScalarParamSpec,
)

gather_and_permute_grad_h_contract = KernelContract(
    kernel_name="gather_and_permute_grad_hidden_activations",
    contract_block=KernelContractBlock(
        holistic_constraints="Solves the Transpose Illusion by gathering scattered partial results into a dense, reduction-ready SoA layout.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Global Barrier.",
        behavioral_invariants=("Gather performs implicit reduction (summation) over the class_chunk dimension.",),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_tile_count", "modules_per_chunk", "total_batch_count", "padded_hidden_count"),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("total_tile_count", "modules_per_chunk", "total_batch_count", "padded_hidden_count"),
            validation_preconditions=("fully populated by Node 11",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count * padded_hidden_count", "padded_total_modules_count"),
            padding_contract=PaddingContract("CACHE", "Trailing dimension padded to padded_total_modules_count for alignment."),
            calculability_proof=("total_batch_count", "padded_hidden_count", "padded_total_modules_count"),
            validation_preconditions=("exact allocation size",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("total_batch_count", "src", "NATURAL"),
        ScalarParamSpec("hidden_count", "src", "NATURAL"),
        ScalarParamSpec("padded_hidden_count", "src", "NATURAL"),
        ScalarParamSpec("total_modules_count", "src", "NATURAL"),
        ScalarParamSpec("padded_total_modules_count", "src", "NATURAL"),
        ScalarParamSpec("num_module_chunks", "src", "NATURAL"),
        ScalarParamSpec("modules_per_chunk", "src", "NATURAL"),
        ScalarParamSpec("num_class_chunks", "src", "NATURAL"),
        ScalarParamSpec("total_tile_count", "src", "NATURAL"),
    ),
    local_memory=(),
    placement=None,
)

aggregate_register_reduce_contract = KernelContract(
    kernel_name="aggregate_register_reduce",
    contract_block=KernelContractBlock(
        holistic_constraints="Operates on scattered (non-contiguous) input partials via an offset list. When used as a stage in a multi-stage reduction tree, operation_type MUST be AGG_MODE_SUM; AGG_MODE_AVERAGE is valid only as a single-stage terminal reduction, as partial-count division at interior stages produces silently incorrect results.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Reduction Engine Stage",
        behavioral_invariants=("Reduction policy (SUM/AVERAGE) controlled by operation_type flag.",),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_partial_collection", flow="src", memory_scope="GLOBAL",
            tensor_shape=("undefined",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=(),
            validation_preconditions=("valid buffer encompassing all offset references",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_CONST_partial_offset_list", flow="src", memory_scope="GLOBAL_CONST",
            tensor_shape=("partial_offset_list_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("partial_offset_list_count",),
            validation_preconditions=("exact offset count",),
            precision_role=None,
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_partial", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("partial_width",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("partial_width",),
            validation_preconditions=("exact allocation size",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("partial_offset_list_count", "src", "NATURAL"),
        ScalarParamSpec("partial_width", "src", "NATURAL"),
        ScalarParamSpec("operation_type", "src", "FLAG"),
    ),
    local_memory=(),
    placement=None,
)

aggregate_local_reduce_contract = KernelContract(
    kernel_name="aggregate_local_reduce",
    contract_block=KernelContractBlock(
        holistic_constraints="Operates on scattered (non-contiguous) input partials via an offset list. When used as a stage in a multi-stage reduction tree, operation_type MUST be AGG_MODE_SUM; AGG_MODE_AVERAGE is valid only as a single-stage terminal reduction, as partial-count division at interior stages produces silently incorrect results.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Reduction Engine Stage / Work-group Parallel",
        behavioral_invariants=("Reduction policy (SUM/AVERAGE) controlled by operation_type flag.",),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_partial_collection", flow="src", memory_scope="GLOBAL",
            tensor_shape=("undefined",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=(),
            validation_preconditions=("valid buffer encompassing all offset references",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_CONST_partial_offset_list", flow="src", memory_scope="GLOBAL_CONST",
            tensor_shape=("partial_offset_list_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("partial_offset_list_count",),
            validation_preconditions=("exact offset count",),
            precision_role=None,
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_partial", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("partial_width",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("partial_width",),
            validation_preconditions=("exact allocation size",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("partial_offset_list_count", "src", "NATURAL"),
        ScalarParamSpec("partial_width", "src", "NATURAL"),
        ScalarParamSpec("operation_type", "src", "FLAG"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=None,
)

clip_intermediate_grad_contract = KernelContract(
    kernel_name="clip_intermediate_grad",
    contract_block=KernelContractBlock(
        holistic_constraints="Core component of the host-driven, recursive clip-aggregation engine.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Reduction Engine Stage Clip Primitive",
        behavioral_invariants=(
            "Epsilon prevents division by zero.",
            "Uses local memory for scalable norm reduction.",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="update_buffer_GLOBAL_intermediate_grad", flow="update", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("exact element count",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("clipping_threshold_t_j", "src", "REAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
        ScalarParamSpec("parameter_count", "src", "NATURAL"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=None,
)

stabilize_reduce_grad_h_contract = KernelContract(
    kernel_name="stabilize_and_reduce_grad_hidden_activations",
    contract_block=KernelContractBlock(
        holistic_constraints="Complete row-wise reduction on the monolithic, contiguous SoA buffer from Node 13.",
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Specialized Reduction Kernel / Global Barrier",
        behavioral_invariants=(
            "Pre-computation: K_plan = min(policy_max_k, get_local_size(0)); num_stages = ceil(log(total_modules)/log(K_plan))",
            "Per-stage: j = num_stages-1-s; T_policy = t_algorithmic + lambda*j*j; T_safety = fp_max/K_actual; final = min(T_policy, T_safety)",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_grad_hidden_activations_permuted_soa", flow="src", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count * padded_hidden_count", "padded_total_modules_count"),
            padding_contract=PaddingContract("CACHE", "Trailing dimension padded for alignment."),
            calculability_proof=("total_batch_count", "padded_hidden_count", "padded_total_modules_count"),
            validation_preconditions=("fully populated by Node 13",),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_summed_grad_hidden_activations", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("total_batch_count * padded_hidden_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("total_batch_count", "padded_hidden_count"),
            validation_preconditions=("exact allocation size",),
            precision_role="compute",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("fp_max", "src", "REAL"),
        ScalarParamSpec("policy_t_algorithmic", "src", "REAL"),
        ScalarParamSpec("policy_lambda", "src", "REAL"),
        ScalarParamSpec("policy_max_k", "src", "NATURAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
        ScalarParamSpec("total_batch_count", "src", "NATURAL"),
        ScalarParamSpec("padded_hidden_count", "src", "NATURAL"),
        ScalarParamSpec("total_modules_count", "src", "NATURAL"),
        ScalarParamSpec("padded_total_modules_count", "src", "NATURAL"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=None,
)

# --- ADR-019: K-Fan-In Reduction Kernel Primitive ---

reduce_k_fan_in_and_clip_contract = KernelContract(
    kernel_name="reduce_k_fan_in_and_clip",
    contract_block=KernelContractBlock(
        holistic_constraints=(
            "Each work-group processes one reduction node. "
            "Reads K partials per node via flat offset list, sums them, "
            "optionally clips per-node, and writes one output vector."
        ),
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Reduction Engine Stage",
        behavioral_invariants=(
            "Per-node L2 clip when clipping_threshold >= 0: "
            "scale = threshold / (norm + epsilon).",
            "Clip bypassed when clipping_threshold < 0 (diagnostic mode).",
            "Zero threshold clips to zero norm (zeroes all gradients).",
            "Sentinel offset 0xFFFFFFFF skips absent partials in tail node.",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_partial_collection",
            flow="src", memory_scope="GLOBAL",
            tensor_shape=("undefined",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=(),
            validation_preconditions=(
                "valid buffer encompassing all offset references",
            ),
            precision_role="storage",
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_CONST_offset_list_flat",
            flow="src", memory_scope="GLOBAL_CONST",
            tensor_shape=("node_count * fan_in",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("node_count", "fan_in"),
            validation_preconditions=(
                "exactly node_count * fan_in uint entries",
            ),
            precision_role=None,
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_stage_partial",
            flow="dest", memory_scope="GLOBAL",
            tensor_shape=("node_count * partial_width",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("node_count", "partial_width"),
            validation_preconditions=("exact allocation size",),
            precision_role="storage",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("fan_in", "src", "NATURAL"),
        ScalarParamSpec("node_count", "src", "NATURAL"),
        ScalarParamSpec("partial_width", "src", "NATURAL"),
        ScalarParamSpec("clipping_threshold", "src", "REAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
    ),
    local_memory=(
        LocalMemorySpec("reduction_tile", "get_local_size(0) * sizeof(COMPUTE_TYPE)"),
    ),
    placement=None,
)
