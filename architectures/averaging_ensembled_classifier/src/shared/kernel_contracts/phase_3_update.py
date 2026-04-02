# src/shared/kernel_contracts/phase_3_update.py
"""Phase 3 (Update) kernel contracts — populated from kernels.cl.h."""
from . import (
    BufferParamSpec, KernelContract, KernelContractBlock, LocalMemorySpec,
    PaddingContract, ScalarParamSpec,
)

normalize_gradients_contract = KernelContract(
    kernel_name="normalize_gradients",
    contract_block=KernelContractBlock(
        holistic_constraints="Generic, element-wise scaling utility for any parameter group's summed gradient buffer.",
        idempotency="Strictly Idempotent",
        synchronization_model="Finalizer Utility / Batch-wide Normalizer.",
        behavioral_invariants=(
            "Element-wise: output[i] = input[i] / (effective_batch_size + epsilon).",
            "Epsilon prevents division by zero.",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_summed_grad", flow="src", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("complete aggregated gradients",),
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_final_grad", flow="dest", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("identical to src",),
        ),
    ),
    scalar_params=(
        ScalarParamSpec("effective_batch_size", "src", "REAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
        ScalarParamSpec("parameter_count", "src", "NATURAL"),
    ),
    local_memory=(),
    placement=None,
)

adam_update_contract = KernelContract(
    kernel_name="adam_update",
    contract_block=KernelContractBlock(
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        idempotency="Fundamentally Non-Idempotent (Stateful)",
        synchronization_model="Stateful Optimizer Update.",
        behavioral_invariants=(
            "Forbidden from using pown or equivalent.",
            "Host provides pre-computed bias correction terms.",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_final_grad", flow="src", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("exact allocation size",),
        ),
        BufferParamSpec(
            name="update_buffer_GLOBAL_parameters", flow="update", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("identical layout to final_grad, m1, m2",),
            precision_role="state",
        ),
        BufferParamSpec(
            name="update_buffer_GLOBAL_m1", flow="update", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("identical layout to other state buffers",),
            precision_role="state",
        ),
        BufferParamSpec(
            name="update_buffer_GLOBAL_m2", flow="update", memory_scope="GLOBAL",
            tensor_shape=("parameter_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("parameter_count",),
            validation_preconditions=("identical layout to other state buffers",),
            precision_role="state",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("learning_rate", "src", "REAL"),
        ScalarParamSpec("beta1_pow_t", "src", "REAL"),
        ScalarParamSpec("beta2_pow_t", "src", "REAL"),
        ScalarParamSpec("beta1", "src", "REAL"),
        ScalarParamSpec("beta2", "src", "REAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
        ScalarParamSpec("parameter_count", "src", "NATURAL"),
    ),
    local_memory=(),
    placement=None,
)

clamp_temperatures_contract = KernelContract(
    kernel_name="clamp_temperatures",
    contract_block=KernelContractBlock(
        holistic_constraints="All constraints are defined by the parameter commentary blocks.",
        idempotency="Fundamentally Non-Idempotent (Stateful)",
        synchronization_model="Finalizer Utility",
        behavioral_invariants=("Enforces temps = clamp(temps, min_value, max_value) for each element.",),
    ),
    buffer_params=(
        BufferParamSpec(
            name="update_buffer_GLOBAL_temps", flow="update", memory_scope="GLOBAL",
            tensor_shape=("total_modules_count",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("total_modules_count",),
            validation_preconditions=("exact allocation size",),
            precision_role="state",
        ),
    ),
    scalar_params=(
        ScalarParamSpec("min_value", "src", "REAL"),
        ScalarParamSpec("max_value", "src", "REAL"),
        ScalarParamSpec("total_modules_count", "src", "NATURAL"),
    ),
    local_memory=(),
    placement=None,
)
