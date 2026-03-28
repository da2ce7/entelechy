# src/shared/problem_type_strategy.py
"""Backend-neutral problem-type strategy abstraction (ADR-011).

NOTE (Phase 0): The ProblemTypeStrategy ABC and its concrete implementations
(CceStrategy, BceStrategy) are placed in the shared layer per ADR-011, but
they currently reference OpenCL-specific KernelSignature types from the backend.
This temporary Plan-boundary violation is resolved in Phase 1 when strategies
produce KernelContract instances instead of KernelSignature instances.
"""

import abc
from dataclasses import dataclass

from ..backends.opencl.launcher_infra import BufferHandle, KernelSignature
from ..backends.opencl.kernel_bindings import (
    ComputeProbsLossCceChunkSignature,
    ComputeProbsLossBceChunkSignature,
    CalculateModuleParamGradsCceSignature,
    CalculateModuleParamGradsBceSignature,
    BackpropErrorToHiddenChunkCceSignature,
    BackpropErrorToHiddenChunkBceSignature,
    CalculateChunkTempGradientsCceSignature,
    CalculateChunkTempGradientsBceSignature,
)


class ProblemTypeStrategy(abc.ABC):
    """
    Function: An abstract contract for a problem type (e.g., CCE, BCE).

    Architectural Mandate:
    This is the embodiment of the classical Strategy Pattern. It replaces messy,
    procedural `if/elif/else` blocks in the core execution logic with a clean,
    polymorphic interface. By doing so, it upholds the Open/Closed Principle:
    the system is open to extension (one can add a new loss function by creating
    a new strategy class) but closed for modification (the `BatchProcessor`
    need never be changed).
    """

    @property
    @abc.abstractmethod
    def required_targets_buffer_name(self) -> str:
        """The canonical name of the buffer this strategy requires for ground truth."""
        pass

    @abc.abstractmethod
    def get_loss_signature(self, **kwargs) -> KernelSignature:
        """A factory for the appropriate loss computation signature (Node 6/7)."""
        pass

    @abc.abstractmethod
    def get_module_grad_signature(self, **kwargs) -> KernelSignature:
        """A factory for the module parameter gradient signature (Node 8)."""
        pass

    @abc.abstractmethod
    def get_hidden_grad_signature(self, **kwargs) -> KernelSignature:
        """A factory for the upstream hidden gradient signature (Node 9)."""
        pass

    @abc.abstractmethod
    def get_temp_grad_signature(self, **kwargs) -> KernelSignature:
        """A factory for the temperature gradient signature (Node 10)."""
        pass


@dataclass(frozen=True)
class CceStrategy(ProblemTypeStrategy):
    """The type-safe, concrete strategy for CCE (single-label classification)."""

    targets_cce_ref: BufferHandle  # Contractually binds this strategy to the integer-typed targets buffer.

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_cce"

    def get_loss_signature(self, **kwargs) -> "ComputeProbsLossCceChunkSignature":
        # CCE uses `loss_out_ref`; discard the BCE-specific `partial_loss_out_ref`.
        kwargs.pop("partial_loss_out_ref", None)
        return ComputeProbsLossCceChunkSignature(**kwargs, target_ref=self.targets_cce_ref)

    def get_module_grad_signature(self, **kwargs) -> "CalculateModuleParamGradsCceSignature":
        return CalculateModuleParamGradsCceSignature(**kwargs, targets_cce_ref=self.targets_cce_ref)

    def get_hidden_grad_signature(self, **kwargs) -> "BackpropErrorToHiddenChunkCceSignature":
        return BackpropErrorToHiddenChunkCceSignature(**kwargs, targets_cce_ref=self.targets_cce_ref)

    def get_temp_grad_signature(self, **kwargs) -> "CalculateChunkTempGradientsCceSignature":
        return CalculateChunkTempGradientsCceSignature(**kwargs, targets_cce_ref=self.targets_cce_ref)


@dataclass(frozen=True)
class BceStrategy(ProblemTypeStrategy):
    """The type-safe, concrete strategy for BCE (multi-label classification)."""

    targets_bce_ref: BufferHandle  # Contractually binds this strategy to the float-typed targets buffer.

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_bce"

    def get_loss_signature(self, **kwargs) -> "ComputeProbsLossBceChunkSignature":
        # BCE uses `partial_loss_out_ref`; discard the CCE-specific `loss_out_ref`.
        kwargs.pop("loss_out_ref", None)
        return ComputeProbsLossBceChunkSignature(**kwargs, target_ref=self.targets_bce_ref)

    def get_module_grad_signature(self, **kwargs) -> "CalculateModuleParamGradsBceSignature":
        return CalculateModuleParamGradsBceSignature(**kwargs, targets_bce_ref=self.targets_bce_ref)

    def get_hidden_grad_signature(self, **kwargs) -> "BackpropErrorToHiddenChunkBceSignature":
        return BackpropErrorToHiddenChunkBceSignature(**kwargs, targets_bce_ref=self.targets_bce_ref)

    def get_temp_grad_signature(self, **kwargs) -> "CalculateChunkTempGradientsBceSignature":
        return CalculateChunkTempGradientsBceSignature(**kwargs, targets_bce_ref=self.targets_bce_ref)


# =========================================================================
# Plan-layer strategies (ADR-011, Phase 1) — no OpenCL imports
# =========================================================================

from .kernel_contracts import KernelContract
from .kernel_contracts.phase_2_learn_A_production import (
    compute_probs_loss_cce_contract,
    compute_probs_loss_bce_contract,
    calculate_module_param_grads_contract,
)
from .kernel_contracts.phase_2_learn_B_processing import (
    backprop_error_to_hidden_contract,
    calculate_chunk_temp_gradients_contract,
)


class PlanProblemTypeStrategy(abc.ABC):
    """Clean plan-layer strategy returning KernelContract instances (ADR-011).

    Replaces the legacy ProblemTypeStrategy (which returns KernelSignature).
    Legacy class is retained until Phase 6 for backward compatibility.
    """

    @property
    @abc.abstractmethod
    def required_targets_buffer_name(self) -> str:
        ...

    @abc.abstractmethod
    def get_loss_contract(self) -> KernelContract:
        ...

    @abc.abstractmethod
    def get_module_grad_contract(self) -> KernelContract:
        ...

    @abc.abstractmethod
    def get_hidden_grad_contract(self) -> KernelContract:
        ...

    @abc.abstractmethod
    def get_temp_grad_contract(self) -> KernelContract:
        ...


class PlanCceStrategy(PlanProblemTypeStrategy):
    """CCE plan strategy — single-label classification."""

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_cce"

    def get_loss_contract(self) -> KernelContract:
        return compute_probs_loss_cce_contract

    def get_module_grad_contract(self) -> KernelContract:
        return calculate_module_param_grads_contract

    def get_hidden_grad_contract(self) -> KernelContract:
        return backprop_error_to_hidden_contract

    def get_temp_grad_contract(self) -> KernelContract:
        return calculate_chunk_temp_gradients_contract


class PlanBceStrategy(PlanProblemTypeStrategy):
    """BCE plan strategy — multi-label classification."""

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_bce"

    def get_loss_contract(self) -> KernelContract:
        return compute_probs_loss_bce_contract

    def get_module_grad_contract(self) -> KernelContract:
        return calculate_module_param_grads_contract

    def get_hidden_grad_contract(self) -> KernelContract:
        return backprop_error_to_hidden_contract

    def get_temp_grad_contract(self) -> KernelContract:
        return calculate_chunk_temp_gradients_contract
