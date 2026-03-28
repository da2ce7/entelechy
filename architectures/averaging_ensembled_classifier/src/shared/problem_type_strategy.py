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
