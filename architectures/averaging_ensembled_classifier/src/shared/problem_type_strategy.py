# src/shared/problem_type_strategy.py
"""Backend-neutral problem-type strategy abstraction (ADR-011).

Defines PlanProblemTypeStrategy and its concrete implementations
(PlanCceStrategy, PlanBceStrategy) that return KernelContract instances
for plan-level dispatch.
"""

import abc
from typing import Literal

from .kernel_contracts import (
    KernelContract,
    compute_probs_loss_cce_chunk,
    compute_probs_loss_bce_chunk,
    calculate_module_param_grads_chunk,
    backprop_error_to_hidden_chunk,
    calculate_chunk_temp_gradients,
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

    @property
    @abc.abstractmethod
    def problem_type_flag(self) -> int:
        """0 for CCE, 1 for BCE — matches the FLAG field in kernel structs."""
        ...

    @abc.abstractmethod
    def get_targets_element_size(self, storage_dtype_size: int) -> int:
        """Return element size for targets buffer (4 for CCE int, storage_dtype_size for BCE)."""
        ...

    @abc.abstractmethod
    def get_targets_precision_role(self) -> Literal["storage", "compute", "state"] | None:
        """Return precision role for targets buffer (None for CCE int, 'storage' for BCE)."""
        ...
        ...


class PlanCceStrategy(PlanProblemTypeStrategy):
    """CCE plan strategy — single-label classification."""

    @property
    def problem_type_flag(self) -> int:
        return 0

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_cce"

    def get_targets_element_size(self, storage_dtype_size: int) -> int:
        return 4  # int32 class indices

    def get_targets_precision_role(self) -> Literal["storage", "compute", "state"] | None:
        return None  # integer-typed, no precision role

    def get_loss_contract(self) -> KernelContract:
        return compute_probs_loss_cce_chunk

    def get_module_grad_contract(self) -> KernelContract:
        return calculate_module_param_grads_chunk

    def get_hidden_grad_contract(self) -> KernelContract:
        return backprop_error_to_hidden_chunk

    def get_temp_grad_contract(self) -> KernelContract:
        return calculate_chunk_temp_gradients


class PlanBceStrategy(PlanProblemTypeStrategy):
    """BCE plan strategy — multi-label classification."""

    @property
    def problem_type_flag(self) -> int:
        return 1

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_bce"

    def get_targets_element_size(self, storage_dtype_size: int) -> int:
        return storage_dtype_size  # multi-hot floats in storage precision

    def get_targets_precision_role(self) -> Literal["storage", "compute", "state"] | None:
        return "storage"  # BCE targets are storage-role (ADR-021)

    def get_loss_contract(self) -> KernelContract:
        return compute_probs_loss_bce_chunk

    def get_module_grad_contract(self) -> KernelContract:
        return calculate_module_param_grads_chunk

    def get_hidden_grad_contract(self) -> KernelContract:
        return backprop_error_to_hidden_chunk

    def get_temp_grad_contract(self) -> KernelContract:
        return calculate_chunk_temp_gradients
