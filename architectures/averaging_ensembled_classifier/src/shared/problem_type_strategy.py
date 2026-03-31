# src/shared/problem_type_strategy.py
"""Backend-neutral problem-type strategy abstraction (ADR-011).

Defines PlanProblemTypeStrategy and its concrete implementations
(PlanCceStrategy, PlanBceStrategy) that return KernelContract instances
for plan-level dispatch.
"""

import abc

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

    @property
    @abc.abstractmethod
    def problem_type_flag(self) -> int:
        """0 for CCE, 1 for BCE — matches the FLAG field in kernel structs."""
        ...


class PlanCceStrategy(PlanProblemTypeStrategy):
    """CCE plan strategy — single-label classification."""

    @property
    def problem_type_flag(self) -> int:
        return 0

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
    def problem_type_flag(self) -> int:
        return 1

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
