# src/shared/kernel_contracts/phase_2_learn_A_production.py
"""Phase 2A (Gradient Production) kernel contracts — stubs populated in Phase 1."""
from dataclasses import dataclass
from . import KernelContract


@dataclass(frozen=True)
class ComputeProbsLossCceContract(KernelContract):
    """Contract for the compute_probs_loss_cce_chunk kernel."""
    pass


@dataclass(frozen=True)
class ComputeProbsLossBceContract(KernelContract):
    """Contract for the compute_probs_loss_bce_chunk kernel."""
    pass


@dataclass(frozen=True)
class CalculateModuleParamGradsCceContract(KernelContract):
    """Contract for the calculate_module_param_grads_cce kernel."""
    pass


@dataclass(frozen=True)
class CalculateModuleParamGradsBceContract(KernelContract):
    """Contract for the calculate_module_param_grads_bce kernel."""
    pass
