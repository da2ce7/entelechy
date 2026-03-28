# src/shared/kernel_contracts/phase_1_act.py
"""Phase 1 (Act) kernel contracts — stubs populated in Phase 1."""
from dataclasses import dataclass
from . import KernelContract


@dataclass(frozen=True)
class ForwardPassContract(KernelContract):
    """Contract for the forward_pass kernel."""
    pass


@dataclass(frozen=True)
class ComputeHiddenMaskContract(KernelContract):
    """Contract for the compute_hidden_mask kernel."""
    pass
