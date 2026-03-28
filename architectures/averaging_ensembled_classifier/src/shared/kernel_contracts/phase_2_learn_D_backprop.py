# src/shared/kernel_contracts/phase_2_learn_D_backprop.py
"""Phase 2D (Shared Backprop) kernel contracts — stubs populated in Phase 1."""
from dataclasses import dataclass
from . import KernelContract


@dataclass(frozen=True)
class BackpropSharedWeightsContract(KernelContract):
    """Contract for the backprop_shared_weights kernel."""
    pass


@dataclass(frozen=True)
class BackpropSharedBiasesContract(KernelContract):
    """Contract for the backprop_shared_biases kernel."""
    pass


@dataclass(frozen=True)
class ClipSharedGradientsContract(KernelContract):
    """Contract for the clip_shared_gradients kernel."""
    pass
