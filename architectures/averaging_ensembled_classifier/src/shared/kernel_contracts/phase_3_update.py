# src/shared/kernel_contracts/phase_3_update.py
"""Phase 3 (Update) kernel contracts — stubs populated in Phase 1."""
from dataclasses import dataclass
from . import KernelContract


@dataclass(frozen=True)
class NormalizeGradientsContract(KernelContract):
    """Contract for the normalize_gradients kernel."""
    pass


@dataclass(frozen=True)
class AdamUpdateContract(KernelContract):
    """Contract for the adam_update kernel."""
    pass


@dataclass(frozen=True)
class ClampTemperaturesContract(KernelContract):
    """Contract for the clamp_temperatures kernel."""
    pass
