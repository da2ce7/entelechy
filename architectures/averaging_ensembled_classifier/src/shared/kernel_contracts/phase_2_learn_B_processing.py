# src/shared/kernel_contracts/phase_2_learn_B_processing.py
"""Phase 2B (Gradient Processing) kernel contracts — stubs populated in Phase 1."""
from dataclasses import dataclass
from . import KernelContract


@dataclass(frozen=True)
class BackpropErrorToHiddenContract(KernelContract):
    """Contract for the backprop_error_to_hidden kernel."""
    pass


@dataclass(frozen=True)
class CalculateChunkTempGradientsContract(KernelContract):
    """Contract for the calculate_chunk_temp_gradients kernel."""
    pass


@dataclass(frozen=True)
class ClipPartialGradientsContract(KernelContract):
    """Contract for the clip_partial_gradients kernel."""
    pass
