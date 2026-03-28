# src/shared/kernel_contracts/phase_2_learn_C_reduction.py
"""Phase 2C (Reduction) kernel contracts — stubs populated in Phase 1."""
from dataclasses import dataclass
from . import KernelContract


@dataclass(frozen=True)
class GatherAndPermuteContract(KernelContract):
    """Contract for the gather_and_permute kernel."""
    pass


@dataclass(frozen=True)
class AggregateRegisterReduceContract(KernelContract):
    """Contract for the aggregate_register_reduce kernel."""
    pass


@dataclass(frozen=True)
class AggregateLocalReduceContract(KernelContract):
    """Contract for the aggregate_local_reduce kernel."""
    pass


@dataclass(frozen=True)
class ClipIntermediateGradContract(KernelContract):
    """Contract for the clip_intermediate_grad kernel."""
    pass


@dataclass(frozen=True)
class StabilizeReduceGradHContract(KernelContract):
    """Contract for the stabilize_and_reduce_grad_h kernel."""
    pass
