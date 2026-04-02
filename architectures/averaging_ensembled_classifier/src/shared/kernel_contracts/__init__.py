# src/shared/kernel_contracts/__init__.py
"""Backend-neutral kernel contracts (ADR-007).

Each contract is a frozen dataclass describing a kernel's interface
independently of any backend.
"""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class PaddingContract:
    """Padding specification (CONTRACT.md Article 3.1)."""
    padding_type: Literal["CACHE", "BANK_CONFLICT_AVOIDANCE", "SIMD", "NONE"]
    formula: str | None


@dataclass(frozen=True)
class BufferParamSpec:
    """Specification for a single buffer parameter in a kernel contract."""
    name: str
    flow: Literal["src", "dest", "update", "sync"]
    memory_scope: Literal["GLOBAL", "LOCAL", "GLOBAL_CONST", "DEVICE_CONST"]
    tensor_shape: tuple[str, ...]
    padding_contract: PaddingContract
    calculability_proof: tuple[str, ...]
    validation_preconditions: tuple[str, ...]
    # None is reserved for integer-typed buffers (e.g. CCE/BCE targets).
    precision_role: Optional[Literal["storage", "compute", "state"]]


@dataclass(frozen=True)
class ScalarParamSpec:
    """Specification for a single scalar parameter in a kernel contract."""
    name: str
    flow: Literal["src", "dest"]
    number_type: Literal["NATURAL", "INTEGER", "REAL", "FLAG"]


@dataclass(frozen=True)
class LocalMemorySpec:
    """Specification for a local memory requirement."""
    name: str
    size_expr: str


@dataclass(frozen=True)
class PlacementContract:
    """Placement strategy specification (CONTRACT.md Article 3.2)."""
    strategy: str
    key_domain: tuple[int, int] | None
    context_params: dict[str, str]


@dataclass(frozen=True)
class KernelContractBlock:
    """Kernel-level contract metadata (CONTRACT.md Article 4)."""
    holistic_constraints: str
    idempotency: Literal[
        "Strictly Idempotent",
        "Associatively Non-Idempotent",
        "Fundamentally Non-Idempotent (Stateful)"
    ]
    synchronization_model: str | None
    behavioral_invariants: tuple[str, ...] | None


@dataclass(frozen=True)
class KernelContract:
    """Backend-neutral kernel interface contract (ADR-007).

    Carries the complete interface specification for plan-construction-time
    validation. Does not carry backend-specific dispatch details — those
    belong in each backend's KernelBinding.
    """
    kernel_name: str
    contract_block: KernelContractBlock
    buffer_params: tuple[BufferParamSpec, ...]
    scalar_params: tuple[ScalarParamSpec, ...]
    local_memory: tuple[LocalMemorySpec, ...]
    placement: PlacementContract | None
