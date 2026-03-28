# src/shared/kernel_contracts/__init__.py
"""Backend-neutral kernel contracts (ADR-007).

Each contract is a frozen dataclass describing a kernel's interface
independently of any backend. Populated in Phase 1.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelContract:
    """Base class for all backend-neutral kernel contracts."""
    kernel_name: str
