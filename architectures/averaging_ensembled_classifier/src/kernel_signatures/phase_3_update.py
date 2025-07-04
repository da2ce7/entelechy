# kernel_signatures/phase_3_update.py

"""
The Definitive, Executable Contracts for the Final Update Phase (Nodes 21, 24, 25).

Jurisdictional Mandate:
This file is the canonical Python-side embodiment of the C-level kernel
contracts for the culmination of the learning cycle. Its jurisdiction covers
the final, irreversible acts of learning: gradient normalization, stateful
parameter updates, and the enforcement of domain-specific constraints.

Architectural Role:
The signatures herein represent the final execution of the Host Orchestrator's
plan.
- `NormalizeGradientsSignature` is the bridge between the raw, summed output of
  the reduction engine and the mathematically correct average gradient required
  for stable mini-batch training.
- `AdamUpdateSignature` is the most architecturally significant. It enforces a
  critical system mandate: the host is solely responsible for computing the
  optimizer's bias correction terms, preventing on-device precision loss and
  guaranteeing long-term numerical stability.
- `ClampTemperaturesSignature` upholds the principle of single-purpose modules,
  acting as a final, domain-specific state modification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Primitives & Core Infrastructure ---
from ..launcher_infra import BufferHandle, KernelSignature, BufferManager
from ..cl_context_manager import DiscoveredArchConstants


# =========================================================================
# === API Clarification Primitives ===
# =========================================================================


@dataclass(frozen=True)
class AdamParameterGroup:
    """
    A helper dataclass to cleanly group all buffers for a single parameter
    group into a single, coherent unit for the Adam optimizer. This simplifies
    the `AdamUpdateSignature` interface and makes its intent self-documenting.
    """

    param_ref: BufferHandle  # The learnable parameter buffer (e.g., weights)
    grad_ref: BufferHandle  # The corresponding final, averaged gradient
    m1_state_ref: BufferHandle  # The first moment vector (momentum)
    m2_state_ref: BufferHandle  # The second moment vector (RMSProp)


# =========================================================================
# === Node 21: Pre-Update Normalization
# =========================================================================


@dataclass(frozen=True)
class NormalizeGradientsSignature(KernelSignature):
    """(Node 21) Signature for the `normalize_gradients` kernel."""

    # --- Injected System Context (The Architectural Mandate) ---
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    # --- Kernel-Specific Buffers & Control Scalars ---
    summed_grad_ref: BufferHandle
    final_grad_out_ref: BufferHandle
    # This value is the result of a host-side reduction over the sample_mask,
    # as mandated by the kernel's `Calculability Proof` contract.
    effective_batch_size: np.float32
    epsilon: np.float32

    # --- Derived Fields ---
    element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives the element count from the destination buffer spec."""
        super().__post_init__()
        shape, _ = self._buffer_mgr.get_spec(self.final_grad_out_ref)
        object.__setattr__(self, "element_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "normalize_gradients"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # This is a simple, element-wise kernel.
        return (int(self.element_count),), None

    def get_args(self) -> List:
        """Returns all 5 arguments in the exact order mandated by `kernels.cl.h`."""
        return [
            self._buffer_mgr.get_cl_buffer(self.summed_grad_ref),
            self._buffer_mgr.get_cl_buffer(self.final_grad_out_ref),
            self.effective_batch_size,
            self.epsilon,
            self.element_count,
        ]


# =========================================================================
# === Node 24 & 25: Optimizer Update & Constraint Enforcement
# =========================================================================


@dataclass(frozen=True)
class AdamUpdateSignature(KernelSignature):
    """(Node 24) Signature for the `adam_update` stateful optimizer kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    param_group: AdamParameterGroup

    # --- Hyperparameters ---
    learning_rate: np.float32
    beta1: np.float32
    beta2: np.float32
    epsilon: np.float32

    # --- THE CRITICAL HOST-SIDE CONTRACT ---
    # These values MUST be pre-computed by the host in high precision (e.g.,
    # float64) to prevent on-device precision loss and underflow during long
    # training runs. This is a non-negotiable architectural mandate.
    beta1_pow_t: np.float32
    beta2_pow_t: np.float32

    # --- Derived Fields ---
    parameter_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives the parameter count from the parameter buffer spec."""
        super().__post_init__()
        shape, _ = self._buffer_mgr.get_spec(self.param_group.param_ref)
        object.__setattr__(self, "parameter_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "adam_update"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # This is an element-wise update applied to all parameters in parallel.
        return (int(self.parameter_count),), None

    def get_args(self) -> List:
        """Assembles all 11 arguments in strict contractual order."""
        pg = self.param_group
        return [
            self._buffer_mgr.get_cl_buffer(pg.grad_ref),
            self._buffer_mgr.get_cl_buffer(pg.param_ref),
            self._buffer_mgr.get_cl_buffer(pg.m1_state_ref),
            self._buffer_mgr.get_cl_buffer(pg.m2_state_ref),
            self.learning_rate,
            self.beta1_pow_t,
            self.beta2_pow_t,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.parameter_count,
        ]


@dataclass(frozen=True)
class ClampTemperaturesSignature(KernelSignature):
    """(Node 25) Signature for the `clamp_temperatures` domain constraint kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    temps_ref: BufferHandle  # This buffer is modified in-place.
    min_val: np.float32
    max_val: np.float32

    element_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        shape, _ = self._buffer_mgr.get_spec(self.temps_ref)
        object.__setattr__(self, "element_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "clamp_temperatures"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        return (int(self.element_count),), None

    def get_args(self) -> List:
        """Returns all 4 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.temps_ref),
            self.min_val,
            self.max_val,
            self.element_count,
        ]
