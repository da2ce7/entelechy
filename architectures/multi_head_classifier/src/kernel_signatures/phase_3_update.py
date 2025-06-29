# kernel_signatures/phase_3_update.py

"""
Concrete KernelSignature Implementations for the Update Phase (Nodes 20, 23, 24).

This file contains the final, canonical implementations for the kernel launch
signatures related to the culmination of the learning cycle: gradient
normalization, parameter updates via the Adam optimizer, and domain-specific
constraint enforcement (clamping).

The `AdamUpdateSignature` is the most architecturally significant class in this
file, as it contractually requires the host to provide pre-computed bias
correction terms, thus solving a critical flaw in the original system design.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..launcher_infra import BufferHandle, KernelSignature, SCALAR_NP_TYPE


# === Pre-Update Normalization (Node 20) ===


# In file: kernel_signatures/phase_3_update.py


@dataclass(frozen=True)
class NormalizeGradientsSignature(KernelSignature):
    """
    (Node 20) Signature for the `normalize_gradients` kernel.

    (REV 2 - Rectified) This version corrects the previous implementation,
    which was missing the mandatory 'epsilon' argument. This signature is
    now in full compliance with the kernel's 5-argument contract.
    """

    summed_grad_ref: BufferHandle
    final_grad_out_ref: BufferHandle
    effective_batch_size: SCALAR_NP_TYPE
    epsilon: SCALAR_NP_TYPE

    # --- Derived Scalar Fields ---
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


# === Optimizer Update & Constraint Enforcement (Nodes 23, 24) ===


@dataclass(frozen=True)
class AdamParameterGroup:
    """A helper dataclass to cleanly group buffers for the Adam optimizer."""

    param_ref: BufferHandle  # The learnable parameter buffer (e.g., weights)
    grad_ref: BufferHandle  # The corresponding final gradient buffer
    m1_state_ref: BufferHandle  # The first moment vector (momentum)
    m2_state_ref: BufferHandle  # The second moment vector (RMSProp)


@dataclass(frozen=True)
class AdamUpdateSignature(KernelSignature):
    """(Node 23) Signature for the `adam_update` kernel."""

    # --- Buffer Handles (grouped for clarity) ---
    param_group: AdamParameterGroup

    # --- Hyperparameters & Control Scalars ---
    learning_rate: SCALAR_NP_TYPE
    beta1: SCALAR_NP_TYPE
    beta2: SCALAR_NP_TYPE
    epsilon: SCALAR_NP_TYPE

    # --- THE CRITICAL HOST-SIDE CONTRACT ---
    # These values MUST be pre-computed by the host in high precision.
    beta1_pow_t: SCALAR_NP_TYPE
    beta2_pow_t: SCALAR_NP_TYPE

    # --- Derived Scalar Fields ---
    parameter_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives the parameter count from the parameter buffer spec."""
        shape, _ = self._buffer_mgr.get_spec(self.param_group.param_ref)
        object.__setattr__(self, "parameter_count", np.uint32(np.prod(shape)))

    @property
    def kernel_name(self) -> str:
        return "adam_update"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        return (int(self.parameter_count),), None

    def get_args(self) -> List:
        """Returns all 11 arguments in exact contractual order."""
        pg = self.param_group
        return [
            self._buffer_mgr.get_cl_buffer(pg.param_ref),
            self._buffer_mgr.get_cl_buffer(pg.grad_ref),
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
    """(Node 24) Signature for the `clamp_temperatures` domain constraint kernel."""

    temps_ref: BufferHandle  # Operates in-place
    min_val: SCALAR_NP_TYPE
    max_val: SCALAR_NP_TYPE
    element_count: np.uint32 = field(init=False)

    def __post_init__(self):
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
