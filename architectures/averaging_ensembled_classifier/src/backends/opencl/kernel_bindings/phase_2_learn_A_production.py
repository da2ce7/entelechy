# kernel_signatures/phase_2_learn_A_production.py

"""
The Definitive, Executable Contracts for the Initial Gradient Production Stage (Nodes 8-10).

Jurisdictional Mandate:
This file is the canonical Python-side embodiment of the C-level kernel
contracts for the first, massively parallel stage of the 'Learn' phase. Its
jurisdiction is to define the immutable, self-sufficient 'artisan' classes
responsible for producing the raw, un-aggregated partial gradients for all
module-specific parameters (`Grad_ModW`, `Grad_ModB`, `Grad_H`, `Grad_Temps`).

Architectural Rectification:
This version formalizes a powerful design pattern: for each kernel whose
logic depends on the problem type (CCE vs. BCE), an internal base class
encapsulates shared logic, while two distinct, public, type-safe classes
are exposed. This is a profound architectural improvement:
  1. It enforces correctness by design, making it impossible to pass the
     wrong type of 'targets' buffer to the wrong kernel variant.
  2. It absolves the higher-level recipes of managing a 'problem_type' flag,
     as the choice of signature class itself encodes this information.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Primitives & Core Infrastructure ---
from ....shared.workload_primitives import WorkTile
from ..launcher_infra import BufferHandle, KernelSignature, BufferManager
from ..context import DiscoveredArchConstants


# =========================================================================
# === Node 8: Calculate Module Parameter Gradients (`Grad_ModW`, `Grad_ModB`)
# =========================================================================


@dataclass(frozen=True)
class _CalculateModuleParamGradsBase(KernelSignature):
    """
    (Internal) A shared base class to enforce consistency and prevent code
    duplication for the Node 8 `calculate_module_param_grads_chunk` kernel.
    This class is an implementation detail and not part of the public API.
    """

    # --- Injected System Context (The Architectural Mandate) ---
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    # --- Kernel-Specific Needs ---
    work_group_size_0: int
    h_ref: BufferHandle
    prob_ref: BufferHandle
    mask_ref: BufferHandle
    gw_out_ref: BufferHandle
    gb_out_ref: BufferHandle
    tile: WorkTile
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    hidden_count: np.uint32
    total_output_class_count: np.uint32
    padded_total_output_class_count: np.uint32
    total_modules_count: np.uint32

    # --- Derived Fields ---
    padded_hidden_count: np.uint32 = field(init=False)
    total_batch_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives physical dimensions from the injected memory context."""
        super().__post_init__()
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        gw_shape, _ = self._buffer_mgr.get_spec(self.gw_out_ref)
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))
        object.__setattr__(self, "total_batch_count", np.uint32(h_shape[0]))
        object.__setattr__(self, "total_tile_count", np.uint32(gw_shape[0]))

    @property
    def kernel_name(self) -> str:
        return "calculate_module_param_grads_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        # WHY: The kernel uses get_group_id() for gradient component mapping
        # (module, hidden, class) and get_local_id(0)/get_local_size(0) for
        # batch reduction within each work-group. The global_size in dim 0
        # must be modules_per_chunk * work_group_size_0 so that there are
        # exactly modules_per_chunk work-groups in that dimension. Dims 1 & 2
        # have local_size = 1, giving one work-group per hidden/class index.
        global_size = (
            self.tile.modules_per_chunk * self.work_group_size_0,
            int(self.hidden_count),
            self.tile.classes_per_chunk,
        )
        local_size = (self.work_group_size_0, 1, 1)
        return global_size, local_size


@dataclass(frozen=True)
class CalculateModuleParamGradsCceSignature(_CalculateModuleParamGradsBase):
    """(Node 8 - CCE) The type-safe, public signature for CCE-based param gradients."""

    targets_cce_ref: BufferHandle  # Contractually accepts the CCE (int) targets buffer.

    def get_args(self) -> List:
        """Assembles arguments, injecting the correct `problem_type` flag (0)."""
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = self.work_group_size_0 * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_cce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.gw_out_ref),
            self._buffer_mgr.get_cl_buffer(self.gb_out_ref),
            np.uint32(0),  # PROBLEM_TYPE_CCE
            np.uint32(self.tile.flat_tile_index),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
            self.total_tile_count,
        ]


@dataclass(frozen=True)
class CalculateModuleParamGradsBceSignature(_CalculateModuleParamGradsBase):
    """(Node 8 - BCE) The type-safe, public signature for BCE-based param gradients."""

    targets_bce_ref: BufferHandle  # Contractually accepts the BCE (float) targets buffer.

    def get_args(self) -> List:
        """Assembles arguments, injecting the correct `problem_type` flag (1)."""
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = self.work_group_size_0 * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_bce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.gw_out_ref),
            self._buffer_mgr.get_cl_buffer(self.gb_out_ref),
            np.uint32(1),  # PROBLEM_TYPE_BCE
            np.uint32(self.tile.flat_tile_index),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
            self.total_tile_count,
        ]


# =========================================================================
# === Node 9: Backpropagate Error to Hidden Layer (`Grad_H`)
# =========================================================================


@dataclass(frozen=True)
class _BackpropErrorToHiddenChunkBase(KernelSignature):
    """(Internal) Shared base for the Node 9 `backprop_error_to_hidden_chunk` kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    prob_ref: BufferHandle
    mask_ref: BufferHandle
    w_mod_ref: BufferHandle
    gh_out_ref: BufferHandle
    tile: WorkTile
    hidden_count: np.uint32
    total_output_class_count: np.uint32

    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_batch_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        w_shape, _ = self._buffer_mgr.get_spec(self.w_mod_ref)
        gh_shape, _ = self._buffer_mgr.get_spec(self.gh_out_ref)
        object.__setattr__(self, "total_modules_count", np.uint32(w_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(w_shape[1]))
        object.__setattr__(self, "padded_total_output_class_count", np.uint32(w_shape[2]))
        object.__setattr__(self, "total_tile_count", np.uint32(gh_shape[0]))
        object.__setattr__(self, "total_batch_count", np.uint32(gh_shape[2]))

    @property
    def kernel_name(self) -> str:
        return "backprop_error_to_hidden_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (
            self.tile.modules_per_chunk,
            int(self.total_batch_count),
            int(self.padded_hidden_count),
        )
        return global_size, None


@dataclass(frozen=True)
class BackpropErrorToHiddenChunkCceSignature(_BackpropErrorToHiddenChunkBase):
    """(Node 9 - CCE) The type-safe, public signature for CCE-based `Grad_H` calculation."""

    targets_cce_ref: BufferHandle

    def get_args(self) -> List:
        return [
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_cce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.w_mod_ref),
            self._buffer_mgr.get_cl_buffer(self.gh_out_ref),
            np.uint32(0),
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
            self.total_tile_count,
        ]


@dataclass(frozen=True)
class BackpropErrorToHiddenChunkBceSignature(_BackpropErrorToHiddenChunkBase):
    """(Node 9 - BCE) The type-safe, public signature for BCE-based `Grad_H` calculation."""

    targets_bce_ref: BufferHandle

    def get_args(self) -> List:
        return [
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_bce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.w_mod_ref),
            self._buffer_mgr.get_cl_buffer(self.gh_out_ref),
            np.uint32(1),
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
            self.total_tile_count,
        ]


# =========================================================================
# === Node 10: Calculate Temperature Gradients (`Grad_Temps`)
# =========================================================================


@dataclass(frozen=True)
class _CalculateChunkTempGradientsBase(KernelSignature):
    """(Internal) Shared base for the Node 10 `calculate_chunk_temp_gradients` kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    work_group_size_0: int
    logit_ref: BufferHandle
    prob_ref: BufferHandle
    mask_ref: BufferHandle
    temp_ref: BufferHandle
    gt_out_ref: BufferHandle
    tile: WorkTile
    total_output_class_count: np.uint32

    padded_total_output_class_count: np.uint32 = field(init=False)
    total_batch_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        logit_shape, _ = self._buffer_mgr.get_spec(self.logit_ref)
        gt_shape, _ = self._buffer_mgr.get_spec(self.gt_out_ref)
        object.__setattr__(self, "total_modules_count", np.uint32(logit_shape[0]))
        object.__setattr__(self, "total_batch_count", np.uint32(logit_shape[1]))
        object.__setattr__(self, "padded_total_output_class_count", np.uint32(logit_shape[2]))
        object.__setattr__(self, "total_tile_count", np.uint32(gt_shape[0]))

    @property
    def kernel_name(self) -> str:
        return "calculate_chunk_temp_gradients"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (self.tile.modules_per_chunk * self.work_group_size_0,)
        local_size = (self.work_group_size_0,)
        return global_size, local_size


@dataclass(frozen=True)
class CalculateChunkTempGradientsCceSignature(_CalculateChunkTempGradientsBase):
    """(Node 10 - CCE) The type-safe, public signature for CCE-based temperature gradients."""

    targets_cce_ref: BufferHandle

    def get_args(self) -> List:
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = self.work_group_size_0 * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.logit_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_cce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.temp_ref),
            self._buffer_mgr.get_cl_buffer(self.gt_out_ref),
            np.uint32(0),
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
            self.total_tile_count,
        ]


@dataclass(frozen=True)
class CalculateChunkTempGradientsBceSignature(_CalculateChunkTempGradientsBase):
    """(Node 10 - BCE) The type-safe, public signature for BCE-based temperature gradients."""

    targets_bce_ref: BufferHandle

    def get_args(self) -> List:
        scalar_size_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        local_mem_size = self.work_group_size_0 * scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.logit_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_bce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.temp_ref),
            self._buffer_mgr.get_cl_buffer(self.gt_out_ref),
            np.uint32(1),
            np.uint32(self.tile.flat_tile_index),
            np.uint32(self.tile.num_class_chunks),
            np.uint32(self.tile.classes_per_chunk),
            np.uint32(self.tile.modules_per_chunk),
            self.total_batch_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
            self.total_tile_count,
        ]
