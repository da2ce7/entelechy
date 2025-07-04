# kernel_signatures/phase_2_learn_A_production.py

"""
Concrete KernelSignature Implementations for Gradient Production (Nodes 8-10).

This file contains the final, canonical implementations for the kernel launch
signatures related to the first and most parallelizable stage of the 'Learn'
phase. These kernels are responsible for calculating the initial, un-aggregated
partial gradients for a single tile of work.

This implementation follows a rectified design pattern: for each kernel that
depends on `problem_type` (CCE vs. BCE), a private base class encapsulates
shared logic, while two public, type-safe derived classes are exposed to the
orchestrator. This eliminates type ambiguity for the `targets` buffer and
removes the need for the orchestrator to manage the problem type flag, thus
enforcing correctness by design.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
from ..workload_primitives import WorkTile
from ..launcher_infra import BufferHandle, KernelSignature


# === Node 8: Calculate Module Parameter Gradients ===


@dataclass(frozen=True)
class _CalculateModuleParamGradsBase(KernelSignature):
    """(Internal) Shared base for Node 8 gradient production (`Grad_ModW`, `Grad_ModB`)."""

    work_group_size_0: int
    scalar_size_bytes: int
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
    padded_hidden_count: np.uint32 = field(init=False)
    total_batch_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        gw_shape, _ = self._buffer_mgr.get_spec(self.gw_out_ref)
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))
        object.__setattr__(self, "total_batch_count", np.uint32(h_shape[0]))
        object.__setattr__(self, "total_tile_count", np.uint32(gw_shape[0]))

    @property
    def kernel_name(self) -> str:
        return "calculate_module_param_grads_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (self.tile.modules_per_chunk, self.hidden_count, self.tile.classes_per_chunk)
        return global_size, None


@dataclass(frozen=True)
class CalculateModuleParamGradsCceSignature(_CalculateModuleParamGradsBase):
    """(Node 8 - CCE) Type-safe signature for CCE-based param gradients."""

    targets_cce_ref: BufferHandle

    def get_args(self) -> List:
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_cce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.gw_out_ref),
            self._buffer_mgr.get_cl_buffer(self.gb_out_ref),
            np.uint32(0),
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
    """(Node 8 - BCE) Type-safe signature for BCE-based param gradients."""

    targets_bce_ref: BufferHandle

    def get_args(self) -> List:
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_ref),
            self._buffer_mgr.get_cl_buffer(self.targets_bce_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.gw_out_ref),
            self._buffer_mgr.get_cl_buffer(self.gb_out_ref),
            np.uint32(1),
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


# === Node 9: Backpropagate Error to Hidden Layer ===


@dataclass(frozen=True)
class _BackpropErrorToHiddenChunkBase(KernelSignature):
    """(Internal) Shared base for Node 9 gradient production (`Grad_H`)."""

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
        global_size = (self.tile.modules_per_chunk, self.total_batch_count, self.padded_hidden_count)
        return global_size, None


@dataclass(frozen=True)
class BackpropErrorToHiddenChunkCceSignature(_BackpropErrorToHiddenChunkBase):
    """(Node 9 - CCE) Type-safe signature for CCE-based `Grad_H` calculation."""

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
    """(Node 9 - BCE) Type-safe signature for BCE-based `Grad_H` calculation."""

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


# === Node 10: Calculate Temperature Gradients ===


@dataclass(frozen=True)
class _CalculateChunkTempGradientsBase(KernelSignature):
    """(Internal) Shared base for Node 10 gradient production (`Grad_Temps`)."""

    work_group_size_0: int
    scalar_size_bytes: int
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
    """(Node 10 - CCE) Type-safe signature for CCE-based temperature gradients."""

    targets_cce_ref: BufferHandle

    def get_args(self) -> List:
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
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
    """(Node 10 - BCE) Type-safe signature for BCE-based temperature gradients."""

    targets_bce_ref: BufferHandle

    def get_args(self) -> List:
        local_mem_size = self.work_group_size_0 * self.scalar_size_bytes
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
