# kernel_signatures/phase_1_act.py

"""
Concrete KernelSignature Implementations for the 'Act' Phase (Nodes 4-7).

This file contains the final, executable implementations for the kernel launch
signatures related to the forward pass and initial loss calculation. Each class
is a direct, Pythonic embodiment of its corresponding C kernel contract defined
in `kernels.cl.h`.

These classes are designed to be instantiated by the Host Orchestrator and
dispatched by the pure KernelExecutor. Their design enforces correctness by
requiring all necessary parameters at instantiation and deriving internal
scalars from the provided buffer specifications.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Local Infrastructure Imports ---
# These are the foundational components upon which these signatures are built.
from ..workload_primitives import WorkTile
from ..launcher_infra import BufferHandle, KernelSignature
from ..memory_layout import _pad_to_multiple


@dataclass(frozen=True)
class ForwardPassSignature(KernelSignature):
    """(Node 4) Signature for the `forward_pass` shared layer kernel."""

    # --- Injected Architectural Constants (Edict #3 Compliance) ---
    simd_width: int
    local_mem_bank_padding: int
    scalar_size_bytes: int

    # --- Buffer Handles (from kernel contract) ---
    in_ref: BufferHandle
    mask_ref: BufferHandle
    w_ref: BufferHandle
    b_ref: BufferHandle
    h_out_ref: BufferHandle
    h_mask_out_ref: BufferHandle

    # --- Control Scalars (from kernel contract) ---
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32

    # --- Derived Scalar Fields (Edict #2 Compliance) ---
    total_batch_count: np.uint32 = field(init=False)
    padded_input_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives dimensional parameters from provided buffer handles."""
        in_shape, _ = self._buffer_mgr.get_spec(self.in_ref)
        h_out_shape, _ = self._buffer_mgr.get_spec(self.h_out_ref)

        # Use object.__setattr__ as the dataclass is frozen (Edict #4)
        object.__setattr__(self, "total_batch_count", np.uint32(in_shape[0]))
        object.__setattr__(self, "padded_input_count", np.uint32(in_shape[1]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_out_shape[1]))

    @property
    def kernel_name(self) -> str:
        return "forward_pass"

    def get_grid(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Calculates the execution grid based on derived dimensions."""
        global_size = (
            _pad_to_multiple(self.batch_chunk_count, self.simd_width),
            self.padded_hidden_count // self.simd_width,
        )
        local_size = (self.simd_width, 1)
        return global_size, local_size

    def get_args(self) -> List:
        """Returns arguments in the exact order mandated by kernels.cl.h (Edict #5)."""
        local_mem_size = self.simd_width * (self.simd_width + self.local_mem_bank_padding) * self.scalar_size_bytes

        return [
            cl.LocalMemory(local_mem_size),
            self._buffer_mgr.get_cl_buffer(self.in_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.w_ref),
            self._buffer_mgr.get_cl_buffer(self.b_ref),
            self._buffer_mgr.get_cl_buffer(self.h_out_ref),
            self._buffer_mgr.get_cl_buffer(self.h_mask_out_ref),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            self.total_batch_count,
            self.padded_input_count,
            self.padded_hidden_count,
        ]


@dataclass(frozen=True)
class RenderLogitsChunkSignature(KernelSignature):
    """(Node 5) Signature for the `render_logits_chunk` kernel."""

    # --- Buffer Handles ---
    h_ref: BufferHandle
    h_mask_ref: BufferHandle
    w_ref: BufferHandle
    b_ref: BufferHandle
    logit_out_ref: BufferHandle

    # --- Control Scalars ---
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    module_chunk_offset: np.uint32
    module_chunk_count: np.uint32
    class_chunk_offset: np.uint32
    class_chunk_count: np.uint32

    # --- Unpadded Dimensions (must be known by orchestrator) ---
    hidden_count: np.uint32
    total_output_class_count: np.uint32

    # --- Derived Scalar Fields ---
    total_batch_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives padded and total dimensional parameters from buffer specs."""
        h_shape, _ = self._buffer_mgr.get_spec(self.h_ref)
        w_shape, _ = self._buffer_mgr.get_spec(self.w_ref)
        object.__setattr__(self, "total_batch_count", np.uint32(h_shape[0]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_shape[1]))
        object.__setattr__(self, "total_modules_count", np.uint32(w_shape[0]))
        object.__setattr__(self, "padded_total_output_class_count", np.uint32(w_shape[2]))

    @property
    def kernel_name(self) -> str:
        return "render_logits_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (self.module_chunk_count, self.batch_chunk_count, self.class_chunk_count)
        return global_size, None

    def get_args(self) -> List:
        """Returns all 17 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.h_ref),
            self._buffer_mgr.get_cl_buffer(self.h_mask_ref),
            self._buffer_mgr.get_cl_buffer(self.w_ref),
            self._buffer_mgr.get_cl_buffer(self.b_ref),
            self._buffer_mgr.get_cl_buffer(self.logit_out_ref),
            self.batch_chunk_offset,
            self.batch_chunk_count,
            self.module_chunk_offset,
            self.module_chunk_count,
            self.class_chunk_offset,
            self.class_chunk_count,
            self.total_batch_count,
            self.hidden_count,
            self.padded_hidden_count,
            self.total_output_class_count,
            self.padded_total_output_class_count,
            self.total_modules_count,
        ]


@dataclass(frozen=True)
class ComputeProbsLossCceChunkSignature(KernelSignature):
    """(Node 6) Signature for the fused CCE loss kernel."""

    # --- Buffer Handles ---
    logit_ref: BufferHandle
    temp_ref: BufferHandle
    target_ref: BufferHandle
    mask_ref: BufferHandle
    prob_out_ref: BufferHandle
    loss_out_ref: BufferHandle

    # --- Control Object ---
    tile: WorkTile

    # --- Unpadded Dimension ---
    total_output_class_count: np.uint32

    # --- Derived Scalar Fields ---
    total_batch_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        logit_shape, _ = self._buffer_mgr.get_spec(self.logit_ref)
        prob_shape, _ = self._buffer_mgr.get_spec(self.prob_out_ref)
        object.__setattr__(self, "total_modules_count", np.uint32(logit_shape[0]))
        object.__setattr__(self, "total_batch_count", np.uint32(logit_shape[1]))
        object.__setattr__(self, "padded_total_output_class_count", np.uint32(logit_shape[2]))
        object.__setattr__(self, "total_tile_count", np.uint32(prob_shape[0]))

    @property
    def kernel_name(self) -> str:
        return "compute_probs_loss_cce_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (self.tile.modules_per_chunk, self.total_batch_count, self.tile.classes_per_chunk)
        return global_size, None

    def get_args(self) -> List:
        """Returns all 15 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.logit_ref),
            self._buffer_mgr.get_cl_buffer(self.temp_ref),
            self._buffer_mgr.get_cl_buffer(self.target_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_out_ref),
            self._buffer_mgr.get_cl_buffer(self.loss_out_ref),
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
class ComputeProbsLossBceChunkSignature(KernelSignature):
    """(Node 7) Signature for the BCE loss kernel."""

    # --- Buffer Handles ---
    logit_ref: BufferHandle
    temp_ref: BufferHandle
    target_ref: BufferHandle
    mask_ref: BufferHandle
    prob_out_ref: BufferHandle
    partial_loss_out_ref: BufferHandle  # Note the difference from CCE

    # --- Control Object ---
    tile: WorkTile

    # --- Unpadded Dimension ---
    total_output_class_count: np.uint32

    # --- Derived Scalar Fields (identical to CCE) ---
    total_batch_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        logit_shape, _ = self._buffer_mgr.get_spec(self.logit_ref)
        prob_shape, _ = self._buffer_mgr.get_spec(self.prob_out_ref)
        object.__setattr__(self, "total_modules_count", np.uint32(logit_shape[0]))
        object.__setattr__(self, "total_batch_count", np.uint32(logit_shape[1]))
        object.__setattr__(self, "padded_total_output_class_count", np.uint32(logit_shape[2]))
        object.__setattr__(self, "total_tile_count", np.uint32(prob_shape[0]))

    @property
    def kernel_name(self) -> str:
        return "compute_probs_loss_bce_chunk"

    def get_grid(self) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        global_size = (self.tile.modules_per_chunk, self.total_batch_count, self.tile.classes_per_chunk)
        return global_size, None

    def get_args(self) -> List:
        """Returns all 15 arguments in exact contractual order."""
        return [
            self._buffer_mgr.get_cl_buffer(self.logit_ref),
            self._buffer_mgr.get_cl_buffer(self.temp_ref),
            self._buffer_mgr.get_cl_buffer(self.target_ref),
            self._buffer_mgr.get_cl_buffer(self.mask_ref),
            self._buffer_mgr.get_cl_buffer(self.prob_out_ref),
            self._buffer_mgr.get_cl_buffer(self.partial_loss_out_ref),
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
