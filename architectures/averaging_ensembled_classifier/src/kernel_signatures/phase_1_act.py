# kernel_signatures/phase_1_act.py

"""
The Definitive, Executable Contract for the 'Act' Phase (Nodes 4-7).

Jurisdictional Mandate:
This file is the canonical Python-side embodiment of the C-level kernel
contracts defined in `kernels.cl.h` for the 'Act' (forward pass and loss
computation) phase of the system's Directed Acyclic Graph (DAG). Each class
herein is not merely a data container, but an immutable, self-sufficient
'artisan' responsible for a single kernel dispatch.

Architectural Role:
These signature classes serve as the sole, verifiable bridge between the
Host Orchestrator's strategic intent and the KernelExecutor's tactical
dispatch. By being instantiated with the system's core context
(`BufferManager`, `DiscoveredArchConstants`), they fulfill the Axiom of Interface
Verifiability, possessing all knowledge required to derive their own execution
parameters and marshal their arguments, thus ensuring correctness by design.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Primitives (The Vocabulary of Work) ---
# These are the low-level, stateless descriptors of the workload.
from ..workload_primitives import WorkTile

# --- Core Infrastructure (The Tools of the Artisan) ---
# These are the foundational components upon which all signatures are built.
from ..launcher_infra import BufferHandle, KernelSignature, BufferManager
from ..memory_layout import _pad_to_multiple
from ..cl_context_manager import DiscoveredArchConstants


@dataclass(frozen=True)
class ForwardPassSignature(KernelSignature):
    """(Node 4) Signature for the `forward_pass` shared layer kernel."""

    # --- Injected System Context (The Architectural Mandate) ---
    # These objects provide the complete memory and hardware context, making the
    # signature self-sufficient and fulfilling its role as a pure 'artisan'.
    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    # --- Kernel-Specific Buffers (The Materials for this Operation) ---
    in_ref: BufferHandle
    mask_ref: BufferHandle
    w_ref: BufferHandle
    b_ref: BufferHandle
    h_out_ref: BufferHandle
    h_mask_out_ref: BufferHandle

    # --- Kernel-Specific Control Scalars (The Blueprint for this Operation) ---
    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32

    # --- Derived Scalar Fields (The Self-Sufficiency Contract) ---
    # These fields are derived internally from the buffer specifications, ensuring
    # the signature is the sole authority on the physical dimensions it commands.
    total_batch_count: np.uint32 = field(init=False)
    padded_input_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)

    def __post_init__(self):
        """Derives physical dimensions from the injected memory context."""
        super().__post_init__()
        in_shape, _ = self._buffer_mgr.get_spec(self.in_ref)
        h_out_shape, _ = self._buffer_mgr.get_spec(self.h_out_ref)

        # Using object.__setattr__ as the dataclass is frozen and immutable.
        object.__setattr__(self, "total_batch_count", np.uint32(in_shape[0]))
        object.__setattr__(self, "padded_input_count", np.uint32(in_shape[1]))
        object.__setattr__(self, "padded_hidden_count", np.uint32(h_out_shape[1]))

    @property
    def kernel_name(self) -> str:
        return "forward_pass"

    def get_grid(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Calculates the execution grid from the injected hardware context."""
        simd = self._arch_consts.simd_width
        global_size = (
            _pad_to_multiple(int(self.batch_chunk_count), simd),
            int(self.padded_hidden_count) // simd,
        )
        local_size = (simd, 1)
        return global_size, local_size

    def get_args(self) -> List:
        """Assembles arguments in the exact order mandated by `kernels.cl.h`."""
        simd = self._arch_consts.simd_width
        scalar_bytes = self._arch_consts.SCALAR_NP_TYPE().itemsize
        # The padding is a fixed architectural constant from the System Contract.
        local_mem_padding = 1
        local_mem_size = simd * (simd + local_mem_padding) * scalar_bytes

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

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    h_ref: BufferHandle
    h_mask_ref: BufferHandle
    w_ref: BufferHandle
    b_ref: BufferHandle
    logit_out_ref: BufferHandle

    batch_chunk_offset: np.uint32
    batch_chunk_count: np.uint32
    module_chunk_offset: np.uint32
    module_chunk_count: np.uint32
    class_chunk_offset: np.uint32
    class_chunk_count: np.uint32

    hidden_count: np.uint32
    total_output_class_count: np.uint32

    total_batch_count: np.uint32 = field(init=False)
    padded_hidden_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
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
        global_size = (
            int(self.module_chunk_count),
            int(self.batch_chunk_count),
            int(self.class_chunk_count),
        )
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
    """(Node 6) Signature for the fused `compute_probs_loss_cce_chunk` kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    logit_ref: BufferHandle
    temp_ref: BufferHandle
    target_ref: BufferHandle
    mask_ref: BufferHandle
    prob_out_ref: BufferHandle
    loss_out_ref: BufferHandle

    tile: WorkTile

    total_output_class_count: np.uint32

    total_batch_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
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
        global_size = (
            self.tile.modules_per_chunk,
            int(self.total_batch_count),
            self.tile.classes_per_chunk,
        )
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
    """(Node 7) Signature for the `compute_probs_loss_bce_chunk` kernel."""

    _buffer_mgr: BufferManager
    _arch_consts: DiscoveredArchConstants

    logit_ref: BufferHandle
    temp_ref: BufferHandle
    target_ref: BufferHandle
    mask_ref: BufferHandle
    prob_out_ref: BufferHandle
    # This kernel is a "PARTIAL" renderer for loss, requiring a later reduction.
    partial_loss_out_ref: BufferHandle

    tile: WorkTile

    total_output_class_count: np.uint32

    total_batch_count: np.uint32 = field(init=False)
    padded_total_output_class_count: np.uint32 = field(init=False)
    total_modules_count: np.uint32 = field(init=False)
    total_tile_count: np.uint32 = field(init=False)

    def __post_init__(self):
        super().__post_init__()
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
        global_size = (
            self.tile.modules_per_chunk,
            int(self.total_batch_count),
            self.tile.classes_per_chunk,
        )
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
