# src/backends/opencl/kernel_bindings/binding_phase_2_learn_D.py
"""KernelBinding adapters for Learn-D streaming shared-layer backprop kernels
(Nodes 17, 18, 19).

Each binding translates the backend-neutral plan node's buffer bindings and
scalar parameters into the concrete argument list required by
``clEnqueueNDRangeKernel``, including local-memory allocations, dispatch
geometry, and tile-index delivery (CONTRACT.md §7).

Reference specification: kernels.cl.h (ADR-013 designation).
Dispatch geometry source: phase_2_learn_D_backprop.cl.c implementation comments.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ..type_mapping import compute_scalar
from .base import KernelBinding


# ---------------------------------------------------------------------------
# Node 17 — backprop_shared_weights_chunk
# ---------------------------------------------------------------------------


class BackpropSharedWeightsChunkBinding(KernelBinding):
    """Binding for ``backprop_shared_weights_chunk`` (Node 17).

    Dispatch geometry::

        global = (padded_input_count × batch_reduction_wg_size,
                  padded_hidden_count)
        local  = (batch_reduction_wg_size, 1)

    Each work-group computes a single scalar dL/dW_{i,j} (the partial
    gradient for one shared weight) by collaboratively reducing
    contributions from all samples in the assigned batch chunk via local
    memory.  ``get_group_id(0)`` → input dimension index,
    ``get_group_id(1)`` → hidden dimension index, ``get_local_id(0)`` →
    batch reduction lane.

    The SIMD-major (SoA) write pattern matches the persistent shared weight
    layout consumed by Node 4's forward pass and updated by Node 24's Adam
    update, ensuring flat-index correspondence between gradient and
    parameter buffers through the layout-agnostic reduction pipeline.

    Precision Boundary Conversion: storage-role inputs widened via
    ``load_storage()``; compute-role gradient consumed directly; partial
    gradient outputs narrowed via ``store_storage()``.  All arithmetic
    exclusively in COMPUTE_TYPE.

    ReLU derivative source is controlled by
    ``src_scalar_FLAG_use_explicit_hidden_mask``.  When 0, the mask is
    derived from stored activations; when 1, the explicit mask buffer is
    read.  The Host MAY pass a minimal stub buffer for
    ``src_buffer_GLOBAL_hidden_mask`` when the flag is 0.

    Padding Zero-Establishment: the kernel writes zero for SIMD-major
    positions corresponding to logical indices ``i >= input_count`` or
    ``h >= hidden_count`` (Initialization Contract: NONE).
    """

    def get_kernel_name(self) -> str:
        return "backprop_shared_weights_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # Partial Renderer — grid is tile-independent.
        wg_size = hardware_simd_width
        padded_input = int(scalar_params["src_scalar_NATURAL_padded_input_count"])
        padded_hidden = int(scalar_params["src_scalar_NATURAL_padded_hidden_count"])
        return (padded_input * wg_size, padded_hidden), (wg_size, 1)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Partial Renderer — batch_chunk_index is the placement key.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        wg_size = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = wg_size * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_input"),
            buf("src_buffer_GLOBAL_hidden_activations"),
            buf("src_buffer_GLOBAL_hidden_mask"),
            u32("src_scalar_FLAG_use_explicit_hidden_mask"),
            buf("src_buffer_GLOBAL_summed_grad_hidden_activations"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("dest_buffer_GLOBAL_partial_grad_weights_shared_simd_major"),
            u32("src_scalar_NATURAL_batch_chunk_offset"),
            u32("src_scalar_NATURAL_batch_chunk_count"),
            u32("src_scalar_NATURAL_batch_chunk_index"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_num_batch_chunks"),
            u32("src_scalar_NATURAL_input_count"),
            u32("src_scalar_NATURAL_padded_input_count"),
            u32("src_scalar_NATURAL_hidden_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_final_grad_hidden_activations_total_count"),
        ]


# ---------------------------------------------------------------------------
# Node 18 — backprop_shared_biases_chunk
# ---------------------------------------------------------------------------


class BackpropSharedBiasesChunkBinding(KernelBinding):
    """Binding for ``backprop_shared_biases_chunk`` (Node 18).

    Dispatch geometry::

        global = (padded_hidden_count × batch_reduction_wg_size)
        local  = (batch_reduction_wg_size)

    Each work-group computes a single scalar dL/dB_j by collaboratively
    reducing contributions from all samples in the assigned batch chunk via
    local memory.  ``get_group_id(0)`` → hidden dimension index,
    ``get_local_id(0)`` → batch reduction lane.  The 1D dispatch is
    simpler and more efficient than a 2D model because the bias gradient
    does not depend on the input dimension.

    Precision Boundary Conversion: storage-role inputs widened via
    ``load_storage()``; compute-role gradient consumed directly; partial
    gradient outputs narrowed via ``store_storage()``.  All arithmetic
    exclusively in COMPUTE_TYPE.

    ReLU derivative source is controlled by
    ``src_scalar_FLAG_use_explicit_hidden_mask``.  When 0, the mask is
    derived from stored activations; when 1, the explicit mask buffer is
    read.  The Host MAY pass a minimal stub buffer for
    ``src_buffer_GLOBAL_hidden_mask`` when the flag is 0.

    Padding Zero-Establishment: the kernel writes zero for positions at
    indices ``>= hidden_count`` (Initialization Contract: NONE).
    """

    def get_kernel_name(self) -> str:
        return "backprop_shared_biases_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # Partial Renderer — grid is tile-independent.
        wg_size = hardware_simd_width
        padded_hidden = int(scalar_params["src_scalar_NATURAL_padded_hidden_count"])
        return (padded_hidden * wg_size,), (wg_size,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Partial Renderer — batch_chunk_index is the placement key.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        wg_size = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = wg_size * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_hidden_activations"),
            buf("src_buffer_GLOBAL_hidden_mask"),
            u32("src_scalar_FLAG_use_explicit_hidden_mask"),
            buf("src_buffer_GLOBAL_summed_grad_hidden_activations"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("dest_buffer_GLOBAL_partial_grad_biases_shared"),
            u32("src_scalar_NATURAL_batch_chunk_offset"),
            u32("src_scalar_NATURAL_batch_chunk_count"),
            u32("src_scalar_NATURAL_batch_chunk_index"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_num_batch_chunks"),
            u32("src_scalar_NATURAL_hidden_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_final_grad_hidden_activations_total_count"),
        ]


# ---------------------------------------------------------------------------
# Node 19 — clip_shared_gradients_chunk
# ---------------------------------------------------------------------------


class ClipSharedGradientsChunkBinding(KernelBinding):
    """Binding for ``clip_shared_gradients_chunk`` (Node 19).

    Dispatch geometry::

        global = (work_group_size)
        local  = (work_group_size)

    A single work-group processes one batch chunk's complete shared gradient
    set — weight and bias gradients treated as one contiguous "virtual
    vector."  Two-pass algorithm: parallel sum-of-squares reduction (joint
    L2 norm), then conditional uniform scaling with placement write to the
    host-specified destination offsets in the collection buffers.

    This kernel fulfills the same stabilization role as Node 11 for the
    module path, but operates within the "True Streaming" backpropagation
    model where gradients are clipped immediately per batch chunk.  The
    host provides explicit write offsets, making this kernel a "dumb"
    numerical primitive that writes to host-specified memory locations.

    Precision Boundary Conversion: storage-role inputs widened via
    ``load_storage()``; storage-role outputs narrowed via
    ``store_storage()``.  All arithmetic exclusively in COMPUTE_TYPE.

    Padding Zero-Preservation: a mathematical consequence of the
    uniform-scaling design — zero inputs map to zero outputs for all
    finite scale factors.
    """

    def get_kernel_name(self) -> str:
        return "clip_shared_gradients_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # Single work-group per chunk; grid is chunk-independent.
        # Single work-group dispatch: the kernel processes the entire chunk's
        # virtual vector with one work-group, striding in increments of
        # get_local_size(0).
        wg_size = hardware_simd_width
        return (wg_size,), (wg_size,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Utility — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        wg_size = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = wg_size * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_partial_grad_weights_shared_simd_major"),
            buf("src_buffer_GLOBAL_partial_grad_biases_shared"),
            buf("dest_buffer_GLOBAL_clipped_partial_grad_weights_shared_simd_major"),
            buf("dest_buffer_GLOBAL_clipped_partial_grad_biases_shared"),
            real("src_scalar_REAL_clipping_threshold_t_pre"),
            real("src_scalar_REAL_epsilon"),
            u32("src_scalar_NATURAL_weights_parameter_count"),
            u32("src_scalar_NATURAL_biases_parameter_count"),
            u32("dest_scalar_NATURAL_weights_write_offset"),
            u32("dest_scalar_NATURAL_biases_write_offset"),
            u32("src_scalar_NATURAL_num_batch_chunks"),
        ]
