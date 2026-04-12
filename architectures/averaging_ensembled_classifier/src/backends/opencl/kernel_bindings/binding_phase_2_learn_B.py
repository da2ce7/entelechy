# src/backends/opencl/kernel_bindings/binding_phase_2_learn_B.py
"""KernelBinding adapters for Learn-B gradient processing kernels (Nodes 11, 13).

Each binding translates the backend-neutral plan node's buffer bindings and
scalar parameters into the concrete argument list required by
``clEnqueueNDRangeKernel``, including local-memory allocations, dispatch
geometry, and tile-index delivery (CONTRACT.md §7).

Reference specification: kernels.cl.h (ADR-013 designation).
Dispatch geometry source: phase_2_learn_B_processing.cl.c implementation comments.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ..type_mapping import compute_scalar
from .base import KernelBinding


# ---------------------------------------------------------------------------
# Node 11 — clip_partial_gradients
# ---------------------------------------------------------------------------


class ClipPartialGradientsBinding(KernelBinding):
    """Binding for ``clip_partial_gradients`` (Node 11).

    Dispatch geometry::

        global = (work_group_size)
        local  = (work_group_size)

    A single work-group processes one logical tile's complete gradient set —
    four physically disjoint buffers (weights, biases, temperatures, hidden
    activations) treated as one contiguous "virtual vector."  Two-pass
    algorithm: parallel sum-of-squares reduction (joint L2 norm), then
    conditional uniform scaling.  The uniform-scaling design is the mechanism
    for Padding Zero-Preservation.

    The clipping threshold source is controlled by
    ``src_scalar_FLAG_use_per_item_norm``: when 0, the global scalar
    ``src_scalar_REAL_clipping_threshold_t_pre`` is used; when 1, the
    per-item buffer is read.  The Orchestration tier MUST always provide a
    valid buffer handle for ``src_buffer_GLOBAL_CONST_clipping_threshold_per_item``
    (a minimal stub when the flag is 0).
    """

    def get_kernel_name(self) -> str:
        return "clip_partial_gradients"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # One work-group per tile; grid is tile-independent.
        # Single work-group dispatch: the kernel processes the entire tile's
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
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])
        real: Callable[[str], Any] = lambda key: compute_scalar(
            scalar_params[key], scalar_params
        )

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        simd = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = simd * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_partial_grad_weights_module"),
            buf("src_buffer_GLOBAL_partial_grad_biases_module"),
            buf("src_buffer_GLOBAL_partial_grad_temps"),
            buf("src_buffer_GLOBAL_partial_grad_hidden_activations_aos"),
            buf("src_buffer_GLOBAL_CONST_clipping_threshold_per_item"),
            buf("dest_buffer_GLOBAL_clipped_partial_grad_weights_module"),
            buf("dest_buffer_GLOBAL_clipped_partial_grad_biases_module"),
            buf("dest_buffer_GLOBAL_clipped_partial_grad_temps"),
            buf("dest_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos"),
            u32("src_scalar_FLAG_use_per_item_norm"),
            real("src_scalar_REAL_clipping_threshold_t_pre"),
            real("src_scalar_REAL_epsilon"),
            # Tile-index delivery: binding → src_scalar_NATURAL_flat_tile_index.
            np.uint32(tile_index),
            u32("src_scalar_NATURAL_num_class_chunks"),
            u32("src_scalar_NATURAL_classes_per_chunk"),
            u32("src_scalar_NATURAL_modules_per_chunk"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_padded_total_output_class_count"),
            u32("src_scalar_NATURAL_total_tile_count"),
        ]


# ---------------------------------------------------------------------------
# Node 13 — gather_and_permute_grad_hidden_activations
# ---------------------------------------------------------------------------


class GatherAndPermuteGradHBinding(KernelBinding):
    """Binding for ``gather_and_permute_grad_hidden_activations`` (Node 13).

    Dispatch geometry::

        global = (total_batch_count × padded_hidden_count,
                  total_modules_count)
        local  = backend-selected (None)

    Each work-item gathers the corresponding partial from all class-chunk
    tiles, sums them (implicit reduction over the class-chunk dimension),
    and writes the final value to the SoA destination buffer.  This
    single-pass approach simultaneously solves the "Transpose Illusion" —
    transforming the chunked AoS tile collection into a dense,
    reduction-ready SoA layout.  This kernel is the canonical Item
    Synchronization Point.
    """

    def get_kernel_name(self) -> str:
        return "gather_and_permute_grad_hidden_activations"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Global Barrier — no tiling.
        total_batch = int(
            scalar_params["src_scalar_NATURAL_total_batch_count"]
        )
        padded_hidden = int(
            scalar_params["src_scalar_NATURAL_padded_hidden_count"]
        )
        total_modules = int(
            scalar_params["src_scalar_NATURAL_total_modules_count"]
        )
        return (total_batch * padded_hidden, total_modules), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Global Barrier — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(
            buffer_bindings[name]
        )
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        return [
            buf("src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos"),
            buf("dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_hidden_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
            u32("src_scalar_NATURAL_padded_total_modules_count"),
            u32("src_scalar_NATURAL_num_module_chunks"),
            u32("src_scalar_NATURAL_modules_per_chunk"),
            u32("src_scalar_NATURAL_num_class_chunks"),
            u32("src_scalar_NATURAL_total_tile_count"),
        ]
