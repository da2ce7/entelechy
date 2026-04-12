# src/backends/opencl/kernel_bindings/binding_phase_1_act.py
"""KernelBinding adapters for Act-phase kernels (Nodes 4, 5, 6, 7).

Each binding translates the backend-neutral plan node's buffer bindings and
scalar parameters into the concrete argument list required by
``clEnqueueNDRangeKernel``, including local-memory allocations, dispatch
geometry, and tile-index delivery (CONTRACT.md §7).

Reference specification: kernels.cl.h (ADR-013 designation).
Dispatch geometry source: phase_1_act.cl.c implementation comments.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from .base import KernelBinding


# ---------------------------------------------------------------------------
# Node 4 — forward_pass
# ---------------------------------------------------------------------------


class ForwardPassBinding(KernelBinding):
    """Binding for ``forward_pass`` (Node 4).

    Dispatch geometry::

        global = (batch_chunk_count × SIMD_WIDTH,
                  padded_hidden_count / SIMD_WIDTH)
        local  = (SIMD_WIDTH, 1)

    Each work-group computes one SIMD-width tile of hidden activations for
    one sample.  Local memory broadcasts the input-vector slice to all
    lanes, reducing global memory traffic by a factor of SIMD_WIDTH.
    """

    def get_kernel_name(self) -> str:
        return "forward_pass"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # Streamable — grid is tile-independent.
        simd = hardware_simd_width
        batch_count = int(scalar_params["src_scalar_NATURAL_batch_chunk_count"])
        padded_hidden = int(scalar_params["src_scalar_NATURAL_padded_hidden_count"])
        return (batch_count * simd, padded_hidden // simd), (simd, 1)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Streamable — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   (SIMD_WIDTH + SIMD_WIDTH²) × sizeof(COMPUTE_TYPE)
        simd = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = (simd + simd * simd) * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_input"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("src_buffer_GLOBAL_CONST_weights_shared_simd_major"),
            buf("src_buffer_GLOBAL_CONST_biases_shared"),
            buf("dest_buffer_GLOBAL_hidden_activations"),
            buf("dest_buffer_GLOBAL_hidden_mask"),
            u32("dest_scalar_FLAG_produce_hidden_mask"),
            u32("src_scalar_NATURAL_batch_chunk_offset"),
            u32("src_scalar_NATURAL_batch_chunk_count"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_input_count"),
            u32("src_scalar_NATURAL_padded_input_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
        ]


# ---------------------------------------------------------------------------
# Node 5 — render_logits_chunk
# ---------------------------------------------------------------------------


class RenderLogitsChunkBinding(KernelBinding):
    """Binding for ``render_logits_chunk`` (Node 5).

    Dispatch geometry::

        global = (module_chunk_count, batch_chunk_count, class_chunk_count)
        local  = backend-selected (None)

    Each work-item computes exactly one logit value — an embarrassingly
    parallel structure over the (module × batch × class) volume.  The
    sparsity-aware dot product elides terms for ReLU-zeroed hidden units.
    """

    def get_kernel_name(self) -> str:
        return "render_logits_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Slice Renderer — no tiling.
        return (
            int(scalar_params["src_scalar_NATURAL_module_chunk_count"]),
            int(scalar_params["src_scalar_NATURAL_batch_chunk_count"]),
            int(scalar_params["src_scalar_NATURAL_class_chunk_count"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        _ = tile_index  # Slice Renderer — no tile placement.
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        return [
            buf("src_buffer_GLOBAL_hidden_activations"),
            buf("src_buffer_GLOBAL_hidden_mask"),
            u32("src_scalar_FLAG_use_explicit_hidden_mask"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("src_buffer_GLOBAL_CONST_weights_module"),
            buf("src_buffer_GLOBAL_CONST_biases_module"),
            buf("dest_buffer_GLOBAL_logits"),
            u32("src_scalar_NATURAL_batch_chunk_offset"),
            u32("src_scalar_NATURAL_batch_chunk_count"),
            u32("src_scalar_NATURAL_module_chunk_offset"),
            u32("src_scalar_NATURAL_module_chunk_count"),
            u32("src_scalar_NATURAL_class_chunk_offset"),
            u32("src_scalar_NATURAL_class_chunk_count"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_hidden_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_total_output_class_count"),
            u32("src_scalar_NATURAL_padded_total_output_class_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
        ]


# ---------------------------------------------------------------------------
# Node 6 — compute_probs_loss_cce_chunk
# ---------------------------------------------------------------------------


class ComputeProbsLossCceChunkBinding(KernelBinding):
    """Binding for ``compute_probs_loss_cce_chunk`` (Node 6).

    Dispatch geometry::

        global = (modules_per_chunk, total_batch_count)
        local  = backend-selected (None)

    Each work-item handles one (module, sample) pair: a serial reduction
    engine for the full numerically-stable Softmax, acting as a Partial
    Renderer for probabilities and a Conditional Writer for CCE loss.
    """

    def get_kernel_name(self) -> str:
        return "compute_probs_loss_cce_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Grid is tile-independent.
        return (
            int(scalar_params["src_scalar_NATURAL_modules_per_chunk"]),
            int(scalar_params["src_scalar_NATURAL_total_batch_count"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        return [
            buf("src_buffer_GLOBAL_logits"),
            buf("src_buffer_GLOBAL_CONST_temps"),
            buf("src_buffer_GLOBAL_targets"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("dest_buffer_GLOBAL_partial_probs"),
            buf("dest_buffer_GLOBAL_final_loss"),
            # Tile-index delivery: binding → src_scalar_NATURAL_flat_tile_index.
            np.uint32(tile_index),
            u32("src_scalar_NATURAL_num_class_chunks"),
            u32("src_scalar_NATURAL_classes_per_chunk"),
            u32("src_scalar_NATURAL_modules_per_chunk"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_total_output_class_count"),
            u32("src_scalar_NATURAL_padded_total_output_class_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
            u32("src_scalar_NATURAL_total_tile_count"),
        ]


# ---------------------------------------------------------------------------
# Node 7 — compute_probs_loss_bce_chunk
# ---------------------------------------------------------------------------


class ComputeProbsLossBceChunkBinding(KernelBinding):
    """Binding for ``compute_probs_loss_bce_chunk`` (Node 7).

    Dispatch geometry::

        global = (modules_per_chunk, total_batch_count)
        local  = backend-selected (None)

    Each work-item handles one (module, sample) pair.  The Sigmoid uses
    the numerically stable two-branch form; BCE loss uses the logit-domain
    identity ``max(x,0) − x·y + log(1 + exp(−|x|))`` to avoid log(0).
    Dual Partial Renderer for both probability and loss outputs.
    """

    def get_kernel_name(self) -> str:
        return "compute_probs_loss_bce_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # Grid is tile-independent.
        return (
            int(scalar_params["src_scalar_NATURAL_modules_per_chunk"]),
            int(scalar_params["src_scalar_NATURAL_total_batch_count"]),
        ), None

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        return [
            buf("src_buffer_GLOBAL_logits"),
            buf("src_buffer_GLOBAL_CONST_temps"),
            buf("src_buffer_GLOBAL_targets"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("dest_buffer_GLOBAL_partial_probs"),
            buf("dest_buffer_GLOBAL_partial_loss"),
            # Tile-index delivery: binding → src_scalar_NATURAL_flat_tile_index.
            np.uint32(tile_index),
            u32("src_scalar_NATURAL_num_class_chunks"),
            u32("src_scalar_NATURAL_classes_per_chunk"),
            u32("src_scalar_NATURAL_modules_per_chunk"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_total_output_class_count"),
            u32("src_scalar_NATURAL_padded_total_output_class_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
            u32("src_scalar_NATURAL_total_tile_count"),
        ]
