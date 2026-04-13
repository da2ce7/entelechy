# src/backends/opencl/kernel_bindings/binding_phase_2_learn_A.py
"""KernelBinding adapters for Learn-A gradient production kernels (Nodes 8, 9, 10).

Each binding translates the backend-neutral plan node's buffer bindings and
scalar parameters into the concrete argument list required by
``clEnqueueNDRangeKernel``, including local-memory allocations, dispatch
geometry, and tile-index delivery (CONTRACT.md §7).

Reference specification: kernels.cl.h (ADR-013 designation).
Dispatch geometry source: phase_2_learn_A_production.cl.c implementation comments.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from .base import KernelBinding


# ---------------------------------------------------------------------------
# Node 8 — calculate_module_param_grads_chunk
# ---------------------------------------------------------------------------


class CalculateModuleParamGradsChunkBinding(KernelBinding):
    """Binding for ``calculate_module_param_grads_chunk`` (Node 8).

    Dispatch geometry::

        global = (modules_per_chunk × batch_reduction_wg_size,
                  padded_hidden_count,
                  classes_per_chunk)
        local  = (batch_reduction_wg_size, 1, 1)

    Each work-group computes a single scalar in the output gradient tensor
    (e.g., dL/dW_{m,h,c}).  Threads collaborate via local memory to perform
    a tree-structured reduction over the batch chunk dimension.  The fused
    weight/bias scheme computes bias gradients only at ``h_idx == 0``
    work-groups, reusing the same local memory tile.

    Problem type (CCE/BCE) is controlled by Strategy A (host-injected FLAG
    scalar ``src_scalar_FLAG_problem_type``).  Per Principle §3
    (Inter-Dispatch Reduction Delegation), each ``(tile, batch_chunk)``
    pair writes to a unique slot; batch-chunk summation is delegated to a
    ``ReductionTreeNode`` inserted before Node 11.
    """

    def get_kernel_name(self) -> str:
        return "calculate_module_param_grads_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # Partial Renderer — grid is tile-independent.
        wg_size = hardware_simd_width
        modules_per_chunk = int(
            scalar_params["src_scalar_NATURAL_modules_per_chunk"]
        )
        padded_hidden = int(
            scalar_params["src_scalar_NATURAL_padded_hidden_count"]
        )
        classes_per_chunk = int(
            scalar_params["src_scalar_NATURAL_classes_per_chunk"]
        )
        return (
            modules_per_chunk * wg_size,
            padded_hidden,
            classes_per_chunk,
        ), (wg_size, 1, 1)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        simd = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = simd * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_hidden_activations"),
            buf("src_buffer_GLOBAL_partial_probs"),
            buf("src_buffer_GLOBAL_targets"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("src_buffer_GLOBAL_CONST_temps"),
            buf("dest_buffer_GLOBAL_partial_grad_weights_module"),
            buf("dest_buffer_GLOBAL_partial_grad_biases_module"),
            u32("src_scalar_FLAG_problem_type"),
            # Tile-index delivery: binding → src_scalar_NATURAL_flat_tile_index.
            np.uint32(tile_index),
            u32("src_scalar_NATURAL_batch_chunk_index"),
            u32("src_scalar_NATURAL_batch_chunk_offset"),
            u32("src_scalar_NATURAL_batch_chunk_count"),
            u32("src_scalar_NATURAL_num_batch_chunks"),
            u32("src_scalar_NATURAL_num_class_chunks"),
            u32("src_scalar_NATURAL_classes_per_chunk"),
            u32("src_scalar_NATURAL_modules_per_chunk"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_hidden_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_total_output_class_count"),
            u32("src_scalar_NATURAL_padded_total_output_class_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
            u32("src_scalar_NATURAL_total_tile_count"),
        ]


# ---------------------------------------------------------------------------
# Node 9 — backprop_error_to_hidden_chunk
# ---------------------------------------------------------------------------


class BackpropErrorToHiddenChunkBinding(KernelBinding):
    """Binding for ``backprop_error_to_hidden_chunk`` (Node 9).

    Dispatch geometry::

        global = (modules_per_chunk, total_batch_count, padded_hidden_count)
        local  = backend-selected (None)

    Each work-item computes a single scalar in the partial upstream gradient
    tensor (Grad_H) by performing a serial dot product over its assigned
    class chunk.  This maps a matrix-vector multiply to embarrassingly
    parallel work-items.  Padding Zero-Establishment writes zeros for
    ``h_idx >= hidden_count`` (Initialization Contract: NOT_REQUIRED).

    The contract mandates no batch-chunking: the downstream Item
    Synchronization Point (Node 13) requires a monolithic collection buffer.

    Problem type (CCE/BCE) is controlled by Strategy A (host-injected FLAG
    scalar ``src_scalar_FLAG_problem_type``).
    """

    def get_kernel_name(self) -> str:
        return "backprop_error_to_hidden_chunk"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index, hardware_simd_width  # 3D map — no tiling concerns.
        return (
            int(scalar_params["src_scalar_NATURAL_modules_per_chunk"]),
            int(scalar_params["src_scalar_NATURAL_total_batch_count"]),
            int(scalar_params["src_scalar_NATURAL_padded_hidden_count"]),
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
            buf("src_buffer_GLOBAL_partial_probs"),
            buf("src_buffer_GLOBAL_targets"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("src_buffer_GLOBAL_CONST_weights_module"),
            buf("src_buffer_GLOBAL_CONST_temps"),
            buf("dest_buffer_GLOBAL_partial_grad_hidden_activations_aos"),
            u32("src_scalar_FLAG_problem_type"),
            # Tile-index delivery: binding → src_scalar_NATURAL_flat_tile_index.
            np.uint32(tile_index),
            u32("src_scalar_NATURAL_num_class_chunks"),
            u32("src_scalar_NATURAL_classes_per_chunk"),
            u32("src_scalar_NATURAL_modules_per_chunk"),
            u32("src_scalar_NATURAL_total_batch_count"),
            u32("src_scalar_NATURAL_hidden_count"),
            u32("src_scalar_NATURAL_padded_hidden_count"),
            u32("src_scalar_NATURAL_total_output_class_count"),
            u32("src_scalar_NATURAL_padded_total_output_class_count"),
            u32("src_scalar_NATURAL_total_modules_count"),
            u32("src_scalar_NATURAL_total_tile_count"),
        ]


# ---------------------------------------------------------------------------
# Node 10 — calculate_chunk_temp_gradients
# ---------------------------------------------------------------------------


class CalculateChunkTempGradientsBinding(KernelBinding):
    """Binding for ``calculate_chunk_temp_gradients`` (Node 10).

    Dispatch geometry::

        global = (modules_per_chunk × batch_reduction_wg_size)
        local  = (batch_reduction_wg_size)

    Each work-group computes the partial temperature gradient for a single
    module within its tile, performing a three-level hierarchical reduction
    (batch × class chunk × work-group tree).  The final chain-rule factor
    (−1/τ²) is applied post-reduction.

    Problem type (CCE/BCE) is controlled by Strategy A (host-injected FLAG
    scalar ``src_scalar_FLAG_problem_type``).
    """

    def get_kernel_name(self) -> str:
        return "calculate_chunk_temp_gradients"

    def compute_grid(
        self,
        tile_index: int,
        scalar_params: dict[str, int | float],
        hardware_simd_width: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        _ = tile_index  # Partial Renderer — grid is tile-independent.
        wg_size = hardware_simd_width
        modules_per_chunk = int(
            scalar_params["src_scalar_NATURAL_modules_per_chunk"]
        )
        return (modules_per_chunk * wg_size,), (wg_size,)

    def marshal_args(
        self,
        get_buffer: Callable[[BufferHandle], cl.Buffer],
        buffer_bindings: dict[str, BufferHandle],
        scalar_params: dict[str, int | float],
        tile_index: int,
    ) -> list[Any]:
        buf: Callable[[str], cl.Buffer] = lambda name: get_buffer(buffer_bindings[name])
        u32: Callable[[str], np.uint32] = lambda key: np.uint32(scalar_params[key])

        # Local-memory sizing (CONTRACT Article 3, Allocation Formula):
        #   get_local_size(0) × sizeof(COMPUTE_TYPE)
        simd = int(scalar_params["SIMD_WIDTH"])
        compute_bytes = int(scalar_params["_compute_type_size_bytes"])
        local_bytes = simd * compute_bytes

        return [
            cl.LocalMemory(local_bytes),
            buf("src_buffer_GLOBAL_logits"),
            buf("src_buffer_GLOBAL_partial_probs"),
            buf("src_buffer_GLOBAL_targets"),
            buf("src_buffer_GLOBAL_sample_mask"),
            buf("src_buffer_GLOBAL_CONST_temps"),
            buf("dest_buffer_GLOBAL_partial_grad_temps"),
            u32("src_scalar_FLAG_problem_type"),
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
