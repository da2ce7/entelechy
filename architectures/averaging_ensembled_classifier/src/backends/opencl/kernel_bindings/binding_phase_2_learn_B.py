# src/backends/opencl/kernel_bindings/binding_phase_2_learn_B.py
"""KernelBinding adapters for Learn-B gradient processing kernels (Nodes 11, 13)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ....shared.memory_layout import pad_to_multiple
from .base import KernelBinding


class ClipPartialGradientsBinding(KernelBinding):
    """Binding for clip_partial_gradients kernel (Node 11)."""

    def get_kernel_name(self) -> str:
        return "clip_partial_gradients"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg_size = int(scalar_params["optimal_workgroup_size_1d_reduction"])
        padded_hidden_count = int(scalar_params["padded_hidden_count"])
        padded_class_count = int(scalar_params["padded_total_output_class_count"])
        modules_per_chunk = int(scalar_params["modules_per_chunk"])
        total_elements = (
            (padded_hidden_count * padded_class_count)  # weights
            + padded_class_count  # biases
            + 1  # temps
            + padded_hidden_count  # hidden_activations
        ) * modules_per_chunk
        global_size = (pad_to_multiple(total_elements, wg_size),)
        local_size = (wg_size,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        wg_size = int(scalar_params["optimal_workgroup_size_1d_reduction"])
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = wg_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["partial_grad_weights_module"]),
            get_buffer(buffer_bindings["partial_grad_biases_module"]),
            get_buffer(buffer_bindings["partial_grad_temps"]),
            get_buffer(buffer_bindings["partial_grad_hidden_activations_aos"]),
            get_buffer(buffer_bindings.get(
                "clipping_threshold_per_item",
                buffer_bindings.get("null_buffer", buffer_bindings["partial_grad_weights_module"]),
            )) if scalar_params.get("use_per_item_norm", 0) else None,
            get_buffer(buffer_bindings["clipped_partial_grad_weights_module"]),
            get_buffer(buffer_bindings["clipped_partial_grad_biases_module"]),
            get_buffer(buffer_bindings["clipped_partial_grad_temps"]),
            get_buffer(buffer_bindings["clipped_partial_grad_hidden_activations_aos"]),
            np.uint32(scalar_params.get("use_per_item_norm", 0)),
            # Negative threshold bypasses clipping (diagnostic mode); do not default to 0.0
            np.float32(scalar_params.get("clipping_threshold_t_pre", -1.0)),
            np.float32(scalar_params["epsilon"]),
            np.uint32(tile_index),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["classes_per_chunk"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]


class GatherAndPermuteGradHBinding(KernelBinding):
    """Binding for gather_and_permute_grad_hidden_activations kernel (Node 13)."""

    def get_kernel_name(self) -> str:
        return "gather_and_permute_grad_hidden_activations"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        total_batch_count = int(scalar_params["total_batch_count"])
        padded_hidden_count = int(scalar_params["padded_hidden_count"])
        padded_total_modules_count = int(scalar_params["padded_total_modules_count"])
        global_size = (
            total_batch_count * padded_hidden_count,
            padded_total_modules_count,
        )
        return global_size, None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["clipped_partial_grad_hidden_activations_aos"]),
            get_buffer(buffer_bindings["clipped_grad_hidden_activations_permuted_soa"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["padded_total_modules_count"]),
            np.uint32(scalar_params["num_module_chunks"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]
