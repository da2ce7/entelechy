# src/backends/opencl/kernel_bindings/binding_phase_2_learn_D.py
"""KernelBinding adapters for Learn-D shared layer backprop kernels (Nodes 17-19)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ....shared.memory_layout import pad_to_multiple
from ..type_mapping import compute_scalar
from .base import KernelBinding


class BackpropSharedWeightsBinding(KernelBinding):
    """Binding for backprop_shared_weights_chunk (Node 17)."""

    def __init__(self, rectangular_tile_dim1: int = 16) -> None:
        self._tile_dim1 = rectangular_tile_dim1

    def get_kernel_name(self) -> str:
        return "backprop_shared_weights_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        padded_input = int(scalar_params["padded_input_count"])
        padded_hidden = int(scalar_params["padded_hidden_count"])
        wg_1 = self._tile_dim1
        global_size = (
            padded_input,
            pad_to_multiple(padded_hidden, wg_1),
        )
        local_size = (1, wg_1)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._tile_dim1 * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["input"]),
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["hidden_mask"]),
            np.uint32(scalar_params["FLAG_use_explicit_hidden_mask"]),
            get_buffer(buffer_bindings["summed_grad_hidden_activations"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_grad_weights_shared"]),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["batch_chunk_index"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["num_batch_chunks"]),
            np.uint32(scalar_params["input_count"]),
            np.uint32(scalar_params["padded_input_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["final_grad_hidden_activations_total_count"]),
        ]


class BackpropSharedBiasesBinding(KernelBinding):
    """Binding for backprop_shared_biases_chunk (Node 18)."""

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "backprop_shared_biases_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        padded_hidden = int(scalar_params["padded_hidden_count"])
        wg = self._workgroup_size
        global_size = (padded_hidden * wg,)
        local_size = (wg,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["hidden_mask"]),
            np.uint32(scalar_params["FLAG_use_explicit_hidden_mask"]),
            get_buffer(buffer_bindings["summed_grad_hidden_activations"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_grad_biases_shared"]),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["batch_chunk_index"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["num_batch_chunks"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["final_grad_hidden_activations_total_count"]),
        ]


class ClipSharedGradientsBinding(KernelBinding):
    """Binding for clip_shared_gradients_chunk (Node 19)."""

    def __init__(self, workgroup_size: int = 256) -> None:
        self._workgroup_size = workgroup_size

    def get_kernel_name(self) -> str:
        return "clip_shared_gradients_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg = self._workgroup_size
        weights_count = int(scalar_params["weights_parameter_count"])
        biases_count = int(scalar_params["biases_parameter_count"])
        total_elements = weights_count + biases_count
        global_size = (pad_to_multiple(total_elements, wg),)
        local_size = (wg,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = self._workgroup_size * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["partial_grad_weights_shared"]),
            get_buffer(buffer_bindings["partial_grad_biases_shared"]),
            get_buffer(buffer_bindings["clipped_partial_grad_weights_shared"]),
            get_buffer(buffer_bindings["clipped_partial_grad_biases_shared"]),
            compute_scalar(scalar_params["clipping_threshold_t_pre"], scalar_params),
            compute_scalar(scalar_params["epsilon"], scalar_params),
            np.uint32(scalar_params["weights_parameter_count"]),
            np.uint32(scalar_params["biases_parameter_count"]),
            np.uint32(scalar_params["weights_write_offset"]),
            np.uint32(scalar_params["biases_write_offset"]),
            np.uint32(scalar_params["num_batch_chunks"]),
        ]
