# src/backends/opencl/kernel_bindings/binding_phase_2_learn_A.py
"""KernelBinding adapters for Learn-A gradient production kernels (Nodes 8-10)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from .base import KernelBinding


class CalculateModuleParamGradsChunkBinding(KernelBinding):
    """Unified binding for calculate_module_param_grads_chunk (Node 8).

    Reads problem_type from scalar_params (0=CCE, 1=BCE).
    """

    def get_kernel_name(self) -> str:
        return "calculate_module_param_grads_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        modules_per_chunk = int(scalar_params["modules_per_chunk"])
        hidden_count = int(scalar_params["hidden_count"])
        classes_per_chunk = int(scalar_params["classes_per_chunk"])
        global_size = (
            modules_per_chunk * wg_size_0,
            hidden_count,
            classes_per_chunk,
        )
        local_size = (wg_size_0, 1, 1)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = wg_size_0 * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_grad_weights_module"]),
            get_buffer(buffer_bindings["partial_grad_biases_module"]),
            np.uint32(scalar_params["problem_type"]),
            np.uint32(tile_index),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["classes_per_chunk"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_output_class_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]


class CalculateModuleParamGradsCceBinding(KernelBinding):
    """Binding for calculate_module_param_grads_chunk (CCE variant, Node 8)."""

    def get_kernel_name(self) -> str:
        return "calculate_module_param_grads_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        modules_per_chunk = int(scalar_params["modules_per_chunk"])
        hidden_count = int(scalar_params["hidden_count"])
        classes_per_chunk = int(scalar_params["classes_per_chunk"])
        global_size = (
            modules_per_chunk * wg_size_0,
            hidden_count,
            classes_per_chunk,
        )
        local_size = (wg_size_0, 1, 1)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = wg_size_0 * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_grad_weights_module"]),
            get_buffer(buffer_bindings["partial_grad_biases_module"]),
            np.uint32(0),  # PROBLEM_TYPE_CCE
            np.uint32(tile_index),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["classes_per_chunk"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_output_class_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]


class CalculateModuleParamGradsBceBinding(KernelBinding):
    """Binding for calculate_module_param_grads_chunk (BCE variant, Node 8)."""

    def get_kernel_name(self) -> str:
        return "calculate_module_param_grads_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        modules_per_chunk = int(scalar_params["modules_per_chunk"])
        hidden_count = int(scalar_params["hidden_count"])
        classes_per_chunk = int(scalar_params["classes_per_chunk"])
        global_size = (
            modules_per_chunk * wg_size_0,
            hidden_count,
            classes_per_chunk,
        )
        local_size = (wg_size_0, 1, 1)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = wg_size_0 * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_grad_weights_module"]),
            get_buffer(buffer_bindings["partial_grad_biases_module"]),
            np.uint32(1),  # PROBLEM_TYPE_BCE
            np.uint32(tile_index),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["classes_per_chunk"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_output_class_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]


class BackpropErrorToHiddenBinding(KernelBinding):
    """Binding for backprop_error_to_hidden_chunk (Strategy A — flag-driven, Node 9)."""

    def get_kernel_name(self) -> str:
        return "backprop_error_to_hidden_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        global_size = (
            int(scalar_params["modules_per_chunk"]),
            int(scalar_params["total_batch_count"]),
            int(scalar_params["padded_hidden_count"]),
        )
        return global_size, None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["weights_module"]),
            get_buffer(buffer_bindings["partial_grad_hidden_activations_aos"]),
            np.uint32(scalar_params["problem_type"]),
            np.uint32(tile_index),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["classes_per_chunk"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_output_class_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]


class CalculateChunkTempGradientsBinding(KernelBinding):
    """Binding for calculate_chunk_temp_gradients (Strategy A — flag-driven, Node 10)."""

    def get_kernel_name(self) -> str:
        return "calculate_chunk_temp_gradients"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        modules_per_chunk = int(scalar_params["modules_per_chunk"])
        global_size = (modules_per_chunk * wg_size_0,)
        local_size = (wg_size_0,)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        wg_size_0 = int(scalar_params["work_group_size_0"])
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_size = wg_size_0 * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["logits"]),
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["temps"]),
            get_buffer(buffer_bindings["partial_grad_temps"]),
            np.uint32(scalar_params["problem_type"]),
            np.uint32(tile_index),
            np.uint32(scalar_params["num_class_chunks"]),
            np.uint32(scalar_params["classes_per_chunk"]),
            np.uint32(scalar_params["modules_per_chunk"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["total_output_class_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_modules_count"]),
            np.uint32(scalar_params["total_tile_count"]),
        ]
