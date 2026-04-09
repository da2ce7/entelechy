# src/backends/opencl/kernel_bindings/binding_phase_1_act.py
"""KernelBinding adapters for Act-phase kernels (Nodes 4-7)."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyopencl as cl

from ....shared.buffer_lifecycle import BufferHandle
from ....shared.memory_layout import pad_to_multiple
from .base import KernelBinding


class ForwardPassBinding(KernelBinding):
    """Binding for the forward_pass kernel (Node 4)."""

    def get_kernel_name(self) -> str:
        return "forward_pass"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        simd = hardware_simd_width
        batch_chunk_count = int(scalar_params["batch_chunk_count"])
        padded_hidden_count = int(scalar_params["padded_hidden_count"])
        global_size = (
            pad_to_multiple(batch_chunk_count, simd),
            padded_hidden_count // simd,
        )
        local_size = (simd, 1)
        return global_size, local_size

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        simd = int(scalar_params.get("simd_width", 16))
        element_size = int(scalar_params.get("element_size", 4))
        local_mem_padding = 1
        local_mem_size = simd * (simd + local_mem_padding) * element_size
        return [
            cl.LocalMemory(local_mem_size),
            get_buffer(buffer_bindings["input"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["weights_shared_simd_major"]),
            get_buffer(buffer_bindings["biases_shared"]),
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["hidden_mask"]),
            np.uint32(scalar_params["FLAG_produce_hidden_mask"]),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["padded_input_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
        ]


class RenderLogitsChunkBinding(KernelBinding):
    """Binding for the render_logits_chunk kernel (Node 5)."""

    def get_kernel_name(self) -> str:
        return "render_logits_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        global_size = (
            int(scalar_params["module_chunk_count"]),
            int(scalar_params["batch_chunk_count"]),
            int(scalar_params["class_chunk_count"]),
        )
        return global_size, None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["hidden_activations"]),
            get_buffer(buffer_bindings["hidden_mask"]),
            np.uint32(scalar_params["FLAG_use_explicit_hidden_mask"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["weights_module"]),
            get_buffer(buffer_bindings["biases_module"]),
            get_buffer(buffer_bindings["logits"]),
            np.uint32(scalar_params["batch_chunk_offset"]),
            np.uint32(scalar_params["batch_chunk_count"]),
            np.uint32(scalar_params["module_chunk_offset"]),
            np.uint32(scalar_params["module_chunk_count"]),
            np.uint32(scalar_params["class_chunk_offset"]),
            np.uint32(scalar_params["class_chunk_count"]),
            np.uint32(scalar_params["total_batch_count"]),
            np.uint32(scalar_params["hidden_count"]),
            np.uint32(scalar_params["padded_hidden_count"]),
            np.uint32(scalar_params["total_output_class_count"]),
            np.uint32(scalar_params["padded_total_output_class_count"]),
            np.uint32(scalar_params["total_modules_count"]),
        ]


class ComputeProbsLossCceBinding(KernelBinding):
    """Binding for the compute_probs_loss_cce_chunk kernel (Node 6)."""

    def get_kernel_name(self) -> str:
        return "compute_probs_loss_cce_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        global_size = (
            int(scalar_params["modules_per_chunk"]),
            int(scalar_params["total_batch_count"]),
            int(scalar_params["classes_per_chunk"]),
        )
        return global_size, None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["logits"]),
            get_buffer(buffer_bindings["temps"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["final_loss"]),
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


class ComputeProbsLossBceBinding(KernelBinding):
    """Binding for the compute_probs_loss_bce_chunk kernel (Node 7)."""

    def get_kernel_name(self) -> str:
        return "compute_probs_loss_bce_chunk"

    def compute_grid(self, tile_index: int, scalar_params: dict[str, int | float], hardware_simd_width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
        global_size = (
            int(scalar_params["modules_per_chunk"]),
            int(scalar_params["total_batch_count"]),
            int(scalar_params["classes_per_chunk"]),
        )
        return global_size, None

    def marshal_args(self, get_buffer: Callable[[BufferHandle], cl.Buffer], buffer_bindings: dict[str, BufferHandle], scalar_params: dict[str, int | float], tile_index: int) -> list[Any]:
        return [
            get_buffer(buffer_bindings["logits"]),
            get_buffer(buffer_bindings["temps"]),
            get_buffer(buffer_bindings["targets"]),
            get_buffer(buffer_bindings["sample_mask"]),
            get_buffer(buffer_bindings["partial_probs"]),
            get_buffer(buffer_bindings["partial_loss"]),
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
