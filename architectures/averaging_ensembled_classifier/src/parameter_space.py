# parameter_space.py

"""
The Definitive Abstraction for the Model's Learnable Parameter Space.

(REV 2 - ARCHITECTURALLY RECTIFIED) This module provides the `ParameterSpace`
class, a fundamental architectural primitive that serves as the declarative
manifest for ALL learnable parameters and their associated data buffers.

This version fulfills its mandate as the single source of truth for buffer
memory layouts. The new public method, `get_all_memory_layouts`, consumes the
logical `ModelSpec` and runtime execution parameters (like batch size and the
tiling grid) to produce a complete dictionary of fully-specified `MemoryLayout`
objects.

This revision perfectly decouples the orchestration logic from memory layout
concerns. The `TrainingOrchestrator` now asks the `ParameterSpace` for the layout
plan instead of constructing it itself, thus upholding the system's core
principles of architectural elegance and separation of concerns.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

import numpy as np

# --- Foundational Imports from Sibling Modules ---
from model_spec import ModelSpec
from memory_layout import MemoryLayout, PaddingStrategy, PaddingType
from workload_primitives import TilingScheme  # Required for sizing collection buffers


@dataclass(frozen=True)
class ParameterFlowConfig:
    """A declarative manifest entry for a single parameter's gradient lifecycle."""

    name: str  # Short, unique name, e.g., "shared_weights"
    param_buffer_name: str
    partial_grad_buffer_name: str
    clipped_partial_grad_buffer_name: str
    summed_grad_buffer_name: str
    final_grad_buffer_name: str
    m1_buffer_name: str
    m2_buffer_name: str
    # A flag to indicate if this gradient has a specialized reduction path
    specialized_reduction: bool = False


class ParameterSpace:
    """
    A manifest-like object that defines the entire learnable parameter space
    and the memory layout contracts for all associated buffers.
    """

    def __init__(self, spec: ModelSpec):
        """Builds the parameter space from the base model specification."""
        self._spec = spec
        self._flows: List[ParameterFlowConfig] = self._generate_flows()

    def __iter__(self) -> Iterator[ParameterFlowConfig]:
        """Allows direct iteration over the flow configurations."""
        return iter(self._flows)

    def _generate_flows(self) -> List[ParameterFlowConfig]:
        """Programmatically generates the manifest of all parameter flows."""
        flows = [
            self._create_flow("shared_weights"),
            self._create_flow("shared_biases"),
            self._create_flow("module_weights"),
            self._create_flow("module_biases"),
            self._create_flow("temperatures"),
            self._create_flow("hidden_activations", specialized_reduction=True),
        ]
        return [f for f in flows if f is not None]

    def _create_flow(self, name: str, specialized_reduction: bool = False) -> ParameterFlowConfig:
        """A factory for a single ParameterFlowConfig object."""
        grad_name = "temps" if name == "temperatures" else name
        return ParameterFlowConfig(
            name=name,
            param_buffer_name=name,
            partial_grad_buffer_name=f"partial_grad_{grad_name}",
            clipped_partial_grad_buffer_name=f"clipped_partial_grad_{grad_name}",
            summed_grad_buffer_name=f"summed_grad_{grad_name}",
            final_grad_buffer_name=f"final_grad_{grad_name}",
            m1_buffer_name=f"m1_{name}",
            m2_buffer_name=f"m2_{name}",
            specialized_reduction=specialized_reduction,
        )

    def get_all_memory_layouts(
        self, batch_size: int, grid: TilingScheme, num_batch_chunks: int
    ) -> Dict[str, MemoryLayout]:
        """
        The single source of truth for buffer layouts.

        This method translates the logical model spec and dynamic execution
        parameters into a complete set of physical memory layout plans,
        fulfilling all kernel padding contracts.

        Args:
            batch_size: The number of items in the current training batch.
            grid: The `TilingScheme` defining the module/class tiling strategy.
            num_batch_chunks: The number of chunks for streaming backpropagation.

        Returns:
            A dictionary mapping canonical buffer names to their `MemoryLayout` objects.
        """
        spec = self._spec
        layouts = {}

        # --- Define Padding Strategies from Kernel Contracts ---
        pad_to_simd = PaddingStrategy(type=PaddingType.ELEMENT_COUNT, value=spec.simd_width)
        pad_to_cache = PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=spec.cache_line_bytes)

        # --- Layouts for Learnable Parameters & Their Direct Derivatives ---
        # These layouts are static and based purely on the ModelSpec.
        layouts["shared_weights"] = MemoryLayout((spec.input_dim, spec.hidden_dim)).add_strategy(pad_to_simd)
        layouts["shared_biases"] = MemoryLayout((spec.hidden_dim,)).add_strategy(pad_to_simd)
        layouts["module_weights"] = MemoryLayout((spec.num_modules, spec.hidden_dim, spec.output_classes)).add_strategy(
            pad_to_cache
        )
        layouts["module_biases"] = MemoryLayout((spec.num_modules, spec.output_classes)).add_strategy(pad_to_cache)
        layouts["temperatures"] = MemoryLayout((spec.num_modules,))

        # Derivative buffers (summed/final grads, optimizer state) share the same layout
        # as their parent parameter buffer.
        for flow in self._flows:
            if flow.name in layouts and not flow.specialized_reduction:
                param_layout = layouts[flow.name]
                layouts[flow.summed_grad_buffer_name] = param_layout
                layouts[flow.final_grad_buffer_name] = param_layout
                layouts[flow.m1_buffer_name] = param_layout
                layouts[flow.m2_buffer_name] = param_layout

        # --- Layouts for Batch-Dependent Dataflow Buffers ---
        layouts["input"] = MemoryLayout((batch_size, spec.input_dim)).add_strategy(pad_to_cache)
        layouts["hidden_activations"] = MemoryLayout((batch_size, spec.hidden_dim)).add_strategy(pad_to_cache)
        layouts["hidden_mask"] = layouts["hidden_activations"]  # Identical layout
        layouts["logits"] = MemoryLayout((spec.num_modules, batch_size, spec.output_classes)).add_strategy(pad_to_cache)
        layouts["sample_mask"] = MemoryLayout((batch_size,))
        # For targets, we define both CCE and BCE layouts since they differ
        layouts["targets_cce"] = MemoryLayout((batch_size,))
        layouts["targets_bce"] = MemoryLayout((batch_size, spec.output_classes)).add_strategy(pad_to_cache)

        # --- Layouts for Partial Gradient Collection Buffers ---
        # These are the most complex, depending on the tiling/chunking strategy.
        # We must use the *maximum* chunk size for allocation.
        max_mods_per_tile = (spec.num_modules + grid.num_module_chunks - 1) // grid.num_module_chunks
        max_cls_per_tile = (spec.output_classes + grid.num_class_chunks - 1) // grid.num_class_chunks
        padded_hidden_dim = layouts["shared_biases"].get_padded_shape(spec.scalar_dtype)[0]

        layouts["partial_grad_module_weights"] = MemoryLayout(
            (grid.total_tiles, max_mods_per_tile, padded_hidden_dim, max_cls_per_tile)
        )
        layouts["partial_grad_module_biases"] = MemoryLayout((grid.total_tiles, max_mods_per_tile, max_cls_per_tile))
        layouts["partial_grad_temps"] = MemoryLayout((grid.total_tiles, max_mods_per_tile))
        layouts["partial_grad_hidden_activations"] = MemoryLayout(
            (grid.total_tiles, max_mods_per_tile, batch_size, padded_hidden_dim)
        )
        # Shared grad collection buffers must account for the streaming chunks
        padded_input_dim = layouts["input"].get_padded_shape(spec.scalar_dtype)[1]
        layouts["partial_grad_shared_weights"] = MemoryLayout((num_batch_chunks, padded_input_dim, padded_hidden_dim))
        layouts["partial_grad_shared_biases"] = MemoryLayout((num_batch_chunks, padded_hidden_dim))

        # Clipped buffers share the same layout as their partial counterparts
        for name in list(layouts.keys()):
            if "partial_grad" in name:
                layouts[name.replace("partial", "clipped")] = layouts[name]

        # --- Layouts for Diagnostic Collection Buffers ---
        layouts["partial_probs"] = MemoryLayout((grid.total_tiles, max_mods_per_tile, batch_size, max_cls_per_tile))
        layouts["partial_loss"] = MemoryLayout((grid.total_tiles, max_mods_per_tile, batch_size))
        # The final loss buffer is a matrix of (modules, samples) for CCE.
        # For BCE, it is a single scalar after full reduction. To unify, we will
        # use a simple scalar buffer for the final reduced BCE loss and retrieve
        # the CCE loss from its direct-write buffer when needed.
        layouts["final_loss"] = MemoryLayout((1,))

        # --- Layouts for Specialized Intermediates ---
        layouts["permuted_grad_h"] = MemoryLayout((batch_size * padded_hidden_dim, spec.num_modules)).add_strategy(
            pad_to_cache
        )  # Pad the module dim for row alignment

        return layouts
