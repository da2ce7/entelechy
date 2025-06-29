# parameter_space.py

"""
The Definitive Abstraction for the Model's Learnable Parameter Space.

This module provides the `ParameterSpace` class, a fundamental architectural
primitive that serves as the declarative manifest for ALL learnable parameters
in the model.

It programmatically generates a complete list of `ParameterFlowConfig` objects
based on an input `ModelSpec`. It is also the single source of truth for the
memory shapes of all parameters and their corresponding gradient buffers.

This abstraction completely decouples the definition of the model's learnable
structure from the orchestration logic that trains it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterator

import numpy as np

from model_spec import ModelSpec


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
    of the model.
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
        # This pattern avoids long, repetitive dataclass instantiations.
        if name == "temperatures":  # A special case with a different grad name
            grad_name = "temps"
        else:
            grad_name = name

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

    def get_param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Returns a dictionary of shapes for all primary parameter buffers."""
        spec = self._spec
        return {
            "shared_weights": (spec.input_dim, spec.padded_hidden_dim),
            "shared_biases": (spec.padded_hidden_dim,),
            "module_weights": (spec.num_modules, spec.padded_hidden_dim, spec.padded_class_dim),
            "module_biases": (spec.num_modules, spec.padded_class_dim),
            "temperatures": (spec.num_modules,),
            "hidden_activations": (1,),  # Not a real parameter, shape is batch-dependent
        }

    def get_partial_grad_shapes(
        self, batch_size: int, total_tiles: int, modules_per_chunk: int, classes_per_chunk: int
    ) -> Dict[str, Tuple[int, ...]]:
        """Returns shapes for the large 'collection' buffers of partial gradients."""
        spec = self._spec
        return {
            "partial_grad_shared_weights": (
                batch_size,
                spec.input_dim,
                spec.padded_hidden_dim,
            ),  # Placeholder for streaming
            "partial_grad_shared_biases": (batch_size, spec.padded_hidden_dim),  # Placeholder for streaming
            "partial_grad_module_weights": (total_tiles, modules_per_chunk, spec.padded_hidden_dim, classes_per_chunk),
            "partial_grad_module_biases": (total_tiles, modules_per_chunk, classes_per_chunk),
            "partial_grad_temps": (total_tiles, modules_per_chunk),
            "partial_grad_hidden_activations": (total_tiles, modules_per_chunk, batch_size, spec.padded_hidden_dim),
        }
