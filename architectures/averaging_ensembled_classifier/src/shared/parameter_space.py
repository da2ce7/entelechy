# parameter_space.py

"""
A Declarative Manifest for the Model's Learnable Parameter Space.

Design Rationale:
This module provides the `ParameterSpace` class, an architectural entity that
endeavors to be the single source of truth for all learnable parameters and
their associated data buffers within the system. Our design goal here is to
rigorously apply the principle of Separation of Concerns. We aim to decouple the
high-level orchestration logic from the nuanced and often error-prone details
of device memory allocation and layout.

Module's Role in the System:
The `ParameterSpace` acts as a planner. It consumes the abstract, logical
`ModelSpec` and translates it into a complete and concrete set of `MemoryLayout`
blueprints. These blueprints are then passed to the `BufferManager`, which
executes the physical allocation. This version has undergone a deep logical
review to ensure its memory plans faithfully reflect the specific computational
contracts defined in the kernel headers.
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List

# --- Foundational Imports from Sibling Architectural Modules ---
from .model_spec import ModelSpec
from .memory_layout import MemoryLayout
from .workload_primitives import TilingScheme


@dataclass(frozen=True)
class ParameterFlowConfig:
    """
    A simple data contract describing a single parameter's gradient lifecycle.

    Our hope is that by grouping the many string-based buffer names associated
    with a single parameter (e.g., its raw value, its gradients, its optimizer
    state), we can improve the clarity and reduce the potential for errors that
    arise from managing these names independently.
    """

    name: str  # A short, unique name, e.g., "shared_weights"
    param_buffer_name: str
    partial_grad_buffer_name: str
    clipped_partial_grad_buffer_name: str
    summed_grad_buffer_name: str
    final_grad_buffer_name: str
    m1_buffer_name: str
    m2_buffer_name: str
    # A flag to note if this parameter has a non-standard reduction path.
    specialized_reduction: bool = False


class ParameterSpace:
    """
    A manifest-like object that defines the entire learnable parameter space
    and generates the memory layout contracts for all associated buffers.
    """

    def __init__(self, spec: ModelSpec):
        """Initializes the space by composing a list of parameter flows."""
        self._spec = spec
        self._flows: List[ParameterFlowConfig] = self._generate_flows()

    def __iter__(self) -> Iterator[ParameterFlowConfig]:
        """Permits direct iteration over the contained flow configurations."""
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
        return flows

    def _create_flow(self, name: str, specialized_reduction: bool = False) -> ParameterFlowConfig:
        """A factory for generating a consistent set of buffer names."""
        grad_name = "temps" if name == "temperatures" else name

        # WHY: The reasoning here is to handle computational paths that deviate
        # from the standard gradient lifecycle. The 'hidden_activations' flow,
        # for instance, has its `summed_grad` produced by a specialized kernel,
        # so it lacks the usual intermediate partial/clipped/final buffers.
        # This conditional logic ensures we don't declare buffer names for
        # memory that will never be allocated, preventing downstream errors.
        if specialized_reduction:
            return ParameterFlowConfig(
                name=name,
                param_buffer_name=name,
                partial_grad_buffer_name="",
                clipped_partial_grad_buffer_name="",
                summed_grad_buffer_name=f"summed_grad_{name}",
                final_grad_buffer_name="",
                m1_buffer_name="",
                m2_buffer_name="",
                specialized_reduction=specialized_reduction,
            )
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
        The method where the translation from logical specification to a
        concrete physical memory plan occurs.
        """
        spec = self._spec
        layouts: Dict[str, MemoryLayout] = {}

        # --- Layouts for Learnable Parameters & Their Direct Derivatives ---
        # WHY: Our design philosophy is that the `ModelSpec` is the single
        # source of truth for calculating padded dimensions. This section reads
        # those values directly to construct layouts, ensuring consistency.

        # WHY: The forward_pass kernel reads weights in SIMD-major layout as
        # W[h_block, feature] = flat[h_block * padded_input * SIMD + feature * SIMD + lane].
        # For SIMD=1 this is flat[h * padded_input + i], i.e. (hidden, padded_input).
        # The buffer must provide padded_hidden * padded_input elements in this
        # (hidden-major, padded-input) physical layout.
        layouts["shared_weights"] = MemoryLayout((spec.padded_hidden_dim, spec.padded_input_dim))
        layouts["shared_biases"] = MemoryLayout((spec.padded_hidden_dim,))
        layouts["module_weights"] = MemoryLayout(
            (spec.num_modules, spec.padded_hidden_dim, spec.padded_class_dim)
        )
        layouts["module_biases"] = MemoryLayout((spec.num_modules, spec.padded_class_dim))
        layouts["temperatures"] = MemoryLayout((spec.num_modules,))

        # WHY: By mathematical necessity, derivative buffers (like optimizer
        # state) must have a shape identical to the parameters they track. This
        # loop programmatically enforces that fundamental contract.
        for flow in self._flows:
            if flow.name in layouts and not flow.specialized_reduction:
                param_layout = layouts[flow.name]
                layouts[flow.summed_grad_buffer_name] = param_layout
                layouts[flow.final_grad_buffer_name] = param_layout
                layouts[flow.m1_buffer_name] = param_layout
                layouts[flow.m2_buffer_name] = param_layout

        # --- Layouts for Batch-Dependent Dataflow Buffers ---
        layouts["input"] = MemoryLayout((batch_size, spec.padded_input_dim))
        layouts["hidden_activations"] = MemoryLayout((batch_size, spec.padded_hidden_dim))
        layouts["hidden_mask"] = layouts["hidden_activations"]
        layouts["logits"] = MemoryLayout((spec.num_modules, batch_size, spec.padded_class_dim))
        layouts["sample_mask"] = MemoryLayout(((batch_size + 31) // 32,))
        layouts["targets_cce"] = MemoryLayout((batch_size,))
        layouts["targets_bce"] = MemoryLayout((batch_size, spec.padded_class_dim))

        # --- Layouts for Partial Gradient Collection Buffers ---
        # WHY: These buffers are designed to hold the scattered results from many
        # parallel kernel invocations. Their size is therefore a direct function
        # of the tiling and chunking strategy defined in the ExecutionPlan.
        max_mods_per_tile = (spec.num_modules + grid.num_module_chunks - 1) // grid.num_module_chunks
        max_cls_per_tile = (spec.output_classes + grid.num_class_chunks - 1) // grid.num_class_chunks

        layouts["partial_grad_module_weights"] = MemoryLayout(
            (grid.total_tiles, max_mods_per_tile, spec.padded_hidden_dim, spec.padded_class_dim)
        )
        layouts["partial_grad_module_biases"] = MemoryLayout((grid.total_tiles, max_mods_per_tile, spec.padded_class_dim))
        layouts["partial_grad_temps"] = MemoryLayout((grid.total_tiles, max_mods_per_tile))
        layouts["partial_grad_hidden_activations"] = MemoryLayout(
            (grid.total_tiles, max_mods_per_tile, batch_size, spec.padded_hidden_dim)
        )

        # WHY: The backprop_shared_weights_chunk kernel (after the transpose fix)
        # writes gradients as flat[j * padded_input + i], matching the weight
        # buffer's (hidden, padded_input) flat layout. Each chunk produces
        # padded_hidden * padded_input elements.
        layouts["partial_grad_shared_weights"] = MemoryLayout(
            (num_batch_chunks, spec.padded_hidden_dim, spec.padded_input_dim)
        )
        layouts["partial_grad_shared_biases"] = MemoryLayout((num_batch_chunks, spec.padded_hidden_dim))

        # WHY: A clipped buffer is merely a value-transformed version of its
        # partial precursor; it is not a shape-transformed one. This loop
        # enforces the contract that their shapes must be identical.
        for name in list(layouts.keys()):
            if "partial_grad" in name:
                layouts[name.replace("partial_grad", "clipped_partial_grad")] = layouts[name]

        # --- Layouts for Diagnostic & Specialized Buffers ---
        layouts["partial_probs"] = MemoryLayout((grid.total_tiles, max_mods_per_tile, batch_size, max_cls_per_tile))
        layouts["partial_loss"] = MemoryLayout((grid.total_tiles, max_mods_per_tile, batch_size))
        # This buffer is for the direct-write CCE loss; BCE reduction uses a transient buffer.
        layouts["final_loss"] = MemoryLayout((spec.num_modules, batch_size))
        # WHY: `final_probs` is the destination of the tile-wise probability
        # reduction (Node 14). Its shape is one partial's worth: the tile
        # dimension has been summed away, leaving per-module, per-sample,
        # per-class probabilities.
        layouts["final_probs"] = MemoryLayout((max_mods_per_tile, batch_size, max_cls_per_tile))

        # Specialized Intermediates
        # WHY: These layouts are dictated entirely by the kernels that consume them.
        # `summed_grad_hidden` matches the hidden activation shape. `permuted_grad_h`
        # has a Struct-of-Arrays (SoA) layout required for its specialized reduction kernel.
        layouts["summed_grad_hidden_activations"] = MemoryLayout((batch_size, spec.padded_hidden_dim))
        layouts["permuted_grad_h"] = MemoryLayout(
            (batch_size * spec.padded_hidden_dim, spec.padded_module_dim)
        )
        layouts["clipping_threshold_per_item"] = MemoryLayout((grid.total_tiles,))

        return layouts
