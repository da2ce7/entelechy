# src/tests/test_buffer_plumbing.py
"""
Unit tests for the complete buffer-name and shape plumbing chain.

Bug-hunting focus:

These tests verify the CONTRACT between:
  ParameterFlowConfig  →  ParameterSpace.get_all_memory_layouts()
                        →  graph_recipes.py (buffer lookups)
                        →  batch_processor.py (elements_per_partial calculations)

Any mismatch in buffer naming, shape interpretation, or elements_per_partial
between these layers will cause silent data corruption or outright KeyErrors.

Strategy: We build the full layout dictionary for each canonical model, then
simulate the buffer lookups that graph_recipes and batch_processor make,
asserting that the names resolve and the element counts are consistent.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

from src.model_spec import Float32ModelSpec
from src.parameter_space import ParameterSpace, ParameterFlowConfig
from src.memory_layout import MemoryLayout
from src.workload_primitives import (
    TilingScheme,
    TiledGather,
    LinearlyChunkedGather,
    ContiguousGather,
)


def _make_tiling(spec: Float32ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


def _get_layouts(spec: Float32ModelSpec, batch_size: int = 150, num_chunks: int = 4):
    ps = ParameterSpace(spec=spec)
    grid = _make_tiling(spec)
    return ps.get_all_memory_layouts(batch_size=batch_size, grid=grid, num_batch_chunks=num_chunks), grid, ps


# =========================================================================
# 1.  All buffers referenced by graph_recipes must exist in the layout dict.
# =========================================================================


class TestGraphRecipeBufferNames:
    """Every buffer name that graph_recipes.py looks up must be present in the layout."""

    # These are the exact names used by graph_recipes.py  (from grep_search).
    RECIPE_BUFFER_NAMES = [
        # Forward pass (execute_forward_pass / build_forward_module_path)
        "input", "sample_mask", "shared_weights", "shared_biases",
        "hidden_activations", "hidden_mask",
        "module_weights", "module_biases", "logits", "temperatures",
        "partial_probs", "final_probs",
        "final_loss", "partial_loss",
        "targets_cce", "targets_bce",

        # Backward module path (build_backward_module_path)
        "partial_grad_module_weights", "partial_grad_module_biases",
        "partial_grad_temps", "partial_grad_hidden_activations",
        "clipped_partial_grad_module_weights", "clipped_partial_grad_module_biases",
        "clipped_partial_grad_temps", "clipped_partial_grad_hidden_activations",

        # Shared backprop (build_shared_backprop_subgraph)
        "clipped_partial_grad_shared_weights", "clipped_partial_grad_shared_biases",

        # Reduction destinations (batch_processor)
        "summed_grad_hidden_activations", "permuted_grad_h",

        # Clipping threshold (PER_ITEM style)
        "clipping_threshold_per_item",
    ]

    @pytest.mark.parametrize("spec_name", ["iris", "hydra", "lexicon"])
    def test_all_recipe_buffers_exist(self, spec_name, iris_spec, hydra_spec, lexicon_spec) -> None:
        specs = {"iris": iris_spec, "hydra": hydra_spec, "lexicon": lexicon_spec}
        spec = specs[spec_name]
        layouts, _, _ = _get_layouts(spec)

        missing = [name for name in self.RECIPE_BUFFER_NAMES if name not in layouts]
        assert not missing, (
            f"[{spec_name}] Buffer names used by graph_recipes but MISSING from layouts:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


# =========================================================================
# 2.  elements_per_partial calculations in batch_processor must be consistent.
# =========================================================================


class TestElementsPerPartial:
    """
    The batch_processor computes `elements_per_partial = prod(shape[1:])`
    for each flow's clipped buffer, then builds a GatherPrimitive with it.
    This test verifies that calculation is self-consistent.
    """

    def test_module_weights_epp(self, iris_spec: Float32ModelSpec) -> None:
        """elements_per_partial for clipped_partial_grad_module_weights."""
        layouts, grid, ps = _get_layouts(iris_spec)
        layout = layouts["clipped_partial_grad_module_weights"]
        shape = layout.logical_shape
        # shape = (total_tiles, max_mods_per_tile, padded_hidden_dim, max_cls_per_tile)
        assert len(shape) == 4, f"Expected 4D, got {len(shape)}D: {shape}"
        epp = int(np.prod(shape[1:]))
        assert epp > 0
        # Must be consistent with TiledGather
        tg = TiledGather(scheme=grid, _elements_per_partial=epp)
        assert tg.num_partials == grid.total_tiles

    def test_module_biases_epp(self, iris_spec: Float32ModelSpec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec)
        shape = layouts["clipped_partial_grad_module_biases"].logical_shape
        assert len(shape) == 3  # (total_tiles, max_mods_per_tile, max_cls_per_tile)
        epp = int(np.prod(shape[1:]))
        assert epp > 0

    def test_temps_epp(self, iris_spec: Float32ModelSpec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec)
        shape = layouts["clipped_partial_grad_temps"].logical_shape
        assert len(shape) == 2  # (total_tiles, max_mods_per_tile)
        epp = int(np.prod(shape[1:]))
        assert epp > 0

    def test_shared_weights_epp_linear_chunked(self, iris_spec: Float32ModelSpec) -> None:
        """Shared weights use LinearlyChunkedGather with num_batch_chunks."""
        layouts, grid, ps = _get_layouts(iris_spec, num_chunks=4)
        shape = layouts["clipped_partial_grad_shared_weights"].logical_shape
        assert len(shape) == 3  # (num_batch_chunks, input_dim, padded_hidden_dim)
        epp = int(np.prod(shape[1:]))
        assert epp > 0
        g = LinearlyChunkedGather(num_chunks=4, elements_per_chunk=epp)
        assert g.num_partials == 4

    def test_shared_biases_epp_linear_chunked(self, iris_spec: Float32ModelSpec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec, num_chunks=4)
        shape = layouts["clipped_partial_grad_shared_biases"].logical_shape
        assert len(shape) == 2  # (num_batch_chunks, padded_hidden_dim)
        epp = int(np.prod(shape[1:]))
        assert epp > 0


# =========================================================================
# 3.  Adam update loop — the flow.name → summed_grads must be handled.
# =========================================================================


class TestAdamUpdateFlowCoverage:
    """
    In build_update_subgraph, for each flow in param_space (excluding
    specialized), the code looks up:
      - flow.param_buffer_name  → named buffer
      - flow.final_grad_buffer_name  → named buffer
      - flow.m1_buffer_name  → named buffer
      - flow.m2_buffer_name  → named buffer
    All must exist.
    """

    @pytest.mark.parametrize("spec_name", ["iris", "hydra"])
    def test_adam_buffers_exist(self, spec_name, iris_spec, hydra_spec) -> None:
        specs = {"iris": iris_spec, "hydra": hydra_spec}
        spec = specs[spec_name]
        layouts, grid, ps = _get_layouts(spec)

        for flow in ps:
            if flow.specialized_reduction:
                continue
            for attr in ["param_buffer_name", "final_grad_buffer_name",
                         "m1_buffer_name", "m2_buffer_name"]:
                name = getattr(flow, attr)
                assert name in layouts, (
                    f"[{spec_name}] Flow '{flow.name}'.{attr} = '{name}' not in layouts"
                )


# =========================================================================
# 4.  Diagnostic aggregation — final_probs shape must be consistent.
# =========================================================================


class TestDiagnosticAggregation:
    """
    execute_diagnostic_aggregation computes:
      partial_probs_shape = bm.get_spec(partial_probs_ref)
      elements_per_prob_partial = prod(partial_probs_shape[1:])
    then does a TiledGather with that epp.
    The destination is final_probs.  Its total element count must be >= epp.
    """

    def test_final_probs_consistent_with_partial(self, iris_spec: Float32ModelSpec) -> None:
        layouts, grid, _ = _get_layouts(iris_spec)
        partial_shape = layouts["partial_probs"].logical_shape
        final_shape = layouts["final_probs"].logical_shape

        epp = int(np.prod(partial_shape[1:]))
        final_elements = int(np.prod(final_shape))

        assert final_elements >= epp, (
            f"final_probs has {final_elements} elements but "
            f"one partial has {epp} (from partial_probs {partial_shape})"
        )

    def test_final_loss_consistent_with_partial(self, iris_spec: Float32ModelSpec) -> None:
        layouts, grid, _ = _get_layouts(iris_spec)
        partial_shape = layouts["partial_loss"].logical_shape
        final_shape = layouts["final_loss"].logical_shape

        epp = int(np.prod(partial_shape[1:]))
        final_elements = int(np.prod(final_shape))

        assert final_elements >= epp, (
            f"final_loss has {final_elements} elements but "
            f"one partial has {epp} (from partial_loss {partial_shape})"
        )


# =========================================================================
# 5.  Permuted grad_h shape must be consistent with clipped partials.
# =========================================================================


class TestPermutedGradHConsistency:
    """
    GatherAndPermuteGradHiddenActivationsSignature asserts:
      aos_shape[2] == total_batch_count
      soa_shape[0] == total_batch_count * padded_hidden_count
      soa_shape[1] == padded_module_dim
    """

    @pytest.mark.parametrize("spec_name,batch_size", [
        ("iris", 150),
        ("hydra", 32),
    ])
    def test_soa_shape_matches_contract(self, spec_name, batch_size, iris_spec, hydra_spec) -> None:
        specs = {"iris": iris_spec, "hydra": hydra_spec}
        spec = specs[spec_name]
        layouts, grid, _ = _get_layouts(spec, batch_size=batch_size)

        # AoS shape
        aos = layouts["clipped_partial_grad_hidden_activations"].logical_shape
        # SoA shape
        soa = layouts["permuted_grad_h"].logical_shape

        assert aos[2] == batch_size, f"AoS dim2={aos[2]} != batch_size={batch_size}"
        assert soa[0] == batch_size * spec.padded_hidden_dim, (
            f"SoA dim0={soa[0]} != batch*padded_hidden={batch_size * spec.padded_hidden_dim}"
        )
        assert soa[1] == spec.padded_module_dim, (
            f"SoA dim1={soa[1]} != padded_module_dim={spec.padded_module_dim}"
        )
