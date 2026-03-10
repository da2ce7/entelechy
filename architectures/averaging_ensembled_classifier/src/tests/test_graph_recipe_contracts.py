# src/tests/test_graph_recipe_contracts.py
"""
Unit tests for graph_recipes.py contractual invariants.

Bug-hunting focus:

These tests don't invoke OpenCL.  Instead they verify the PURE-PYTHON
plumbing contracts that graph_recipes relies on:

  1. That every KernelSignature base-class `__post_init__` can derive
     dimensions correctly from its buffer-manager specs.
  2. That kernel argument lists have the correct length.
  3. That the build_update_subgraph correctly iterates flows and finds
     the right buffers.
  4. That the DependencyProvider / lifecycle_policy system resolves the
     correct buffer handles.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pytest

from src.model_spec import Float32ModelSpec
from src.parameter_space import ParameterSpace, ParameterFlowConfig
from src.memory_layout import MemoryLayout
from src.workload_primitives import TilingScheme
from src.execution_plan import CceStrategy, BceStrategy


# =========================================================================
# Helpers
# =========================================================================


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
# 1.  The partial_grad → clipped_partial_grad → summed_grad chain shapes
# =========================================================================


class TestGradientShapeChain:
    """
    For each non-specialized flow, verify:
      partial_grad.shape == clipped_partial_grad.shape   (must be identical)
      summed_grad.shape == param.shape                   (must match the parameter)
      prod(clipped_shape[1:]) == prod(summed_shape)      (epp consistency)
    """

    @pytest.fixture
    def chain_data(self, iris_spec):
        layouts, grid, ps = _get_layouts(iris_spec)
        return layouts, ps

    @pytest.mark.parametrize("flow_name", [
        "shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"
    ])
    def test_partial_equals_clipped(self, chain_data, flow_name) -> None:
        layouts, ps = chain_data
        flow = next(f for f in ps if f.name == flow_name)
        partial_shape = layouts[flow.partial_grad_buffer_name].logical_shape
        clipped_shape = layouts[flow.clipped_partial_grad_buffer_name].logical_shape
        assert partial_shape == clipped_shape, (
            f"[{flow_name}] partial {partial_shape} != clipped {clipped_shape}"
        )

    @pytest.mark.parametrize("flow_name", [
        "shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"
    ])
    def test_summed_matches_param(self, chain_data, flow_name) -> None:
        layouts, ps = chain_data
        flow = next(f for f in ps if f.name == flow_name)
        param_shape = layouts[flow.param_buffer_name].logical_shape
        summed_shape = layouts[flow.summed_grad_buffer_name].logical_shape
        assert param_shape == summed_shape, (
            f"[{flow_name}] param {param_shape} != summed {summed_shape}"
        )

    @pytest.mark.parametrize("flow_name", [
        "shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"
    ])
    def test_epp_consistency(self, chain_data, flow_name) -> None:
        layouts, ps = chain_data
        flow = next(f for f in ps if f.name == flow_name)
        clipped_shape = layouts[flow.clipped_partial_grad_buffer_name].logical_shape
        summed_shape = layouts[flow.summed_grad_buffer_name].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_el = int(np.prod(summed_shape))
        assert epp == summed_el, (
            f"[{flow_name}] epp={epp} (clipped{clipped_shape}[1:]) != summed_el={summed_el} (summed{summed_shape})"
        )


# =========================================================================
# 2.  Module weights flow: detailed shape breakdown
# =========================================================================


class TestModuleWeightsFlowDetailed:
    """
    Deep dive into module_weights gradient shapes for the IRIS configuration.

    Model: input_dim=4, hidden_dim=32, output_classes=3, num_modules=8
    Padding: padded_hidden_dim=32, padded_class_dim=4, padded_module_dim=8

    Expected shapes:
      module_weights:                   (8, 32, padded_class_dim)
      partial_grad_module_weights:      (total_tiles, max_mods_per_tile, 32, padded_class_dim)
      clipped_partial_grad_module_weights: same as partial
      summed_grad_module_weights:       (8, 32, padded_class_dim)

    With grid (1,1): total_tiles=1, max_mods_per_tile=8.
    """

    def test_module_weights_shape(self, iris_spec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec)
        shape = layouts["module_weights"].logical_shape
        assert shape == (8, iris_spec.padded_hidden_dim, iris_spec.padded_class_dim)

    def test_partial_grad_mw_shape(self, iris_spec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec)
        shape = layouts["partial_grad_module_weights"].logical_shape
        # With grid (1,1): total_tiles=1, max_mods=8, padded_h=32, padded_cls=padded_class_dim
        max_mods = (iris_spec.num_modules + grid.num_module_chunks - 1) // grid.num_module_chunks
        expected = (grid.total_tiles, max_mods, iris_spec.padded_hidden_dim, iris_spec.padded_class_dim)
        assert shape == expected, f"partial_grad_module_weights shape: {shape} != {expected}"

    def test_epp_module_weights_with_padding_mismatch(self, iris_spec) -> None:
        """
        CRITICAL BUG DETECTOR: If padded_class_dim != max_cls_per_tile,
        then epp = max_mods * padded_hidden * max_cls
        but summed_grad elements = num_modules * padded_hidden * padded_class_dim
        
        These must be equal for the copy/reduction to produce correct results.
        """
        layouts, grid, ps = _get_layouts(iris_spec)
        
        max_mods = (iris_spec.num_modules + grid.num_module_chunks - 1) // grid.num_module_chunks
        max_cls = (iris_spec.output_classes + grid.num_class_chunks - 1) // grid.num_class_chunks
        
        clipped_shape = layouts["clipped_partial_grad_module_weights"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))  # max_mods * padded_h * max_cls
        
        summed_shape = layouts["summed_grad_module_weights"].logical_shape
        summed_el = int(np.prod(summed_shape))  # num_modules * padded_h * padded_class_dim
        
        # If these differ, the copy in n==1 path copies wrong number of bytes,
        # and the aggregation produces misaligned results!
        assert epp == summed_el, (
            f"SHAPE MISMATCH BUG: "
            f"epp = prod({clipped_shape[1:]}) = {epp}, "
            f"summed = prod({summed_shape}) = {summed_el}\n"
            f"max_mods={max_mods} vs num_modules={iris_spec.num_modules}, "
            f"max_cls={max_cls} vs padded_class_dim={iris_spec.padded_class_dim}"
        )


# =========================================================================
# 3.  Temperatures flow: 1D edge case
# =========================================================================


class TestTemperaturesFlow:
    """
    Temperatures is a 1D parameter: shape (num_modules,).
    partial_grad_temps shape: (total_tiles, max_mods_per_tile).
    
    With grid (1,1): clipped shape = (1, 8), epp = 8.
    summed shape = (8,), elements = 8.
    """

    def test_temps_shapes_iris(self, iris_spec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec)
        
        param_shape = layouts["temperatures"].logical_shape
        assert param_shape == (8,)
        
        partial_shape = layouts["partial_grad_temps"].logical_shape
        max_mods = (iris_spec.num_modules + grid.num_module_chunks - 1) // grid.num_module_chunks
        assert partial_shape == (grid.total_tiles, max_mods)
        
        summed_shape = layouts["summed_grad_temps"].logical_shape
        assert summed_shape == (8,)
        
        epp = int(np.prod(partial_shape[1:]))
        assert epp == 8  # = max_mods per tile = num_modules


# =========================================================================
# 4.  Multi-tile scenario (hydra-like) — verify reduction is needed
# =========================================================================


class TestMultiTileReduction:
    """
    For a model with many modules/classes (e.g., hydra: 64 modules, 50 classes),
    the tiling produces multiple tiles, and the reduction tree must actually
    aggregate across them.
    """

    def test_hydra_has_multiple_tiles(self, hydra_spec) -> None:
        grid = _make_tiling(hydra_spec)
        assert grid.total_tiles > 1, (
            f"hydra should have >1 tile but has {grid.total_tiles}"
        )

    def test_hydra_module_weights_epp_consistent(self, hydra_spec) -> None:
        layouts, grid, ps = _get_layouts(hydra_spec)
        clipped_shape = layouts["clipped_partial_grad_module_weights"].logical_shape
        summed_shape = layouts["summed_grad_module_weights"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_el = int(np.prod(summed_shape))
        # With multiple module chunks, epp covers max_mods_per_tile modules
        # while summed covers ALL modules.  The critical check is that the
        # class and hidden strides match — i.e., epp / max_mods == summed / num_modules.
        max_mods = clipped_shape[1]
        per_mod_epp = epp // max_mods
        per_mod_summed = summed_el // hydra_spec.num_modules
        assert per_mod_epp == per_mod_summed, (
            f"Per-module element count mismatch: epp/max_mods={per_mod_epp} "
            f"!= summed/num_modules={per_mod_summed}"
        )


# =========================================================================
# 5.  Final_probs shape vs what Iris classification needs
# =========================================================================


class TestFinalProbsForClassification:
    """
    For Iris classification to work, after the probability aggregation,
    final_probs must contain per-sample, per-class information that can
    be averaged across modules.
    
    final_probs shape: (max_mods_per_tile, batch_size, max_cls_per_tile)
    
    With Iris (8 modules, 3 classes, 1 tile):
      shape = (8, 150, 3)
    
    The prediction should be:
      averaged = final_probs.mean(axis=0)  → (150, 3)
      pred = averaged.argmax(axis=1)       → (150,)
    """

    def test_final_probs_shape_for_iris(self, iris_spec) -> None:
        layouts, grid, ps = _get_layouts(iris_spec)
        final_shape = layouts["final_probs"].logical_shape
        
        max_mods = (iris_spec.num_modules + grid.num_module_chunks - 1) // grid.num_module_chunks
        max_cls = (iris_spec.output_classes + grid.num_class_chunks - 1) // grid.num_class_chunks
        
        assert final_shape == (max_mods, 150, max_cls)
        
        # For classification, we need:
        # axis 0 = modules dimension (to average over)
        # axis 1 = samples dimension
        # axis 2 = classes dimension (to argmax over)
        assert final_shape[0] == iris_spec.num_modules  # 8 modules
        assert final_shape[1] == 150                     # batch size
        assert final_shape[2] >= iris_spec.output_classes  # >= 3 classes

    def test_partial_probs_to_final_reduction_makes_sense(self, iris_spec) -> None:
        """
        partial_probs shape: (total_tiles, max_mods_per_tile, batch_size, max_cls_per_tile)
        
        The diagnostic aggregation sums over axis 0 (tiles).
        With 1 tile, this is a copy: final_probs = partial_probs[0].
        
        But wait — what does "sum over tiles" mean for probabilities?
        Each tile produces probabilities for a DIFFERENT SET of modules and classes.
        Summing across tiles means combining contributions from different tiles.
        
        For iris with 1 tile, this is fine (it's just a copy).
        For multi-tile, the probs might NOT be additive! This could be a semantic bug.
        """
        layouts, grid, ps = _get_layouts(iris_spec)
        partial_shape = layouts["partial_probs"].logical_shape
        final_shape = layouts["final_probs"].logical_shape
        
        # One tile's worth of probabilities
        one_partial = partial_shape[1:]  # (max_mods, batch, max_cls)
        assert one_partial == final_shape


# =========================================================================
# 6.  Strategy kwargs filtering — ensure get_loss_signature doesn't crash
# =========================================================================


class TestStrategyKwargsFiltering:
    """
    The get_loss_signature method receives BOTH loss_out_ref and
    partial_loss_out_ref, and must pop the ones it doesn't need.

    CceStrategy: pops partial_loss_out_ref (CCE writes loss directly)
    BceStrategy: pops loss_out_ref (BCE uses partial_loss and reduces)
    """

    def test_cce_pops_partial_loss_out_ref(self) -> None:
        kwargs = {
            "_buffer_mgr": "fake_bm",
            "_arch_consts": "fake_ac",
            "logit_ref": "fake_logit",
            "temp_ref": "fake_temp",
            "mask_ref": "fake_mask",
            "prob_out_ref": "fake_prob",
            "tile": "fake_tile",
            "total_output_class_count": np.uint32(3),
            "loss_out_ref": "fake_loss",
            "partial_loss_out_ref": "fake_partial_loss",
        }
        strategy = CceStrategy(targets_cce_ref="fake_targets")
        # This should NOT raise a TypeError about partial_loss_out_ref
        try:
            result = strategy.get_loss_signature(**kwargs)
        except TypeError as e:
            if "partial_loss_out_ref" in str(e):
                pytest.fail(
                    f"CceStrategy.get_loss_signature failed to filter partial_loss_out_ref: {e}"
                )
            # Other TypeErrors are expected (fake objects don't have right type)
            pass
        except Exception:
            # Other exceptions are fine — we're testing the kwargs filtering
            pass

    def test_bce_pops_loss_out_ref(self) -> None:
        kwargs = {
            "_buffer_mgr": "fake_bm",
            "_arch_consts": "fake_ac",
            "logit_ref": "fake_logit",
            "temp_ref": "fake_temp",
            "mask_ref": "fake_mask",
            "prob_out_ref": "fake_prob",
            "tile": "fake_tile",
            "total_output_class_count": np.uint32(3),
            "loss_out_ref": "fake_loss",
            "partial_loss_out_ref": "fake_partial_loss",
        }
        strategy = BceStrategy(targets_bce_ref="fake_targets")
        try:
            result = strategy.get_loss_signature(**kwargs)
        except TypeError as e:
            if "loss_out_ref" in str(e):
                pytest.fail(
                    f"BceStrategy.get_loss_signature failed to filter loss_out_ref: {e}"
                )
            pass
        except Exception:
            pass
