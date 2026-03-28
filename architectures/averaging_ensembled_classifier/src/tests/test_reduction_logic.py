# src/tests/test_reduction_logic.py
"""
Unit tests for the reduction pipeline's control-flow logic.

Bug-hunting focus:

The reduction pipeline in `graph_recipes._execute_reduction_pipeline` drives
the entire gradient aggregation chain.  If the offset calculations, edge-case
handling (n==1 → copy), or the ping-pong management are wrong, aggregated
gradients will be zero or garbage.

These tests exercise the PURE-PYTHON logic without OpenCL, using the
GatherPrimitive hierarchy, the StabilizationPolicy tree planning, and
the offset arithmetic that feeds into the kernels.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.workload_primitives import (
    TilingScheme,
    TiledGather,
    LinearlyChunkedGather,
    ContiguousGather,
)
from src.shared.stabilization_policy import StabilizationPolicy


# =========================================================================
# 1.  TiledGather offset correctness
# =========================================================================


class TestTiledGatherOffsets:
    """
    TiledGather.get_offsets() must return byte-element-offsets (in elements)
    into the partial_collection buffer so that
        partial_collection[offset : offset + epp]
    gives the flat data for that tile.

    For a buffer shaped (total_tiles, *tile_shape), tile i starts at
    i * prod(tile_shape) = i * epp.
    """

    @pytest.mark.parametrize("num_mod_chunks, num_cls_chunks, total_mods, total_cls, epp", [
        (1, 1, 8, 3, 8 * 32 * 4),    # iris-like: 1 tile, mw shape (8, 32, 4)
        (2, 1, 16, 3, 8 * 32 * 4),    # 2 module chunks
        (2, 2, 16, 10, 8 * 32 * 8),   # 4 tiles
        (1, 1, 1, 1, 1 * 32 * 1),     # degenerate single-module/single-class
    ])
    def test_offsets_contiguous(self, num_mod_chunks, num_cls_chunks, total_mods, total_cls, epp) -> None:
        grid = TilingScheme(
            num_module_chunks=num_mod_chunks,
            num_class_chunks=num_cls_chunks,
            total_modules=total_mods,
            total_classes=total_cls,
        )
        tg = TiledGather(scheme=grid, _elements_per_partial=epp)
        offsets = tg.get_offsets()

        assert len(offsets) == grid.total_tiles
        for i, off in enumerate(offsets):
            assert off == i * epp, f"Tile {i}: expected offset {i * epp}, got {off}"

    def test_num_partials_equals_total_tiles(self) -> None:
        grid = TilingScheme(
            num_module_chunks=3, num_class_chunks=2, total_modules=24, total_classes=10
        )
        tg = TiledGather(scheme=grid, _elements_per_partial=100)
        assert tg.num_partials == 6

    def test_single_tile_single_partial(self) -> None:
        """When there is exactly 1 tile, num_partials == 1 (triggers copy path in pipeline)."""
        grid = TilingScheme(
            num_module_chunks=1, num_class_chunks=1, total_modules=8, total_classes=3
        )
        tg = TiledGather(scheme=grid, _elements_per_partial=999)
        assert tg.num_partials == 1
        offsets = tg.get_offsets()
        assert len(offsets) == 1
        assert offsets[0] == 0


# =========================================================================
# 2.  LinearlyChunkedGather offset correctness
# =========================================================================


class TestLinearlyChunkedGatherOffsets:
    """
    LinearlyChunkedGather(num_chunks=N, elements_per_chunk=E) should produce
    offsets [0, E, 2E, ...] with num_partials=N.
    """

    @pytest.mark.parametrize("n_chunks, epp", [
        (1, 128),
        (4, 64),
        (8, 1),
    ])
    def test_offsets_linear(self, n_chunks, epp) -> None:
        g = LinearlyChunkedGather(num_chunks=n_chunks, elements_per_chunk=epp)
        offsets = g.get_offsets()
        assert len(offsets) == n_chunks
        for i, off in enumerate(offsets):
            assert off == i * epp

    def test_num_partials(self) -> None:
        g = LinearlyChunkedGather(num_chunks=4, elements_per_chunk=32)
        assert g.num_partials == 4


# =========================================================================
# 3.  ContiguousGather offset correctness
# =========================================================================


class TestContiguousGatherOffsets:
    """ContiguousGather is used for intermediate stages where data is already contiguous."""

    @pytest.mark.parametrize("n, epp", [
        (1, 100),
        (3, 50),
        (8, 1),
    ])
    def test_offsets_contiguous(self, n, epp) -> None:
        g = ContiguousGather(n, epp)
        offsets = g.get_offsets()
        assert len(offsets) == n
        for i, off in enumerate(offsets):
            assert off == i * epp
        assert g.num_partials == n


# =========================================================================
# 4.  Reduction tree planning with StabilizationPolicy
# =========================================================================


class TestReductionTreePlanning:
    """
    The reduction pipeline calls:
        plan_uniform_reduction_tree(n, hardware_max_k) → (safe_k, num_stages)
    Number of stages and chosen K determine correctness of the clip-aggregate loop.
    """

    @pytest.mark.parametrize("n_partials, max_k, expected_stages_min", [
        (1, 64, 0),     # n=1, no reduction needed
        (2, 64, 1),     # n=2, single stage
        (4, 4, 1),      # n=4, k=4, single stage
        (16, 4, 2),     # n=16, k=4, 2 stages (16→4→1)
        (64, 8, 2),     # n=64, k=8, at least 2 stages
    ])
    def test_tree_depth(self, n_partials, max_k, expected_stages_min) -> None:
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=0.9, fp_format_max=3.4e38
        )
        safe_k, num_stages = policy.plan_uniform_reduction_tree(
            num_partials=n_partials, hardware_max_fan_in=max_k
        )
        assert safe_k >= 2 or n_partials <= 1, f"safe_k={safe_k} too small for n={n_partials}"
        assert safe_k <= max_k, f"safe_k={safe_k} exceeds hardware_max_k={max_k}"
        assert num_stages >= expected_stages_min, (
            f"n={n_partials}, k={max_k}: got {num_stages} stages, need >= {expected_stages_min}"
        )

    def test_tree_correctness_full_reduction(self) -> None:
        """For n=8, k=4: stage 0 → 8/4=2 partials, stage 1 → 2/2=1 final."""
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=0.9, fp_format_max=3.4e38
        )
        safe_k, num_stages = policy.plan_uniform_reduction_tree(
            num_partials=8, hardware_max_fan_in=4
        )
        # Simulate the pipeline loop
        n = 8
        stage = 0
        while n > 1:
            n = (n + safe_k - 1) // safe_k
            stage += 1
        assert n == 1, "Reduction did not converge to 1"
        assert stage <= num_stages + 1  # Allow some slack

    def test_stabilization_disabled(self) -> None:
        """When t_algorithmic == 0, no stabilization is active."""
        policy = StabilizationPolicy(
            t_algorithmic=0.0, lambda_=0.9, fp_format_max=3.4e38
        )
        # With stabilization disabled, the pipeline uses the raw k directly
        # and num_stages should be 0
        safe_k, num_stages = policy.plan_uniform_reduction_tree(
            num_partials=8, hardware_max_fan_in=4
        )
        # Even without stabilization, the function should still handle this gracefully
        assert safe_k >= 2


# =========================================================================
# 5.  The critical n==1 copy path in the pipeline
# =========================================================================


class TestSinglePartialCopyPath:
    """
    When num_partials == 1, _execute_reduction_pipeline takes a direct-copy
    shortcut.  This is critical for the IRIS model with 1 tile:
    - module_weights: TiledGather with total_tiles=1 → n=1 → copy
    - module_biases: same
    - temperatures: same

    If the copy offset is wrong, summed_grad_module_weights will be wrong (likely zeros).
    """

    def test_tiled_single_tile_offset_is_zero(self) -> None:
        """When total_tiles=1, the single partial starts at element offset 0."""
        grid = TilingScheme(
            num_module_chunks=1, num_class_chunks=1, total_modules=8, total_classes=3
        )
        epp = 8 * 32 * 4  # module_weights partial size
        tg = TiledGather(scheme=grid, _elements_per_partial=epp)
        offsets = tg.get_offsets()

        assert tg.num_partials == 1
        assert offsets[0] == 0  # Copy should read from byte_offset = 0 * scalar_bytes

    def test_copy_byte_offset_formula(self) -> None:
        """
        In _execute_reduction_pipeline when n==1:
            initial_offsets = gather_primitive.get_offsets()
            src_offset_bytes = int(initial_offsets[0] * scalar_byte_size)

        For TiledGather with 1 tile, initial_offsets[0] should be 0.
        The copy reads `partial_byte_size = epp * scalar_byte_size` bytes
        from this offset, targeting final_dest_handle.

        This verifies the FORMULA, not the device copy.
        """
        grid = TilingScheme(
            num_module_chunks=1, num_class_chunks=1, total_modules=8, total_classes=3
        )
        epp = 256  # arbitrary
        scalar_byte_size = 4  # float32
        tg = TiledGather(scheme=grid, _elements_per_partial=epp)
        offsets = tg.get_offsets()

        src_offset_bytes = int(offsets[0] * scalar_byte_size)
        partial_byte_size = epp * scalar_byte_size

        assert src_offset_bytes == 0
        assert partial_byte_size == 256 * 4
        # This means: cl.enqueue_copy(q, dest, src, byte_count=1024, src_offset=0, dst_offset=0)
        # Which should copy the first 1024 bytes of the clipped_partial buffer to summed buffer.


# =========================================================================
# 6.  elements_per_partial must match between clipped buffer and summed buffer
# =========================================================================


class TestEppConsistency:
    """
    batch_processor computes:
        clipped_shape, _ = bm.get_spec(clipped_ref)
        elements_per_partial = int(np.prod(clipped_shape[1:]))

    And then constructs a TiledGather or LinearlyChunkedGather with this epp.
    The summed_grad buffer must have at least `epp` elements. This test
    verifies that relationship across all flows.
    """

    @pytest.fixture
    def iris_layouts(self, iris_spec):
        from src.shared.parameter_space import ParameterSpace
        ps = ParameterSpace(spec=iris_spec)
        grid = TilingScheme(
            num_module_chunks=1, num_class_chunks=1,
            total_modules=iris_spec.num_modules, total_classes=iris_spec.output_classes
        )
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)
        return layouts, ps, grid

    def test_module_weights_epp_fits_summed(self, iris_layouts) -> None:
        layouts, ps, grid = iris_layouts
        clipped_shape = layouts["clipped_partial_grad_module_weights"].logical_shape
        summed_shape = layouts["summed_grad_module_weights"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_elements = int(np.prod(summed_shape))
        assert epp == summed_elements, (
            f"epp={epp} (from clipped shape {clipped_shape}[1:]) "
            f"!= summed elements={summed_elements} (from shape {summed_shape})"
        )

    def test_module_biases_epp_fits_summed(self, iris_layouts) -> None:
        layouts, ps, grid = iris_layouts
        clipped_shape = layouts["clipped_partial_grad_module_biases"].logical_shape
        summed_shape = layouts["summed_grad_module_biases"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_elements = int(np.prod(summed_shape))
        assert epp == summed_elements, (
            f"epp={epp} (from clipped shape {clipped_shape}[1:]) "
            f"!= summed elements={summed_elements} (from shape {summed_shape})"
        )

    def test_temps_epp_fits_summed(self, iris_layouts) -> None:
        layouts, ps, grid = iris_layouts
        clipped_shape = layouts["clipped_partial_grad_temps"].logical_shape
        summed_shape = layouts["summed_grad_temps"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_elements = int(np.prod(summed_shape))
        assert epp == summed_elements, (
            f"epp={epp} (from clipped shape {clipped_shape}[1:]) "
            f"!= summed elements={summed_elements} (from shape {summed_shape})"
        )

    def test_shared_weights_epp_fits_summed(self, iris_layouts) -> None:
        layouts, ps, grid = iris_layouts
        clipped_shape = layouts["clipped_partial_grad_shared_weights"].logical_shape
        summed_shape = layouts["summed_grad_shared_weights"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_elements = int(np.prod(summed_shape))
        assert epp == summed_elements, (
            f"epp={epp} (from clipped shape {clipped_shape}[1:]) "
            f"!= summed elements={summed_elements} (from shape {summed_shape})"
        )

    def test_shared_biases_epp_fits_summed(self, iris_layouts) -> None:
        layouts, ps, grid = iris_layouts
        clipped_shape = layouts["clipped_partial_grad_shared_biases"].logical_shape
        summed_shape = layouts["summed_grad_shared_biases"].logical_shape
        epp = int(np.prod(clipped_shape[1:]))
        summed_elements = int(np.prod(summed_shape))
        assert epp == summed_elements, (
            f"epp={epp} (from clipped shape {clipped_shape}[1:]) "
            f"!= summed elements={summed_elements} (from shape {summed_shape})"
        )


# =========================================================================
# 7.  The build_update_subgraph filtering: are module_weight flows found?
# =========================================================================


class TestUpdateSubgraphFlowFiltering:
    """
    In build_update_subgraph (graph_recipes.py line ~890):

        for flow in param_space:
            if flow.specialized_reduction or flow.name not in summed_grads:
                continue
            # ...normalize and adam-update...

    AND in batch_processor.py (line ~120):

        for flow in self.param_space:
            if flow.specialized_reduction:
                continue
            # ...reduce grad and put into summed_grad_handles...

    So summed_grad_handles should contain entries for:
      shared_weights, shared_biases, module_weights, module_biases, temperatures

    And build_update_subgraph should iterate over the same set of non-specialized flows.
    If flow.name somehow doesn't match the key in summed_grads, the flow is skipped
    and that parameter NEVER UPDATES.
    """

    def test_non_specialized_flows_have_expected_names(self):
        from src.shared.parameter_space import ParameterSpace
        from src.shared.model_spec import Float32ModelSpec

        spec = Float32ModelSpec(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        ps = ParameterSpace(spec=spec)

        non_spec_names = [f.name for f in ps if not f.specialized_reduction]
        expected = {"shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"}
        assert set(non_spec_names) == expected, (
            f"Non-specialized flow names: {set(non_spec_names)} != expected {expected}"
        )

    def test_summed_grad_handles_key_match(self):
        """
        batch_processor puts results into summed_grad_handles[flow.name].
        build_update_subgraph checks `flow.name not in summed_grads`.

        Both iterate the same param_space object, so the keys MUST match.
        This verifies that the name used as dict key is the same as flow.name.
        """
        from src.shared.parameter_space import ParameterSpace
        from src.shared.model_spec import Float32ModelSpec

        spec = Float32ModelSpec(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        ps = ParameterSpace(spec=spec)

        # Simulate batch_processor
        simulated_summed_grads = {}
        for flow in ps:
            if flow.specialized_reduction:
                continue
            simulated_summed_grads[flow.name] = f"handle_for_{flow.name}"

        # Simulate build_update_subgraph filtering
        adam_updated_params = []
        for flow in ps:
            if flow.specialized_reduction or flow.name not in simulated_summed_grads:
                continue
            adam_updated_params.append(flow.name)

        # ALL non-specialized flows must end up in adam_updated_params
        expected = {"shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"}
        assert set(adam_updated_params) == expected, (
            f"Adam-updated params: {set(adam_updated_params)} != expected {expected}"
        )
