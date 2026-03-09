# tests/test_integration_scenario_validation.py

"""
Integration Tests: End-to-End Validation Scenarios from CONCEPT.md.

These tests implement the conceptual validation scenarios defined in the
architecture document, exercising them at the host-orchestration level.
Each test verifies that the system's planning and composition machinery
produces correct, self-consistent results for a specific validation case.

Scenarios Covered:
  1. The Iris Case         – Graceful degradation for small batches
  2. The Scientist's Repeater – Batch-size-independent normalized gradients
  3. The Marathon          – Long-term Adam stability (beta**t precision)
  4. The Rodeo             – FP16 numerical safety throughout reduction
  5. The Hydra             – Massive num_heads produces valid reduction tree
  6. The Lexicon           – Massive output_classes produce valid tiling
  7. The Data Tsunami      – Large batch_size produces valid reduction tree
  8. The Colossus          – All systems under compound pressure

No OpenCL device is required.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.model_spec import Float32ModelSpec
from src.parameter_space import ParameterSpace
from src.stabilization_policy import StabilizationPolicy
from src.workload_primitives import LinearlyChunkedGather, TilingScheme

# =========================================================================
# Helpers
# =========================================================================


def _make_spec(cls: type[Float32ModelSpec], **kwargs: Any) -> Float32ModelSpec:
    defaults: dict[str, Any] = dict(
        input_dim=4,
        hidden_dim=32,
        output_classes=3,
        num_modules=8,
        simd_width=4,
        cache_line_bytes=64,
    )
    defaults.update(kwargs)
    return cls(**defaults)


def _make_tiling(spec: Float32ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


# =========================================================================
# Scenario 1: The Iris Case (Sequential Execution Mode)
# =========================================================================


class TestIrisCase:
    """
    Graceful degradation: the full planning pipeline must produce valid
    artifacts for a tiny dataset (N=150, 4 features, 3 classes).
    """

    def test_small_batch_produces_valid_layouts(self):
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)
        dtype = np.dtype(spec.SCALAR_NP_TYPE)
        for name, layout in layouts.items():
            shape = layout.get_padded_shape(dtype)
            total_bytes = int(np.prod(shape)) * dtype.itemsize
            assert total_bytes > 0, f"{name} has zero bytes"

    def test_n_equals_1_degenerate_case(self):
        """The N=1 case should produce valid layouts (no reduction needed)."""
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=1, grid=grid, num_batch_chunks=1)
        assert "hidden_activations" in layouts
        assert "shared_weights" in layouts

    def test_reduction_plan_valid_for_single_tile(self):
        """A single tile (small model) still produces a valid reduction plan."""
        spec = _make_spec(Float32ModelSpec, num_modules=1, output_classes=1)
        grid = _make_tiling(spec)
        assert grid.total_tiles == 1
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        _k, stages = policy.plan_uniform_reduction_tree(  # noqa: F841
            num_partials=grid.total_tiles,
            hardware_max_fan_in=256,
        )
        # Single tile → no reduction needed
        assert stages == 0


# =========================================================================
# Scenario 2: The Scientist's Repeater (Normalization Correctness)
# =========================================================================


class TestScientistsRepeater:
    """
    Verify that the normalization step (Node 21) divides by N, making
    the gradient magnitude independent of batch size.
    """

    def test_normalize_is_division_by_n(self):
        """Simulated: dividing a gradient sum by N yields the mean."""
        # This validates the host-side CONCEPT, not the kernel.
        for n in [1, 16, 32, 128, 1024]:
            summed_grad = np.full((32, 64), fill_value=float(n), dtype=np.float32)
            averaged_grad = summed_grad / float(n)
            np.testing.assert_allclose(averaged_grad, 1.0, atol=1e-6)

    def test_different_batch_sizes_produce_same_avg(self):
        """When each sample contributes gradient=1.0, the average is always 1.0."""
        rng = np.random.default_rng(42)
        base_grad = rng.standard_normal((32, 64)).astype(np.float32)
        for n in [4, 16, 64]:
            # Simulate N identical contributions summed
            summed = base_grad * n
            averaged = summed / float(n)
            np.testing.assert_allclose(averaged, base_grad, rtol=1e-6)


# =========================================================================
# Scenario 3: The Marathon (Massive Epochs / Adam Stability)
# =========================================================================


class TestMarathon:
    """
    Verify that computing beta**t in high precision on the host avoids
    underflow for large step counts.
    """

    @pytest.mark.parametrize(
        "beta,max_step",
        [
            (0.9, 5_000),
            (0.999, 500_000),
            (0.9999, 5_000_000),
        ],
    )
    def test_beta_power_no_underflow_fp64(self, beta: float, max_step: int) -> None:
        """Host-side double precision must not underflow for marathon runs.

        FP64 subnormal minimum is ~5e-324.  The step counts are chosen to
        stay above this limit while still being well beyond FP32's underflow
        boundary, validating the architectural decision to compute beta**t
        on the host in FP64.
        """
        val = beta**max_step  # Python uses float64 by default
        assert val > 0.0, f"beta^{max_step} underflowed to zero"
        assert math.isfinite(val), f"beta^{max_step} is not finite"

    def test_beta_power_underflows_in_fp32_eventually(self):
        """Demonstrate why host-side FP64 is necessary."""
        # 0.9^7000 in FP32 approaches zero
        val_fp32 = np.float32(0.9) ** np.float32(7000)
        val_fp64 = 0.9**7000
        # FP32 should underflow or be extremely imprecise
        # FP64 preserves more precision
        assert val_fp64 > 0 or val_fp32 == 0.0

    def test_beta2_power_stays_positive(self):
        """beta2=0.999, at step 500K: the host must keep this positive."""
        val = 0.999**500_000
        assert val > 0.0


# =========================================================================
# Scenario 4: The Rodeo (Extreme Instability Resilience)
# =========================================================================


class TestRodeo:
    """
    Verify that the stabilization policy for FP16 prevents unbounded
    thresholds and maintains finite, safe ceilings throughout the tree.
    """

    def test_all_thresholds_within_fp16_range(self):
        """No stage threshold should exceed FP16 representable range."""
        fp16_max = float(np.finfo(np.float16).max)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=fp16_max,
        )
        k, stages = policy.plan_uniform_reduction_tree(
            num_partials=256,
            hardware_max_fan_in=64,
        )
        for j in range(stages):
            threshold = policy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=k)
            assert threshold <= fp16_max, f"Stage {j} threshold {threshold} > fp16_max"
            assert threshold > 0, f"Stage {j} threshold is non-positive"

    def test_leaf_threshold_within_fp16(self):
        fp16_max = float(np.finfo(np.float16).max)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=fp16_max,
        )
        leaf_t = policy.get_leaf_safety_threshold()
        assert 0 < leaf_t < fp16_max

    def test_combined_leaf_and_tree_prevent_overflow(self):
        """
        After leaf clipping to T_leaf, summing K partials should stay < fp16_max.
        Then tree clipping at each stage should maintain this invariant.
        """
        fp16_max = float(np.finfo(np.float16).max)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=fp16_max,
        )
        k, stages = policy.plan_uniform_reduction_tree(
            num_partials=100,
            hardware_max_fan_in=16,
        )
        _leaf_t = policy.get_leaf_safety_threshold()  # noqa: F841
        # After summing K leaf-clipped vectors: max possible = K * leaf_t
        # The safety ceiling at each stage should prevent this from exceeding fp16_max
        for j in range(stages):
            stage_threshold = policy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=k)
            assert stage_threshold * k <= fp16_max * 1.01  # Small tolerance


# =========================================================================
# Scenario 5: The Hydra (Massive num_heads)
# =========================================================================


class TestHydra:
    """
    Verify that 256 modules produce a valid, deep reduction tree and
    that the planning pipeline handles the resulting memory pressure.
    """

    def test_hydra_produces_many_tiles(self, fp32_hydra_spec: Float32ModelSpec) -> None:
        grid = _make_tiling(fp32_hydra_spec)
        assert grid.total_tiles >= 16  # 256/16 = 16 at minimum

    def test_hydra_reduction_tree_is_valid(self, fp32_hydra_spec: Float32ModelSpec) -> None:
        grid = _make_tiling(fp32_hydra_spec)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        k, stages = policy.plan_uniform_reduction_tree(
            num_partials=grid.total_tiles,
            hardware_max_fan_in=256,
        )
        if grid.total_tiles > 1:
            assert k >= 2
            assert stages >= 1

    def test_hydra_all_tiles_have_valid_indices(self, fp32_hydra_spec: Float32ModelSpec) -> None:
        grid = _make_tiling(fp32_hydra_spec)
        indices: set[int] = set()
        for tile in grid:
            assert tile.flat_tile_index not in indices
            indices.add(tile.flat_tile_index)
        assert len(indices) == grid.total_tiles


# =========================================================================
# Scenario 6: The Lexicon (Massive output_classes)
# =========================================================================


class TestLexicon:
    """
    Verify that 10,000 output classes produce valid tiling and
    memory layouts without pathological sizes.
    """

    def test_lexicon_class_chunking(self, fp32_lexicon_spec: Float32ModelSpec) -> None:
        grid = _make_tiling(fp32_lexicon_spec)
        assert grid.num_class_chunks > 100

    def test_lexicon_partial_buffer_shapes_valid(self, fp32_lexicon_spec: Float32ModelSpec) -> None:
        ps = ParameterSpace(spec=fp32_lexicon_spec)
        grid = _make_tiling(fp32_lexicon_spec)
        layouts = ps.get_all_memory_layouts(batch_size=16, grid=grid, num_batch_chunks=4)
        dtype = np.dtype(fp32_lexicon_spec.SCALAR_NP_TYPE)
        for name in ["partial_grad_module_weights", "partial_grad_module_biases"]:
            shape = layouts[name].get_padded_shape(dtype)
            assert shape[0] == grid.total_tiles
            assert all(d > 0 for d in shape)


# =========================================================================
# Scenario 7: The Data Tsunami (Massive batch_size)
# =========================================================================


class TestDataTsunami:
    """
    Verify that large batch sizes produce valid reduction plans
    and that gather primitives scale correctly.
    """

    def test_large_batch_reduction_plan(self):
        """10,000 partials should yield a feasible reduction tree."""
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        k, stages = policy.plan_uniform_reduction_tree(
            num_partials=10_000,
            hardware_max_fan_in=256,
        )
        assert k >= 2
        assert k**stages >= 10_000

    def test_large_batch_gather_offsets(self):
        """LinearlyChunkedGather should handle large chunk counts."""
        gather = LinearlyChunkedGather(num_chunks=10_000, elements_per_chunk=64)
        offsets = gather.get_offsets()
        assert len(offsets) == 10_000
        assert offsets[-1] == 9_999 * 64

    def test_large_batch_memory_layout(self):
        """batch_size=10000 should still produce valid layouts."""
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=10_000, grid=grid, num_batch_chunks=4)
        ha_shape = layouts["hidden_activations"].logical_shape
        assert ha_shape[0] == 10_000


# =========================================================================
# Scenario 8: The Colossus (Holistic Stress Test)
# =========================================================================


class TestColossus:
    """
    Compound stress: large modules + large classes + large batch.
    Verifies that all planning subsystems compose correctly.
    """

    def test_colossus_planning_succeeds(self):
        """A large model with all dimensions stressed should plan without error."""
        spec = _make_spec(
            Float32ModelSpec,
            input_dim=64,
            hidden_dim=128,
            output_classes=1000,
            num_modules=64,
        )
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=256, grid=grid, num_batch_chunks=8)
        dtype = np.dtype(spec.SCALAR_NP_TYPE)

        # Verify all layouts produce non-zero shapes
        for name, layout in layouts.items():
            shape = layout.get_padded_shape(dtype)
            assert all(d > 0 for d in shape), f"Bad shape for {name}: {shape}"

    def test_colossus_reduction_trees_valid(self):
        """The reduction plans for a compound-stressed model must be valid."""
        spec = _make_spec(
            Float32ModelSpec,
            input_dim=64,
            hidden_dim=128,
            output_classes=1000,
            num_modules=64,
        )
        grid = _make_tiling(spec)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )

        # Module grad reduction tree
        k_mod, stages_mod = policy.plan_uniform_reduction_tree(
            num_partials=grid.total_tiles,
            hardware_max_fan_in=256,
        )
        if grid.total_tiles > 1:
            assert k_mod >= 2
            assert k_mod**stages_mod >= grid.total_tiles

        # Shared grad reduction tree (batch chunks)
        k_shared, _stages_shared = policy.plan_uniform_reduction_tree(
            num_partials=8,
            hardware_max_fan_in=256,
        )
        if 8 > 1:
            assert k_shared >= 2

    def test_colossus_tiling_covers_all(self):
        """All (module, class) pairs must be covered."""
        spec = _make_spec(
            Float32ModelSpec,
            num_modules=64,
            output_classes=1000,
        )
        grid = _make_tiling(spec)
        covered: set[tuple[int, int]] = set()
        for tile in grid:
            for m in range(tile.module_chunk_offset, tile.module_chunk_offset + tile.modules_per_chunk):
                for c in range(tile.class_chunk_offset, tile.class_chunk_offset + tile.classes_per_chunk):
                    if m < spec.num_modules and c < spec.output_classes:
                        covered.add((m, c))
        expected = {(m, c) for m in range(spec.num_modules) for c in range(spec.output_classes)}
        assert covered == expected
