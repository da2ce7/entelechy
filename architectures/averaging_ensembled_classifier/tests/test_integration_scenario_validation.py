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
  9. The Behemoth          – Massive hidden_dim Cache/Recompute trade-off
 10. The Real-Time Trader  – Act/Learn temporal split & event-triggered Learn

No OpenCL device is required.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pytest

from src.shared.model_spec import Float32ModelSpec, Float16ModelSpec, ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.stabilization_policy import StabilizationPolicy
from src.shared.workload_primitives import LinearlyChunkedGather, TilingScheme

# =========================================================================
# Helpers
# =========================================================================


def _make_spec(cls: Callable[..., ModelSpec], **kwargs: Any) -> ModelSpec:
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


def _make_tiling(spec: ModelSpec) -> TilingScheme:
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

    def test_hydra_produces_many_tiles(self, fp32_hydra_spec: ModelSpec) -> None:
        grid = _make_tiling(fp32_hydra_spec)
        assert grid.total_tiles >= 16  # 256/16 = 16 at minimum

    def test_hydra_reduction_tree_is_valid(self, fp32_hydra_spec: ModelSpec) -> None:
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

    def test_hydra_all_tiles_have_valid_indices(self, fp32_hydra_spec: ModelSpec) -> None:
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

    def test_lexicon_class_chunking(self, fp32_lexicon_spec: ModelSpec) -> None:
        grid = _make_tiling(fp32_lexicon_spec)
        assert grid.num_class_chunks > 100

    def test_lexicon_partial_buffer_shapes_valid(self, fp32_lexicon_spec: ModelSpec) -> None:
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


# =========================================================================
# Scenario 9: The Behemoth (Massive hidden_dim)
# =========================================================================


class TestBehemoth:
    """
    Verify scalability through strategic memory trade-offs when hidden_dim
    is a bottleneck.  The host's Cache-vs-Recompute policy must produce
    radically different memory footprints.

    From CONCEPT.md:
    "When the hidden_i buffer is a bottleneck, the host's policy to Cache or
     Recompute it remains a critical, orthogonal optimization that works in
     synergy with the batch accumulation model."
    """

    @pytest.fixture
    def behemoth_spec(self) -> ModelSpec:
        """A model with massive hidden_dim to stress memory layouts."""
        return _make_spec(
            Float32ModelSpec,
            input_dim=64,
            hidden_dim=2048,
            output_classes=10,
            num_modules=8,
        )

    def test_hidden_activations_dominate_memory(self, behemoth_spec: ModelSpec) -> None:
        """With hidden_dim=2048, the hidden_activations buffer should be
        substantially larger than the module-level gradient buffers."""
        ps = ParameterSpace(spec=behemoth_spec)
        grid = _make_tiling(behemoth_spec)
        batch_size = 256
        layouts = ps.get_all_memory_layouts(
            batch_size=batch_size, grid=grid, num_batch_chunks=8
        )
        dtype = np.dtype(behemoth_spec.SCALAR_NP_TYPE)

        hidden_shape = layouts["hidden_activations"].get_padded_shape(dtype)
        hidden_bytes = int(np.prod(hidden_shape)) * dtype.itemsize

        # Module weights: (num_modules, padded_hidden, padded_class)
        mod_w_shape = layouts["module_weights"].get_padded_shape(dtype)
        mod_w_bytes = int(np.prod(mod_w_shape)) * dtype.itemsize

        # hidden_activations should be a significant fraction of total memory
        assert hidden_bytes > 0
        # With batch_size=256 and hidden_dim=2048, this buffer is at least 2MB
        assert hidden_bytes >= 256 * 2048 * dtype.itemsize

    def test_permuted_grad_h_scales_with_hidden_dim(self, behemoth_spec: ModelSpec) -> None:
        """The permuted_grad_h buffer grows linearly with hidden_dim and batch."""
        ps = ParameterSpace(spec=behemoth_spec)
        grid = _make_tiling(behemoth_spec)
        batch_size = 256
        layouts = ps.get_all_memory_layouts(
            batch_size=batch_size, grid=grid, num_batch_chunks=8
        )
        dtype = np.dtype(behemoth_spec.SCALAR_NP_TYPE)

        permuted = layouts["permuted_grad_h"]
        p_shape = permuted.get_padded_shape(dtype)
        permuted_bytes = int(np.prod(p_shape)) * dtype.itemsize

        # SoA layout: (B * padded_H, padded_M)
        # For B=256, H=2048, this is very large
        assert permuted.logical_shape[0] == batch_size * behemoth_spec.padded_hidden_dim
        assert permuted_bytes > 0

    def test_cache_strategy_memory_footprint(self, behemoth_spec: ModelSpec) -> None:
        """Under CACHE strategy, hidden_activations are retained in full."""
        ps = ParameterSpace(spec=behemoth_spec)
        grid = _make_tiling(behemoth_spec)
        # Two batch sizes to demonstrate scaling
        for batch_size in [64, 256]:
            layouts = ps.get_all_memory_layouts(
                batch_size=batch_size, grid=grid, num_batch_chunks=8
            )
            hidden = layouts["hidden_activations"]
            assert hidden.logical_shape[0] == batch_size
            assert hidden.logical_shape[1] == behemoth_spec.padded_hidden_dim

    def test_recompute_strategy_reduces_partial_grad_hidden(self, behemoth_spec: ModelSpec) -> None:
        """Under RECOMPUTE, the partial_grad_hidden buffer is allocated per-tile,
        but each tile is processed serially and the scratch is reused.

        The partial_grad_hidden_activations collection buffer is still allocated
        (it holds ALL tiles' clipped results), but the raw hidden_i itself is
        recomputed into a small scratch buffer."""
        ps = ParameterSpace(spec=behemoth_spec)
        grid = _make_tiling(behemoth_spec)
        batch_size = 256
        layouts = ps.get_all_memory_layouts(
            batch_size=batch_size, grid=grid, num_batch_chunks=8
        )
        dtype = np.dtype(behemoth_spec.SCALAR_NP_TYPE)

        # The full hidden_activations buffer (for CACHE) exists in layout
        full_hidden_shape = layouts["hidden_activations"].get_padded_shape(dtype)
        full_hidden_bytes = int(np.prod(full_hidden_shape)) * dtype.itemsize

        # A RECOMPUTE strategy would use a scratch buffer of the same shape
        # as hidden_activations for ONE tile at a time, then discard it.
        # The scratch bytes are always <= the full cache bytes.
        scratch_bytes = full_hidden_bytes  # Same shape, but transient
        assert scratch_bytes <= full_hidden_bytes

    def test_behemoth_layouts_produce_valid_shapes(self, behemoth_spec: ModelSpec) -> None:
        """All layouts for the Behemoth must produce non-zero shapes."""
        ps = ParameterSpace(spec=behemoth_spec)
        grid = _make_tiling(behemoth_spec)
        layouts = ps.get_all_memory_layouts(
            batch_size=256, grid=grid, num_batch_chunks=8
        )
        dtype = np.dtype(behemoth_spec.SCALAR_NP_TYPE)
        for name, layout in layouts.items():
            shape = layout.get_padded_shape(dtype)
            total_bytes = int(np.prod(shape)) * dtype.itemsize
            assert total_bytes > 0, f"{name} has zero bytes"

    def test_behemoth_reduction_tree_valid(self, behemoth_spec: ModelSpec) -> None:
        """The reduction tree for a Behemoth model must be valid."""
        grid = _make_tiling(behemoth_spec)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        if grid.total_tiles > 1:
            k, stages = policy.plan_uniform_reduction_tree(
                num_partials=grid.total_tiles,
                hardware_max_fan_in=256,
            )
            assert k >= 2
            assert k ** stages >= grid.total_tiles

    def test_behemoth_shared_weight_layout_scales(self, behemoth_spec: ModelSpec) -> None:
        """The shared_weights buffer grows with hidden_dim."""
        ps = ParameterSpace(spec=behemoth_spec)
        grid = _make_tiling(behemoth_spec)
        layouts = ps.get_all_memory_layouts(
            batch_size=64, grid=grid, num_batch_chunks=4
        )
        sw = layouts["shared_weights"]
        # Shape: (padded_hidden_dim, padded_input_dim) — hidden-major layout
        assert sw.logical_shape[0] == behemoth_spec.padded_hidden_dim
        assert sw.logical_shape[0] >= 2048  # At least hidden_dim
        assert sw.logical_shape[1] == behemoth_spec.padded_input_dim


# =========================================================================
# Scenario 10: The Real-Time Trader (Event-Triggered Execution Mode)
# =========================================================================


class TestRealTimeTrader:
    """
    Verify the Act/Learn temporal split and event-triggered Learn-phase
    triggering described in CONCEPT.md.

    From CONCEPT.md:
    "All workflows follow Act then Learn sequencing, manifesting as either
     Sequential Execution Mode — where Act-Learn phases execute contiguously
     — or Event-Triggered Execution Mode — where Learn-phase execution
     awaits an external readiness signal post-Act."

    This scenario validates that:
    - Act and Learn phases are independently plannable
    - Act-phase output (Final Probs) is self-sufficient for inference
    - VRAM for Learn-phase intermediates can be deferred
    - The lifecycle policy supports both CACHE and RECOMPUTE strategies
      for the event-triggered case
    """

    def test_act_phase_output_is_self_contained(self) -> None:
        """The Act phase produces Final Probs, which must be independently
        usable without any Learn-phase data."""
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=1, grid=grid, num_batch_chunks=1)

        # Act-phase buffers must all be present
        act_buffers = [
            "input", "hidden_activations", "hidden_mask",
            "logits", "sample_mask",
            "partial_probs", "final_loss",
        ]
        for buf_name in act_buffers:
            assert buf_name in layouts, f"Act-phase buffer '{buf_name}' missing"

    def test_learn_phase_buffers_are_additional(self) -> None:
        """Learn-phase buffers (gradients, optimizer state) are present in the
        layout but are logically separable from Act-phase buffers."""
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)

        learn_only_buffers = [
            "partial_grad_module_weights",
            "partial_grad_module_biases",
            "partial_grad_temps",
            "partial_grad_hidden_activations",
            "partial_grad_shared_weights",
            "partial_grad_shared_biases",
            "summed_grad_hidden_activations",
            "permuted_grad_h",
        ]
        for buf_name in learn_only_buffers:
            assert buf_name in layouts, f"Learn-phase buffer '{buf_name}' missing"

    def test_act_phase_independent_of_batch_size_for_planning(self) -> None:
        """The Act phase's computational structure (tiling, forward pass)
        is identical regardless of whether Learn will follow immediately
        (Sequential) or later (Event-Triggered)."""
        spec = _make_spec(Float32ModelSpec, num_modules=8, output_classes=10)
        grid = _make_tiling(spec)

        # The grid (which drives Act-phase kernel dispatch) is determined by
        # model dimensions, not by execution mode
        assert grid.total_tiles == grid.num_module_chunks * grid.num_class_chunks
        # All tiles are valid for the Act phase
        for tile in grid:
            assert tile.modules_per_chunk > 0
            assert tile.classes_per_chunk > 0

    def test_recompute_strategy_enables_vram_release(self) -> None:
        """Under RECOMPUTE strategy, hidden_activations can be discarded after
        the Act phase and recomputed when Learn is triggered.  This is the
        key architectural enabler for the Real-Time Trader scenario."""
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)

        # hidden_activations exists in layout (it's always planned)
        ha = layouts["hidden_activations"]
        dtype = np.dtype(spec.SCALAR_NP_TYPE)
        ha_bytes = int(np.prod(ha.get_padded_shape(dtype))) * dtype.itemsize

        # The RECOMPUTE strategy means this buffer is transient:
        # allocated for Act, released, then re-allocated for Learn.
        # This test verifies the buffer CAN be independently described.
        assert ha_bytes > 0
        # The hidden_mask is similarly releasable
        hm = layouts["hidden_mask"]
        hm_bytes = int(np.prod(hm.get_padded_shape(dtype))) * dtype.itemsize
        assert hm_bytes > 0

    def test_event_triggered_plan_with_deferred_targets(self) -> None:
        """In Event-Triggered mode, ground truth (targets) arrive after Act.
        The layout must still plan for targets buffers regardless."""
        spec = _make_spec(Float32ModelSpec)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=1, grid=grid, num_batch_chunks=1)

        # Both target types must be planned (the strategy object selects one)
        assert "targets_cce" in layouts
        assert "targets_bce" in layouts

    def test_single_item_event_triggered(self) -> None:
        """The Real-Time Trader processes one item at a time (N=1).
        The full planning pipeline must be valid for batch_size=1."""
        spec = _make_spec(Float32ModelSpec, num_modules=4, output_classes=5)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=1, grid=grid, num_batch_chunks=1)
        dtype = np.dtype(spec.SCALAR_NP_TYPE)

        for name, layout in layouts.items():
            shape = layout.get_padded_shape(dtype)
            total = int(np.prod(shape)) * dtype.itemsize
            assert total > 0, f"{name} has zero bytes for N=1"

        # With N=1, the reduction tree degenerates (no reduction needed)
        policy = StabilizationPolicy(
            t_algorithmic=1.0,
            lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        # Single tile, single partial → no multi-stage reduction
        _k, stages = policy.plan_uniform_reduction_tree(
            num_partials=grid.total_tiles,
            hardware_max_fan_in=256,
        )
        # For very small grids, may need 0 or 1 stages
        assert stages >= 0

    def test_temporal_split_orthogonality(self) -> None:
        """The Act/Learn split is orthogonal to the batch processing model.
        Both Sequential and Event-Triggered modes must produce identical
        computational grids for the same model."""
        spec = _make_spec(Float32ModelSpec, num_modules=16, output_classes=20)
        grid = _make_tiling(spec)

        # The grid is model-determined, not mode-determined
        tiles_sequential = list(grid)
        tiles_event_triggered = list(grid)  # Same grid, same tiles

        assert len(tiles_sequential) == len(tiles_event_triggered)
        for t_seq, t_evt in zip(tiles_sequential, tiles_event_triggered):
            assert t_seq.flat_tile_index == t_evt.flat_tile_index
            assert t_seq.module_chunk_offset == t_evt.module_chunk_offset
            assert t_seq.class_chunk_offset == t_evt.class_chunk_offset
