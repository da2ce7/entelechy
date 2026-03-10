# tests/bench_host_planning.py
from __future__ import annotations

"""
Benchmarks: Host-Side Planning Primitives.

These benchmarks exercise the computational hot-paths of the host-side
planning layer — the code that runs every batch to compute tiling layouts,
stabilization thresholds, memory plans, and parameter manifests.

No OpenCL device is required.  All benchmarks are pure host-side Python/NumPy.

Run with:
    pytest tests/bench_host_planning.py --benchmark-only
    pytest tests/bench_host_planning.py --benchmark-only --benchmark-sort=fullname
    pytest tests/bench_host_planning.py --benchmark-only --benchmark-group-by=group
"""

import math

import numpy as np
import pytest

from src.model_spec import Float32ModelSpec, Float16ModelSpec, ModelSpec
from src.parameter_space import ParameterSpace
from src.stabilization_policy import StabilizationPolicy
from src.memory_layout import MemoryLayout, PaddingStrategy, PaddingType
from src.workload_primitives import (
    ContiguousGather,
    LinearlyChunkedGather,
    TiledGather,
    TilingScheme,
)


# =========================================================================
# Canonical Model Configurations (mirroring conftest.py scenarios)
# =========================================================================

_SPECS: dict[str, Float32ModelSpec] = {
    "iris": Float32ModelSpec(
        input_dim=4, hidden_dim=32, output_classes=3,
        num_modules=8, simd_width=4, cache_line_bytes=64,
    ),
    "hydra": Float32ModelSpec(
        input_dim=16, hidden_dim=64, output_classes=10,
        num_modules=256, simd_width=4, cache_line_bytes=64,
    ),
    "lexicon": Float32ModelSpec(
        input_dim=8, hidden_dim=32, output_classes=10_000,
        num_modules=4, simd_width=4, cache_line_bytes=64,
    ),
}


def _make_tiling(spec: ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


# =========================================================================
# 1. StabilizationPolicy Benchmarks
# =========================================================================


class TestBenchStabilizationPolicy:
    """Benchmarks for the quadratic-policy reduction planning hot-path."""

    # --- plan_uniform_reduction_tree ---

    @pytest.mark.benchmark(group="stabilization-plan-tree")
    @pytest.mark.parametrize("num_partials", [8, 256, 10_000, 100_000], ids=lambda n: f"N={n}")
    def test_bench_plan_uniform_reduction_tree(self, benchmark, num_partials: int) -> None:
        """Planning cost scales with log(N) internally; verify wall-clock."""
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        result = benchmark(policy.plan_uniform_reduction_tree, num_partials, 256)
        k, stages = result
        assert k >= 2
        assert stages >= 0

    @pytest.mark.benchmark(group="stabilization-plan-tree")
    def test_bench_plan_tree_fp16_constrained(self, benchmark) -> None:
        """FP16's narrow range forces tighter constraints — measure overhead."""
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            fp_format_max=float(np.finfo(np.float16).max),
        )
        result = benchmark(policy.plan_uniform_reduction_tree, 10_000, 256)
        k, stages = result
        assert k >= 2

    # --- get_threshold_for_generic_stage ---

    @pytest.mark.benchmark(group="stabilization-threshold")
    @pytest.mark.parametrize("stage_j", [0, 3, 10, 50], ids=lambda j: f"j={j}")
    def test_bench_get_threshold_for_generic_stage(self, benchmark, stage_j: int) -> None:
        """Per-stage threshold lookup — called once per reduction stage."""
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        result = benchmark(policy.get_threshold_for_generic_stage, stage_j, 64)
        assert result > 0.0

    # --- get_leaf_safety_threshold ---

    @pytest.mark.benchmark(group="stabilization-threshold")
    def test_bench_get_leaf_safety_threshold(self, benchmark) -> None:
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        result = benchmark(policy.get_leaf_safety_threshold)
        assert result > 0.0

    # --- get_specialized_reduction_policy_k ---

    @pytest.mark.benchmark(group="stabilization-policy-k")
    @pytest.mark.parametrize(
        "user_k,hw_max",
        [(32, 256), (256, 256), (1024, 64)],
        ids=["user<hw", "user==hw", "user>hw"],
    )
    def test_bench_get_specialized_reduction_policy_k(
        self, benchmark, user_k: int, hw_max: int,
    ) -> None:
        policy = StabilizationPolicy(
            t_algorithmic=1.0, lambda_=1.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        result = benchmark(policy.get_specialized_reduction_policy_k, user_k, hw_max)
        assert result >= 2


# =========================================================================
# 2. TilingScheme Benchmarks
# =========================================================================


class TestBenchTilingScheme:
    """Benchmarks for tile generation across model scales."""

    @pytest.mark.benchmark(group="tiling-iterate")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_iterate_all_tiles(self, benchmark, scale: str) -> None:
        """Measure full tile iteration (the hot-path for dispatch planning)."""
        spec = _SPECS[scale]
        scheme = _make_tiling(spec)
        tiles = benchmark(lambda: list(scheme))
        assert len(tiles) == scheme.total_tiles

    @pytest.mark.benchmark(group="tiling-single")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_get_single_tile(self, benchmark, scale: str) -> None:
        """Single tile lookup — used in inner dispatch loops."""
        spec = _SPECS[scale]
        scheme = _make_tiling(spec)
        tile = benchmark(scheme.get_tile, 0, 0)
        assert tile.flat_tile_index == 0

    @pytest.mark.benchmark(group="tiling-iterate")
    def test_bench_iterate_extreme_grid(self, benchmark) -> None:
        """Stress test: 64 module chunks × 625 class chunks = 40 000 tiles."""
        scheme = TilingScheme(
            num_module_chunks=64,
            num_class_chunks=625,
            total_modules=1024,
            total_classes=10_000,
        )
        tiles = benchmark(lambda: list(scheme))
        assert len(tiles) == 40_000


# =========================================================================
# 3. GatherPrimitive Offset Generation Benchmarks
# =========================================================================


class TestBenchGatherPrimitives:
    """Benchmarks for offset array generation (consumed by reduction kernels)."""

    # --- LinearlyChunkedGather ---

    @pytest.mark.benchmark(group="gather-offsets")
    @pytest.mark.parametrize("n_chunks", [8, 256, 10_000], ids=lambda n: f"N={n}")
    def test_bench_linear_gather_offsets(self, benchmark, n_chunks: int) -> None:
        gather = LinearlyChunkedGather(num_chunks=n_chunks, elements_per_chunk=64)
        offsets = benchmark(gather.get_offsets)
        assert offsets.shape == (n_chunks,)
        assert offsets.dtype == np.uint32

    # --- TiledGather ---

    @pytest.mark.benchmark(group="gather-offsets")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_tiled_gather_offsets(self, benchmark, scale: str) -> None:
        spec = _SPECS[scale]
        scheme = _make_tiling(spec)
        gather = TiledGather(scheme=scheme, _elements_per_partial=128)
        offsets = benchmark(gather.get_offsets)
        assert offsets.shape == (scheme.total_tiles,)

    # --- ContiguousGather ---

    @pytest.mark.benchmark(group="gather-offsets")
    @pytest.mark.parametrize("n_partials", [8, 256, 10_000], ids=lambda n: f"N={n}")
    def test_bench_contiguous_gather_offsets(self, benchmark, n_partials: int) -> None:
        gather = ContiguousGather(_num_partials=n_partials, _elements_per_partial=64)
        offsets = benchmark(gather.get_offsets)
        assert offsets.shape == (n_partials,)


# =========================================================================
# 4. MemoryLayout Benchmarks
# =========================================================================


class TestBenchMemoryLayout:
    """Benchmarks for padding resolution on representative buffer shapes."""

    @pytest.mark.benchmark(group="memory-layout")
    def test_bench_no_padding(self, benchmark) -> None:
        """Baseline: layout resolution with no padding strategies."""
        layout = MemoryLayout((150, 32))
        result = benchmark(layout.get_padded_shape, np.float32)
        assert result == (150, 32)

    @pytest.mark.benchmark(group="memory-layout")
    def test_bench_simd_padding(self, benchmark) -> None:
        """Single ELEMENT_COUNT strategy (SIMD alignment)."""
        layout = MemoryLayout((150, 30))
        layout.add_strategy(PaddingStrategy(PaddingType.ELEMENT_COUNT, 4, -1))
        result = benchmark(layout.get_padded_shape, np.float32)
        assert result[1] % 4 == 0

    @pytest.mark.benchmark(group="memory-layout")
    def test_bench_cache_line_padding(self, benchmark) -> None:
        """Single BYTE_ALIGNMENT strategy (cache-line alignment)."""
        layout = MemoryLayout((150, 13))
        layout.add_strategy(PaddingStrategy(PaddingType.BYTE_ALIGNMENT, 64, -1))
        result = benchmark(layout.get_padded_shape, np.float32)
        assert (result[1] * 4) % 64 == 0

    @pytest.mark.benchmark(group="memory-layout")
    def test_bench_compound_padding(self, benchmark) -> None:
        """Compound: SIMD + BYTE_ALIGNMENT in sequence (realistic for this arch)."""
        layout = MemoryLayout((150, 30))
        layout.add_strategy(PaddingStrategy(PaddingType.ELEMENT_COUNT, 4, -1))
        layout.add_strategy(PaddingStrategy(PaddingType.BYTE_ALIGNMENT, 64, -1))
        result = benchmark(layout.get_padded_shape, np.float32)
        assert result[1] % 4 == 0
        assert (result[1] * 4) % 64 == 0

    @pytest.mark.benchmark(group="memory-layout")
    def test_bench_high_rank_layout(self, benchmark) -> None:
        """4D tensor layout (partial grad collection buffer shape)."""
        layout = MemoryLayout((16, 8, 150, 30))
        layout.add_strategy(PaddingStrategy(PaddingType.ELEMENT_COUNT, 4, -1))
        layout.add_strategy(PaddingStrategy(PaddingType.BYTE_ALIGNMENT, 64, -1))
        result = benchmark(layout.get_padded_shape, np.float32)
        assert len(result) == 4

    @pytest.mark.benchmark(group="memory-layout")
    def test_bench_fp16_cache_line(self, benchmark) -> None:
        """FP16 doubles elements-per-cache-line — verify no overhead cliff."""
        layout = MemoryLayout((150, 13))
        layout.add_strategy(PaddingStrategy(PaddingType.BYTE_ALIGNMENT, 64, -1))
        result = benchmark(layout.get_padded_shape, np.float16)
        assert (result[1] * 2) % 64 == 0


# =========================================================================
# 5. ParameterSpace Benchmarks
# =========================================================================


class TestBenchParameterSpace:
    """Benchmarks for parameter manifest construction and layout generation."""

    @pytest.mark.benchmark(group="param-space-init")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_parameter_space_construction(self, benchmark, scale: str) -> None:
        """Construction time includes flow generation."""
        spec = _SPECS[scale]
        ps = benchmark(ParameterSpace, spec)
        assert len(list(ps)) > 0

    @pytest.mark.benchmark(group="param-space-layouts")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_get_all_memory_layouts(self, benchmark, scale: str) -> None:
        """Full layout generation — the heaviest single-call planning step."""
        spec = _SPECS[scale]
        ps = ParameterSpace(spec)
        grid = _make_tiling(spec)

        def _generate():
            return ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)

        layouts = benchmark(_generate)
        assert len(layouts) > 10  # Expect many buffers

    @pytest.mark.benchmark(group="param-space-layouts")
    def test_bench_get_all_memory_layouts_large_batch(self, benchmark) -> None:
        """Layout generation with a large batch size (stress byte calculations)."""
        spec = _SPECS["hydra"]
        ps = ParameterSpace(spec)
        grid = _make_tiling(spec)

        def _generate():
            return ps.get_all_memory_layouts(batch_size=4096, grid=grid, num_batch_chunks=64)

        layouts = benchmark(_generate)
        assert "input" in layouts

    @pytest.mark.benchmark(group="param-space-iteration")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_iterate_parameter_flows(self, benchmark, scale: str) -> None:
        """Iteration over flows — used by orchestrator for dispatch planning."""
        spec = _SPECS[scale]
        ps = ParameterSpace(spec)
        flows = benchmark(lambda: list(ps))
        assert len(flows) > 0


# =========================================================================
# 6. Integrated Planning Pipeline Benchmarks
# =========================================================================


class TestBenchIntegratedPlanning:
    """
    End-to-end benchmarks that compose multiple primitives, mirroring the
    actual planning sequence the host orchestrator executes per batch.
    """

    @pytest.mark.benchmark(group="integrated-pipeline")
    @pytest.mark.parametrize("scale", ["iris", "hydra", "lexicon"])
    def test_bench_full_planning_pipeline(self, benchmark, scale: str) -> None:
        """
        Spec -> ParameterSpace -> TilingScheme -> MemoryLayouts -> ReductionPlan.

        This is the full host-side planning path, executed once per batch.
        """
        spec = _SPECS[scale]
        fp_max = float(np.finfo(np.float32).max)

        def _plan():
            ps = ParameterSpace(spec)
            grid = _make_tiling(spec)
            layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)
            tiles = list(grid)
            policy = StabilizationPolicy(
                t_algorithmic=1.0, lambda_=1.0, fp_format_max=fp_max,
            )
            k, stages = policy.plan_uniform_reduction_tree(grid.total_tiles, 256)
            thresholds = [
                policy.get_threshold_for_generic_stage(j, k)
                for j in range(stages)
            ]
            return layouts, tiles, k, stages, thresholds

        result = benchmark(_plan)
        layouts, tiles, k, stages, thresholds = result
        assert len(layouts) > 10
        assert len(tiles) > 0
        # k=1 is valid when total_tiles==1 (trivial single-partial case)
        assert k >= 1

    @pytest.mark.benchmark(group="integrated-pipeline")
    def test_bench_colossus_planning_pipeline(self, benchmark) -> None:
        """
        Stress test: Colossus-scale model (256 modules × 10 000 classes).

        Exercises the planning pipeline at a scale that stresses both
        tiling iteration and reduction tree planning.
        """
        spec = Float32ModelSpec(
            input_dim=32, hidden_dim=128, output_classes=10_000,
            num_modules=256, simd_width=4, cache_line_bytes=64,
        )
        fp_max = float(np.finfo(np.float32).max)

        def _plan():
            ps = ParameterSpace(spec)
            grid = _make_tiling(spec)
            layouts = ps.get_all_memory_layouts(batch_size=256, grid=grid, num_batch_chunks=8)
            tiles = list(grid)
            policy = StabilizationPolicy(
                t_algorithmic=1.0, lambda_=1.0, fp_format_max=fp_max,
            )
            k, stages = policy.plan_uniform_reduction_tree(grid.total_tiles, 256)
            leaf_threshold = policy.get_leaf_safety_threshold()
            return layouts, tiles, k, stages, leaf_threshold

        result = benchmark(_plan)
        layouts, tiles, k, stages, leaf_threshold = result
        assert len(tiles) == _make_tiling(spec).total_tiles
        assert leaf_threshold > 0.0
