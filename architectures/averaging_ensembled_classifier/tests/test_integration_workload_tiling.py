# tests/test_integration_workload_tiling.py
from __future__ import annotations

"""
Integration Tests: TilingScheme + GatherPrimitive + Placement Contract.

These tests verify the interplay between the workload partitioning primitives
(TilingScheme, WorkTile) and the indirection/gather abstractions that
underpin the Recursive Clip-Aggregation Engine.

Target contracts validated:
  - The Placement Contract: flat_tile_index uniqueness and coverage
  - The Indirection Contract: offset_list correctness and non-overlap
  - TilingScheme <-> TiledGather consistency

No OpenCL device is required.
"""

import numpy as np
import pytest

from src.model_spec import Float32ModelSpec
from src.workload_primitives import (
    ContiguousGather,
    GatherPrimitive,
    LinearlyChunkedGather,
    TiledGather,
    TilingScheme,
)

# =========================================================================
# 1. Placement Contract
# =========================================================================


class TestPlacementContract:
    """Verify that the Placement Contract guarantees unique, complete indexing."""

    def test_flat_tile_indices_are_unique(self, iris_tiling: TilingScheme) -> None:
        """Every tile must have a unique flat_tile_index."""
        indices = [tile.flat_tile_index for tile in iris_tiling]
        assert len(indices) == len(set(indices))

    def test_flat_tile_indices_cover_range(self, iris_tiling: TilingScheme) -> None:
        """flat_tile_index values must span [0, total_tiles)."""
        indices = sorted(tile.flat_tile_index for tile in iris_tiling)
        assert indices == list(range(iris_tiling.total_tiles))

    def test_total_tiles_equals_product_of_chunks(self, iris_tiling: TilingScheme) -> None:
        assert iris_tiling.total_tiles == iris_tiling.num_module_chunks * iris_tiling.num_class_chunks

    @pytest.mark.parametrize("n_mods,n_cls", [(1, 1), (3, 1), (1, 5), (7, 13), (256, 10000)])
    def test_all_elements_covered(self, n_mods: int, n_cls: int) -> None:
        """Every (module, class) coordinate must be owned by exactly one tile."""
        grid = TilingScheme(
            num_module_chunks=(n_mods + 15) // 16,
            num_class_chunks=(n_cls + 15) // 16,
            total_modules=n_mods,
            total_classes=n_cls,
        )
        covered_modules: set[int] = set()
        covered_classes_per_mod: dict[int, set[int]] = {m: set() for m in range(n_mods)}

        for tile in grid:
            for m in range(tile.module_chunk_offset, tile.module_chunk_offset + tile.modules_per_chunk):
                if m < n_mods:
                    covered_modules.add(m)
                    for c in range(tile.class_chunk_offset, tile.class_chunk_offset + tile.classes_per_chunk):
                        if c < n_cls:
                            covered_classes_per_mod[m].add(c)

        assert covered_modules == set(range(n_mods))
        for m in range(n_mods):
            assert covered_classes_per_mod[m] == set(range(n_cls)), f"Module {m} missing class indices"

    def test_tile_chunks_do_not_exceed_total(self):
        """Tile offsets + chunk sizes must not exceed actual total dimensions."""
        grid = TilingScheme(
            num_module_chunks=3,
            num_class_chunks=2,
            total_modules=7,
            total_classes=5,
        )
        for tile in grid:
            assert tile.module_chunk_offset + tile.modules_per_chunk <= grid.total_modules
            assert tile.class_chunk_offset + tile.classes_per_chunk <= grid.total_classes


# =========================================================================
# 2. Indirection Contract (GatherPrimitive Implementations)
# =========================================================================


class TestIndirectionContract:
    """Verify that GatherPrimitive subclasses produce valid offset lists."""

    def test_tiled_gather_offsets_non_overlapping(self, iris_tiling: TilingScheme) -> None:
        """Offsets in a TiledGather must not overlap."""
        elements_per = 100
        gather = TiledGather(scheme=iris_tiling, _elements_per_partial=elements_per)
        offsets = gather.get_offsets()

        assert len(offsets) == iris_tiling.total_tiles
        # Check no two offset ranges overlap
        ranges = [(int(o), int(o) + elements_per) for o in offsets]
        ranges.sort()
        for i in range(1, len(ranges)):
            assert ranges[i][0] >= ranges[i - 1][1], f"Overlap between tile {i - 1} and {i}"

    def test_tiled_gather_offsets_are_contiguous(self, iris_tiling: TilingScheme) -> None:
        """TiledGather offsets should be sequential multiples of elements_per_partial."""
        elements_per = 50
        gather = TiledGather(scheme=iris_tiling, _elements_per_partial=elements_per)
        offsets = gather.get_offsets()
        expected = np.arange(iris_tiling.total_tiles, dtype=np.uint32) * elements_per
        np.testing.assert_array_equal(offsets, expected)

    def test_linearly_chunked_gather_offsets(self):
        """LinearlyChunkedGather offsets must be sequential and non-overlapping."""
        gather = LinearlyChunkedGather(num_chunks=8, elements_per_chunk=256)
        offsets = gather.get_offsets()
        assert len(offsets) == 8
        expected = np.arange(8, dtype=np.uint32) * 256
        np.testing.assert_array_equal(offsets, expected)

    def test_contiguous_gather_offsets(self):
        """ContiguousGather offsets should be a simple arithmetic progression."""
        gather = ContiguousGather(_num_partials=4, _elements_per_partial=100)
        offsets = gather.get_offsets()
        expected = np.arange(4, dtype=np.uint32) * 100
        np.testing.assert_array_equal(offsets, expected)

    def test_gather_properties_consistent(self):
        """All gather primitives report consistent num_partials and elements_per_partial."""
        gathers: list[GatherPrimitive] = [
            TiledGather(
                scheme=TilingScheme(2, 3, 20, 30),
                _elements_per_partial=50,
            ),
            LinearlyChunkedGather(num_chunks=5, elements_per_chunk=200),
            ContiguousGather(_num_partials=7, _elements_per_partial=128),
        ]
        for gather in gathers:
            offsets = gather.get_offsets()
            assert len(offsets) == gather.num_partials
            assert offsets.dtype == np.uint32


# =========================================================================
# 3. TilingScheme + GatherPrimitive Integration (Hydra scale)
# =========================================================================


class TestTilingGatherIntegration:
    """Verify tiling-to-gather consistency at scale."""

    def test_hydra_tiling_produces_many_tiles(self, fp32_hydra_spec: Float32ModelSpec) -> None:
        """256 modules chunked into groups of 16 → 16 module chunks."""
        grid = TilingScheme(
            num_module_chunks=(fp32_hydra_spec.num_modules + 15) // 16,
            num_class_chunks=(fp32_hydra_spec.output_classes + 15) // 16,
            total_modules=fp32_hydra_spec.num_modules,
            total_classes=fp32_hydra_spec.output_classes,
        )
        assert grid.total_tiles == grid.num_module_chunks * grid.num_class_chunks
        gather = TiledGather(scheme=grid, _elements_per_partial=42)
        assert gather.num_partials == grid.total_tiles

    def test_lexicon_tiling_many_class_chunks(self, fp32_lexicon_spec: Float32ModelSpec) -> None:
        """10,000 classes chunked into groups of 16 → 625 class chunks."""
        grid = TilingScheme(
            num_module_chunks=(fp32_lexicon_spec.num_modules + 15) // 16,
            num_class_chunks=(fp32_lexicon_spec.output_classes + 15) // 16,
            total_modules=fp32_lexicon_spec.num_modules,
            total_classes=fp32_lexicon_spec.output_classes,
        )
        assert grid.num_class_chunks == (10_000 + 15) // 16
        # Verify all tiles iterate cleanly
        count = sum(1 for _ in grid)
        assert count == grid.total_tiles
