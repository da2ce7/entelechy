# src/tests/test_workload_primitives.py
"""
Unit tests for workload tiling primitives: TilingScheme, WorkTile,
and the various GatherPrimitive implementations.

Bug-hunting focus:
* TiledGather.get_offsets() must produce monotonically increasing offsets
  that are correct multiples of elements_per_partial.
* LinearlyChunkedGather offsets must cover the full collection buffer stride.
* WorkTile must correctly clip last-chunk sizes to the actual total.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.workload_primitives import (
    TilingScheme,
    WorkTile,
    TiledGather,
    LinearlyChunkedGather,
    ContiguousGather,
    SCALAR_UINT_TYPE,
)
from src.model_spec import Float32ModelSpec


# =========================================================================
# 1. TilingScheme
# =========================================================================


class TestTilingScheme:

    def test_single_tile(self) -> None:
        """If everything fits in one chunk, total_tiles must be 1."""
        ts = TilingScheme(num_module_chunks=1, num_class_chunks=1, total_modules=8, total_classes=3)
        assert ts.total_tiles == 1
        tiles = list(ts)
        assert len(tiles) == 1

    def test_total_tiles_matches_product(self) -> None:
        ts = TilingScheme(num_module_chunks=3, num_class_chunks=4, total_modules=48, total_classes=40)
        assert ts.total_tiles == 12

    def test_tiles_cover_all_modules_and_classes(self) -> None:
        """The union of all tiles' module and class ranges must cover the full space."""
        ts = TilingScheme(num_module_chunks=3, num_class_chunks=2, total_modules=40, total_classes=10)
        tiles = list(ts)

        covered_mods = set()
        covered_cls = set()
        for t in tiles:
            for m in range(t.module_chunk_offset, t.module_chunk_offset + t.modules_per_chunk):
                covered_mods.add(m)
            for c in range(t.class_chunk_offset, t.class_chunk_offset + t.classes_per_chunk):
                covered_cls.add(c)

        assert covered_mods == set(range(40)), f"Modules not fully covered: {covered_mods}"
        assert covered_cls == set(range(10)), f"Classes not fully covered: {covered_cls}"

    def test_last_chunk_clips_to_remainder(self) -> None:
        """The last tile in each dimension must clip to the actual remainder."""
        ts = TilingScheme(num_module_chunks=3, num_class_chunks=2, total_modules=10, total_classes=7)
        last_tile = ts.get_tile(2, 1)
        # chunk_size for modules = ceil(10/3) = 4
        # last module offset = 2*4 = 8, remaining = 10 - 8 = 2
        assert last_tile.modules_per_chunk <= 4
        assert last_tile.module_chunk_offset + last_tile.modules_per_chunk <= 10
        # class chunk_size = ceil(7/2) = 4
        # last class offset = 1*4 = 4, remaining = 7 - 4 = 3
        assert last_tile.classes_per_chunk <= 4
        assert last_tile.class_chunk_offset + last_tile.classes_per_chunk <= 7

    def test_flat_tile_index_is_unique(self) -> None:
        """Each tile must have a unique flat_tile_index."""
        ts = TilingScheme(num_module_chunks=4, num_class_chunks=3, total_modules=60, total_classes=30)
        indices = [t.flat_tile_index for t in ts]
        assert len(indices) == len(set(indices)), f"Duplicate flat indices: {indices}"

    @pytest.mark.parametrize("num_mod_chunks,num_cls_chunks,total_mods,total_cls", [
        (1, 1, 8, 3),
        (1, 1, 256, 10),
        (16, 1, 256, 10),
        (1, 625, 4, 10_000),
    ])
    def test_no_zero_size_tiles(self, num_mod_chunks, num_cls_chunks, total_mods, total_cls) -> None:
        """Every tile must have at least one module and one class."""
        ts = TilingScheme(
            num_module_chunks=num_mod_chunks,
            num_class_chunks=num_cls_chunks,
            total_modules=total_mods,
            total_classes=total_cls,
        )
        for tile in ts:
            assert tile.modules_per_chunk > 0, f"Zero modules in tile {tile.flat_tile_index}"
            assert tile.classes_per_chunk > 0, f"Zero classes in tile {tile.flat_tile_index}"


# =========================================================================
# 2. GatherPrimitive implementations
# =========================================================================


class TestTiledGather:

    def test_offsets_are_correct(self) -> None:
        """Each offset must equal tile_index * elements_per_partial."""
        ts = TilingScheme(num_module_chunks=2, num_class_chunks=3, total_modules=16, total_classes=30)
        epp = 256
        tg = TiledGather(scheme=ts, _elements_per_partial=epp)
        offsets = tg.get_offsets()
        assert offsets.dtype == SCALAR_UINT_TYPE
        assert len(offsets) == ts.total_tiles
        expected = np.arange(ts.total_tiles, dtype=SCALAR_UINT_TYPE) * epp
        np.testing.assert_array_equal(offsets, expected)

    def test_num_partials_matches_tiles(self) -> None:
        ts = TilingScheme(num_module_chunks=3, num_class_chunks=2, total_modules=30, total_classes=10)
        tg = TiledGather(scheme=ts, _elements_per_partial=100)
        assert tg.num_partials == ts.total_tiles


class TestLinearlyChunkedGather:

    def test_offsets_are_strided(self) -> None:
        g = LinearlyChunkedGather(num_chunks=4, elements_per_chunk=128)
        offsets = g.get_offsets()
        assert offsets.dtype == SCALAR_UINT_TYPE
        expected = np.array([0, 128, 256, 384], dtype=SCALAR_UINT_TYPE)
        np.testing.assert_array_equal(offsets, expected)

    def test_num_partials(self) -> None:
        g = LinearlyChunkedGather(num_chunks=7, elements_per_chunk=32)
        assert g.num_partials == 7
        assert g.elements_per_partial == 32


class TestContiguousGather:

    def test_offsets_contiguous(self) -> None:
        g = ContiguousGather(_num_partials=5, _elements_per_partial=64)
        offsets = g.get_offsets()
        expected = np.arange(5, dtype=SCALAR_UINT_TYPE) * 64
        np.testing.assert_array_equal(offsets, expected)
