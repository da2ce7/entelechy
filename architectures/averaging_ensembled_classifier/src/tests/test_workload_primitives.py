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

from src.shared.workload_primitives import (
    TilingScheme,
    WorkTile,
    TiledGather,
    LinearlyChunkedGather,
    ContiguousGather,
    ModuleChunkGather,
    ModuleBufferKind,
    SCALAR_UINT_TYPE,
)
from src.shared.model_spec import ModelSpec


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


# =========================================================================
# 3. ModuleChunkGather (ADR-030 Module-Chunk Isolation)
# =========================================================================


class TestModuleChunkGather:
    """Tests for ModuleChunkGather per Phase 13 plan (Step 13.18).

    ModuleChunkGather implements the Module-Chunk Isolation Invariant:
    offset lists for parameter-gradient reductions SHALL contain only
    tile offsets belonging to a single module chunk.
    """

    def test_basic_offset_correctness_weights(self) -> None:
        """For num_module_chunks=3, num_class_chunks=4, verify correct offsets per chunk."""
        ts = TilingScheme(
            num_module_chunks=3,
            num_class_chunks=4,
            total_modules=30,
            total_classes=40,
            padded_hidden_count=256,
            padded_total_output_class_count=48,
        )
        epp = ts.elements_per_tile(ModuleBufferKind.WEIGHTS)  # modules_per_chunk * hidden

        # Module chunk 0 gets tiles [0,1,2,3] (class chunks 0-3 for module chunk 0)
        mcg0 = ModuleChunkGather(ts, module_chunk_index=0, buffer_kind=ModuleBufferKind.WEIGHTS)
        offsets0 = mcg0.get_offsets()
        assert len(offsets0) == 4
        expected0 = np.array([0, 1, 2, 3], dtype=SCALAR_UINT_TYPE) * epp
        np.testing.assert_array_equal(offsets0, expected0)

        # Module chunk 1 gets tiles [4,5,6,7]
        mcg1 = ModuleChunkGather(ts, module_chunk_index=1, buffer_kind=ModuleBufferKind.WEIGHTS)
        offsets1 = mcg1.get_offsets()
        expected1 = np.array([4, 5, 6, 7], dtype=SCALAR_UINT_TYPE) * epp
        np.testing.assert_array_equal(offsets1, expected1)

        # Module chunk 2 gets tiles [8,9,10,11]
        mcg2 = ModuleChunkGather(ts, module_chunk_index=2, buffer_kind=ModuleBufferKind.WEIGHTS)
        offsets2 = mcg2.get_offsets()
        expected2 = np.array([8, 9, 10, 11], dtype=SCALAR_UINT_TYPE) * epp
        np.testing.assert_array_equal(offsets2, expected2)

    def test_degeneration_equivalence_weights(self) -> None:
        """When num_module_chunks=1, ModuleChunkGather(0) === TiledGather."""
        ts = TilingScheme(
            num_module_chunks=1,
            num_class_chunks=4,
            total_modules=16,
            total_classes=40,
            padded_hidden_count=128,
            padded_total_output_class_count=48,
        )
        epp = ts.elements_per_tile(ModuleBufferKind.WEIGHTS)

        mcg = ModuleChunkGather(ts, module_chunk_index=0, buffer_kind=ModuleBufferKind.WEIGHTS)
        tg = TiledGather(scheme=ts, _elements_per_partial=epp)

        np.testing.assert_array_equal(mcg.get_offsets(), tg.get_offsets())
        assert mcg.num_partials == tg.num_partials
        assert mcg.elements_per_partial == tg.elements_per_partial

    def test_degeneration_equivalence_biases(self) -> None:
        """When num_module_chunks=1, ModuleChunkGather(0) === TiledGather for biases."""
        ts = TilingScheme(
            num_module_chunks=1,
            num_class_chunks=4,
            total_modules=16,
            total_classes=40,
            padded_hidden_count=128,
            padded_total_output_class_count=48,
        )
        epp = ts.elements_per_tile(ModuleBufferKind.BIASES)

        mcg = ModuleChunkGather(ts, module_chunk_index=0, buffer_kind=ModuleBufferKind.BIASES)
        tg = TiledGather(scheme=ts, _elements_per_partial=epp)

        np.testing.assert_array_equal(mcg.get_offsets(), tg.get_offsets())

    def test_degeneration_equivalence_temps(self) -> None:
        """When num_module_chunks=1, ModuleChunkGather(0) === TiledGather for temps."""
        ts = TilingScheme(
            num_module_chunks=1,
            num_class_chunks=4,
            total_modules=16,
            total_classes=40,
            padded_hidden_count=128,
            padded_total_output_class_count=48,
        )
        epp = ts.elements_per_tile(ModuleBufferKind.TEMPS)

        mcg = ModuleChunkGather(ts, module_chunk_index=0, buffer_kind=ModuleBufferKind.TEMPS)
        tg = TiledGather(scheme=ts, _elements_per_partial=epp)

        np.testing.assert_array_equal(mcg.get_offsets(), tg.get_offsets())

    def test_boundary_validation_negative_index(self) -> None:
        """Verify module_chunk_index=-1 raises ValueError."""
        ts = TilingScheme(
            num_module_chunks=3,
            num_class_chunks=4,
            total_modules=30,
            total_classes=40,
        )
        with pytest.raises(ValueError, match="module_chunk_index"):
            ModuleChunkGather(ts, module_chunk_index=-1, buffer_kind=ModuleBufferKind.WEIGHTS)

    def test_boundary_validation_index_too_large(self) -> None:
        """Verify module_chunk_index=num_module_chunks raises ValueError."""
        ts = TilingScheme(
            num_module_chunks=3,
            num_class_chunks=4,
            total_modules=30,
            total_classes=40,
        )
        with pytest.raises(ValueError, match="module_chunk_index"):
            ModuleChunkGather(ts, module_chunk_index=3, buffer_kind=ModuleBufferKind.WEIGHTS)

    def test_num_partials_equals_num_class_chunks(self) -> None:
        """num_partials must equal num_class_chunks regardless of module_chunk_index."""
        ts = TilingScheme(
            num_module_chunks=3,
            num_class_chunks=5,
            total_modules=30,
            total_classes=50,
        )
        for m in range(3):
            mcg = ModuleChunkGather(ts, module_chunk_index=m, buffer_kind=ModuleBufferKind.WEIGHTS)
            assert mcg.num_partials == 5

    def test_elements_per_partial_delegates_to_scheme_weights(self) -> None:
        """elements_per_partial must equal TilingScheme.elements_per_tile(WEIGHTS)."""
        ts = TilingScheme(
            num_module_chunks=2,
            num_class_chunks=3,
            total_modules=20,
            total_classes=30,
            padded_hidden_count=256,
            padded_total_output_class_count=32,
        )
        mcg = ModuleChunkGather(ts, module_chunk_index=0, buffer_kind=ModuleBufferKind.WEIGHTS)
        assert mcg.elements_per_partial == ts.elements_per_tile(ModuleBufferKind.WEIGHTS)

    def test_elements_per_partial_delegates_to_scheme_biases(self) -> None:
        """elements_per_partial must equal TilingScheme.elements_per_tile(BIASES)."""
        ts = TilingScheme(
            num_module_chunks=2,
            num_class_chunks=3,
            total_modules=20,
            total_classes=30,
            padded_hidden_count=256,
            padded_total_output_class_count=32,
        )
        mcg = ModuleChunkGather(ts, module_chunk_index=0, buffer_kind=ModuleBufferKind.BIASES)
        assert mcg.elements_per_partial == ts.elements_per_tile(ModuleBufferKind.BIASES)

    def test_elements_per_partial_delegates_to_scheme_temps(self) -> None:
        """elements_per_partial must equal TilingScheme.elements_per_tile(TEMPS)."""
        ts = TilingScheme(
            num_module_chunks=2,
            num_class_chunks=3,
            total_modules=20,
            total_classes=30,
            padded_hidden_count=256,
            padded_total_output_class_count=32,
        )
        mcg = ModuleChunkGather(ts, module_chunk_index=1, buffer_kind=ModuleBufferKind.TEMPS)
        assert mcg.elements_per_partial == ts.elements_per_tile(ModuleBufferKind.TEMPS)
