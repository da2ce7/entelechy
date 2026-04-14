# src/shared/workload_primitives.py
"""Foundational, stateless primitives for workload decomposition.

This module provides the vocabulary that the Policy tier's plan-construction
logic uses to describe how computational work is partitioned, how partial
results are scattered through collection buffers, and how parameter buffers
are sliced for per-chunk optimizer dispatch.

Every type is a frozen dataclass — an immutable descriptor carrying no device
references, no mutable state, and no active computation beyond host-side
offset arithmetic.

Architectural contracts encoded
───────────────────────────────
**Placement Contract** (CONTRACT §3.5):
    ``TilingGeometry`` produces tile coordinates whose flat indices serve
    as placement keys for Partial Renderer kernels (``grid_mod_cls``).

**Indirection Contract** (CONCEPT.md §2):
    ``GatherDescriptor`` subclasses produce the element-offset arrays that
    the Recursive Reduction Engine uses to locate scattered partials
    without intermediate device-to-device copies.

**Module-Chunk Isolation Invariant** (ADR-030):
    ``ModuleChunkGather`` structurally restricts each reduction tree's
    offset list to tiles belonging to a single module chunk.
    Cross-module-chunk reduction is architecturally prohibited.

Organisation
────────────
§1  ``ChunkDecomposition`` — 1-D dimension partitioning.
§2  ``TileCoord``, ``TilingGeometry`` — 2-D (module × class) grid.
§3  ``GatherDescriptor``, ``StridedGather``, ``ModuleChunkGather`` —
    reduction-engine offset-list construction.
§4  ``ParameterSlice`` — optimizer slice-addressed dispatch geometry.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterator

import numpy as np

__all__ = [
    "ChunkDecomposition",
    "GatherDescriptor",
    "ModuleChunkGather",
    "ParameterSlice",
    "StridedGather",
    "TileCoord",
    "TilingGeometry",
]

_U32 = np.uint32


# ─── Internal ────────────────────────────────────────────────────────


def _ceildiv(a: int, b: int) -> int:
    """⌈a / b⌉.  *b* must be positive."""
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════════
# §1  Chunk Decomposition — 1-D dimension partitioning
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ChunkDecomposition:
    """Describes how *total* elements are split into *num_chunks* chunks.

    Chunks have equal maximum size (ceiling division); the last chunk
    may contain fewer elements.  This is the universal building block
    for batch chunking, module chunking, class chunking, and streaming
    iteration — any 1-D dimension that the Host Orchestrator partitions.

    Parameters
    ----------
    total:
        Number of elements in the dimension (≥ 0).
    num_chunks:
        Number of partitions (≥ 1).

    Examples
    --------
    >>> d = ChunkDecomposition(total=10, num_chunks=3)
    >>> d.chunk_size
    4
    >>> [d.count_for(i) for i in range(3)]
    [4, 4, 2]
    """

    total: int
    num_chunks: int

    def __post_init__(self) -> None:
        if self.total < 0:
            raise ValueError(f"total must be ≥ 0, got {self.total}")
        if self.num_chunks < 1:
            raise ValueError(f"num_chunks must be ≥ 1, got {self.num_chunks}")

    # ── Derived geometry ──────────────────────────────────────────

    @property
    def chunk_size(self) -> int:
        """Maximum elements per chunk: ⌈total / num_chunks⌉."""
        return _ceildiv(self.total, self.num_chunks) if self.total else 0

    def offset_for(self, index: int) -> int:
        """Starting element offset for chunk *index*."""
        return index * self.chunk_size

    def count_for(self, index: int) -> int:
        """Actual element count for chunk *index* (last may be smaller)."""
        return max(0, min(self.chunk_size, self.total - self.offset_for(index)))

    # ── Factories ─────────────────────────────────────────────────

    @classmethod
    def from_chunk_size(cls, total: int, chunk_size: int) -> ChunkDecomposition:
        """Construct from a desired *chunk_size*, deriving *num_chunks*.

        >>> ChunkDecomposition.from_chunk_size(total=10, chunk_size=4)
        ChunkDecomposition(total=10, num_chunks=3)
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")
        return cls(total=total, num_chunks=max(1, _ceildiv(total, chunk_size)))

    @classmethod
    def single(cls, total: int) -> ChunkDecomposition:
        """One chunk containing all elements (no decomposition)."""
        return cls(total=total, num_chunks=1)

    @classmethod
    def per_element(cls, total: int) -> ChunkDecomposition:
        """One chunk per element (finest granularity).

        Used for the True Streaming backpropagation model (Nodes 17–19),
        where each streaming iteration processes one sample.
        """
        return cls(total=total, num_chunks=max(1, total))


# ═══════════════════════════════════════════════════════════════════
# §2  Tiling Geometry — 2-D (module × class) grid
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TileCoord:
    """A tile's coordinate and derived extent in the problem grid.

    Created by :meth:`TilingGeometry.tile`; not typically constructed
    directly.  Fields mirror the kernel scalar parameters used by the
    ``grid_mod_cls`` placement strategy (CONTRACT §3.5.3).
    """

    module_chunk: int       # Module-chunk ordinal
    class_chunk: int        # Class-chunk ordinal
    flat_index: int         # Row-major: mc × num_class_chunks + cc

    module_offset: int      # First module in this chunk
    module_count: int       # Modules in this chunk (≤ modules_per_chunk)
    class_offset: int       # First class in this chunk
    class_count: int        # Classes in this chunk (≤ classes_per_chunk)


@dataclass(frozen=True)
class TilingGeometry:
    """Partitions the (module × class) problem space into tiles.

    Composed from two independent :class:`ChunkDecomposition` instances.
    Tiles are indexed in row-major order::

        flat_index = module_chunk × classes.num_chunks + class_chunk

    The total tile count is ``modules.num_chunks × classes.num_chunks``.

    This class owns the decomposition geometry.  It does *not* carry
    buffer-specific information (padded dimensions, element counts per
    tile) — those are computed by the plan builder from ``ModelSpec``.

    Parameters
    ----------
    modules:
        Module-dimension decomposition.
    classes:
        Class-dimension decomposition.
    """

    modules: ChunkDecomposition
    classes: ChunkDecomposition

    # ── Core geometry ─────────────────────────────────────────────

    @property
    def total_tiles(self) -> int:
        """Total number of tiles in the grid."""
        return self.modules.num_chunks * self.classes.num_chunks

    # ── Tile access ───────────────────────────────────────────────

    def tile(self, module_chunk: int, class_chunk: int) -> TileCoord:
        """Build a :class:`TileCoord` for a specific grid position."""
        return TileCoord(
            module_chunk=module_chunk,
            class_chunk=class_chunk,
            flat_index=module_chunk * self.classes.num_chunks + class_chunk,
            module_offset=self.modules.offset_for(module_chunk),
            module_count=self.modules.count_for(module_chunk),
            class_offset=self.classes.offset_for(class_chunk),
            class_count=self.classes.count_for(class_chunk),
        )

    def tiles(self) -> Iterator[TileCoord]:
        """All tiles in flat-index (row-major) order."""
        for mc in range(self.modules.num_chunks):
            for cc in range(self.classes.num_chunks):
                yield self.tile(mc, cc)

    def tiles_for_module_chunk(
        self, module_chunk: int
    ) -> tuple[TileCoord, ...]:
        """Tiles belonging to a single module chunk (ADR-030).

        Because tiles are row-major, a module chunk's tiles are a
        contiguous subsequence of length ``classes.num_chunks``.
        """
        return tuple(
            self.tile(module_chunk, cc)
            for cc in range(self.classes.num_chunks)
        )

    # ── Gather factories ──────────────────────────────────────────

    def gather_all_tiles(self, elements_per_tile: int) -> StridedGather:
        """A :class:`StridedGather` over every tile in the grid.

        Used for diagnostic reductions (Node 14) where all tiles'
        partials are summed regardless of module-chunk affiliation.
        """
        return StridedGather(
            count=self.total_tiles,
            partial_width=elements_per_tile,
        )

    def gather_module_chunk(
        self, module_chunk: int, elements_per_tile: int
    ) -> ModuleChunkGather:
        """A :class:`ModuleChunkGather` scoped to one module chunk.

        Used for parameter-gradient reduction trees (Nodes 15, optimizer
        paths) that must respect the Module-Chunk Isolation Invariant
        (ADR-030).
        """
        return ModuleChunkGather(
            geometry=self,
            module_chunk=module_chunk,
            elements_per_tile=elements_per_tile,
        )


# ═══════════════════════════════════════════════════════════════════
# §3  Gather Descriptors — reduction engine offset-list contract
# ═══════════════════════════════════════════════════════════════════


class GatherDescriptor(abc.ABC):
    """Abstract layout declaration for partials in a collection buffer.

    The Recursive Reduction Engine (CONCEPT.md §2) locates partial
    results in a flat collection buffer via an element-offset array —
    the Indirection Contract.  A ``GatherDescriptor`` produces that
    array, enabling scatter→gather reduction without intermediate
    device-to-device copies.

    Subclasses encode specific scatter patterns; the Policy tier
    selects the appropriate subclass at plan-construction time.
    """

    @property
    @abc.abstractmethod
    def num_partials(self) -> int:
        """Number of partial results in the collection."""

    @property
    @abc.abstractmethod
    def elements_per_partial(self) -> int:
        """Scalar element count per partial vector."""

    @abc.abstractmethod
    def offsets(self) -> np.ndarray:
        """Element-offset array (``uint32``) into the collection buffer.

        Returns a 1-D ``ndarray`` of length :attr:`num_partials`.
        Entry *i* is the starting element index of partial *i*.
        The reduction kernel reads :attr:`elements_per_partial`
        consecutive elements starting at each offset.
        """


@dataclass(frozen=True)
class StridedGather(GatherDescriptor):
    r"""Uniformly-spaced partials in a collection buffer.

    Offset formula::

        offset[i] = base_element + i × partial_width

    This single class subsumes three formerly distinct patterns:

    * **Linear batch chunking** — streaming-path partials
      (``StridedGather(num_chunks, elements_per_chunk)``).
    * **All-tile diagnostic** — every tile in the grid
      (``StridedGather(total_tiles, elements_per_tile)``).
    * **Prior-stage contiguous** — ping-pong reduction intermediates
      (``StridedGather(num_nodes, elements_per_node)``).

    Parameters
    ----------
    count:
        Number of partials (≥ 0).
    partial_width:
        Element count per partial; also the element stride between
        consecutive partial starts (≥ 1).
    base_element:
        Element offset of the first partial (default 0).
    """

    count: int
    partial_width: int
    base_element: int = 0

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError(f"count must be ≥ 0, got {self.count}")
        if self.partial_width < 1:
            raise ValueError(
                f"partial_width must be ≥ 1, got {self.partial_width}"
            )
        if self.base_element < 0:
            raise ValueError(
                f"base_element must be ≥ 0, got {self.base_element}"
            )

    @property
    def num_partials(self) -> int:
        return self.count

    @property
    def elements_per_partial(self) -> int:
        return self.partial_width

    def offsets(self) -> np.ndarray:
        a = np.arange(self.count, dtype=_U32) * _U32(self.partial_width)
        if self.base_element:
            a += _U32(self.base_element)
        return a


@dataclass(frozen=True)
class ModuleChunkGather(GatherDescriptor):
    """Gathers only the tiles belonging to a single module chunk.

    Structural encoding of the **Module-Chunk Isolation Invariant**
    (ADR-030): the resulting offset list references only tiles whose
    flat indices fall within one module chunk's contiguous row in the
    tiling grid.  Cross-module-chunk reduction is architecturally
    prohibited.

    Because tiles are row-major::

        flat_index ∈ [mc × C,  mc × C + C)       C = classes.num_chunks
        offset[i]  = (mc × C + i) × elements_per_tile

    Parameters
    ----------
    geometry:
        The tiling geometry governing tile layout.
    module_chunk:
        Module-chunk ordinal to gather (0-indexed).
    elements_per_tile:
        Element count per tile in the collection buffer.
    """

    geometry: TilingGeometry
    module_chunk: int
    elements_per_tile: int

    def __post_init__(self) -> None:
        n = self.geometry.modules.num_chunks
        if not 0 <= self.module_chunk < n:
            raise ValueError(
                f"module_chunk {self.module_chunk} out of [0, {n})"
            )
        if self.elements_per_tile < 1:
            raise ValueError(
                f"elements_per_tile must be ≥ 1, got {self.elements_per_tile}"
            )

    @property
    def num_partials(self) -> int:
        """One partial per class chunk within this module chunk."""
        return self.geometry.classes.num_chunks

    @property
    def elements_per_partial(self) -> int:
        return self.elements_per_tile

    def offsets(self) -> np.ndarray:
        base = self.module_chunk * self.geometry.classes.num_chunks
        return (
            (np.arange(self.num_partials, dtype=_U32) + _U32(base))
            * _U32(self.elements_per_tile)
        )


# ═══════════════════════════════════════════════════════════════════
# §4  Parameter Slice — optimizer dispatch geometry (ADR-030)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParameterSlice:
    """Contiguous sub-range ``[offset, offset + count)`` of a parameter buffer.

    Nodes 24 and 25 (``adam_update``, ``clamp_temperatures``) address a
    slice within the full parameter buffer via ``(parameter_offset,
    parameter_count, total_parameter_count)``.  Per-module-chunk dispatch
    (ADR-030) produces one ``ParameterSlice`` per chunk.

    Invariant: ``offset + count ≤ total``.

    Parameters
    ----------
    offset:
        Element index of the first parameter in the slice.
    count:
        Number of elements in the slice.
    total:
        Total elements in the full (un-sliced) parameter buffer.
    """

    offset: int
    count: int
    total: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"offset must be ≥ 0, got {self.offset}")
        if self.count < 0:
            raise ValueError(f"count must be ≥ 0, got {self.count}")
        if self.offset + self.count > self.total:
            raise ValueError(
                f"[{self.offset}, {self.offset + self.count}) "
                f"exceeds total {self.total}"
            )

    @property
    def end(self) -> int:
        """One-past-the-end element index."""
        return self.offset + self.count

    # ── Factories ─────────────────────────────────────────────────

    @classmethod
    def from_chunk(
        cls,
        decomp: ChunkDecomposition,
        chunk_index: int,
        elements_per_item: int,
    ) -> ParameterSlice:
        """Derive a slice from a :class:`ChunkDecomposition`.

        Maps chunk *chunk_index*'s item range to an element range
        in a buffer of ``decomp.total × elements_per_item`` elements.

        Parameters
        ----------
        decomp:
            The dimension decomposition (e.g. module chunking).
        chunk_index:
            Which chunk to slice.
        elements_per_item:
            Element count per logical item (e.g.
            ``padded_hidden × padded_classes`` for a per-module weight
            buffer).

        Examples
        --------
        >>> modules = ChunkDecomposition(total=5, num_chunks=2)
        >>> ParameterSlice.from_chunk(modules, 0, elements_per_item=100)
        ParameterSlice(offset=0, count=300, total=500)
        >>> ParameterSlice.from_chunk(modules, 1, elements_per_item=100)
        ParameterSlice(offset=300, count=200, total=500)
        """
        item_off = decomp.offset_for(chunk_index)
        item_cnt = decomp.count_for(chunk_index)
        return cls(
            offset=item_off * elements_per_item,
            count=item_cnt * elements_per_item,
            total=decomp.total * elements_per_item,
        )

    @classmethod
    def full(cls, total: int) -> ParameterSlice:
        """Slice spanning the entire buffer (``num_module_chunks == 1``)."""
        return cls(offset=0, count=total, total=total)
