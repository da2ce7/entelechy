# workload_primitives.py

"""
A Collection of Foundational, Stateless Primitives for Describing Workloads.

This module provides the primitive data structures used by the planning and
execution layers to describe the shape and layout of computational work. These
classes are pure data containers and contracts, embodying the "descriptor"
pattern, and contain no active logic.

They are foundational "Layer 1" artifacts, serving as the vocabulary for the
higher-level components.
"""
import abc
from dataclasses import dataclass

import numpy as np

# --- Local Type Imports ---
# These are defined here for clarity and to avoid circular dependencies.
SCALAR_UINT_TYPE = np.uint32


# === Section 1: Workload Partitioning (Tiling) Primitives ===


@dataclass(frozen=True)
class WorkTile:
    """Defines a single, independent unit of work for tiled kernels."""

    module_chunk_index: int
    class_chunk_index: int
    flat_tile_index: int
    num_class_chunks: int
    module_chunk_offset: int
    modules_per_chunk: int
    class_chunk_offset: int
    classes_per_chunk: int


@dataclass(frozen=True)
class TilingScheme:
    """
    A factory for producing WorkTile objects that partition a 2D problem space.
    """

    num_module_chunks: int
    num_class_chunks: int
    total_modules: int
    total_classes: int

    @property
    def total_tiles(self) -> int:
        return self.num_module_chunks * self.num_class_chunks

    def __iter__(self):
        for m_idx in range(self.num_module_chunks):
            for c_idx in range(self.num_class_chunks):
                yield self.get_tile(m_idx, c_idx)

    def get_tile(self, m_idx: int, c_idx: int) -> "WorkTile":
        mod_chunk_size = (self.total_modules + self.num_module_chunks - 1) // self.num_module_chunks
        cls_chunk_size = (self.total_classes + self.num_class_chunks - 1) // self.num_class_chunks
        mod_offset = m_idx * mod_chunk_size
        num_mods = min(mod_chunk_size, self.total_modules - mod_offset)
        cls_offset = c_idx * cls_chunk_size
        num_cls = min(cls_chunk_size, self.total_classes - cls_offset)
        flat_idx = m_idx * self.num_class_chunks + c_idx
        return WorkTile(m_idx, c_idx, flat_idx, self.num_class_chunks, mod_offset, num_mods, cls_offset, num_cls)


# === Section 2: The Gather Abstraction for Reductions ===


class GatherPrimitive(abc.ABC):
    """
    An abstract contract representing a collection of partials to be gathered.

    This is a declarative primitive that tells the SystemPlanner *how* a
    set of partials is laid out, allowing it to correctly construct the DAG for
    a reduction sub-graph.
    """

    @property
    @abc.abstractmethod
    def num_partials(self) -> int:
        """The total number of partials in the collection."""
        pass

    @property
    @abc.abstractmethod
    def elements_per_partial(self) -> int:
        """The number of scalar elements in a single partial."""
        pass

    @abc.abstractmethod
    def get_offsets(self) -> np.ndarray:
        """
        Returns a host-side numpy array of uints, where each element is the
        starting OFFSET (in elements, not bytes) of a partial relative to the
        start of its collection buffer.
        """
        pass


@dataclass(frozen=True)
class LinearlyChunkedGather(GatherPrimitive):
    """
    Represents partials scattered into a collection buffer according to a
    linear chunking of a primary dimension (e.g., batch).
    """

    num_chunks: int
    elements_per_chunk: int

    @property
    def num_partials(self) -> int:
        return self.num_chunks

    @property
    def elements_per_partial(self) -> int:
        return self.elements_per_chunk

    def get_offsets(self) -> np.ndarray:
        """The offset is the chunk index multiplied by the element stride."""
        return np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) * self.elements_per_partial


@dataclass(frozen=True)
class TiledGather(GatherPrimitive):
    """
    Represents partials scattered across a collection buffer according to a
    tiled placement strategy (e.g., `grid_mod_cls`).
    """

    scheme: TilingScheme
    _elements_per_partial: int

    @property
    def num_partials(self) -> int:
        return self.scheme.total_tiles

    @property
    def elements_per_partial(self) -> int:
        return self._elements_per_partial

    def get_offsets(self) -> np.ndarray:
        """The offset is simply the tile index multiplied by the element stride."""
        return np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) * self.elements_per_partial


@dataclass(frozen=True)
class ContiguousGather(GatherPrimitive):
    """
    Represents partials that are laid out contiguously in memory, such as the
    output of a previous reduction stage in a ping-pong buffer.
    """

    _num_partials: int
    _elements_per_partial: int

    @property
    def num_partials(self) -> int:
        return self._num_partials

    @property
    def elements_per_partial(self) -> int:
        return self._elements_per_partial

    def get_offsets(self) -> np.ndarray:
        """The offsets are a simple linear progression from the start of the buffer."""
        return np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) * self.elements_per_partial
