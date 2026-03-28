# memory_layout.py

"""
The Contractual Memory Layout Abstraction.

Jurisdictional Mandate:
This module provides the canonical, compositional abstractions for translating a
tensor's logical, human-intelligible shape into its final, device-optimized
physical memory layout. Its sole jurisdiction is to serve as the physical
manifestation of the `Padding Contract` rules defined within the system's C-level
kernel headers (`kernels.cl.h`).

Architectural Role:
This module provides the primitive 'nouns'—`PaddingType`, `PaddingStrategy`, and
`MemoryLayout`—that allow the `ParameterSpace` to declaratively specify memory
requirements. The `MemoryLayout` object is the final blueprint, consumed by the
`BufferManager`, which executes the allocation. This architecture upholds the
principle of "explicit is better than implicit," replacing scattered, ad-hoc
padding logic with a centralized, verifiable, and compositional planning system.
"""

import enum
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# =========================================================================
# === Foundational Primitives (The Vocabulary of Physical Form)         ===
# =========================================================================


class PaddingType(enum.Enum):
    """
    The canonical, type-safe representation of the *reason* for applying padding.

    Contractual Role:
    This enum translates the string literals from a kernel's `Padding Contract`
    block into a controlled, explicit vocabulary, ensuring clarity of intent and
    preventing errors from ambiguous or misspelled strategy types.
    """

    # WHY: Represents the null case where no padding is required.
    NONE = enum.auto()
    # WHY: Represents padding to meet a specific element count, typically to
    # align a dimension with the hardware's natural SIMD vector width.
    ELEMENT_COUNT = enum.auto()
    # WHY: Represents padding a row's total size in bytes to a boundary,
    # typically to align with a hardware cache line for optimal memory access.
    BYTE_ALIGNMENT = enum.auto()


@dataclass(frozen=True)
class PaddingStrategy:
    """
    Defines a single, atomic padding rule for one dimension of a tensor.

    Contractual Role:
    This is the fundamental, immutable building block of a memory layout plan.
    It is a pure data contract, carrying a single, unambiguous instruction. A
    complex `MemoryLayout` is constructed by composing these simple rules.
    """

    # WHY: Enforces that the strategy's purpose is chosen from the controlled
    # vocabulary defined in `PaddingType`, preventing ambiguity.
    type: PaddingType
    # WHY: Carries the primitive integer value of the constraint (e.g., the
    # SIMD width `8`, or the cache line size `128` bytes).
    value: int
    # WHY: Explicitly declares which tensor dimension this rule applies to,
    # conforming to standard list indexing (e.g., -1 for the last dimension).
    # This removes all ambiguity in multi-dimensional padding scenarios.
    target_dim_idx: int = -1


def _pad_to_multiple(dim: int, multiple: int) -> int:
    """A pure, stateless utility to calculate the next highest multiple."""
    # WHY: A defensive guard. If the padding multiple is zero, it would cause a
    # division-by-zero error. This ensures the function is robust.
    if multiple is None or multiple == 0:
        return dim
    return (dim + multiple - 1) // multiple * multiple


# =========================================================================
# === The Core Abstraction: The Declarative Memory Layout Plan          ===
# =========================================================================


class MemoryLayout:
    """
    Encapsulates the complete, compositional layout plan for a device buffer.

    Contractual Role:
    This object is the declarative blueprint passed from the `ParameterSpace` to
    the `BufferManager`. It serves as the explicit, unambiguous instruction for
    how a buffer's memory must be physically arranged.
    """

    def __init__(self, logical_shape: Tuple[int, ...]):
        """Initializes a layout plan with its base logical shape."""
        # WHY: A defensive check to enforce the contract that a logical shape
        # must be composed of non-negative integers from the moment of creation.
        if not all(isinstance(d, int) and d >= 0 for d in logical_shape):
            raise ValueError("Logical shape must be a tuple of non-negative integers.")
        self.logical_shape: Tuple[int, ...] = logical_shape
        self.strategies: List[PaddingStrategy] = []

    def add_strategy(self, strategy: PaddingStrategy) -> "MemoryLayout":
        """
        Applies a padding strategy to the plan.

        Architectural Mandate:
        This method returns `self` to enable a fluent, compositional interface.
        This allows the `ParameterSpace` to describe complex, multi-rule layouts
        in a highly readable, declarative style, improving maintainability.
        """
        # WHY: Ensures that only valid `PaddingStrategy` objects can be added
        # to the plan, upholding the integrity of the layout contract.
        if not isinstance(strategy, PaddingStrategy):
            raise TypeError("Can only add objects of type PaddingStrategy.")
        self.strategies.append(strategy)
        return self

    def get_padded_shape(self, dtype: np.dtype) -> Tuple[int, ...]:
        """
        Synthesizes the final physical (padded) shape from all applied strategies.

        Architectural Mandate:
        This is the method where the declarative plan is made concrete. It is
        contractually obligated to apply all strategies in the order they were
        added, allowing for the correct resolution of compound padding rules.
        """
        # WHY: A trivial performance optimization. If there are no strategies,
        # the physical shape is identical to the logical shape.
        if not self.strategies:
            return self.logical_shape

        padded_shape = list(self.logical_shape)
        # WHY: The size of the fundamental data type is essential for any
        # byte-based alignment calculations. This makes the method self-sufficient.
        element_size_bytes = np.dtype(dtype).itemsize

        for strategy in self.strategies:
            if strategy.type == PaddingType.NONE:
                continue

            dim_idx = strategy.target_dim_idx
            if dim_idx < 0:
                dim_idx += len(padded_shape)

            # WHY: A defensive check ensuring the strategy is valid for the rank
            # of the tensor being planned, preventing runtime indexing errors.
            if not (0 <= dim_idx < len(padded_shape)):
                raise IndexError(
                    f"PaddingStrategy has invalid target_dim_idx {strategy.target_dim_idx} "
                    f"for shape with {len(padded_shape)} dimensions."
                )

            current_dim_size = padded_shape[dim_idx]

            if strategy.type == PaddingType.ELEMENT_COUNT:
                padded_shape[dim_idx] = _pad_to_multiple(current_dim_size, strategy.value)

            elif strategy.type == PaddingType.BYTE_ALIGNMENT:
                # WHY: This check enforces a critical architectural constraint.
                # For a contiguous, row-major memory layout, padding a row's
                # byte-stride is only physically achievable by increasing the
                # number of elements in the *last* dimension of that row-slice.
                # This check makes a subtle but fundamental rule explicit.
                if dim_idx != len(padded_shape) - 1:
                    raise ValueError(
                        "BYTE_ALIGNMENT padding is only logically sound for the last dimension "
                        "in a contiguous row-major memory layout."
                    )

                row_elements = current_dim_size
                row_bytes = row_elements * element_size_bytes
                padded_row_bytes = _pad_to_multiple(row_bytes, strategy.value)

                # WHY: A CONTRACTUAL VERIFICATION. This is a critical sanity check.
                # It is physically impossible for a padded row's byte count to
                # not be perfectly divisible by the size of a single element. This
                # check prevents the creation of a physically unrealizable plan.
                if padded_row_bytes % element_size_bytes != 0:
                    raise ValueError(
                        f"Byte alignment padding of {strategy.value} bytes is invalid. "
                        f"It results in a padded row of {padded_row_bytes} bytes, which is not "
                        f"divisible by the element size of {element_size_bytes} bytes."
                    )

                padded_shape[dim_idx] = padded_row_bytes // element_size_bytes

        return tuple(padded_shape)
