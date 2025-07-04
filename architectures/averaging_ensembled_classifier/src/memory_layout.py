# memory_layout.py

"""
The Definitive Implementation of the Contractual Memory Layout Abstraction.

This module provides the classes necessary to explicitly and unambiguously
define the physical memory layout of device buffers, as dictated by the
Padding Contracts within `kernels.cl.h`.

Its abstractions are designed to be composed by a Host Orchestrator and
executed by a `BufferManager`, ensuring that the host-side memory allocation
is a direct, verifiable reflection of device-side requirements.

This module contains:
- PaddingType: A canonical enum for the *reason* of padding.
- PaddingStrategy: An atomic rule for padding a single dimension.
- MemoryLayout: A compositional plan for a buffer's complete physical layout.
"""

import enum
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# --- Foundational Enums and Dataclasses ---


class PaddingType(enum.Enum):
    """The reason for applying a padding strategy, derived from kernel contracts.

    This enum is the canonical, type-safe representation of the `Type` key
    in a kernel's `Padding Contract` block, providing clarity of intent.
    """

    NONE = enum.auto()
    # Pad a dimension to be a multiple of N elements (e.g., for SIMD width).
    ELEMENT_COUNT = enum.auto()
    # Pad a dimension's row stride to be a multiple of N bytes (e.g., for cache alignment).
    BYTE_ALIGNMENT = enum.auto()


@dataclass(frozen=True)
class PaddingStrategy:
    """Defines a single, atomic padding rule for one dimension of a tensor.

    This is the fundamental building block of a memory layout plan. A complex
    layout is constructed by composing one or more of these simple rules.
    """

    # The type of padding rule to apply.
    type: PaddingType
    # The value to pad to (e.g., SIMD_WIDTH, or 128 for byte alignment).
    value: int
    # The index of the dimension to apply this rule to. Conforms to Python's
    # list indexing; e.g., -1 means the last dimension.
    target_dim_idx: int = -1


def _pad_to_multiple(dim: int, multiple: int) -> int:
    """A pure utility function to pad a dimension to the nearest multiple."""
    if multiple is None or multiple == 0:
        # Avoid division by zero; no padding if multiple is zero or None.
        return dim
    return (dim + multiple - 1) // multiple * multiple


# --- The Core Abstraction: The MemoryLayout Class ---


class MemoryLayout:
    """Encapsulates the complete layout plan for a device buffer.

    This object is constructed by the Host Orchestrator and passed to the
    BufferManager. It serves as the explicit, unambiguous instruction for
    how a buffer's memory should be physically arranged, replacing any
    implicit, role-based logic.
    """

    def __init__(self, logical_shape: Tuple[int, ...]):
        """Initializes a layout plan with its base logical shape."""
        if not all(isinstance(d, int) and d >= 0 for d in logical_shape):
            raise ValueError("Logical shape must be a tuple of non-negative integers.")
        self.logical_shape: Tuple[int, ...] = logical_shape
        self.strategies: List[PaddingStrategy] = []

    def add_strategy(self, strategy: PaddingStrategy) -> "MemoryLayout":
        """Applies a padding strategy.

        This method returns `self` to allow for elegant, readable chaining
        of multiple padding rules, enabling the description of complex layouts.
        Example:
            layout = MemoryLayout((10, 17))
                .add_strategy(PaddingStrategy(..., target_dim_idx=0))
                .add_strategy(PaddingStrategy(..., target_dim_idx=1))
        """
        if not isinstance(strategy, PaddingStrategy):
            raise TypeError("Can only add objects of type PaddingStrategy.")
        self.strategies.append(strategy)
        return self

    def get_padded_shape(self, dtype: np.dtype) -> Tuple[int, ...]:
        """Calculates the final physical (padded) shape based on all applied strategies."""
        if not self.strategies:
            return self.logical_shape

        padded_shape = list(self.logical_shape)
        element_size_bytes = dtype().itemsize

        for strategy in self.strategies:
            if strategy.type == PaddingType.NONE:
                continue

            dim_idx = strategy.target_dim_idx
            if dim_idx < 0:
                # Convert negative index (e.g., -1) to a positive list index
                dim_idx += len(padded_shape)

            if not (0 <= dim_idx < len(padded_shape)):
                raise IndexError(
                    f"PaddingStrategy has invalid target_dim_idx {strategy.target_dim_idx} "
                    f"for shape with {len(padded_shape)} dimensions."
                )

            current_dim_size = padded_shape[dim_idx]

            if strategy.type == PaddingType.ELEMENT_COUNT:
                padded_shape[dim_idx] = _pad_to_multiple(current_dim_size, strategy.value)
            elif strategy.type == PaddingType.BYTE_ALIGNMENT:
                # In a row-major layout, padding a row's byte-stride is achieved by
                # increasing the number of elements in the last dimension of that row-slice.
                if dim_idx != len(padded_shape) - 1:
                    raise ValueError(
                        "BYTE_ALIGNMENT padding is only logically sound for the last dimension "
                        "in a contiguous row-major memory layout."
                    )

                row_elements = current_dim_size
                row_bytes = row_elements * element_size_bytes

                # Pad the byte count of the row to the nearest multiple
                padded_row_bytes = _pad_to_multiple(row_bytes, strategy.value)

                # CONTRACTUAL VERIFICATION: The alignment must be a multiple of the element size.
                if padded_row_bytes % element_size_bytes != 0:
                    raise ValueError(
                        f"Byte alignment padding of {strategy.value} bytes is invalid. "
                        f"It results in a padded row of {padded_row_bytes} bytes, which is not "
                        f"divisible by the element size of {element_size_bytes} bytes."
                    )

                # Convert the padded byte count back to a padded element count
                padded_shape[dim_idx] = padded_row_bytes // element_size_bytes

        return tuple(padded_shape)


if __name__ == "__main__":
    # This block serves as a live demonstration and informal verification of the abstraction.

    print("--- Memory Layout Abstraction Demonstration ---")

    # --- Architectural Constants (as would be loaded by the orchestrator) ---
    SIMD_WIDTH = 8
    CACHE_LINE_BYTES = 128
    SCALAR_NP_TYPE = np.float32

    print(f"\nArchitectural Constants:")
    print(f"  - SIMD Width: {SIMD_WIDTH} elements")
    print(f"  - Cache Line: {CACHE_LINE_BYTES} bytes")
    print(f"  - Scalar Type: {SCALAR_NP_TYPE.__name__} ({SCALAR_NP_TYPE().itemsize} bytes)")
    print("-" * 45)

    # --- Case 1: Shared Biases (Simple Element Count Padding) ---
    # Kernel Contract: {Type: SIMD, Formula: "Padded to SIMD_WIDTH"}
    HIDDEN_DIM = 250
    print(f"\nCase 1: Shared Biases (1D Element Padding)")
    biases_layout = MemoryLayout(logical_shape=(HIDDEN_DIM,)).add_strategy(
        PaddingStrategy(type=PaddingType.ELEMENT_COUNT, value=SIMD_WIDTH)
    )
    padded_biases_shape = biases_layout.get_padded_shape(SCALAR_NP_TYPE)
    print(f"  Logical Shape : {biases_layout.logical_shape}")
    print(f"  Strategy      : Pad last dim to {SIMD_WIDTH} elements")
    print(f"  Padded Shape  : {padded_biases_shape}  <-- Correct, 250 padded to 256")
    assert padded_biases_shape == (256,)

    # --- Case 2: Input Data (Byte Alignment Padding) ---
    # Kernel Contract: {Type: CACHE, Formula: "Pad row stride to 128-byte alignment"}
    BATCH_SIZE = 32
    INPUT_DIM = 75
    print(f"\nCase 2: Input Data (2D Byte Alignment)")
    input_layout = MemoryLayout(logical_shape=(BATCH_SIZE, INPUT_DIM)).add_strategy(
        PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=CACHE_LINE_BYTES)
    )
    padded_input_shape = input_layout.get_padded_shape(SCALAR_NP_TYPE)
    print(f"  Logical Shape : {input_layout.logical_shape}")
    print(f"  Strategy      : Pad last dim's byte stride to {CACHE_LINE_BYTES} bytes")
    # Verification math: 75 elements * 4 bytes/element = 300 bytes.
    # Padded to next multiple of 128 is 384 bytes.
    # 384 bytes / 4 bytes/element = 96 elements.
    print(f"  Padded Shape  : {padded_input_shape}  <-- Correct, (32, 96)")
    assert padded_input_shape == (32, 96)

    # --- Case 3: Shared Weights (Compound Padding) ---
    # A hypothetical contract requiring two strategies on the same buffer.
    print(f"\nCase 3: Shared Weights (2D Compound Padding)")
    weights_layout = (
        MemoryLayout(logical_shape=(INPUT_DIM, HIDDEN_DIM))
        .add_strategy(
            PaddingStrategy(
                type=PaddingType.ELEMENT_COUNT,
                value=SIMD_WIDTH,
                target_dim_idx=1,  # Pad the second dimension (hidden_dim) first
            )
        )
        .add_strategy(
            PaddingStrategy(
                type=PaddingType.BYTE_ALIGNMENT,
                value=CACHE_LINE_BYTES,
                target_dim_idx=1,  # Then, pad the same dimension's stride for cache
            )
        )
    )
    padded_weights_shape = weights_layout.get_padded_shape(SCALAR_NP_TYPE)
    print(f"  Logical Shape : {weights_layout.logical_shape}")
    print(f"  Strategy 1    : Pad dim 1 to {SIMD_WIDTH} elements -> shape becomes (75, 256)")
    print(f"  Strategy 2    : Pad dim 1's byte stride to {CACHE_LINE_BYTES} bytes")
    # Verification math: (75, 250) -> element pad -> (75, 256).
    # Stride of dim 1 is now 256 elements * 4 bytes/element = 1024 bytes.
    # 1024 is already a multiple of 128, so byte alignment padding has no further effect.
    print(f"  Padded Shape  : {padded_weights_shape}  <-- Correct, (75, 256)")
    assert padded_weights_shape == (75, 256)

    print("\n--- All demonstration cases passed. Abstraction is sound. ---")
