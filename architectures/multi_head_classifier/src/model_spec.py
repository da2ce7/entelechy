# model_spec.py

"""
The Definitive Implementation of the Model Specification Primitive.

This module provides the `ModelSpec` class, a foundational abstraction that
encapsulates the architectural "shape" of the neural network. Its sole purpose
is to serve as the single source of truth for the model's logical dimensions
(e.g., input, hidden, output) and to provide the translation of these into
the physical (padded) dimensions required by the device kernels.

By isolating this primitive, we decouple the model's static architecture from
the dynamic training and execution logic.
"""

from dataclasses import dataclass, field
import numpy as np

# A shared type definition is appropriate here as it's part of the spec's contract.
SCALAR_DTYPE = np.float32


@dataclass(frozen=True)
class ModelSpec:
    """
    A primitive that defines the fundamental architectural shape of the model.

    This object is the single source of truth for the model's dimensions and
    is responsible for calculating the physical (padded) dimensions required
    by the device kernels, based on architectural constraints.
    """

    # Logical (unpadded) dimensions
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int

    # Architectural constraints for padding calculations
    simd_width: int
    cache_line_bytes: int
    scalar_dtype: np.dtype = SCALAR_DTYPE

    @field(init=False)
    @property
    def padded_hidden_dim(self) -> int:
        """Pads the hidden dimension to be an even multiple of the SIMD width."""
        if self.simd_width == 0:
            return self.hidden_dim
        return (self.hidden_dim + self.simd_width - 1) // self.simd_width * self.simd_width

    @field(init=False)
    @property
    def padded_input_dim(self) -> int:
        """Pads the input dimension's row stride for cache-line alignment."""
        if self.cache_line_bytes == 0:
            return self.input_dim
        row_bytes = self.input_dim * self.scalar_dtype().itemsize
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        return padded_row_bytes // self.scalar_dtype().itemsize

    @field(init=False)
    @property
    def padded_class_dim(self) -> int:
        """Pads the class dimension's row stride for cache-line alignment."""
        if self.cache_line_bytes == 0:
            return self.output_classes
        row_bytes = self.output_classes * self.scalar_dtype().itemsize
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        return padded_row_bytes // self.scalar_dtype().itemsize


if __name__ == "__main__":
    # This block serves as a live demonstration and unit test of the abstraction.
    print("--- ModelSpec Abstraction: Live Demonstration & Verification ---")

    # --- Define a hypothetical model architecture ---
    INPUT_FEATURES = 75
    HIDDEN_UNITS = 250
    OUTPUT_LABELS = 17
    MODULE_COUNT = 32
    SIMD = 8
    CACHE = 128

    # --- Create the specification object ---
    spec = ModelSpec(
        input_dim=INPUT_FEATURES,
        hidden_dim=HIDDEN_UNITS,
        output_classes=OUTPUT_LABELS,
        num_modules=MODULE_COUNT,
        simd_width=SIMD,
        cache_line_bytes=CACHE,
    )
    print(f"\nCreated ModelSpec for a {spec.input_dim} -> {spec.hidden_dim} -> {spec.output_classes} model.")
    print(f"  - Logical hidden dim: {spec.hidden_dim}")
    print(f"  - Padded hidden dim:  {spec.padded_hidden_dim} (to be a multiple of {spec.simd_width})")
    print(f"\n  - Logical input dim:  {spec.input_dim}")
    print(f"  - Padded input dim:   {spec.padded_input_dim} (row stride padded to {spec.cache_line_bytes} bytes)")
    print(f"\n  - Logical class dim:  {spec.output_classes}")
    print(f"  - Padded class dim:   {spec.padded_class_dim} (row stride padded to {spec.cache_line_bytes} bytes)")

    # --- Verification ---
    # 250 padded to a multiple of 8 is 256.
    assert spec.padded_hidden_dim == 256
    # An input row is 75 elements * 4 bytes/element = 300 bytes.
    # To align to a 128-byte boundary, pad to 384 bytes.
    # 384 bytes / 4 bytes/element = 96 elements.
    assert spec.padded_input_dim == 96
    # A class row is 17 elements * 4 bytes/element = 68 bytes.
    # To align to a 128-byte boundary, pad to 128 bytes.
    # 128 bytes / 4 bytes/element = 32 elements.
    assert spec.padded_class_dim == 32

    print("\n--- All demonstration cases passed. Abstraction is sound. ---")
