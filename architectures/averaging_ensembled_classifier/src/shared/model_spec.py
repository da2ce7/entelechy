# model_spec.py

"""
This module provides the `ModelSpec` class
hierarchy. It has been refactored to embody its role as a precision-aware,
abstract contract for the model's logical architecture.

Key Architectural Guarantees:
- `ModelSpec` is now an Abstract Base Class inheriting from `PrecisionContext`.
  This enforces the rule that any model specification MUST have a defined
  precision, making it impossible to create an ambiguous, "precision-less" spec.
- The padding calculation properties (`padded_..._dim`) are now fully
  self-contained. They correctly use the `SCALAR_NP_TYPE` property provided by
  their own inherited precision context, eliminating dependencies on external
  objects for type information.
- This creates a verifiable, compile-time link between the logical blueprint
  (the model's shape) and its physical memory requirements on the device.
"""

import abc
from dataclasses import dataclass, field
import numpy as np

from ..arch_primitives import PrecisionContext, Float32Context, Float16Context

@dataclass(frozen=True)
class ModelSpec(PrecisionContext, abc.ABC):
    """
    An *abstract* contract for the model's logical architecture.
    """
    # Logical (unpadded) dimensions
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int

    # Architectural constraints for padding calculations
    simd_width: int
    cache_line_bytes: int

    @property
    def padded_hidden_dim(self) -> int:
        """Pads the hidden dimension to be an even multiple of the SIMD width."""
        if self.simd_width <= 0:
            return self.hidden_dim
        return (self.hidden_dim + self.simd_width - 1) // self.simd_width * self.simd_width

    @property
    def padded_input_dim(self) -> int:
        """Pads the input dimension's row stride for cache-line alignment."""
        if self.cache_line_bytes <= 0:
            return self.input_dim
        # This now correctly and safely uses the SCALAR_NP_TYPE from its own context.
        item_size = self.SCALAR_NP_TYPE().itemsize
        row_bytes = self.input_dim * item_size
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        # Ensure the padded size is still a valid multiple of the element size.
        if padded_row_bytes % item_size != 0:
            raise ValueError(f"Cache line padding resulted in indivisible byte count for dtype.")
        return padded_row_bytes // item_size

    @property
    def padded_class_dim(self) -> int:
        """Pads the class dimension's row stride for cache-line alignment."""
        if self.cache_line_bytes <= 0:
            return self.output_classes
        item_size = self.SCALAR_NP_TYPE().itemsize
        row_bytes = self.output_classes * item_size
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        if padded_row_bytes % item_size != 0:
            raise ValueError(f"Cache line padding resulted in indivisible byte count for dtype.")
        return padded_row_bytes // item_size

    @property
    def padded_module_dim(self) -> int:
        """Pads the module dimension's row stride for cache-line alignment."""
        if self.cache_line_bytes <= 0:
            return self.num_modules
        item_size = self.SCALAR_NP_TYPE().itemsize
        row_bytes = self.num_modules * item_size
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        if padded_row_bytes % item_size != 0:
            raise ValueError(f"Cache line padding resulted in indivisible byte count for dtype.")
        return padded_row_bytes // item_size


@dataclass(frozen=True)
class Float32ModelSpec(ModelSpec, Float32Context):
    """A concrete model specification for 32-bit float precision."""
    pass


@dataclass(frozen=True)
class Float16ModelSpec(ModelSpec, Float16Context):
    """A concrete model specification for 16-bit float precision."""
    pass
