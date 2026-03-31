# model_spec.py

"""
Backend-neutral model specification (ADR-008).

ModelSpec is a frozen dataclass with PrecisionConfig composition.
Backward-compatible properties delegate to self.precision for code
that references the old PrecisionContext ABC interface.
"""

from dataclasses import dataclass
from typing import Any, Type

import numpy as np

from .precision_config import PrecisionConfig


@dataclass(frozen=True)
class ModelSpec:
    """Backend-neutral model configuration (ADR-008).

    PrecisionConfig composition replaces PrecisionContext ABC inheritance.
    """
    precision: PrecisionConfig
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    simd_width: int
    cache_line_bytes: int

    # ------------------------------------------------------------------
    # Backward-compatible properties (deprecated — use self.precision.*)
    # ------------------------------------------------------------------

    @property
    def SCALAR_NP_TYPE(self) -> Type[np.floating]:
        """Deprecated: Use self.precision.numpy_dtype."""
        return self.precision.numpy_dtype.type

    @property
    def SCALAR_C_TYPE_NAME(self) -> str:
        """Deprecated: Use self.precision.numpy_dtype."""
        if self.precision.numpy_dtype == np.dtype(np.float16):
            return "half"
        return "float"

    # ------------------------------------------------------------------
    # Padding calculations (unchanged from Phase 0)
    # ------------------------------------------------------------------

    @property
    def padded_hidden_dim(self) -> int:
        if self.simd_width <= 0:
            return self.hidden_dim
        return (self.hidden_dim + self.simd_width - 1) // self.simd_width * self.simd_width

    @property
    def padded_input_dim(self) -> int:
        if self.cache_line_bytes <= 0:
            return self.input_dim
        item_size = np.dtype(self.precision.numpy_dtype).itemsize
        row_bytes = self.input_dim * item_size
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        if padded_row_bytes % item_size != 0:
            raise ValueError("Cache line padding resulted in indivisible byte count for dtype.")
        return padded_row_bytes // item_size

    @property
    def padded_class_dim(self) -> int:
        if self.cache_line_bytes <= 0:
            return self.output_classes
        item_size = np.dtype(self.precision.numpy_dtype).itemsize
        row_bytes = self.output_classes * item_size
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        if padded_row_bytes % item_size != 0:
            raise ValueError("Cache line padding resulted in indivisible byte count for dtype.")
        return padded_row_bytes // item_size

    @property
    def padded_module_dim(self) -> int:
        if self.cache_line_bytes <= 0:
            return self.num_modules
        item_size = np.dtype(self.precision.numpy_dtype).itemsize
        row_bytes = self.num_modules * item_size
        padded_row_bytes = (row_bytes + self.cache_line_bytes - 1) // self.cache_line_bytes * self.cache_line_bytes
        if padded_row_bytes % item_size != 0:
            raise ValueError("Cache line padding resulted in indivisible byte count for dtype.")
        return padded_row_bytes // item_size

    # ------------------------------------------------------------------
    # Factory classmethods (replace Float32ModelSpec / Float16ModelSpec)
    # ------------------------------------------------------------------

    @classmethod
    def float32(cls, **kwargs: Any) -> "ModelSpec":
        return cls(precision=PrecisionConfig.float32(), **kwargs)

    @classmethod
    def float16(cls, **kwargs: Any) -> "ModelSpec":
        return cls(precision=PrecisionConfig.float16(), **kwargs)
