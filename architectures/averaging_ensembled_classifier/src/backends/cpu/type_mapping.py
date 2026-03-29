"""PrecisionConfig → numpy dtype mapping for CPU backend (ADR-008)."""
from __future__ import annotations

import numpy as np

from ...shared.precision_config import PrecisionConfig


def get_numpy_dtype(precision: PrecisionConfig) -> np.dtype:
    """Map PrecisionConfig to the numpy dtype for buffer allocation."""
    return precision.numpy_dtype


def get_c_type_name(precision: PrecisionConfig) -> str:
    """Map PrecisionConfig to the C type name (diagnostic purposes)."""
    if precision.numpy_dtype == np.float32:
        return "float"
    if precision.numpy_dtype == np.float16:
        return "half"
    raise ValueError(f"Unsupported precision: {precision.numpy_dtype}")
