# src/shared/precision_config.py
"""Backend-neutral precision configuration (ADR-008)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable precision configuration for plan construction.

    Replaces the PrecisionContext ABC hierarchy. Consumed by ModelSpec,
    StabilizationPolicy, and the plan builder.
    """
    numpy_dtype: np.dtype
    fp_format_max: float
    epsilon: float

    @classmethod
    def float32(cls) -> "PrecisionConfig":
        finfo = np.finfo(np.float32)
        return cls(numpy_dtype=np.dtype(np.float32),
                   fp_format_max=float(finfo.max),
                   epsilon=float(finfo.eps))

    @classmethod
    def float16(cls) -> "PrecisionConfig":
        finfo = np.finfo(np.float16)
        return cls(numpy_dtype=np.dtype(np.float16),
                   fp_format_max=float(finfo.max),
                   epsilon=float(finfo.eps))
