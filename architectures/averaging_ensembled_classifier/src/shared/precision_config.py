# src/shared/precision_config.py
"""Backend-neutral precision configuration (ADR-008, ADR-020, ADR-022 §1)."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable three-role precision configuration for plan construction.

    Three independent dtype roles separate bandwidth, arithmetic fidelity,
    and optimizer-state stability concerns (ADR-020 §4.1):
      - storage_dtype:  element type for storage-role buffers (bandwidth lever)
      - compute_dtype:  element type for arithmetic (fidelity lever)
      - state_dtype:    element type for optimizer state (stability lever)

    Consumed by ModelSpec, StabilizationPolicy, and the plan builder.
    """
    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype

    storage_fp_format_max: float
    compute_fp_format_max: float
    compute_epsilon: float

    def __post_init__(self) -> None:
        assert self.storage_dtype.itemsize <= self.compute_dtype.itemsize, (
            "storage precision must not be wider than compute precision"
        )
        assert self.storage_dtype.itemsize <= self.state_dtype.itemsize, (
            "storage precision must not be wider than state precision"
        )

    @classmethod
    def float32(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=float(np.finfo(np.float32).max),
            compute_fp_format_max=float(np.finfo(np.float32).max),
            compute_epsilon=float(np.finfo(np.float32).eps),
        )

    @classmethod
    def float16(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float16),
            storage_fp_format_max=float(np.finfo(np.float16).max),
            compute_fp_format_max=float(np.finfo(np.float16).max),
            compute_epsilon=float(np.finfo(np.float16).eps),
        )

    @classmethod
    def mixed_f16_f32(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=float(np.finfo(np.float16).max),
            compute_fp_format_max=float(np.finfo(np.float32).max),
            compute_epsilon=float(np.finfo(np.float32).eps),
        )

    @classmethod
    def float64(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float64),
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(np.finfo(np.float64).max),
            compute_fp_format_max=float(np.finfo(np.float64).max),
            compute_epsilon=1e-15,
        )

    @classmethod
    def mixed_f32_f64_state(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(np.finfo(np.float32).max),
            compute_fp_format_max=float(np.finfo(np.float32).max),
            compute_epsilon=float(np.finfo(np.float32).eps),
        )

    @classmethod
    def mixed_f16_f64_state(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(np.finfo(np.float16).max),
            compute_fp_format_max=float(np.finfo(np.float32).max),
            compute_epsilon=float(np.finfo(np.float32).eps),
        )

    @classmethod
    def mixed_f32_f64(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(np.finfo(np.float32).max),
            compute_fp_format_max=float(np.finfo(np.float64).max),
            compute_epsilon=1e-15,
        )
