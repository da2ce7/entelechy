# src/shared/precision_config.py
"""Backend-neutral precision configuration (ADR-008, ADR-020, ADR-022 §1, ADR-025)."""
from dataclasses import dataclass

import ml_dtypes
import numpy as np

# FP8 dtype references (ADR-025 §1)
# ml_dtypes exports type objects (not dtype instances) — wrap with np.dtype() for consistency
# These constants ARE np.dtype instances — use directly without re-wrapping
FP8_E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)
FP8_E5M2 = np.dtype(ml_dtypes.float8_e5m2)
FP8_DTYPES = frozenset({FP8_E4M3, FP8_E5M2})


@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable three-role precision configuration for plan construction.

    Three independent dtype roles separate bandwidth, arithmetic fidelity,
    and optimizer-state stability concerns (ADR-020 §4.1):
      - storage_dtype:  element type for storage-role buffers (bandwidth lever)
      - compute_dtype:  element type for arithmetic (fidelity lever)
      - state_dtype:    element type for optimizer state (stability lever)

    All fields are required (no defaults). Factory classmethods provide
    the standard configurations; direct construction is supported but
    requires all fields to be specified explicitly.

    Consumed by ModelSpec, StabilizationPolicy, and the plan builder.
    """
    # --- Core dtype fields (3 roles) ---
    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype

    # --- Storage-role derived constants ---
    storage_fp_format_max: float
    storage_fp_min_positive: float
    storage_mantissa_bits: int

    # --- Compute-role derived constants ---
    compute_fp_format_max: float
    compute_epsilon: float

    def __post_init__(self) -> None:
        # Normalize dtype fields to np.dtype instances for reliable comparison
        object.__setattr__(self, 'storage_dtype', np.dtype(self.storage_dtype))
        object.__setattr__(self, 'compute_dtype', np.dtype(self.compute_dtype))
        object.__setattr__(self, 'state_dtype', np.dtype(self.state_dtype))

        # FP8 role constraints: storage-only (never compute or state)
        if self.compute_dtype in FP8_DTYPES:
            raise ValueError(
                f"FP8 compute is architecturally prohibited: "
                f"hardware lacks native FP8 arithmetic. "
                f"Use FP16, FP32, or FP64 for compute_dtype, "
                f"got {self.compute_dtype}."
            )
        if self.state_dtype in FP8_DTYPES:
            raise ValueError(
                f"FP8 state is architecturally prohibited: "
                f"EMA updates require higher precision. "
                f"Use FP16, FP32, or FP64 for state_dtype, "
                f"got {self.state_dtype}."
            )

        # Existing invariants (storage ≤ compute, storage ≤ state)
        if self.storage_dtype.itemsize > self.compute_dtype.itemsize:
            raise ValueError(
                f"storage_dtype ({self.storage_dtype}) cannot be wider than "
                f"compute_dtype ({self.compute_dtype})"
            )
        if self.storage_dtype.itemsize > self.state_dtype.itemsize:
            raise ValueError(
                f"storage_dtype ({self.storage_dtype}) cannot be wider than "
                f"state_dtype ({self.state_dtype})"
            )

    @classmethod
    def float32(cls) -> "PrecisionConfig":
        f32_info = np.finfo(np.float32)
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=float(f32_info.max),
            storage_fp_min_positive=float(f32_info.smallest_subnormal),
            storage_mantissa_bits=f32_info.nmant,
            compute_fp_format_max=float(f32_info.max),
            compute_epsilon=float(f32_info.eps),
        )

    @classmethod
    def mixed_f16_f32(cls) -> "PrecisionConfig":
        """FP16 storage, FP32 compute, FP32 state.

        Recommended replacement for the deleted float16() factory.
        Provides FP16 bandwidth savings with FP32 compute fidelity and FP32 state stability.
        """
        f16_info = np.finfo(np.float16)
        f32_info = np.finfo(np.float32)
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=float(f16_info.max),
            storage_fp_min_positive=float(f16_info.smallest_subnormal),
            storage_mantissa_bits=f16_info.nmant,
            compute_fp_format_max=float(f32_info.max),
            compute_epsilon=float(f32_info.eps),
        )

    @classmethod
    def float64(cls) -> "PrecisionConfig":
        f64_info = np.finfo(np.float64)
        return cls(
            storage_dtype=np.dtype(np.float64),
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(f64_info.max),
            storage_fp_min_positive=float(f64_info.smallest_subnormal),
            storage_mantissa_bits=f64_info.nmant,
            compute_fp_format_max=float(f64_info.max),
            compute_epsilon=float(f64_info.eps),
        )

    @classmethod
    def mixed_f32_f64_state(cls) -> "PrecisionConfig":
        f32_info = np.finfo(np.float32)
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(f32_info.max),
            storage_fp_min_positive=float(f32_info.smallest_subnormal),
            storage_mantissa_bits=f32_info.nmant,
            compute_fp_format_max=float(f32_info.max),
            compute_epsilon=float(f32_info.eps),
        )

    @classmethod
    def mixed_f16_f64_state(cls) -> "PrecisionConfig":
        """FP16 storage, FP32 compute, FP64 state for extended optimizer stability."""
        f16_info = np.finfo(np.float16)
        f32_info = np.finfo(np.float32)
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(f16_info.max),
            storage_fp_min_positive=float(f16_info.smallest_subnormal),
            storage_mantissa_bits=f16_info.nmant,
            compute_fp_format_max=float(f32_info.max),
            compute_epsilon=float(f32_info.eps),
        )

    @classmethod
    def mixed_f32_f64(cls) -> "PrecisionConfig":
        f32_info = np.finfo(np.float32)
        f64_info = np.finfo(np.float64)
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=float(f32_info.max),
            storage_fp_min_positive=float(f32_info.smallest_subnormal),
            storage_mantissa_bits=f32_info.nmant,
            compute_fp_format_max=float(f64_info.max),
            compute_epsilon=float(f64_info.eps),
        )

    # --- FP8 E4M3 Factories (ADR-025 §2.3) ---

    @classmethod
    def fp8_e4m3(cls) -> "PrecisionConfig":
        """E4M3 storage (8-bit, max=448), FP32 compute, FP32 state.

        Maximum bandwidth configuration for standard training.
        4× storage compression vs. FP32, 2× vs. FP16.
        """
        f32_info = np.finfo(np.float32)
        return cls(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=448.0,
            storage_fp_min_positive=0.001953125,  # 2^-9
            storage_mantissa_bits=3,
            compute_fp_format_max=float(f32_info.max),
            compute_epsilon=float(f32_info.eps),
        )

    @classmethod
    def fp8_e4m3_f16(cls) -> "PrecisionConfig":
        """E4M3 storage (8-bit), FP16 compute, FP32 state."""
        f16_info = np.finfo(np.float16)
        return cls(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=448.0,
            storage_fp_min_positive=0.001953125,  # 2^-9
            storage_mantissa_bits=3,
            compute_fp_format_max=float(f16_info.max),
            compute_epsilon=float(f16_info.eps),
        )

    @classmethod
    def fp8_e4m3_f64(cls) -> "PrecisionConfig":
        """E4M3 storage (8-bit), FP64 compute, FP64 state."""
        f64_info = np.finfo(np.float64)
        return cls(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=448.0,
            storage_fp_min_positive=0.001953125,  # 2^-9
            storage_mantissa_bits=3,
            compute_fp_format_max=float(f64_info.max),
            compute_epsilon=float(f64_info.eps),
        )

    # --- FP8 E5M2 Factories (ADR-025 §2.3) ---

    @classmethod
    def fp8_e5m2(cls) -> "PrecisionConfig":
        """E5M2 storage (8-bit, max=57344), FP32 compute, FP32 state.

        Maximum bandwidth with wider dynamic range.
        """
        f32_info = np.finfo(np.float32)
        return cls(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=57344.0,
            storage_fp_min_positive=0.0000152587890625,  # 2^-16, exact
            storage_mantissa_bits=2,
            compute_fp_format_max=float(f32_info.max),
            compute_epsilon=float(f32_info.eps),
        )

    @classmethod
    def fp8_e5m2_f16(cls) -> "PrecisionConfig":
        """E5M2 storage (8-bit), FP16 compute, FP32 state."""
        f16_info = np.finfo(np.float16)
        return cls(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=57344.0,
            storage_fp_min_positive=0.0000152587890625,  # 2^-16, exact
            storage_mantissa_bits=2,
            compute_fp_format_max=float(f16_info.max),
            compute_epsilon=float(f16_info.eps),
        )

    @classmethod
    def fp8_e5m2_f64(cls) -> "PrecisionConfig":
        """E5M2 storage (8-bit), FP64 compute, FP64 state."""
        f64_info = np.finfo(np.float64)
        return cls(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
            storage_fp_format_max=57344.0,
            storage_fp_min_positive=0.0000152587890625,  # 2^-16, exact
            storage_mantissa_bits=2,
            compute_fp_format_max=float(f64_info.max),
            compute_epsilon=float(f64_info.eps),
        )
