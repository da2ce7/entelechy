# src/shared/precision_config.py
"""Backend-neutral precision configuration (CONTRACT §5.2, ADR-020, ADR-025).

Decomposes numeric precision into three independent roles:

* **storage_dtype** — bandwidth optimisation (the narrowest format that
  preserves sufficient information for the downstream operation)
* **compute_dtype** — arithmetic fidelity (the format used for all
  transformative arithmetic)
* **state_dtype** — long-term stability (the format for optimizer moment
  vectors and learnable parameters)

A configuration where all three roles share a type
(e.g. ``PrecisionConfig.float32()``) is a parameterisation — not a
distinct mode.

Derived scalar constants (``storage_fp_format_max``, ``compute_epsilon``,
etc.) are computed automatically from the primary dtype fields in
``__post_init__`` and cannot be supplied independently.  This eliminates
the possibility of inconsistent direct construction.

FP8 storage requires the ``ml_dtypes`` package.  When absent, all
non-FP8 configurations remain fully functional; FP8 factory methods
raise ``ImportError`` at call time.

Authoritative source
────────────────────
CONTRACT.md §5.2 (Revision 10)
CONCEPT.md  §2   (Primacy of Memory Strategy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ═════════════════════════════════════════════════════════════════════
# FP8 dtype support (optional dependency)
#
# ml_dtypes is required only for FP8 storage configurations.  When
# absent, the module-level constants are None, FP8_DTYPES is empty,
# and all non-FP8 code paths remain fully functional.
# ═════════════════════════════════════════════════════════════════════

try:
    import ml_dtypes as _ml_dtypes

    FP8_E4M3: np.dtype | None = np.dtype(_ml_dtypes.float8_e4m3fn)
    FP8_E5M2: np.dtype | None = np.dtype(_ml_dtypes.float8_e5m2)
    FP8_DTYPES: frozenset[np.dtype] = frozenset({FP8_E4M3, FP8_E5M2})
except ImportError:
    FP8_E4M3 = None
    FP8_E5M2 = None
    FP8_DTYPES = frozenset()

# ═════════════════════════════════════════════════════════════════════
# Type aliases
# ═════════════════════════════════════════════════════════════════════

MaskStrategy = Literal["explicit", "recompute"]
"""CONTRACT §5.2.3 — hidden-mask lifecycle strategy.

``"explicit"``:  Node 4 materialises the mask buffer capturing
                 compute-precision derivative truth before storage
                 narrowing.  Consuming kernels read the mask directly.
``"recompute"``: No mask buffer allocated.  Consuming kernels derive
                 the mask from stored activations (``activation > 0``).
                 Valid only when ``storage_dtype == compute_dtype``.
"""

# ═════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════


def _storage_fp_constants(dt: np.dtype) -> tuple[float, float, int]:
    """Derive ``(format_max, min_positive_subnormal, mantissa_bits)``.

    FP8 values are architectural constants from CONTRACT §5.2.6.
    Standard IEEE types delegate to ``np.finfo``.
    """
    if FP8_E4M3 is not None and dt == FP8_E4M3:
        return (448.0, 0.001953125, 3)  # max, 2⁻⁹, 3-bit mantissa
    if FP8_E5M2 is not None and dt == FP8_E5M2:
        return (57344.0, 0.0000152587890625, 2)  # max, 2⁻¹⁶, 2-bit mantissa
    info = np.finfo(dt)
    return (float(info.max), float(info.smallest_subnormal), info.nmant)


def _require_fp8(factory_name: str) -> None:
    """Raise ``ImportError`` if ``ml_dtypes`` is not available."""
    if FP8_E4M3 is None:
        raise ImportError(
            f"PrecisionConfig.{factory_name}() requires the ml_dtypes "
            f"package: pip install ml_dtypes"
        )


# ═════════════════════════════════════════════════════════════════════
# PrecisionConfig
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable three-role precision configuration for plan construction.

    The constructor accepts **only** the three primary dtype fields.
    All derived scalar constants are computed in ``__post_init__``
    from the primaries and cannot be overridden.

    Parameters
    ----------
    storage_dtype:
        Element type for storage-role buffers (bandwidth lever).
    compute_dtype:
        Element type for arithmetic operations (fidelity lever).
        FP8 is architecturally prohibited.
    state_dtype:
        Element type for optimizer state (stability lever).
        FP8 is architecturally prohibited.

    Construction invariants (CONTRACT §5.2.4)
    ──────────────────────────────────────────
    1. ``storage_dtype.itemsize ≤ compute_dtype.itemsize``
    2. ``storage_dtype.itemsize ≤ state_dtype.itemsize``
    3. FP8 is storage-only (never compute or state).

    The state role has **no** ordering constraint relative to compute —
    ``state_dtype.itemsize`` may be >, ==, or < ``compute_dtype.itemsize``.
    """

    # ── Primary fields (3 roles) — sole constructor arguments ─────

    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype

    # ── Derived storage-role constants (CONTRACT §5.2.2) ──────────

    storage_fp_format_max: float = field(init=False)
    """Maximum finite value representable in the storage format."""

    storage_fp_min_positive: float = field(init=False)
    """Minimum positive subnormal in the storage format."""

    storage_mantissa_bits: int = field(init=False)
    """Mantissa bit count in the storage format."""

    # ── Derived compute-role constants (CONTRACT §5.2.2) ──────────

    compute_fp_format_max: float = field(init=False)
    """Maximum finite value in the compute format."""

    compute_epsilon: float = field(init=False)
    """Machine epsilon for the compute format."""

    # ── Validation & derivation ───────────────────────────────────

    def __post_init__(self) -> None:
        # Normalise to np.dtype instances for reliable comparison.
        object.__setattr__(self, "storage_dtype", np.dtype(self.storage_dtype))
        object.__setattr__(self, "compute_dtype", np.dtype(self.compute_dtype))
        object.__setattr__(self, "state_dtype", np.dtype(self.state_dtype))

        # Invariant: FP8 is storage-only (CONTRACT §5.2.4 #3).
        if self.compute_dtype in FP8_DTYPES:
            raise ValueError(
                f"FP8 compute is architecturally prohibited: hardware lacks "
                f"native FP8 arithmetic.  Use FP16, FP32, or FP64 for "
                f"compute_dtype, got {self.compute_dtype}."
            )
        if self.state_dtype in FP8_DTYPES:
            raise ValueError(
                f"FP8 state is architecturally prohibited: EMA updates "
                f"require higher precision.  Use FP16, FP32, or FP64 for "
                f"state_dtype, got {self.state_dtype}."
            )

        # Invariant: storage ≤ compute (CONTRACT §5.2.4 #1).
        if self.storage_dtype.itemsize > self.compute_dtype.itemsize:
            raise ValueError(
                f"storage_dtype ({self.storage_dtype}) cannot be wider than "
                f"compute_dtype ({self.compute_dtype})"
            )

        # Invariant: storage ≤ state (CONTRACT §5.2.4 #2).
        if self.storage_dtype.itemsize > self.state_dtype.itemsize:
            raise ValueError(
                f"storage_dtype ({self.storage_dtype}) cannot be wider than "
                f"state_dtype ({self.state_dtype})"
            )

        # Derive storage-role constants.
        s_max, s_min, s_mant = _storage_fp_constants(self.storage_dtype)
        object.__setattr__(self, "storage_fp_format_max", s_max)
        object.__setattr__(self, "storage_fp_min_positive", s_min)
        object.__setattr__(self, "storage_mantissa_bits", s_mant)

        # Derive compute-role constants.
        # FP8 compute is rejected above, so np.finfo is always valid.
        c_info = np.finfo(self.compute_dtype)
        object.__setattr__(self, "compute_fp_format_max", float(c_info.max))
        object.__setattr__(self, "compute_epsilon", float(c_info.eps))

    # ── Mask strategy (CONTRACT §5.2.3) ───────────────────────────

    @property
    def mask_strategy(self) -> MaskStrategy:
        """Hidden-mask lifecycle strategy derived from the precision roles.

        When ``storage_dtype != compute_dtype``, the precision boundary
        can destroy derivative information for activations near the
        storage format's quantisation floor.  The mask must be
        explicitly materialised to preserve compute-precision truth.

        When ``storage_dtype == compute_dtype``, the mask is bit-
        identical to ``activation > 0`` and can be recomputed from
        the stored activations.
        """
        if self.storage_dtype != self.compute_dtype:
            return "explicit"
        return "recompute"

    # ═════════════════════════════════════════════════════════════════
    # Factory classmethods (CONTRACT §5.2.5)
    # ═════════════════════════════════════════════════════════════════

    # ── Standard IEEE configurations ──────────────────────────────

    @classmethod
    def float32(cls) -> PrecisionConfig:
        """FP32 storage, FP32 compute, FP32 state.  Default reference."""
        dt = np.dtype(np.float32)
        return cls(storage_dtype=dt, compute_dtype=dt, state_dtype=dt)

    @classmethod
    def float64(cls) -> PrecisionConfig:
        """FP64 storage, FP64 compute, FP64 state.  Validation reference."""
        dt = np.dtype(np.float64)
        return cls(storage_dtype=dt, compute_dtype=dt, state_dtype=dt)

    # ── Mixed IEEE configurations ─────────────────────────────────

    @classmethod
    def mixed_f16_f32(cls) -> PrecisionConfig:
        """FP16 storage, FP32 compute, FP32 state.  Production mixed-precision."""
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
        )

    @classmethod
    def mixed_f32_f64(cls) -> PrecisionConfig:
        """FP32 storage, FP64 compute, FP64 state.  High-fidelity scientific."""
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
        )

    @classmethod
    def mixed_f32_f64_state(cls) -> PrecisionConfig:
        """FP32 storage, FP32 compute, FP64 state.  Extended-stability training."""
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float64),
        )

    @classmethod
    def mixed_f16_f64_state(cls) -> PrecisionConfig:
        """FP16 storage, FP32 compute, FP64 state.  Bandwidth + extended stability."""
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float64),
        )

    # ── FP8 E4M3 configurations (CONTRACT §5.2.5, §5.2.6) ────────

    @classmethod
    def fp8_e4m3(cls) -> PrecisionConfig:
        """E4M3 storage, FP32 compute, FP32 state.  4× compression vs FP32."""
        _require_fp8("fp8_e4m3")
        assert FP8_E4M3 is not None  # narrowing; guarded by _require_fp8
        return cls(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
        )

    @classmethod
    def fp8_e4m3_f16(cls) -> PrecisionConfig:
        """E4M3 storage, FP16 compute, FP32 state."""
        _require_fp8("fp8_e4m3_f16")
        assert FP8_E4M3 is not None
        return cls(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float32),
        )

    @classmethod
    def fp8_e4m3_f64(cls) -> PrecisionConfig:
        """E4M3 storage, FP64 compute, FP64 state.  Full FP64 fidelity."""
        _require_fp8("fp8_e4m3_f64")
        assert FP8_E4M3 is not None
        return cls(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
        )

    # ── FP8 E5M2 configurations (CONTRACT §5.2.5, §5.2.6) ────────

    @classmethod
    def fp8_e5m2(cls) -> PrecisionConfig:
        """E5M2 storage, FP32 compute, FP32 state.  Wider dynamic range."""
        _require_fp8("fp8_e5m2")
        assert FP8_E5M2 is not None
        return cls(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
        )

    @classmethod
    def fp8_e5m2_f16(cls) -> PrecisionConfig:
        """E5M2 storage, FP16 compute, FP32 state."""
        _require_fp8("fp8_e5m2_f16")
        assert FP8_E5M2 is not None
        return cls(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float32),
        )

    @classmethod
    def fp8_e5m2_f64(cls) -> PrecisionConfig:
        """E5M2 storage, FP64 compute, FP64 state."""
        _require_fp8("fp8_e5m2_f64")
        assert FP8_E5M2 is not None
        return cls(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(np.float64),
            state_dtype=np.dtype(np.float64),
        )
