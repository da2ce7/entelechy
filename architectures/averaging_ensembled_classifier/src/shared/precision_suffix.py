"""Canonical kernel variant suffix definitions (ADR-024 §4.1, ADR-025 §5.3).

Single source of truth for the three-axis precision suffix convention
``s{storage}c{compute}x{state}`` used across all backends.

Both the CPU and Vulkan backends import from here rather than maintaining
their own duplicated suffix maps.  The canonical form carries no leading
underscore; backends that require one (e.g., Vulkan SPIR-V file names)
prepend it at the point of use.
"""
from __future__ import annotations

import numpy as np

from .precision_config import FP8_DTYPES, FP8_E4M3, FP8_E5M2

# ── Three-axis precision suffixes ─────────────────────────────────────
# Format: s{storage_bits}c{compute_bits}x{state_bits}
# FP8 storage encodes the exponent count: s8e4 = E4M3, s8e5 = E5M2.

PRECISION_SUFFIXES: tuple[str, ...] = (
    "s16c16x16", "s16c16x32", "s16c16x64",
    "s16c32x16", "s16c32x32", "s16c32x64",
    "s16c64x16", "s16c64x32", "s16c64x64",
    "s32c32x32", "s32c32x64",
    "s32c64x32", "s32c64x64",
    "s64c64x64",
    # FP8 E4M3 storage (ADR-025 §5.3)
    "s8e4c32x32", "s8e4c32x64", "s8e4c64x32", "s8e4c64x64",
    # FP8 E5M2 storage (ADR-025 §5.3)
    "s8e5c32x32", "s8e5c32x64", "s8e5c64x32", "s8e5c64x64",
)

# FP8 variants with FP16 compute — conditionally available per platform.
FP8_FP16_PRECISION_SUFFIXES: tuple[str, ...] = (
    # FP16 compute variants
    "s8e4c16x32", "s8e4c16x64", "s8e4c16x16",
    "s8e5c16x32", "s8e5c16x64", "s8e5c16x16",
    # FP16 state variants (FP32/FP64 compute)
    "s8e4c32x16", "s8e4c64x16",
    "s8e5c32x16", "s8e5c64x16",
)

# Compute-only suffixes (for shaders that have no storage-role bindings).
COMPUTE_ONLY_SUFFIXES: tuple[str, ...] = ("c16", "c32", "c64")

# ── Suffix map: (storage_type, compute_type, state_type) → suffix ────

_SUFFIX_MAP: dict[tuple[type, type, type], str] = {
    (np.float16, np.float16, np.float16): "s16c16x16",
    (np.float16, np.float16, np.float32): "s16c16x32",
    (np.float16, np.float16, np.float64): "s16c16x64",
    (np.float16, np.float32, np.float16): "s16c32x16",
    (np.float16, np.float32, np.float32): "s16c32x32",
    (np.float16, np.float32, np.float64): "s16c32x64",
    (np.float16, np.float64, np.float16): "s16c64x16",
    (np.float16, np.float64, np.float32): "s16c64x32",
    (np.float16, np.float64, np.float64): "s16c64x64",
    (np.float32, np.float32, np.float32): "s32c32x32",
    (np.float32, np.float32, np.float64): "s32c32x64",
    (np.float32, np.float64, np.float32): "s32c64x32",
    (np.float32, np.float64, np.float64): "s32c64x64",
    (np.float64, np.float64, np.float64): "s64c64x64",
}

_FP8_SUFFIX_MAP: dict[tuple[np.dtype, type, type], str] = {
    # E4M3 variants
    (FP8_E4M3, np.float16, np.float16): "s8e4c16x16",
    (FP8_E4M3, np.float16, np.float32): "s8e4c16x32",
    (FP8_E4M3, np.float16, np.float64): "s8e4c16x64",
    (FP8_E4M3, np.float32, np.float16): "s8e4c32x16",
    (FP8_E4M3, np.float32, np.float32): "s8e4c32x32",
    (FP8_E4M3, np.float32, np.float64): "s8e4c32x64",
    (FP8_E4M3, np.float64, np.float16): "s8e4c64x16",
    (FP8_E4M3, np.float64, np.float32): "s8e4c64x32",
    (FP8_E4M3, np.float64, np.float64): "s8e4c64x64",
    # E5M2 variants
    (FP8_E5M2, np.float16, np.float16): "s8e5c16x16",
    (FP8_E5M2, np.float16, np.float32): "s8e5c16x32",
    (FP8_E5M2, np.float16, np.float64): "s8e5c16x64",
    (FP8_E5M2, np.float32, np.float16): "s8e5c32x16",
    (FP8_E5M2, np.float32, np.float32): "s8e5c32x32",
    (FP8_E5M2, np.float32, np.float64): "s8e5c32x64",
    (FP8_E5M2, np.float64, np.float16): "s8e5c64x16",
    (FP8_E5M2, np.float64, np.float32): "s8e5c64x32",
    (FP8_E5M2, np.float64, np.float64): "s8e5c64x64",
}


def precision_to_suffix(
    storage_dtype: np.dtype,
    compute_dtype: np.dtype,
    state_dtype: np.dtype,
) -> str:
    """Map a three-axis precision configuration to its canonical kernel suffix.

    Returns a bare suffix string (no leading underscore), e.g. ``"s32c32x32"``
    or ``"s8e4c32x32"``.

    Raises ``ValueError`` if the dtype triple has no defined suffix.
    """
    if storage_dtype in FP8_DTYPES:
        fp8_key = (storage_dtype, compute_dtype.type, state_dtype.type)
        suffix = _FP8_SUFFIX_MAP.get(fp8_key)
        if suffix is None:
            raise ValueError(
                f"No kernel variant for FP8 storage={storage_dtype}, "
                f"compute={compute_dtype}, state={state_dtype}"
            )
        return suffix

    key = (storage_dtype.type, compute_dtype.type, state_dtype.type)
    suffix = _SUFFIX_MAP.get(key)
    if suffix is None:
        raise ValueError(
            f"No kernel variant for storage={storage_dtype}, "
            f"compute={compute_dtype}, state={state_dtype}"
        )
    return suffix


def compute_only_suffix(compute_dtype: np.dtype) -> str:
    """Map a compute dtype to its compute-only kernel suffix.

    Returns a bare suffix string (no leading underscore), e.g. ``"c32"``.
    """
    if compute_dtype == np.dtype(np.float64):
        return "c64"
    if compute_dtype == np.dtype(np.float16):
        return "c16"
    return "c32"
