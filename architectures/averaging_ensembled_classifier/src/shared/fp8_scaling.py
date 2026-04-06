# src/shared/fp8_scaling.py
"""FP8 scaling utilities (ADR-025 §8).

The host manages per-tensor scale factors to ensure values fit within
FP8's limited dynamic range. Gradients are already constrained by the
Quadratic Scaling Policy; activations may require explicit scaling.
"""

import warnings
from typing import NamedTuple

import numpy as np

from .precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

# Increment if FP8ScaleInfo structure changes (for checkpoint compatibility)
FP8_SCALES_VERSION = 1


class FP8ScaleInfo(NamedTuple):
    """Scale information for FP8 quantization.

    Note: This is a NamedTuple for immutability and easy serialization.
    Use `._asdict()` for JSON/pickle serialization, `FP8ScaleInfo(**d)` to restore.
    """
    scale: float           # Multiply by this before quantization
    inv_scale: float       # Multiply by this after dequantization
    storage_max: float     # FP8 format max (448 or 57344)


def compute_fp8_scale(
    tensor: np.ndarray,
    precision: PrecisionConfig,
    headroom: float = 0.9
) -> FP8ScaleInfo:
    """Compute scale factor to map tensor values into FP8 range.

    Args:
        tensor: Values to be stored in FP8 format.
        precision: Must have FP8 storage dtype.
        headroom: Fraction of FP8 max to use (default 0.9 for saturation margin).
                  Must be in (0, 1].

    Returns:
        FP8ScaleInfo with scale factors and diagnostics.

    Raises:
        ValueError: If precision lacks FP8 storage dtype or headroom is not in (0, 1].
    """
    if precision.storage_dtype not in (FP8_E4M3, FP8_E5M2):
        raise ValueError(
            f"compute_fp8_scale requires FP8 storage dtype, "
            f"got {precision.storage_dtype}"
        )

    if not (0.0 < headroom <= 1.0):
        raise ValueError(
            f"headroom must be in (0, 1], got {headroom}. "
            f"headroom=0 causes division by zero; headroom>1 allows values "
            f"beyond FP8 max, causing saturation."
        )

    storage_max = precision.storage_fp_format_max
    target_range = storage_max * headroom

    # Guard against zero-size arrays (np.nanmax raises ValueError on empty input)
    if tensor.size == 0:
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )

    # Use nanmax to ignore NaN values when computing tensor maximum.
    # If the entire tensor is NaN, nanmax returns nan (with RuntimeWarning).
    with np.errstate(invalid='ignore'):  # Suppress nanmax RuntimeWarning for all-NaN case
        tensor_max = np.nanmax(np.abs(tensor))

    # Handle all-NaN tensor case: return identity scale, let kernel NaN→zero mapping handle it
    if np.isnan(tensor_max):
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )

    if tensor_max < 1e-10:
        # Near-zero tensor — no scaling needed, inverse is identity
        # WARNING: With scale=1.0, all values in this tensor will quantize to zero
        # in FP8 (since tensor_max < FP8 min subnormal for both E4M3 and E5M2).
        # This is intentional for near-zero activations but should be visible
        # during debugging.
        warnings.warn(
            f"compute_fp8_scale: tensor_max={tensor_max:.2e} < 1e-10. "
            f"All values will quantize to FP8 zero (no gradient contribution). "
            f"This is expected for near-dead activations but may indicate "
            f"upstream numerical issues if seen frequently.",
            stacklevel=2,
        )
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )

    if tensor_max <= target_range:
        # Already fits — no scaling needed.
        # DESIGN DECISION: We intentionally never scale UP (i.e., scale > 1.0).
        # Small tensors carry negligible gradient information, and amplifying them
        # into the FP8 range would inflate quantization noise on values that
        # contribute little to the update. The identity scale (1.0) is correct:
        # small values map to small FP8 codes or zero, which is the desired
        # behavior for near-zero activations.
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )

    # Scale down to fit
    scale = target_range / tensor_max
    inv_scale = 1.0 / scale

    return FP8ScaleInfo(
        scale=scale,
        inv_scale=inv_scale,
        storage_max=storage_max,
    )


def apply_fp8_scale(
    tensor: np.ndarray,
    scale_info: FP8ScaleInfo
) -> np.ndarray:
    """Scale tensor values for FP8 storage."""
    return tensor * scale_info.scale


def unapply_fp8_scale(
    tensor: np.ndarray,
    scale_info: FP8ScaleInfo
) -> np.ndarray:
    """Restore original scale after FP8 dequantization."""
    return tensor * scale_info.inv_scale
