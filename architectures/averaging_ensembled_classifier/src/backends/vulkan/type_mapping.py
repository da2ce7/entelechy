"""PrecisionConfig → Vulkan type mapping (ADR-008)."""
from __future__ import annotations

import numpy as np

from ...shared.precision_config import PrecisionConfig


def get_numpy_dtype(precision: PrecisionConfig) -> np.dtype:
    """Map PrecisionConfig to the numpy dtype for buffer allocation."""
    return precision.storage_dtype


def get_element_size(precision: PrecisionConfig) -> int:
    """Element size in bytes for the given precision."""
    return int(precision.storage_dtype.itemsize)


def get_specialization_scalar_is_half(precision: PrecisionConfig) -> int:
    """Return the SPEC_PROBLEM_TYPE-style flag for scalar type.

    0 = FP32 (float), 1 = FP16 (half) — currently only FP32 is supported.
    """
    if precision.compute_dtype == np.dtype(np.float32):
        return 0
    raise NotImplementedError(
        f"Vulkan backend does not yet support {precision.compute_dtype}. "
        "FP16 requires VK_KHR_shader_float16_int8 and shaderFloat16 feature."
    )
