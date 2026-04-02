"""PrecisionConfig → Vulkan type mapping (ADR-008, ADR-022 §7.3)."""
from __future__ import annotations

import numpy as np

from ...shared.precision_config import PrecisionConfig


def get_storage_dtype(precision: PrecisionConfig) -> np.dtype:
    """Map PrecisionConfig to the numpy dtype for storage-role buffers."""
    return precision.storage_dtype


def get_compute_dtype(precision: PrecisionConfig) -> np.dtype:
    """Map PrecisionConfig to the numpy dtype for compute-role buffers."""
    return precision.compute_dtype


def get_state_dtype(precision: PrecisionConfig) -> np.dtype:
    """Map PrecisionConfig to the numpy dtype for state-role buffers."""
    return precision.state_dtype


def get_storage_element_size(precision: PrecisionConfig) -> int:
    """Element size in bytes for storage-role buffers."""
    return int(precision.storage_dtype.itemsize)


def get_storage_is_half(precision: PrecisionConfig) -> int:
    """Return 1 if storage dtype is FP16, 0 otherwise."""
    return 1 if precision.storage_dtype == np.dtype(np.float16) else 0


def get_state_is_half(precision: PrecisionConfig) -> int:
    """Return 1 if state dtype is FP16, 0 otherwise."""
    return 1 if precision.state_dtype == np.dtype(np.float16) else 0


def get_compute_is_double(precision: PrecisionConfig) -> int:
    """Return 1 if compute dtype is FP64, 0 otherwise."""
    return 1 if precision.compute_dtype == np.dtype(np.float64) else 0


def get_state_is_double(precision: PrecisionConfig) -> int:
    """Return 1 if state dtype is FP64, 0 otherwise."""
    return 1 if precision.state_dtype == np.dtype(np.float64) else 0


def get_storage_is_double(precision: PrecisionConfig) -> int:
    """Return 1 if storage dtype is FP64, 0 otherwise."""
    return 1 if precision.storage_dtype == np.dtype(np.float64) else 0
