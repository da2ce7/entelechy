"""PrecisionConfig → numpy dtype mapping for CPU backend (ADR-008, ADR-022 §7.3)."""
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
