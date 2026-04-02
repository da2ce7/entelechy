# src/backends/opencl/type_mapping.py
"""PrecisionConfig -> OpenCL compiler flags and dtype mapping (ADR-008, ADR-020 §7.1)."""
from __future__ import annotations

import numpy as np

from ...shared.precision_config import PrecisionConfig
from ...shared.hardware_profile import HardwareProfile


def build_compiler_flags(
    precision: PrecisionConfig,
    hardware: HardwareProfile,
    c_tile_size: int,
) -> list[str]:
    """Produce OpenCL -D compiler flags from plan-level configuration.

    Generates flags for all CONTRACT.md Article 6 mandatory symbols (amended by ADR-020 §3.6):
    STORAGE_TYPE, COMPUTE_TYPE, STATE_TYPE, STORAGE_TYPE_IS_HALF, COMPUTE_TYPE_IS_HALF,
    SIMD_WIDTH, C_TILE_SIZE, NUMERICAL_STABILITY_EPSILON, LOCAL_MEM_BANK_PADDING.
    
    Also includes transitional aliases (ADR-021 §3) for backward compatibility during migration.
    """
    storage_cl = _dtype_to_cl_type(precision.storage_dtype)
    compute_cl = _dtype_to_cl_type(precision.compute_dtype)
    state_cl = _dtype_to_cl_type(precision.state_dtype)
    storage_is_half = 1 if precision.storage_dtype == np.dtype(np.float16) else 0
    compute_is_half = 1 if precision.compute_dtype == np.dtype(np.float16) else 0
    eps = _epsilon_literal(precision.compute_epsilon, compute_is_half)
    
    flags = [
        f"-DSTORAGE_TYPE={storage_cl}",
        f"-DCOMPUTE_TYPE={compute_cl}",
        f"-DSTATE_TYPE={state_cl}",
        f"-DSTORAGE_TYPE_IS_HALF={storage_is_half}",
        f"-DCOMPUTE_TYPE_IS_HALF={compute_is_half}",
        f"-DSIMD_WIDTH={hardware.simd_width}",
        f"-DC_TILE_SIZE={c_tile_size}",
        f"-DNUMERICAL_STABILITY_EPSILON={eps}",
        "-DLOCAL_MEM_BANK_PADDING=1",
    ]
    return flags


def _dtype_to_cl_type(dtype: np.dtype) -> str:
    """Map numpy dtype to OpenCL C type name."""
    mapping = {
        np.dtype(np.float32): "float",
        np.dtype(np.float16): "half",
    }
    return mapping[dtype]


def _epsilon_literal(epsilon: float, is_half: int) -> str:
    """Format epsilon as appropriate C literal for the target precision."""
    if is_half:
        return str(epsilon)  # No 'f' suffix for half literals
    return f"{epsilon}f"

