# src/backends/opencl/type_mapping.py
"""PrecisionConfig -> OpenCL compiler flags and dtype mapping (ADR-008, ADR-020 §7.1, ADR-025 §6.1)."""
from __future__ import annotations

import numpy as np

from ...shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2, FP8_DTYPES
from ...shared.hardware_profile import HardwareProfile


def build_compiler_flags(
    precision: PrecisionConfig,
    hardware: HardwareProfile,
    c_tile_size: int,
) -> list[str]:
    """Produce OpenCL -D compiler flags from plan-level configuration.

    Generates flags for all CONTRACT.md Article 6 mandatory symbols (amended by ADR-020 §3.6, ADR-024 §2, ADR-025 §3):
    STORAGE_TYPE, COMPUTE_TYPE, STATE_TYPE,
    STORAGE_TYPE_IS_FP8, STORAGE_TYPE_IS_E4M3, STORAGE_TYPE_IS_E5M2,
    STORAGE_TYPE_IS_HALF, STORAGE_TYPE_IS_FLOAT, STORAGE_TYPE_IS_DOUBLE,
    COMPUTE_TYPE_IS_HALF, COMPUTE_TYPE_IS_FLOAT, COMPUTE_TYPE_IS_DOUBLE,
    STATE_TYPE_IS_HALF, STATE_TYPE_IS_FLOAT, STATE_TYPE_IS_DOUBLE,
    SIMD_WIDTH, C_TILE_SIZE, NUMERICAL_STABILITY_EPSILON, LOCAL_MEM_BANK_PADDING.
    """
    storage_cl = _dtype_to_cl_type(precision.storage_dtype)
    compute_cl = _dtype_to_cl_type(precision.compute_dtype)
    state_cl = _dtype_to_cl_type(precision.state_dtype)
    
    # Storage role type flags (complete taxonomy)
    is_fp8 = precision.storage_dtype in FP8_DTYPES
    storage_is_half = 1 if precision.storage_dtype == np.dtype(np.float16) else 0
    storage_is_float = 1 if precision.storage_dtype == np.dtype(np.float32) else 0
    storage_is_double = 1 if precision.storage_dtype == np.dtype(np.float64) else 0
    is_e4m3 = 1 if precision.storage_dtype == FP8_E4M3 else 0
    is_e5m2 = 1 if precision.storage_dtype == FP8_E5M2 else 0
    
    # Compute role type flags (complete taxonomy)
    compute_is_half = 1 if precision.compute_dtype == np.dtype(np.float16) else 0
    compute_is_float = 1 if precision.compute_dtype == np.dtype(np.float32) else 0
    compute_is_double = 1 if precision.compute_dtype == np.dtype(np.float64) else 0
    
    # State role type flags (complete taxonomy)
    state_is_half = 1 if precision.state_dtype == np.dtype(np.float16) else 0
    state_is_float = 1 if precision.state_dtype == np.dtype(np.float32) else 0
    state_is_double = 1 if precision.state_dtype == np.dtype(np.float64) else 0
    
    eps = _epsilon_literal(precision.compute_epsilon, compute_is_half, compute_is_double)
    
    flags = [
        f"-DSTORAGE_TYPE={storage_cl}",
        f"-DCOMPUTE_TYPE={compute_cl}",
        f"-DSTATE_TYPE={state_cl}",
        f"-DSTORAGE_TYPE_IS_FP8={1 if is_fp8 else 0}",
        f"-DSTORAGE_TYPE_IS_E4M3={is_e4m3}",
        f"-DSTORAGE_TYPE_IS_E5M2={is_e5m2}",
        f"-DSTORAGE_TYPE_IS_HALF={storage_is_half}",
        f"-DSTORAGE_TYPE_IS_FLOAT={storage_is_float}",
        f"-DSTORAGE_TYPE_IS_DOUBLE={storage_is_double}",
        f"-DCOMPUTE_TYPE_IS_HALF={compute_is_half}",
        f"-DCOMPUTE_TYPE_IS_FLOAT={compute_is_float}",
        f"-DCOMPUTE_TYPE_IS_DOUBLE={compute_is_double}",
        f"-DSTATE_TYPE_IS_HALF={state_is_half}",
        f"-DSTATE_TYPE_IS_FLOAT={state_is_float}",
        f"-DSTATE_TYPE_IS_DOUBLE={state_is_double}",
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
        np.dtype(np.float64): "double",
        FP8_E4M3: "uchar",  # FP8 E4M3 as byte (software emulation, ADR-025 §6.1)
        FP8_E5M2: "uchar",  # FP8 E5M2 as byte (software emulation, ADR-025 §6.1)
    }
    return mapping[dtype]


def _epsilon_literal(epsilon: float, is_half: int, is_double: int) -> str:
    """Format epsilon as appropriate C literal for the target precision."""
    if is_half:
        return str(epsilon)  # No 'f' suffix for half literals
    if is_double:
        return str(epsilon)  # No suffix for double literals in OpenCL C
    return f"{epsilon}f"

