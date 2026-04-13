# src/backends/opencl/type_mapping.py
"""PrecisionConfig -> OpenCL compiler flags and dtype mapping (ADR-008, ADR-020 §7.1, ADR-025 §6.1)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2, FP8_DTYPES
from ...shared.hardware_profile import HardwareProfile


def build_compiler_flags(
    precision: PrecisionConfig,
    hardware: HardwareProfile,
    c_tile_size: int,
) -> list[str]:
    """Produce OpenCL -D compiler flags from plan-level configuration.

    Generates flags for all CONTRACT.md Article 6 mandatory symbols.
    LOCAL_MEM_BANK_PADDING is intentionally NOT emitted — it is a
    kernel-internal constant defined in kernels.cl.h (CONTRACT Article 6,
    Retired Build-Time Symbols).
    """
    storage_cl = _dtype_to_cl_type(precision.storage_dtype)
    compute_cl = _dtype_to_cl_type(precision.compute_dtype)
    state_cl = _dtype_to_cl_type(precision.state_dtype)

    # Storage role type flags (complete taxonomy)
    is_fp8 = precision.storage_dtype in FP8_DTYPES
    storage_is_half = int(precision.storage_dtype == np.dtype(np.float16))
    storage_is_float = int(precision.storage_dtype == np.dtype(np.float32))
    storage_is_double = int(precision.storage_dtype == np.dtype(np.float64))
    is_e4m3 = int(precision.storage_dtype == FP8_E4M3)
    is_e5m2 = int(precision.storage_dtype == FP8_E5M2)

    # Compute role type flags
    compute_is_half = int(precision.compute_dtype == np.dtype(np.float16))
    compute_is_float = int(precision.compute_dtype == np.dtype(np.float32))
    compute_is_double = int(precision.compute_dtype == np.dtype(np.float64))

    # State role type flags
    state_is_half = int(precision.state_dtype == np.dtype(np.float16))
    state_is_float = int(precision.state_dtype == np.dtype(np.float32))
    state_is_double = int(precision.state_dtype == np.dtype(np.float64))

    eps = _epsilon_literal(precision.compute_epsilon, compute_is_half, compute_is_double)

    return [
        f"-DSTORAGE_TYPE={storage_cl}",
        f"-DCOMPUTE_TYPE={compute_cl}",
        f"-DSTATE_TYPE={state_cl}",
        f"-DSTORAGE_TYPE_IS_FP8={int(is_fp8)}",
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
        # NOTE: LOCAL_MEM_BANK_PADDING is NOT emitted here.
        # It is a kernel-internal constant (#define in kernels.cl.h).
        # CONTRACT Article 6 Retired Build-Time Symbols: "Build system
        # MUST NOT provide via -D."
    ]


def _dtype_to_cl_type(dtype: np.dtype[Any]) -> str:
    """Map numpy dtype to OpenCL C type name."""
    mapping = {
        np.dtype(np.float32): "float",
        np.dtype(np.float16): "half",
        np.dtype(np.float64): "double",
        FP8_E4M3: "uchar",
        FP8_E5M2: "uchar",
    }
    return mapping[dtype]


def _epsilon_literal(epsilon: float, is_half: int, is_double: int) -> str:
    """Format epsilon as an appropriate C literal for the target precision.

    Uses standard suffixes: 'h' for half (cl_khr_fp16), 'f' for float,
    none for double.
    """
    if is_half:
        return f"{epsilon}h"
    if is_double:
        return str(epsilon)
    return f"{epsilon}f"


def compute_scalar(
    value: float, scalar_params: Mapping[str, object],
) -> np.number[Any]:
    """Convert a float to the active COMPUTE_TYPE numpy scalar.

    Ensures COMPUTE_TYPE scalar parameters (thresholds, epsilon, etc.)
    are marshalled with the correct precision per ADR-024.
    """
    compute_dtype = scalar_params.get("_compute_dtype")
    if compute_dtype is None:
        return np.float32(value)
    if compute_dtype == np.float16:
        return np.float16(value)
    if compute_dtype == np.float64:
        return np.float64(value)
    return np.float32(value)
