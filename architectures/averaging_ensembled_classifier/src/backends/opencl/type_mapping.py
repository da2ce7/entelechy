# src/backends/opencl/type_mapping.py
"""PrecisionConfig -> OpenCL compiler flags and dtype mapping (ADR-008)."""
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

    Generates flags for all CONTRACT.md Article 6 mandatory symbols:
    SCALAR_TYPE, SIMD_WIDTH, C_TILE_SIZE, SCALAR_IS_HALF,
    NUMERICAL_STABILITY_EPSILON, LOCAL_MEM_BANK_PADDING.
    """
    cl_type = numpy_dtype_to_cl_type_name(precision)
    is_half = 1 if precision.numpy_dtype == np.dtype(np.float16) else 0
    eps = f"{precision.epsilon}f" if not is_half else str(precision.epsilon)
    return [
        f"-DSCALAR_TYPE={cl_type}",
        f"-DSIMD_WIDTH={hardware.simd_width}",
        f"-DC_TILE_SIZE={c_tile_size}",
        f"-DSCALAR_IS_HALF={is_half}",
        f"-DNUMERICAL_STABILITY_EPSILON={eps}",
        "-DLOCAL_MEM_BANK_PADDING=1",
    ]


def numpy_dtype_to_cl_type_name(precision: PrecisionConfig) -> str:
    """Map PrecisionConfig.numpy_dtype to OpenCL C type name."""
    mapping = {
        np.dtype(np.float32): "float",
        np.dtype(np.float16): "half",
    }
    return mapping[precision.numpy_dtype]
