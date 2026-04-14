# src/shared/kernel_contracts/__init__.py
"""Backend-neutral kernel contracts (ADR-007).

Re-exports all types and contract instances from the authoritative
``kernels_cl_h.py`` header mirror.
"""

# ── Core types ────────────────────────────────────────────────────────
from .kernels_cl_h import Dim as Dim
from .kernels_cl_h import PaddingType as PaddingType
from .kernels_cl_h import PrecisionRole as PrecisionRole
from .kernels_cl_h import InitContract as InitContract
from .kernels_cl_h import PlacementContract as PlacementContract
from .kernels_cl_h import BufferParam as BufferParam
from .kernels_cl_h import ScalarParam as ScalarParam
from .kernels_cl_h import KernelContract as KernelContract

# ── Constants ─────────────────────────────────────────────────────────
from .kernels_cl_h import PROBLEM_TYPE_CCE as PROBLEM_TYPE_CCE
from .kernels_cl_h import PROBLEM_TYPE_BCE as PROBLEM_TYPE_BCE
from .kernels_cl_h import AGG_MODE_SUM as AGG_MODE_SUM
from .kernels_cl_h import AGG_MODE_AVERAGE as AGG_MODE_AVERAGE
from .kernels_cl_h import SENTINEL_ABSENT_PARTIAL as SENTINEL_ABSENT_PARTIAL

# ── Registry ──────────────────────────────────────────────────────────
from .kernels_cl_h import KERNEL_REGISTRY as KERNEL_REGISTRY
