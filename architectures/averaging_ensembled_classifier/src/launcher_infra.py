# src/launcher_infra.py — SHIM (Phase 0)
# This module has moved to src/backends/opencl/launcher_infra.py. This shim
# re-exports all public symbols for backward compatibility. Remove in Phase 6.
#
# Canonical location: src/backends/opencl/launcher_infra.py

from .backends.opencl.launcher_infra import *  # noqa: F401,F403
