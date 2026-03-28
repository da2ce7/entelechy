# src/cl_context_manager.py — SHIM (Phase 0)
# This module has moved to src/backends/opencl/context.py. This shim re-exports
# all public symbols for backward compatibility. Remove in Phase 6.
#
# Canonical location: src/backends/opencl/context.py

from .backends.opencl.context import *  # noqa: F401,F403
