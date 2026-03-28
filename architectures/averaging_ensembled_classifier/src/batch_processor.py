# src/batch_processor.py — SHIM (Phase 0)
# This module has moved to src/backends/opencl/batch_processor.py. This shim
# re-exports all public symbols for backward compatibility. Remove in Phase 6.
#
# Canonical location: src/backends/opencl/batch_processor.py

from .backends.opencl.batch_processor import *  # noqa: F401,F403
