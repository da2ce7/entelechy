# kernel_signatures/__init__.py — SHIM (Phase 0)
# This package has moved to src/backends/opencl/kernel_bindings/. This shim
# re-exports all public symbols for backward compatibility. Remove in Phase 6.
#
# Canonical location: src/backends/opencl/kernel_bindings/

from ..backends.opencl.kernel_bindings import *  # noqa: F401,F403
