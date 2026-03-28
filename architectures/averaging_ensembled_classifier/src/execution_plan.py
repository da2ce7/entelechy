# src/execution_plan.py — SHIM (Phase 0)
# This module has moved to src/backends/opencl/execution_plan.py. This shim
# re-exports all public symbols for backward compatibility. Remove in Phase 6.
#
# Canonical location: src/backends/opencl/execution_plan.py
# ProblemTypeStrategy family: src/shared/problem_type_strategy.py

from .backends.opencl.execution_plan import *  # noqa: F401,F403
from .shared.problem_type_strategy import *  # noqa: F401,F403
