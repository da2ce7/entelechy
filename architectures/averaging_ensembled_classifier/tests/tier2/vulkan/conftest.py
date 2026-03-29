# tests/tier2/vulkan/conftest.py
"""Session-scoped Vulkan fixtures for Tier 2 kernel correctness tests.

Stub — becomes functional when Phase 5 delivers the Vulkan renderer.
"""
from __future__ import annotations

import pytest

try:
    from src._build_config import BACKEND_VULKAN  # type: ignore[import-not-found]
except ImportError:
    BACKEND_VULKAN = False

if not BACKEND_VULKAN:
    pytest.skip("Vulkan backend not available", allow_module_level=True)

# Session-scoped Vulkan fixtures will be populated in Phase 5.
# The renderer import is deferred until the module is not skipped.
#
# Expected fixtures (mirroring CPU and OpenCL conftest patterns):
#   - hardware_profile: Session-scoped Vulkan hardware profile
#   - precision_fp32: PrecisionConfig.float32()
#   - renderer_fp32: Session-scoped VulkanPlanRenderer for FP32 tests
