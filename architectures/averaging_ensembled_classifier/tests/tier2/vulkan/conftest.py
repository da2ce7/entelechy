# tests/tier2/vulkan/conftest.py
"""Session-scoped Vulkan fixtures for Tier 2 kernel correctness tests."""
from __future__ import annotations

import pytest

try:
    from src._build_config import BACKEND_VULKAN  # type: ignore[import-not-found]
except ImportError:
    BACKEND_VULKAN = False

if not BACKEND_VULKAN:
    pytest.skip("Vulkan backend not available", allow_module_level=True)

from src.backends.vulkan.context import VulkanContext  # noqa: E402
from src.backends.vulkan.discovery import discover_hardware  # noqa: E402
from src.backends.vulkan.renderer import VulkanPlanRenderer  # noqa: E402
from src.shared.precision_config import PrecisionConfig  # noqa: E402


@pytest.fixture(scope="session")
def vulkan_context():
    """Session-scoped Vulkan context."""
    ctx = VulkanContext(enable_validation=True)
    yield ctx
    ctx.destroy()


@pytest.fixture(scope="session")
def hardware_profile(vulkan_context):
    """Session-scoped Vulkan hardware profile."""
    return discover_hardware(vulkan_context)


@pytest.fixture(scope="session")
def precision_fp32():
    return PrecisionConfig.float32()


@pytest.fixture(scope="session")
def renderer_fp32(vulkan_context):
    """Session-scoped VulkanPlanRenderer for FP32 tests (CCE)."""
    renderer = VulkanPlanRenderer(context=vulkan_context, problem_type=0)
    yield renderer
    renderer.destroy()


@pytest.fixture(scope="session")
def renderer_fp32_bce(vulkan_context):
    """Session-scoped VulkanPlanRenderer for FP32 BCE tests."""
    renderer = VulkanPlanRenderer(context=vulkan_context, problem_type=1)
    yield renderer
    renderer.destroy()

