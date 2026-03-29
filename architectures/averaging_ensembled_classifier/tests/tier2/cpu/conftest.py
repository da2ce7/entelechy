# tests/tier2/cpu/conftest.py
"""Session-scoped CPU fixtures for Tier 2 kernel correctness tests."""
from __future__ import annotations

import pytest

from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]

if not BACKEND_CPU:
    pytest.skip("CPU backend not available", allow_module_level=True)

from src.backends.cpu.discovery import discover_hardware, detect_thread_count  # noqa: E402
from src.backends.cpu.renderer import CPUPlanRenderer  # noqa: E402


@pytest.fixture(scope="session")
def hardware_profile():
    """Session-scoped CPU hardware profile."""
    return discover_hardware()


@pytest.fixture(scope="session")
def thread_count():
    """Auto-detected thread count for the test host."""
    return detect_thread_count()


@pytest.fixture(scope="session")
def precision_fp32():
    from src.shared.precision_config import PrecisionConfig
    return PrecisionConfig.float32()


@pytest.fixture(scope="session")
def renderer_fp32():
    """Session-scoped CPUPlanRenderer for FP32 tests.

    Uses auto-detected thread count. Session scoping amortizes
    the library load and layout verification cost across all tests.
    """
    return CPUPlanRenderer()


@pytest.fixture
def renderer_single_thread():
    """Per-test CPUPlanRenderer with a single worker thread.

    Uses threads=1 to exercise the serial execution path.
    """
    return CPUPlanRenderer(thread_count=1)
