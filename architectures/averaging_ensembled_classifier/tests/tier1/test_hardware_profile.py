# tests/tier1/test_hardware_profile.py
"""HardwareProfile field semantics."""
import pytest

from src.shared.hardware_profile import HardwareProfile


class TestConstruction:
    def test_basic(self):
        hp = HardwareProfile(
            simd_width=16, cache_line_bytes=64,
            max_reduce_fan_in=256, max_local_mem_bytes=65536,
            global_mem_bytes=4 * 1024**3,
        )
        assert hp.simd_width == 16
        assert hp.max_reduce_fan_in == 256

    def test_cpu_backend_none_local_mem(self):
        hp = HardwareProfile(
            simd_width=4, cache_line_bytes=64,
            max_reduce_fan_in=128, max_local_mem_bytes=None,
            global_mem_bytes=16 * 1024**3,
        )
        assert hp.max_local_mem_bytes is None

    def test_frozen(self):
        hp = HardwareProfile(
            simd_width=16, cache_line_bytes=64,
            max_reduce_fan_in=256, max_local_mem_bytes=65536,
            global_mem_bytes=4 * 1024**3,
        )
        with pytest.raises(AttributeError):
            hp.simd_width = 32  # type: ignore[misc]
