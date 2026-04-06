# tests/tier2/cpu/test_fp8_cpu_availability.py
"""Test graceful failure when _Float16 is unavailable (ADR-025 §5.3)."""
from __future__ import annotations

import pytest

from src._build_config import BACKEND_CPU  # type: ignore[import-not-found]

if not BACKEND_CPU:
    pytest.skip("CPU backend not available", allow_module_level=True)

try:
    from src.backends.cpu._fp8_variants import AVAILABLE_FP8_VARIANTS, HAS_FLOAT16
except ImportError:
    AVAILABLE_FP8_VARIANTS: list[str] = []
    HAS_FLOAT16: bool = False


class TestFP8VariantAvailability:
    """Validate FP8 variant availability reported by the build system."""

    def test_fp32_compute_variants_always_available(self):
        """FP8 variants with FP32/FP64 compute are always compiled."""
        required = [
            "s8e4c32x32", "s8e4c32x64", "s8e4c64x64",
            "s8e5c32x32", "s8e5c32x64", "s8e5c64x64",
        ]
        for suffix in required:
            assert suffix in AVAILABLE_FP8_VARIANTS, (
                f"FP8 variant {suffix} should always be available"
            )

    @pytest.mark.skipif(not HAS_FLOAT16, reason="_Float16 not available")
    def test_fp16_compute_variants_when_available(self):
        """FP16 compute variants are present when _Float16 is supported."""
        fp16_variants = [
            "s8e4c16x32", "s8e4c16x64",
            "s8e5c16x32", "s8e5c16x64",
        ]
        for suffix in fp16_variants:
            assert suffix in AVAILABLE_FP8_VARIANTS, (
                f"FP8+FP16 variant {suffix} should be available when HAS_FLOAT16=True"
            )

    @pytest.mark.skipif(HAS_FLOAT16, reason="_Float16 IS available")
    def test_fp16_compute_variants_absent_when_unavailable(self):
        """FP16 compute variants are absent when _Float16 is not supported."""
        fp16_variants = [
            "s8e4c16x32", "s8e4c16x64",
            "s8e5c16x32", "s8e5c16x64",
        ]
        for suffix in fp16_variants:
            assert suffix not in AVAILABLE_FP8_VARIANTS, (
                f"FP8+FP16 variant {suffix} should not be available when HAS_FLOAT16=False"
            )

    def test_available_variants_count(self):
        """At least 6 FP8 variants are always available."""
        fp8_variants = [v for v in AVAILABLE_FP8_VARIANTS if v.startswith("s8")]
        assert len(fp8_variants) >= 6, (
            f"Expected at least 6 FP8 variants, got {len(fp8_variants)}: {fp8_variants}"
        )


class TestFP16ComputeAvailability:
    """Validate graceful failure when _Float16 not available."""

    @pytest.mark.skipif(HAS_FLOAT16, reason="_Float16 IS available; skip unavailability test")
    def test_c16_suffix_not_in_available_variants(self):
        """When _Float16 unavailable, c16 suffixes are not listed."""
        c16_variants = [v for v in AVAILABLE_FP8_VARIANTS if "c16" in v]
        assert len(c16_variants) == 0, (
            f"c16 variants should not be available without _Float16: {c16_variants}"
        )

    @pytest.mark.skipif(not HAS_FLOAT16, reason="_Float16 not available; skip availability test")
    def test_c16_suffix_in_available_variants(self):
        """When _Float16 available, c16 suffixes are listed."""
        c16_variants = [v for v in AVAILABLE_FP8_VARIANTS if "c16" in v]
        assert len(c16_variants) == 4, (
            f"Expected 4 c16 variants with _Float16: {c16_variants}"
        )
