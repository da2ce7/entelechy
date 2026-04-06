# tests/tier2/test_fp8_cross_backend.py
"""Cross-backend FP8 conversion consistency.

Ensures LUT-based (CPU, OpenCL) and arithmetic (Vulkan) backends produce
bit-identical FP8→FP32 decode values, and that all agree with the Python
ml_dtypes reference for all 256 bit patterns in both E4M3 and E5M2 formats.

Strategy:
- CPU and OpenCL: parse their generated LUT header files (.gen.h).
- Vulkan: GPU-dispatch the fp8_roundtrip SPIR-V shader.
- Python: ml_dtypes as the authoritative IEEE FP8 reference.

Finite patterns are compared bit-exact. NaN patterns are verified separately
(backends intentionally map NaN → 0.0 per ADR-025 §7).
"""
from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

import ml_dtypes

from src.shared.precision_config import PrecisionConfig
from tests.conftest import BUILD_CONFIG

_ARCH_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Generated LUT file paths
_CPU_LUT_PATH = (
    _ARCH_ROOT / "src" / "backends" / "cpu" / "kernel_sources" / "cpu_fp8_lut.gen.h"
)
_OPENCL_LUT_PATH = _ARCH_ROOT / "kernels" / "fp8_lut.gen.h"


# ── Helpers ──────────────────────────────────────────────────────────


def _parse_lut_from_header(
    header_path: pathlib.Path, array_name: str,
) -> np.ndarray:
    """Extract a float[256] array from a generated .gen.h file."""
    text = header_path.read_text()
    pattern = rf"{re.escape(array_name)}\[256\]\s*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {array_name}[256] in {header_path}")
    values: list[float] = []
    for tok in match.group(1).split(","):
        tok = tok.strip().rstrip("f")
        if tok:
            values.append(float(tok))
    assert len(values) == 256, f"Expected 256 values in {array_name}, got {len(values)}"
    return np.array(values, dtype=np.float32)


def _python_reference(fp8_format: str) -> np.ndarray:
    """ml_dtypes FP32 decode for all 256 FP8 bit patterns."""
    dtype = (
        ml_dtypes.float8_e4m3fn if fp8_format == "e4m3" else ml_dtypes.float8_e5m2
    )
    return np.arange(256, dtype=np.uint8).view(dtype).astype(np.float32)


def _cpu_lut(fp8_format: str) -> np.ndarray | None:
    """CPU LUT decode for all 256 patterns, or None if unavailable."""
    if not BUILD_CONFIG.get("cpu", False) or not _CPU_LUT_PATH.exists():
        return None
    name = (
        "cpu_fp8_e4m3_to_float_lut"
        if fp8_format == "e4m3"
        else "cpu_fp8_e5m2_to_float_lut"
    )
    return _parse_lut_from_header(_CPU_LUT_PATH, name)


def _opencl_lut(fp8_format: str) -> np.ndarray | None:
    """OpenCL LUT decode for all 256 patterns, or None if unavailable."""
    if not BUILD_CONFIG.get("opencl", False) or not _OPENCL_LUT_PATH.exists():
        return None
    name = (
        "fp8_e4m3_to_float_lut"
        if fp8_format == "e4m3"
        else "fp8_e5m2_to_float_lut"
    )
    return _parse_lut_from_header(_OPENCL_LUT_PATH, name)


def _find_spv(shader_name: str, suffix: str) -> pathlib.Path | None:
    """Find compiled .spv file for a Vulkan shader variant."""
    spv_name = f"{shader_name}{suffix}.spv"
    candidates = [
        _ARCH_ROOT / d / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name
        for d in ("builddir-vulkan", "builddir")
    ]
    build_dir = _ARCH_ROOT / "build"
    if build_dir.is_dir():
        for child in build_dir.iterdir():
            if child.is_dir() and child.name.startswith("cp"):
                candidates.append(
                    child / "src" / "backends" / "vulkan" / "kernel_sources" / spv_name
                )
    for c in candidates:
        if c.exists():
            return c
    return None


def _vulkan_roundtrip(
    fp8_format: str, input_f32: np.ndarray,
) -> np.ndarray | None:
    """Dispatch Vulkan fp8_roundtrip shader for the given FP32 values.

    Returns the GPU-roundtripped FP32 output, or None if Vulkan is
    unavailable or the SPIR-V file is missing.
    """
    if not BUILD_CONFIG.get("vulkan", False):
        return None

    suffix = "_s8e4c32x32" if fp8_format == "e4m3" else "_s8e5c32x32"
    spv_path = _find_spv("fp8_roundtrip", suffix)
    if spv_path is None:
        return None

    try:
        from src.backends.vulkan.context import VulkanContext
        from tests.tier2.vulkan.test_fp8_vulkan import _dispatch_roundtrip
    except ImportError:
        return None

    ctx = VulkanContext(enable_validation=False)
    try:
        return _dispatch_roundtrip(ctx, spv_path, input_f32, np.float32)
    finally:
        ctx.destroy()


def _nan_indices(fp8_format: str) -> list[int]:
    """Return byte indices of NaN bit patterns for the given FP8 format.

    NaN patterns map to 0.0 in backends.  Does NOT include inf patterns
    (E5M2 only): those map to ±max_finite, not zero.
    """
    if fp8_format == "e4m3":
        # E4M3fn: mant=7 at exp=15 → NaN (0x7F positive, 0xFF negative)
        return [0x7F, 0xFF]
    # E5M2: exp=0x1F (31) with mant!=0 → NaN; mant==0 → ±inf (not NaN)
    result = []
    for i in range(256):
        exp5 = (i >> 2) & 0x1F
        mant2 = i & 0x3
        if exp5 == 0x1F and mant2 != 0:
            result.append(i)
    return result


def _inf_indices(fp8_format: str) -> list[int]:
    """Return byte indices of ±inf bit patterns (E5M2 only).

    Inf patterns map to ±max_finite in backends (saturation).
    E4M3fn has no infinity representation.
    """
    if fp8_format == "e4m3":
        return []
    # E5M2: exp=0x1F, mant=0 → ±inf (0x7C = +inf, 0xFC = -inf)
    return [0x7C, 0xFC]


# ── Tests ────────────────────────────────────────────────────────────


def _available_backends() -> set[str]:
    """Return set of backends available for FP8 testing."""
    return {b for b in ("cpu", "opencl", "vulkan") if BUILD_CONFIG.get(b, False)}


pytestmark = pytest.mark.skipif(
    len(_available_backends()) < 2,
    reason="Cross-backend test requires ≥2 FP8-capable backends",
)


class TestCrossBackendFP8Consistency:
    """ADR-025 §9: All backends must produce identical FP8↔FP32 decode."""

    @pytest.mark.parametrize("fp8_format", ["e4m3", "e5m2"])
    def test_finite_patterns_match_across_backends(self, fp8_format):
        """Finite FP8 bit patterns decode identically across all backends."""
        python_ref = _python_reference(fp8_format)
        finite_mask = np.isfinite(python_ref)
        python_finite = python_ref[finite_mask]

        results: dict[str, np.ndarray] = {"python": python_finite}

        cpu = _cpu_lut(fp8_format)
        if cpu is not None:
            results["cpu"] = cpu[finite_mask]

        opencl = _opencl_lut(fp8_format)
        if opencl is not None:
            results["opencl"] = opencl[finite_mask]

        vulkan = _vulkan_roundtrip(fp8_format, python_finite)
        if vulkan is not None:
            results["vulkan"] = vulkan

        if len(results) < 2:
            pytest.skip(f"Need ≥2 backends, have: {set(results.keys())}")

        for name, values in results.items():
            if name == "python":
                continue
            np.testing.assert_array_equal(
                python_finite,
                values,
                err_msg=(
                    f"FP8 {fp8_format} finite-pattern decode mismatch: "
                    f"python vs {name}"
                ),
            )

    @pytest.mark.parametrize("fp8_format", ["e4m3", "e5m2"])
    def test_nan_patterns_map_to_zero_in_luts(self, fp8_format):
        """NaN bit patterns map to 0.0 in CPU and OpenCL LUTs."""
        nan_idx = _nan_indices(fp8_format)

        cpu = _cpu_lut(fp8_format)
        opencl = _opencl_lut(fp8_format)

        if cpu is None and opencl is None:
            pytest.skip("No LUT-based backend available")

        for idx in nan_idx:
            if cpu is not None:
                assert cpu[idx] == 0.0, (
                    f"CPU LUT {fp8_format} index 0x{idx:02X}: "
                    f"expected 0.0, got {cpu[idx]}"
                )
            if opencl is not None:
                assert opencl[idx] == 0.0, (
                    f"OpenCL LUT {fp8_format} index 0x{idx:02X}: "
                    f"expected 0.0, got {opencl[idx]}"
                )

    @pytest.mark.parametrize("fp8_format", ["e4m3", "e5m2"])
    def test_inf_patterns_saturate_to_max_in_luts(self, fp8_format):
        """±inf bit patterns (E5M2 only) map to ±max_finite in LUTs."""
        inf_idx = _inf_indices(fp8_format)
        if not inf_idx:
            pytest.skip(f"{fp8_format} has no inf representation")

        max_val = 57344.0  # E5M2 max finite

        cpu = _cpu_lut(fp8_format)
        opencl = _opencl_lut(fp8_format)

        if cpu is None and opencl is None:
            pytest.skip("No LUT-based backend available")

        for idx in inf_idx:
            sign = -1.0 if idx >= 0x80 else 1.0
            expected = sign * max_val
            if cpu is not None:
                assert cpu[idx] == expected, (
                    f"CPU LUT {fp8_format} index 0x{idx:02X}: "
                    f"expected {expected}, got {cpu[idx]}"
                )
            if opencl is not None:
                assert opencl[idx] == expected, (
                    f"OpenCL LUT {fp8_format} index 0x{idx:02X}: "
                    f"expected {expected}, got {opencl[idx]}"
                )

    def test_cpu_opencl_lut_identical(self):
        """CPU and OpenCL LUTs are bit-identical for all 256×2 patterns."""
        cpu_e4 = _cpu_lut("e4m3")
        cpu_e5 = _cpu_lut("e5m2")
        opencl_e4 = _opencl_lut("e4m3")
        opencl_e5 = _opencl_lut("e5m2")

        if cpu_e4 is None or opencl_e4 is None:
            pytest.skip("Both CPU and OpenCL backends required")

        np.testing.assert_array_equal(
            cpu_e4, opencl_e4,
            err_msg="E4M3 LUT mismatch: cpu vs opencl",
        )
        np.testing.assert_array_equal(
            cpu_e5, opencl_e5,
            err_msg="E5M2 LUT mismatch: cpu vs opencl",
        )
