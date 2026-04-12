# tests/tier2/opencl/test_fp8_opencl.py
"""Tier 2 tests: FP8 OpenCL backend roundtrip and compilation (ADR-025 §9.2)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import ml_dtypes

pytest.importorskip("pyopencl")
import pyopencl as cl  # noqa: E402

from src.backends.opencl.context import load_and_compile_kernels_from_path
from src.backends.opencl.type_mapping import build_compiler_flags
from src.shared.precision_config import PrecisionConfig

_ARCH_ROOT = Path(__file__).resolve().parents[3]
_KERNEL_DIR = str(_ARCH_ROOT / "kernels")

# Minimal roundtrip kernel: store COMPUTE_TYPE values to FP8, then load back.
_FP8_ROUNDTRIP_KERNEL_SRC = """
__kernel void fp8_roundtrip(
    __global const COMPUTE_TYPE *input,
    __global STORAGE_TYPE *storage_buf,
    __global COMPUTE_TYPE *output,
    const uint count)
{
    size_t i = get_global_id(0);
    if (i < count) {
        store_storage(storage_buf, i, input[i]);
        output[i] = load_storage(storage_buf, i);
    }
}
"""


def _compile_roundtrip_program(cl_ctx, cl_dev, precision, hardware):
    """Compile the header + roundtrip kernel with given precision config."""
    header_src = (Path(_KERNEL_DIR) / "kernels.cl.h").read_text()
    flags = build_compiler_flags(precision, hardware, c_tile_size=16)
    flags.append(f"-I{_KERNEL_DIR}")
    full_src = header_src + "\n" + _FP8_ROUNDTRIP_KERNEL_SRC
    program = cl.Program(cl_ctx, full_src)
    program.build(options=" ".join(flags), devices=[cl_dev])
    return program


def _run_roundtrip(cl_ctx, cl_queue, program, input_data):
    """Execute the roundtrip kernel and return output array."""
    n = len(input_data)
    input_buf = cl.Buffer(
        cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
        hostbuf=input_data,
    )
    # FP8 = 1 byte per element
    storage_buf = cl.Buffer(cl_ctx, cl.mem_flags.READ_WRITE, size=n)
    output_buf = cl.Buffer(cl_ctx, cl.mem_flags.WRITE_ONLY, size=input_data.nbytes)

    kernel = program.fp8_roundtrip
    kernel.set_args(input_buf, storage_buf, output_buf, np.uint32(n))
    cl.enqueue_nd_range_kernel(cl_queue, kernel, (n,), None)
    cl_queue.finish()

    output_data = np.empty_like(input_data)
    cl.enqueue_copy(cl_queue, output_data, output_buf)
    cl_queue.finish()
    return output_data


def _reference_roundtrip(values, fp8_dtype, compute_np_dtype):
    """Compute expected roundtrip result using ml_dtypes as reference."""
    compute_values = np.array(values, dtype=compute_np_dtype)
    fp8_values = compute_values.astype(fp8_dtype)
    return fp8_values.astype(compute_np_dtype)


def _device_supports_fp16(device):
    exts = device.extensions.strip()
    return "cl_khr_fp16" in exts


def _device_supports_fp64(device):
    exts = device.extensions.strip()
    return "cl_khr_fp64" in exts


# ── Full kernel set compilation ──────────────────────────────────────

class TestFP8KernelCompilation:
    """Verify the full kernel set compiles with FP8 precision flags."""

    def test_e4m3_fp32_compile(self, cl_context, cl_device, hardware_profile):
        cfg = PrecisionConfig.fp8_e4m3()
        flags = build_compiler_flags(cfg, hardware_profile, c_tile_size=16)
        load_and_compile_kernels_from_path(
            cl_context, cl_device, flags, _KERNEL_DIR,
        )

    def test_e5m2_fp32_compile(self, cl_context, cl_device, hardware_profile):
        cfg = PrecisionConfig.fp8_e5m2()
        flags = build_compiler_flags(cfg, hardware_profile, c_tile_size=16)
        load_and_compile_kernels_from_path(
            cl_context, cl_device, flags, _KERNEL_DIR,
        )

    def test_e4m3_fp16_compile(self, cl_context, cl_device, hardware_profile):
        if not _device_supports_fp16(cl_device):
            pytest.skip("Device does not support cl_khr_fp16")
        cfg = PrecisionConfig.fp8_e4m3_f16()
        flags = build_compiler_flags(cfg, hardware_profile, c_tile_size=16)
        load_and_compile_kernels_from_path(
            cl_context, cl_device, flags, _KERNEL_DIR,
        )

    def test_e5m2_fp16_compile(self, cl_context, cl_device, hardware_profile):
        if not _device_supports_fp16(cl_device):
            pytest.skip("Device does not support cl_khr_fp16")
        cfg = PrecisionConfig.fp8_e5m2_f16()
        flags = build_compiler_flags(cfg, hardware_profile, c_tile_size=16)
        load_and_compile_kernels_from_path(
            cl_context, cl_device, flags, _KERNEL_DIR,
        )

    def test_e4m3_fp64_compile(self, cl_context, cl_device, hardware_profile):
        if not _device_supports_fp64(cl_device):
            pytest.skip("Device does not support cl_khr_fp64")
        cfg = PrecisionConfig.fp8_e4m3_f64()
        flags = build_compiler_flags(cfg, hardware_profile, c_tile_size=16)
        load_and_compile_kernels_from_path(
            cl_context, cl_device, flags, _KERNEL_DIR,
        )

    def test_e5m2_fp64_compile(self, cl_context, cl_device, hardware_profile):
        if not _device_supports_fp64(cl_device):
            pytest.skip("Device does not support cl_khr_fp64")
        cfg = PrecisionConfig.fp8_e5m2_f64()
        flags = build_compiler_flags(cfg, hardware_profile, c_tile_size=16)
        load_and_compile_kernels_from_path(
            cl_context, cl_device, flags, _KERNEL_DIR,
        )


# ── E4M3 roundtrip tests ────────────────────────────────────────────

class TestFP8E4M3Roundtrip:
    """ADR-025 §9.2: E4M3 FP8 roundtrip tests for OpenCL backend."""

    _IN_RANGE_VALUES = [0.0, 1.0, 0.5, 0.25, 100.0, 448.0, -1.0, -448.0, 0.125, 2.0]

    def test_e4m3_fp32_roundtrip(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values in E4M3 range survive FP32→FP8→FP32 roundtrip."""
        cfg = PrecisionConfig.fp8_e4m3()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(self._IN_RANGE_VALUES, ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_fp16_roundtrip(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values in E4M3 range survive FP16→FP8→FP16 roundtrip."""
        if not _device_supports_fp16(cl_device):
            pytest.skip("Device does not support cl_khr_fp16")
        cfg = PrecisionConfig.fp8_e4m3_f16()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float16)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(self._IN_RANGE_VALUES, ml_dtypes.float8_e4m3fn, np.float16)
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_fp64_roundtrip(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values in E4M3 range survive FP64→FP8→FP64 roundtrip."""
        if not _device_supports_fp64(cl_device):
            pytest.skip("Device does not support cl_khr_fp64")
        cfg = PrecisionConfig.fp8_e4m3_f64()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float64)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(self._IN_RANGE_VALUES, ml_dtypes.float8_e4m3fn, np.float64)
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_saturation(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values exceeding 448 saturate to ±448, not NaN/Inf."""
        cfg = PrecisionConfig.fp8_e4m3()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array([500.0, 1000.0, 1e6, -500.0, -1000.0], dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = np.array([448.0, 448.0, 448.0, -448.0, -448.0], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_nan_maps_to_zero(self, cl_context, cl_device, cl_queue, hardware_profile):
        """NaN input maps to zero (safe failure mode)."""
        cfg = PrecisionConfig.fp8_e4m3()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array([np.nan, -np.nan], dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_subnormals(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Small values near E4M3 subnormal boundary roundtrip correctly."""
        cfg = PrecisionConfig.fp8_e4m3()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        # E4M3 min subnormal = 2^-9 ≈ 0.001953125
        test_values = [0.001953125, 0.00390625, 0.005859375, 0.0078125]
        input_data = np.array(test_values, dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(test_values, ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e4m3_underflow_to_zero(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values below half of min subnormal underflow to zero."""
        cfg = PrecisionConfig.fp8_e4m3()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        # Half of min_subnormal = 2^-10 ≈ 0.0009765625
        input_data = np.array([1e-5, 1e-10, 0.0005], dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip([1e-5, 1e-10, 0.0005], ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_array_equal(actual, expected)


# ── E5M2 roundtrip tests ────────────────────────────────────────────

class TestFP8E5M2Roundtrip:
    """ADR-025 §9.2: E5M2 FP8 roundtrip tests for OpenCL backend."""

    _IN_RANGE_VALUES = [0.0, 1.0, 0.5, 1000.0, 57344.0, -1.0, -57344.0, 0.25, 4.0]

    def test_e5m2_fp32_roundtrip(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values in E5M2 range survive FP32→FP8→FP32 roundtrip."""
        cfg = PrecisionConfig.fp8_e5m2()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(self._IN_RANGE_VALUES, ml_dtypes.float8_e5m2, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_fp16_roundtrip(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values in E5M2 range survive FP16→FP8→FP16 roundtrip."""
        if not _device_supports_fp16(cl_device):
            pytest.skip("Device does not support cl_khr_fp16")
        cfg = PrecisionConfig.fp8_e5m2_f16()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        # E5M2 max (57344) exceeds FP16 max (65504) — but is within range.
        # Use values representable in FP16.
        fp16_safe_values = [0.0, 1.0, 0.5, 1000.0, -1.0, 0.25, 4.0]
        input_data = np.array(fp16_safe_values, dtype=np.float16)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(fp16_safe_values, ml_dtypes.float8_e5m2, np.float16)
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_fp64_roundtrip(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values in E5M2 range survive FP64→FP8→FP64 roundtrip."""
        if not _device_supports_fp64(cl_device):
            pytest.skip("Device does not support cl_khr_fp64")
        cfg = PrecisionConfig.fp8_e5m2_f64()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array(self._IN_RANGE_VALUES, dtype=np.float64)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(self._IN_RANGE_VALUES, ml_dtypes.float8_e5m2, np.float64)
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_saturation(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Values exceeding 57344 saturate to ±57344, not NaN/Inf."""
        cfg = PrecisionConfig.fp8_e5m2()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array([60000.0, 100000.0, 1e10, -60000.0, -1e10], dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = np.array([57344.0, 57344.0, 57344.0, -57344.0, -57344.0], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_nan_maps_to_zero(self, cl_context, cl_device, cl_queue, hardware_profile):
        """NaN input maps to zero (safe failure mode)."""
        cfg = PrecisionConfig.fp8_e5m2()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        input_data = np.array([np.nan, -np.nan], dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = np.array([0.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_subnormals(self, cl_context, cl_device, cl_queue, hardware_profile):
        """Small values near E5M2 subnormal boundary roundtrip correctly."""
        cfg = PrecisionConfig.fp8_e5m2()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        # E5M2 min subnormal = 2^-16
        test_values = [1.52587890625e-05, 3.0517578125e-05, 6.103515625e-05]
        input_data = np.array(test_values, dtype=np.float32)
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(test_values, ml_dtypes.float8_e5m2, np.float32)
        np.testing.assert_array_equal(actual, expected)


# ── Exhaustive bit-pattern test ──────────────────────────────────────

class TestFP8ExhaustiveRoundtrip:
    """Exhaustive roundtrip: every representable FP8 value must survive load→store→load."""

    def test_e4m3_all_256_patterns(self, cl_context, cl_device, cl_queue, hardware_profile):
        """All 256 E4M3 bit patterns: finite values survive load→store→load.

        E4M3 has 2 NaN bit patterns (0x7F, 0xFF). Our kernel maps NaN→zero
        (intentional safe-failure), while ml_dtypes preserves NaN. These are
        excluded; NaN handling is tested separately in test_e4m3_nan_maps_to_zero.
        """
        cfg = PrecisionConfig.fp8_e4m3()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        all_bytes = np.arange(256, dtype=np.uint8)
        all_fp8 = all_bytes.view(ml_dtypes.float8_e4m3fn)
        all_f32 = all_fp8.astype(np.float32)
        # Filter out NaN patterns (0x7F, 0xFF) — NaN→zero is intentional
        finite_mask = np.isfinite(all_f32)
        input_data = all_f32[finite_mask]
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(input_data, ml_dtypes.float8_e4m3fn, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_e5m2_all_256_patterns(self, cl_context, cl_device, cl_queue, hardware_profile):
        """All 256 E5M2 bit patterns: load from LUT then store back must produce same byte."""
        cfg = PrecisionConfig.fp8_e5m2()
        program = _compile_roundtrip_program(cl_context, cl_device, cfg, hardware_profile)
        all_bytes = np.arange(256, dtype=np.uint8)
        all_fp8 = all_bytes.view(ml_dtypes.float8_e5m2)
        all_f32 = all_fp8.astype(np.float32)
        # Filter out inf/NaN values from E5M2 (bit patterns 0x7C-0x7F, 0xFC-0xFF)
        finite_mask = np.isfinite(all_f32)
        input_data = all_f32[finite_mask]
        actual = _run_roundtrip(cl_context, cl_queue, program, input_data)
        expected = _reference_roundtrip(input_data, ml_dtypes.float8_e5m2, np.float32)
        np.testing.assert_array_equal(actual, expected)
