# tests/tier1/test_fp8_precision_config.py
"""FP8 precision configuration tests (ADR-025, Phase 9A).

Tests cover:
- Precision role constraints (FP8 storage-only, FP16 state permitted)
- All valid FP8 storage + compute/state combinations
- FP8 factory classmethods
- FP8 derived constants
- Golden reference values via ml_dtypes
- E5M2 defensive loading (inf/NaN handling)
"""
from __future__ import annotations

import numpy as np
import pytest
import ml_dtypes

from src.shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2


class TestPrecisionRoleConstraints:
    """ADR-025 §2.2: Precision role constraints (FP8 storage-only)."""

    _F32 = np.dtype(np.float32)
    _F32_MAX = float(np.finfo(np.float32).max)
    _F32_MIN_POS = float(np.finfo(np.float32).smallest_subnormal)
    _F32_EPS = float(np.finfo(np.float32).eps)

    def test_fp8_e4m3_compute_rejection(self):
        """FP8 compute raises ValueError."""
        with pytest.raises(ValueError, match="FP8 compute is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=FP8_E4M3,
                state_dtype=self._F32,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=448.0,
                compute_epsilon=0.125,
            )

    def test_fp8_e5m2_compute_rejection(self):
        """FP8 compute raises ValueError (E5M2 variant)."""
        with pytest.raises(ValueError, match="FP8 compute is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=FP8_E5M2,
                state_dtype=self._F32,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=57344.0,
                compute_epsilon=0.25,
            )

    def test_fp8_e4m3_state_rejection(self):
        """FP8 state raises ValueError."""
        with pytest.raises(ValueError, match="FP8 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=self._F32,
                state_dtype=FP8_E4M3,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=self._F32_MAX,
                compute_epsilon=self._F32_EPS,
            )

    def test_fp8_e5m2_state_rejection(self):
        """FP8 state raises ValueError (E5M2 variant)."""
        with pytest.raises(ValueError, match="FP8 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=self._F32,
                state_dtype=FP8_E5M2,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=self._F32_MAX,
                compute_epsilon=self._F32_EPS,
            )

    def test_fp16_state_construction_valid(self):
        """FP16 state constructs successfully via direct construction (ADR-020 three-role model).

        The float16() factory is deleted (common footgun), but direct construction
        with state_dtype=np.float16 remains valid — users who explicitly choose
        FP16 state accept the precision limitations.
        """
        f16_info = np.finfo(np.float16)
        cfg = PrecisionConfig(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float16),
            storage_fp_format_max=float(f16_info.max),
            storage_fp_min_positive=float(f16_info.smallest_subnormal),
            storage_mantissa_bits=f16_info.nmant,
            compute_fp_format_max=float(f16_info.max),
            compute_epsilon=float(f16_info.eps),
        )
        assert cfg.state_dtype == np.dtype(np.float16)

    def test_fp8_storage_with_fp16_state_valid(self):
        """FP8 storage + FP16 compute + FP16 state is valid (itemsize: 1 ≤ 2 ≤ 2)."""
        f16_info = np.finfo(np.float16)
        cfg = PrecisionConfig(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float16),
            storage_fp_format_max=448.0,
            storage_fp_min_positive=0.001953125,
            storage_mantissa_bits=3,
            compute_fp_format_max=float(f16_info.max),
            compute_epsilon=float(f16_info.eps),
        )
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.state_dtype == np.dtype(np.float16)


class TestFP8ValidCombinations:
    """ADR-025: All valid FP8 storage + compute/state combinations."""

    @pytest.mark.parametrize("compute_dtype,state_dtype", [
        (np.float16, np.float32),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ])
    def test_e4m3_valid_combinations(self, compute_dtype, state_dtype):
        """E4M3 storage accepts all valid compute/state combinations."""
        compute_info = np.finfo(compute_dtype)
        cfg = PrecisionConfig(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(compute_dtype),
            state_dtype=np.dtype(state_dtype),
            storage_fp_format_max=448.0,
            storage_fp_min_positive=0.001953125,
            storage_mantissa_bits=3,
            compute_fp_format_max=float(compute_info.max),
            compute_epsilon=float(compute_info.eps),
        )
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(compute_dtype)
        assert cfg.state_dtype == np.dtype(state_dtype)

    @pytest.mark.parametrize("compute_dtype,state_dtype", [
        (np.float16, np.float32),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ])
    def test_e5m2_valid_combinations(self, compute_dtype, state_dtype):
        """E5M2 storage accepts all valid compute/state combinations."""
        compute_info = np.finfo(compute_dtype)
        cfg = PrecisionConfig(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(compute_dtype),
            state_dtype=np.dtype(state_dtype),
            storage_fp_format_max=57344.0,
            storage_fp_min_positive=0.0000152587890625,
            storage_mantissa_bits=2,
            compute_fp_format_max=float(compute_info.max),
            compute_epsilon=float(compute_info.eps),
        )
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(compute_dtype)
        assert cfg.state_dtype == np.dtype(state_dtype)


class TestFP8Factories:
    """ADR-025 §2.3: FP8 factory classmethods."""

    def test_fp8_e4m3_factory(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(np.float32)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e5m2_factory(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(np.float32)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e4m3_f16_factory(self):
        cfg = PrecisionConfig.fp8_e4m3_f16()
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(np.float16)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e5m2_f16_factory(self):
        cfg = PrecisionConfig.fp8_e5m2_f16()
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(np.float16)
        assert cfg.state_dtype == np.dtype(np.float32)

    def test_fp8_e4m3_f64_factory(self):
        cfg = PrecisionConfig.fp8_e4m3_f64()
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(np.float64)
        assert cfg.state_dtype == np.dtype(np.float64)

    def test_fp8_e5m2_f64_factory(self):
        cfg = PrecisionConfig.fp8_e5m2_f64()
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(np.float64)
        assert cfg.state_dtype == np.dtype(np.float64)


class TestFP8DerivedConstants:
    """ADR-025 §2.4: FP8 derived constants."""

    def test_e4m3_storage_fp_format_max(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_fp_format_max == 448.0

    def test_e5m2_storage_fp_format_max(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_fp_format_max == 57344.0

    def test_e4m3_storage_mantissa_bits(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_mantissa_bits == 3

    def test_e5m2_storage_mantissa_bits(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_mantissa_bits == 2

    def test_e4m3_buffer_sizing(self):
        """FP8 storage buffer is 1/4 size of FP32."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp32 = PrecisionConfig.float32()
        assert cfg_fp8.storage_dtype.itemsize == 1
        assert cfg_fp32.storage_dtype.itemsize == 4
        assert cfg_fp8.storage_dtype.itemsize * 4 == cfg_fp32.storage_dtype.itemsize


class TestFP8GoldenReference:
    """Golden reference tests using ml_dtypes directly."""

    def test_e4m3_roundtrip_exact_values(self):
        """Known E4M3 bit patterns produce expected float values."""
        test_cases = [
            (0x00, 0.0),
            (0x38, 1.0),
            (0x3C, 1.5),
            (0x40, 2.0),
            (0x7E, 448.0),
            (0x80, -0.0),
            (0xB8, -1.0),
            (0xFE, -448.0),
        ]
        for bits, expected in test_cases:
            arr = np.array([bits], dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
            actual = float(arr[0])
            assert actual == expected, f"E4M3 0x{bits:02X}: expected {expected}, got {actual}"

    def test_e5m2_roundtrip_exact_values(self):
        """Known E5M2 bit patterns produce expected float values."""
        test_cases = [
            (0x00, 0.0),
            (0x3C, 1.0),
            (0x40, 2.0),
            (0x7B, 57344.0),
            (0x80, -0.0),
            (0xBC, -1.0),
            (0xFB, -57344.0),
        ]
        for bits, expected in test_cases:
            arr = np.array([bits], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
            actual = float(arr[0])
            assert actual == expected, f"E5M2 0x{bits:02X}: expected {expected}, got {actual}"

    def test_e4m3_no_infinities(self):
        """E4M3fn has no infinities (only 2 NaN patterns: 0x7F, 0xFF)."""
        all_bits = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        all_float = all_bits.astype(np.float32)
        assert not np.any(np.isinf(all_float)), "E4M3fn should have no inf values"
        # Exactly 254 finite values (256 - 2 NaN at 0x7F and 0xFF)
        finite_count = np.sum(np.isfinite(all_float))
        assert finite_count == 254, (
            f"E4M3fn should have 254 finite bit patterns, got {finite_count}"
        )

    def test_e5m2_special_values(self):
        """E5M2 has IEEE-like inf/NaN (unlike E4M3fn which is all-finite)."""
        all_bits = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        all_float = all_bits.astype(np.float32)

        finite_count = np.sum(np.isfinite(all_float))
        assert finite_count == 248, (
            f"E5M2 should have 248 finite bit patterns, got {finite_count}"
        )

        assert np.isinf(all_float[0x7C]), "E5M2 0x7C should be +inf"
        assert np.isinf(all_float[0xFC]), "E5M2 0xFC should be -inf"
        assert np.isnan(all_float[0x7D]), "E5M2 0x7D should be NaN"
        assert np.isnan(all_float[0xFF]), "E5M2 0xFF should be NaN"

    def test_ml_dtypes_version_compatibility(self):
        """ml_dtypes version is compatible with expected FP8 behavior."""
        version = tuple(int(x) for x in ml_dtypes.__version__.split('.')[:2])
        assert version >= (0, 2), f"ml_dtypes {ml_dtypes.__version__} < 0.2.0"


class TestE5M2DefensiveLoading:
    """E5M2 inf/NaN handling — validates ml_dtypes behavior for LUT generation."""

    def test_e5m2_inf_indices_return_inf(self):
        """ml_dtypes E5M2 +inf indices (0x7C) produce inf."""
        arr_inf = np.array([0x7C], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        assert np.isinf(float(arr_inf[0])), "ml_dtypes E5M2 0x7C should be +inf"

    def test_e5m2_nan_indices_return_nan(self):
        """ml_dtypes E5M2 NaN indices produce NaN."""
        nan_indices = [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]
        for idx in nan_indices:
            arr_nan = np.array([idx], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
            assert np.isnan(float(arr_nan[0])), f"ml_dtypes E5M2 0x{idx:02X} should be NaN"
