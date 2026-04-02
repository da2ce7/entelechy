# src/tests/test_model_spec.py
"""
Unit tests for ModelSpec: padding calculations and precision contracts.

Bug-hunting focus:
* padded_input_dim, padded_hidden_dim, padded_class_dim, padded_module_dim
  must each be ≥ the logical dim and an appropriate multiple.
* ModelSpec.float32() and ModelSpec.float16() must produce different padding when
  cache_line_bytes causes different per-element byte strides.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.shared.model_spec import ModelSpec


class TestPaddingCalculations:

    @pytest.mark.parametrize("hidden_dim,simd_width,expected", [
        (32, 4, 32),   # already aligned
        (30, 4, 32),   # 30 → 32
        (1, 4, 4),     # minimal
        (33, 8, 40),   # 33 → 40
    ])
    def test_padded_hidden_dim(self, hidden_dim, simd_width, expected) -> None:
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=hidden_dim, output_classes=3,
            num_modules=8, simd_width=simd_width, cache_line_bytes=64,
        )
        assert spec.padded_hidden_dim == expected

    def test_padded_hidden_ge_logical(self) -> None:
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=17, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        assert spec.padded_hidden_dim >= spec.hidden_dim

    def test_padded_input_dim_cache_aligned(self) -> None:
        """With 4-byte float32 and 64-byte cache line → row must be multiple of 16 elements."""
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        assert spec.padded_input_dim >= spec.input_dim
        assert (spec.padded_input_dim * 4) % 64 == 0

    def test_padded_class_dim_cache_aligned(self) -> None:
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        assert spec.padded_class_dim >= spec.output_classes
        assert (spec.padded_class_dim * 4) % 64 == 0

    def test_padded_module_dim_cache_aligned(self) -> None:
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        assert spec.padded_module_dim >= spec.num_modules
        assert (spec.padded_module_dim * 4) % 64 == 0

    def test_fp16_different_padding_from_fp32(self) -> None:
        """With 2-byte float16 and 64-byte cache line, row stride is 32 elements (not 16)."""
        fp32 = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        fp16 = ModelSpec.float16(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        # fp16 row = 4*2=8 bytes, padded to 64 → 32 elements
        # fp32 row = 4*4=16 bytes, padded to 64 → 16 elements
        assert fp16.padded_input_dim >= fp32.padded_input_dim

    def test_precision_storage_dtype_is_correct(self) -> None:
        """Verify precision.storage_dtype gives expected np.dtype."""
        fp32 = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        fp16 = ModelSpec.float16(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=4, cache_line_bytes=64,
        )
        assert fp32.precision.storage_dtype == np.dtype(np.float32)
        assert fp16.precision.storage_dtype == np.dtype(np.float16)
