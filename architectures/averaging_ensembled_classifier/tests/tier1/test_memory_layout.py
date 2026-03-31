# tests/tier1/test_memory_layout.py
"""SIMD-aware layout and padding calculations via ModelSpec."""
import numpy as np

from src.shared.model_spec import ModelSpec


class TestPaddingCalculations:
    def test_padded_hidden_simd_aligned(self):
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=30, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        assert spec.padded_hidden_dim % 16 == 0
        assert spec.padded_hidden_dim >= 30

    def test_padded_input_cache_aligned(self):
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        assert (spec.padded_input_dim * 4) % 64 == 0

    def test_fp16_wider_padding(self):
        fp32 = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        fp16 = ModelSpec.float16(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        assert fp16.padded_input_dim >= fp32.padded_input_dim


class TestModelSpecComposition:
    def test_precision_field(self):
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        assert spec.precision.numpy_dtype == np.dtype(np.float32)

    def test_backward_compat_scalar_np_type(self):
        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        assert spec.SCALAR_NP_TYPE == np.float32

    def test_backward_compat_c_type_name(self):
        fp32 = ModelSpec.float32(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        fp16 = ModelSpec.float16(
            input_dim=4, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=16, cache_line_bytes=64,
        )
        assert fp32.SCALAR_C_TYPE_NAME == "float"
        assert fp16.SCALAR_C_TYPE_NAME == "half"

    def test_factory_classmethod(self):
        spec = ModelSpec.float16(
            input_dim=8, hidden_dim=16, output_classes=5,
            num_modules=2, simd_width=4, cache_line_bytes=64,
        )
        assert spec.precision.numpy_dtype == np.dtype(np.float16)
