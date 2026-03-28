# tests/tier1/test_precision_config.py
"""PrecisionConfig correctness."""
import numpy as np
import pytest

from src.shared.precision_config import PrecisionConfig


class TestFactoryMethods:
    def test_float32(self):
        pc = PrecisionConfig.float32()
        assert pc.numpy_dtype == np.dtype(np.float32)
        assert pc.fp_format_max == float(np.finfo(np.float32).max)
        assert pc.epsilon == float(np.finfo(np.float32).eps)

    def test_float16(self):
        pc = PrecisionConfig.float16()
        assert pc.numpy_dtype == np.dtype(np.float16)
        assert pc.fp_format_max == float(np.finfo(np.float16).max)
        assert pc.epsilon == float(np.finfo(np.float16).eps)

    def test_frozen(self):
        pc = PrecisionConfig.float32()
        with pytest.raises(AttributeError):
            pc.epsilon = 0.0  # type: ignore[misc]
