# src/tests/test_memory_layout.py
"""
Unit tests for MemoryLayout and PaddingStrategy: padding calculations,
shape transformations, and edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.memory_layout import MemoryLayout, PaddingStrategy, PaddingType, _pad_to_multiple


class TestPadToMultiple:

    @pytest.mark.parametrize("dim,mult,expected", [
        (10, 4, 12),
        (16, 4, 16),
        (0, 4, 0),
        (1, 1, 1),
        (7, 8, 8),
        (33, 16, 48),
    ])
    def test_correct(self, dim, mult, expected) -> None:
        assert _pad_to_multiple(dim, mult) == expected

    def test_zero_multiple_returns_original(self) -> None:
        assert _pad_to_multiple(10, 0) == 10

    def test_none_multiple_returns_original(self) -> None:
        assert _pad_to_multiple(10, None) == 10  # type: ignore[arg-type]


class TestMemoryLayout:

    def test_no_strategies_returns_logical(self) -> None:
        ml = MemoryLayout((150, 16))
        shape = ml.get_padded_shape(np.dtype(np.float32))
        assert shape == (150, 16)

    def test_element_count_padding(self) -> None:
        ml = MemoryLayout((150, 30))
        ml.add_strategy(PaddingStrategy(type=PaddingType.ELEMENT_COUNT, value=4, target_dim_idx=-1))
        shape = ml.get_padded_shape(np.dtype(np.float32))
        assert shape[-1] == 32  # 30 → 32 (next multiple of 4)
        assert shape[0] == 150

    def test_byte_alignment_padding(self) -> None:
        ml = MemoryLayout((150, 3))
        # 3 * 4bytes = 12 bytes, padded to 64 → 64/4 = 16
        ml.add_strategy(PaddingStrategy(type=PaddingType.BYTE_ALIGNMENT, value=64, target_dim_idx=-1))
        shape = ml.get_padded_shape(np.dtype(np.float32))
        assert shape[-1] == 16

    def test_fluent_api(self) -> None:
        ml = MemoryLayout((10, 10))
        result = ml.add_strategy(PaddingStrategy(type=PaddingType.NONE, value=0))
        assert result is ml  # fluent returns self

    def test_negative_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryLayout((-1, 10))
