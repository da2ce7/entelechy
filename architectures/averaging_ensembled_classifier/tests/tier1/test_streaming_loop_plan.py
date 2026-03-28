# tests/tier1/test_streaming_loop_plan.py
"""StreamingLoopPlan stride arithmetic and constraints."""
import pytest

from src.shared.streaming_loop_plan import (
    IterationDimension,
    ParameterStride,
    ScratchBufferSpec,
    StreamingLoopPlan,
)


class TestIterationDimension:
    def test_basic(self):
        dim = IterationDimension(total_extent=100, chunk_count=10, chunk_size=10)
        assert dim.chunk_count * dim.chunk_size >= dim.total_extent

    def test_last_chunk_smaller(self):
        dim = IterationDimension(total_extent=7, chunk_count=3, chunk_size=3)
        assert dim.chunk_count * dim.chunk_size >= dim.total_extent


class TestParameterStride:
    def test_stride_arithmetic(self):
        stride = ParameterStride(param_name="offset", base=0, stride=16)
        for i in range(5):
            assert stride.base + i * stride.stride == i * 16


class TestScratchBufferSpec:
    def test_unique_names(self):
        s1 = ScratchBufferSpec(logical_name="a", size_bytes=100, shape=(10,))
        s2 = ScratchBufferSpec(logical_name="b", size_bytes=200, shape=(20,))
        assert s1.logical_name != s2.logical_name


class TestStreamingLoopPlan:
    def test_construction(self):
        plan = StreamingLoopPlan(
            iteration=IterationDimension(total_extent=64, chunk_count=64, chunk_size=1),
            body=("node_a", "node_b"),
            parameter_strides=(ParameterStride("batch_offset", 0, 1),),
            scratch_buffers=(),
            constant_scalars={"lr": 0.001},
        )
        assert len(plan.body) == 2
        assert plan.constant_scalars["lr"] == 0.001

    def test_frozen(self):
        plan = StreamingLoopPlan(
            iteration=IterationDimension(total_extent=1, chunk_count=1, chunk_size=1),
            body=("a",),
            parameter_strides=(),
            scratch_buffers=(),
            constant_scalars={},
        )
        with pytest.raises(AttributeError):
            plan.body = ("changed",)  # type: ignore[misc]
