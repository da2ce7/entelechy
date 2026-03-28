# tests/tier1/test_buffer_lifecycle.py
"""BufferHandle, BufferDescriptor, and BufferRole tests."""
import pytest

from src.shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole


class TestBufferHandle:
    def test_is_int(self):
        h = BufferHandle(42)
        assert isinstance(h, int)
        assert h == 42

    def test_distinct_handles(self):
        assert BufferHandle(0) != BufferHandle(1)


class TestBufferRole:
    def test_all_roles_exist(self):
        assert BufferRole.MODEL_STATE is not None
        assert BufferRole.BATCH_INPUT is not None
        assert BufferRole.BATCH_INTERMEDIATE is not None
        assert BufferRole.BATCH_OUTPUT is not None


class TestBufferDescriptor:
    def test_frozen(self):
        h = BufferHandle(0)
        d = BufferDescriptor(
            handle=h, logical_name="test", padded_shape=(4, 16),
            element_size_bytes=4, size_bytes=256, role=BufferRole.BATCH_INTERMEDIATE,
            producing_node="n1", consumers=frozenset({"n2"}), last_consumer="n2",
        )
        assert d.size_bytes == 256
        with pytest.raises(AttributeError):
            d.size_bytes = 0  # type: ignore[misc]

    def test_model_state_no_producer(self):
        h = BufferHandle(0)
        d = BufferDescriptor(
            handle=h, logical_name="weights", padded_shape=(128, 64),
            element_size_bytes=4, size_bytes=128 * 64 * 4,
            role=BufferRole.MODEL_STATE,
            producing_node=None, consumers=frozenset(), last_consumer=None,
        )
        assert d.producing_node is None
