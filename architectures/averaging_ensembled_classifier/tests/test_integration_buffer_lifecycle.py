# tests/test_integration_buffer_lifecycle.py

"""
Integration Tests: BufferManager + PingPongManager + HostView Lifecycle.

These tests verify the device memory management layer operates correctly
in concert with higher-level primitives.  The BufferManager and
PingPongManager are exercised with real PyOpenCL buffers when available,
and with host-side contract verification when not.

Target CONCEPT.md Contracts:
  - Buffer creation from MemoryLayout plans
  - Opaque BufferHandle token system
  - Transient buffer acquire/release lifecycle
  - PingPong buffer management for reduction trees
  - HostView padding-aware data retrieval

No OpenCL device is required for the structural/contract tests.
The transfer tests are marked with `requires_opencl`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from src.memory_layout import MemoryLayout

# Provide Pylance with the real types for static analysis.
if TYPE_CHECKING:
    import pyopencl as cl

    from src.launcher_infra import BufferHandle, BufferManager, HostView, PingPongManager

# Conditional OpenCL imports — launcher_infra requires pyopencl
try:
    import pyopencl as cl  # noqa: F811

    from src.launcher_infra import BufferHandle, BufferManager, HostView, PingPongManager  # noqa: F811

    _has_cl = True
except ImportError:
    _has_cl = False

from tests.conftest import requires_opencl

# =========================================================================
# 1. BufferHandle Contracts (no device required)
# =========================================================================


class TestBufferHandleContract:
    """Verify the opaque handle token system."""

    def test_handle_equality(self):
        h1 = BufferHandle(id=0)
        h2 = BufferHandle(id=0)
        h3 = BufferHandle(id=1)
        assert h1 == h2
        assert h1 != h3

    def test_handle_is_hashable(self):
        h = BufferHandle(id=42)
        d = {h: "value"}
        assert d[BufferHandle(id=42)] == "value"

    def test_handle_is_frozen(self):
        h = BufferHandle(id=0)
        with pytest.raises(AttributeError):
            h.id = 1  # type: ignore[misc]


# =========================================================================
# 2. HostView Padding-Aware Retrieval (no device required)
# =========================================================================


class TestHostViewContract:
    """Verify that HostView correctly un-pads data."""

    def test_host_view_slicing_2d(self):
        padded_shape = (10, 32)
        real_shape = (10, 24)
        view = HostView(padded_shape=padded_shape, dtype=np.float32, real_shape=real_shape)
        # Simulate filling the padded buffer
        view.host_data[:] = np.arange(np.prod(padded_shape), dtype=np.float32).reshape(padded_shape)
        result = view.get()
        assert result.shape == real_shape

    def test_host_view_slicing_3d(self):
        padded_shape = (4, 8, 16)
        real_shape = (4, 5, 10)
        view = HostView(padded_shape=padded_shape, dtype=np.float32, real_shape=real_shape)
        view.host_data[:] = 1.0
        result = view.get()
        assert result.shape == real_shape

    def test_host_view_no_padding(self):
        shape = (5, 10)
        view = HostView(padded_shape=shape, dtype=np.float32, real_shape=shape)
        view.host_data[:] = 42.0
        result = view.get()
        assert result.shape == shape
        np.testing.assert_array_equal(result, 42.0)


# =========================================================================
# 3. BufferManager Lifecycle (requires OpenCL)
# =========================================================================


@requires_opencl
class TestBufferManagerLifecycle:
    """Verify named and transient buffer creation, lookup, and release."""

    @pytest.fixture
    def bm(self) -> BufferManager:
        ctx = cl.create_some_context(interactive=False)
        return BufferManager(ctx)

    def test_create_named_buffer(self, bm: BufferManager) -> None:
        layout = MemoryLayout((10, 32))
        handle = bm.create_named_buffer("test_buf", layout, np.float32)
        assert isinstance(handle, BufferHandle)

    def test_get_handle_by_name(self, bm: BufferManager) -> None:
        layout = MemoryLayout((10, 32))
        handle = bm.create_named_buffer("my_buf", layout, np.float32)
        retrieved = bm.get_handle_by_name("my_buf")
        assert retrieved == handle

    def test_duplicate_name_raises(self, bm: BufferManager) -> None:
        layout = MemoryLayout((10,))
        bm.create_named_buffer("dup", layout, np.float32)
        with pytest.raises(ValueError, match="already exists"):
            bm.create_named_buffer("dup", layout, np.float32)

    def test_missing_name_raises(self, bm: BufferManager) -> None:
        with pytest.raises(KeyError):
            bm.get_handle_by_name("nonexistent")

    def test_transient_buffer_lifecycle(self, bm: BufferManager) -> None:
        handle = bm.acquire_transient_buffer(1024)
        assert isinstance(handle, BufferHandle)
        cl_buf = bm.get_cl_buffer(handle)
        assert cl_buf is not None
        bm.release_transient_buffer(handle)
        with pytest.raises(KeyError):
            bm.get_cl_buffer(handle)

    def test_get_spec(self, bm: BufferManager) -> None:
        layout = MemoryLayout((4, 16))
        handle = bm.create_named_buffer("spec_test", layout, np.float32)
        shape, dtype = bm.get_spec(handle)
        assert shape == (4, 16)
        assert dtype == np.float32


# =========================================================================
# 4. PingPongManager Lifecycle (requires OpenCL)
# =========================================================================


@requires_opencl
class TestPingPongManagerLifecycle:
    """Verify the ping-pong buffer management pattern used in reduction trees."""

    @pytest.fixture
    def pp_ctx(self) -> tuple[BufferManager, PingPongManager]:
        ctx = cl.create_some_context(interactive=False)
        bm = BufferManager(ctx)
        pp = PingPongManager()
        return bm, pp

    def test_initialize_and_get_io(self, pp_ctx: tuple[BufferManager, PingPongManager]) -> None:
        bm, pp = pp_ctx
        pp.initialize(bm, max_bytes=4096)
        inp, out = pp.get_io()
        assert isinstance(inp, BufferHandle)
        assert isinstance(out, BufferHandle)
        assert inp != out

    def test_swap_changes_io_roles(self, pp_ctx: tuple[BufferManager, PingPongManager]) -> None:
        bm, pp = pp_ctx
        pp.initialize(bm, max_bytes=4096)
        inp1, out1 = pp.get_io()
        pp.swap()
        inp2, out2 = pp.get_io()
        assert inp2 == out1
        assert out2 == inp1

    def test_double_swap_restores_original(self, pp_ctx: tuple[BufferManager, PingPongManager]) -> None:
        bm, pp = pp_ctx
        pp.initialize(bm, max_bytes=4096)
        inp_orig, out_orig = pp.get_io()
        pp.swap()
        pp.swap()
        inp_restored, out_restored = pp.get_io()
        assert inp_restored == inp_orig
        assert out_restored == out_orig

    def test_use_before_init_raises(self, pp_ctx: tuple[BufferManager, PingPongManager]) -> None:
        _, pp = pp_ctx
        with pytest.raises(RuntimeError, match="initialized"):
            pp.get_io()

    def test_double_init_raises(self, pp_ctx: tuple[BufferManager, PingPongManager]) -> None:
        bm, pp = pp_ctx
        pp.initialize(bm, max_bytes=4096)
        with pytest.raises(RuntimeError, match="already initialized"):
            pp.initialize(bm, max_bytes=4096)

    def test_release_allows_reinit(self, pp_ctx: tuple[BufferManager, PingPongManager]) -> None:
        bm, pp = pp_ctx
        pp.initialize(bm, max_bytes=4096)
        pp.release()
        # After release, re-initialization should work
        pp.initialize(bm, max_bytes=2048)
        inp, out = pp.get_io()
        assert inp != out
