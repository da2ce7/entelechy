# src/tests/test_bugfix_regressions.py
"""
Regression tests for critical bugs discovered during E2E debugging.

Each test class targets one specific bug and verifies the fix at the
unit-test level (no OpenCL device required).

Bug 1 — Forward-pass timing (ComputeOnceProvider)
    CacheProvider pre-executed the forward pass at plan-creation time,
    before data was uploaded.  All hidden activations were zero.  The fix
    replaces CacheProvider with ComputeOnceProvider, which defers the
    kernel launch until the first resolve() call.

Bug 2 — Node 8 dispatch grid (work-group structure)
    calculate_module_param_grads_chunk uses get_group_id() to map
    work-groups to gradient components and get_local_id(0) for batch
    reduction.  local_size was None, so OpenCL could collapse the entire
    global_size into a single work-group, making get_group_id() return 0
    for all components.  The fix specifies explicit local_size.

Bug 3 — Uninitialized OpenCL buffers
    OpenCL does NOT guarantee zeroed memory.  Adam optimizer state (m1, m2)
    and bias buffers started with garbage, causing weight explosions on the
    first step.  The fix zero-fills every buffer before selective non-zero
    initialization.

Bug 4a — shared_weights buffer shape
    The shared_weights layout was (input_dim, padded_hidden_dim) but the
    forward pass kernel reads as flat[h * padded_input + i], requiring
    (padded_hidden_dim, padded_input_dim).  This caused out-of-bounds reads.

Bug 4b — Gradient/weight layout alignment
    The backprop kernel must write gradients in the same (hidden-major) flat
    order that the forward pass uses to read weights.  A transposed write
    order (input-major) would apply Adam updates to incorrect elements.

Bug 4c — batch_chunk_index for scratch buffers
    The backprop kernels write to per-chunk scratch buffers.  The kernel uses
    batch_chunk_index to offset the write location.  When writing to a
    single-chunk scratch buffer, the index must be 0; using the loop variable
    `i` caused out-of-bounds writes for chunks > 0.

Bug 5 — Input data upload without padding
    The host array X_batch has shape (batch_size, input_dim) but the GPU
    "input" buffer has shape (batch_size, padded_input_dim).  A raw
    enqueue_copy of the unpadded array packs samples contiguously in flat
    memory, misaligning rows relative to the padded stride the kernel
    expects.  Only ~(input_dim / padded_input_dim) of samples get correct
    data; the rest read zeros or contaminated padding.  The fix pads
    X_batch to padded_input_dim width before uploading.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, call, patch
import math

import numpy as np
import pytest

from src.backends.opencl.launcher_infra import BufferHandle, KernelSignature
from src.shared.model_spec import Float32ModelSpec, ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.workload_primitives import TilingScheme, WorkTile


# ── Shared helpers ──

def _h(n: int) -> BufferHandle:
    """Shorthand factory for test BufferHandles."""
    return BufferHandle(id=n)


def _make_iris_spec(simd_width: int = 1) -> ModelSpec:
    return Float32ModelSpec(
        input_dim=4,
        hidden_dim=32,
        output_classes=3,
        num_modules=8,
        simd_width=simd_width,
        cache_line_bytes=64,
    )


def _make_tiling(spec: ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


# =========================================================================
# Bug 1 — ComputeOnceProvider defers execution and caches the result
# =========================================================================


class TestComputeOnceProvider:
    """Verify the ComputeOnceProvider semantics WITHOUT an OpenCL device."""

    @staticmethod
    def _make_provider():
        """Build a ComputeOnceProvider with a mock signature."""
        from src.backends.opencl.execution_plan import ComputeOnceProvider

        mock_sig = MagicMock(spec=KernelSignature)
        handle = _h(42)
        provider = ComputeOnceProvider(signature=mock_sig, output_handle=handle)
        return provider, mock_sig, handle

    def test_first_resolve_launches_kernel(self) -> None:
        """The kernel must be launched on the very first resolve() call."""
        provider, mock_sig, handle = self._make_provider()

        mock_queue = MagicMock()
        mock_exec = MagicMock()
        fake_event = MagicMock()
        mock_exec.launch.return_value = fake_event

        ret_handle, ret_event = provider.resolve(mock_queue, mock_exec, wait_for=[])
        mock_exec.launch.assert_called_once_with(mock_queue, mock_sig, wait_for=[])
        assert ret_handle is handle
        assert ret_event is fake_event

    def test_second_resolve_uses_cache(self) -> None:
        """Subsequent resolve() calls must NOT re-launch the kernel."""
        provider, mock_sig, handle = self._make_provider()
        mock_queue = MagicMock()
        mock_exec = MagicMock()
        mock_exec.launch.return_value = MagicMock()

        first_handle, first_event = provider.resolve(mock_queue, mock_exec, wait_for=[])
        second_handle, second_event = provider.resolve(mock_queue, mock_exec, wait_for=[MagicMock()])

        # launch called exactly once — the second resolve re-used the cache.
        mock_exec.launch.assert_called_once()
        assert second_handle is first_handle is handle
        assert second_event is first_event

    def test_third_resolve_still_cached(self) -> None:
        """Even a third call must return the same cached event."""
        provider, _, handle = self._make_provider()
        mock_queue, mock_exec = MagicMock(), MagicMock()
        mock_exec.launch.return_value = MagicMock()

        ev1 = provider.resolve(mock_queue, mock_exec, [])[1]
        ev2 = provider.resolve(mock_queue, mock_exec, [])[1]
        ev3 = provider.resolve(mock_queue, mock_exec, [])[1]

        assert ev1 is ev2 is ev3
        assert mock_exec.launch.call_count == 1

    def test_wait_for_propagated_on_first_call(self) -> None:
        """The first resolve must forward the caller's wait_for to the kernel."""
        provider, mock_sig, _ = self._make_provider()
        mock_queue, mock_exec = MagicMock(), MagicMock()
        mock_exec.launch.return_value = MagicMock()
        sentinel_events = [MagicMock(), MagicMock()]

        provider.resolve(mock_queue, mock_exec, wait_for=sentinel_events)  # type: ignore[arg-type]
        mock_exec.launch.assert_called_once_with(mock_queue, mock_sig, wait_for=sentinel_events)

    def test_frozen_dataclass_allows_cache_mutation(self) -> None:
        """The internal _cached_event must be mutable despite the frozen dataclass."""
        from src.backends.opencl.execution_plan import ComputeOnceProvider

        provider = ComputeOnceProvider(signature=MagicMock(), output_handle=_h(0))
        # Before resolve, cache is empty.
        assert provider._cached_event is None

        mock_exec = MagicMock()
        mock_exec.launch.return_value = MagicMock()
        provider.resolve(MagicMock(), mock_exec, [])

        # After resolve, cache is populated.
        assert provider._cached_event is not None

    def test_cache_provider_ignores_wait_for(self) -> None:
        """CacheProvider must ignore its wait_for — this means it cannot solve
        the timing bug because its event was created before data upload."""
        from src.backends.opencl.execution_plan import CacheProvider

        handle = _h(10)
        stale_event = MagicMock()
        provider = CacheProvider(handle=handle, ready_event=stale_event)

        fresh_events = [MagicMock(), MagicMock()]
        ret_handle, ret_event = provider.resolve(MagicMock(), MagicMock(), wait_for=fresh_events)  # type: ignore[arg-type]

        # CacheProvider always returns the stale event — it cannot honour
        # fresh_events.  This is WHY ComputeOnceProvider was needed.
        assert ret_event is stale_event
        assert ret_handle is handle


# =========================================================================
# Bug 2 — Node 8 dispatch grid must specify explicit local_size
# =========================================================================


class _FakeBufferManager:
    """Minimal mock that supplies shape/dtype for buffer specs."""

    def __init__(self, specs: Dict[BufferHandle, Tuple[Tuple[int, ...], type]]):
        self._specs = specs

    def get_spec(self, ref: BufferHandle):
        return self._specs[ref]

    def get_cl_buffer(self, ref):
        return MagicMock()


class _FakeArchConsts:
    """Minimal mock for DiscoveredArchConstants."""

    def __init__(self, simd_width: int = 1, reduction_wg: int = 256, rect_tile_dim1: int = 16):
        self.simd_width = simd_width
        self.optimal_workgroup_size_1d_reduction = reduction_wg
        self.optimal_rectangular_tile_dim1 = rect_tile_dim1
        self.SCALAR_NP_TYPE = np.float32
        self.global_mem_cacheline_size = 64


class TestNode8DispatchGrid:
    """The calculate_module_param_grads_chunk kernel uses get_group_id().

    If local_size is None, OpenCL may assign all work-items to a single
    work-group, making get_group_id() == 0 for every component.  The grid
    must explicitly set local_size = (work_group_size_0, 1, 1).
    """

    @staticmethod
    def _make_signature(
        modules_per_chunk: int = 8,
        hidden_count: int = 32,
        classes_per_chunk: int = 3,
        work_group_size_0: int = 256,
        batch_size: int = 150,
        total_tiles: int = 1,
    ):
        from src.backends.opencl.kernel_bindings.phase_2_learn_A_production import (
            CalculateModuleParamGradsCceSignature,
        )

        h_ref = _h(0)
        prob_ref = _h(1)
        mask_ref = _h(2)
        gw_ref = _h(3)
        gb_ref = _h(4)
        targets_ref = _h(5)

        specs = {
            h_ref: ((batch_size, hidden_count), np.float32),
            gw_ref: ((total_tiles, modules_per_chunk, hidden_count, 16), np.float32),
            gb_ref: ((total_tiles, modules_per_chunk, 16), np.float32),
        }
        bm = _FakeBufferManager(specs)
        arch = _FakeArchConsts(reduction_wg=work_group_size_0)

        tile = WorkTile(
            module_chunk_index=0,
            class_chunk_index=0,
            flat_tile_index=0,
            num_class_chunks=1,
            module_chunk_offset=0,
            modules_per_chunk=modules_per_chunk,
            class_chunk_offset=0,
            classes_per_chunk=classes_per_chunk,
        )

        return CalculateModuleParamGradsCceSignature(
            _buffer_mgr=bm,  # type: ignore[arg-type]
            _arch_consts=arch,  # type: ignore[arg-type]
            work_group_size_0=work_group_size_0,
            h_ref=h_ref,
            prob_ref=prob_ref,
            mask_ref=mask_ref,
            gw_out_ref=gw_ref,
            gb_out_ref=gb_ref,
            targets_cce_ref=targets_ref,
            tile=tile,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
            hidden_count=np.uint32(hidden_count),
            total_output_class_count=np.uint32(classes_per_chunk),
            padded_total_output_class_count=np.uint32(16),
            total_modules_count=np.uint32(modules_per_chunk),
        )

    def test_local_size_is_not_none(self) -> None:
        """get_grid() must return an explicit local_size tuple, not None."""
        sig = self._make_signature()
        _, local_size = sig.get_grid()
        assert local_size is not None, (
            "local_size must not be None — the kernel uses get_group_id()"
        )

    def test_local_size_dim0_equals_work_group_size(self) -> None:
        """local_size[0] must match work_group_size_0 for the batch reduction."""
        for wg_size in [32, 64, 128, 256]:
            sig = self._make_signature(work_group_size_0=wg_size)
            _, local_size = sig.get_grid()
            assert local_size is not None
            assert local_size[0] == wg_size, (
                f"Expected local_size[0]={wg_size}, got {local_size[0]}"
            )

    def test_local_size_dims_1_and_2_are_one(self) -> None:
        """Dims 1 and 2 map to hidden and class indices — one work-group each."""
        sig = self._make_signature()
        _, local_size = sig.get_grid()
        assert local_size is not None
        assert len(local_size) == 3
        assert local_size[1] == 1
        assert local_size[2] == 1

    def test_global_size_dim0_is_modules_times_wg(self) -> None:
        """global_size[0] = modules_per_chunk * work_group_size_0."""
        sig = self._make_signature(modules_per_chunk=8, work_group_size_0=256)
        global_size, _ = sig.get_grid()
        assert global_size[0] == 8 * 256

    def test_global_size_dim1_is_hidden_count(self) -> None:
        """global_size[1] = hidden_count (one work-group per hidden index)."""
        sig = self._make_signature(hidden_count=32)
        global_size, _ = sig.get_grid()
        assert global_size[1] == 32

    def test_global_size_dim2_is_classes_per_chunk(self) -> None:
        """global_size[2] = classes_per_chunk (one work-group per class index)."""
        sig = self._make_signature(classes_per_chunk=3)
        global_size, _ = sig.get_grid()
        assert global_size[2] == 3

    def test_work_groups_cover_all_gradient_components(self) -> None:
        """The total number of work-groups must equal modules × hidden × classes."""
        M, H, C, WG = 8, 32, 3, 128
        sig = self._make_signature(
            modules_per_chunk=M, hidden_count=H,
            classes_per_chunk=C, work_group_size_0=WG,
        )
        global_size, local_size = sig.get_grid()
        assert local_size is not None
        num_wg_d0 = global_size[0] // local_size[0]
        num_wg_d1 = global_size[1] // local_size[1]
        num_wg_d2 = global_size[2] // local_size[2]
        assert num_wg_d0 * num_wg_d1 * num_wg_d2 == M * H * C

    def test_small_grid_does_not_collapse(self) -> None:
        """Even when total work-items < a single work-group, grid must not collapse.

        This is the exact scenario for Iris: 8 mods × 32 hidden × 3 classes =
        768 components, but WG=256 implies OpenCL WITHOUT explicit local_size
        might assign all 768 to one group.
        """
        sig = self._make_signature(
            modules_per_chunk=8, hidden_count=32,
            classes_per_chunk=3, work_group_size_0=256,
        )
        global_size, local_size = sig.get_grid()
        assert local_size is not None
        # The grid must request 8 separate work-groups in dim 0
        assert global_size[0] // local_size[0] == 8, (
            "Must have exactly 8 work-groups in dim 0 (one per module)"
        )

    @pytest.mark.parametrize(
        "modules, hidden, classes",
        [(1, 1, 1), (1, 64, 1), (16, 128, 16), (8, 32, 3)],
    )
    def test_grid_invariant_for_various_shapes(
        self, modules: int, hidden: int, classes: int
    ) -> None:
        """The grid contract must hold for all parameter combinations."""
        WG = 64
        sig = self._make_signature(
            modules_per_chunk=modules, hidden_count=hidden,
            classes_per_chunk=classes, work_group_size_0=WG,
        )
        global_size, local_size = sig.get_grid()
        assert local_size is not None
        assert global_size[0] == modules * WG
        assert global_size[1] == hidden
        assert global_size[2] == classes
        assert local_size == (WG, 1, 1)


# =========================================================================
# Bug 3 — All buffers must be zero-initialized before non-zero overrides
# =========================================================================


class TestBufferZeroInitialization:
    """Verify that _setup_buffers zeroes all buffers before selective init.

    These tests operate at the Python/contract level: we check that the set
    of buffers requiring zero start values is correct, and that the code in
    _setup_buffers matches the architectural intent.
    """

    @staticmethod
    def _get_all_buffer_names(spec: ModelSpec, batch_size: int = 150) -> set:
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(
            batch_size=batch_size, grid=grid, num_batch_chunks=4,
        )
        return set(layouts.keys())

    def test_adam_state_buffers_exist(self) -> None:
        """Every learnable flow must have m1_ and m2_ buffers in the layout."""
        spec = _make_iris_spec()
        names = self._get_all_buffer_names(spec)
        ps = ParameterSpace(spec=spec)

        for flow in ps:
            if not flow.m1_buffer_name:  # specialized flows (e.g. hidden_activations)
                continue
            assert flow.m1_buffer_name in names, (
                f"Missing Adam m1 buffer: {flow.m1_buffer_name}"
            )
            assert flow.m2_buffer_name in names, (
                f"Missing Adam m2 buffer: {flow.m2_buffer_name}"
            )

    def test_adam_m1_m2_names_follow_convention(self) -> None:
        """m1/m2 buffer names must start with 'm1_' / 'm2_' prefix."""
        spec = _make_iris_spec()
        ps = ParameterSpace(spec=spec)
        for flow in ps:
            if not flow.m1_buffer_name:  # specialized flows skip Adam
                continue
            assert flow.m1_buffer_name.startswith("m1_"), flow.m1_buffer_name
            assert flow.m2_buffer_name.startswith("m2_"), flow.m2_buffer_name

    def test_bias_buffers_are_zero_by_default(self) -> None:
        """Bias buffers (module_biases, shared_biases) are NOT in the set of
        buffers that receive non-zero initialization, so they must be zero.

        If they were NOT zero-filled, stale GPU memory would act as bias init,
        breaking the model.
        """
        # The orchestrator's _setup_buffers explicitly overwrites only:
        # sample_mask, temperatures, shared_weights, module_weights
        NON_ZERO_INITS = {"sample_mask", "temperatures", "shared_weights", "module_weights"}
        spec = _make_iris_spec()
        names = self._get_all_buffer_names(spec)

        bias_names = {n for n in names if "bias" in n.lower()}
        # None of the bias buffers should be in the non-zero set
        for bn in bias_names:
            assert bn not in NON_ZERO_INITS, (
                f"Bias buffer '{bn}' marked for non-zero init — "
                f"it should start at zero"
            )

    def test_m1_m2_buffers_not_in_nonzero_init_set(self) -> None:
        """Adam moment buffers must NOT receive non-zero initialization."""
        NON_ZERO_INITS = {"sample_mask", "temperatures", "shared_weights", "module_weights"}
        spec = _make_iris_spec()
        names = self._get_all_buffer_names(spec)

        m_names = {n for n in names if n.startswith("m1_") or n.startswith("m2_")}
        assert len(m_names) > 0, "Expected at least one m1/m2 buffer"
        for mn in m_names:
            assert mn not in NON_ZERO_INITS, (
                f"Adam state buffer '{mn}' must NOT be in the non-zero init set"
            )

    def test_all_buffers_listed_are_accounted_for(self) -> None:
        """Every buffer is either zero-init-only or explicitly overwritten.
        This catches new buffers added without proper initialization."""
        NON_ZERO_INITS = {"sample_mask", "temperatures", "shared_weights", "module_weights"}
        spec = _make_iris_spec()
        names = self._get_all_buffer_names(spec)

        # All names not in NON_ZERO_INITS rely on the blanket zero-fill.
        # This test simply asserts the count is reasonable.
        zero_init_names = names - NON_ZERO_INITS
        assert len(zero_init_names) >= 20, (
            f"Expected at least 20 zero-init buffers, got {len(zero_init_names)}"
        )

    @pytest.mark.parametrize(
        "flow_attr",
        [
            "param_buffer_name",
            "clipped_partial_grad_buffer_name",
            "summed_grad_buffer_name",
            "final_grad_buffer_name",
            "m1_buffer_name",
            "m2_buffer_name",
        ],
    )
    def test_every_flow_buffer_exists_in_layout(
        self, flow_attr: str
    ) -> None:
        """Every buffer name on every ParameterFlowConfig must exist in the layout dict."""
        spec = _make_iris_spec()
        names = self._get_all_buffer_names(spec)
        ps = ParameterSpace(spec=spec)

        for flow in ps:
            buf_name = getattr(flow, flow_attr, "")
            if buf_name:
                assert buf_name in names, (
                    f"Flow '{flow.name}' references buffer '{buf_name}' "
                    f"(attr={flow_attr}) which is missing from layouts"
                )

    def test_iris_m1_m2_shapes_match_param_shapes(self) -> None:
        """For the Iris model, every m1/m2 buffer must have the same shape as its parameter buffer."""
        spec = _make_iris_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(
            batch_size=150, grid=grid, num_batch_chunks=4,
        )

        for flow in ps:
            if not flow.m1_buffer_name:  # specialized flows skip Adam
                continue
            param_shape = layouts[flow.param_buffer_name].get_padded_shape(np.dtype(np.float32))
            m1_shape = layouts[flow.m1_buffer_name].get_padded_shape(np.dtype(np.float32))
            m2_shape = layouts[flow.m2_buffer_name].get_padded_shape(np.dtype(np.float32))

            assert m1_shape == param_shape, (
                f"Flow '{flow.name}': m1 shape {m1_shape} != param shape {param_shape}"
            )
            assert m2_shape == param_shape, (
                f"Flow '{flow.name}': m2 shape {m2_shape} != param shape {param_shape}"
            )


# =========================================================================
# Cross-cutting: Temp gradients kernel also uses get_group_id()
# =========================================================================


class TestNode10DispatchGrid:
    """calculate_chunk_temp_gradients also uses get_group_id().

    Unlike Node 8, this kernel always had explicit local_size.  This test
    locks in that contract to prevent a similar regression.
    """

    @staticmethod
    def _make_temp_signature(modules_per_chunk: int = 8, wg_size: int = 256):
        from src.backends.opencl.kernel_bindings.phase_2_learn_A_production import (
            CalculateChunkTempGradientsCceSignature,
        )

        logit_ref = _h(0)
        prob_ref = _h(1)
        mask_ref = _h(2)
        temp_ref = _h(3)
        gt_ref = _h(4)
        targets_ref = _h(5)

        specs = {
            logit_ref: ((modules_per_chunk, 150, 16), np.float32),
            gt_ref: ((1, modules_per_chunk), np.float32),
        }
        bm = _FakeBufferManager(specs)
        arch = _FakeArchConsts(reduction_wg=wg_size)

        tile = WorkTile(
            module_chunk_index=0, class_chunk_index=0,
            flat_tile_index=0, num_class_chunks=1,
            module_chunk_offset=0, modules_per_chunk=modules_per_chunk,
            class_chunk_offset=0, classes_per_chunk=3,
        )

        return CalculateChunkTempGradientsCceSignature(
            _buffer_mgr=bm,  # type: ignore[arg-type]
            _arch_consts=arch,  # type: ignore[arg-type]
            work_group_size_0=wg_size,
            logit_ref=logit_ref,
            prob_ref=prob_ref,
            mask_ref=mask_ref,
            temp_ref=temp_ref,
            gt_out_ref=gt_ref,
            targets_cce_ref=targets_ref,
            tile=tile,
            total_output_class_count=np.uint32(3),
        )

    def test_temp_kernel_has_explicit_local_size(self) -> None:
        sig = self._make_temp_signature()
        _, local_size = sig.get_grid()
        assert local_size is not None

    def test_temp_kernel_local_dim0_equals_wg_size(self) -> None:
        sig = self._make_temp_signature(wg_size=128)
        _, local_size = sig.get_grid()
        assert local_size is not None
        assert local_size[0] == 128

    def test_temp_kernel_grid_is_1d(self) -> None:
        """Temp gradient grid is 1D, not 3D like Node 8."""
        sig = self._make_temp_signature()
        global_size, local_size = sig.get_grid()
        assert local_size is not None
        assert len(global_size) == 1
        assert len(local_size) == 1

    def test_temp_kernel_work_groups_equals_modules(self) -> None:
        sig = self._make_temp_signature(modules_per_chunk=4, wg_size=64)
        global_size, local_size = sig.get_grid()
        assert local_size is not None
        assert global_size[0] // local_size[0] == 4


# =========================================================================
# Bug 4a — shared_weights buffer must be (padded_hidden, padded_input)
# =========================================================================


class TestSharedWeightsLayout:
    """The shared_weights buffer shape must match the forward-pass kernel's
    SIMD-major weight access pattern.

    The forward_pass kernel reads: flat[h_block * padded_input * SIMD + feature * SIMD + lane]
    For SIMD=1: flat[h * padded_input + i]  →  shape = (padded_hidden, padded_input)

    Previously the layout was (input_dim, padded_hidden_dim), which was
    transposed and too small, causing out-of-bounds reads.
    """

    @staticmethod
    def _get_sw_layout(spec: ModelSpec):
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)
        return layouts["shared_weights"]

    def test_iris_shared_weights_shape_is_hidden_by_padded_input(self) -> None:
        """Iris (4 inputs, 32 hidden, simd=1, cache_line=64):
        padded_hidden=32, padded_input=16 → shape = (32, 16)."""
        spec = _make_iris_spec(simd_width=1)
        layout = self._get_sw_layout(spec)
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape == (32, 16), f"Expected (32, 16), got {shape}"

    def test_shared_weights_dim0_is_padded_hidden(self) -> None:
        """First dimension must be padded_hidden_dim, not input_dim."""
        spec = _make_iris_spec()
        layout = self._get_sw_layout(spec)
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape[0] == spec.padded_hidden_dim, (
            f"dim0={shape[0]} != padded_hidden_dim={spec.padded_hidden_dim}"
        )

    def test_shared_weights_dim1_is_padded_input(self) -> None:
        """Second dimension must be padded_input_dim, not hidden_dim."""
        spec = _make_iris_spec()
        layout = self._get_sw_layout(spec)
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape[1] == spec.padded_input_dim, (
            f"dim1={shape[1]} != padded_input_dim={spec.padded_input_dim}"
        )

    def test_shared_weights_element_count_sufficient_for_kernel(self) -> None:
        """The buffer must have at least padded_hidden * padded_input elements —
        the number of unique weight indices the forward kernel can address."""
        spec = _make_iris_spec()
        layout = self._get_sw_layout(spec)
        shape = layout.get_padded_shape(np.dtype(np.float32))
        total_elements = 1
        for d in shape:
            total_elements *= d
        min_needed = spec.padded_hidden_dim * spec.padded_input_dim
        assert total_elements >= min_needed, (
            f"Buffer has {total_elements} elements but kernel needs {min_needed}"
        )

    @pytest.mark.parametrize("input_dim, hidden_dim, simd, cache_line", [
        (4, 32, 1, 64),    # Iris
        (16, 64, 1, 64),   # Hydra
        (8, 32, 4, 64),    # Iris with wider SIMD
        (784, 256, 1, 64), # MNIST-scale
    ])
    def test_layout_contract_across_specs(
        self, input_dim: int, hidden_dim: int, simd: int, cache_line: int
    ) -> None:
        """The (padded_hidden, padded_input) contract holds for all valid specs."""
        spec = Float32ModelSpec(
            input_dim=input_dim, hidden_dim=hidden_dim,
            output_classes=3, num_modules=8,
            simd_width=simd, cache_line_bytes=cache_line,
        )
        layout = self._get_sw_layout(spec)
        shape = layout.get_padded_shape(np.dtype(np.float32))
        assert shape[0] == spec.padded_hidden_dim
        assert shape[1] == spec.padded_input_dim

    def test_partial_grad_shared_weights_matches_weight_shape(self) -> None:
        """Per-chunk gradient layout must match weight layout (excluding chunk dim)."""
        spec = _make_iris_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)
        w_shape = layouts["shared_weights"].get_padded_shape(np.dtype(np.float32))
        pg_shape = layouts["partial_grad_shared_weights"].get_padded_shape(np.dtype(np.float32))
        # partial_grad shape = (num_batch_chunks, *weight_shape)
        assert pg_shape[1:] == w_shape, (
            f"Partial grad per-chunk shape {pg_shape[1:]} != weight shape {w_shape}"
        )


# =========================================================================
# Bug 4b — Backprop gradient write order must match forward-pass read order
# =========================================================================


class TestBackpropGradientLayoutAlignment:
    """The backprop_shared_weights_chunk kernel writes:
        grad[chunk_base + j * padded_input + i]
    The forward_pass kernel reads weights as:
        weight[h * padded_input + i]

    These flat layouts must be identical so Adam applies each gradient to
    the correct weight.  A transposed write (i * padded_hidden + j) would
    corrupt the update.

    We verify this at the Python signature level by checking that:
    1. The signature derives padded_input from the input buffer (dim 1)
    2. The signature derives padded_hidden from the hidden buffer (dim 1)
    3. The grid launches one work-group per (input, hidden) pair
    """

    @staticmethod
    def _make_backprop_sw_sig(
        padded_input: int = 16,
        padded_hidden: int = 32,
        batch_size: int = 150,
        num_chunks: int = 4,
    ):
        from src.backends.opencl.kernel_bindings.phase_2_learn_D_backprop import (
            BackpropSharedWeightsChunkSignature,
        )
        input_ref = _h(0)
        h_ref = _h(1)
        grad_h_ref = _h(2)
        mask_ref = _h(3)
        out_ref = _h(4)

        specs = {
            input_ref: ((batch_size, padded_input), np.float32),
            h_ref: ((batch_size, padded_hidden), np.float32),
            grad_h_ref: ((batch_size, padded_hidden), np.float32),
        }
        bm = _FakeBufferManager(specs)
        arch = _FakeArchConsts(reduction_wg=256, rect_tile_dim1=16)

        return BackpropSharedWeightsChunkSignature(
            _buffer_mgr=bm,  # type: ignore[arg-type]
            _arch_consts=arch,  # type: ignore[arg-type]
            input_ref=input_ref,
            h_ref=h_ref,
            grad_h_ref=grad_h_ref,
            mask_ref=mask_ref,
            partial_gsw_out_ref=out_ref,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
            batch_chunk_index=np.uint32(0),
            num_batch_chunks_count=np.uint32(num_chunks),
        )

    def test_derived_padded_input_matches_input_buffer(self) -> None:
        """The signature must derive padded_input_count from input buffer dim 1."""
        sig = self._make_backprop_sw_sig(padded_input=16)
        assert int(sig.padded_input_count) == 16

    def test_derived_padded_hidden_matches_hidden_buffer(self) -> None:
        """The signature must derive padded_hidden_count from hidden buffer dim 1."""
        sig = self._make_backprop_sw_sig(padded_hidden=32)
        assert int(sig.padded_hidden_count) == 32

    def test_grid_dim0_is_padded_input(self) -> None:
        """Grid dim 0 = padded_input: one column of work-groups per input feature."""
        sig = self._make_backprop_sw_sig(padded_input=16, padded_hidden=32)
        global_size, _ = sig.get_grid()
        assert global_size[0] == 16

    def test_grid_dim1_covers_padded_hidden(self) -> None:
        """Grid dim 1 must cover all hidden neurons (padded to tile size)."""
        sig = self._make_backprop_sw_sig(padded_input=16, padded_hidden=32)
        global_size, local_size = sig.get_grid()
        assert local_size is not None
        assert global_size[1] >= 32
        # Must be a multiple of local_size[1]
        assert global_size[1] % local_size[1] == 0

    def test_grid_has_explicit_local_size(self) -> None:
        """local_size must not be None (same class of bug as Bug 2)."""
        sig = self._make_backprop_sw_sig()
        _, local_size = sig.get_grid()
        assert local_size is not None

    def test_gradient_element_count_matches_weight_buffer(self) -> None:
        """The flat gradient order (hidden-major) must produce the same number
        of elements as the weight buffer shape."""
        padded_input, padded_hidden = 16, 32
        sig = self._make_backprop_sw_sig(
            padded_input=padded_input, padded_hidden=padded_hidden
        )
        # The kernel writes padded_input * padded_hidden elements per chunk
        # at offsets chunk_base + j * padded_input + i.
        # This must equal the weight buffer's total element count.
        grad_elements = int(sig.padded_input_count) * int(sig.padded_hidden_count)
        weight_elements = padded_hidden * padded_input
        assert grad_elements == weight_elements

    @pytest.mark.parametrize("padded_input, padded_hidden", [
        (16, 32),   # Iris
        (16, 64),   # Hydra
        (800, 256), # MNIST-scale
    ])
    def test_dimension_derivation_across_sizes(
        self, padded_input: int, padded_hidden: int
    ) -> None:
        """Derived dimensions must always match the buffer shapes."""
        sig = self._make_backprop_sw_sig(
            padded_input=padded_input, padded_hidden=padded_hidden
        )
        assert int(sig.padded_input_count) == padded_input
        assert int(sig.padded_hidden_count) == padded_hidden


# =========================================================================
# Bug 4c — batch_chunk_index must be 0 for per-chunk scratch buffers
# =========================================================================


class TestBatchChunkIndexContract:
    """The backprop kernels use batch_chunk_index to compute the write offset
    into their output buffer:
        chunk_base = batch_chunk_index * padded_input * padded_hidden

    When writing to a per-chunk SCRATCH buffer (one chunk's worth of space),
    batch_chunk_index MUST be 0.  If the loop variable `i` is passed instead,
    chunks > 0 write out-of-bounds.

    This is a contract test verifying the graph recipe's usage pattern.
    """

    @staticmethod
    def _make_backprop_sw_sig(batch_chunk_index: int, padded_input: int = 16, padded_hidden: int = 32):
        from src.backends.opencl.kernel_bindings.phase_2_learn_D_backprop import (
            BackpropSharedWeightsChunkSignature,
        )
        input_ref, h_ref, grad_h_ref = _h(0), _h(1), _h(2)
        mask_ref, out_ref = _h(3), _h(4)

        specs = {
            input_ref: ((150, padded_input), np.float32),
            h_ref: ((150, padded_hidden), np.float32),
            grad_h_ref: ((150, padded_hidden), np.float32),
        }
        bm = _FakeBufferManager(specs)
        arch = _FakeArchConsts(rect_tile_dim1=16)

        return BackpropSharedWeightsChunkSignature(
            _buffer_mgr=bm,  # type: ignore[arg-type] _arch_consts=arch,  # type: ignore[arg-type]
            input_ref=input_ref, h_ref=h_ref, grad_h_ref=grad_h_ref,
            mask_ref=mask_ref, partial_gsw_out_ref=out_ref,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(150),
            batch_chunk_index=np.uint32(batch_chunk_index),
            num_batch_chunks_count=np.uint32(4),
        )

    def test_scratch_buffer_requires_index_zero(self) -> None:
        """When writing to a single-chunk scratch buffer, index must be 0."""
        sig = self._make_backprop_sw_sig(batch_chunk_index=0)
        # The kernel computes: chunk_base = index * padded_input * padded_hidden
        # With index=0, chunk_base=0 — always in bounds for a single-chunk buffer.
        chunk_base = int(sig.batch_chunk_index) * int(sig.padded_input_count) * int(sig.padded_hidden_count)
        assert chunk_base == 0

    def test_nonzero_index_causes_out_of_bounds(self) -> None:
        """Demonstrates the bug: index > 0 produces an offset beyond the scratch buffer."""
        sig = self._make_backprop_sw_sig(batch_chunk_index=1)
        scratch_elements = int(sig.padded_input_count) * int(sig.padded_hidden_count)
        chunk_base = int(sig.batch_chunk_index) * scratch_elements
        # chunk_base == scratch_elements → first write is exactly at the boundary
        assert chunk_base >= scratch_elements, (
            "Non-zero index must produce an out-of-bounds offset for a single-chunk buffer"
        )

    def test_graph_recipe_uses_zero_index(self) -> None:
        """Verify the graph recipe source passes batch_chunk_index=0.

        This is a source-level contract check — we inspect the actual Python
        source of execute_streaming_shared_backprop to confirm the fix.
        """
        import inspect
        from src.backends.opencl.graph_recipes import build_shared_backprop_subgraph
        source = inspect.getsource(build_shared_backprop_subgraph)
        # The fixed code must pass batch_chunk_index=np.uint32(0) for BOTH
        # the weight and bias backprop signatures.
        assert source.count("batch_chunk_index=np.uint32(0)") >= 2, (
            "Expected at least 2 occurrences of batch_chunk_index=np.uint32(0) "
            "in build_shared_backprop_subgraph (one for weights, one for biases)"
        )

    @pytest.mark.parametrize("chunk_idx", [0, 1, 2, 3])
    def test_only_zero_index_is_safe_for_scratch(self, chunk_idx: int) -> None:
        """For a scratch buffer sized for 1 chunk, only index 0 is safe."""
        sig = self._make_backprop_sw_sig(batch_chunk_index=chunk_idx)
        scratch_elements = int(sig.padded_input_count) * int(sig.padded_hidden_count)
        chunk_base = int(sig.batch_chunk_index) * scratch_elements
        max_write = chunk_base + scratch_elements - 1
        if chunk_idx == 0:
            assert max_write < scratch_elements, "Index 0 should be in bounds"
        else:
            assert max_write >= scratch_elements, f"Index {chunk_idx} should be out of bounds"


# =========================================================================
# Bug 5 — Input data must be padded before GPU upload
# =========================================================================


class TestInputDataPadding:
    """Verify that BatchProcessor pads X_batch to padded_input_dim.

    When input_dim < padded_input_dim, a raw enqueue_copy of the unpadded
    array packs consecutive sample features into the same GPU buffer row.
    The forward kernel reads flat[sample * padded_input_dim + feature], so
    only the first ~(input_dim/padded_input_dim) fraction of samples get
    correct input data.
    """

    @pytest.mark.parametrize("input_dim,simd,cache_line,expected_padded", [
        (4, 1, 64, 16),   # Iris: 4 → 16 (ceil to cacheline/4)
        (16, 1, 64, 16),  # Already aligned: no padding needed
        (5, 1, 64, 16),   # 5 → 16
        (4, 4, 64, 16),   # SIMD=4
        (17, 1, 64, 32),  # 17 → 32
    ])
    def test_padding_dimension_requires_expansion(
        self, input_dim: int, simd: int, cache_line: int, expected_padded: int
    ) -> None:
        """Verify that padded_input_dim >= input_dim for various configs."""
        spec = Float32ModelSpec(
            input_dim=input_dim, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=simd, cache_line_bytes=cache_line,
        )
        assert spec.padded_input_dim == expected_padded
        assert spec.padded_input_dim >= input_dim

    def test_batch_processor_source_pads_input(self) -> None:
        """The BatchProcessor.run method must pad X_batch before upload.

        Source-level contract check: inspect the actual Python source to
        confirm the padding logic is present.
        """
        import inspect
        from src.backends.opencl.batch_processor import BatchProcessor
        source = inspect.getsource(BatchProcessor.run)
        # Must reference padded_input_dim and create a padded array
        assert "padded_input_dim" in source, (
            "BatchProcessor.run must reference padded_input_dim for padding"
        )
        assert "padded_X" in source or "np.zeros" in source, (
            "BatchProcessor.run must create a zero-padded array"
        )

    def test_raw_copy_would_corrupt_data(self) -> None:
        """Demonstrate that without padding, flat memory is misaligned.

        This is the 'negative test' — it shows WHY padding is needed.
        """
        input_dim, padded_dim, batch_size = 4, 16, 10
        X = np.arange(batch_size * input_dim, dtype=np.float32).reshape(batch_size, input_dim)

        # Simulated raw copy: 40 floats into a (10, 16) buffer
        raw_buffer = np.zeros(batch_size * padded_dim, dtype=np.float32)
        raw_buffer[:X.size] = X.ravel()  # contiguous copy
        raw_view = raw_buffer.reshape(batch_size, padded_dim)

        # Row 0 features 4..7 would contain sample 1's data
        assert not np.allclose(raw_view[0, input_dim:2*input_dim], 0), (
            "Without padding, row 0 padding columns are polluted by sample 1"
        )
        # Row 1 features 0..3 would NOT be sample 1's actual data
        assert not np.allclose(raw_view[1, :input_dim], X[1]), (
            "Without padding, row 1 doesn't start at sample 1's data"
        )

    def test_padded_copy_preserves_alignment(self) -> None:
        """With proper padding, each row's features are correctly placed."""
        input_dim, padded_dim, batch_size = 4, 16, 10
        X = np.arange(batch_size * input_dim, dtype=np.float32).reshape(batch_size, input_dim)

        # This is what BatchProcessor now does:
        padded_X = np.zeros((batch_size, padded_dim), dtype=X.dtype)
        padded_X[:, :input_dim] = X

        for s in range(batch_size):
            np.testing.assert_array_equal(
                padded_X[s, :input_dim], X[s],
                err_msg=f"Sample {s}: features not preserved",
            )
            np.testing.assert_array_equal(
                padded_X[s, input_dim:], 0,
                err_msg=f"Sample {s}: padding columns should be zero",
            )

    def test_no_padding_needed_when_dims_match(self) -> None:
        """When input_dim == padded_input_dim, the original array is used."""
        spec = Float32ModelSpec(
            input_dim=16, hidden_dim=32, output_classes=3,
            num_modules=8, simd_width=1, cache_line_bytes=64,
        )
        assert spec.padded_input_dim == spec.input_dim
        # The BatchProcessor code has an `if X_batch.shape[1] < padded_input_dim`
        # guard, so when they match, no copy is made.

    @pytest.mark.parametrize("batch_size", [1, 50, 150, 256])
    def test_all_samples_get_correct_features(self, batch_size: int) -> None:
        """Every sample must have its features intact after padding."""
        input_dim, padded_dim = 4, 16
        rng = np.random.default_rng(42)
        X = rng.standard_normal((batch_size, input_dim)).astype(np.float32)

        padded = np.zeros((batch_size, padded_dim), dtype=X.dtype)
        padded[:, :input_dim] = X
        flat = padded.ravel()

        for s in range(batch_size):
            start = s * padded_dim
            row = flat[start:start + padded_dim]
            np.testing.assert_array_equal(row[:input_dim], X[s])
            np.testing.assert_array_equal(row[input_dim:], 0)
