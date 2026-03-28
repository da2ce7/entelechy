# tests/bench_opencl_backend.py
from __future__ import annotations

"""
Benchmarks: OpenCL Backend Kernel Dispatch & Device Interaction.

These benchmarks exercise the system's OpenCL backend hot-paths — the code
that compiles kernels, transfers data, dispatches compute, and executes the
reduction engine on a live device.

An OpenCL device IS required.  All benchmarks are skipped gracefully when no
GPU/accelerator is available.

Run with:
    pytest tests/bench_opencl_backend.py --benchmark-only
    pytest tests/bench_opencl_backend.py --benchmark-only --benchmark-sort=fullname
    pytest tests/bench_opencl_backend.py --benchmark-only --benchmark-group-by=group
"""

import os
import math
from typing import TYPE_CHECKING, Any, List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Provide Pylance with the real types for static analysis.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    import pyopencl as cl

    from src.backends.opencl.context import (
        OpenCLContextManager,
        Float32ComputeEnvironment,
        Float32DiscoveredArchConstants,
    )
    from src.arch_primitives import Float32Context
    from src.shared.model_spec import Float32ModelSpec
    from src.shared.parameter_space import ParameterSpace
    from src.shared.memory_layout import MemoryLayout
    from src.shared.workload_primitives import TilingScheme
    from src.backends.opencl.launcher_infra import (
        BufferManager,
        KernelExecutor,
        BufferHandle,
        HostView,
    )
    from src.backends.opencl.kernel_bindings import (
        ForwardPassSignature,
        RenderLogitsChunkSignature,
        NormalizeGradientsSignature,
        AdamUpdateSignature,
        AdamParameterGroup,
        ClampTemperaturesSignature,
        AggregateRegisterReduceSignature,
        AggregateLocalReduceSignature,
        ClipIntermediateGradSignature,
    )
    from src.backends.opencl.compute_patterns import AggregationManager

# ---------------------------------------------------------------------------
# Conditional OpenCL imports — the entire module is skipped when absent.
# ---------------------------------------------------------------------------
try:
    import pyopencl as cl  # noqa: F811

    _has_opencl = True
    try:
        _ctx = cl.create_some_context(interactive=False)
        _has_opencl_device = len(_ctx.devices) > 0
        del _ctx
    except Exception:
        _has_opencl_device = False
except ImportError:
    _has_opencl = False
    _has_opencl_device = False

pytestmark = pytest.mark.skipif(
    not (_has_opencl and _has_opencl_device),
    reason="No OpenCL device available",
)

if _has_opencl and _has_opencl_device:
    from src.backends.opencl.context import (  # noqa: F811
        OpenCLContextManager,
        Float32ComputeEnvironment,
        Float32DiscoveredArchConstants,
    )
    from src.arch_primitives import Float32Context  # noqa: F811
    from src.shared.model_spec import Float32ModelSpec  # noqa: F811
    from src.shared.parameter_space import ParameterSpace  # noqa: F811
    from src.shared.memory_layout import MemoryLayout  # noqa: F811
    from src.shared.workload_primitives import TilingScheme  # noqa: F811
    from src.backends.opencl.launcher_infra import (  # noqa: F811
        BufferManager,
        KernelExecutor,
        BufferHandle,
        HostView,
    )
    from src.backends.opencl.kernel_bindings import (  # noqa: F811
        ForwardPassSignature,
        RenderLogitsChunkSignature,
        NormalizeGradientsSignature,
        AdamUpdateSignature,
        AdamParameterGroup,
        ClampTemperaturesSignature,
        AggregateRegisterReduceSignature,
        AggregateLocalReduceSignature,
        ClipIntermediateGradSignature,
    )
    from src.backends.opencl.compute_patterns import AggregationManager  # noqa: F811


# =========================================================================
# Shared Fixtures
# =========================================================================

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "kernels",
)


@pytest.fixture(scope="session")
def compute_env() -> Float32ComputeEnvironment:
    """Build a full FP32 compute environment once per test session."""
    manager = OpenCLContextManager(kernel_source_dir=_KERNEL_DIR)
    return manager.build_and_discover(Float32Context)


@pytest.fixture(scope="session")
def queue(compute_env: Float32ComputeEnvironment) -> cl.CommandQueue:
    return compute_env.cl_bundle.queue


@pytest.fixture(scope="session")
def program(compute_env: Float32ComputeEnvironment) -> cl.Program:
    return compute_env.cl_bundle.program


@pytest.fixture(scope="session")
def arch_consts(compute_env: Float32ComputeEnvironment) -> Float32DiscoveredArchConstants:
    return compute_env.arch_consts


@pytest.fixture
def bm(compute_env: Float32ComputeEnvironment) -> BufferManager:
    """A fresh BufferManager per test (cheap — no device alloc until use)."""
    return BufferManager(compute_env.cl_bundle.context)


@pytest.fixture
def ex(program: cl.Program) -> KernelExecutor:
    return KernelExecutor(program)


# --- Canonical model specifications (same as bench_host_planning) ---

_IRIS_SPEC = None
_HYDRA_SPEC = None


def _get_iris_spec(arch_consts: Float32DiscoveredArchConstants) -> Float32ModelSpec:
    return Float32ModelSpec(
        input_dim=4, hidden_dim=32, output_classes=3,
        num_modules=8, simd_width=arch_consts.simd_width,
        cache_line_bytes=arch_consts.global_mem_cacheline_size,
    )


def _get_hydra_spec(arch_consts: Float32DiscoveredArchConstants) -> Float32ModelSpec:
    return Float32ModelSpec(
        input_dim=16, hidden_dim=64, output_classes=10,
        num_modules=256, simd_width=arch_consts.simd_width,
        cache_line_bytes=arch_consts.global_mem_cacheline_size,
    )


def _make_tiling(spec: Float32ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


def _allocate_model_buffers(
    bm: BufferManager,
    spec: Float32ModelSpec,
    batch_size: int,
) -> None:
    """
    Helper: allocate the minimal set of named buffers required by the
    benchmark kernels, matching the canonical layout plan from ParameterSpace.
    """
    ps = ParameterSpace(spec)
    grid = _make_tiling(spec)
    layouts = ps.get_all_memory_layouts(batch_size=batch_size, grid=grid, num_batch_chunks=4)
    for name, layout in layouts.items():
        bm.create_named_buffer(name, layout, np.float32)


# =========================================================================
# 1. Environment Bootstrap Benchmarks
# =========================================================================


class TestBenchEnvironmentBootstrap:
    """Benchmarks for the full OpenCL context + program build pipeline."""

    @pytest.mark.benchmark(group="env-bootstrap")
    def test_bench_build_and_discover_fp32(self, benchmark: Any) -> None:
        """
        Full FP32 bootstrap: context creation, kernel compilation, hardware
        discovery.  This is the system's cold-start cost.
        """
        manager = OpenCLContextManager(kernel_source_dir=_KERNEL_DIR)
        env = benchmark(manager.build_and_discover, Float32Context)
        assert isinstance(env, Float32ComputeEnvironment)
        assert env.arch_consts.simd_width >= 1


# =========================================================================
# 2. Data Transfer Benchmarks
# =========================================================================


class TestBenchDataTransfer:
    """Benchmarks for host <-> device memory transfers at realistic scales."""

    @pytest.mark.benchmark(group="transfer-h2d")
    @pytest.mark.parametrize(
        "shape",
        [(150, 16), (150, 64), (4096, 64), (4096, 128)],
        ids=["150x16", "150x64", "4096x64", "4096x128"],
    )
    def test_bench_host_to_device(self, benchmark: Any, compute_env: Float32ComputeEnvironment, shape: Tuple[int, ...]) -> None:
        """Host-to-device upload — the per-batch data staging cost."""
        ctx = compute_env.cl_bundle.context
        q = compute_env.cl_bundle.queue
        host_data = np.random.randn(*shape).astype(np.float32)
        buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=host_data.nbytes)

        def _upload():
            evt = cl.enqueue_copy(q, buf, host_data)
            evt.wait()

        benchmark(_upload)
        buf.release()

    @pytest.mark.benchmark(group="transfer-d2h")
    @pytest.mark.parametrize(
        "shape",
        [(150, 16), (150, 64), (4096, 64)],
        ids=["150x16", "150x64", "4096x64"],
    )
    def test_bench_device_to_host(self, benchmark: Any, compute_env: Float32ComputeEnvironment, shape: Tuple[int, ...]) -> None:
        """Device-to-host readback — the diagnostic/inference retrieval cost."""
        ctx = compute_env.cl_bundle.context
        q = compute_env.cl_bundle.queue
        host_data = np.zeros(shape, dtype=np.float32)
        buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=host_data.nbytes)
        # Seed the buffer with something
        cl.enqueue_copy(q, buf, np.ones(shape, dtype=np.float32)).wait()

        def _download():
            evt = cl.enqueue_copy(q, host_data, buf)
            evt.wait()

        benchmark(_download)
        buf.release()

    @pytest.mark.benchmark(group="transfer-roundtrip")
    @pytest.mark.parametrize(
        "num_elements",
        [1_000, 100_000, 1_000_000],
        ids=["1K", "100K", "1M"],
    )
    def test_bench_roundtrip_hostview(self, benchmark: Any, compute_env: Float32ComputeEnvironment, num_elements: int) -> None:
        """
        End-to-end HostView roundtrip: upload, then readback through the
        padding-aware HostView abstraction.
        """
        ctx = compute_env.cl_bundle.context
        q = compute_env.cl_bundle.queue
        # Use a 2D shape with a realistic trailing dimension
        cols = 64
        rows = max(1, num_elements // cols)
        padded_cols = cols  # no padding needed when already aligned
        shape = (rows, padded_cols)
        real_shape = (rows, cols)
        host_src = np.random.randn(*shape).astype(np.float32)
        buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=host_src.nbytes)
        cl.enqueue_copy(q, buf, host_src).wait()

        view = HostView(padded_shape=shape, dtype=np.float32, real_shape=real_shape)

        def _roundtrip():
            evt = view.enqueue_read(q, buf)
            evt.wait()
            return view.get()

        result = benchmark(_roundtrip)
        assert result.shape == real_shape


# =========================================================================
# 3. Buffer Management Benchmarks
# =========================================================================


class TestBenchBufferManagement:
    """Benchmarks for device buffer allocation via BufferManager."""

    @pytest.mark.benchmark(group="buffer-alloc")
    @pytest.mark.parametrize(
        "num_buffers",
        [10, 50, 100],
        ids=["N=10", "N=50", "N=100"],
    )
    def test_bench_create_named_buffers(self, benchmark: Any, compute_env: Float32ComputeEnvironment, num_buffers: int) -> None:
        """Named buffer allocation throughput — the per-run setup cost."""
        ctx = compute_env.cl_bundle.context

        def _create_all():
            bm = BufferManager(ctx)
            for i in range(num_buffers):
                layout = MemoryLayout((64, 128))
                bm.create_named_buffer(f"buf_{i}", layout, np.float32)

        benchmark(_create_all)

    @pytest.mark.benchmark(group="buffer-alloc")
    def test_bench_full_model_buffer_setup_iris(self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants) -> None:
        """Complete buffer allocation plan for the Iris model (small)."""
        ctx = compute_env.cl_bundle.context
        spec = _get_iris_spec(arch_consts)

        def _setup():
            bm = BufferManager(ctx)
            _allocate_model_buffers(bm, spec, batch_size=150)

        benchmark(_setup)

    @pytest.mark.benchmark(group="buffer-alloc")
    def test_bench_full_model_buffer_setup_hydra(self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants) -> None:
        """Complete buffer allocation plan for the Hydra model (large)."""
        ctx = compute_env.cl_bundle.context
        spec = _get_hydra_spec(arch_consts)

        def _setup():
            bm = BufferManager(ctx)
            _allocate_model_buffers(bm, spec, batch_size=32)

        benchmark(_setup)

    @pytest.mark.benchmark(group="buffer-alloc")
    @pytest.mark.parametrize(
        "size_bytes",
        [4096, 65536, 1_048_576, 16_777_216],
        ids=["4KB", "64KB", "1MB", "16MB"],
    )
    def test_bench_transient_buffer_lifecycle(self, benchmark: Any, compute_env: Float32ComputeEnvironment, size_bytes: int) -> None:
        """Transient buffer acquire + release cycle (reduction scratch space)."""
        ctx = compute_env.cl_bundle.context

        def _cycle():
            bm = BufferManager(ctx)
            h = bm.acquire_transient_buffer(size_bytes)
            bm.release_transient_buffer(h)

        benchmark(_cycle)


# =========================================================================
# 4. Kernel Dispatch Benchmarks — Forward Pass
# =========================================================================


class TestBenchForwardPassKernel:
    """Benchmarks for the shared-layer forward_pass kernel dispatch."""

    @pytest.mark.benchmark(group="kernel-forward")
    @pytest.mark.parametrize(
        "batch_size",
        [32, 150, 512],
        ids=["BS=32", "BS=150", "BS=512"],
    )
    def test_bench_forward_pass_iris(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Forward pass kernel for the Iris-scale model at varying batch sizes."""
        spec = _get_iris_spec(arch_consts)
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        _allocate_model_buffers(bm, spec, batch_size)

        # Upload random input data
        in_buf = bm.get_cl_buffer("input")
        host_input = np.random.randn(batch_size, spec.padded_input_dim).astype(np.float32)
        cl.enqueue_copy(q, in_buf, host_input).wait()

        sig = ForwardPassSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            in_ref=bm.get_handle_by_name("input"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_ref=bm.get_handle_by_name("shared_weights"),
            b_ref=bm.get_handle_by_name("shared_biases"),
            h_out_ref=bm.get_handle_by_name("hidden_activations"),
            h_mask_out_ref=bm.get_handle_by_name("hidden_mask"),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 5. Kernel Dispatch Benchmarks — Reduction Engine
# =========================================================================


class TestBenchReductionKernels:
    """
    Benchmarks for the tiered aggregation kernels that power the reduction
    engine — the hottest path in the backward pass.
    """

    def _setup_reduction_buffers(
        self, bm: BufferManager, num_partials: int, width: int,
    ) -> Tuple[BufferHandle, BufferHandle, BufferHandle, np.ndarray]:
        """Helper: allocate collection, offset list, and destination buffers."""
        element_bytes = np.dtype(np.float32).itemsize

        # Collection buffer: N partials × width elements each
        collection_h = bm.acquire_transient_buffer(num_partials * width * element_bytes)
        # Offset list: N uint32 offsets
        offsets = (np.arange(num_partials, dtype=np.uint32) * np.uint32(width))
        offset_h = bm.acquire_transient_buffer(offsets.nbytes)
        # Destination: width elements
        dest_h = bm.acquire_transient_buffer(width * element_bytes)

        return collection_h, offset_h, dest_h, offsets

    @pytest.mark.benchmark(group="kernel-reduction-register")
    @pytest.mark.parametrize("num_partials", [4, 8, 16], ids=lambda n: f"N={n}")
    @pytest.mark.parametrize("width", [128, 1024, 4096], ids=lambda w: f"W={w}")
    def test_bench_aggregate_register_reduce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_partials: int, width: int,
    ) -> None:
        """Register-tier reduction: small N, varying output width."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        collection_h, offset_h, dest_h, offsets = self._setup_reduction_buffers(bm, num_partials, width)

        # Upload the offset list
        cl.enqueue_copy(q, bm.get_cl_buffer(offset_h), offsets).wait()

        sig = AggregateRegisterReduceSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            partial_collection_ref=collection_h,
            partial_offset_list_ref=offset_h,
            dest_ref=dest_h,
            partial_offset_list_count=np.uint32(num_partials),
            partial_width=np.uint32(width),
            operation_type=np.uint32(0),  # AGG_MODE_SUM
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    @pytest.mark.benchmark(group="kernel-reduction-local")
    @pytest.mark.parametrize("num_partials", [32, 128, 256], ids=lambda n: f"N={n}")
    @pytest.mark.parametrize("width", [128, 1024], ids=lambda w: f"W={w}")
    def test_bench_aggregate_local_reduce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_partials: int, width: int,
    ) -> None:
        """Local-memory-tier reduction: larger N, representative widths."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        collection_h, offset_h, dest_h, offsets = self._setup_reduction_buffers(bm, num_partials, width)
        cl.enqueue_copy(q, bm.get_cl_buffer(offset_h), offsets).wait()

        sig = AggregateLocalReduceSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            partial_collection_ref=collection_h,
            partial_offset_list_ref=offset_h,
            dest_ref=dest_h,
            partial_offset_list_count=np.uint32(num_partials),
            partial_width=np.uint32(width),
            operation_type=np.uint32(0),  # AGG_MODE_SUM
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    @pytest.mark.benchmark(group="kernel-reduction-clip")
    @pytest.mark.parametrize(
        "num_elements",
        [256, 4096, 65536],
        ids=["256", "4K", "64K"],
    )
    def test_bench_clip_intermediate_grad(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_elements: int,
    ) -> None:
        """Intermediate gradient clipping — called once per reduction stage."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        # Create a named buffer so get_spec works for the signature
        layout = MemoryLayout((num_elements,))
        handle = bm.create_named_buffer("clip_target", layout, np.float32)

        # Seed with non-zero data
        data = np.random.randn(num_elements).astype(np.float32)
        cl.enqueue_copy(q, bm.get_cl_buffer(handle), data).wait()

        sig = ClipIntermediateGradSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            intermediate_grad_ref=handle,
            clipping_threshold_t_j=np.float32(1.0),
            epsilon=np.float32(1e-7),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 6. Kernel Dispatch Benchmarks — Update Phase
# =========================================================================


class TestBenchUpdateKernels:
    """
    Benchmarks for the final update-phase kernels: normalization,
    Adam optimizer step, and temperature clamping.
    """

    @pytest.mark.benchmark(group="kernel-normalize")
    @pytest.mark.parametrize(
        "num_elements",
        [128, 4096, 65536],
        ids=["128", "4K", "64K"],
    )
    def test_bench_normalize_gradients(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_elements: int,
    ) -> None:
        """Gradient normalization — element-wise divide by effective batch size."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        layout = MemoryLayout((num_elements,))
        summed_h = bm.create_named_buffer("summed_grad", layout, np.float32)
        final_h = bm.create_named_buffer("final_grad", layout, np.float32)

        sig = NormalizeGradientsSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            summed_grad_ref=summed_h,
            final_grad_out_ref=final_h,
            effective_batch_size=np.float32(150.0),
            epsilon=np.float32(1e-7),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    @pytest.mark.benchmark(group="kernel-adam")
    @pytest.mark.parametrize(
        "num_elements",
        [128, 4096, 65536],
        ids=["128", "4K", "64K"],
    )
    def test_bench_adam_update(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_elements: int,
    ) -> None:
        """Adam optimizer step — the most frequent per-parameter kernel."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        layout = MemoryLayout((num_elements,))
        param_h = bm.create_named_buffer("param", layout, np.float32)
        grad_h = bm.create_named_buffer("grad", layout, np.float32)
        m1_h = bm.create_named_buffer("m1", layout, np.float32)
        m2_h = bm.create_named_buffer("m2", layout, np.float32)

        sig = AdamUpdateSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            param_group=AdamParameterGroup(
                param_ref=param_h, grad_ref=grad_h,
                m1_state_ref=m1_h, m2_state_ref=m2_h,
            ),
            learning_rate=np.float32(0.001),
            beta1=np.float32(0.9),
            beta2=np.float32(0.999),
            epsilon=np.float32(1e-7),
            beta1_pow_t=np.float32(0.9 ** 10),
            beta2_pow_t=np.float32(0.999 ** 10),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    @pytest.mark.benchmark(group="kernel-clamp")
    @pytest.mark.parametrize(
        "num_modules",
        [8, 64, 256],
        ids=["M=8", "M=64", "M=256"],
    )
    def test_bench_clamp_temperatures(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_modules: int,
    ) -> None:
        """Temperature clamping — a lightweight element-wise constraint kernel."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        layout = MemoryLayout((num_modules,))
        temps_h = bm.create_named_buffer("temps", layout, np.float32)

        # Seed with values that span below and above the clamp range
        data = np.linspace(0.01, 20.0, num_modules).astype(np.float32)
        cl.enqueue_copy(q, bm.get_cl_buffer(temps_h), data).wait()

        sig = ClampTemperaturesSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            temps_ref=temps_h,
            min_val=np.float32(0.1),
            max_val=np.float32(10.0),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 7. AggregationManager Tier-Selection Benchmarks
# =========================================================================


class TestBenchAggregationManager:
    """
    Benchmarks for the AggregationManager's tier-selection dispatch, which
    automatically picks the optimal kernel (register vs. local memory) based
    on the number of partials.
    """

    def _setup(self, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_partials: int, width: int):
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)
        agg = AggregationManager(ex=ex, bm=bm, arch_consts=arch_consts)

        element_bytes = np.dtype(np.float32).itemsize
        collection_h = bm.acquire_transient_buffer(num_partials * width * element_bytes)
        offsets = np.arange(num_partials, dtype=np.uint32) * np.uint32(width)
        offset_h = bm.acquire_transient_buffer(offsets.nbytes)
        dest_h = bm.acquire_transient_buffer(width * element_bytes)
        cl.enqueue_copy(q, bm.get_cl_buffer(offset_h), offsets).wait()

        return q, agg, collection_h, offset_h, dest_h

    @pytest.mark.benchmark(group="aggregation-manager")
    @pytest.mark.parametrize("num_partials", [4, 16], ids=lambda n: f"N={n}")
    def test_bench_agg_manager_register_tier(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_partials: int,
    ) -> None:
        """AggregationManager with N <= 16 (register path)."""
        width = 1024
        q, agg, coll_h, off_h, dest_h = self._setup(compute_env, arch_consts, num_partials, width)

        def _dispatch():
            evt = agg.execute_stage(
                queue=q,
                collection_ref=coll_h,
                offset_list_ref=off_h,
                num_partials_to_reduce=num_partials,
                elements_per_partial=width,
                destination_ref=dest_h,
                wait_for=[],
            )
            evt.wait()

        benchmark(_dispatch)

    @pytest.mark.benchmark(group="aggregation-manager")
    @pytest.mark.parametrize("num_partials", [32, 128, 256], ids=lambda n: f"N={n}")
    def test_bench_agg_manager_local_tier(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants, num_partials: int,
    ) -> None:
        """AggregationManager with N > 16 (local memory path)."""
        width = 1024
        q, agg, coll_h, off_h, dest_h = self._setup(compute_env, arch_consts, num_partials, width)

        def _dispatch():
            evt = agg.execute_stage(
                queue=q,
                collection_ref=coll_h,
                offset_list_ref=off_h,
                num_partials_to_reduce=num_partials,
                elements_per_partial=width,
                destination_ref=dest_h,
                wait_for=[],
            )
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 8. Composed Pipeline Benchmarks
# =========================================================================


class TestBenchComposedPipelines:
    """
    End-to-end benchmarks that compose multiple OpenCL operations, mirroring
    realistic multi-kernel dispatch patterns.
    """

    @pytest.mark.benchmark(group="pipeline-forward")
    def test_bench_forward_plus_logits_iris(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants,
    ) -> None:
        """
        Forward pass + logit rendering for a single tile — the fused "Act"
        phase hot-path for the Iris model.
        """
        spec = _get_iris_spec(arch_consts)
        batch_size = 150
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        _allocate_model_buffers(bm, spec, batch_size)

        # Seed input data
        host_input = np.random.randn(batch_size, spec.padded_input_dim).astype(np.float32)
        cl.enqueue_copy(q, bm.get_cl_buffer("input"), host_input).wait()

        grid = _make_tiling(spec)
        tile = grid.get_tile(0, 0)

        fwd_sig = ForwardPassSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            in_ref=bm.get_handle_by_name("input"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_ref=bm.get_handle_by_name("shared_weights"),
            b_ref=bm.get_handle_by_name("shared_biases"),
            h_out_ref=bm.get_handle_by_name("hidden_activations"),
            h_mask_out_ref=bm.get_handle_by_name("hidden_mask"),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
        )

        logit_sig = RenderLogitsChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            h_ref=bm.get_handle_by_name("hidden_activations"),
            h_mask_ref=bm.get_handle_by_name("hidden_mask"),
            w_ref=bm.get_handle_by_name("module_weights"),
            b_ref=bm.get_handle_by_name("module_biases"),
            logit_out_ref=bm.get_handle_by_name("logits"),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
            module_chunk_offset=np.uint32(tile.module_chunk_offset),
            module_chunk_count=np.uint32(tile.modules_per_chunk),
            class_chunk_offset=np.uint32(tile.class_chunk_offset),
            class_chunk_count=np.uint32(tile.classes_per_chunk),
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
        )

        def _pipeline():
            fwd_evt = ex.launch(q, fwd_sig)
            logit_evt = ex.launch(q, logit_sig, wait_for=[fwd_evt])
            logit_evt.wait()

        benchmark(_pipeline)

    @pytest.mark.benchmark(group="pipeline-reduction")
    @pytest.mark.parametrize(
        "total_partials,k",
        [(8, 4), (64, 8), (256, 16)],
        ids=["8/k=4", "64/k=8", "256/k=16"],
    )
    def test_bench_multi_stage_reduction(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants,
        total_partials: int, k: int,
    ) -> None:
        """
        Multi-stage sum-then-clip reduction — the full Recursive
        Clip-Aggregation Engine hot-path with realistic fan-in.
        """
        width = 1024
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        element_bytes = np.dtype(np.float32).itemsize

        # Allocate the collection of partials
        collection_h = bm.acquire_transient_buffer(total_partials * width * element_bytes)
        # Two scratch buffers for ping-pong between stages
        scratch_a = bm.acquire_transient_buffer(total_partials * width * element_bytes)
        scratch_b = bm.acquire_transient_buffer(total_partials * width * element_bytes)

        # Compute number of stages and pre-allocate all clip buffers + offset
        # lists OUTSIDE the benchmark loop to avoid create_named_buffer collisions
        # on repeated iterations.
        num_stages = math.ceil(math.log(total_partials) / math.log(k))

        # Pre-compute the stage plan: for each stage, the number of groups
        # and per-group offset arrays.
        stage_plan: List[Tuple[int, List[Tuple[BufferHandle, int]], BufferHandle]] = []
        n = total_partials
        for stage_j in range(num_stages):
            groups_out = max(1, math.ceil(n / k))
            group_offsets: List[Tuple[BufferHandle, int]] = []
            for g in range(groups_out):
                start = g * k
                count = min(k, n - start)
                if count > 1:
                    offsets = np.arange(start, start + count, dtype=np.uint32) * np.uint32(width)
                    off_h = bm.acquire_transient_buffer(offsets.nbytes)
                    cl.enqueue_copy(q, bm.get_cl_buffer(off_h), offsets).wait()
                    group_offsets.append((off_h, count))
            # Named buffer for the clip step
            clip_layout = MemoryLayout((groups_out * width,))
            clip_h = bm.create_named_buffer(f"clip_stage_{stage_j}", clip_layout, np.float32)
            stage_plan.append((groups_out, group_offsets, clip_h))
            n = groups_out

        # Pre-build clip signatures (they are frozen dataclasses, safe to reuse)
        clip_sigs = [
            ClipIntermediateGradSignature(
                _buffer_mgr=bm, _arch_consts=arch_consts,
                intermediate_grad_ref=clip_h,
                clipping_threshold_t_j=np.float32(1.0),
                epsilon=np.float32(1e-7),
            )
            for _, _, clip_h in stage_plan
        ]

        def _multi_stage():
            src = collection_h
            dst = scratch_a
            other = scratch_b

            for stage_j, (_groups_out, group_offsets, _clip_h) in enumerate(stage_plan):
                for off_h, count in group_offsets:
                    sig = AggregateRegisterReduceSignature(
                        _buffer_mgr=bm, _arch_consts=arch_consts,
                        partial_collection_ref=src,
                        partial_offset_list_ref=off_h,
                        dest_ref=dst,
                        partial_offset_list_count=np.uint32(count),
                        partial_width=np.uint32(width),
                        operation_type=np.uint32(0),
                    )
                    ex.launch(q, sig)

                ex.launch(q, clip_sigs[stage_j])

                # Swap for next stage
                src, dst = dst, other
                other = src

            q.finish()

        benchmark(_multi_stage)

    @pytest.mark.benchmark(group="pipeline-update")
    def test_bench_full_update_step_iris(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment, arch_consts: Float32DiscoveredArchConstants,
    ) -> None:
        """
        Complete update phase for the Iris model: normalize all gradient
        flows, run Adam on each parameter group, and clamp temperatures.
        """
        spec = _get_iris_spec(arch_consts)
        batch_size = 150
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        _allocate_model_buffers(bm, spec, batch_size)

        # The parameter groups to update (matching ParameterSpace flows)
        param_names = ["shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"]

        # Build signatures for normalization and Adam update of each group
        norm_sigs: List[NormalizeGradientsSignature] = []
        adam_sigs: List[AdamUpdateSignature] = []
        for name in param_names:
            grad_name = "temps" if name == "temperatures" else name
            summed_name = f"summed_grad_{grad_name}"
            final_name = f"final_grad_{grad_name}"
            m1_name = f"m1_{name}"
            m2_name = f"m2_{name}"

            norm_sigs.append(NormalizeGradientsSignature(
                _buffer_mgr=bm, _arch_consts=arch_consts,
                summed_grad_ref=bm.get_handle_by_name(summed_name),
                final_grad_out_ref=bm.get_handle_by_name(final_name),
                effective_batch_size=np.float32(float(batch_size)),
                epsilon=np.float32(1e-7),
            ))

            adam_sigs.append(AdamUpdateSignature(
                _buffer_mgr=bm, _arch_consts=arch_consts,
                param_group=AdamParameterGroup(
                    param_ref=bm.get_handle_by_name(name),
                    grad_ref=bm.get_handle_by_name(final_name),
                    m1_state_ref=bm.get_handle_by_name(m1_name),
                    m2_state_ref=bm.get_handle_by_name(m2_name),
                ),
                learning_rate=np.float32(0.001),
                beta1=np.float32(0.9),
                beta2=np.float32(0.999),
                epsilon=np.float32(1e-7),
                beta1_pow_t=np.float32(0.9 ** 10),
                beta2_pow_t=np.float32(0.999 ** 10),
            ))

        clamp_sig = ClampTemperaturesSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            temps_ref=bm.get_handle_by_name("temperatures"),
            min_val=np.float32(0.1),
            max_val=np.float32(10.0),
        )

        def _update_step():
            events: List[Any] = []
            # Phase 1: Normalize all gradient flows in parallel
            for sig in norm_sigs:
                events.append(ex.launch(q, sig))

            # Phase 2: Adam update for each parameter group (depends on normalize)
            adam_events: List[Any] = []
            for sig in adam_sigs:
                adam_events.append(ex.launch(q, sig, wait_for=events))

            # Phase 3: Clamp temperatures (depends on Adam update of temperatures)
            clamp_evt = ex.launch(q, clamp_sig, wait_for=adam_events)
            clamp_evt.wait()

        benchmark(_update_step)
