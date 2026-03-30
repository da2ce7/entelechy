# tests/bench_opencl_backend_learn.py
from __future__ import annotations

"""
Benchmarks: OpenCL Backend — Learn Phase, Loss Kernels & Composed Pipelines.

These benchmarks cover the DAG nodes NOT exercised by `bench_opencl_backend.py`,
completing the system's device-side benchmark coverage against CONCEPT.md.

Coverage map (CONCEPT.md Node → Benchmark Section):
  Act Phase:
    (5) render_logits_chunk           → §1  TestBenchRenderLogitsKernel
    (6) compute_probs_loss_cce_chunk  → §2  TestBenchLossKernels
    (7) compute_probs_loss_bce_chunk  → §2  TestBenchLossKernels
   (14) Diagnostic reduction (Probs)  → §3  TestBenchDiagnosticReduction

  Learn Phase I — Gradient Production:
    (8)  calculate_module_param_grads → §4  TestBenchGradientProductionKernels
    (9)  backprop_error_to_hidden     → §4  TestBenchGradientProductionKernels
   (10)  calculate_chunk_temp_grads   → §4  TestBenchGradientProductionKernels

  Learn Phase I — Gradient Processing:
   (11) clip_partial_gradients        → §5  TestBenchClipPartialGradients
   (13) gather_and_permute_grad_h     → §6  TestBenchGatherAndPermute

  Learn Phase II — Specialized Reduction:
   (16) stabilize_and_reduce_grad_h   → §7  TestBenchStabilizeReduceGradH

  Learn Phase III — Shared Layer Backprop:
   (17) backprop_shared_weights_chunk → §8  TestBenchSharedLayerBackprop
   (18) backprop_shared_biases_chunk  → §8  TestBenchSharedLayerBackprop
   (19) clip_shared_gradients_chunk   → §8  TestBenchSharedLayerBackprop

  Composed Pipelines:
    Act pipeline (4→5→6)              → §9  TestBenchComposedLearnPipelines
    Gradient production tile pipeline → §9  TestBenchComposedLearnPipelines
    Streaming backprop loop (17→18→19)→ §9  TestBenchComposedLearnPipelines

An OpenCL device IS required.  All benchmarks are skipped gracefully when no
GPU/accelerator is available.

Run with:
    pytest tests/bench_opencl_backend_learn.py --benchmark-only
    pytest tests/bench_opencl_backend_learn.py --benchmark-only --benchmark-sort=fullname
    pytest tests/bench_opencl_backend_learn.py --benchmark-only --benchmark-group-by=group
"""

import os
from typing import TYPE_CHECKING, Any, List

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
    from src.shared.model_spec import Float32ModelSpec, ModelSpec
    from src.shared.parameter_space import ParameterSpace
    from src.shared.workload_primitives import TilingScheme
    from src.backends.opencl.launcher_infra import (
        BufferManager,
        KernelExecutor,
    )
    from src.backends.opencl.kernel_bindings import (
        ForwardPassSignature,
        RenderLogitsChunkSignature,
        ComputeProbsLossCceChunkSignature,
        ComputeProbsLossBceChunkSignature,
        CalculateModuleParamGradsCceSignature,
        BackpropErrorToHiddenChunkCceSignature,
        CalculateChunkTempGradientsCceSignature,
        ClipPartialGradientsGlobalNormSignature,
        GradientHandles,
        GatherAndPermuteGradHiddenActivationsSignature,
        StabilizeAndReduceGradHiddenActivationsSignature,
        BackpropSharedWeightsChunkSignature,
        BackpropSharedBiasesChunkSignature,
        SharedGradientHandles,
        ClipSharedGradientsChunkSignature,
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
    from src.shared.workload_primitives import TilingScheme  # noqa: F811
    from src.backends.opencl.launcher_infra import (  # noqa: F811
        BufferManager,
        KernelExecutor,
    )
    from src.backends.opencl.kernel_bindings import (  # noqa: F811
        ForwardPassSignature,
        RenderLogitsChunkSignature,
        ComputeProbsLossCceChunkSignature,
        ComputeProbsLossBceChunkSignature,
        CalculateModuleParamGradsCceSignature,
        BackpropErrorToHiddenChunkCceSignature,
        CalculateChunkTempGradientsCceSignature,
        ClipPartialGradientsGlobalNormSignature,
        GradientHandles,
        GatherAndPermuteGradHiddenActivationsSignature,
        StabilizeAndReduceGradHiddenActivationsSignature,
        BackpropSharedWeightsChunkSignature,
        BackpropSharedBiasesChunkSignature,
        SharedGradientHandles,
        ClipSharedGradientsChunkSignature,
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


# --- Canonical model specifications ---

def _get_iris_spec(arch_consts: Float32DiscoveredArchConstants) -> ModelSpec:
    return Float32ModelSpec(
        input_dim=4, hidden_dim=32, output_classes=3,
        num_modules=8, simd_width=arch_consts.simd_width,
        cache_line_bytes=arch_consts.global_mem_cacheline_size,
    )


def _make_tiling(spec: ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


def _allocate_model_buffers(
    bm: BufferManager,
    spec: ModelSpec,
    batch_size: int,
    num_batch_chunks: int = 4,
) -> TilingScheme:
    """
    Allocate the full set of named buffers required by the benchmark kernels.
    Returns the TilingScheme used.
    """
    ps = ParameterSpace(spec)
    grid = _make_tiling(spec)
    layouts = ps.get_all_memory_layouts(
        batch_size=batch_size, grid=grid, num_batch_chunks=num_batch_chunks,
    )
    for name, layout in layouts.items():
        bm.create_named_buffer(name, layout, np.float32)
    return grid


def _seed_buffer(q: cl.CommandQueue, bm: BufferManager, name: str, rng: np.random.Generator | None = None) -> None:
    """Upload random FP32 data into a named buffer."""
    shape, _dtype = bm.get_spec(bm.get_handle_by_name(name))
    if rng is None:
        rng = np.random.default_rng(42)
    data = rng.standard_normal(shape).astype(np.float32)
    cl.enqueue_copy(q, bm.get_cl_buffer(name), data).wait()


def _seed_buffer_uniform(q: cl.CommandQueue, bm: BufferManager, name: str, low: float, high: float) -> None:
    """Upload uniform-random FP32 data into a named buffer."""
    shape, _ = bm.get_spec(bm.get_handle_by_name(name))
    data = np.random.default_rng(42).uniform(low, high, shape).astype(np.float32)
    cl.enqueue_copy(q, bm.get_cl_buffer(name), data).wait()


def _seed_targets_cce(q: cl.CommandQueue, bm: BufferManager, batch_size: int, num_classes: int) -> None:
    """Upload valid CCE integer targets."""
    data = np.random.default_rng(42).integers(0, num_classes, size=(batch_size,)).astype(np.float32)
    cl.enqueue_copy(q, bm.get_cl_buffer("targets_cce"), data).wait()


def _seed_mask_all_active(q: cl.CommandQueue, bm: BufferManager, batch_size: int) -> None:
    """Upload a mask that marks all samples as active."""
    data = np.ones(batch_size, dtype=np.float32)
    cl.enqueue_copy(q, bm.get_cl_buffer("sample_mask"), data).wait()


# =========================================================================
# 1. Kernel Dispatch — render_logits_chunk (Node 5, standalone)
# =========================================================================


class TestBenchRenderLogitsKernel:
    """Standalone benchmarks for the render_logits_chunk kernel (Node 5)."""

    @pytest.mark.benchmark(group="kernel-render-logits")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_render_logits_chunk(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Logit rendering for a single tile — the inner-loop Act-phase kernel."""
        spec = _get_iris_spec(arch_consts)
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size)
        _seed_buffer(q, bm, "hidden_activations")

        tile = grid.get_tile(0, 0)
        sig = RenderLogitsChunkSignature(
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

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 2. Kernel Dispatch — Loss Kernels (Nodes 6 & 7)
# =========================================================================


class TestBenchLossKernels:
    """Benchmarks for the CCE and BCE loss+probability kernels."""

    def _setup_loss_prereqs(
        self, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants,
        spec: ModelSpec, batch_size: int,
    ):
        """Common setup: allocate buffers and seed inputs for loss kernels."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size)
        _seed_buffer(q, bm, "logits")
        _seed_buffer_uniform(q, bm, "temperatures", 0.5, 2.0)
        _seed_targets_cce(q, bm, batch_size, spec.output_classes)
        _seed_mask_all_active(q, bm, batch_size)

        return q, bm, ex, grid

    @pytest.mark.benchmark(group="kernel-loss-cce")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_compute_probs_loss_cce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 6 — fused softmax + CCE loss for a single tile."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, grid = self._setup_loss_prereqs(
            compute_env, arch_consts, spec, batch_size,
        )
        tile = grid.get_tile(0, 0)

        sig = ComputeProbsLossCceChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            logit_ref=bm.get_handle_by_name("logits"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            target_ref=bm.get_handle_by_name("targets_cce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            prob_out_ref=bm.get_handle_by_name("partial_probs"),
            loss_out_ref=bm.get_handle_by_name("final_loss"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    @pytest.mark.benchmark(group="kernel-loss-bce")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_compute_probs_loss_bce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 7 — probability + partial BCE loss for a single tile."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, grid = self._setup_loss_prereqs(
            compute_env, arch_consts, spec, batch_size,
        )
        tile = grid.get_tile(0, 0)

        # Seed BCE (float) targets instead of CCE (int) targets
        bce_data = np.random.default_rng(42).uniform(
            0.0, 1.0, bm.get_spec(bm.get_handle_by_name("targets_bce"))[0],
        ).astype(np.float32)
        cl.enqueue_copy(q, bm.get_cl_buffer("targets_bce"), bce_data).wait()

        sig = ComputeProbsLossBceChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            logit_ref=bm.get_handle_by_name("logits"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            target_ref=bm.get_handle_by_name("targets_bce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            prob_out_ref=bm.get_handle_by_name("partial_probs"),
            partial_loss_out_ref=bm.get_handle_by_name("partial_loss"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 3. Diagnostic Reduction — Probability Aggregation (Node 14 path)
# =========================================================================


class TestBenchDiagnosticReduction:
    """
    Benchmarks for the diagnostic (non-gradient) reduction path: aggregating
    partial probabilities.  This exercises the Recursive Aggregation Engine
    with AGG_MODE_SUM but WITHOUT the interleaved clip step.
    """

    @pytest.mark.benchmark(group="pipeline-diagnostic-reduction")
    @pytest.mark.parametrize(
        "num_tiles,width",
        [(4, 512), (8, 1024)],
        ids=["4tiles/W=512", "8tiles/W=1024"],
    )
    def test_bench_diagnostic_prob_reduction(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants,
        num_tiles: int, width: int,
    ) -> None:
        """Pure sum reduction of partial probabilities — no clipping stage."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        element_bytes = np.dtype(np.float32).itemsize

        collection_h = bm.acquire_transient_buffer(num_tiles * width * element_bytes)
        offsets = (np.arange(num_tiles, dtype=np.uint32) * np.uint32(width))
        offset_h = bm.acquire_transient_buffer(offsets.nbytes)
        dest_h = bm.acquire_transient_buffer(width * element_bytes)

        cl.enqueue_copy(q, bm.get_cl_buffer(offset_h), offsets).wait()

        # Seed with plausible probability-scale data
        prob_data = np.random.default_rng(42).uniform(
            0.0, 1.0, (num_tiles * width,),
        ).astype(np.float32)
        cl.enqueue_copy(q, bm.get_cl_buffer(collection_h), prob_data).wait()

        agg = AggregationManager(ex=ex, bm=bm, arch_consts=arch_consts)

        def _dispatch():
            evt = agg.execute_stage(
                queue=q,
                collection_ref=collection_h,
                offset_list_ref=offset_h,
                num_partials_to_reduce=num_tiles,
                elements_per_partial=width,
                destination_ref=dest_h,
                wait_for=[],
            )
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 4. Kernel Dispatch — Gradient Production (Nodes 8, 9, 10)
# =========================================================================


class TestBenchGradientProductionKernels:
    """
    Benchmarks for the massively-parallel gradient production kernels
    that constitute Phase I of the Learn process.
    """

    def _setup_gradient_prereqs(
        self, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants,
        spec: ModelSpec, batch_size: int,
    ):
        """Common setup for gradient production benchmarks."""
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size)

        # Seed all input buffers needed by Nodes 8, 9, 10
        rng = np.random.default_rng(42)
        _seed_buffer(q, bm, "hidden_activations", rng)
        _seed_buffer(q, bm, "partial_probs", rng)
        _seed_buffer(q, bm, "logits", rng)
        _seed_buffer_uniform(q, bm, "temperatures", 0.5, 2.0)
        _seed_targets_cce(q, bm, batch_size, spec.output_classes)
        _seed_mask_all_active(q, bm, batch_size)

        tile = grid.get_tile(0, 0)
        return q, bm, ex, grid, tile

    # --- Node 8: calculate_module_param_grads ---

    @pytest.mark.benchmark(group="kernel-grad-module-params")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_calculate_module_param_grads_cce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 8 (CCE) — partial Grad_ModW & Grad_ModB for a single tile."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, _grid, tile = self._setup_gradient_prereqs(
            compute_env, arch_consts, spec, batch_size,
        )

        sig = CalculateModuleParamGradsCceSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
            h_ref=bm.get_handle_by_name("hidden_activations"),
            prob_ref=bm.get_handle_by_name("partial_probs"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            gw_out_ref=bm.get_handle_by_name("partial_grad_module_weights"),
            gb_out_ref=bm.get_handle_by_name("partial_grad_module_biases"),
            tile=tile,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
            padded_total_output_class_count=np.uint32(spec.padded_class_dim),
            total_modules_count=np.uint32(spec.num_modules),
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    # --- Node 9: backprop_error_to_hidden ---

    @pytest.mark.benchmark(group="kernel-grad-hidden")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_backprop_error_to_hidden_cce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 9 (CCE) — partial Grad_H for a single tile."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, _grid, tile = self._setup_gradient_prereqs(
            compute_env, arch_consts, spec, batch_size,
        )

        sig = BackpropErrorToHiddenChunkCceSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            prob_ref=bm.get_handle_by_name("partial_probs"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_mod_ref=bm.get_handle_by_name("module_weights"),
            gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
            tile=tile,
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    # --- Node 10: calculate_chunk_temp_gradients ---

    @pytest.mark.benchmark(group="kernel-grad-temps")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_calculate_chunk_temp_grads_cce(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 10 (CCE) — partial Grad_Temps for a single tile."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, _grid, tile = self._setup_gradient_prereqs(
            compute_env, arch_consts, spec, batch_size,
        )

        sig = CalculateChunkTempGradientsCceSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
            logit_ref=bm.get_handle_by_name("logits"),
            prob_ref=bm.get_handle_by_name("partial_probs"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            gt_out_ref=bm.get_handle_by_name("partial_grad_temps"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 5. Kernel Dispatch — clip_partial_gradients (Node 11)
# =========================================================================


class TestBenchClipPartialGradients:
    """
    Benchmarks for the foundational stability primitive — the leaf-level,
    group-wise clip applied to the complete gradient vector of a single tile.
    """

    @pytest.mark.benchmark(group="kernel-clip-partial")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_clip_partial_gradients_global_norm(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """
        Node 11 — leaf-level group-wise clip with a global threshold.
        Exercises the full Grad_ModW + Grad_ModB + Grad_Temps + Grad_H vector.
        """
        spec = _get_iris_spec(arch_consts)
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size)
        tile = grid.get_tile(0, 0)

        # Seed partial gradient buffers with data that will trigger clipping
        rng = np.random.default_rng(42)
        for name in [
            "partial_grad_module_weights", "partial_grad_module_biases",
            "partial_grad_temps", "partial_grad_hidden_activations",
        ]:
            _seed_buffer(q, bm, name, rng)

        handles = GradientHandles(
            grad_weights_module=bm.get_handle_by_name("partial_grad_module_weights"),
            grad_biases_module=bm.get_handle_by_name("partial_grad_module_biases"),
            grad_temps=bm.get_handle_by_name("partial_grad_temps"),
            grad_hidden_activations_aos=bm.get_handle_by_name("partial_grad_hidden_activations"),
            clipped_grad_weights_module=bm.get_handle_by_name("clipped_partial_grad_module_weights"),
            clipped_grad_biases_module=bm.get_handle_by_name("clipped_partial_grad_module_biases"),
            clipped_grad_temps=bm.get_handle_by_name("clipped_partial_grad_temps"),
            clipped_grad_hidden_activations_aos=bm.get_handle_by_name("clipped_partial_grad_hidden_activations"),
        )

        sig = ClipPartialGradientsGlobalNormSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            handles=handles,
            tile=tile,
            epsilon=np.float32(1e-7),
            clipping_threshold_global=np.float32(1.0),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 6. Kernel Dispatch — gather_and_permute_grad_h (Node 13)
# =========================================================================


class TestBenchGatherAndPermute:
    """
    Benchmarks for the canonical Item Synchronization Point — the AoS→SoA
    permutation of clipped Grad_H partials.
    """

    @pytest.mark.benchmark(group="kernel-gather-permute")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_gather_and_permute_grad_h(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 13 — gather scattered clipped Grad_H AoS → contiguous SoA."""
        spec = _get_iris_spec(arch_consts)
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        _grid = _allocate_model_buffers(bm, spec, batch_size)

        # Seed the clipped partials buffer
        _seed_buffer(q, bm, "clipped_partial_grad_hidden_activations")

        max_mods_per_tile = (
            (spec.num_modules + _grid.num_module_chunks - 1) // _grid.num_module_chunks
        )

        sig = GatherAndPermuteGradHiddenActivationsSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            clipped_partials_aos_ref=bm.get_handle_by_name("clipped_partial_grad_hidden_activations"),
            permuted_soa_out_ref=bm.get_handle_by_name("permuted_grad_h"),
            total_modules_count=np.uint32(spec.num_modules),
            hidden_count=np.uint32(spec.hidden_dim),
            total_batch_count=np.uint32(batch_size),
            num_module_chunks=np.uint32(_grid.num_module_chunks),
            modules_per_chunk=np.uint32(max_mods_per_tile),
            num_class_chunks=np.uint32(_grid.num_class_chunks),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 7. Kernel Dispatch — stabilize_and_reduce_grad_h (Node 16)
# =========================================================================


class TestBenchStabilizeReduceGradH:
    """
    Benchmarks for the specialized, self-contained Grad_H reduction engine
    that applies the Quadratic Scaling Policy internally.
    """

    @pytest.mark.benchmark(group="kernel-stabilize-reduce-h")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_stabilize_and_reduce_grad_h(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """
        Node 16 — the specialized, policy-aware reduction of Grad_H from
        the permuted SoA layout to a final (batch, padded_hidden) result.
        """
        spec = _get_iris_spec(arch_consts)
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        _allocate_model_buffers(bm, spec, batch_size)

        # Seed the permuted SoA buffer
        _seed_buffer(q, bm, "permuted_grad_h")

        sig = StabilizeAndReduceGradHiddenActivationsSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            permuted_soa_in_ref=bm.get_handle_by_name("permuted_grad_h"),
            final_grad_h_out_ref=bm.get_handle_by_name("summed_grad_hidden_activations"),
            fp_max=np.float32(3.4e38),
            policy_t_algorithmic=np.float32(1.0),
            policy_lambda=np.float32(0.1),
            policy_max_k=np.uint32(16),
            epsilon=np.float32(1e-7),
            total_batch_count=np.uint32(batch_size),
            padded_hidden_count=np.uint32(spec.padded_hidden_dim),
            total_modules_count=np.uint32(spec.num_modules),
            padded_total_modules_count=np.uint32(spec.padded_module_dim),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 8. Kernel Dispatch — Shared Layer Backprop (Nodes 17, 18, 19)
# =========================================================================


class TestBenchSharedLayerBackprop:
    """
    Benchmarks for the streaming shared-layer backpropagation kernels
    that constitute Phase III of the Learn process.
    """

    def _setup_shared_backprop(
        self, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants,
        spec: ModelSpec, batch_size: int,
    ):
        """Common setup for shared backprop benchmarks."""
        num_batch_chunks = 4
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size, num_batch_chunks)

        rng = np.random.default_rng(42)
        _seed_buffer(q, bm, "input", rng)
        _seed_buffer(q, bm, "hidden_activations", rng)
        _seed_buffer(q, bm, "summed_grad_hidden_activations", rng)
        _seed_mask_all_active(q, bm, batch_size)

        chunk_size = (batch_size + num_batch_chunks - 1) // num_batch_chunks
        return q, bm, ex, grid, num_batch_chunks, chunk_size

    # --- Node 17: backprop_shared_weights_chunk ---

    @pytest.mark.benchmark(group="kernel-backprop-shared-weights")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_backprop_shared_weights_chunk(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 17 — partial Grad_SW for a single batch chunk."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, _grid, num_batch_chunks, chunk_size = self._setup_shared_backprop(
            compute_env, arch_consts, spec, batch_size,
        )

        actual_count = min(chunk_size, batch_size)
        sig = BackpropSharedWeightsChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            input_ref=bm.get_handle_by_name("input"),
            h_ref=bm.get_handle_by_name("hidden_activations"),
            grad_h_ref=bm.get_handle_by_name("summed_grad_hidden_activations"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            partial_gsw_out_ref=bm.get_handle_by_name("partial_grad_shared_weights"),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(actual_count),
            batch_chunk_index=np.uint32(0),
            num_batch_chunks_count=np.uint32(num_batch_chunks),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    # --- Node 18: backprop_shared_biases_chunk ---

    @pytest.mark.benchmark(group="kernel-backprop-shared-biases")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_backprop_shared_biases_chunk(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 18 — partial Grad_SB for a single batch chunk."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, _grid, num_batch_chunks, chunk_size = self._setup_shared_backprop(
            compute_env, arch_consts, spec, batch_size,
        )

        actual_count = min(chunk_size, batch_size)
        sig = BackpropSharedBiasesChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            h_ref=bm.get_handle_by_name("hidden_activations"),
            grad_h_ref=bm.get_handle_by_name("summed_grad_hidden_activations"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            partial_gsb_out_ref=bm.get_handle_by_name("partial_grad_shared_biases"),
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(actual_count),
            batch_chunk_index=np.uint32(0),
            num_batch_chunks_count=np.uint32(num_batch_chunks),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)

    # --- Node 19: clip_shared_gradients_chunk ---

    @pytest.mark.benchmark(group="kernel-clip-shared")
    @pytest.mark.parametrize(
        "batch_size", [32, 150], ids=["BS=32", "BS=150"],
    )
    def test_bench_clip_shared_gradients_chunk(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, batch_size: int,
    ) -> None:
        """Node 19 — clip the concatenated Grad_SW+Grad_SB for a single chunk."""
        spec = _get_iris_spec(arch_consts)
        q, bm, ex, _grid, num_batch_chunks, _chunk_size = self._setup_shared_backprop(
            compute_env, arch_consts, spec, batch_size,
        )

        # Seed partial gradient scratch buffers
        _seed_buffer(q, bm, "partial_grad_shared_weights")
        _seed_buffer(q, bm, "partial_grad_shared_biases")

        handles = SharedGradientHandles(
            grad_weights_shared_chunk=bm.get_handle_by_name("partial_grad_shared_weights"),
            grad_biases_shared_chunk=bm.get_handle_by_name("partial_grad_shared_biases"),
            clipped_grad_weights_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_weights"),
            clipped_grad_biases_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_biases"),
        )

        sig = ClipSharedGradientsChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            handles=handles,
            clipping_threshold_global=np.float32(1.0),
            epsilon=np.float32(1e-7),
            dest_weights_write_offset_elements=np.uint32(0),
            dest_biases_write_offset_elements=np.uint32(0),
            num_batch_chunks=np.uint32(num_batch_chunks),
        )

        def _dispatch():
            evt = ex.launch(q, sig)
            evt.wait()

        benchmark(_dispatch)


# =========================================================================
# 9. Composed Pipeline Benchmarks
# =========================================================================


class TestBenchComposedLearnPipelines:
    """
    End-to-end benchmarks that compose multiple kernel dispatches, mirroring
    realistic multi-kernel patterns from the DAG.
    """

    # --- Full Act pipeline: forward_pass → render_logits → CCE loss ---

    @pytest.mark.benchmark(group="pipeline-full-act")
    def test_bench_full_act_phase_iris(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants,
    ) -> None:
        """
        Composed Act phase: forward_pass (4) → render_logits (5) → CCE loss (6).
        The complete inference hot-path before the host sync point.
        """
        spec = _get_iris_spec(arch_consts)
        batch_size = 150
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size)

        rng = np.random.default_rng(42)
        _seed_buffer(q, bm, "input", rng)
        _seed_buffer(q, bm, "shared_weights", rng)
        _seed_buffer(q, bm, "shared_biases", rng)
        _seed_buffer(q, bm, "module_weights", rng)
        _seed_buffer(q, bm, "module_biases", rng)
        _seed_buffer_uniform(q, bm, "temperatures", 0.5, 2.0)
        _seed_targets_cce(q, bm, batch_size, spec.output_classes)
        _seed_mask_all_active(q, bm, batch_size)

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

        loss_sig = ComputeProbsLossCceChunkSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            logit_ref=bm.get_handle_by_name("logits"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            target_ref=bm.get_handle_by_name("targets_cce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            prob_out_ref=bm.get_handle_by_name("partial_probs"),
            loss_out_ref=bm.get_handle_by_name("final_loss"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )

        def _pipeline():
            fwd_evt = ex.launch(q, fwd_sig)
            logit_evt = ex.launch(q, logit_sig, wait_for=[fwd_evt])
            loss_evt = ex.launch(q, loss_sig, wait_for=[logit_evt])
            loss_evt.wait()

        benchmark(_pipeline)

    # --- Gradient production tile:  8 + 9 + 10 → 11 (one tile) ---

    @pytest.mark.benchmark(group="pipeline-grad-tile")
    def test_bench_gradient_production_tile_iris(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants,
    ) -> None:
        """
        Composed Phase I tile: calculate_module_param_grads (8) +
        backprop_error_to_hidden (9) + calculate_chunk_temp_grads (10) →
        clip_partial_gradients (11).
        """
        spec = _get_iris_spec(arch_consts)
        batch_size = 150
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        grid = _allocate_model_buffers(bm, spec, batch_size)
        tile = grid.get_tile(0, 0)

        rng = np.random.default_rng(42)
        _seed_buffer(q, bm, "hidden_activations", rng)
        _seed_buffer(q, bm, "partial_probs", rng)
        _seed_buffer(q, bm, "logits", rng)
        _seed_buffer_uniform(q, bm, "temperatures", 0.5, 2.0)
        _seed_targets_cce(q, bm, batch_size, spec.output_classes)
        _seed_mask_all_active(q, bm, batch_size)

        sig_8 = CalculateModuleParamGradsCceSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
            h_ref=bm.get_handle_by_name("hidden_activations"),
            prob_ref=bm.get_handle_by_name("partial_probs"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            gw_out_ref=bm.get_handle_by_name("partial_grad_module_weights"),
            gb_out_ref=bm.get_handle_by_name("partial_grad_module_biases"),
            tile=tile,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(batch_size),
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
            padded_total_output_class_count=np.uint32(spec.padded_class_dim),
            total_modules_count=np.uint32(spec.num_modules),
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
        )

        sig_9 = BackpropErrorToHiddenChunkCceSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            prob_ref=bm.get_handle_by_name("partial_probs"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_mod_ref=bm.get_handle_by_name("module_weights"),
            gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
            tile=tile,
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
        )

        sig_10 = CalculateChunkTempGradientsCceSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
            logit_ref=bm.get_handle_by_name("logits"),
            prob_ref=bm.get_handle_by_name("partial_probs"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            gt_out_ref=bm.get_handle_by_name("partial_grad_temps"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
        )

        handles_11 = GradientHandles(
            grad_weights_module=bm.get_handle_by_name("partial_grad_module_weights"),
            grad_biases_module=bm.get_handle_by_name("partial_grad_module_biases"),
            grad_temps=bm.get_handle_by_name("partial_grad_temps"),
            grad_hidden_activations_aos=bm.get_handle_by_name("partial_grad_hidden_activations"),
            clipped_grad_weights_module=bm.get_handle_by_name("clipped_partial_grad_module_weights"),
            clipped_grad_biases_module=bm.get_handle_by_name("clipped_partial_grad_module_biases"),
            clipped_grad_temps=bm.get_handle_by_name("clipped_partial_grad_temps"),
            clipped_grad_hidden_activations_aos=bm.get_handle_by_name("clipped_partial_grad_hidden_activations"),
        )

        sig_11 = ClipPartialGradientsGlobalNormSignature(
            _buffer_mgr=bm, _arch_consts=arch_consts,
            handles=handles_11,
            tile=tile,
            epsilon=np.float32(1e-7),
            clipping_threshold_global=np.float32(1.0),
        )

        def _pipeline():
            # Nodes 8, 9, 10 are independent and can be enqueued without waits
            evt_8 = ex.launch(q, sig_8)
            evt_9 = ex.launch(q, sig_9)
            evt_10 = ex.launch(q, sig_10)
            # Node 11 waits for all three
            evt_11 = ex.launch(q, sig_11, wait_for=[evt_8, evt_9, evt_10])
            evt_11.wait()

        benchmark(_pipeline)

    # --- Streaming shared backprop loop: (17 + 18) → 19, iterated ---

    @pytest.mark.benchmark(group="pipeline-streaming-backprop")
    @pytest.mark.parametrize(
        "num_batch_chunks", [2, 4], ids=["chunks=2", "chunks=4"],
    )
    def test_bench_streaming_shared_backprop_loop(
        self, benchmark: Any, compute_env: Float32ComputeEnvironment,
        arch_consts: Float32DiscoveredArchConstants, num_batch_chunks: int,
    ) -> None:
        """
        Composed Phase III streaming loop: for each batch chunk, dispatch
        backprop_shared_weights (17) + backprop_shared_biases (18) →
        clip_shared_gradients (19).  Mirrors the True Streaming model.
        """
        spec = _get_iris_spec(arch_consts)
        batch_size = 150
        q = compute_env.cl_bundle.queue
        bm = BufferManager(compute_env.cl_bundle.context)
        ex = KernelExecutor(compute_env.cl_bundle.program)

        _grid = _allocate_model_buffers(bm, spec, batch_size, num_batch_chunks)

        rng = np.random.default_rng(42)
        _seed_buffer(q, bm, "input", rng)
        _seed_buffer(q, bm, "hidden_activations", rng)
        _seed_buffer(q, bm, "summed_grad_hidden_activations", rng)
        _seed_buffer(q, bm, "shared_weights", rng)
        _seed_mask_all_active(q, bm, batch_size)

        chunk_size = (batch_size + num_batch_chunks - 1) // num_batch_chunks

        # Pre-compute the shapes we need for write offsets
        sw_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_shared_weights"))
        sb_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_shared_biases"))
        sw_elements_per_chunk = int(np.prod(sw_shape[1:]))  # input_dim * padded_hidden_dim
        sb_elements_per_chunk = int(np.prod(sb_shape[1:]))  # padded_hidden_dim

        # Build all signatures for each chunk upfront
        sigs_17: List[BackpropSharedWeightsChunkSignature] = []
        sigs_18: List[BackpropSharedBiasesChunkSignature] = []
        sigs_19: List[ClipSharedGradientsChunkSignature] = []

        for i in range(num_batch_chunks):
            offset = i * chunk_size
            count = min(chunk_size, batch_size - offset)
            if count <= 0:
                break

            sigs_17.append(BackpropSharedWeightsChunkSignature(
                _buffer_mgr=bm, _arch_consts=arch_consts,
                input_ref=bm.get_handle_by_name("input"),
                h_ref=bm.get_handle_by_name("hidden_activations"),
                grad_h_ref=bm.get_handle_by_name("summed_grad_hidden_activations"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsw_out_ref=bm.get_handle_by_name("partial_grad_shared_weights"),
                batch_chunk_offset=np.uint32(offset),
                batch_chunk_count=np.uint32(count),
                batch_chunk_index=np.uint32(i),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            ))

            sigs_18.append(BackpropSharedBiasesChunkSignature(
                _buffer_mgr=bm, _arch_consts=arch_consts,
                h_ref=bm.get_handle_by_name("hidden_activations"),
                grad_h_ref=bm.get_handle_by_name("summed_grad_hidden_activations"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsb_out_ref=bm.get_handle_by_name("partial_grad_shared_biases"),
                batch_chunk_offset=np.uint32(offset),
                batch_chunk_count=np.uint32(count),
                batch_chunk_index=np.uint32(i),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            ))

            handles_19 = SharedGradientHandles(
                grad_weights_shared_chunk=bm.get_handle_by_name("partial_grad_shared_weights"),
                grad_biases_shared_chunk=bm.get_handle_by_name("partial_grad_shared_biases"),
                clipped_grad_weights_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_weights"),
                clipped_grad_biases_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_biases"),
            )

            sigs_19.append(ClipSharedGradientsChunkSignature(
                _buffer_mgr=bm, _arch_consts=arch_consts,
                handles=handles_19,
                clipping_threshold_global=np.float32(1.0),
                epsilon=np.float32(1e-7),
                dest_weights_write_offset_elements=np.uint32(i * sw_elements_per_chunk),
                dest_biases_write_offset_elements=np.uint32(i * sb_elements_per_chunk),
                num_batch_chunks=np.uint32(num_batch_chunks),
            ))

        def _pipeline():
            for i in range(len(sigs_17)):
                # 17 and 18 are independent within a chunk
                evt_17 = ex.launch(q, sigs_17[i])
                evt_18 = ex.launch(q, sigs_18[i])
                # 19 waits for both before clipping
                evt_19 = ex.launch(q, sigs_19[i], wait_for=[evt_17, evt_18])
                # Each chunk's clip must complete before the next chunk
                # overwrites the scratch partial buffers
                evt_19.wait()
            q.finish()

        benchmark(_pipeline)
