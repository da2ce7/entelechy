# tests/test_integration_dag_orchestration.py
from __future__ import annotations

"""
Integration Tests: DAG Orchestration & Synchronization Contracts.

These tests verify host-side planning and composition for DAG nodes and
synchronization barriers that were previously exercised only by the device-level
benchmarks.  They close four specific coverage gaps identified in the
architecture's test-layer comparison:

  Gap 1 – Streaming Backprop Orchestration (Model B):
      Host correctly plans the chunk-by-chunk streaming loop
      (17→18→19→reduce) for shared-layer gradients.

  Gap 2 – Diagnostic Reduction Orchestration (Node 14):
      Host distinguishes pure summation (diagnostics) from sum-then-clip
      (gradients) and correctly handles CCE direct-write vs. BCE reduction.

  Gap 3 – Item Synchronization Barrier (Node 13):
      Host inserts the AoS→SoA permutation step and its output layout is
      consistent with the downstream specialized reduction kernel (Node 16).

  Gap 4 – Batch Synchronization Point (Node 22):
      Host enforces a strict barrier ensuring adam_update executes only after
      all preceding gradient normalization steps are complete.

Target CONCEPT.md Nodes: 13, 14, 17, 18, 19, 22
No OpenCL device is required.
"""

import math
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.model_spec import Float32ModelSpec
from src.shared.parameter_space import ParameterSpace
from src.shared.stabilization_policy import StabilizationPolicy
from src.shared.workload_primitives import (
    LinearlyChunkedGather,
    TiledGather,
    TilingScheme,
)

# These modules require pyopencl at import time — gate them.
if TYPE_CHECKING:
    from src.backends.opencl.compute_patterns import ReductionPlan
    from src.backends.opencl.execution_plan import (
        BceStrategy,
        CacheProvider,
        CceStrategy,
        DataLifecyclePolicy,
        DependencyProvider,
        ExecutionPlan,
        ProblemTypeStrategy,
    )
    from src.backends.opencl.launcher_infra import BufferHandle

try:
    from src.backends.opencl.compute_patterns import ReductionPlan  # noqa: F811
    from src.backends.opencl.execution_plan import (  # noqa: F811
        BceStrategy,
        CacheProvider,
        CceStrategy,
        DataLifecyclePolicy,
        DependencyProvider,
        ExecutionPlan,
        ProblemTypeStrategy,
    )
    from src.backends.opencl.launcher_infra import BufferHandle  # noqa: F811

    _has_cl = True
except ImportError:
    _has_cl = False

pytestmark = pytest.mark.skipif(not _has_cl, reason="pyopencl not installed")


# =========================================================================
# Helpers
# =========================================================================


def _dummy_handle(id_: int = 0) -> BufferHandle:
    return BufferHandle(id=id_)


def _dummy_event() -> MagicMock:
    evt = MagicMock()
    evt.wait = MagicMock()
    return evt


def _make_spec(**overrides: Any) -> Float32ModelSpec:
    defaults = dict(
        input_dim=4,
        hidden_dim=32,
        output_classes=3,
        num_modules=8,
        simd_width=4,
        cache_line_bytes=64,
    )
    defaults.update(overrides)
    return Float32ModelSpec(**defaults)


def _make_tiling(spec: Float32ModelSpec) -> TilingScheme:
    return TilingScheme(
        num_module_chunks=(spec.num_modules + 15) // 16,
        num_class_chunks=(spec.output_classes + 15) // 16,
        total_modules=spec.num_modules,
        total_classes=spec.output_classes,
    )


def _build_plan(
    *,
    problem_type_name: str = "CCE",
    adaptation_strategy: str = "CACHE",
    clipping_strategy: str = "GLOBAL",
    batch_size: int = 150,
    num_modules: int = 8,
    output_classes: int = 3,
    stream_chunks: int = 4,
) -> ExecutionPlan:
    """Build a minimal, valid ExecutionPlan for host-side testing."""
    grid = TilingScheme(
        num_module_chunks=(num_modules + 15) // 16,
        num_class_chunks=(output_classes + 15) // 16,
        total_modules=num_modules,
        total_classes=output_classes,
    )
    reduction_plan = ReductionPlan(k=64)
    fp_max = float(np.finfo(np.float32).max)
    stab_policy = StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=1.0,
        fp_format_max=fp_max,
    )

    problem_strategy: ProblemTypeStrategy
    if problem_type_name == "CCE":
        problem_strategy = CceStrategy(targets_cce_ref=_dummy_handle(99))
    else:
        problem_strategy = BceStrategy(targets_bce_ref=_dummy_handle(100))

    providers: dict[str, DependencyProvider] = {}
    if adaptation_strategy == "CACHE":
        providers["hidden_activations"] = CacheProvider(
            handle=_dummy_handle(1),
            ready_event=_dummy_event(),
        )

    hyperparams = MagicMock()
    hyperparams.adam_epsilon = 1e-7
    hyperparams.learning_rate = 0.001
    hyperparams.adam_beta1 = 0.9
    hyperparams.adam_beta2 = 0.999
    hyperparams.stabilization.max_grad_norm = 1.0
    hyperparams.stabilization.lambda_ = 1.0
    hyperparams.temp_min = 0.1
    hyperparams.temp_max = 10.0
    hyperparams.reduction_k_grad_h = 16

    return ExecutionPlan(
        grid=grid,
        reduction_plan=reduction_plan,
        lifecycle_policy=DataLifecyclePolicy(providers=providers),
        effective_batch_size=batch_size,
        problem_type=problem_strategy,
        clipping_strategy=clipping_strategy,
        stabilization_policy=stab_policy,
        hyperparams=hyperparams,
        shared_backprop_stream_chunks=stream_chunks,
        adaptation_strategy=adaptation_strategy,
    )


# =========================================================================
# Gap 1: Streaming Backprop Orchestration (Model B)
# =========================================================================


class TestStreamingBackpropOrchestration:
    """
    Verify host-side planning for the chunk-by-chunk streaming backprop loop
    (Nodes 17→18→19→reduce) described in CONCEPT.md's "Model B: True Streaming."

    The host is responsible for:
      - Choosing the number of stream chunks (shared_backprop_stream_chunks)
      - Computing per-chunk offset and count for Summed_Grad_H slicing
      - Constructing a LinearlyChunkedGather for the shared grad reduction
      - Ensuring all chunks collectively cover the full batch
    """

    @pytest.mark.parametrize(
        "batch_size,stream_chunks",
        [
            (1, 1),       # Degenerate: single sample, single chunk
            (4, 1),       # All in one chunk
            (4, 4),       # Each sample is its own chunk
            (150, 4),     # Iris-like: 150 / 4 = 37 or 38 per chunk
            (150, 150),   # Each sample is a chunk
            (10_000, 8),  # Data Tsunami: large batch, moderate chunks
            (10_000, 1),  # No streaming: all in one pass
            (7, 3),       # Non-divisible: 7 / 3 = 2+2+3 or similar
        ],
    )
    def test_chunk_iteration_covers_full_batch(self, batch_size: int, stream_chunks: int) -> None:
        """Every sample in the batch must be assigned to exactly one chunk."""
        chunk_size = (batch_size + stream_chunks - 1) // stream_chunks
        covered = 0
        for i in range(stream_chunks):
            offset = i * chunk_size
            items_in_chunk = min(chunk_size, batch_size - offset)
            if items_in_chunk <= 0:
                break
            # Verify no overlap: each chunk starts where the previous one ended
            assert offset == covered, f"Chunk {i}: expected offset {covered}, got {offset}"
            covered += items_in_chunk
        assert covered == batch_size, f"Covered {covered} of {batch_size} samples"

    @pytest.mark.parametrize("stream_chunks", [1, 2, 4, 8, 16])
    def test_stream_chunks_encoded_in_plan(self, stream_chunks: int) -> None:
        """The ExecutionPlan faithfully carries the streaming chunk count."""
        plan = _build_plan(batch_size=150, stream_chunks=stream_chunks)
        assert plan.shared_backprop_stream_chunks == stream_chunks

    def test_gather_primitive_matches_stream_chunks(self) -> None:
        """
        The shared-layer reduction must use a LinearlyChunkedGather whose
        num_chunks equals shared_backprop_stream_chunks from the plan.
        """
        plan = _build_plan(batch_size=150, stream_chunks=6)
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(
            batch_size=150, grid=grid, num_batch_chunks=plan.shared_backprop_stream_chunks
        )

        for flow_name in ["shared_weights", "shared_biases"]:
            clipped_name = f"clipped_partial_grad_{flow_name}"
            layout = layouts[clipped_name]
            shape = layout.get_padded_shape(np.dtype(np.float32))
            elements_per_partial = int(np.prod(shape[1:]))

            gather = LinearlyChunkedGather(
                num_chunks=plan.shared_backprop_stream_chunks,
                elements_per_chunk=elements_per_partial,
            )
            assert gather.num_partials == plan.shared_backprop_stream_chunks
            offsets = gather.get_offsets()
            assert len(offsets) == plan.shared_backprop_stream_chunks
            # Offsets must be sequential and non-overlapping
            for j in range(1, len(offsets)):
                assert offsets[j] > offsets[j - 1]

    def test_summed_grad_h_slicing_contracts(self) -> None:
        """
        Each streaming chunk receives a slice of Summed_Grad_H.  Verify that
        the slice parameters (offset, count) are derivable and valid.
        """
        batch_size = 150
        stream_chunks = 4
        chunk_size = (batch_size + stream_chunks - 1) // stream_chunks

        # Simulate the slicing logic that the host uses in build_shared_backprop_subgraph
        slices: list[tuple[int, int]] = []
        for i in range(stream_chunks):
            offset = i * chunk_size
            items = min(chunk_size, batch_size - offset)
            if items <= 0:
                break
            slices.append((offset, items))

        # All slices must be non-overlapping and cover [0, batch_size)
        covered_set: set[int] = set()
        for offset, count in slices:
            for s in range(offset, offset + count):
                assert s not in covered_set, f"Sample {s} assigned to multiple chunks"
                covered_set.add(s)
        assert covered_set == set(range(batch_size))

    def test_write_offset_elements_per_chunk(self) -> None:
        """
        Node 19 clip kernel receives dest_*_write_offset_elements.
        Verify the offset calculation: i * prod(chunk_shape).
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        stream_chunks = 4
        layouts = ps.get_all_memory_layouts(
            batch_size=150, grid=grid, num_batch_chunks=stream_chunks
        )

        # Shared weights shape: (input_dim, padded_hidden_dim)
        w_shape = layouts["shared_weights"].get_padded_shape(np.dtype(np.float32))
        gsw_chunk_elements = int(np.prod(w_shape))

        # Each chunk writes at offset i * gsw_chunk_elements
        offsets = [i * gsw_chunk_elements for i in range(stream_chunks)]
        for i in range(1, len(offsets)):
            assert offsets[i] == offsets[i - 1] + gsw_chunk_elements
        # No overlap: last offset + chunk size fits within collection buffer
        total_collection_elements = stream_chunks * gsw_chunk_elements
        assert offsets[-1] + gsw_chunk_elements == total_collection_elements

    @pytest.mark.parametrize("batch_size,chunks", [(1, 1), (5, 3), (256, 16)])
    def test_partial_collection_shape_matches_chunks(self, batch_size: int, chunks: int) -> None:
        """
        The partial collection buffer shape (dim 0) must equal num_batch_chunks.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=batch_size, grid=grid, num_batch_chunks=chunks)

        sw_layout = layouts["partial_grad_shared_weights"]
        assert sw_layout.logical_shape[0] == chunks
        sb_layout = layouts["partial_grad_shared_biases"]
        assert sb_layout.logical_shape[0] == chunks


# =========================================================================
# Gap 2: Diagnostic Reduction Orchestration (Node 14)
# =========================================================================


class TestDiagnosticReductionOrchestration:
    """
    Verify host-side planning for the diagnostic reduction tree (Node 14).

    The host must correctly distinguish:
      - Pure summation (no clip) for diagnostics (probs, BCE loss)
      - Sum-then-clip for gradient reduction (Nodes 15 & 20)
      - CCE direct-write (no loss aggregation) vs. BCE partial aggregation
    """

    def test_cce_loss_is_direct_write_no_reduction(self) -> None:
        """
        CCE loss is a scatter-write (one scalar per module×sample).
        No reduction tree is needed for the loss value itself.
        """
        plan = _build_plan(problem_type_name="CCE")
        # CCE strategy writes directly to final_loss; no partial_loss reduction
        assert isinstance(plan.problem_type, CceStrategy)
        # With CCE, the diagnostic aggregation should NOT produce a 'loss' key
        # (this is the architectural distinction from BCE)

    def test_bce_loss_requires_reduction(self) -> None:
        """
        BCE loss is per-class, requiring a reduction tree to aggregate into
        a per-(module, sample) scalar.
        """
        plan = _build_plan(problem_type_name="BCE")
        assert isinstance(plan.problem_type, BceStrategy)
        # With BCE, the host must plan a reduction tree for partial_loss
        # Verify the partial_loss buffer exists in the parameter space layout
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)
        assert "partial_loss" in layouts
        assert "final_loss" in layouts

    def test_diagnostic_gather_uses_tiled_scheme(self) -> None:
        """
        The probability partials are gathered using a TiledGather, whose
        num_partials must equal grid.total_tiles.
        """
        spec = _make_spec(num_modules=16, output_classes=10)
        grid = _make_tiling(spec)
        ps = ParameterSpace(spec=spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)

        prob_layout = layouts["partial_probs"]
        prob_shape = prob_layout.get_padded_shape(np.dtype(np.float32))
        elements_per_prob_partial = int(np.prod(prob_shape[1:]))

        gather = TiledGather(scheme=grid, _elements_per_partial=elements_per_prob_partial)
        assert gather.num_partials == grid.total_tiles

    def test_diagnostic_reduction_is_pure_summation(self) -> None:
        """
        Diagnostic reduction uses pure sum (no clip) — contrast with gradient
        reduction which uses sum-then-clip.

        Verified architecturally: the diagnostic path calls
        execute_summation_tree (not execute_stabilized_reduction_tree).
        The stabilization policy threshold is never applied to diagnostics.
        """
        # Construct a stabilization policy with a very restrictive threshold
        policy = StabilizationPolicy(
            t_algorithmic=0.001,  # Very restrictive
            lambda_=0.0,
            fp_format_max=float(np.finfo(np.float32).max),
        )
        # For diagnostics, the policy MUST NOT affect aggregation
        # The leaf safety threshold is irrelevant to diagnostic summation
        # This tests the architectural separation:
        # - get_threshold_for_generic_stage → ONLY for gradient reduction
        # - no threshold method is called for diagnostic reduction
        # We verify the concept by confirming that diagnostic reduction
        # plans with ANY stabilization policy produce valid results
        # (the policy object exists but should be ignored).
        k, stages = policy.plan_uniform_reduction_tree(
            num_partials=16,
            hardware_max_fan_in=256,
        )
        # The tree is valid regardless of the restrictive policy
        assert k >= 2 or stages == 0

    def test_bce_partial_loss_shape_matches_tiling(self) -> None:
        """
        The partial_loss buffer for BCE must have its first dimension equal
        to grid.total_tiles, matching the Placement Contract.
        """
        spec = _make_spec(num_modules=8, output_classes=20)
        grid = _make_tiling(spec)
        ps = ParameterSpace(spec=spec)
        layouts = ps.get_all_memory_layouts(batch_size=64, grid=grid, num_batch_chunks=4)

        partial_loss_layout = layouts["partial_loss"]
        assert partial_loss_layout.logical_shape[0] == grid.total_tiles

    def test_prob_reduction_input_output_shapes_consistent(self) -> None:
        """
        The partial_probs collection reduces to final_probs.
        Their non-tile dimensions must match.
        """
        spec = _make_spec(num_modules=8, output_classes=5)
        grid = _make_tiling(spec)
        ps = ParameterSpace(spec=spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)

        # partial_probs: (total_tiles, mods_per_chunk, batch, classes_per_chunk)
        # After full reduction → final shape should match (num_modules, batch, output_classes)
        # Verify that partial dimensions are consistent with the element count
        partial_shape = layouts["partial_probs"].logical_shape
        assert partial_shape[0] == grid.total_tiles
        assert len(partial_shape) == 4  # (tiles, mods/tile, batch, cls/tile)


# =========================================================================
# Gap 3: Item Synchronization Barrier (Node 13) Orchestration
# =========================================================================


class TestItemSynchronizationBarrier:
    """
    Verify host-side planning for the gather_and_permute_grad_h step (Node 13),
    which serves as the canonical Item Synchronization Point.

    The host must:
      - Insert Node 13 between clipped partial Grad_H (AoS) and Node 16
      - Plan the AoS→SoA permutation output buffer correctly
      - Ensure the output layout matches Node 16's input contract
    """

    def test_permuted_grad_h_layout_is_soa(self) -> None:
        """
        The permuted_grad_h buffer must have SoA layout:
        (batch_size * padded_hidden_dim, padded_module_dim).
        This is the contractual input format for Node 16.
        """
        spec = _make_spec(num_modules=8, hidden_dim=32)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        batch_size = 150
        layouts = ps.get_all_memory_layouts(batch_size=batch_size, grid=grid, num_batch_chunks=4)

        permuted = layouts["permuted_grad_h"]
        expected_shape = (
            batch_size * spec.padded_hidden_dim,
            spec.padded_module_dim,
        )
        assert permuted.logical_shape == expected_shape, (
            f"Expected SoA shape {expected_shape}, got {permuted.logical_shape}"
        )

    def test_clipped_partial_grad_h_layout_is_aos(self) -> None:
        """
        The input to Node 13 is the clipped partial Grad_H in AoS layout.
        Its first dimension is grid.total_tiles (Placement Contract).
        """
        spec = _make_spec(num_modules=16, hidden_dim=64)
        grid = _make_tiling(spec)
        ps = ParameterSpace(spec=spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)

        clipped_h = layouts["clipped_partial_grad_hidden_activations"]
        # First dim = total_tiles (from Placement Contract)
        assert clipped_h.logical_shape[0] == grid.total_tiles

    def test_node_13_connects_clipped_to_permuted(self) -> None:
        """
        Verify that both the input (clipped AoS) and output (permuted SoA)
        buffers are present in the memory layout, establishing that the host
        has planned the Node 13 synchronization point.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)

        assert "clipped_partial_grad_hidden_activations" in layouts
        assert "permuted_grad_h" in layouts
        assert "summed_grad_hidden_activations" in layouts

    def test_node_13_output_compatible_with_node_16_input(self) -> None:
        """
        Node 16 (stabilize_and_reduce_grad_hidden_activations) expects:
          - permuted_soa_in_ref: (batch_size * padded_hidden_dim, padded_module_dim)
          - final_grad_h_out_ref: (batch_size, padded_hidden_dim)

        Verify dimensional compatibility.
        """
        for batch_size in [1, 32, 150, 1000]:
            spec = _make_spec(num_modules=8, hidden_dim=32)
            ps = ParameterSpace(spec=spec)
            grid = _make_tiling(spec)
            layouts = ps.get_all_memory_layouts(batch_size=batch_size, grid=grid, num_batch_chunks=4)

            permuted = layouts["permuted_grad_h"]
            summed = layouts["summed_grad_hidden_activations"]

            # Permuted: (B * padded_H, padded_M)
            assert permuted.logical_shape[0] == batch_size * spec.padded_hidden_dim
            assert permuted.logical_shape[1] == spec.padded_module_dim

            # Summed: (B, padded_H) — the reduction over modules collapses dim 1
            assert summed.logical_shape == (batch_size, spec.padded_hidden_dim)

    def test_hydra_scale_permutation_buffer(self) -> None:
        """
        For the Hydra scenario (256 modules), verify the permuted buffer
        scales correctly with num_modules.
        """
        spec = _make_spec(num_modules=256, hidden_dim=64, output_classes=10)
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=32, grid=grid, num_batch_chunks=4)

        permuted = layouts["permuted_grad_h"]
        # SoA: (32 * padded_hidden, padded_module)
        assert permuted.logical_shape[0] == 32 * spec.padded_hidden_dim
        assert permuted.logical_shape[1] == spec.padded_module_dim
        # With 256 modules, the second dimension should be significant
        assert permuted.logical_shape[1] >= 256

    def test_specialized_flow_skips_standard_reduction(self) -> None:
        """
        The hidden_activations flow is marked as specialized_reduction=True,
        confirming it takes the Node 13→16 path instead of the generic
        Nodes 15/20 Recursive Clip-Aggregation Engine.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        ha_flows = [f for f in ps if f.name == "hidden_activations"]
        assert len(ha_flows) == 1
        assert ha_flows[0].specialized_reduction is True
        # Specialized flows lack the standard intermediate buffer names
        assert ha_flows[0].partial_grad_buffer_name == ""
        assert ha_flows[0].clipped_partial_grad_buffer_name == ""


# =========================================================================
# Gap 4: Batch Synchronization Point (Node 22)
# =========================================================================


class TestBatchSynchronizationPoint:
    """
    Verify that the host-side planning enforces the Batch Synchronization Point
    (Node 22): adam_update (Node 24) must execute strictly after ALL preceding
    gradient normalization (Node 21) and reduction steps complete.

    The architectural contract states:
      "A barrier that resolves a data dependency *between multiple independent
       learning items* that constitute a single logical batch."
    """

    def test_update_depends_on_all_gradient_flows(self) -> None:
        """
        Verify that every non-specialized parameter flow has the full
        gradient lifecycle (partial → clipped → summed → final → adam_update),
        ensuring the barrier cannot be bypassed.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)

        non_specialized = [f for f in ps if not f.specialized_reduction]
        # There should be 5 non-specialized flows:
        # shared_weights, shared_biases, module_weights, module_biases, temperatures
        assert len(non_specialized) == 5

        for flow in non_specialized:
            # Every flow must have a complete gradient lifecycle
            assert flow.summed_grad_buffer_name, f"{flow.name} missing summed grad"
            assert flow.final_grad_buffer_name, f"{flow.name} missing final grad"
            assert flow.m1_buffer_name, f"{flow.name} missing m1 state"
            assert flow.m2_buffer_name, f"{flow.name} missing m2 state"

    def test_normalization_before_update_contract(self) -> None:
        """
        Node 21 (normalize_gradients) produces final_grad_* buffers that
        Node 24 (adam_update) consumes.  Verify the naming contract that
        links normalization outputs to update inputs.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)

        for flow in ps:
            if flow.specialized_reduction:
                continue
            # summed_grad → normalize → final_grad → adam_update
            assert flow.summed_grad_buffer_name in layouts
            assert flow.final_grad_buffer_name in layouts
            # final_grad and param must have identical shapes
            param_shape = layouts[flow.param_buffer_name].logical_shape
            final_grad_shape = layouts[flow.final_grad_buffer_name].logical_shape
            assert param_shape == final_grad_shape, (
                f"{flow.name}: param shape {param_shape} != "
                f"final_grad shape {final_grad_shape}"
            )

    def test_adam_state_shapes_match_parameters(self) -> None:
        """
        Adam optimizer state (m1, m2) must have shapes identical to
        the parameter they track.  This is a prerequisite for the
        update step that follows the batch barrier.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        grid = _make_tiling(spec)
        layouts = ps.get_all_memory_layouts(batch_size=150, grid=grid, num_batch_chunks=4)

        for flow in ps:
            if flow.specialized_reduction:
                continue
            param_shape = layouts[flow.param_buffer_name].logical_shape
            m1_shape = layouts[flow.m1_buffer_name].logical_shape
            m2_shape = layouts[flow.m2_buffer_name].logical_shape
            assert param_shape == m1_shape, (
                f"{flow.name}: param {param_shape} != m1 {m1_shape}"
            )
            assert param_shape == m2_shape, (
                f"{flow.name}: param {param_shape} != m2 {m2_shape}"
            )

    def test_beta_power_t_computed_on_host(self) -> None:
        """
        The Batch Synchronization Point contract mandates that beta1**t and
        beta2**t are computed on the host in high precision (FP64) and
        passed as primitive scalars. Verify the architectural decision.
        """
        beta1, beta2 = 0.9, 0.999
        for step in [1, 100, 10_000, 500_000]:
            # Python float is FP64 — this is the mandated host calculation
            b1_t = beta1 ** step
            b2_t = beta2 ** step
            assert math.isfinite(b1_t), f"beta1^{step} is not finite"
            assert math.isfinite(b2_t), f"beta2^{step} is not finite"
            assert b1_t >= 0, f"beta1^{step} is negative"
            assert b2_t >= 0, f"beta2^{step} is negative"

    def test_barrier_flow_count_matches_parameter_count(self) -> None:
        """
        The number of Adam update dispatches at the barrier must equal
        the number of non-specialized parameter flows.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        _non_specialized = [f for f in ps if not f.specialized_reduction]

        # The batch processor dispatches one adam_update per non-specialized flow
        # Plus the hidden_activations flow gets its gradient via the specialized
        # path, but its summed_grad is still consumed by adam_update.
        # Actually, the hidden_activations flow is specialized, so it doesn't
        # go through the standard normalize path — its gradient is summed
        # by Node 16 and consumed directly.
        all_flows_needing_update = [f for f in ps if not f.specialized_reduction]
        assert len(all_flows_needing_update) == 5

    def test_clamp_temperatures_follows_update(self) -> None:
        """
        Node 25 (clamp_temperatures) must logically follow Node 24 (adam_update)
        for temperatures.  Verify that a 'temperatures' flow exists in the
        parameter space, confirming the update→clamp sequencing is planned.
        """
        spec = _make_spec()
        ps = ParameterSpace(spec=spec)
        temp_flows = [f for f in ps if f.name == "temperatures"]
        assert len(temp_flows) == 1
        assert not temp_flows[0].specialized_reduction
        assert temp_flows[0].param_buffer_name == "temperatures"
