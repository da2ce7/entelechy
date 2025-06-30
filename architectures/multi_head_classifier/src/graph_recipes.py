# graph_recipes.py

"""
A Toolbox of Canonical Sub-Graph Recipes.

(First Draft) This module provides a set of pure, stateless functions that
encapsulate the fundamental "recipes" of the training algorithm. Each function
takes all necessary context and dependencies, and orchestrates a sequence of
kernel launches on the device command queue, returning the final completion event.

This file is the direct result of refactoring the procedural logic out of the
PhaseExecutor classes to create a library of reusable, testable, and stateless
computational patterns.
"""


from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from execution_plan import ExecutionPlan, WorkTile
from launcher_infra import BufferManager, KernelExecutor, BufferHandle, SCALAR_NP_TYPE, PingPongManager
from kernel_signatures import *
from model_spec import ModelSpec
from parameter_space import ParameterSpace
from compute_patterns import AggregationManager, ReductionPlan
from workload_primitives import GatherPrimitive, ContiguousGather


# --- Standardized Dependency Bundle ---
@dataclass(frozen=True)
class Services:
    """A simple container for passing core system components for dependency injection."""

    q: cl.CommandQueue
    ex: KernelExecutor
    bm: BufferManager
    model_spec: ModelSpec
    arch_consts: Dict[str, int]


# =========================================================================
# === Recipes for Forward Pass & Tiled Gradient Production              ===
# =========================================================================


def execute_forward_pass(svs: Services, batch_size: int, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
    """
    Recipe for the initial shared layer forward pass (Node 4).
    This logic is extracted from the original `ForwardPassExecutor` class.

    Args:
        svs: The bundle of core system services (queue, executor, etc.).
        batch_size: The number of items in this batch.
        deps: A list of `cl.Event` objects to wait for before executing.

    Returns:
        A tuple containing the BufferHandle for the output and the final cl.Event.
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    spec = svs.model_spec

    sig = ForwardPassSignature(
        buffer_mgr=bm,
        simd_width=spec.simd_width,
        local_mem_bank_padding=1,
        scalar_size_bytes=np.dtype(spec.scalar_dtype).itemsize,
        in_ref=bm.get_handle_by_name("input"),
        mask_ref=bm.get_handle_by_name("sample_mask"),
        w_ref=bm.get_handle_by_name("shared_weights"),
        b_ref=bm.get_handle_by_name("shared_biases"),
        h_out_ref=bm.get_handle_by_name("hidden_activations"),
        h_mask_out_ref=bm.get_handle_by_name("hidden_mask"),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(batch_size),
    )
    event = ex.launch(q, sig, wait_for=deps)
    return sig.h_out_ref, event


def build_module_path_subgraph(svs: Services, tile: WorkTile, plan: ExecutionPlan, deps: List[cl.Event]) -> cl.Event:
    """
    Recipe for the parallel per-tile module path (Nodes 5-11).
    This logic is extracted from the original `ModulePathExecutor` class. It
    orchestrates the complex sequence: Logits -> Probs/Loss -> Partial Grads -> Clipping.

    Args:
        svs: The bundle of core system services (queue, executor, etc.).
        tile: The `WorkTile` object defining this specific unit of work.
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        deps: A list of `cl.Event` objects to wait for before executing.

    Returns:
        The final `cl.Event` that signals the completion of clipping for this tile.
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    spec, h_params = svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts

    # 1. Resolve `hidden_activations` dependency using the plan's defined policy.
    #    This allows the recipe to be agnostic to whether the data is cached or recomputed.
    h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
    h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)

    # 2. Launch Logits Rendering (Node 5)
    logits_sig = RenderLogitsChunkSignature(
        buffer_mgr=bm,
        h_ref=h_ref,
        h_mask_ref=bm.get_handle_by_name("hidden_mask"),
        w_ref=bm.get_handle_by_name("module_weights"),
        b_ref=bm.get_handle_by_name("module_biases"),
        logit_out_ref=bm.get_handle_by_name("logits"),
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(plan.effective_batch_size),
        module_chunk_offset=np.uint32(tile.module_chunk_offset),
        module_chunk_count=np.uint32(tile.modules_per_chunk),
        class_chunk_offset=np.uint32(tile.class_chunk_offset),
        class_chunk_count=np.uint32(tile.classes_per_chunk),
        hidden_count=np.uint32(spec.hidden_dim),
        total_output_class_count=np.uint32(spec.output_classes),
    )
    logits_evt = ex.launch(q, logits_sig, wait_for=[h_ready_evt])

    # 3. Launch Loss & Probabilities (Nodes 6 or 7) - Dynamically selected based on the plan
    if plan.problem_type == "CCE":
        loss_sig = ComputeProbsLossCceChunkSignature(
            buffer_mgr=bm,
            logit_ref=logits_sig.logit_out_ref,
            temp_ref=bm.get_handle_by_name("temperatures"),
            target_ref=bm.get_handle_by_name("targets_cce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            prob_out_ref=bm.get_handle_by_name("partial_probs"),
            loss_out_ref=bm.get_handle_by_name("final_loss"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )
    else:  # BCE
        loss_sig = ComputeProbsLossBceChunkSignature(
            buffer_mgr=bm,
            logit_ref=logits_sig.logit_out_ref,
            temp_ref=bm.get_handle_by_name("temperatures"),
            target_ref=bm.get_handle_by_name("targets_bce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            prob_out_ref=bm.get_handle_by_name("partial_probs"),
            partial_loss_out_ref=bm.get_handle_by_name("partial_loss"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )
    loss_evt = ex.launch(q, loss_sig, wait_for=[logits_evt])

    # 4. Launch Raw Partial Gradient Kernels (Nodes 8, 9, 10) - Dynamically
    wgs0 = arch_consts.get("work_group_size_0", 256)
    scalar_bytes = spec.scalar_dtype().itemsize
    padded_class_dim = spec.padded_class_dim

    if plan.problem_type == "CCE":
        grad_mod_sig = CalculateModuleParamGradsCceSignature(
            bm,
            wgs0,
            scalar_bytes,
            h_ref,
            loss_sig.prob_out_ref,
            bm.get_handle_by_name("targets_cce"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("partial_grad_module_weights"),
            bm.get_handle_by_name("partial_grad_module_biases"),
            tile,
            np.uint32(0),
            np.uint32(plan.effective_batch_size),
            np.uint32(spec.hidden_dim),
            np.uint32(spec.output_classes),
            np.uint32(padded_class_dim),
            np.uint32(spec.num_modules),
        )
        grad_h_sig = BackpropErrorToHiddenChunkCceSignature(
            bm,
            loss_sig.prob_out_ref,
            bm.get_handle_by_name("targets_cce"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("module_weights"),
            bm.get_handle_by_name("partial_grad_hidden_activations"),
            tile,
            np.uint32(spec.hidden_dim),
            np.uint32(spec.output_classes),
        )
        grad_t_sig = CalculateChunkTempGradientsCceSignature(
            bm,
            wgs0,
            scalar_bytes,
            logits_sig.logit_out_ref,
            loss_sig.prob_out_ref,
            bm.get_handle_by_name("targets_cce"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("temperatures"),
            bm.get_handle_by_name("partial_grad_temps"),
            tile,
            np.uint32(spec.output_classes),
        )
    else:  # BCE
        grad_mod_sig = CalculateModuleParamGradsBceSignature(
            bm,
            wgs0,
            scalar_bytes,
            h_ref,
            loss_sig.prob_out_ref,
            bm.get_handle_by_name("targets_bce"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("partial_grad_module_weights"),
            bm.get_handle_by_name("partial_grad_module_biases"),
            tile,
            np.uint32(0),
            np.uint32(plan.effective_batch_size),
            np.uint32(spec.hidden_dim),
            np.uint32(spec.output_classes),
            np.uint32(padded_class_dim),
            np.uint32(spec.num_modules),
        )
        grad_h_sig = BackpropErrorToHiddenChunkBceSignature(
            bm,
            loss_sig.prob_out_ref,
            bm.get_handle_by_name("targets_bce"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("module_weights"),
            bm.get_handle_by_name("partial_grad_hidden_activations"),
            tile,
            np.uint32(spec.hidden_dim),
            np.uint32(spec.output_classes),
        )
        grad_t_sig = CalculateChunkTempGradientsBceSignature(
            bm,
            wgs0,
            scalar_bytes,
            logits_sig.logit_out_ref,
            loss_sig.prob_out_ref,
            bm.get_handle_by_name("targets_bce"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("temperatures"),
            bm.get_handle_by_name("partial_grad_temps"),
            tile,
            np.uint32(spec.output_classes),
        )

    grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=[loss_evt])
    grad_h_evt = ex.launch(q, grad_h_sig, wait_for=[loss_evt])
    grad_t_evt = ex.launch(q, grad_t_sig, wait_for=[loss_evt])

    # 5. Launch Gradient Clipping (Node 11) - Dynamically selected based on the plan
    grad_handles = GradientHandles(
        grad_weights_module=grad_mod_sig.gw_out_ref,
        grad_biases_module=grad_mod_sig.gb_out_ref,
        grad_temps=grad_t_sig.gt_out_ref,
        grad_hidden_activations_aos=grad_h_sig.gh_out_ref,
        clipped_grad_weights_module=bm.get_handle_by_name("clipped_partial_grad_module_weights"),
        clipped_grad_biases_module=bm.get_handle_by_name("clipped_partial_grad_module_biases"),
        clipped_grad_temps=bm.get_handle_by_name("clipped_partial_grad_temps"),
        clipped_grad_hidden_activations_aos=bm.get_handle_by_name("clipped_partial_grad_hidden_activations"),
    )

    if plan.clipping_strategy == "GLOBAL":
        clip_sig = ClipPartialGradientsGlobalNormSignature(
            buffer_mgr=bm,
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            handles=grad_handles,
            tile=tile,
            max_norm_global=SCALAR_NP_TYPE(h_params.max_grad_norm),
            epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
    else:  # 'PER_ITEM'
        clip_sig = ClipPartialGradientsPerItemNormSignature(
            buffer_mgr=bm,
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            handles=grad_handles,
            tile=tile,
            max_norm_per_item_ref=bm.get_handle_by_name("max_norm_per_item"),
            epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
        )

    clip_event = ex.launch(q, clip_sig, wait_for=[grad_mod_evt, grad_h_evt, grad_t_evt])
    # The final event of the entire recipe is returned, signaling that this tile is complete.
    return clip_event


# =========================================================================
# === Recipes for Backpropagation & Specialized Pipelines               ===
# =========================================================================


def build_shared_backprop_subgraph(svs: Services, plan: ExecutionPlan, deps: List[cl.Event]) -> cl.Event:
    """
    Recipe for the streaming shared layer backpropagation (Nodes 17-19).
    This logic is extracted from the original `SharedBackpropExecutor` class. It
    iteratively computes and clips partial gradients for the shared weights and
    biases, chunk by chunk.

    Args:
        svs: The bundle of core system services (queue, executor, etc.).
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        deps: A list of `cl.Event` objects to wait for before executing.

    Returns:
        A single `cl.Event` that signals when all chunks have been processed.
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    spec, h_params = svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts
    num_batch_chunks = plan.shared_backprop_stream_chunks
    final_chunk_events = []

    # 1. Resolve Batch-Wide Dependencies (Once, outside the loop)
    grad_h_provider = plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
    grad_h_ref, grad_h_ready_evt = grad_h_provider.resolve(q, ex, wait_for=deps)
    h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
    h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)
    deps_for_all_chunks = [grad_h_ready_evt, h_ready_evt]

    # 2. Acquire Transient Scratch Buffers for Raw Gradients
    w_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_weights"))
    b_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_biases"))
    gsw_chunk_shape = (w_shape[0], w_shape[1])
    gsb_chunk_shape = (b_shape[0],)
    gsw_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gsw_chunk_shape) * spec.scalar_dtype().itemsize))
    gsb_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gsb_chunk_shape) * spec.scalar_dtype().itemsize))

    # 3. Main Streaming Loop over batch chunks
    batch_size = plan.effective_batch_size
    chunk_size = (batch_size + num_batch_chunks - 1) // num_batch_chunks
    for i in range(num_batch_chunks):
        batch_offset = i * chunk_size
        items_in_chunk = min(chunk_size, batch_size - batch_offset)
        if items_in_chunk <= 0:
            continue

        # Step 3a: Compute Raw Partials (Nodes 17 & 18) into SCRATCH buffers
        gsw_sig = BackpropSharedWeightsChunkSignature(
            bm,
            arch_consts.get("work_group_size_1", 16),
            spec.scalar_dtype().itemsize,
            bm.get_handle_by_name("input"),
            h_ref,
            grad_h_ref,
            bm.get_handle_by_name("sample_mask"),
            gsw_scratch_ref,
            np.uint32(batch_offset),
            np.uint32(items_in_chunk),
            np.uint32(i),
            np.uint32(num_batch_chunks),
        )
        gsw_evt = ex.launch(q, gsw_sig, wait_for=deps_for_all_chunks)
        gsb_sig = BackpropSharedBiasesChunkSignature(
            bm,
            arch_consts.get("work_group_size_0", 256),
            spec.scalar_dtype().itemsize,
            h_ref,
            grad_h_ref,
            bm.get_handle_by_name("sample_mask"),
            gsb_scratch_ref,
            np.uint32(batch_offset),
            np.uint32(items_in_chunk),
            np.uint32(i),
            np.uint32(num_batch_chunks),
        )
        gsb_evt = ex.launch(q, gsb_sig, wait_for=deps_for_all_chunks)

        # Step 3b: Clip Raw Partials (Node 19) from scratch into final COLLECTION buffers
        shared_grad_handles = SharedGradientHandles(
            grad_weights_shared_chunk=gsw_scratch_ref,
            grad_biases_shared_chunk=gsb_scratch_ref,
            clipped_grad_weights_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_weights"),
            clipped_grad_biases_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_biases"),
        )
        clip_sig = ClipSharedGradientsChunkSignature(
            bm,
            arch_consts.get("work_group_size_0", 256),
            spec.scalar_dtype().itemsize,
            shared_grad_handles,
            SCALAR_NP_TYPE(h_params.max_grad_norm),
            SCALAR_NP_TYPE(h_params.adam_epsilon),
            np.uint32(i * np.prod(gsw_chunk_shape)),
            np.uint32(i * np.prod(gsb_chunk_shape)),
            np.uint32(num_batch_chunks),
        )
        clip_evt = ex.launch(q, clip_sig, wait_for=[gsw_evt, gsb_evt])
        final_chunk_events.append(clip_evt)

    # 4. Release Transient Resources and return a single synchronization event
    bm.release_transient_buffer(gsw_scratch_ref)
    bm.release_transient_buffer(gsb_scratch_ref)
    return cl.WaitForEvents(final_chunk_events)


def execute_grad_h_streaming_pipeline(
    svs: Services, plan: ExecutionPlan, deps: List[cl.Event]
) -> Tuple[BufferHandle, cl.Event]:
    """
    Recipe for the specialized "Model A: Accumulate via Recompute" strategy.
    This logic is extracted from the original `GradHStreamingExecutor`. It manages
    the full recompute->bprop->clip->permute->reduce pipeline for `Grad_H`.

    Args:
        svs: The bundle of core system services (queue, executor, etc.).
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        deps: A list of `cl.Event` objects to wait for before executing.

    Returns:
        A tuple of (final_summed_grad_h_handle, final_completion_event).
    """
    q, ex, bm = svs.q, svs.ex, svs.q
    spec, h_params = svs.model_spec, plan.hyperparams
    arch_consts, grid = svs.arch_consts, plan.grid
    scalar_bytes = spec.scalar_dtype().itemsize

    # 1. Acquire Monolithic Intermediates & Transient Scratch Buffers
    clipped_aos_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")
    permuted_soa_ref = bm.get_handle_by_name("permuted_grad_h")
    final_summed_grad_h_ref = bm.get_handle_by_name("summed_grad_hidden_activations")

    h_chunk_shape, _ = bm.get_spec(bm.get_handle_by_name("hidden_activations"))
    gh_chunk_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_hidden_activations"))
    gh_chunk_shape = (1, gh_chunk_shape[1], gh_chunk_shape[2], gh_chunk_shape[3])

    h_chunk_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_chunk_shape) * scalar_bytes))
    h_mask_chunk_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_chunk_shape) * scalar_bytes))
    raw_gh_chunk_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gh_chunk_shape) * scalar_bytes))

    # 2. Loop over module/class tiles to accumulate clipped partials
    all_clip_events = []
    for tile in grid:
        fwd_pass_sig = ForwardPassSignature(
            bm,
            spec.simd_width,
            1,
            scalar_bytes,
            bm.get_handle_by_name("input"),
            bm.get_handle_by_name("sample_mask"),
            bm.get_handle_by_name("shared_weights"),
            bm.get_handle_by_name("shared_biases"),
            h_chunk_scratch_ref,
            h_mask_chunk_scratch_ref,
            np.uint32(0),
            np.uint32(plan.effective_batch_size),
        )
        h_ready_evt = ex.launch(q, fwd_pass_sig, wait_for=deps)

        if plan.problem_type == "CCE":
            grad_h_sig = BackpropErrorToHiddenChunkCceSignature(
                bm,
                bm.get_handle_by_name("partial_probs"),
                bm.get_handle_by_name("targets_cce"),
                bm.get_handle_by_name("sample_mask"),
                bm.get_handle_by_name("module_weights"),
                raw_gh_chunk_scratch_ref,
                tile,
                np.uint32(spec.hidden_dim),
                np.uint32(spec.output_classes),
            )
        else:  # BCE
            grad_h_sig = BackpropErrorToHiddenChunkBceSignature(
                bm,
                bm.get_handle_by_name("partial_probs"),
                bm.get_handle_by_name("targets_bce"),
                bm.get_handle_by_name("sample_mask"),
                bm.get_handle_by_name("module_weights"),
                raw_gh_chunk_scratch_ref,
                tile,
                np.uint32(spec.hidden_dim),
                np.uint32(spec.output_classes),
            )

        raw_gh_ready_evt = ex.launch(q, grad_h_sig, wait_for=[h_ready_evt] + deps)

        # Instantiate GradientHandles to reflect that this pipeline ONLY processes
        # the grad_hidden_activations. The other handles are passed as None,
        # signaling to a robust signature that they are unused. This prevents
        # the kernel from reading garbage from the main collection buffers.
        grad_handles = GradientHandles(
            grad_weights_module=None,
            grad_biases_module=None,
            grad_temps=None,
            grad_hidden_activations_aos=raw_gh_chunk_scratch_ref,
            clipped_grad_weights_module=None,
            clipped_grad_biases_module=None,
            clipped_grad_temps=None,
            clipped_grad_hidden_activations_aos=clipped_aos_ref,
        )

        clip_sig = ClipPartialGradientsGlobalNormSignature(
            bm,
            arch_consts.get("work_group_size_0", 256),
            scalar_bytes,
            grad_handles,
            tile,
            SCALAR_NP_TYPE(h_params.max_grad_norm),
            SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
        clip_evt = ex.launch(q, clip_sig, wait_for=[raw_gh_ready_evt])
        all_clip_events.append(clip_evt)

    # 3. Release transient resources now that accumulation is complete
    bm.release_transient_buffer(h_chunk_scratch_ref)
    bm.release_transient_buffer(h_mask_chunk_scratch_ref)
    bm.release_transient_buffer(raw_gh_chunk_scratch_ref)
    all_clips_done = cl.WaitForEvents(all_clip_events)

    # 4. Permute the now-filled monolithic buffer (Node 13)
    permute_sig = GatherAndPermuteGradHSignature(
        bm,
        clipped_aos_ref,
        permuted_soa_ref,
        np.uint32(spec.num_modules),
        np.uint32(spec.hidden_dim),
        np.uint32(plan.effective_batch_size),
        np.uint32(plan.grid.num_module_chunks),
        np.uint32(plan.grid.get_tile(0, 0).modules_per_chunk),
        np.uint32(plan.grid.num_class_chunks),
    )
    permute_evt = ex.launch(q, permute_sig, wait_for=[all_clips_done])

    # 5. Reduce the permuted buffer (Node 16)
    reduce_sig = ReduceGradHOverModulesSignature(
        bm,
        arch_consts.get("work_group_size_0", 256),
        scalar_bytes,
        permuted_soa_ref,
        final_summed_grad_h_ref,
        np.uint32(spec.num_modules),
    )
    final_reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])

    return final_summed_grad_h_ref, final_reduce_evt


# =========================================================================
# === Recipes for Aggregation & Final Update                            ===
# =========================================================================


def execute_reduction_tree(
    svs: Services,
    reduction_plan: ReductionPlan,
    gather_primitive: GatherPrimitive,
    partial_collection_ref: BufferHandle,
    final_dest_handle: BufferHandle,
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """
    A pure, stateless recipe for executing a complete, multi-stage log_K(N) reduction.
    This complex logic is extracted from the original `ReductionTreeExecutor`.

    Args:
        svs: The bundle of core system services (queue, executor, etc.).
        reduction_plan: The `ReductionPlan` defining fan-in and other config.
        gather_primitive: The GatherPrimitive describing the layout of the partials.
        partial_collection_ref: Handle to the buffer containing all partials to be reduced.
        final_dest_handle: Handle to the dense buffer for the final result.
        wait_for: A list of `cl.Event` objects to wait for before executing.

    Returns:
        A single `cl.Event` that signals the completion of the entire reduction tree.
    """
    wait_for = wait_for or []
    q, ex, bm = svs.q, svs.ex, svs.bm
    n = gather_primitive.num_partials

    if n <= 0:
        user_event = cl.UserEvent(q.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

    # This helper is now a nested function, preserving a clean module namespace.
    def _create_offset_list(offsets_host: np.ndarray, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        offset_list_ref = bm.acquire_transient_buffer(offsets_host.nbytes)
        evt = cl.enqueue_copy(q, bm.get_cl_buffer(offset_list_ref), offsets_host, wait_for=deps)
        return offset_list_ref, evt

    elements_per_partial = gather_primitive.elements_per_partial
    scalar_byte_size = bm.get_spec(final_dest_handle)[1]().itemsize
    partial_byte_size = elements_per_partial * scalar_byte_size

    initial_offsets_host = gather_primitive.get_offsets()

    if n == 1:
        src_offset_bytes = int(initial_offsets_host[0] * scalar_byte_size)
        return cl.enqueue_copy_buffer(
            q,
            src=bm.get_cl_buffer(partial_collection_ref),
            dst=bm.get_cl_buffer(final_dest_handle),
            byte_count=partial_byte_size,
            src_offset=src_offset_bytes,
            dst_offset=0,
            wait_for=wait_for,
        )

    # --- Setup for N > 1 Reduction Tree ---
    ppm = PingPongManager()
    agg_mgr = AggregationManager(ex, bm, svs.arch_consts)
    stage_output_partials = (n + reduction_plan.k - 1) // reduction_plan.k
    ppm_buffer_size_bytes = stage_output_partials * partial_byte_size
    ppm.initialize(bm, max_bytes=ppm_buffer_size_bytes)
    transient_handles = []
    try:
        offset_list_ref, upload_evt = _create_offset_list(initial_offsets_host, wait_for)
        transient_handles.append(offset_list_ref)

        current_n = n
        current_collection_ref = partial_collection_ref
        current_deps = [upload_evt]

        while current_n > reduction_plan.k:
            stage_dest_ref, _ = ppm.get_io()
            stage_event = agg_mgr.execute_stage(
                q,
                current_collection_ref,
                offset_list_ref,
                current_n,
                elements_per_partial,
                stage_dest_ref,
                current_deps,
            )
            current_deps = [stage_event]

            next_n = (current_n + reduction_plan.k - 1) // reduction_plan.k
            current_collection_ref = stage_dest_ref
            next_gather_primitive = ContiguousGather(next_n, elements_per_partial)
            next_offsets_host = next_gather_primitive.get_offsets()
            offset_list_ref, upload_evt = _create_offset_list(next_offsets_host, current_deps)
            transient_handles.append(offset_list_ref)
            current_deps = [upload_evt]
            current_n = next_n
            ppm.swap()

        final_stage_event = agg_mgr.execute_stage(
            q, current_collection_ref, offset_list_ref, current_n, elements_per_partial, final_dest_handle, current_deps
        )
        return final_stage_event

    finally:
        ppm.release()
        for handle in transient_handles:
            bm.release_transient_buffer(handle)


def execute_specialized_grad_h_reduction(
    svs: Services, plan: ExecutionPlan, deps: List[cl.Event]
) -> Tuple[BufferHandle, cl.Event]:
    """
    Recipe for the specialized permutation and reduction of Grad_H.
    Extracted from the original `ReductionPhaseExecutor.run_single_flow`.
    """
    q, ex, bm = svs.q, svs.ex, svs.q
    spec, arch_consts = svs.model_spec, svs.arch_consts

    clipped_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")
    permuted_ref = bm.get_handle_by_name("permuted_grad_h")
    summed_ref = bm.get_handle_by_name("summed_grad_hidden_activations")

    # Step 1: Gather & Permute (Node 13)
    permute_sig = GatherAndPermuteGradHSignature(
        bm,
        clipped_ref,
        permuted_ref,
        np.uint32(spec.num_modules),
        np.uint32(spec.hidden_dim),
        np.uint32(plan.effective_batch_size),
        np.uint32(plan.grid.num_module_chunks),
        np.uint32(plan.grid.get_tile(0, 0).modules_per_chunk),
        np.uint32(plan.grid.num_class_chunks),
    )
    permute_evt = ex.launch(q, permute_sig, wait_for=deps)

    # Step 2: Reduce the permuted buffer (Node 16)
    reduce_sig = ReduceGradHOverModulesSignature(
        bm,
        arch_consts.get("work_group_size_0", 256),
        spec.scalar_dtype().itemsize,
        permuted_ref,
        summed_ref,
        np.uint32(spec.num_modules),
    )
    reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])

    return summed_ref, reduce_evt


def build_update_subgraph(
    svs: Services,
    param_space: ParameterSpace,
    step: int,
    summed_grads: Dict[str, BufferHandle],
    plan: ExecutionPlan,
    deps: List[cl.Event],
) -> cl.Event:
    """
    Recipe for the final, batch-wide update stage (Nodes 21, 24, 25).
    Extracted from the original `UpdatePhaseExecutor`.

    Args:
        svs: The bundle of core system services.
        param_space: The manifest defining all learnable parameters.
        step: The current global training step for Adam bias correction.
        summed_grads: A dictionary mapping parameter names to their summed gradient buffers.
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        deps: A list of `cl.Event` objects to wait for before executing.

    Returns:
        A singe `cl.Event` signaling the completion of the entire update sequence.
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    h_params = plan.hyperparams

    # Step 1 (Node 21): Normalize all summed gradients
    norm_events, final_grad_handles = [], {}
    for flow in param_space:
        if flow.specialized_reduction or flow.name not in summed_grads:
            continue

        sig = NormalizeGradientsSignature(
            bm,
            summed_grads[flow.name],
            bm.get_handle_by_name(flow.final_grad_buffer_name),
            SCALAR_NP_TYPE(plan.effective_batch_size),
            SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
        norm_events.append(ex.launch(q, sig, wait_for=deps))
        final_grad_handles[flow.name] = sig.final_grad_out_ref

    if "hidden_activations" in summed_grads:
        final_grad_handles["hidden_activations"] = summed_grads["hidden_activations"]

    all_norm_evt = (
        cl.WaitForEvents(norm_events)
        if norm_events
        else cl.UserEvent(q.context).set_status(cl.command_execution_status.COMPLETE)
    )

    # Step 2 (Node 24): Apply Adam Optimizer Update
    beta1_t = SCALAR_NP_TYPE(h_params.adam_beta1**step)
    beta2_t = SCALAR_NP_TYPE(h_params.adam_beta2**step)
    update_events = []
    for flow in param_space:
        # Guard to ensure we only update parameters for which
        # a final gradient was actually produced and is available.
        if flow.specialized_reduction or flow.name not in final_grad_handles:
            continue

        pg = AdamParameterGroup(
            bm.get_handle_by_name(flow.param_buffer_name),
            final_grad_handles[flow.name],
            bm.get_handle_by_name(flow.m1_buffer_name),
            bm.get_handle_by_name(flow.m2_buffer_name),
        )
        adam_sig = AdamUpdateSignature(
            bm,
            pg,
            SCALAR_NP_TYPE(h_params.learning_rate),
            SCALAR_NP_TYPE(h_params.adam_beta1),
            SCALAR_NP_TYPE(h_params.adam_beta2),
            SCALAR_NP_TYPE(h_params.adam_epsilon),
            beta1_t,
            beta2_t,
        )
        update_events.append(ex.launch(q, adam_sig, wait_for=[all_norm_evt]))

    all_updates_evt = cl.WaitForEvents(update_events)

    # Step 3 (Node 25): Final Clamping
    clamp_sig = ClampTemperaturesSignature(
        bm, bm.get_handle_by_name("temperatures"), SCALAR_NP_TYPE(h_params.temp_min), SCALAR_NP_TYPE(h_params.temp_max)
    )
    clamp_evt = ex.launch(q, clamp_sig, wait_for=[all_updates_evt])

    return clamp_evt
