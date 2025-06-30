# graph_recipes.py

"""
A Toolbox of Canonical Sub-Graph Recipes.

This module provides a set of pure, stateless functions
that encapsulate the fundamental "recipes" of the training algorithm. Each function
takes all necessary context and dependencies, and orchestrates a sequence of
kernel launches on the device command queue, returning the final completion event.
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


def build_forward_module_path(
    svs: Services, tile: WorkTile, plan: ExecutionPlan, h_ref: BufferHandle, h_ready_evt: cl.Event
) -> Tuple[BufferHandle, cl.Event]:
    """
    (NEW) Recipe for the forward-pass part of the module path (Nodes 5-7).
    Its sole purpose is to produce the final `logits` and the intermediate
    `partial_probs` buffers required by the backpropagation stage.

    This recipe is ALWAYS executed, regardless of the adaptation strategy, as
    the backward pass always needs the `partial_probs` as input.

    Args:
        svs: The bundle of core system services.
        tile: The `WorkTile` defining this specific unit of work.
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        h_ref: The handle to the `hidden_activations` buffer (cached or recomputed).
        h_ready_evt: The event signaling that the `hidden_activations` buffer is ready.

    Returns:
        A tuple of (handle_to_partial_probs, completion_event_for_this_recipe).
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    spec = svs.model_spec

    # 1. Launch Logits Rendering (Node 5)
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

    # 2. Launch Loss & Probabilities (Nodes 6 or 7) - Dynamically selected
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

    # Return the handle to the probabilities buffer and the final event for this chain.
    return loss_sig.prob_out_ref, loss_evt


# =========================================================================
# === Recipes for Backpropagation & Specialized Pipelines               ===
# =========================================================================


def build_backward_module_path(
    svs: Services,
    tile: WorkTile,
    plan: ExecutionPlan,
    h_ref: BufferHandle,
    prob_ref: BufferHandle,
    prior_deps: List[cl.Event],
) -> cl.Event:
    """
    (NEW) Recipe for the backward-pass part of the module path (Nodes 8-11).
    This computes and clips all module-path partial gradients.

    This recipe should ONLY be executed when the plan's strategy is "CACHE". In
    "RECOMPUTE" mode, this work is subsumed by the Grad_H streaming pipeline.

    Args:
        svs: The bundle of core system services.
        tile: The `WorkTile` object defining this specific unit of work.
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        h_ref: The handle to the `hidden_activations` buffer.
        prob_ref: The handle to the `partial_probs` buffer produced by the forward path.
        prior_deps: A list of events to wait for (must include the forward path's event).

    Returns:
        The final `cl.Event` that signals the completion of clipping for this tile.
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    spec, h_params = svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts

    # 1. Launch Raw Partial Gradient Kernels (Nodes 8, 9, 10) - Dynamically
    wgs0 = arch_consts.get("work_group_size_0", 256)
    scalar_bytes = spec.scalar_dtype().itemsize
    padded_class_dim = spec.padded_class_dim

    if plan.problem_type == "CCE":
        grad_mod_sig = CalculateModuleParamGradsCceSignature(
            buffer_mgr=bm,
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            h_ref=h_ref,
            prob_ref=prob_ref,
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            gw_out_ref=bm.get_handle_by_name("partial_grad_module_weights"),
            gb_out_ref=bm.get_handle_by_name("partial_grad_module_biases"),
            tile=tile,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(plan.effective_batch_size),
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
            padded_total_output_class_count=np.uint32(padded_class_dim),
            total_modules_count=np.uint32(spec.num_modules),
        )
        grad_h_sig = BackpropErrorToHiddenChunkCceSignature(
            buffer_mgr=bm,
            prob_ref=prob_ref,
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_mod_ref=bm.get_handle_by_name("module_weights"),
            gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
            tile=tile,
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
        )
        grad_t_sig = CalculateChunkTempGradientsCceSignature(
            buffer_mgr=bm,
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            logit_ref=bm.get_handle_by_name("logits"),  # Logits were created in the forward pass
            prob_ref=prob_ref,
            targets_cce_ref=bm.get_handle_by_name("targets_cce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            gt_out_ref=bm.get_handle_by_name("partial_grad_temps"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )
    else:  # BCE
        grad_mod_sig = CalculateModuleParamGradsBceSignature(
            buffer_mgr=bm,
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            h_ref=h_ref,
            prob_ref=prob_ref,
            targets_bce_ref=bm.get_handle_by_name("targets_bce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            gw_out_ref=bm.get_handle_by_name("partial_grad_module_weights"),
            gb_out_ref=bm.get_handle_by_name("partial_grad_module_biases"),
            tile=tile,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(plan.effective_batch_size),
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
            padded_total_output_class_count=np.uint32(padded_class_dim),
            total_modules_count=np.uint32(spec.num_modules),
        )
        grad_h_sig = BackpropErrorToHiddenChunkBceSignature(
            buffer_mgr=bm,
            prob_ref=prob_ref,
            targets_bce_ref=bm.get_handle_by_name("targets_bce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_mod_ref=bm.get_handle_by_name("module_weights"),
            gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
            tile=tile,
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
        )
        grad_t_sig = CalculateChunkTempGradientsBceSignature(
            buffer_mgr=bm,
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            logit_ref=bm.get_handle_by_name("logits"),
            prob_ref=prob_ref,
            targets_bce_ref=bm.get_handle_by_name("targets_bce"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            temp_ref=bm.get_handle_by_name("temperatures"),
            gt_out_ref=bm.get_handle_by_name("partial_grad_temps"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes),
        )

    grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=prior_deps)
    grad_h_evt = ex.launch(q, grad_h_sig, wait_for=prior_deps)
    grad_t_evt = ex.launch(q, grad_t_sig, wait_for=prior_deps)

    # 2. Launch Gradient Clipping (Node 11) - Dynamically selected based on the plan
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
    return clip_event


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
            buffer_mgr=bm,
            work_group_size_1=arch_consts.get("work_group_size_1", 16),
            scalar_size_bytes=spec.scalar_dtype().itemsize,
            input_ref=bm.get_handle_by_name("input"),
            h_ref=h_ref,
            grad_h_ref=grad_h_ref,
            mask_ref=bm.get_handle_by_name("sample_mask"),
            partial_gsw_out_ref=gsw_scratch_ref,
            batch_chunk_offset=np.uint32(batch_offset),
            batch_chunk_count=np.uint32(items_in_chunk),
            batch_chunk_index=np.uint32(i),
            num_batch_chunks_count=np.uint32(num_batch_chunks),
        )
        gsw_evt = ex.launch(q, gsw_sig, wait_for=deps_for_all_chunks)
        gsb_sig = BackpropSharedBiasesChunkSignature(
            buffer_mgr=bm,
            work_group_size_0=arch_consts.get("work_group_size_0", 256),
            scalar_size_bytes=spec.scalar_dtype().itemsize,
            h_ref=h_ref,
            grad_h_ref=grad_h_ref,
            mask_ref=bm.get_handle_by_name("sample_mask"),
            partial_gsb_out_ref=gsb_scratch_ref,
            batch_chunk_offset=np.uint32(batch_offset),
            batch_chunk_count=np.uint32(items_in_chunk),
            batch_chunk_index=np.uint32(i),
            num_batch_chunks_count=np.uint32(num_batch_chunks),
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
            buffer_mgr=bm,
            work_group_size_0=arch_consts.get("work_group_size_0", 256),
            scalar_size_bytes=spec.scalar_dtype().itemsize,
            handles=shared_grad_handles,
            max_norm_global=SCALAR_NP_TYPE(h_params.max_grad_norm),
            epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
            dest_weights_write_offset_elements=np.uint32(i * np.prod(gsw_chunk_shape)),
            dest_biases_write_offset_elements=np.uint32(i * np.prod(gsb_chunk_shape)),
            num_batch_chunks=np.uint32(num_batch_chunks),
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
    Recipe for "Model A: Accumulate via Recompute".

    Implements the full sequence within its loop: for each
    tile, it recomputes hidden activations, then computes ALL required partial
    gradients (Module Weights, Biases, Temps, AND Hidden Activations), before
    calling the holistic clipping kernel (Node 11). This fulfills the contract
    that clipping is performed on the complete gradient vector for an item.

    This function represents the complete, self-contained pipeline for producing
    the `summed_grad_hidden_activations` under memory-constrained conditions.

    Returns:
        A tuple of (final_summed_grad_h_handle, final_completion_event).
    """
    q, ex, bm = svs.q, svs.ex, svs.q
    spec, h_params = svs.model_spec, plan.hyperparams
    arch_consts, grid = svs.arch_consts, plan.grid
    scalar_bytes = spec.scalar_dtype().itemsize
    padded_class_dim = spec.padded_class_dim
    wgs0 = arch_consts.get("work_group_size_0", 256)

    # 1. Acquire Monolithic Collection Buffers & Sized Transient Scratch Buffers
    # -- Final Monolithic Destinations (will be populated iteratively)
    clipped_gw_out_ref = bm.get_handle_by_name("clipped_partial_grad_module_weights")
    clipped_gb_out_ref = bm.get_handle_by_name("clipped_partial_grad_module_biases")
    clipped_gt_out_ref = bm.get_handle_by_name("clipped_partial_grad_temps")
    clipped_gh_out_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")

    # -- Sizing Info (get physical spec from collection buffers)
    h_full_shape, _ = bm.get_spec(bm.get_handle_by_name("hidden_activations"))
    gw_full_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_module_weights"))
    gb_full_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_module_biases"))
    gt_full_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_temps"))
    gh_full_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_hidden_activations"))

    # -- Scratch Buffers (sized for a single tile's worth of data)
    h_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_full_shape) * scalar_bytes))
    h_mask_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_full_shape) * scalar_bytes))
    raw_gw_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gw_full_shape[1:]) * scalar_bytes))
    raw_gb_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gb_full_shape[1:]) * scalar_bytes))
    raw_gt_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gt_full_shape[1:]) * scalar_bytes))
    raw_gh_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gh_full_shape[1:]) * scalar_bytes))

    all_clip_events = []

    # 2. Loop over module/class tiles to accumulate clipped partials
    for tile in grid:
        # Step 2a: Recompute hidden activations for the entire batch into a scratch buffer
        fwd_pass_sig = ForwardPassSignature(
            buffer_mgr=bm,
            simd_width=spec.simd_width,
            local_mem_bank_padding=1,
            scalar_size_bytes=scalar_bytes,
            in_ref=bm.get_handle_by_name("input"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_ref=bm.get_handle_by_name("shared_weights"),
            b_ref=bm.get_handle_by_name("shared_biases"),
            h_out_ref=h_scratch_ref,
            h_mask_out_ref=h_mask_scratch_ref,
            batch_chunk_offset=np.uint32(0),
            batch_chunk_count=np.uint32(plan.effective_batch_size),
        )
        h_ready_evt = ex.launch(q, fwd_pass_sig, wait_for=deps)
        grad_calc_deps = [h_ready_evt] + deps

        # Step 2b: Compute ALL partial gradients using the recomputed hidden state
        if plan.problem_type == "CCE":
            grad_mod_sig = CalculateModuleParamGradsCceSignature(
                buffer_mgr=bm,
                work_group_size_0=wgs0,
                scalar_size_bytes=scalar_bytes,
                h_ref=h_scratch_ref,
                prob_ref=bm.get_handle_by_name("partial_probs"),
                targets_cce_ref=bm.get_handle_by_name("targets_cce"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                gw_out_ref=raw_gw_scratch_ref,
                gb_out_ref=raw_gb_scratch_ref,
                tile=tile,
                batch_chunk_offset=np.uint32(0),
                batch_chunk_count=np.uint32(plan.effective_batch_size),
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
                padded_total_output_class_count=np.uint32(padded_class_dim),
                total_modules_count=np.uint32(spec.num_modules),
            )
            grad_t_sig = CalculateChunkTempGradientsCceSignature(
                buffer_mgr=bm,
                work_group_size_0=wgs0,
                scalar_size_bytes=scalar_bytes,
                logit_ref=bm.get_handle_by_name("logits"),
                prob_ref=bm.get_handle_by_name("partial_probs"),
                targets_cce_ref=bm.get_handle_by_name("targets_cce"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                temp_ref=bm.get_handle_by_name("temperatures"),
                gt_out_ref=raw_gt_scratch_ref,
                tile=tile,
                total_output_class_count=np.uint32(spec.output_classes),
            )
            grad_h_sig = BackpropErrorToHiddenChunkCceSignature(
                buffer_mgr=bm,
                prob_ref=bm.get_handle_by_name("partial_probs"),
                targets_cce_ref=bm.get_handle_by_name("targets_cce"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                w_mod_ref=bm.get_handle_by_name("module_weights"),
                gh_out_ref=raw_gh_scratch_ref,
                tile=tile,
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
            )
        else:
            raise NotImplementedError("BCE path for GradH streaming pipeline not yet implemented.")

        grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=grad_calc_deps)
        grad_t_evt = ex.launch(q, grad_t_sig, wait_for=grad_calc_deps)
        grad_h_evt = ex.launch(q, grad_h_sig, wait_for=grad_calc_deps)

        all_raw_grads_ready = [grad_mod_evt, grad_t_evt, grad_h_evt]

        # Step 2c: Construct a valid handle group for the clipping kernel
        grad_handles = GradientHandles(
            grad_weights_module=raw_gw_scratch_ref,
            grad_biases_module=raw_gb_scratch_ref,
            grad_temps=raw_gt_scratch_ref,
            grad_hidden_activations_aos=raw_gh_scratch_ref,
            clipped_grad_weights_module=clipped_gw_out_ref,
            clipped_grad_biases_module=clipped_gb_out_ref,
            clipped_grad_temps=clipped_gt_out_ref,
            clipped_grad_hidden_activations_aos=clipped_gh_out_ref,
        )

        # Step 2d: Use the holistic clipping kernel (Node 11)
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
        else:
            raise NotImplementedError("Per-item norm clipping for GradH streaming not yet implemented.")

        clip_evt = ex.launch(q, clip_sig, wait_for=all_raw_grads_ready)
        all_clip_events.append(clip_evt)

    # 3. Release all transient scratch buffers after the loop completes
    bm.release_transient_buffer(h_scratch_ref)
    bm.release_transient_buffer(h_mask_scratch_ref)
    bm.release_transient_buffer(raw_gw_scratch_ref)
    bm.release_transient_buffer(raw_gb_scratch_ref)
    bm.release_transient_buffer(raw_gt_scratch_ref)
    bm.release_transient_buffer(raw_gh_scratch_ref)

    all_clips_done = cl.WaitForEvents(all_clip_events)

    # 4. Permute the now-filled monolithic buffer (Node 13)
    permuted_soa_ref = bm.get_handle_by_name("permuted_grad_h")
    permute_sig = GatherAndPermuteGradHSignature(
        buffer_mgr=bm,
        clipped_partials_aos_ref=clipped_gh_out_ref,
        permuted_soa_out_ref=permuted_soa_ref,
        total_modules_count=np.uint32(spec.num_modules),
        hidden_count=np.uint32(spec.hidden_dim),
        total_batch_count=np.uint32(plan.effective_batch_size),
        num_module_chunks_count=np.uint32(plan.grid.num_module_chunks),
        modules_per_chunk_count=np.uint32(plan.grid.get_tile(0, 0).modules_per_chunk),
        num_class_chunks_count=np.uint32(plan.grid.num_class_chunks),
    )
    permute_evt = ex.launch(q, permute_sig, wait_for=[all_clips_done])

    # 5. Reduce the permuted buffer (Node 16)
    final_summed_grad_h_ref = bm.get_handle_by_name("summed_grad_hidden_activations")
    reduce_sig = ReduceGradHOverModulesSignature(
        buffer_mgr=bm,
        work_group_size_0=wgs0,
        scalar_size_bytes=scalar_bytes,
        permuted_soa_in_ref=permuted_soa_ref,
        final_grad_h_out_ref=final_summed_grad_h_ref,
        total_modules_count=np.uint32(spec.num_modules),
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
    agg_mgr = AggregationManager(ex=ex, bm=bm, arch_consts=svs.arch_consts)

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
        buffer_mgr=bm,
        clipped_partials_aos_ref=clipped_ref,
        permuted_soa_out_ref=permuted_ref,
        total_modules_count=np.uint32(spec.num_modules),
        hidden_count=np.uint32(spec.hidden_dim),
        total_batch_count=np.uint32(plan.effective_batch_size),
        num_module_chunks_count=np.uint32(plan.grid.num_module_chunks),
        modules_per_chunk_count=np.uint32(plan.grid.get_tile(0, 0).modules_per_chunk),
        num_class_chunks_count=np.uint32(plan.grid.num_class_chunks),
    )
    permute_evt = ex.launch(q, permute_sig, wait_for=deps)

    # Step 2: Reduce the permuted buffer (Node 16)
    reduce_sig = ReduceGradHOverModulesSignature(
        buffer_mgr=bm,
        work_group_size_0=arch_consts.get("work_group_size_0", 256),
        scalar_size_bytes=spec.scalar_dtype().itemsize,
        permuted_soa_in_ref=permuted_ref,
        final_grad_h_out_ref=summed_ref,
        total_modules_count=np.uint32(spec.num_modules),
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
            buffer_mgr=bm,
            summed_grad_ref=summed_grads[flow.name],
            final_grad_out_ref=bm.get_handle_by_name(flow.final_grad_buffer_name),
            effective_batch_size=SCALAR_NP_TYPE(plan.effective_batch_size),
            epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
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
            param_ref=bm.get_handle_by_name(flow.param_buffer_name),
            grad_ref=final_grad_handles[flow.name],
            m1_state_ref=bm.get_handle_by_name(flow.m1_buffer_name),
            m2_state_ref=bm.get_handle_by_name(flow.m2_buffer_name),
        )
        adam_sig = AdamUpdateSignature(
            buffer_mgr=bm,
            param_group=pg,
            learning_rate=SCALAR_NP_TYPE(h_params.learning_rate),
            beta1=SCALAR_NP_TYPE(h_params.adam_beta1),
            beta2=SCALAR_NP_TYPE(h_params.adam_beta2),
            epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
            beta1_pow_t=beta1_t,
            beta2_pow_t=beta2_t,
        )
        update_events.append(ex.launch(q, adam_sig, wait_for=[all_norm_evt]))

    all_updates_evt = cl.WaitForEvents(update_events)

    # Step 3 (Node 25): Final Clamping
    clamp_sig = ClampTemperaturesSignature(
        buffer_mgr=bm,
        temps_ref=bm.get_handle_by_name("temperatures"),
        min_val=SCALAR_NP_TYPE(h_params.temp_min),
        max_val=SCALAR_NP_TYPE(h_params.temp_max),
    )
    clamp_evt = ex.launch(q, clamp_sig, wait_for=[all_updates_evt])

    return clamp_evt
