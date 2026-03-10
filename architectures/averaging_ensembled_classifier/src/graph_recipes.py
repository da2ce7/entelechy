# graph_recipes.py

"""
A Grimoire of Canonical Sub-Graph Recipes.

Jurisdictional Mandate:
This module provides the definitive, stateless library of computational
"recipes" that constitute the core of the training algorithm. Each function
herein is a pure, self-contained orchestrator for a specific, logical piece of
the system's Directed Acyclic Graph (DAG). Its jurisdiction is exclusively the
sequencing of kernel launches; it owns no state and makes no strategic
decisions.

Architectural Role:
This module is the system's "Workshop." It is where the abstract, declarative
`ExecutionPlan` authored by the `TrainingOrchestrator` is translated into a
concrete sequence of device commands. These recipes are the tactical
fulfillment of the strategic plan, embodying the principles of:
  - Polymorphism: Deferring to `ProblemTypeStrategy` objects to select the
    correct kernel signatures, eliminating conditional logic.
  - Contractual Adherence: Instantiating each `KernelSignature` class with the
    high-level context it requires to fulfill its own contract.
  - Composition: Building complex dataflows (like the Recursive
    Clip-Aggregation Engine) by composing simpler, stateless recipes.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, cast, Callable

import numpy as np
import pyopencl as cl

# --- Foundational Imports from Sibling Modules ---
# WHY: This module is a consumer of the system's core contracts and primitives.
# It requires the vocabulary of the plan, the infrastructure for execution,
# the full library of kernel signatures, and the primitives for describing workloads.
from .execution_plan import ExecutionPlan
from .launcher_infra import (
    Services,
    BufferManager,
    KernelExecutor,
    BufferHandle,
    PingPongManager,
    HostView,
    KernelSignature,
)
from .kernel_signatures import (
    ForwardPassSignature,
    RenderLogitsChunkSignature,
    ComputeProbsLossCceChunkSignature,
    ComputeProbsLossBceChunkSignature,
    CalculateModuleParamGradsCceSignature,
    CalculateModuleParamGradsBceSignature,
    BackpropErrorToHiddenChunkCceSignature,
    BackpropErrorToHiddenChunkBceSignature,
    CalculateChunkTempGradientsCceSignature,
    CalculateChunkTempGradientsBceSignature,
    ClipPartialGradientsGlobalNormSignature,
    ClipPartialGradientsPerItemNormSignature,
    GradientHandles,
    GatherAndPermuteGradHiddenActivationsSignature,
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
    ClipIntermediateGradSignature,
    StabilizeAndReduceGradHiddenActivationsSignature,
    BackpropSharedWeightsChunkSignature,
    BackpropSharedBiasesChunkSignature,
    SharedGradientHandles,
    ClipSharedGradientsChunkSignature,
    NormalizeGradientsSignature,
    AdamParameterGroup,
    AdamUpdateSignature,
    ClampTemperaturesSignature,
)
from .model_spec import ModelSpec
from .parameter_space import ParameterSpace
from .compute_patterns import AggregationManager, ReductionPlan
from .workload_primitives import GatherPrimitive, ContiguousGather, TiledGather, WorkTile, LinearlyChunkedGather


# =========================================================================
# === Section 1: Forward Pass & Diagnostic Aggregation Recipes          ===
# =========================================================================
# The recipes in this section handle the "Act" phase of the computation,
# from the initial input layer through to the final aggregation of
# probabilities and diagnostic loss values.
# =========================================================================


def execute_forward_pass(svs: Services, batch_size: int, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
    """Recipe for the initial shared layer forward pass (Node 4)."""
    bm, ex, q = svs.bm, svs.ex, svs.q

    # The signature is instantiated with high-level context, allowing it
    # to derive its own low-level parameters, fulfilling its contract.
    sig = ForwardPassSignature(
        _buffer_mgr=bm,
        _arch_consts=svs.arch_consts,
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
    """Recipe for the module-level forward pass (Nodes 5-7), producing logits and partial probabilities."""
    bm, ex, q, spec = svs.bm, svs.ex, svs.q, svs.model_spec

    # Node 5: Render Logits.
    logits_sig = RenderLogitsChunkSignature(
        _buffer_mgr=bm,
        _arch_consts=svs.arch_consts,
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

    # Node 6/7: Compute Loss & Probabilities via Polymorphic Delegation.
    # WHY: This single call to the strategy object replaces a complex if/else
    # block, making the recipe agnostic to the specific loss function. This
    # embodies the Open/Closed Principle: the system is extensible without
    # modification of this core logic.
    loss_sig_base = plan.problem_type.get_loss_signature(
        _buffer_mgr=bm,
        _arch_consts=svs.arch_consts,
        logit_ref=logits_sig.logit_out_ref,
        temp_ref=bm.get_handle_by_name("temperatures"),
        mask_ref=bm.get_handle_by_name("sample_mask"),
        prob_out_ref=bm.get_handle_by_name("partial_probs"),
        tile=tile,
        total_output_class_count=np.uint32(spec.output_classes),
        # Pass handles for ALL possible outputs; the specific signature
        # constructor will ignore the ones it doesn't need.
        loss_out_ref=bm.get_handle_by_name("final_loss"),
        partial_loss_out_ref=bm.get_handle_by_name("partial_loss"),
    )
    # WHY: The cast is a pragmatic necessity. The ABC promises a generic
    # `KernelSignature`, but we know the concrete implementation will have a
    # `prob_out_ref`. This cast makes that knowledge explicit to the type checker.
    loss_sig = cast(Union[ComputeProbsLossCceChunkSignature, ComputeProbsLossBceChunkSignature], loss_sig_base)
    loss_evt = ex.launch(q, loss_sig, wait_for=[logits_evt])

    return loss_sig.prob_out_ref, loss_evt


def execute_diagnostic_aggregation(svs: Services, plan: ExecutionPlan, deps: List[cl.Event]) -> Dict[str, cl.Event]:
    """Recipe to aggregate diagnostic results from the Act phase (Node 14)."""
    bm = svs.bm
    completion_events: Dict[str, cl.Event] = {}

    # Aggregate probabilities (a universal requirement).
    partial_probs_ref = bm.get_handle_by_name("partial_probs")
    final_probs_ref = bm.get_handle_by_name("final_probs")
    partial_probs_shape, _ = bm.get_spec(partial_probs_ref)
    elements_per_prob_partial = int(np.prod(partial_probs_shape[1:]))
    prob_gather_prim = TiledGather(scheme=plan.grid, _elements_per_partial=elements_per_prob_partial)

    prob_agg_evt = execute_summation_tree(
        svs=svs,
        plan=plan,
        gather_primitive=prob_gather_prim,
        partial_collection_ref=partial_probs_ref,
        final_dest_handle=final_probs_ref,
        wait_for=deps,
    )
    completion_events["probs"] = prob_agg_evt

    # Conditionally aggregate loss (only for BCE, as CCE is direct-write).
    if plan.problem_type.required_targets_buffer_name == "targets_bce":
        partial_loss_ref = bm.get_handle_by_name("partial_loss")
        final_loss_ref = bm.get_handle_by_name("final_loss")
        partial_loss_shape, _ = bm.get_spec(partial_loss_ref)
        elements_per_loss_partial = int(np.prod(partial_loss_shape[1:]))
        loss_gather_prim = TiledGather(scheme=plan.grid, _elements_per_partial=elements_per_loss_partial)

        loss_agg_evt = execute_summation_tree(
            svs=svs,
            plan=plan,
            gather_primitive=loss_gather_prim,
            partial_collection_ref=partial_loss_ref,
            final_dest_handle=final_loss_ref,
            wait_for=deps,
        )
        completion_events["loss"] = loss_agg_evt
    return completion_events


# =========================================================================
# === Section 2: Backward Pass - Module Gradient & Specialized Paths    ===
# =========================================================================
# This section contains the recipes for the first part of the 'Learn' phase,
# focusing on the module-specific gradients. It includes both the 'CACHE' and
# 'RECOMPUTE' strategies, as well as the specialized reduction path for Grad_H.
# =========================================================================


def build_backward_module_path(
    svs: Services,
    tile: WorkTile,
    plan: ExecutionPlan,
    h_ref: BufferHandle,
    prob_ref: BufferHandle,
    prior_deps: List[cl.Event],
) -> cl.Event:
    """Recipe for the CACHE-based module backward pass (Nodes 8-11), producing clipped partial gradients."""
    bm, ex, q, spec, h_params = svs.bm, svs.ex, svs.q, svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts

    # Nodes 8, 9, 10: Compute raw partial gradients via polymorphic delegation.
    grad_mod_sig_base = plan.problem_type.get_module_grad_signature(
        _buffer_mgr=bm,
        _arch_consts=arch_consts,
        work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
        h_ref=h_ref,
        prob_ref=prob_ref,
        mask_ref=bm.get_handle_by_name("sample_mask"),
        gw_out_ref=bm.get_handle_by_name("partial_grad_module_weights"),
        gb_out_ref=bm.get_handle_by_name("partial_grad_module_biases"),
        tile=tile,
        batch_chunk_offset=np.uint32(0),
        batch_chunk_count=np.uint32(plan.effective_batch_size),
        hidden_count=np.uint32(spec.hidden_dim),
        total_output_class_count=np.uint32(spec.output_classes),
        padded_total_output_class_count=np.uint32(spec.padded_class_dim),
        total_modules_count=np.uint32(spec.num_modules),
    )
    grad_mod_sig = cast(
        Union[CalculateModuleParamGradsCceSignature, CalculateModuleParamGradsBceSignature], grad_mod_sig_base
    )

    grad_h_sig_base = plan.problem_type.get_hidden_grad_signature(
        _buffer_mgr=bm,
        _arch_consts=arch_consts,
        prob_ref=prob_ref,
        mask_ref=bm.get_handle_by_name("sample_mask"),
        w_mod_ref=bm.get_handle_by_name("module_weights"),
        gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
        tile=tile,
        hidden_count=np.uint32(spec.hidden_dim),
        total_output_class_count=np.uint32(spec.output_classes),
    )
    grad_h_sig = cast(
        Union[BackpropErrorToHiddenChunkCceSignature, BackpropErrorToHiddenChunkBceSignature], grad_h_sig_base
    )

    grad_t_sig_base = plan.problem_type.get_temp_grad_signature(
        _buffer_mgr=bm,
        _arch_consts=arch_consts,
        work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
        logit_ref=bm.get_handle_by_name("logits"),
        prob_ref=prob_ref,
        mask_ref=bm.get_handle_by_name("sample_mask"),
        temp_ref=bm.get_handle_by_name("temperatures"),
        gt_out_ref=bm.get_handle_by_name("partial_grad_temps"),
        tile=tile,
        total_output_class_count=np.uint32(spec.output_classes),
    )
    grad_t_sig = cast(
        Union[CalculateChunkTempGradientsCceSignature, CalculateChunkTempGradientsBceSignature], grad_t_sig_base
    )

    grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=prior_deps)
    grad_h_evt = ex.launch(q, grad_h_sig, wait_for=prior_deps)
    grad_t_evt = ex.launch(q, grad_t_sig, wait_for=prior_deps)

    # Node 11: The foundational stability primitive, clipping the raw gradients.
    # WHY: A clipping strategy (e.g., GLOBAL vs PER_ITEM) is chosen at the
    # highest level and encoded in the ExecutionPlan. This recipe simply executes
    # that strategy by selecting the appropriate signature class, again freeing
    # this tactical layer from making strategic decisions.
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

    clip_sig: KernelSignature
    if plan.clipping_strategy == "GLOBAL":
        clip_sig = ClipPartialGradientsGlobalNormSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            handles=grad_handles,
            tile=tile,
            clipping_threshold_global=spec.SCALAR_NP_TYPE(plan.stabilization_policy.get_leaf_safety_threshold()),
            epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
    else:  # 'PER_ITEM'
        clip_sig = ClipPartialGradientsPerItemNormSignature(
            _buffer_mgr=bm,
            _arch_consts=arch_consts,
            handles=grad_handles,
            tile=tile,
            clipping_threshold_per_item_ref=bm.get_handle_by_name("clipping_threshold_per_item"),
            epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
        )

    clip_event = ex.launch(q, clip_sig, wait_for=[grad_mod_evt, grad_h_evt, grad_t_evt])
    return clip_event


def build_streaming_module_grad_path(svs: Services, plan: ExecutionPlan, deps: List[cl.Event]) -> Dict[str, cl.Event]:
    """Recipe for the RECOMPUTE-based "Accumulate via Recompute" model (Nodes 8, 9, 10, 11)."""
    q, ex, bm, spec, h_params, grid = svs.q, svs.ex, svs.bm, svs.model_spec, plan.hyperparams, plan.grid
    arch_consts = svs.arch_consts
    scalar_bytes = spec.SCALAR_NP_TYPE().itemsize

    # Acquire handles for final COLLECTION and transient SCRATCH buffers.
    clipped_gw_out_ref = bm.get_handle_by_name("clipped_partial_grad_module_weights")
    clipped_gb_out_ref = bm.get_handle_by_name("clipped_partial_grad_module_biases")
    clipped_gt_out_ref = bm.get_handle_by_name("clipped_partial_grad_temps")
    clipped_gh_out_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")

    h_full_shape, _ = bm.get_spec(bm.get_handle_by_name("hidden_activations"))
    gw_coll_shape, _ = bm.get_spec(clipped_gw_out_ref)
    gb_coll_shape, _ = bm.get_spec(clipped_gb_out_ref)
    gt_coll_shape, _ = bm.get_spec(clipped_gt_out_ref)
    gh_coll_shape, _ = bm.get_spec(clipped_gh_out_ref)

    h_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_full_shape) * scalar_bytes))
    h_mask_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_full_shape) * scalar_bytes))
    raw_gw_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gw_coll_shape[1:]) * scalar_bytes))
    raw_gb_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gb_coll_shape[1:]) * scalar_bytes))
    raw_gt_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gt_coll_shape[1:]) * scalar_bytes))
    raw_gh_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gh_coll_shape[1:]) * scalar_bytes))

    scratch_handles: List[BufferHandle] = [
        h_scratch_ref,
        h_mask_scratch_ref,
        raw_gw_scratch_ref,
        raw_gb_scratch_ref,
        raw_gt_scratch_ref,
        raw_gh_scratch_ref,
    ]

    try:
        all_clip_events: List[cl.Event] = []
        for tile in grid:
            # Recompute hidden activations into scratch space.
            fwd_pass_sig = ForwardPassSignature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
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

            # Compute raw partial gradients into scratch buffers.
            grad_mod_sig = plan.problem_type.get_module_grad_signature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
                work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
                h_ref=h_scratch_ref,
                prob_ref=bm.get_handle_by_name("partial_probs"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                gw_out_ref=raw_gw_scratch_ref,
                gb_out_ref=raw_gb_scratch_ref,
                tile=tile,
                batch_chunk_offset=np.uint32(0),
                batch_chunk_count=np.uint32(plan.effective_batch_size),
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
                padded_total_output_class_count=np.uint32(spec.padded_class_dim),
                total_modules_count=np.uint32(spec.num_modules),
            )
            grad_h_sig = plan.problem_type.get_hidden_grad_signature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
                prob_ref=bm.get_handle_by_name("partial_probs"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                w_mod_ref=bm.get_handle_by_name("module_weights"),
                gh_out_ref=raw_gh_scratch_ref,
                tile=tile,
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
            )
            grad_t_sig = plan.problem_type.get_temp_grad_signature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
                work_group_size_0=arch_consts.optimal_workgroup_size_1d_reduction,
                logit_ref=bm.get_handle_by_name("logits"),
                prob_ref=bm.get_handle_by_name("partial_probs"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                temp_ref=bm.get_handle_by_name("temperatures"),
                gt_out_ref=raw_gt_scratch_ref,
                tile=tile,
                total_output_class_count=np.uint32(spec.output_classes),
            )

            grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=grad_calc_deps)
            grad_t_evt = ex.launch(q, grad_t_sig, wait_for=grad_calc_deps)
            grad_h_evt = ex.launch(q, grad_h_sig, wait_for=grad_calc_deps)
            all_raw_grads_ready = [grad_mod_evt, grad_t_evt, grad_h_evt]

            # Clip from scratch into final collection buffers.
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

            clip_sig: KernelSignature
            if plan.clipping_strategy == "GLOBAL":
                clip_sig = ClipPartialGradientsGlobalNormSignature(
                    _buffer_mgr=bm,
                    _arch_consts=arch_consts,
                    handles=grad_handles,
                    tile=tile,
                    clipping_threshold_global=spec.SCALAR_NP_TYPE(
                        plan.stabilization_policy.get_leaf_safety_threshold()
                    ),
                    epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
                )
            else:
                clip_sig = ClipPartialGradientsPerItemNormSignature(
                    _buffer_mgr=bm,
                    _arch_consts=arch_consts,
                    handles=grad_handles,
                    tile=tile,
                    clipping_threshold_per_item_ref=bm.get_handle_by_name("clipping_threshold_per_item"),
                    epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
                )
            clip_event = ex.launch(q, clip_sig, wait_for=all_raw_grads_ready)
            all_clip_events.append(clip_event)

        # WHY: This is the dialect-corrected synchronization point. A barrier is
        # the correct tool to join multiple parallel streams (the clip events for
        # each tile) into a single event that signals the entire phase is complete.
        all_clips_done_evt = (
            cl.enqueue_barrier(q, wait_for=all_clip_events) if all_clip_events else cl.enqueue_marker(q, wait_for=deps)
        )
        return {
            "clipped_partial_grad_module_weights": all_clips_done_evt,
            "clipped_partial_grad_module_biases": all_clips_done_evt,
            "clipped_partial_grad_temps": all_clips_done_evt,
            "clipped_partial_grad_hidden_activations": all_clips_done_evt,
        }
    finally:
        # WHY: A contractually obligated cleanup step. Using a `finally` block
        # guarantees that we release all transient memory, even if an error
        # occurs, preventing VRAM leaks and upholding architectural robustness.
        for handle in scratch_handles:
            bm.release_transient_buffer(handle)


def build_final_grad_h_reduction_path(
    svs: Services, plan: ExecutionPlan, wait_for: List[cl.Event]
) -> Tuple[BufferHandle, cl.Event]:
    """Recipe for permuting and reducing `Grad_H` (Nodes 13 & 16), the culmination of the Item Synchronization path."""
    q, ex, bm, spec, h_params = svs.q, svs.ex, svs.bm, svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts

    clipped_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")
    permuted_ref = bm.get_handle_by_name("permuted_grad_h")
    summed_ref = bm.get_handle_by_name("summed_grad_hidden_activations")

    # Node 13: Gather scattered partials and permute to reduction-ready SoA layout.
    permute_sig = GatherAndPermuteGradHiddenActivationsSignature(
        _buffer_mgr=bm,
        _arch_consts=arch_consts,
        clipped_partials_aos_ref=clipped_ref,
        permuted_soa_out_ref=permuted_ref,
        total_modules_count=np.uint32(spec.num_modules),
        hidden_count=np.uint32(spec.hidden_dim),
        total_batch_count=np.uint32(plan.effective_batch_size),
        num_module_chunks_count=np.uint32(plan.grid.num_module_chunks),
        modules_per_chunk_count=np.uint32(plan.grid.get_tile(0, 0).modules_per_chunk),
        num_class_chunks_count=np.uint32(plan.grid.num_class_chunks),
    )
    permute_evt = ex.launch(q, permute_sig, wait_for=wait_for)

    # Node 16: Reduce the permuted buffer using the specialized, policy-aware kernel.
    policy_k = plan.stabilization_policy.get_specialized_reduction_policy_k(
        user_policy_k=h_params.reduction_k_grad_h,
        hardware_max_fan_in=arch_consts.optimal_workgroup_size_1d_reduction,
    )
    reduce_sig = StabilizeAndReduceGradHiddenActivationsSignature(
        _buffer_mgr=bm,
        _arch_consts=arch_consts,
        permuted_soa_in_ref=permuted_ref,
        final_grad_h_out_ref=summed_ref,
        fp_max=spec.SCALAR_NP_TYPE(plan.stabilization_policy.fp_format_max),
        policy_t_algorithmic=spec.SCALAR_NP_TYPE(h_params.stabilization.max_grad_norm),
        policy_lambda=spec.SCALAR_NP_TYPE(h_params.stabilization.lambda_),
        policy_max_k=np.uint32(policy_k),
        epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
        total_batch_count=np.uint32(plan.effective_batch_size),
        padded_hidden_count=np.uint32(spec.padded_hidden_dim),
        total_modules_count=np.uint32(spec.num_modules),
        padded_total_modules_count=np.uint32(spec.padded_module_dim),
    )
    reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])

    return summed_ref, reduce_evt


# =========================================================================
# === Section 3: Backward Pass - Shared Layer                           ===
# =========================================================================
# This section contains the recipes for the second part of the 'Learn' phase.
# It includes the streaming backpropagation for the shared layer.
# =========================================================================


def build_shared_backprop_subgraph(svs: Services, plan: ExecutionPlan, deps: List[cl.Event]) -> cl.Event:
    """Recipe for the streaming shared layer backpropagation (Nodes 17-19)."""
    bm, ex, q, spec, h_params = svs.bm, svs.ex, svs.q, svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts
    num_batch_chunks = plan.shared_backprop_stream_chunks
    final_chunk_events: List[cl.Event] = []

    # Resolve batch-wide dependencies once, outside the streaming loop.
    grad_h_provider = plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
    grad_h_ref, grad_h_ready_evt = grad_h_provider.resolve(q, ex, wait_for=deps)
    h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
    h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)
    deps_for_all_chunks = [grad_h_ready_evt, h_ready_evt]

    # Acquire transient scratch buffers for raw partial gradients.
    w_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_weights"))
    b_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_biases"))
    gsw_chunk_shape = (w_shape[0], w_shape[1])
    gsb_chunk_shape = (b_shape[0],)
    gsw_scratch_ref = bm.acquire_transient_buffer(
        int(np.prod(gsw_chunk_shape) * spec.SCALAR_NP_TYPE().itemsize),
        shape=gsw_chunk_shape,
        dtype=spec.SCALAR_NP_TYPE,
    )
    gsb_scratch_ref = bm.acquire_transient_buffer(
        int(np.prod(gsb_chunk_shape) * spec.SCALAR_NP_TYPE().itemsize),
        shape=gsb_chunk_shape,
        dtype=spec.SCALAR_NP_TYPE,
    )

    try:
        # Main streaming loop over batch chunks.
        batch_size = plan.effective_batch_size
        chunk_size = (batch_size + num_batch_chunks - 1) // num_batch_chunks
        for i in range(num_batch_chunks):
            batch_offset = i * chunk_size
            items_in_chunk = min(chunk_size, batch_size - batch_offset)
            if items_in_chunk <= 0:
                continue

            # Nodes 17 & 18: Compute raw partials into SCRATCH buffers.
            # WHY: batch_chunk_index=0 because the kernel uses it to compute a
            # write offset into its output buffer. Since the output is a per-chunk
            # SCRATCH buffer (not the full collection), the offset must be zero.
            # The clip kernel (Node 19) handles the placement into the collection.
            gsw_sig = BackpropSharedWeightsChunkSignature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
                input_ref=bm.get_handle_by_name("input"),
                h_ref=h_ref,
                grad_h_ref=grad_h_ref,
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsw_out_ref=gsw_scratch_ref,
                batch_chunk_offset=np.uint32(batch_offset),
                batch_chunk_count=np.uint32(items_in_chunk),
                batch_chunk_index=np.uint32(0),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            )
            gsw_evt = ex.launch(q, gsw_sig, wait_for=deps_for_all_chunks)
            gsb_sig = BackpropSharedBiasesChunkSignature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
                h_ref=h_ref,
                grad_h_ref=grad_h_ref,
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsb_out_ref=gsb_scratch_ref,
                batch_chunk_offset=np.uint32(batch_offset),
                batch_chunk_count=np.uint32(items_in_chunk),
                batch_chunk_index=np.uint32(0),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            )
            gsb_evt = ex.launch(q, gsb_sig, wait_for=deps_for_all_chunks)

            # Node 19: Clip from scratch into the final COLLECTION buffers.
            shared_grad_handles = SharedGradientHandles(
                grad_weights_shared_chunk=gsw_scratch_ref,
                grad_biases_shared_chunk=gsb_scratch_ref,
                clipped_grad_weights_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_weights"),
                clipped_grad_biases_shared_collection=bm.get_handle_by_name("clipped_partial_grad_shared_biases"),
            )
            clip_sig = ClipSharedGradientsChunkSignature(
                _buffer_mgr=bm,
                _arch_consts=arch_consts,
                handles=shared_grad_handles,
                clipping_threshold_global=spec.SCALAR_NP_TYPE(plan.stabilization_policy.get_leaf_safety_threshold()),
                epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
                dest_weights_write_offset_elements=np.uint32(i * np.prod(gsw_chunk_shape)),
                dest_biases_write_offset_elements=np.uint32(i * np.prod(gsb_chunk_shape)),
                num_batch_chunks=np.uint32(num_batch_chunks),
            )
            clip_evt = ex.launch(q, clip_sig, wait_for=[gsw_evt, gsb_evt])
            final_chunk_events.append(clip_evt)

    finally:
        bm.release_transient_buffer(gsw_scratch_ref)
        bm.release_transient_buffer(gsb_scratch_ref)

    if not final_chunk_events:
        return cl.enqueue_marker(q, wait_for=deps)
    return cl.enqueue_barrier(q, wait_for=final_chunk_events)


# =========================================================================
# === Section 4: The Core Reduction Engine & Its Public Recipes         ===
# =========================================================================
# This section contains the unified reduction pipeline and the two distinct,
# intention-revealing public recipes that use it.
# =========================================================================


def _execute_reduction_pipeline(
    svs: "Services",
    plan: "ExecutionPlan",
    gather_primitive: "GatherPrimitive",
    partial_collection_ref: "BufferHandle",
    final_dest_handle: "BufferHandle",
    # The contract for the pluggable stage processor function.
    stage_processor: Callable[[BufferHandle, BufferHandle, int, BufferHandle, int, List[cl.Event]], cl.Event],
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """
    (Internal) The core reduction pipeline. Manages the loop, resources, and
    data flow, but delegates the per-stage computational logic to the provided
    'stage_processor' function.
    """
    wait_for = wait_for or []
    q, bm, spec = svs.q, svs.bm, svs.model_spec
    n = gather_primitive.num_partials
    reduction_plan = plan.reduction_plan

    # --- Edge Case Handling ---
    if n <= 0:
        user_event = cl.UserEvent(q.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

    elements_per_partial = gather_primitive.elements_per_partial
    scalar_byte_size = spec.SCALAR_NP_TYPE().itemsize
    partial_byte_size = elements_per_partial * scalar_byte_size

    # WHY: An elegant optimization. If there is only one partial to "reduce,"
    # the correct action is a direct memory copy, not a redundant kernel call.
    # This avoids the overhead of launching a kernel for a no-op aggregation.
    if n == 1:
        initial_offsets = gather_primitive.get_offsets()
        src_offset_bytes = int(initial_offsets[0] * scalar_byte_size)
        # WHY: This is the dialect-corrected function for all copies, including
        # device-to-device. It's a top-level function in the `pyopencl` module.
        return cl.enqueue_copy(
            q,
            dest=bm.get_cl_buffer(final_dest_handle),
            src=bm.get_cl_buffer(partial_collection_ref),
            byte_count=partial_byte_size,
            src_offset=src_offset_bytes,
            dst_offset=0,
            wait_for=wait_for,
        )

    # --- Resource & Loop Management (centralized logic) ---
    ppm = PingPongManager()
    transient_handles: List[BufferHandle] = []

    def _create_offset_list(offsets_host: np.ndarray, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        offset_list_ref = bm.acquire_transient_buffer(offsets_host.nbytes)
        transient_handles.append(offset_list_ref)
        evt = cl.enqueue_copy(q, bm.get_cl_buffer(offset_list_ref), offsets_host, wait_for=deps)
        return offset_list_ref, evt

    try:
        # Determine the reduction plan (K and num_stages) based on whether stabilization is active.
        if plan.stabilization_policy.t_algorithmic > 0:
            safe_k, num_stages = plan.stabilization_policy.plan_uniform_reduction_tree(
                num_partials=n, hardware_max_fan_in=reduction_plan.k
            )
        else:
            safe_k, num_stages = reduction_plan.k, 0

        stage_output_partials = (n + safe_k - 1) // safe_k
        ppm.initialize(bm, max_bytes=stage_output_partials * partial_byte_size)

        current_n = n
        current_collection_ref = partial_collection_ref
        initial_offsets_host = gather_primitive.get_offsets()
        offset_list_ref, upload_evt = _create_offset_list(initial_offsets_host, wait_for)
        current_deps = [upload_evt]
        stage_idx = 0

        while current_n > 1:
            stage_output_n = (current_n + safe_k - 1) // safe_k
            is_final_stage = stage_output_n == 1
            stage_dest_ref = final_dest_handle if is_final_stage else ppm.get_io()[1]

            # >>> DELEGATION POINT <<<
            # The pipeline calls the provided 'stage_processor' to do the actual work.
            stage_j = num_stages - 1 - stage_idx if num_stages > 0 else 0
            stage_event = stage_processor(
                current_collection_ref,
                offset_list_ref,
                current_n,
                stage_dest_ref,
                stage_j,  # Pass stage_j for stabilization logic
                current_deps,
            )

            if is_final_stage:
                return stage_event

            # Prepare for next stage (identical logic)
            current_n = stage_output_n
            current_collection_ref = stage_dest_ref
            current_deps = [stage_event]
            next_gather_primitive = ContiguousGather(current_n, elements_per_partial)
            offset_list_ref, upload_evt = _create_offset_list(next_gather_primitive.get_offsets(), current_deps)
            current_deps = [upload_evt]
            ppm.swap()
            stage_idx += 1

        # Safeguard return for cases where loop doesn't run (n <= safe_k)
        user_event = cl.UserEvent(q.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event
    finally:
        ppm.release()
        for handle in transient_handles:
            bm.release_transient_buffer(handle)


def execute_stabilized_reduction_tree(
    svs: "Services",
    plan: "ExecutionPlan",
    gather_primitive: "GatherPrimitive",
    partial_collection_ref: "BufferHandle",
    final_dest_handle: "BufferHandle",
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """Recipe for the Recursive Clip-Aggregation Engine (Nodes 15 & 20), the core stabilized reduction tool."""
    spec, h_params = svs.model_spec, plan.hyperparams
    agg_mgr = AggregationManager(ex=svs.ex, bm=svs.bm, arch_consts=svs.arch_consts)

    def _process_stabilized_stage(
        collection_ref: BufferHandle,
        offset_list_ref: BufferHandle,
        num_partials: int,
        dest_ref: BufferHandle,
        stage_j: int,
        deps: List[cl.Event],
    ) -> cl.Event:
        """A nested function defining the 'sum-then-clip' operation for a single stage."""
        # 1. SUM
        sum_event = agg_mgr.execute_stage(
            svs.q,
            collection_ref,
            offset_list_ref,
            num_partials,
            gather_primitive.elements_per_partial,
            dest_ref,
            deps,
        )
        # 2. CLIP
        threshold_t_j = plan.stabilization_policy.get_threshold_for_generic_stage(
            stage_j=stage_j, runtime_fan_in_k=num_partials
        )
        clip_sig = ClipIntermediateGradSignature(
            _buffer_mgr=svs.bm,
            _arch_consts=svs.arch_consts,
            intermediate_grad_ref=dest_ref,
            clipping_threshold_t_j=spec.SCALAR_NP_TYPE(threshold_t_j),
            epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
        return svs.ex.launch(svs.q, clip_sig, wait_for=[sum_event])

    return _execute_reduction_pipeline(
        svs,
        plan,
        gather_primitive,
        partial_collection_ref,
        final_dest_handle,
        stage_processor=_process_stabilized_stage,
        wait_for=wait_for,
    )


def execute_summation_tree(
    svs: "Services",
    plan: "ExecutionPlan",
    gather_primitive: "GatherPrimitive",
    partial_collection_ref: "BufferHandle",
    final_dest_handle: "BufferHandle",
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """A pure recipe for a non-stabilized, log_K(N) summation tree (e.g., for diagnostics)."""
    agg_mgr = AggregationManager(ex=svs.ex, bm=svs.bm, arch_consts=svs.arch_consts)

    def _process_summation_stage(
        collection_ref: BufferHandle,
        offset_list_ref: BufferHandle,
        num_partials: int,
        dest_ref: BufferHandle,
        stage_j: int,  # This parameter is unused but required by the pipeline's contract
        deps: List[cl.Event],
    ) -> cl.Event:
        """A nested function defining the simpler 'sum-only' operation for a single stage."""
        return agg_mgr.execute_stage(
            svs.q,
            collection_ref,
            offset_list_ref,
            num_partials,
            gather_primitive.elements_per_partial,
            dest_ref,
            deps,
        )

    return _execute_reduction_pipeline(
        svs,
        plan,
        gather_primitive,
        partial_collection_ref,
        final_dest_handle,
        stage_processor=_process_summation_stage,
        wait_for=wait_for,
    )


def compute_effective_batch_size(
    svs: "Services", plan: "ExecutionPlan", deps: List[cl.Event]
) -> Tuple[HostView, cl.Event]:
    """Recipe to compute the effective batch size by summing the sample_mask."""
    bm, spec = svs.bm, svs.model_spec
    scalar_byte_size = spec.SCALAR_NP_TYPE().itemsize
    result_buffer_ref = bm.acquire_transient_buffer(scalar_byte_size)
    batch_size = plan.effective_batch_size

    try:
        gather_prim = LinearlyChunkedGather(num_chunks=batch_size, elements_per_chunk=1)
        sample_mask_ref = bm.get_handle_by_name("sample_mask")

        reduction_complete_evt = execute_summation_tree(
            svs=svs,
            plan=plan,
            gather_primitive=gather_prim,
            partial_collection_ref=sample_mask_ref,
            final_dest_handle=result_buffer_ref,
            wait_for=deps,
        )

        host_view = HostView(padded_shape=(1,), dtype=spec.SCALAR_NP_TYPE, real_shape=(1,))
        download_complete_evt = host_view.enqueue_read(
            queue=svs.q,
            cl_buffer=bm.get_cl_buffer(result_buffer_ref),
            wait_for=[reduction_complete_evt],
        )
        return host_view, download_complete_evt
    finally:
        bm.release_transient_buffer(result_buffer_ref)


# =========================================================================
# === Section 5: Finalization & Update Recipes                          ===
# =========================================================================
# This final section contains the recipes for the culmination of a learning
# step: normalizing the batch-wide gradients and applying the stateful
# optimizer update.
# =========================================================================


def build_update_subgraph(
    svs: Services,
    param_space: ParameterSpace,
    step: int,
    summed_grads: Dict[str, BufferHandle],
    effective_batch_size: float,
    plan: ExecutionPlan,
    deps: List[cl.Event],
) -> cl.Event:
    """Recipe for the final, batch-wide update stage (Nodes 21, 24, 25)."""
    bm, ex, q, spec = svs.bm, svs.ex, svs.q, svs.model_spec
    h_params = plan.hyperparams
    norm_events: List[cl.Event] = []
    final_grad_handles: Dict[str, BufferHandle] = {}

    # Node 21: Normalize all applicable summed gradients.
    for flow in param_space:
        if flow.specialized_reduction or flow.name not in summed_grads:
            continue
        sig = NormalizeGradientsSignature(
            _buffer_mgr=bm,
            _arch_consts=svs.arch_consts,
            summed_grad_ref=summed_grads[flow.name],
            final_grad_out_ref=bm.get_handle_by_name(flow.final_grad_buffer_name),
            effective_batch_size=spec.SCALAR_NP_TYPE(effective_batch_size),
            epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
        norm_events.append(ex.launch(q, sig, wait_for=deps))
        final_grad_handles[flow.name] = sig.final_grad_out_ref

    if "hidden_activations" in summed_grads:
        final_grad_handles["hidden_activations"] = summed_grads["hidden_activations"]

    # This barrier joins all parallel normalization streams.
    all_norm_evt = cl.enqueue_barrier(q, wait_for=norm_events) if norm_events else cl.enqueue_marker(q, wait_for=deps)

    # Node 24: Apply Adam Optimizer Update.
    # WHY: A CRITICAL ARCHITECTURAL MANDATE. The sensitive exponentiation is
    # performed here on the host, in high precision, and passed as a primitive
    # scalar to the kernel. This prevents on-device precision loss and underflow
    # during long training runs, guaranteeing numerical stability indefinitely.
    beta1_t = spec.SCALAR_NP_TYPE(h_params.adam_beta1**step)
    beta2_t = spec.SCALAR_NP_TYPE(h_params.adam_beta2**step)
    update_events: List[cl.Event] = []
    for flow in param_space:
        if flow.specialized_reduction or flow.name not in final_grad_handles:
            continue
        pg = AdamParameterGroup(
            param_ref=bm.get_handle_by_name(flow.param_buffer_name),
            grad_ref=final_grad_handles[flow.name],
            m1_state_ref=bm.get_handle_by_name(flow.m1_buffer_name),
            m2_state_ref=bm.get_handle_by_name(flow.m2_buffer_name),
        )
        adam_sig = AdamUpdateSignature(
            _buffer_mgr=bm,
            _arch_consts=svs.arch_consts,
            param_group=pg,
            learning_rate=spec.SCALAR_NP_TYPE(h_params.learning_rate),
            beta1=spec.SCALAR_NP_TYPE(h_params.adam_beta1),
            beta2=spec.SCALAR_NP_TYPE(h_params.adam_beta2),
            epsilon=spec.SCALAR_NP_TYPE(h_params.adam_epsilon),
            beta1_pow_t=beta1_t,
            beta2_pow_t=beta2_t,
        )
        update_events.append(ex.launch(q, adam_sig, wait_for=[all_norm_evt]))

    # This is the final Batch Synchronization Point before the cycle ends.
    all_updates_evt = cl.enqueue_barrier(q, wait_for=update_events) if update_events else all_norm_evt

    # Node 25: Final Clamping for domain-specific constraints.
    clamp_sig = ClampTemperaturesSignature(
        _buffer_mgr=bm,
        _arch_consts=svs.arch_consts,
        temps_ref=bm.get_handle_by_name("temperatures"),
        min_val=spec.SCALAR_NP_TYPE(h_params.temp_min),
        max_val=spec.SCALAR_NP_TYPE(h_params.temp_max),
    )
    return ex.launch(q, clamp_sig, wait_for=[all_updates_evt])
