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
from .execution_plan import ExecutionPlan
from .launcher_infra import Services, BufferManager, KernelExecutor, BufferHandle, PingPongManager
from .kernel_signatures import *
from .model_spec import ModelSpec
from .parameter_space import ParameterSpace
from .compute_patterns import AggregationManager, ReductionPlan
from .workload_primitives import GatherPrimitive, ContiguousGather, TiledGather, WorkTile


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
    Recipe for the forward-pass part of the module path.
    Its sole purpose is to produce the final `logits` and the intermediate
    `partial_probs` buffers required by the backpropagation stage.

    This version uses the polymorphic `ProblemTypeStrategy`
    from the `ExecutionPlan`, eliminating conditional logic for loss function selection.

    Args:
        svs: The bundle of core system services.
        tile: The `WorkTile` defining this specific unit of work.
        plan: The `ExecutionPlan` containing strategy and hyperparameter details.
        effective_batch_size: The actual calculated batch size, used for normalization.
        h_ref: The handle to the `hidden_activations` buffer (cached or recomputed).
        h_ready_evt: The event signaling that the `hidden_activations` buffer is ready.

    Returns:
        A tuple of (handle_to_partial_probs, completion_event_for_this_recipe).
    """
    bm, ex, q = svs.bm, svs.ex, svs.q
    spec = svs.model_spec

    # 1. Launch Logits Rendering (Node 5)
    logits_sig = RenderLogitsChunkSignature(
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

    # 2. Launch Loss & Probabilities (Nodes 6 or 7) via polymorphic delegation.
    # The `get_loss_signature` factory method on the strategy object is now
    # solely responsible for selecting the correct kernel signature and injecting
    # its own specific dependencies (e.g., the correct targets buffer).
    # This recipe is now completely ignorant of CCE vs. BCE.
    loss_sig = plan.problem_type.get_loss_signature(
        # Pass all common arguments.
        buffer_mgr=svs.bm,
        logit_ref=logits_sig.logit_out_ref,
        temp_ref=bm.get_handle_by_name("temperatures"),
        mask_ref=bm.get_handle_by_name("sample_mask"),
        prob_out_ref=bm.get_handle_by_name("partial_probs"),
        tile=tile,
        total_output_class_count=np.uint32(spec.output_classes),
        # Pass handles for ALL possible outputs. The strategy's specific
        # signature constructor will ignore the one it doesn't need.
        loss_out_ref=bm.get_handle_by_name("final_loss"),
        partial_loss_out_ref=bm.get_handle_by_name("partial_loss"),
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
    Recipe for the backward-pass part of the module path (Nodes 8-11).

    This uses the polymorphic `ProblemTypeStrategy`
    from the `ExecutionPlan`, eliminating conditional logic for selecting gradient
    computation kernels. This recipe should ONLY be executed when the plan's strategy is "CACHE".

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

    # 1. Launch Raw Partial Gradient Kernels (Nodes 8, 9, 10) via polymorphic delegation.
    # The recipe is now ignorant of "CCE" vs "BCE". It simply asks the strategy
    # object in the plan to provide the correct signature.
    wgs0 = arch_consts.get("work_group_size_0", 256)
    scalar_bytes = spec.scalar_dtype().itemsize
    padded_class_dim = spec.padded_class_dim

    grad_mod_sig = plan.problem_type.get_module_grad_signature(
        buffer_mgr=svs.bm,
        work_group_size_0=wgs0,
        scalar_size_bytes=scalar_bytes,
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
        padded_total_output_class_count=np.uint32(padded_class_dim),
        total_modules_count=np.uint32(spec.num_modules),
    )

    grad_h_sig = plan.problem_type.get_hidden_grad_signature(
        buffer_mgr=svs.bm,
        prob_ref=prob_ref,
        mask_ref=bm.get_handle_by_name("sample_mask"),
        w_mod_ref=bm.get_handle_by_name("module_weights"),
        gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
        tile=tile,
        hidden_count=np.uint32(spec.hidden_dim),
        total_output_class_count=np.uint32(spec.output_classes),
    )

    grad_t_sig = plan.problem_type.get_temp_grad_signature(
        buffer_mgr=svs.bm,
        work_group_size_0=wgs0,
        scalar_size_bytes=scalar_bytes,
        logit_ref=bm.get_handle_by_name("logits"),
        prob_ref=prob_ref,
        mask_ref=bm.get_handle_by_name("sample_mask"),
        temp_ref=bm.get_handle_by_name("temperatures"),
        gt_out_ref=bm.get_handle_by_name("partial_grad_temps"),
        tile=tile,
        total_output_class_count=np.uint32(spec.output_classes),
    )

    grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=prior_deps)
    grad_h_evt = ex.launch(q, grad_h_sig, wait_for=prior_deps)
    grad_t_evt = ex.launch(q, grad_t_sig, wait_for=prior_deps)

    # 2. Launch Gradient Clipping (Node 11) - This logic remains the same as it is
    # already agnostic to the CCE/BCE problem type.
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
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            handles=grad_handles,
            tile=tile,
            clipping_threshold_global=SCALAR_NP_TYPE(plan.stabilization_policy.get_leaf_safety_threshold()),
            epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
        )
    else:  # 'PER_ITEM'
        clip_sig = ClipPartialGradientsPerItemNormSignature(
            work_group_size_0=wgs0,
            scalar_size_bytes=scalar_bytes,
            handles=grad_handles,
            tile=tile,
            clipping_threshold_per_item_ref=bm.get_handle_by_name("clipping_threshold_per_item"),
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
            work_group_size_0=arch_consts.get("work_group_size_0", 256),
            scalar_size_bytes=spec.scalar_dtype().itemsize,
            handles=shared_grad_handles,
            clipping_threshold_global=SCALAR_NP_TYPE(plan.stabilization_policy.get_leaf_safety_threshold()),
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


def execute_stabilized_reduction_tree(
    svs: "Services",
    plan: "ExecutionPlan",
    gather_primitive: "GatherPrimitive",
    partial_collection_ref: "BufferHandle",
    final_dest_handle: "BufferHandle",
    wait_for: Optional[List[cl.Event]] = None,
) -> cl.Event:
    """
    A pure, stateless recipe for executing a complete, multi-stage log_K(N)
    reduction that is fully governed by the system's dynamic stabilization policy.

    Architectural Mandate:
    This recipe is the canonical implementation of the "Recursive Clip-Aggregation
    Engine" for gradient parameters (Nodes 15 & 20). It is contractually
    obligated to perform an atomic "sum-then-clip" pattern at each stage of
    the reduction tree. For each layer of the tree, it will:
      1. SUM a set of K partials into a transient intermediate buffer.
      2. CLIP that intermediate buffer in-place, using a unique threshold T_j
         that is calculated for that specific stage by the StabilizationPolicy.
      3. Use the resulting clipped buffer as the input for the subsequent stage.

    This ensures that the entire reduction process adheres to the user-defined
    algorithmic constraints and the hardware's physical safety limits.

    Args:
        svs: The bundle of core system services (queue, executor, etc.).
        reduction_plan: The `ReductionPlan` defining hardware fan-in and other config.
        stabilization_policy: The authoritative policy object for calculating thresholds.
        gather_primitive: The GatherPrimitive describing the layout of the initial partials.
        partial_collection_ref: Handle to the buffer containing all partials to be reduced.
        final_dest_handle: Handle to the dense buffer for the final result.
        wait_for: A list of `cl.Event` objects to wait for before executing.

    Returns:
        A single `cl.Event` that signals the completion of the entire reduction tree.
    """
    wait_for = wait_for or []
    q, ex, bm = svs.q, svs.ex, svs.bm
    arch_consts = svs.arch_consts
    stabilization_policy = plan.stabilization_policy
    reduction_plan = plan.reduction_plan
    h_params = plan.hyperparams
    agg_mgr = AggregationManager(ex=ex, bm=bm, arch_consts=arch_consts)

    n = gather_primitive.num_partials

    # --- Handle Edge Cases ---
    if n <= 0:
        # No work to do, return a completed event immediately.
        user_event = cl.UserEvent(q.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

    elements_per_partial = gather_primitive.elements_per_partial
    spec, dtype = bm.get_spec(final_dest_handle)
    scalar_byte_size = dtype().itemsize
    partial_byte_size = elements_per_partial * scalar_byte_size

    if n == 1:
        # A "reduction" of one partial is just a direct memory copy.
        initial_offsets = gather_primitive.get_offsets()
        src_offset_bytes = int(initial_offsets[0] * scalar_byte_size)
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
    transient_handles = []

    # This helper function encapsulates the creation and upload of the indirection table.
    def _create_offset_list(offsets_host: np.ndarray, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        offset_list_ref = bm.acquire_transient_buffer(offsets_host.nbytes)
        transient_handles.append(offset_list_ref)  # Track for cleanup
        evt = cl.enqueue_copy(q, bm.get_cl_buffer(offset_list_ref), offsets_host, wait_for=deps)
        return offset_list_ref, evt

    try:
        # --- Pre-computation & Planning Step (Action 2.2) ---
        safe_k, num_stages = stabilization_policy.plan_uniform_reduction_tree(
            num_partials=n, hardware_max_fan_in=reduction_plan.k
        )

        # Allocate ping-pong buffers large enough for the output of the first stage.
        stage_output_partials = (n + safe_k - 1) // safe_k
        ppm_buffer_size_bytes = stage_output_partials * partial_byte_size
        ppm.initialize(bm, max_bytes=ppm_buffer_size_bytes)

        # Prepare the initial set of inputs for the first loop iteration.
        current_n = n
        current_collection_ref = partial_collection_ref
        initial_offsets_host = gather_primitive.get_offsets()
        offset_list_ref, upload_evt = _create_offset_list(initial_offsets_host, wait_for)
        current_deps = [upload_evt]
        stage_idx = 0

        # --- Main Reduction Loop (implements Action 2.3) ---
        while current_n > 1:
            stage_output_n = (current_n + safe_k - 1) // safe_k
            is_final_stage = stage_output_n == 1

            # Determine the destination for this stage's SUM operation.
            # If it's the last stage, write directly to the final destination.
            stage_sum_dest_ref = final_dest_handle if is_final_stage else ppm.get_io()[1]

            # --- 1. SUM: Aggregate the current set of partials ---
            sum_event = agg_mgr.execute_stage(
                q,
                current_collection_ref,
                offset_list_ref,
                current_n,
                elements_per_partial,
                stage_sum_dest_ref,
                current_deps,
            )

            # --- 2. CLIP: Stabilize the intermediate sum ---
            # The stage index 'j' is the number of steps away from the ROOT node.
            stage_j = num_stages - 1 - stage_idx
            threshold_t_j = stabilization_policy.get_threshold_for_generic_stage(
                stage_j=stage_j, runtime_fan_in_k=current_n
            )

            clip_sig = ClipIntermediateGradSignature(
                work_group_size_0=arch_consts.get("work_group_size_0", 256),
                scalar_size_bytes=scalar_byte_size,
                intermediate_grad_ref=stage_sum_dest_ref,
                clipping_threshold_t_j=SCALAR_NP_TYPE(threshold_t_j),
                epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
            )
            clip_event = ex.launch(q, clip_sig, wait_for=[sum_event])

            if is_final_stage:
                # The reduction is complete. Return the final clipping event.
                return clip_event

            # --- 3. Prepare Inputs for the Next Stage ---
            current_n = stage_output_n
            current_collection_ref = stage_sum_dest_ref
            current_deps = [clip_event]

            # The outputs of each stage are contiguous, so we use a ContiguousGather
            # primitive to generate the next set of offsets.
            next_gather_primitive = ContiguousGather(current_n, elements_per_partial)
            next_offsets_host = next_gather_primitive.get_offsets()
            offset_list_ref, upload_evt = _create_offset_list(next_offsets_host, current_deps)
            current_deps = [upload_evt]

            ppm.swap()
            stage_idx += 1

        # This part of the code should theoretically be unreachable if n > 1,
        # as the loop handles the final stage. Return a completed event as a safeguard.
        user_event = cl.UserEvent(q.context)
        user_event.set_status(cl.command_execution_status.COMPLETE)
        return user_event

    finally:
        # Ensure all transient resources are released, regardless of exceptions.
        ppm.release()
        for handle in transient_handles:
            bm.release_transient_buffer(handle)


def build_streaming_module_grad_path(svs: Services, plan: ExecutionPlan, deps: List[cl.Event]) -> Dict[str, cl.Event]:
    """
    (Nodes 8, 9, 10, 11) The canonical recipe for the "Accumulate via Recompute" model.

    Architectural Mandate:
    This recipe fulfills the computational contract for the module gradient path
    under the `RECOMPUTE_GRAD_H` adaptation strategy. Its purpose is to produce the complete,
    clipped collections of all partial gradients (`Grad_ModW/B`, `Grad_Temps`, AND `Grad_H`)
    without ever holding the full `hidden_activations` buffer in VRAM. It achieves
    this via a host-driven streaming loop that, for each tile, recomputes inputs,
    computes raw partials into scratch space, and scatters clipped results into final
    collection buffers.

    Returns:
        A dictionary mapping the canonical names of the final, clipped partial
        gradient collection buffers to the single `cl.Event` signaling their simultaneous
        completion.
    """
    q, ex, bm = svs.q, svs.ex, svs.q
    spec, h_params, grid = svs.model_spec, plan.hyperparams, plan.grid
    arch_consts = svs.arch_consts
    scalar_bytes = spec.scalar_dtype().itemsize
    wgs0 = arch_consts.get("work_group_size_0", 256)

    # --- Step 1: Resource Acquisition ---
    # Acquire handles for the final, monolithic COLLECTION buffers this recipe populates.
    clipped_gw_out_ref = bm.get_handle_by_name("clipped_partial_grad_module_weights")
    clipped_gb_out_ref = bm.get_handle_by_name("clipped_partial_grad_module_biases")
    clipped_gt_out_ref = bm.get_handle_by_name("clipped_partial_grad_temps")
    clipped_gh_out_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")

    # Acquire transient SCRATCH buffers. These are reused in every loop iteration.
    h_full_shape, _ = bm.get_spec(bm.get_handle_by_name("hidden_activations"))
    gw_coll_shape, _ = bm.get_spec(clipped_gw_out_ref)
    gb_coll_shape, _ = bm.get_spec(clipped_gb_out_ref)
    gt_coll_shape, _ = bm.get_spec(clipped_gt_out_ref)
    gh_coll_shape, _ = bm.get_spec(clipped_gh_out_ref)

    # Scratch buffer for recomputed hidden activations must be full-sized to satisfy
    # the contract of the ForwardPassSignature.
    h_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_full_shape) * scalar_bytes))
    h_mask_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_full_shape) * scalar_bytes))
    # Scratch buffers for raw partial gradients only need to hold one tile's worth of data.
    raw_gw_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gw_coll_shape[1:]) * scalar_bytes))
    raw_gb_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gb_coll_shape[1:]) * scalar_bytes))
    raw_gt_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gt_coll_shape[1:]) * scalar_bytes))
    raw_gh_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gh_coll_shape[1:]) * scalar_bytes))

    scratch_handles = [
        h_scratch_ref,
        h_mask_scratch_ref,
        raw_gw_scratch_ref,
        raw_gb_scratch_ref,
        raw_gt_scratch_ref,
        raw_gh_scratch_ref,
    ]

    try:
        all_clip_events = []
        # --- Step 2: Host-Driven Streaming Loop ---
        for tile in grid:
            # Step 2a: Recompute hidden activations into scratch space (Node 4 logic)
            fwd_pass_sig = ForwardPassSignature(
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
            grad_calc_deps = [h_ready_evt]

            # Step 2b: Compute all raw partial gradients for this tile into scratch buffers
            grad_mod_sig = plan.problem_type.get_module_grad_signature(
                buffer_mgr=svs.bm,
                work_group_size_0=wgs0,
                scalar_size_bytes=scalar_bytes,
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
                buffer_mgr=svs.bm,
                prob_ref=bm.get_handle_by_name("partial_probs"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                w_mod_ref=bm.get_handle_by_name("module_weights"),
                gh_out_ref=raw_gh_scratch_ref,
                tile=tile,
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
            )
            grad_t_sig = plan.problem_type.get_temp_grad_signature(
                buffer_mgr=svs.bm,
                work_group_size_0=wgs0,
                scalar_size_bytes=scalar_bytes,
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

            # Step 2c: Perform the holistic clip, scattering from scratch to the final collection buffers.
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

            if plan.clipping_strategy == "GLOBAL":
                clip_sig = ClipPartialGradientsGlobalNormSignature(
                    work_group_size_0=wgs0,
                    scalar_size_bytes=scalar_bytes,
                    handles=grad_handles,
                    tile=tile,
                    clipping_threshold_global=SCALAR_NP_TYPE(plan.stabilization_policy.get_leaf_safety_threshold()),
                    epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
                )
            else:  # 'PER_ITEM'
                clip_sig = ClipPartialGradientsPerItemNormSignature(
                    work_group_size_0=wgs0,
                    scalar_size_bytes=scalar_bytes,
                    handles=grad_handles,
                    tile=tile,
                    clipping_threshold_per_item_ref=bm.get_handle_by_name("clipping_threshold_per_item"),
                    epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
                )

            clip_event = ex.launch(q, clip_sig, wait_for=all_raw_grads_ready)
            all_clip_events.append(clip_event)

        # --- Step 3: Synchronization and Return ---
        all_clips_done_evt = cl.WaitForEvents(all_clip_events)

        # Return a dictionary that maps the canonical names of the four final
        # collection buffers to the single event that signals their completion.
        return {
            "clipped_partial_grad_module_weights": all_clips_done_evt,
            "clipped_partial_grad_module_biases": all_clips_done_evt,
            "clipped_partial_grad_temps": all_clips_done_evt,
            "clipped_partial_grad_hidden_activations": all_clips_done_evt,
        }

    finally:
        # --- Step 4: Resource Cleanup ---
        # A contractually obligated step to prevent VRAM leakage from transient allocations.
        for handle in scratch_handles:
            bm.release_transient_buffer(handle)


def build_final_grad_h_reduction_path(
    svs: Services, plan: ExecutionPlan, deps: List[cl.Event]
) -> Tuple[BufferHandle, cl.Event]:
    """
    (Nodes 13, 16) The canonical recipe for permuting and reducing Grad_H.

    Architectural Mandate:
    This recipe is a simple, stateless sequencer for the two kernels that form
    the Item Synchronization Point and the specialized reduction for `Grad_H`. It
    is the terminal step in the `RECOMPUTE_GRAD_H` data path, consuming the
    clipped partials produced by `build_streaming_module_grad_path` and yielding
    the final `summed_grad_hidden_activations` tensor.

    Returns:
        A tuple of (handle_to_summed_grad_h, completion_event_for_this_recipe).
    """
    q, ex, bm = svs.q, svs.ex, svs.bm
    spec, h_params = svs.model_spec, plan.hyperparams
    arch_consts = svs.arch_consts

    # Acquire handles for the three buffers involved in this specific pipeline.
    clipped_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")
    permuted_ref = bm.get_handle_by_name("permuted_grad_h")
    summed_ref = bm.get_handle_by_name("summed_grad_hidden_activations")

    # Step 1: Gather scattered partials and permute into a reduction-ready SoA layout (Node 13).
    permute_sig = GatherAndPermuteGradHiddenActivationsSignature(
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

    # Step 2: Reduce the permuted buffer using the specialized, policy-aware kernel (Node 16).
    wgs0 = arch_consts.get("work_group_size_0", 256)
    policy_k = plan.stabilization_policy.get_specialized_reduction_policy_k(
        user_policy_k=h_params.reduction_k_grad_h,
        hardware_max_fan_in=wgs0,
    )
    reduce_sig = StabilizeAndReduceGradHiddenActivationsSignature(
        work_group_size_0=wgs0,
        scalar_size_bytes=spec.scalar_dtype().itemsize,
        permuted_soa_in_ref=permuted_ref,
        final_grad_h_out_ref=summed_ref,
        fp_max=SCALAR_NP_TYPE(plan.stabilization_policy.fp_format_max),
        policy_t_algorithmic=SCALAR_NP_TYPE(h_params.max_grad_norm),
        policy_lambda=SCALAR_NP_TYPE(h_params.stabilization_lambda),
        policy_max_k=np.uint32(policy_k),
        epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
        total_batch_count=np.uint32(plan.effective_batch_size),
        padded_hidden_count=np.uint32(spec.padded_hidden_dim),
        total_modules_count=np.uint32(spec.num_modules),
        padded_total_modules_count=np.uint32(spec.padded_module_dim),
    )
    reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])

    return summed_ref, reduce_evt


# =========================================================================
# === Recipes for Aggregation & Final Update                            ===
# =========================================================================


def execute_summation_tree(
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


def compute_effective_batch_size(
    svs: "Services",
    batch_size: int,
    deps: List[cl.Event],
) -> Tuple[HostView, cl.Event]:
    """
    (Node 21 Dependency) A stateless recipe to compute the effective batch size.

    Architectural Mandate:
    This recipe is the sole, authoritative implementation of the `Calculability Proof`
    for the `src_scalar_REAL_effective_batch_size` parameter in the `normalize_gradients`
    kernel contract. Its purpose is to perform a reduction sum over the `sample_mask`
    buffer on the device and make the resulting single scalar value available to the host.

    Role:
    As a foundational recipe, it abstracts away the complexity of reduction and
    asynchronous data retrieval. It is invoked by the `BatchProcessor` to acquire
    the correct normalization factor before launching the final update subgraph.

    Return Value Contract:
    This function returns a tuple containing:
    1.  A `HostView` object: An asynchronous handle to the host-side result.
    2.  A `cl.Event`: The final event that signals the completion of the device-to-host
        copy. The caller MUST wait for this event before calling `.get()` on the
        HostView object to safely retrieve the scalar value.

    Args:
        svs: The bundle of core system services.
        batch_size: The logical size of the batch, used to define the gather primitive.
        deps: A list of `cl.Event` objects to wait for before executing (e.g., the
              event from uploading the `sample_mask` data).

    Returns:
        A tuple of (HostView, cl.Event) for asynchronous result retrieval.
    """
    bm, spec = svs.bm, svs.model_spec

    # 1. Acquire a transient buffer to hold the single scalar result of the reduction.
    #    This buffer is managed with a try/finally block to guarantee its release.
    scalar_byte_size = spec.scalar_dtype().itemsize
    result_buffer_ref = bm.acquire_transient_buffer(scalar_byte_size)

    try:
        # 2. Define the reduction plan and the memory layout of the source data.
        #    We model the 1D sample_mask as `batch_size` partials, each of size 1.
        #    This allows us to reuse the generic reduction engine.
        wgs0 = svs.arch_consts.get("work_group_size_0", 256)
        reduction_plan = ReductionPlan(k=wgs0)
        gather_prim = LinearlyChunkedGather(num_chunks=batch_size, elements_per_chunk=1)
        sample_mask_ref = bm.get_handle_by_name("sample_mask")

        # 3. Launch the generic summation tree recipe to perform the reduction on the device.
        reduction_complete_evt = execute_summation_tree(
            svs=svs,
            reduction_plan=reduction_plan,
            gather_primitive=gather_prim,
            partial_collection_ref=sample_mask_ref,
            final_dest_handle=result_buffer_ref,
            wait_for=deps,
        )

        # 4. Prepare for the asynchronous download of the single scalar result.
        #    The HostView acts as a padded numpy array on the host, ready to receive data.
        host_view = HostView(
            padded_shape=(1,),
            dtype=spec.scalar_dtype,
            real_shape=(1,),
        )

        # 5. Enqueue the non-blocking copy from the transient device buffer to the host view.
        download_complete_evt = host_view.enqueue_read(
            queue=svs.q,
            cl_buffer=bm.get_cl_buffer(result_buffer_ref),
            wait_for=[reduction_complete_evt],
        )

        # 6. Return the host view and the final event. The caller now owns the
        #    responsibility of waiting for the event and extracting the data.
        return host_view, download_complete_evt

    finally:
        # 7. This is a contractually obligated step. The transient buffer used for
        #    the reduction result MUST be released to prevent VRAM leakage.
        bm.release_transient_buffer(result_buffer_ref)


def build_update_subgraph(
    svs: Services,
    param_space: ParameterSpace,
    step: int,
    summed_grads: Dict[str, BufferHandle],
    effective_batch_size: float,
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
            summed_grad_ref=summed_grads[flow.name],
            final_grad_out_ref=bm.get_handle_by_name(flow.final_grad_buffer_name),
            effective_batch_size=SCALAR_NP_TYPE(effective_batch_size),
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
        temps_ref=bm.get_handle_by_name("temperatures"),
        min_val=SCALAR_NP_TYPE(h_params.temp_min),
        max_val=SCALAR_NP_TYPE(h_params.temp_max),
    )
    clamp_evt = ex.launch(q, clamp_sig, wait_for=[all_updates_evt])

    return clamp_evt


def execute_diagnostic_aggregation(svs: Services, plan: ExecutionPlan, deps: List[cl.Event]) -> Dict[str, cl.Event]:
    """
    (Node 14) A stateless recipe to aggregate all diagnostic results from the Act phase.

    Architectural Mandate:
    This function is the canonical implementation of the "Reduction (Act Results)"
    step. It is responsible for consuming the scattered partial results generated during
    the forward pass and reducing them into their final, batch-wide forms.
    It guarantees the availability of `Final Probs` and, when applicable, `Final BCE Loss`.

    This centralization enables two critical architectural improvements:
    1.  It simplifies the `ProblemTypeStrategy` contracts, allowing them to be pure
        "signature factories" without DAG-building logic.
    2.  It allows the `BatchProcessor` to remain a pure "Conductor," ignorant of the
        specific aggregation needs of different loss functions.

    Supersedes:
    - The `BceStrategy.build_loss_aggregation_subgraph` method, which is now obsolete.

    Args:
        svs: The bundle of core system services.
        plan: The `ExecutionPlan` containing strategy and layout information.
        deps: A list of `cl.Event` objects to wait for (e.g., from the forward pass).

    Returns:
        A dictionary mapping aggregated artifact names ("probs", "loss") to the
        `cl.Event` signaling their completion.
    """
    bm = svs.bm
    completion_events: Dict[str, cl.Event] = {}

    # --- 1. Aggregate Probabilities (Universal Requirement) ---
    # This step is always executed to produce the final, batch-wide probability tensor,
    # which is essential for inference, validation, and debugging.
    partial_probs_ref = bm.get_handle_by_name("partial_probs")
    final_probs_ref = bm.get_handle_by_name("final_probs")

    # Determine the memory footprint of a single partial probability tile from the
    # collection buffer's specification.
    partial_probs_shape, _ = bm.get_spec(partial_probs_ref)
    elements_per_prob_partial = int(np.prod(partial_probs_shape[1:]))

    # Instantiate the Gather Primitive that perfectly describes the memory layout
    # of the scattered source partials for the reduction engine.
    prob_gather_prim = TiledGather(scheme=plan.grid, _elements_per_partial=elements_per_prob_partial)

    # Launch the reusable, generic summation tree recipe.
    prob_agg_evt = execute_summation_tree(
        svs=svs,
        reduction_plan=plan.reduction_plan,
        gather_primitive=prob_gather_prim,
        partial_collection_ref=partial_probs_ref,
        final_dest_handle=final_probs_ref,
        wait_for=deps,
    )
    completion_events["probs"] = prob_agg_evt

    # --- 2. Conditionally Aggregate Loss (For BCE Problem Type) ---
    # We use the strategy's required buffer name as a clean, contract-driven check.
    if plan.problem_type.required_targets_buffer_name == "targets_bce":
        partial_loss_ref = bm.get_handle_by_name("partial_loss")
        final_loss_ref = bm.get_handle_by_name("final_loss")

        partial_loss_shape, _ = bm.get_spec(partial_loss_ref)
        elements_per_loss_partial = int(np.prod(partial_loss_shape[1:]))

        loss_gather_prim = TiledGather(scheme=plan.grid, _elements_per_partial=elements_per_loss_partial)

        loss_agg_evt = execute_summation_tree(
            svs=svs,
            reduction_plan=plan.reduction_plan,
            gather_primitive=loss_gather_prim,
            partial_collection_ref=partial_loss_ref,
            final_dest_handle=final_loss_ref,
            wait_for=deps,
        )
        completion_events["loss"] = loss_agg_evt

    return completion_events
