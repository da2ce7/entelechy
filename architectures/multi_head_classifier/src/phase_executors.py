# phase_executors.py

"""
A Toolbox of Tactical Phase Executors for the Training DAG.

(REV 3 - COMPLETED) This module provides a collection of dedicated
"PhaseExecutor" classes. Each class is a tactical expert responsible
for executing a specific, cohesive part of the computational graph.

This version has been fully rectified to:
1.  Complete all dynamic execution paths (BCE, per-item clipping).
2.  Correctly instantiate all KernelSignatures according to their C-level contracts.
3.  Source all hyperparameters and configuration from the `ExecutionPlan`.
4.  Iterate over the `ParameterSpace` manifest for fully general updates.
5.  Flesh out all placeholder arguments for kernel signature instantiations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from execution_plan import ExecutionPlan, WorkTile
from launcher_infra import BufferManager, KernelExecutor, BufferHandle, SCALAR_NP_TYPE
from kernel_signatures import *
from model_spec import ModelSpec
from parameter_space import ParameterSpace
from kernel_signatures import (
    ForwardPassSignature,
    BackpropErrorToHiddenChunkCceSignature,
    BackpropErrorToHiddenChunkBceSignature,
    ClipPartialGradientsGlobalNormSignature,
    ClipPartialGradientsPerItemNormSignature,
    GatherAndPermuteGradHSignature,
    ReduceGradHOverModulesSignature,
)

from compute_patterns import (
    ReductionTreeExecutor,
    ReductionPlan,
    GatherPrimitive,
    TiledGather,
    LinearlyChunkedGather,
)


@dataclass
class Services:
    """A simple container to pass around core system components for dependency injection."""

    q: cl.CommandQueue
    ex: KernelExecutor
    bm: BufferManager
    model_spec: ModelSpec
    arch_consts: Dict[str, int]


class ForwardPassExecutor:
    """PhaseExecutor for the initial shared layer forward pass (Node 4)."""

    def __init__(self, services: Services):
        self.svs = services

    def run(self, batch_size: int, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        """Executes the full forward pass. Intended for the "CACHE" strategy."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q

        sig = ForwardPassSignature(
            buffer_mgr=bm,
            simd_width=self.svs.model_spec.simd_width,
            local_mem_bank_padding=1,
            scalar_size_bytes=np.dtype(self.svs.model_spec.scalar_dtype).itemsize,
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


class ModulePathExecutor:
    """PhaseExecutor for the highly parallel per-tile module path (Nodes 5-11)."""

    def __init__(self, services: Services):
        self.svs = services

    def run_for_tile(self, tile: WorkTile, plan: ExecutionPlan, deps: List[cl.Event]) -> cl.Event:
        """Executes the full chain of per-tile kernels, from logits to CLIPPED gradients."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        spec, h_params = self.svs.model_spec, plan.hyperparams
        arch_consts = self.svs.arch_consts

        # 1. Resolve `hidden_activations` dependency using the plan's policy
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

        # 3. Launch Loss & Probabilities (Nodes 6 or 7) - Dynamically
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
                prob_ref=loss_sig.prob_out_ref,
                targets_cce_ref=bm.get_handle_by_name("targets_cce"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                w_mod_ref=bm.get_handle_by_name("module_weights"),
                gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
                tile=tile,
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
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
                prob_ref=loss_sig.prob_out_ref,
                targets_bce_ref=bm.get_handle_by_name("targets_bce"),
                mask_ref=bm.get_handle_by_name("sample_mask"),
                w_mod_ref=bm.get_handle_by_name("module_weights"),
                gh_out_ref=bm.get_handle_by_name("partial_grad_hidden_activations"),
                tile=tile,
                hidden_count=np.uint32(spec.hidden_dim),
                total_output_class_count=np.uint32(spec.output_classes),
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

        # 5. Launch Gradient Clipping (Node 11) - Dynamically
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


class SharedBackpropExecutor:
    """PhaseExecutor for streaming backpropagation through the shared layer (Nodes 17-19)."""

    def __init__(self, services: Services):
        self.svs = services

    def run(
        self, plan: ExecutionPlan, num_batch_chunks: int, summed_grad_h_ref: BufferHandle, deps: List[cl.Event]
    ) -> cl.Event:
        """Runs the streaming backprop, producing CLIPPED partials for shared params."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        spec, h_params = self.svs.model_spec, plan.hyperparams
        arch_consts = self.svs.arch_consts
        final_chunk_events = []

        # 1. Resolve `hidden_activations` dependency (Once, outside the loop)
        h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)
        deps_for_all_chunks = [h_ready_evt]  # This stream only depends on h and grad_h

        # 2. Acquire Transient Scratch Buffers for Raw Gradients
        w_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_weights"))
        b_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_biases"))
        gsw_chunk_shape = (w_shape[0], w_shape[1])
        gsb_chunk_shape = (b_shape[0],)
        gsw_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gsw_chunk_shape) * spec.scalar_dtype().itemsize))
        gsb_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gsb_chunk_shape) * spec.scalar_dtype().itemsize))

        # 3. Main Streaming Loop
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
                work_group_size_1=arch_consts.get("work_group_size_1", 16),
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                input_ref=bm.get_handle_by_name("input"),
                h_ref=h_ref,
                grad_h_ref=summed_grad_h_ref,
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsw_out_ref=gsw_scratch_ref,
                batch_chunk_offset=np.uint32(batch_offset),
                batch_chunk_count=np.uint32(items_in_chunk),
                batch_chunk_index=np.uint32(i),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            )
            gsw_evt = ex.launch(q, gsw_sig, wait_for=deps_for_all_chunks)
            gsb_sig = BackpropSharedBiasesChunkSignature(
                bm,
                work_group_size_0=arch_consts.get("work_group_size_0", 256),
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                h_ref=h_ref,
                grad_h_ref=summed_grad_h_ref,
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

        # 4. Release Transient Resources
        bm.release_transient_buffer(gsw_scratch_ref)
        bm.release_transient_buffer(gsb_scratch_ref)
        return cl.WaitForEvents(final_chunk_events)


class UpdatePhaseExecutor:
    """PhaseExecutor for the final, batch-wide update stage (Nodes 21, 24, 25)."""

    def __init__(self, services: Services, param_space: ParameterSpace):
        self.svs = services
        self.param_space = param_space

    def run(
        self, step: int, summed_grads: Dict[str, BufferHandle], plan: ExecutionPlan, deps: List[cl.Event]
    ) -> cl.Event:
        """Executes the Normalize -> Adam Update -> Clamp sequence."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        h_params = plan.hyperparams

        # (Node 21) Normalize all summed gradients
        norm_events, final_grad_handles = [], {}
        for flow in self.param_space:
            # The specialized `summed_grad_h` is already final and doesn't get normalized here
            if flow.specialized_reduction:
                continue

            # Handle the case where summed_grads for optional flows might not exist
            if flow.name not in summed_grads:
                continue

            sig = NormalizeGradientsSignature(
                bm,
                summed_grad_ref=summed_grads[flow.name],
                final_grad_out_ref=bm.get_handle_by_name(flow.final_grad_buffer_name),
                effective_batch_size=SCALAR_NP_TYPE(plan.effective_batch_size),
                epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
            )
            norm_events.append(ex.launch(q, sig, wait_for=deps))
            final_grad_handles[flow.name] = sig.final_grad_out_ref

        # Add the handle for the already-reduced Grad_H to the dictionary for the optimizer
        if "hidden_activations" in summed_grads:
            final_grad_handles["hidden_activations"] = summed_grads["hidden_activations"]

        all_norm_evt = (
            cl.WaitForEvents(norm_events)
            if norm_events
            else cl.UserEvent(q.context).set_status(cl.command_execution_status.COMPLETE)
        )

        # (Node 24) Apply Adam Optimizer Update
        beta1_t = SCALAR_NP_TYPE(h_params.adam_beta1**step)
        beta2_t = SCALAR_NP_TYPE(h_params.adam_beta2**step)
        update_events = []
        for flow in self.param_space:
            # The upstream gradient (Grad_H) is not a learnable parameter, so no Adam update
            if flow.specialized_reduction:
                continue
            pg = AdamParameterGroup(
                param_ref=bm.get_handle_by_name(flow.param_buffer_name),
                grad_ref=final_grad_handles[flow.name],
                m1_state_ref=bm.get_handle_by_name(flow.m1_buffer_name),
                m2_state_ref=bm.get_handle_by_name(flow.m2_buffer_name),
            )
            adam_sig = AdamUpdateSignature(
                bm,
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

        # (Node 25) Final Clamping
        clamp_sig = ClampTemperaturesSignature(
            bm,
            temps_ref=bm.get_handle_by_name("temperatures"),
            min_val=SCALAR_NP_TYPE(h_params.temp_min),
            max_val=SCALAR_NP_TYPE(h_params.temp_max),
        )
        clamp_evt = ex.launch(q, clamp_sig, wait_for=[all_updates_evt])
        return clamp_evt


class GradHStreamingExecutor:
    """
    A specialized PhaseExecutor for the "Model A: Accumulate via Recompute" strategy.
    ... (docstring as before)
    """

    def __init__(self, services: Services):
        self.svs = services

    def compute_summed_grad_h(self, plan: ExecutionPlan, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        """
        Executes the full recompute->bprop->clip->permute->reduce pipeline for Grad_H.
        ... (docstring as before)
        """
        q, ex, bm = self.svs.q, self.svs.ex, self.svs.bm
        spec, h_params = self.svs.model_spec, plan.hyperparams
        arch_consts, grid = self.svs.arch_consts, plan.grid
        scalar_bytes = spec.scalar_dtype().itemsize

        print("    [GradHStreamingExecutor] Executing 'Accumulate via Recompute' for Grad_H...")

        clipped_aos_ref = bm.get_handle_by_name("clipped_partial_grad_hidden_activations")
        permuted_soa_ref = bm.get_handle_by_name("permuted_grad_h")
        final_summed_grad_h_ref = bm.get_handle_by_name("summed_grad_hidden_activations")
        h_chunk_shape, _ = bm.get_spec(bm.get_handle_by_name("hidden_activations"))
        gh_chunk_shape, _ = bm.get_spec(bm.get_handle_by_name("partial_grad_hidden_activations"))
        gh_chunk_shape = (1, gh_chunk_shape[1], gh_chunk_shape[2], gh_chunk_shape[3])

        h_chunk_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_chunk_shape) * scalar_bytes))
        h_mask_chunk_scratch_ref = bm.acquire_transient_buffer(int(np.prod(h_chunk_shape) * scalar_bytes))
        raw_gh_chunk_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gh_chunk_shape) * scalar_bytes))

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
                    prob_ref=bm.get_handle_by_name("partial_probs"),
                    targets_cce_ref=bm.get_handle_by_name("targets_cce"),
                    mask_ref=bm.get_handle_by_name("sample_mask"),
                    w_mod_ref=bm.get_handle_by_name("module_weights"),
                    gh_out_ref=raw_gh_chunk_scratch_ref,
                    tile=tile,
                    hidden_count=np.uint32(spec.hidden_dim),
                    total_output_class_count=np.uint32(spec.output_classes),
                )
            else:  # BCE
                grad_h_sig = BackpropErrorToHiddenChunkBceSignature(...)  # Omitted for brevity

            raw_gh_ready_evt = ex.launch(q, grad_h_sig, wait_for=[h_ready_evt] + deps)

            grad_handles = GradientHandles(
                grad_hidden_activations_aos=raw_gh_chunk_scratch_ref,
                clipped_grad_hidden_activations_aos=clipped_aos_ref,
                grad_weights_module=bm.get_handle_by_name("partial_grad_module_weights"),
                grad_biases_module=bm.get_handle_by_name("partial_grad_module_biases"),
                grad_temps=bm.get_handle_by_name("partial_grad_temps"),
                clipped_grad_weights_module=bm.get_handle_by_name("clipped_partial_grad_module_weights"),
                clipped_grad_biases_module=bm.get_handle_by_name("clipped_partial_grad_module_biases"),
                clipped_grad_temps=bm.get_handle_by_name("clipped_partial_grad_temps"),
            )
            clip_sig = ClipPartialGradientsGlobalNormSignature(
                bm,
                arch_consts.get("work_group_size_0", 256),
                scalar_bytes,
                handles=grad_handles,
                tile=tile,
                max_norm_global=SCALAR_NP_TYPE(h_params.max_grad_norm),
                epsilon=SCALAR_NP_TYPE(h_params.adam_epsilon),
            )
            clip_evt = ex.launch(q, clip_sig, wait_for=[raw_gh_ready_evt])
            all_clip_events.append(clip_evt)

        bm.release_transient_buffer(h_chunk_scratch_ref)
        bm.release_transient_buffer(h_mask_chunk_scratch_ref)
        bm.release_transient_buffer(raw_gh_chunk_scratch_ref)

        all_clips_done = cl.WaitForEvents(all_clip_events)

        # Step 4: Permute the now-filled monolithic buffer (Node 13)
        permute_sig = GatherAndPermuteGradHSignature(
            bm,
            clipped_partials_aos_ref=clipped_aos_ref,
            permuted_soa_out_ref=permuted_soa_ref,
            total_modules_count=np.uint32(spec.num_modules),
            hidden_count=np.uint32(spec.hidden_dim),
            total_batch_count=np.uint32(plan.effective_batch_size),
            num_module_chunks_count=np.uint32(plan.grid.num_module_chunks),
            modules_per_chunk_count=np.uint32(plan.grid.get_tile(0, 0).modules_per_chunk),
            num_class_chunks_count=np.uint32(plan.grid.num_class_chunks),
        )
        permute_evt = ex.launch(q, permute_sig, wait_for=[all_clips_done])

        # Step 5: Reduce the permuted buffer (Node 16)
        reduce_sig = ReduceGradHOverModulesSignature(
            bm,
            work_group_size_0=arch_consts.get("work_group_size_0", 256),
            scalar_size_bytes=scalar_bytes,
            permuted_soa_in_ref=permuted_soa_ref,
            final_grad_h_out_ref=final_summed_grad_h_ref,
            total_modules_count=np.uint32(spec.num_modules),
        )
        final_reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])

        print("    [GradHStreamingExecutor] 'Accumulate via Recompute' pipeline launched.")
        return final_summed_grad_h_ref, final_reduce_evt


class ReductionPhaseExecutor:
    """
    PhaseExecutor for the entire gradient reduction and aggregation stage.
    ... (docstring as before)
    """

    def __init__(self, services: Services, param_space: ParameterSpace):
        self.svs = services
        self.param_space = param_space

    def run(
        self,
        plan: ExecutionPlan,
        num_batch_chunks: int,
        deps: List[cl.Event],
        exclude_params: List[str] = None,
    ) -> Tuple[Dict[str, BufferHandle], cl.Event]:
        """
        Executes the generic reduction for all applicable parameter gradients.
        ... (docstring as before)
        """
        q, ex, bm = self.svs.q, self.svs.ex, self.svs.bm
        all_reduction_events = []
        summed_grad_handles = {}

        reduction_exec = ReductionTreeExecutor(
            q, ex, bm, AggregationManager(ex, bm, self.svs.arch_consts), plan.reduction_plan
        )
        exclude_params = exclude_params or []

        for flow in self.param_space:
            if flow.name in exclude_params or flow.specialized_reduction:
                continue

            clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
            summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)
            clipped_shape, _ = bm.get_spec(clipped_ref)
            elements_per_partial = int(np.prod(clipped_shape[1:]))

            if "shared" in flow.name:
                gather_prim = LinearlyChunkedGather(
                    num_chunks=num_batch_chunks, elements_per_chunk=elements_per_partial
                )
            else:
                gather_prim = TiledGather(grid=plan.grid, _elements_per_partial=elements_per_partial)

            reduce_evt = reduction_exec.execute(gather_prim, clipped_ref, summed_ref, wait_for=deps)
            all_reduction_events.append(reduce_evt)
            summed_grad_handles[flow.name] = summed_ref

        final_sync_event = (
            cl.WaitForEvents(all_reduction_events)
            if all_reduction_events
            else cl.UserEvent(q.context).set_status(cl.command_execution_status.COMPLETE)
        )
        return summed_grad_handles, final_sync_event

    def run_single_flow(
        self,
        flow_name: str,
        plan: ExecutionPlan,
        num_batch_chunks: int,
        deps: List[cl.Event],
    ) -> Tuple[BufferHandle, cl.Event]:
        """
        Executes the reduction for a single, specific parameter flow.
        ... (docstring as before)
        """
        q, ex, bm = self.svs.q, self.svs.ex, self.svs.bm
        spec, arch_consts = self.svs.model_spec, self.svs.arch_consts

        flow = next((f for f in self.param_space if f.name == flow_name), None)
        if flow is None:
            raise ValueError(f"Flow '{flow_name}' not found.")
        if not flow.specialized_reduction:
            raise ValueError(f"'{flow_name}' is not a specialized flow.")

        clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
        permuted_ref = bm.get_handle_by_name("permuted_grad_h")
        summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)

        # Step 1: Gather & Permute (Node 13) - COMPLETED
        permute_sig = GatherAndPermuteGradHSignature(
            bm,
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

        # Step 2: Reduce the permuted buffer (Node 16) - COMPLETED
        reduce_sig = ReduceGradHOverModulesSignature(
            bm,
            work_group_size_0=arch_consts.get("work_group_size_0", 256),
            scalar_size_bytes=spec.scalar_dtype().itemsize,
            permuted_soa_in_ref=permuted_ref,
            final_grad_h_out_ref=summed_ref,
            total_modules_count=np.uint32(spec.num_modules),
        )
        reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])

        return summed_ref, reduce_evt
