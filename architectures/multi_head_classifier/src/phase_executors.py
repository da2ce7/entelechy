# phase_executors.py

"""
A Toolbox of Tactical Phase Executors for the Training DAG.

This module provides a collection of dedicated "PhaseExecutor" classes. Each
class is a tactical expert responsible for executing a specific, cohesive part
of the computational graph (e.g., the forward pass, the per-tile module path,
the final update phase).

These classes are instantiated and called by the high-level `BatchProcessor`
(the "Conductor"), allowing the main orchestration logic to remain clean and
declarative. They encapsulate the details of creating and launching kernel
signatures.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from execution_plan import ExecutionPlan, WorkTile
from launcher_infra import BufferManager, KernelExecutor, BufferHandle
from kernel_signatures import *
from model_spec import ModelSpec


@dataclass
class Services:
    """A simple container to pass around core system components for dependency injection."""

    q: cl.CommandQueue
    ex: KernelExecutor
    bm: BufferManager
    model_spec: ModelSpec


class ForwardPassExecutor:
    """PhaseExecutor for the initial shared layer forward pass (Node 4)."""

    def __init__(self, services: Services):
        self.svs = services

    def run(self, batch_size: int, deps: List[cl.Event]) -> Tuple[BufferHandle, cl.Event]:
        """
        Executes the full forward pass. Intended for the "CACHE" strategy.
        Returns the handle to the output buffer and the completion event.
        """
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q

        sig = ForwardPassSignature(
            buffer_mgr=bm,
            # Injected constants
            simd_width=self.svs.model_spec.simd_width,
            local_mem_bank_padding=1,  # From contract
            scalar_size_bytes=np.dtype(self.svs.model_spec.scalar_dtype).itemsize,
            # Buffer Handles
            in_ref=bm.get_handle_by_name("input"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            w_ref=bm.get_handle_by_name("shared_weights"),
            b_ref=bm.get_handle_by_name("shared_biases"),
            h_out_ref=bm.get_handle_by_name("hidden_activations"),
            h_mask_out_ref=bm.get_handle_by_name("hidden_mask"),
            # Control Scalars
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
        spec = self.svs.model_spec
        arch_consts = plan.lifecycle_policy.compute_env.arch_consts  # Assumed accessible

        # 1. --- Resolve `hidden_activations` dependency using the plan's policy ---
        # This part of the code is already architecturally sound.
        h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)

        # 2. --- Launch Logits Rendering (Node 5) ---
        logits_sig = RenderLogitsChunkSignature(
            buffer_mgr=bm,
            h_ref=h_ref,
            h_mask_ref=bm.get_handle_by_name("hidden_mask"),
            w_ref=bm.get_handle_by_name("module_weights"),
            b_ref=bm.get_handle_by_name("module_biases"),
            logit_out_ref=bm.get_handle_by_name("logits"),
            # --- Per-tile control scalars ---
            batch_chunk_offset=np.uint32(0),  # Whole batch processed for this dependency
            batch_chunk_count=np.uint32(plan.effective_batch_size),
            module_chunk_offset=np.uint32(tile.module_chunk_offset),
            module_chunk_count=np.uint32(tile.modules_per_chunk),
            class_chunk_offset=np.uint32(tile.class_chunk_offset),
            class_chunk_count=np.uint32(tile.classes_per_chunk),
            # --- Global dimensional scalars ---
            hidden_count=np.uint32(spec.hidden_dim),
            total_output_class_count=np.uint32(spec.output_classes),
        )
        logits_evt = ex.launch(q, logits_sig, wait_for=[h_ready_evt])

        # 3. --- Launch Loss & Probabilities (Nodes 6 or 7) - Dynamically ---
        # Query the plan to decide which loss path to take.
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

        # 4. --- Launch Raw Partial Gradient Kernels (Nodes 8, 9, 10) - Dynamically ---
        # The choice of gradient kernel is also dependent on the problem type.
        if plan.problem_type == "CCE":
            grad_mod_sig = CalculateModuleParamGradsCceSignature(bm, ...)  # etc.
            grad_h_sig = BackpropErrorToHiddenChunkCceSignature(bm, ...)  # etc.
            grad_t_sig = CalculateChunkTempGradientsCceSignature(bm, ...)  # etc.
        else:  # BCE
            grad_mod_sig = CalculateModuleParamGradsBceSignature(bm, ...)  # etc.
            grad_h_sig = BackpropErrorToHiddenChunkBceSignature(bm, ...)  # etc.
            grad_t_sig = CalculateChunkTempGradientsBceSignature(bm, ...)  # etc.

        grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=[loss_evt])
        grad_h_evt = ex.launch(q, grad_h_sig, wait_for=[loss_evt])
        grad_t_evt = ex.launch(q, grad_t_sig, wait_for=[loss_evt])

        # 5. --- Launch Gradient Clipping (Node 11) - Dynamically ---
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

        # Query the plan to decide which clipping strategy to use.
        if plan.clipping_strategy == "GLOBAL":
            clip_sig = ClipPartialGradientsGlobalNormSignature(
                buffer_mgr=bm,
                work_group_size_0=arch_consts.get("work_group_size_0", 256),
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                handles=grad_handles,
                tile=tile,
                max_norm_global=np.float32(plan.max_grad_norm),
                epsilon=np.float32(plan.epsilon),
            )
        else:  # 'PER_ITEM'
            clip_sig = ClipPartialGradientsPerItemNormSignature(
                bm,
                # ...
                max_norm_per_item_ref=bm.get_handle_by_name("max_norm_per_item"),
                # ...
            )

        clip_event = ex.launch(q, clip_sig, wait_for=[grad_mod_evt, grad_h_evt, grad_t_evt])

        return clip_event


class SharedBackpropExecutor:
    """PhaseExecutor for streaming backpropagation through the shared layer (Nodes 17-19)."""

    def __init__(self, services: Services):
        self.svs = services

    def run(self, plan: ExecutionPlan, num_batch_chunks: int, deps: List[cl.Event]) -> cl.Event:
        """
        Runs the streaming backprop, producing CLIPPED partials for shared params.

        Returns a single event that completes when all chunks have been processed
        and their clipped partials are written to the collection buffers.
        """
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        spec = self.svs.model_spec
        arch_consts = plan.lifecycle_policy.compute_env.arch_consts  # Assumed accessible

        final_chunk_events = []

        # 1. --- Resolve Batch-Wide Dependencies (Once) ---
        # The fully summed Grad_H is required by all chunks, so we resolve it once upfront.
        print("    [SharedBProp] Resolving summed_grad_hidden_activations dependency...")
        grad_h_provider = plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
        grad_h_ref, grad_h_ready_evt = grad_h_provider.resolve(q, ex, wait_for=deps)

        # The Host Orchestrator decides the "Cache vs Recompute" strategy for hidden activations.
        # We query the provider here to get the handle to the full buffer if it's cached.
        h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)

        # 2. --- Acquire Transient Scratch Buffers ---
        # It's an architectural anti-pattern to write un-clipped gradients into a final
        # collection buffer. We use transient scratch space for one chunk's worth of raw output.
        gsw_chunk_shape = (spec.padded_input_dim, spec.padded_hidden_dim)
        gsb_chunk_shape = (spec.padded_hidden_dim,)
        gsw_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gsw_chunk_shape) * spec.scalar_dtype().itemsize))
        gsb_scratch_ref = bm.acquire_transient_buffer(int(np.prod(gsb_chunk_shape) * spec.scalar_dtype().itemsize))

        # 3. --- Main Streaming Loop ---
        batch_size = plan.effective_batch_size
        chunk_size = (batch_size + num_batch_chunks - 1) // num_batch_chunks

        for i in range(num_batch_chunks):
            batch_offset = i * chunk_size
            items_in_chunk = min(chunk_size, batch_size - batch_offset)
            if items_in_chunk <= 0:
                continue

            deps_for_chunk = [grad_h_ready_evt, h_ready_evt]

            # --- Step 3a: Compute Raw Partials (Nodes 17 & 18) ---
            # These kernels read from the full input/hidden/grad_h buffers but process only
            # a slice defined by the chunk scalars, writing their output to the scratch buffers.

            # EXPANDED: Node 17 `backprop_shared_weights_chunk`
            gsw_sig = BackpropSharedWeightsChunkSignature(
                buffer_mgr=bm,
                # Architectural Constants
                work_group_size_1=arch_consts.get("work_group_size_1", 16),
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                # Buffer Handles
                input_ref=bm.get_handle_by_name("input"),
                h_ref=h_ref,
                grad_h_ref=grad_h_ref,
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsw_out_ref=gsw_scratch_ref,  # CRITICAL: Write to scratch buffer
                # Control Scalars
                batch_chunk_offset=np.uint32(batch_offset),
                batch_chunk_count=np.uint32(items_in_chunk),
                batch_chunk_index=np.uint32(i),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            )
            gsw_evt = ex.launch(q, gsw_sig, wait_for=deps_for_chunk)

            # EXPANDED: Node 18 `backprop_shared_biases_chunk`
            gsb_sig = BackpropSharedBiasesChunkSignature(
                buffer_mgr=bm,
                # Architectural Constants
                work_group_size_0=arch_consts.get("work_group_size_0", 256),
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                # Buffer Handles
                h_ref=h_ref,
                grad_h_ref=grad_h_ref,
                mask_ref=bm.get_handle_by_name("sample_mask"),
                partial_gsb_out_ref=gsb_scratch_ref,  # CRITICAL: Write to scratch buffer
                # Control Scalars
                batch_chunk_offset=np.uint32(batch_offset),
                batch_chunk_count=np.uint32(items_in_chunk),
                batch_chunk_index=np.uint32(i),
                num_batch_chunks_count=np.uint32(num_batch_chunks),
            )
            gsb_evt = ex.launch(q, gsb_sig, wait_for=deps_for_chunk)

            # --- Step 3b: Clip Raw Partials and Place into Collection (Node 19) ---
            # This kernel reads from the scratch buffers and writes the clipped result
            # into the correct slice of the final, large collection buffer.
            dest_gsw_offset_elements = np.uint32(i * np.prod(gsw_chunk_shape))
            dest_gsb_offset_elements = np.uint32(i * np.prod(gsb_chunk_shape))

            # The clip kernel needs a unified offset for its concatenated view.
            # This is a small logical detail for the implementation. For simplicity,
            # let's assume separate offsets or a modified kernel is used.
            # Here we demonstrate the core idea.
            clip_sig = ClipSharedGradientsChunkSignature(
                buffer_mgr=bm,
                work_group_size_0=arch_consts.get("work_group_size_0", 256),
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                gsw_partial_in_ref=gsw_scratch_ref,
                gsb_partial_in_ref=gsb_scratch_ref,
                clipped_gsw_out_ref=bm.get_handle_by_name("clipped_partial_grad_shared_weights"),
                clipped_gsb_out_ref=bm.get_handle_by_name("clipped_partial_grad_shared_biases"),
                max_norm_global=np.float32(1.0),  # This would come from hyperparams
                epsilon=np.float32(1e-7),  # This would come from hyperparams
                dest_output_offset_elements=dest_gsw_offset_elements,  # Passing the crucial offset
            )
            clip_evt = ex.launch(q, clip_sig, wait_for=[gsw_evt, gsb_evt])
            final_chunk_events.append(clip_evt)

        # 4. --- Release Transient Resources ---
        bm.release_transient_buffer(gsw_scratch_ref)
        bm.release_transient_buffer(gsb_scratch_ref)

        # Return a single event that waits for all clipping operations to finish.
        return cl.WaitForEvents(final_chunk_events)


class UpdatePhaseExecutor:
    """PhaseExecutor for the final, batch-wide update stage (Nodes 20, 23-24)."""

    def __init__(self, services: Services):
        self.svs = services

    def run(
        self, step: int, summed_grads: Dict[str, BufferHandle], plan: ExecutionPlan, deps: List[cl.Event]
    ) -> cl.Event:
        """Executes the Normalize -> Adam Update -> Clamp sequence."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        param_names = ["shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"]

        # --- Normalize all summed gradients (Node 20) ---
        norm_events = []
        for name in param_names:
            norm_sig = NormalizeGradientsSignature(
                bm,
                summed_grad_ref=summed_grads[name],
                final_grad_out_ref=bm.get_handle_by_name(f"final_grad_{name}"),
                effective_batch_size=np.float32(plan.effective_batch_size),
                epsilon=np.float32(1e-7),
            )
            norm_events.append(ex.launch(q, norm_sig, wait_for=deps))
        all_norm_evt = cl.WaitForEvents(norm_events)

        # --- Apply Adam Optimizer Update (Node 23) ---
        beta1_t = np.float32(0.9**step)
        beta2_t = np.float32(0.999**step)
        update_events = []
        for name in param_names:
            pg = AdamParameterGroup(
                param_ref=bm.get_handle_by_name(name),
                grad_ref=bm.get_handle_by_name(f"final_grad_{name}"),
                m1_state_ref=bm.get_handle_by_name(f"m1_{name}"),
                m2_state_ref=bm.get_handle_by_name(f"m2_{name}"),
            )
            adam_sig = AdamUpdateSignature(
                bm,
                param_group=pg,
                learning_rate=np.float32(0.001),
                beta1=np.float32(0.9),
                beta2=np.float32(0.999),
                epsilon=np.float32(1e-7),
                beta1_pow_t=beta1_t,
                beta2_pow_t=beta2_t,
            )
            update_events.append(ex.launch(q, adam_sig, wait_for=[all_norm_evt]))
        all_updates_evt = cl.WaitForEvents(update_events)

        # --- Final Clamping (Node 24) ---
        clamp_sig = ClampTemperaturesSignature(
            bm, temps_ref=bm.get_handle_by_name("temperatures"), min_val=np.float32(0.1), max_val=np.float32(10.0)
        )
        clamp_evt = ex.launch(q, clamp_sig, wait_for=[all_updates_evt])

        return clamp_evt
