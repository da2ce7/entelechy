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
            local_mem_bank_padding=1, # From contract
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

        # 1. Resolve `hidden_activations` dependency using the plan's policy
        h_provider = plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=deps)

        # 2. Launch Logits Rendering (Node 5)
        logits_sig = RenderLogitsChunkSignature(bm,
            h_ref=h_ref, h_mask_ref=..., w_ref=..., b_ref=..., logit_out_ref=..., # Simplified
            # ... all other necessary scalars ...
        )
        logits_evt = ex.launch(q, logits_sig, wait_for=[h_ready_evt])

        # 3. Launch Loss & Probabilities (Node 6)
        loss_sig = ComputeProbsLossCceChunkSignature(bm,
            logit_ref=logits_sig.logit_out_ref,
            temp_ref=bm.get_handle_by_name("temperatures"),
            target_ref=bm.get_handle_by_name("targets"),
            mask_ref=bm.get_handle_by_name("sample_mask"),
            prob_out_ref=bm.get_handle_by_name("partial_probs"),
            loss_out_ref=bm.get_handle_by_name("final_loss"),
            tile=tile,
            total_output_class_count=np.uint32(spec.output_classes)
        )
        loss_evt = ex.launch(q, loss_sig, wait_for=[logits_evt])

        # 4. Launch Raw (Un-clipped) Partial Gradient Kernels (Nodes 8, 9, 10)
        grad_mod_sig = CalculateModuleParamGradsCceSignature(bm, h_ref=h_ref, prob_ref=loss_sig.prob_out_ref, tile=tile, ...)
        grad_mod_evt = ex.launch(q, grad_mod_sig, wait_for=[loss_evt])

        grad_h_sig = BackpropErrorToHiddenChunkCceSignature(bm, prob_ref=loss_sig.prob_out_ref, tile=tile, ...)
        grad_h_evt = ex.launch(q, grad_h_sig, wait_for=[loss_evt])

        grad_t_sig = CalculateChunkTempGradientsCceSignature(bm, prob_ref=loss_sig.prob_out_ref, logit_ref=logits_sig.logit_out_ref, tile=tile,...)
        grad_t_evt = ex.launch(q, grad_t_sig, wait_for=[loss_evt])

        # 5. Launch Gradient Clipping (Node 11)
        grad_handles = GradientHandles(
            grad_weights_module=grad_mod_sig.gw_out_ref,
            grad_biases_module=grad_mod_sig.gb_out_ref,
            grad_temps=grad_t_sig.gt_out_ref,
            grad_hidden_activations_aos=grad_h_sig.gh_out_ref,
            clipped_grad_weights_module=bm.get_handle_by_name("clipped_partial_grad_weights_module"),
            clipped_grad_biases_module=bm.get_handle_by_name("clipped_partial_grad_biases_module"),
            clipped_grad_temps=bm.get_handle_by_name("clipped_partial_grad_temps"),
            clipped_grad_hidden_activations_aos=bm.get_handle_by_name("clipped_partial_grad_hidden_activations_aos")
        )
        clip_sig = ClipPartialGradientsGlobalNormSignature(bm, handles=grad_handles, tile=tile, ...)
        clip_event = ex.launch(q, clip_sig, wait_for=[grad_mod_evt, grad_h_evt, grad_t_evt])

        return clip_event

class SharedBackpropExecutor:
    """PhaseExecutor for streaming backpropagation through the shared layer (Nodes 17-18)."""
    def __init__(self, services: Services):
        self.svs = services

    def run(self, plan: ExecutionPlan, num_batch_chunks: int, deps: List[cl.Event]) -> cl.Event:
        """Runs the streaming backprop, producing clipped partials for shared params."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        chunk_completion_events = []

        # Resolve the fully summed Grad_H dependency once.
        grad_h_provider = plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
        grad_h_ref, grad_h_ready_evt = grad_h_provider.resolve(q, ex, wait_for=deps)

        for i in range(num_batch_chunks):
            # In a real impl, this would resolve providers for input_chunk_i, hidden_chunk_i etc.
            # This is the "True Streaming" model in action.
            input_chunk_provider=...
            input_ref, input_ready_evt = input_chunk_provider.resolve(...)
            hidden_chunk_provider=...
            hidden_ref, hidden_ready_evt = hidden_chunk_provider.resolve(...)

            deps_for_chunk = [grad_h_ready_evt, input_ready_evt, hidden_ready_evt]

            # Launch kernels to get raw partials SW and SB for this chunk (Nodes 17, 18)
            gsw_sig = BackpropSharedWeightsChunkSignature(bm, input_ref=input_ref, h_ref=hidden_ref, grad_h_ref=grad_h_ref, ...)
            gsw_evt = ex.launch(q, gsw_sig, wait_for=deps_for_chunk)
            gsb_sig = BackpropSharedBiasesChunkSignature(bm, h_ref=hidden_ref, grad_h_ref=grad_h_ref, ...)
            gsb_evt = ex.launch(q, gsb_sig, wait_for=deps_for_chunk)

            # Immediately clip these new partials
            # This would require a specialized version of ClipPartialGradients
            clip_sw_sb_sig = ...
            clip_evt = ex.launch(q, clip_sw_sb_sig, wait_for=[gsw_evt, gsb_evt])
            chunk_completion_events.append(clip_evt)

        return cl.WaitForEvents(chunk_completion_events)


class UpdatePhaseExecutor:
    """PhaseExecutor for the final, batch-wide update stage (Nodes 20, 23-24)."""
    def __init__(self, services: Services):
        self.svs = services

    def run(self, step: int, summed_grads: Dict[str, BufferHandle], plan: ExecutionPlan, deps: List[cl.Event]) -> cl.Event:
        """Executes the Normalize -> Adam Update -> Clamp sequence."""
        bm, ex, q = self.svs.bm, self.svs.ex, self.svs.q
        param_names = ["shared_weights", "shared_biases", "module_weights", "module_biases", "temperatures"]

        # --- Normalize all summed gradients (Node 20) ---
        norm_events = []
        for name in param_names:
            norm_sig = NormalizeGradientsSignature(bm,
                summed_grad_ref=summed_grads[name],
                final_grad_out_ref=bm.get_handle_by_name(f"final_grad_{name}"),
                effective_batch_size=np.float32(plan.effective_batch_size),
                epsilon=np.float32(1e-7)
            )
            norm_events.append(ex.launch(q, norm_sig, wait_for=deps))
        all_norm_evt = cl.WaitForEvents(norm_events)

        # --- Apply Adam Optimizer Update (Node 23) ---
        beta1_t = np.float32(0.9 ** step)
        beta2_t = np.float32(0.999 ** step)
        update_events = []
        for name in param_names:
            pg = AdamParameterGroup(
                param_ref=bm.get_handle_by_name(name),
                grad_ref=bm.get_handle_by_name(f"final_grad_{name}"),
                m1_state_ref=bm.get_handle_by_name(f"m1_{name}"),
                m2_state_ref=bm.get_handle_by_name(f"m2_{name}")
            )
            adam_sig = AdamUpdateSignature(bm, param_group=pg,
                learning_rate=np.float32(0.001),
                beta1=np.float32(0.9), beta2=np.float32(0.999),
                epsilon=np.float32(1e-7), beta1_pow_t=beta1_t, beta2_pow_t=beta2_t)
            update_events.append(ex.launch(q, adam_sig, wait_for=[all_norm_evt]))
        all_updates_evt = cl.WaitForEvents(update_events)

        # --- Final Clamping (Node 24) ---
        clamp_sig = ClampTemperaturesSignature(bm,
            temps_ref=bm.get_handle_by_name("temperatures"),
            min_val=np.float32(0.1),
            max_val=np.float32(10.0)
        )
        clamp_evt = ex.launch(q, clamp_sig, wait_for=[all_updates_evt])

        return clamp_evt
