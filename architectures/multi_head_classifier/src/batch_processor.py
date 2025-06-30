# batch_processor.py

"""
The Definitive Implementation of the Batch Processing Conductor.

This module provides the `BatchProcessor` class, a transient object whose sole
responsibility is to execute a single, complete `ExecutionPlan`. It acts as the
"Conductor" of the training process, directly consuming the pure, stateless functions
from the `graph_recipes` module to orchestrate the entire computational sequence.

This class is the embodiment of Jurisdictional Purity: it contains no
algorithmic logic itself, only the high-level orchestration logic to sequence
the recipes correctly based on the strategic plan it is given.
"""

from typing import Dict, List

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from execution_plan import ExecutionPlan
from launcher_infra import Services
from parameter_space import ParameterSpace
from . import graph_recipes as recipes
from workload_primitives import TiledGather, LinearlyChunkedGather


class BatchProcessor:
    """
    A transient object that executes a single, complete ExecutionPlan.
    This class is the "Conductor" of the DAG execution.
    """

    def __init__(self, services: Services, param_space: ParameterSpace, plan: ExecutionPlan):
        """
        Initializes the conductor. It no longer needs a team of executors,
        as it is now a self-sufficient orchestrator of recipes.
        """
        self.svs = services
        self.plan = plan
        self.param_space = param_space

    def run(self, X_batch: np.ndarray, y_batch: np.ndarray, step: int) -> cl.Event:
        """
        Executes the full training step by directly calling the canonical recipes.
        The logic is a clean, linear sequence of computational phases.
        """
        q, bm = self.svs.q, self.svs.ex
        print("    [Conductor] Beginning batch processing...")

        # --- Phase 1: Initial Data Uploads & Strategic Dependency Resolution ---
        upload_x_evt = cl.enqueue_copy(q, bm.get_cl_buffer("input"), X_batch)
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer("targets_cce"), y_batch)
        initial_deps = [upload_x_evt, upload_y_evt]

        # The Conductor blindly resolves the dependency for `hidden_activations`.
        # The plan's policy dictates whether this returns a cached handle instantly
        # or launches a recomputation kernel.
        h_provider = self.plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, self.svs.ex, wait_for=initial_deps)

        # --- Phase 2: Parallel Partial Gradient Production ---
        # The Conductor launches all the necessary forward and backward passes
        # to produce the full set of clipped, partial gradients.

        # 2a. Run the forward module path for every tile to get probabilities. This
        # is required by all subsequent backprop steps.
        prob_results = [
            recipes.build_forward_module_path(self.svs, tile, self.plan, h_ref, h_ready_evt) for tile in self.plan.grid
        ]
        prob_events = [res[1] for res in prob_results]
        all_probs_ready_evt = cl.WaitForEvents(prob_events)

        # 2b. Run the appropriate backward paths based on the strategic plan.
        if self.plan.adaptation_strategy == "CACHE":
            # In CACHE mode, we run a distinct backward pass for the module path.
            prob_refs = [res[0] for res in prob_results]
            bwd_mod_events = [
                recipes.build_backward_module_path(
                    self.svs, tile, self.plan, h_ref, prob_refs[i], [all_probs_ready_evt]
                )
                for i, tile in enumerate(self.plan.grid)
            ]
            all_module_grads_clipped_evt = cl.WaitForEvents(bwd_mod_events)
        else:  # "RECOMPUTE_GRAD_H"
            # In RECOMPUTE_GRAD_H mode, the specialized streaming pipeline for Grad_H
            # will handle the module-path gradients implicitly. We simply pass the
            # probability-ready event as the dependency for this stage.
            all_module_grads_clipped_evt = all_probs_ready_evt

        # 2c. Launch the shared-layer backpropagation stream. This is always run.
        shared_grads_clipped_evt = recipes.build_shared_backprop_subgraph(self.svs, self.plan, deps=[h_ready_evt])

        # Synchronization point: Wait for all partial gradients to be produced.
        all_partials_ready_evt = cl.WaitForEvents([all_module_grads_clipped_evt, shared_grads_clipped_evt])
        print("    [Conductor] Sync Point 1: All partial gradients produced and clipped.")

        # --- Phase 3: Gradient Aggregation (Reduction) ---
        reduction_events = []
        summed_grad_handles = {}

        # 3a. Aggregate the shared-path gradients using the generic reduction tree.
        for flow in self.param_space:
            if flow.name in ["shared_weights", "shared_biases"]:
                clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
                summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)
                clipped_shape, _ = bm.get_spec(clipped_ref)
                elements = int(np.prod(clipped_shape[1:]))
                gather_prim = LinearlyChunkedGather(self.plan.shared_backprop_stream_chunks, elements)
                evt = recipes.execute_reduction_tree(
                    self.svs, self.plan.reduction_plan, gather_prim, clipped_ref, summed_ref, [all_partials_ready_evt]
                )
                reduction_events.append(evt)
                summed_grad_handles[flow.name] = summed_ref

        # 3b. Aggregate the module-path gradients ONLY if not in recompute mode.
        if self.plan.adaptation_strategy == "CACHE":
            for flow in self.param_space:
                if flow.name in ["module_weights", "module_biases", "temperatures"]:
                    clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
                    summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)
                    clipped_shape, _ = bm.get_spec(clipped_ref)
                    elements = int(np.prod(clipped_shape[1:]))
                    gather_prim = TiledGather(self.plan.grid, elements)
                    evt = recipes.execute_reduction_tree(
                        self.svs,
                        self.plan.reduction_plan,
                        gather_prim,
                        clipped_ref,
                        summed_ref,
                        [all_partials_ready_evt],
                    )
                    reduction_events.append(evt)
                    summed_grad_handles[flow.name] = summed_ref

        # 3c. Resolve the dependency for `summed_grad_hidden_activations`.
        # The plan's policy dictates whether this calls the simple permute-and-reduce recipe
        # (for CACHE) or the entire streaming pipeline (for RECOMPUTE_GRAD_H).
        grad_h_provider = self.plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
        grad_h_handle, grad_h_evt = grad_h_provider.resolve(q, self.svs.ex, wait_for=[all_partials_ready_evt])
        reduction_events.append(grad_h_evt)
        summed_grad_handles["hidden_activations"] = grad_h_handle

        all_grads_summed_evt = cl.WaitForEvents(reduction_events)
        print("    [Conductor] Sync Point 2: All gradients aggregated.")

        # --- Phase 4: Final Batch-Wide Update ---
        final_event = recipes.build_update_subgraph(
            svs=self.svs,
            param_space=self.param_space,
            step=step,
            summed_grads=summed_grad_handles,
            plan=self.plan,
            deps=[all_grads_summed_evt],
        )
        print("    [Conductor] Final update phase launched.")

        return final_event
