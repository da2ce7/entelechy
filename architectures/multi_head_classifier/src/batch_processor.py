# batch_processor.py

"""
(REV 2) This module provides the `BatchProcessor`,
the "Conductor" of the training process, which executes a single `ExecutionPlan`.

This version is architecturally complete. It is now a pure Conductor, fully
decoupled from the specifics of any problem type. It delegates all
strategy-specific decisions (such as target buffer selection and loss
aggregation) to the polymorphic `ProblemTypeStrategy` object provided in the plan.
This fulfills the system's core principles by making the `BatchProcessor` a
generic sequencer of abstract recipes.
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
        Initializes the conductor. It is a self-sufficient orchestrator of recipes.
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

        # Simply ask the plan's strategy for the correct buffer name.
        targets_buffer_name = self.plan.problem_type.required_targets_buffer_name
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer(targets_buffer_name), y_batch)

        initial_deps = [upload_x_evt, upload_y_evt]

        # The Conductor blindly resolves the dependency for `hidden_activations`.
        # The plan's policy dictates whether this returns a cached handle instantly
        # or launches a recomputation kernel.
        h_provider = self.plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, self.svs.ex, wait_for=initial_deps)

        # --- Phase 2: Parallel Partial Gradient Production ---
        # The Conductor launches all the necessary forward and backward passes
        # to produce the full set of clipped, partial gradients.

        # 2a. Run the forward module path for every tile to get probabilities.
        prob_results = [
            recipes.build_forward_module_path(self.svs, tile, self.plan, h_ref, h_ready_evt) for tile in self.plan.grid
        ]
        prob_events = [res[1] for res in prob_results]
        all_probs_ready_evt = cl.WaitForEvents(prob_events)

        # 2b. Run the appropriate backward paths based on the strategic plan.
        if self.plan.adaptation_strategy == "CACHE":
            prob_refs = [res[0] for res in prob_results]
            bwd_mod_events = [
                recipes.build_backward_module_path(
                    self.svs, tile, self.plan, h_ref, prob_refs[i], [all_probs_ready_evt]
                )
                for i, tile in enumerate(self.plan.grid)
            ]
            all_module_grads_clipped_evt = cl.WaitForEvents(bwd_mod_events)
        else:
            # If not caching, the grad_h stream will handle this path.
            # We just need to sync on the probability calculations.
            all_module_grads_clipped_evt = all_probs_ready_evt

        # 2c. Launch the shared-layer backpropagation stream. This is always run.
        shared_grads_clipped_evt = recipes.build_shared_backprop_subgraph(self.svs, self.plan, deps=[h_ready_evt])

        # Synchronization point: Wait for all parallel tracks to finish their partials.
        all_partials_ready_evt = cl.WaitForEvents([all_module_grads_clipped_evt, shared_grads_clipped_evt])
        print("    [Conductor] Sync Point 1: All partial gradients produced and clipped.")

        # --- Phase 3: Aggregation via Differentiated Reduction ---
        reduction_events = []
        summed_grad_handles = {}

        # Blindly delegate the entire loss aggregation step to the strategy.
        # The `build_loss_aggregation_subgraph` method will do nothing if no
        # reduction is needed (e.g., for CCE), returning None.
        loss_sum_evt = self.plan.problem_type.build_loss_aggregation_subgraph(
            svs=self.svs, plan=self.plan, deps=[all_probs_ready_evt]
        )
        if loss_sum_evt:
            reduction_events.append(loss_sum_evt)

        # Use the STABILIZED reduction engine for all gradient parameters.
        print("    [Conductor] Aggregating gradients (stabilized reduction)...")
        # 3a. Aggregate the shared-path gradients.
        for flow in self.param_space:
            if flow.name in ["shared_weights", "shared_biases"]:
                clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
                summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)
                clipped_shape, _ = bm.get_spec(clipped_ref)
                elements_per_partial = int(np.prod(clipped_shape[1:]))
                gather_prim = LinearlyChunkedGather(self.plan.shared_backprop_stream_chunks, elements_per_partial)

                evt = recipes.execute_stabilized_reduction_tree(
                    svs=self.svs,
                    plan=self.plan,
                    gather_primitive=gather_prim,
                    partial_collection_ref=clipped_ref,
                    final_dest_handle=summed_ref,
                    wait_for=[all_partials_ready_evt],
                )
                reduction_events.append(evt)
                summed_grad_handles[flow.name] = summed_ref

        # 3b. Aggregate the module-path gradients. This logic must run for BOTH "CACHE"
        #     and "RECOMPUTE_GRAD_H" strategies, as both produce the necessary partials.
        for flow in self.param_space:
            if flow.name in ["module_weights", "module_biases", "temperatures"]:
                clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
                summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)
                clipped_shape, _ = bm.get_spec(clipped_ref)
                elements_per_partial = int(np.prod(clipped_shape[1:]))
                gather_prim = TiledGather(self.plan.grid, elements_per_partial)

                evt = recipes.execute_stabilized_reduction_tree(
                    svs=self.svs,
                    plan=self.plan,
                    gather_primitive=gather_prim,
                    partial_collection_ref=clipped_ref,
                    final_dest_handle=summed_ref,
                    wait_for=[all_partials_ready_evt],
                )
                reduction_events.append(evt)
                summed_grad_handles[flow.name] = summed_ref

        # 3c. Resolve the dependency for `summed_grad_hidden_activations`.
        grad_h_provider = self.plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
        grad_h_handle, grad_h_evt = grad_h_provider.resolve(q, self.svs.ex, wait_for=[all_partials_ready_evt])
        reduction_events.append(grad_h_evt)

        all_grads_summed_evt = cl.WaitForEvents(reduction_events)
        print("    [Conductor] Sync Point 2: All data aggregated.")

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
