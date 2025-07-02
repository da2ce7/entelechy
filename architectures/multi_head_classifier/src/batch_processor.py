# batch_processor.py

from typing import Dict, Tuple

import numpy as np
import pyopencl as cl

# --- Architectural Imports ---
from execution_plan import ExecutionPlan
from launcher_infra import Services, HostView
from parameter_space import ParameterSpace
from . import graph_recipes as recipes
from workload_primitives import TiledGather, LinearlyChunkedGather


class BatchProcessor:
    """
    A transient object that executes a single, complete ExecutionPlan.
    This class is the "Conductor" of the DAG execution. It sequences the
    high-level recipes, but delegates all strategic and tactical decisions
    to the `ExecutionPlan` and the recipes themselves.
    """

    def __init__(self, services: Services, param_space: ParameterSpace, plan: ExecutionPlan):
        """
        Initializes the conductor.

        Args:
            services: A bundle of core system services (queue, executor, etc.).
            param_space: The manifest defining all learnable parameters.
            plan: The immutable `ExecutionPlan` for this specific batch.
        """
        self.svs = services
        self.plan = plan
        self.param_space = param_space

    def run(self, X_batch: np.ndarray, y_batch: np.ndarray, step: int) -> Tuple[cl.Event, cl.Event, HostView]:
        """
        Executes the full training step by directly calling the canonical recipes.
        The logic is a clean, linear sequence of computational phases, with two
        primary, parallel event chains: one for the learning update and one for
        asynchronous diagnostic retrieval.

        Returns:
            A tuple containing the events and handles for asynchronous interaction:
            1. `final_learn_event`: Signals completion of the entire backprop and update cycle.
            2. `inference_event`: Signals completion of the D2H copy for `Final Probs`.
            3. `final_probs_view`: The `HostView` object to call `.get()` on after
                                   `inference_event` completes.
        """
        q, bm = self.svs.q, self.svs.ex

        # Phase 1: Initial Data Uploads & Async Dependency Calculation
        upload_x_evt = cl.enqueue_copy(q, bm.get_cl_buffer("input"), X_batch)
        targets_buffer_name = self.plan.problem_type.required_targets_buffer_name
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer(targets_buffer_name), y_batch)

        effective_bs_view, effective_bs_ready_evt = recipes.compute_effective_batch_size(
            svs=self.svs,
            batch_size=X_batch.shape[0],
            deps=[upload_y_evt],
        )

        h_provider = self.plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, self.svs.ex, wait_for=[upload_x_evt, upload_y_evt])

        # Phase 2: Parallel Partial Gradient Production
        prob_results = [
            recipes.build_forward_module_path(self.svs, tile, self.plan, h_ref, h_ready_evt) for tile in self.plan.grid
        ]
        prob_events = [res[1] for res in prob_results]
        all_probs_ready_evt = cl.WaitForEvents(prob_events)

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
            all_module_grads_clipped_evt = all_probs_ready_evt

        shared_grads_clipped_evt = recipes.build_shared_backprop_subgraph(self.svs, self.plan, deps=[h_ready_evt])

        all_partials_ready_evt = cl.WaitForEvents([all_module_grads_clipped_evt, shared_grads_clipped_evt])

        # Phase 3: Aggregation & Result Retrieval
        reduction_events = []
        summed_grad_handles = {}

        diag_agg_events = recipes.execute_diagnostic_aggregation(
            svs=self.svs, plan=self.plan, deps=[all_probs_ready_evt]
        )
        reduction_events.extend(diag_agg_events.values())

        final_probs_ref = self.svs.bm.get_handle_by_name("final_probs")
        final_probs_shape, dtype = self.svs.bm.get_spec(final_probs_ref)
        final_probs_view = HostView(padded_shape=final_probs_shape, dtype=dtype, real_shape=final_probs_shape)
        inference_event = final_probs_view.enqueue_read(
            q, bm.get_cl_buffer(final_probs_ref), wait_for=[diag_agg_events["probs"]]
        )

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

        grad_h_provider = self.plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
        grad_h_handle, grad_h_evt = grad_h_provider.resolve(q, self.svs.ex, wait_for=[all_partials_ready_evt])
        reduction_events.append(grad_h_evt)
        summed_grad_handles["hidden_activations"] = grad_h_handle

        all_grads_summed_evt = cl.WaitForEvents(reduction_events)

        # Phase 4: Final Batch-Wide Update
        effective_bs_ready_evt.wait()
        effective_batch_size_scalar = effective_bs_view.get()[0]

        update_deps = [all_grads_summed_evt, effective_bs_ready_evt]
        final_learn_event = recipes.build_update_subgraph(
            svs=self.svs,
            param_space=self.param_space,
            step=step,
            summed_grads=summed_grad_handles,
            effective_batch_size=float(effective_batch_size_scalar),
            deps=update_deps,
        )

        return final_learn_event, inference_event, final_probs_view
