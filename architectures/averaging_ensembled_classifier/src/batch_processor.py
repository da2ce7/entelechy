# batch_processor.py

"""
The Definitive Conductor of a Single, Complete Execution Plan.

Jurisdictional Mandate:
This module provides the `BatchProcessor`, a class whose sole purpose is to
execute a single, immutable `ExecutionPlan`. It acts as the "Conductor" in the
system's orchestra, responsible for sequencing the high-level computational
"recipes" (found in `graph_recipes.py`).

Architectural Role:
The `BatchProcessor` embodies the principle of delegated responsibility. It
makes no strategic decisions; its role is purely tactical. It receives the
authoritative plan from the `TrainingOrchestrator` and translates it into a
series of asynchronous commands on the device queue. This class is transient,
created for a single training step and then discarded, ensuring that no state
is carried between batches.
"""

from typing import Dict, List, Tuple

import numpy as np
import pyopencl as cl

# --- Foundational Imports from Sibling Architectural Modules ---
from .execution_plan import ExecutionPlan
from .launcher_infra import Services, HostView, BufferHandle
from .parameter_space import ParameterSpace
from .workload_primitives import GatherPrimitive, TiledGather, LinearlyChunkedGather
from . import graph_recipes as recipes


class BatchProcessor:
    """A transient object that executes a single, complete ExecutionPlan."""

    def __init__(self, services: Services, param_space: ParameterSpace, plan: ExecutionPlan):
        """Initializes the conductor via dependency injection of core services."""
        self.svs = services
        self.plan = plan
        self.param_space = param_space

    def run(self, X_batch: np.ndarray, y_batch: np.ndarray, step: int) -> Tuple[cl.Event, cl.Event, HostView]:
        """
        Executes the full training step by orchestrating the canonical recipes.

        Architectural Mandate:
        The logic herein is a direct, linear expression of the computational
        DAG defined in the `CONCEPT.md` document. It is responsible for
        managing the data dependencies between recipes, primarily by using
        `cl.enqueue_barrier` to create explicit synchronization points between
        the major phases of the computation.
        """
        q, bm, ex = self.svs.q, self.svs.bm, self.svs.ex

        # --- Phase 1: Initial Data Uploads & Async Dependency Calculation ---
        # WHY: The GPU "input" buffer has shape (batch_size, padded_input_dim),
        # but the host array X_batch has shape (batch_size, input_dim) where
        # input_dim <= padded_input_dim.  A raw enqueue_copy of the unpadded
        # array would pack samples contiguously in flat memory, misaligning
        # rows relative to the padded stride the kernel expects.  We must
        # expand to padded width so that each row occupies exactly
        # padded_input_dim elements, with zeros in the padding columns.
        padded_input_dim = self.svs.model_spec.padded_input_dim
        if X_batch.shape[1] < padded_input_dim:
            padded_X = np.zeros(
                (X_batch.shape[0], padded_input_dim), dtype=X_batch.dtype
            )
            padded_X[:, :X_batch.shape[1]] = X_batch
        else:
            padded_X = X_batch
        upload_x_evt = cl.enqueue_copy(q, bm.get_cl_buffer("input"), padded_X)
        targets_buffer_name = self.plan.problem_type.required_targets_buffer_name
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer(targets_buffer_name), y_batch)

        effective_bs_view, effective_bs_ready_evt = recipes.compute_effective_batch_size(
            svs=self.svs,
            plan=self.plan,
            deps=[upload_y_evt],
        )

        h_provider = self.plan.lifecycle_policy.get_provider("hidden_activations")
        h_ref, h_ready_evt = h_provider.resolve(q, ex, wait_for=[upload_x_evt, upload_y_evt])

        # --- Phase 2: Parallel Partial Gradient Production ---
        prob_results = [
            recipes.build_forward_module_path(self.svs, tile, self.plan, h_ref, h_ready_evt) for tile in self.plan.grid
        ]
        prob_events = [res[1] for res in prob_results]
        # WHY: This barrier is a critical synchronization point. It ensures that
        # all parallel forward-pass computations are complete before any
        # backpropagation begins, fulfilling a fundamental data dependency in the DAG.
        all_probs_ready_evt = cl.enqueue_barrier(q, wait_for=prob_events)

        if self.plan.adaptation_strategy == "CACHE":
            prob_refs = [res[0] for res in prob_results]
            bwd_mod_events = [
                recipes.build_backward_module_path(
                    self.svs, tile, self.plan, h_ref, prob_refs[i], [all_probs_ready_evt]
                )
                for i, tile in enumerate(self.plan.grid)
            ]
            all_module_grads_clipped_evt = cl.enqueue_barrier(q, wait_for=bwd_mod_events)
        elif self.plan.adaptation_strategy == "RECOMPUTE_GRAD_H":
            module_grad_events_map = recipes.build_streaming_module_grad_path(
                svs=self.svs, plan=self.plan, deps=[all_probs_ready_evt]
            )
            all_module_grads_clipped_evt = module_grad_events_map["clipped_partial_grad_hidden_activations"]
        else:
            raise ValueError(f"Unknown adaptation strategy: '{self.plan.adaptation_strategy}'")

        shared_grads_clipped_evt = recipes.build_shared_backprop_subgraph(self.svs, self.plan, deps=[h_ready_evt])

        # WHY: This is the second major synchronization point, joining the two
        # independent gradient streams (module path and shared path).
        all_partials_ready_evt = cl.enqueue_barrier(q, wait_for=[all_module_grads_clipped_evt, shared_grads_clipped_evt])

        # --- Phase 3: Aggregation & Result Retrieval ---
        reduction_events: List[cl.Event] = []
        summed_grad_handles: Dict[str, BufferHandle] = {}

        diag_agg_events = recipes.execute_diagnostic_aggregation(
            svs=self.svs, plan=self.plan, deps=[all_probs_ready_evt]
        )
        reduction_events.extend(diag_agg_events.values())

        final_probs_ref = bm.get_handle_by_name("final_probs")
        final_probs_shape, dtype = bm.get_spec(final_probs_ref)
        final_probs_view = HostView(padded_shape=final_probs_shape, dtype=dtype, real_shape=final_probs_shape)
        inference_event = final_probs_view.enqueue_read(
            q, bm.get_cl_buffer(final_probs_ref), wait_for=[diag_agg_events["probs"]]
        )

        for flow in self.param_space:
            if flow.specialized_reduction:
                continue

            # WHY: This explicit type hint clarifies our intent to the static
            # analyzer. We are declaring that `gather_prim` will hold an object
            # that conforms to the `GatherPrimitive` abstract contract, resolving
            # the apparent type conflict between assignments in the loop.
            gather_prim: GatherPrimitive
            clipped_ref = bm.get_handle_by_name(flow.clipped_partial_grad_buffer_name)
            summed_ref = bm.get_handle_by_name(flow.summed_grad_buffer_name)
            clipped_shape, _ = bm.get_spec(clipped_ref)
            elements_per_partial = int(np.prod(clipped_shape[1:]))

            if flow.name in ["shared_weights", "shared_biases"]:
                gather_prim = LinearlyChunkedGather(self.plan.shared_backprop_stream_chunks, elements_per_partial)
            else:
                gather_prim = TiledGather(self.plan.grid, _elements_per_partial=elements_per_partial)

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
        grad_h_handle, grad_h_evt = grad_h_provider.resolve(q, ex, wait_for=[all_partials_ready_evt])
        reduction_events.append(grad_h_evt)
        summed_grad_handles["hidden_activations"] = grad_h_handle

        all_grads_summed_evt = cl.enqueue_barrier(q, wait_for=reduction_events)

        # --- Phase 4: Final Batch-Wide Update ---
        # The .get() call implicitly blocks until the calculation is complete.
        effective_batch_size_scalar = effective_bs_view.get()[0]

        final_learn_event = recipes.build_update_subgraph(
            svs=self.svs,
            param_space=self.param_space,
            step=step,
            summed_grads=summed_grad_handles,
            plan=self.plan,
            effective_batch_size=float(effective_batch_size_scalar),
            deps=[all_grads_summed_evt, effective_bs_ready_evt],
        )

        return final_learn_event, inference_event, final_probs_view
