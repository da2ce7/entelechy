# main_orchestrator.py

"""
The Definitive, Unified Streaming Classification Engine (Host Implementation).

(REV 3 - ARCHITECTURALLY FINALIZED) This version represents the fully harmonized
architecture, including the implementation of the memory-constrained "Accumulate
via Recompute" streaming model for Grad_H.

The `TrainingOrchestrator` is a pure strategist, authoring high-level,
declarative execution plans.

The `BatchProcessor` is a pure conductor, supervising a linear sequence
of expert `PhaseExecutor` objects, each responsible for one cohesive stage of
the training algorithm. The final separation of concerns is complete.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import pyopencl as cl
from sklearn.datasets import load_iris

# --- Foundational & Architectural Imports ---
from model_spec import ModelSpec, SCALAR_DTYPE
from cl_context_manager import OpenCLContextManager, ComputeEnvironment
from parameter_space import ParameterSpace
from execution_plan import *
from launcher_infra import *
from memory_layout import *
from kernel_signatures import *
from compute_patterns import *
from phase_executors import *

# This constant is specific to the execution strategy of the BatchProcessor
SHARED_BACKPROP_STREAM_CHUNKS = 4


# === Dataclasses for Structured Configuration ===
@dataclass(frozen=True)
class TrainingHyperparams:
    """A structured container for all training hyperparameters."""

    epochs: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float
    temp_min: float
    temp_max: float


class BatchProcessor:
    """
    A transient object that executes a single, complete ExecutionPlan.
    This class is the "Conductor" of the DAG execution.
    """

    def __init__(self, services: Services, param_space: ParameterSpace, plan: ExecutionPlan):
        """Initializes the conductor and its team of expert phase executors."""
        self.svs = services
        self.plan = plan
        self.param_space = param_space
        self.mod_path_exec = ModulePathExecutor(services)
        self.shared_bprop_exec = SharedBackpropExecutor(services)
        self.reduction_exec = ReductionPhaseExecutor(services, param_space)
        self.update_exec = UpdatePhaseExecutor(services, param_space)

    def run(self, X_batch: np.ndarray, y_batch: np.ndarray, step: int) -> cl.Event:
        """
        Executes the full training step in a clean, supervised, linear sequence.
        """
        q, bm = self.svs.q, self.svs.bm
        print("    [Conductor] Beginning batch processing...")

        # Initial data uploads
        upload_x_evt = cl.enqueue_copy(q, bm.get_cl_buffer("input"), X_batch)
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer("targets_cce"), y_batch)
        initial_deps = [upload_x_evt, upload_y_evt]

        # --- Phase 1: Parallel Partial Gradient Production ---
        # The standard module path and shared backprop paths still produce their partials.
        # If Grad_H is being streamed, it will rely on the partial_probs produced in this phase.
        tile_completion_events = [
            self.mod_path_exec.run_for_tile(tile, self.plan, deps=initial_deps) for tile in self.plan.grid
        ]
        all_tiles_clipped_evt = cl.WaitForEvents(tile_completion_events)

        # The shared backprop path needs `hidden_activations`, which it resolves via its own provider.
        # However, its primary dependency is the *final* summed_grad_h.
        grad_h_provider = self.plan.lifecycle_policy.get_provider("summed_grad_hidden_activations")
        summed_grad_h_ref, summed_grad_h_evt = grad_h_provider.resolve(q, self.svs.ex, wait_for=initial_deps)

        shared_partials_clipped_evt = self.shared_bprop_exec.run(
            plan=self.plan,
            num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS,
            summed_grad_h_ref=summed_grad_h_ref,
            deps=[summed_grad_h_evt],
        )
        print("    [Conductor] Sync Point 1: All partial gradients clipped.")

        # --- Phase 2: Gradient Aggregation ---
        all_partials_ready_deps = [all_tiles_clipped_evt, shared_partials_clipped_evt]
        summed_grad_handles = {}
        reduction_events = [summed_grad_h_evt]  # Start with the event from the Grad_H provider
        summed_grad_handles["hidden_activations"] = summed_grad_h_ref

        # The standard reduction executor now only processes the *other* gradients.
        other_handles, other_event = self.reduction_exec.run(
            plan=self.plan,
            num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS,
            deps=all_partials_ready_deps,
            exclude_params=["hidden_activations"],  # CRITICAL: Exclude Grad_H
        )
        summed_grad_handles.update(other_handles)
        reduction_events.append(other_event)

        all_grads_summed_evt = cl.WaitForEvents(reduction_events)
        print("    [Conductor] Sync Point 2: All gradients aggregated.")

        # --- Phase 3: Final Batch-Wide Update ---
        final_event = self.update_exec.run(
            step=step, summed_grads=summed_grad_handles, plan=self.plan, deps=[all_grads_summed_evt]
        )
        print("    [Conductor] Final update phase launched.")

        return final_event


class TrainingOrchestrator:
    """
    The top-level System Owner. This class is the "Strategist." It authors
    the ExecutionPlan but delegates all tactical execution.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        param_space: ParameterSpace,
        compute_env: ComputeEnvironment,
        hyperparams: TrainingHyperparams,
        adaptation_strategy: str,
        batch_size: int,
    ):
        self.model_spec = model_spec
        self.param_space = param_space
        self.compute_env = compute_env
        self.hyperparams = hyperparams
        self.adaptation_strategy = adaptation_strategy
        self.global_step = 1

        bm = BufferManager(self.compute_env.cl_bundle.context)
        ex = KernelExecutor(self.compute_env.cl_bundle.program)
        agg_mgr = AggregationManager(ex, bm, self.compute_env.arch_consts)
        self.services = Services(
            q=self.compute_env.cl_bundle.queue,
            ex=ex,
            bm=bm,
            agg_mgr=agg_mgr,
            model_spec=model_spec,
            arch_consts=self.compute_env.arch_consts,
        )
        self._setup_buffers(batch_size)

    def _setup_buffers(self, batch_size: int):
        """Creates all buffers by consuming the authoritative memory layouts
        from the ParameterSpace manifest."""
        bm, spec = self.services.bm, self.model_spec
        print("INFO: Setting up all buffers from ParameterSpace manifest...")

        grid_for_sizing = ExecutionGrid(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        all_layouts = self.param_space.get_all_memory_layouts(
            batch_size=batch_size, grid=grid_for_sizing, num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS
        )
        for name, layout in all_layouts.items():
            bm.create_named_buffer(name, layout, spec.scalar_dtype)
        print("INFO: All buffers created successfully from manifest.")

    def _create_execution_plan(self, batch_size: int) -> ExecutionPlan:
        """Authors the plan. This is now a pure, high-level strategic method."""
        svs, spec, h = self.services, self.model_spec, self.hyperparams
        policy_providers = {}

        # 1. Define Work Partitioning and Reduction Strategies
        grid = ExecutionGrid(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        reduction_plan = ReductionPlan(k=self.compute_env.arch_consts.get("optimal_tile_size", 16))

        # 2. Define the strategic DATA LIFECYCLE policies
        print(f"  [Orchestrator] Authoring plan with strategy: '{self.adaptation_strategy}'")

        if self.adaptation_strategy == "CACHE":
            # --- High VRAM Strategy ---
            # Pre-compute `hidden_activations` and cache the result.
            fwd_exec = ForwardPassExecutor(svs)
            h_ref, h_ready_evt = fwd_exec.run(batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)

            # `summed_grad_h` is not provided here; it will be computed by the default
            # `ReductionPhaseExecutor` path, which is what we want for this strategy.

        elif self.adaptation_strategy == "RECOMPUTE_GRAD_H":
            # --- Low VRAM Strategy: "Accumulate via Recompute" ---

            # For this strategy, `hidden_activations` are ephemeral. We create a
            # `RecomputeProvider` so any part of the DAG that needs them can get
            # them on-demand by re-running the forward pass.
            h_shape, h_dtype = svs.bm.get_spec(svs.bm.get_handle_by_name("hidden_activations"))
            recompute_out_ref = svs.bm.acquire_transient_buffer(int(np.prod(h_shape) * h_dtype().itemsize))
            recompute_mask_ref = svs.bm.acquire_transient_buffer(int(np.prod(h_shape) * h_dtype().itemsize))
            recompute_sig = ForwardPassSignature(
                svs.bm,
                simd_width=spec.simd_width,
                local_mem_bank_padding=1,
                scalar_size_bytes=spec.scalar_dtype().itemsize,
                in_ref=svs.bm.get_handle_by_name("input"),
                mask_ref=svs.bm.get_handle_by_name("sample_mask"),
                w_ref=svs.bm.get_handle_by_name("shared_weights"),
                b_ref=svs.bm.get_handle_by_name("shared_biases"),
                h_out_ref=recompute_out_ref,
                h_mask_out_ref=recompute_mask_ref,
                batch_chunk_offset=np.uint32(0),
                batch_chunk_count=np.uint32(batch_size),
            )
            policy_providers["hidden_activations"] = RecomputeProvider(
                signature=recompute_sig, output_handle=recompute_out_ref
            )

            # The calculation of `summed_grad_h` is now a complex, staged process.
            # We instantiate its specialist executor and create a placeholder for it.
            grad_h_stream_executor = GradHStreamingExecutor(svs)
            policy_providers["_grad_h_executor_placeholder"] = grad_h_stream_executor

        else:
            raise ValueError(f"Unknown adaptation strategy: '{self.adaptation_strategy}'")

        # 3. Assemble the Final, Immutable ExecutionPlan
        # This object is created first, then refined to solve the circular dependency.
        plan = ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            effective_batch_size=batch_size,
            problem_type="CCE",
            clipping_strategy="GLOBAL",
            hyperparams=self.hyperparams,
            param_space=self.param_space,
            adaptation_strategy=self.adaptation_strategy,
        )

        # 4. Finalize the `StagedComputationProvider` for RECOMPUTE_GRAD_H
        # This resolves the circular dependency where the provider needs the final plan.
        if self.adaptation_strategy == "RECOMPUTE_GRAD_H":
            executor_placeholder = plan.lifecycle_policy.providers.pop("_grad_h_executor_placeholder")
            # Create a callable that binds the specialist executor's method to the final plan.
            provider_fn = partial(executor_placeholder.compute_summed_grad_h, plan=plan)
            plan.lifecycle_policy.providers["summed_grad_hidden_activations"] = StagedComputationProvider(
                computation_fn=provider_fn
            )
        else:  # For CACHE strategy, the default reduction path IS the provider.
            reduction_exec = ReductionPhaseExecutor(svs, self.param_space)
            provider_fn = partial(
                reduction_exec.run_single_flow, flow_name="hidden_activations", plan=plan, num_batch_chunks=0
            )
            plan.lifecycle_policy.providers["summed_grad_hidden_activations"] = StagedComputationProvider(
                computation_fn=provider_fn
            )

        return plan

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """The main training loop."""
        h = self.hyperparams
        batch_size = X_train.shape[0]

        print(f"\n--- Beginning Training Run: {h.epochs} epochs ---")
        for epoch in range(h.epochs):
            print(f"\n--- Epoch {epoch+1}/{h.epochs} ---")
            plan = self._create_execution_plan(batch_size)
            processor = BatchProcessor(self.services, self.param_space, plan)
            final_event = processor.run(X_train, y_train, self.global_step)
            final_event.wait()
            print(f"  Epoch {epoch+1} complete.")
            self.global_step += 1
        print("\n--- Training Finished ---")


# =========================================================================
# === The Application Entry Point & System Assembler ===
# =========================================================================
if __name__ == "__main__":
    # --- 1. Define The LOGICAL Experiment ---
    print("--- Step 1: Defining Logical Experiment ---")
    LOGICAL_HIDDEN_DIM = 32
    LOGICAL_NUM_MODULES = 8
    # Select the strategy here: "CACHE" or "RECOMPUTE_GRAD_H"
    ADAPTATION_STRATEGY = "RECOMPUTE_GRAD_H"
    KERNEL_SOURCE_DIR = "./"

    HYPERPARAMS = TrainingHyperparams(
        epochs=5,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-7,
        max_grad_norm=1.0,
        temp_min=0.1,
        temp_max=10.0,
    )

    # --- 2. Build the SELF-CONFIGURING Compute Environment ---
    print("\n--- Step 2: Building Self-Configuring Compute Environment ---")
    try:
        # Corrected for the provided file structure (single kernels.cl.h)
        manager = OpenCLContextManager(kernel_source_dir=KERNEL_SOURCE_DIR)
        compute_env = manager.build_and_discover()
        print(f"  Successfully built environment. Discovered Constants: {compute_env.arch_consts}")
    except (cl.RuntimeError, FileNotFoundError) as e:
        print(f"\nFATAL: Could not build OpenCL environment: {e}")
        exit(1)

    # --- 3. Load Data ---
    print("\n--- Step 3: Loading Data ---")
    iris = load_iris()
    X_train_data = iris.data.astype(SCALAR_DTYPE)
    y_train_data = iris.target.astype(np.int32)
    batch_size = X_train_data.shape[0]

    # --- 4. Create the Model's Architectural Specification (`ModelSpec`) ---
    print("\n--- Step 4: Creating Architectural ModelSpec ---")
    iris_model_spec = ModelSpec(
        input_dim=X_train_data.shape[1],
        output_classes=len(np.unique(y_train_data)),
        hidden_dim=LOGICAL_HIDDEN_DIM,
        num_modules=LOGICAL_NUM_MODULES,
        simd_width=compute_env.arch_consts.get("simd_width"),
        cache_line_bytes=compute_env.arch_consts.get("global_mem_cacheline_size"),
    )
    print(f"  Final ModelSpec created:\n{iris_model_spec}")

    # --- 5. Create the Learnable Parameter Manifest (`ParameterSpace`) ---
    print("\n--- Step 5: Building Learnable ParameterSpace from ModelSpec ---")
    param_space = ParameterSpace(spec=iris_model_spec)
    print("  ParameterSpace manifest created successfully.")

    # --- 6. Instantiate and Run the Orchestrator ---
    print("\n--- Step 6: Instantiating and Running the Orchestrator ---")
    orchestrator = TrainingOrchestrator(
        model_spec=iris_model_spec,
        param_space=param_space,
        compute_env=compute_env,
        hyperparams=HYPERPARAMS,
        adaptation_strategy=ADAPTATION_STRATEGY,
        batch_size=batch_size,
    )
    orchestrator.train(X_train_data, y_train_data)

    print("\n--- Run Finished ---")
