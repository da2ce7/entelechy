# main_orchestrator.py

"""
The Definitive, Unified Streaming Classification Engine (Host Implementation).

(REV 2 - ARCHITECTURALLY FINALIZED) This version represents the fully harmonized
architecture, where every component strictly adheres to its designated role.

The `TrainingOrchestrator` is now a pure strategist, authoring high-level,
declarative execution plans.

The `BatchProcessor` is now a pure conductor, supervising a linear sequence
of expert `PhaseExecutor` objects, each responsible for one cohesive stage of
the training algorithm. The final separation of concerns is complete.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
        self.plan = plan
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
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer("targets_cce"), y_batch)  # Assumes CCE for demo
        initial_deps = [upload_x_evt, upload_y_evt]

        # --- Phase 1: Parallel Partial Gradient Production ---
        tile_completion_events = [
            self.mod_path_exec.run_for_tile(tile, self.plan, deps=initial_deps) for tile in self.plan.grid
        ]
        all_tiles_clipped_evt = cl.WaitForEvents(tile_completion_events)
        shared_partials_clipped_evt = self.shared_bprop_exec.run(
            plan=self.plan, num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS, deps=initial_deps
        )
        print("    [Conductor] Sync Point 1: All partial gradients clipped.")

        # --- Phase 2: Gradient Aggregation ---
        all_partials_ready_deps = [all_tiles_clipped_evt, shared_partials_clipped_evt]
        summed_grad_handles, all_grads_summed_evt = self.reduction_exec.run(
            plan=self.plan, num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS, deps=all_partials_ready_deps
        )
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
            model_spec=model_spec,
            arch_consts=self.compute_env.arch_consts,
        )
        self._setup_buffers(batch_size)

    def _setup_buffers(self, batch_size: int):
        """Creates all buffers by consuming the authoritative memory layouts
        from the ParameterSpace manifest."""
        bm, spec = self.services.bm, self.model_spec
        print("INFO: Setting up all buffers from ParameterSpace manifest...")

        # Create a grid just for sizing the partial gradient buffers
        grid_for_sizing = ExecutionGrid(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )

        # Get the complete, authoritative dictionary of memory layouts
        all_layouts = self.param_space.get_all_memory_layouts(
            batch_size=batch_size, grid=grid_for_sizing, num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS
        )

        # Create all buffers by iterating through the manifest
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

        # 2. Define the strategic DATA LIFECYCLE policy for hidden_activations
        print(f"  [Orchestrator] Authoring plan with '{self.adaptation_strategy}' strategy for hidden activations.")
        fwd_exec = ForwardPassExecutor(svs)
        if self.adaptation_strategy == "CACHE":
            h_ref, h_ready_evt = fwd_exec.run(batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)
        else:  # "RECOMPUTE"
            # The RECOMPUTE strategy still implies a single, full recomputation in this design.
            # A chunk-by-chunk recompute would be a further evolution.
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

        # 3. Assemble the Final, Immutable ExecutionPlan
        return ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            effective_batch_size=batch_size,
            # Pass through the strategic policies for the tactical executors to use.
            problem_type="CCE",
            clipping_strategy="GLOBAL",
            hyperparams=self.hyperparams,
            param_space=self.param_space,
        )

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
    ADAPTATION_STRATEGY = "CACHE"
    KERNEL_SOURCE_DIR = "./"  # Assuming kernels are in the current directory for demo

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
        manager = OpenCLContextManager(
            kernel_source_dir=KERNEL_SOURCE_DIR, cl_filename="kernels.cl.h"
        )  # Refined for demo
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
