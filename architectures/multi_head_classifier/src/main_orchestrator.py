# main_orchestrator.py

"""
The Definitive, Unified Streaming Classification Engine (Host Implementation).

This version represents the final, harmonized architecture. The `__main__`
block acts as the 'System Assembler,' defining the logical experiment, using the
`OpenCLContextManager` to discover hardware and build the compute environment,
and then injecting these complete, ready-to-use resources into the
`TrainingOrchestrator`, which is purely focused on the logic of training.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pyopencl as cl
from sklearn.datasets import load_iris

# --- Foundational & Architectural Imports ---
from model_spec import ModelSpec, SCALAR_DTYPE
from cl_context_manager import OpenCLContextManager, ComputeEnvironment
from parameter_space import ParameterSpace, ParameterFlowConfig
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


@dataclass(frozen=True)
class ParameterFlowConfig:
    """A declarative manifest entry for a single parameter's gradient lifecycle."""

    name: str
    param_buffer_name: str
    partial_grad_buffer_name: str
    clipped_partial_grad_buffer_name: str
    summed_grad_buffer_name: str
    final_grad_buffer_name: str
    m1_buffer_name: str
    m2_buffer_name: str


class BatchProcessor:
    """A transient object that executes a single, complete ExecutionPlan."""

    def __init__(self, services: Services, plan: ExecutionPlan):
        self.svs = services
        self.plan = plan
        self.mod_path_exec = ModulePathExecutor(services)
        self.shared_bprop_exec = SharedBackpropExecutor(services)
        self.update_exec = UpdatePhaseExecutor(services)

    def run(self, X_batch: np.ndarray, y_batch: np.ndarray, step: int) -> cl.Event:
        q, bm, ex = self.svs.q, self.svs.bm, self.svs.ex
        print("    [Conductor] Beginning batch processing...")

        upload_x_evt = cl.enqueue_copy(q, bm.get_cl_buffer("input"), X_batch)
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer("targets"), y_batch)
        initial_deps = [upload_x_evt, upload_y_evt]

        tile_completion_events = [
            self.mod_path_exec.run_for_tile(tile, self.plan, deps=initial_deps) for tile in self.plan.grid
        ]
        shared_partials_clipped_evt = self.shared_bprop_exec.run(
            plan=self.plan, num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS, deps=initial_deps
        )

        all_partials_ready_deps = tile_completion_events + [shared_partials_clipped_evt]
        print("    [Conductor] Sync Point 1: All partial gradients clipped.")

        summed_grad_handles, summed_grad_events = {}, []
        provider_keys = [f"summed_{flow.name}" for flow in TrainingOrchestrator.PARAMETER_FLOW_MANIFEST]
        for key in provider_keys:
            provider = self.plan.lifecycle_policy.get_provider(key)
            grad_name = key.replace("summed_", "")
            handle, ready_evt = provider.resolve(q, ex, wait_for=all_partials_ready_deps)
            summed_grad_handles[grad_name] = handle
            summed_grad_events.append(ready_evt)

        all_grads_summed_evt = cl.WaitForEvents(summed_grad_events)
        print("    [Conductor] Sync Point 2: All gradients aggregated.")

        final_event = self.update_exec.run(step, summed_grad_handles, self.plan, deps=[all_grads_summed_evt])
        print("    [Conductor] Final update phase launched.")

        return final_event


class TrainingOrchestrator:
    """
    The top-level System Owner. Receives a complete blueprint of the model
    and compute environment, and orchestrates the training process.

    This class is now 'pure' in the sense that it contains no hardcoded knowledge
    of the model's learnable structure; that is all encapsulated within the
    injected `ParameterSpace` object.
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
        """
        Initializes the orchestrator with its complete, pre-built dependencies.

        Args:
            model_spec: The architectural specification of the model.
            param_space: The declarative manifest of all learnable parameters.
            compute_env: The pre-built, self-configured compute environment.
            hyperparams: A dataclass containing all training hyperparameters.
            adaptation_strategy: The high-level strategy ("CACHE" or "RECOMPUTE").
            batch_size: The number of samples per batch.
        """
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
            q=self.compute_env.cl_bundle.queue, ex=ex, bm=bm, agg_mgr=agg_mgr, model_spec=model_spec
        )

        self._setup_buffers(batch_size)

    def _setup_buffers(self, batch_size: int):
        """Creates all buffers, now driven by the ParameterSpace manifest."""
        bm, spec = self.services.bm, self.model_spec
        print("INFO: Setting up all buffers from ParameterSpace manifest...")

        # Create a grid just for sizing the partial gradient buffers
        grid_for_sizing = ExecutionGrid(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        # Get shape dictionaries from the authority (the ParameterSpace)
        param_shapes = self.param_space.get_param_shapes()
        partial_grad_shapes = self.param_space.get_partial_grad_shapes(
            batch_size=batch_size,
            total_tiles=grid_for_sizing.total_tiles,
            modules_per_chunk=grid_for_sizing.get_tile(0, 0).modules_per_chunk,
            classes_per_chunk=grid_for_sizing.get_tile(0, 0).classes_per_chunk,
        )

        # Create all parameter-related buffers by iterating through the manifest
        for flow in self.param_space:
            p_shape = param_shapes.get(flow.name)
            pg_shape = partial_grad_shapes.get(flow.partial_grad_buffer_name)

            if p_shape and not flow.specialized_reduction:  # `hidden_activations` is not a real parameter
                bm.create_named_buffer(flow.param_buffer_name, MemoryLayout(p_shape), spec.scalar_dtype)
                bm.create_named_buffer(flow.summed_grad_buffer_name, MemoryLayout(p_shape), spec.scalar_dtype)
                bm.create_named_buffer(flow.final_grad_buffer_name, MemoryLayout(p_shape), spec.scalar_dtype)
                bm.create_named_buffer(flow.m1_buffer_name, MemoryLayout(p_shape), spec.scalar_dtype)
                bm.create_named_buffer(flow.m2_buffer_name, MemoryLayout(p_shape), spec.scalar_dtype)
            if pg_shape:
                bm.create_named_buffer(flow.partial_grad_buffer_name, MemoryLayout(pg_shape), spec.scalar_dtype)
                bm.create_named_buffer(flow.clipped_partial_grad_buffer_name, MemoryLayout(pg_shape), spec.scalar_dtype)

        print("INFO: All buffers created successfully from manifest.")

    def _create_execution_plan(self, batch_size: int) -> ExecutionPlan:
        """Authors the plan, now cleanly driven by the ParameterSpace."""
        svs, spec = self.services, self.model_spec
        policy_providers = {}
        grid = ExecutionGrid(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        reduction_plan = ReductionPlan(k=self.compute_env.arch_consts.optimal_tile_size)

        if self.adaptation_strategy == "CACHE":
            fwd_exec = ForwardPassExecutor(svs)
            h_ref, h_ready_evt = fwd_exec.run(batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)
        else:  # "RECOMPUTE"
            recompute_sig = ForwardPassSignature(...)
            policy_providers["hidden_activations"] = RecomputeProvider(signature=recompute_sig, output_handle=...)

        # Create reduction providers for all parameters defined in the manifest
        for flow in self.param_space:

            def make_reduction_fn(current_flow=flow):
                def fn(q, ex, deps):
                    print(f"      [Provider] Executing reduction for {current_flow.name}...")
                    if current_flow.specialized_reduction:
                        # Specialized Grad_H Path
                        permute_sig = GatherAndPermuteGradHSignature(svs.bm, ...)
                        permute_evt = ex.launch(q, permute_sig, wait_for=deps)
                        reduce_sig = ReduceGradHOverModulesSignature(svs.bm, ...)
                        reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])
                        return svs.bm.get_handle_by_name(current_flow.summed_grad_buffer_name), reduce_evt
                    else:
                        # Generic Reduction Path using the ReductionTreeExecutor
                        clipped_ref = svs.bm.get_handle_by_name(current_flow.clipped_partial_grad_buffer_name)
                        shape, _ = svs.bm.get_spec(clipped_ref)
                        elements_per_partial = np.prod(shape[1:]) if len(shape) > 1 else 1
                        gather_prim = TiledGather(grid=grid, elements_per_partial=int(elements_per_partial))
                        reduction_exec = ReductionTreeExecutor(q, ex, svs.bm, svs.agg_mgr, reduction_plan)
                        dest_h = svs.bm.get_handle_by_name(current_flow.summed_grad_buffer_name)
                        return reduction_exec.execute(gather_prim, clipped_ref, dest_h, deps)

                return fn

            # e.g., key becomes "summed_shared_weights"
            provider_key = current_flow.summed_grad_buffer_name.replace("summed_grad_", "summed_")
            policy_providers[provider_key] = StagedComputationProvider(computation_fn=make_reduction_fn())

        return ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            effective_batch_size=batch_size,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """The main training loop."""
        h = self.hyperparams
        print(f"\n--- Beginning Training Run: {h.epochs} epochs ---")
        for epoch in range(h.epochs):
            print(f"\n--- Epoch {epoch+1}/{h.epochs} ---")
            plan = self._create_execution_plan(batch_size=X_train.shape[0])
            processor = BatchProcessor(self.services, plan)
            final_event = processor.run(X_train, y_train, self.global_step)
            # final_event.wait() # Would be uncommented in a real run
            print(f"  Epoch {epoch+1} complete (Simulated).")
            self.global_step += 1
        print("\n--- Training Finished ---")


# =========================================================================
# === The Application Entry Point & System Assembler ===
# =========================================================================
if __name__ == "__main__":
    # --- 1. Define The LOGICAL Experiment ---
    # This is the single, high-level configuration section for a specific run.
    print("--- Step 1: Defining Logical Experiment ---")
    LOGICAL_HIDDEN_DIM = 32
    LOGICAL_NUM_MODULES = 8
    ADAPTATION_STRATEGY = "CACHE"
    KERNEL_SOURCE_DIR = "./kernels"
    # All hyperparameters are defined cleanly in a structured object.
    HYPERPARAMS = TrainingHyperparams(
        epochs=5, learning_rate=0.001, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-7
    )

    # --- 2. Build the SELF-CONFIGURING Compute Environment ---
    print("\n--- Step 2: Building Self-Configuring Compute Environment ---")
    # This is the single point of interaction with hardware and filesystems.
    # It will fail gracefully if OpenCL or the kernel directory is not found.
    try:
        manager = OpenCLContextManager(kernel_source_dir=KERNEL_SOURCE_DIR)
        compute_env = manager.build_and_discover()
        print(f"  Successfully built environment. Discovered Constants: {compute_env.arch_consts}")
    except (cl.RuntimeError, FileNotFoundError) as e:
        print(f"\nFATAL: Could not build OpenCL environment: {e}")
        print("Please ensure OpenCL drivers are installed and the kernel directory is correct.")
        exit(1)

    # --- 3. Load Data ---
    print("\n--- Step 3: Loading Data ---")
    iris = load_iris()
    X_train_data = iris.data.astype(SCALAR_DTYPE)
    y_train_data = iris.target.astype(np.int32)
    batch_size = X_train_data.shape[0]

    # --- 4. Create the Model's Architectural Specification (`ModelSpec`) ---
    # This fuses the logical experiment parameters with the discovered hardware parameters.
    print("\n--- Step 4: Creating Architectural ModelSpec ---")
    iris_model_spec = ModelSpec(
        input_dim=X_train_data.shape[1],
        output_classes=len(np.unique(y_train_data)),
        hidden_dim=LOGICAL_HIDDEN_DIM,
        num_modules=LOGICAL_NUM_MODULES,
        simd_width=compute_env.arch_consts.simd_width,
        cache_line_bytes=compute_env.arch_consts.global_mem_cacheline_size,
    )
    print(f"  Final ModelSpec created:\n{iris_model_spec}")

    # --- 5. Create the Learnable Parameter Manifest (`ParameterSpace`) ---
    # This uses the architectural spec to programmatically define the entire
    # space of learnable parameters and their associated buffers.
    print("\n--- Step 5: Building Learnable ParameterSpace from ModelSpec ---")
    param_space = ParameterSpace(spec=iris_model_spec)
    print("  ParameterSpace manifest created successfully.")

    # --- 6. Instantiate and Run the Orchestrator ---
    # Dependency injection is now perfect. The orchestrator receives its final,
    # immutable, pre-built dependencies and has no knowledge of their construction.
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
