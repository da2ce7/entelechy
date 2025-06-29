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
from execution_plan import *
from launcher_infra import *
from memory_layout import *
from kernel_signatures import *
from compute_patterns import *
from phase_executors import *

# This constant is specific to the execution strategy of the BatchProcessor
SHARED_BACKPROP_STREAM_CHUNKS = 4


# === A Dataclass for Training Hyperparameters ===
@dataclass(frozen=True)
class TrainingHyperparams:
    """A structured container for all training hyperparameters."""

    epochs: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float


class BatchProcessor:
    """
    A transient object that executes a single, complete ExecutionPlan.

    As the "Conductor," this class interprets the computational graph for one
    training step. It does not perform any computation itself, but instead
    delegates to the specialized `PhaseExecutor`s and manages the flow of
    OpenCL event dependencies between them to ensure correct execution order.
    """

    def __init__(self, services: Services, plan: ExecutionPlan):
        """
        Initializes the Conductor with its dependencies and the plan to execute.

        Args:
            services: The container of shared, persistent system resources.
            plan: The immutable, declarative manifest for this specific batch.
        """
        self.svs = services
        self.plan = plan

        # The Conductor instantiates the tactical experts it will delegate to.
        self.mod_path_exec = ModulePathExecutor(services)
        self.shared_bprop_exec = SharedBackpropExecutor(services)
        self.update_exec = UpdatePhaseExecutor(services)

    def run(self, X_batch: np.ndarray, y_batch: np.ndarray, step: int) -> cl.Event:
        """
        Executes the entire training DAG for one batch of data.

        Args:
            X_batch: The input feature data for the batch.
            y_batch: The target label data for the batch.
            step: The current global training step, used for optimizer calculations.

        Returns:
            A single OpenCL event that signals the completion of all work for
            this batch.
        """
        q, bm, ex = self.svs.q, self.svs.bm, self.svs.ex
        print("    [Conductor] Beginning batch processing...")

        # === Stage 1: Initial Data Upload ===
        # Copy host data to their respective device buffers. These uploads can
        # happen in parallel.
        upload_x_evt = cl.enqueue_copy(q, bm.get_cl_buffer("input"), X_batch)
        upload_y_evt = cl.enqueue_copy(q, bm.get_cl_buffer("targets"), y_batch)
        initial_deps = [upload_x_evt, upload_y_evt]

        # === Stage 2: Parallel Partial Gradient Calculation ===
        # Launch the two main parallel branches of the learning phase.

        # Branch A: Per-Tile Module Path (Nodes 5-11)
        # This loop launches all work for the module-specific parameters. Each call
        # to `run_for_tile` is a large, independent chunk of work.
        tile_completion_events = [
            self.mod_path_exec.run_for_tile(tile, self.plan, deps=initial_deps) for tile in self.plan.grid
        ]

        # Branch B: Streaming Shared Backpropagation (Nodes 17-18 -> Clip)
        # This launches the work for the shared layer parameters.
        shared_partials_clipped_evt = self.shared_bprop_exec.run(
            plan=self.plan, num_batch_chunks=SHARED_BACKPROP_STREAM_CHUNKS, deps=initial_deps
        )

        # === Stage 3: First Synchronization Point (Post-Clipping) ===
        # All subsequent reduction operations depend on ALL partial gradients
        # (from both Branch A and B) being fully calculated and clipped.
        all_partials_ready_deps = tile_completion_events + [shared_partials_clipped_evt]
        print("    [Conductor] Sync Point 1: All partial gradients clipped.")

        # === Stage 4: Reduction Phase ===
        # Resolve all 'summed_grad_*' dependencies from the ExecutionPlan. The
        # `provider.resolve` call triggers the complex reduction logic (e.g.,
        # the specialized Grad_H path or the generic ReductionTreeExecutor).
        summed_grad_handles: Dict[str, BufferHandle] = {}
        summed_grad_events: List[cl.Event] = []
        grad_provider_names = [p for p in self.plan.lifecycle_policy.providers if "summed" in p]

        for provider_name in grad_provider_names:
            provider = self.plan.lifecycle_policy.get_provider(provider_name)
            # The 'grad_name' is used to index the final dictionary of summed handles.
            grad_name = provider_name.replace("summed_", "")
            handle, ready_evt = provider.resolve(q, ex, wait_for=all_partials_ready_deps)
            summed_grad_handles[grad_name] = handle
            summed_grad_events.append(ready_evt)

        # === Stage 5: Second Synchronization Point (Batch Sync) ===
        # We must wait for all reduction/aggregation operations to complete
        # before we can normalize and apply the final gradients.
        all_grads_summed_evt = cl.WaitForEvents(summed_grad_events)
        print("    [Conductor] Sync Point 2: All gradients aggregated.")

        # === Stage 6: Final Update Phase (Nodes 20-24) ===
        # Delegate to the final expert to normalize gradients and update parameters.
        final_event = self.update_exec.run(step, summed_grad_handles, self.plan, deps=[all_grads_summed_evt])
        print("    [Conductor] Final update phase launched.")

        return final_event


# ====== Tier 1: The System Owner (`TrainingOrchestrator`) ======


class TrainingOrchestrator:
    """
    The top-level System Owner. Owns all resources and makes strategic decisions
    by authoring an ExecutionPlan for each training step.

    It receives a pre-built compute environment and has no knowledge of how it
    was compiled or from where. Its job is to orchestrate the high-level logic
    of the training loop.
    """

    # This class-level map defines which gradients exist and what parameter they
    # correspond to. It's a central piece of configuration for the reduction phase.
    GRADIENT_PARAM_MAP = {
        "grad_shared_weights": "shared_weights",
        "grad_shared_biases": "shared_biases",
        "grad_module_weights": "module_weights",
        "grad_module_biases": "module_biases",
        "grad_temps": "temperatures",
        "grad_hidden_activations": "hidden_activations",
    }

    def __init__(
        self,
        model_spec: ModelSpec,
        compute_env: ComputeEnvironment,
        hyperparams: TrainingHyperparams,
        adaptation_strategy: str,
        batch_size: int,
    ):
        """
        Initializes the orchestrator with all its dependencies.

        Args:
            model_spec: The architectural specification of the model.
            compute_env: The pre-built, self-configured compute environment.
            hyperparams: A dataclass containing all training hyperparameters.
            adaptation_strategy: The high-level strategy ("CACHE" or "RECOMPUTE").
            batch_size: The number of samples per batch.
        """
        self.model_spec = model_spec
        self.compute_env = compute_env
        self.hyperparams = hyperparams
        self.adaptation_strategy = adaptation_strategy
        self.global_step = 1

        # --- Instantiate Core Services from Pre-Built Resources ---
        # The constructor is clean. It just unpacks its dependencies.
        bm = BufferManager(self.compute_env.cl_bundle.context)
        ex = KernelExecutor(self.compute_env.cl_bundle.program)
        agg_mgr = AggregationManager(ex, bm, arch_consts=self.compute_env.arch_consts)
        self.services = Services(
            q=self.compute_env.cl_bundle.queue, ex=ex, bm=bm, agg_mgr=agg_mgr, model_spec=model_spec
        )

        self._setup_buffers(batch_size)

    def _setup_buffers(self, batch_size: int):
        """
        Creates all persistent buffers using the ModelSpec for dimensions.
        """
        bm = self.services.bm
        spec = self.model_spec
        print(
            f"INFO: Setting up buffers for batch size {batch_size} and model spec: "
            f"{spec.input_dim} -> {spec.hidden_dim} -> {spec.output_classes}"
        )

        # --- Tiling Plan for Partial Buffer Sizing ---
        num_module_chunks = (spec.num_modules + 15) // 16
        num_class_chunks = (spec.output_classes + 15) // 16
        grid_for_sizing = ExecutionGrid(num_module_chunks, num_class_chunks, spec.num_modules, spec.output_classes)
        total_tiles = grid_for_sizing.total_tiles
        first_tile = grid_for_sizing.get_tile(0, 0)
        modules_per_chunk = first_tile.modules_per_chunk
        classes_per_chunk = first_tile.classes_per_chunk

        # === Foundational Inputs & Parameters ===
        bm.create_named_buffer("input", MemoryLayout((batch_size, spec.padded_input_dim)), spec.scalar_dtype)
        bm.create_named_buffer("targets", MemoryLayout((batch_size,)), np.int32)
        param_shapes = {
            "shared_weights": (spec.input_dim, spec.padded_hidden_dim),
            "shared_biases": (spec.padded_hidden_dim,),
            "module_weights": (spec.num_modules, spec.padded_hidden_dim, spec.padded_class_dim),
            "module_biases": (spec.num_modules, spec.padded_class_dim),
            "temperatures": (spec.num_modules,),
        }
        for name, shape in param_shapes.items():
            bm.create_named_buffer(name, MemoryLayout(shape), spec.scalar_dtype)
            bm.create_named_buffer(f"m1_{name}", MemoryLayout(shape), spec.scalar_dtype)
            bm.create_named_buffer(f"m2_{name}", MemoryLayout(shape), spec.scalar_dtype)

        # === Intermediates ===
        bm.create_named_buffer(
            "hidden_activations", MemoryLayout((batch_size, spec.padded_hidden_dim)), spec.scalar_dtype
        )
        bm.create_named_buffer(
            "logits", MemoryLayout((spec.num_modules, batch_size, spec.padded_class_dim)), spec.scalar_dtype
        )
        probs_shape = (total_tiles, modules_per_chunk, batch_size, classes_per_chunk)
        bm.create_named_buffer("partial_probs", MemoryLayout(probs_shape), spec.scalar_dtype)

        # === Partial & Clipped Gradient Collections ===
        partial_grad_shapes = {
            "grad_module_weights": (total_tiles, modules_per_chunk, spec.padded_hidden_dim, classes_per_chunk),
            "grad_module_biases": (total_tiles, modules_per_chunk, classes_per_chunk),
            "grad_temps": (total_tiles, modules_per_chunk),
            "grad_hidden_activations": (total_tiles, modules_per_chunk, batch_size, spec.padded_hidden_dim),
        }
        for name, shape in partial_grad_shapes.items():
            bm.create_named_buffer(f"partial_{name}", MemoryLayout(shape), spec.scalar_dtype)
            bm.create_named_buffer(f"clipped_partial_{name}", MemoryLayout(shape), spec.scalar_dtype)

        # === Final Aggregated Gradients ===
        for name, shape in param_shapes.items():
            bm.create_named_buffer(f"summed_grad_{name}", MemoryLayout(shape), spec.scalar_dtype)
            bm.create_named_buffer(f"final_grad_{name}", MemoryLayout(shape), spec.scalar_dtype)

        print("INFO: All buffers created successfully.")

    def _create_execution_plan(self, batch_size: int) -> ExecutionPlan:
        """The STRATEGIC heart of the orchestrator. Authors the plan for the conductor."""
        print("  [Strategist] Authoring ExecutionPlan...")
        svs = self.services
        spec = self.model_spec
        policy_providers = {}

        grid = ExecutionGrid(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        reduction_plan = ReductionPlan(k=self.compute_env.arch_consts.optimal_tile_size)

        # --- Dynamic Adaptation: Decide how to provide `hidden_activations` ---
        if self.adaptation_strategy == "CACHE":
            print("  [Strategist] Policy: CACHE hidden_activations.")
            fwd_pass_exec = ForwardPassExecutor(svs)
            h_ref, h_ready_evt = fwd_pass_exec.run(batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)
        else:  # "RECOMPUTE"
            print("  [Strategist] Policy: RECOMPUTE hidden_activations.")
            recompute_sig = ForwardPassSignature(...)  # Configured for a specific chunk
            policy_providers["hidden_activations"] = RecomputeProvider(signature=recompute_sig, output_handle=...)

        # --- Encapsulate Specialized Grad_H Path ---
        def compute_summed_grad_h(q, ex, deps):
            print("      [Provider] Executing staged computation for Grad_H...")
            permute_sig = GatherAndPermuteGradHSignature(svs.bm, ...)
            permute_evt = ex.launch(q, permute_sig, wait_for=deps)
            reduce_sig = ReduceGradHOverModulesSignature(svs.bm, ...)
            reduce_evt = ex.launch(q, reduce_sig, wait_for=[permute_evt])
            return svs.bm.get_handle_by_name("summed_grad_hidden_activations"), reduce_evt

        policy_providers["summed_grad_hidden_activations"] = StagedComputationProvider(
            computation_fn=compute_summed_grad_h
        )

        # --- Encapsulate Generic Reductions for other gradients ---
        for grad_name, param_name in self.GRADIENT_PARAM_MAP.items():
            if "hidden" in grad_name:
                continue

            def make_reduction_fn(g_name=grad_name, p_name=param_name):
                def fn(q, ex, deps):
                    print(f"      [Provider] Executing reduction for {g_name}...")
                    clipped_partial_ref = svs.bm.get_handle_by_name(f"clipped_partial_{g_name}")
                    shape, _ = svs.bm.get_spec(clipped_partial_ref)
                    elements_per_partial = np.prod(shape[1:])
                    gather_prim = TiledGather(grid=grid, elements_per_partial=int(elements_per_partial))
                    reduction_exec = ReductionTreeExecutor(q, ex, svs.bm, svs.agg_mgr, reduction_plan)
                    dest_h = svs.bm.get_handle_by_name(f"summed_{g_name}")
                    return reduction_exec.execute(gather_prim, clipped_partial_ref, dest_h, deps)

                return fn

            policy_providers[f"summed_{grad_name}"] = StagedComputationProvider(computation_fn=make_reduction_fn())

        return ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            effective_batch_size=batch_size,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """The main training loop, driven by the ModelSpec."""
        h = self.hyperparams
        print(f"\n--- Beginning Training Run: {h.epochs} epochs ---")
        for epoch in range(h.epochs):
            print(f"\n--- Epoch {epoch+1}/{h.epochs} ---")
            plan = self._create_execution_plan(batch_size=X_train.shape[0])
            processor = BatchProcessor(self.services, plan)
            final_event = processor.run(X_train, y_train, self.global_step)
            # final_event.wait() # Would be active in a real run
            print(f"  Epoch {epoch+1} complete (Simulated).")
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
    KERNEL_SOURCE_DIR = "./kernels"
    # All hyperparameters are now defined cleanly together.
    HYPERPARAMS = TrainingHyperparams(
        epochs=5, learning_rate=0.001, adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-7
    )

    # --- 2. Build the SELF-CONFIGURING Compute Environment ---
    print("\n--- Step 2: Building Self-Configuring Compute Environment ---")
    # This is now the single point of interaction with the hardware and filesystem.
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

    # --- 4. Create the Final Model Specification ---
    # Fuses the logical experiment parameters with the discovered hardware parameters.
    print("\n--- Step 4: Fusing Logical and Hardware Specs into ModelSpec ---")
    iris_model_spec = ModelSpec(
        input_dim=X_train_data.shape[1],
        output_classes=len(np.unique(y_train_data)),
        hidden_dim=LOGICAL_HIDDEN_DIM,
        num_modules=LOGICAL_NUM_MODULES,
        simd_width=compute_env.arch_consts.simd_width,
        cache_line_bytes=compute_env.arch_consts.global_mem_cacheline_size,
    )
    print(f"  Final ModelSpec created:\n{iris_model_spec}")

    # --- 5. Instantiate and Run the Orchestrator ---
    # Dependency injection is now perfect. The orchestrator receives its final,
    # immutable, pre-built dependencies.
    print("\n--- Step 5: Instantiating and Running the Orchestrator ---")
    orchestrator = TrainingOrchestrator(
        model_spec=iris_model_spec,
        compute_env=compute_env,
        hyperparams=HYPERPARAMS,
        adaptation_strategy=ADAPTATION_STRATEGY,
    )
    orchestrator.train(X_train_data, y_train_data)

    print("\n--- Run Finished ---")
