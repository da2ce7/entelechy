# main_orchestrator.py

"""
The Definitive, Unified Streaming Classification Engine (Host Implementation).

(REV 5) This file contains the top-level
TrainingOrchestrator, which acts as a pure "Strategist". It is responsible for
defining the model, managing the training lifecycle (e.g., epochs), and
authoring high-level, declarative execution plans.

It delegates all tactical, step-by-step execution to the `BatchProcessor`,
which it instantiates on a per-batch basis. This file serves as the primary
entry point and high-level controller for the application.
"""

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
from compute_patterns import ReductionPlan
from workload_primitives import TilingScheme
from . import graph_recipes as recipes
from .batch_processor import BatchProcessor


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
    stabilization_lambda: float = 1.0
    reduction_k_grad_h: int = 16


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

        # Create the canonical Services bundle to be passed around
        bm = BufferManager(self.compute_env.cl_bundle.context)
        ex = KernelExecutor(self.compute_env.cl_bundle.program)
        self.services = Services(
            q=self.compute_env.cl_bundle.queue,
            ex=ex,
            bm=bm,
            model_spec=model_spec,
            arch_consts=self.compute_env.arch_consts,
        )
        self.stream_chunks = 4  # This remains a strategic decision
        self._setup_buffers(batch_size, self.stream_chunks)

    def _setup_buffers(self, batch_size: int, num_stream_chunks: int):
        """Creates all buffers by consuming the authoritative memory layouts."""
        bm, spec = self.services.bm, self.model_spec
        print("INFO: Setting up all buffers from ParameterSpace manifest...")
        grid = TilingScheme(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        all_layouts = self.param_space.get_all_memory_layouts(
            batch_size=batch_size, grid=grid, num_batch_chunks=num_stream_chunks
        )
        for name, layout in all_layouts.items():
            bm.create_named_buffer(name, layout, spec.scalar_dtype)
        print("INFO: All buffers created successfully from manifest.")

    def _create_execution_plan(self, batch_size: int) -> ExecutionPlan:
        """
        Authors the plan. This is a pure, high-level strategic method that
        encapsulates all strategic decisions for a single batch run.
        """
        svs, spec = self.services, self.model_spec
        policy_providers = {}

        # 1. Define Work Partitioning and Reduction Strategies
        grid = TilingScheme(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        reduction_plan = ReductionPlan(k=self.compute_env.arch_consts.get("optimal_tile_size", 16))

        # 2. Define the strategic DATA LIFECYCLE policies
        print(f"  [Orchestrator] Authoring plan with strategy: '{self.adaptation_strategy}'")

        if self.adaptation_strategy == "CACHE":
            # For CACHE, h is pre-computed and its provider is a simple CacheProvider.
            h_ref, h_ready_evt = recipes.execute_forward_pass(svs, batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)
            # The provider for summed_grad_h uses the simple permute-and-reduce recipe.
            provider_fn = partial(recipes.execute_specialized_grad_h_reduction, svs=svs, plan=plan)
            policy_providers["summed_grad_hidden_activations"] = StagedComputationProvider(computation_fn=provider_fn)

        elif self.adaptation_strategy == "RECOMPUTE_GRAD_H":
            # For RECOMPUTE, h is provided on-demand by a RecomputeProvider.
            # We must define the signature for the recomputation kernel.
            recompute_handle = svs.bm.acquire_transient_buffer(...)  # Size logic omitted for brevity
            recompute_mask = svs.bm.acquire_transient_buffer(...)
            recompute_sig = ForwardPassSignature(
                buffer_mgr=svs.bm,
                # ... all params for ForwardPassSignature ...
            )
            policy_providers["hidden_activations"] = RecomputeProvider(
                signature=recompute_sig, output_handle=recompute_handle
            )
            # The provider for summed_grad_h is the entire streaming pipeline recipe.
            provider_fn = partial(recipes.execute_grad_h_streaming_pipeline, svs=svs, plan=plan)
            policy_providers["summed_grad_hidden_activations"] = StagedComputationProvider(computation_fn=provider_fn)
        else:
            raise ValueError(f"Unknown adaptation strategy: '{self.adaptation_strategy}'")

        # 3. Assemble the Final, Immutable ExecutionPlan
        plan = ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            effective_batch_size=batch_size,
            problem_type="CCE",
            clipping_strategy="GLOBAL",
            hyperparams=self.hyperparams,
            shared_backprop_stream_chunks=self.stream_chunks,
        )
        # We must re-bind the provider functions here because 'plan' was not defined
        # when the partials were first created.
        for provider in policy_providers.values():
            if isinstance(provider, StagedComputationProvider):
                provider.computation_fn.keywords["plan"] = plan

        return plan

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """The main training loop. Now a clean, high-level controller."""
        h = self.hyperparams
        batch_size = X_train.shape[0]

        print(f"\n--- Beginning Training Run: {h.epochs} epochs ---")
        for epoch in range(h.epochs):
            print(f"\n--- Epoch {epoch+1}/{h.epochs} ---")

            # Step 1: Author the high-level strategy for this batch.
            plan = self._create_execution_plan(batch_size)

            # Step 2: Instantiate a dedicated conductor to execute the plan.
            processor = BatchProcessor(self.services, self.param_space, plan)

            # Step 3: Run the processor and wait for it to complete the full batch.
            final_event = processor.run(X_train, y_train, self.global_step)
            final_event.wait()

            print(f"  Epoch {epoch+1} complete.")
            self.global_step += 1
        print("\n--- Training Finished ---")


# =========================================================================
# === The Application Entry Point & System Assembler ===
# =========================================================================
if __name__ == "__main__":
    print("--- Step 1: Defining Logical Experiment ---")
    LOGICAL_HIDDEN_DIM = 32
    LOGICAL_NUM_MODULES = 8
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

    print("\n--- Step 2: Building Self-Configuring Compute Environment ---")
    try:
        manager = OpenCLContextManager(kernel_source_dir=KERNEL_SOURCE_DIR)
        compute_env = manager.build_and_discover()
        print(f"  Successfully built environment. Discovered Constants: {compute_env.arch_consts}")
    except (cl.RuntimeError, FileNotFoundError) as e:
        print(f"\nFATAL: Could not build OpenCL environment: {e}")
        exit(1)

    print("\n--- Step 3: Loading Data ---")
    iris = load_iris()
    X_train_data = iris.data.astype(SCALAR_DTYPE)
    y_train_data = iris.target.astype(np.int32)
    batch_size = X_train_data.shape[0]

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

    print("\n--- Step 5: Building Learnable ParameterSpace from ModelSpec ---")
    param_space = ParameterSpace(spec=iris_model_spec)
    print("  ParameterSpace manifest created successfully.")

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
