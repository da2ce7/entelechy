# main_orchestrator.py

# This module serves as the highest level of host-side control, embodying the
# "Strategist" pattern. Its sole purpose is to translate high-level experimental
# goals (model shape, hyperparameters, adaptation strategies) into a single,
# declarative "ExecutionPlan" for a training batch. It owns the training
# lifecycle (epochs) but delegates all per-batch tactical execution.

from dataclasses import dataclass
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import pyopencl as cl
from sklearn.datasets import load_iris

# --- Foundational Architectural Primitives ---
# These imports represent the core "nouns" of the system architecture. The
# Orchestrator's role is to compose these primitives into a coherent plan.
from .model_spec import ModelSpec, SCALAR_DTYPE
from .cl_context_manager import OpenCLContextManager, ComputeEnvironment
from .parameter_space import ParameterSpace
from .execution_plan import (
    ExecutionPlan,
    DataLifecyclePolicy,
    CacheProvider,
    RecomputeProvider,
    StagedComputationProvider,
    CceStrategy,
    BceStrategy,
)
from .launcher_infra import Services, BufferManager, KernelExecutor
from .kernel_signatures import ForwardPassSignature
from .compute_patterns import ReductionPlan
from .workload_primitives import TilingScheme
from .stabilization_policy import StabilizationPolicy
from .batch_processor import BatchProcessor
from . import graph_recipes as recipes

@dataclass(frozen=True)
class StabilizationConfig:
    """A structured configuration primitive for stabilization policy."""

    max_grad_norm: float
    lambda_: float


@dataclass(frozen=True)
class TrainingHyperparams:
    """
    The canonical, type-safe manifest for all training hyperparameters.
    Nesting the `StabilizationConfig` enforces a logical grouping, making the
    configuration hierarchy clear and explicit.
    """

    epochs: int
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    temp_min: float
    temp_max: float
    stabilization: StabilizationConfig
    reduction_k_grad_h: int = 16


class TrainingOrchestrator:
    """
    The System Owner and "Strategist."

    This class instantiates and owns all lower-level services. It is responsible
    for authoring the strategic `ExecutionPlan` but delegates all tactical,
    step-by-step execution to the `BatchProcessor`, which it creates on a
    per-batch basis.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        param_space: ParameterSpace,
        compute_env: ComputeEnvironment,
        hyperparams: TrainingHyperparams,
        adaptation_strategy: str,
        problem_type_name: str,
        clipping_strategy_name: str,
        batch_size: int,
    ):
        # The constructor acts as the "System Assembler," gathering and owning
        # all foundational components required for the entire training run.
        self.model_spec = model_spec
        self.param_space = param_space
        self.compute_env = compute_env
        self.hyperparams = hyperparams
        self.adaptation_strategy = adaptation_strategy
        self.problem_type_name = problem_type_name
        self.clipping_strategy_name = clipping_strategy_name
        self.global_step = 1

        # The `Services` bundle is a key dependency injection pattern, allowing
        # core components to be passed cleanly through the system layers.
        bm = BufferManager(self.compute_env.cl_bundle.context)
        ex = KernelExecutor(self.compute_env.cl_bundle.program)
        self.services = Services(
            q=self.compute_env.cl_bundle.queue,
            ex=ex,
            bm=bm,
            model_spec=model_spec,
            arch_consts=compute_env.arch_consts,
        )

        self.stream_chunks = 4
        self._setup_buffers(batch_size, self.stream_chunks)

    def _setup_buffers(self, batch_size: int, num_stream_chunks: int):
        """
        Orchestrates memory allocation by delegating to the `ParameterSpace`.
        This fulfills the "Primacy of Memory Strategy" by ensuring that memory
        layout decisions are centralized and contract-driven, not scattered
        throughout the application logic.
        """
        bm, spec = self.services.bm, self.model_spec
        grid = TilingScheme(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        # The Orchestrator asks the ParameterSpace for the complete memory plan...
        all_layouts = self.param_space.get_all_memory_layouts(
            batch_size=batch_size, grid=grid, num_batch_chunks=num_stream_chunks
        )
        # ...and instructs the BufferManager to execute that plan.
        for name, layout in all_layouts.items():
            bm.create_named_buffer(name, layout, spec.scalar_dtype)

    def _create_execution_plan(self, batch_size: int) -> ExecutionPlan:
        """
        Authors the complete, immutable execution strategy for a single batch.

        This method is the heart of the "Strategist" role. It makes no kernel
        calls itself. Instead, it composes high-level policy and strategy
        objects into a single `ExecutionPlan` manifest.
        """
        svs, spec, h_params = self.services, self.model_spec, self.hyperparams
        policy_providers = {}

        # --- Phase 1: Formulate Strategic Primitives ---
        grid = TilingScheme(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        reduction_plan = ReductionPlan(k=self.compute_env.arch_consts.get("optimal_tile_size", 16))

        # --- Phase 2: Instantiate Polymorphic Strategy Objects ---
        if self.problem_type_name == "CCE":
            problem_strategy = CceStrategy(targets_cce_ref=svs.bm.get_handle_by_name("targets_cce"))
        elif self.problem_type_name == "BCE":
            problem_strategy = BceStrategy(targets_bce_ref=svs.bm.get_handle_by_name("targets_bce"))
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type_name}")

        fp_max = np.finfo(spec.scalar_dtype).max
        stabilization_policy = StabilizationPolicy(
            t_algorithmic=h_params.stabilization.max_grad_norm,
            lambda_=h_params.stabilization.lambda_,
            fp_format_max=fp_max,
        )

        plan_placeholder = None

        # --- Phase 3: Define Data Lifecycle Policies (`CACHE` vs. `RECOMPUTE`) ---
        # The `StagedComputationProvider` below points to a unified recipe. This is
        # the architectural simplification: the logic for permuting and reducing `Grad_H`
        # is identical regardless of how its partials were created.
        summed_grad_h_provider_fn = partial(recipes.build_final_grad_h_reduction_path, svs=svs, plan=plan_placeholder)
        policy_providers["summed_grad_hidden_activations"] = StagedComputationProvider(
            computation_fn=summed_grad_h_provider_fn
        )

        # Now, we define the policy for the `hidden_activations` themselves based on strategy.
        if self.adaptation_strategy == "CACHE":
            h_ref, h_ready_evt = recipes.execute_forward_pass(svs, batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)

        elif self.adaptation_strategy == "RECOMPUTE_GRAD_H":
            # In this strategy, `hidden_activations` are transient scratch space internal
            # to the `build_streaming_module_grad_path` recipe. There is no persistent
            # `hidden_activations` buffer available to the Conductor, so no provider
            # is defined for it. This is architecturally correct.
            pass
        else:
            raise ValueError(f"Unknown adaptation strategy: '{self.adaptation_strategy}'")

        # --- Phase 4: Assemble and Finalize the ExecutionPlan ---
        plan = ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            # This is a placeholder; the Conductor calculates the true value.
            effective_batch_size=batch_size,
            problem_type=problem_strategy,
            clipping_strategy=self.clipping_strategy_name,
            stabilization_policy=stabilization_policy,
            hyperparams=self.hyperparams,
            shared_backprop_stream_chunks=self.stream_chunks,
            adaptation_strategy=self.adaptation_strategy,
        )

        # Re-bind the provider function with the now-complete `plan` object.
        for provider in policy_providers.values():
            if isinstance(provider, StagedComputationProvider):
                provider.computation_fn.keywords["plan"] = plan

        return plan

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        The main training loop. It is a pure controller that demonstrates the
        system's core operational pattern: Plan -> Execute.
        """
        batch_size = X_train.shape[0]

        for epoch in range(self.hyperparams.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.hyperparams.epochs}, Step {self.global_step} ---")

            # 1. Author the complete, high-level strategy for this batch.
            plan = self._create_execution_plan(batch_size)

            # 2. Instantiate a dedicated "Conductor" to execute the plan.
            processor = BatchProcessor(self.services, self.param_space, plan)

            # 3. Delegate execution and receive handles to the two event chains.
            learn_event, infer_event, probs_view = processor.run(X_train, y_train, self.global_step)

            # The host can choose to wait on the chains separately or together.
            # Here, we block for the learning to complete before the next step.
            learn_event.wait()
            print(f"--- Step {self.global_step} Complete. ---")

            # Example of using the inference event for non-blocking diagnostics.
            # In a real app, this could be passed to a separate logging thread.
            # infer_event.wait()
            # print("    [Host] Inference results are ready.")
            # final_probs = probs_view.get()

            self.global_step += 1


if __name__ == "__main__":
    # This block serves as the "System Assembler" at the application's
    # entry point. It defines the experiment's configuration, instantiates
    # all necessary architectural components, and initiates the training process.

    # --- 1. Experiment Configuration ---
    LOGICAL_HIDDEN_DIM = 32
    LOGICAL_NUM_MODULES = 8
    ADAPTATION_STRATEGY = "CACHE"  # Switch to "RECOMPUTE_GRAD_H" to test other path
    PROBLEM_TYPE = "CCE"
    CLIPPING_STRATEGY = "GLOBAL"
    KERNEL_SOURCE_DIR = "./"

    HYPERPARAMS = TrainingHyperparams(
        epochs=5,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-7,
        temp_min=0.1,
        temp_max=10.0,
        stabilization=StabilizationConfig(
            max_grad_norm=1.0,
            lambda_=1.0,
        ),
        reduction_k_grad_h=16,
    )

    # --- 2. System Assembly ---
    try:
        manager = OpenCLContextManager(kernel_source_dir=KERNEL_SOURCE_DIR)
        compute_env = manager.build_and_discover()
    except Exception as e:
        print(f"FATAL: Could not build OpenCL environment: {e}")
        exit(1)

    iris = load_iris()
    X_train_data = iris.data.astype(SCALAR_DTYPE)
    y_train_data = iris.target.astype(np.int32)
    batch_size = X_train_data.shape[0]

    iris_model_spec = ModelSpec(
        input_dim=X_train_data.shape[1],
        output_classes=len(np.unique(y_train_data)),
        hidden_dim=LOGICAL_HIDDEN_DIM,
        num_modules=LOGICAL_NUM_MODULES,
        simd_width=compute_env.arch_consts.get("simd_width"),
        cache_line_bytes=compute_env.arch_consts.get("global_mem_cacheline_size"),
    )

    param_space = ParameterSpace(spec=iris_model_spec)

    # --- 3. Orchestration and Execution ---
    orchestrator = TrainingOrchestrator(
        model_spec=iris_model_spec,
        param_space=param_space,
        compute_env=compute_env,
        hyperparams=HYPERPARAMS,
        adaptation_strategy=ADAPTATION_STRATEGY,
        problem_type_name=PROBLEM_TYPE,
        clipping_strategy_name=CLIPPING_STRATEGY,
        batch_size=batch_size,
    )
    orchestrator.train(X_train_data, y_train_data)
