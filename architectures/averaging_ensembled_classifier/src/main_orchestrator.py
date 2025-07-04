# main_orchestrator.py

"""
The System's "Strategist": The Definitive Host-Side Orchestrator.

Jurisdictional Mandate:
This module constitutes the highest level of host-side control. Its sole
and sacred jurisdiction is to translate a high-level experimental goal
(defined by model shape, hyperparameters, and adaptation strategies) into a
single, immutable, and declarative `ExecutionPlan`. It owns the training
lifecycle (i.e., the epoch loop) but humbly delegates all per-batch tactical
execution to subordinate modules.

Architectural Role:
This class embodies the "Strategist" pattern. It assembles the system's core
services, authors the strategic plan, and then passes that plan to a
transient `BatchProcessor` to be conducted. It makes no direct kernel calls;
its work is the pure art of composition, ensuring that the system's execution
is a direct and verifiable reflection of its stated intent.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, cast
from functools import partial

import numpy as np
import pyopencl as cl
from sklearn.datasets import load_iris
from sklearn.utils import Bunch

# --- Foundational Architectural Primitives ---
# WHY: The Orchestrator's primary function is to compose these primitive "nouns"
# into a coherent, executable "sentence" (the ExecutionPlan).
from .model_spec import ModelSpec, Float32ModelSpec, Float16ModelSpec
from .cl_context_manager import (
    OpenCLContextManager,
    ComputeEnvironment,
    Float32ComputeEnvironment,
    Float16ComputeEnvironment,
    Float32Context,
    Float16Context,
)
from .parameter_space import ParameterSpace
from .execution_plan import (
    ExecutionPlan,
    DataLifecyclePolicy,
    DependencyProvider,
    CacheProvider,
    StagedComputationProvider,
    ProblemTypeStrategy,
    CceStrategy,
    BceStrategy,
)
from .launcher_infra import Services, BufferManager, KernelExecutor, BufferHandle
from .compute_patterns import ReductionPlan
from .workload_primitives import TilingScheme
from .stabilization_policy import StabilizationPolicy
from .batch_processor import BatchProcessor
from . import graph_recipes as recipes


@dataclass(frozen=True)
class StabilizationConfig:
    """A structured configuration primitive for the stabilization policy."""
    max_grad_norm: float
    lambda_: float


@dataclass(frozen=True)
class TrainingHyperparams:
    """The canonical, type-safe manifest for all training hyperparameters."""
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
    """The System Owner, "Strategist," and author of the ExecutionPlan."""

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
        # WHY: The constructor acts as the "System Assembler," gathering and
        # owning all foundational components for the entire training run.
        self.model_spec = model_spec
        self.param_space = param_space
        self.compute_env = compute_env
        self.hyperparams = hyperparams
        self.adaptation_strategy = adaptation_strategy
        self.problem_type_name = problem_type_name
        self.clipping_strategy_name = clipping_strategy_name
        self.global_step = 1

        # WHY: The `Services` bundle is a key dependency injection pattern.
        # It allows core components to be passed cleanly and explicitly
        # through the system layers, avoiding global state.
        bm = BufferManager(self.compute_env.cl_bundle.context)
        ex = KernelExecutor(self.compute_env.cl_bundle.program)
        self.services = Services(
            q=self.compute_env.cl_bundle.queue, ex=ex, bm=bm, model_spec=model_spec, arch_consts=compute_env.arch_consts
        )

        self.stream_chunks = 4
        self._setup_buffers(batch_size, self.stream_chunks)

    def _setup_buffers(self, batch_size: int, num_stream_chunks: int):
        """Orchestrates memory allocation by delegating to the ParameterSpace."""
        # WHY: This method upholds the "Primacy of Memory Strategy." The
        # Orchestrator does not decide layouts; it asks the `ParameterSpace`
        # for the complete memory plan and instructs the `BufferManager` to
        # execute that plan, perfectly separating strategic intent from
        # implementation details.
        bm, spec = self.services.bm, self.model_spec
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
            bm.create_named_buffer(name, layout, spec.SCALAR_NP_TYPE)

    def _create_execution_plan(self, batch_size: int) -> ExecutionPlan:
        """Authors the complete, immutable execution strategy for one batch."""
        svs, spec, h_params = self.services, self.model_spec, self.hyperparams
        policy_providers: Dict[str, DependencyProvider] = {}

        # --- Phase 1: Formulate Strategic Primitives ---
        grid = TilingScheme(
            num_module_chunks=(spec.num_modules + 15) // 16,
            num_class_chunks=(spec.output_classes + 15) // 16,
            total_modules=spec.num_modules,
            total_classes=spec.output_classes,
        )
        reduction_plan = ReductionPlan(k=self.compute_env.arch_consts.optimal_workgroup_size_1d_reduction)

        # --- Phase 2: Instantiate Polymorphic Strategy Objects ---
        # WHY: This block uses the Strategy Pattern to select the correct set of
        # kernel factories, making the rest of the system blissfully unaware
        # of the specific loss function being used (CCE vs. BCE).
        problem_strategy: ProblemTypeStrategy
        if self.problem_type_name == "CCE":
            problem_strategy = CceStrategy(targets_cce_ref=svs.bm.get_handle_by_name("targets_cce"))
        else:
            problem_strategy = BceStrategy(targets_bce_ref=svs.bm.get_handle_by_name("targets_bce"))

        fp_max = np.finfo(spec.SCALAR_NP_TYPE).max
        stabilization_policy = StabilizationPolicy(
            t_algorithmic=h_params.stabilization.max_grad_norm,
            lambda_=h_params.stabilization.lambda_,
            fp_format_max=float(fp_max),
        )

        # --- Phase 3: Define Data Lifecycle Policies (The Core of Adaptation) ---

        # WHY: This function and the `partial` object below are an elegant
        # solution to a circular dependency. The StagedComputationProvider needs a
        # reference to the final `ExecutionPlan`, but it must be created *before*
        # the plan itself is constructed. This placeholder approach resolves the paradox.
        def _resolve_grad_h_with_plan(
            queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event], *, plan: ExecutionPlan
        ) -> Tuple[BufferHandle, cl.Event]:
            return recipes.build_final_grad_h_reduction_path(svs, plan, wait_for)

        policy_providers["summed_grad_hidden_activations"] = StagedComputationProvider(
            computation_fn=partial(_resolve_grad_h_with_plan)
        )

        # Here, the high-level adaptation strategy is translated into a concrete
        # `DependencyProvider` for the `hidden_activations` themselves.
        if self.adaptation_strategy == "CACHE":
            h_ref, h_ready_evt = recipes.execute_forward_pass(svs, batch_size, deps=[])
            policy_providers["hidden_activations"] = CacheProvider(handle=h_ref, ready_event=h_ready_evt)
        elif self.adaptation_strategy == "RECOMPUTE_GRAD_H":
            # Architecturally correct: no provider is defined. `hidden_activations`
            # becomes a transient internal artifact of the recipe.
            pass
        else:
            raise ValueError(f"Unknown adaptation strategy: '{self.adaptation_strategy}'")

        # --- Phase 4: Assemble and Finalize the ExecutionPlan ---
        plan = ExecutionPlan(
            grid=grid,
            reduction_plan=reduction_plan,
            lifecycle_policy=DataLifecyclePolicy(providers=policy_providers),
            effective_batch_size=batch_size,
            problem_type=problem_strategy,
            clipping_strategy=self.clipping_strategy_name,
            stabilization_policy=stabilization_policy,
            hyperparams=self.hyperparams,
            shared_backprop_stream_chunks=self.stream_chunks,
            adaptation_strategy=self.adaptation_strategy,
        )

        # The Final Polish: The completed `plan` is now bound to the placeholder function.
        # This replaces the `cast` with a verifiable `isinstance` check. This is
        # not a command to the type checker, but a question. If the provider is
        # of the expected type, the type checker understands that its attributes
        # are safe to access within the `if` block. This preserves the
        # unbroken chain of static verification.
        for provider in policy_providers.values():
            if isinstance(provider, StagedComputationProvider):
                provider.computation_fn.keywords["plan"] = plan

        return plan

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """The main training loop, demonstrating the Plan -> Execute pattern."""
        batch_size = X_train.shape[0]

        for epoch in range(self.hyperparams.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.hyperparams.epochs}, Step {self.global_step} ---")

            # 1. Author the complete, high-level strategy for this batch.
            plan = self._create_execution_plan(batch_size)

            # 2. Instantiate a dedicated "Conductor" to execute the plan.
            processor = BatchProcessor(self.services, self.param_space, plan)

            # 3. Delegate execution and receive asynchronous event handles.
            learn_event, infer_event, probs_view = processor.run(X_train, y_train, self.global_step)

            # Blocks for learning to complete before the next step.
            learn_event.wait()
            print(f"--- Step {self.global_step} Complete. ---")
            self.global_step += 1


if __name__ == "__main__":
    # This block serves as the "System Assembler" at the application's entry point.
    # It defines the experiment, instantiates components, and starts the process.

    # --- 1. Experiment Configuration ---
    LOGICAL_HIDDEN_DIM = 32
    LOGICAL_NUM_MODULES = 8
    ADAPTATION_STRATEGY = "CACHE"
    PROBLEM_TYPE = "CCE"
    CLIPPING_STRATEGY = "GLOBAL"
    KERNEL_SOURCE_DIR = "architectures/averaging_ensembled_classifier/kernels"

    HYPERPARAMS = TrainingHyperparams(
        epochs=5,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-7,
        temp_min=0.1,
        temp_max=10.0,
        stabilization=StabilizationConfig(max_grad_norm=1.0, lambda_=1.0),
        reduction_k_grad_h=16,
    )

    # --- 2. Precision-Aware System Assembly ---
    try:
        manager = OpenCLContextManager(kernel_source_dir=KERNEL_SOURCE_DIR)
        # The first critical decision: select the precision context. This choice
        # dictates the concrete types for the rest of the assembly process.
        compute_env = manager.build_and_discover(Float32Context)
    except Exception as e:
        print(f"FATAL: Could not build OpenCL environment: {e}")
        exit(1)

    # Load data and use the *concrete* compute environment's type for casting.
    iris = cast(Bunch, load_iris())
    X_train_data = iris.data.astype(compute_env.SCALAR_NP_TYPE)
    y_train_data = iris.target.astype(np.int32)
    batch_size = X_train_data.shape[0]

    # Use the concrete environment type to select the concrete ModelSpec. This
    # creates a verifiable, end-to-end chain of type consistency.
    iris_model_spec: ModelSpec
    if isinstance(compute_env, Float32ComputeEnvironment):
        iris_model_spec = Float32ModelSpec(
            input_dim=X_train_data.shape[1],
            output_classes=len(np.unique(y_train_data)),
            hidden_dim=LOGICAL_HIDDEN_DIM,
            num_modules=LOGICAL_NUM_MODULES,
            simd_width=compute_env.arch_consts.simd_width,
            cache_line_bytes=compute_env.arch_consts.global_mem_cacheline_size,
        )
    else:  # Example for extending to other precisions
        iris_model_spec = Float16ModelSpec(
            input_dim=X_train_data.shape[1],
            output_classes=len(np.unique(y_train_data)),
            hidden_dim=LOGICAL_HIDDEN_DIM,
            num_modules=LOGICAL_NUM_MODULES,
            simd_width=compute_env.arch_consts.simd_width,
            cache_line_bytes=compute_env.arch_consts.global_mem_cacheline_size,
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
