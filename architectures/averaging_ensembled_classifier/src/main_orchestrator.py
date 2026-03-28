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
from .shared.model_spec import ModelSpec, Float32ModelSpec, Float16ModelSpec
from .backends.opencl.context import (
    OpenCLContextManager,
    ComputeEnvironment,
    Float32ComputeEnvironment,
    Float16ComputeEnvironment,
    Float32Context,
    Float16Context,
)
from .shared.parameter_space import ParameterSpace
from .backends.opencl.execution_plan import (
    ExecutionPlan,
    DataLifecyclePolicy,
    DependencyProvider,
    CacheProvider,
    ComputeOnceProvider,
    StagedComputationProvider,
)
from .shared.problem_type_strategy import (
    ProblemTypeStrategy,
    CceStrategy,
    BceStrategy,
)
from .backends.opencl.kernel_bindings import ForwardPassSignature
from .backends.opencl.launcher_infra import Services, BufferManager, KernelExecutor, BufferHandle
from .backends.opencl.compute_patterns import ReductionPlan
from .shared.workload_primitives import TilingScheme
from .shared.stabilization_policy import StabilizationPolicy
from .backends.opencl.batch_processor import BatchProcessor
from .backends.opencl import graph_recipes as recipes


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

        # --- Zero-initialize ALL buffers ---
        # WHY: OpenCL's clCreateBuffer with CL_MEM_READ_WRITE does NOT
        # guarantee zero-initialized memory.  Buffers may contain arbitrary
        # leftover GPU data.  This is critical for Adam optimizer state
        # (m1, m2 must start at zero), bias buffers, and any accumulation
        # buffer.  We zero everything, then selectively overwrite the
        # buffers that require non-zero starting values below.
        q = self.compute_env.cl_bundle.queue
        for name in all_layouts:
            ref = bm.get_handle_by_name(name)
            buf_shape, buf_dtype = bm.get_spec(ref)
            zeros = np.zeros(buf_shape, dtype=buf_dtype)
            cl.enqueue_copy(q, bm.get_cl_buffer(ref), zeros)

        # --- Initialize buffers that require non-zero starting values ---
        rng = np.random.default_rng(42)

        # WHY: Every sample is "active" in full-batch training. The mask must
        # be all-ones so that loss/gradient kernels process every sample.
        sample_mask = np.ones(batch_size, dtype=spec.SCALAR_NP_TYPE)
        cl.enqueue_copy(q, bm.get_cl_buffer("sample_mask"), sample_mask)

        # WHY: Temperatures control the sharpness of the softmax per module.
        # An initial value of 1.0 gives a standard softmax; zero would cause
        # division-by-zero in the kernel.
        temps = np.ones(spec.num_modules, dtype=spec.SCALAR_NP_TYPE)
        cl.enqueue_copy(q, bm.get_cl_buffer("temperatures"), temps)

        # WHY: Learnable weight matrices require non-zero initialization so
        # that the model breaks symmetry and gradient flow is non-trivial.
        # We use Xavier (Glorot) uniform initialization scaled to the logical
        # dimensions, writing into the full padded buffer shape so that
        # padding elements remain zero and do not affect computation.

        # Shared layer: shape (padded_hidden_dim, padded_input_dim)
        # WHY: The forward_pass kernel reads W[h, i] = flat[h * padded_input + i],
        # so the numpy array with shape (padded_hidden, padded_input) stores
        # arr[h, i] at flat[h * padded_input + i] — matching the kernel exactly.
        sw_shape, _ = bm.get_spec(bm.get_handle_by_name("shared_weights"))
        limit_sw = np.sqrt(6.0 / (spec.input_dim + spec.hidden_dim))
        sw_init = np.zeros(sw_shape, dtype=spec.SCALAR_NP_TYPE)
        sw_init[:spec.hidden_dim, :spec.input_dim] = rng.uniform(
            -limit_sw, limit_sw, (spec.hidden_dim, spec.input_dim)
        ).astype(spec.SCALAR_NP_TYPE)
        cl.enqueue_copy(q, bm.get_cl_buffer("shared_weights"), sw_init)

        # Module layer: shape (num_modules, padded_hidden_dim, padded_class_dim)
        mw_shape, _ = bm.get_spec(bm.get_handle_by_name("module_weights"))
        limit_mw = np.sqrt(6.0 / (spec.hidden_dim + spec.output_classes))
        mw_init = np.zeros(mw_shape, dtype=spec.SCALAR_NP_TYPE)
        mw_init[:, :spec.hidden_dim, :spec.output_classes] = rng.uniform(
            -limit_mw, limit_mw, (spec.num_modules, spec.hidden_dim, spec.output_classes)
        ).astype(spec.SCALAR_NP_TYPE)
        cl.enqueue_copy(q, bm.get_cl_buffer("module_weights"), mw_init)

        q.finish()

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
            # WHY: ComputeOnceProvider instead of CacheProvider. The forward pass
            # must execute AFTER data is uploaded to the input buffer (which happens
            # during BatchProcessor.run). Pre-executing at plan-creation time would
            # read uninitialised input data, producing all-zero hidden activations.
            # ComputeOnceProvider defers the launch until the first resolve() call
            # (which is made by BatchProcessor with data-upload events in wait_for),
            # then caches the result for subsequent consumers.
            fwd_sig = ForwardPassSignature(
                _buffer_mgr=svs.bm,
                _arch_consts=svs.arch_consts,
                in_ref=svs.bm.get_handle_by_name("input"),
                mask_ref=svs.bm.get_handle_by_name("sample_mask"),
                w_ref=svs.bm.get_handle_by_name("shared_weights"),
                b_ref=svs.bm.get_handle_by_name("shared_biases"),
                h_out_ref=svs.bm.get_handle_by_name("hidden_activations"),
                h_mask_out_ref=svs.bm.get_handle_by_name("hidden_mask"),
                batch_chunk_offset=np.uint32(0),
                batch_chunk_count=np.uint32(batch_size),
            )
            policy_providers["hidden_activations"] = ComputeOnceProvider(
                signature=fwd_sig,
                output_handle=svs.bm.get_handle_by_name("hidden_activations"),
            )
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

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """The main training loop, demonstrating the Plan -> Execute pattern.

        Returns the final epoch's class-probability matrix (batch_size × output_classes)
        after all learning events have completed.
        """
        batch_size = X_train.shape[0]
        final_probs: np.ndarray = np.empty(0)

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
            infer_event.wait()
            final_probs = probs_view.get()
            print(f"--- Step {self.global_step} Complete. ---")
            self.global_step += 1

        return final_probs


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
