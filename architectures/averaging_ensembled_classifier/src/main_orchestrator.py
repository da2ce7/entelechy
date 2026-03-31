# main_orchestrator.py

"""
The System's Strategist: Backend-Agnostic Training Orchestration.

Translates an experimental configuration into immutable ExecutionPlans
and delegates execution to a PlanRenderer. All backend interaction is
mediated through the PlanRenderer Protocol — the orchestrator never
imports backend-specific modules directly (except for backend selection
in the entry point).

Phase 6: Rewritten to use plan-model dispatch exclusively (Option B:
direct PlanRenderer). No legacy module references remain.
"""

from dataclasses import dataclass

import numpy as np

from .shared.model_spec import ModelSpec
from .shared.parameter_space import ParameterSpace
from .shared.precision_config import PrecisionConfig
from .shared.hardware_profile import HardwareProfile
from .shared.stabilization_policy import StabilizationPolicy
from .shared.plan_builder import build_act_plan, build_learn_plan
from .shared.plan_renderer import PlanRenderer
from .shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy, PlanProblemTypeStrategy


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


class TrainingOrchestrator:
    """Backend-agnostic training orchestrator using plan-model dispatch.

    Accepts any PlanRenderer implementation. Constructs immutable
    ExecutionPlans via the shared-layer PlanBuilder and delegates
    execution to the renderer.
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        hardware: HardwareProfile,
        renderer: PlanRenderer,
        hyperparams: TrainingHyperparams,
        problem_type_name: str,
        batch_size: int,
        activation_lifecycle: str = "recompute",
    ):
        self.model_spec = model_spec
        self.param_space = ParameterSpace(spec=model_spec)
        self.hardware = hardware
        self.renderer = renderer
        self.hyperparams = hyperparams
        self.batch_size = batch_size
        self.activation_lifecycle = activation_lifecycle

        self.strategy: PlanProblemTypeStrategy
        if problem_type_name == "CCE":
            self.strategy = PlanCceStrategy()
        elif problem_type_name == "BCE":
            self.strategy = PlanBceStrategy()
        else:
            raise ValueError(f"Unknown problem type: {problem_type_name!r}")

        fp_max = float(np.finfo(model_spec.precision.numpy_dtype).max)
        self.stabilization_policy = StabilizationPolicy(
            t_algorithmic=hyperparams.stabilization.max_grad_norm,
            lambda_=hyperparams.stabilization.lambda_,
            fp_format_max=fp_max,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Run the training loop, returning final-epoch class probabilities.

        Each epoch: build Act plan → render → build Learn plan → render → wait.
        """
        final_probs: np.ndarray = np.empty(0)

        for epoch in range(self.hyperparams.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.hyperparams.epochs} ---")

            # Act phase: forward pass + inference
            act_plan = build_act_plan(
                model_spec=self.model_spec,
                hardware=self.hardware,
                strategy=self.strategy,
                batch_size=self.batch_size,
            )
            act_futures = self.renderer.render(act_plan)

            # Extract probabilities from Act phase
            if "final_probs" in act_futures:
                final_probs = act_futures["final_probs"].result()

            # Learn phase: gradient computation + parameter update
            learn_plan = build_learn_plan(
                model_spec=self.model_spec,
                hardware=self.hardware,
                strategy=self.strategy,
                batch_size=self.batch_size,
                policy=self.stabilization_policy,
                activation_lifecycle=self.activation_lifecycle,
            )
            learn_futures = self.renderer.render(learn_plan)

            # Wait for all learn-phase futures to complete
            for future in learn_futures.values():
                future.wait()

            print(f"--- Epoch {epoch + 1} Complete. ---")

        return final_probs


if __name__ == "__main__":
    import os

    # --- 1. Experiment Configuration ---
    HIDDEN_DIM = 32
    NUM_MODULES = 8
    PROBLEM_TYPE = "CCE"

    HYPERPARAMS = TrainingHyperparams(
        epochs=5,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-7,
        temp_min=0.1,
        temp_max=10.0,
        stabilization=StabilizationConfig(max_grad_norm=1.0, lambda_=1.0),
    )

    # --- 2. Backend selection (CPU as default; OpenCL/Vulkan as alternatives) ---
    BACKEND = os.environ.get("AEC_BACKEND", "cpu").lower()
    ARCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    KERNEL_DIR = os.path.join(ARCH_ROOT, "kernels")

    if BACKEND == "cpu":
        from .backends.cpu.renderer import CPUPlanRenderer
        from .backends.cpu.discovery import discover_hardware
        hardware = discover_hardware()
        renderer = CPUPlanRenderer()

    elif BACKEND == "opencl":
        import pyopencl as cl
        from .backends.opencl.discovery import discover_hardware as ocl_discover
        from .backends.opencl.context import load_and_compile_kernels_from_path
        from .backends.opencl.renderer import OpenCLPlanRenderer
        from .backends.opencl.type_mapping import build_compiler_flags
        from .backends.opencl.kernel_bindings.dispatch_table import build_dispatch_table
        from .backends.opencl.kernel_bindings.binding_phase_2_learn_C import (
            AggregateRegisterReduceBinding,
            AggregateLocalReduceBinding,
            ClipIntermediateGradBinding,
            ReduceKFanInAndClipBinding,
        )

        ctx = cl.create_some_context(interactive=False)
        device = ctx.devices[0]
        queue = cl.CommandQueue(ctx, device)
        hardware = ocl_discover(device)
        prec = PrecisionConfig.float32()
        flags = build_compiler_flags(prec, hardware, c_tile_size=16)
        program = load_and_compile_kernels_from_path(ctx, device, flags, KERNEL_DIR)
        dispatch_table = build_dispatch_table()
        renderer = OpenCLPlanRenderer(
            context=ctx, queue=queue, program=program,
            kernel_bindings=dispatch_table, hardware=hardware,
        )
        renderer.set_reduction_bindings(
            register_reduce=AggregateRegisterReduceBinding(),
            local_reduce=AggregateLocalReduceBinding(),
            clip_intermediate=ClipIntermediateGradBinding(),
            k_fan_in=ReduceKFanInAndClipBinding(),
        )

    elif BACKEND == "vulkan":
        from .backends.vulkan.context import VulkanContext
        from .backends.vulkan.renderer import VulkanPlanRenderer
        from .backends.vulkan.discovery import discover_hardware as vk_discover

        vk_ctx = VulkanContext(enable_validation=False)
        hardware = vk_discover(vk_ctx)
        renderer = VulkanPlanRenderer(context=vk_ctx)

    else:
        raise ValueError(f"Unknown backend: {BACKEND!r}. Use 'cpu', 'opencl', or 'vulkan'.")

    # --- 3. Model assembly ---
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_train = iris.data.astype(np.float32)  # type: ignore[union-attr]
    y_train = iris.target.astype(np.int32)  # type: ignore[union-attr]

    model_spec = ModelSpec.float32(
        input_dim=X_train.shape[1],
        output_classes=len(np.unique(y_train)),
        hidden_dim=HIDDEN_DIM,
        num_modules=NUM_MODULES,
        simd_width=hardware.simd_width,
        cache_line_bytes=hardware.cache_line_bytes,
    )

    # --- 4. Orchestration ---
    orchestrator = TrainingOrchestrator(
        model_spec=model_spec,
        hardware=hardware,
        renderer=renderer,
        hyperparams=HYPERPARAMS,
        problem_type_name=PROBLEM_TYPE,
        batch_size=X_train.shape[0],
    )
    orchestrator.train(X_train, y_train)
