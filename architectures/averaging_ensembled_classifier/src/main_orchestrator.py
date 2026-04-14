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

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .shared.fp8_scaling import (
    FP8ScaleInfo,
    FP8_SCALES_VERSION,
    compute_fp8_scale,
    apply_fp8_scale,
    unapply_fp8_scale,
)
from .shared.model_spec import ModelSpec
from .shared.parameter_space import ParameterSpace
from .shared.precision_config import PrecisionConfig, FP8_DTYPES
from .shared.hardware_profile import HardwareProfile
from .shared.stabilization_policy import StabilizationPolicy
from .shared.plan_builder import build_act_plan, build_learn_plan
from .shared.plan_renderer import PlanRenderer
from .shared.problem_type_spec import PlanCceStrategy, PlanBceStrategy, PlanProblemTypeStrategy


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
        activation_lifecycle: Literal["cache", "recompute"] = "recompute",
    ):
        self.model_spec = model_spec
        self.param_space = ParameterSpace(spec=model_spec)
        self.hardware = hardware
        self.renderer = renderer
        self.hyperparams = hyperparams
        self.batch_size = batch_size
        self.activation_lifecycle: Literal["cache", "recompute"] = activation_lifecycle

        self.strategy: PlanProblemTypeStrategy
        if problem_type_name == "CCE":
            self.strategy = PlanCceStrategy()
        elif problem_type_name == "BCE":
            self.strategy = PlanBceStrategy()
        else:
            raise ValueError(f"Unknown problem type: {problem_type_name!r}")

        self.stabilization_policy = StabilizationPolicy(
            t_algorithmic=hyperparams.stabilization.max_grad_norm,
            lambda_=hyperparams.stabilization.lambda_,
            compute_fp_format_max=model_spec.precision.compute_fp_format_max,
        )

        # FP8 support (ADR-025 §8)
        self.is_fp8_storage = model_spec.precision.storage_dtype in FP8_DTYPES
        # Per-buffer scale tracking (for FP8).
        # CONSTRAINT: Keys must be static buffer names derived from plan node names
        # (deterministic and bounded). Do NOT use per-step or per-iteration keys —
        # that would cause unbounded memory growth. There is intentionally no
        # eviction policy; the dict size is bounded by the number of plan nodes.
        self._activation_scales: dict[str, FP8ScaleInfo] = {}

    # ------------------------------------------------------------------
    # FP8 scaling helpers (ADR-025 §8)
    # ------------------------------------------------------------------

    def _prepare_storage_buffer(
        self,
        name: str,
        tensor: np.ndarray,
        precision_role: str,
    ) -> np.ndarray:
        """Prepare tensor for storage-role buffer.

        For FP8, applies scaling if needed and tracks scale for inverse.
        """
        if precision_role != "storage":
            return tensor

        if not self.is_fp8_storage:
            return tensor

        # Compute and apply FP8 scaling
        scale_info = compute_fp8_scale(tensor, self.model_spec.precision)
        self._activation_scales[name] = scale_info

        # Invariant: compute_fp8_scale returns scale=1.0 as a Python float literal
        # (identity path), so this exact-equality check is safe — no floating-point
        # rounding can produce a near-1.0 value that should be treated as identity.
        if scale_info.scale != 1.0:
            return apply_fp8_scale(tensor, scale_info)
        return tensor

    def _restore_from_storage(
        self,
        name: str,
        tensor: np.ndarray,
    ) -> np.ndarray:
        """Restore tensor values after reading from storage-role buffer."""
        if name not in self._activation_scales:
            return tensor

        scale_info = self._activation_scales[name]
        # Same invariant as _prepare_storage_buffer: inv_scale=1.0 is exact identity.
        if scale_info.inv_scale != 1.0:
            return unapply_fp8_scale(tensor, scale_info)
        return tensor

    def _validate_gradient_fits_fp8(self, gradient: np.ndarray) -> None:
        """Assert gradients are within FP8 range (should pass due to QSP)."""
        if not self.is_fp8_storage:
            return

        max_grad = np.abs(gradient).max()
        storage_max = self.model_spec.precision.storage_fp_format_max

        if max_grad > storage_max:
            # This should not happen under correct Quadratic Scaling Policy
            warnings.warn(
                f"Gradient max {max_grad:.2e} exceeds FP8 max {storage_max}. "
                f"Values will saturate. Check stabilization policy."
            )

    # ------------------------------------------------------------------
    # Checkpoint support (ADR-025 §8)
    # ------------------------------------------------------------------

    def checkpoint_state(self) -> dict[str, Any]:
        """Export state for checkpointing."""
        state: dict[str, Any] = {
            "training_step": getattr(self, '_training_step', 0),
        }
        if self.is_fp8_storage:
            # Convert NamedTuple to dict for JSON/pickle serialization
            # Embed version INSIDE the fp8_scales dict (not as sibling key) for atomicity
            state["fp8_scales"] = {
                "_version": FP8_SCALES_VERSION,  # Reserved metadata key (leading underscore)
                "scales": {
                    name: info._asdict() for name, info in self._activation_scales.items()
                },
            }
        return state

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._training_step: int = state.get("training_step", 0)  # pyright: ignore[reportUnknownMemberType]

        # Restore FP8 scales with version checking and structure validation
        if "fp8_scales" in state:
            fp8_data: dict[str, Any] = state["fp8_scales"]
            checkpoint_version: Any = fp8_data.get("_version", 0)

            if checkpoint_version != FP8_SCALES_VERSION:
                warnings.warn(
                    f"FP8 scale format mismatch: checkpoint v{checkpoint_version}, "
                    f"current v{FP8_SCALES_VERSION}. Scales will be recomputed."
                )
                self._activation_scales = {}  # Recompute on next forward pass
            else:
                # Defensive construction: catch TypeError if FP8ScaleInfo structure changed
                try:
                    scales_dict: dict[str, Any] = fp8_data.get("scales", {})
                    self._activation_scales = {
                        name: FP8ScaleInfo(**info) for name, info in scales_dict.items()
                    }
                except TypeError as e:
                    warnings.warn(
                        f"FP8ScaleInfo structure incompatible with checkpoint: {e}. "
                        f"Scales will be recomputed."
                    )
                    self._activation_scales = {}
        else:
            # Pre-FP8 checkpoint (Phase 8 or earlier) — no fp8_scales key is valid
            self._activation_scales = {}

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
            if "inference_event" in act_futures:
                final_probs = act_futures["inference_event"].result()
                # Ensemble averaging: if the retrieval returns per-module
                # probabilities (M, N, C), average across modules.
                if final_probs.ndim == 3:
                    final_probs = final_probs.mean(axis=0)

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
    from sklearn.datasets import load_iris  # pyright: ignore[reportMissingTypeStubs]
    iris: Any = load_iris()  # pyright: ignore[reportUnknownVariableType]
    X_train: np.ndarray = iris.data.astype(np.float32)  # type: ignore[union-attr]  # pyright: ignore[reportUnknownMemberType]
    y_train: np.ndarray = iris.target.astype(np.int32)  # type: ignore[union-attr]  # pyright: ignore[reportUnknownMemberType]

    model_spec = ModelSpec.float32(
        input_dim=int(X_train.shape[1]),
        output_classes=int(np.unique(y_train).shape[0]),
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
        batch_size=int(X_train.shape[0]),
    )
    orchestrator.train(X_train, y_train)
