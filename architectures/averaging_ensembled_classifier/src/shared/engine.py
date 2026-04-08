# src/shared/engine.py
"""Engine class — user-facing entry point for inference and training (ADR-018).

The Engine is the orchestration-tier component that produces WorkTickets for
submitted batches. It is stateless between ticket lifecycles (Choice 5A: ephemeral).

Shared-layer purity:
    This module imports only from Python stdlib, numpy, and other src/shared/
    modules. Backend-specific imports are deferred to the renderer_factory
    callable, which is injected at construction time or lazily resolved.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from .hardware_profile import HardwareProfile
from .model_spec import ModelSpec
from .optimizer_config import OptimizerConfig
from .plan_builder import build_act_plan, build_learn_plan
from .plan_renderer import PlanRenderer
from .precision_config import PrecisionConfig
from .problem_type_strategy import PlanCceStrategy, PlanProblemTypeStrategy
from .retrieval_future import RetrievalFuture
from .stabilization_policy import StabilizationPolicy
from .ticket import WorkTicket

if TYPE_CHECKING:
    from .parameter_space import ParameterSpace


def _default_renderer_factory(
    backend: str,
    hardware: HardwareProfile,
    precision: PrecisionConfig,
) -> PlanRenderer:
    """Default renderer factory with lazy backend imports.

    This function preserves shared-layer purity by importing backend modules
    only when called. The Engine module itself contains no backend imports.

    Backend preference order: CPU > OpenCL > Vulkan
    This ordering prioritizes the CPU backend for deterministic CI
    reproducibility and debuggability.

    Args:
        backend: Backend selector ("auto", "cpu", "opencl", "vulkan").
        hardware: Hardware profile for renderer configuration.
        precision: Precision configuration.

    Returns:
        A PlanRenderer instance for the selected backend.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    # Import build config to determine available backends
    try:
        from src._build_config import BACKEND_CPU, BACKEND_OPENCL, BACKEND_VULKAN  # type: ignore[import-not-found]
    except ImportError:
        raise RuntimeError(
            "_build_config.py not found. Run 'meson setup builddir' and then "
            "'ninja -C builddir' before using the Engine."
        )

    # Resolve "auto" to concrete backend
    if backend == "auto":
        if BACKEND_CPU:
            backend = "cpu"
        elif BACKEND_OPENCL:
            backend = "opencl"
        elif BACKEND_VULKAN:
            backend = "vulkan"
        else:
            raise RuntimeError(
                "No backends available. Build with at least one of: "
                "-Daec_backend_cpu=enabled, -Daec_backend_opencl=enabled, "
                "-Daec_backend_vulkan=enabled"
            )

    # Dispatch to backend-specific factory
    if backend == "cpu":
        if not BACKEND_CPU:
            raise RuntimeError("CPU backend not available. Rebuild with -Daec_backend_cpu=enabled")
        from src.backends.cpu import CPUPlanRenderer
        return CPUPlanRenderer()

    elif backend == "opencl":
        if not BACKEND_OPENCL:
            raise RuntimeError("OpenCL backend not available. Rebuild with -Daec_backend_opencl=enabled")
        # OpenCL requires complex initialization (context, queue, program).
        # For now, raise NotImplementedError; users can provide a custom renderer_factory.
        raise NotImplementedError(
            "OpenCL backend auto-initialization is not yet implemented. "
            "Provide a custom renderer_factory or use backend='cpu'."
        )

    elif backend == "vulkan":
        if not BACKEND_VULKAN:
            raise RuntimeError("Vulkan backend not available. Rebuild with -Daec_backend_vulkan=enabled")
        # Vulkan requires complex initialization (context, pipelines).
        # For now, raise NotImplementedError; users can provide a custom renderer_factory.
        raise NotImplementedError(
            "Vulkan backend auto-initialization is not yet implemented. "
            "Provide a custom renderer_factory or use backend='cpu'."
        )

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Expected one of: auto, cpu, opencl, vulkan")


class Engine:
    """User-facing entry point for inference and training (ADR-018).

    The Engine is stateless between ticket lifecycles (Choice 5A: ephemeral).
    It holds configuration (model spec, hardware profile, precision) and
    produces WorkTickets for submitted batches.

    Example usage (Event-Triggered mode):
        >>> engine = Engine(model_spec, parameter_space, hardware)
        >>> ticket = engine.submit(x_data)          # Returns immediately
        >>> prediction = ticket.get_prediction()   # Blocks on Act completion
        >>> ticket.resolve(y_data).wait()          # Dispatch Learn, wait

    Example usage (Sequential mode):
        >>> prediction = engine.train_batch(x_data, y_data)  # All-in-one
    """

    def __init__(
        self,
        model_spec: ModelSpec,
        parameter_space: "ParameterSpace",
        hardware_profile: HardwareProfile,
        *,
        precision: PrecisionConfig | None = None,
        backend: str = "auto",
        strategy: PlanProblemTypeStrategy | None = None,
        policy: StabilizationPolicy | None = None,
        optimizer: OptimizerConfig | None = None,
        renderer_factory: Callable[[str, HardwareProfile, PrecisionConfig], PlanRenderer] | None = None,
    ) -> None:
        """Initialize the Engine.

        Args:
            model_spec: Model architecture specification.
            parameter_space: Learnable parameters (weights, biases).
            hardware_profile: Target hardware characteristics.
            precision: Precision configuration (default: model_spec.precision).
            backend: Backend selector ("auto", "cpu", "opencl", "vulkan").
            strategy: Problem type strategy (default: PlanCceStrategy).
            policy: Stabilization policy for gradient clipping (default: auto-configured).
            optimizer: Optimizer hyperparameters (default: Adam with lr=0.001,
                β₁=0.9, β₂=0.999, precision-aware ε). See OptimizerConfig.
            renderer_factory: Optional callable to construct a PlanRenderer from
                (backend, hardware, precision). If None, uses default factory.
                This parameter preserves shared-layer purity by deferring
                backend imports to the factory implementation.
        """
        self._model_spec = model_spec
        self._parameter_space = parameter_space
        self._hardware_profile = hardware_profile
        self._precision = precision if precision is not None else model_spec.precision
        self._strategy = strategy if strategy is not None else PlanCceStrategy()

        # Configure stabilization policy if not provided
        if policy is None:
            self._policy = StabilizationPolicy(
                t_algorithmic=1.0,
                lambda_=0.1,
                compute_fp_format_max=self._precision.compute_fp_format_max,
            )
        else:
            self._policy = policy

        # Configure optimizer (ADR-029)
        self._optimizer = optimizer if optimizer is not None else OptimizerConfig()

        # Adam step counter (incremented each learn dispatch)
        self._adam_step: int = 0

        # Create renderer via factory
        factory = renderer_factory if renderer_factory is not None else _default_renderer_factory
        self._renderer = factory(backend, hardware_profile, self._precision)

    def submit(self, x_data: NDArray[np.floating]) -> WorkTicket:
        """Submit a batch for inference (Act phase).

        Returns immediately with a PENDING WorkTicket. The Act plan is
        dispatched to the device; call ticket.get_prediction() to block
        on completion.

        Args:
            x_data: Input data matrix, shape (N, D_in).

        Returns:
            WorkTicket in PENDING state.
        """
        act_future = self._dispatch_act_plan(x_data)
        return WorkTicket(x_data, act_future, self)

    def train_batch(
        self,
        x_data: NDArray[np.floating],
        y_data: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """Convenience method: submit, infer, resolve, and wait in one call.

        Equivalent to:
            ticket = engine.submit(x_data)
            prediction = ticket.get_prediction()
            ticket.resolve(y_data).wait()
            return prediction

        This is the minimal-ceremony pattern for sequential batch training.

        Args:
            x_data: Input data matrix, shape (N, D_in).
            y_data: Ground truth labels, shape (N,).

        Returns:
            Prediction probabilities, shape (N, C).
        """
        ticket = self.submit(x_data)
        prediction = ticket.get_prediction()
        ticket.resolve(y_data).wait()
        return prediction

    # -------------------------------------------------------------------
    # Internal methods (not part of public API)
    # -------------------------------------------------------------------

    def _dispatch_act_plan(self, x_data: NDArray[np.floating]) -> RetrievalFuture:
        """Build and render an Act-only plan, returning the inference future.

        This method constructs the Act plan (forward pass + inference retrieval)
        and dispatches it via the renderer. Input data is injected into the
        plan's ``input_data`` buffer via ``data_injections``.

        Args:
            x_data: Input data matrix, shape (N, D_in).

        Returns:
            RetrievalFuture for the inference_retrieval node.
        """
        batch_size = x_data.shape[0]
        plan = build_act_plan(
            self._model_spec,
            self._hardware_profile,
            self._strategy,
            batch_size,
        )

        # Inject input data and sample mask into plan buffers
        sample_mask = np.ones(batch_size, dtype=np.float32)
        injections = {
            "input_data": x_data,
            "sample_mask": sample_mask,
        }

        futures = self._renderer.render(plan, data_injections=injections)

        # The Act plan's terminal node has event_name="inference_event"
        return futures["inference_event"]

    def _dispatch_learn_plan(
        self,
        x_data: NDArray[np.floating],
        y_data: NDArray[np.integer],
    ) -> RetrievalFuture:
        """Build and render a Learn-only plan, returning the final batch future.

        Under Choice 4A (recompute), the Learn plan recomputes the forward pass.
        Input data and ground truth labels are injected into the plan's buffers
        via ``data_injections``.

        Args:
            x_data: Input data matrix, shape (N, D_in).
            y_data: Ground truth labels, shape (N,).

        Returns:
            RetrievalFuture for the final_batch_retrieval node.
        """
        batch_size = x_data.shape[0]
        self._adam_step += 1
        plan = build_learn_plan(
            self._model_spec,
            self._hardware_profile,
            self._strategy,
            batch_size,
            self._policy,
            activation_lifecycle="recompute",
            optimizer=self._optimizer,
            adam_step=self._adam_step,
        )

        # Inject input data, targets, and sample mask into plan buffers
        sample_mask = np.ones(batch_size, dtype=np.float32)
        injections: dict[str, NDArray] = {
            "input_data": x_data,
            "sample_mask": sample_mask,
            self._strategy.required_targets_buffer_name: y_data,
        }

        futures = self._renderer.render(plan, data_injections=injections)

        # The Learn plan's terminal node has event_name="final_batch_event"
        return futures["final_batch_event"]
