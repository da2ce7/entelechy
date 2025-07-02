# execution_plan.py

"""
(REV 2) The Definitive Implementation of the Strategic Execution Plan Abstraction.

This version introduces a major architectural enhancement: the `ProblemTypeStrategy`
abstraction. This pattern replaces the brittle "stringly-typed" approach to
selecting loss functions (`problem_type="CCE"`) with a robust, polymorphic
class hierarchy.

This change perfectly decouples the sequence-oriented recipes from the
implementation details of any given loss function. The recipes no longer need
`if/else` blocks to handle different problem types; they simply ask the strategy
object provided in the plan to create the correct kernel signature, upholding
the Open/Closed Principle.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable

# --- Architectural Imports ---
try:
    import pyopencl as cl
    from .launcher_infra import BufferHandle, KernelSignature, KernelExecutor
    from .compute_patterns import ReductionPlan
    from .workload_primitives import TilingScheme, WorkTile

    # Import all signatures needed by the strategy factories
    from .kernel_signatures import (
        ComputeProbsLossCceChunkSignature,
        ComputeProbsLossBceChunkSignature,
        CalculateModuleParamGradsCceSignature,
        CalculateModuleParamGradsBceSignature,
        BackpropErrorToHiddenChunkCceSignature,
        BackpropErrorToHiddenChunkBceSignature,
        CalculateChunkTempGradientsCceSignature,
        CalculateChunkTempGradientsBceSignature,
    )
except ImportError:
    # Create mock types for standalone review and documentation generation
    cl = type("cl", (), {"Event": type("Event", (), {})})
    BufferHandle = type("BufferHandle", (), {"id": int})
    KernelSignature = type("KernelSignature", (), {})
    KernelExecutor = type("KernelExecutor", (), {})
    TilingScheme = type("TilingScheme", (), {})
    ReductionPlan = type("ReductionPlan", (), {})
    WorkTile = type("WorkTile", (), {})


# === Abstraction Level 1: The DependencyProvider Contract ===


class DependencyProvider(abc.ABC):
    """
    An abstract contract describing how a data dependency is fulfilled.

    This pattern is the heart of the dynamic adaptation strategy. An object
    implementing this interface represents a promise to provide a specific
    buffer handle and the event signaling its readiness, without exposing
    *how* that is achieved (be it from a cache or through on-demand computation).
    """

    @abc.abstractmethod
    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        """
        The core action. Fulfills the promise to provide the dependency.

        Args:
            queue: The OpenCL command queue for potential kernel launches.
            ex: The KernelExecutor for launching kernels.
            wait_for: A list of events that must complete before this resolution
                      can begin.

        Returns:
            A tuple containing:
            1. The `BufferHandle` for the requested data.
            2. The `cl.Event` that signals when the data is ready for consumption.
        """
        pass


@dataclass(frozen=True)
class CacheProvider(DependencyProvider):
    """
    A concrete provider for dependencies that are already computed and resident
    in VRAM. This is the "do nothing" strategy.
    """

    handle: BufferHandle
    ready_event: cl.Event

    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        """
        Simply returns the pre-computed handle and event. Ignores all inputs
        as no new computation is required.
        """
        return self.handle, self.ready_event


@dataclass(frozen=True)
class RecomputeProvider(DependencyProvider):
    """
    A concrete provider for dependencies that are computed on-demand by
    launching a single kernel.
    """

    signature: KernelSignature
    output_handle: BufferHandle

    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        """
        Launches the stored kernel signature and returns its output handle
        and the resulting completion event.
        """
        event = ex.launch(queue, self.signature, wait_for=wait_for)
        return self.output_handle, event


@dataclass(frozen=True)
class StagedComputationProvider(DependencyProvider):
    """
    A powerful, generic provider for complex dependencies that require a
    multi-step computation sequence (e.g., a permutation followed by a reduction).
    """

    # A callable that encapsulates the entire multi-kernel launch sequence.
    # It must adhere to the same signature as the `resolve` method.
    computation_fn: Callable[[cl.CommandQueue, KernelExecutor, List[cl.Event]], Tuple[BufferHandle, cl.Event]]

    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        """
        Delegates the entire resolution logic to the provided computation function.
        This allows for arbitrary complexity without polluting the provider itself.
        """
        return self.computation_fn(queue, ex, wait_for)


# === Abstraction Level 2: The Policy Registries ===


@dataclass(frozen=True)
class DataLifecyclePolicy:
    """
    A registry that maps a conceptual buffer's name to its concrete provider.

    This object acts as a "switchboard" for the BatchProcessor. When a dependency
    is needed, the processor queries this policy to get the correct provider,
    then asks that provider to resolve the dependency.
    """

    providers: Dict[str, DependencyProvider] = field(default_factory=dict)

    def get_provider(self, buffer_name: str) -> DependencyProvider:
        if buffer_name not in self.providers:
            raise KeyError(f"No lifecycle policy defined for buffer: '{buffer_name}'")
        return self.providers[buffer_name]


# === (NEW) Abstraction: The ProblemTypeStrategy Contract ===


class ProblemTypeStrategy(abc.ABC):
    """
    An abstract contract for a problem type (e.g., CCE, BCE).

    This object acts as a "factory" for the specific kernel signatures required
    by a given loss function, allowing recipes to be written polymorphically
    without needing to know the details of CCE or BCE.
    """

    @abc.abstractmethod
    def get_loss_signature(self, **kwargs) -> KernelSignature:
        """Returns the appropriate signature for Node 6 or 7."""
        pass

    @abc.abstractmethod
    def get_module_grad_signature(self, **kwargs) -> KernelSignature:
        """Returns the appropriate signature for Node 8."""
        pass

    @abc.abstractmethod
    def get_hidden_grad_signature(self, **kwargs) -> KernelSignature:
        """Returns the appropriate signature for Node 9."""
        pass

    @abc.abstractmethod
    def get_temp_grad_signature(self, **kwargs) -> KernelSignature:
        """Returns the appropriate signature for Node 10."""
        pass


@dataclass(frozen=True)
class CceStrategy(ProblemTypeStrategy):
    """The concrete strategy for CCE (single-label classification)."""

    targets_cce_ref: BufferHandle

    def get_loss_signature(self, **kwargs) -> "ComputeProbsLossCceChunkSignature":
        return ComputeProbsLossCceChunkSignature(target_ref=self.targets_cce_ref, **kwargs)

    def get_module_grad_signature(self, **kwargs) -> "CalculateModuleParamGradsCceSignature":
        return CalculateModuleParamGradsCceSignature(targets_cce_ref=self.targets_cce_ref, **kwargs)

    def get_hidden_grad_signature(self, **kwargs) -> "BackpropErrorToHiddenChunkCceSignature":
        return BackpropErrorToHiddenChunkCceSignature(targets_cce_ref=self.targets_cce_ref, **kwargs)

    def get_temp_grad_signature(self, **kwargs) -> "CalculateChunkTempGradientsCceSignature":
        return CalculateChunkTempGradientsCceSignature(targets_cce_ref=self.targets_cce_ref, **kwargs)


@dataclass(frozen=True)
class BceStrategy(ProblemTypeStrategy):
    """The concrete strategy for BCE (multi-label classification)."""

    targets_bce_ref: BufferHandle

    def get_loss_signature(self, **kwargs) -> "ComputeProbsLossBceChunkSignature":
        return ComputeProbsLossBceChunkSignature(target_ref=self.targets_bce_ref, **kwargs)

    def get_module_grad_signature(self, **kwargs) -> "CalculateModuleParamGradsBceSignature":
        return CalculateModuleParamGradsBceSignature(targets_bce_ref=self.targets_bce_ref, **kwargs)

    def get_hidden_grad_signature(self, **kwargs) -> "BackpropErrorToHiddenChunkBceSignature":
        return BackpropErrorToHiddenChunkBceSignature(targets_bce_ref=self.targets_bce_ref, **kwargs)

    def get_temp_grad_signature(self, **kwargs) -> "CalculateChunkTempGradientsBceSignature":
        return CalculateChunkTempGradientsBceSignature(targets_bce_ref=self.targets_bce_ref, **kwargs)


# === Abstraction Level 3: The Complete Batch Manifest ===


@dataclass(frozen=True)
class ExecutionPlan:
    """
    (REV 2) The single, immutable manifest describing the complete strategy for
    executing one training batch.
    """

    grid: TilingScheme
    reduction_plan: ReductionPlan
    lifecycle_policy: DataLifecyclePolicy
    effective_batch_size: int
    # The `problem_type` field is now a polymorphic strategy object.
    problem_type: "ProblemTypeStrategy"
    clipping_strategy: str  # e.g., 'GLOBAL' or 'PER_ITEM'
    hyperparams: "TrainingHyperparams"  # Forward ref for type hint
    shared_backprop_stream_chunks: int
