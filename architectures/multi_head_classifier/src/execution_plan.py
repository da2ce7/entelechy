# execution_plan.py

"""
The Definitive Implementation of the Strategic Execution Plan Abstraction.

This module provides the data structures to create a complete, declarative
manifest for a single training batch run. This formally decouples strategic
planning (deciding *how* to run the batch) from tactical execution (actually
running it), which is a cornerstone of the system's architectural elegance.

This is the canonical home for the "Dynamic Adaptation" logic (e.g., Cache vs.
Recompute). This dynamic behavior is encapsulated by the DependencyProvider
pattern, allowing the main orchestrator to remain clean, declarative, and
unaware of the underlying fulfillment strategy for its data dependencies.

The primary export of this module is the `ExecutionPlan` class, which serves
as the single, immutable "program" for a `BatchProcessor` to execute.
"""

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable

# --- Architectural Imports ---
# These are imported for type hinting and to show the dependencies.
# In a real project, these would resolve to the actual class definitions.
try:
    import pyopencl as cl
    from .launcher_infra import BufferHandle, KernelSignature, KernelExecutor
    from .compute_patterns import ReductionPlan
    from .workload_primitives import TilingScheme
except ImportError:
    # Create mock types for standalone review and demonstration
    cl = type("cl", (), {"Event": type("Event", (), {})})
    BufferHandle = type("BufferHandle", (), {"id": int})
    KernelSignature = type("KernelSignature", (), {})
    KernelExecutor = type("KernelExecutor", (), {})
    TilingScheme = type("TilingScheme", (), {})
    ReductionPlan = type("ReductionPlan", (), {})


# === Abstraction Level 1: The DependencyProvider Contract ===
# This is the core pattern that enables dynamic adaptation.


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


# === Abstraction Level 2: The Policy Registry ===


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
        """Retrieves the provider strategy for a given conceptual buffer."""
        if buffer_name not in self.providers:
            raise KeyError(f"No lifecycle policy defined for buffer: '{buffer_name}'")
        return self.providers[buffer_name]


# === Abstraction Level 3: The Complete Batch Manifest ===


@dataclass(frozen=True)
class ExecutionPlan:
    """
    The single, immutable manifest describing the complete strategy for
    executing one training batch.

    This object is created by the `TrainingOrchestrator` based on its high-level
    assessment of the problem and system resources. It is then passed to a
    `BatchProcessor`, which executes it without question.
    """

    grid: TilingScheme
    reduction_plan: ReductionPlan
    lifecycle_policy: DataLifecyclePolicy
    effective_batch_size: int
    problem_type: str  # e.g., 'CCE' or 'BCE'
    clipping_strategy: str  # e.g., 'GLOBAL' or 'PER_ITEM'
    hyperparams: "TrainingHyperparams"  # Forward ref for type hint
    shared_backprop_stream_chunks: int
