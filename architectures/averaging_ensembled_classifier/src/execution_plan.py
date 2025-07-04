# execution_plan.py

"""
A Module Defining the Abstract Vocabulary of Strategic Intent.

Jurisdictional Mandate:
This module is the definitive source for the abstract data structures that
constitute an "Execution Plan." It does not contain logic for execution;
rather, it provides the formal, immutable contracts that describe a strategy
for execution. Its jurisdiction is to define the "what," leaving the "how" to
other, subordinate layers of the system.

Architectural Role:
This module provides the formal bridge between the high-level "Strategist"
(the `TrainingOrchestrator`) and the tactical "Conductor" (the
`BatchProcessor`). The classes herein are the system's lingua franca—a
declarative, verifiable language for expressing a complete computational plan
for a single training batch. It is through the instantiation of these humble
primitives that the principles of dynamic adaptation and polymorphic
correctness are made manifest.
"""

import abc
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pyopencl as cl

# --- Local Infrastructure & Primitive Imports ---
# These are the foundational components upon which our strategic contracts are built.
from .launcher_infra import BufferHandle, KernelSignature, KernelExecutor, Services
from .compute_patterns import ReductionPlan
from .workload_primitives import TilingScheme, WorkTile, TiledGather
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

# --- Type-Checking Guard for Circular Dependencies ---
# A humble acknowledgment of a necessary complexity. For the static type
# checker to understand our complete contracts, it must see these types.
# To prevent a circular import at runtime, we place them behind this guard.
if TYPE_CHECKING:
    from .stabilization_policy import StabilizationPolicy
    from .main_orchestrator import TrainingHyperparams


# =========================================================================
# === Abstraction 1: The DependencyProvider Contract                      ===
# =========================================================================


class DependencyProvider(abc.ABC):
    """
    Function: An abstract contract for the fulfillment of a data dependency.

    Architectural Mandate:
    This class embodies a profound architectural principle: the decoupling of
    the *need* for a resource from the *method of its acquisition*. An object
    adhering to this contract makes a simple promise: to provide a buffer and a
    signal of its readiness. It deliberately hides the complexity of whether
    that data is retrieved from a cache or recomputed on demand, thereby
    enabling the system's core adaptive memory strategies.
    """

    @abc.abstractmethod
    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        """Fulfills the promise to provide the dependency."""
        pass


@dataclass(frozen=True)
class CacheProvider(DependencyProvider):
    """The 'do nothing' strategy: provides a dependency that is already resident in device memory."""

    handle: BufferHandle
    ready_event: cl.Event

    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        # WHY: As the dependency is already computed, we simply return the
        # existing handle and event, ignoring all other inputs. This is the
        # lowest-overhead path, chosen when memory permits.
        return self.handle, self.ready_event


@dataclass(frozen=True)
class RecomputeProvider(DependencyProvider):
    """The 'on-demand' strategy: provides a dependency by launching a single kernel."""

    signature: KernelSignature
    output_handle: BufferHandle

    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        # WHY: This provider encapsulates a single computational step, trading
        # VRAM for compute time. It is the tactical fulfillment of the system's
        # choice to prioritize survival over speed under memory pressure.
        event = ex.launch(queue, self.signature, wait_for=wait_for)
        return self.output_handle, event


@dataclass(frozen=True)
class StagedComputationProvider(DependencyProvider):
    """A generic, powerful provider for complex, multi-kernel dependency chains."""

    computation_fn: partial

    def resolve(
        self, queue: cl.CommandQueue, ex: KernelExecutor, wait_for: List[cl.Event]
    ) -> Tuple[BufferHandle, cl.Event]:
        # Delegates the entire complex resolution to the injected partial function.
        return self.computation_fn(queue, ex, wait_for)


@dataclass(frozen=True)
class DataLifecyclePolicy:
    """A registry mapping a conceptual buffer to its concrete fulfillment strategy."""
    # WHY: This object acts as a "switchboard" for the Conductor. It translates
    # a logical request ("I need `summed_grad_hidden_activations`") into a
    # concrete `DependencyProvider`, which orchestrates the necessary steps.
    providers: Dict[str, DependencyProvider] = field(default_factory=dict)

    def get_provider(self, buffer_name: str) -> DependencyProvider:
        if buffer_name not in self.providers:
            raise KeyError(f"No lifecycle policy defined for buffer: '{buffer_name}'")
        return self.providers[buffer_name]


# =========================================================================
# === Abstraction 2: The ProblemTypeStrategy Contract                 ===
# =========================================================================


class ProblemTypeStrategy(abc.ABC):
    """
    Function: An abstract contract for a problem type (e.g., CCE, BCE).

    Architectural Mandate:
    This is the embodiment of the classical Strategy Pattern. It replaces messy,
    procedural `if/elif/else` blocks in the core execution logic with a clean,
    polymorphic interface. By doing so, it upholds the Open/Closed Principle:
    the system is open to extension (one can add a new loss function by creating
    a new strategy class) but closed for modification (the `BatchProcessor`
    need never be changed).
    """

    @property
    @abc.abstractmethod
    def required_targets_buffer_name(self) -> str:
        """The canonical name of the buffer this strategy requires for ground truth."""
        pass

    @abc.abstractmethod
    def get_loss_signature(self, **kwargs) -> KernelSignature:
        """A factory for the appropriate loss computation signature (Node 6/7)."""
        pass

    @abc.abstractmethod
    def get_module_grad_signature(self, **kwargs) -> KernelSignature:
        """A factory for the module parameter gradient signature (Node 8)."""
        pass

    @abc.abstractmethod
    def get_hidden_grad_signature(self, **kwargs) -> KernelSignature:
        """A factory for the upstream hidden gradient signature (Node 9)."""
        pass

    @abc.abstractmethod
    def get_temp_grad_signature(self, **kwargs) -> KernelSignature:
        """A factory for the temperature gradient signature (Node 10)."""
        pass


@dataclass(frozen=True)
class CceStrategy(ProblemTypeStrategy):
    """The type-safe, concrete strategy for CCE (single-label classification)."""

    targets_cce_ref: BufferHandle  # Contractually binds this strategy to the integer-typed targets buffer.

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_cce"

    def get_loss_signature(self, **kwargs) -> "ComputeProbsLossCceChunkSignature":
        return ComputeProbsLossCceChunkSignature(**kwargs, target_ref=self.targets_cce_ref)

    def get_module_grad_signature(self, **kwargs) -> "CalculateModuleParamGradsCceSignature":
        return CalculateModuleParamGradsCceSignature(**kwargs, targets_cce_ref=self.targets_cce_ref)

    def get_hidden_grad_signature(self, **kwargs) -> "BackpropErrorToHiddenChunkCceSignature":
        return BackpropErrorToHiddenChunkCceSignature(**kwargs, targets_cce_ref=self.targets_cce_ref)

    def get_temp_grad_signature(self, **kwargs) -> "CalculateChunkTempGradientsCceSignature":
        return CalculateChunkTempGradientsCceSignature(**kwargs, targets_cce_ref=self.targets_cce_ref)


@dataclass(frozen=True)
class BceStrategy(ProblemTypeStrategy):
    """The type-safe, concrete strategy for BCE (multi-label classification)."""

    targets_bce_ref: BufferHandle  # Contractually binds this strategy to the float-typed targets buffer.

    @property
    def required_targets_buffer_name(self) -> str:
        return "targets_bce"

    def get_loss_signature(self, **kwargs) -> "ComputeProbsLossBceChunkSignature":
        return ComputeProbsLossBceChunkSignature(**kwargs, target_ref=self.targets_bce_ref)

    def get_module_grad_signature(self, **kwargs) -> "CalculateModuleParamGradsBceSignature":
        return CalculateModuleParamGradsBceSignature(**kwargs, targets_bce_ref=self.targets_bce_ref)

    def get_hidden_grad_signature(self, **kwargs) -> "BackpropErrorToHiddenChunkBceSignature":
        return BackpropErrorToHiddenChunkBceSignature(**kwargs, targets_bce_ref=self.targets_bce_ref)

    def get_temp_grad_signature(self, **kwargs) -> "CalculateChunkTempGradientsBceSignature":
        return CalculateChunkTempGradientsBceSignature(**kwargs, targets_bce_ref=self.targets_bce_ref)


# =========================================================================
# === Abstraction 3: The Complete Batch Manifest                        ===
# =========================================================================


@dataclass(frozen=True)
class ExecutionPlan:
    """
    Function: The single, immutable manifest describing the complete
    strategy for executing one training batch.

    Architectural Role:
    This is the final artifact of the Strategist's work. It is a sacred,
    declarative contract given to the Conductor. It contains no active logic,
    only a complete and verifiable description of the work to be done,
    encompassing the spatial partitioning of the problem (`grid`), the memory
    and temporal trade-offs (`lifecycle_policy`, `adaptation_strategy`), and
    the core algorithmic choices (`problem_type`, `stabilization_policy`).
    """

    # The spatial partitioning of the problem space.
    grid: TilingScheme
    # The hardware-aware fan-in plan for reduction kernels.
    reduction_plan: ReductionPlan
    # The policy for fulfilling data dependencies (cache vs. recompute).
    lifecycle_policy: DataLifecyclePolicy
    # The actual number of valid samples in the batch.
    effective_batch_size: int
    # The polymorphic strategy object for the loss function and gradients.
    problem_type: "ProblemTypeStrategy"
    # The strategy for applying the initial, leaf-level clipping.
    clipping_strategy: str
    # The policy object for staged reduction stabilization.
    stabilization_policy: "StabilizationPolicy"
    # A manifest of all training hyperparameters.
    hyperparams: "TrainingHyperparams"
    # The top-level memory adaptation strategy (e.g., 'CACHE' or 'RECOMPUTE_GRAD_H').
    adaptation_strategy: str
    # The number of chunks for the streaming shared-layer backpropagation.
    shared_backprop_stream_chunks: int
