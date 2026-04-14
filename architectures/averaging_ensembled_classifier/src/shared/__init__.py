# src/shared/__init__.py
"""Backend-neutral shared-layer types and abstractions."""

# Plan node types
from .plan_types import (
    KernelDispatchNode,
    ReductionTreeNode,
    StreamingLoopNode,
    BarrierNode,
    RetrievalNode,
    PlanNode,
    ExecutionPlan,
    PlanValidationError,
)

# Buffer lifecycle
from .buffer_lifecycle import BufferHandle, BufferRole, BufferDescriptor

# Reduction tree
from .reduction_tree_plan import ReductionTreePlan

# Streaming loop
from .streaming_loop_plan import (
    StreamingLoopPlan,
    IterationDimension,
    ParameterStride,
    ScratchBufferSpec,
)

# Retrieval protocol
from .retrieval_future import RetrievalFuture

# Kernel contracts
from .kernel_contracts import KernelContract

# Plan renderer protocol
from .plan_renderer import PlanRenderer

# Configuration types
from .hardware_profile import HardwareProfile
from .precision_config import PrecisionConfig, MaskStrategy
from .model_spec import ModelSpec
from .parameter_space import ParameterSpace
from .stabilization_policy import StabilizationPolicy
from .optimizer_config import OptimizerConfig
from .problem_type_spec import ProblemTypeSpec, PlanCceStrategy, PlanBceStrategy

# Plan builder
from .plan_builder import build_act_plan, build_learn_plan

# ---------------------------------------------------------------------------
# User-facing API (ADR-018)
# ---------------------------------------------------------------------------
from .ticket import WorkTicket, LearnHandle, TicketState, InvalidTicketStateError
from .engine import Engine

__all__ = [
    # Plan node types
    "KernelDispatchNode",
    "ReductionTreeNode",
    "StreamingLoopNode",
    "BarrierNode",
    "RetrievalNode",
    "PlanNode",
    "ExecutionPlan",
    "PlanValidationError",
    # Buffer lifecycle
    "BufferHandle",
    "BufferRole",
    "BufferDescriptor",
    # Reduction tree
    "ReductionTreePlan",
    # Streaming loop
    "StreamingLoopPlan",
    "IterationDimension",
    "ParameterStride",
    "ScratchBufferSpec",
    # Retrieval protocol
    "RetrievalFuture",
    # Kernel contracts
    "KernelContract",
    # Plan renderer protocol
    "PlanRenderer",
    # Configuration types
    "HardwareProfile",
    "PrecisionConfig",
    "MaskStrategy",
    "ModelSpec",
    "ParameterSpace",
    "StabilizationPolicy",
    "OptimizerConfig",
    "PlanProblemTypeStrategy",
    "PlanCceStrategy",
    "PlanBceStrategy",
    # Plan builder
    "build_act_plan",
    "build_learn_plan",
    # User-facing API (ADR-018)
    "WorkTicket",
    "LearnHandle",
    "TicketState",
    "InvalidTicketStateError",
    "Engine",
]
