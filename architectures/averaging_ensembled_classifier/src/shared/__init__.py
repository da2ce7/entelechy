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
from .precision_config import PrecisionConfig

# Plan builder
from .plan_builder import build_act_plan, build_learn_plan
