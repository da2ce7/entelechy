# src/shared/plan_types.py
"""Plan node types and ExecutionPlan container (ADR-002)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Union

from .buffer_lifecycle import BufferDescriptor, BufferHandle
from .hardware_profile import HardwareProfile
from .kernel_contracts import KernelContract
from .precision_config import PrecisionConfig
from .reduction_tree_plan import ReductionTreePlan
from .streaming_loop_plan import StreamingLoopPlan


@dataclass(frozen=True)
class KernelDispatchNode:
    """A single logical kernel invocation (ADR-002)."""
    node_id: str
    depends_on: frozenset[str]
    kernel_name: str
    contract: KernelContract
    buffer_bindings: dict[str, BufferHandle]
    scalar_params: dict[str, int | float]
    tile_count: int
    local_work_size: int | None
    placement_strategy: str | None


@dataclass(frozen=True)
class ReductionTreeNode:
    """A multi-stage log_K(N) reduction tree (ADR-002, ADR-003)."""
    node_id: str
    depends_on: frozenset[str]
    reduction_plan: ReductionTreePlan


@dataclass(frozen=True)
class StreamingLoopNode:
    """A parametric loop over a chunk-indexed body (ADR-002, ADR-004)."""
    node_id: str
    depends_on: frozenset[str]
    streaming_plan: StreamingLoopPlan


@dataclass(frozen=True)
class BarrierNode:
    """A named synchronization point (ADR-002)."""
    node_id: str
    depends_on: frozenset[str]
    barrier_name: str


@dataclass(frozen=True)
class RetrievalNode:
    """A host-accessible result extraction point (ADR-002, ADR-010)."""
    node_id: str
    depends_on: frozenset[str]
    source_buffer: BufferHandle
    logical_shape: tuple[int, ...]
    event_name: str


PlanNode = Union[
    KernelDispatchNode,
    ReductionTreeNode,
    StreamingLoopNode,
    BarrierNode,
    RetrievalNode,
]


class PlanValidationError(Exception):
    """Raised when an ExecutionPlan fails structural validation."""


def _validate_plan(
    nodes: dict[str, PlanNode],
    buffers: dict[BufferHandle, BufferDescriptor],
    topological_order: tuple[str, ...],
) -> None:
    """Enforce ADR-002 structural invariants."""
    node_ids = set(nodes.keys())

    # 1. Unique node IDs (guaranteed by dict, but verify topo order matches)
    if set(topological_order) != node_ids:
        raise PlanValidationError(
            "topological_order does not match the set of node IDs in nodes."
        )

    # 2. Every depends_on target exists
    for nid, node in nodes.items():
        for dep in node.depends_on:
            if dep not in node_ids:
                raise PlanValidationError(
                    f"Node {nid!r} depends on {dep!r}, which does not exist."
                )

    # 3. DAG acyclicity via Kahn's algorithm
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for nid, node in nodes.items():
        for dep in node.depends_on:
            adjacency[dep].append(nid)
            in_degree[nid] += 1

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for successor in adjacency[current]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if visited != len(node_ids):
        raise PlanValidationError("Dependency graph contains a cycle.")

    # 4. topological_order consistent with edges
    order_index = {nid: i for i, nid in enumerate(topological_order)}
    for nid, node in nodes.items():
        for dep in node.depends_on:
            if order_index[dep] >= order_index[nid]:
                raise PlanValidationError(
                    f"topological_order inconsistency: {dep!r} must precede {nid!r}."
                )

    # 5. Buffer reference validity
    buffer_handles = set(buffers.keys())
    for nid, node in nodes.items():
        if isinstance(node, KernelDispatchNode):
            for param_name, handle in node.buffer_bindings.items():
                if handle not in buffer_handles:
                    raise PlanValidationError(
                        f"Node {nid!r} references BufferHandle {handle} "
                        f"(param {param_name!r}) not in buffers."
                    )
        elif isinstance(node, RetrievalNode):
            if node.source_buffer not in buffer_handles:
                raise PlanValidationError(
                    f"RetrievalNode {nid!r} references BufferHandle "
                    f"{node.source_buffer} not in buffers."
                )
        elif isinstance(node, ReductionTreeNode):
            rp = node.reduction_plan
            if rp.source_buffer not in buffer_handles:
                raise PlanValidationError(
                    f"ReductionTreeNode {nid!r} source_buffer "
                    f"{rp.source_buffer} not in buffers."
                )
            if rp.destination_buffer not in buffer_handles:
                raise PlanValidationError(
                    f"ReductionTreeNode {nid!r} destination_buffer "
                    f"{rp.destination_buffer} not in buffers."
                )

    # 6. StreamingLoopNode body must reference KernelDispatchNode instances
    for nid, node in nodes.items():
        if isinstance(node, StreamingLoopNode):
            for body_id in node.streaming_plan.body:
                if body_id not in nodes:
                    raise PlanValidationError(
                        f"StreamingLoopNode {nid!r} body references "
                        f"{body_id!r} which does not exist."
                    )
                if not isinstance(nodes[body_id], KernelDispatchNode):
                    raise PlanValidationError(
                        f"StreamingLoopNode {nid!r} body references "
                        f"{body_id!r} which is not a KernelDispatchNode."
                    )


@dataclass(frozen=True)
class ExecutionPlan:
    """An immutable, backend-neutral execution plan (ADR-002).

    The plan is a DAG of typed nodes connected by explicit dependency edges.
    """
    nodes: dict[str, PlanNode]
    buffers: dict[BufferHandle, BufferDescriptor]
    topological_order: tuple[str, ...]
    precision: PrecisionConfig
    hardware: HardwareProfile

    def __post_init__(self) -> None:
        _validate_plan(self.nodes, self.buffers, self.topological_order)
