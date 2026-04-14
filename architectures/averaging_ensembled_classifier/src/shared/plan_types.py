# src/shared/plan_types.py
"""Backend-neutral execution plan — node types and plan container.

This module defines the closed five-type DAG vocabulary (CONCEPT.md §6,
ADR-002) and the ``ExecutionPlan`` container.  Construction-time
validation guarantees structural integrity: no plan reaches a backend
unless it is acyclic, consistently ordered, and referentially sound.

Design decisions
────────────────
**Two-level node namespace.**  Top-level DAG nodes are the plan's
concurrency and scheduling units.  Streaming-loop body nodes are
*owned* by their parent ``StreamingLoopNode`` and do not appear in
the top-level namespace.  Body nodes carry intra-body dependency
edges (intra-iteration ordering); the loop node carries external
edges (inter-node gating).  This eliminates the impedance mismatch
between an atomic loop node in the DAG and its internal dispatch
sequence.

**Execution order is the committed dispatch sequence.**  The Policy
tier records the specific topological sort it chose —
``ExecutionPlan.execution_order`` — which validation confirms is
consistent with the dependency graph.  Backends may exploit
parallelism within this ordering but must respect its precedence
constraints.

**Precomputed buffers are first-class.**  Host-computed constant data
(e.g. Node 16's threshold schedule) is carried by the plan so the
Orchestration tier knows which buffers need host→device upload of
pre-computed content versus zero-initialisation or kernel production.

**Common node base.**  All five node types inherit ``node_id`` and
``depends_on`` from ``PlanNodeBase``, eliminating field repetition
and enabling ``isinstance(node, PlanNodeBase)`` for generic type
guards.

**Minimal query surface.**  The plan is a data structure, not an
analysis framework.  Methods are limited to projections that every
consumer needs; higher-level analysis belongs in separate modules.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .buffer_lifecycle import BufferDescriptor, BufferHandle
from .hardware_profile import HardwareProfile
from .kernel_contracts import KernelContract
from .precision_config import PrecisionConfig
from .reduction_tree_plan import ReductionTreePlan
from .streaming_loop_plan import StreamingLoopPlan

__all__ = [
    "BarrierNode",
    "ExecutionPlan",
    "KernelDispatchNode",
    "PlanNode",
    "PlanNodeBase",
    "PlanValidationError",
    "ReductionTreeNode",
    "RetrievalNode",
    "StreamingLoopNode",
]


# ═══════════════════════════════════════════════════════════════════
# §1  Node Types — closed taxonomy (CONCEPT.md §6, ADR-002)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PlanNodeBase:
    """Fields common to every node in the execution plan.

    Subclassed by the five concrete node types — never instantiated
    directly.

    Parameters
    ----------
    node_id:
        Unique within the plan's top-level namespace (or within a
        ``StreamingLoopNode``'s body namespace for body nodes).
    depends_on:
        Node IDs that must complete before this node may execute.
        For top-level nodes, targets are other top-level IDs.
        For body nodes, targets are sibling body-node IDs
        (intra-iteration ordering); the parent loop's own
        ``depends_on`` gates iteration start.
    """

    node_id: str
    depends_on: frozenset[str]


@dataclass(frozen=True)
class KernelDispatchNode(PlanNodeBase):
    """A single logical kernel invocation with buffer bindings.

    ``tile_count`` specifies logical parallelism — whether the backend
    renders this as N independent dispatches, one dispatch with N
    work-groups, or N thread-pool tasks is a rendering concern.
    """

    kernel_name: str
    contract: KernelContract
    buffer_bindings: dict[str, BufferHandle]
    scalar_params: dict[str, int | float]
    tile_count: int
    local_work_size: int | None = None
    placement_strategy: str | None = None


@dataclass(frozen=True)
class ReductionTreeNode(PlanNodeBase):
    """A multi-stage log_K(N) reduction tree.

    Rendered atomically by the backend, which selects kernel tiers
    (register-reduce vs. local-reduce) and manages intermediate
    buffers.  The tree's source and destination buffers are carried
    by the ``reduction_plan``.
    """

    reduction_plan: ReductionTreePlan


@dataclass(frozen=True)
class StreamingLoopNode(PlanNodeBase):
    """A parametric loop over a chunk-indexed body of dispatch nodes.

    ``body_nodes`` are *owned* by this loop — they do **not** appear
    in the plan's top-level node namespace.  The loop is atomic from
    the DAG's perspective: upstream dependencies gate when any body
    dispatch may begin; downstream dependents wait until the complete
    iteration series finishes.

    Within each iteration the backend dispatches body nodes
    respecting their intra-body ``depends_on`` edges and applies
    ``streaming_plan.parameter_strides`` to advance per-iteration
    scalar values.

    ``streaming_plan.body`` carries the body node IDs for the
    streaming-plan's own metadata; validation confirms it matches
    ``tuple(bn.node_id for bn in body_nodes)``.
    """

    streaming_plan: StreamingLoopPlan
    body_nodes: tuple[KernelDispatchNode, ...]


@dataclass(frozen=True)
class BarrierNode(PlanNodeBase):
    """A named synchronisation point — pure sequencing construct.

    Carries no dispatch payload.  Joins upstream dependency edges
    so that downstream nodes can express a fan-in dependency with
    a single target ID.
    """

    barrier_name: str


@dataclass(frozen=True)
class RetrievalNode(PlanNodeBase):
    """A host-accessible result extraction point.

    Specifies the source buffer, expected shape, and the named
    event it signals upon completion.
    """

    source_buffer: BufferHandle
    logical_shape: tuple[int, ...]
    event_name: str


PlanNode = (
    KernelDispatchNode
    | ReductionTreeNode
    | StreamingLoopNode
    | BarrierNode
    | RetrievalNode
)
"""Union of the five plan node types — closed taxonomy per CONCEPT.md §6."""


# ═══════════════════════════════════════════════════════════════════
# §2  Validation
# ═══════════════════════════════════════════════════════════════════


class PlanValidationError(Exception):
    """Raised when an ``ExecutionPlan`` fails structural validation."""


def _buffer_handles_of(node: PlanNode) -> frozenset[BufferHandle]:
    """Every ``BufferHandle`` referenced by *node* at any nesting level."""
    handles: set[BufferHandle] = set()

    if isinstance(node, KernelDispatchNode):
        handles.update(node.buffer_bindings.values())

    elif isinstance(node, ReductionTreeNode):
        rp = node.reduction_plan
        handles.add(rp.source_buffer)
        handles.add(rp.destination_buffer)

    elif isinstance(node, StreamingLoopNode):
        for body_node in node.body_nodes:
            handles.update(body_node.buffer_bindings.values())

    elif isinstance(node, RetrievalNode):
        handles.add(node.source_buffer)

    # BarrierNode references no buffers.
    return frozenset(handles)


def _validate(plan: ExecutionPlan) -> None:
    """Enforce all structural invariants on a freshly constructed plan.

    Raises ``PlanValidationError`` on the first violation detected.

    Invariant families
    ──────────────────
    **S — Structural:** ID uniqueness, namespace isolation, cross-
    reference consistency between ``body_nodes`` and
    ``streaming_plan.body``.

    **D — Dependency:** edge targets exist, no cycles, execution
    order respects edges.

    **B — Buffer referential:** every ``BufferHandle`` referenced by
    any node (including loop bodies) and by ``precomputed_buffers``
    exists in the plan's ``buffers`` namespace.
    """
    nodes = plan.nodes
    order = plan.execution_order
    top_ids = frozenset(nodes)
    known_handles = frozenset(plan.buffers)

    # ── S1: No duplicates in execution_order ──────────────────────

    if len(order) != len(set(order)):
        seen: set[str] = set()
        dupes: list[str] = []
        for nid in order:
            if nid in seen:
                dupes.append(nid)
            seen.add(nid)
        raise PlanValidationError(
            f"Duplicate entries in execution_order: {dupes}"
        )

    # ── S2: execution_order matches top-level node IDs exactly ────

    order_set = frozenset(order)
    if order_set != top_ids:
        missing = top_ids - order_set
        extra = order_set - top_ids
        raise PlanValidationError(
            f"execution_order / nodes mismatch — "
            f"missing from order: {missing or '∅'}, "
            f"extra in order: {extra or '∅'}"
        )

    # ── S3: Body node IDs don't collide with top-level or peers ───

    all_body_ids: set[str] = set()
    for nid, node in nodes.items():
        if not isinstance(node, StreamingLoopNode):
            continue
        local_ids: set[str] = set()
        for bn in node.body_nodes:
            if bn.node_id in top_ids:
                raise PlanValidationError(
                    f"Body node {bn.node_id!r} in loop {nid!r} "
                    f"collides with a top-level node ID."
                )
            if bn.node_id in local_ids:
                raise PlanValidationError(
                    f"Duplicate body node ID {bn.node_id!r} in "
                    f"loop {nid!r}."
                )
            local_ids.add(bn.node_id)
        if overlap := all_body_ids & local_ids:
            raise PlanValidationError(
                f"Body node IDs {overlap} appear in multiple loops."
            )
        all_body_ids |= local_ids

    # ── S4: streaming_plan.body matches body_nodes ────────────────

    for nid, node in nodes.items():
        if not isinstance(node, StreamingLoopNode):
            continue
        declared = node.streaming_plan.body
        actual = tuple(bn.node_id for bn in node.body_nodes)
        if declared != actual:
            raise PlanValidationError(
                f"StreamingLoopNode {nid!r}: streaming_plan.body "
                f"{declared} ≠ body_nodes IDs {actual}."
            )

    # ── D1: Top-level depends_on targets are top-level IDs ────────

    for nid, node in nodes.items():
        bad = node.depends_on - top_ids
        if bad:
            raise PlanValidationError(
                f"Node {nid!r} depends on {bad}, which are not "
                f"top-level node IDs."
            )

    # ── D2: Body depends_on targets are sibling body IDs ──────────

    for nid, node in nodes.items():
        if not isinstance(node, StreamingLoopNode):
            continue
        sibling_ids = frozenset(bn.node_id for bn in node.body_nodes)
        for bn in node.body_nodes:
            bad = bn.depends_on - sibling_ids
            if bad:
                raise PlanValidationError(
                    f"Body node {bn.node_id!r} in loop {nid!r} "
                    f"depends on {bad}, which are not sibling body "
                    f"nodes.  External dependencies belong on the "
                    f"loop node itself."
                )

    # ── D3: Top-level DAG is acyclic (Kahn's algorithm) ───────────

    in_deg: dict[str, int] = {nid: 0 for nid in top_ids}
    successors: dict[str, list[str]] = {nid: [] for nid in top_ids}
    for nid, node in nodes.items():
        for dep in node.depends_on:
            successors[dep].append(nid)
            in_deg[nid] += 1

    frontier: deque[str] = deque(
        nid for nid, d in in_deg.items() if d == 0
    )
    visited = 0
    while frontier:
        visited += 1
        for succ in successors[frontier.popleft()]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                frontier.append(succ)

    if visited != len(top_ids):
        raise PlanValidationError(
            "Top-level dependency graph contains a cycle."
        )

    # ── D4: execution_order respects dependency edges ─────────────

    pos = {nid: i for i, nid in enumerate(order)}
    for nid, node in nodes.items():
        for dep in node.depends_on:
            if pos[dep] >= pos[nid]:
                raise PlanValidationError(
                    f"execution_order violation: {dep!r} (position "
                    f"{pos[dep]}) must precede {nid!r} (position "
                    f"{pos[nid]})."
                )

    # ── B1: All buffer handles referenced by nodes exist ──────────

    for nid, node in nodes.items():
        missing = _buffer_handles_of(node) - known_handles
        if missing:
            raise PlanValidationError(
                f"Node {nid!r} references BufferHandles {missing} "
                f"not in the plan's buffer namespace."
            )

    # ── B2: precomputed_buffers handles exist in buffers ──────────

    missing_pre = frozenset(plan.precomputed_buffers) - known_handles
    if missing_pre:
        raise PlanValidationError(
            f"precomputed_buffers references BufferHandles "
            f"{missing_pre} not in the plan's buffer namespace."
        )


# ═══════════════════════════════════════════════════════════════════
# §3  ExecutionPlan
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ExecutionPlan:
    """An immutable, backend-neutral execution plan.

    The plan is a directed acyclic graph of typed, immutable node
    descriptors connected by explicit dependency edges.  Construction-
    time validation (``_validate``) guarantees the plan is structurally
    sound before any backend sees it.

    Top-level nodes — those in ``nodes`` — are the plan's concurrency
    and scheduling units.  Streaming-loop body nodes are owned by
    their parent ``StreamingLoopNode`` and occupy a separate namespace.

    Parameters
    ----------
    nodes:
        Top-level node ID → node mapping.
    buffers:
        Buffer handle → descriptor for every device-side buffer in
        the plan.  Lifetimes are plan-prescribed; the Orchestration
        tier manages allocation and deallocation accordingly.
    execution_order:
        The Policy tier's committed top-level dispatch sequence —
        a specific topological sort of the dependency graph.  Backends
        may exploit parallelism within this ordering but must respect
        its precedence constraints.
    precision:
        Active precision configuration for type resolution.
    hardware:
        Hardware profile for dispatch geometry decisions.
    precomputed_buffers:
        Host-computed constant data uploaded before dispatch (e.g.
        Node 16's threshold schedule).  Every handle key must exist
        in ``buffers``.  Empty when no precomputation is needed.
    """

    nodes: dict[str, PlanNode]
    buffers: dict[BufferHandle, BufferDescriptor]
    execution_order: tuple[str, ...]
    precision: PrecisionConfig
    hardware: HardwareProfile
    precomputed_buffers: dict[BufferHandle, np.ndarray] = field(
        default_factory=dict,
    )

    def __post_init__(self) -> None:
        _validate(self)

    # ── Lookup ────────────────────────────────────────────────────

    def __getitem__(self, node_id: str) -> PlanNode:
        """Look up a top-level node by ID.

        Raises ``KeyError`` for unknown IDs — the same contract as
        a dict, because that's what callers expect.
        """
        return self.nodes[node_id]

    def __contains__(self, node_id: str) -> bool:
        """Whether *node_id* is a top-level node."""
        return node_id in self.nodes

    def __len__(self) -> int:
        """Number of top-level nodes."""
        return len(self.nodes)

    # ── Iteration ─────────────────────────────────────────────────

    def in_execution_order(self) -> Iterator[tuple[str, PlanNode]]:
        """Yield ``(node_id, node)`` pairs in the committed order."""
        for nid in self.execution_order:
            yield nid, self.nodes[nid]

    def kernel_dispatches(self) -> Iterator[KernelDispatchNode]:
        """Every ``KernelDispatchNode``, including loop body nodes.

        Yields top-level dispatch nodes in execution order.  When a
        ``StreamingLoopNode`` is encountered, its body nodes are
        expanded inline at the loop's position.

        ``ReductionTreeNode`` internal dispatches are not included —
        the backend renders those from the ``ReductionTreePlan``.
        """
        for nid in self.execution_order:
            node = self.nodes[nid]
            if isinstance(node, KernelDispatchNode):
                yield node
            elif isinstance(node, StreamingLoopNode):
                yield from node.body_nodes
