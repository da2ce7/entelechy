# src/shared/plan_types.py
"""Backend-neutral execution plan — node types and plan container.

This module defines the closed five-type DAG vocabulary (CONCEPT.md §6,
ADR-002) and the ``ExecutionPlan`` container.  Construction-time
validation guarantees structural integrity: no plan reaches a backend
unless it is acyclic, consistently ordered, and referentially sound.

Dispatch parameterisation
─────────────────────────
``DispatchGrid`` models tile-parallel dispatch as a Cartesian product
over named axes, each with independent per-dispatch parameter strides.
This is the tile-dispatch counterpart of ``StreamingLoopPlan``: both
specify how scalar parameters vary across invocations.  The semantic
difference is concurrency:

* **Grid dispatches** are independent and may execute in parallel.
* **Streaming iterations** are sequential (body nodes may overwrite
  scratch buffers between iterations).

A ``ScalarStride`` declares a scalar parameter that advances linearly::

    value(dispatch_i) = scalar_params[param_name] + i × stride

where ``scalar_params`` carries the base (iteration-0) value.  This
convention is shared with ``ParameterStride`` (streaming_loop_plan.py),
eliminating the invariant-maintenance burden of a redundant ``base``
field.

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

**N-dimensional dispatch grid.**  ``DispatchGrid`` expresses 1-D
(tile index), 2-D (tile × batch chunk), or higher-dimensional
decomposition as a parameterisation — adding an axis — rather than
a structural rework.  ``SINGLE_DISPATCH`` is the zero-axis sentinel
for kernels with no tile parallelism.
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
    "DispatchAxis",
    "DispatchGrid",
    "ExecutionPlan",
    "KernelDispatchNode",
    "PlanNode",
    "PlanNodeBase",
    "PlanValidationError",
    "ReductionTreeNode",
    "RetrievalNode",
    "SINGLE_DISPATCH",
    "ScalarStride",
    "StreamingLoopNode",
]


# ═══════════════════════════════════════════════════════════════════
# §1  Dispatch Grid — typed per-tile parameter variation
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScalarStride:
    """A scalar parameter that advances linearly across dispatches.

    For dispatch *i* along the axis that owns this stride::

        value(i) = scalar_params[param_name] + i × stride

    where ``scalar_params`` is the hosting ``KernelDispatchNode``'s
    constant-parameter dict, carrying the base (iteration-0) value.

    This type is the dispatch-grid counterpart of ``ParameterStride``
    (streaming_loop_plan.py).  Both share the same ``param_name`` +
    ``stride`` semantics and the "base lives in ``scalar_params``"
    convention.

    Parameters
    ----------
    param_name:
        Full canonical scalar parameter name (CONTRACT §2.3).
        Must exist in the hosting node's ``scalar_params``.
    stride:
        Per-dispatch increment along the owning axis.
    """

    param_name: str
    stride: int | float


@dataclass(frozen=True)
class DispatchAxis:
    """One independent dimension of the dispatch parameter space.

    For a given axis whose ordinal within the grid is *k*, the
    per-axis dispatch index ``i_k`` ranges over ``[0, extent)``.
    Each stride in ``strides`` produces::

        scalar_params[s.param_name] + i_k × s.stride

    Axes iterate in row-major order within a ``DispatchGrid``:
    the **last** axis is the fastest-varying dimension.  For a
    2-axis grid ``(A, B)``, flat dispatch index *d* decomposes as::

        i_A = d // B.extent
        i_B = d %  B.extent

    Parameters
    ----------
    name:
        Human-readable axis identifier for debugging, logging,
        and plan-inspection tooling (e.g. ``"tile"``,
        ``"batch_chunk"``).
    extent:
        Number of positions along this axis (≥ 1).
    strides:
        Scalar parameters that vary along this axis.  A parameter
        may appear in at most one axis — cross-axis duplication is
        detected by ``DispatchGrid.__post_init__``.
    """

    name: str
    extent: int
    strides: tuple[ScalarStride, ...]

    def __post_init__(self) -> None:
        if self.extent < 1:
            raise ValueError(
                f"DispatchAxis {self.name!r}: extent must be ≥ 1, "
                f"got {self.extent}"
            )


@dataclass(frozen=True)
class DispatchGrid:
    """N-dimensional grid of independent, parallel dispatches.

    Total dispatches = product of all axis extents.  An empty
    ``axes`` tuple (the ``SINGLE_DISPATCH`` sentinel) signals a
    single dispatch with no parameter variation.

    All dispatches described by a grid are **independent** — the
    backend is free to execute them concurrently, sequentially,
    or via native dispatch-grid features (e.g.
    ``vkCmdDispatch(ext_0, ext_1, 1)``).  This is the defining
    semantic difference from ``StreamingLoopPlan``, whose
    iterations are sequential.

    Mapping to GPU dispatch models
    ──────────────────────────────
    * **OpenCL:** N imperative ``clEnqueueNDRange`` calls with
      per-dispatch scalar overrides.
    * **Vulkan:** A 1-D axis maps to ``vkCmdDispatch(extent, 1, 1)``;
      the tile index is derivable from ``gl_WorkGroupID.x``.  2-D
      maps to ``vkCmdDispatch(ext_0, ext_1, 1)``.
    * **CPU:** N thread-pool tasks, one per grid point.

    Parameters
    ----------
    axes:
        Ordered tuple of independent dispatch dimensions.  Row-major:
        the last axis is the fastest-varying.  Empty for single-
        dispatch nodes.
    """

    axes: tuple[DispatchAxis, ...]

    def __post_init__(self) -> None:
        # No scalar parameter may be strided by two different axes.
        seen: dict[str, str] = {}  # param_name → axis_name
        for ax in self.axes:
            for s in ax.strides:
                if s.param_name in seen:
                    raise ValueError(
                        f"Parameter {s.param_name!r} is strided by "
                        f"both axis {seen[s.param_name]!r} and "
                        f"{ax.name!r}"
                    )
                seen[s.param_name] = ax.name

    # ── Derived properties ────────────────────────────────────────

    @property
    def total_dispatches(self) -> int:
        """Product of all axis extents.  1 when ``axes`` is empty."""
        result = 1
        for a in self.axes:
            result *= a.extent
        return result

    @property
    def is_single_dispatch(self) -> bool:
        """Whether this grid describes exactly one dispatch."""
        return not self.axes or self.total_dispatches == 1

    def all_strides(self) -> tuple[ScalarStride, ...]:
        """Flat collection of every ``ScalarStride`` across all axes."""
        return tuple(s for a in self.axes for s in a.strides)


SINGLE_DISPATCH: DispatchGrid = DispatchGrid(axes=())
"""Zero-axis sentinel — a single dispatch with no parameter variation.

Default value for ``KernelDispatchNode.grid`` when no tiling is needed.
"""


# ═══════════════════════════════════════════════════════════════════
# §2  Node Types — closed taxonomy (CONCEPT.md §6, ADR-002)
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

    A ``grid`` with multiple axes or extent > 1 specifies
    tile-parallel dispatch — independent invocations whose scalar
    parameters vary according to the grid's ``ScalarStride``
    declarations.  Whether the backend renders these as N
    ``clEnqueueNDRange`` calls, one ``vkCmdDispatch(N,1,1)``, or
    N thread-pool tasks is a rendering concern.

    Parameters
    ----------
    kernel_name:
        Kernel function name (key into ``KERNEL_REGISTRY``).
    contract:
        The ``KernelContract`` governing this kernel's interface.
    buffer_bindings:
        Mapping from contract buffer parameter names to plan-level
        ``BufferHandle`` instances.
    scalar_params:
        Mapping from contract scalar parameter names to base
        (iteration-0 / dispatch-0) values.  Tile-varying parameters
        are listed in ``grid`` strides; streaming-varying parameters
        are listed in the parent ``StreamingLoopPlan``'s strides.
    grid:
        Dispatch decomposition.  ``SINGLE_DISPATCH`` (the default)
        means one invocation with no parameter variation.
    local_work_size:
        Work-group size hint for GPU backends.  ``None`` defers to
        the backend's default.
    """

    kernel_name: str
    contract: KernelContract
    buffer_bindings: dict[str, BufferHandle]
    scalar_params: dict[str, int | float]
    grid: DispatchGrid = field(default_factory=lambda: SINGLE_DISPATCH)
    local_work_size: int | None = None

    @property
    def tile_count(self) -> int:
        """Total dispatch count derived from the grid geometry.

        Convenience property preserving the intuitive "how many
        dispatches?" query.  Returns 1 for ``SINGLE_DISPATCH``.
        """
        return self.grid.total_dispatches


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
# §3  Validation
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


def _check_grid_strides(
    node_id: str,
    node: KernelDispatchNode,
) -> list[str]:
    """Verify grid stride parameters exist in ``scalar_params``.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []
    for stride in node.grid.all_strides():
        if stride.param_name not in node.scalar_params:
            errors.append(
                f"Node {node_id!r}: grid stride references "
                f"{stride.param_name!r} not in scalar_params"
            )
    return errors


def _check_streaming_strides(
    loop_id: str,
    loop: StreamingLoopNode,
) -> list[str]:
    """Verify streaming parameter strides reference valid body scalars.

    Returns a list of error messages (empty if valid).
    ``StreamingLoopPlan.__post_init__`` already checks that
    ``target_nodes ⊆ body``; this additionally checks that each
    targeted body node's ``scalar_params`` contains the strided
    parameter.
    """
    errors: list[str] = []
    body_by_id = {bn.node_id: bn for bn in loop.body_nodes}
    for ps in loop.streaming_plan.parameter_strides:
        for target_nid in ps.target_nodes:
            bn = body_by_id.get(target_nid)
            if bn is None:
                # Target not found — already caught by
                # StreamingLoopPlan.__post_init__, but defense-in-depth.
                errors.append(
                    f"Loop {loop_id!r}: ParameterStride for "
                    f"{ps.param_name!r} targets {target_nid!r}, "
                    f"which is not a body node"
                )
            elif ps.param_name not in bn.scalar_params:
                errors.append(
                    f"Loop {loop_id!r}: ParameterStride for "
                    f"{ps.param_name!r} targets body node "
                    f"{target_nid!r}, which lacks that scalar param"
                )
    return errors


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

    **G — Grid & stride consistency:** dispatch grid strides
    reference parameters that exist in ``scalar_params``; streaming
    loop strides reference parameters that exist in their target
    body nodes' ``scalar_params``.
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

    # ── G1: Dispatch grid strides reference valid scalar params ───

    for nid, node in nodes.items():
        if isinstance(node, KernelDispatchNode):
            errors = _check_grid_strides(nid, node)
            if errors:
                raise PlanValidationError(errors[0])

        elif isinstance(node, StreamingLoopNode):
            for bn in node.body_nodes:
                errors = _check_grid_strides(bn.node_id, bn)
                if errors:
                    raise PlanValidationError(errors[0])

    # ── G2: Streaming strides reference valid body scalar params ──

    for nid, node in nodes.items():
        if isinstance(node, StreamingLoopNode):
            errors = _check_streaming_strides(nid, node)
            if errors:
                raise PlanValidationError(errors[0])


# ═══════════════════════════════════════════════════════════════════
# §4  ExecutionPlan
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
