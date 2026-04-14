# src/shared/streaming_loop_plan.py
"""Stride-based parametric streaming loop specification (ADR-004).

A streaming loop expresses the True Streaming backpropagation model
(CONCEPT.md §11, Model B): recompute inputs, compute partial gradients,
clip immediately, and feed clipped partials into the reduction engine —
all within a single parametric loop.

Iterations are **sequential** — body nodes may overwrite scratch buffers
between iterations, creating intra-iteration data dependencies.  This is
the defining semantic difference from ``DispatchGrid`` (plan_types.py),
whose dispatches are independent and may execute in parallel.

Per-iteration parameter advancement
────────────────────────────────────
Each ``ParameterStride`` declares a scalar parameter that varies
linearly across iterations.  For iteration *i*::

    value(i) = scalar_params[param_name] + i × stride

where ``scalar_params`` is the body node's constant-parameter dict,
carrying the base (iteration-0) value.  The ``target_nodes`` field
scopes the stride to specific body nodes; non-targeted body nodes
receive the unmodified base value from their ``scalar_params``.

This convention — base lives in ``scalar_params``, not in the stride
object — is shared with ``ScalarStride`` (plan_types.py) and eliminates
the invariant-maintenance burden of a redundant ``base`` field.

Authoritative sources
─────────────────────
ADR-004    — Streaming loop plan specification
CONCEPT.md §11 — Model B: True Streaming
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "IterationDimension",
    "ParameterStride",
    "ScratchBufferSpec",
    "StreamingLoopPlan",
]


@dataclass(frozen=True)
class IterationDimension:
    """Describes the dimension over which the streaming loop iterates.

    Parameters
    ----------
    total_extent:
        Total elements in the iterated dimension (e.g. ``batch_size``).
    chunk_count:
        Number of iterations (chunks).
    chunk_size:
        Elements per iteration.  The last iteration may process
        fewer than ``chunk_size`` elements when ``total_extent``
        is not evenly divisible; the renderer must account for this
        via per-iteration ``batch_chunk_count`` scalars.
    """

    total_extent: int
    chunk_count: int
    chunk_size: int

    def __post_init__(self) -> None:
        if self.total_extent < 0:
            raise ValueError(
                f"total_extent must be ≥ 0, got {self.total_extent}"
            )
        if self.chunk_count < 1:
            raise ValueError(
                f"chunk_count must be ≥ 1, got {self.chunk_count}"
            )
        if self.chunk_size < 1:
            raise ValueError(
                f"chunk_size must be ≥ 1, got {self.chunk_size}"
            )


@dataclass(frozen=True)
class ParameterStride:
    """A scalar parameter that advances linearly across streaming iterations.

    For iteration *i*, the parameter's value is::

        scalar_params[param_name] + i × stride

    where ``scalar_params`` is the body node's constant-parameter dict,
    carrying the base (iteration-0) value.

    ``target_nodes`` scopes which body nodes receive the per-iteration
    advancement.  Body nodes listed in ``target_nodes`` have their
    ``scalar_params[param_name]`` overridden with the computed value;
    unlisted body nodes retain the unmodified base value.

    This type is the streaming-loop counterpart of ``ScalarStride``
    (plan_types.py).  Both share the same ``param_name`` + ``stride``
    semantics and the same "base lives in ``scalar_params``" convention.
    The additional ``target_nodes`` field reflects the streaming loop's
    multi-body-node structure — a dispatch grid's strides apply
    uniformly to a single dispatch.

    Parameters
    ----------
    param_name:
        Full canonical scalar parameter name (CONTRACT §2.3).
        Must exist in the ``scalar_params`` dict of every body node
        listed in ``target_nodes``.
    stride:
        Per-iteration increment.
    target_nodes:
        Body node IDs to which this stride applies.  Must be a
        non-empty subset of ``StreamingLoopPlan.body``.
    """

    param_name: str
    stride: int | float
    target_nodes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.target_nodes:
            raise ValueError(
                f"target_nodes must be non-empty for stride on "
                f"{self.param_name!r}"
            )


@dataclass(frozen=True)
class ScratchBufferSpec:
    """Declares a renderer-internal scratch buffer used within the loop body.

    The renderer allocates this buffer once and reuses it across
    iterations.  These are NOT plan-level buffers (ADR-009 two-tier
    scope) — they do not appear in ``ExecutionPlan.buffers``.

    Parameters
    ----------
    logical_name:
        Human-readable identifier for debugging and logging.
    size_bytes:
        Allocation size in bytes.
    shape:
        Logical shape for renderer-side dimension tracking.
    """

    logical_name: str
    size_bytes: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class StreamingLoopPlan:
    """Stride-based parametric specification for a streaming loop.

    The ``body`` is an ordered sequence of ``KernelDispatchNode`` IDs.
    The renderer iterates ``iteration.chunk_count`` times, dispatching
    body nodes in order each iteration, applying ``parameter_strides``
    to advance per-iteration scalar values.

    Intra-body ordering is enforced by the body nodes' ``depends_on``
    edges (sibling body-node IDs).  External gating (e.g. dependence
    on Node 16's completion) is expressed on the parent
    ``StreamingLoopNode``'s own ``depends_on``.

    Parameters
    ----------
    iteration:
        Dimension geometry for the iteration axis.
    body:
        Ordered body node IDs.  Must match ``tuple(bn.node_id for bn
        in StreamingLoopNode.body_nodes)``.
    parameter_strides:
        Scalars that advance per iteration, scoped to specific body
        nodes.
    scratch_buffers:
        Renderer-internal scratch buffers allocated once and reused
        across iterations.
    constant_scalars:
        Loop-invariant scalar values available to the renderer.
        These supplement (do not override) body nodes'
        ``scalar_params``.
    """

    iteration: IterationDimension
    body: tuple[str, ...]
    parameter_strides: tuple[ParameterStride, ...]
    scratch_buffers: tuple[ScratchBufferSpec, ...]
    constant_scalars: dict[str, float]

    def __post_init__(self) -> None:
        if not self.body:
            raise ValueError("body must contain at least one node ID")

        # Verify stride target_nodes are subsets of body.
        body_ids = frozenset(self.body)
        for ps in self.parameter_strides:
            bad = ps.target_nodes - body_ids
            if bad:
                raise ValueError(
                    f"ParameterStride for {ps.param_name!r} targets "
                    f"{bad}, which are not body node IDs"
                )
