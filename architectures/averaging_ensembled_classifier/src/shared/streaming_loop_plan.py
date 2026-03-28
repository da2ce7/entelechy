# src/shared/streaming_loop_plan.py
"""Stride-based parametric streaming loop specification (ADR-004)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class IterationDimension:
    """Describes the dimension over which the loop iterates (ADR-004)."""
    total_extent: int
    chunk_count: int
    chunk_size: int


@dataclass(frozen=True)
class ParameterStride:
    """Describes how a single scalar parameter varies across iterations (ADR-004)."""
    param_name: str
    base: int
    stride: int


@dataclass(frozen=True)
class ScratchBufferSpec:
    """Declares a renderer-internal scratch buffer used within the loop body (ADR-004).

    The renderer allocates this buffer once and reuses it across iterations.
    These are NOT plan-level buffers (ADR-009 two-tier scope).
    """
    logical_name: str
    size_bytes: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class StreamingLoopPlan:
    """Stride-based parametric specification for a streaming loop (ADR-004).

    The body is a flat sequence of KernelDispatchNode node_ids. The renderer
    instantiates per-chunk parameter values from base + index * stride.
    """
    iteration: IterationDimension
    body: tuple[str, ...]
    parameter_strides: tuple[ParameterStride, ...]
    scratch_buffers: tuple[ScratchBufferSpec, ...]
    constant_scalars: dict[str, float]
