# src/shared/buffer_lifecycle.py
"""Plan-level buffer lifecycle types (ADR-009)."""
from dataclasses import dataclass
from enum import Enum, auto


class BufferHandle(int):
    """Opaque integer identifier for a plan-level buffer.

    BufferHandles are assigned by the plan builder during plan construction.
    They are meaningless outside the plan that created them — each plan has
    its own handle namespace.
    """
    pass


class BufferRole(Enum):
    """Lifecycle role classification for plan-level buffers (ADR-009)."""
    MODEL_STATE = auto()
    BATCH_INPUT = auto()
    BATCH_INTERMEDIATE = auto()
    BATCH_OUTPUT = auto()


@dataclass(frozen=True)
class BufferDescriptor:
    """Plan-level buffer declaration with lifetime annotations (ADR-009).

    The Policy tier computes lifetime intervals at plan-construction time.
    The Orchestration tier uses these intervals for physical allocation
    without re-analyzing the DAG.
    """
    handle: BufferHandle
    logical_name: str
    padded_shape: tuple[int, ...]
    element_size_bytes: int
    size_bytes: int
    role: BufferRole
    producing_node: str | None
    consumers: frozenset[str]
    last_consumer: str | None
