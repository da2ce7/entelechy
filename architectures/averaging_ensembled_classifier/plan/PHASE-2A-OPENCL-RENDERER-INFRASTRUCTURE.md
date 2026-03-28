# Phase 2A: OpenCL Renderer Infrastructure — Detailed Plan

**Status:** Not started  
**Phase:** 2A of 6 (sub-phase A of 3)  
**Objective:** Establish the `PlanRenderer` Protocol in the shared layer and implement the OpenCL backend's core rendering infrastructure — context management, device discovery, buffer allocation, retrieval futures, and the top-level `OpenCLPlanRenderer` that traverses an `ExecutionPlan` DAG and dispatches via PyOpenCL. This sub-phase produces a renderer that can traverse the plan but delegates individual kernel dispatch to stub bindings (populated in Phase 2B).  
**Governing ADRs:** ADR-001 (backend abstraction boundary), ADR-006 (HardwareProfile population), ADR-009 (buffer lifecycle / allocation), ADR-010 (RetrievalFuture implementation), ADR-012 (module factoring), ADR-014 (build system — OpenCL always enabled), ADR-015 (interop — PyOpenCL established)  
**Rollback gate:** Tier 1 green (no regressions) + structural smoke tests for renderer infrastructure pass. Full OpenCL Tier 2 gate deferred to Phase 2C.  
**Dependencies:** Phase 1 (plan model) — **COMPLETE** (332 tests green: 192 Phase 0 + 140 Tier 1).

### Phase 1 Deliverables Consumed Here

Phase 2A is the first consumer of the Phase 1 plan model from the backend side. The following shared-layer types cross the boundary:

| Shared Type | Phase 2A Consumption |
| :--- | :--- |
| `ExecutionPlan` | Input to `OpenCLPlanRenderer.render()` |
| `KernelDispatchNode` | Dispatched via `KernelBinding` (stub in 2A, populated in 2B) |
| `ReductionTreeNode` | Rendered by reduction tree rendering loop (stub in 2A, logic in 2B) |
| `StreamingLoopNode` | Rendered by streaming loop rendering loop (stub in 2A, logic in 2B) |
| `BarrierNode` | Translated to `cl.Event` join point |
| `RetrievalNode` | Triggers async D2H transfer → `OpenCLRetrievalFuture` |
| `BufferDescriptor` | Drives `OpenCLBufferAllocator` allocation decisions |
| `BufferHandle` | Mapped to physical `cl.Buffer` objects by the allocator |
| `BufferRole` | Determines allocation strategy (persistent vs. per-batch) |
| `HardwareProfile` | Populated by `discovery.py` from OpenCL device queries |
| `PrecisionConfig` | Drives OpenCL build flags (`-DSCALAR_TYPE`, `-DSCALAR_IS_HALF`) |
| `RetrievalFuture` | Protocol implemented by `OpenCLRetrievalFuture` |
| `ReductionTreePlan` | Consumed by the reduction tree rendering loop |
| `StreamingLoopPlan` | Consumed by the streaming loop rendering loop |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State After Phase 1](#2-current-state-after-phase-1)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 2A.1: Define `PlanRenderer` Protocol → `src/shared/plan_renderer.py`](#step-2a1-define-planrenderer-protocol--srcsharedplan_rendererpy)
   - [Step 2A.2: Implement `discovery.py` → `src/backends/opencl/discovery.py`](#step-2a2-implement-discoverypy--srcbackendsopencldiscoverypy)
   - [Step 2A.3: Implement `type_mapping.py` → `src/backends/opencl/type_mapping.py`](#step-2a3-implement-type_mappingpy--srcbackendsopenclttype_mappingpy)
   - [Step 2A.4: Refactor `context.py` → `src/backends/opencl/context.py`](#step-2a4-refactor-contextpy--srcbackendsopenclcontextpy)
   - [Step 2A.5: Implement `buffer_allocator.py` → `src/backends/opencl/buffer_allocator.py`](#step-2a5-implement-buffer_allocatorpy--srcbackendsopenclbuffer_allocatorpy)
   - [Step 2A.6: Implement `OpenCLRetrievalFuture` → `src/backends/opencl/retrieval.py`](#step-2a6-implement-openclretrievalfuture--srcbackendsopenclretrievalpy)
   - [Step 2A.7: Implement `OpenCLPlanRenderer` (DAG traversal) → `src/backends/opencl/renderer.py`](#step-2a7-implement-openclplanrenderer-dag-traversal--srcbackendsopenclrendererpy)
   - [Step 2A.8: Update `src/backends/opencl/__init__.py` exports](#step-2a8-update-srcbackendsopencl__init__py-exports)
   - [Step 2A.9: Update shared `__init__.py` for `PlanRenderer`](#step-2a9-update-shared-__init__py-for-planrenderer)
   - [Step 2A.10: Write infrastructure smoke tests](#step-2a10-write-infrastructure-smoke-tests)
   - [Step 2A.11: Validate rollback gate](#step-2a11-validate-rollback-gate)
5. [Module Inventory](#5-module-inventory)
6. [`PlanRenderer` Protocol Specification](#6-planrenderer-protocol-specification)
7. [Buffer Allocation Strategy](#7-buffer-allocation-strategy)
8. [Event Chain Model](#8-event-chain-model)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

- Defining the `PlanRenderer` Protocol in `src/shared/plan_renderer.py` — the sole contract between the shared layer and any backend (ADR-001).
- Implementing `src/backends/opencl/discovery.py` — populating `HardwareProfile` from OpenCL device queries (`cl.device_info`), including computing `max_reduce_fan_in` from `max_work_group_size` and local memory constraints (ADR-006).
- Implementing `src/backends/opencl/type_mapping.py` — mapping `PrecisionConfig` to OpenCL compiler flags (`-DSCALAR_TYPE=float`, `-DSIMD_WIDTH=16`, etc.) and `numpy.dtype` ↔ OpenCL type name resolution.
- Refactoring `src/backends/opencl/context.py` — adapting the existing `CLContextManager` to expose a clean context/queue lifecycle that the renderer consumes, decoupled from legacy `Services` pattern.
- Implementing `src/backends/opencl/buffer_allocator.py` — translating `BufferDescriptor` plan-level declarations into physical `cl.Buffer` allocations, with lifetime management driven by `BufferRole` and producer/consumer annotations.
- Implementing `src/backends/opencl/retrieval.py` — `OpenCLRetrievalFuture` satisfying the `RetrievalFuture` Protocol via `cl.Event`-gated async D2H read with padding-stripping.
- Implementing `src/backends/opencl/renderer.py` — the `OpenCLPlanRenderer` class that traverses `ExecutionPlan.topological_order`, dispatches each node type through its corresponding rendering logic, manages the `cl.Event` dependency graph, and returns `RetrievalFuture` instances for `RetrievalNode`s.
- Writing infrastructure-level smoke tests that validate the renderer can traverse a simple plan without failing (full kernel-level correctness deferred to Phase 2C).

### Out of scope

- Implementing `KernelBinding` dispatch adapters for individual kernels (Phase 2B).
- Writing per-kernel Tier 2 correctness tests (Phase 2C).
- Implementing Vulkan or CPU backends.
- Modifying any shared-layer code beyond adding the `PlanRenderer` Protocol.
- Modifying kernel source files (`kernels/*.cl.c`, `kernels/kernels.cl.h`).
- Implementing the `WorkTicket` / `LearnHandle` / `Engine` user-facing API (parallel workstream).

### Key constraint: backward compatibility

The existing OpenCL backend code in `src/backends/opencl/` (migrated from Phase 0) must continue to operate. The legacy `batch_processor.py`, `graph_recipes.py`, `execution_plan.py`, `launcher_infra.py`, and `compute_patterns.py` modules are not modified or removed — they coexist with the new renderer infrastructure until Phase 6. Existing tests (332 green) must remain green.

### Key constraint: renderer is a pure consumer of the plan

The renderer receives an immutable `ExecutionPlan` and never modifies it. All plan construction occurs in the shared layer (`plan_builder.py`). The renderer's role is strictly Orchestration-tier: translate plan nodes into native PyOpenCL dispatch calls.

---

## 2. Current State After Phase 1

### OpenCL backend (`src/backends/opencl/`)

| Module | Status | Notes |
| :--- | :--- | :--- |
| `__init__.py` | Minimal | Re-exports from legacy modules |
| `context.py` | Legacy | `CLContextManager` with `Services` pattern; creates `cl.Context`, `cl.CommandQueue`. Tightly coupled to legacy dispatch model. |
| `compute_patterns.py` | Legacy | Imperative kernel dispatch patterns using `cl.Event` chaining |
| `execution_plan.py` | Legacy | `DependencyProvider` / `CacheProvider` / `ComputeOnceProvider` — OpenCL-specific plan execution with `cl.Event` resolution |
| `graph_recipes.py` | Legacy | Recipe functions composing kernel launches into sub-DAG patterns (reduction trees, streaming loops). ~984 lines. |
| `launcher_infra.py` | Legacy | `Services`, `BufferManager`, `KernelExecutor` — the OpenCL dispatch infrastructure |
| `batch_processor.py` | Legacy | Top-level `BatchProcessor.run()` orchestrating Act/Learn phases through `graph_recipes` |
| `kernel_bindings/` | Legacy | `KernelSignature` subclasses per kernel — fused validation + OpenCL argument marshalling |

### Not yet created (Phase 2A deliverables)

| Module | Purpose |
| :--- | :--- |
| `src/shared/plan_renderer.py` | `PlanRenderer` Protocol |
| `src/backends/opencl/discovery.py` | `HardwareProfile` population from OpenCL device |
| `src/backends/opencl/type_mapping.py` | `PrecisionConfig` → OpenCL compiler flags |
| `src/backends/opencl/buffer_allocator.py` | `BufferDescriptor` → `cl.Buffer` allocation |
| `src/backends/opencl/retrieval.py` | `OpenCLRetrievalFuture` |
| `src/backends/opencl/renderer.py` | `OpenCLPlanRenderer` |

---

## 3. Target Deliverables

After Phase 2A completes:

```
src/shared/
├── plan_renderer.py                    # NEW: PlanRenderer Protocol
└── ... (unchanged Phase 1 modules)

src/backends/opencl/
├── __init__.py                         # UPDATED: exports new modules
├── renderer.py                         # NEW: OpenCLPlanRenderer
├── retrieval.py                        # NEW: OpenCLRetrievalFuture
├── buffer_allocator.py                 # NEW: OpenCLBufferAllocator
├── discovery.py                        # NEW: HardwareProfile population
├── type_mapping.py                     # NEW: PrecisionConfig → OpenCL flags
├── context.py                          # MODIFIED: adapted for renderer consumption
├── batch_processor.py                  # UNCHANGED (legacy)
├── compute_patterns.py                 # UNCHANGED (legacy)
├── execution_plan.py                   # UNCHANGED (legacy)
├── graph_recipes.py                    # UNCHANGED (legacy)
├── launcher_infra.py                   # UNCHANGED (legacy)
└── kernel_bindings/                    # UNCHANGED (legacy; adapted in Phase 2B)
```

---

## 4. Task Breakdown

### Step 2A.1: Define `PlanRenderer` Protocol → `src/shared/plan_renderer.py`

**Action:** Create `src/shared/plan_renderer.py` containing the `PlanRenderer` Protocol — the sole interface through which the shared layer (and eventually the `Engine`) interacts with any backend.

**Governing ADR:** ADR-001 (backend abstraction boundary — "Each backend exposes a `PlanRenderer` interface — the sole contract between the shared layer and any backend").

**Protocol definition:**

```python
from typing import Protocol, runtime_checkable

from .plan_types import ExecutionPlan
from .retrieval_future import RetrievalFuture


@runtime_checkable
class PlanRenderer(Protocol):
    """Protocol for backend-specific plan rendering (ADR-001).

    Each backend implements this protocol to translate an immutable
    ExecutionPlan into native dispatch calls. The renderer is the
    Orchestration-tier entry point.
    """

    def render(self, plan: ExecutionPlan) -> dict[str, RetrievalFuture]:
        """Render an execution plan using the backend's native dispatch model.

        Traverses the plan's topological order, dispatching each node
        according to its type. Returns a mapping of event_name → RetrievalFuture
        for each RetrievalNode in the plan.

        The renderer:
        - Allocates physical buffers from BufferDescriptors
        - Dispatches KernelDispatchNodes via KernelBindings
        - Renders ReductionTreeNodes as multi-stage kernel sequences
        - Renders StreamingLoopNodes as parametric dispatch loops
        - Translates BarrierNodes into native synchronization
        - Creates RetrievalFutures for RetrievalNodes

        The plan is not modified. All mutable state (buffers, events,
        futures) is renderer-internal.
        """
        ...
```

**Design decision — Protocol vs. ABC:** `Protocol` (with `@runtime_checkable`) rather than `abc.ABC` because:
- The shared layer never instantiates a renderer — it receives one.
- `isinstance` checks via `runtime_checkable` are sufficient for validation.
- Avoids forcing backends to inherit from a shared base class, preserving per-backend implementation freedom.
- Consistent with `RetrievalFuture` (also a Protocol).

**Additional methods (deferred):** The `Engine` (ADR-018, parallel workstream) may require additional `PlanRenderer` methods such as `upload_model_state()` and `download_model_state()`. These are not defined here — they will be added when the `Engine` workstream reaches the renderer boundary. Phase 2A defines only `render()`.

**Validation:** `isinstance(OpenCLPlanRenderer(), PlanRenderer)` returns `True`.

---

### Step 2A.2: Implement `discovery.py` → `src/backends/opencl/discovery.py`

**Action:** Create `src/backends/opencl/discovery.py` containing the function that queries an OpenCL device and produces a `HardwareProfile`.

**Governing ADR:** ADR-006 (HardwareProfile with policy-input naming). DESIGN.md §3.4 specifies the field semantics.

**Function signature:**

```python
import pyopencl as cl

from ...shared.hardware_profile import HardwareProfile


def discover_hardware(device: cl.Device) -> HardwareProfile:
    """Populate a HardwareProfile from an OpenCL device's capabilities.

    Computes max_reduce_fan_in from the device's work-group and local memory
    constraints — this is the Orchestration-tier derivation that the Policy
    tier consumes as an abstract budget.
    """
    ...
```

**Field derivation logic:**

| HardwareProfile field | OpenCL device query | Derivation |
| :--- | :--- | :--- |
| `simd_width` | `device.get_info(cl.device_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE)` | Direct (preferred wavefront/warp size) |
| `cache_line_bytes` | `device.get_info(cl.device_info.GLOBAL_MEM_CACHELINE_SIZE)` | Direct; fallback to `64` if device reports `0` |
| `max_reduce_fan_in` | Derived from `MAX_WORK_GROUP_SIZE` and `LOCAL_MEM_SIZE` | `min(max_work_group_size, local_mem_size // (element_size * 2))` — the maximum number of partials a single workgroup can reduce using local memory ping-pong. Capped by architectural limit if needed. |
| `max_local_mem_bytes` | `device.get_info(cl.device_info.LOCAL_MEM_SIZE)` | Direct |
| `global_mem_bytes` | `device.get_info(cl.device_info.GLOBAL_MEM_SIZE)` | Direct |

**Key derivation — `max_reduce_fan_in`:** The Policy tier uses this value as the reduction batch size $K$ upper bound. The OpenCL backend computes it from:
- `MAX_WORK_GROUP_SIZE` — hardware limit on work-items per workgroup.
- `LOCAL_MEM_SIZE` — local memory available per workgroup; the local-reduce kernel requires `2 × K × element_size` bytes for ping-pong reduction.
- The final value is `min(max_wg_size, local_mem // (2 * element_size))`, respecting both constraints. This is a conservative bound; the renderer may further tune at dispatch time.

**Validation:** `discover_hardware(device)` returns a valid `HardwareProfile` with all fields populated. Integration test (requires OpenCL device) verifies plausible values.

---

### Step 2A.3: Implement `type_mapping.py` → `src/backends/opencl/type_mapping.py`

**Action:** Create `src/backends/opencl/type_mapping.py` containing mapping functions between `PrecisionConfig` and OpenCL compiler symbols.

**Purpose:** The OpenCL backend compiles kernel sources at runtime via `clBuildProgram`. The compiler needs `-D` flags from CONTRACT.md Article 6 (Mandatory Build-Time Symbols). This module translates the plan's `PrecisionConfig` and `HardwareProfile` into those flags.

**Function signatures:**

```python
from ...shared.precision_config import PrecisionConfig
from ...shared.hardware_profile import HardwareProfile


def build_compiler_flags(
    precision: PrecisionConfig,
    hardware: HardwareProfile,
    c_tile_size: int,
) -> list[str]:
    """Produce OpenCL -D compiler flags from plan-level configuration.

    Generates flags for all CONTRACT.md Article 6 mandatory symbols:
    SCALAR_TYPE, SIMD_WIDTH, C_TILE_SIZE, SCALAR_IS_HALF,
    NUMERICAL_STABILITY_EPSILON.
    """
    ...


def numpy_dtype_to_cl_type_name(precision: PrecisionConfig) -> str:
    """Map PrecisionConfig.numpy_dtype to OpenCL C type name.

    np.float32 → "float", np.float16 → "half".
    """
    ...
```

**Flag generation:**

| Flag | Source | Example |
| :--- | :--- | :--- |
| `-DSCALAR_TYPE=float` | `PrecisionConfig.numpy_dtype` | `np.float32` → `float`; `np.float16` → `half` |
| `-DSIMD_WIDTH=16` | `HardwareProfile.simd_width` | |
| `-DC_TILE_SIZE=32` | `c_tile_size` parameter | |
| `-DSCALAR_IS_HALF=0` | `PrecisionConfig.numpy_dtype` | `1` if `float16`, else `0` |
| `-DNUMERICAL_STABILITY_EPSILON=1e-7` | `PrecisionConfig.epsilon` | |

**Validation:** `build_compiler_flags(PrecisionConfig.float32(), hw, 32)` returns the expected list of flag strings. Edge case: FP16 flags correctly set `SCALAR_IS_HALF=1`.

---

### Step 2A.4: Refactor `context.py` → `src/backends/opencl/context.py`

**Action:** Adapt the existing `CLContextManager` in `src/backends/opencl/context.py` to expose a clean interface for the new renderer, while preserving backward compatibility with the legacy `Services` pattern.

**Current state:** `context.py` contains the `CLContextManager` (or equivalent) that creates `cl.Context` and `cl.CommandQueue`. The legacy `launcher_infra.py` `Services` class bundles context, queue, `BufferManager`, `KernelExecutor`, and `DiscoveredArchConstants` into a single object.

**Target state:** Add a lightweight factory function or class that the renderer can use independently:

```python
from dataclasses import dataclass

import pyopencl as cl


@dataclass
class OpenCLContext:
    """Holds the OpenCL context, queue, and compiled program.

    Created by the renderer at initialization. Immutable after construction
    (aside from the reference-counted OpenCL objects themselves).
    """
    context: cl.Context
    queue: cl.CommandQueue
    device: cl.Device
    program: cl.Program | None  # Set after kernel compilation
```

**Key decisions:**

1. **Do not remove the legacy `CLContextManager` or `Services`.** They remain functional for the legacy `batch_processor.py` path. Phase 6 removes them.
2. **The renderer creates `OpenCLContext` via standard PyOpenCL `cl.create_some_context()` or from an explicitly provided `cl.Context`.** This supports both interactive (device selection dialog) and programmatic (CI, specific device) usage.
3. **Kernel compilation** (`cl.Program` construction from source files) is handled here or in the renderer's initialization. The program is built once with the flags from `type_mapping.py` and stored in `OpenCLContext.program`.

**Backward compatibility:** The legacy `Services` class can optionally wrap an `OpenCLContext` to bridge old and new code during the transition. This is a convenience, not a requirement — the two paths are independent.

**Validation:** `OpenCLContext` can be constructed from a valid `cl.Context` and `cl.CommandQueue`.

---

### Step 2A.5: Implement `buffer_allocator.py` → `src/backends/opencl/buffer_allocator.py`

**Action:** Create `src/backends/opencl/buffer_allocator.py` containing the `OpenCLBufferAllocator` that translates plan-level `BufferDescriptor`s into physical `cl.Buffer` allocations.

**Governing ADR:** ADR-009 (buffer lifecycle — plan-prescribed lifetime intervals).

**Class design:**

```python
import pyopencl as cl

from ...shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole


class OpenCLBufferAllocator:
    """Allocates and manages physical cl.Buffer objects for a single plan.

    Consumes BufferDescriptors from the plan and creates cl.Buffer objects
    with appropriate memory flags. Manages the handle → cl.Buffer mapping.

    Lifetime: one allocator instance per render() call. When the allocator
    is released, all non-MODEL_STATE buffers are freed.
    """

    def __init__(self, context: cl.Context, queue: cl.CommandQueue) -> None:
        ...

    def allocate_plan_buffers(
        self, descriptors: dict[BufferHandle, BufferDescriptor]
    ) -> None:
        """Allocate cl.Buffers for all descriptors in the plan."""
        ...

    def get_buffer(self, handle: BufferHandle) -> cl.Buffer:
        """Retrieve the physical cl.Buffer for a plan-level handle."""
        ...

    def upload(
        self, handle: BufferHandle, data: "NDArray", wait_for: list[cl.Event] | None = None
    ) -> cl.Event:
        """Enqueue an async host→device transfer for a BATCH_INPUT buffer."""
        ...

    def enqueue_read(
        self, handle: BufferHandle, wait_for: list[cl.Event] | None = None
    ) -> tuple[cl.Event, "NDArray"]:
        """Enqueue an async device→host transfer. Returns (event, host_buffer)."""
        ...

    def release_non_persistent(self) -> None:
        """Release all buffers except MODEL_STATE (which persist across batches)."""
        ...
```

**Allocation strategy by `BufferRole`:**

| `BufferRole` | `cl.mem_flags` | Lifetime | Notes |
| :--- | :--- | :--- | :--- |
| `MODEL_STATE` | `READ_WRITE` | Persists across `render()` calls | Weights, biases, Adam state. Allocated once; the renderer maintains a persistent handle mapping across batches. |
| `BATCH_INPUT` | `READ_ONLY \| COPY_HOST_PTR` or `READ_ONLY` + async upload | Per-batch | Input data, targets, sample masks. Uploaded at render start. |
| `BATCH_INTERMEDIATE` | `READ_WRITE` | Per-batch (freed after last consumer) | Partials, activations, intermediate gradients. |
| `BATCH_OUTPUT` | `READ_WRITE` | Per-batch (freed after retrieval read) | Final probs, loss values. Read back to host via `RetrievalNode`. |

**Renderer-internal buffers (not plan-level):**

The `OpenCLBufferAllocator` also handles renderer-internal allocations requested during rendering:
- **Reduction tree intermediate buffers:** Ping-pong buffers for multi-stage reduction. Sized from `ReductionTreePlan` parameters. Not described by `BufferDescriptor`s — the renderer allocates them on demand.
- **Streaming loop scratch buffers:** Sized from `ScratchBufferSpec` in the `StreamingLoopPlan`. Allocated once, reused across iterations.
- **Offset list device buffers:** `initial_offset_list` from `ReductionTreePlan` uploaded to device memory for stage 0.

These renderer-internal buffers are tracked separately and freed when the allocator is released.

**Validation:** Allocate buffers for a small plan; verify handle→buffer mapping; verify role-based `mem_flags`; verify release frees non-persistent buffers.

---

### Step 2A.6: Implement `OpenCLRetrievalFuture` → `src/backends/opencl/retrieval.py`

**Action:** Create `src/backends/opencl/retrieval.py` containing `OpenCLRetrievalFuture`, the OpenCL implementation of the `RetrievalFuture` Protocol (ADR-010).

**Protocol contract (from Phase 1):**

| Method | OpenCL implementation |
| :--- | :--- |
| `node_id` (property) | Returns the `RetrievalNode.node_id` that produced this future |
| `wait()` | Calls `self._event.wait()` on the D2H transfer event. Idempotent. |
| `result()` | Calls `wait()` if not yet complete. Returns the unpadded numpy array — the renderer strips SIMD/cache padding from the raw device readback before storing the result. |
| `release()` | Marks the host buffer and device staging buffer as reclaimable. After release, `result()` raises an error. |

**Implementation sketch:**

```python
import numpy as np
import pyopencl as cl
from numpy.typing import NDArray

from ...shared.retrieval_future import RetrievalFuture


class OpenCLRetrievalFuture:
    """OpenCL implementation of RetrievalFuture (ADR-010).

    Wraps a cl.Event from an async D2H enqueue_read. The renderer
    constructs this when processing a RetrievalNode.
    """

    def __init__(
        self,
        node_id: str,
        event: cl.Event,
        host_buffer: NDArray,
        logical_shape: tuple[int, ...],
    ) -> None:
        self._node_id = node_id
        self._event = event
        self._host_buffer = host_buffer
        self._logical_shape = logical_shape
        self._released = False
        self._result_cache: NDArray | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        if not self._released:
            self._event.wait()

    def result(self) -> NDArray[np.floating]:
        if self._released:
            raise RuntimeError("RetrievalFuture has been released")
        if self._result_cache is None:
            self.wait()
            # Strip padding: extract logical shape from padded host buffer
            slices = tuple(slice(0, s) for s in self._logical_shape)
            self._result_cache = np.array(
                self._host_buffer.reshape(self._padded_shape())[slices],
                copy=True,
            )
        return self._result_cache

    def release(self) -> None:
        self._released = True
        self._host_buffer = None  # type: ignore[assignment]
        self._result_cache = None
        self._event = None  # type: ignore[assignment]
```

**Padding-stripping responsibility:** Per ADR-010, the renderer owns padding-stripping. The `OpenCLRetrievalFuture` receives the padded host buffer (from `clEnqueueReadBuffer`) and the `logical_shape` (from the `RetrievalNode`). On `.result()`, it extracts the unpadded region matching the logical shape.

**Protocol compliance:** `isinstance(OpenCLRetrievalFuture(...), RetrievalFuture)` returns `True` (runtime_checkable).

**Validation:** Construct a future with a mock event; verify `.wait()`, `.result()`, `.release()` lifecycle; verify padding-stripping.

---

### Step 2A.7: Implement `OpenCLPlanRenderer` (DAG traversal) → `src/backends/opencl/renderer.py`

**Action:** Create `src/backends/opencl/renderer.py` containing the `OpenCLPlanRenderer` — the Orchestration-tier entry point that traverses an `ExecutionPlan` and dispatches each node via PyOpenCL.

**This is the central deliverable of Phase 2A.** It establishes the plan rendering loop, event chain management, and buffer lifecycle orchestration. Individual kernel dispatch is delegated to `KernelBinding` instances (populated in Phase 2B).

**Class design:**

```python
import pyopencl as cl

from ...shared.plan_types import (
    ExecutionPlan, KernelDispatchNode, ReductionTreeNode,
    StreamingLoopNode, BarrierNode, RetrievalNode, PlanNode,
)
from ...shared.plan_renderer import PlanRenderer
from ...shared.retrieval_future import RetrievalFuture
from .buffer_allocator import OpenCLBufferAllocator
from .retrieval import OpenCLRetrievalFuture


class OpenCLPlanRenderer:
    """OpenCL implementation of PlanRenderer (ADR-001).

    Renders an ExecutionPlan by traversing its topological order and
    dispatching each node using PyOpenCL's imperative enqueue model.
    """

    def __init__(
        self,
        context: cl.Context,
        queue: cl.CommandQueue,
        program: cl.Program,
        kernel_bindings: dict,  # kernel_name → KernelBinding (populated in Phase 2B)
    ) -> None:
        ...

    def render(self, plan: ExecutionPlan) -> dict[str, RetrievalFuture]:
        ...
```

**DAG traversal algorithm:**

The renderer iterates through `plan.topological_order` and processes each node according to its type. The key internal state is the **event map**: a `dict[str, cl.Event | list[cl.Event]]` mapping `node_id` → the `cl.Event`(s) that signal the node's completion.

```
for node_id in plan.topological_order:
    node = plan.nodes[node_id]
    wait_for = collect_dependency_events(node.depends_on, event_map)

    match node:
        case KernelDispatchNode():
            event = dispatch_kernel(node, wait_for)
            event_map[node_id] = event

        case ReductionTreeNode():
            event = render_reduction_tree(node, wait_for)
            event_map[node_id] = event

        case StreamingLoopNode():
            event = render_streaming_loop(node, wait_for)
            event_map[node_id] = event

        case BarrierNode():
            # No dispatch — the barrier's completion event is the join
            # of all dependency events. A marker event suffices.
            event = create_marker_event(wait_for)
            event_map[node_id] = event

        case RetrievalNode():
            future = create_retrieval_future(node, wait_for)
            futures[node.event_name] = future
            event_map[node_id] = future._event

return futures
```

**Per-node-type rendering logic:**

| Node Type | Rendering Strategy | Event Output |
| :--- | :--- | :--- |
| `KernelDispatchNode` | Look up `KernelBinding` by `node.kernel_name`. For `tile_count` tiles: loop, set per-tile `flat_tile_index` scalar, enqueue `clEnqueueNDRange`. Collect all tile events. | Last tile event (or user event joining all tile events) |
| `ReductionTreeNode` | Delegate to `_render_reduction_tree()` — multi-stage loop with kernel tier selection, intermediate buffer management, offset list upload. Detailed in Phase 2B. | Final stage completion event |
| `StreamingLoopNode` | Delegate to `_render_streaming_loop()` — per-chunk parameter instantiation from `ParameterStride`, body node dispatch per chunk. Detailed in Phase 2B. | Final chunk completion event |
| `BarrierNode` | No dispatch. Create `cl.UserEvent` (or `clEnqueueMarkerWithWaitList`) that completes when all `depends_on` events complete. | Marker event |
| `RetrievalNode` | `enqueue_read` the source buffer with dependency events. Create `OpenCLRetrievalFuture` wrapping the read event, host buffer, and logical shape. | Read completion event |

**OpenCL imperative dispatch model (DESIGN.md §7.1):**

The OpenCL backend dispatches per-tile via `clEnqueueNDRange`. For a `KernelDispatchNode` with `tile_count=N`:

```python
events = []
for tile_idx in range(node.tile_count):
    # Set tile-specific scalars
    tile_scalars = {**node.scalar_params}
    # The KernelBinding adds flat_tile_index and derives grid/offset
    event = binding.dispatch(
        queue, program, allocator, tile_scalars, tile_idx,
        node.buffer_bindings, node.local_work_size, wait_for=wait_for,
    )
    events.append(event)
    # Each tile depends on the same upstream events, not on sibling tiles
```

This per-tile loop is the defining characteristic of the OpenCL dispatch model. Vulkan's single-dispatch model eliminates it; CPU's `pool_dispatch_and_wait` replaces it.

**Phase 2A scope:** The renderer's per-node-type methods are implemented as stubs that raise `NotImplementedError("KernelBinding not yet registered")` for `KernelDispatchNode`, `ReductionTreeNode`, and `StreamingLoopNode`. The `BarrierNode` and `RetrievalNode` paths are fully functional — they depend only on `cl.Event` management and the buffer allocator, not on kernel bindings. Phase 2B populates the bindings and enables full dispatch.

**Validation:** Construct a plan containing only `BarrierNode`s and `RetrievalNode`s; verify the renderer traverses without error and returns valid futures.

---

### Step 2A.8: Update `src/backends/opencl/__init__.py` exports

**Action:** Update the OpenCL backend's `__init__.py` to export the new modules' public API.

**Exported symbols:**

```python
from .renderer import OpenCLPlanRenderer
from .discovery import discover_hardware
from .buffer_allocator import OpenCLBufferAllocator
from .retrieval import OpenCLRetrievalFuture
from .type_mapping import build_compiler_flags, numpy_dtype_to_cl_type_name
```

---

### Step 2A.9: Update shared `__init__.py` for `PlanRenderer`

**Action:** Add `PlanRenderer` to `src/shared/__init__.py` exports.

```python
from .plan_renderer import PlanRenderer
```

---

### Step 2A.10: Write infrastructure smoke tests

**Action:** Create smoke tests that validate the renderer infrastructure without requiring full kernel bindings.

**Test location:** `tests/tier2/opencl/test_renderer_infrastructure.py`

**Test cases:**

1. **`test_discover_hardware`** — `discover_hardware(device)` returns a `HardwareProfile` with `simd_width > 0`, `cache_line_bytes > 0`, `max_reduce_fan_in > 1`, `global_mem_bytes > 0`.
2. **`test_build_compiler_flags_fp32`** — Flags contain `-DSCALAR_TYPE=float`, `-DSCALAR_IS_HALF=0`, correct epsilon.
3. **`test_build_compiler_flags_fp16`** — Flags contain `-DSCALAR_TYPE=half`, `-DSCALAR_IS_HALF=1`.
4. **`test_buffer_allocator_lifecycle`** — Allocate buffers from a small set of descriptors; verify `get_buffer()` returns valid `cl.Buffer`; verify `release_non_persistent()` frees intermediates.
5. **`test_retrieval_future_protocol`** — `OpenCLRetrievalFuture` satisfies `isinstance(..., RetrievalFuture)`.
6. **`test_renderer_barrier_only_plan`** — Construct a plan with barriers and a retrieval node; render; verify future lifecycle.

**Skip logic:** All tests use `@pytest.mark.skipif(not _build_config.BACKEND_OPENCL, reason="OpenCL not available")`.

---

### Step 2A.11: Validate rollback gate

**Action:** Run the complete test suite and confirm no regressions.

**Procedure:**

1. `pytest tests/tier1/ -v` — all Tier 1 tests pass (no shared-layer regressions).
2. `pytest src/tests/ tests/ -v` — all existing tests (332) pass.
3. `pytest tests/tier2/opencl/test_renderer_infrastructure.py -v` — infrastructure smoke tests pass.

---

## 5. Module Inventory

| Module | Content | Lines (est.) |
| :--- | :--- | :--- |
| `src/shared/plan_renderer.py` | `PlanRenderer` Protocol | 30–50 |
| `src/backends/opencl/discovery.py` | `discover_hardware()` | 40–60 |
| `src/backends/opencl/type_mapping.py` | Flag generation, dtype mapping | 40–60 |
| `src/backends/opencl/buffer_allocator.py` | `OpenCLBufferAllocator` | 150–250 |
| `src/backends/opencl/retrieval.py` | `OpenCLRetrievalFuture` | 60–80 |
| `src/backends/opencl/renderer.py` | `OpenCLPlanRenderer` (DAG traversal, stubs) | 200–350 |

---

## 6. `PlanRenderer` Protocol Specification

The `PlanRenderer` Protocol is deliberately minimal in Phase 2A. It exposes a single `render()` method. Additional methods are added as the `Engine` workstream (ADR-018) matures:

| Method | Phase | Purpose |
| :--- | :--- | :--- |
| `render(plan) → dict[str, RetrievalFuture]` | 2A | Dispatch an execution plan |
| `upload_model_state(params) → None` | Engine workstream | Upload initial model weights to device |
| `download_model_state() → dict` | Engine workstream | Read current model weights from device |

The Protocol evolves via the standard process: shared-layer Protocol definition first, then per-backend implementation.

---

## 7. Buffer Allocation Strategy

The allocator separates **persistent** and **per-batch** buffer pools:

```
┌─────────────────────────────┐
│   Persistent Pool           │   MODEL_STATE buffers
│   (lives across batches)    │   Allocated once at engine init
│   Weights, Biases, Adam M/V │   Updated in-place by adam_update
└─────────────────────────────┘

┌─────────────────────────────┐
│   Per-Batch Pool            │   BATCH_INPUT + BATCH_INTERMEDIATE + BATCH_OUTPUT
│   (allocated per render)    │   Freed at render() end (or on release())
│                             │
│   ┌───────────────────────┐ │
│   │ BATCH_INPUT           │ │   Uploaded at render start
│   │ (input_data, targets) │ │
│   └───────────────────────┘ │
│   ┌───────────────────────┐ │
│   │ BATCH_INTERMEDIATE    │ │   Allocated on demand during traversal
│   │ (partials, grads)     │ │   Freed after last_consumer completes
│   └───────────────────────┘ │
│   ┌───────────────────────┐ │
│   │ BATCH_OUTPUT          │ │   Read to host via RetrievalNode
│   │ (final_probs)         │ │   Freed on future.release()
│   └───────────────────────┘ │
└─────────────────────────────┘
```

**Eager vs. lazy allocation:** The allocator pre-allocates all plan buffers at the start of `render()` (eager). This avoids allocation latency mid-traversal and enables the OpenCL driver to optimize memory placement. The alternative (lazy allocation at `producing_node` time) defers allocation but risks fragmentation and adds allocation latency to the critical path.

---

## 8. Event Chain Model

The OpenCL backend's synchronization model translates plan dependency edges into `cl.Event` wait lists:

```
Plan DAG Edge                          OpenCL Realization
─────────────────────────────          ──────────────────
node_A.depends_on = {node_X, node_Y}   wait_for=[event_X, event_Y]
```

**Event lifecycle:**
1. Each dispatched node produces one or more `cl.Event`s (one per tile for `KernelDispatchNode`).
2. Downstream nodes collect dependency events via `depends_on` lookup in the event map.
3. `BarrierNode`s produce a marker event joining all upstream events.
4. `RetrievalNode`s produce a read event that signals D2H completion.

**Multi-tile event aggregation:** A `KernelDispatchNode` with `tile_count=N` produces N events. The renderer may either:
- (A) Pass all N events as `wait_for` to downstream nodes, or
- (B) Create a single marker/user event that joins the N tile events, and pass only that.

Option (B) is preferred to avoid quadratic `wait_for` list growth in deeply tiled plans. The renderer creates a `clEnqueueMarkerWithWaitList` after the tile loop.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| `cl.UserEvent` / marker semantics differ across OpenCL drivers | Medium | Medium | Use `clEnqueueMarkerWithWaitList` instead of `cl.UserEvent` where possible; test on multiple drivers in CI |
| Buffer allocation exceeds device `GLOBAL_MEM_SIZE` for large models | Low | High | Allocator checks total allocation against `HardwareProfile.global_mem_bytes` before allocating; raises descriptive error |
| Legacy `Services`/`BufferManager` conflicts with new allocator | Medium | Low | The two systems are independent — new renderer uses `OpenCLBufferAllocator`; legacy uses `BufferManager`. No shared mutable state. |
| Async D2H read returns before padding-stripping in `OpenCLRetrievalFuture` | Low | High | `.result()` always calls `.wait()` first; padding-stripping occurs after wait. |
| `max_reduce_fan_in` derivation is too conservative on some devices | Medium | Low | Plan builder can use a fan-in smaller than the maximum; the derivation provides an upper bound. Runtime heuristics in Phase 2B may refine. |
