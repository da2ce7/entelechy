# Phase 1: Plan Model — Detailed Plan

**Status: ✅ IMPLEMENTED** — 140 Tier 1 tests pass (332 total with Phase 0). Rollback gate validated.  
**Phase:** 1 of 6  
**Objective:** Implement the shared orchestration layer's plan data structures — the backend-neutral execution plan model — and validate them with Tier 1 tests. The plan model is usable but not yet consumed by any backend renderer.  
**Governing ADRs:** ADR-002 (plan node types), ADR-003 (reduction tree plan), ADR-004 (streaming loop plan), ADR-006 (HardwareProfile finalization), ADR-007 (KernelContract/KernelBinding split), ADR-008 (PrecisionConfig finalization), ADR-009 (buffer lifecycle), ADR-010 (RetrievalFuture Protocol), ADR-011 (CCE/BCE strategy delegation cleanup)  
**Rollback gate:** Tier 1 green — all plan construction, contract validation, buffer lifecycle, reduction tree plan, streaming loop plan, strategy delegation, memory layout, and precision config tests pass.  
**Dependencies:** Phase 0 (directory structure and module relocation). Phase 0 is **COMPLETE** (192 tests green).

### Implementation Notes

- **ModelSpec composition:** `ModelSpec` switched from `PrecisionContext` ABC inheritance to `PrecisionConfig` composition. Backward-compatible properties (`SCALAR_NP_TYPE`, `SCALAR_C_TYPE_NAME`) delegate to `self.precision`. Factory classmethods `ModelSpec.float32()` and `ModelSpec.float16()` added. Deprecated module-level `Float32ModelSpec()` and `Float16ModelSpec()` factory functions preserved until Phase 6.
- **ProblemTypeStrategy:** New `PlanProblemTypeStrategy` protocol + `PlanCceStrategy`/`PlanBceStrategy` implementations added alongside legacy `CceStrategy`/`BceStrategy` classes. Legacy classes remain for the existing OpenCL backend; clean removal deferred to Phase 6.
- **HardwareProfile:** Aligned to DESIGN.md §3.4 — fields renamed to `simd_width`, `cache_line_bytes`, `max_reduce_fan_in`, `max_local_mem_bytes`, `global_mem_bytes`. Backward-compatible properties added for Phase 0 field names.
- **KernelContract hierarchy:** Full contract field sets populated across all seven per-phase modules. `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, `PlacementContract`, `KernelContractBlock` spec types introduced in `kernel_contracts/__init__.py`.
- **Plan builder:** `build_act_plan()` and `build_learn_plan()` implemented with full DAG construction: 7 nodes (Act plan), 20+ nodes (Learn plan). Both plans include buffer lifecycle annotations, reduction trees with threshold scheduling, streaming loops with parametric strides, barriers, and retrieval nodes.
- **RetrievalFuture:** Implemented as a `@runtime_checkable` Protocol with `node_id` property, `wait()`, `result()`, and `release()` methods.
- **Test count:** 140 Tier 1 tests across 11 test modules covering all Phase 1 deliverables.

### Phase 0 Deferred Items Resolved Here

Phase 0 left several items explicitly deferred to Phase 1:

| Deferred Item | Phase 0 State | Phase 1 Resolution |
| :--- | :--- | :--- |
| `arch_primitives.py` → `PrecisionConfig` composition | `arch_primitives.py` remains at `src/` root; `model_spec.py` inherits from `PrecisionContext` ABC | Step 1.3: `ModelSpec` switches from inheritance (`PrecisionContext` ABC) to composition (`PrecisionConfig` frozen dataclass) |
| `ProblemTypeStrategy` references `KernelSignature` types | Copy+shim approach — strategies import OpenCL `KernelSignature` types | Step 1.7: Strategies produce `KernelContract` instances; OpenCL `KernelSignature` imports removed |
| `KernelContract` stubs → full field sets | Stub `KernelContract` base class with only `kernel_name: str` | Step 1.6: Full contract field sets populated per ADR-007 |
| `HardwareProfile` fields diverge from DESIGN.md §3.4 | Phase 0 stub has `max_work_group_size`, `max_alloc_bytes`; DESIGN.md §3.4 specifies `max_reduce_fan_in`, `max_local_mem_bytes` | Step 1.2: HardwareProfile aligned to DESIGN.md §3.4 specification |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Current State After Phase 0](#2-current-state-after-phase-0)
3. [Target Deliverables](#3-target-deliverables)
4. [Task Breakdown](#4-task-breakdown)
   - [Step 1.1: Implement plan node types → `src/shared/plan_types.py`](#step-11-implement-plan-node-types--srcsharedplan_typespy)
   - [Step 1.2: Finalize `HardwareProfile` → `src/shared/hardware_profile.py`](#step-12-finalize-hardwareprofile--srcsharedhardware_profilepy)
   - [Step 1.3: Finalize `PrecisionConfig` and `ModelSpec` composition](#step-13-finalize-precisionconfig-and-modelspec-composition)
   - [Step 1.4: Implement buffer lifecycle → `src/shared/buffer_lifecycle.py`](#step-14-implement-buffer-lifecycle--srcsharedbuffer_lifecyclepy)
   - [Step 1.5: Implement `ReductionTreePlan` → `src/shared/reduction_tree_plan.py`](#step-15-implement-reductiontreeplan--srcsharedreduction_tree_planpy)
   - [Step 1.6: Implement `StreamingLoopPlan` → `src/shared/streaming_loop_plan.py`](#step-16-implement-streamingloopplan--srcsharedstreaming_loop_planpy)
   - [Step 1.7: Implement `RetrievalFuture` Protocol → `src/shared/retrieval_future.py`](#step-17-implement-retrievalfuture-protocol--srcsharedretrieval_futurepy)
   - [Step 1.8: Populate `KernelContract` field sets → `src/shared/kernel_contracts/`](#step-18-populate-kernelcontract-field-sets--srcsharedkernel_contracts)
   - [Step 1.9: Clean `ProblemTypeStrategy` — remove OpenCL coupling](#step-19-clean-problemtypestrategy--remove-opencl-coupling)
   - [Step 1.10: Implement plan builder → `src/shared/plan_builder.py`](#step-110-implement-plan-builder--srcsharedplan_builderpy)
   - [Step 1.11: Update `src/shared/__init__.py` exports](#step-111-update-srcshared__init__py-exports)
   - [Step 1.12: Write Tier 1 tests](#step-112-write-tier-1-tests)
   - [Step 1.13: Validate rollback gate](#step-113-validate-rollback-gate)
5. [New Module Inventory](#5-new-module-inventory)
6. [Data Type Reference](#6-data-type-reference)
7. [Plan Builder Design](#7-plan-builder-design)
8. [Tier 1 Test Specification](#8-tier-1-test-specification)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

- Implementing the five plan node types as frozen dataclasses (ADR-002): `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode`, `BarrierNode`, `RetrievalNode`.
- Implementing `BufferHandle`, `BufferRole`, `BufferDescriptor` (ADR-009).
- Implementing `ReductionTreePlan` with pre-computed threshold schedule (ADR-003).
- Implementing `StreamingLoopPlan` with stride-based parametric specification (ADR-004).
- Implementing `RetrievalFuture` Protocol (ADR-010).
- Populating `KernelContract` frozen dataclasses with full field sets (ADR-007): abstract parameter manifests, buffer shape expectations, placement strategies, local memory requirements, validation preconditions, calculability proofs.
- Cleaning the `ProblemTypeStrategy` hierarchy to produce `KernelContract` instances instead of OpenCL `KernelSignature` instances (ADR-011).
- Aligning `HardwareProfile` to DESIGN.md §3.4 (ADR-006): renaming/adding fields to match plan-construction role names.
- Replacing the `PrecisionContext` ABC inheritance in `ModelSpec` with `PrecisionConfig` composition (ADR-008).
- Implementing the plan builder — the shared-layer logic that constructs `ExecutionPlan` DAGs from `ModelSpec`, `HardwareProfile`, `PrecisionConfig`, and `StabilizationPolicy` inputs.
- Writing Tier 1 tests for all of the above.

### Out of scope

- Implementing any `PlanRenderer` or `KernelBinding` (Phases 2–5).
- Writing Tier 2 or Tier 3 tests (Phases 2–5, Phase 4).
- Backend-specific code of any kind — no OpenCL, Vulkan, or CPU imports in any Phase 1 deliverable.
- The `WorkTicket` / `LearnHandle` / `Engine` user-facing API (parallel User-Facing API workstream, Phase 4-adjacent).
- Modifying kernel source files (`kernels/*.cl.c`, `kernels/kernels.cl.h`).
- Creating `src/backends/vulkan/` or `src/backends/cpu/` directories.

### Key constraint: shared-layer purity

Every new module created in Phase 1 resides in `src/shared/` and imports **only** from:
- Python stdlib
- `numpy`
- Other `src/shared/` modules

No `src/backends/` imports are permitted. The plan model is the data-structure boundary between the Policy tier and the Orchestration tier (ADR-001). Phase 1 validates this boundary.

### Key constraint: behavioral compatibility

The existing 407 tests from Phase 0 must continue to pass. Phase 1 adds new modules and modifies shared-layer types (`HardwareProfile`, `PrecisionConfig`, `ModelSpec`, `ProblemTypeStrategy`) — these modifications must not break the existing OpenCL backend's test suite. Where a type change would break existing code, a compatibility adapter is introduced alongside the new type, and the old interface is deprecated (not removed) until Phase 6.

---

## 2. Current State After Phase 0

### Shared layer (`src/shared/`)

| Module | Status | Notes |
| :--- | :--- | :--- |
| `__init__.py` | Minimal | Empty docstring |
| `hardware_profile.py` | Stub | Fields: `cache_line_bytes`, `max_local_memory_bytes`, `max_work_group_size`, `simd_width`, `max_alloc_bytes`, `global_memory_bytes`. Diverges from DESIGN.md §3.4. |
| `precision_config.py` | Complete | `PrecisionConfig` frozen dataclass with `float32()` and `float16()` factory methods. |
| `model_spec.py` | Moved | Still inherits from `PrecisionContext` ABC via `..arch_primitives` shim. |
| `parameter_space.py` | Moved | No issues. |
| `memory_layout.py` | Moved | No issues. |
| `stabilization_policy.py` | Moved | No issues. `plan_uniform_reduction_tree()` logic is reused by the plan builder. |
| `workload_primitives.py` | Moved | `TilingScheme`, `WorkTile`, `GatherPrimitive` — consumed by the plan builder. |
| `problem_type_strategy.py` | Coupled | Imports OpenCL `KernelSignature` types; methods return `KernelSignature` instances. Temporary Plan-boundary violation. |
| `kernel_contracts/__init__.py` | Stub | `KernelContract` base class with only `kernel_name: str`. |
| `kernel_contracts/phase_*.py` | Stubs | Empty contract placeholders per phase file. |

### Not yet created (Phase 1 deliverables)

| Module | ADR |
| :--- | :--- |
| `plan_types.py` | ADR-002 |
| `buffer_lifecycle.py` | ADR-009 |
| `reduction_tree_plan.py` | ADR-003 |
| `streaming_loop_plan.py` | ADR-004 |
| `retrieval_future.py` | ADR-010 |
| `plan_builder.py` | ADR-002, ADR-003, ADR-004, ADR-009 |

---

## 3. Target Deliverables

After Phase 1 completes, the `src/shared/` directory gains the following:

```
src/shared/
├── __init__.py                     # Updated: exports plan model public API
├── plan_types.py                   # NEW: Five node types + ExecutionPlan container
├── buffer_lifecycle.py             # NEW: BufferHandle, BufferRole, BufferDescriptor
├── reduction_tree_plan.py          # NEW: ReductionTreePlan frozen dataclass
├── streaming_loop_plan.py          # NEW: StreamingLoopPlan, IterationDimension, ParameterStride, ScratchBufferSpec
├── retrieval_future.py             # NEW: RetrievalFuture Protocol
├── plan_builder.py                 # NEW: Plan construction logic
├── hardware_profile.py             # MODIFIED: aligned to DESIGN.md §3.4
├── precision_config.py             # Unchanged (already complete)
├── model_spec.py                   # MODIFIED: PrecisionConfig composition replaces PrecisionContext inheritance
├── parameter_space.py              # Unchanged
├── memory_layout.py                # Unchanged
├── stabilization_policy.py         # Unchanged (consumed by plan_builder)
├── workload_primitives.py          # Unchanged (consumed by plan_builder)
├── problem_type_strategy.py        # MODIFIED: returns KernelContract instances; no OpenCL imports
└── kernel_contracts/               # MODIFIED: full field sets populated
    ├── __init__.py                 # Updated: full KernelContract base + spec types
    ├── phase_1_act.py              # Populated
    ├── phase_2_learn_A_production.py # Populated
    ├── phase_2_learn_B_processing.py # Populated
    ├── phase_2_learn_C_reduction.py  # Populated
    ├── phase_2_learn_D_backprop.py   # Populated
    └── phase_3_update.py             # Populated
```

And the Tier 1 test directory:

```
tests/
└── tier1/                          # NEW
    ├── conftest.py                 # Tier 1 fixtures
    ├── test_plan_types.py          # Node type construction & validation
    ├── test_buffer_lifecycle.py    # BufferHandle/Descriptor lifecycle
    ├── test_reduction_tree_plan.py # ReductionTreePlan structure & thresholds
    ├── test_streaming_loop_plan.py # StreamingLoopPlan strides & iteration
    ├── test_kernel_contracts.py    # Contract field completeness & validation
    ├── test_plan_builder.py        # End-to-end plan construction
    ├── test_precision_config.py    # PrecisionConfig correctness
    ├── test_hardware_profile.py    # HardwareProfile field semantics
    ├── test_problem_type_strategy.py # Strategy → KernelContract delegation
    └── test_memory_layout.py       # SIMD-aware layout & padding
```

---

## 4. Task Breakdown

### Step 1.1: Implement plan node types → `src/shared/plan_types.py`

**Action:** Create `src/shared/plan_types.py` containing the five plan node frozen dataclasses and the `ExecutionPlan` container.

**Governing ADR:** ADR-002 (Option B — typed node hierarchy with explicit dependency edges).

**Data types to implement:**

```python
@dataclass(frozen=True)
class KernelDispatchNode:
    """A single logical kernel invocation (ADR-002)."""
    node_id: str
    depends_on: frozenset[str]
    kernel_name: str
    contract: "KernelContract"
    buffer_bindings: dict[str, "BufferHandle"]  # contract param name → BufferHandle
    scalar_params: dict[str, int | float]       # contract param name → value
    tile_count: int
    local_work_size: int | None                 # None = let backend decide
    placement_strategy: str | None              # e.g. "grid_mod_cls", "linear_batch"

@dataclass(frozen=True)
class ReductionTreeNode:
    """A multi-stage log_K(N) reduction tree, rendered atomically by the
    backend (ADR-002, ADR-003)."""
    node_id: str
    depends_on: frozenset[str]
    reduction_plan: "ReductionTreePlan"

@dataclass(frozen=True)
class StreamingLoopNode:
    """A parametric loop over a chunk-indexed body of KernelDispatchNode
    references (ADR-002, ADR-004)."""
    node_id: str
    depends_on: frozenset[str]
    streaming_plan: "StreamingLoopPlan"

@dataclass(frozen=True)
class BarrierNode:
    """A named synchronization point joining upstream edges; no dispatch
    payload (ADR-002)."""
    node_id: str
    depends_on: frozenset[str]
    barrier_name: str

@dataclass(frozen=True)
class RetrievalNode:
    """A host-accessible result extraction point (ADR-002, ADR-010)."""
    node_id: str
    depends_on: frozenset[str]
    source_buffer: "BufferHandle"
    logical_shape: tuple[int, ...]
    event_name: str
```

**`PlanNode` union type:**

```python
PlanNode = KernelDispatchNode | ReductionTreeNode | StreamingLoopNode | BarrierNode | RetrievalNode
```

**`ExecutionPlan` container:**

```python
@dataclass(frozen=True)
class ExecutionPlan:
    """An immutable, backend-neutral execution plan (ADR-002).

    The plan is a DAG of typed nodes connected by explicit dependency edges.
    Dependency edges are the concurrency specification — nodes with no
    dependency relationship may execute in parallel.
    """
    nodes: dict[str, PlanNode]                       # node_id → PlanNode
    buffers: dict["BufferHandle", "BufferDescriptor"] # handle → descriptor
    topological_order: tuple[str, ...]               # pre-computed traversal order
    precision: "PrecisionConfig"
    hardware: "HardwareProfile"
```

**Validation rules (enforced at `ExecutionPlan` construction):**

1. Every `node_id` in any `depends_on` set must exist in `nodes`.
2. The dependency graph must be acyclic (verified via topological sort).
3. Every `BufferHandle` referenced in `buffer_bindings` or `source_buffer` must exist in `buffers`.
4. `topological_order` must be consistent with the dependency edges.
5. Node IDs must be unique.

**Validation:** Unit tests construct valid and invalid plans; invalid plans raise descriptive errors.

---

### Step 1.2: Finalize `HardwareProfile` → `src/shared/hardware_profile.py`

**Action:** Align `HardwareProfile` to the DESIGN.md §3.4 specification. The Phase 0 stub has fields named for hardware measurements (`max_work_group_size`, `max_alloc_bytes`). DESIGN.md specifies plan-construction-role-named fields.

**Before (Phase 0 stub):**
```python
@dataclass(frozen=True)
class HardwareProfile:
    cache_line_bytes: int
    max_local_memory_bytes: int
    max_work_group_size: int
    simd_width: int
    max_alloc_bytes: int
    global_memory_bytes: int
```

**After (Phase 1):**
```python
@dataclass(frozen=True)
class HardwareProfile:
    """Immutable snapshot of hardware capabilities (ADR-006, DESIGN.md §3.4).

    Each backend populates this from its native device queries.
    Field names describe Policy-tier consumption role, not hardware origin.
    """
    simd_width: int
    cache_line_bytes: int
    max_reduce_fan_in: int            # Backend-computed: max K for reduction trees
    max_local_mem_bytes: int | None   # None if backend has no local memory concept
    global_mem_bytes: int
```

**Field semantics:**

| Field | Plan-construction role | Source |
| :--- | :--- | :--- |
| `simd_width` | Tile size rounding, SoA layout stride | Backend device query |
| `cache_line_bytes` | Padding alignment for CACHE-type padding contracts | Backend device query |
| `max_reduce_fan_in` | Reduction batch size $K$ upper bound | Backend-computed from work-group limits, local memory, and SIMD width |
| `max_local_mem_bytes` | Local memory budget for local-reduce kernel tier selection (renderer concern, but plan builder uses it for scratch buffer sizing) | Backend device query; `None` for CPU backend |
| `global_mem_bytes` | Buffer allocation budget; chunk count planning | Backend device query |

**Key change:** `max_reduce_fan_in` replaces `max_work_group_size`. The plan builder never needs `max_work_group_size` directly — it needs the derived fan-in limit, which each backend computes from its own native constraints. This respects the Policy/Orchestration boundary: the backend (Orchestration tier) performs the hardware-specific derivation; the plan builder (Policy tier) consumes the abstract result.

**Compatibility:** Existing code that references `max_work_group_size` or `max_alloc_bytes` is in the OpenCL backend (`src/backends/opencl/`). The backend's `discovery.py` (created in Phase 2) computes `max_reduce_fan_in` from `max_work_group_size` and exposes it through the `HardwareProfile`. Phase 1 changes only the shared-layer type — the OpenCL backend adapts in Phase 2.

**Impact on existing tests:** If any existing test directly constructs a `HardwareProfile` with the old field names, it must be updated. Assess impact before proceeding.

**Validation:** `HardwareProfile(simd_width=16, cache_line_bytes=64, max_reduce_fan_in=256, max_local_mem_bytes=65536, global_mem_bytes=4294967296)` constructs successfully. The CPU case `max_local_mem_bytes=None` also constructs.

---

### Step 1.3: Finalize `PrecisionConfig` and `ModelSpec` composition

**Action:** Replace the `PrecisionContext` ABC inheritance in `ModelSpec` with `PrecisionConfig` composition.

**Before (Phase 0 state):**
```python
# src/shared/model_spec.py
from ..arch_primitives import PrecisionContext

class ModelSpec(PrecisionContext, abc.ABC):
    ...
```

The `PrecisionContext` ABC in `src/arch_primitives.py` defines `SCALAR_NP_TYPE`, `SCALAR_C_TYPE_NAME`, `FP_FORMAT_MAX`, `FP_FORMAT_EPSILON` as abstract properties, with `Float32Context` and `Float16Context` as concrete subclasses. `Float32ModelSpec` inherits from both `ModelSpec` and `Float32Context`.

**After (Phase 1):**
```python
# src/shared/model_spec.py
from .precision_config import PrecisionConfig

@dataclass(frozen=True)
class ModelSpec:
    """Backend-neutral model configuration (ADR-008)."""
    precision: PrecisionConfig
    num_modules: int
    num_classes: int
    input_features: int
    hidden_features: int
    # ... remaining model geometry fields
```

`ModelSpec` becomes a plain frozen dataclass with `precision: PrecisionConfig` as a composed field. Code that previously accessed `model_spec.SCALAR_NP_TYPE` now accesses `model_spec.precision.numpy_dtype`.

**Migration strategy for existing code:**

1. Add a `precision: PrecisionConfig` field to `ModelSpec`.
2. Provide backward-compatible properties on `ModelSpec` that delegate to `self.precision`:
   ```python
   @property
   def SCALAR_NP_TYPE(self) -> np.dtype:
       """Deprecated: Use self.precision.numpy_dtype."""
       return self.precision.numpy_dtype
   ```
3. These compatibility properties are removed in Phase 6.
4. Update `Float32ModelSpec` and `Float16ModelSpec` to be factory functions or classmethods that construct `ModelSpec` with the appropriate `PrecisionConfig`:
   ```python
   @classmethod
   def float32(cls, **kwargs) -> "ModelSpec":
       return cls(precision=PrecisionConfig.float32(), **kwargs)
   ```

**Impact on existing tests:** Tests that construct `Float32ModelSpec(...)` or `Float16ModelSpec(...)` must be updated to use the new factory pattern. The backward-compatible properties ensure tests that access `.SCALAR_NP_TYPE` etc. still work.

**Validation:** All existing tests pass; new Tier 1 tests validate `PrecisionConfig` composition behavior.

---

### Step 1.4: Implement buffer lifecycle → `src/shared/buffer_lifecycle.py`

**Action:** Create `src/shared/buffer_lifecycle.py` containing `BufferHandle`, `BufferRole`, and `BufferDescriptor`.

**Governing ADR:** ADR-009 (Option A — plan-prescribed lifetime intervals).

**Data types to implement:**

```python
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
    MODEL_STATE = auto()       # Persists across batches (weights, biases, optimizer state)
    BATCH_INPUT = auto()       # Uploaded per batch (training data, targets)
    BATCH_INTERMEDIATE = auto() # Temporary computation product (partials, activations)
    BATCH_OUTPUT = auto()      # Returned to the host (final probs, loss)


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
    producing_node: str | None        # None for MODEL_STATE and BATCH_INPUT (externally populated)
    consumers: frozenset[str]
    last_consumer: str | None         # None if no consumers (dead buffer — plan validation error)
```

**Plan-time validation invariants (enforced by plan builder):**

1. **Single-producer:** Every `BATCH_INTERMEDIATE` buffer has exactly one `producing_node`.
2. **Coverage:** Every buffer consumed by a node either has a `producing_node` or is `MODEL_STATE`/`BATCH_INPUT`.
3. **Shape consistency:** Producer output shape matches every consumer's expected input shape (via `KernelContract` tensor shapes).
4. **Acyclic buffer flows:** No circular producer→consumer chains (guaranteed by DAG acyclicity).
5. **No dead intermediates:** Every `BATCH_INTERMEDIATE` buffer has at least one consumer.

**Two-tier buffer scope boundary (ADR-003, ADR-004, ADR-009):**

| Scope | Described by | Example |
| :--- | :--- | :--- |
| **Plan-level** | `BufferDescriptor` in `ExecutionPlan.buffers` | Hidden activations, partial gradients, final probs |
| **Renderer-internal** | Not in the plan; allocated by the backend during rendering | Reduction tree intermediate ping-pong buffers, streaming loop scratch buffers |

`ReductionTreeNode` intermediate buffers and `StreamingLoopNode` scratch buffers (`ScratchBufferSpec`) are renderer-internal. The plan declares their size requirements (so the renderer knows how much to allocate) but does not assign `BufferHandle`s to them.

**Validation:** Construct `BufferDescriptor` instances; verify frozen immutability; verify `BufferRole` enum values.

---

### Step 1.5: Implement `ReductionTreePlan` → `src/shared/reduction_tree_plan.py`

**Action:** Create `src/shared/reduction_tree_plan.py` containing `ReductionTreePlan`.

**Governing ADR:** ADR-003 (Option C — parametric header with pre-computed threshold schedule).

**Data type:**

```python
from dataclasses import dataclass
from typing import Literal

from .buffer_lifecycle import BufferHandle


@dataclass(frozen=True)
class ReductionTreePlan:
    """Parametric descriptor for a multi-stage log_K(N) reduction tree (ADR-003).

    Pre-computed by the Policy tier. Consumed atomically by the backend
    renderer, which selects kernel tiers (register-reduce vs. local-reduce)
    per stage and manages intermediate buffers.
    """
    num_partials: int                                  # N (total input partials)
    fan_in: int                                      # K (uniform reduction batch size)
    num_stages: int                                    # ceil(log_K(N))
    elements_per_partial: int                          # Scalar count per partial result
    initial_offset_list: tuple[int, ...]               # Placement-dependent, non-trivial
    tree_variant: Literal["sum", "sum_and_clip"]       # Diagnostic (sum-only) or stabilized
    threshold_schedule: tuple[float | None, ...]       # Per-stage T_j (leaf→root), None for sum-only
    partial_width: int                                 # Width of each partial result element
    source_buffer: BufferHandle
    destination_buffer: BufferHandle
```

**Threshold schedule computation:**

The plan builder computes the threshold schedule using `StabilizationPolicy`:

For `tree_variant == "sum_and_clip"`:
- Stage $j$ (where $j = \text{num\_stages} - 1$ is the leaf, $j = 0$ is the root):
  - Policy threshold: $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$
  - Safety ceiling: $T_{\text{safety},j} = \text{FP\_FORMAT\_MAX} / K$
  - Final threshold: $T_j = \min(T_{\text{policy}}, T_{\text{safety}})$
  - Clamped by minimum: $T_j = \max(T_j, T_{\text{min}})$
- `threshold_schedule` is a tuple of length `num_stages`, ordered leaf→root.

For `tree_variant == "sum"`:
- `threshold_schedule` is a tuple of `None` values of length `num_stages`.

**Plan-time validation invariants:**

1. `fan_in >= 2`.
2. $K^{\text{num\_stages}} \geq N$ (sufficient stages to reduce all partials).
3. `len(threshold_schedule) == num_stages`.
4. `len(initial_offset_list) == num_partials`.
5. For `"sum_and_clip"`: threshold monotonicity $T_{j+1} \geq T_j$ for $\lambda \geq 0$.
6. For `"sum_and_clip"`: safety bound $T_j \leq \text{FP\_FORMAT\_MAX} / K$ at every stage.
7. `source_buffer` and `destination_buffer` are valid `BufferHandle`s in the plan's buffer namespace.

**Renderer contract (not implemented here — documented for Phase 2+):**

The renderer receives the `ReductionTreePlan` and:
1. Creates intermediate ping-pong buffers (renderer-internal, not plan-level).
2. Uploads `initial_offset_list` to device memory (for stage 0).
3. For each stage, selects kernel tier (register-reduce vs. local-reduce) based on `fan_in` and hardware-specific crossover heuristic.
4. Computes contiguous intermediate offset lists for stages > 0.
5. Executes stages leaf→root, synchronizing between stages.

**Validation:** Construct valid and invalid `ReductionTreePlan` instances; verify threshold schedule computation against `StabilizationPolicy`.

---

### Step 1.6: Implement `StreamingLoopPlan` → `src/shared/streaming_loop_plan.py`

**Action:** Create `src/shared/streaming_loop_plan.py` containing `StreamingLoopPlan`, `IterationDimension`, `ParameterStride`, and `ScratchBufferSpec`.

**Governing ADR:** ADR-004 (Option B — stride-based parametric specification).

**Data types:**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class IterationDimension:
    """Describes the dimension over which the loop iterates (ADR-004)."""
    total_extent: int       # Total size of the iterated dimension
    chunk_count: int        # Number of chunks
    chunk_size: int         # Items per chunk (uniform; last chunk may be smaller)


@dataclass(frozen=True)
class ParameterStride:
    """Describes how a single scalar parameter varies across iterations (ADR-004)."""
    param_name: str         # The parameter's name in the body's KernelDispatchNodes
    base: int               # Value at chunk index 0
    stride: int             # Additive delta per chunk index
    # Per-chunk value: base + chunk_index * stride


@dataclass(frozen=True)
class ScratchBufferSpec:
    """Declares a renderer-internal scratch buffer used within the loop body (ADR-004).

    The renderer allocates this buffer once and reuses it across iterations.
    These are NOT plan-level buffers (ADR-009 two-tier scope).
    """
    logical_name: str       # Name referenced by body nodes' buffer_bindings
    size_bytes: int         # Required size per chunk iteration
    shape: tuple[int, ...]  # Logical tensor shape


@dataclass(frozen=True)
class StreamingLoopPlan:
    """Stride-based parametric specification for a streaming loop (ADR-004).

    The body is a flat sequence of KernelDispatchNode node_ids. The renderer
    instantiates per-chunk parameter values from base + index * stride.
    """
    iteration: IterationDimension
    body: tuple[str, ...]                           # Ordered KernelDispatchNode node_ids
    parameter_strides: tuple[ParameterStride, ...]
    scratch_buffers: tuple[ScratchBufferSpec, ...]
    constant_scalars: dict[str, float]              # Scalars constant across all iterations
```

**Constraints from ADR-002 (Complexity Ceiling):**

- `body` contains only `KernelDispatchNode` references — no nested `StreamingLoopNode`, `ReductionTreeNode`, `BarrierNode`, or `RetrievalNode`.
- This constraint is enforced at `ExecutionPlan` construction time: nodes referenced by `StreamingLoopPlan.body` are verified to be `KernelDispatchNode` instances.

**Plan-time validation invariants:**

1. `iteration.chunk_count > 0`.
2. `iteration.chunk_size > 0`.
3. `iteration.chunk_size * iteration.chunk_count >= iteration.total_extent`.
4. `body` is non-empty.
5. All `body` node_ids reference `KernelDispatchNode` instances in the plan.
6. All `ParameterStride.param_name` values reference parameters that exist in the body nodes' `scalar_params`.
7. `scratch_buffers` have unique `logical_name` values.

**Validation:** Construct valid and invalid `StreamingLoopPlan` instances; verify stride arithmetic for edge cases (last chunk smaller than `chunk_size`).

---

### Step 1.7: Implement `RetrievalFuture` Protocol → `src/shared/retrieval_future.py`

**Action:** Create `src/shared/retrieval_future.py` containing the `RetrievalFuture` Protocol.

**Governing ADR:** ADR-010 (Option A — `RetrievalFuture` Protocol with padding-aware numpy result).

**Type definition:**

```python
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RetrievalFuture(Protocol):
    """Protocol for host-side observation of device results (ADR-010).

    Each backend implements this protocol for its native completion
    signaling mechanism. The renderer owns padding-stripping — .result()
    returns the unpadded numpy array with logical shape.
    """

    @property
    def node_id(self) -> str:
        """The RetrievalNode's node_id that produced this future."""
        ...

    def wait(self) -> None:
        """Block until the data is host-accessible. Idempotent."""
        ...

    def result(self) -> NDArray[np.floating]:
        """Return the retrieved data as an unpadded numpy array.

        Implicitly calls .wait() if the transfer has not completed.
        The renderer applies padding-stripping before storing the result.
        """
        ...

    def release(self) -> None:
        """Signal that the host has finished consuming the result.

        The renderer may reclaim physical memory backing this result
        after release() is called.
        """
        ...
```

**Backend implementation notes (documented here, implemented in Phases 2–5):**

| Backend | `.wait()` | `.result()` | `.release()` |
| :--- | :--- | :--- | :--- |
| OpenCL | `event.wait()` | Numpy slice of host buffer | Mark host buffer reclaimable |
| Vulkan | `vkWaitForFences(fence)` | Numpy from mapped staging pointer | Mark staging buffer reclaimable |
| CPU | No-op (synchronous) | Numpy view over compute buffer | Decrement allocation reference |

**Validation:** `runtime_checkable` allows `isinstance` checks. Tier 1 tests create a mock implementation and verify the protocol contract.

---

### Step 1.8: Populate `KernelContract` field sets → `src/shared/kernel_contracts/`

**Action:** Expand the `KernelContract` base class and populate per-phase contract frozen dataclasses with full field sets per ADR-007.

**Governing ADR:** ADR-007 (Option A — extract shared validation into `KernelContract`; abstract placement keys).

**`KernelContract` base class expansion:**

```python
@dataclass(frozen=True)
class BufferParamSpec:
    """Specification for a single buffer parameter in a kernel contract."""
    name: str                                  # CONTRACT.md Article 2.1 grammar name
    flow: Literal["src", "dest", "update", "sync"]
    memory_scope: Literal["GLOBAL", "LOCAL", "GLOBAL_CONST", "DEVICE_CONST"]
    tensor_shape: tuple[str, ...]              # Symbolic dimension expressions
    padding_contract: "PaddingContract"
    calculability_proof: tuple[str, ...]        # Parameter names used in proof
    validation_preconditions: tuple[str, ...]   # Constraint expressions

@dataclass(frozen=True)
class ScalarParamSpec:
    """Specification for a single scalar parameter in a kernel contract."""
    name: str                                  # CONTRACT.md Article 2.2 grammar name
    flow: Literal["src", "dest"]
    number_type: Literal["NATURAL", "INTEGER", "REAL", "FLAG"]

@dataclass(frozen=True)
class LocalMemorySpec:
    """Specification for a local memory requirement."""
    name: str
    size_expr: str                             # Symbolic expression over scalar params

@dataclass(frozen=True)
class PlacementContract:
    """Placement strategy specification (CONTRACT.md Article 3.2)."""
    strategy: str                              # "grid_mod_cls", "linear_batch", "linear_generic"
    key_domain: tuple[int, int] | None         # (min, max) of the placement key
    context_params: dict[str, str]             # Additional params needed for placement

@dataclass(frozen=True)
class PaddingContract:
    """Padding specification (CONTRACT.md Article 3.1)."""
    padding_type: Literal["CACHE", "BANK_CONFLICT_AVOIDANCE", "SIMD", "NONE"]
    formula: str | None                        # Human-readable formula, None for NONE

@dataclass(frozen=True)
class KernelContractBlock:
    """Kernel-level contract metadata (CONTRACT.md Article 4)."""
    holistic_constraints: str
    idempotency: Literal[
        "Strictly Idempotent",
        "Associatively Non-Idempotent",
        "Fundamentally Non-Idempotent (Stateful)"
    ]
    synchronization_model: str | None          # e.g. "Streamable", "Partial Renderer"
    behavioral_invariants: tuple[str, ...] | None

@dataclass(frozen=True)
class KernelContract:
    """Backend-neutral kernel interface contract (ADR-007).

    Carries the complete interface specification for plan-construction-time
    validation. Does not carry backend-specific dispatch details — those
    belong in each backend's KernelBinding.
    """
    kernel_name: str
    contract_block: KernelContractBlock
    buffer_params: tuple[BufferParamSpec, ...]
    scalar_params: tuple[ScalarParamSpec, ...]
    local_memory: tuple[LocalMemorySpec, ...]
    placement: PlacementContract | None
```

**Per-phase contract population:**

Each per-phase contract module defines concrete `KernelContract` instances for the kernels in that phase, populated from the `@kernel_contract` and `@param` blocks in `kernels.cl.h`. The contracts are **data** — they are frozen dataclass instances without methods, constructed from the specification in CONTRACT.md and `kernels.cl.h`.

| Module | Contracts |
| :--- | :--- |
| `phase_1_act.py` | `forward_pass_contract`, `compute_hidden_mask_contract`, `render_logits_chunk_contract` |
| `phase_2_learn_A_production.py` | `compute_probs_loss_cce_contract`, `compute_probs_loss_bce_contract`, `calculate_module_param_grads_cce_contract`, `calculate_module_param_grads_bce_contract` |
| `phase_2_learn_B_processing.py` | `backprop_error_to_hidden_cce_contract`, `backprop_error_to_hidden_bce_contract`, `calculate_chunk_temp_gradients_cce_contract`, `calculate_chunk_temp_gradients_bce_contract`, `clip_partial_gradients_contract` |
| `phase_2_learn_C_reduction.py` | `gather_and_permute_grad_h_contract`, `aggregate_register_reduce_contract`, `aggregate_local_reduce_contract`, `clip_intermediate_grad_contract`, `stabilize_reduce_grad_h_contract` |
| `phase_2_learn_D_backprop.py` | `backprop_shared_weights_contract`, `backprop_shared_biases_contract`, `clip_shared_gradients_contract` |
| `phase_3_update.py` | `normalize_gradients_contract`, `adam_update_contract`, `clamp_temperatures_contract` |

**Extraction methodology:** For each kernel, consult `kernels.cl.h` for the `@kernel_contract` block and each `@param` commentary, and translate them into the `KernelContract` frozen dataclass form. The abstract placement key model (ADR-007) replaces backend-specific tile index parameters: `flat_tile_index` is NOT in the contract; the `PlacementContract` specifies the strategy abstractly.

**Validation:** Tier 1 tests verify that each contract's fields are non-empty, placement strategies are from the canonical set, and all references in calculability proofs map to named parameters within the same contract (closed system — CONTRACT.md Article 1.4.1).

---

### Step 1.9: Clean `ProblemTypeStrategy` — remove OpenCL coupling

**Action:** Modify `src/shared/problem_type_strategy.py` to remove all OpenCL backend imports. Strategy methods return `KernelContract` identifiers or `str` kernel names instead of `KernelSignature` instances.

**Before (Phase 0):**
```python
from ..backends.opencl.launcher_infra import BufferHandle, KernelSignature
from ..backends.opencl.kernel_bindings import (
    ComputeProbsLossCceChunkSignature, ...
)

class ProblemTypeStrategy(abc.ABC):
    @abc.abstractmethod
    def get_loss_signature(self, **kwargs) -> KernelSignature: ...
```

**After (Phase 1):**
```python
from .kernel_contracts import KernelContract
from .kernel_contracts.phase_2_learn_A_production import (
    compute_probs_loss_cce_contract,
    compute_probs_loss_bce_contract,
    ...
)

class ProblemTypeStrategy(abc.ABC):
    @property
    @abc.abstractmethod
    def required_targets_buffer_name(self) -> str: ...

    @abc.abstractmethod
    def get_loss_contract(self) -> KernelContract: ...

    @abc.abstractmethod
    def get_module_grad_contract(self) -> KernelContract: ...

    @abc.abstractmethod
    def get_hidden_grad_contract(self) -> KernelContract: ...

    @abc.abstractmethod
    def get_temp_grad_contract(self) -> KernelContract: ...

class CceStrategy(ProblemTypeStrategy):
    def get_loss_contract(self) -> KernelContract:
        return compute_probs_loss_cce_contract  # Module-level constant

class BceStrategy(ProblemTypeStrategy):
    def get_loss_contract(self) -> KernelContract:
        return compute_probs_loss_bce_contract
```

**Compatibility:** The existing OpenCL backend code that calls `strategy.get_loss_signature(...)` is in `src/backends/opencl/`. Since the existing tests exercise the legacy path through `batch_processor.py`, a compatibility adapter must be provided:

1. The old `ProblemTypeStrategy` interface (returning `KernelSignature`) is preserved as `LegacyProblemTypeStrategy` in `src/backends/opencl/` — a thin wrapper that maps the new contract-returning interface to `KernelSignature` construction. This wrapper is used by the legacy `batch_processor.py` during the transition.
2. Alternatively: the existing `problem_type_strategy.py` is left as-is with OpenCL imports (the Phase 0 state) and a *new* `ProblemTypeStrategy` with the clean contract-returning interface is introduced alongside it. The old version is deprecated and Phase 6 removes it.

**Decision:** Option 2 — introduce the clean `ProblemTypeStrategy` as a new class (`PlanProblemTypeStrategy`) in `src/shared/problem_type_strategy.py` alongside the legacy `ProblemTypeStrategy` (which retains its OpenCL imports). This avoids breaking the existing 407 tests. The legacy class is removed in Phase 6.

**Validation:** `from src.shared.problem_type_strategy import PlanCceStrategy, PlanBceStrategy` succeeds without any OpenCL imports in the import chain.

---

### Step 1.10: Implement plan builder → `src/shared/plan_builder.py`

**Action:** Create `src/shared/plan_builder.py` containing the plan construction logic that assembles `ExecutionPlan` DAGs from model configuration, hardware profile, and stabilization parameters.

**This is the most complex deliverable in Phase 1.** The plan builder translates the CONCEPT.md DAG into a concrete `ExecutionPlan` composed of the five node types.

**Plan builder inputs:**

| Input | Type | Source |
| :--- | :--- | :--- |
| Model geometry | `ModelSpec` | User configuration |
| Hardware capabilities | `HardwareProfile` | Backend discovery |
| Precision config | `PrecisionConfig` | User configuration / `ModelSpec.precision` |
| Stabilization policy | `StabilizationPolicy` | User configuration |
| Problem type strategy | `PlanProblemTypeStrategy` (CCE or BCE) | User configuration |
| Batch size | `int` | Per-invocation |
| Activation lifecycle | `Literal["cache", "recompute"]` | User configuration / plan builder heuristic |

**Plan builder outputs:**

| Output | Type | Description |
| :--- | :--- | :--- |
| Act plan | `ExecutionPlan` | Forward pass through aggregate probs → inference retrieval |
| Learn plan | `ExecutionPlan` | Gradient production → reduction → update → final retrieval |

**High-level construction logic (Act plan):**

1. Allocate `BufferHandle`s for MODEL_STATE buffers (shared weights, module weights/biases, temperatures, Adam state).
2. Allocate handles for BATCH_INPUT buffers (input data, sample mask, targets).
3. Create `KernelDispatchNode` for `forward_pass` (Node 4) with tile decomposition from `TilingScheme`.
4. Create `KernelDispatchNode` for `render_logits_chunk` (Node 5).
5. Create `KernelDispatchNode`(s) for loss computation (Nodes 6/7) — CCE or BCE, selected by strategy.
6. Create `ReductionTreeNode` for diagnostic aggregation (Node 14) — probs and optionally BCE loss. `tree_variant = "sum"`.
7. Create `RetrievalNode` for `inference_retrieval` — source buffer is aggregated final probs.
8. Compute buffer lifecycle annotations (`BufferDescriptor`) for all allocated handles.
9. Validate and assemble `ExecutionPlan`.

**High-level construction logic (Learn plan):**

1. Reuse MODEL_STATE buffer handles (persistent across plans).
2. Allocate handles for BATCH_INPUT buffers (input data, sample mask, targets — same as Act).
3. **Phase I (Hierarchical Gradient Generation):**
   - Create `KernelDispatchNode`s for Nodes 8, 9, 10 (module/hidden/temp gradient computation) — per-tile parallel.
   - Create `KernelDispatchNode` for Node 11 (clip partial gradients).
   - If `activation_lifecycle == "recompute"`: wrap Phase I in a `StreamingLoopNode` (Model A recompute path).
4. Create `BarrierNode` for Item Synchronization Point (Node 13 — `gather_and_permute_grad_h` gates this).
5. Create `KernelDispatchNode` for `gather_and_permute_grad_h` (Node 13).
6. **Phase II (Specialized & Collective Aggregation):**
   - Create `KernelDispatchNode` for `stabilize_reduce_grad_h` (Node 16) — single dispatch with internal multi-stage reduction.
   - Create `ReductionTreeNode` for Module+Temp gradient reduction (Node 15) — `tree_variant = "sum_and_clip"`.
7. **Phase III (Streaming Backprop):**
   - Create `StreamingLoopNode` containing Nodes 17→18→19 (backprop shared weights/biases + clip).
8. **Phase IV (Final Aggregation):**
   - Create `ReductionTreeNode` for Shared gradient reduction (Node 20) — `tree_variant = "sum_and_clip"`.
   - Create `KernelDispatchNode` for `normalize_gradients` (Node 21).
9. Create `BarrierNode` for Batch Synchronization Point (Node 22).
10. **Phase V (Parameter Update):**
    - Create `KernelDispatchNode`s for `adam_update` (Node 24) — one per parameter group (shared, module, temps).
    - Create `KernelDispatchNode` for `clamp_temperatures` (Node 25).
11. Create `RetrievalNode` for `final_batch_retrieval`.
12. Compute buffer lifecycle annotations.
13. Validate and assemble `ExecutionPlan`.

**Internal dependencies consumed from existing shared modules:**

| Module | Consumed API |
| :--- | :--- |
| `stabilization_policy.py` | `StabilizationPolicy.plan_uniform_reduction_tree()`, threshold computation methods |
| `workload_primitives.py` | `TilingScheme`, `WorkTile`, `GatherPrimitive` |
| `memory_layout.py` | SIMD padding calculations, `PaddingStrategy` |
| `model_spec.py` | Model geometry fields |
| `problem_type_strategy.py` | `PlanProblemTypeStrategy` for CCE/BCE kernel selection |
| `kernel_contracts/` | Per-kernel `KernelContract` instances for node construction |

**Buffer handle allocation strategy:**

The plan builder maintains a monotonically increasing counter for `BufferHandle` allocation within each plan. Handles start at 0 and increment. Two plans have independent handle namespaces.

**Validation:** `build_act_plan(model_spec, hw, precision, policy, strategy, batch_size)` returns a structurally valid `ExecutionPlan`. Node count, dependency edges, and buffer descriptors match expected values for a given model geometry and batch size.

---

### Step 1.11: Update `src/shared/__init__.py` exports

**Action:** Update `src/shared/__init__.py` to export the plan model's public API.

**Exported symbols:**

```python
# Plan node types
from .plan_types import (
    KernelDispatchNode,
    ReductionTreeNode,
    StreamingLoopNode,
    BarrierNode,
    RetrievalNode,
    PlanNode,
    ExecutionPlan,
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

# Configuration types
from .hardware_profile import HardwareProfile
from .precision_config import PrecisionConfig

# Plan builder
from .plan_builder import build_act_plan, build_learn_plan
```

**Validation:** `from src.shared import ExecutionPlan, KernelDispatchNode, BufferHandle` etc. all succeed.

---

### Step 1.12: Write Tier 1 tests

**Action:** Create `tests/tier1/` directory with Tier 1 test modules.

**Governing ADR:** ADR-016 (Tier 1 — host-side plan correctness; always runs; no backend required).

See [§8 Tier 1 Test Specification](#8-tier-1-test-specification) for the complete test inventory.

**Test infrastructure:**

- `tests/tier1/conftest.py` — shared fixtures: factory functions for `ModelSpec`, `HardwareProfile`, `PrecisionConfig`, `StabilizationPolicy` with sensible defaults; a standard `model_spec` fixture for canonical test geometry.
- All Tier 1 tests are pure Python — no device, no OpenCL, no ctypes. They run everywhere.
- Tests use `pytest.mark.tier1` marker for selective execution.

**Validation:** `pytest tests/tier1/ -v` — all Tier 1 tests pass.

---

### Step 1.13: Validate rollback gate

**Action:** Run the complete test suite (existing + new Tier 1) and confirm green.

**Procedure:**

1. `cd architectures/averaging_ensembled_classifier`
2. `pytest tests/tier1/ -v` — all new Tier 1 tests pass.
3. `pytest src/tests/ tests/ -v` — all existing tests (407) + new Tier 1 tests pass.
4. `python -c "from src.shared import ExecutionPlan, KernelDispatchNode, BufferHandle, ReductionTreePlan"` — imports work.
5. `meson setup builddir --wipe && meson compile -C builddir` — build succeeds.

**Gate criteria (ADR-017):**

- Tier 1 green: all plan construction, contract validation, buffer lifecycle, reduction tree plan, streaming loop plan, strategy delegation, memory layout, and precision config tests pass.
- All existing tests remain green (no regressions from Phase 0).

---

## 5. New Module Inventory

| Module | ADR | Content | Lines (est.) |
| :--- | :--- | :--- | :--- |
| `src/shared/plan_types.py` | ADR-002 | Five node types + `ExecutionPlan` container + validation | 200–300 |
| `src/shared/buffer_lifecycle.py` | ADR-009 | `BufferHandle`, `BufferRole`, `BufferDescriptor` | 60–80 |
| `src/shared/reduction_tree_plan.py` | ADR-003 | `ReductionTreePlan` frozen dataclass | 40–60 |
| `src/shared/streaming_loop_plan.py` | ADR-004 | `StreamingLoopPlan`, `IterationDimension`, `ParameterStride`, `ScratchBufferSpec` | 60–80 |
| `src/shared/retrieval_future.py` | ADR-010 | `RetrievalFuture` Protocol | 30–40 |
| `src/shared/plan_builder.py` | — | Plan construction logic (Act + Learn plans) | 500–800 |
| `src/shared/kernel_contracts/__init__.py` | ADR-007 | Expanded: `KernelContract` + spec types | 100–150 |
| `src/shared/kernel_contracts/phase_*.py` | ADR-007 | Full contract instances (6 modules) | 150–300 each |
| `tests/tier1/*.py` | ADR-016 | Tier 1 test suite (10+ modules) | 100–200 each |

---

## 6. Data Type Reference

Complete type dependency graph for the plan model:

```
PrecisionConfig ─────────────────────────────────────────┐
HardwareProfile ─────────────────────────────────────────┤
                                                         │
BufferHandle (int) ───────────┐                          │
BufferRole (Enum) ────────────┤                          │
PaddingContract ──────────────┤                          │
BufferParamSpec ──────────────┤                          │
ScalarParamSpec ──────────────┤                          │
LocalMemorySpec ──────────────┤                          │
PlacementContract ────────────┤                          │
KernelContractBlock ──────────┤                          │
KernelContract ───────────────┤                          │
                              │                          │
IterationDimension ───────────┤                          │
ParameterStride ──────────────┤                          │
ScratchBufferSpec ────────────┤                          │
                              │                          │
BufferDescriptor ─────────────┤                          │
ReductionTreePlan ────────────┤                          │
StreamingLoopPlan ────────────┤                          │
                              │                          │
KernelDispatchNode ───────┐   │                          │
ReductionTreeNode ────────┤   │                          │
StreamingLoopNode ────────┤   │                          │
BarrierNode ──────────────┤   │                          │
RetrievalNode ────────────┤   │                          │
                          │   │                          │
PlanNode (Union) ─────────┤   │                          │
                          │   │                          │
ExecutionPlan ────────────┴───┴──────────────────────────┘
  .nodes: dict[str, PlanNode]
  .buffers: dict[BufferHandle, BufferDescriptor]
  .topological_order: tuple[str, ...]
  .precision: PrecisionConfig
  .hardware: HardwareProfile
```

All types are frozen dataclasses (immutable). The plan model is a pure data structure — no methods, no backend types, no executable objects beyond the `RetrievalFuture` Protocol (which is only a structural type specification, not an implementation).

---

## 7. Plan Builder Design

### 7.1 Public API

```python
def build_act_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
) -> ExecutionPlan:
    """Construct an Act-phase (forward pass + inference retrieval) plan."""

def build_learn_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
    activation_lifecycle: Literal["cache", "recompute"] = "recompute",
) -> ExecutionPlan:
    """Construct a Learn-phase (gradient production → update) plan."""
```

### 7.2 Internal Structure

The plan builder is organized as a sequence of sub-builders, each responsible for a DAG segment:

| Sub-builder | DAG Segment | Nodes Created |
| :--- | :--- | :--- |
| `_build_forward_pass` | Act: forward pass + logits | Nodes 4, 5 |
| `_build_loss_computation` | Act: loss + probs | Nodes 6/7 |
| `_build_diagnostic_aggregation` | Act: reduction tree for probs | Node 14 (ReductionTreeNode) |
| `_build_inference_retrieval` | Act: host retrieval | RetrievalNode |
| `_build_gradient_production` | Learn Phase I: per-tile grads + clip | Nodes 8, 9, 10, 11 |
| `_build_recompute_loop` | Learn Phase I: optional StreamingLoopNode | StreamingLoopNode (Model A) |
| `_build_item_sync` | Learn Phase I→II gate | Node 13 + BarrierNode |
| `_build_specialized_reduction` | Learn Phase II: grad_h reduction | Node 16 |
| `_build_module_temp_reduction` | Learn Phase II: module+temp reduction | Node 15 (ReductionTreeNode) |
| `_build_streaming_backprop` | Learn Phase III: shared grad streaming | StreamingLoopNode (Nodes 17→18→19) |
| `_build_shared_reduction` | Learn Phase IV: shared grad reduction | Node 20 (ReductionTreeNode) |
| `_build_normalize` | Learn Phase IV: normalization | Node 21 |
| `_build_batch_sync` | Learn Phase IV→V gate | BarrierNode (Node 22) |
| `_build_parameter_update` | Learn Phase V: Adam + clamp | Nodes 24, 25 |
| `_build_final_retrieval` | Learn Phase V: host retrieval | RetrievalNode |

### 7.3 Buffer Allocation

The plan builder uses a `_BufferAllocator` helper:

```python
class _BufferAllocator:
    """Internal helper for monotonically allocating BufferHandles."""

    def __init__(self) -> None:
        self._next_id: int = 0
        self._descriptors: dict[BufferHandle, BufferDescriptor] = {}

    def allocate(
        self,
        logical_name: str,
        padded_shape: tuple[int, ...],
        element_size_bytes: int,
        role: BufferRole,
    ) -> BufferHandle:
        """Allocate a new BufferHandle and register its descriptor."""
        ...

    def set_producer(self, handle: BufferHandle, node_id: str) -> None: ...
    def add_consumer(self, handle: BufferHandle, node_id: str) -> None: ...
    def finalize(self) -> dict[BufferHandle, BufferDescriptor]: ...
```

The `finalize()` method computes `last_consumer` for each buffer from the topological order and validates the single-producer and coverage invariants.

### 7.4 Reduction Tree Construction

The plan builder delegates to `StabilizationPolicy` for reduction tree planning:

1. Call `stabilization_policy.plan_uniform_reduction_tree(num_partials, hardware.max_reduce_fan_in)` to get `fan_in` and `num_stages`.
2. Compute the initial offset list from the upstream `GatherPrimitive` (placement-dependent scatter pattern).
3. Compute the threshold schedule using the policy's threshold methods for each stage.
4. Construct `ReductionTreePlan` and wrap in `ReductionTreeNode`.

### 7.5 Tiling Integration

The plan builder uses `workload_primitives.TilingScheme` to determine tile decomposition:

1. Construct `TilingScheme` from `ModelSpec` geometry, `HardwareProfile.simd_width`, and batch size.
2. Extract `tile_count`, per-tile parameter values, and placement strategies.
3. Use these to populate `KernelDispatchNode.tile_count`, `.scalar_params`, and `.placement_strategy`.

---

## 8. Tier 1 Test Specification

### 8.1 Test Module Inventory

| Test Module | Scope | Key Assertions |
| :--- | :--- | :--- |
| `test_plan_types.py` | Node type construction & validation | Valid construction; frozen immutability; dependency edge validation; DAG acyclicity detection; topological sort correctness |
| `test_buffer_lifecycle.py` | BufferHandle/Descriptor lifecycle | Handle uniqueness; role classification; single-producer invariant; coverage invariant; last_consumer computation |
| `test_reduction_tree_plan.py` | ReductionTreePlan structure | Fan-in safety; stage count sufficiency; threshold schedule monotonicity; safety bound clamping; initial offset list length |
| `test_streaming_loop_plan.py` | StreamingLoopPlan strides | Stride arithmetic; last-chunk handling; body node type enforcement; scratch buffer uniqueness |
| `test_kernel_contracts.py` | Contract field completeness | Every contract has non-empty `buffer_params`; placement strategies are canonical; calculability proof closure (all referenced params exist in the contract) |
| `test_plan_builder.py` | End-to-end plan construction | Act plan node count and structure; Learn plan node count and structure; correct dependency edges between phases; buffer lifecycle coverage |
| `test_precision_config.py` | PrecisionConfig correctness | `float32()` and `float16()` factory values; `fp_format_max` and `epsilon` match numpy `finfo` |
| `test_hardware_profile.py` | HardwareProfile semantics | Construction with all fields; `max_local_mem_bytes=None` for CPU; frozen immutability |
| `test_problem_type_strategy.py` | Strategy → KernelContract delegation | CCE strategy returns CCE contracts; BCE strategy returns BCE contracts; no OpenCL imports in import chain |
| `test_memory_layout.py` | SIMD-aware layout & padding | Padding calculations; cache-line alignment; SIMD width rounding |

### 8.2 Test Fixture Design

**Canonical model geometry** (used across most Tier 1 tests):

```python
@pytest.fixture
def model_spec():
    return ModelSpec(
        precision=PrecisionConfig.float32(),
        num_modules=4,
        num_classes=10,
        input_features=128,
        hidden_features=64,
        # ... remaining fields
    )

@pytest.fixture
def hardware():
    return HardwareProfile(
        simd_width=16,
        cache_line_bytes=64,
        max_reduce_fan_in=256,
        max_local_mem_bytes=65536,
        global_mem_bytes=4 * 1024**3,
    )

@pytest.fixture
def policy(model_spec):
    return StabilizationPolicy(
        t_algorithmic=1.0,
        lambda_=0.1,
        fp_format_max=model_spec.precision.fp_format_max,
    )
```

### 8.3 Key Test Scenarios

**Plan structure validation (test_plan_builder.py):**

1. **Minimal Act plan:** 1 module, 2 classes, batch=1. Verify nodes: forward_pass → render_logits → loss → reduction → retrieval. Verify dependency edges form a valid DAG.
2. **Act plan with tiling:** 4 modules, 10 classes, batch=64, SIMD=16. Verify tile counts from `TilingScheme`. Verify placement strategies on tiled nodes.
3. **Learn plan (recompute):** Verify StreamingLoopNode wraps Phase I. Verify ReductionTreeNode instances for Nodes 14, 15, 20. Verify BarrierNode instances for Item Sync and Batch Sync.
4. **Learn plan (cache):** Verify no StreamingLoopNode for Phase I (activation caching).
5. **Buffer lifetime correctness:** For every `BATCH_INTERMEDIATE` buffer in the Learn plan, verify `producing_node` exists and `last_consumer` is topologically downstream.
6. **CCE vs. BCE divergence:** Verify Act plan selects `compute_probs_loss_cce_contract` for CCE strategy and `compute_probs_loss_bce_contract` for BCE.

**Reduction tree validation (test_reduction_tree_plan.py):**

7. **Small tree:** N=4, K=2 → 2 stages. Verify threshold schedule length = 2.
8. **Threshold monotonicity:** For $\lambda > 0$, verify $T_{\text{leaf}} \geq T_{\text{root}}$.
9. **Safety clamping:** For FP16 (`fp_format_max ≈ 65504`), verify no threshold exceeds `65504 / K`.
10. **Diagnostic tree:** `tree_variant="sum"` → all thresholds are `None`.
11. **Large fan-in:** K=256, N=65536 → 3 stages. Verify stage count = $\lceil \log_{256}(65536) \rceil = 3$.

**Contract completeness (test_kernel_contracts.py):**

12. **All contracts populated:** Every per-phase module exports at least one `KernelContract` instance.
13. **Calculability proof closure:** For each contract, every parameter name referenced in `calculability_proof` fields exists in the contract's `buffer_params` or `scalar_params` names.
14. **Placement strategy canonical:** Every `PlacementContract.strategy` is one of `"grid_mod_cls"`, `"linear_batch"`, `"linear_generic"`.

---

## 9. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| R1 | `HardwareProfile` field changes break existing tests | Medium | Medium | Assess existing test usage of `HardwareProfile` before modifying. If tests construct `HardwareProfile` directly, update them. Alternatively, provide both old and new constructors temporarily. |
| R2 | `ModelSpec` composition change breaks 407 existing tests | High | High | Backward-compatible properties (`.SCALAR_NP_TYPE`, `.FP_FORMAT_MAX` etc.) delegate to `self.precision`. Factory classmethods replace subclass constructors. All existing call sites continue to work. |
| R3 | `ProblemTypeStrategy` cleanup breaks legacy `batch_processor.py` | High | High | Decision: introduce clean `PlanProblemTypeStrategy` alongside legacy `ProblemTypeStrategy`. Both coexist until Phase 6. No existing interface is modified. |
| R4 | Plan builder logic is incorrect — DAG has wrong structure | Medium | High | Tier 1 tests validate plan structure against the CONCEPT.md DAG exhaustively. Small, focused test cases isolate each sub-builder. |
| R5 | `KernelContract` population is incomplete — missing parameters | Medium | Medium | Cross-reference each contract against `kernels.cl.h` `@kernel_contract` and `@param` blocks during population. Tier 1 tests check calculability proof closure. |
| R6 | Circular import between new plan modules | Low | High | All new modules depend only on other `src/shared/` modules and follow a strict layering: `buffer_lifecycle` → `reduction_tree_plan`/`streaming_loop_plan` → `plan_types` → `plan_builder`. No cycles. |
| R7 | `StabilizationPolicy` API insufficient for plan builder | Low | Medium | `StabilizationPolicy` already has `plan_uniform_reduction_tree()` and threshold computation methods. The plan builder's reduction tree construction delegates to these. If the API is insufficient, extend `StabilizationPolicy` with new pure methods — this is a shared-layer internal change. |
| R8 | Scope creep — plan builder grows beyond Phase 1 scope | Medium | Medium | The plan builder produces structurally valid plans that are NOT rendered. Rendering correctness is Phase 2+. Phase 1 validates structure only. |

---

## Appendix: Execution Order Summary

The steps have the following dependency structure:

```
Step 1.1  (plan node types)
    │
    ├──► Step 1.4  (buffer lifecycle)  ← depends on 1.1 (BufferHandle used by nodes)
    │       │
    │       ├──► Step 1.5  (ReductionTreePlan)  ← depends on 1.4 (BufferHandle)
    │       └──► Step 1.6  (StreamingLoopPlan)   ← depends on 1.4 (ScratchBufferSpec)
    │
    ├──► Step 1.7  (RetrievalFuture Protocol)   ← independent
    │
    ├──► Step 1.8  (KernelContract field sets)  ← depends on 1.1 (contracts embedded in nodes)
    │
Step 1.2  (HardwareProfile finalization)        ← independent
    │
Step 1.3  (PrecisionConfig + ModelSpec composition) ← independent
    │
Step 1.9  (ProblemTypeStrategy cleanup)         ← depends on 1.8 (uses KernelContract)
    │
    ▼
Step 1.10 (plan builder)  ← depends on 1.1–1.9 (assembles all types)
    │
    ▼
Step 1.11 (update __init__.py exports)  ← depends on all new modules existing
    │
    ▼
Step 1.12 (Tier 1 tests)  ← depends on all implementations
    │
    ▼
Step 1.13 (validate rollback gate)  ← depends on all above
```

**Parallelizable groups:**
- Steps 1.1, 1.2, 1.3, 1.7 are independent of each other and can proceed in parallel.
- Steps 1.4, 1.5, 1.6 are sequential (buffer lifecycle → plan sub-types) but independent of 1.2, 1.3, 1.7.
- Step 1.8 can begin once Step 1.1 provides the `KernelContract` spec types.
- Step 1.9 depends on Step 1.8 (contracts must exist for the strategy to return them).
- Step 1.10 (plan builder) is the convergence point requiring all preceding steps.
- Steps 1.11, 1.12, 1.13 are strictly sequential and run after all implementations.
