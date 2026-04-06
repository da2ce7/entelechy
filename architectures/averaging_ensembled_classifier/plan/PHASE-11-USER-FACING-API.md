# Phase 11: User-Facing API — Detailed Plan

**Status:** Complete (All steps implemented)  
**Phase:** 11 (Phase 4-adjacent parallel workstream)  
**Objective:** Implement the user-facing API surface — the `WorkTicket`, `LearnHandle`, and `Engine` abstractions — that expose the architecture's Event-Triggered and Sequential execution modes through a unified, ergonomic interface.  
**Governing ADRs:** ADR-018 (user-facing API — WorkTicket pattern, future-based concurrency, explicit batch submission, recompute buffer lifecycle, ephemeral data model), ADR-001 (backend abstraction boundary), ADR-002 (plan node types), ADR-009 (buffer lifecycle), ADR-010 (D2H transfer & RetrievalFuture Protocol), ADR-012 (module factoring), ADR-017 (migration path)  
**Rollback gate:** ✅ PASSED — All WorkTicket state transitions, Engine submission patterns, Act/Learn plan construction, and convenience method tests pass across CPU backend.  
**Dependencies:** Phase 1 (plan model complete), Phase 2 or Phase 3 (at least one renderer available for integration testing). Phase 11 can begin once Phase 1 is complete and proceeds in parallel with other workstreams per ADR-018 Choice 7C.

### Relationship to Other Phases

Phase 11 is positioned as a Phase 4-adjacent parallel workstream per ADR-018 Choice 7C. It consumes infrastructure from earlier phases and cross-pollinates with the test harness:

| Phase | Phase 11 interaction |
| :--- | :--- |
| Phase 0 (Foundation) | Phase 11 consumes the directory skeleton and `_build_config.py.in` template. |
| Phase 1 (Plan Model) | Phase 11 consumes plan node types, buffer lifecycle, RetrievalFuture Protocol. Plan builder is extended with Act-only and Learn-only plan construction modes. |
| Phase 2 (OpenCL Adapter) | Phase 11 integration tests exercise the ticket lifecycle against the OpenCL renderer. |
| Phase 3 (CPU Backend) | Phase 11 integration tests exercise the ticket lifecycle against the CPU renderer. |
| Phase 4 (Test Harness) | Cross-pollination: ticket lifecycle tests validate plan builder usage patterns; test fixtures provide canonical model configurations. |
| Phase 5 (Vulkan Backend) | Phase 11 integration tests exercise the ticket lifecycle against the Vulkan renderer (when available). |
| Phase 6 (Legacy Removal) | The existing `TrainingOrchestrator.train()` surface coexists until Phase 6 retires it. |

### ADR-018 Decisions Summary

The following choices are **decided** and govern this phase:

| Choice | Decision | Implication |
| :--- | :--- | :--- |
| **1: Unit of intent** | 1A: Stateful ticket (WorkTicket pattern) | User receives a mutable, stateful object whose lifecycle mirrors Act/Learn temporal split. |
| **2: Concurrency model** | 2B: Future-based (non-blocking, sync-compatible) | `submit()` and `resolve()` return immediately; blocking occurs on `get_prediction()` and `learn_handle.wait()`. |
| **3: Batch composition** | 3A: Explicit batch submission | One `submit()` produces one ticket representing one batch. Engine dispatches one plan per call. |
| **4: Buffer lifecycle** | 4A: Recompute (no cross-plan sharing) | Learn plan recomputes forward pass; Act plan buffers released on `get_prediction()`. No VRAM held during Act→Learn gap. |
| **5: Data persistence** | 5A: Ephemeral (no persistence) | Tickets are transient. No registry, no engine-side state between calls. |
| **6: Observability** | **OPEN** | Per-batch diagnostics attachment to LearnHandle is architecturally compatible but not yet decided. |
| **7: Migration positioning** | 7C: Parallel workstream | Developed behind feature flag; now enabled by default. |

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Target Deliverables](#2-target-deliverables)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 11.1: Create feature flag in `_build_config.py.in`](#step-111-create-feature-flag-in-_build_configpyin)
   - [Step 11.2: Implement ticket state machine → `src/shared/ticket.py`](#step-112-implement-ticket-state-machine--srcsharedticketpy)
   - [Step 11.3: Implement Engine class → `src/shared/engine.py`](#step-113-implement-engine-class--srcsharedenginepy)
   - [Step 11.4: Validate Act plan construction ✓](#step-114-validate-act-plan-construction-)
   - [Step 11.5: Validate Learn plan construction ✓](#step-115-validate-learn-plan-construction-)
   - [Step 11.6: Implement `train_batch()` convenience method](#step-116-implement-train_batch-convenience-method)
   - [Step 11.7: Update `src/shared/__init__.py` exports](#step-117-update-srcshared__init__py-exports)
   - [Step 11.8: Write ticket lifecycle unit tests](#step-118-write-ticket-lifecycle-unit-tests)
   - [Step 11.9: Write Engine integration tests](#step-119-write-engine-integration-tests)
   - [Step 11.10: Validate rollback gate](#step-1110-validate-rollback-gate)
4. [Data Type Reference](#4-data-type-reference)
5. [WorkTicket State Machine](#5-workticket-state-machine)
6. [Engine Architecture](#6-engine-architecture)
7. [Plan Construction Modes](#7-plan-construction-modes)
8. [Test Specification](#8-test-specification)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

- Implementing the `WorkTicket` class with the four-state lifecycle (PENDING → ACT_COMPLETE → RESOLVED → CONSUMED).
- Implementing the `LearnHandle` class wrapping the Learn plan's `RetrievalFuture`.
- Implementing the `Engine` class with `submit()`, `train_batch()` methods.
- ~~Extending the plan builder to construct Act-only and Learn-only plans~~ — **complete**: `build_act_plan()` and `build_learn_plan()` already exist.
- Implementing the mapping from ticket state to plan construction and rendering.
- Writing unit tests for ticket state transitions and integration tests for the full Engine→Ticket→Plan→Renderer flow.
- ~~Feature-flag gating the new API behind `FEATURE_TICKET_API` in `_build_config.py`.~~ — **promoted**: flag now defaults to `true`; ticket API is unconditionally available.

### Out of scope

- Implementing any `PlanRenderer` — Phases 2, 3, 5.
- Backend-specific code of any kind — no OpenCL, Vulkan, or CPU imports in any Phase 11 deliverable (shared-layer purity).
- Choice 6 (Observability) resolution — diagnostics attachment remains an open choice. This phase implements the ticket/engine infrastructure that Choice 6 would build upon.
- Multi-epoch training loops or data persistence (Choice 5A: ephemeral).
- Automatic batching or accumulation (Choice 3A: explicit batch submission).
- `asyncio` integration (Choice 2B: sync-compatible futures).
- Modifying kernel source files.
- The existing `TrainingOrchestrator.train()` API — it remains functional and is not modified. Users can choose either surface during the coexistence period.

### Key constraint: shared-layer purity

Every new module created in Phase 11 resides in `src/shared/` and imports **only** from:
- Python stdlib
- `numpy`
- Other `src/shared/` modules

No `src/backends/` imports are permitted. The Engine and WorkTicket are orchestration-tier components that consume the `PlanRenderer` Protocol (ADR-001). The renderer implementation is injected at construction time or resolved via a backend factory.

### Key constraint: non-breaking addition

The existing `TrainingOrchestrator.train()` API continues to function. Phase 11 adds a parallel entry point; it does not replace the existing surface. Both APIs coexist until Phase 6 (Legacy Removal) consolidates to the ticket-based surface.

### Key constraint: feature-flag gated — NOW PROMOTED

~~The ticket API is gated behind `FEATURE_TICKET_API` in `_build_config.py`. When the flag is `False`, the `Engine` class and related imports may raise `ImportError` or be absent. Tests for the ticket API are skipped when the flag is disabled. This follows the ADR-017 parallel-development model.~~

**Update (Phase 11 complete):** The ticket API is now unconditionally available. The feature flag defaults to `true` in `meson.options`, and `src/shared/__init__.py` imports the ticket API directly without conditional checks. The flag remains in `_build_config.py` for diagnostic purposes but is no longer used for gating imports.

---

## 2. Target Deliverables

After Phase 11 completes, the `src/shared/` directory gains:

```
src/shared/
├── ticket.py                       # NEW: WorkTicket, LearnHandle, TicketState
├── engine.py                       # NEW: Engine class
├── plan_builder.py                 # EXISTS: build_act_plan(), build_learn_plan() already implemented
├── __init__.py                     # MODIFIED: exports Engine, WorkTicket, LearnHandle
```

Additionally:
- `meson.options` gains `feature_ticket_api` option.
- `_build_config.py.in` gains `FEATURE_TICKET_API` flag.
- `tests/tier1/` gains ticket lifecycle test files.
- `tests/tier2/` gains engine integration test files (backend-specific).

---

## 3. Task Breakdown

### Step 11.1: Create feature flag in `_build_config.py.in`

**Action:** Add the `FEATURE_TICKET_API` flag to the build configuration template.

**File:** `architectures/averaging_ensembled_classifier/meson.options`

```meson
option('feature_ticket_api', type: 'boolean', value: true,
       description: 'WorkTicket / Engine user-facing API (ADR-018) — enabled by default since Phase 11 complete')
```

**File:** `architectures/averaging_ensembled_classifier/src/_build_config.py.in` (template)

```python
# Feature flags (ADR-017 parallel development)
FEATURE_TICKET_API: bool = @FEATURE_TICKET_API@
```

**Validation:** `ninja -C builddir` regenerates `_build_config.py` with the new flag.

---

### Step 11.2: Implement ticket state machine → `src/shared/ticket.py`

**Action:** Create the `WorkTicket` and `LearnHandle` classes with the four-state lifecycle.

**Module:** `src/shared/ticket.py`

**Types to implement:**

```python
from enum import Enum, auto
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .retrieval_future import RetrievalFuture
    from .engine import Engine


class InvalidTicketStateError(Exception):
    """Raised when a ticket operation is invalid for the current state."""


class TicketState(Enum):
    """WorkTicket lifecycle states (ADR-018 Choice 1A)."""
    PENDING = auto()       # Act plan dispatched, inference not yet complete
    ACT_COMPLETE = auto()  # Inference complete, prediction cached
    RESOLVED = auto()      # Learn plan dispatched
    CONSUMED = auto()      # Learn complete, ticket lifecycle terminated

class LearnHandle:
    """Handle for observing Learn phase completion (ADR-018)."""
    
    def __init__(self, future: "RetrievalFuture", ticket: "WorkTicket") -> None: ...
    def wait(self) -> None:
        """Block until Learn phase completes. Transitions ticket to CONSUMED."""
        ...
    # Future: diagnostics properties (Choice 6) would attach here.

class WorkTicket:
    """
    Stateful ticket representing a unit of user intent (ADR-018 Choice 1A).
    
    Lifecycle: PENDING → ACT_COMPLETE → RESOLVED → CONSUMED
    
    The ticket captures input data at submit time, ensuring system-enforced
    Act→Learn association even if the original array is mutated.
    """
    
    def __init__(
        self,
        x_data: NDArray[np.floating],
        act_future: "RetrievalFuture",
        engine: "Engine",
    ) -> None: ...
    
    @property
    def state(self) -> TicketState: ...
    
    def get_prediction(self) -> NDArray[np.floating]:
        """
        Block until Act phase completes and return prediction probabilities.
        
        Transitions: PENDING → ACT_COMPLETE (first call)
        Subsequent calls return cached prediction.
        
        Device buffers are released upon first call (Choice 4A: recompute).
        """
        ...
    
    def resolve(self, y_data: NDArray[np.integer]) -> LearnHandle:
        """
        Submit ground truth and dispatch Learn plan.
        
        Transitions: ACT_COMPLETE → RESOLVED
        (If called in PENDING state, implicitly waits for Act completion first.)
        
        Returns a LearnHandle for observing Learn phase completion.
        """
        ...
```

**Key implementation details:**

1. **Data capture:** `__init__` stores a *copy* of `x_data` (via `np.copy(x_data)`) to ensure Act→Learn association is preserved even if the caller mutates the original array.

2. **Future consumption:** `get_prediction()` calls `act_future.result()` to obtain the unpadded prediction matrix, then calls `act_future.release()` to free device buffers. The prediction is cached for subsequent calls.

3. **State enforcement:** Methods assert valid state transitions and raise `InvalidTicketStateError` for invalid operations (e.g., `resolve()` on a CONSUMED ticket).

4. **Engine reference:** The ticket holds a reference to its parent `Engine` to dispatch the Learn plan via `engine._dispatch_learn_plan()`.

**Validation:** Unit tests for all state transitions, invalid state errors, data capture immutability.

---

### Step 11.3: Implement Engine class → `src/shared/engine.py`

**Action:** Create the `Engine` class that serves as the user-facing entry point.

**Module:** `src/shared/engine.py`

**Type to implement:**

```python
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .model_spec import ModelSpec
    from .hardware_profile import HardwareProfile
    from .precision_config import PrecisionConfig
    from .parameter_space import ParameterSpace

class Engine:
    """
    User-facing entry point for inference and training (ADR-018).
    
    The Engine is stateless between ticket lifecycles (Choice 5A: ephemeral).
    It holds configuration (model spec, hardware profile, precision) and
    produces WorkTickets for submitted batches.
    """
    
    def __init__(
        self,
        model_spec: "ModelSpec",
        parameter_space: "ParameterSpace",
        hardware_profile: "HardwareProfile",
        *,
        precision: "PrecisionConfig | None" = None,
        backend: str = "auto",
        renderer_factory: "Callable[[str], PlanRenderer] | None" = None,
    ) -> None:
        """
        Initialize the Engine.
        
        Args:
            model_spec: Model architecture specification.
            parameter_space: Learnable parameters (weights, biases).
            hardware_profile: Target hardware characteristics.
            precision: Precision configuration (default: model_spec.precision).
            backend: Backend selector ("auto", "opencl", "cpu", "vulkan").
            renderer_factory: Optional callable to construct a PlanRenderer from
                backend name. If None, uses lazy-imported default factory.
                This parameter preserves shared-layer purity by deferring
                backend imports to the factory implementation.
        """
        ...
    
    def submit(self, x_data: NDArray[np.floating]) -> "WorkTicket":
        """
        Submit a batch for inference (Act phase).
        
        Returns immediately with a PENDING WorkTicket. The Act plan is
        dispatched to the device; call ticket.get_prediction() to block
        on completion.
        
        Args:
            x_data: Input data matrix, shape (N, D_in).
        
        Returns:
            WorkTicket in PENDING state.
        """
        ...
    
    def train_batch(
        self,
        x_data: NDArray[np.floating],
        y_data: NDArray[np.integer],
    ) -> NDArray[np.floating]:
        """
        Convenience method: submit, infer, resolve, and wait in one call.
        
        Equivalent to:
            ticket = engine.submit(x_data)
            prediction = ticket.get_prediction()
            ticket.resolve(y_data).wait()
            return prediction
        
        This is the minimal-ceremony pattern for sequential batch training.
        
        Args:
            x_data: Input data matrix, shape (N, D_in).
            y_data: Ground truth labels, shape (N,).
        
        Returns:
            Prediction probabilities, shape (N, C).
        """
        ...
    
    # Internal methods (not part of public API)
    def _dispatch_act_plan(self, x_data: NDArray[np.floating]) -> "RetrievalFuture": ...
    def _dispatch_learn_plan(
        self,
        x_data: NDArray[np.floating],
        y_data: NDArray[np.integer],
    ) -> "RetrievalFuture": ...
```

**Key implementation details:**

1. **Backend resolution:** The `backend` parameter selects a renderer factory. `"auto"` probes `_build_config.py` for enabled backends in preference order: **CPU > OpenCL > Vulkan** during development. This ordering prioritizes the CPU backend for deterministic CI reproducibility and debuggability; production deployments typically override with explicit backend selection or pass a custom `renderer_factory`.

2. **Renderer injection:** The optional `renderer_factory` parameter accepts a callable `(backend_name: str) -> PlanRenderer`. This preserves shared-layer purity: the Engine module contains no backend imports. The default factory is lazily imported from a thin adapter module that handles the `src/backends/` imports.

3. **Renderer lifecycle:** The Engine holds a single `PlanRenderer` instance for its lifetime. The renderer is created at `__init__` time via the backend factory.

4. **Plan dispatch:** `_dispatch_act_plan()` calls `build_act_plan()`, then `renderer.render(plan)`. The returned `RetrievalFuture` is passed to the WorkTicket.

5. **Stateless between tickets:** The Engine holds no per-ticket state. All ticket-specific data is owned by the WorkTicket instance.

**Validation:** Unit tests for Engine construction, backend selection; integration tests for full submission flow.

---

### Step 11.4: Validate Act plan construction ✓

**Status:** ✅ Complete

**Action:** Verify that `build_act_plan()` exists and meets the Engine's requirements.

**File:** `src/shared/plan_builder.py`

**Existing implementation:** The function `build_act_plan()` is already implemented and exported from `src/shared/__init__.py`.

**Signature (actual):**

```python
def build_act_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
) -> ExecutionPlan:
    """Construct an Act-phase (forward pass + inference retrieval) plan."""
```

**Verified properties:**

- Constructs forward pass, render_logits, softmax, probability aggregation nodes
- Includes inference retrieval as terminal synchronization point
- Does NOT include loss computation, gradient calculation, or parameter updates
- Buffer lifecycle is single-plan scoped (ADR-009 compliant)

**Validation:** Existing tier1/test_plan_builder.py covers Act plan structure.

---

### Step 11.5: Validate Learn plan construction ✓

**Status:** ✅ Complete

**Action:** Verify that `build_learn_plan()` exists and meets the Engine's requirements.

**File:** `src/shared/plan_builder.py`

**Existing implementation:** The function `build_learn_plan()` is already implemented and exported from `src/shared/__init__.py`.

**Signature (actual):**

```python
def build_learn_plan(
    model_spec: ModelSpec,
    hardware: HardwareProfile,
    strategy: PlanProblemTypeStrategy,
    batch_size: int,
    policy: StabilizationPolicy,
    activation_lifecycle: Literal["cache", "recompute"] = "recompute",
) -> ExecutionPlan:
    """Construct a Learn-phase (gradient production → update) plan."""
```

**Verified properties:**

- Supports `activation_lifecycle="recompute"` mode (Choice 4A: recompute)
- When recomputing, includes forward pass nodes before gradient computation
- Full learning phase: loss, backprop, reduction trees, parameter updates
- Final batch retrieval as terminal synchronization point

**Design note:** The `activation_lifecycle` parameter allows the Engine to explicitly request recompute mode for Event-Triggered execution. The recompute cost is minimal (one forward matmul) relative to the Learn phase work.

**Validation:** Existing tier1/test_plan_builder.py covers Learn plan structure.

---

### Step 11.6: Implement `train_batch()` convenience method

**Action:** The `train_batch()` method is defined in Step 11.3. This step validates its implementation wraps the ticket lifecycle correctly.

**Expected behavior:**

```python
def train_batch(self, x_data, y_data):
    ticket = self.submit(x_data)
    prediction = ticket.get_prediction()
    ticket.resolve(y_data).wait()
    return prediction
```

**Validation:** Integration test confirming `train_batch()` produces identical results to manual ticket lifecycle.

---

### Step 11.7: Update `src/shared/__init__.py` exports

**Action:** Export the new public API types.

**File:** `src/shared/__init__.py`

**Additions (updated — unconditional since Phase 11 complete):**

```python
# User-facing API (ADR-018)
from .ticket import WorkTicket, LearnHandle, TicketState, InvalidTicketStateError
from .engine import Engine

__all__ = [
    # ... existing exports ...
    # User-facing API (ADR-018)
    "WorkTicket",
    "LearnHandle",
    "TicketState",
    "InvalidTicketStateError",
    "Engine",
]
```

**Validation:** Import `from src.shared import Engine` succeeds unconditionally.

---

### Step 11.8: Write ticket lifecycle unit tests

**Action:** Create Tier 1 tests for the WorkTicket state machine.

**File:** `tests/tier1/test_ticket_lifecycle.py`

**Test cases:**

| Test | Description |
| :--- | :--- |
| `test_ticket_initial_state_is_pending` | Freshly created ticket is in PENDING state. |
| `test_get_prediction_transitions_to_act_complete` | `get_prediction()` on PENDING ticket transitions to ACT_COMPLETE. |
| `test_get_prediction_is_idempotent` | Multiple `get_prediction()` calls return same cached value. |
| `test_resolve_transitions_to_resolved` | `resolve()` on ACT_COMPLETE ticket transitions to RESOLVED. |
| `test_resolve_on_pending_waits_for_act` | `resolve()` on PENDING ticket implicitly transitions through ACT_COMPLETE. |
| `test_learn_handle_wait_transitions_to_consumed` | `learn_handle.wait()` transitions ticket to CONSUMED. |
| `test_resolve_on_consumed_raises` | `resolve()` on CONSUMED ticket raises `InvalidTicketStateError`. |
| `test_double_resolve_raises` | Second `resolve()` call raises `InvalidTicketStateError`. |
| `test_data_capture_is_copy` | Mutating original array does not affect ticket's stored data. |
| `test_get_prediction_on_resolved_returns_cached` | `get_prediction()` on RESOLVED ticket returns cached prediction. |
| `test_get_prediction_on_consumed_returns_cached` | `get_prediction()` on CONSUMED ticket returns cached prediction. |

**Mock strategy:** Tests use a mock `RetrievalFuture` that returns predetermined values without device interaction.

---

### Step 11.9: Write Engine integration tests

**Action:** Create Tier 2 tests exercising the Engine against actual renderers.

**Files:**
- `tests/tier2/cpu/test_engine_cpu.py`
- `tests/tier2/opencl/test_engine_opencl.py`
- `tests/tier2/vulkan/test_engine_vulkan.py` (stub until Phase 5)

**Test cases:**

| Test | Description |
| :--- | :--- |
| `test_submit_returns_pending_ticket` | `engine.submit()` returns a WorkTicket in PENDING state. |
| `test_get_prediction_returns_valid_probabilities` | Predictions are valid probability distributions (sum to 1, non-negative). |
| `test_full_ticket_lifecycle` | PENDING → get_prediction → resolve → wait → CONSUMED. |
| `test_train_batch_convenience` | `engine.train_batch()` completes without error, returns predictions. |
| `test_train_batch_matches_manual_lifecycle` | Predictions from `train_batch()` match manual ticket lifecycle. |
| `test_sequential_tickets` | Multiple tickets can be created and resolved sequentially. |
| `test_overlapping_tickets` | Ticket N+1's Act can overlap with ticket N's Learn (pipeline parallelism). |

**Fixture strategy:** Uses existing Tier 2 fixtures (iris-scale model, analytical data generators).

---

### Step 11.10: Validate rollback gate

**Action:** Run the full test suite and confirm rollback gate is satisfied.

**Rollback gate criteria:**

1. All Tier 1 ticket lifecycle tests pass.
2. All Tier 2 Engine integration tests pass for enabled backends.
3. Existing tests (Phases 0–10) remain green.
4. Feature flag correctly gates the API.

**Command:**

```bash
cd architectures/averaging_ensembled_classifier
pytest tests/ -v 2>&1 | tee /tmp/phase11-gate.log
grep -E "passed|failed|error" /tmp/phase11-gate.log
```

---

## 4. Data Type Reference

### TicketState (Enum)

| Value | Description |
| :--- | :--- |
| `PENDING` | Act plan dispatched, inference not yet complete. Ticket created but `get_prediction()` not yet called. |
| `ACT_COMPLETE` | Act phase complete, prediction cached. Device buffers released. Ready for `resolve()`. |
| `RESOLVED` | Learn plan dispatched. Ground truth provided. LearnHandle returned. |
| `CONSUMED` | Learn phase complete. Ticket lifecycle terminated. No further operations permitted. |

### WorkTicket

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `_x_data` | `NDArray[np.floating]` | Captured input data (copy). |
| `_act_future` | `RetrievalFuture` | Future for Act plan's inference retrieval. |
| `_engine` | `Engine` | Parent engine reference for Learn dispatch. |
| `_state` | `TicketState` | Current lifecycle state. |
| `_cached_prediction` | `NDArray[np.floating] | None` | Prediction cache after Act completes. |

### InvalidTicketStateError (Exception)

Raised when a ticket operation is invalid for the current state. Examples:
- `resolve()` on a CONSUMED ticket
- `resolve()` called twice on the same ticket

### LearnHandle

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `_future` | `RetrievalFuture` | Future for Learn plan's final_batch retrieval. |
| `_ticket` | `WorkTicket` | Associated ticket reference for state transition. |

### Engine

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `_model_spec` | `ModelSpec` | Model architecture configuration. |
| `_parameter_space` | `ParameterSpace` | Learnable parameters. |
| `_hardware_profile` | `HardwareProfile` | Target hardware characteristics. |
| `_precision` | `PrecisionConfig` | Numeric precision configuration. |
| `_renderer` | `PlanRenderer` | Backend renderer instance. |

---

## 5. WorkTicket State Machine

```
                    submit()
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  ┌─────────┐                                │
    │  │ PENDING │◄───────────────────────────────┘
    │  └────┬────┘
    │       │
    │       │ get_prediction()
    │       │ [block on Act future, cache result, release buffers]
    │       ▼
    │  ┌─────────────┐
    │  │ ACT_COMPLETE │
    │  └──────┬──────┘
    │         │
    │         │ resolve(y_data)
    │         │ [dispatch Learn plan, return LearnHandle]
    │         ▼
    │    ┌──────────┐
    │    │ RESOLVED │
    │    └────┬─────┘
    │         │
    │         │ learn_handle.wait()
    │         │ [block on Learn future, release buffers]
    │         ▼
    │    ┌──────────┐
    └────│ CONSUMED │
         └──────────┘
```

**Special case:** `resolve()` called in PENDING state implicitly calls `get_prediction()` first, transitioning through ACT_COMPLETE before reaching RESOLVED.

**Idempotent reads:** `get_prediction()` returns the cached prediction when called in any state after ACT_COMPLETE (i.e., ACT_COMPLETE, RESOLVED, or CONSUMED). No state transition occurs; the method is purely a read of cached data.

---

## 6. Engine Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         User Code                            │
│  ticket = engine.submit(x_data)                              │
│  probs = ticket.get_prediction()                             │
│  ticket.resolve(y_data).wait()                               │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         Engine                               │
│  ┌────────────────┐  ┌─────────────────┐  ┌───────────────┐  │
│  │ ModelSpec      │  │ ParameterSpace  │  │ HardwareProfile│ │
│  └────────────────┘  └─────────────────┘  └───────────────┘  │
│                              │                               │
│                              ▼                               │
│                    ┌─────────────────┐                       │
│                    │  Plan Builder   │                       │
│                    │ ┌─────────────┐ │                       │
│                    │ │ Act-only    │ │                       │
│                    │ │ Learn-only  │ │                       │
│                    │ │ Combined    │ │                       │
│                    │ └─────────────┘ │                       │
│                    └────────┬────────┘                       │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │  PlanRenderer   │ ◄─── Backend-specific │
│                    │  (Protocol)     │      implementation   │
│                    └────────┬────────┘                       │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │ RetrievalFuture │                       │
│                    └─────────────────┘                       │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                       WorkTicket                             │
│  State: PENDING → ACT_COMPLETE → RESOLVED → CONSUMED         │
│  Data: x_data (copy), cached_prediction, act_future          │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. Plan Construction Modes

The plan builder supports three construction modes:

| Mode | Function | Use Case | Terminal Node |
| :--- | :--- | :--- | :--- |
| **Combined** | `build_execution_plan()` | Sequential mode: `train_batch()` | `final_batch_retrieval` |
| **Act-only** | `build_act_plan()` | Event-Triggered: `submit()` | `inference_retrieval` |
| **Learn-only** | `build_learn_plan()` | Event-Triggered: `resolve()` | `final_batch_retrieval` |

### Act-only Plan Node Inventory

| Node ID | Type | Description |
| :--- | :--- | :--- |
| 1 | `KernelDispatchNode` | `forward_pass` — compute hidden activations |
| 2–(M+1) | `KernelDispatchNode` | `module_forward_pass` × M modules |
| M+2 | `KernelDispatchNode` | `aggregate_probabilities` |
| M+3 | `RetrievalNode` | `inference_retrieval` |

### Learn-only Plan Node Inventory

The Learn-only plan recomputes the forward pass (Choice 4A) then performs the full learning phase:

| Node ID | Type | Description |
| :--- | :--- | :--- |
| 1 | `KernelDispatchNode` | `forward_pass` — recompute hidden activations |
| 2–(M+1) | `KernelDispatchNode` | `module_forward_pass` × M modules (recomputed) |
| M+2 | `KernelDispatchNode` | `aggregate_probabilities` (recomputed) |
| M+3 | `KernelDispatchNode` | `compute_loss` |
| M+4 | `KernelDispatchNode` | `backprop_output_layer` |
| M+5–(2M+4) | `KernelDispatchNode` | `module_gradient` × M modules |
| 2M+5 | `KernelDispatchNode` | `shared_weight_gradient` |
| 2M+6–... | `ReductionTreeNode` | Gradient aggregation trees |
| ... | `KernelDispatchNode` | `parameter_update` |
| final | `RetrievalNode` | `final_batch_retrieval` |

See [DESIGN.md](../DESIGN.md) §Plan Node Taxonomy for the complete combined-plan node inventory.

---

## 8. Test Specification

### Tier 1 Tests (Host-side, no backend)

| File | Test Count | Coverage |
| :--- | :--- | :--- |
| `test_ticket_lifecycle.py` | ~15 | WorkTicket state transitions, error cases, data capture |
| `test_engine_construction.py` | ~8 | Engine initialization, backend selection logic |
| `test_act_only_plan.py` | ~10 | Act-only plan structure, buffer lifecycle |
| `test_learn_only_plan.py` | ~10 | Learn-only plan structure, recompute verification |

**Total Tier 1:** ~45 tests

### Tier 2 Tests (Per-backend integration)

| File | Test Count | Coverage |
| :--- | :--- | :--- |
| `test_engine_cpu.py` | ~8 | Full ticket lifecycle against CPU renderer |
| `test_engine_opencl.py` | ~8 | Full ticket lifecycle against OpenCL renderer |
| `test_engine_vulkan.py` | ~8 (stub) | Full ticket lifecycle against Vulkan renderer |

**Total Tier 2:** ~24 tests (per enabled backend)

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Plan builder API churn during Phase 1 | Low | Medium | ADRs 001–010 heavily constrain plan builder shape. Monitor Phase 1 for unexpected revisions; pause ticket work if interface destabilizes. |
| RetrievalFuture Protocol extension needed | Low | Low | ADR-010 defines complete Protocol. If extension is needed, formalize per CONCEPT.md §1 (Architectural Elegance Feedback). |
| Backend selection complexity | Medium | Low | Start with simple preference order. Defer sophisticated selection (device capabilities, load balancing) to future work. |
| Choice 6 (Observability) interaction | Medium | Low | LearnHandle is designed to accommodate future diagnostics. Avoid committing to specific diagnostic properties until Choice 6 is decided. |
| Test infrastructure dependency on Phase 4 | Low | Low | Tier 1 tests are self-contained. Tier 2 tests reuse existing fixtures. No hard dependency on Phase 4 deliverables. |
| Feature flag maintenance burden | Low | Low | Single flag, single conditional import block. Removed entirely at Phase 6 when legacy API is retired. |

---

## References

- [ADR-018: User-Facing API](../adr/ADR-018-user-facing-api.md) — governing ADR for this phase
- [ADR-001: Backend Abstraction Boundary](../adr/ADR-001-backend-abstraction-boundary.md) — PlanRenderer Protocol, jurisdictional separation
- [ADR-002: Plan Node Types](../adr/ADR-002-plan-node-types-and-synchronization-structure.md) — five-node taxonomy, ExecutionPlan structure
- [ADR-009: Buffer Lifecycle](../adr/ADR-009-buffer-lifecycle-in-the-plan-model.md) — single-plan buffer lifetime, BufferDescriptor
- [ADR-010: RetrievalFuture Protocol](../adr/ADR-010-d2h-transfer-and-phase-sync-points.md) — `.wait()` / `.result()` / `.release()` pattern
- [ADR-012: Module Factoring](../adr/ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` directory structure
- [ADR-017: Migration Path](../adr/ADR-017-incremental-migration-path.md) — feature-flag gated parallel development
- [CONCEPT.md](../CONCEPT.md) — §4 Asynchronous Host Interaction, §5 Unified Execution Model
