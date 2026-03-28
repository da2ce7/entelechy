# ADR-018: User-Facing API

**Status:** ACCEPTED  
**Date:** 2026-03-28  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002, ADR-009, ADR-010, ADR-012, ADR-017  
**Blocks:** —

---

## Context

The existing ADR chain (ADR-001 through ADR-017) establishes the internal architecture: a shared orchestration layer producing immutable execution plans consumed by per-backend renderers. The plan model (ADR-002), buffer lifecycle (ADR-009), retrieval protocol (ADR-010), and migration path (ADR-017) are all decided. What has not been decided is the **user-facing API** — the surface through which application code submits data, requests inference, triggers learning, and observes results.

### The current user-facing surface

The existing API is minimal and synchronous:

```python
orchestrator = TrainingOrchestrator(model_spec, param_space, compute_env, hyperparams, ...)
final_probs = orchestrator.train(X_train, y_train)
```

`TrainingOrchestrator.train()` owns the epoch loop, constructs one `ExecutionPlan` per epoch, delegates to a transient `BatchProcessor`, and blocks on `learn_event.wait()` before proceeding to the next epoch. The caller has no access to per-epoch inference results during training, no ability to submit data incrementally, and no ability to decouple inference from learning temporally.

### The architectural seam

CONCEPT.md §4 (Asynchronous Host Interaction) and §5 (Unified Execution Model) define two execution modes:

- **Sequential Execution Mode:** Act and Learn phases execute contiguously for pre-labeled batches. The host submits `(x, y)` pairs and waits for the full Act+Learn cycle to complete.
- **Event-Triggered Execution Mode:** The Act phase executes immediately upon data submission. The Learn phase is deferred until ground truth arrives from an external source.

ADR-010 demonstrates both modes using the `RetrievalFuture` Protocol. For Event-Triggered mode, it shows a **split-plan pattern**: `renderer.render(act_plan)` followed later by `renderer.render(learn_plan)` when ground truth arrives. The infrastructure for temporal decoupling of Act from Learn exists at the plan/renderer level. What is missing is a user-facing abstraction that exposes this capability.

### The DESIGN.md vision (historical context)

DESIGN.md (December 2025) proposed a `WorkTicket` / `LearnTicket` API with a global mutable `networkx.DiGraph` and an `AsyncConductor` scanning loop. The subsequent ADR work chose a different internal architecture (immutable plans, backend-owned rendering, no `networkx`, no `asyncio` conductor). The `WorkTicket` *concept* — a persistent promise representing a unit of user intent, with temporal decoupling between inference and learning — has architectural merit independent of the internal execution model it was originally proposed for. This ADR realizes that concept on top of the existing plan-model architecture.

### What this ADR must decide

This ADR identifies the design choices that determine the user-facing API surface. Each choice is presented with its options, trade-offs, and interactions with the existing ADR chain.

**Partial resolution (2026-03-28):** Choice 1 (Unit of user intent) is decided as **Option 1A: Stateful ticket (WorkTicket pattern)**. Choice 2 (Concurrency model) is decided as **Option 2B: Future-based (non-blocking, sync-compatible)**. Choice 3 (Batch composition model) is decided as **Option 3A: Explicit batch submission**. Choice 4 (Cross-plan buffer lifecycle) is decided as **Option 4A: Recompute**. Choice 5 (Data persistence and epoch iteration) is decided as **Option 5A: Ephemeral (no persistence)**. Choice 7 (Migration positioning) is decided as **Option 7C: Parallel workstream (Phase 4-adjacent)**. The remaining choice (6) is open.

---

## Decision Drivers

1. **CONCEPT.md §4 (Asynchronous Host Interaction).** The `inference_event` and `final_batch_event` are named synchronization points. The API must expose both Sequential and Event-Triggered execution modes. **Satisfied by Choice 1A:** the ticket's state transitions correspond directly to these synchronization points — `PENDING → ACT_COMPLETE` maps to `inference_event`; `RESOLVED → CONSUMED` maps to `final_batch_event`.

2. **CONCEPT.md §5 (Unified Execution Model).** All workflows follow Act/Learn sequencing regardless of backend. The API must not bifurcate into separate "batch training" and "online inference" surfaces — the same primitives must serve both modes. **Satisfied by Choice 1A:** the ticket expresses both modes through identical syntax; the only difference is whether the user inspects the prediction before calling `resolve()`.

3. **CONCEPT.md §1 (Architectural Elegance Feedback).** If the user-facing API creates tension with the plan model's contracts, the response is to extend the architecture — not to expose plan-model internals as user-visible leaky abstractions.

4. **ADR-001 (Backend Abstraction Boundary).** The user-facing API must not expose any backend-specific type. No `cl.Event`, `VkFence`, or backend renderer appears in the user's import surface.

5. **ADR-002 (Plan Node Types).** The five-node taxonomy is a closed set. The user-facing API may produce different plan *configurations* (Act-only, Learn-only, combined) but must not require new plan node types. **Satisfied by Choices 1A + 4A:** the ticket maps to existing plan configurations; recompute eliminates the need for cross-plan buffer retention nodes.

5b. **ADR-010 (RetrievalFuture Protocol).** The `RetrievalFuture` Protocol defines `.wait()` / `.result()` / `.release()` — a non-blocking, sync-compatible future pattern at the renderer level. The user-facing concurrency model should be consistent with this existing pattern. **Satisfied by Choice 2B:** the ticket's lifecycle methods (`submit` returns immediately, `get_prediction` blocks on demand) lift this same deferred-observation pattern to the user surface.

6. **ADR-009 (Buffer Lifecycle).** Each plan's `BufferDescriptor` namespace is self-contained. The recompute decision (Choice 4A, decided) preserves this invariant: Act and Learn plans submitted independently share no device-side buffers, and ADR-009 requires no extension. The ticket (Choice 1A, decided) holds only host-side state; its existence has no impact on device-side buffer lifetimes.

7. **ADR-010 (D2H Transfer & Sync Points).** The `RetrievalFuture` Protocol is the sole host-facing type for observing device results. The ticket consumes `RetrievalFuture` internally — `get_prediction()` maps to the Act plan's `RetrievalFuture.result()` and `RetrievalFuture.release()`. The user interacts with the ticket, not with `RetrievalFuture` directly. This wrapping adds semantic value (lifecycle management, Act→Learn data association) rather than mere indirection.

8. **ADR-012 (Module Factoring).** ADR-012 defines `src/shared/` and `src/backends/<name>/`. The user-facing API module(s) belong in the shared layer. The `WorkTicket` type and the engine that produces it are shared-layer components. Their relationship to the existing `orchestrator.py` (the renamed `main_orchestrator.py`) must be resolved.

9. **ADR-017 (Migration Path).** The migration is organized into Phases 0–6. The user-facing API is not addressed by any existing phase. Its introduction must be positioned relative to the existing phase structure — either as an extension of an existing phase or as a new Phase 7.

10. **CONCEPT.md Validation Scenarios.** The "Iris Case" (Sequential), "Real-Time Trader" (Event-Triggered), and "Data Tsunami" (massive batch) scenarios are canonical. The API must handle all three without mode-specific branching in application code. **Satisfied by Choice 1A:** see Degenerate Case Validation below.

---

## Design Choices

### Choice 1: Unit of user intent — DECIDED

> **Decision (2026-03-28): Option 1A — Stateful ticket (WorkTicket pattern).**

The user receives a mutable, stateful object whose lifecycle mirrors the Act/Learn temporal split:

```python
ticket = engine.submit(x_data)                    # → WorkTicket (PENDING)
probs = ticket.get_prediction()                    # blocks until Act completes
learn_handle = ticket.resolve(y_data)              # → LearnHandle
learn_handle.wait()                                # blocks until Learn completes
```

The ticket transitions through states: `PENDING → ACT_COMPLETE → RESOLVED → CONSUMED`. The engine manages the mapping from ticket state to plan construction and rendering.

A convenience method provides minimal-ceremony batch training:

```python
result = engine.train_batch(X_train, y_train)      # combined Act+Learn, returns predictions
```

#### Rationale

- **System-enforced Act→Learn association.** The ticket is the linkage between an inference and its subsequent learning step. `engine.submit(x_data)` captures the host-side input data reference inside the ticket. `ticket.resolve(y_data)` constructs the Learn plan using the captured reference — not a user-re-specified copy. This eliminates silent bugs where input data is mutated or swapped between Act and Learn phases. With the rejected Option 1B, `learn(x_data, y_data)` requires the user to re-specify `x_data`; if the underlying array has been overwritten in-place, the Learn plan recomputes the forward pass on different data than was inferred, producing silently incorrect gradients.

- **Unified Execution Model (CONCEPT.md §5).** The ticket natively expresses both execution modes through identical syntax:
  - Sequential: `submit(x)` → `resolve(y)` (resolve immediately, prediction may or may not be inspected)
  - Event-Triggered: `submit(x)` → `get_prediction()` → *(time passes)* → `resolve(y)`

  No mode flag, no conditional branching, no separate API surface.

- **Temporal split as first-class concept (CONCEPT.md §4).** The Act/Learn temporal split is architecturally fundamental. The ticket's state machine (`PENDING → ACT_COMPLETE → RESOLVED → CONSUMED`) directly reifies this split as a user-visible object. The state transitions correspond to the named synchronization points: `inference_event` (PENDING → ACT_COMPLETE) and `final_batch_event` (RESOLVED → CONSUMED).

- **Lightweight host-side object (Choice 4A synergy).** With the recompute decision, the ticket holds only host-side state: the input data reference, the cached prediction (after Act completes), and the ticket's current state. No device buffers are retained, no VRAM is held, no cross-plan coordination is needed.

- **Natural anchor for remaining design choices.** The ticket provides a consistent unit of identity for batch composition (Choice 3: tickets as accumulable items), data persistence (Choice 5: resolved tickets as persistent registry entries), and observability (Choice 6: diagnostics attached to LearnHandle). These interactions are not forced — each remaining choice has options that don't use the ticket as an anchor — but the ticket provides a natural gravitational center.

#### Internal mechanism

The ticket interacts with the plan model and retrieval protocol as follows:

1. **`engine.submit(x_data)`** — Constructs and renders an Act-only plan. The renderer returns a `RetrievalFuture` for `"inference_retrieval"`. The ticket stores the future and a reference to the host-side input data. Returns immediately with a PENDING ticket.

2. **`ticket.get_prediction()`** — Calls `future.result()` to obtain the unpadded numpy array, then calls `future.release()` to free device-side buffers (Choice 4A: no buffers are retained). The ticket transitions to ACT_COMPLETE and caches the prediction. Subsequent calls return the cached value.

3. **`ticket.resolve(y_data)`** — Constructs and renders a Learn-only plan using the stored `x_data` and provided `y_data`. The Learn plan recomputes `hidden_activations` from `x_data` (Choice 4A). The ticket transitions to RESOLVED and returns a `LearnHandle` wrapping the Learn plan's `RetrievalFuture` for `"final_batch_retrieval"`.

4. **`learn_handle.wait()`** — Blocks on the Learn plan's `RetrievalFuture.wait()`. Upon return, calls `RetrievalFuture.release()`. The ticket transitions to CONSUMED.

Note: `resolve()` on a ticket in PENDING state implicitly waits for Act to complete before constructing the Learn plan, transitioning through ACT_COMPLETE.

#### Rejected: Option 1B — Explicit two-call API

The user would call `engine.infer(x_data)` then `engine.learn(x_data, y_data)` as independent operations.

**Reasons for rejection:**
- No system-enforced linkage between inference and learning. The user re-specifies `x_data` in `learn()` and must maintain the association manually — fragile under array mutation.
- Mirrors the internal split-plan mechanism (ADR-010), but the split-plan pattern is a renderer-level concern. Surfacing it directly violates ADR-001's jurisdictional separation.
- Both execution modes are expressible but with less safety and more ceremony than the ticket model.

#### Rejected: Option 1C — Plan-level API (direct plan exposure)

The user would construct execution plans and submit them to a renderer directly.

**Reasons for rejection:**
- Exposes plan-model vocabulary (`RetrievalNode` names, `BufferDescriptor` semantics, plan topology) to application code, violating ADR-001's jurisdictional separation.
- CONCEPT.md §1 (Architectural Elegance Feedback) directs that internal contracts should not become user-visible surfaces. Plan inspection may be valuable as a debug capability (relevant to Choice 6) but not as the primary API idiom.
- Highest ceremony for simple cases: the Iris Case requires 6+ lines of plan-building and rendering code.

---

### Choice 2: Concurrency model — DECIDED

> **Decision (2026-03-28): Option 2B — Future-based (non-blocking, sync-compatible).**

With the WorkTicket model decided (Choice 1A), the concurrency question becomes: what are the blocking semantics of the ticket's lifecycle methods?

The ticket defines four user-visible operations: `engine.submit()`, `ticket.get_prediction()`, `ticket.resolve()`, and `learn_handle.wait()`. Choice 2 determines which of these block the calling thread and which return immediately with a handle for deferred observation.

#### Rejected: Option 2A — Synchronous (blocking)

All user-facing methods block until their phase completes.

```python
ticket = engine.submit(x_data)         # blocks until Act completes; ticket in ACT_COMPLETE
probs = ticket.get_prediction()        # returns immediately (prediction cached)
ticket.resolve(y_data)                 # blocks until Learn completes; ticket in CONSUMED
```

**Reasons for rejection:**
- Collapses the ticket's four-state machine (PENDING → ACT_COMPLETE → RESOLVED → CONSUMED) to three by hiding PENDING from the user — `submit()` blocks past it. This retroactively suppresses part of the Choice 1A decision: the ticket ceases to be a handle for deferred observation and becomes a result container.
- Multi-item pipeline parallelism requires the user to manage threads manually. The host cannot overlap inference of item N+1 with learning of item N without threading.
- 2B is a strict superset: calling `get_prediction()` immediately after `submit()` degenerates to 2A behavior. Any code valid under 2A is valid under 2B, but not the reverse.

#### Option 2B: Future-based (non-blocking, sync-compatible) ✓ DECIDED

`submit()` and `resolve()` return immediately. The user decides when to block by calling `get_prediction()` or `learn_handle.wait()`.

```python
ticket = engine.submit(x_data)         # returns immediately; ticket in PENDING
# ... do other work ...
probs = ticket.get_prediction()        # blocks until Act completes (may already be done)
learn_handle = ticket.resolve(y_data)  # returns immediately; Learn plan dispatched
# ... do other work ...
learn_handle.wait()                    # blocks until Learn completes
```

#### Rationale

- **The ticket is inherently a future.** Choice 1A decided that `submit()` returns a `WorkTicket` in PENDING state — a handle whose result is not yet available. Option 2B embraces this by making all transitions lazy-observable: the user holds the handle and decides when to block. Option 2A would suppress this by blocking inside `submit()`, making PENDING invisible.

- **Consistent with renderer-level pattern (ADR-010).** The `RetrievalFuture` Protocol already provides non-blocking, sync-compatible deferred observation at the renderer level (`.wait()` / `.result()` / `.release()`). Option 2B lifts this same pattern to the user surface. The ticket's `get_prediction()` maps to `RetrievalFuture.result()` + `.release()`; `learn_handle.wait()` maps to the Learn plan's `RetrievalFuture.wait()`. The user-facing and renderer-facing concurrency models are structurally identical.

- **Strict superset of synchronous behavior.** Calling `get_prediction()` immediately after `submit()` degenerates to Option 2A behavior. No user loses functionality. But the reverse is not true: 2A cannot express "hold multiple in-flight tickets" or pipeline parallelism without threads.

- **Pipeline parallelism without threads.** The user can hold tickets for multiple items simultaneously, submitting item N+1's Act while item N's Learn is in flight. With Choice 4A (recompute), each plan's buffers are self-contained — overlapping plans requires no cross-plan buffer coordination.

- **No asyncio dependency.** Compatible with all Python environments — synchronous scripts, Jupyter notebooks, non-async frameworks. The existing codebase has zero `asyncio` usage; 2B requires no new concurrency framework.

- **Implementation can start synchronous and evolve.** The first implementation can render plans synchronously inside `submit()` and enqueue them on the device. The ticket's `get_prediction()` then blocks on the device event. This is functionally close to 2A in wall-clock behavior, but the API contract is future-based. Later, if background plan construction becomes valuable, the contract doesn't change — only the internals do.

- **Abandoned ticket cost is bounded (Choice 4A synergy).** With recompute decided, no device memory is tied to ticket lifetime. If a user calls `submit()` and never consumes the ticket, the Act plan's device buffers are freed after the `RetrievalFuture`'s device transfer completes regardless. The only uncollected resource is the host-side prediction buffer inside the future, which is reclaimed by garbage collection. Documentation should note this; a `ticket.cancel()` method or `__del__` cleanup may be specified.

#### Rejected: Option 2C — Native asyncio

Ticket lifecycle methods are coroutines.

```python
ticket = await engine.submit(x_data)   # awaits Act completion; ticket in ACT_COMPLETE
probs = ticket.get_prediction()        # returns immediately
await ticket.resolve(y_data)           # awaits Learn completion
```

**Reasons for rejection:**
- Hard dependency on an `asyncio` event loop. Excludes users who cannot or do not want to use `asyncio` (Jupyter notebooks with synchronous cells, simple scripts, integration into non-async frameworks).
- The current codebase has zero `asyncio` usage. The `RetrievalFuture` Protocol (ADR-010) is synchronous (`.wait()` blocks). Bridging `RetrievalFuture` into `asyncio` requires an adapter — implementation cost with no corresponding architectural benefit over 2B.
- Can be added later as a thin async wrapper over 2B's future-based contract if demand materializes, without breaking changes.

#### Rejected: Option 2D — Dual-mode (sync + async)

Provide both synchronous and asynchronous entry points. The sync API blocks; the async API returns awaitables. Both dispatch through the same plan builder and renderer.

```python
# Sync
ticket = engine.submit(x_data)                # blocks until Act completes
ticket.resolve(y_data)                         # blocks until Learn completes

# Async
ticket = await engine.submit_async(x_data)    # awaits Act completion
await ticket.resolve_async(y_data)             # awaits Learn completion
```

**Reasons for rejection:**
- Doubles the API surface (`submit` + `submit_async`, `resolve` + `resolve_async`). Two entry points per ticket operation with potentially subtle behavioral differences.
- Risk of a fractured ecosystem where examples, documentation, and community code use inconsistent styles.
- The marginal benefit over 2B is native `await` syntax — slightly more ergonomic but insufficient to justify the maintenance burden. Async wrappers can be added non-breakingly over 2B later.

---

### Choice 3: Batch composition model — DECIDED

> **Decision (2026-03-28): Option 3A — Explicit batch submission.**

How does the user express "train on these N items as a single batch"?

With the WorkTicket model decided (Choice 1A), the fundamental question is whether a ticket represents a *batch* (matrix submission) or an *individual item* (vector submission), and how items are collected into plans.

#### Option 3A: Explicit batch submission ✓ DECIDED

The user constructs a batch and submits it as a unit. A single ticket represents the entire batch.

```python
ticket = engine.submit(X_batch)                # N items, one ticket
probs = ticket.get_prediction()                # (N × C) probability matrix
ticket.resolve(y_batch).wait()                 # one reduction tree, one parameter update
```

The convenience method `engine.train_batch(X, y)` wraps this pattern.

#### Rationale

- **Direct continuation of decided choices.** Choice 1A commits to one plan construction per `submit()` call; Choice 2B commits to immediate plan dispatch. Option 3A respects both: one `submit()` → one Act plan → one ticket → one prediction matrix. Options 3B and 3C require either deferring plan construction (breaking 2B's immediate-dispatch contract) or dispatching per-item plans (degenerating the reduction tree and multiplying kernel launch overhead).
- **Kernel architecture is batch-native.** The plan's `ReductionTreeNode` (ADR-003) performs $\log_K(N)$ aggregation over N items. Per-item plans (Option 3B) reduce this to trivial single-item reductions, discarding the architecture's primary efficiency mechanism.
- **Aligns with the current `orchestrator.train()` pattern** — one batch → one plan → one render. Migration (Choice 7) is simpler when the batch granularity is unchanged.
- **User owns batching.** The engine does not accumulate items or maintain internal state between `submit()` calls. The engine is stateless between ticket lifecycles (synergy with Choice 5A).
- **Event-Triggered single-item case is a degenerate batch.** The Trader scenario submits a 1-row matrix (`market_snapshot.reshape(1, -1)`). This is slightly more ceremonial than a raw vector submission but semantically correct and architecturally consistent.
- **Streaming accumulation gap acknowledged.** The scenario of many concurrent Event-Triggered items needing individual predictions followed by batched learning is not natively expressible. Per CONCEPT.md §1 (Architectural Elegance Feedback), this pattern should be formalized as a first-class primitive (e.g., a user-space `BatchAccumulator` utility) if and when real workloads demonstrate the need — not pre-built speculatively.

#### Rejected: Option 3B — Ticket accumulation with explicit flush

The user submits items individually, each producing a ticket. An explicit `flush()` triggers plan construction and rendering for all accumulated tickets as a single batch.

```python
tickets = [engine.submit(x) for x in stream]   # one ticket per item, Act plans dispatched
engine.flush(y_labels)                          # constructs batch Learn plan for all pending tickets
```

**Reasons for rejection:**
- Each individual `submit(x)` triggers a separate 1-item Act plan — a full plan graph construction, buffer allocation, kernel launch, and D2H transfer for a single row. For N items, this produces N independent kernel launches instead of one batched matmul. The reduction tree (ADR-003) degenerates to trivial single-item passes.
- Contradicts Choice 2B's immediate-dispatch contract: `submit()` dispatches a plan immediately, but the per-item plan is architecturally wasteful. Deferring dispatch until `flush()` would violate the decided `submit()` semantics.
- Introduces engine-side accumulation state (the set of unresolved tickets), making the engine stateful between calls.

#### Rejected: Option 3C — Automatic batching with configurable policy

The engine automatically batches items according to a policy (e.g., batch size threshold, time window, or explicit trigger):

```python
engine = Engine(batch_policy=BatchWhenFull(size=64))
for x, y in stream:
    engine.submit(x, y)    # triggers training when 64 tickets accumulate
```

**Reasons for rejection:**
- Fundamentally incompatible with Choice 2B's immediate-dispatch contract. `submit()` cannot dispatch a plan immediately if the engine is waiting for the accumulation threshold. `ticket.get_prediction()` would block until *other items arrive* — a single prediction is held hostage by future data that may not exist, breaking the Event-Triggered mode's "submit one, infer immediately" guarantee.
- The batching policy is a new configuration dimension with no existing primitives in the ADR chain. Defining and composing policies (size-based, time-based, manual trigger) is a design effort disproportionate to the current architecture's scope.
- Reduces user control over exactly when training occurs. The user cannot easily say "train on exactly these 37 items right now."

---

### Choice 4: Cross-plan buffer lifecycle — DECIDED

> **Decision (2026-03-28): Option 4A — Recompute (no cross-plan sharing).**

When Act and Learn plans are submitted independently (as the WorkTicket model permits with temporal separation), intermediate device-side buffers from the Act plan may be needed by the subsequent Learn plan.

#### Option 4A: Recompute (no cross-plan sharing) ✓ DECIDED

The Learn plan always recomputes any intermediate values it needs. The Act plan's device buffers are released when the Act `RetrievalFuture` is released (i.e., when `ticket.get_prediction()` caches the result). No buffer persists across plan boundaries.

**Rationale:**

- **ADR-009 preservation.** Each plan's `BufferDescriptor` namespace remains fully self-contained. No extension to the single-plan buffer lifetime model is required.
- **Negligible cost.** The recomputed work is one `forward_pass` kernel — a single-layer tiled matrix-vector multiply ($O(N \times D_{\text{in}} \times D_{\text{hidden}})$). The Learn phase already contains work of equal or greater cost: `backprop_shared_weights_chunk` (Node 17) performs the same $O(N \times D_{\text{in}} \times D_{\text{hidden}})$ computation, and `calculate_module_param_grads_chunk` (Node 8) scales as $O(N \times M \times D_{\text{hidden}} \times C)$ — a factor of $\frac{M \times C}{D_{\text{in}}}$ larger. The forward pass is a small fraction of total Learn cost for any non-trivial model configuration.
- **CONCEPT.md validation.** The "Real-Time Trader" scenario explicitly validates this approach — it "assesses the system's ability to release VRAM after the Act phase and efficiently recompute intermediates for the Learn phase."
- **Minimal VRAM footprint.** Device memory is freed promptly after Act completion. No VRAM is held during the temporal gap between inference and ground-truth arrival — which may span seconds to hours in Event-Triggered mode.
- **No stale-activation risk.** If model parameters are updated by another batch's Learn between this item's Act and Learn, the recomputed forward pass uses the current parameters. Caching would produce stale activations silently.
- **Architectural simplicity.** Eliminates the need for cross-plan buffer retention protocols, timeout/eviction policies, and per-item cache lifecycle management — none of which have existing primitives in the ADR chain.
- **WorkTicket synergy (Choice 1A).** The ticket holds only host-side state (input data reference, cached prediction, state enum). No device-side resource is tied to ticket lifetime. Tickets can exist for arbitrary durations in ACT_COMPLETE state without VRAM cost.

#### Rejected: Option 4B — Cached (cross-plan buffer retention)

The Act plan's `hidden_activations` buffer would be retained on the device after Act completes, with the Learn plan referencing the existing allocation.

**Reasons for rejection:**
- Requires extending ADR-009's single-plan buffer lifetime model with a new "buffer retained by host intent" concept.
- VRAM held during the Act→Learn temporal gap conflicts with adaptive memory management (CONCEPT.md "Grace of Adaptive Memory").
- Stale activations: if model parameters are updated between Act and Learn, cached activations are silently invalid.
- Demands timeout/eviction policy design for the case where ground truth never arrives.
- The computational savings (one matmul) do not justify the architectural complexity for a single-hidden-layer model.

#### Rejected: Option 4C — User-controlled (explicit cache/recompute per submission)

The user would specify per-submission whether to cache or recompute.

**Reasons for rejection:**
- Pushes an internal architectural concern (activation caching) to the user surface, requiring users to understand device-side memory trade-offs.
- Inherits all of Option 4B's architectural costs (ADR-009 extension, eviction policy, stale-activation risk) while adding per-call API complexity.
- The recompute cost is low enough that user-level control over this trade-off is unnecessary.

---

### Choice 5: Data persistence and epoch iteration — DECIDED

> **Decision (2026-03-28): Option 5A — Ephemeral (no persistence).**

Does the system remember previously submitted data across epoch boundaries?

#### Option 5A: Ephemeral (no persistence) ✓ DECIDED

Each submission is a one-shot event. The system processes it and forgets it. For multi-epoch training, the user resubmits the full dataset each epoch.

```python
for epoch in range(num_epochs):
    result = engine.train_batch(X_train, y_train)
```

#### Rationale

- **Natural partner of Choice 3A.** With explicit batch submission decided, the user already owns the data and the training loop. The engine receives a batch, processes it, and returns results. No engine-side state accumulates between calls. Adding persistence would introduce a stateful registry that contradicts this stateless-engine property.
- **Simplest implementation — no registry, no lifecycle management.** No new shared-layer component is needed. The engine's only persistent state is the model parameters (managed by `ParameterSpace`).
- **Matches the current `TrainingOrchestrator.train()` pattern exactly.** Migration (Choice 7) is simpler when the data lifecycle is unchanged.
- **Tickets are transient.** Consumed after a single Act+Learn cycle. The ticket's lifecycle terminates at CONSUMED; no resurrection or re-planning occurs. This is consistent with Choice 4A (recompute) — no device-side state persists, and now no host-side registry persists either.
- **H2D transfer cost is acceptable for current scale.** Every epoch re-uploads data to the device. For Iris-scale (150 × 4 floats = 2.4 KB) and even moderate datasets, this cost is negligible relative to kernel execution time. For future large-dataset scenarios, persistent device-side data management can be formalized as a new architectural primitive per CONCEPT.md §1 if real workloads demonstrate the need.
- **User retains full control.** Learning rate scheduling, validation splits, early stopping, data augmentation, and curriculum ordering are all natural in a user-owned loop. Engine-managed persistence (Options 5B/5C) would require hooks or callbacks for these capabilities.

#### Rejected: Option 5B — Persistent registry (LearnTicket pattern)

Resolved tickets persist in a registry. Each epoch, the engine re-plans and re-renders all active tickets. The user can retire individual tickets to remove their data from future training.

```python
ticket = engine.submit(x_data)
ticket.resolve(y_data)                 # (x, y) pair persists in registry
# ... many epochs later ...
ticket.retire()                        # data removed from future training
```

**Reasons for rejection:**
- Introduces a stateful registry component in the shared layer, contradicting the stateless-engine property established by Choice 3A. The engine would need to track all active `(x, y)` pairs and include them in each epoch's plan — a fundamentally different execution model from batch-at-a-time processing.
- With Choice 3A (explicit batch), tickets represent batches, not individual items. A persistent batch-level ticket lacks the sub-batch granularity needed for `retire()` to be useful (cannot retire item 37 of a 150-item batch).
- Device-side data management becomes non-trivial: persistent device buffers, eviction policies, and `retire()` semantics during in-flight plans are all undefined.
- No canonical CONCEPT.md scenario requires persistent data management. The "Interactive Scientist" use case this enables is not among the three validated scenarios.

#### Rejected: Option 5C — Dataset abstraction (engine-managed data)

The engine manages a mutable dataset object. The user adds/removes items. Epoch iteration is engine-controlled.

```python
ds = engine.create_dataset()
ds.add(x1, y1)
ds.add(x2, y2)
engine.train(ds, epochs=10)            # engine iterates internally
ds.remove(item_id=0)
engine.train(ds, epochs=5)             # continues without removed item
```

**Reasons for rejection:**
- Introduces a `Dataset` abstraction not currently in the architecture, with undefined relationships to `ParameterSpace`, `ModelSpec`, the plan builder, and the ticket model.
- Training loop ownership moves to the engine, requiring hooks or callbacks for per-epoch logic (learning rate scheduling, validation, early stopping). This contradicts 5A's user-owns-the-loop principle and Choice 3A's explicit-submission model.
- The ticket model (Choice 1A) and the dataset abstraction operate at fundamentally different granularities. Reconciling whether tickets are Dataset entries or whether `train()` bypasses the ticket model internally creates unnecessary conceptual friction.
- Combines naturally with Option 3C (auto-batching), which is rejected. Without auto-batching, the dataset abstraction provides little beyond what `train_batch(X, y)` already offers.

---

### Choice 6: Observability surface

What runtime information does the system expose to the user during and after training?

With the WorkTicket model decided (Choice 1A), the `LearnHandle` returned by `ticket.resolve()` is the natural carrier for per-step diagnostic information.

#### Option 6A: Minimal (results only)

The user receives prediction probabilities (via `ticket.get_prediction()`) and learns when training completes (via `learn_handle.wait()`). No internal metrics are exposed.

**Considerations:**
- Simplest to implement and maintain.
- Users who need loss values, gradient norms, or per-epoch statistics must compute them externally from the returned probabilities and known labels.
- Sufficient for the current Iris-scale validation scenarios.

#### Option 6B: Per-batch diagnostics

The `LearnHandle` exposes diagnostic properties available after completion:

```python
learn_handle = ticket.resolve(y_data)
learn_handle.wait()
print(learn_handle.loss, learn_handle.effective_batch_size)
```

**Considerations:**
- The diagnostic aggregation tree (Node 14) already computes aggregated probabilities and loss on-device. Exposing these to the user requires only an additional `RetrievalNode` in the Learn plan and corresponding `RetrievalFuture` consumed by the `LearnHandle`.
- Does not require new device-side computation — these values are already produced.
- The set of available diagnostics is fixed by the plan's `RetrievalNode` inventory. Adding new diagnostics requires plan-level changes.
- The `LearnHandle` becomes the natural diagnostics carrier — no separate "result" type needed.

#### Option 6C: Structured metrics with callback/streaming

The engine exposes a metrics stream or accepts callbacks for real-time monitoring:

```python
engine.on_batch_complete(lambda metrics: wandb.log(metrics))
for epoch in range(num_epochs):
    engine.train_batch(X, y)
```

**Considerations:**
- Most flexible for integration with experiment tracking systems (Weights & Biases, TensorBoard, MLflow).
- Callback invocation timing and threading semantics must be defined. Does the callback run on the main thread? On a background thread? Within an `asyncio` task?
- The callback interface must be backend-agnostic — no backend type may appear in the metrics object.
- DESIGN.md proposed six specific metrics (`dag.nodes_ready`, `conductor.act_queue_depth`, etc.) tied to its `networkx` DAG model. The ADR architecture has no global DAG, so equivalent metrics would be plan-level (nodes per plan, buffer count) and renderer-level (kernel execution time, transfer latency). Defining these is a separate effort.

---

### Choice 7: Migration positioning — DECIDED

> **Decision (2026-03-28): Option 7C — Parallel workstream (Phase 4-adjacent).**

Where does the user-facing API fit in the ADR-017 migration path?

#### Rejected: Option 7A — Phase 1 extension

The user-facing API is defined alongside the plan model in Phase 1. Plan builder is designed from the start to support Act-only, Learn-only, and combined plans.

**Reasons for rejection:**
- Phase 1 scope increases significantly. The plan builder must handle three plan construction modes (Act-only, Learn-only, combined) before any backend renderer exists to test against. This means debugging the plan builder's fundamentals and the ticket API's plan-splitting logic simultaneously.
- The plan builder's API surface is already substantially constrained by ADRs 001–010. User-facing requirements do not need to be co-designed — they can be layered on once the plan builder's core API stabilizes.

#### Rejected: Option 7B — New Phase 7 (post-migration)

The user-facing API is a post-migration deliverable. Phases 0–6 complete the multi-backend refactoring with the existing synchronous `train()` API. Phase 7 introduces the ticket model on top of the proven plan-model infrastructure.

**Reasons for rejection:**
- Delays the Event-Triggered execution mode — the architecture's most distinctive capability — until all six migration phases complete. The "Real-Time Trader" scenario is one of three canonical CONCEPT.md validation cases; deferring it to post-migration deprioritizes a core architectural identity.
- The existing `train()` API continues to work throughout Phases 0–6 (a genuine advantage), but this is equally true under Option 7C — the ticket API develops behind its own feature flag without disturbing the existing surface.
- Lower risk in isolation, but the delay cost is disproportionate. The plan model's contract surface is already heavily constrained by ADRs 001–010; the risk of building against it before Phase 6 completes is low.

#### Option 7C: Parallel workstream (Phase 4-adjacent) ✓ DECIDED

The user-facing API develops in parallel with the test harness (Phase 4), consuming plan builder and renderer interfaces as they stabilize.

#### Rationale

- **ADR-017 already chose feature-flag gated parallel development (Option C).** The migration strategy explicitly enables independent workstreams behind `_build_config.py` flags. The ticket API slots naturally into this model — its own flag, independently toggleable, testable against the plan builder as it stabilizes. Using the infrastructure that's already been decided for its intended purpose.

- **The plan builder's contract surface is substantially determined.** Ten accepted ADRs (001–010) constrain the five-node taxonomy, buffer lifecycle, RetrievalFuture protocol, and rendering interface. The remaining design freedom in Phase 1 is narrow. Building the ticket layer against this contract surface is low-risk.

- **Cross-pollination with the test harness (Phase 4).** Both the test harness and the ticket API sit above the plan model and consume the `PlanRenderer` interface. Developing them in parallel provides mutual validation: test scenarios stress the ticket's plan-construction patterns, and the ticket API provides realistic plan builder usage that exercises the harness.

- **Act-only and Learn-only plan construction is an additive extension.** Phase 1 delivers combined Act+Learn plans first (the existing `TrainingOrchestrator.train()` pattern). The ticket workstream adds split-plan construction modes once the plan builder's core API stabilizes. If the ticket work slips, no other phase is blocked — the existing `train()` API continues to function with the feature flag off.

- **Event-Triggered mode is available earlier.** Users can exercise the temporal Act/Learn decoupling as soon as the ticket workstream's feature flag is promoted, rather than waiting for all migration phases to complete.

#### Risk

Option 7C requires the plan builder's interface to stabilize early enough that the ticket workstream can build against it. If Phase 1's plan builder API churns significantly, the ticket work is rework-prone. This risk is mitigated by the heavy constraint that ADRs 001–010 place on the plan builder's shape, but should be monitored: if Phase 1 introduces unexpected API revisions, the ticket workstream should pause until the interface re-stabilizes rather than building against a moving target.

---

## Interaction Map

The choices are not independent. Key interactions:

| Choice | Interacts With | Nature of Interaction |
| :--- | :--- | :--- |
| **1 (Unit of intent)** | **4 (Buffer lifecycle)** | **Resolved (both decided).** With 4A (recompute) and 1A (ticket) decided, the ticket holds no device-side state. All Act-plan buffers are released unconditionally when `get_prediction()` caches the result. The ticket's arbitrary lifetime in ACT_COMPLETE state carries zero VRAM cost. |
| **1 (Unit of intent)** | **2 (Concurrency)** | **Resolved (both decided).** With 1A (ticket) and 2B (future-based) decided, the ticket's lifecycle methods are non-blocking dispatchers (`submit`, `resolve` return immediately) and the user controls when to block (`get_prediction`, `learn_handle.wait`). The ticket's four-state machine is fully observable. Pipeline parallelism — overlapping Act of item N+1 with Learn of item N — is expressible without threads. |
| **1 (Unit of intent)** | **3 (Batch composition)** | **Resolved (both decided).** With 1A (ticket) and 3A (explicit batch) decided, one `submit()` produces one ticket representing one batch. The ticket's Act→Learn association operates at the batch level. The engine dispatches one plan per `submit()` call — no accumulation, no deferred dispatch. |
| **1 (Unit of intent)** | **5 (Data persistence)** | **Resolved (both decided).** With 1A (ticket) and 5A (ephemeral) decided, tickets are transient — consumed after a single Act+Learn cycle. No registry, no `retire()`, no engine-side state between calls. The ticket's lifecycle terminates at CONSUMED. |
| **1 (Unit of intent)** | **6 (Observability)** | The `LearnHandle` returned by `ticket.resolve()` is a natural carrier for per-step diagnostics (Option 6B). Option 6A ignores this; Option 6C (callbacks) operates orthogonally to the ticket model. |
| **2 (Concurrency)** | **4 (Buffer lifecycle)** | **Resolved (both decided).** With 2B (future-based) and 4A (recompute) decided, overlapping Act of item N+1 with Learn of item N is straightforward: each plan's buffers are self-contained and independently allocated/freed. Abandoned tickets cost only host-side memory (no VRAM leak). |
| **2 (Concurrency)** | **7 (Migration)** | **Resolved (both decided).** With 2B (future-based) and 7C (parallel workstream) decided, the concurrency model requires no new framework dependency and no adapter layer. The ticket workstream develops behind its own feature flag, consuming the plan builder's interface as it stabilizes. 2B can be implemented with the same synchronous plan-rendering internals used today — no migration-phase dependency on async infrastructure. |
| **3 (Batch composition)** | **5 (Data persistence)** | **Resolved (both decided).** 3A (explicit batch) and 5A (ephemeral) are natural partners: the user owns the data and the loop, the engine is stateless between calls. The rejected pairings (3C+5C, 3B+5B) would have introduced engine-side state accumulation. |
| **4 (Buffer lifecycle)** | **ADR-009** | **Resolved.** Option 4A (recompute) is decided. No ADR-009 changes required. |
| **6 (Observability)** | **ADR-002** | Options 6B/6C may require additional `RetrievalNode`s in the plan (e.g., for per-batch loss retrieval). This is an additive change to plan construction, not a node taxonomy change. |

---

## Degenerate Case Validation

The decided combination (Choices 1A + 2B + 3A + 4A + 5A) must handle the three canonical scenarios from CONCEPT.md. All remaining open choices (6, 7) must preserve these properties regardless of which options are selected.

### Scenario: The Iris Case (Sequential Mode)

Pre-labeled batch of 150 items.

```python
engine = Engine(model_spec, hyperparams)
for epoch in range(num_epochs):
    result = engine.train_batch(X_train, y_train)
print(result.predictions)
```

3 lines of user code (excluding engine construction). The convenience method wraps the ticket lifecycle internally. **Under 5-line ceremony threshold.** ✓

### Scenario: The Real-Time Trader (Event-Triggered Mode)

Input data arrives; inference must be immediate. Ground truth arrives later.

```python
ticket = engine.submit(market_snapshot)        # returns immediately; Act plan dispatched
prediction = ticket.get_prediction()           # blocks until Act completes
execute_trade(prediction)

# ... seconds to minutes pass ...

learn_handle = ticket.resolve(actual_outcome)  # returns immediately; Learn plan dispatched
learn_handle.wait()                            # blocks until Learn completes
```

The ticket captures `market_snapshot` at submit time. Even if the original array is subsequently overwritten, the ticket holds its own reference. **System-enforced Act→Learn association (Choice 1A).** ✓

`submit()` returns immediately; the trader can perform pre-inference work while the Act plan executes on-device. `resolve()` returns immediately; the trader can continue while the Learn plan runs. **Non-blocking dispatch (Choice 2B).** ✓

All device buffers are released when `get_prediction()` returns. No VRAM is held during the temporal gap. **Minimal VRAM footprint (Choice 4A).** ✓

### Scenario: The Data Tsunami (Massive Batch)

N = 10,000+ items in a single training batch.

```python
ticket = engine.submit(X_massive)              # returns immediately; Act plan dispatched
probs = ticket.get_prediction()                # blocks until Act completes; (N × C) matrix
learn_handle = ticket.resolve(y_massive)       # returns immediately; Learn plan dispatched
learn_handle.wait()                            # reduction tree handles log_K(N) aggregation
```

One ticket, one Act plan, one Learn plan. No per-item overhead. The plan's `ReductionTreeNode` handles the aggregation. **Batch-efficient.** ✓

The user can submit the massive batch and perform host-side work (e.g., preparing the next batch or computing metrics on a previous result) while the Act plan executes. **Non-blocking dispatch (Choice 2B).** ✓

---

## References

- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §4 Asynchronous Host Interaction; §5 Unified Execution Model; Validation Scenarios (Iris, Real-Time Trader, Data Tsunami)
- [CONTRACT.md](../CONTRACT.md) — Article 1.1 Jurisdictional Separation; Article 1.4 Collaborative Interface Verifiability
- [DESIGN.md](../DESIGN.md) — WorkTicket / LearnTicket vision; AsyncConductor proposal; three canonical scenarios (historical context, not binding)
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; PlanRenderer interface; no backend types in shared layer
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — five-node closed taxonomy; RetrievalNode for host-observable points
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — single-plan buffer lifetime model; BufferDescriptor with producing_node/last_consumer; two-tier buffer scope
- [ADR-010: D2H Transfer & Phase Sync Points](ADR-010-d2h-transfer-and-phase-sync-points.md) — RetrievalFuture Protocol; Sequential and Event-Triggered consumption patterns; split-plan pattern for Event-Triggered mode
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` + `src/backends/<name>/` directory structure; shared-layer module inventory
- [ADR-017: Incremental Migration Path](ADR-017-incremental-migration-path.md) — Phase 0–6 structure; feature-flag gated sequencing; tier-based rollback gates
