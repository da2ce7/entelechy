# ADR-018: User-Facing API

**Status:** PROPOSED  
**Date:** 2026-03-10  
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

DESIGN.md (December 2025) proposed a `WorkTicket` / `LearnTicket` API with a global mutable `networkx.DiGraph` and an `AsyncConductor` scanning loop. The subsequent ADR work chose a different internal architecture (immutable plans, backend-owned rendering, no `networkx`, no `asyncio` conductor). The `WorkTicket` *concept* — a persistent promise representing a unit of user intent, with temporal decoupling between inference and learning — has architectural merit independent of the internal execution model it was originally proposed for. This ADR examines how such a concept could be realized on top of the existing plan-model architecture, without presupposing the DESIGN.md implementation.

### What this ADR must decide

This ADR identifies the design choices that determine the user-facing API surface. Each choice is presented with its options, trade-offs, and interactions with the existing ADR chain. No option is selected; the decision is deferred to a future revision.

---

## Decision Drivers

1. **CONCEPT.md §4 (Asynchronous Host Interaction).** The `inference_event` and `final_batch_event` are named synchronization points. The API must expose both Sequential and Event-Triggered execution modes. The choice of API idiom determines how naturally Event-Triggered mode is expressed.

2. **CONCEPT.md §5 (Unified Execution Model).** All workflows follow Act/Learn sequencing regardless of backend. The API must not bifurcate into separate "batch training" and "online inference" surfaces — the same primitives must serve both modes.

3. **CONCEPT.md §1 (Architectural Elegance Feedback).** If the user-facing API creates tension with the plan model's contracts, the response is to extend the architecture — not to expose plan-model internals as user-visible leaky abstractions.

4. **ADR-001 (Backend Abstraction Boundary).** The user-facing API must not expose any backend-specific type. No `cl.Event`, `VkFence`, or backend renderer appears in the user's import surface.

5. **ADR-002 (Plan Node Types).** The five-node taxonomy is a closed set. The user-facing API may produce different plan *configurations* (Act-only, Learn-only, combined) but must not require new plan node types.

6. **ADR-009 (Buffer Lifecycle).** Each plan's `BufferDescriptor` namespace is self-contained. If the API permits Act plans and Learn plans to be submitted independently, cross-plan buffer persistence introduces a new lifecycle concern that ADR-009's current model does not address.

7. **ADR-010 (D2H Transfer & Sync Points).** The `RetrievalFuture` Protocol is the sole host-facing type for observing device results. The user-facing API builds on top of this — it does not replace it or wrap it unnecessarily.

8. **ADR-012 (Module Factoring).** ADR-012 defines `src/shared/` and `src/backends/<name>/`. The user-facing API module(s) belong in the shared layer. Their location and relationship to the existing `orchestrator.py` (the renamed `main_orchestrator.py`) must be resolved.

9. **ADR-017 (Migration Path).** The migration is organized into Phases 0–6. The user-facing API is not addressed by any existing phase. Its introduction must be positioned relative to the existing phase structure — either as an extension of an existing phase or as a new Phase 7.

10. **CONCEPT.md Validation Scenarios.** The "Iris Case" (Sequential), "Real-Time Trader" (Event-Triggered), and "Data Tsunami" (massive batch) scenarios are canonical. The API must handle all three without mode-specific branching in application code.

---

## Design Choices

### Choice 1: Unit of user intent

The fundamental question: what object does the user hold that represents "I have data and I want the system to do something with it"?

#### Option 1A: Stateful ticket (WorkTicket pattern)

The user receives a mutable, stateful object whose lifecycle mirrors the Act/Learn temporal split:

```python
ticket = engine.submit(x_data)                    # → WorkTicket (PENDING)
probs = ticket.get_prediction()                    # blocks until Act completes
learn_handle = ticket.resolve(y_data)              # → LearnHandle (LEARNING)
learn_handle.wait()                                # blocks until Learn completes
```

The ticket transitions through states: `PENDING → ACT_COMPLETE → RESOLVED → CONSUMED`. The engine manages the mapping from ticket state to plan construction and rendering.

**Considerations:**
- Directly models the temporal gap between inference and learning.
- The ticket object becomes the natural anchor for per-item operations: cancel, retire, inspect.
- Introduces mutable state in the user-facing layer, which contrasts with the immutable plan model underneath.
- Must define what happens to device-side resources (buffers, intermediate activations) while a ticket is in the `ACT_COMPLETE` state awaiting resolution. This interacts with ADR-009 (buffer lifecycle).

#### Option 1B: Explicit two-call API

The user explicitly constructs and submits Act and Learn requests as separate operations:

```python
act_result = engine.infer(x_data)                  # → ActResult (contains RetrievalFuture)
probs = act_result.get_probabilities()             # blocks until Act completes
engine.learn(x_data, y_data)                       # blocks until Learn completes
```

No persistent ticket object. Each call is independent. The engine's internal state (e.g., whether to cache or recompute activations) is configured at engine construction time, not per-call.

**Considerations:**
- Simpler mental model — no state machine, no ticket lifecycle.
- Directly mirrors the ADR-010 split-plan pattern (`render(act_plan)` then `render(learn_plan)`).
- The user must manually associate `infer()` and `learn()` calls for the same data. There is no system-enforced linkage between an inference and its subsequent learning step.
- Batch training (Scenario A) requires the user to call both `infer()` and `learn()` for every batch — more ceremony than the current `train()` API.

#### Option 1C: Plan-level API (direct plan exposure)

The user constructs or requests execution plans and submits them to a renderer:

```python
act_plan = engine.build_act_plan(x_data)
futures = engine.render(act_plan)
probs = futures["inference_retrieval"].result()

learn_plan = engine.build_learn_plan(x_data, y_data)
futures = engine.render(learn_plan)
futures["final_batch_retrieval"].wait()
```

**Considerations:**
- Maximum transparency — the user sees exactly what the system will do.
- Enables advanced use cases: plan inspection, plan caching, plan serialization.
- Exposes plan-model vocabulary (`RetrievalNode` names, `RetrievalFuture` Protocol) to the user. This may violate the principle of keeping the plan model as an internal contract.
- Highest ceremony for simple use cases. The "Iris Case" (submit batch, get results) becomes multi-step.

---

### Choice 2: Concurrency model

How does the user-facing API relate to Python's concurrency primitives?

#### Option 2A: Synchronous (blocking)

All user-facing methods block until their results are available. Event-Triggered mode is expressed by calling `infer()` (blocks until Act completes), then later calling `learn()` (blocks until Learn completes).

```python
probs = engine.infer(x_data)          # blocks
# ... time passes, ground truth arrives ...
engine.learn(x_data, y_data)           # blocks
```

**Considerations:**
- Simplest to implement and reason about.
- Compatible with all Python environments — no `asyncio` event loop required.
- The "Real-Time Trader" scenario is expressible: the host blocks on `infer()`, processes the result, optionally blocks on `learn()`. But the host cannot overlap inference of item N+1 with learning of item N without threading.
- Multi-item pipeline parallelism (infer item 2 while learning item 1) requires the user to manage threads manually.

#### Option 2B: Future-based (non-blocking, sync-compatible)

User-facing methods return `Future`-like handles. The user decides when to block.

```python
infer_handle = engine.submit_infer(x_data)    # non-blocking, returns handle
# ... do other work ...
probs = infer_handle.result()                  # blocks when needed

learn_handle = engine.submit_learn(x_data, y_data)
learn_handle.wait()
```

**Considerations:**
- The `RetrievalFuture` Protocol (ADR-010) already provides this pattern at the renderer level. Option 2B lifts it to the user-facing level.
- Enables pipeline parallelism without threads: the user holds multiple outstanding handles.
- Does not require `asyncio`. Compatible with synchronous code — calling `.result()` immediately degenerates to Option 2A.
- The "Real-Time Trader" can hold an inference handle, process results, then submit learning.
- Handle lifecycle management falls on the user. Forgetting to `.wait()` or `.release()` a handle is a resource leak. ADR-010's `release()` semantics propagate to the user surface.

#### Option 2C: Native asyncio

User-facing methods are coroutines. The API is designed for use within an `asyncio` event loop.

```python
probs = await engine.infer(x_data)
# ... await external event ...
await engine.learn(x_data, y_data)
```

**Considerations:**
- Most natural for Event-Triggered workflows — `await` directly models "do this, then wait for something else, then do that."
- Enables high-throughput pipelines: `asyncio.gather()` for multiple inferences, `asyncio.Queue` for streaming ground-truth arrival.
- Requires the user to run an `asyncio` event loop. This is a hard dependency on a specific concurrency framework.
- The current codebase has zero `asyncio` usage. The `RetrievalFuture` Protocol (ADR-010) is synchronous (`.wait()` blocks). Bridging `RetrievalFuture` into `asyncio` requires an adapter (analogous to DESIGN.md's `awaitable_cl_event`, but at the Future level rather than the OpenCL level).
- Excludes users who cannot or do not want to use `asyncio` (Jupyter notebooks with synchronous cells, simple scripts, integration into non-async frameworks).

#### Option 2D: Dual-mode (sync + async)

Provide both synchronous and asynchronous entry points. The sync API blocks; the async API returns awaitables. Both dispatch through the same plan builder and renderer.

```python
# Sync
probs = engine.infer(x_data)

# Async
probs = await engine.infer_async(x_data)
```

**Considerations:**
- Maximum compatibility — every user can use the API they prefer.
- Implementation complexity: two entry points per operation, potentially with subtle behavioral differences (e.g., does `infer()` hold the GIL while waiting? does `infer_async()` release it?).
- Risk of a fractured ecosystem where examples, documentation, and community code use inconsistent styles.

---

### Choice 3: Batch composition model

How does the user express "train on these N items as a single batch"?

#### Option 3A: Explicit batch submission

The user constructs a batch and submits it as a unit:

```python
engine.train_batch(X_batch, y_batch)    # N items, one reduction tree, one update
```

The engine internally constructs a plan with N gradient subgraphs → `ReductionTreeNode` → `ParameterUpdateNode`.

**Considerations:**
- Simplest for the common case — bulk supervised training.
- The user is responsible for batching. The engine does not accumulate items.
- Aligns with the current `orchestrator.train()` pattern (one batch → one plan → one render).
- Does not naturally integrate with the ticket model (Choice 1A) where items arrive individually.

#### Option 3B: Ticket accumulation with explicit flush

The user submits items individually. The engine accumulates them. An explicit `flush()` triggers plan construction and rendering for the accumulated batch.

```python
for x, y in stream:
    engine.submit(x, y)
engine.flush()          # constructs plan for all accumulated items, renders, waits
```

**Considerations:**
- Natural for streaming data arrival.
- The engine must maintain an accumulation buffer — a staging area for unprocessed items.
- `flush()` semantics must be defined: does it block? return a handle? What happens if new items arrive during flush?
- Interacts with Choice 1: if tickets are stateful (1A), each submitted item has a ticket; `flush()` resolves all pending tickets. If the API is two-call (1B), accumulation is engine-internal.

#### Option 3C: Automatic batching with configurable policy

The engine automatically batches items according to a policy (e.g., batch size threshold, time window, or explicit trigger):

```python
engine = Engine(batch_policy=BatchWhenFull(size=64))
for x, y in stream:
    engine.submit(x, y)    # triggers training when 64 items accumulate
```

**Considerations:**
- Most convenient for streaming workloads — the user never manually batches.
- The batching policy is a new configuration dimension. Defining and composing policies (size-based, time-based, manual trigger) is a design effort in itself.
- Reduces user control over exactly when training occurs. The user cannot easily say "train on exactly these 37 items right now."
- Policy-triggered plan construction happens asynchronously from the user's perspective — even in a synchronous API, the `submit()` call may or may not trigger a training step.

---

### Choice 4: Cross-plan buffer lifecycle

When Act and Learn plans are submitted independently (any option in Choice 1 that permits temporal separation), intermediate device-side buffers from the Act plan may be needed by the subsequent Learn plan. How is this managed?

#### Option 4A: Recompute (no cross-plan sharing)

The Learn plan always recomputes any intermediate values it needs. The Act plan's device buffers are released when the Act `RetrievalFuture` is released. No buffer persists across plan boundaries.

**Considerations:**
- No extension to ADR-009 required. Each plan's `BufferDescriptor` namespace is fully self-contained.
- Redundant computation: the forward pass runs twice (once in Act, once in Learn).
- CONCEPT.md's "Real-Time Trader" scenario explicitly validates this approach — it "assesses the system's ability to release VRAM after the Act phase and efficiently recompute intermediates for the Learn phase."
- Device memory is freed promptly after Act completion. Minimum VRAM footprint between phases.

#### Option 4B: Cached (cross-plan buffer retention)

The Act plan's `hidden_activations` buffer is retained on the device after Act completes. The Learn plan's buffer descriptors reference the existing allocation. A protocol governs when the cached buffer is released.

**Considerations:**
- Avoids redundant forward-pass computation in Learn. Faster Learn-phase execution for large models.
- Requires extending ADR-009: buffer lifetimes can now span multiple plans. The `producing_node` and `last_consumer` model assumes a single plan scope. A new concept — "buffer retained by host intent" — must be formalized.
- Device memory is held between Act and Learn submissions. If the temporal gap is long (minutes, hours in an event-driven system), VRAM is consumed with no active computation. This conflicts with adaptive memory management (CONCEPT.md "Grace of Adaptive Memory").
- Must define a timeout or eviction policy: if ground truth never arrives, cached buffers must eventually be released.
- Must define what happens if model parameters are updated (by another batch's Learn) between this item's Act and Learn — the cached activations are stale.

#### Option 4C: User-controlled (explicit cache/recompute per submission)

The user specifies per-submission whether to cache or recompute:

```python
ticket = engine.submit(x_data, cache_activations=True)
# Act plan retains hidden_activations
# ...
ticket.resolve(y_data)      # Learn plan reuses cached activations
```

**Considerations:**
- Maximum user control — the user knows their latency/memory trade-off.
- Pushes an internal architectural decision (cache vs. recompute) to the user surface. Users must understand what "activations" are and why caching them matters.
- Interaction with Choice 1: only Option 1A (tickets) provides a natural anchor for per-item cache decisions. Options 1B and 1C would need separate mechanism.

---

### Choice 5: Data persistence and epoch iteration

Does the system remember previously submitted data across epoch boundaries?

#### Option 5A: Ephemeral (no persistence)

Each submission is a one-shot event. The system processes it and forgets it. For multi-epoch training, the user resubmits the full dataset each epoch.

```python
for epoch in range(num_epochs):
    engine.train_batch(X_train, y_train)
```

**Considerations:**
- Simplest implementation — no registry, no lifecycle management.
- Matches the current `TrainingOrchestrator.train()` pattern exactly.
- Requires the user to own the training loop and data management.
- Every epoch re-uploads data to the device (H2D transfer cost). For large datasets, this is significant.

#### Option 5B: Persistent registry (LearnTicket pattern)

Resolved tickets persist in a registry. Each epoch, the engine re-plans and re-renders all active tickets. The user can retire individual tickets to remove their data from future training.

```python
ticket = engine.submit(x_data)
learn = ticket.resolve(y_data)        # (x, y) pair persists
# ... many epochs later ...
learn.retire()                         # data removed from future training
```

**Considerations:**
- Enables the "Interactive Scientist" scenario — individual data points can be added or removed mid-training.
- The registry is a new stateful component in the shared layer. It must track all active `(x, y)` pairs and include them in each epoch's plan.
- Device-side data management becomes non-trivial: are `(x, y)` pairs kept on device permanently? Re-uploaded each epoch? Managed by a device-side data pool?
- Plan size grows linearly with the number of active tickets. For large persistent datasets, plan construction time and plan size may become bottlenecks.
- `retire()` semantics during an in-flight Learn plan must be defined — does retirement take effect immediately (mid-batch) or at the next epoch boundary?

#### Option 5C: Dataset abstraction (engine-managed data)

The engine manages a mutable dataset object. The user adds/removes items. Epoch iteration is engine-controlled.

```python
ds = engine.create_dataset()
ds.add(x1, y1)
ds.add(x2, y2)
engine.train(ds, epochs=10)            # engine iterates internally
ds.remove(item_id=0)
engine.train(ds, epochs=5)             # continues without removed item
```

**Considerations:**
- Clean separation: the dataset is the user's data management surface; the engine is the training surface.
- The dataset can optimize device-side storage (persistent device buffers, incremental uploads).
- Introduces a new abstraction (`Dataset`) not currently in the architecture. Its relationship to `ParameterSpace`, `ModelSpec`, and the plan builder must be defined.
- Training loop ownership moves to the engine. Users who want custom per-epoch logic (learning rate scheduling, validation, early stopping) need hooks or callbacks.

---

### Choice 6: Observability surface

What runtime information does the system expose to the user during and after training?

#### Option 6A: Minimal (results only)

The user receives prediction probabilities and learns when training completes. No internal metrics are exposed.

**Considerations:**
- Simplest to implement and maintain.
- Users who need loss values, gradient norms, or per-epoch statistics must compute them externally from the returned probabilities and known labels.
- Sufficient for the current Iris-scale validation scenarios.

#### Option 6B: Per-batch diagnostics

Each training step returns a diagnostics object containing: aggregated loss, effective batch size, and any values already computed on-device (via the diagnostic aggregation tree — Node 14).

```python
result = engine.train_batch(X, y)
print(result.loss, result.effective_batch_size)
```

**Considerations:**
- The diagnostic aggregation tree (Node 14) already computes aggregated probabilities and loss on-device. Exposing these to the user requires only an additional `RetrievalNode` in the plan and corresponding `RetrievalFuture` in the return value.
- Does not require new device-side computation — these values are already produced.
- The set of available diagnostics is fixed by the plan's `RetrievalNode` inventory. Adding new diagnostics requires plan-level changes.

#### Option 6C: Structured metrics with callback/streaming

The engine exposes a metrics stream or accepts callbacks for real-time monitoring:

```python
engine.on_batch_complete(lambda metrics: wandb.log(metrics))
engine.train(ds, epochs=10)
```

**Considerations:**
- Most flexible for integration with experiment tracking systems (Weights & Biases, TensorBoard, MLflow).
- Callback invocation timing and threading semantics must be defined. Does the callback run on the main thread? On a background thread? Within an `asyncio` task?
- The callback interface must be backend-agnostic — no backend type may appear in the metrics object.
- DESIGN.md proposed six specific metrics (`dag.nodes_ready`, `conductor.act_queue_depth`, etc.) tied to its `networkx` DAG model. The ADR architecture has no global DAG, so equivalent metrics would be plan-level (nodes per plan, buffer count) and renderer-level (kernel execution time, transfer latency). Defining these is a separate effort.

---

### Choice 7: Migration positioning

Where does the user-facing API fit in the ADR-017 migration path?

#### Option 7A: Phase 1 extension

The user-facing API is defined alongside the plan model in Phase 1. Plan builder is designed from the start to support Act-only, Learn-only, and combined plans.

**Considerations:**
- The plan builder's API surface is influenced by user-facing requirements from the start. No retrofitting.
- Phase 1 scope increases significantly. Plan builder must handle three plan modes before any backend renderer exists to test against.

#### Option 7B: New Phase 7 (post-migration)

The user-facing API is a post-migration deliverable. Phases 0–6 complete the multi-backend refactoring with the existing synchronous `train()` API. Phase 7 introduces the new user surface on top of the proven plan-model infrastructure.

**Considerations:**
- Lower risk — the plan model and renderers are battle-tested before the user API is layered on.
- The existing `train()` API continues to work throughout Phases 0–6. No user-visible API break during migration.
- Delays the Event-Triggered execution mode's availability. Users cannot use split-phase workflows until Phase 7.

#### Option 7C: Parallel workstream (Phase 4-adjacent)

The user-facing API develops in parallel with the test harness (Phase 4), consuming plan builder and renderer interfaces as they stabilize.

**Considerations:**
- The test harness (Phase 4) and the user-facing API both sit above the plan model. Developing them in parallel enables cross-pollination — test scenarios validate the user API, and the user API tests stress the plan builder.
- Requires the plan builder's interface to be stable enough to build against. Same dependency as Phases 2/3/5 on Phase 1.
- Option C (feature-flag gated) from ADR-017 naturally accommodates this: the user API is behind its own feature flag, independently toggleable.

---

## Interaction Map

The choices are not independent. Key interactions:

| Choice | Interacts With | Nature of Interaction |
| :--- | :--- | :--- |
| **1 (Unit of intent)** | **4 (Buffer lifecycle)** | Only Option 1A (tickets) provides a natural per-item anchor for cache/recompute decisions. Option 1B requires a separate mechanism. |
| **1 (Unit of intent)** | **5 (Data persistence)** | Option 5B (persistent registry) is most natural with Option 1A (tickets). Option 5A (ephemeral) is most natural with Option 1B (two-call). |
| **1 (Unit of intent)** | **3 (Batch composition)** | Option 3B (accumulation with flush) requires either tickets (1A) or engine-internal staging. Option 3A (explicit batch) works with any Choice 1 option. |
| **2 (Concurrency)** | **4 (Buffer lifecycle)** | Async concurrency (2C/2D) enables overlapping Act of item N+1 with Learn of item N — but this requires either Option 4B (cached buffers for item N's Act) or Option 4A (recompute for item N's Learn) to coexist with item N+1's Act. The interaction between concurrent items' buffer lifetimes is non-trivial. |
| **2 (Concurrency)** | **7 (Migration)** | Option 2C (native asyncio) is a significant implementation and represents a larger scope expansion than 2A/2B, affecting migration positioning. |
| **3 (Batch composition)** | **5 (Data persistence)** | Option 3C (automatic batching) combines naturally with Option 5C (dataset abstraction). Option 3A (explicit batch) combines naturally with Option 5A (ephemeral). |
| **4 (Buffer lifecycle)** | **ADR-009** | Option 4B (cached) requires an extension to ADR-009's single-plan buffer lifetime model. Option 4A (recompute) requires no ADR-009 changes. |
| **6 (Observability)** | **ADR-002** | Options 6B/6C may require additional `RetrievalNode`s in the plan (e.g., for per-batch loss retrieval). This is an additive change to plan construction, not a node taxonomy change. |

---

## Degenerate Case Validation

Any chosen combination must handle the three canonical scenarios from CONCEPT.md:

### Scenario: The Iris Case (Sequential Mode)

- Pre-labeled batch of 150 items.
- The user submits `(X, y)` and receives final probabilities and a "training complete" signal.
- Minimum ceremony — if the API requires more than 5 lines for this case, it is over-engineered.

### Scenario: The Real-Time Trader (Event-Triggered Mode)

- Input data arrives. Inference must be immediate.
- Ground truth arrives later (seconds to minutes).
- The API must support: submit `x` → get prediction → (time passes) → submit `y` → learning happens.
- Must not require the user to manually manage device buffers between Act and Learn.

### Scenario: The Data Tsunami (Massive Batch)

- N = 10,000+ items in a single training batch.
- The plan's `ReductionTreeNode` handles the `log_K(N)` aggregation.
- The API must allow the user to submit a large batch without per-item overhead dominating.

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
