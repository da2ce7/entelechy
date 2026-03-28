# ADR-001: Backend Abstraction Boundary

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
**Blocks:** ADR-002, ADR-003, ADR-004, ADR-005, ADR-006, ADR-010, ADR-011, ADR-012, ADR-016, ADR-018

---

## Context

The current implementation is pervasively coupled to `pyopencl`. The types `cl.Event`, `cl.CommandQueue`, `cl.Buffer`, and `cl.Context` appear in the signatures of nearly every host-side module—`execution_plan.py`, `graph_recipes.py`, `batch_processor.py`, `launcher_infra.py`, and `compute_patterns.py`. Adding a CPU or Vulkan backend cannot be a localized change; it requires defining a formal abstraction boundary.

### Coupling inventory

| Module               | OpenCL types in public signatures                                          |
| :------------------- | :------------------------------------------------------------------------- |
| `launcher_infra.py`  | `Services.q: cl.CommandQueue`; `BufferManager.__init__(cl.Context)`;       |
|                      | `BufferManager.get_cl_buffer() → cl.Buffer`; `KernelExecutor.launch() → cl.Event` |
| `execution_plan.py`  | `DependencyProvider.resolve(cl.CommandQueue, ..., List[cl.Event]) → (BufferHandle, cl.Event)` |
|                      | `CacheProvider.ready_event: cl.Event`; `ComputeOnceProvider._cached_event: cl.Event` |
| `graph_recipes.py`   | Every recipe accepts `List[cl.Event]` and returns `cl.Event`               |
| `batch_processor.py` | `BatchProcessor.run() → Tuple[cl.Event, cl.Event, HostView]`              |

### Backend execution model divergence

The three target backends have fundamentally different execution semantics:

| Property              | OpenCL                           | Vulkan                                          | CPU                                  |
| :-------------------- | :------------------------------- | :---------------------------------------------- | :----------------------------------- |
| Dispatch model        | Per-tile `clEnqueueNDRange`      | Single `vkCmdDispatch(N, 1, 1)` for N tiles     | `pool_dispatch_and_wait` per node    |
| Sync primitive        | `cl_event` (per-enqueue token)   | `VkPipelineBarrier` (structural) + `VkFence` (host) | Function return (synchronous)     |
| Dispatch granularity  | Host loops over tiles            | GPU hardware scheduler distributes workgroups    | Thread pool claims tasks atomically  |
| Command model         | Imperative enqueue-and-execute   | Record ≠ execute (command buffers)               | Direct function calls                |
| Tile index            | Host scalar `flat_tile_index`    | `gl_WorkGroupID.x` (implicit)                   | `task_index` parameter               |

The core tension: these are not superficial API differences. They represent structurally distinct execution paradigms, particularly the OpenCL imperative "enqueue-and-execute" model vs. Vulkan's deferred "record-then-submit" model vs. CPU's synchronous "call-and-return" model.

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback):** Optimization pressure that violates core principles is signal for incomplete architectural modeling—not justification for exceptions. Forcing Vulkan into OpenCL's imperative enqueue model violates this principle.

2. **CONCEPT.md §5 (Unified Execution Model):** All workflows follow Act/Learn sequencing regardless of backend. This mandates a shared orchestration layer expressing the phase structure.

3. **CONCEPT.md §4 (Asynchronous Host Interaction):** The `inference_event` and `final_batch_event` are named synchronization points that must be expressible in the shared layer.

4. **CONTRACT.md Article 1.1 (Jurisdictional Separation):** Syntactic structure (dispatch mechanics) and semantic contracts (DAG dependencies) are distinct jurisdictions. The abstraction boundary must separate them.

5. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability):** All validation occurs pre-dispatch. Validation logic is inherently backend-neutral and must not be duplicated.

6. **Performance fidelity:** Each backend's native execution model provides optimizations that a lowest-common-denominator abstraction would forfeit. Vulkan's single-dispatch-for-N-tiles and command buffer batching; CPU's zero-copy zero-transfer direct memory; OpenCL's implicit driver scheduling.

---

## Options Considered

### Option A: Low-level — Abstract the dispatch/sync primitives

Define abstract `Event`, `Queue`, `Buffer` types. Each module (`graph_recipes.py`, `compute_patterns.py`) continues to compose individual kernel launches against abstract types.

**Advantages:**
- Smallest code diff from current state.
- Preserves the current per-tile imperative dispatch model.

**Disadvantages:**
- Forces Vulkan into an emulation of the imperative enqueue model, defeating its batched-recording advantage. Vulkan's single `vkCmdDispatch(N, 1, 1)` would be split into N host-side calls, forfeiting the entire point of the Vulkan backend.
- The abstract `Event` type cannot faithfully represent Vulkan pipeline barriers, which are structural commands in a recording—not returnable per-dispatch tokens.
- CPU's synchronous `pool_dispatch_and_wait` model would require wrapping every return value in a trivially-resolved event object, imposing conceptual overhead with no benefit.
- Violates CONCEPT.md §1: forcing backend-incompatible semantics through a thin type alias is precisely the kind of contract compromise that Architectural Elegance Feedback warns against.

### Option B: High-level — Abstract at the DAG/Phase level

The shared orchestration layer (`execution_plan.py`, `batch_processor.py`) produces an abstract execution plan describing the full DAG structure. Each backend receives the complete plan and renders it natively—OpenCL enqueues imperatively, Vulkan records a command buffer, CPU calls `pool_dispatch_and_wait` in sequence.

**Advantages:**
- Faithfully honors each backend's native execution model. Vulkan records its loops at command-buffer time; OpenCL enqueues imperatively; CPU calls synchronously.
- Naturally separates "what to compute" (shared) from "how to dispatch" (backend).
- Aligns with the Vulkan backend document's architecture, which already shows the host orchestrator producing a plan that the backend renders.
- The DAG plan is a first-class, inspectable data structure—enabling future optimizations like plan caching, cross-backend plan comparison, and offline validation.

**Disadvantages:**
- Requires the largest refactoring of `graph_recipes.py`—from "execute this now" to "describe this for later execution." The current 984-line module mixes DAG logic and OpenCL dispatch.
- The plan representation must be expressive enough to capture streaming loops (Phase III), conditional reduction tree depths, and dynamic chunk counts.
- Risk of an overly abstract plan language that becomes its own complexity burden.

### Option C: Hybrid — Abstract at the recipe level

Each recipe function (`execute_forward_pass`, `build_streaming_module_grad_path`, etc.) becomes a method on a backend-specific object. The `BatchProcessor` orchestrates through a backend-neutral interface of recipe calls. The recipes themselves are backend-specific implementations.

**Advantages:**
- Moderate refactoring scope—the existing recipe function signatures define the natural abstraction boundary.
- The `BatchProcessor` remains the shared orchestrator, calling `backend.execute_forward_pass(...)` instead of `execute_forward_pass(svs, ...)`.
- Each backend has full freedom in its recipe implementation. The Vulkan backend can record an entire recipe into a command buffer segment; the CPU backend can issue `pool_dispatch_and_wait` calls; OpenCL can enqueue imperatively.
- Naturally composable: a recipe for Node 15 (reduction tree) encapsulates the full dispatch/sync strategy for that sub-DAG.
- Does not require inventing a new plan representation language.

**Disadvantages:**
- Risks code duplication across backends for the orchestration logic *within* a recipe (e.g., the streaming loop structure in Phase III).
- The boundary is less formally defined than Option B's data-structure plan—correctness relies on recipe implementations honoring the same DAG contracts.
- Intra-recipe dependency structure is implicit in each backend's implementation rather than explicit in a shared plan.

---

## Analysis

### Eliminating Option A

Option A is eliminated. The Vulkan backend's execution model is structurally incompatible with per-dispatch abstract events. Forcing Vulkan into OpenCL emulation:

- Converts a single `vkCmdDispatch(N, 1, 1)` into N host-side calls + N barrier insertions, losing the hardware scheduler's workgroup distribution.
- Makes pipeline barriers—structural commands in a recording—impossible to represent as per-dispatch return tokens.
- Directly violates CONCEPT.md §1: the optimization pressure (Vulkan's batched model) contradicts the constraint (per-dispatch event abstraction), which is the textbook signal that the abstraction is incomplete.

### Choosing between B and C

Both B and C respect backend execution semantics. The decisive factors:

**Shared computation that must not diverge.** The reduction tree planning (stage count, fan-in, offset lists, Quadratic Scaling Policy thresholds), streaming loop structure (chunk count, per-chunk dependencies), and activation lifecycle decisions (cache vs. recompute) are computationally non-trivial and must produce identical results across backends. Under Option C, each recipe implementation must independently reproduce this logic or call shared helper functions, creating either duplication or an informal "shared planning layer" that Option B makes explicit.

**Recording-time vs. execution-time loops.** Vulkan's Phase III streaming loop is recorded into a command buffer at recording time with per-chunk `vkCmdPushConstants` + `vkCmdDispatch` + `vkCmdPipelineBarrier`. Option B's plan naturally expresses "loop N times over this sub-DAG with these parameters"—the Vulkan backend interprets this as recording instructions. Option C would express this as a `backend.execute_streaming_backprop(chunks=N)` call, which works but hides the loop structure from the shared layer entirely.

**Inspection and validation.** A first-class plan data structure (Option B) enables:
- Cross-backend plan comparison (same model → same logical plan → different renderings).
- Offline plan validation against CONTRACT.md and CONCEPT.md invariants.
- Plan serialisation for debugging and reproducibility.

Option C's implicit plans are only observable through execution side effects.

**Pragmatic risk.** Option B requires inventing a plan representation that is expressive enough for all three backends. If the representation is too rigid, it will force backends into unnatural patterns—replicating Option A's failure at a higher level. If too flexible, it becomes a generic AST with no verification benefit.

### The decisive principle

CONCEPT.md §1 (Architectural Elegance Feedback) is the tiebreaker. Option B formalises the execution plan as a first-class architectural primitive. When a new backend-specific optimization creates tension with the plan representation (e.g., Vulkan's single-dispatch model), the response is to extend the plan vocabulary—not to work around it. The plan representation evolves to subsume valid optimizations as first-class nodes. This is exactly the process prescribed by §1.

Option C, by hiding backend execution behind opaque recipe methods, loses the ability to formally reason about execution structure. Optimisation tensions are resolved inside backend code, invisible to the shared layer.

---

## Decision

**Option B: Abstract at the DAG/Phase level.**

The shared orchestration layer produces a backend-neutral execution plan expressed as a data structure. Each backend receives this plan and renders it using its native execution model.

### Boundary definition

The system is split into two layers with a plan data structure at the interface:

```
┌─────────────────────────────────────────────────────────┐
│                 Shared Orchestration Layer               │
│                                                         │
│  ModelSpec, MemoryLayout, ParameterSpace, Tiling,       │
│  StabilizationPolicy, ReductionTreePlan, ExecutionPlan, │
│  BatchProcessor (plan construction only)                │
│                                                         │
│  Produces: ExecutionPlan (backend-neutral data)         │
└───────────────────────┬─────────────────────────────────┘
                        │  plan data structure
┌───────────────────────┼─────────────────────────────────┐
│                 Backend Execution Layer                  │
│                                                         │
│  Consumes: ExecutionPlan                                │
│  Owns: context/queue, buffer allocation, kernel         │
│        dispatch, synchronization, D2H transfer          │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  OpenCL  │  │  Vulkan  │  │   CPU    │              │
│  │ Backend  │  │ Backend  │  │ Backend  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Three-tier jurisdictional model

The two-layer boundary above governs *where code lives*. Examining the existing OpenCL implementation reveals a finer-grained pattern in how decisions actually flow through the system — three tiers of jurisdiction that distribute differently depending on the computation:

| Tier | Decides | Illustrative ownership (OpenCL) |
| :--- | :------ | :------------------------------ |
| **Policy** | *What* to compute and *under what constraints*. | `StabilizationPolicy` resolves fan-in `K`, stage counts, threshold schedules. `ParameterSpace` determines tile decompositions. `ExecutionPlan` selects activation lifecycle. |
| **Orchestration** | *How* to sequence and dispatch the computation. | `graph_recipes.py` loops over reduction stages, selects register vs. local kernel tier (`max_reg_agg = 16`), manages ping-pong buffers, uploads offset lists via `cl.enqueue_copy`, chains `cl.Event`s between stages. |
| **Execution** | *The computation itself*. | `aggregate_register_reduce` / `aggregate_local_reduce` kernels sum partials. `clip_intermediate_grad` clips. `stabilize_and_reduce_grad_hidden_activations` (Node 16) runs its own internal `log_K(M)` reduction. |

The two-layer boundary (shared / backend) maps cleanly to this: the Policy tier is always shared; the Orchestration and Execution tiers are always backend-side. But the *distribution* of work across the three tiers varies by node type — the existing OpenCL code already demonstrates this:

**Host-orchestrated reduction trees (Nodes 14, 15, 20).** All three tiers have distinct actors. `StabilizationPolicy` computes the threshold schedule and `K` (Policy). `_execute_reduction_pipeline` in `graph_recipes.py` drives the stage loop — selecting kernel tier, allocating intermediate buffers, uploading offset lists, threading `cl.Event` chains (Orchestration). The `aggregate_*` and `clip_intermediate_grad` kernels execute individual stages (Execution).

**Specialized kernels (Node 16).** The Orchestration tier collapses into the Execution tier. `StabilizationPolicy` still computes `policy_max_k`, and the plan still carries `T_algorithmic`, `λ`, and `fp_max` as scalar parameters (Policy). But the kernel itself manages its internal multi-stage reduction — selecting its own fan-in, running its own `log_K(M)` loop, inserting its own synchronization. The host's role reduces to a single `KernelExecutor.launch()` call. This collapse is possible because Node 13 (`gather_and_permute_grad_h`) guarantees contiguous input, eliminating the inter-stage logistics — scattered offset lists, intermediate buffer allocation — that otherwise require host participation.

**Simple kernel dispatches (Nodes 4–11, 17–19, 21, 24, 25).** The Orchestration tier is minimal. `graph_recipes.py` issues a single `KernelExecutor.launch()` with the appropriate event dependencies. Policy determines the parameters; execution is a single kernel invocation.

This observed pattern is significant for the abstraction boundary because it reveals that **the Policy tier is the only tier that is invariant across all node types and all backends**. In every case — whether the host drives a multi-stage loop, a kernel manages its own internal reduction, or a simple dispatch occurs — the shared layer is responsible for setting the constraints. The Orchestration and Execution tiers redistribute freely depending on the node's structure and the backend's native capabilities.

The plan boundary (this ADR) therefore separates Policy from Orchestration. The kernel contract boundary (CONTRACT.md) separates Orchestration from Execution. When a future backend discovers an optimization that shifts work between these latter two tiers — e.g., a Vulkan backend fusing adjacent sum-and-clip stages into a single command buffer segment, or a CPU backend inlining the stage loop — CONCEPT.md §1 (Architectural Elegance Feedback) applies: if the shift creates tension with the plan vocabulary, it is formalized as a plan-level primitive, not hidden inside the renderer.

### What crosses the boundary

The execution plan carries:

- **Phase structure:** Act and Learn phases with their node sequences.
- **Node descriptors:** For each DAG node—the kernel identity, logical input/output buffer names, tile decomposition, and dependency edges.
- **Reduction tree plans:** Stage count, fan-in per stage, offset lists, and per-stage thresholds (Quadratic Scaling Policy).
- **Streaming loop descriptors:** Chunk count, per-chunk parameter deltas, and per-chunk sub-DAG structure.
- **Activation lifecycle policy:** Cache vs. recompute decisions, expressed as provider descriptors rather than executable `DependencyProvider` objects.
- **Synchronization points:** Named barriers (`inference_event`, `final_batch_event`) that each backend must honour.

### What does NOT cross the boundary

- `cl.Event`, `cl.CommandQueue`, `cl.Buffer`, `cl.Context` — or equivalents from any backend.
- Kernel argument binding details (positional args, push constants, descriptor sets).
- Dispatch mechanics (enqueue calls, command buffer recording, `pool_dispatch_and_wait`).
- Memory transfer implementation (staging buffers, zero-copy, `clEnqueueCopy`).

### Backend rendering contract

Each backend implements a `PlanRenderer` (or equivalent) that:

1. Accepts a fully constructed `ExecutionPlan`.
2. Allocates device memory for all buffers named in the plan.
3. Renders each phase by interpreting node descriptors through its native execution model.
4. Signals the named synchronization points.
5. Returns host-accessible results at the plan-specified retrieval points.

The renderer is free to:
- Batch multiple nodes into a single dispatch (Vulkan's single-dispatch model).
- Fuse adjacent barrier-free nodes (Vulkan's hazard elision).
- Execute nodes synchronously and skip barrier insertion (CPU).
- Enqueue nodes imperatively with event chains (OpenCL).

---

## Consequences

### Positive

- **Backend fidelity.** Vulkan records command buffers with structural barriers. CPU calls `pool_dispatch_and_wait` sequentially. OpenCL enqueues with event dependencies. No backend is forced into an unnatural execution model.
- **Shared correctness.** Reduction tree planning, Quadratic Scaling thresholds, streaming loop structure, and activation lifecycle decisions are computed once in the shared layer. No risk of cross-backend divergence in these critical computations.
- **Inspectable plans.** The execution plan is a data structure that can be logged, compared across backends, validated offline, and serialised for debugging.
- **Extensibility.** New backends require implementing a renderer—not modifying the shared orchestration layer. New plan node types are added by extending the plan vocabulary per CONCEPT.md §1.

### Negative

- **Large initial refactoring of `graph_recipes.py`.** The current 984-line module must be split: DAG-description logic moves to the shared layer; OpenCL dispatch logic becomes the first backend renderer. This is the riskiest single change in the migration.
- **Plan representation design cost.** The plan data structure must be expressive enough for streaming loops, conditional reduction depths, and dynamic chunk counts—without becoming an ad-hoc DSL. Careful design is required.
- **`execution_plan.py` rework.** The current `DependencyProvider` hierarchy embeds `cl.Event` and `KernelExecutor.launch()` calls. These must be replaced with declarative provider descriptors that the backend interprets.

### Migration implications

Per ADR-017 (Incremental Migration Path), this decision validates the proposed phasing:

1. **Phase 0:** Extract backend-neutral `Protocol` types from current OpenCL code. No behavioural change.
2. **Phase 1:** Purify the shared layer—remove all `pyopencl` imports from `execution_plan.py`, `stabilization_policy.py`, `workload_primitives.py`, `model_spec.py`, `memory_layout.py`, `parameter_space.py`.
3. **Phase 2:** Refactor `graph_recipes.py` into plan construction (shared) + plan rendering (OpenCL backend). This is the critical step.
4. **Phase 3–4:** Implement CPU and Vulkan renderers against the established plan interface.

---

## References

- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback, §4 Asynchronous Host Interaction, §5 Unified Execution Model
- [CONTRACT.md](../CONTRACT.md) — Article 1.1 Jurisdictional Separation, Article 1.4 Collaborative Interface Verifiability
- [CPU_BACKEND.md](../CPU_BACKEND.md) — Threading model, `pool_dispatch_and_wait` execution
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — Command buffer strategy, single-dispatch parallelism
- [ADR_PLAN.md](../ADR_PLAN.md) — ADR-001 problem statement and option enumeration
- [ADR-018: User-Facing API](ADR-018-user-facing-api.md) — user-facing API must not expose backend-specific types; `WorkTicket`/`Engine` sit above the plan boundary in the shared layer
