# ADR-002: Plan Node Types & Synchronization Structure

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001  
**Blocks:** ADR-009, ADR-010

---

## Context

ADR-001 established that the shared orchestration layer produces a backend-neutral execution plan expressed as a data structure, and each backend receives this plan and renders it using its native execution model. The plan boundary stops at each kernel's public interface: plan nodes describe kernel identity, logical buffer bindings, tile decomposition, dependency edges, and policy parameters.

This ADR defines the concrete vocabulary of plan node types and the synchronization structure that replaces the `cl.Event` objects currently threading through every module.

### The current state

The existing implementation expresses execution structure implicitly through three coupled mechanisms:

1. **`cl.Event` chains.** Every recipe function in `graph_recipes.py` accepts `List[cl.Event]` and returns `cl.Event`. Dependencies between nodes are encoded as event wait-lists threaded through function call chains. There is no inspectable dependency graph — the DAG exists only as transient runtime state in the OpenCL driver's event queue.

2. **`cl.enqueue_barrier` calls.** The `BatchProcessor` creates synchronization points (e.g., `all_probs_ready_evt`, `all_module_grads_clipped_evt`, `all_partials_ready_evt`, `all_grads_summed_evt`) by inserting barriers that join multiple event streams. These correspond to the architecture's named synchronization concepts but are expressed as unnamed, ad-hoc OpenCL primitives.

3. **`DependencyProvider` objects.** The `ExecutionPlan`'s `DataLifecyclePolicy` maps buffer names to providers (`CacheProvider`, `RecomputeProvider`, `ComputeOnceProvider`, `StagedComputationProvider`), each of which embeds `cl.Event` and `KernelExecutor.launch()` calls in its `resolve()` method. These carry both the lifecycle decision (shared concern) and the dispatch mechanism (backend concern) in one object.

All three mechanisms must be replaced by a backend-neutral plan representation.

### What the plan must express

From the CONCEPT.md DAG and the `BatchProcessor`'s current execution flow, the plan must capture:

1. **Kernel dispatch nodes** — individual kernel invocations with their logical buffer bindings, tile decomposition, and scalar parameters. These are the fundamental units of work (Nodes 4–11, 13, 16–19, 21, 24, 25).

2. **Reduction tree structures** — multi-stage `log_K(N)` reduction trees with fan-in, offset lists, per-stage thresholds, and optional clip steps. Kernel tier selection is an Orchestration-tier concern (ADR-001 §Three-tier jurisdictional model) resolved by the renderer, not the plan. These correspond to the Recursive Clip-Aggregation Engine (Nodes 14, 15, 20). See ADR-003.

3. **Streaming loop structures** — parametric loops over a chunk-indexed sub-DAG. Phase III's streaming backpropagation (Nodes 17→18→19, iterated per chunk) and the Model A recompute path (Nodes 4→5→6/7→8→9→10→11, iterated per tile under `RECOMPUTE_GRAD_H`). See ADR-004.

4. **Named synchronization points** — the Item Synchronization Point (Node 13), the Batch Synchronization Point (Node 22), and the two host-visible barriers (`inference_event`, `final_batch_event`).

5. **Host retrieval points** — locations where device results become host-accessible (`Final Probs` after the Act phase, `effective_batch_size` for the normalization scalar).

6. **Dependency edges** — the data-flow relationships between all the above, replacing the implicit `cl.Event` chains.

7. **Activation lifecycle policy** — the Cache vs. Recompute decision, expressed as a plan-level structural choice (presence or absence of recompute loops) rather than as executable `DependencyProvider` objects.

---

## Decision Drivers

1. **ADR-001 (Backend Abstraction Boundary):** The plan is a backend-neutral data structure. No `cl.Event`, `cl.CommandQueue`, `cl.Buffer`, or equivalent types from any backend may appear in the plan vocabulary.

2. **CONCEPT.md §1 (Architectural Elegance Feedback):** The plan node taxonomy is a finite, enumerated vocabulary. When a new optimization requires a construct beyond this vocabulary, the response is to extend the taxonomy as a new first-class node type — not to stretch existing node types beyond their semantic scope.

3. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability):** Pre-dispatch validation must remain in the shared layer. Each node must carry sufficient contract data for the shared layer to validate correctness at plan-construction time, without requiring a backend.

4. **CONCEPT.md §4 & §5 (Asynchronous Host Interaction & Unified Execution Model):** The plan must express the Act/Learn phase structure, the named synchronization points (`inference_event`, `final_batch_event`, Item Synchronization Point, Batch Synchronization Point), and the host's temporal decoupling from device execution.

5. **Backend dispatch divergence (ADR-001 §Backend execution model divergence):** A `KernelDispatchNode` with `tile_count=N` must not prescribe dispatch granularity. OpenCL renders it as N `clEnqueueNDRange` calls; Vulkan renders it as one `vkCmdDispatch(N, 1, 1)`; CPU renders it as N `pool_dispatch_and_wait` calls.

---

## Options Considered

### Option A: Flat, homogeneous node list

All plan elements are a single `PlanNode` type with a `kind` discriminator field (`"kernel"`, `"barrier"`, `"retrieval"`, `"reduction"`, `"loop"`). Nodes carry a union of all possible fields, with unused fields set to `None`.

**Advantages:**
- Minimal type system complexity. Plan iteration is a single loop over one type.
- Serialisation is trivial — one JSON schema.

**Disadvantages:**
- Loses static type safety. A `"barrier"` node with unused `kernel_identity` and `tile_count` fields is misleading and invites errors.
- Validation must perform dynamic dispatch on the `kind` field to check field invariants, reimplementing a type system at runtime.
- The distinction between structurally different node categories (an atomic kernel dispatch vs. a multi-stage reduction tree vs. a parametric loop) is fundamental, not incidental. Erasing it into a union type obscures the plan's semantic structure.

### Option B: Typed node hierarchy with explicit dependency edges

A small, closed set of typed node classes, each carrying exactly the fields its semantics require. Dependency edges are expressed as references between nodes (by node identity or index). The plan is a DAG of these typed nodes.

**Advantages:**
- Static type safety enforces field invariants. A `BarrierNode` cannot have a `tile_count`.
- Each node type's validation logic is self-contained.
- The plan's semantic structure is immediately visible from its types.
- Aligns with the Complexity Ceiling Constraint (ADR_PLAN.md): the taxonomy is a fixed enumeration, not an open hierarchy.

**Disadvantages:**
- Requires defining 5 node types instead of 1.
- Serialisation requires a discriminated union scheme.

### Option C: Two-level plan — phase descriptors containing node sequences

The plan is structured as a sequence of `PhaseDescriptor` objects (Act, Learn Phase I, Learn Phase II, etc.), each containing an ordered list of nodes. Dependencies are implicit in the ordering within and between phases.

**Advantages:**
- Directly mirrors the CONCEPT.md phase structure.
- Phase-level barriers are structural (phase boundaries) rather than explicit nodes.

**Disadvantages:**
- Intra-phase concurrency is invisible. Within Phase I, Nodes 8, 9, and 10 can execute in parallel within a tile — but a flat, ordered list within the phase either loses this information or requires separate concurrency annotations.
- The `BatchProcessor`'s current flow reveals that some dependency relationships cross phase boundaries in non-trivial ways (e.g., `hidden_activations` computed in Act are consumed in Learn Phase I and Phase III). Phase-based structuring either duplicates these cross-phase references or requires a separate inter-phase wiring mechanism — which converges on the explicit edge model from Option B.
- Reduction trees and streaming loops do not fit cleanly as "phases" — they are sub-structures within phases.

---

## Analysis

### Eliminating Option A

Option A is eliminated. The plan node taxonomy serves as a formal contract between the shared orchestration layer and every backend renderer. A dynamically-typed union of all possible fields forfeits the static guarantees that make this contract enforceable. Every backend renderer would need to defensively validate that a node claiming `kind="barrier"` does not contain contradictory kernel dispatch fields — duplicating validation logic that the type system can enforce for free.

### Eliminating Option C

Option C conflates two orthogonal concerns: the phase structure of the DAG (a semantic property defined by CONCEPT.md) and the dependency structure between nodes (a structural property of the computation). The existing DAG has dependency edges that interleave across phases (e.g., `hidden_activations` flows from Act into Learn Phases I and III; `Summed_Grad_H` flows from Learn Phase II into Phase III). Encoding these as inter-phase cross-references recreates the explicit edge model of Option B but with additional phase-container overhead and no additional expressiveness.

Furthermore, the Named Synchronization Points (Node 13, Node 22) are natural `BarrierNode` instances in the dependency DAG — they are not phase boundaries. Node 13 (Item Synchronization Point) occurs within Learn Phase I and gates the transition to Phase II for `Grad_H`, but other flows (Module and Temperature gradient reductions) proceed through Phase II concurrently without waiting for Node 13. This intra-phase branching and joining is precisely what a DAG with typed nodes and explicit edges captures.

### Choosing Option B

Option B provides static type safety, direct correspondence to the CONCEPT.md DAG, and clean separation of node semantics. Each node type carries exactly the data its role requires. The plan is a DAG — not a flat list and not a phase hierarchy — because the architecture's dependency structure is genuinely a DAG with concurrent branches, joins, and named barriers that do not align with a strict phase decomposition.

---

## Decision

**Option B: Typed node hierarchy with explicit dependency edges.**

The execution plan is a directed acyclic graph of typed, immutable node descriptors connected by explicit dependency edges. Each node type is a frozen dataclass carrying exactly the fields its semantics require.

### Node type taxonomy

| Node Type              | Semantics                                                                                           | CONCEPT.md Correspondence                                      |
| :--------------------- | :-------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- |
| `KernelDispatchNode`   | A single logical kernel invocation with buffer bindings, scalar parameters, tile decomposition, and placement strategy. Node 16 appears here despite its internal multi-stage reduction because ADR-001's plan boundary stops at the kernel's public interface — the Orchestration and Execution tiers collapse into a single kernel dispatch (see ADR-001 §Three-tier jurisdictional model, ADR-005). | Nodes 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17, 18, 19, 21, 24, 25 |
| `ReductionTreeNode`    | A multi-stage reduction tree (fan-in, offset lists, per-stage thresholds). Rendered atomically by the backend, which selects kernel tiers and manages intermediate buffers. Internal structure defined in ADR-003. | Nodes 14, 15, 20                                               |
| `StreamingLoopNode`    | A parametric loop over a chunk-indexed sub-DAG body. Specifies chunk count, per-chunk parameter deltas, and the flat body sequence. | Phase III streaming (Nodes 17→18→19); Model A recompute path   |
| `BarrierNode`          | A named synchronization point that joins multiple upstream dependency edges. Carries no kernel dispatch — it is a pure sequencing construct. | Node 13 (Item Sync), Node 22 (Batch Sync)                     |
| `RetrievalNode`        | A host-accessible result extraction point. Specifies the source buffer, expected shape, and the named event it signals. | Node 23 (`inference_event`), `final_batch_event`               |

This is a **closed taxonomy**. No additional node types may be introduced without a formal ADR. Per CONCEPT.md §1 (Architectural Elegance Feedback): when an optimization requires a construct beyond this vocabulary, implementation is suspended, the pattern is formalized as a documented extension, and a new node type is introduced through a revised ADR.

### Node identity and dependency edges

Each node in the plan carries a unique `node_id: str` (e.g., `"forward_pass"`, `"item_sync_barrier"`, `"batch_sync_barrier"`, `"inference_retrieval"`). Dependency edges are expressed as a set of upstream node identifiers:

```python
@dataclass(frozen=True)
class KernelDispatchNode:
    node_id: str
    depends_on: FrozenSet[str]
    kernel_identity: str
    buffer_bindings: Dict[str, str]     # param_name → logical_buffer_name
    scalar_params: Dict[str, Any]       # param_name → value
    tile_count: int
    placement_strategy: Optional[str]   # e.g., "grid_mod_cls", "linear_batch"
    contract: "KernelContract"          # Shared-layer validation data
```

The backend renderer resolves `depends_on` to its native synchronization mechanism:
- **OpenCL:** Each upstream node's completion event is added to the `wait_for` list.
- **Vulkan:** A `vkCmdPipelineBarrier` is inserted between producer and consumer command buffer entries.
- **CPU:** Nodes are topologically sorted and executed sequentially; dependency edges constrain the sort order.

### Synchronization structure

ADR-001 eliminates `cl.Event` from the shared layer. The four named synchronization points from CONCEPT.md are represented as follows:

| Synchronization Point       | Node Type        | `node_id`                | Upstream Dependencies                                                                          |
| :-------------------------- | :--------------- | :----------------------- | :--------------------------------------------------------------------------------------------- |
| Item Synchronization Point  | `BarrierNode`    | `"item_sync_barrier"`    | All `clip_partial_gradients` (Node 11) dispatches for the current item                         |
| Batch Synchronization Point | `BarrierNode`    | `"batch_sync_barrier"`   | `normalize_gradients` (Node 21) for all parameter groups                                       |
| `inference_event`           | `RetrievalNode`  | `"inference_retrieval"`  | Diagnostic aggregation (Node 14) for probabilities                                             |
| `final_batch_event`         | `RetrievalNode`  | `"final_batch_retrieval"`| `clamp_temperatures` (Node 25), all `adam_update` (Node 24) dispatches                         |

The `BarrierNode` is purely structural — it carries no dispatch payload. It exists so that downstream nodes can declare a single dependency on the barrier rather than enumerating all upstream nodes individually. This serves both correctness (a single, auditable join point) and readability (the plan mirrors the CONCEPT.md DAG's named barriers).

The `RetrievalNode` marks a host-observable completion point. Its `node_id` corresponds to the named event from CONCEPT.md §4. The renderer must signal host-side availability at this point. The mechanism (blocking call, Future-like handle, or zero-cost pointer access) is a backend concern decided in ADR-010.

### Concurrency annotations

The plan's dependency edges inherently encode concurrency. Nodes with no dependency relationship (neither directly nor transitively connected) may execute in parallel. The renderer is free to exploit this:

- Within a tile, Nodes 8, 9, and 10 produce independent partial gradients from the same inputs. Their `depends_on` sets reference the same upstream forward-pass node, but not each other. A renderer may dispatch them concurrently.
- The diagnostic aggregation (Node 14) depends only on Act-phase outputs. It can proceed in parallel with the Learn phase's gradient production.
- The `inference_retrieval` node depends on the probability aggregation (Node 14), not on any Learn-phase nodes. The host can observe inference results while learning continues.

No explicit concurrency annotation is required. The DAG structure is the concurrency specification.

### KernelDispatchNode design principles

#### Dispatch granularity independence

A `KernelDispatchNode` with `tile_count=N` specifies "this kernel is invoked across N tiles." It does **not** specify whether this is N sequential dispatches, one dispatch with N workgroups, or N thread-pool tasks. The `tile_count` is a logical property of the computation; the dispatch granularity is a rendering concern.

The `placement_strategy` field (e.g., `"grid_mod_cls"`, `"linear_batch"`) specifies the abstract strategy by which each tile determines its position in the output buffer. Per ADR-007 (KernelSignature Contract/Binding Split), the mechanism — host-provided `flat_tile_index` scalar (OpenCL), `gl_WorkGroupID.x` (Vulkan), or `task_index` parameter (CPU) — is a binding-level concern, not a plan-level concern.

#### Contract embedding

Each `KernelDispatchNode` carries a `contract: KernelContract` reference. The `KernelContract` (defined in ADR-007) encodes the kernel's validated shapes, padding requirements, calculability proofs, and validation preconditions from CONTRACT.md Article 1.4. Plan construction validates the contract at build time. The backend renderer trusts the contract's validity and proceeds directly to binding and dispatch.

This satisfies CONTRACT.md Article 1.4a (Host Proof Obligation): all validation occurs in the shared layer, at plan-construction time, exactly once.

### ReductionTreeNode design principles

The `ReductionTreeNode` encapsulates the full Recursive Clip-Aggregation Engine for one parameter flow. Its internal structure — stage count, per-stage fan-in, offset lists, optional clip steps, and Quadratic Scaling Policy thresholds — is defined in ADR-003.

The `ReductionTreeNode` is the canonical example of how the three-tier jurisdictional model (ADR-001 §Three-tier jurisdictional model) distributes across all three tiers with distinct actors. In the existing OpenCL code, `StabilizationPolicy` computes the threshold schedule and fan-in (Policy); `_execute_reduction_pipeline` in `graph_recipes.py` drives the stage loop, selects kernel tier, and manages intermediate buffers (Orchestration); `aggregate_*` and `clip_intermediate_grad` kernels execute individual stages (Execution). The `ReductionTreePlan` (ADR-003) is the data structure that carries the Policy tier's output across the plan boundary to the Orchestration tier.

The plan carries the reduction tree as a single, typed node rather than expanding it into individual `KernelDispatchNode`s because:

1. The tree is a **semantically atomic unit**. Its stages are not independently meaningful — they collectively implement a specific reduction strategy (stabilized or diagnostic).
2. The backend must render the tree as a contiguous unit. Vulkan records all stages into a command buffer segment with pipeline barriers between stages. OpenCL enqueues with intra-tree event chains. CPU calls sequentially. Expanding the tree into individual plan nodes would force the backend to reconstruct the tree structure from flat nodes — an information-destroying round-trip.
3. Per ADR-003, the tree's internal structure (stage plans) is a nested data structure within the `ReductionTreeNode`, not a separate set of plan nodes.

### StreamingLoopNode design principles

The `StreamingLoopNode` encapsulates the streaming loop body and iteration parameters. Its design is defined in ADR-004.

Key constraint (the **Complexity Ceiling**): `StreamingLoopNode.body` is a flat sequence of `KernelDispatchNode`s. It may not contain another `StreamingLoopNode`, a `ReductionTreeNode`, a `BarrierNode`, or a `RetrievalNode`. If a future optimization requires nested loops or loop-internal barriers, CONCEPT.md §1 mandates formalizing that as a new node type — not relaxing this constraint.

### Activation lifecycle in the plan

The Cache vs. Recompute decision (CONCEPT.md §5) is expressed structurally in the plan, not as an executable `DependencyProvider` object:

- **Cache strategy:** The plan contains `KernelDispatchNode`s for `forward_pass` (Node 4) and `render_logits_chunk` (Node 5) executed once. Downstream nodes reference these outputs by buffer name. No recomputation nodes appear.
- **Recompute strategy:** The plan contains `StreamingLoopNode`s whose bodies include recomputation kernels (Nodes 4, 5) alongside the gradient kernels (Nodes 8, 9, 10, 11). The loop implicitly expresses "compute, use, and discard."

The `DataLifecyclePolicy` and its `DependencyProvider` subclasses (`CacheProvider`, `RecomputeProvider`, `ComputeOnceProvider`, `StagedComputationProvider`) are replaced by the plan's structural representation. The plan builder makes the lifecycle decision; the plan encodes its consequence as DAG topology.

---

## Consequences

### Positive

- **Inspectable dependency structure.** The plan's dependency edges are a first-class data structure that can be topologically sorted, visualized, validated against CONCEPT.md invariants, and compared across backends — all without executing anything.
- **Static type safety.** Each node type enforces its own field invariants. A `BarrierNode` cannot accidentally carry kernel dispatch parameters. A `RetrievalNode` cannot be missing its source buffer specification.
- **Backend rendering freedom.** The typed DAG with explicit edges gives each renderer complete freedom in its execution strategy — the Orchestration and Execution tiers (ADR-001 §Three-tier jurisdictional model) are entirely backend-owned. OpenCL builds event chains. Vulkan records a command buffer in topological order with pipeline barriers at dependency edges. CPU executes a topological sort. No renderer is constrained by another's execution model.
- **Single validation pass.** Contract validation occurs once at plan-construction time (CONTRACT.md Article 1.4a). The renderer receives a pre-validated plan and proceeds directly to dispatch.
- **Concurrency from structure.** The DAG's dependency edges are the complete and sufficient specification of concurrency constraints. No separate concurrency annotation layer is needed.
- **Named synchronization fidelity.** The CONCEPT.md synchronization points (Item Sync, Batch Sync, `inference_event`, `final_batch_event`) are directly represented as typed nodes with specific `node_id` values, preserving their architectural significance.

### Negative

- **Plan construction complexity.** The plan builder must construct a fully-wired DAG with correct dependency edges, node types, and contract data. This replaces the current approach of threading `cl.Event` objects through function calls — which is simpler to write (just pass the event) but impossible to inspect or validate.
- **Five node types to maintain.** Each node type requires its own validation logic, serialisation schema, and backend rendering path. This is a modest but real maintenance surface. The Complexity Ceiling Constraint bounds this: no new node types without a formal ADR.
- **`DependencyProvider` dissolution.** The current `CacheProvider` / `RecomputeProvider` / `ComputeOnceProvider` / `StagedComputationProvider` hierarchy, which elegantly encapsulates lifecycle decisions as polymorphic objects, is dissolved. The plan builder absorbs this logic, emitting different DAG topologies for different lifecycle strategies. The polymorphism moves from runtime to plan-construction time — a net architectural improvement, but a loss of the current code's local elegance.

### Migration implications

Per ADR-017 (Incremental Migration Path), this decision affects Phase 2:

1. **Define node types** as frozen dataclasses in `execution_plan.py`. No existing code is modified — these are additive definitions.
2. **Implement `ExecutionPlanBuilder`** that produces a plan DAG from `ModelSpec` + `HardwareProfile` + batch parameters. This parallels the current `TrainingOrchestrator`'s plan-construction logic but outputs typed nodes instead of `DependencyProvider` objects.
3. **Write plan-level tests** (ADR-016, layer 1) that validate DAG structure, dependency correctness, reduction tree depth, and synchronization point placement for known inputs.
4. **Phase 3** (OpenCL Plan Renderer) then consumes this plan, replacing the current `graph_recipes.py` dispatch functions with a renderer that walks the DAG and dispatches via OpenCL.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — foundational decision on plan-level abstraction; three-tier jurisdictional model (Policy / Orchestration / Execution)
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback, §4 Asynchronous Host Interaction, §5 Unified Execution Model, DAG diagram
- [CONTRACT.md](../CONTRACT.md) — Article 1.1 Jurisdictional Separation, Article 1.4 Collaborative Interface Verifiability, Article 3.2 Placement Contract
- [ADR_PLAN.md](../ADR_PLAN.md) — ADR-002 problem statement and narrowing analysis
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `ReductionTreeNode` internal structure (pending)
- [ADR-004: Streaming Loop Representation](ADR-004-streaming-loop-representation.md) — `StreamingLoopNode` design (pending)
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` definition (pending)
