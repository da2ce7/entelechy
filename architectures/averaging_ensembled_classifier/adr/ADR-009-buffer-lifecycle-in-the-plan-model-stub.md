# ADR-009: Buffer Lifecycle in the Plan Model

**Status:** STUB (NARROWED — `BufferHandle` as universal plan-level token; backend allocates)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-002, ADR-003, ADR-004  
**Blocks:** ADR-010, ADR-012

---

## Context

The plan references buffers by logical name via `buffer_bindings: Dict[str, str]` in each `KernelDispatchNode` (ADR-002). Plan construction never allocates — it declares "this node produces buffer X with shape Y and padding Z." The backend's `PlanRenderer` maps logical buffer names to physical allocations.

ADR-002 further resolves the activation lifecycle dimension: the Cache vs. Recompute decision is expressed structurally in the plan DAG — presence or absence of recompute `StreamingLoopNode`s — not as executable `DependencyProvider` objects. The `CacheProvider` / `RecomputeProvider` / `ComputeOnceProvider` / `StagedComputationProvider` hierarchy is dissolved. Buffer lifetimes are fully determined by the DAG topology at plan-construction time.

The existing `BufferHandle` token is the natural plan→renderer handoff mechanism. The plan builder assigns handles; the renderer allocates backing memory.

---

## Narrowed Direction

`BufferHandle` as universal plan-level token; backend allocates physical memory.

---

## Upstream Constraint: Two-Tier Buffer Scope

ADR-003 establishes a critical distinction between **plan-level buffers** and **renderer-internal buffers**, reinforced by ADR-004:

- **Plan-level buffers** are named in the plan data structure — `source_buffer` and `destination_buffer` on `ReductionTreeNode`, `buffer_bindings` on `KernelDispatchNode`s, and `collection_buffers` on `StreamingLoopNode`s. Their lifetimes are properties of the DAG topology and belong in the shared layer.
- **Renderer-internal buffers** are allocated by the backend during rendering and never appear in the plan. For `ReductionTreeNode`s: intermediate stage buffers (ping-pong) and uploaded offset-list buffers. For `StreamingLoopNode`s: per-iteration scratch buffers (declared via `ScratchBufferSpec` entries, but allocation is renderer-owned).

The plan's buffer lifetime model governs only plan-level buffers. Renderer-internal buffers are entirely Orchestration-tier concerns.

---

## Remaining Decision

Does the plan prescribe buffer *reuse* for plan-level buffers, or does it leave lifetime management entirely to the renderer?

- **(A) Plan prescribes lifetimes.** The plan annotates each plan-level buffer with its producing node and last-consuming node. The renderer uses this information to optimize memory reuse. The shared layer has full DAG visibility and can compute optimal lifetimes.

- **(B) Renderer owns lifetimes.** The plan declares buffers but not their lifetimes. Each renderer analyzes the plan to determine reuse opportunities. Risk: duplicated analysis logic across backends.

Option A is strongly favored. Plan-level buffer lifetime is a property of the DAG, not the dispatch model. ADR-002's explicit dependency edges make lifetime computation straightforward.

---

## Tensions

- Vulkan backends may further optimize by suballocating from large `VkDeviceMemory` blocks. The plan's lifetime annotations enable this without prescribing it.
- The two-tier buffer scope means the plan's memory footprint estimate will undercount actual device memory usage — renderer-internal intermediates are invisible. This tension is accommodated by ADR-003's rendering contract (intermediate buffers derivable from `num_stages`, `fan_in`, `elements_per_partial`) and ADR-004's `ScratchBufferSpec` entries (sizes declared explicitly). Together, these make renderer-internal memory overhead *estimable* from plan data alone.
- ADR-004's `collection_buffers` on `StreamingLoopNode` persist beyond the streaming loop for consumption by downstream `ReductionTreeNode`s (Node 20). Option A's lifetime annotations must model the producing node as the `StreamingLoopNode` itself, not individual iterations.

---

## References

- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — explicit dependency edges; `DependencyProvider` dissolution
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — two-tier buffer scope; renderer-internal intermediates
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `ScratchBufferSpec`; `collection_buffers` lifetime
