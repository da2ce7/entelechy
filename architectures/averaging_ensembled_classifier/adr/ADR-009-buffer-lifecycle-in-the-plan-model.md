# ADR-009: Buffer Lifecycle in the Plan Model

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-002, ADR-003, ADR-004  
**Blocks:** ADR-010, ADR-012

---

## Context

ADR-002 establishes that plan nodes reference buffers by logical name: `KernelDispatchNode.buffer_bindings: Dict[str, str]` maps kernel parameter names to logical buffer names, `ReductionTreeNode` carries `source_buffer` and `destination_buffer` names, and `StreamingLoopNode` carries `collection_buffers`. Plan construction never allocates device memory — it declares what buffers exist, what shapes they have, and which nodes produce and consume them. The backend's `PlanRenderer` maps these logical declarations to physical allocations.

ADR-003 and ADR-004 jointly establish the **two-tier buffer scope** distinction:

- **Plan-level buffers** are named in the plan data structure and have lifetimes governed by the DAG topology. These include all buffers referenced by `buffer_bindings`, `source_buffer`, `destination_buffer`, and `collection_buffers`.
- **Renderer-internal buffers** are allocated by the backend during rendering and never appear in the plan's buffer namespace. For `ReductionTreeNode`s: intermediate stage buffers (ping-pong) and uploaded offset-list buffers (ADR-003). For `StreamingLoopNode`s: per-iteration scratch buffers declared via `ScratchBufferSpec` entries, with allocation renderer-owned (ADR-004).

The existing `BufferHandle` type in `launcher_infra.py` is coupled to OpenCL — it is created by `BufferManager.__init__(cl.Context)` and resolved to `cl.Buffer` via `BufferManager.get_cl_buffer()`. The multi-backend refactoring (ADR-001) requires a backend-neutral `BufferHandle` that lives in the shared layer.

ADR-002 further resolves the activation lifecycle dimension: the Cache vs. Recompute decision is expressed structurally in the plan DAG — presence or absence of recompute `StreamingLoopNode`s — not as executable `DependencyProvider` objects. The `CacheProvider` / `RecomputeProvider` / `ComputeOnceProvider` / `StagedComputationProvider` hierarchy is dissolved. Buffer lifetimes are fully determined by the DAG topology at plan-construction time.

### The design question

The plan must carry a formal namespace of all plan-level buffers — their names, shapes, types, and roles. The open question is whether the plan also carries **lifetime annotations** (producing node, last consuming node) for each buffer, enabling the renderer to optimize physical memory reuse without re-analyzing the DAG.

The shared layer has full visibility into the DAG at plan-construction time. It can compute buffer lifetimes from the dependency edges and each node's read/write bindings (the `src_`/`dest_`/`update_` flow prefixes from CONTRACT.md Article 2.1 and the `KernelContract` flow declarations from ADR-007). The renderer also receives the DAG — but deriving lifetimes requires replaying this kernel-contract analysis, which is Policy-tier knowledge.

### What varies across backends

Only the physical allocation and memory reuse mechanics differ:

- **OpenCL:** Discrete `cl.Buffer` allocations via `cl.Buffer(ctx, flags, size)`. Memory reuse via buffer pooling (`BufferManager.acquire_transient_buffer` / `release_transient_buffer`).
- **Vulkan:** Suballocated from large `VkDeviceMemory` blocks. Non-overlapping buffers can share the same memory region if their lifetimes do not overlap. Lifetime intervals directly enable optimal packing.
- **CPU:** `malloc`'d arrays or memory-pool allocation. Zero-copy for host-visible buffers. May prefer arena allocation with explicit reset points.

---

## Decision Drivers

1. **ADR-001 (Plan as data structure):** The plan boundary is a pure data structure. `BufferHandle` must be backend-neutral — no `cl.Buffer`, `VkDeviceMemory`, or pointer types may appear in the plan.

2. **ADR-001 (Shared correctness — Policy tier):** ADR-001 §Three-tier jurisdictional model identifies the Policy tier as the only tier invariant across all node types and backends. Buffer lifetime computation requires DAG analysis and kernel-contract awareness (which bindings are reads vs. writes) — this is Policy-tier computation. Duplicating it across renderers risks cross-backend divergence, the exact failure mode ADR-001 warns against.

3. **ADR-002 (Dependency edges):** Explicit dependency edges make lifetime computation straightforward. The producing node and all consuming nodes for each buffer are fully determined by the DAG topology and `buffer_bindings` mappings.

4. **ADR-003, ADR-004 (Two-tier buffer scope):** The plan's buffer namespace governs only plan-level buffers. Renderer-internal buffers (intermediate reduction stages, scratch buffers) are outside its scope. This ADR formalizes this boundary as a first-class invariant.

5. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability):** The plan must carry sufficient data to validate buffer correctness at plan-construction time: shape consistency between producer and consumer, single-producer coverage, and acyclic buffer flows.

6. **Backend allocation divergence:** Vulkan's suballocation model benefits directly from plan-level lifetime intervals for packing non-overlapping buffers into shared memory regions. OpenCL's discrete allocations benefit from knowing when buffers become dead for pooled reuse. CPU's arena allocators benefit from knowing the high-water mark of simultaneously live buffer bytes. The plan must enable all three approaches without constraining any.

7. **CONCEPT.md §1 (Architectural Elegance Feedback):** The buffer lifecycle model is a first-class architectural primitive. If an optimization requires a construct beyond this model (e.g., aliased buffers for in-place operations beyond `update_` flow), the response is to extend the vocabulary — not to bypass it.

---

## Options Considered

### Option A: Plan-prescribed lifetime intervals

The plan carries a `BufferDescriptor` for each plan-level buffer, annotated with `producing_node` (the node whose dispatch creates the buffer's contents) and `last_consumer` (the topologically-last node to read from it). The renderer uses these intervals to compute optimal physical memory reuse — without re-analyzing the DAG or kernel contracts.

```python
@dataclass(frozen=True)
class BufferDescriptor:
    handle: BufferHandle
    logical_name: str
    padded_shape: Tuple[int, ...]
    element_size_bytes: int
    size_bytes: int
    role: BufferRole
    producing_node: Optional[str]
    consumers: FrozenSet[str]
    last_consumer: Optional[str]
```

**Advantages:**
- Lifetime computation is performed once by the shared layer. No renderer reimplements the DAG analysis or kernel-contract flow-prefix interpretation.
- Full plan-time validation: every consumed buffer has a producer; no buffer is produced by two different nodes; lifetime intervals are consistent with the DAG's topological order.
- Vulkan directly uses lifetime intervals for suballocation packing — non-overlapping buffers can share the same `VkDeviceMemory` region.
- The renderer derives its reuse schedule (graph coloring or equivalent) from lifetime intervals with trivial arithmetic, not DAG traversal.

**Disadvantages:**
- Plan size increases by $O(B)$ where $B$ is the number of plan-level buffers (one descriptor per buffer). This is modest — typically 40–60 buffers for the averaging ensembled classifier.
- The `last_consumer` annotation assumes the shared layer's topological order. If a renderer reorders independent nodes (e.g., for latency hiding), the annotation may be conservative — a buffer could potentially be freed earlier than `last_consumer` suggests. But conservative lifetimes are always safe; the renderer is free to tighten them.

### Option B: Renderer-owned lifetimes

The plan declares buffers (name, shape, type, role) but not their lifetimes. Each renderer independently analyzes the DAG to determine buffer liveness at each execution point.

```python
@dataclass(frozen=True)
class BufferDescriptor:
    handle: BufferHandle
    logical_name: str
    padded_shape: Tuple[int, ...]
    element_size_bytes: int
    size_bytes: int
    role: BufferRole
    # No lifetime annotations.
```

**Advantages:**
- Simpler plan data structure — fewer fields per descriptor.
- Renderer has maximum flexibility to derive lifetimes based on its own execution order, which may differ from the shared layer's topological sort.

**Disadvantages:**
- **Duplicated analysis logic.** Every renderer must independently inspect `buffer_bindings`, resolve kernel contract flow prefixes (`src_` = read, `dest_` = write, `update_` = read-write), and perform a topological walk to determine producer/consumer relationships. This is Policy-tier knowledge the shared layer already possesses.
- **Validation gap.** Without lifetime annotations, the shared layer cannot verify at plan-construction time that: every consumed buffer has a producing node; no buffer is produced by two different nodes (single-producer invariant); buffer types and shapes are consistent between producer and consumers. These checks require the same analysis that Option A pre-computes.
- **ADR-001's "shared correctness" principle.** Buffer lifetime computation is non-trivial and kernel-contract-dependent. Duplicating it across renderers is the exact "risk of cross-backend divergence" that ADR-001 warns against.

### Option C: Plan-prescribed allocation coloring

The plan computes an explicit buffer-to-physical-slot assignment. Non-overlapping buffers are mapped to the same physical slot identity, creating an optimal memory reuse schedule.

```python
@dataclass(frozen=True)
class BufferDescriptor:
    handle: BufferHandle
    logical_name: str
    padded_shape: Tuple[int, ...]
    element_size_bytes: int
    size_bytes: int
    role: BufferRole
    physical_slot: int            # Buffers with the same slot share physical memory
    producing_node: Optional[str]
    last_consumer: Optional[str]
```

**Advantages:**
- Maximum memory optimization decided once. The renderer allocates $N$ physical slots and maps handles to slots with zero analysis.
- Deterministic across backends — guarantees identical memory reuse decisions.

**Disadvantages:**
- **Over-constrains the renderer.** The coloring is optimal only under the shared layer's topological order and size-based packing heuristic. A renderer may prefer a fundamentally different allocation strategy:
  - Vulkan suballocates from a single large `VkDeviceMemory` block with device-specific alignment requirements (e.g., `minStorageBufferOffsetAlignment`) that the shared layer cannot know.
  - CPU may alias buffers into a single large arena with OS-level page management.
  - OpenCL may benefit from driver-specific buffer pooling heuristics.
- **The coloring depends on backend-specific granularity.** Physical allocation granularity differs per backend: page-aligned on CPU, 256-byte aligned on some Vulkan devices, driver-determined on OpenCL. A coloring computed in the shared layer would either use a lowest-common-denominator alignment (wasting memory) or require backend-specific alignment parameters in the shared layer (violating the abstraction boundary).
- **Violates rendering freedom.** ADR-001 states the renderer is free to reorder independent nodes, batch dispatches, and apply backend-specific scheduling. If the renderer reorders execution, a pre-computed coloring based on the shared layer's topological order may assign two simultaneously-live buffers to the same physical slot — producing incorrect aliasing. The coloring would need to be conservative (never alias buffers that *could* overlap under any valid ordering), which nullifies the optimization benefit.
- **Crosses the Policy/Orchestration boundary.** The Policy tier determines *what* buffers exist and *when* they are live. The Orchestration tier determines *where* to place them in physical memory. Slot assignment is a physical placement decision.

---

## Analysis

### Eliminating Option B

Option B pushes lifetime analysis to every renderer, duplicating shared-layer knowledge. The analysis parallels ADR-003's elimination of its Option B (which embedded a live `StabilizationPolicy` object in the plan, pushing threshold computation to render time):

- ADR-003 rejected deferring threshold computation because it "pushes threshold computation to render time, forfeiting the shared layer's ability to validate the reduction tree at plan construction." Here, deferring lifetime computation forfeits the shared layer's ability to validate buffer flow correctness at plan construction — the same structural failure.

The lifetime of a buffer depends on two inputs: the DAG topology (which nodes exist and their dependency edges) and the kernel contracts (which `buffer_bindings` entries are reads vs. writes, determined by the `src_`/`dest_`/`update_` flow prefixes). Both are Policy-tier data. Under ADR-001's three-tier model, the Policy tier computes non-trivial shared values; the Orchestration tier handles dispatch mechanics.

The validation argument is decisive. The plan builder can verify at construction time:
- **Single-producer invariant:** Every `BATCH_INTERMEDIATE` buffer has exactly one producing node.
- **Coverage:** Every buffer consumed by a node has a producing node (or is `MODEL_STATE`/`BATCH_INPUT`).
- **Shape consistency:** The producing node's output shape matches every consumer's expected input shape (using `KernelContract` tensor shapes from ADR-007).
- **Acyclic buffer flows:** No circular producer→consumer chains exist (guaranteed by DAG acyclicity, but the buffer flow is a useful secondary check).

Without lifetime annotations, these checks must be deferred to render time or omitted entirely. Deferring violates CONTRACT.md Article 1.4 (pre-dispatch validation in the shared layer).

### Eliminating Option C

Option C over-specifies the plan by prescribing physical allocation — a rendering concern. The analysis parallels ADR-003's elimination of its Option A (which prescribed kernel tier selection):

- ADR-003 rejected prescribing register-vs-local kernel tier because "the register/local crossover heuristic differs fundamentally across backends." Here, physical allocation strategy differs fundamentally: Vulkan packs buffers into large device memory blocks with device-specific alignment; CPU uses OS memory management with page granularity; OpenCL uses driver-managed discrete buffers.

- The coloring is fragile under execution reordering. Consider two buffers $A$ and $B$ whose lifetimes do not overlap under the shared layer's topological sort. Option C assigns them to the same physical slot. If the renderer schedules independent node groups concurrently (permitted by ADR-002's "Concurrency from structure" principle), both buffers may be simultaneously live — and the aliasing is incorrect. The only safe coloring under arbitrary valid orderings is the trivially conservative "never alias," which provides no benefit.

- Worse, backend-specific physical constraints (alignment, page size, memory type requirements) cannot be incorporated into a shared-layer coloring without leaking backend details into the plan. This is the exact boundary violation that ADR-001 prevents.

### Choosing Option A

Option A separates shared from backend concerns at the Policy/Orchestration boundary — the same boundary established by ADR-003 and ADR-004:

**Policy tier pre-computes (carried in the plan):**
- Buffer declarations: logical name, padded shape, element size, total size.
- Buffer role classification: `MODEL_STATE`, `BATCH_INPUT`, `BATCH_INTERMEDIATE`, `BATCH_OUTPUT`.
- Lifetime intervals: `producing_node`, `consumers`, `last_consumer` — derived from DAG topology and kernel contract flow analysis.

**Orchestration tier derives (at render time):**
- Physical allocation: `cl.Buffer`, `VkDeviceMemory` suballocation, `malloc`, arena, or pool.
- Memory reuse schedule: graph coloring, linear scan, or backend-specific packing — based on lifetime intervals, allocation alignment, and memory type constraints.
- Buffer binding: mapping `BufferHandle` → physical address for each kernel dispatch.

This boundary satisfies all decision drivers:

1. **ADR-001:** `BufferHandle` and `BufferDescriptor` are backend-neutral frozen data.
2. **ADR-001 (shared correctness):** Lifetime computation is performed once; no renderer reimplements it.
3. **ADR-002:** `buffer_bindings` references logical names; the namespace resolves them to `BufferHandle`s.
4. **ADR-003, ADR-004:** The two-tier scope is formalized — plan-level buffers have descriptors; renderer-internal buffers (reduction intermediates, scratch buffers) do not.
5. **CONTRACT.md 1.4:** Full plan-time validation of buffer flow correctness.
6. **Backend freedom:** The renderer's physical allocation strategy is unconstrained.

### The two-tier scope formalization

ADR-003's rendering contract specifies that intermediate stage buffers are derived from `elements_per_partial` and per-stage output count — the renderer allocates them. ADR-004's `ScratchBufferSpec` declares per-iteration scratch buffer sizes — the renderer allocates them. This ADR formalizes the boundary:

A buffer is **plan-level** if and only if it appears in the plan's `BufferDescriptor` tuple. All other device memory allocations are **renderer-internal**. The plan builder never creates descriptors for:
- Reduction tree intermediate stage buffers (ping-pong buffers, uploaded offset lists).
- Streaming loop scratch buffers (declared in `ScratchBufferSpec`, allocation renderer-owned).
- Backend-specific staging buffers (Vulkan's staging memory for D2H transfers).

This means the plan's total `size_bytes` across all `BufferDescriptor`s undercounts actual device memory usage. This tension is accommodated by ADR-003's rendering contract (intermediate buffer sizes derivable from `num_stages`, `fan_in`, `elements_per_partial`) and ADR-004's `ScratchBufferSpec` entries (sizes declared explicitly). Together, these make renderer-internal memory overhead *estimable* from plan data alone — the shared layer can compute a lower bound (plan-level) and an estimate (plan-level + derivable renderer-internal) without requiring backend participation.

### Buffer role semantics

The `BufferRole` classification governs how the renderer interprets lifetime annotations:

| Role | Lifetime Semantics | Lifetime Annotations | Reuse Eligible |
| :--- | :----------------- | :------------------- | :------------- |
| `MODEL_STATE` | Persistent across batches. Externally initialized (host upload or checkpoint load). Modified in-place by `adam_update` (the `update_` flow prefix). | `producing_node = None`. `consumers` includes all reading nodes plus the `adam_update` node. `last_consumer = None`. | No — always allocated. |
| `BATCH_INPUT` | Uploaded by the host before each batch execution. Read-only during the DAG. | `producing_node = None`. `consumers` = all reading nodes. `last_consumer` = topologically-last reader. | No — live for entire batch. |
| `BATCH_INTERMEDIATE` | Produced by exactly one DAG node. Consumed by one or more downstream nodes. Dead after the last consumer completes. | `producing_node` = the producing node's `node_id`. `consumers` = all reading nodes' `node_id`s. `last_consumer` = topologically-last consumer. | **Yes** — primary target for memory reuse. |
| `BATCH_OUTPUT` | Produced by a DAG node. Read back to host via a `RetrievalNode`. | `producing_node` = the producing node's `node_id`. `last_consumer` = the `RetrievalNode`'s `node_id`. | No — persists until host retrieval. |

Only `BATCH_INTERMEDIATE` buffers participate in memory reuse optimization. The renderer can determine which `BATCH_INTERMEDIATE` buffers have non-overlapping lifetimes and map them to the same physical memory.

### StreamingLoopNode collection buffers

ADR-004's `StreamingLoopNode.collection_buffers` (e.g., `clipped_partial_grad_shared_weights`, `clipped_partial_grad_shared_biases`) persist beyond the streaming loop for consumption by downstream `ReductionTreeNode`s (Node 20). The lifetime annotations model the producing node as the `StreamingLoopNode` itself — not the individual body `KernelDispatchNode`s, which are internal to the loop and execute multiple times (each iteration writing to a different slice). The buffer as a whole becomes available when the `StreamingLoopNode` completes.

Similarly, for `ReductionTreeNode`, the `destination_buffer` is produced by the `ReductionTreeNode` (the node's completion makes the final reduced result available). The `source_buffer` is consumed by the `ReductionTreeNode` (the node reads from it during its internal multi-stage reduction). The internal intermediate buffers are renderer-internal and carry no plan-level descriptors.

---

## Decision

**Option A: Plan-prescribed lifetime intervals.**

The plan carries a `BufferDescriptor` for each plan-level buffer, annotated with producing node, consumer set, and topologically-last consumer. The renderer uses these annotations to optimize physical memory allocation and reuse at render time. The plan builder computes all lifetime annotations from the DAG topology and kernel contract flow analysis, exactly once.

### Data structures

```python
import enum
from dataclasses import dataclass
from typing import Optional, Tuple, FrozenSet


class BufferRole(enum.Enum):
    """Classifies a plan-level buffer by its lifecycle scope.

    Determines how the renderer interprets lifetime annotations and
    whether the buffer is eligible for physical memory reuse.
    """
    MODEL_STATE = "model_state"
    BATCH_INPUT = "batch_input"
    BATCH_INTERMEDIATE = "batch_intermediate"
    BATCH_OUTPUT = "batch_output"


@dataclass(frozen=True)
class BufferHandle:
    """
    An opaque, immutable token identifying a plan-level buffer.

    Assigned by the plan builder at plan-construction time.
    Mapped to physical device memory by the renderer at render time.
    Backend-neutral — carries no device type, address, or allocation detail.

    Invariant: id is unique within a plan's buffer namespace.
    """
    id: int


@dataclass(frozen=True)
class BufferDescriptor:
    """
    The complete declaration of a single plan-level buffer.

    Pure data. Computed by the plan builder. Consumed by the renderer.
    Carries shape, type, role, and lifetime annotations sufficient
    for the renderer to allocate physical memory and determine reuse.
    """
    handle: BufferHandle
    """Opaque token for cross-referencing this buffer across plan nodes.
    Plan nodes reference buffers by logical_name in buffer_bindings;
    the plan's buffer namespace resolves names to handles."""

    logical_name: str
    """Canonical name (e.g., 'hidden_activations', 'partial_grad_module_weights').
    Matches the names used in KernelDispatchNode.buffer_bindings,
    ReductionTreeNode.source_buffer / destination_buffer, and
    StreamingLoopNode.collection_buffers.
    Invariant: unique within the plan's buffer namespace."""

    padded_shape: Tuple[int, ...]
    """Physical, padded tensor shape. Computed by MemoryLayout from the
    kernel's Padding Contract (CONTRACT.md Article 3.1). The renderer
    allocates prod(padded_shape) * element_size_bytes bytes."""

    element_size_bytes: int
    """Size of one scalar element in bytes. Derived from PrecisionConfig
    (ADR-008): e.g., 2 for float16, 4 for float32, 4 for int32.
    The renderer maps this to its native type system."""

    size_bytes: int
    """Total physical buffer size in bytes.
    Invariant: size_bytes == prod(padded_shape) * element_size_bytes."""

    role: BufferRole
    """Lifecycle classification. Determines lifetime semantics and
    reuse eligibility (see BufferRole table in Analysis)."""

    producing_node: Optional[str]
    """node_id of the plan node that writes this buffer's contents.

    None for MODEL_STATE (persistent, externally initialized) and
    BATCH_INPUT (uploaded by host before plan execution).

    For KernelDispatchNode outputs: the node's node_id.
    For StreamingLoopNode collection_buffers: the StreamingLoopNode's
    node_id (the loop as a whole produces the collection).
    For ReductionTreeNode destination_buffer: the ReductionTreeNode's
    node_id (the tree as a whole produces the final result).

    Invariant: for BATCH_INTERMEDIATE, producing_node is not None.
    Invariant: at most one producing node per buffer (single-assignment)."""

    consumers: FrozenSet[str]
    """node_ids of all plan nodes that read from this buffer.

    Includes KernelDispatchNode sources (src_ bindings),
    ReductionTreeNode source_buffer references,
    StreamingLoopNode body nodes reading from collection_buffers of
    an upstream loop, and RetrievalNode source references.

    For MODEL_STATE buffers with update_ access (adam_update):
    the update node appears in consumers (it reads the current value)."""

    last_consumer: Optional[str]
    """node_id of the topologically-last consumer in the plan's DAG.
    Computed by the plan builder via topological sort of all consumers.

    None for MODEL_STATE (always live — lifetime not plan-managed).

    For BATCH_INTERMEDIATE: the node after which the buffer's physical
    memory may be reclaimed or reused by the renderer.
    For BATCH_OUTPUT: the RetrievalNode's node_id (buffer must persist
    until host retrieval completes).
    For BATCH_INPUT: the topologically-last reading node."""
```

### Plan construction contract

The plan builder constructs a `BufferDescriptor` for every plan-level buffer by:

1. **Enumerating all buffer names.** Scan the plan's DAG nodes: every `buffer_bindings` value, every `source_buffer` and `destination_buffer`, every `collection_buffers` value, and every `RetrievalNode` source. Collect the unique set of logical buffer names.

2. **Resolving shapes and sizes.** For each buffer name, look up the `MemoryLayout` from `ParameterSpace.get_all_memory_layouts()` and the element size from `PrecisionConfig` (ADR-008). Compute `padded_shape`, `element_size_bytes`, and `size_bytes`.

3. **Classifying roles.** Assign `BufferRole` based on the buffer's position in the architecture:
   - `MODEL_STATE`: learnable parameters (`shared_weights`, `shared_biases`, `module_weights`, `module_biases`, `temperatures`) and optimizer state (`m1_*`, `m2_*`).
   - `BATCH_INPUT`: host-uploaded per-batch data (`input`, `targets_cce`/`targets_bce`, `sample_mask`).
   - `BATCH_OUTPUT`: buffers consumed by `RetrievalNode`s (`final_probs`, `final_loss` via `inference_retrieval`).
   - `BATCH_INTERMEDIATE`: all other buffers (`hidden_activations`, `logits`, `partial_grad_*`, `clipped_partial_grad_*`, `summed_grad_*`, `final_grad_*`, `permuted_grad_h`, `hidden_mask`).

4. **Computing lifetime annotations.** For each buffer:
   - **`producing_node`:** Identify the unique node whose `buffer_bindings` maps a `dest_` or `update_` flow-prefixed parameter to this buffer name. For `ReductionTreeNode` outputs: the tree node's `node_id`. For `StreamingLoopNode` collection buffers: the loop node's `node_id`. For `MODEL_STATE` and `BATCH_INPUT`: `None`.
   - **`consumers`:** Collect all `node_id`s whose `buffer_bindings` map a `src_` flow-prefixed parameter to this buffer name, plus any `ReductionTreeNode`s that reference it as `source_buffer`, plus any `RetrievalNode`s that reference it. For `MODEL_STATE` buffers modified by `adam_update`: include the update node.
   - **`last_consumer`:** Topologically sort the plan's DAG. Among all entries in `consumers`, select the one latest in topological order. For `MODEL_STATE`: `None`.

5. **Assigning handles.** Assign a unique `BufferHandle(id=i)` to each buffer, with `id` values allocated sequentially starting from 0.

6. **Validating invariants.** At construction time:
   - Every `logical_name` is unique across all descriptors.
   - Every `handle.id` is unique across all descriptors.
   - For `BATCH_INTERMEDIATE` buffers: `producing_node is not None`.
   - Single-producer invariant: no two nodes produce the same buffer.
   - Coverage: every buffer referenced in any node's `buffer_bindings` (as a `src_` parameter) has a descriptor with a `producing_node` or is `MODEL_STATE`/`BATCH_INPUT`.
   - `size_bytes == prod(padded_shape) * element_size_bytes` for every descriptor.
   - `last_consumer` is in `consumers` (or both are `None`).
   - `last_consumer` is topologically after `producing_node` in the DAG (when both are present).

### Backend rendering contract

The renderer interprets the plan's `BufferDescriptor` tuple as the Orchestration tier (ADR-001 §Three-tier jurisdictional model). It performs the following steps:

1. **Allocate physical memory.** For each `BufferDescriptor`, allocate `size_bytes` of device memory using the backend's native mechanism:
   - **OpenCL:** `cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=size_bytes)`.
   - **Vulkan:** Suballocate from a `VkDeviceMemory` block with appropriate memory type and alignment.
   - **CPU:** `malloc(size_bytes)` or arena allocation.

2. **Optimize reuse (optional).** The renderer may use the `role`, `producing_node`, and `last_consumer` annotations to identify `BATCH_INTERMEDIATE` buffers with non-overlapping lifetimes and map them to the same physical allocation. The algorithm is the renderer's choice — graph coloring, linear scan, first-fit-decreasing, or no reuse at all (allocate every buffer independently). The plan does not prescribe reuse; it enables it.

3. **Build the handle-to-physical map.** Construct a mapping from `BufferHandle` → native buffer reference (`cl.Buffer`, `VkBuffer` offset, pointer). This map is the renderer's internal state, not visible to the shared layer.

4. **Resolve buffer bindings at dispatch time.** When dispatching a `KernelDispatchNode`, resolve each `buffer_bindings` entry: logical_name → `BufferHandle` (via the namespace) → physical buffer (via the handle map). Bind the physical buffer to the kernel parameter using the backend's native mechanism (`clSetKernelArg`, push constants + descriptor sets, C function arguments).

5. **Respect lifetime boundaries.** The renderer must not reclaim or reuse a buffer's physical memory before its `last_consumer` node has completed execution (as determined by the backend's synchronization model). After the `last_consumer` completes, the physical memory is eligible for reuse. For `MODEL_STATE` buffers: never reclaim.

### Relationship to the ExecutionPlan

The plan's buffer namespace is a top-level field on the `ExecutionPlan`:

```python
@dataclass(frozen=True)
class ExecutionPlan:
    # ... existing fields from ADR-002 (nodes, dependency edges) ...
    buffer_descriptors: Tuple[BufferDescriptor, ...]
    """The complete plan-level buffer namespace.
    Every buffer referenced by any plan node has a descriptor here.
    Renderer-internal buffers (reduction intermediates, scratch buffers)
    are absent — they are the renderer's concern."""
```

Plan nodes continue to reference buffers by `logical_name` (strings) in their `buffer_bindings`, `source_buffer`, `destination_buffer`, and `collection_buffers` fields. The renderer resolves names to `BufferHandle`s via the descriptor tuple, and handles to physical allocations via its internal map.

### Plan-level memory footprint estimation

The shared layer can compute a **plan-level memory footprint** as the sum of `size_bytes` across all `BufferDescriptor`s. This undercounts actual device memory usage because renderer-internal buffers are absent. However, the shared layer can estimate the renderer-internal overhead:

- **Reduction tree intermediates:** For each `ReductionTreeNode` with plan `p`: the intermediate buffer sizes are derivable from `p.elements_per_partial`, `p.fan_in`, and `p.num_stages` (ADR-003 §Backend rendering contract). The ping-pong scheme requires at most two intermediate buffers at any stage.
- **Streaming loop scratch:** For each `StreamingLoopNode`'s `ScratchBufferSpec` entries: the `size_bytes` field gives the exact per-buffer allocation (ADR-004).
- **Offset list uploads:** Each `ReductionTreeNode`'s `initial_offset_list` requires `len(initial_offset_list) * sizeof(int)` bytes on device.

The sum of plan-level + estimated renderer-internal gives the shared layer a reliable **total memory estimate** for pre-flight validation — verifying the plan fits within the `HardwareProfile`'s available memory (ADR-006) before committing to execution.

---

## Consequences

### Positive

- **Single computation of lifetime intervals.** Producing nodes, consumer sets, and last-consumer annotations are computed once by the shared layer from DAG topology and kernel contract flow analysis. No renderer reimplements this logic. Cross-backend divergence in lifetime determination is structurally prevented.

- **Full plan-time validation.** All buffer flow invariants — single-producer, coverage, shape consistency, topological ordering of producer before last-consumer — are verifiable at plan-construction time without any backend. This satisfies CONTRACT.md Article 1.4.

- **Backend allocation freedom.** Physical allocation, memory reuse scheduling, and buffer binding are Orchestration-tier concerns. Each backend applies its native mechanism: Vulkan suballocates with device-specific alignment; OpenCL allocates discrete buffers; CPU uses arena or pool allocation. The plan enables all strategies without constraining any.

- **Vulkan suballocation directly enabled.** Vulkan's `VkDeviceMemory` packing benefits directly from lifetime intervals. The renderer identifies non-overlapping `BATCH_INTERMEDIATE` buffers and suballocates them from a shared memory region — a critical optimization for memory-constrained GPUs. The plan's annotations make this a straightforward computation, not a DAG re-analysis.

- **Two-tier buffer scope formalized.** The boundary between plan-level buffers (have descriptors) and renderer-internal buffers (no descriptors, sizes derivable from plan data) is a first-class invariant. This clarifies the memory accounting model: plan-level is exact; renderer-internal is estimable.

- **Inspectable buffer flow.** The `BufferDescriptor` tuple is a complete, ordered record of every plan-level buffer's identity, shape, role, and lifetime. It can be logged, compared across training steps, and validated against expected buffer counts — all without executing any kernels.

- **Consistent architectural pattern.** The lifetime-intervals-in-the-plan pattern follows ADR-003's thresholds-in-the-plan pattern and ADR-004's strides-in-the-plan pattern: the shared layer pre-computes what is non-trivial (lifetimes); the renderer derives what is trivial (physical allocation from sizes) or backend-specific (memory type, alignment, reuse schedule).

### Negative

- **Renderer must still compute reuse schedule.** The lifetime intervals are inputs to a memory reuse algorithm (e.g., graph coloring), which the renderer must implement. This is a deliberate consequence of preserving backend allocation freedom — it is trivial arithmetic, not DAG analysis.

- **Conservative last-consumer under reordering.** If a renderer reorders independent nodes, a buffer's `last_consumer` may be later than the earliest point at which the buffer is actually dead. The renderer is free to tighten lifetimes by re-analyzing the DAG under its actual execution order — the plan's annotations are an upper bound on lifetime extent, always safe.

- **Plan size.** Each `BufferDescriptor` adds a fixed-size record to the plan. For the averaging ensembled classifier, this is ~40–60 descriptors — negligible relative to the `initial_offset_list` arrays in `ReductionTreePlan` or the `buffer_bindings` dictionaries on `KernelDispatchNode`s.

- **MODEL_STATE annotations are informational only.** `MODEL_STATE` buffers carry `producing_node = None` and `last_consumer = None` because their lifetimes are not plan-managed — they persist across batches. The annotations exist for completeness and validation (ensuring all consumers are listed) but do not drive reuse decisions. This is a minor redundancy justified by namespace completeness.

### Migration implications

Per ADR-002's migration path:

1. **Define the data structures** (`BufferRole`, `BufferHandle`, `BufferDescriptor`) as frozen types in the shared layer (e.g., `shared/buffer_handles.py` per ADR-012's proposed structure). The `BufferHandle` type is extracted from `launcher_infra.py` into the shared layer, removing its OpenCL dependency. No existing code is modified at this step.

2. **Add buffer namespace construction** to the `ExecutionPlanBuilder`: for each plan being built, enumerate all buffer names from the DAG nodes, resolve shapes from `ParameterSpace.get_all_memory_layouts()`, classify roles, compute producing nodes and consumer sets from `buffer_bindings` and kernel contract flow prefixes, and determine `last_consumer` via topological analysis. The resulting `Tuple[BufferDescriptor, ...]` is stored on the `ExecutionPlan`.

3. **Write plan-level tests** (ADR-016, layer 1) that validate:
   - Single-producer invariant: no buffer has two producing nodes.
   - Coverage: every consumed buffer has a producing node or is `MODEL_STATE`/`BATCH_INPUT`.
   - Shape consistency: producer and consumer shapes agree for every buffer.
   - Topological ordering: `producing_node` precedes `last_consumer` in the DAG.
   - Name and handle uniqueness across all descriptors.
   - `size_bytes` correctness: `prod(padded_shape) * element_size_bytes`.
   - Role classification consistency: `MODEL_STATE` buffers correspond to learnable parameters and optimizer state; `BATCH_INPUT` buffers correspond to host-uploaded data.
   - Memory footprint estimation: plan-level + renderer-internal estimate fits within `HardwareProfile` available memory.

4. **OpenCL renderer** (Phase 3 of ADR-001 migration) interprets `BufferDescriptor` tuples by replacing the current `BufferManager.create_named_buffer()` calls. The renderer iterates over descriptors, allocates `cl.Buffer`s, and builds the handle-to-`cl.Buffer` map. Memory reuse (if implemented) uses lifetime intervals from the descriptors.

5. **CPU and Vulkan renderers** (Phase 4 & 5) consume the same `BufferDescriptor` tuples with their native allocation models. The Vulkan renderer's suballocation packing is the primary beneficiary of lifetime annotations.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; shared correctness requirement; three-tier jurisdictional model (Policy / Orchestration / Execution)
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `buffer_bindings` on `KernelDispatchNode`; `DependencyProvider` dissolution; `RetrievalNode` definition
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — two-tier buffer scope; renderer-internal intermediates (ping-pong, offset lists); parametric header precedent
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `ScratchBufferSpec` (renderer-internal scratch); `collection_buffers` lifetime semantics; stride-based parametric pattern
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` flow declarations; buffer shape consistency validation
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` frozen dataclass; `element_size_bytes` derivation
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §5 Host Orchestrator & Execution Policies (activation lifecycle, chunk definition); §8 Reduction Planning & Rendering
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 2.1 Buffer Name Grammar (flow prefixes: `src_`, `dest_`, `update_`); Article 3.1 Padding Contract
