# ADR-030: Module-Chunk Isolation in Parameter-Gradient Reduction Trees

**Status:** PROPOSED
**Date:** 2026-04-08
**Deciders:** —
**Triggered by:** Discovery that `TiledGather` conflates gradients for disjoint parameter sets when `num_module_chunks > 1`
**Depends on:** ADR-002 (Plan Node Types), ADR-003 (Reduction Tree Plan), ADR-007 (Kernel Signature Contract/Binding Split), ADR-009 (Buffer Lifecycle), ADR-019 (K-Fan-In Reduction Kernel Primitive)
**Constrains:** `kernels.cl.h` (`adam_update`, `clamp_temperatures`), `build_learn_plan()` Phase II and Phase V construction, optimizer state buffer allocation
**Amends:** CONCEPT.md §2 (Recursive Aggregation Engine), CONCEPT.md §4 (Synchronization Barriers), Kernel Contract for Node 24 (`adam_update`), Kernel Contract for Node 25 (`clamp_temperatures`), Node 22 (Batch Synchronization Barrier)

---

## Decision Summary

| Decision Axis              | Selected Option                                                                              | Rationale                                                                                                         |
| :------------------------- | :------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- |
| Gather primitive           | **Option C** — New `ModuleChunkGather` primitive                                             | Structural enforcement; respects Architectural Elegance Feedback                                                  |
| Plan builder restructuring | Per-module-chunk sub-graphs in Phases II and V                                               | Enables per-chunk parallelism; eliminates monolithic barrier                                                      |
| Kernel contract amendment  | Add `parameter_offset` and `total_parameter_count` to `adam_update` and `clamp_temperatures` | Enables slice-indexed dispatch into monolithic state buffers; preserves Article 1.4.1 Calculability Proof closure |
| Barrier decomposition      | Node 22 retired as a discrete barrier; property preserved by DAG dependency edges            | Removes false serialization; per-chunk dependency chains suffice                                                  |

---

## Context

### The Defect

When the Host Orchestrator decomposes the module dimension into `num_module_chunks > 1` chunks, the system's `grid_mod_cls` placement strategy assigns each tile a `flat_tile_index` that encodes both a `module_chunk_index` and a `class_chunk_index`:

```
flat_tile_index = module_chunk_index * num_class_chunks + class_chunk_index
```

Tiles from the **same** module chunk but different class chunks write to non-overlapping class positions within the same `modules_per_chunk`-wide parameter slice. Their element-wise sum correctly fills the class dimension — this is the documented behavior of the `ZERO_REQUIRED` initialization pattern on Node 8's output buffers.

However, tiles from **different** module chunks write gradient values for **disjoint physical parameters** at the **same positional indices** within their respective `elements_per_partial`-wide regions. If the Indirection Contract's offset list for a reduction tree mixes tiles from different module chunks into a single reduction node, the element-wise summation **conflates gradients for different parameters**, producing silent, catastrophic corruption of all module-specific parameter updates.

### Scope of Impact

The defect affects only the parameter-gradient reduction trees:

| Reduction Path             | Affected? | Reason                                                                                                              |
| :------------------------- | :-------- | :------------------------------------------------------------------------------------------------------------------ |
| **Node 15 (Grad_ModW)**    | **Yes**   | Tiles index by `(module_chunk, class_chunk)`                                                                        |
| **Node 15 (Grad_ModB)**    | **Yes**   | Same indexing                                                                                                       |
| **Node 15 (Grad_Temps)**   | **Yes**   | Same indexing                                                                                                       |
| Node 14 (Probs, Loss)      | No        | Diagnostic reduction; output layout is `(module, batch, class)` — tile placement encodes the correct final position |
| Node 16 (Grad_H)           | No        | Permuted to contiguous SoA by Node 13; reduces over _all_ modules simultaneously via a dedicated kernel             |
| Node 20 (Grad_SW, Grad_SB) | No        | Shared-layer gradients have no module dimension; use `linear_batch` placement                                       |

The defect is **latent when `num_module_chunks == 1`** (the common case for small model configurations), because all tiles belong to the single module chunk and no cross-contamination is possible. It manifests only under memory pressure that forces module-dimension decomposition.

### Root Cause

The existing `TiledGather` primitive returns offsets for **all** tiles across **all** module chunks, treating the flat tile index space as a homogeneous collection. The Recursive Aggregation Engine (CONCEPT.md §2) then constructs a reduction tree over this undifferentiated set, violating the implicit invariant that tiles being summed must correspond to the same physical parameters.

No explicit invariant in CONCEPT.md, CONTRACT.md, or the kernel headers prohibits this cross-module-chunk mixing. The constraint is implicit in the mathematical structure of the gradient computation but is not encoded in the system's formal contracts.

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** The defect is a vocabulary gap: the gather primitive set lacks a construct that expresses module-chunk scoping. Per §1, the response is to formalize the pattern as a documented architectural primitive — not to filter offset lists ad-hoc or constrain the plan builder.

2. **CONCEPT.md §2 (Recursive, Tiered Aggregation Engine).** The reduction engine's correctness depends on the Indirection Contract delivering offset lists that refer to compatible partials. Module-chunk isolation is a precondition for this compatibility for parameter-gradient paths.

3. **CONCEPT.md §3 (Modular, "Dumb" Kernels).** The `adam_update` and `clamp_temperatures` kernels must remain stateless and host-parameterized. Slice-indexed dispatch via a `parameter_offset` scalar is consistent with existing patterns (`forward_pass` has `batch_chunk_offset`, `render_logits_chunk` has `module_chunk_offset`/`class_chunk_offset`).

4. **ADR-002 (Plan Node Types & Synchronization Structure).** ADR-002 specifies Node 22 as a `BarrierNode` gating all `adam_update` dispatches. Per-module-chunk sub-graphs require re-evaluating whether this monolithic barrier remains necessary or becomes a false serialization point.

5. **ADR-003 (Reduction Tree Plan).** The `ReductionTreePlan` consumes a `GatherPrimitive`'s offset list to construct staged reduction. The new gather primitive must conform to the `GatherPrimitive` interface established by ADR-003.

6. **ADR-009 (Buffer Lifecycle).** Optimizer state buffers (`m1`, `m2`) are `MODEL_STATE` lifecycle role — persistent across batches. Per-module-chunk dispatch requires these buffers to be full-model-sized, with each dispatch indexing into them via offset.

7. **ADR-019 (K-Fan-In Reduction Kernel Primitive).** Multi-stage reduction trees using `reduce_k_fan_in_and_clip` are correct per-node — but only if the offset lists feeding them respect module-chunk boundaries. This ADR provides the upstream guarantee that ADR-019's kernel receives well-formed inputs.

8. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability).** The new gather primitive and amended kernel contracts must be fully validatable at plan-construction time with calculability proofs for all buffer dimensions and preconditions for all scalar parameters. Per Article 1.4.1, all parameters used in Calculability Proofs and Validation Preconditions must exist in the interface.

---

## Options Considered

### Option A: Constrain plan builder to `num_module_chunks == 1`

Force the plan builder to never decompose the module dimension into multiple chunks. When memory is insufficient for the full module set, raise an error rather than chunking.

**Advantages:**

- Zero code changes to gather primitives or kernels.
- Eliminates the broken code path entirely.

**Disadvantages:**

- **Violates CONCEPT.md §1.** Constraining the system to avoid a vocabulary gap rather than extending the formal primitive set is architecturally inadmissible.
- **Breaks the Primacy of Memory Strategy.** Module-dimension decomposition is the orchestrator's primary tool for fitting large model configurations into bounded device memory. Removing it renders The Hydra scenario unreachable.
- **Masks the invariant.** The constraint that partials within a reduction tree must correspond to the same physical parameters is a real, permanent property of the system's mathematics. Hiding it behind a plan-builder limit defers the problem rather than encoding it.

### Option B: Filter `TiledGather` offset lists at plan-build time

Retain the existing `TiledGather` primitive but add a post-hoc filter in the plan builder that partitions the returned offset list by module chunk, constructing separate reduction trees per partition.

**Advantages:**

- Minimal code change — the gather primitive is untouched; only the plan builder adds a partitioning step.
- Produces correct reduction trees.

**Disadvantages:**

- **Violates CONCEPT.md §1.** The module-chunk isolation invariant exists only as implicit logic in the plan builder, not as a structural property of the gather primitive. A future caller of `TiledGather` would have no contract-level indication that its offset list contains mixed module chunks.
- **Fragile.** Any plan-builder refactoring that restructures the Phase II loop must re-derive the partitioning logic. The invariant is not self-enforcing.
- **Semantically misleading.** `TiledGather.get_offsets()` returns offsets for all tiles, but only a subset is valid for any given reduction tree. The primitive's interface promises more than is contractually safe to consume.

### Option C: New `ModuleChunkGather` primitive with per-chunk sub-graphs

Introduce a new `GatherPrimitive` subclass — `ModuleChunkGather` — that structurally produces offset lists scoped to a single module chunk. The plan builder constructs `num_module_chunks` independent sub-graphs, each consuming its own `ModuleChunkGather`. Kernel contracts for `adam_update` and `clamp_temperatures` are amended with `parameter_offset` and `total_parameter_count` scalars for slice-indexed dispatch into monolithic state buffers.

**Advantages:**

- **Structural enforcement.** The invariant is encoded in the gather primitive's type — it is impossible to construct a `ModuleChunkGather` that spans multiple module chunks.
- **Respects CONCEPT.md §1.** The optimization pressure (per-chunk reduction) is formalized as a first-class primitive, reified through revised contracts, and implemented through host-controlled parameter switches.
- **Enables per-chunk parallelism.** Independent sub-graphs have no cross-dependencies, permitting the backend to exploit concurrency per CONCEPT.md §6.
- **Eliminates Node 22's monolithic barrier.** Each per-chunk update chain depends only on its own normalized gradient — no global synchronization is needed between module chunks.
- **Preserves Article 1.4.1 closure.** The addition of `total_parameter_count` keeps all buffer-shape proofs and bounds preconditions expressible in terms of interface parameters.
- **Clean degeneration.** When `num_module_chunks == 1`, the loop executes once, `parameter_offset = 0`, `parameter_count = total_parameter_count`, and all behavior is identical to the pre-fix state.

**Disadvantages:**

- **Kernel interface changes.** `adam_update` and `clamp_temperatures` gain `parameter_offset` and `total_parameter_count` parameters — a breaking change to `kernels.cl.h` requiring updates across all three backends.
- **Plan builder restructuring.** Phases II and V change from single-tree to per-module-chunk loops.
- **Buffer sizing correction.** Optimizer state buffers must be resized from per-chunk to full-model, which may require migration logic for existing checkpoints.

### Option D: Parameterize `TiledGather` with `module_chunk_index`

Add an optional `module_chunk_index` parameter to the existing `TiledGather` class. When provided, `get_offsets()` returns only the tiles for that module chunk; when absent, it returns all tiles (preserving current behavior for non-parameter-gradient paths).

**Advantages:**

- Reuses existing class hierarchy — no new `GatherPrimitive` subclass needed.
- Backward-compatible: callers that omit the parameter get unchanged behavior.

**Disadvantages:**

- **Dual-mode primitive.** A single class with optional filtering conflates two semantically distinct operations: "gather all tiles" and "gather one module chunk's tiles." The presence or absence of a parameter determines whether the offset list is safe for parameter-gradient reduction — this is invisible at the type level.
- **Violates CONCEPT.md §3 (Modular, "Dumb" Kernels, applied to host primitives).** Modal behavior controlled by an optional parameter is the host-side equivalent of a "smart kernel." The gather primitive should express exactly one semantic.
- **Weaker validation.** Type-level enforcement is lost: nothing prevents a caller from passing `module_chunk_index=None` to a parameter-gradient reduction tree, recreating the original defect.

---

## Analysis

### Eliminating Option A

Option A is architecturally inadmissible. CONCEPT.md §1 prescribes that vocabulary gaps be resolved by extending the formal primitive set, not by constraining the system to avoid the gap. Module-dimension decomposition is a core capability for large model configurations (The Hydra scenario). Removing it converts a vocabulary problem into a capability regression.

### Eliminating Option B

Option B produces correct reduction trees but encodes the invariant as implicit plan-builder logic rather than as a structural property of the gather primitive. This violates CONCEPT.md §1's three-step protocol: the invariant must be _formalized_ as a documented primitive, not hidden in orchestration code. A future plan-builder refactoring that calls `TiledGather` without the ad-hoc filter would silently reintroduce the defect.

### Comparing Options C and D

Both options correctly partition offset lists by module chunk. The distinction is structural enforcement:

- **Option D** (parameterized `TiledGather`) makes the partition a runtime behavior toggle on an existing type. The type system cannot distinguish "all-tile gather" from "module-chunk gather." A caller that forgets the parameter gets silently incorrect offsets for parameter-gradient reduction.

- **Option C** (new `ModuleChunkGather`) makes the partition a type-level property. The primitive's constructor requires `module_chunk_index` — there is no mode in which it returns cross-chunk offsets. Plan-builder code that constructs a parameter-gradient reduction tree with a `ModuleChunkGather` is correct by construction; code that attempts to use a `TiledGather` for the same purpose is a type error.

The additional cost of Option C (a new class, kernel amendments, buffer resizing) is real but bounded. The kernel `parameter_offset` pattern is precedented (`forward_pass`, `render_logits_chunk`), and the buffer resizing is a one-time correction that the Buffer Lifecycle model (ADR-009) already accommodates.

---

## Decision

**Option C: New `ModuleChunkGather` primitive with per-module-chunk sub-graphs**, applied through the Architectural Elegance Feedback protocol:

1. **Suspend** — no ad-hoc offset-list filtering or conditional reduction logic.
2. **Formalize** — `ModuleChunkGather` is the documented primitive expressing module-chunk isolation.
3. **Reify** — implemented through the gather contract, plan builder loop, and amended kernel contracts.

### 1. `ModuleChunkGather` Primitive

#### 1.1 Definition

```python
@dataclass(frozen=True)
class ModuleChunkGather(GatherPrimitive):
    """Gathers only the tiles belonging to a single module chunk.

    The TilingScheme orders tiles as:
        flat_tile_index = module_chunk_index * num_class_chunks + class_chunk_index

    Each module chunk's tiles are therefore a contiguous subsequence of length
    num_class_chunks, starting at offset (module_chunk_index * num_class_chunks).

    This primitive encodes the Module-Chunk Isolation Invariant (ADR-030) as a
    structural property of the gather operation: only tiles contributing gradients
    for the same physical parameter slice are included in a single reduction tree.
    """
    scheme: TilingScheme
    module_chunk_index: int
    buffer_kind: ModuleBufferKind

    def __post_init__(self):
        if not (0 <= self.module_chunk_index < self.scheme.num_module_chunks):
            raise ValueError(
                f"module_chunk_index {self.module_chunk_index} out of range "
                f"[0, {self.scheme.num_module_chunks})"
            )

    @property
    def num_partials(self) -> int:
        return self.scheme.num_class_chunks

    @property
    def elements_per_partial(self) -> int:
        return self.scheme.elements_per_tile(self.buffer_kind)

    def get_offsets(self) -> np.ndarray:
        base = self.module_chunk_index * self.scheme.num_class_chunks
        return (
            (np.arange(self.num_partials, dtype=SCALAR_UINT_TYPE) + base)
            * self.elements_per_partial
        )
```

#### 1.2 `ModuleBufferKind` Enumeration

```python
class ModuleBufferKind(enum.Enum):
    """Identifies the parameter-gradient buffer for elements_per_tile dispatch."""
    WEIGHTS = "weights"   # modules_per_chunk × padded_hidden_count × padded_total_output_class_count
    BIASES  = "biases"    # modules_per_chunk × padded_total_output_class_count
    TEMPS   = "temps"     # modules_per_chunk
```

`TilingScheme.elements_per_tile(kind)` returns the product of the appropriate dimensions. This calculation belongs to `TilingScheme` because it is the canonical descriptor for workload partitioning. Implementation requires adding `modules_per_chunk`, `padded_hidden_count`, and `padded_total_output_class_count` as stored fields — `modules_per_chunk` is currently computed transiently inside `get_tile()` via ceiling division, and the two padded-dimension fields do not yet exist. These fields are workload-shape descriptors already known at tiling construction time in the plan builder.

#### 1.3 `TiledGather` Scope Clarification

`TiledGather` **remains the correct primitive** for non-parameter-gradient reduction paths. Node 14's diagnostic reduction (Probs, Loss) gathers all tiles across all module chunks — the output layout encodes the correct final position per tile, so cross-module-chunk summation is mathematically valid. `TiledGather` is deprecated **solely for parameter-gradient reduction trees** (Nodes 15 Weight/Bias/Temp paths), where cross-module-chunk mixing conflates gradients for disjoint parameter sets.

Plan-builder code that constructs a parameter-gradient `ReductionTreeNode` should accept only `ModuleChunkGather` instances; code that constructs a diagnostic `ReductionTreeNode` should continue to accept `TiledGather`.

#### 1.4 Offset Arithmetic Verification

For a configuration with `num_module_chunks=3`, `num_class_chunks=4`:

```
Total tiles: 12
Module chunk 0: tiles [0, 1, 2, 3]     → offsets [0, 1, 2, 3] × epp
Module chunk 1: tiles [4, 5, 6, 7]     → offsets [4, 5, 6, 7] × epp
Module chunk 2: tiles [8, 9, 10, 11]   → offsets [8, 9, 10, 11] × epp
```

Each module chunk's reduction tree receives exactly `num_class_chunks` tiles. Those tiles wrote non-overlapping class positions for the same `modules_per_chunk` parameters (Node 8's `ZERO_REQUIRED` initialization fills gaps). Element-wise summation correctly reconstructs the full gradient for the module chunk's parameter slice. ✓

#### 1.5 Degeneration

When `num_module_chunks == 1`:

- `module_chunk_index = 0`
- `num_partials = num_class_chunks = total_tile_count`
- `get_offsets()` returns `[0, 1, ..., total_tile_count-1] × epp`
- This equals the full set of tile offsets formerly returned by `TiledGather`
- The plan builder loop (§2) executes once
- All downstream behavior is identical to the pre-fix state ✓

### 2. Plan Builder Restructuring

#### 2.1 Phase II: Per-Module-Chunk Gradient Reduction

The current Phase II constructs a single reduction tree per parameter group over all tiles. The amended Phase II constructs `num_module_chunks` independent sub-graphs:

```python
for m in range(num_module_chunks):
    # Gather tiles for this module chunk
    gather_w = ModuleChunkGather(scheme, m, ModuleBufferKind.WEIGHTS)
    gather_b = ModuleChunkGather(scheme, m, ModuleBufferKind.BIASES)
    gather_t = ModuleChunkGather(scheme, m, ModuleBufferKind.TEMPS)

    # Build reduction trees (ReductionTreeNode)
    reduce_w[m] = build_reduction_tree(gather_w, clipped_grad_modw_collection, policy)
    reduce_b[m] = build_reduction_tree(gather_b, clipped_grad_modb_collection, policy)
    reduce_t[m] = build_reduction_tree(gather_t, clipped_grad_temps_collection, policy)

    # Normalize (KernelDispatchNode)
    norm_w[m] = normalize_gradients(reduce_w[m].output, param_count_w_chunk)
    norm_b[m] = normalize_gradients(reduce_b[m].output, param_count_b_chunk)
    norm_t[m] = normalize_gradients(reduce_t[m].output, param_count_t_chunk)
```

Each sub-graph is a self-contained `ReductionTreeNode` → `KernelDispatchNode` chain with no dependency edges connecting it to other module-chunk sub-graphs.

#### 2.2 Phase V: Per-Module-Chunk Parameter Update

```python
for m in range(num_module_chunks):
    offset_w = m * modules_per_chunk * padded_hidden_count * padded_total_output_class_count
    offset_b = m * modules_per_chunk * padded_total_output_class_count
    offset_t = m * modules_per_chunk

    # adam_update dispatches (KernelDispatchNode)
    adam_w[m] = adam_update(
        final_grad=norm_w[m].output,
        parameters=model_weights_module,       # full-model buffer
        m1=m1_module_weights,                  # full-model buffer
        m2=m2_module_weights,                  # full-model buffer
        parameter_offset=offset_w,
        parameter_count=param_count_w_chunk,
        total_parameter_count=total_param_count_w,
        ...
    )
    adam_b[m] = adam_update(
        final_grad=norm_b[m].output,
        parameters=model_biases_module,
        m1=m1_module_biases,
        m2=m2_module_biases,
        parameter_offset=offset_b,
        parameter_count=param_count_b_chunk,
        total_parameter_count=total_param_count_b,
        ...
    )
    adam_t[m] = adam_update(
        final_grad=norm_t[m].output,
        parameters=model_temps,
        m1=m1_temps,
        m2=m2_temps,
        parameter_offset=offset_t,
        parameter_count=param_count_t_chunk,
        total_parameter_count=total_param_count_t,
        ...
    )

    clamp_t[m] = clamp_temperatures(
        temps=model_temps,
        parameter_offset=offset_t,
        parameter_count=param_count_t_chunk,
        total_parameter_count=total_param_count_t,
        ...
    )

# Shared-layer dispatches (no module dimension — offset is zero)
adam_sw = adam_update(
    final_grad=norm_sw.output,
    parameters=model_weights_shared,
    m1=m1_shared_weights,
    m2=m2_shared_weights,
    parameter_offset=0,
    parameter_count=total_shared_weight_count,
    total_parameter_count=total_shared_weight_count,
    ...
)
adam_sb = adam_update(
    final_grad=norm_sb.output,
    parameters=model_biases_shared,
    m1=m1_shared_biases,
    m2=m2_shared_biases,
    parameter_offset=0,
    parameter_count=total_shared_bias_count,
    total_parameter_count=total_shared_bias_count,
    ...
)
```

The shared-layer dispatches require no chunking, but their call sites must pass the amended signature with `parameter_offset=0` and `total_parameter_count = parameter_count`.

#### 2.3 Dependency Graph

```
Phase I (all tiles complete)
    │
    ├── Module chunk 0:  reduce_w[0] → norm_w[0] → adam_w[0] ──┐
    │                    reduce_b[0] → norm_b[0] → adam_b[0] ──┤
    │                    reduce_t[0] → norm_t[0] → adam_t[0] → clamp_t[0] ──┐
    │                                                                        │
    ├── Module chunk 1:  reduce_w[1] → norm_w[1] → adam_w[1] ──┤           │
    │                    reduce_b[1] → norm_b[1] → adam_b[1] ──┤           │
    │                    reduce_t[1] → norm_t[1] → adam_t[1] → clamp_t[1] ──┤
    │                                                                        │
    ├── ...                                                                  │
    │                                                                        │
    ├── Shared path:     reduce_sw → norm_sw → adam_sw ─────────┤           │
    │                    reduce_sb → norm_sb → adam_sb ─────────┤           │
    │                                                            │           │
    └────────────────────────────────────────────────────────────┴───────────┘
                                                                     │
                                                              final_batch_event
```

All `num_module_chunks` sub-graphs plus the shared-layer sub-graph are independent and may execute in parallel. The backend is free to exploit this concurrency per CONCEPT.md §6 ("Nodes with no dependency relationship may execute in parallel").

#### 2.4 Amended Phase V Mermaid Fragment

The following replaces the Phase V subgraph in CONCEPT.md's Architectural Blueprint. The former `B22` barrier node is removed; dependency edges flow directly from each `FINAL_Grad_*[m]` to its consuming `adam_update`:

```mermaid
subgraph PhaseV["<b>Phase V:</b> Parameter Update (Per-Module-Chunk)"]
    style PhaseV phase_box

    subgraph ModChunkLoop["Per Module Chunk 'm'"]
        style ModChunkLoop domain_instance

        K24_ModW_m["(24) adam_update<br/>(ModW[m])"]:::kernel
        K24_ModB_m["(24) adam_update<br/>(ModB[m])"]:::kernel
        K24_Temps_m["(24) adam_update<br/>(Temps[m])"]:::kernel

        FINAL_Grad_Mod_m["Final Avg Grad<br/>(Mod[m])"]:::final_data --> K24_ModW_m
        FINAL_Grad_Mod_m --> K24_ModB_m
        FINAL_Grad_Temps_m["Final Avg Grad<br/>(Temps[m])"]:::final_data --> K24_Temps_m
        P_Adam --> K24_ModW_m & K24_ModB_m & K24_Temps_m

        K24_Temps_m --> K25_m["(25) clamp_temperatures<br/>(Temps[m])"]:::kernel
    end

    K24_Shared["(24) adam_update (Shared W)"]:::kernel
    K24_SharedB["(24) adam_update (Shared B)"]:::kernel
    FINAL_Grad_S --> K24_Shared & K24_SharedB
    P_Adam --> K24_Shared & K24_SharedB
end

K24_ModW_m & K24_ModB_m & K25_m & K24_Shared & K24_SharedB --> EV_Final
```

### 3. Kernel Contract Amendments

#### 3.1 `adam_update` (Node 24): Add `parameter_offset` and `total_parameter_count`

Per-module-chunk dispatch requires indexing into monolithic model-state and optimizer-state buffers at a chunk-specific offset. The scalar-offset pattern is consistent with existing kernel contracts (`forward_pass` has `batch_chunk_offset`/`total_batch_count`, `render_logits_chunk` has `module_chunk_offset`/`total_modules_count`). The addition of `total_parameter_count` preserves the defense-in-depth pattern established throughout the kernel set, where slice-indexed kernels receive both the offset/count pair and the total extent, enabling device-side bounds assertions per CONTRACT.md Article 1.4.b. The kernel remains stateless and "dumb" — the host computes and provides the offset.

**Amendment to `adam_update` contract:**

Add the following parameters, placed before `src_scalar_NATURAL_parameter_count`:

```c
    /**
     * @param src_scalar_NATURAL_parameter_offset The element offset into the parameter,
     *        m1, and m2 buffers at which this dispatch's slice begins.
     *        - Calculability Proof: [Host-side calculation: module_chunk_index ×
     *          elements_per_module_chunk for the parameter group being processed.
     *          For shared-layer dispatches: 0.]
     *        - Validation Preconditions: [1] Must satisfy (parameter_offset +
     *          parameter_count) <= total_parameter_count.
     *          [2] The final_grad buffer is zero-indexed and contains exactly
     *          parameter_count elements — no offset is applied to it.
     */
    uint src_scalar_NATURAL_parameter_offset,
```

Add the following parameter after `src_scalar_NATURAL_parameter_count`:

```c
    /**
     * @param src_scalar_NATURAL_total_parameter_count The total element count of the
     *        parameter, m1, and m2 buffers. Serves as a device-side bounds-check
     *        parameter per Article 1.4.b (defense-in-depth).
     *        - Calculability Proof: [Host-side: total model parameter count for this
     *          parameter group]
     *        - Validation Preconditions: [1] (parameter_offset + parameter_count) <=
     *          total_parameter_count. [2] Must equal the total allocation size (in
     *          elements) of the update_buffer_GLOBAL_parameters, m1, and m2 buffers.
     */
    uint src_scalar_NATURAL_total_parameter_count,
```

**Amended behavioral invariant (access pattern):**

```
parameters[parameter_offset + i]  for i ∈ [0, parameter_count)
m1[parameter_offset + i]          for i ∈ [0, parameter_count)
m2[parameter_offset + i]          for i ∈ [0, parameter_count)
final_grad[i]                     for i ∈ [0, parameter_count)  (zero-indexed)
```

**State-role buffer commentary amendment.** The `Tensor Shape` and `Calculability Proof` for `update_buffer_GLOBAL_parameters`, `update_buffer_GLOBAL_m1`, and `update_buffer_GLOBAL_m2` are amended:

```
- Tensor Shape: (src_scalar_NATURAL_total_parameter_count)
- Calculability Proof: [src_scalar_NATURAL_total_parameter_count]
- Validation Preconditions: [1] Host shall allocate exactly
  [src_scalar_NATURAL_total_parameter_count * sizeof(STATE_TYPE)] bytes.
  [2] (src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <=
  src_scalar_NATURAL_total_parameter_count.
  [3] The physical memory layout across the accessed slice must be
  identical to `final_grad`, `m1`, and `m2` buffers.
```

The `final_grad` buffer remains `(src_scalar_NATURAL_parameter_count)` — it is a per-dispatch intermediate, not a slice of a monolithic buffer.

**When `num_module_chunks == 1`:** `parameter_offset = 0`, `parameter_count = total_parameter_count`. The kernel's access pattern reduces to `buf[0 + i] = buf[i]`, identical to pre-amendment behavior. ✓

#### 3.2 `clamp_temperatures` (Node 25): Add `parameter_offset` and `total_parameter_count`

The same offset pattern applies to `clamp_temperatures`, which modifies a slice of the monolithic temperatures buffer after its corresponding `adam_update` completes. Per-module-chunk dispatch of `clamp_temperatures` enables fine-grained dependency edges (each `clamp_t[m]` depends only on `adam_t[m]`, not on all temperature updates).

**Amendment to `clamp_temperatures` contract:**

Rename `src_scalar_NATURAL_total_modules_count` to `src_scalar_NATURAL_parameter_count` for consistency with the general slice-indexed dispatch pattern. Add two new parameters:

```c
    /**
     * @param src_scalar_NATURAL_parameter_offset The element offset into the
     *        temperatures buffer at which this dispatch's slice begins.
     *        - Calculability Proof: [Host-side calculation: module_chunk_index ×
     *          modules_per_chunk. For single-chunk dispatch: 0.]
     *        - Validation Preconditions: (parameter_offset + parameter_count) <=
     *          total_parameter_count.
     */
    uint src_scalar_NATURAL_parameter_offset,

    /**
     * @param src_scalar_NATURAL_parameter_count The number of temperature elements
     *        to clamp in this dispatch.
     *        - Calculability Proof: [Host-side: modules_per_chunk for per-chunk dispatch,
     *          or total_modules_count for single dispatch]
     *        - Validation Preconditions: Must be >= 1.
     */
    uint src_scalar_NATURAL_parameter_count,

    /**
     * @param src_scalar_NATURAL_total_parameter_count The total element count of
     *        the temperatures buffer. Serves as a device-side bounds-check parameter
     *        per Article 1.4.b (defense-in-depth).
     *        - Calculability Proof: [Host-side: total_modules_count]
     *        - Validation Preconditions: [1] (parameter_offset + parameter_count) <=
     *          total_parameter_count. [2] Must equal the total allocation size (in
     *          elements) of update_buffer_GLOBAL_temps.
     */
    uint src_scalar_NATURAL_total_parameter_count,
```

**Buffer commentary amendment.** The `Tensor Shape` for `update_buffer_GLOBAL_temps` is amended:

```
- Tensor Shape: (src_scalar_NATURAL_total_parameter_count)
- Calculability Proof: [src_scalar_NATURAL_total_parameter_count]
- Validation Preconditions: [1] Host shall allocate exactly
  [src_scalar_NATURAL_total_parameter_count * sizeof(STATE_TYPE)] bytes.
  [2] (src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <=
  src_scalar_NATURAL_total_parameter_count.
```

**Amended behavioral invariant (access pattern):**

```
temps[parameter_offset + i]  for i ∈ [0, parameter_count)
```

**Alternative considered (single post-all-updates dispatch).** `clamp_temperatures` could remain a single dispatch gated on all temperature-related `adam_update` completions. This is simpler but re-introduces a synchronization bottleneck that defeats the per-module-chunk parallelism. The per-chunk dispatch is preferred for architectural consistency.

#### 3.3 Full Amended Signatures

```c
__kernel void adam_update(
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_final_grad,
    __global STATE_TYPE         *update_buffer_GLOBAL_parameters,
    __global STATE_TYPE         *update_buffer_GLOBAL_m1,
    __global STATE_TYPE         *update_buffer_GLOBAL_m2,
    COMPUTE_TYPE src_scalar_REAL_learning_rate,
    COMPUTE_TYPE src_scalar_REAL_beta1_pow_t,
    COMPUTE_TYPE src_scalar_REAL_beta2_pow_t,
    COMPUTE_TYPE src_scalar_REAL_beta1,
    COMPUTE_TYPE src_scalar_REAL_beta2,
    COMPUTE_TYPE src_scalar_REAL_epsilon,
    uint         src_scalar_NATURAL_parameter_offset,         // NEW (ADR-030)
    uint         src_scalar_NATURAL_parameter_count,
    uint         src_scalar_NATURAL_total_parameter_count);   // NEW (ADR-030)
```

```c
__kernel void clamp_temperatures(
    __global STATE_TYPE *update_buffer_GLOBAL_temps,
    COMPUTE_TYPE src_scalar_REAL_min_value,
    COMPUTE_TYPE src_scalar_REAL_max_value,
    uint         src_scalar_NATURAL_parameter_offset,         // NEW (ADR-030)
    uint         src_scalar_NATURAL_parameter_count,          // RENAMED from total_modules_count
    uint         src_scalar_NATURAL_total_parameter_count);   // NEW (ADR-030)
```

### 4. Node 22 Barrier Retirement

#### 4.1 Current State

CONCEPT.md defines Node 22 as a **Batch Synchronization Barrier** — a single `BarrierNode` that gates all `adam_update` dispatches. ADR-002 specifies this as the canonical `BarrierNode` instance `"batch_sync_barrier"`. With the original single-reduction-tree design, this monolithic barrier correctly ensures all normalized gradients for all parameter groups are available before any update fires.

#### 4.2 Amended State

With per-module-chunk sub-graphs, Node 22's monolithic barrier is both unnecessary and constraining:

- **Unnecessary:** Each `adam_update[m]` depends only on its own `normalize[m]` output. There is no cross-module-chunk data dependency requiring global synchronization.
- **Constraining:** A monolithic barrier serializes all module-chunk update chains, preventing the backend from exploiting the per-module-chunk parallelism.

Node 22 is **retired as a discrete barrier node**. It is removed from the plan's node set and from the Mermaid diagram.

The synchronization property that Node 22 formerly expressed — "parameter updates occur only after their gradient dependencies are fully resolved" — is now a **structural property of the plan DAG**, satisfied by the direct dependency edges from each `normalize_gradients[m]` output to its consuming `adam_update[m]`. No explicit `BarrierNode` is needed.

The only global synchronization point that remains is `final_batch_event`, which collects all terminal nodes (all `adam_update` completions, all `clamp_temperatures` completions) as a `RetrievalNode`. `final_batch_event` is the sole plan-wide synchronization point and serves as the observable successor of the former Node 22.

#### 4.3 CONCEPT.md Amendment

The "Batch Synchronization Point" definition in the Fundamental Synchronization Barriers section is amended:

> **Batch Synchronization Point:** A synchronization property ensuring that parameter updates occur only after their gradient dependencies are fully resolved. ~~This manifests as a single monolithic `BarrierNode` (Node 22) that gates all `adam_update` dispatches.~~ When the module dimension is decomposed into chunks, this property is satisfied by per-sub-graph dependency edges rather than a single monolithic barrier: each `adam_update` dispatch's dependency on its own `normalize_gradients` output expresses the synchronization contract directly in the DAG structure. When `num_module_chunks == 1`, the property is trivially satisfied — a single dependency edge from each normalized gradient to its corresponding `adam_update` is equivalent to the former barrier.
>
> The `final_batch_event` `RetrievalNode` remains the sole plan-wide synchronization point, collecting all terminal update nodes. It is the observable successor of the former Node 22 barrier.
>
> - **Canonical Example:** The per-module-chunk dependency edges from `normalize_gradients[m]` to `adam_update[m]` for each parameter group and module chunk index `m`, converging at `final_batch_event`.

### 5. Buffer Sizing Corrections

#### 5.1 Model-State and Optimizer-State Buffers

The model-state buffers (weights, biases, temperatures) are already full-model-sized. The optimizer-state buffers (`m1`, `m2` for each parameter group) must match:

| Buffer              | Incorrect Size              | Correct Size                  | Lifecycle Role |
| :------------------ | :-------------------------- | :---------------------------- | :------------- |
| `m1_module_weights` | `modules_per_chunk × H × C` | `num_modules × H_pad × C_pad` | `MODEL_STATE`  |
| `m2_module_weights` | `modules_per_chunk × H × C` | `num_modules × H_pad × C_pad` | `MODEL_STATE`  |
| `m1_module_biases`  | `modules_per_chunk × C`     | `num_modules × C_pad`         | `MODEL_STATE`  |
| `m2_module_biases`  | `modules_per_chunk × C`     | `num_modules × C_pad`         | `MODEL_STATE`  |
| `m1_temps`          | `modules_per_chunk`         | `num_modules`                 | `MODEL_STATE`  |
| `m2_temps`          | `modules_per_chunk`         | `num_modules`                 | `MODEL_STATE`  |

These buffers persist across batches (`MODEL_STATE` lifecycle role per ADR-009) and are indexed by `parameter_offset` in each `adam_update` dispatch.

#### 5.2 Intermediate Gradient Buffers

The reduction tree output buffers and normalized gradient buffers are **per-module-chunk intermediates** (`BATCH_INTERMEDIATE` lifecycle role). They are sized to the chunk:

| Buffer                 | Size                                | Lifecycle Role       |
| :--------------------- | :---------------------------------- | :------------------- |
| `summed_grad_modw[m]`  | `modules_per_chunk × H_pad × C_pad` | `BATCH_INTERMEDIATE` |
| `summed_grad_modb[m]`  | `modules_per_chunk × C_pad`         | `BATCH_INTERMEDIATE` |
| `summed_grad_temps[m]` | `modules_per_chunk`                 | `BATCH_INTERMEDIATE` |
| `final_grad_modw[m]`   | Same as summed                      | `BATCH_INTERMEDIATE` |
| `final_grad_modb[m]`   | Same as summed                      | `BATCH_INTERMEDIATE` |
| `final_grad_temps[m]`  | Same as summed                      | `BATCH_INTERMEDIATE` |

These are allocated per-module-chunk and freed after their consuming `adam_update` dispatch completes. When `num_module_chunks == 1`, there is one set of intermediates sized to the full model — identical to current behavior.

#### 5.3 Checkpoint Migration

Existing checkpoints serialized with `num_module_chunks > 1` may contain per-chunk-sized optimizer state buffers. This ADR does not prescribe a migration path because:

1. The defect means any checkpoint produced with `num_module_chunks > 1` contains corrupted optimizer state (the summed gradients that populated `m1`/`m2` were incorrect). Migrating the buffer layout would preserve the corruption.
2. The correct recovery is to re-initialize optimizer state (`m1 = 0`, `m2 = 0`) and resume training. This is standard practice when optimizer state is invalidated.

Implementations should detect the size mismatch at checkpoint load time and raise a clear diagnostic indicating that optimizer state re-initialization is required. Model parameter buffers (weights, biases, temperatures) are unaffected — they are already full-model-sized and were updated with corrupted gradients, but their values are not structurally incompatible with the corrected system.

### 6. Formal Invariant

The following invariant is added to CONCEPT.md §2 (The Recursive, Tiered Aggregation Engine), immediately following the Indirection Contract description:

> **Module-Chunk Isolation Invariant (ADR-030).** For parameter-gradient reduction trees (Nodes 15 and associated optimizer paths), the offset lists constructed by the Host Orchestrator SHALL contain only tile offsets belonging to a **single module chunk**. Cross-module-chunk reduction is architecturally prohibited because it conflates gradients for disjoint parameter sets. This invariant is structurally enforced by the `ModuleChunkGather` primitive, which produces offset lists scoped to a single module chunk's tiles. When `num_module_chunks == 1`, the invariant is trivially satisfied — all tiles belong to the single module chunk.
>
> `TiledGather` remains the correct primitive for non-parameter-gradient reductions (Node 14), where the output layout encodes the correct final position per tile and cross-module-chunk summation is mathematically valid. The hidden-gradient reduction (Node 16) is also exempt: it operates on a permuted SoA buffer that has already collapsed the module dimension via Node 13's class-chunk summation.

### 7. Validation

#### 7.1 Existing Scenario Coverage

The following existing validation scenarios now exercise module-chunk isolation:

- **The Hydra (Massive `num_heads`):** With `num_heads` large enough to force `num_module_chunks > 1`, this scenario validates that per-module-chunk reduction trees produce correct summed gradients, that `adam_update` correctly indexes into monolithic state buffers via `parameter_offset`, and that convergence matches a single-module-chunk baseline.

- **The Colossus (Holistic Stress Test):** Already tests "the orchestrator's robustness by forcing it to compose a single, valid execution plan that correctly handles [all] reduction paths." Now additionally validates the per-module-chunk sub-graph parallelism.

#### 7.2 New Scenario

> **Scenario: The Hydra's Spine (Module-Chunk Isolation Validation)**
>
> - **Description:** A training task with `num_modules` chosen to force `num_module_chunks >= 2` at the target memory budget. Two runs are executed: (A) the orchestrator's natural chunking (`num_module_chunks = M`), and (B) a reference run with artificially increased memory budget forcing `num_module_chunks = 1`.
> - **Validation Focus:** Confirms that per-parameter weight updates are identical (within floating-point tolerance) between runs (A) and (B). This validates that the `ModuleChunkGather` primitive correctly isolates tile subsets, that the per-module-chunk reduction trees produce the same summed gradients as the monolithic tree, and that `adam_update`'s `parameter_offset` indexing is correct.
> - **Key Insight:** Proves that module-dimension decomposition is a **transparent scheduling decision** — it changes the plan's structure but not its mathematical output. The user observes identical training dynamics regardless of `num_module_chunks`, confirming that the Module-Chunk Isolation Invariant is correctly enforced.

---

## Impact Summary

| Component                     | Change                                                                                               | Breaking?                         |
| :---------------------------- | :--------------------------------------------------------------------------------------------------- | :-------------------------------- |
| `ModuleChunkGather`           | New gather primitive                                                                                 | No (additive)                     |
| `ModuleBufferKind`            | New enumeration                                                                                      | No (additive)                     |
| `TiledGather`                 | Deprecated for parameter-gradient reductions; remains valid for Node 14 diagnostic reductions        | Scope clarification               |
| Plan builder (Phase II)       | Per-module-chunk reduction loop                                                                      | Internal restructuring            |
| Plan builder (Phase V)        | Per-module-chunk update loop                                                                         | Internal restructuring            |
| `adam_update` contract        | Add `parameter_offset` and `total_parameter_count` scalars                                           | **Yes** (kernel interface change) |
| `clamp_temperatures` contract | Add `parameter_offset` and `total_parameter_count`; rename `total_modules_count` → `parameter_count` | **Yes** (kernel interface change) |
| Node 22                       | Retired as discrete barrier; synchronization property expressed by DAG dependency edges              | Architectural amendment           |
| `final_batch_event`           | Now sole plan-wide synchronization point; observable successor of former Node 22                     | Clarification                     |
| Optimizer state buffers       | Sized to full model                                                                                  | Allocation change                 |
| Existing checkpoints          | `num_module_chunks > 1` checkpoints require optimizer state re-initialization                        | Migration note                    |
| CONCEPT.md §2                 | Module-Chunk Isolation Invariant added                                                               | Documentary                       |
| CONCEPT.md §4 (barriers)      | Batch Synchronization Point redefined; Node 22 retired                                               | Documentary                       |
| Mermaid diagram (Phase V)     | Per-module-chunk update sub-graphs; Node 22 removed                                                  | Documentary                       |
| `kernels.cl.h`                | Two kernel signatures amended                                                                        | **Yes** (all backends)            |

All three backends (OpenCL, Vulkan, CPU) require implementation updates for the two amended kernel signatures. The `parameter_offset = 0`, `parameter_count = total_parameter_count` degeneration ensures backward compatibility with existing test fixtures when `num_module_chunks == 1`.
