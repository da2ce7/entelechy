# Recommendation: Module-Chunk Isolation Invariant

## ADR-028: Module-Chunk Isolation in Parameter-Gradient Reduction Trees

**Status:** Proposed
**Supersedes:** None
**Amends:** CONCEPT.md §4 (Recursive Aggregation Engine), §11 (Host Orchestrator), Kernel Contract for Node 24 (`adam_update`), Kernel Contract for Node 25 (`clamp_temperatures`), Node 22 (Batch Synchronization Barrier)

---

### 1. Context & Problem Statement

#### 1.1 The Defect

When the Host Orchestrator decomposes the module dimension into `num_module_chunks > 1` chunks, the system's `grid_mod_cls` placement strategy assigns each tile a `flat_tile_index` that encodes both a `module_chunk_index` and a `class_chunk_index`:

```
flat_tile_index = module_chunk_index * num_class_chunks + class_chunk_index
```

Tiles from the **same** module chunk but different class chunks write to non-overlapping class positions within the same `modules_per_chunk`-wide parameter slice. Their element-wise sum correctly fills the class dimension — this is the documented behavior of the `ZERO_REQUIRED` initialization pattern on Node 8's output buffers.

However, tiles from **different** module chunks write gradient values for **disjoint physical parameters** at the **same positional indices** within their respective `elements_per_partial`-wide regions. If the Indirection Contract's offset list for a reduction tree mixes tiles from different module chunks into a single reduction node, the element-wise summation **conflates gradients for different parameters**, producing silent, catastrophic corruption of all module-specific parameter updates.

#### 1.2 Scope of Impact

The defect affects only the parameter-gradient reduction trees:

| Reduction Path | Affected? | Reason |
|:---|:---|:---|
| **Node 15 (Grad_ModW)** | **Yes** | Tiles index by `(module_chunk, class_chunk)` |
| **Node 15 (Grad_ModB)** | **Yes** | Same indexing |
| **Node 15 (Grad_Temps)** | **Yes** | Same indexing |
| Node 14 (Probs, Loss) | No | Diagnostic reduction; output layout is `(module, batch, class)` — tile placement encodes the correct final position |
| Node 16 (Grad_H) | No | Permuted to contiguous SoA by Node 13; reduces over *all* modules simultaneously via a dedicated kernel |
| Node 20 (Grad_SW, Grad_SB) | No | Shared-layer gradients have no module dimension; use `linear_batch` placement |

The defect is **latent when `num_module_chunks == 1`** (the common case for small model configurations), because all tiles belong to the single module chunk and no cross-contamination is possible. It manifests only under memory pressure that forces module-dimension decomposition.

#### 1.3 Root Cause

The existing `TiledGather` primitive returns offsets for **all** tiles across **all** module chunks, treating the flat tile index space as a homogeneous collection. The Recursive Aggregation Engine (CONCEPT.md §2) then constructs a reduction tree over this undifferentiated set, violating the implicit invariant that tiles being summed must correspond to the same physical parameters.

No explicit invariant in CONCEPT.md, CONTRACT.md, or the kernel headers prohibits this cross-module-chunk mixing. The constraint is implicit in the mathematical structure of the gradient computation but is not encoded in the system's formal contracts.

---

### 2. Decision

#### 2.1 Principle

Module-chunk isolation is formalized as a **first-class structural property** of the tiling decomposition, not an ad-hoc filter on offset lists. The fix introduces a new gather primitive, amends two kernel contracts, decomposes a synchronization barrier, and corrects buffer sizing — all within the existing plan node vocabulary.

Per CONCEPT.md §1 (Architectural Elegance Feedback):

1. **Suspend** — no ad-hoc offset-list filtering or conditional reduction logic
2. **Formalize** — `ModuleChunkGather` is the documented primitive expressing module-chunk isolation
3. **Reify** — implemented through the gather contract, plan builder loop, and amended kernel contracts

#### 2.2 Solution Components

The fix consists of five coordinated changes:

1. **New Gather Primitive:** `ModuleChunkGather`
2. **Plan Builder Restructuring:** Per-module-chunk sub-graphs in Phases II and V
3. **Kernel Contract Amendment:** `adam_update` (Node 24) and `clamp_temperatures` (Node 25)
4. **Barrier Decomposition:** Node 22 dissolves into per-sub-graph dependency edges
5. **Buffer Sizing Correction:** Optimizer state buffers sized to full model

---

### 3. New Gather Primitive: `ModuleChunkGather`

#### 3.1 Definition

```python
@dataclass(frozen=True)
class ModuleChunkGather(GatherPrimitive):
    """Gathers only the tiles belonging to a single module chunk.

    The TilingScheme orders tiles as:
        flat_tile_index = module_chunk_index * num_class_chunks + class_chunk_index

    Each module chunk's tiles are therefore a contiguous subsequence of length
    num_class_chunks, starting at offset (module_chunk_index * num_class_chunks).

    This primitive encodes the Module-Chunk Isolation Invariant (ADR-028) as a
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
            (np.arange(self.num_partials, dtype=np.uint32) + base)
            * self.elements_per_partial
        )
```

#### 3.2 `ModuleBufferKind` Enumeration

```python
class ModuleBufferKind(enum.Enum):
    """Identifies the parameter-gradient buffer for elements_per_tile dispatch."""
    WEIGHTS = "weights"   # modules_per_chunk × padded_hidden_count × padded_total_output_class_count
    BIASES  = "biases"    # modules_per_chunk × padded_total_output_class_count
    TEMPS   = "temps"     # modules_per_chunk
```

`TilingScheme.elements_per_tile(kind)` returns the product of the appropriate dimensions. This calculation belongs to `TilingScheme` because it already holds `modules_per_chunk`, `padded_hidden_count`, and `padded_total_output_class_count`.

#### 3.3 Offset Arithmetic Verification

For a configuration with `num_module_chunks=3`, `num_class_chunks=4`:

```
Total tiles: 12
Module chunk 0: tiles [0, 1, 2, 3]     → offsets [0, 1, 2, 3] × epp
Module chunk 1: tiles [4, 5, 6, 7]     → offsets [4, 5, 6, 7] × epp
Module chunk 2: tiles [8, 9, 10, 11]   → offsets [8, 9, 10, 11] × epp
```

Each module chunk's reduction tree receives exactly `num_class_chunks` tiles. Those tiles wrote non-overlapping class positions for the same `modules_per_chunk` parameters (Node 8's `ZERO_REQUIRED` initialization fills gaps). Element-wise summation correctly reconstructs the full gradient for the module chunk's parameter slice. ✓

#### 3.4 Degeneration

When `num_module_chunks == 1`:
- `module_chunk_index = 0`
- `num_partials = num_class_chunks = total_tile_count`
- `get_offsets()` returns `[0, 1, ..., total_tile_count-1] × epp`
- This equals the full set of tile offsets formerly returned by `TiledGather`
- The plan builder loop (§4) executes once
- All downstream behavior is identical to the pre-fix state ✓

---

### 4. Plan Builder Restructuring

#### 4.1 Phase II: Per-Module-Chunk Gradient Reduction

The current Phase II constructs a single reduction tree per parameter group over all tiles. The amended Phase II constructs `num_module_chunks` independent sub-graphs:

```
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

#### 4.2 Phase V: Per-Module-Chunk Parameter Update

```
for m in range(num_module_chunks):
    offset_w = m * modules_per_chunk * padded_hidden_count * padded_total_output_class_count
    offset_b = m * modules_per_chunk * padded_total_output_class_count
    offset_t = m * modules_per_chunk

    # adam_update dispatches (KernelDispatchNode)
    adam_w[m] = adam_update(
        final_grad=norm_w[m].output,
        parameters=model_weights_module,  # full-model buffer
        m1=m1_module_weights,             # full-model buffer
        m2=m2_module_weights,             # full-model buffer
        parameter_offset=offset_w,
        parameter_count=param_count_w_chunk,
        ...
    )
    adam_b[m] = adam_update(
        final_grad=norm_b[m].output,
        parameters=model_biases_module,
        m1=m1_module_biases,
        m2=m2_module_biases,
        parameter_offset=offset_b,
        parameter_count=param_count_b_chunk,
        ...
    )
    adam_t[m] = adam_update(
        final_grad=norm_t[m].output,
        parameters=model_temps,
        m1=m1_temps,
        m2=m2_temps,
        parameter_offset=offset_t,
        parameter_count=param_count_t_chunk,
        ...
    )
```

The shared-layer `adam_update` dispatches (`Grad_SW`, `Grad_SB`) remain unchanged — they have no module dimension.

#### 4.3 Dependency Graph

The amended Phase II–V dependency structure:

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

---

### 5. Kernel Contract Amendments

#### 5.1 `adam_update` (Node 24): Add `parameter_offset`

**Rationale.** Per-module-chunk dispatch requires indexing into monolithic model-state and optimizer-state buffers at a chunk-specific offset. The scalar-offset pattern is consistent with existing kernel contracts (`forward_pass` has `batch_chunk_offset`, `render_logits_chunk` has `module_chunk_offset`/`class_chunk_offset`). The kernel remains stateless and "dumb" — the host computes and provides the offset.

**Amendment to `adam_update` contract:**

Add the following parameter before `src_scalar_NATURAL_parameter_count`:

```c
    /**
     * @param src_scalar_NATURAL_parameter_offset The element offset into the parameter,
     *        m1, and m2 buffers at which this dispatch's slice begins.
     *        - Calculability Proof: [Host-side calculation: module_chunk_index ×
     *          elements_per_module_chunk for the parameter group being processed]
     *        - Validation Preconditions: [1] Must satisfy (parameter_offset +
     *          parameter_count) <= total element count of the parameter buffer.
     *          [2] The final_grad buffer is zero-indexed and contains exactly
     *          parameter_count elements — no offset is applied to it.
     */
    uint src_scalar_NATURAL_parameter_offset,
```

**Amended behavioral invariant (access pattern):**

```
parameters[parameter_offset + i]  for i ∈ [0, parameter_count)
m1[parameter_offset + i]          for i ∈ [0, parameter_count)
m2[parameter_offset + i]          for i ∈ [0, parameter_count)
final_grad[i]                     for i ∈ [0, parameter_count)  (zero-indexed)
```

**State-role buffer commentary amendment.** The `Tensor Shape` for `update_buffer_GLOBAL_parameters`, `update_buffer_GLOBAL_m1`, and `update_buffer_GLOBAL_m2` changes from `(src_scalar_NATURAL_parameter_count)` to:

```
- Tensor Shape: (total model parameter count for this parameter group)
- Validation Preconditions: [1] Host shall allocate the full model-sized
  buffer. [2] (src_scalar_NATURAL_parameter_offset +
  src_scalar_NATURAL_parameter_count) <= total element count.
```

The `final_grad` buffer remains `(src_scalar_NATURAL_parameter_count)` — it is a per-dispatch intermediate, not a slice of a monolithic buffer.

**When `num_module_chunks == 1`:** `parameter_offset = 0`, `parameter_count = full model size`. The kernel's access pattern reduces to `buf[0 + i] = buf[i]`, identical to pre-amendment behavior. ✓

#### 5.2 `clamp_temperatures` (Node 25): Add `parameter_offset`

**Rationale.** The same offset pattern applies to `clamp_temperatures`, which modifies a slice of the monolithic temperatures buffer after its corresponding `adam_update` completes. Per-module-chunk dispatch of `clamp_temperatures` enables fine-grained dependency edges (each `clamp_t[m]` depends only on `adam_t[m]`, not on all temperature updates).

**Amendment to `clamp_temperatures` contract:**

Add the following parameter:

```c
    /**
     * @param src_scalar_NATURAL_parameter_offset The element offset into the
     *        temperatures buffer at which this dispatch's slice begins.
     *        - Calculability Proof: [Host-side calculation: module_chunk_index ×
     *          modules_per_chunk]
     *        - Validation Preconditions: (parameter_offset + parameter_count) <=
     *          total_modules_count, where parameter_count is the number of
     *          elements to clamp in this dispatch.
     */
    uint src_scalar_NATURAL_parameter_offset,
```

Rename `src_scalar_NATURAL_total_modules_count` to `src_scalar_NATURAL_parameter_count` for consistency with the general pattern. The buffer commentary for `update_buffer_GLOBAL_temps` changes to reflect full-model allocation with slice access.

**Alternative (single post-all-updates dispatch).** `clamp_temperatures` could remain a single dispatch gated on all temperature-related `adam_update` completions. This is simpler but re-introduces a synchronization bottleneck that defeats the per-module-chunk parallelism. The per-chunk dispatch is preferred for architectural consistency.

#### 5.3 Full Amended `adam_update` Signature

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
    uint         src_scalar_NATURAL_parameter_offset,  // NEW
    uint         src_scalar_NATURAL_parameter_count);
```

#### 5.4 Full Amended `clamp_temperatures` Signature

```c
__kernel void clamp_temperatures(
    __global STATE_TYPE *update_buffer_GLOBAL_temps,
    COMPUTE_TYPE src_scalar_REAL_min_value,
    COMPUTE_TYPE src_scalar_REAL_max_value,
    uint         src_scalar_NATURAL_parameter_offset,  // NEW
    uint         src_scalar_NATURAL_parameter_count);  // RENAMED from total_modules_count
```

---

### 6. Node 22 Barrier Decomposition

#### 6.1 Current State

CONCEPT.md defines Node 22 as a **Batch Synchronization Barrier** — a single `BarrierNode` that gates all `adam_update` dispatches. With the original single-reduction-tree design, this monolithic barrier correctly ensures all normalized gradients for all parameter groups are available before any update fires.

#### 6.2 Amended State

With per-module-chunk sub-graphs, Node 22's monolithic barrier is both unnecessary and constraining:

- **Unnecessary:** Each `adam_update[m]` depends only on its own `normalize[m]` output. There is no cross-module-chunk data dependency requiring global synchronization.
- **Constraining:** A monolithic barrier serializes all module-chunk update chains, preventing the backend from exploiting the per-module-chunk parallelism.

**Decision:** Node 22 is dissolved into **implicit dependency edges** expressed by the plan's DAG structure. Each `adam_update` dispatch depends on its direct predecessor (`normalize_gradients` for its parameter group and module chunk). No explicit `BarrierNode` is needed.

The only global synchronization point that remains is `final_batch_event`, which collects all terminal nodes (all `adam_update` completions, all `clamp_temperatures` completions) as a `RetrievalNode`. This is already the existing design — `final_batch_event` is the Plan's terminal fan-in.

#### 6.3 CONCEPT.md Amendment

The "Batch Synchronization Point" definition in the Fundamental Synchronization Barriers section is amended:

> **Batch Synchronization Point:** ~~A barrier that resolves a data dependency between multiple independent learning items that constitute a single logical batch.~~ **A structural property of the plan DAG** expressing that parameter updates occur only after their gradient dependencies are fully resolved. When the module dimension is decomposed into chunks, this property is satisfied by per-sub-graph dependency edges rather than a single monolithic barrier. The `final_batch_event` `RetrievalNode` remains the sole plan-wide synchronization point, collecting all terminal update nodes.

The Mermaid diagram's Phase V is amended to show per-module-chunk `adam_update` dispatches with independent dependency chains, converging at `final_batch_event`.

---

### 7. Buffer Sizing Corrections

#### 7.1 Model-State and Optimizer-State Buffers

The model-state buffers (weights, biases, temperatures) are already full-model-sized. The optimizer-state buffers (`m1`, `m2` for each parameter group) must match:

| Buffer | Incorrect Size | Correct Size | Lifecycle Role |
|:---|:---|:---|:---|
| `m1_module_weights` | `modules_per_chunk × H × C` | `num_modules × H_pad × C_pad` | `MODEL_STATE` |
| `m2_module_weights` | `modules_per_chunk × H × C` | `num_modules × H_pad × C_pad` | `MODEL_STATE` |
| `m1_module_biases` | `modules_per_chunk × C` | `num_modules × C_pad` | `MODEL_STATE` |
| `m2_module_biases` | `modules_per_chunk × C` | `num_modules × C_pad` | `MODEL_STATE` |
| `m1_temps` | `modules_per_chunk` | `num_modules` | `MODEL_STATE` |
| `m2_temps` | `modules_per_chunk` | `num_modules` | `MODEL_STATE` |

These buffers persist across batches (`MODEL_STATE` lifecycle role) and are indexed by `parameter_offset` in each `adam_update` dispatch.

#### 7.2 Intermediate Gradient Buffers

The reduction tree output buffers and normalized gradient buffers are **per-module-chunk intermediates** (`BATCH_INTERMEDIATE` lifecycle role). They are sized to the chunk:

| Buffer | Size | Lifecycle Role |
|:---|:---|:---|
| `summed_grad_modw[m]` | `modules_per_chunk × H_pad × C_pad` | `BATCH_INTERMEDIATE` |
| `summed_grad_modb[m]` | `modules_per_chunk × C_pad` | `BATCH_INTERMEDIATE` |
| `summed_grad_temps[m]` | `modules_per_chunk` | `BATCH_INTERMEDIATE` |
| `final_grad_modw[m]` | Same as summed | `BATCH_INTERMEDIATE` |
| `final_grad_modb[m]` | Same as summed | `BATCH_INTERMEDIATE` |
| `final_grad_temps[m]` | Same as summed | `BATCH_INTERMEDIATE` |

These are allocated per-module-chunk and freed after their consuming `adam_update` dispatch completes. When `num_module_chunks == 1`, there is one set of intermediates sized to the full model — identical to current behavior.

---

### 8. Formal Invariant

The following invariant is added to CONCEPT.md §2 (The Recursive, Tiered Aggregation Engine), immediately following the Indirection Contract description:

> **Module-Chunk Isolation Invariant (ADR-028).** For parameter-gradient reduction trees (Nodes 15 and associated optimizer paths), the offset lists constructed by the Host Orchestrator SHALL contain only tile offsets belonging to a **single module chunk**. Cross-module-chunk reduction is architecturally prohibited because it conflates gradients for disjoint parameter sets. This invariant is structurally enforced by the `ModuleChunkGather` primitive, which produces offset lists scoped to a single module chunk's tiles. When `num_module_chunks == 1`, the invariant is trivially satisfied — all tiles belong to the single module chunk.
>
> The diagnostic reduction (Node 14) and the hidden-gradient reduction (Node 16) are exempt from this invariant: Node 14's output layout encodes the correct final position per tile, and Node 16 operates on a permuted SoA buffer that has already collapsed the module dimension via Node 13's class-chunk summation.

---

### 9. Validation

#### 9.1 Existing Scenarios

The following existing validation scenarios now exercise module-chunk isolation:

- **The Hydra (Massive `num_heads`):** With `num_heads` large enough to force `num_module_chunks > 1`, this scenario validates that per-module-chunk reduction trees produce correct summed gradients, that `adam_update` correctly indexes into monolithic state buffers via `parameter_offset`, and that convergence matches a single-module-chunk baseline.

- **The Colossus (Holistic Stress Test):** Already tests "the orchestrator's robustness by forcing it to compose a single, valid execution plan that correctly handles [all] reduction paths." Now additionally validates the per-module-chunk sub-graph parallelism.

#### 9.2 New Scenario

> **Scenario: The Hydra's Spine (Module-Chunk Isolation Validation)**
>
> - **Description:** A training task with `num_modules` chosen to force `num_module_chunks ≥ 2` at the target memory budget. Two runs are executed: (A) the orchestrator's natural chunking (`num_module_chunks = M`), and (B) a reference run with artificially increased memory budget forcing `num_module_chunks = 1`.
> - **Validation Focus:** Confirms that per-parameter weight updates are identical (within floating-point tolerance) between runs (A) and (B). This validates that the `ModuleChunkGather` primitive correctly isolates tile subsets, that the per-module-chunk reduction trees produce the same summed gradients as the monolithic tree, and that `adam_update`'s `parameter_offset` indexing is correct.
> - **Key Insight:** Proves that module-dimension decomposition is a **transparent scheduling decision** — it changes the plan's structure but not its mathematical output. The user observes identical training dynamics regardless of `num_module_chunks`, confirming that the Module-Chunk Isolation Invariant is correctly enforced.

---

### 10. Impact Summary

| Component | Change | Breaking? |
|:---|:---|:---|
| `ModuleChunkGather` | New gather primitive | No (additive) |
| `TiledGather` | No longer used for parameter-gradient reductions | Deprecation candidate |
| Plan builder (Phase II) | Per-module-chunk reduction loop | Internal restructuring |
| Plan builder (Phase V) | Per-module-chunk update loop | Internal restructuring |
| `adam_update` contract | Add `parameter_offset` scalar | **Yes** (kernel interface change) |
| `clamp_temperatures` contract | Add `parameter_offset`, rename count scalar | **Yes** (kernel interface change) |
| Node 22 | Dissolved into DAG dependency edges | Architectural amendment |
| Optimizer state buffers | Sized to full model | Allocation change |
| CONCEPT.md §2 | Module-Chunk Isolation Invariant added | Documentary |
| CONCEPT.md §4 (barriers) | Batch Synchronization Point redefined | Documentary |
| Mermaid diagram (Phase V) | Per-module-chunk update sub-graphs | Documentary |
| `kernels.cl.h` | Two kernel signatures amended | **Yes** (all backends) |

All three backends (OpenCL, Vulkan, CPU) require implementation updates for the two amended kernel signatures. The `parameter_offset = 0` degeneration ensures backward compatibility with existing test fixtures when `num_module_chunks == 1`.
