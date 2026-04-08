# Phase 13: Module-Chunk Isolation in Parameter-Gradient Reduction Trees

**Status:** Not started  
**Phase:** 13  
**Objective:** Fix silent gradient corruption when `num_module_chunks > 1` by introducing the `ModuleChunkGather` primitive, restructuring the plan builder to construct per-module-chunk reduction and update sub-graphs, amending the `adam_update` and `clamp_temperatures` kernel contracts with `parameter_offset` and `total_parameter_count` scalars, dissolving the monolithic Node 22 barrier into DAG dependency edges, and validating with the Hydra's Spine convergence scenario.  
**Governing ADR:** ADR-030 (Module-Chunk Isolation)  
**Rollback gate:** All existing tests pass. A new Hydra's Spine scenario confirms identical parameter updates (within floating-point tolerance) between `num_module_chunks > 1` and `num_module_chunks == 1` runs. When `num_module_chunks == 1`, the plan builder produces a DAG structurally equivalent to the pre-fix state (single-iteration loop, `parameter_offset = 0`).  
**Dependencies:** Phase 1 (Plan Model — complete), Phase 3 (CPU Backend — complete), Phase 10 (State-Precision Accumulation — complete), Phase 12A (Optimizer Hyperparameter Configuration — complete). All three backend renderers must be updated for the amended kernel signatures.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 13.1: Add `ModuleChunkGather` and `ModuleBufferKind` to `workload_primitives.py`](#step-131-add-modulechunkgather-and-modulebufferkind-to-workload_primitivespy)
   - [Step 13.2: Amend `adam_update` kernel contract in `phase_3_update.py`](#step-132-amend-adam_update-kernel-contract-in-phase_3_updatepy)
   - [Step 13.3: Amend `clamp_temperatures` kernel contract in `phase_3_update.py`](#step-133-amend-clamp_temperatures-kernel-contract-in-phase_3_updatepy)
   - [Step 13.4: Restructure plan builder Phases II and IV — per-module-chunk reduction and normalization](#step-134-restructure-plan-builder-phases-ii-and-iv--per-module-chunk-reduction-and-normalization)
   - [Step 13.5: Restructure plan builder Phase V — per-module-chunk update and barrier dissolution](#step-135-restructure-plan-builder-phase-v--per-module-chunk-update-and-barrier-dissolution)
   - [Step 13.6: Correct optimizer state buffer sizing in plan builder](#step-136-correct-optimizer-state-buffer-sizing-in-plan-builder)
   - [Step 13.7: Amend OpenCL `adam_update` kernel](#step-137-amend-opencl-adam_update-kernel)
   - [Step 13.8: Amend OpenCL `clamp_temperatures` kernel](#step-138-amend-opencl-clamp_temperatures-kernel)
   - [Step 13.9: Amend CPU `adam_update` kernel](#step-139-amend-cpu-adam_update-kernel)
   - [Step 13.10: Amend CPU `clamp_temperatures` kernel](#step-1310-amend-cpu-clamp_temperatures-kernel)
   - [Step 13.11: Amend Vulkan `adam_update` shader](#step-1311-amend-vulkan-adam_update-shader)
   - [Step 13.12: Amend Vulkan `clamp_temperatures` shader](#step-1312-amend-vulkan-clamp_temperatures-shader)
   - [Step 13.13: Update OpenCL renderer for amended signatures](#step-1313-update-opencl-renderer-for-amended-signatures)
   - [Step 13.14: Update CPU renderer for amended signatures](#step-1314-update-cpu-renderer-for-amended-signatures)
   - [Step 13.15: Update Vulkan renderer for amended signatures](#step-1315-update-vulkan-renderer-for-amended-signatures)
   - [Step 13.16: Amend CONCEPT.md](#step-1316-amend-conceptmd)
   - [Step 13.17: Amend CONTRACT.md](#step-1317-amend-contractmd)
   - [Step 13.18: Write Tier 1 tests for `ModuleChunkGather`](#step-1318-write-tier-1-tests-for-modulechunkgather)
   - [Step 13.19: Update existing Tier 1 plan builder tests](#step-1319-update-existing-tier-1-plan-builder-tests)
   - [Step 13.20: Write Tier 2 kernel tests for `parameter_offset` and `total_parameter_count`](#step-1320-write-tier-2-kernel-tests-for-parameter_offset-and-total_parameter_count)
   - [Step 13.21: Implement Hydra's Spine convergence scenario](#step-1321-implement-hydras-spine-convergence-scenario)
   - [Step 13.22: Validate rollback gate](#step-1322-validate-rollback-gate)
4. [Migration Order Rationale](#4-migration-order-rationale)
5. [Degeneration Verification](#5-degeneration-verification)
6. [Risk Register](#6-risk-register)

---

## 1. Scope & Constraints

### In scope

- Adding `ModuleChunkGather` (a new `GatherPrimitive` subclass) and `ModuleBufferKind` (a new enum) to `src/shared/workload_primitives.py`.
- Amending the `adam_update` kernel contract to add `parameter_offset` and `total_parameter_count` (before and after `parameter_count` respectively), and amending the state-role buffer tensor shapes from `(parameter_count)` to `(total_parameter_count)`.
- Amending the `clamp_temperatures` kernel contract to add `parameter_offset` and `total_parameter_count`, and rename `total_modules_count` → `parameter_count`.
- Restructuring `build_learn_plan()` Phase II to construct `num_module_chunks` independent reduction sub-graphs per parameter group (module weights, module biases, temperatures) using `ModuleChunkGather`.
- Restructuring `build_learn_plan()` Phase IV to construct `num_module_chunks` independent `normalize_gradients` dispatches per module-dimension parameter group, each depending only on its own module chunk's reduction tree rather than a coarse `norm_deps` set.
- Restructuring `build_learn_plan()` Phase V to construct `num_module_chunks` independent `adam_update` and `clamp_temperatures` dispatches with chunk-specific `parameter_offset` values.
- Dissolving Node 22 (`batch_sync_barrier`) into per-sub-graph DAG dependency edges.
- Correcting optimizer state buffer (`m1`, `m2`) allocation from per-chunk to full-model size.
- Amending the authoritative kernel declarations and `@kernel_contract` annotations in `kernels/kernels.cl.h` for both `adam_update` and `clamp_temperatures`.
- Amending the OpenCL implementations in `kernels/phase_3_update.cl.c` for both `adam_update` and `clamp_temperatures` kernel signatures and access patterns.
- Amending CPU `src/backends/cpu/kernel_sources/phase_3_update.inc` kernel signatures and access patterns, and the corresponding struct definitions in `src/backends/cpu/kernel_sources/cpu_kernels.h`.
- Amending CPU FFI type definitions in `src/backends/cpu/_ffi_types.py` to add `parameter_offset` and `total_parameter_count` to the `AdamUpdateArgs` and `ClampTemperaturesArgs` struct layouts.
- Amending Vulkan `src/backends/vulkan/kernel_sources/adam_update.comp` and `clamp_temperatures.comp` shader signatures and access patterns.
- Updating all three backend renderers (OpenCL, CPU, Vulkan) to pass `parameter_offset` and `total_parameter_count` when dispatching `adam_update` and `clamp_temperatures`.
- Adding the Module-Chunk Isolation Invariant to CONCEPT.md §2.
- Amending the Batch Synchronization Point definition in CONCEPT.md §4.
- Adding the Hydra's Spine convergence test scenario.
- Writing Tier 1 tests for `ModuleChunkGather` and updated plan builder output.
- Writing Tier 2 tests for the amended kernel access patterns with non-zero `parameter_offset`.

### Out of scope

- Changes to non-parameter-gradient reduction trees (Node 14 probs/loss, Node 16 Grad_H) — these are exempt from the Module-Chunk Isolation Invariant per ADR-030 §Scope of Impact.
- Changes to shared-layer `adam_update` dispatches (Grad_SW, Grad_SB) — these have no module dimension and require no per-chunk loop. However, their scalar dictionaries must be updated to pass the new `parameter_offset=0` and `total_parameter_count=parameter_count` parameters to match the amended kernel contract.
- Deprecation or removal of `TiledGather` — it remains valid for non-parameter-gradient gather operations. Formal deprecation is a future consideration.
- Changes to `PrecisionConfig`, `ReductionTreePlan`, or `StreamingLoopPlan` — the fix operates at the gather primitive and plan builder level.
- Checkpoint migration logic for resized optimizer state buffers — out of scope for this phase; existing checkpoints saved with `num_module_chunks == 1` are already full-model-sized.

### Key constraint: degeneration equivalence

When `num_module_chunks == 1`, the amended plan builder must produce a DAG that is structurally equivalent to the pre-fix state. The per-module-chunk loop executes once, `parameter_offset = 0`, `parameter_count = full model size`, and all downstream behavior is identical. This ensures no behavioral regression for the common single-chunk configuration.

### Key constraint: kernel interface breaking changes

The `adam_update` and `clamp_temperatures` kernel signatures change across all three backends. This is a coordinated cross-backend change — all renderers must be updated in the same phase to maintain the contract invariant that kernel signatures match their `KernelContract` specifications.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/shared/workload_primitives.py` | Contains `GatherPrimitive` ABC, `TiledGather`, `LinearlyChunkedGather`, `ContiguousGather`, `TilingScheme`. No `ModuleChunkGather` or `ModuleBufferKind`. |
| `src/shared/plan_builder.py` | `build_learn_plan()` Phase II constructs single `TiledGather` per parameter group over all tiles. Phase IV constructs all five `normalize_gradients` dispatches sharing a single coarse `norm_deps` set (false serialization). Phase V uses monolithic `BarrierNode("batch_sync_barrier")` gating all updates. `adam_update` dispatches pass `parameter_count` but no `parameter_offset`. `clamp_temperatures` dispatch passes `total_modules_count`. |
| `src/shared/kernel_contracts/phase_3_update.py` | `adam_update_contract` has 7 scalar params (no `parameter_offset` or `total_parameter_count`). `clamp_temperatures_contract` has 3 scalar params with `total_modules_count` (not `parameter_count`). State-role buffer shapes are `(parameter_count)`. |
| `kernels/kernels.cl.h` | Authoritative algorithmic specification (ADR-013). `adam_update` declaration has 7 scalar params (no `parameter_offset` or `total_parameter_count`). `clamp_temperatures` declaration has `src_scalar_NATURAL_total_modules_count`. Buffer `@param` annotations use `(src_scalar_NATURAL_parameter_count)` tensor shapes for state-role buffers. |
| `kernels/phase_3_update.cl.c` | OpenCL `adam_update` and `clamp_temperatures` kernels. `adam_update` indexes `parameters[i]`, `m1[i]`, `m2[i]` directly. `clamp_temperatures` indexes `temps[i]` directly. |
| `src/backends/cpu/kernel_sources/phase_3_update.inc` | CPU `adam_update` and `clamp_temperatures` — same direct indexing pattern. |
| `src/backends/cpu/kernel_sources/cpu_kernels.h` | `DECLARE_PRECISION_STRUCTS` macro defining `AdamUpdateArgs` (6 REAL scalars + 1 NATURAL scalar `parameter_count`) and `ClampTemperaturesArgs` (2 REAL scalars + 1 NATURAL scalar `total_modules_count`). |
| `src/backends/cpu/_ffi_types.py` | Python ctypes struct layouts for `AdamUpdateArgs` and `ClampTemperaturesArgs`, mirroring the C header field order. |
| `src/backends/vulkan/kernel_sources/adam_update.comp` | Vulkan `adam_update` shader — same direct indexing pattern. |
| `src/backends/vulkan/kernel_sources/clamp_temperatures.comp` | Vulkan `clamp_temperatures` shader — same direct indexing pattern. |
| `CONCEPT.md` §2 | Recursive Aggregation Engine description. No Module-Chunk Isolation Invariant. |
| `CONCEPT.md` §4 | Synchronization Barriers. Node 22 defined as monolithic Batch Synchronization Barrier. |

---

## 3. Task Breakdown

---

### Step 13.1: Add `ModuleChunkGather` and `ModuleBufferKind` to `workload_primitives.py`

**Governing authority:** ADR-030 §1  
**File:** `src/shared/workload_primitives.py`

Add a new `ModuleBufferKind` enum and `ModuleChunkGather` frozen dataclass after the existing `TiledGather` definition.

#### `ModuleBufferKind`

```python
class ModuleBufferKind(enum.Enum):
    """Identifies the parameter-gradient buffer for elements_per_tile dispatch."""
    WEIGHTS = "weights"   # modules_per_chunk × padded_hidden_count × padded_total_output_class_count
    BIASES  = "biases"    # modules_per_chunk × padded_total_output_class_count
    TEMPS   = "temps"     # modules_per_chunk
```

#### `ModuleChunkGather`

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

This requires adding an `elements_per_tile(kind: ModuleBufferKind) -> int` method to `TilingScheme`. Implementation requires adding `modules_per_chunk`, `padded_hidden_count`, and `padded_total_output_class_count` as stored fields — `modules_per_chunk` is currently computed transiently inside `get_tile()` via ceiling division, and the two padded-dimension fields do not yet exist. The new method codifies the per-buffer-kind element count:

- `WEIGHTS`: `modules_per_chunk × padded_hidden_count × padded_total_output_class_count`
- `BIASES`: `modules_per_chunk × padded_total_output_class_count`
- `TEMPS`: `modules_per_chunk`

These fields are workload-shape descriptors already known at tiling construction time in the plan builder. The preferred approach is adding them directly to `TilingScheme` rather than introducing a `ModelSpec` reference.

**Verification:** Unit tests confirm `ModuleChunkGather.get_offsets()` returns the expected tile subset for each module chunk. Verify degeneration: when `num_module_chunks == 1`, offsets equal `TiledGather.get_offsets()`.

---

### Step 13.2: Amend `adam_update` kernel contract in `phase_3_update.py`

**Governing authority:** ADR-030 §3.1  
**File:** `src/shared/kernel_contracts/phase_3_update.py`

Add `parameter_offset` and `total_parameter_count` scalar parameters to `adam_update_contract`, positioned before and after `parameter_count` respectively:

```python
ScalarParamSpec("parameter_offset", "src", "NATURAL"),
ScalarParamSpec("parameter_count", "src", "NATURAL"),
ScalarParamSpec("total_parameter_count", "src", "NATURAL"),
```

Amend the three state-role `BufferParamSpec` entries (`update_buffer_GLOBAL_parameters`, `update_buffer_GLOBAL_m1`, `update_buffer_GLOBAL_m2`) to reflect full-model buffer sizing:

- `tensor_shape`: changes from `("parameter_count",)` to `("total_parameter_count",)`.
- `validation_preconditions`: add `"(parameter_offset + parameter_count) <= total_parameter_count"`.
- `calculability_proof`: change to `("total_parameter_count",)`.

The `src_buffer_GLOBAL_final_grad` buffer remains `(parameter_count)` — it is a per-dispatch intermediate.

**Verification:** Existing tests that construct `adam_update_contract` scalars will fail (expected — they must be updated in Step 13.19). The contract's scalar count increments from 7 to 9.

---

### Step 13.3: Amend `clamp_temperatures` kernel contract in `phase_3_update.py`

**Governing authority:** ADR-030 §3.2  
**File:** `src/shared/kernel_contracts/phase_3_update.py`

1. Add `parameter_offset` and `total_parameter_count` scalar parameters.
2. Rename `total_modules_count` → `parameter_count` for consistency with the general pattern.
3. Update the buffer's `tensor_shape` and `calculability_proof` to use `total_parameter_count`.

```python
scalar_params=(
    ScalarParamSpec("min_value", "src", "REAL"),
    ScalarParamSpec("max_value", "src", "REAL"),
    ScalarParamSpec("parameter_offset", "src", "NATURAL"),
    ScalarParamSpec("parameter_count", "src", "NATURAL"),
    ScalarParamSpec("total_parameter_count", "src", "NATURAL"),
),
```

The buffer spec for `update_buffer_GLOBAL_temps` changes `tensor_shape` from `("total_modules_count",)` to `("total_parameter_count",)` and adds `validation_preconditions` for slice-access bounds: `"(parameter_offset + parameter_count) <= total_parameter_count"`. This uses the same generic shape descriptor as the amended `adam_update` state-role buffers (Step 13.2).

**Verification:** Same as Step 13.2 — downstream tests that reference `total_modules_count` will fail until updated.

---

### Step 13.4: Restructure plan builder Phases II and IV — per-module-chunk reduction and normalization

**Governing authority:** ADR-030 §2.1  
**File:** `src/shared/plan_builder.py`

> **Scope note:** ADR-030 explicitly mandates per-module-chunk sub-graphs (§2.1) and Node 22 barrier dissolution (§4). The Phase IV dependency decoupling below extends beyond the ADR's explicit text but is a necessary consequence: without it, the monolithic `norm_deps` set creates a false serialization bottleneck one hop before the dissolved barrier, defeating the per-module-chunk parallelism the ADR intends to enable.

#### Phase II: Per-module-chunk reduction trees

Replace the three single `TiledGather` → `ReductionTreeNode` constructions for module weights, module biases, and temperatures with a loop over `range(num_module_chunks)`:

```python
for m in range(num_module_chunks):
    gather_w = ModuleChunkGather(scheme, m, ModuleBufferKind.WEIGHTS)
    gather_b = ModuleChunkGather(scheme, m, ModuleBufferKind.BIASES)
    gather_t = ModuleChunkGather(scheme, m, ModuleBufferKind.TEMPS)

    # Build reduction trees per module chunk
    # Node IDs become "reduce_mod_grads_m0", "reduce_mod_grads_m1", etc.
    # When num_module_chunks == 1, the single iteration produces the same
    # sub-graph structure — node IDs gain a "_m0" suffix.
```

Each `ReductionTreeNode` gets a module-chunk-suffixed `node_id` (e.g., `"reduce_mod_grads_m0"`). The corresponding `normalize_gradients` dispatches also gain suffixes.

**Buffer allocation changes:**

- Reduction tree output buffers (`b_summed_grad_mod`, `b_summed_grad_mod_biases`, `b_summed_grad_temps`) become per-module-chunk: `b_summed_grad_mod_m{i}`, sized to `modules_per_chunk × ...` (chunk-sized, `BATCH_INTERMEDIATE` lifecycle).
- Normalized gradient buffers similarly become per-module-chunk.

**Dependency structure:** Each per-module-chunk reduction sub-graph depends on `"item_sync_barrier"` (same as current). Sub-graphs for different module chunks have no cross-dependencies.

#### Phase IV: Per-module-chunk normalization with decoupled dependencies

The current Phase IV constructs `normalize_gradients` dispatches for all five parameter groups (module weights, module biases, temperatures, shared weights, shared biases), all sharing a single coarse dependency set:

```python
norm_deps = frozenset({
    "reduce_shared_grads", "reduce_shared_bias_grads",
    "reduce_mod_grads", "reduce_mod_bias_grads",
    "reduce_temp_grads"})
```

This creates false serialization: `normalize_gradients_module` waits on `reduce_shared_grads` and vice versa. Dissolving Node 22 alone (Step 13.5) would leave an equivalent bottleneck one hop earlier.

The restructured Phase IV:

1. **Module-dimension normalize dispatches** become per-module-chunk, inside the same `range(num_module_chunks)` loop. Each depends only on its own reduction tree:
   - `normalize_gradients_module_m{i}` depends on `{"reduce_mod_grads_m{i}"}`
   - `normalize_gradients_module_biases_m{i}` depends on `{"reduce_mod_bias_grads_m{i}"}`
   - `normalize_gradients_temps_m{i}` depends on `{"reduce_temp_grads_m{i}"}`

2. **Shared-layer normalize dispatches** remain singular (no module dimension) and depend only on their own reduction trees:
   - `normalize_gradients_shared` depends on `{"reduce_shared_grads"}`
   - `normalize_gradients_shared_biases` depends on `{"reduce_shared_bias_grads"}`

The monolithic `norm_deps` set is eliminated entirely.

**Verification:** Tier 1 plan builder tests verify that the learn plan contains `num_module_chunks` reduction tree nodes and `num_module_chunks` normalize dispatch nodes per module-dimension parameter group. Each normalize dispatch depends only on its own reducer. For `num_module_chunks == 1`, the plan has one tree and one normalize per group (degeneration equivalence). Shared-layer normalizes have no dependency on module reduction trees.

---

### Step 13.5: Restructure plan builder Phase V — per-module-chunk update and barrier dissolution

**Governing authority:** ADR-030 §§2.2, 4  
**File:** `src/shared/plan_builder.py`

#### Barrier dissolution

Remove the `BarrierNode("batch_sync_barrier")` construction. Each `adam_update` dispatch now depends directly on its specific `normalize_gradients` predecessor rather than on a monolithic barrier.

#### Per-module-chunk update loop

Replace the single `adam_update_module`, `adam_update_module_biases`, `adam_update_temps`, and `clamp_temperatures` dispatches with a loop over `range(num_module_chunks)`:

```python
for m in range(num_module_chunks):
    offset_w = m * modules_per_chunk * padded_hidden_count * padded_total_output_class_count
    offset_b = m * modules_per_chunk * padded_total_output_class_count
    offset_t = m * modules_per_chunk

    # adam_update dispatches: node IDs "adam_update_module_m0", etc.
    # Each depends on its own "normalize_gradients_module_m{m}"
    # Scalars include parameter_offset=offset_w, parameter_count=chunk_param_count,
    #   total_parameter_count=full_model_param_count
    ...

    # clamp_temperatures dispatches: node IDs "clamp_temperatures_m0", etc.
    # Each depends on its own "adam_update_temps_m{m}"
    # Scalars include parameter_offset=offset_t, parameter_count=modules_per_chunk,
    #   total_parameter_count=total_modules
    ...
```

**Shared-layer updates** (`adam_update_shared`, `adam_update_shared_biases`) are unchanged in their logical role — they have no module dimension. However, their scalar dictionaries must include the new parameters to match the amended contract: `parameter_offset=0`, `total_parameter_count=parameter_count`. Their dependency changes from `{"batch_sync_barrier"}` to their own respective `normalize_gradients` predecessors: `adam_update_shared` depends on `{"normalize_gradients_shared"}`, `adam_update_shared_biases` depends on `{"normalize_gradients_shared_biases"}`.

#### Final retrieval amendment

The `final_batch_retrieval` node's `depends_on` set expands to include all per-module-chunk terminal nodes:

```python
depends_on=frozenset(
    {"adam_update_shared", "adam_update_shared_biases"}
    | {f"adam_update_module_m{m}" for m in range(num_module_chunks)}
    | {f"adam_update_module_biases_m{m}" for m in range(num_module_chunks)}
    | {f"clamp_temperatures_m{m}" for m in range(num_module_chunks)}
)
```

#### Topological order amendment

The static `topo_list` must be replaced with dynamically constructed ordering that interleaves per-module-chunk nodes. The topological ordering constraint is: for each module chunk `m`, `reduce_*_m{m}` → `normalize_*_m{m}` → `adam_update_*_m{m}` → (for temps) `clamp_temperatures_m{m}`. Cross-chunk ordering is unconstrained.

The recommended algorithm is:

1. **Static prefix:** Retain the fixed Phase I node sequence (`forward_pass` through `clip_partial_grads`) and the Phase II/III shared-path nodes (`gather_permute_grad_h`, `item_sync_barrier`, `stabilize_reduce_grad_h`, streaming loop nodes, `reduce_shared_grads`, `reduce_shared_bias_grads`).
2. **Per-module-chunk loop:** For each `m` in `range(num_module_chunks)`, append the chain: `reduce_mod_grads_m{m}`, `reduce_mod_bias_grads_m{m}`, `reduce_temp_grads_m{m}`.
3. **Shared normalizes and updates:** Append `normalize_gradients_shared`, `normalize_gradients_shared_biases`, `adam_update_shared`, `adam_update_shared_biases`.
4. **Per-module-chunk normalizes and updates:** For each `m`, append: `normalize_gradients_module_m{m}`, `normalize_gradients_module_biases_m{m}`, `normalize_gradients_temps_m{m}`, `adam_update_module_m{m}`, `adam_update_module_biases_m{m}`, `adam_update_temps_m{m}`, `clamp_temperatures_m{m}`.
5. **Static suffix:** Append `final_batch_retrieval`.

This preserves a valid topological order (every node appears after all its dependencies) while keeping the per-chunk chains grouped for readability. The shared updates are placed before per-module-chunk updates because their dependencies (shared normalizes) resolve earlier; a backend executing nodes sequentially in this order avoids unnecessarily delaying them. A DAG validation pass (see Risk R3 mitigation) confirms correctness. Note that the backend retains full scheduling freedom per CONCEPT.md §6 — any valid topological order may be used at execution time.

**Verification:** Tier 1 tests confirm no `batch_sync_barrier` node exists in the plan. Each `adam_update` node depends only on its direct predecessor. The DAG is acyclic. `final_batch_retrieval` depends on all terminal update nodes.

---

### Step 13.6: Correct optimizer state buffer sizing in plan builder

**Governing authority:** ADR-030 §5  
**File:** `src/shared/plan_builder.py`

Ensure optimizer state buffers (`b_m1_module`, `b_m2_module`, `b_m1_module_biases`, `b_m2_module_biases`, `b_m1_temps`, `b_m2_temps`) are allocated at **full model size**, not per-chunk size:

| Buffer | Correct size |
|:---|:---|
| `b_m1_module`, `b_m2_module` | `num_modules × padded_hidden_count × padded_total_output_class_count` |
| `b_m1_module_biases`, `b_m2_module_biases` | `num_modules × padded_total_output_class_count` |
| `b_m1_temps`, `b_m2_temps` | `num_modules` |

These are `MODEL_STATE` lifecycle role buffers — persistent across batches. Each per-module-chunk `adam_update` dispatch indexes into them via `parameter_offset`.

**Verification:** Inspect buffer descriptors in the plan. For `num_module_chunks > 1`, state buffer sizes exceed the per-chunk gradient buffer sizes. For `num_module_chunks == 1`, state and gradient buffer sizes are equal (degeneration).

---

### Step 13.7: Amend OpenCL `adam_update` kernel

**Governing authority:** ADR-030 §3.1, §3.3  
**Files:** `kernels/kernels.cl.h`, `kernels/phase_3_update.cl.c`

#### `kernels.cl.h` (authoritative declaration)

Amend the `adam_update` declaration (line 2193) to add `uint src_scalar_NATURAL_parameter_offset` before and `uint src_scalar_NATURAL_total_parameter_count` after `src_scalar_NATURAL_parameter_count` (see ADR-030 §3.3 for full amended signature). Amend the `@param` annotations for state-role buffers (`update_buffer_GLOBAL_parameters`, `update_buffer_GLOBAL_m1`, `update_buffer_GLOBAL_m2`):

- `Tensor Shape`: change from `(src_scalar_NATURAL_parameter_count)` to `(src_scalar_NATURAL_total_parameter_count)`.
- `Calculability Proof`: change to `[src_scalar_NATURAL_total_parameter_count]`.
- `Validation Preconditions`: add `(src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <= src_scalar_NATURAL_total_parameter_count`.

The `src_buffer_GLOBAL_final_grad` annotations remain `(src_scalar_NATURAL_parameter_count)` — it is a per-dispatch intermediate.

#### `kernels/phase_3_update.cl.c` (OpenCL implementation)

Add `uint src_scalar_NATURAL_parameter_offset` and `uint src_scalar_NATURAL_total_parameter_count` parameters to the `adam_update` kernel signature, positioned before and after `src_scalar_NATURAL_parameter_count` respectively.

Amend the kernel's access pattern from:

```c
parameters[i]  →  parameters[parameter_offset + i]
m1[i]          →  m1[parameter_offset + i]
m2[i]          →  m2[parameter_offset + i]
final_grad[i]  // unchanged — zero-indexed per-dispatch intermediate
```

The loop bounds remain `[0, parameter_count)`. Only the state-role buffer indexing gains the offset.

**Verification:** Compile the OpenCL kernel. Existing Tier 2 tests with `parameter_offset = 0` produce identical results.

---

### Step 13.8: Amend OpenCL `clamp_temperatures` kernel

**Governing authority:** ADR-030 §3.2, §3.3  
**Files:** `kernels/kernels.cl.h`, `kernels/phase_3_update.cl.c`

#### `kernels.cl.h` (authoritative declaration)

Amend the `clamp_temperatures` declaration (line 2253) to add `uint src_scalar_NATURAL_parameter_offset` and `uint src_scalar_NATURAL_total_parameter_count`, and rename `src_scalar_NATURAL_total_modules_count` → `src_scalar_NATURAL_parameter_count`. Amend the `@param` annotation for `update_buffer_GLOBAL_temps`:

- `Tensor Shape`: change from `(src_scalar_NATURAL_total_modules_count)` to `(src_scalar_NATURAL_total_parameter_count)`.
- `Calculability Proof`: change to `[src_scalar_NATURAL_total_parameter_count]`.
- `Validation Preconditions`: add `(src_scalar_NATURAL_parameter_offset + src_scalar_NATURAL_parameter_count) <= src_scalar_NATURAL_total_parameter_count`.

#### `kernels/phase_3_update.cl.c` (OpenCL implementation)

Add `uint src_scalar_NATURAL_parameter_offset` and `uint src_scalar_NATURAL_total_parameter_count` parameters. Rename `src_scalar_NATURAL_total_modules_count` → `src_scalar_NATURAL_parameter_count`.

Amend access pattern:

```c
temps[i]  →  temps[parameter_offset + i]
```

Loop bounds remain `[0, parameter_count)`.

**Verification:** Same as Step 13.7.

---

### Step 13.9: Amend CPU `adam_update` kernel

**Governing authority:** ADR-030 §3.1, §3.3  
**Files:** `src/backends/cpu/kernel_sources/phase_3_update.inc`, `src/backends/cpu/kernel_sources/cpu_kernels.h`

Mirror the OpenCL amendment: add `parameter_offset` and `total_parameter_count` fields to the `AdamUpdateArgs` struct in `cpu_kernels.h` (before and after `parameter_count` respectively, yielding the field order `..., epsilon, parameter_offset, parameter_count, total_parameter_count`), and amend state-buffer indexing to `[parameter_offset + i]` in `phase_3_update.inc`. The field ordering is ABI-critical for the FFI bridge.

**Verification:** Rebuild the CPU shared library (`ninja -C builddir`). Existing Tier 2 CPU tests pass with `parameter_offset = 0`.

---

### Step 13.10: Amend CPU `clamp_temperatures` kernel

**Governing authority:** ADR-030 §3.2, §3.3  
**Files:** `src/backends/cpu/kernel_sources/phase_3_update.inc`, `src/backends/cpu/kernel_sources/cpu_kernels.h`

Mirror the OpenCL amendment: add `parameter_offset` and `total_parameter_count` fields to `ClampTemperaturesArgs` in `cpu_kernels.h` (yielding the field order `..., min_value, max_value, parameter_offset, parameter_count, total_parameter_count`), rename `total_modules_count` → `parameter_count`, and amend indexing in `phase_3_update.inc`.

**Verification:** Same as Step 13.9.

---

### Step 13.11: Amend Vulkan `adam_update` shader

**Governing authority:** ADR-030 §3.1, §3.3  
**File:** `src/backends/vulkan/kernel_sources/adam_update.comp`

Add `parameter_offset` and `total_parameter_count` to the push constants layout. Amend buffer indexing to `[parameter_offset + gl_GlobalInvocationID.x]` for state-role buffers.

**Verification:** Recompile the Vulkan SPIR-V module. Existing Tier 2 Vulkan tests pass with `parameter_offset = 0`.

---

### Step 13.12: Amend Vulkan `clamp_temperatures` shader

**Governing authority:** ADR-030 §3.2, §3.3  
**File:** `src/backends/vulkan/kernel_sources/clamp_temperatures.comp`

Mirror Step 13.11: add `parameter_offset` and `total_parameter_count` to push constants, amend buffer indexing, rename count field.

**Verification:** Same as Step 13.11.

---

### Step 13.13: Update OpenCL renderer for amended signatures

**Governing authority:** ADR-030 §3  
**File:** `src/backends/opencl/renderer.py` (or equivalent binding dispatch module)

Update the `adam_update` and `clamp_temperatures` dispatch sites to pass `parameter_offset` and `total_parameter_count` as kernel arguments. The plan builder provides both in the scalar params dict; the renderer reads them and passes them to the OpenCL `clSetKernelArg` calls in the correct positions.

**Verification:** OpenCL Tier 2 tests pass. The renderer correctly maps the contract's scalar parameter ordering to the kernel's argument positions.

---

### Step 13.14: Update CPU renderer for amended signatures

**Governing authority:** ADR-030 §3  
**Files:** `src/backends/cpu/renderer.py` (or equivalent FFI dispatch module), `src/backends/cpu/_ffi_types.py`

Update the CPU backend's function call sites to pass `parameter_offset` and `total_parameter_count` to the C function. The CPU backend uses FFI (ctypes) — the `AdamUpdateArgs` and `ClampTemperaturesArgs` struct definitions in `_ffi_types.py` must be updated to include both `parameter_offset` and `total_parameter_count` fields at the correct positions (before and after `parameter_count` respectively), matching the amended `cpu_kernels.h` layout.

**Verification:** CPU Tier 2 tests pass. Rebuild the shared library first (`ninja -C builddir`).

---

### Step 13.15: Update Vulkan renderer for amended signatures

**Governing authority:** ADR-030 §3  
**File:** `src/backends/vulkan/renderer.py` (or equivalent push-constant dispatch module)

Update the Vulkan push constants struct to include `parameter_offset` and `total_parameter_count` for `adam_update` and `clamp_temperatures` dispatches. The push constants layout in the renderer must match the amended GLSL layout.

**Verification:** Vulkan Tier 2 tests pass.

---

### Step 13.16: Amend CONCEPT.md

**Governing authority:** ADR-030 §§2.4, 4, 6  
**File:** `CONCEPT.md`

#### Amendment 13.16.1 — Module-Chunk Isolation Invariant (§2)

Add the following invariant to §2 (The Recursive, Tiered Aggregation Engine), immediately after the Indirection Contract description:

> **Module-Chunk Isolation Invariant (ADR-030).** For parameter-gradient reduction trees (Nodes 15 and associated optimizer paths), the offset lists constructed by the Host Orchestrator SHALL contain only tile offsets belonging to a **single module chunk**. Cross-module-chunk reduction is architecturally prohibited because it conflates gradients for disjoint parameter sets. This invariant is structurally enforced by the `ModuleChunkGather` primitive, which produces offset lists scoped to a single module chunk's tiles. When `num_module_chunks == 1`, the invariant is trivially satisfied — all tiles belong to the single module chunk.
>
> `TiledGather` remains the correct primitive for non-parameter-gradient reductions (Node 14), where the output layout encodes the correct final position per tile and cross-module-chunk summation is mathematically valid. The hidden-gradient reduction (Node 16) is also exempt: it operates on a permuted SoA buffer that has already collapsed the module dimension via Node 13's class-chunk summation.

#### Amendment 13.16.2 — Batch Synchronization Point (§4)

Replace the Batch Synchronization Point barrier definition with:

> **Batch Synchronization Point:** A synchronization property ensuring that parameter updates occur only after their gradient dependencies are fully resolved. When the module dimension is decomposed into chunks, this property is satisfied by per-sub-graph dependency edges rather than a single monolithic barrier: each `adam_update` dispatch's dependency on its own `normalize_gradients` output expresses the synchronization contract directly in the DAG structure. When `num_module_chunks == 1`, the property is trivially satisfied — a single dependency edge from each normalized gradient to its corresponding `adam_update` is equivalent to the former barrier.
>
> The `final_batch_event` `RetrievalNode` remains the sole plan-wide synchronization point, collecting all terminal update nodes. It is the observable successor of the former Node 22 barrier.
>
> - **Canonical Example:** The per-module-chunk dependency edges from `normalize_gradients[m]` to `adam_update[m]` for each parameter group and module chunk index `m`, converging at `final_batch_event`.

#### Amendment 13.16.3 — Mermaid Diagram (Phase V)

Update the Phase V section of the Mermaid diagram to show per-module-chunk `adam_update` dispatches converging at `final_batch_event`.

**Verification:** Documentary consistency review. Cross-reference with ADR-030 text.

---

### Step 13.17: Amend CONTRACT.md

**Governing authority:** ADR-030 §3  
**File:** `CONTRACT.md`

Update the Node 24 (`adam_update`) contract documentation to reflect the `parameter_offset` parameter, amended buffer sizing, and slice-indexed access pattern. Update the Node 25 (`clamp_temperatures`) contract documentation for the renamed `parameter_count` and added `parameter_offset`.

**Verification:** Documentary consistency with `phase_3_update.py` contract objects and kernel source signatures.

---

### Step 13.18: Write Tier 1 tests for `ModuleChunkGather`

**Governing authority:** ADR-030 §1, ADR-016  
**File:** `tests/tier1/test_workload_primitives.py` (new tests in existing module, or new module)

Test cases:

1. **Basic offset correctness.** For `num_module_chunks=3, num_class_chunks=4`, verify that `ModuleChunkGather(scheme, m=0, ...)` returns offsets for tiles `[0,1,2,3]`, `m=1` returns `[4,5,6,7]`, `m=2` returns `[8,9,10,11]`, each scaled by `elements_per_partial`.
2. **Degeneration equivalence.** When `num_module_chunks=1`, verify `ModuleChunkGather(scheme, 0, kind).get_offsets()` equals `TiledGather(scheme, epp).get_offsets()` for each `ModuleBufferKind`.
3. **Boundary validation.** Verify `ModuleChunkGather(scheme, m=-1, ...)` and `ModuleChunkGather(scheme, m=num_module_chunks, ...)` raise `ValueError`.
4. **`num_partials` property.** Verify it equals `num_class_chunks` regardless of `module_chunk_index`.
5. **`elements_per_partial` property.** Verify it delegates correctly to `TilingScheme.elements_per_tile(kind)` for each `ModuleBufferKind` variant.

**Verification:** All new tests green.

---

### Step 13.19: Update existing Tier 1 plan builder tests

**Governing authority:** ADR-030, ADR-016  
**File:** `tests/tier1/test_plan_builder.py` (or equivalent)

Existing tests that construct or inspect the learn plan will break due to:

1. Removal of `batch_sync_barrier` node.
2. Node ID changes (`adam_update_module` → `adam_update_module_m0`, etc.).
3. New `parameter_offset` scalar in `adam_update` and `clamp_temperatures` dispatches.
4. Changed `clamp_temperatures` scalar name (`total_modules_count` → `parameter_count`).
5. Changed dependency edges (no barrier, direct predecessor dependencies).
6. Changed topological order.

Update all affected assertions to reflect the amended plan structure. Add new assertions verifying:

- Per-module-chunk node multiplicity: `num_module_chunks` reduction tree nodes per parameter group.
- Dependency edge correctness: each `adam_update_module_m{i}` depends on `normalize_gradients_module_m{i}`.
- No `batch_sync_barrier` node present.
- `final_batch_retrieval` depends on the full set of terminal update nodes.

**Verification:** All Tier 1 tests green.

---

### Step 13.20: Write Tier 2 kernel tests for `parameter_offset` and `total_parameter_count`

**Governing authority:** ADR-030 §3, ADR-016  
**File:** `tests/tier2/` (appropriate backend-specific test modules)

Test cases per backend:

1. **Non-zero offset correctness.** Allocate a full-model-sized parameter buffer pre-filled with known values. Dispatch `adam_update` with `parameter_offset = N`, `parameter_count = M`, `total_parameter_count = buffer_size`. Verify only elements `[N, N+M)` are updated; elements outside this range are unchanged.
2. **Boundary safety.** Dispatch with `parameter_offset + parameter_count == total_parameter_count` (maximum valid slice). Verify no out-of-bounds access.
3. **`clamp_temperatures` offset correctness.** Same structure: pre-filled buffer, non-zero offset, verify only the slice is clamped.
4. **Degeneration (offset=0) equivalence.** Verify that `parameter_offset=0, total_parameter_count=parameter_count` produces bitwise-identical results to the pre-amendment kernel behavior.
5. **`total_parameter_count` bounds invariant.** Verify that `(parameter_offset + parameter_count) <= total_parameter_count` holds for all test dispatches. If the backend supports device-side assertions, test that violation of this invariant is detected.

**Verification:** All new Tier 2 tests green on each available backend.

---

### Step 13.21: Implement Hydra's Spine convergence scenario

**Governing authority:** ADR-030 §7.2  
**File:** `tests/convergence/test_convergence_module_chunk_isolation.py` (new)

Implement the "Hydra's Spine" convergence scenario:

- **Configuration:** `num_modules` chosen to force `num_module_chunks >= 2` at the test's memory budget.
- **Run A:** Natural chunking (`num_module_chunks = M > 1`).
- **Run B:** Reference run with artificially increased memory budget forcing `num_module_chunks = 1`.
- **Assertion:** Per-parameter weight updates are identical within floating-point tolerance between runs A and B after each training step.
- **Key validation:** Module-dimension decomposition is a transparent scheduling decision — it changes the plan's structure but not its mathematical output.

Mark with `@pytest.mark.convergence_fast` if execution time permits, otherwise `@pytest.mark.convergence_full`.

**Verification:** The Hydra's Spine scenario passes, confirming that `ModuleChunkGather`-based reduction produces identical results to monolithic reduction.

---

### Step 13.22: Validate rollback gate

**Governing authority:** ADR-030  
**File:** All test modules

Run the full test suite:

```bash
pytest tests/ -x --tb=short 2>&1 | tee /tmp/phase13-rollback.log
grep -E "passed|failed|error" /tmp/phase13-rollback.log
```

The rollback gate requires:

1. All pre-existing tests pass (possibly with updated assertions from Step 13.19).
2. All new Tier 1 tests from Step 13.18 pass.
3. All new Tier 2 tests from Step 13.20 pass.
4. The Hydra's Spine convergence scenario (Step 13.21) passes.
5. When `num_module_chunks == 1`, the plan builder produces structurally equivalent output to the pre-fix state (modulo node ID suffixes).

**Verification:** Zero test failures. Rollback gate confirmed.

---

## 4. Migration Order Rationale

The step ordering follows a strict dependency chain:

| Step | Dependency rationale |
|:---|:---|
| 13.1 (Gather primitive) | Foundation — no dependencies, provides the `ModuleChunkGather` type consumed by the plan builder. |
| 13.2–13.3 (Kernel contracts) | Must precede plan builder changes because the plan builder imports contracts and passes scalars matching the contract's parameter list. |
| 13.4–13.5 (Plan builder) | Consumes the new gather primitive (13.1) and amended contracts (13.2–13.3). Step 13.4 restructures Phases II and IV (reduction + normalization); Step 13.5 restructures Phase V (updates + barrier dissolution). Must precede renderer updates because the plan builder now emits `parameter_offset` and `total_parameter_count` in dispatch scalar dicts. |
| 13.6 (Buffer sizing) | Logically part of plan builder restructuring; split out for clarity. Depends on 13.4–13.5. |
| 13.7–13.12 (Kernel sources) | Independent of plan builder changes — can be done in parallel with 13.4–13.6. Grouped by backend for reviewer convenience. |
| 13.13–13.15 (Renderers) | Must follow both kernel source amendments (13.7–13.12) and plan builder amendments (13.4–13.6), because renderers bridge the two. |
| 13.16–13.17 (Authority docs) | Documentary — can be done at any point but logically follow implementation. |
| 13.18–13.20 (Tests) | Must follow all implementation steps they exercise. |
| 13.21 (Convergence) | Exercises the full pipeline end-to-end; must follow all implementation and renderer changes. |
| 13.22 (Rollback gate) | Terminal validation — depends on everything. |

**Recommended implementation order:** 13.1 → 13.2–13.3 → 13.7–13.12 (kernel sources, all backends) → 13.4–13.6 (plan builder) → 13.13–13.15 (renderers) → 13.16–13.17 (docs) → 13.18–13.20 (tests) → 13.21 (convergence) → 13.22 (gate).

Kernel source amendments (13.7–13.12) can proceed in parallel with plan builder restructuring (13.4–13.6) because they are independent codebases. The renderers (13.13–13.15) are the convergence point.

---

## 5. Degeneration Verification

The `num_module_chunks == 1` degeneration is a critical correctness property. The following table maps each changed component to its degeneration behavior:

| Component | `num_module_chunks == 1` behavior | Equivalent to pre-fix? |
|:---|:---|:---|
| `ModuleChunkGather(scheme, 0, kind)` | Returns all tile offsets (= `TiledGather` output) | ✓ |
| Plan builder Phase II loop | Executes once; produces one `ReductionTreeNode` per param group | ✓ (node IDs gain `_m0` suffix) |
| Plan builder Phase IV normalize deps | Single normalize per group; each depends on its sole reducer | Functionally equivalent (narrower dependency sets but same effective ordering since all reducers complete before any normalize in both pre-fix and post-fix DAGs) |
| Plan builder Phase V loop | Executes once; `parameter_offset = 0`, `parameter_count = full model` | ✓ (node IDs gain `_m0` suffix) |
| Node 22 barrier | Absent; `adam_update_*_m0` depends on `normalize_*_m0` | Functionally equivalent (same serialization via direct dependency) |
| `adam_update` kernel | `parameter_offset = 0`, `total_parameter_count = parameter_count`; access pattern `buf[0 + i] = buf[i]` | ✓ (identical codegen) |
| `clamp_temperatures` kernel | `parameter_offset = 0`, `total_parameter_count = parameter_count`; access pattern `temps[0 + i] = temps[i]` | ✓ (identical codegen) |
| Optimizer state buffers | Full model size = chunk size (only one chunk) | ✓ |

The only visible difference is node ID suffixes (`_m0`). Downstream consumers (renderers, tests) must tolerate this suffix in the single-chunk case.

---

## 6. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|:---|:---|:---|:---|:---|
| R1 | `TilingScheme` requires additional fields for `elements_per_tile()`, potentially breaking existing construction sites | Medium | Medium | Add fields with defaults or factory method. Audit all `TilingScheme` construction sites (plan builder, tests) and update. |
| R2 | Node ID suffix change (`_m0`) breaks downstream test assertions or renderer lookup logic | High | Low | Systematic find-and-replace in tests. Renderer dispatches by contract name, not node ID — verify. |
| R3 | Topological order construction becomes complex with dynamic per-chunk nodes | Medium | High | Build topological order programmatically per the algorithm in Step 13.5. Add a validation pass to Tier 1 tests that verifies every node in `topological_order` appears after all nodes in its `depends_on` set. Gate the rollback on this check. An invalid order could silently produce wrong results if a node executes before its dependency. |
| R4 | Push constant struct alignment changes in Vulkan break pipeline layout | Low | High | Review `VkPushConstantRange` sizing. Add `uint parameter_offset` and `uint total_parameter_count` fields at consistent offsets. Validate with `vkCreatePipelineLayout`. |
| R5 | Existing checkpoints with per-chunk optimizer state buffers become incompatible | Low | Low | Out of scope per §1 — existing checkpoints saved with `num_module_chunks == 1` are already full-model-sized. Document the incompatibility for any hypothetical multi-chunk checkpoints. |
| R6 | The `clamp_temperatures` rename (`total_modules_count` → `parameter_count`) cascades through more sites than anticipated | Medium | Low | `grep -r total_modules_count` across all backends and tests to find all references before starting. |
| R7 | Per-module-chunk intermediate buffer count increases memory pressure | Low | Medium | These buffers are `BATCH_INTERMEDIATE` — the allocator can reuse them across chunks if the plan builder sequences chunks rather than parallelizing them. The backend retains scheduling freedom per CONCEPT.md §6. |
| R8 | Residual false serialization from shared `norm_deps` set if Phase IV dependency decoupling (Step 13.4) is incomplete | Medium | Medium | The monolithic `norm_deps` set must be fully eliminated — each normalize dispatch must depend only on its own reduction tree. Tier 1 tests should assert that `normalize_gradients_shared` has no dependency on any `reduce_mod_*` node, and vice versa. |
