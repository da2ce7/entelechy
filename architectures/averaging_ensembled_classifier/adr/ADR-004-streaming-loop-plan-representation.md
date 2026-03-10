# ADR-004: Streaming Loop Plan Representation

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002, ADR-003  
**Blocks:** ADR-009, ADR-012

---

## Context

ADR-002 establishes `StreamingLoopNode` as one of the five canonical plan node types, subject to the **Complexity Ceiling Constraint**: the `StreamingLoopNode.body` is a flat sequence of `KernelDispatchNode`s only — it may not contain another `StreamingLoopNode`, a `ReductionTreeNode`, a `BarrierNode`, or a `RetrievalNode`. ADR-002 defers the concrete representation of the loop's internal structure to this ADR.

The streaming loop pattern appears in two distinct contexts in the CONCEPT.md DAG:

| Instance                     | CONCEPT.md Model | Nodes in Body                   | Iteration Dimension         | Per-Chunk Data Path                                                         |
| :--------------------------- | :--------------- | :------------------------------ | :-------------------------- | :-------------------------------------------------------------------------- |
| Phase III shared backprop    | Model B          | 17 → 18 → 19                   | Batch (linear chunks)       | Recompute `hidden_i` slice → `backprop_shared_weights` → `backprop_shared_biases` → `clip_shared_gradients_chunk` |
| Phase I recompute path       | Model A          | 4 → 8, 9, 10 → 11              | Grid tiles (module × class) | Recompute `hidden_i` → compute per-tile raw grads → clip into collection    |

Both involve a host-side loop that iterates over chunks of a larger problem, dispatching a per-chunk kernel sequence. The backends handle this loop radically differently:

- **OpenCL:** Host Python loop; per-chunk `clEnqueueNDRange` calls with event chains.
- **Vulkan:** Loop at *recording time*; per-chunk `vkCmdPushConstants` + `vkCmdDispatch` + `vkCmdPipelineBarrier`, baked into the command buffer.
- **CPU:** Host loop; per-chunk `pool_dispatch_and_wait`.

### What varies per chunk

Examining the existing OpenCL implementation (`build_shared_backprop_subgraph` in `graph_recipes.py`), the per-chunk parameter variation for Phase III is:

| Parameter                         | Variation Pattern                  | Derivation                                  |
| :-------------------------------- | :--------------------------------- | :------------------------------------------ |
| `batch_chunk_offset`              | `i × chunk_size`                   | Arithmetic progression (base=0, stride=chunk_size) |
| `batch_chunk_count`               | `min(chunk_size, total - offset)`  | Derivable from chunk_size, total, and index |
| `dest_weights_write_offset`       | `i × elements_per_sw_partial`      | Arithmetic progression                      |
| `dest_biases_write_offset`        | `i × elements_per_sb_partial`      | Arithmetic progression                      |
| `clipping_threshold_global`       | Constant                           | Same for all chunks                         |

For the Model A recompute path (`build_streaming_module_grad_path`), each iteration operates on a different `WorkTile` from the `TilingScheme`, providing tile-specific offsets and counts. The per-tile parameters are computed from the tile's position in the grid — not arbitrary values, but deterministic functions of the tile index.

### The design tension

The same tension identified in ADR-003 applies: how much per-chunk data to pre-compute in the Policy tier versus how much to leave derivable by the renderer at the Orchestration tier. The threshold schedule analogy from ADR-003 provides the guiding principle:

> If per-chunk values are non-trivial (policy-dependent), pre-compute them; if they are trivially derivable from the chunk index and a stride, leave them to the renderer.

For `StreamingLoopNode`, every per-chunk parameter delta examined in the existing code is either constant or a simple arithmetic progression derivable from the chunk index and a stride. No per-chunk value requires a policy computation, a non-trivial offset list, or a placement-dependent scatter pattern — unlike the `ReductionTreePlan`'s initial offset list, which *is* non-trivial. This observation strongly favors a stride-based parametric specification.

A secondary tension: the Model A recompute path iterates over a 2D grid (module × class chunks) rather than a 1D batch dimension. The loop representation must accommodate both linear and grid-based iteration without introducing a second node type — per ADR-002's Complexity Ceiling, `StreamingLoopNode` is the sole loop construct.

---

## Decision Drivers

1. **ADR-001 (Three-tier jurisdictional model — Policy tier):** The Policy tier computes non-trivial shared values; the Orchestration tier handles dispatch mechanics. For streaming loops, the chunk count, body sequence, and per-chunk parameter strides are Policy-tier outputs. Per-chunk dispatch (event chains, pipeline barriers, synchronization) is an Orchestration-tier concern.

2. **ADR-002 (Complexity Ceiling Constraint):** The body is a flat sequence of `KernelDispatchNode`s. No nested loops, no reduction trees, no barriers within the body. This constraint is the reason a simple body representation suffices.

3. **ADR-003 (Parametric header precedent):** ADR-003 establishes the representation pattern: carry the Policy tier's pre-computed output as pure data in a frozen dataclass; leave trivially derivable Orchestration-tier data to the renderer. The `ReductionTreePlan` carries non-trivial thresholds but omits contiguous intermediate offset lists. The `StreamingLoopPlan` should follow the same boundary: carry the iteration structure and stride specifications; omit per-chunk instantiated parameter lists.

4. **CONCEPT.md §1 (Architectural Elegance Feedback):** The streaming loop is a first-class architectural primitive. Its plan representation must be a formal vocabulary element, not an ad-hoc expansion.

5. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability):** The plan must carry sufficient data to validate the streaming loop at plan-construction time: chunk count > 0, body is non-empty, strides are non-negative, and the final chunk's offset + count does not exceed the source dimension.

6. **Vulkan command buffer structure preservation:** Option B (pre-expanded flat DAG) was eliminated by the ADR_PLAN analysis: it inflates plan size linearly with chunk count, prevents Vulkan from recognizing loop structure, and loses the semantic signal that N sequences are structurally identical. The chosen representation must preserve the loop as a single semantic unit that Vulkan can record as a contiguous command buffer segment.

---

## Options Considered

### Option A: Pre-materialized per-chunk descriptor array

The `StreamingLoopPlan` contains an explicit array of `ChunkDescriptor` frozen dataclasses — one per chunk. Each descriptor carries the fully instantiated scalar parameters for that chunk's body invocation.

```python
@dataclass(frozen=True)
class ChunkDescriptor:
    chunk_index: int
    scalar_overrides: Dict[str, int]
    # e.g., {"batch_chunk_offset": 64, "batch_chunk_count": 32,
    #        "dest_weights_write_offset": 2048, "dest_biases_write_offset": 16}

@dataclass(frozen=True)
class StreamingLoopPlan:
    chunk_count: int
    chunks: Tuple[ChunkDescriptor, ...]
    body: Tuple[KernelDispatchNodeTemplate, ...]
    scratch_buffers: Tuple[ScratchBufferSpec, ...]
```

**Advantages:**
- Maximum plan completeness. The renderer iterates over descriptors with zero derivation.
- Every per-chunk parameter is inspectable and validatable at plan-construction time.

**Disadvantages:**
- **Inflates plan size linearly with chunk count.** For a batch of 1024 items with chunk size 32, the plan carries 32 `ChunkDescriptor`s, each containing redundant data that is a trivial function of the chunk index. The `ReductionTreePlan` avoided this for the same reason — contiguous intermediate offset lists were not materialized because they carry zero information beyond the stage's input count.
- **Defeats Vulkan's loop recognition.** A Vulkan renderer scanning for loop structure must reconstruct it from the array of descriptors by detecting that they form an arithmetic progression. This inverts the natural information flow — the shared layer *knows* this is a loop, destroys that knowledge by expanding it, and the renderer must *recover* it.
- **Carries redundant data.** Every per-chunk parameter delta in the existing code is derivable from `base + index × stride`. Pre-materializing these values provides zero information gain over the stride specification, analogous to ADR-003's elimination of contiguous intermediate offset lists.
- **Cannot represent the Model A grid iteration natively.** Grid iteration (module × class chunks) is a 2D pattern. Flattening it into a 1D descriptor array loses the grid structure that the TilingScheme expresses. The renderer must reconstruct 2D indexing from the flat array.

### Option B: Stride-based parametric specification

The `StreamingLoopPlan` carries the iteration structure (chunk count, iteration dimension) and a stride table describing how each parameterized scalar changes per iteration. The body is a tuple of `KernelDispatchNode` templates with parameterized bindings. The renderer instantiates per-chunk parameters at render time by applying the strides.

```python
@dataclass(frozen=True)
class IterationDimension:
    """Describes the dimension over which the loop iterates."""
    total_extent: int           # Total size of the iterated dimension
    chunk_count: int            # Number of chunks
    chunk_size: int             # Items per chunk (uniform; last chunk may be smaller)

@dataclass(frozen=True)
class ParameterStride:
    """Describes how a single scalar parameter varies across iterations."""
    param_name: str             # The parameter's name in the body's KernelDispatchNodes
    base: int                   # Value at chunk index 0
    stride: int                 # Additive delta per chunk index
    # Per-chunk value: base + chunk_index * stride

@dataclass(frozen=True)
class ScratchBufferSpec:
    """Declares a renderer-internal scratch buffer used within the loop body."""
    logical_name: str           # Name referenced by body nodes' buffer_bindings
    size_bytes: int             # Required size per chunk iteration
    shape: Tuple[int, ...]      # Logical tensor shape
    # The renderer allocates this buffer once and reuses it across iterations.

@dataclass(frozen=True)
class StreamingLoopPlan:
    """
    The internal structure of a StreamingLoopNode.

    Computed by the shared orchestration layer. Consumed by backend renderers.
    Pure data — no methods, no backend types, no executable objects.
    """
    iteration: IterationDimension
    """The dimension decomposition for the loop."""

    body: Tuple[str, ...]
    """Ordered sequence of KernelDispatchNode node_ids that constitute the
    loop body. These reference nodes defined as children of the
    StreamingLoopNode. Each node is a template — its parameterized scalars
    are instantiated per iteration by the renderer using the stride table."""

    parameter_strides: Tuple[ParameterStride, ...]
    """Stride specifications for all parameters that vary across iterations.
    Parameters not listed here are constant across all iterations and retain
    their values from the body's KernelDispatchNode definitions."""

    scratch_buffers: Tuple[ScratchBufferSpec, ...]
    """Scratch buffers allocated by the renderer for intra-iteration use.
    These are not plan-level buffers (ADR-009) — they are renderer-internal
    (ADR-003's two-tier buffer scope). Declared here so the renderer knows
    their required sizes and shapes."""

    constant_scalars: Dict[str, float]
    """Scalar parameters that are constant across all iterations but specific
    to the streaming loop context (e.g., clipping_threshold_global). These
    augment the body nodes' scalar_params."""
```

**Advantages:**
- **Compact and semantically complete.** The plan carries the iteration structure and stride table — exactly the non-trivial shared computation. Plan size is O(1) in chunk count, not O(N).
- **Preserves loop semantics.** The renderer receives a single `StreamingLoopNode` with explicit iteration structure. Vulkan records the loop body once and emits per-chunk `vkCmdPushConstants` + `vkCmdDispatch` calls with stride-derived values. OpenCL loops in Python. CPU loops in C. Each backend preserves its native dispatch model.
- **Full plan-time validation.** The shared layer validates: `chunk_count > 0`, `chunk_size × chunk_count >= total_extent`, strides are consistent with buffer shapes, scratch buffer sizes are sufficient.
- **Naturally accommodates both iteration patterns.** Linear batch iteration uses a single `IterationDimension` with `total_extent = batch_size`. Grid iteration uses `total_extent = num_tiles` with strides derived from the `TilingScheme`'s 2D→1D flattening. The stride table maps the flat chunk index to the appropriate per-tile parameters.
- **Follows ADR-003's boundary.** Pre-compute what is non-trivial (iteration structure, strides, scratch buffer specs); leave what is trivially derivable (per-chunk instantiated values) to the renderer.

**Disadvantages:**
- The renderer must perform per-chunk arithmetic (`base + index × stride`) for each parameterized scalar. This is trivial — a multiply-add per parameter per iteration — but it is a computation.
- The stride model assumes parameters vary linearly with chunk index. Non-linear variation (e.g., exponentially growing chunk sizes) would require extending the model. Per CONCEPT.md §1, such an extension would be formalized as a new stride type, not worked around.
- The `last_chunk_items_count` (which may be smaller than `chunk_size`) is not directly in the stride table — it is derivable from `total_extent`, `chunk_count`, and `chunk_size`. The renderer must handle this edge case.

### Option C: Callback-based chunk parameter factory

The `StreamingLoopPlan` carries the iteration count and a callable factory that produces per-chunk parameter dictionaries.

```python
@dataclass(frozen=True)
class StreamingLoopPlan:
    chunk_count: int
    body: Tuple[str, ...]
    chunk_param_factory: Callable[[int], Dict[str, int]]
    # chunk_param_factory(chunk_index) -> {"batch_chunk_offset": ..., ...}
```

**Advantages:**
- Maximum flexibility. Any per-chunk variation pattern — linear, quadratic, lookup-table — is expressible.
- The factory encapsulates the shared layer's chunk-parameter logic.

**Disadvantages:**
- **Embeds a callable in the plan.** This is the same boundary violation that eliminated ADR-003 Option B: a callable with methods crosses the plan-as-data-structure boundary (ADR-001). The plan becomes an API surface, not a data artifact.
- **Validation gap.** The shared layer cannot validate the factory's outputs at plan-construction time without calling it for every chunk index — defeating the purpose of a compact specification.
- **Serialization barrier.** A closure or partial function cannot be straightforwardly serialized, logged, or compared across plan instances. The plan loses its inspectability.
- **Forces renderer coupling.** Each renderer must call the factory, coupling backend code to the shared layer's callable interface rather than consuming pure data.

---

## Analysis

### Eliminating Option A

Option A over-specifies the plan by materializing per-chunk parameter values that are trivially derivable from the chunk index and a stride. The analysis parallels ADR-003's elimination of its Option A:

- ADR-003 rejected carrying per-stage contiguous offset lists because they carry zero information beyond the stage's input count (always `[0, 1, ..., N-1]`).
- Here, carrying per-chunk `batch_chunk_offset = [0, 32, 64, ...]` carries zero information beyond "base=0, stride=32."

Furthermore, Option A actively destroys semantic structure. The Vulkan backend's single most important optimization for streaming loops is recording the body once and varying only push constants per iteration. A pre-materialized descriptor array obscures this loop structure, forcing the renderer to reconstruct it — an inversion of the natural information flow.

Plan size under Option A grows as O(chunk_count × parameters_per_chunk). For the Phase III streaming loop with 4 varying parameters and 64 batch chunks, this is 256 redundant integers. For the Model A recompute path with e.g. 128 tiles and 8 varying parameters, this is 1024 redundant integers. The stride-based specification represents the same information in O(parameters) space — 4 or 8 `ParameterStride` entries, regardless of chunk count.

### Eliminating Option C

Option C embeds a callable factory in the plan — the exact boundary violation that eliminated ADR-003 Option B. The arguments are identical:

- ADR-003 rejected embedding a live `StabilizationPolicy` object because it crosses the plan-as-data-structure boundary (ADR-001: "the shared orchestration layer produces a backend-neutral execution plan expressed as a **data structure**").
- A callable factory is an executable component with implicit contract (input: chunk index; output: parameter dict). Each renderer depends on this callable's interface, coupling backend code to the shared layer's implementation.

The validation argument is equally decisive: Option B's stride table is fully inspectable at plan-construction time — the shared layer can verify that `base + (chunk_count - 1) × stride` does not produce an out-of-bounds offset. Option C's factory requires exhaustive probing to achieve the same guarantee.

### Choosing Option B

Option B separates shared from backend concerns at the Policy/Orchestration boundary — the same boundary established in ADR-003:

**Policy tier pre-computes (carried in the plan):**
- Chunk count and chunk size — resolved by the host memory assessment.
- The body's kernel sequence — which kernels execute per iteration.
- The stride table — how each parameterized scalar varies with chunk index. These strides are derived from `MemoryLayout` shapes, `TilingScheme` geometry, and precision parameters.
- Scratch buffer specifications — size and shape of per-iteration transient buffers (analogous to `gsw_scratch_ref` and `gsb_scratch_ref` in the current OpenCL code).
- Constant scalars — loop-wide invariant parameters like `clipping_threshold_global`.

**Orchestration tier derives (at render time):**
- Per-chunk instantiated parameter values — `base + chunk_index × stride` for each `ParameterStride`.
- Last-chunk item count — `min(chunk_size, total_extent - chunk_index × chunk_size)`.
- Scratch buffer allocation — `cl.Buffer`, mapped staging, or stack allocation.
- Per-chunk synchronization — event chains, pipeline barriers, or implicit sequencing.
- Upload mechanism for per-chunk scalars — `clSetKernelArg`, `vkCmdPushConstants`, or C function arguments.

This boundary satisfies all decision drivers:

1. **ADR-001:** Non-trivial shared computation (iteration structure, strides) is computed once; dispatch mechanics are backend-specific.
2. **ADR-002:** The body remains a flat sequence of `KernelDispatchNode` references — no nested constructs.
3. **ADR-003:** The pattern is consistent — parametric header with pre-computed shared data, trivial derivation left to the renderer.
4. **Vulkan fidelity:** The renderer sees a single loop with stride-parameterized iterations and can record the body as a contiguous command buffer segment.
5. **Plan-time validation:** The stride table and iteration dimension are fully inspectable at construction time.

### The iteration dimension generalization

The `IterationDimension` dataclass abstracts over the two iteration patterns:

- **Phase III (linear batch):** `total_extent = batch_size`, `chunk_count = shared_backprop_stream_chunks`, `chunk_size = ceil(batch_size / chunk_count)`. Strides are simple products of `chunk_size` and per-element counts.

- **Model A (grid tiles):** `total_extent = grid.total_tiles`, `chunk_count = grid.total_tiles`, `chunk_size = 1`. Each "chunk" is one tile. Strides encode the flattened 2D→1D mapping: e.g., `module_chunk_offset` has `base=0, stride=modules_per_chunk`; `class_chunk_offset` follows the grid's row-major pattern.

For Model A's 2D grid iteration, the stride model requires a slight extension: some parameters depend on the 2D position `(m_idx, c_idx)`, not just the flat index. Two approaches handle this:

**(a) Flatten the grid.** The plan builder flattens the 2D grid into a 1D iteration with `chunk_count = num_module_chunks × num_class_chunks`. Parameters that vary along the module dimension have stride `s_m` applied every `num_class_chunks` iterations; parameters that vary along the class dimension have stride `s_c` applied cyclically. This requires a `modular_stride` extension:

```python
@dataclass(frozen=True)
class ParameterStride:
    param_name: str
    base: int
    stride: int
    period: Optional[int] = None   # If set: value = base + (chunk_index % period) * stride
    outer_stride: Optional[int] = None  # If set: value += (chunk_index // period) * outer_stride
```

**(b) Use `chunk_size = 1` with pre-computed strides.** Since each iteration maps to exactly one tile, the plan builder can pre-compute a compact per-tile parameter table. This is Option A applied only to the tile-parametric values — but since `chunk_size = 1` and each tile has unique geometry (potentially non-uniform module/class counts for the last tile in each dimension), the per-tile parameters may not follow a simple stride pattern when tile sizes are non-uniform.

The chosen approach is **(a) with the period extension**, because it keeps the representation parametric and O(1) in chunk count. The `period` and `outer_stride` fields handle the 2D→1D decomposition for uniformly-sized tiles. For the last-row and last-column edge cases (where tile sizes differ), the renderer derives the clamped values from `iteration.total_extent` and the grid dimensions — analogous to deriving the last chunk's `items_in_chunk`.

If future workloads introduce non-uniform tile geometries that cannot be expressed via the stride/period model, CONCEPT.md §1 (Architectural Elegance Feedback) applies: suspend the workaround, formalize the non-uniform iteration as a new `IterationDimension` variant, and extend the plan vocabulary through a revised ADR.

### Scratch buffer scope

The `ScratchBufferSpec` entries in the `StreamingLoopPlan` follow ADR-003's two-tier buffer scope distinction:

- **Plan-level buffers** (the collection buffers `clipped_partial_grad_shared_weights`, `clipped_partial_grad_shared_biases`) are declared in the `StreamingLoopNode`'s `buffer_bindings` and have lifetimes governed by the DAG topology — they are ADR-009's concern.
- **Renderer-internal buffers** (the per-iteration scratch buffers `gsw_scratch_ref`, `gsb_scratch_ref` in the current code) are declared in the `ScratchBufferSpec` entries. They are allocated by the renderer, reused across iterations, and never appear in the plan's DAG-level buffer namespace. Their specification in the plan serves only to inform the renderer of the required size and shape — the allocation mechanism (transient `cl.Buffer`, suballocated `VkDeviceMemory`, stack array) is an Orchestration-tier concern.

This mirrors `ReductionTreeNode`'s intermediate stage buffers: ADR-003 explicitly places them in the renderer-internal tier the renderer derives their sizes from `elements_per_partial` and per-stage output count. Here, the plan builder derives scratch buffer sizes from `MemoryLayout` shapes, and the renderer allocates them using its native mechanism.

### The edge case: chunk_count = 1

When `chunk_count = 1`, the loop degenerates to a single iteration — the body executes once with `chunk_index = 0`. All per-chunk parameters take their `base` values. The renderer may optimize by skipping loop overhead, but this is a rendering concern. The plan representation is valid and self-consistent for `chunk_count = 1`.

### The edge case: total_extent not evenly divisible

When `total_extent` is not evenly divisible by `chunk_count`, the last chunk processes fewer items: `items_in_last_chunk = total_extent - (chunk_count - 1) × chunk_size`. The renderer derives this from the `IterationDimension` fields. Strides remain valid — they describe parameter *offsets*, not item counts. The per-chunk item count is a rendering concern derived from the iteration dimension, not a stride.

---

## Decision

**Option B: Stride-based parametric specification.**

The `StreamingLoopPlan` is a frozen dataclass embedded as the internal data structure of a `StreamingLoopNode` (ADR-002). It carries the iteration dimension, the body's kernel sequence (as references to child `KernelDispatchNode` IDs), a stride table for per-chunk parameter variation, scratch buffer specifications, and constant scalars. The renderer derives per-chunk parameter values and manages per-iteration dispatch, synchronization, and scratch buffer allocation at render time.

### Data structure

```python
@dataclass(frozen=True)
class IterationDimension:
    """
    Describes the dimension decomposition for a streaming loop.

    Pure data. Computed by the shared layer from host memory assessment
    (batch chunking) or TilingScheme (grid iteration).
    """
    total_extent: int
    """Total size of the iterated dimension (e.g., batch_size or total_tiles).
    Invariant: total_extent >= 1."""

    chunk_count: int
    """Number of loop iterations.
    Invariant: chunk_count >= 1."""

    chunk_size: int
    """Uniform items per chunk. The last chunk may process fewer items:
    min(chunk_size, total_extent - chunk_index * chunk_size).
    Invariant: chunk_size >= 1.
    Invariant: (chunk_count - 1) * chunk_size < total_extent <= chunk_count * chunk_size."""


@dataclass(frozen=True)
class ParameterStride:
    """
    Describes how a single scalar parameter varies across loop iterations.

    The per-chunk value is computed as:
        value = base + (chunk_index % period) * stride
                     + (chunk_index // period) * outer_stride

    For simple linear progression (the common case):
        period = chunk_count, outer_stride = 0
        => value = base + chunk_index * stride

    For 2D grid iteration flattened to 1D (inner dimension = class, outer = module):
        period = num_class_chunks (inner loop length)
        stride = class_chunk_stride
        outer_stride = module_chunk_stride
    """
    param_name: str
    """The parameter's name as it appears in the body's KernelDispatchNode
    scalar_params dictionaries."""

    base: int
    """Value at chunk_index = 0."""

    stride: int
    """Additive delta per unit of (chunk_index % period)."""

    period: int
    """The period of the inner cycle. For simple linear iteration,
    set to chunk_count (or any value >= chunk_count).
    Invariant: period >= 1."""

    outer_stride: int = 0
    """Additive delta per complete period cycle (chunk_index // period).
    For simple linear iteration, this is 0."""


@dataclass(frozen=True)
class ScratchBufferSpec:
    """
    Declares a renderer-internal scratch buffer used within the loop body.

    This is a renderer-internal buffer (ADR-003 two-tier scope) — it never
    appears in the plan's DAG-level buffer namespace. The plan declares its
    required size and shape so the renderer can allocate appropriately.
    The renderer allocates once and reuses across iterations.
    """
    logical_name: str
    """Name referenced by body nodes' buffer_bindings for this scratch slot."""

    size_bytes: int
    """Required buffer size in bytes for a single iteration's use."""

    shape: Tuple[int, ...]
    """Logical tensor shape of the scratch buffer's contents."""


@dataclass(frozen=True)
class StreamingLoopPlan:
    """
    The internal structure of a StreamingLoopNode.

    Computed by the shared orchestration layer. Consumed by backend renderers.
    Pure data — no methods, no backend types, no executable objects.
    """
    iteration: IterationDimension
    """The dimension decomposition for the loop."""

    body: Tuple[str, ...]
    """Ordered sequence of KernelDispatchNode node_ids that constitute the
    loop body. These reference child nodes defined within the StreamingLoopNode.
    Each node is a template — its parameterized scalars are instantiated
    per iteration by the renderer using the stride table.
    Invariant: len(body) >= 1.
    Invariant: body contains only KernelDispatchNode references
    (Complexity Ceiling, ADR-002)."""

    parameter_strides: Tuple[ParameterStride, ...]
    """Stride specifications for all parameters that vary across iterations.
    Parameters not listed here are constant across all iterations and retain
    their values from the body KernelDispatchNode definitions.
    Each ParameterStride.param_name must reference a scalar_param that exists
    in at least one body node."""

    scratch_buffers: Tuple[ScratchBufferSpec, ...]
    """Scratch buffers allocated by the renderer for intra-iteration use.
    These are renderer-internal buffers (ADR-003 two-tier buffer scope).
    The plan declares their specs; the renderer owns their allocation and
    lifecycle. May be empty if the body needs no scratch space."""

    constant_scalars: Tuple[Tuple[str, float], ...]
    """Scalar parameters that are constant across all iterations but specific
    to the streaming loop context. These augment (or override) the body nodes'
    scalar_params for every iteration. Represented as (name, value) pairs.
    Example: ('clipping_threshold_global', 0.5)."""
```

### Plan construction contract

The plan builder constructs a `StreamingLoopPlan` by:

1. **Resolving the iteration dimension.** For Phase III: the host memory assessment determines `shared_backprop_stream_chunks` (the chunk count), and `chunk_size = ceil(batch_size / chunk_count)`. For Model A: `chunk_count = grid.total_tiles`, `chunk_size = 1`, `total_extent = grid.total_tiles`.

2. **Defining the body.** The plan builder creates `KernelDispatchNode` templates for the loop body — one per kernel in the per-chunk sequence. For Phase III: three nodes corresponding to Nodes 17, 18, 19. For Model A: nodes corresponding to Nodes 4, 8, 9, 10, 11. These nodes carry their full `buffer_bindings`, `KernelContract` references, and non-varying `scalar_params`. The varying parameters are set to placeholder values (e.g., their `base` values) — the renderer overrides them per iteration using the stride table.

3. **Computing the stride table.** For each scalar parameter that varies across iterations, the plan builder computes its `ParameterStride`:
   - Phase III `batch_chunk_offset`: `base=0, stride=chunk_size, period=chunk_count`.
   - Phase III `dest_weights_write_offset`: `base=0, stride=elements_per_sw_partial, period=chunk_count`.
   - Phase III `dest_biases_write_offset`: `base=0, stride=elements_per_sb_partial, period=chunk_count`.
   - Model A `module_chunk_offset`: `base=0, stride=modules_per_chunk, period=num_class_chunks, outer_stride=modules_per_chunk`.
   - Model A `class_chunk_offset`: `base=0, stride=classes_per_chunk, period=num_class_chunks, outer_stride=0`.
   - Model A `flat_tile_index`: `base=0, stride=1, period=chunk_count` (simple linear).

4. **Declaring scratch buffers.** The plan builder computes scratch buffer sizes from `MemoryLayout` shapes: e.g., Phase III needs `gsw_scratch` of size `prod(shared_weights_shape) × scalar_bytes` and `gsb_scratch` of size `prod(shared_biases_shape) × scalar_bytes`. Model A needs `h_scratch`, `h_mask_scratch`, and per-gradient-type scratch buffers.

5. **Setting constant scalars.** Loop-wide constants like `clipping_threshold_global` and `epsilon` are recorded in `constant_scalars`.

6. **Validating invariants.** At construction time:
   - `chunk_count >= 1`.
   - `chunk_size >= 1`.
   - `(chunk_count - 1) * chunk_size < total_extent <= chunk_count * chunk_size`.
   - `len(body) >= 1`.
   - Every `ParameterStride.param_name` references a `scalar_param` that exists in at least one body node.
   - Every `ParameterStride.period >= 1`.
   - For each stride: `base + (chunk_count - 1) % period * stride + (chunk_count - 1) // period * outer_stride >= 0` (no negative parameter values for offset-like parameters).
   - Every `ScratchBufferSpec.size_bytes > 0`.

### Backend rendering contract

The renderer interprets a `StreamingLoopPlan` as the Orchestration tier (ADR-001 §Three-tier jurisdictional model). For each iteration `i` where `0 <= i < chunk_count`:

1. **Compute per-chunk parameters.** For each `ParameterStride` in `parameter_strides`:
   ```
   value = base + (i % period) * stride + (i // period) * outer_stride
   ```

2. **Compute per-chunk item count.** `items = min(chunk_size, total_extent - i * chunk_size)`. If `items <= 0`, skip this iteration (defensive; should not occur if invariants hold).

3. **Override body node scalars.** For each `KernelDispatchNode` in `body`, override `scalar_params[param_name]` with the computed per-chunk value where applicable, and merge in `constant_scalars`.

4. **Dispatch the body sequence.** Execute each kernel in `body` order, using the renderer's native dispatch mechanism.

5. **Insert intra-iteration synchronization.** The body kernels within a single iteration have implicit sequential dependency (the body is ordered). The renderer inserts its native synchronization:
   - **OpenCL:** Each kernel launch returns a `cl.Event`; the next kernel's `wait_for` includes it.
   - **Vulkan:** `vkCmdPipelineBarrier` between body kernel dispatches within one iteration.
   - **CPU:** Implicit — `pool_dispatch_and_wait` is synchronous.

6. **Insert inter-iteration synchronization.** Between iterations, the renderer ensures the previous iteration's final body kernel has completed before the next iteration begins (required because scratch buffers are reused). The mechanism is the same as intra-iteration synchronization.

7. **Manage scratch buffers.** Allocate scratch buffers declared in `scratch_buffers` once before the loop. Bind them to the body nodes' `buffer_bindings` using the `logical_name` key. Reuse across iterations. Deallocate after the loop completes.

### Relationship to `StreamingLoopNode`

The `StreamingLoopNode` (ADR-002) embeds a `StreamingLoopPlan` alongside its standard node fields:

```python
@dataclass(frozen=True)
class StreamingLoopNode:
    node_id: str
    depends_on: FrozenSet[str]
    loop_plan: StreamingLoopPlan
    body_nodes: Tuple[KernelDispatchNode, ...]
    """The KernelDispatchNode templates referenced by loop_plan.body.
    These are child nodes owned by the StreamingLoopNode — they do not
    appear as independent nodes in the top-level DAG."""
    collection_buffers: Dict[str, str]
    """Mapping from logical collection buffer names to plan-level buffer IDs.
    These are the plan-level output buffers that the streaming loop populates
    across iterations (e.g., clipped_partial_grad_shared_weights)."""
```

The nesting mirrors `ReductionTreeNode` / `ReductionTreePlan` (ADR-003): the node owns its DAG position and dependencies; the embedded plan owns the iteration structure. The `body_nodes` tuple carries the full `KernelDispatchNode` definitions (with `KernelContract` references for plan-time validation); the `loop_plan.body` tuple carries only their `node_id` strings for compact iteration reference.

---

## Consequences

### Positive

- **Compact, O(1)-in-chunk-count representation.** The plan carries the iteration structure and stride table — a fixed number of `ParameterStride` entries regardless of how many chunks the loop executes. This contrasts with Option A's O(N) materialization, analogous to ADR-003's decision not to materialize contiguous intermediate offset lists.

- **Preserved loop semantics.** The `StreamingLoopNode` is a single semantic entity in the plan DAG. Vulkan records it as one contiguous command buffer segment. OpenCL and CPU render it as an explicit host loop. The renderer sees the loop structure directly — it does not reconstruct it from an expanded descriptor array.

- **Full plan-time validation.** All iteration invariants — chunk count consistency, stride bound checks, body non-emptiness, scratch buffer sizing — are verifiable at plan-construction time without any backend. This satisfies CONTRACT.md Article 1.4.

- **Backend rendering freedom.** Per-chunk dispatch, synchronization, and scratch buffer allocation are Orchestration-tier concerns. Each backend applies its native mechanism: OpenCL event chains, Vulkan pipeline barriers, CPU sequential calls. The plan does not prescribe them.

- **Consistent architectural pattern.** The parametric-header-with-stride-table pattern follows ADR-003's parametric-header-with-threshold-schedule pattern. The shared layer pre-computes what is non-trivial; the renderer derives what is trivial. The two-tier buffer scope is consistently applied.

- **Unified representation for both streaming models.** Both Phase III (linear batch) and Model A (grid tiles) are expressed through the same `StreamingLoopPlan` structure, with the `period` and `outer_stride` fields accommodating 2D→1D flattened iteration. No second node type is needed.

### Negative

- **Renderer must perform per-chunk arithmetic.** The multiply-add per parameter per iteration is trivial but non-zero. Option A would eliminate it at the cost of linear plan inflation and destroyed loop semantics. The trade-off favors Option B.

- **2D iteration encoding is indirect.** Grid-based iteration (Model A) is encoded via the `period`/`outer_stride` fields of `ParameterStride`, which is less immediately intuitive than a native 2D iteration construct. The representation is correct and complete, but a reader must understand the 2D→1D flattening convention. This is documented in the `ParameterStride` docstring and the plan construction contract.

- **Stride model assumes affine variation.** If a future workload introduces per-chunk parameters that vary non-linearly with the chunk index (e.g., geometrically growing chunk sizes for adaptive streaming), the stride model must be extended. Per CONCEPT.md §1 (Architectural Elegance Feedback), this extension would be a formal vocabulary revision, not an ad-hoc workaround.

- **Last-chunk edge case is renderer-derived.** The potentially smaller last chunk's item count is not explicit in the stride table — it is derived from `IterationDimension` fields. This is analogous to `ReductionTreePlan`'s per-stage input counts (derived from `num_partials`, `fan_in`, and stage index). The derivation is trivial: `min(chunk_size, total_extent - i * chunk_size)`.

### Migration implications

Per ADR-002's migration path:

1. **Define the data structures** (`IterationDimension`, `ParameterStride`, `ScratchBufferSpec`, `StreamingLoopPlan`) as frozen dataclasses in the shared layer (e.g., `execution_plan.py`). No existing code is modified.

2. **Add plan-construction logic** to the `ExecutionPlanBuilder`: for each streaming loop (Phase III shared backprop, Model A recompute path), compute the `IterationDimension`, build the body's `KernelDispatchNode` templates, derive the stride table from `MemoryLayout` shapes and `TilingScheme` geometry, and construct a `StreamingLoopPlan`.

3. **Write plan-level tests** (ADR-016, layer 1) that validate:
   - Iteration dimension consistency: `(chunk_count - 1) * chunk_size < total_extent`.
   - Stride bound checks: no per-chunk parameter value is negative or exceeds buffer dimensions.
   - Body non-emptiness and Complexity Ceiling compliance (only `KernelDispatchNode` references).
   - Scratch buffer size consistency with `MemoryLayout`-derived shapes.
   - Round-trip: for a known `ModelSpec` + `HardwareProfile`, verify that the stride table produces the same per-chunk parameters as the current `build_shared_backprop_subgraph` function.

4. **OpenCL renderer** (Phase 3 of ADR-001 migration) interprets `StreamingLoopPlan` by replacing the current `build_shared_backprop_subgraph` and `build_streaming_module_grad_path` functions in `graph_recipes.py`. The renderer iterates over `chunk_count`, applies strides, allocates scratch buffers as transient `cl.Buffer`s (inheriting `BufferManager.acquire_transient_buffer`'s current scheme), and dispatches with event chains.

5. **CPU and Vulkan renderers** (Phase 4 & 5) consume the same `StreamingLoopPlan` with their native dispatch models: Vulkan records per-chunk push constants into a command buffer segment; CPU calls per-chunk C functions sequentially.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; shared correctness requirement; three-tier jurisdictional model (Policy / Orchestration / Execution)
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `StreamingLoopNode` definition; Complexity Ceiling Constraint; body is flat sequence of `KernelDispatchNode`s only
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — parametric header precedent; two-tier buffer scope distinction; elimination of trivially-derivable per-stage data
- [CONCEPT.md](../CONCEPT.md) — §2 Model A (Accumulate via Recompute) and Model B (True Streaming); §5 Host Orchestrator & Execution Policies (chunk definition, activation lifecycle); §1 Architectural Elegance Feedback
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 3.2 Placement Contract (`linear_batch`, `grid_mod_cls` strategies)
- [ADR_PLAN.md](../ADR_PLAN.md) — ADR-004 problem statement and narrowing analysis
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — plan-level buffer lifetime annotations; two-tier buffer scope formalization
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — physical placement of `StreamingLoopPlan` in `execution_plan.py` (pending)
