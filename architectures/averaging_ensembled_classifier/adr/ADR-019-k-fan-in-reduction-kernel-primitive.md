# ADR-019: K-Fan-In Reduction Kernel Primitive

**Status:** ACCEPTED  
**Date:** 2026-03-29  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-003, ADR-007, ADR-013  
**Blocks:** —  
**Triggered by:** CPU vs OpenCL Kernel Audit — Finding 2 (Reduction Tree Clip Granularity)

---

## Context

The CPU vs OpenCL Kernel Audit (2026-03-29) identified a structural divergence in the multi-stage reduction tree implementation between the CPU and OpenCL backends. The divergence is masked when `num_stages == 1` (the common case for small tiling grids) but produces incorrect results when `num_stages > 1`. Investigation reveals three compounding defects, all rooted in a vocabulary gap in the kernel primitive set.

### The architectural specification

CONCEPT.md §2 specifies the Recursive Clip-Aggregation Engine as a `log_K(N)` reduction tree where `K` — the Reduction Batch Size — "defines the width of the parallel kernel front at each reduction stage." CONCEPT.md §3.3 specifies that the Quadratic Scaling Policy "set[s] an independent, optimal clipping threshold at each layer of the reduction tree." ADR-003 formalizes this as a `ReductionTreePlan` carrying `fan_in_K`, `num_stages`, and a pre-computed `threshold_schedule`.

The specification is unambiguous: each stage of the tree reduces groups of K partials into `ceil(N_stage / K)` intermediate nodes, and each intermediate node is independently clipped before being fed to the next stage.

### Defect A: The aggregate kernels are all-to-one reducers

Both `aggregate_register_reduce` and `aggregate_local_reduce` (`kernels/phase_2_learn_C_reduction.cl.c`) accept a `partial_offset_list_count` and iterate through the **entire** offset list, producing a **single** output vector of `partial_width` elements. Their `@kernel_contract` in `kernels.cl.h` specifies the destination as:

> *Tensor Shape: (src_scalar_NATURAL_partial_width)*

This is a single vector — not `ceil(N/K)` vectors. The kernel vocabulary contains no primitive capable of K-fan-in partial reduction. It can only perform all-to-one reduction.

### Defect B: The OpenCL renderer loop assumes K-fan-in outputs that don't exist

`_render_reduction_tree` in `src/backends/opencl/renderer.py` computes `output_N = ceil(current_N / K)` and constructs contiguous offset lists for subsequent stages. But the aggregate kernel dispatched at stage 0 already reduced ALL `num_partials` inputs to a single output. For `num_stages > 1`:

- Stage 0 passes all N offsets → kernel reduces all N to 1 result (not `ceil(N/K)` results)
- Stage 1+ constructs an offset list referencing `output_N` contiguous entries in the ping buffer, but only entry 0 contains valid data; the remainder is uninitialized

This is a **silent data corruption** path — subsequent stages accumulate uninitialized memory into the final result.

### Defect C: `clip_intermediate_grad` has wrong granularity for multi-node stages

Even if Defect A were resolved, `clip_intermediate_grad` (`kernels.cl.h`, Node 15b/20b) computes a single L2 norm across its entire `parameter_count`-element input buffer and applies a single scaling factor. Its contract states:

> *"It atomically computes an L2 norm over its entire input buffer and conditionally scales that buffer in-place."*

In a multi-node stage with `ceil(N/K)` output nodes, the renderer would need to clip each node's `partial_width` elements independently. The existing `clip_intermediate_grad` kernel treats the entire buffer as one vector — computing one L2 norm and applying one scale to all `ceil(N/K) × partial_width` elements jointly. This conflates the per-node gradient directions that the Quadratic Scaling Policy is designed to preserve independently.

The CPU's `task_reduce_and_clip_node` (`src/backends/cpu/kernel_sources/phase_2_learn_C_reduction.c`) correctly implements per-node semantics: each task processes one reduction node, computing its own L2 norm over `partial_width` elements and clipping independently.

### Why this is masked today

For `num_stages == 1`, the tree performs a single full reduction of all partials to one output, followed by one clip. There is only one output node, so per-node clip ≡ per-buffer clip ≡ the CPU behavior. The plan builder (`src/shared/plan_builder.py`) calls `policy.plan_uniform_reduction_tree(num_partials, hardware.max_reduce_fan_in)` — when `max_reduce_fan_in >= num_partials`, the result is always `(K=num_partials, num_stages=1)`, and the defect is unreachable.

Multi-stage trees appear when `num_partials > max_reduce_fan_in`, which occurs for large `num_module_chunks × num_class_chunks` products with moderate hardware `max_reduce_fan_in` values.

### Architectural Elegance Feedback

Per CONCEPT.md §1:

> *When emergent efficiency gains contradict current constraints:*
> 1. *Suspend implementation of the optimization*
> 2. *Formalize the pattern as a documented architectural primitive*
> 3. *Reify the optimization through revised contracts & DAG extensions*

The all-to-one aggregate kernel is not wrong — it is an incomplete vocabulary. The architecture's execution model specifies K-fan-in staged reduction as a core operation (CONCEPT.md §2), but the kernel set lacks a primitive that implements it. The `_render_reduction_tree` loop in the OpenCL renderer attempted to orchestrate K-fan-in behavior on top of an all-to-one primitive — precisely the kind of ad-hoc workaround that §1 prohibits.

This ADR formalizes the K-fan-in reduction-with-clip as a first-class kernel primitive and revises the rendering contract to consume it.

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** The vocabulary gap must be resolved by extending the formal kernel primitive set, not by renderer-level workarounds or plan-builder constraints.

2. **CONCEPT.md §2 (Recursive, Tiered Aggregation Engine).** The `log_K(N)` tree structure with K-fan-in at each stage is a core architectural concept. The kernel set must natively express it.

3. **CONCEPT.md §3 (Modular, "Dumb" Kernels).** Kernels are "simple, single-purpose modules." A fused reduce-and-clip kernel for one reduction node is a natural single-purpose unit — it performs exactly the sum-then-clip operation that CONCEPT.md §3.3 describes as the atomic building block of the stabilized reduction tree.

4. **ADR-003 (ReductionTreePlan — Backend Rendering Contract).** ADR-003 §Backend rendering contract specifies that the renderer "conditionally dispatch[es] the clip kernel" per stage. This assumes one clip dispatch per stage operating on one output vector. The rendering contract must be updated to reflect per-node clip semantics when `ceil(N/K) > 1`.

5. **ADR-013 (Kernel Source Strategy).** New kernels require specification in `kernels.cl.h` (algorithmic authority), implementation in each backend's kernel source directory, a `KernelContract` in `src/shared/kernel_contracts/`, and per-backend bindings per ADR-007.

6. **ADR-001 (Three-Tier Jurisdictional Model).** The decision of *whether* to fuse aggregate + clip into one kernel is a rendering concern (Orchestration tier) — the plan specifies *what* computation (sum K partials, then clip), not *how many kernel launches* implement it. However, the kernel vocabulary itself must be rich enough to express the operation.

7. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability).** The new kernel's contract must be fully validatable at plan-construction time, with calculability proofs for all buffer dimensions and preconditions for all scalar parameters.

---

## Options Considered

### Option A: Constrain the plan builder to `num_stages == 1` for GPU backends

Force `max_reduce_fan_in` to always exceed `num_partials` for GPU backends, ensuring single-stage trees. The existing all-to-one kernels are correct for this case.

**Advantages:**
- Zero kernel changes. Zero contract changes. Immediate fix.
- Eliminates the broken code path entirely.

**Disadvantages:**
- **Violates CONCEPT.md §1.** This is a constraint that masks incomplete vocabulary rather than extending the architecture. It converts a `log_K(N)` engine into an `O(N)` flat engine for GPU backends — directly contradicting CONCEPT.md §2's scalability claim.
- **Breaks architectural uniformity.** The CPU backend implements genuine multi-stage trees; GPU backends would be artificially limited. This introduces a behavioral asymmetry that cannot be validated by the shared layer.
- **Memory constraint.** Single-stage reduction requires the entire partial collection and all offsets to be resident simultaneously, which may exceed GPU global memory for very large tiling grids.
- **Blocks future scaling.** As models grow, `num_partials` will exceed any reasonable `max_reduce_fan_in` limit. This option defers the problem rather than solving it.

### Option B: Multiple invocations of existing all-to-one kernels per stage

At each stage, the renderer dispatches `ceil(N/K)` separate calls to the existing aggregate kernel, each with a K-element sub-slice of the offset list, targeting a different offset in the output buffer. Then dispatches `ceil(N/K)` separate calls to `clip_intermediate_grad`, each targeting one `partial_width`-element slice of the output buffer.

**Advantages:**
- No kernel changes. Reuses existing primitives exactly as contracted.
- Produces mathematically correct per-node reduction and per-node clip.

**Disadvantages:**
- **`O(N)` kernel launches per stage.** Each of the `ceil(N/K)` nodes requires two kernel dispatches (aggregate + clip). For a stage with 64 nodes, this is 128 `clEnqueueNDRange` calls — dominated by launch overhead, not computation.
- **Offset list management complexity.** Each sub-invocation needs its own uploaded offset sub-list or a way to reference a sub-slice of the stage's full offset list. The current kernels read from offset 0 of their offset buffer — they have no `base_offset` parameter.
- **Violates CONCEPT.md §3.** The "dumb kernel, smart orchestrator" principle means the orchestrator should be composing a small number of powerful primitives, not micro-managing per-node dispatch. `ceil(N/K)` invocations per stage is orchestrator overreach.

### Option C: New K-fan-in reduce-and-clip kernel primitive

Introduce a new kernel — `reduce_k_fan_in_and_clip` — as a first-class primitive in `kernels.cl.h`. This kernel processes `node_count` independent reduction nodes in a single dispatch. Each node sums K partials (from a flat, contiguous offset list with K offsets per node), computes a per-node L2 norm, and conditionally clips independently. The existing all-to-one `aggregate_register_reduce` and `aggregate_local_reduce` kernels are retained for `num_stages == 1` (where they are correct and optimal).

**Advantages:**
- **Directly expresses the architectural concept.** The kernel is the physical realization of CONCEPT.md §2's per-stage K-fan-in reduction with §3.3's per-node clip.
- **Single dispatch per stage.** One `clEnqueueNDRange` call processes all `ceil(N/K)` nodes at a stage, exploiting GPU parallelism naturally.
- **Fused sum-then-clip eliminates intermediate state.** The un-clipped intermediate sum never needs to be globally visible — it exists only in registers or local memory within the kernel. This eliminates one buffer write and one buffer read per node per stage.
- **Per-node parallelism is GPU-natural.** Each node's reduction is independent, mapping directly to GPU work-groups. Within each work-group, the K-partial summation and L2 norm use local memory reduction — the same pattern already proven in `aggregate_local_reduce` and `clip_intermediate_grad`.
- **Minimal plan model impact.** The `ReductionTreePlan` (ADR-003) already carries `fan_in_K`, `num_stages`, `threshold_schedule`, and `initial_offset_list`. No new fields are needed. The renderer derives per-stage `node_count` and offset sub-list base indices trivially.
- **CPU parity is structural.** The CPU's `task_reduce_and_clip_node` already implements exactly this per-node fused semantic. The new OpenCL kernel is its direct GPU counterpart.

**Disadvantages:**
- **New kernel to specify, implement, and test across all backends.** This is non-trivial engineering: a new `@kernel_contract` in `kernels.cl.h`, a new `KernelContract` in `src/shared/kernel_contracts/phase_2_learn_C_reduction.py`, OpenCL C implementation, GLSL compute shader, and per-backend bindings.
- **Tiered implementation complexity.** For stages where K is small (≤ hardware crossover), a register-based variant is optimal; for large K, a local-memory variant is needed. This may require two tiered implementations of the fused kernel, mirroring the existing register/local split in the aggregate kernels.
- **Diagnostic trees don't need clip.** For `tree_variant == "diagnostic"` (Node 14), the clip step is a no-op. The fused kernel either carries a conditional branch or diagnostic trees continue using the existing all-to-one kernels (which are correct for the single-stage diagnostic case).

### Option D: Separate K-fan-in aggregate kernel + batched clip kernel

Split the fused operation into two new kernels: (1) `aggregate_k_fan_in` produces `node_count` output vectors of `partial_width` elements each, and (2) `clip_batched` independently clips each of the `node_count` vectors.

**Advantages:**
- Preserves the existing two-kernel sum-then-clip pattern familiar to the architecture.
- Each kernel is simpler to implement and test independently.
- Diagnostic trees can skip the clip kernel entirely (matching the current architecture).

**Disadvantages:**
- **Two dispatches per stage** instead of one, with a global memory write/read of the un-clipped intermediate between them.
- **Intermediate state is globally visible.** The un-clipped sum must be written to global memory by the aggregate kernel and read back by the clip kernel. For large `node_count × partial_width`, this is significant bandwidth waste.
- **Loss of register-level fusion.** The L2 norm computation could reuse the accumulator from the summation if fused; as separate kernels, the clip kernel must re-read data it didn't produce.

---

## Analysis

### Eliminating Option A

Option A is architecturally inadmissible. CONCEPT.md §1 (Architectural Elegance Feedback) prescribes that vocabulary gaps be resolved by extending the formal primitive set, not by constraining the system to avoid the gap. Limiting GPU backends to single-stage reduction contradicts the `log_K(N)` scalability that CONCEPT.md §2 identifies as a core architectural property.

Option A is acceptable only as a **temporary safety constraint** during the implementation period of the chosen solution — it prevents the broken multi-stage code path from executing while the new kernel is developed. It is not a resolution.

### Eliminating Option B

Option B produces correct results but at unacceptable orchestration complexity. `ceil(N/K)` separate kernel dispatches per stage means the OpenCL command queue accumulates hundreds of small kernel launches for large reduction trees. GPU kernel launch overhead (typically 5–20 μs per `clEnqueueNDRange`) would dominate the actual computation time, negating any benefit of GPU execution for the reduction phase.

Furthermore, the existing aggregate kernels' offset lists start at index 0 — they provide no mechanism to select a sub-range of a larger flat offset list. Adding a `base_offset` parameter to the existing kernels would modify their contracts, affecting all existing uses.

### Comparing Options C and D

The core question is whether sum-then-clip should be fused or separate.

**The mathematical argument for fusion.** CONCEPT.md §3.3 describes the sum-then-clip operation at each tree level as the atomic building block: "the system employs a policy-driven approach to set an independent, optimal clipping threshold at each layer of the reduction tree." The "layer" is the joint sum+clip operation — not sum alone or clip alone. Fusing them makes the kernel's computational scope match the mathematical specification's conceptual scope.

**The performance argument for fusion.** The un-clipped intermediate sum is consumed exactly once (by the clip), immediately after production. Writing it to global memory and reading it back is pure waste — the intermediate can live entirely in registers or local memory within a fused kernel. For a stage with 64 nodes and `partial_width = 4096`, fusion eliminates 64 × 4096 × 4 = 1 MB of redundant global memory traffic per stage.

**The complexity argument against fusion.** Fusion requires the kernel to perform both the indirection-based gather-sum and the L2 norm reduction within a single dispatch. This is more complex than either operation alone but not fundamentally novel — the CPU's `task_reduce_and_clip_node` already implements exactly this fusion, and the OpenCL `stabilize_and_reduce_grad_hidden_activations` (Node 16) implements an even more complex fused multi-stage reduction.

**The diagnostic tree argument.** Diagnostic trees (`tree_variant == "diagnostic"`, Node 14) perform sum-only reduction with no clip. For diagnostic trees:
- If `num_stages == 1`: the existing all-to-one `aggregate_register_reduce` / `aggregate_local_reduce` kernels are correct and optimal. No change needed.
- If `num_stages > 1`: the new K-fan-in kernel is needed for correct multi-node staging, but the clip should be skipped. Two sub-options: (a) the fused kernel accepts a `clip_threshold` parameter where a sentinel value (e.g., `FP_FORMAT_MAX` or `0.0`) disables clipping, or (b) a separate `aggregate_k_fan_in` kernel (without clip) is provided for diagnostic trees.

Sub-option (a) introduces a conditional branch in the kernel — a minor violation of CONCEPT.md §3's "no complex branching" principle, though the branch is a single boolean comparison at the end of the kernel, not a structural fork. Sub-option (b) adds a second kernel but preserves purity. Given that diagnostic reduction with `num_stages > 1` is an edge case (diagnostic partials are typically small in number), and that the threshold sentinel pattern is simple and well-precedented (the host already conditionally dispatches clip), sub-option (a) is acceptable.

### The role of the existing all-to-one kernels

The existing `aggregate_register_reduce` and `aggregate_local_reduce` are correct and optimal for their original design point: reducing ALL partials to a single output in a single stage. After this ADR:

- **Single-stage trees** (`num_stages == 1`): continue to use the existing all-to-one aggregate kernels followed by `clip_intermediate_grad`. No change.
- **Multi-stage trees** (`num_stages > 1`): use the new `reduce_k_fan_in_and_clip` kernel at each stage.

The kernel tier selection (which variant to use at stage 0 vs. stage 1) remains an Orchestration-tier concern per ADR-003 Decision Driver 5.

---

## Decision

**Option C: New K-fan-in reduce-and-clip kernel primitive**, with the following specifications.

### Immediate safety constraint (Phase A)

Until the new kernel is implemented and validated, add a plan-time assertion in the shared layer that prevents construction of multi-stage `ReductionTreePlan` instances for backends that lack the K-fan-in primitive. This is a temporary constraint, not a permanent architectural limitation. It documents the gap explicitly rather than allowing the broken renderer code path to execute silently:

```python
# In plan_builder.py or plan validation
if plan.num_stages > 1 and not backend_capabilities.supports_k_fan_in_reduce:
    raise PlanValidationError(
        f"Multi-stage reduction tree ({plan.num_stages} stages, "
        f"{plan.num_partials} partials) requires K-fan-in reduction primitive. "
        f"Backend does not yet support this. See ADR-019."
    )
```

### New kernel specification (Phase B)

A new kernel `reduce_k_fan_in_and_clip` is added to the kernel vocabulary in `kernels.cl.h`. This kernel is the multi-node, fused counterpart to the existing single-node pipeline of `aggregate_*` + `clip_intermediate_grad`.

#### Algorithmic specification

The kernel processes `node_count` independent reduction nodes in a single dispatch. For each node `n` in `[0, node_count)`:

1. **Gather and sum.** Read K partials from the source buffer via the offset list (offsets `[n*K, n*K+1, ..., n*K+K-1]`; sentinel offset `0xFFFFFFFF` indicates an absent partial for the final node when `N` is not divisible by `K`). Accumulate all `partial_width` elements into a per-node sum vector.

2. **Per-node L2 clip.** Compute the L2 norm of the per-node sum vector:

$$\text{norm}_n = \sqrt{\sum_{e=0}^{\text{partial\_width}-1} \text{sum}_n[e]^2}$$

If $\text{norm}_n > T_j$ (the stage's clipping threshold), scale the vector:

$$\text{sum}_n[e] \leftarrow \text{sum}_n[e] \cdot \frac{T_j}{\text{norm}_n + \varepsilon}$$

If $T_j = 0$ (sentinel for diagnostic trees), skip the clip entirely.

3. **Write output.** Write the (possibly clipped) sum vector to the destination buffer at offset `n * partial_width`.

#### Kernel contract

```c
/**
 * @brief (Node 14, 15a, 20a — multi-stage) Reduces groups of K scattered
 *        partials into independent output nodes with optional per-node L2 clip.
 * @kernel_contract
 *        - Holistic Constraints: "Each work-group processes one reduction node.
 *          The kernel reads K partials per node from the source buffer via an
 *          offset list, sums them, optionally clips the result per-node, and
 *          writes one output vector of partial_width elements. Supports absent
 *          partials via sentinel offset 0xFFFFFFFF for the tail node."
 *        - Behavioral Invariants: "When clipping_threshold > 0, per-node L2
 *          clip is applied: scale = threshold / (norm + epsilon). When
 *          clipping_threshold == 0, clip is bypassed (diagnostic mode).
 *          Epsilon prevents division by zero."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 */
__kernel void reduce_k_fan_in_and_clip(
    __local  COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,
    __global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint        *src_buffer_GLOBAL_CONST_offset_list_flat,
    __global COMPUTE_TYPE       *dest_buffer_GLOBAL_stage_output,
    uint src_scalar_NATURAL_fan_in_K,
    uint src_scalar_NATURAL_node_count,
    uint src_scalar_NATURAL_partial_width,
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold,
    COMPUTE_TYPE src_scalar_REAL_epsilon);
```

**Buffer contracts:**

| Parameter | Flow | Tensor Shape | Notes |
|:---|:---|:---|:---|
| `src_buffer_GLOBAL_partial_collection` | src | (undefined) | The memory pool. Valid buffer encompassing all offset references. |
| `src_buffer_GLOBAL_CONST_offset_list_flat` | src | (`node_count * fan_in_K`) | Flat offset list. K consecutive entries per node. Sentinel `0xFFFFFFFF` for absent partials. |
| `dest_buffer_GLOBAL_stage_output` | dest | (`node_count * partial_width`) | Contiguous output. Node `n` writes at `[n * partial_width, (n+1) * partial_width)`. |

**Scalar contracts:**

| Parameter | Type | Constraint |
|:---|:---|:---|
| `fan_in_K` | NATURAL | `>= 2` |
| `node_count` | NATURAL | `>= 1` |
| `partial_width` | NATURAL | `>= 1` |
| `clipping_threshold` | REAL | `>= 0.0`. Value `0.0` disables clip (diagnostic mode). |
| `epsilon` | REAL | Small positive constant (e.g., `1e-7`). |

**Dispatch grid:**

- **Global size:** `(partial_width, node_count)` — conceptually, one work-group per node. Within each work-group, threads collaborate on the K-partial summation and L2 norm reduction via local memory.
- **Local size:** `(min(partial_width, hardware_workgroup_size), 1)` — the Orchestration tier selects based on hardware.

The `partial_width` dimension parallelizes element-wise summation across threads within a work-group. The L2 norm reduction across elements uses a standard local-memory parallel reduction (the same pattern as `clip_intermediate_grad`).

### GPU execution model

Each work-group is assigned to one reduction node. Threads within the work-group are assigned to elements of the `partial_width` vector. The execution proceeds:

1. **Phase 1 — Gather and accumulate (element-parallel).** Each thread `t` (where `t = get_local_id(0)`, `t < partial_width`) accumulates element `t` across all K input partials:

```
accum[t] = 0
for k in 0..fan_in_K:
    offset = offset_list[node_id * fan_in_K + k]
    if offset != 0xFFFFFFFF:
        accum[t] += src[offset + t]
```

For `partial_width > workgroup_size`, threads stride across elements.

2. **Phase 2 — Per-node L2 norm (local memory reduction).** Each thread computes `accum[t]²` and contributes to a local-memory parallel reduction to compute `sum_sq`. Thread 0 computes `norm = sqrt(sum_sq)` and the scaling factor, broadcasting via local memory.

3. **Phase 3 — Conditional clip and write (element-parallel).** If `norm > threshold` and `threshold > 0`, each thread scales its element: `accum[t] *= scale`. Write to `dest[node_id * partial_width + t]`.

This three-phase model is a natural composition of the existing patterns from `aggregate_local_reduce` (Phase 1) and `clip_intermediate_grad` (Phases 2–3), fused into a single kernel dispatch with no intermediate global memory traffic.

### Updated rendering contract for ADR-003

ADR-003 §Backend rendering contract step 4–5 is revised. For each stage `s`:

**Single-stage tree** (`num_stages == 1`): unchanged. Dispatch existing `aggregate_register_reduce` or `aggregate_local_reduce` (all-to-one), followed by `clip_intermediate_grad` if `threshold_schedule[0]` is not `None`.

**Multi-stage tree** (`num_stages > 1`): dispatch `reduce_k_fan_in_and_clip` with:
- `fan_in_K = plan.fan_in_K`
- `node_count = ceil(current_N / K)`
- `partial_width = plan.partial_width`
- `clipping_threshold = threshold_schedule[s]` (or `0.0` for diagnostic trees)
- `epsilon` from precision configuration
- Offset list: for `s == 0`, the initial offset list expanded to `node_count * K` entries (padding the final node with sentinels). For `s > 0`, contiguous offsets derived from the previous stage's output layout.

Insert backend-native synchronization between stages. Manage intermediate buffers via ping-pong allocation.

### KernelContract specification

A new `reduce_k_fan_in_and_clip_contract` `KernelContract` frozen dataclass is added to `src/shared/kernel_contracts/phase_2_learn_C_reduction.py`:

```python
reduce_k_fan_in_and_clip_contract = KernelContract(
    kernel_name="reduce_k_fan_in_and_clip",
    contract_block=KernelContractBlock(
        holistic_constraints=(
            "Each work-group processes one reduction node. "
            "Reads K partials per node via flat offset list, sums them, "
            "optionally clips per-node, and writes one output vector."
        ),
        idempotency="Associatively Non-Idempotent",
        synchronization_model="Reduction Engine Stage",
        behavioral_invariants=(
            "Per-node L2 clip when clipping_threshold > 0: "
            "scale = threshold / (norm + epsilon).",
            "Clip bypassed when clipping_threshold == 0 (diagnostic mode).",
            "Sentinel offset 0xFFFFFFFF skips absent partials in tail node.",
        ),
    ),
    buffer_params=(
        BufferParamSpec(
            name="src_buffer_GLOBAL_partial_collection",
            flow="src", memory_scope="GLOBAL",
            tensor_shape=("undefined",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=(),
            validation_preconditions=(
                "valid buffer encompassing all offset references",
            ),
        ),
        BufferParamSpec(
            name="src_buffer_GLOBAL_CONST_offset_list_flat",
            flow="src", memory_scope="GLOBAL_CONST",
            tensor_shape=("node_count * fan_in_K",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("node_count", "fan_in_K"),
            validation_preconditions=(
                "exactly node_count * fan_in_K uint entries",
            ),
        ),
        BufferParamSpec(
            name="dest_buffer_GLOBAL_stage_output",
            flow="dest", memory_scope="GLOBAL",
            tensor_shape=("node_count * partial_width",),
            padding_contract=PaddingContract("NONE", None),
            calculability_proof=("node_count", "partial_width"),
            validation_preconditions=("exact allocation size",),
        ),
    ),
    scalar_params=(
        ScalarParamSpec("fan_in_K", "src", "NATURAL"),
        ScalarParamSpec("node_count", "src", "NATURAL"),
        ScalarParamSpec("partial_width", "src", "NATURAL"),
        ScalarParamSpec("clipping_threshold", "src", "REAL"),
        ScalarParamSpec("epsilon", "src", "REAL"),
    ),
    local_memory=(
        LocalMemorySpec(
            "reduction_tile",
            "get_local_size(0) * sizeof(COMPUTE_TYPE)",
        ),
    ),
    placement=None,
)
```

### Relationship to existing kernels

| Kernel | After ADR-019 |
|:---|:---|
| `aggregate_register_reduce` | **Retained.** Used for single-stage trees and wherever all-to-one reduction is correct. |
| `aggregate_local_reduce` | **Retained.** Same as above, for large N. |
| `clip_intermediate_grad` | **Retained.** Used for single-stage trees (one clip dispatch post-aggregate). |
| `reduce_k_fan_in_and_clip` | **New.** Used for multi-stage trees at each stage. Replaces the broken aggregate+clip sequence. |

The existing kernels are not deprecated — they are optimal for the single-stage case and remain the correct choice when the renderer determines that `num_stages == 1`.

### CPU backend alignment

The CPU backend's `execute_reduction_tree` and `task_reduce_and_clip_node` already implement exactly the per-node fused reduce-and-clip semantic. No changes to the CPU kernel sources are needed. The CPU renderer's `_render_reduction_tree` continues to call `execute_reduction_tree`, which uses the existing C implementation.

The semantic alignment is now explicit:

| Concept | OpenCL | CPU |
|:---|:---|:---|
| K-fan-in reduce + per-node clip | `reduce_k_fan_in_and_clip` (GPU kernel) | `task_reduce_and_clip_node` (thread pool task) |
| All-to-one reduce + buffer clip | `aggregate_*` + `clip_intermediate_grad` | N/A (CPU always uses per-node) |

---

## Consequences

### Positive

- **Resolves the structural divergence.** The OpenCL and CPU backends produce mathematically identical results for multi-stage reduction trees. Tier 3 parity tests are unblocked.
- **Kernel vocabulary matches architecture.** The `reduce_k_fan_in_and_clip` kernel is the direct physical realization of CONCEPT.md §2's per-stage K-fan-in reduction and §3.3's per-node clip — a first-class primitive, not an orchestrator workaround.
- **"Dumb kernel" principle preserved.** The new kernel is stateless, single-purpose, and parametric. It does not know its position in the tree, the tree's depth, or the stabilization policy. It receives a threshold and applies it.
- **Fused dispatch eliminates intermediate bandwidth.** The un-clipped sum lives in registers/local memory, never touching global memory. For large `node_count × partial_width`, this is significant bandwidth savings compared to separate aggregate + clip dispatches.
- **No plan model changes.** The `ReductionTreePlan` (ADR-003) carries all data the renderer needs. No new fields, no changed semantics. The new kernel is a rendering concern — the plan specifies *what*, the kernel implements *how*.
- **Backward compatible.** Single-stage trees continue using the existing optimized kernels. The new kernel is additive.

### Negative

- **New kernel implementation and testing across 3 backends.** The OpenCL C and GLSL implementations must be written and validated. The CPU already implements the semantic, but a formal `KernelContract` alignment ensures the C function's parameter names and ordering match the contract.
- **Tiered variants may be needed.** For small K (≤ crossover), a register-only variant of the fused kernel may be desirable to avoid local memory overhead. Whether to implement one or two tiers is a Phase B rendering decision — the contract supports both.
- **Diagnostic sentinel.** The `clipping_threshold == 0.0` sentinel for disabling clip is a minor semantic overload. An explicit `operation_type` flag (as in the existing aggregate kernels) would be purer but adds a parameter for a distinction that matters only to diagnostic trees (which are typically single-stage anyway).

### Implementation phases

| Phase | Scope | Gate |
|:---|:---|:---|
| **A — Safety constraint** | Add plan-time assertion preventing multi-stage trees for backends lacking K-fan-in support. | Immediate. |
| **B — Kernel specification** | Add `@kernel_contract` to `kernels.cl.h`. Add `KernelContract` to `src/shared/kernel_contracts/phase_2_learn_C_reduction.py`. | ADR-019 accepted. |
| **C — OpenCL implementation** | Implement `reduce_k_fan_in_and_clip` in `kernels/phase_2_learn_C_reduction.cl.c`. Add binding in `src/backends/opencl/kernel_bindings/phase_2_learn_C_reduction.py`. | Tier 2 tests pass for new kernel in isolation. |
| **D — Renderer integration** | Update `_render_reduction_tree` in `src/backends/opencl/renderer.py` to dispatch `reduce_k_fan_in_and_clip` for `num_stages > 1`. Remove safety constraint from Phase A. | Tier 2 tests pass for multi-stage trees. |
| **E — Parity validation** | Run Tier 3 cross-backend parity tests with model configurations that produce `num_stages > 1`. | Tier 3 parity gate (ADR-017 Phase 6). |

### ADR-003 amendment

ADR-003 §Backend rendering contract, step 5, is amended to read:

> *5. **Per-stage reduction dispatch.** For single-stage trees: dispatch the existing all-to-one aggregate kernel followed by `clip_intermediate_grad` (unchanged). For multi-stage trees: dispatch `reduce_k_fan_in_and_clip` (ADR-019) with `fan_in_K`, per-stage `node_count`, `partial_width`, and `threshold_schedule[s]`. The kernel performs fused K-fan-in summation and per-node L2 clip in a single dispatch.*

---

## References

- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §2 Recursive, Tiered Aggregation Engine; §3.3 Quadratic Scaling Policy; §3 Modular, "Dumb" Kernels
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `ReductionTreePlan` data structure; backend rendering contract
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` specification pattern
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — `kernels.cl.h` as algorithmic authority; per-backend source organization
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — Three-tier jurisdictional model; shared correctness
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability
- [CPU vs OpenCL Kernel Audit](../doc_archive/CPU_VS_OPENCL_KERNEL_AUDIT.md) — Finding 2: Reduction Tree Clip Granularity
