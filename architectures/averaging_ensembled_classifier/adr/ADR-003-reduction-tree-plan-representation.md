# ADR-003: Reduction Tree Plan Representation

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002  
**Blocks:** ADR-009

---

## Context

ADR-002 establishes `ReductionTreeNode` as one of the five canonical plan node types and defines its architectural role: a multi-stage reduction tree rendered atomically by the backend. The tree's stages are not individual plan nodes — they are internal structure within a single typed node. ADR-002 defers the concrete representation of this internal structure to this ADR.

The Recursive Clip-Aggregation Engine (CONCEPT.md §2, §3) manifests as three distinct `ReductionTreeNode` instances in the execution plan:

| Instance     | CONCEPT.md Node | Data Reduced                              | Stage Pattern    |
| :----------- | :-------------- | :---------------------------------------- | :--------------- |
| Diagnostic   | Node 14         | `PARTIAL_Probs`, `PARTIALS_Loss_BCE`      | Sum-only         |
| Module grads | Node 15         | `Clipped_PARTIAL_Grad_ModW/ModB/Temp`     | Sum-then-clip    |
| Shared grads | Node 20         | `Clipped_PARTIAL_Grad_SW/SB` (streamed)   | Sum-then-clip    |

### The mathematical structure

The tree's mathematical structure is identical across all backends:

- **Fan-in `K`:** The uniform reduction batch size, resolved by `StabilizationPolicy.plan_uniform_reduction_tree()` from three constraints: user intent, hardware maximum, and mathematical safety (CONCEPT.md §3.4). The same `K` applies to every stage.
- **Stage count:** $\lceil \log_K(N) \rceil$, where `N` is the number of input partials.
- **Offset lists:** Integer arrays of memory displacements consumed by the `aggregate_*` kernels via the Indirection Contract (CONCEPT.md §2). The initial offset list is non-trivial — its layout depends on the upstream placement strategy (scattered partial offsets from `grid_mod_cls`, linear chunks from streaming). Subsequent stages always reduce contiguous intermediate results.
- **Per-stage thresholds:** For stabilized trees, each stage `j` (where `j=0` is the root) uses the Quadratic Scaling Policy: $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$, clamped by the dynamic safety ceiling $T_{\text{safety},j} = \text{FP\_FORMAT\_MAX} / K_j$ (CONCEPT.md §3.4). For diagnostic trees, no thresholds apply.
- **Kernel tier selection:** Each stage selects between `aggregate_register_reduce` (fan-in ≤ hardware crossover threshold) and `aggregate_local_reduce` (fan-in > threshold). The crossover is a hardware-informed heuristic.
- **Intermediate buffer sizing:** Each intermediate stage produces $\lceil N_{\text{current}} / K \rceil$ partials, each of `elements_per_partial` scalars.

### What varies across backends

Only the dispatch and synchronization mechanics differ:

- **OpenCL:** Per-stage `clEnqueueNDRange` calls with `cl.Event` chains; offset lists uploaded via `cl.enqueue_copy` to `cl.Buffer`; intermediate buffers allocated as `cl.Buffer`.
- **Vulkan:** All stages recorded into a command buffer segment with `vkCmdPipelineBarrier` between stages; offset lists in mapped staging buffers; intermediate buffers suballocated from `VkDeviceMemory`.
- **CPU:** Sequential `pool_dispatch_and_wait` calls; offset lists are direct pointers into host memory; intermediate buffers are `malloc`'d arrays.

### The design tension

The shared orchestration layer must compute the tree's mathematical structure exactly once (ADR-001: "shared correctness"). The question is how much of this structure to pre-compute and embed in the plan versus how much to leave the renderer to derive.

- The initial offset list is non-trivial and placement-dependent. It cannot be derived by the renderer.
- The stabilization thresholds are non-trivial policy computations involving the Quadratic Scaling Policy and safety clamping. Duplicating this across renderers contradicts ADR-001's shared-computation principle.
- Intermediate stage offset lists are always contiguous (`[0, 1, 2, ..., N_stage-1]`) — trivially derivable from the stage's input count.
- Kernel tier selection (register vs. local) is a simple comparison of the stage fan-in against a hardware-dependent crossover threshold. This threshold may differ per backend — a GPU's register/local tradeoff differs fundamentally from a CPU's L1 cache/SIMD tradeoff.

---

## Decision Drivers

1. **ADR-001 (Three-tier jurisdictional model — Policy tier):** ADR-001 §Three-tier jurisdictional model observes that in the existing OpenCL code, the Policy tier (shared layer) is the only tier invariant across all node types and backends. Reduction tree planning — stage count, fan-in, Quadratic Scaling Policy thresholds — is Policy-tier computation. The plan must carry this computation's results, not require renderers (Orchestration tier) to recompute them.

2. **ADR-002 (ReductionTreeNode atomicity):** The tree is a single typed node, not a sequence of `KernelDispatchNode`s. Its internal structure is a nested data structure within the `ReductionTreeNode`. The renderer must interpret this structure natively without reconstructing it from flat nodes.

3. **CONCEPT.md §1 (Architectural Elegance Feedback):** The `ReductionTreePlan` representation is a first-class architectural primitive. If an optimization creates tension with this representation (e.g., a backend-specific fusion of adjacent sum-and-clip stages), the response is to extend the plan vocabulary — not to bypass it.

4. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability):** Pre-dispatch validation must be possible in the shared layer. The plan must carry sufficient data to verify reduction tree correctness (stage count consistency, threshold monotonicity, fan-in safety bounds) at plan-construction time.

5. **Backend rendering freedom (ADR-001 — Orchestration tier):** Kernel tier selection depends on hardware-specific crossover heuristics. Per ADR-001's three-tier model, kernel tier selection is an Orchestration-tier concern — the existing OpenCL code's `max_reg_agg = 16` in `AggregationManager` is a backend-specific heuristic that other backends need not share. The plan must not prescribe register-vs-local decisions that are fundamentally backend-dependent.

---

## Options Considered

### Option A: Fully materialized per-stage descriptor array

The `ReductionTreePlan` contains an explicit array of `ReductionStageDescriptor` frozen dataclasses — one per stage, ordered from leaf to root. Each descriptor carries: the stage's fan-in count, the complete offset list (as an integer array), the kernel tier indicator (`"register"` or `"local"`), the clip threshold (`T_j` or `None` for diagnostic stages), and the intermediate output buffer shape.

```python
@dataclass(frozen=True)
class ReductionStageDescriptor:
    stage_index: int                    # j (0 = root, num_stages-1 = leaf)
    fan_in: int                         # K for this stage
    offset_list: Tuple[int, ...]        # Memory displacements
    kernel_tier: str                    # "register" or "local"
    clip_threshold: Optional[float]     # T_j, or None for sum-only
    output_partial_count: int           # ceil(input_count / fan_in)

@dataclass(frozen=True)
class ReductionTreePlan:
    stages: Tuple[ReductionStageDescriptor, ...]
    elements_per_partial: int
    source_buffer: str                  # Logical buffer name
    tree_variant: str                   # "diagnostic" or "stabilized"
```

**Advantages:**
- Maximum plan completeness. The renderer iterates over descriptors with zero derivation.
- Every aspect of the tree is inspectable and validatable at plan-construction time.
- Serialisation is straightforward — each stage is fully self-contained.

**Disadvantages:**
- **Prescribes kernel tier selection.** The `kernel_tier` field bakes a hardware-dependent heuristic (the register/local crossover point, currently 16 for OpenCL) into the shared plan. A CPU backend's crossover — or the absence of one entirely — differs fundamentally. This violates ADR-001's rendering freedom.
- **Carries redundant data.** Intermediate stage offset lists are always contiguous integers `[0, 1, ..., N-1]`. Embedding them inflates plan size for no information gain (they are trivially derivable from the stage's input count).
- **Fan-in redundancy.** The fan-in is uniform across all stages (by design of `plan_uniform_reduction_tree`). Carrying it per-stage implies it could vary, which is misleading.
- **Breaks the plan→renderer jurisdictional boundary.** Kernel tier is a dispatch concern (which kernel binary to invoke), not a plan concern (what computation to perform). Deciding it in the shared layer conflates what ADR-001 separated.

### Option B: Compact parametric specification

The `ReductionTreePlan` carries only tree-level parameters. The renderer derives all per-stage details at render time.

```python
@dataclass(frozen=True)
class ReductionTreePlan:
    num_partials: int                   # N (total input partials)
    fan_in: int                         # K (uniform)
    num_stages: int                     # ceil(log_K(N))
    elements_per_partial: int
    initial_offset_list: Tuple[int, ...]  # Placement-dependent
    tree_variant: str                   # "diagnostic" or "stabilized"
    stabilization_policy: Optional["StabilizationPolicy"]  # For threshold computation
    source_buffer: str                  # Logical buffer name
```

**Advantages:**
- Maximally compact. The plan carries only non-derivable data plus the policy object.
- Full rendering freedom — kernel tier, intermediate offsets, and thresholds are all renderer-determined.

**Disadvantages:**
- **Embeds a live policy object in the plan.** The `StabilizationPolicy` instance crosses the plan boundary. ADR-001 specifies the plan as a backend-neutral *data structure*. A policy object with methods (`get_threshold_for_generic_stage()`) is an executable component, not data. This is a boundary violation.
- **Forces threshold recomputation per backend.** Each renderer must call `policy.get_threshold_for_generic_stage()` for every stage of every tree. This duplicates the policy's threshold logic at render time — the exact "risk of cross-backend divergence" that ADR-001 warns against.
- **Validation gap.** The shared layer cannot validate threshold correctness (monotonicity, safety bounds) at plan-construction time because the thresholds are not yet computed.

### Option C: Parametric header with pre-computed threshold schedule

The `ReductionTreePlan` carries tree-level parameters and the pre-computed per-stage threshold schedule. Trivially derivable per-stage data (contiguous intermediate offsets, kernel tier) is left to the renderer.

```python
@dataclass(frozen=True)
class ReductionTreePlan:
    num_partials: int                       # N (total input partials)
    fan_in: int                             # K (uniform, safe)
    num_stages: int                         # ceil(log_K(N))
    elements_per_partial: int
    initial_offset_list: Tuple[int, ...]    # Placement-dependent, non-trivial
    tree_variant: str                       # "diagnostic" or "stabilized"
    threshold_schedule: Tuple[Optional[float], ...]  # Per-stage T_j (leaf→root), None for diagnostic
    source_buffer: str                      # Logical buffer name
```

**Advantages:**
- Carries exactly the non-trivial shared computation: fan-in `K`, stage count, initial offsets, and the threshold schedule.
- The threshold schedule is pure data (a tuple of floats) — no executable objects cross the plan boundary.
- Full plan-time validation: the shared layer can verify threshold monotonicity ($T_{j+1} \geq T_j$ for $\lambda \geq 0$), safety bounds ($T_j \leq \text{FP\_FORMAT\_MAX} / K$), and stage count consistency ($K^{\text{num\_stages}} \geq N$).
- Kernel tier selection remains a renderer concern — each backend applies its own hardware-specific crossover heuristic.
- Intermediate offset lists (always contiguous) are not carried — the renderer derives them from the stage's input count.

**Disadvantages:**
- The renderer must still derive two things per stage: the intermediate offset list and the kernel tier. These are both trivial (one is an iota sequence, the other is a comparison against a local constant), but they are derivations nonetheless.
- The `tree_variant` field is technically redundant with `threshold_schedule` (a diagnostic tree has all-`None` entries). However, the field serves as an explicit semantic marker and validation target.

---

## Analysis

### Eliminating Option A

Option A over-specifies the plan by prescribing kernel tier selection — a dispatch concern that belongs in the renderer. The register/local crossover point is a hardware heuristic:

- **OpenCL (current):** `max_reg_agg = 16`, informed by GPU register file pressure.
- **Vulkan:** Potentially different, determined by subgroup size and shared memory latency.
- **CPU:** No meaningful register/local distinction. The crossover maps to SIMD width and L1 cache line boundaries — a fundamentally different concern.

Embedding this decision in the shared plan either forces a lowest-common-denominator choice (violating ADR-001's backend fidelity principle) or requires backend-specific overrides in the plan (violating the backend-neutral constraint). Neither is acceptable.

Furthermore, carrying per-stage offset lists for intermediate stages is information-free. After stage 0 (which uses the non-trivial initial offsets), every subsequent stage reduces contiguous intermediate results. The offset list for stage $s$ with $N_s$ input partials is always `(0, 1, 2, ..., N_s - 1)`. Pre-computing and storing these arrays provides zero information beyond the stage's input count.

### Eliminating Option B

Option B embeds a live `StabilizationPolicy` object in the plan. The policy has methods — `get_threshold_for_generic_stage()`, `plan_uniform_reduction_tree()` — that encapsulate the Quadratic Scaling Policy mathematics and safety clamping logic. Including this object in the plan crosses the data-structure boundary established by ADR-001:

> "The shared orchestration layer produces a backend-neutral execution plan expressed as a **data structure**."

A data structure with embedded methods is not a data structure — it is an API surface. Each renderer would depend on the policy's method signatures, coupling backend code to the shared layer's implementation details rather than its data contracts.

More critically, Option B pushes threshold computation to render time, forfeiting the shared layer's ability to validate the reduction tree at plan construction. ADR-001's foundational consequence — "shared correctness" — requires that non-trivial shared computations are performed once and their results carried in the plan. The threshold schedule is precisely such a computation: it applies the Quadratic Scaling Policy ($T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$), enforces the safety clamp ($\min(T_j, \text{FP\_FORMAT\_MAX} / K)$), and respects the minimum threshold floor (`min_threshold`). Duplicating this across renderers invites divergence.

### Choosing Option C

Option C separates shared from backend concerns at exactly the right boundary — the Policy/Orchestration boundary identified in ADR-001 §Three-tier jurisdictional model:

**Policy tier pre-computes (carried in the plan):**
- Fan-in `K` — resolved from user intent, hardware limits, and mathematical safety by `StabilizationPolicy.plan_uniform_reduction_tree()`.
- Stage count — derived from `K` and `N`.
- Initial offset list — computed from the upstream gather primitive (placement-dependent, non-trivial).
- Threshold schedule — the Quadratic Scaling Policy output for every stage, including safety clamping. This is the core shared computation.

**Orchestration tier derives (at render time):**
- Intermediate offset lists — always contiguous, trivially derived from per-stage input count.
- Kernel tier — hardware-specific crossover heuristic applied to the uniform fan-in `K`.
- Intermediate buffer allocation — from `elements_per_partial` and per-stage output count.
- Upload mechanism for offset lists — `cl.enqueue_copy`, mapped staging buffer, or direct pointer.
- Synchronization between stages — event chains, pipeline barriers, or implicit sequencing.

This boundary maps directly to the three-tier model: Policy-tier output (thresholds, fan-in, offsets) crosses the plan boundary as pure data; the Orchestration tier interprets it through the backend's native execution model.

### The `tree_variant` field

The `tree_variant` field (`"diagnostic"` or `"stabilized"`) is semantically redundant with the `threshold_schedule` — a diagnostic tree has all-`None` entries; a stabilized tree has all-`float` entries. The field is retained because:

1. **Validation hook.** The plan builder asserts consistency: `tree_variant == "diagnostic"` iff all entries in `threshold_schedule` are `None`. This catches construction bugs where a stabilized tree accidentally omits a threshold.
2. **Renderer dispatch.** A renderer rendering a diagnostic tree can skip clip kernel setup entirely — it does not need to inspect the threshold schedule to discover this. The field is a constant-time semantic signal.
3. **Architectural correspondence.** The CONCEPT.md distinction between "Diagnostic Reduction (Node 14)" and "Gradient Reduction (Nodes 15 & 20)" is a first-class architectural concept, not an incidental implementation detail. The plan should preserve this distinction explicitly.

### The edge case: N = 1

When `N = 1`, there is no reduction to perform — the single input partial is the result. The current implementation handles this as a direct memory copy (see `_execute_reduction_pipeline` in `graph_recipes.py`). In the plan model, this case is handled by the plan builder, not the `ReductionTreePlan`:

- For `N = 1`, the plan builder emits a `KernelDispatchNode` (or equivalent copy construct) instead of a `ReductionTreeNode`. The `ReductionTreePlan` is never constructed with `num_stages = 0` — this invariant is enforced at plan-construction time.

### The edge case: N ≤ K (single-stage tree)

When `1 < N ≤ K`, the tree has exactly one stage (the root). The `threshold_schedule` has exactly one entry — `T_0 = T_{\text{algorithmic}}` for stabilized trees (since $j = 0$ implies $T_j = T_{\text{algorithmic}} + \lambda \cdot 0 = T_{\text{algorithmic}}$, clamped by safety). The initial offset list serves as both the first and final stage's input.

---

## Decision

**Option C: Parametric header with pre-computed threshold schedule.**

The `ReductionTreePlan` is a frozen dataclass embedded as the internal data structure of a `ReductionTreeNode` (ADR-002). It carries the tree-level parameters and the pre-computed threshold schedule. The renderer derives trivially derivable per-stage data (contiguous intermediate offsets, kernel tier selection) at render time.

### Data structure

```python
@dataclass(frozen=True)
class ReductionTreePlan:
    """
    The internal structure of a ReductionTreeNode.

    Computed by the shared orchestration layer. Consumed by backend renderers.
    Pure data — no methods, no backend types, no executable policy objects.
    """
    num_partials: int
    """Total input partial count (N). Invariant: num_partials > 1."""

    fan_in: int
    """Uniform, safe fan-in (K). Computed by StabilizationPolicy.plan_uniform_reduction_tree().
    Invariant: fan_in >= 2."""

    num_stages: int
    """Number of reduction stages. Invariant: fan_in ** num_stages >= num_partials."""

    elements_per_partial: int
    """Element count per partial (e.g., number of weight or gradient scalars).
    All partials in the tree share this width."""

    initial_offset_list: Tuple[int, ...]
    """Memory displacements for the leaf stage's gather operation.
    Computed from the upstream placement strategy (e.g., grid_mod_cls scatter pattern).
    Length equals num_partials. Non-trivial — placement-dependent."""

    tree_variant: str
    """'diagnostic' (sum-only, Node 14) or 'stabilized' (sum-then-clip, Nodes 15 & 20).
    Invariant: consistent with threshold_schedule (see below)."""

    threshold_schedule: Tuple[Optional[float], ...]
    """Pre-computed per-stage clip thresholds, ordered leaf-to-root
    (index 0 = leaf stage, index num_stages-1 = root stage).
    Length equals num_stages.

    For 'stabilized' trees: every entry is a float T_j, computed as
        min(T_algorithmic + lambda * j², FP_FORMAT_MAX / K),
    where j is the stage's distance from the root (j=0 at root, j=num_stages-1 at leaves).

    For 'diagnostic' trees: every entry is None.

    Invariant (stabilized, lambda >= 0): thresholds are monotonically
    non-increasing from leaf to root (T_leaf >= T_{leaf-1} >= ... >= T_root).
    This reflects the Quadratic Scaling Policy's funnel shape."""

    source_buffer: str
    """Logical name of the buffer containing the input partial collection."""
```

### Plan construction contract

The plan builder constructs a `ReductionTreePlan` by:

1. **Resolving `K` and `num_stages`.** Calling `StabilizationPolicy.plan_uniform_reduction_tree(num_partials, hardware_max_fan_in)`, which synthesizes user intent, hardware limits, and the mathematical safety constraint (CONCEPT.md §3.4) into a globally safe `(K, num_stages)` pair.

2. **Computing the initial offset list.** The upstream gather primitive (scattered offsets from placement, linear chunks from streaming) provides the integer displacement array. This is the only non-contiguous offset list in the tree.

3. **Computing the threshold schedule.** For stabilized trees: iterating `j` from `num_stages - 1` (leaf) to `0` (root) and calling `StabilizationPolicy.get_threshold_for_generic_stage(stage_j=j, runtime_fan_in_k=K)` for each stage. The result is a tuple of floats ordered leaf-to-root. For diagnostic trees: a tuple of `None` values.

4. **Validating invariants.** At construction time:
   - `num_partials > 1` (the N=1 case is handled as a direct copy, not a reduction tree).
   - `fan_in >= 2`.
   - `fan_in ** num_stages >= num_partials`.
   - `len(initial_offset_list) == num_partials`.
   - `len(threshold_schedule) == num_stages`.
   - `tree_variant == "diagnostic"` iff all entries in `threshold_schedule` are `None`.
   - `tree_variant == "stabilized"` iff all entries in `threshold_schedule` are `float`.
   - For stabilized trees with $\lambda \geq 0$: threshold monotonicity (leaf ≥ root).
   - Every threshold ≤ `FP_FORMAT_MAX / K` (safety ceiling).

### Backend rendering contract

The renderer interprets a `ReductionTreePlan` as the Orchestration tier (ADR-001 §Three-tier jurisdictional model): it sequences and dispatches the computation using the backend's native execution model, consuming the Policy tier's pre-computed plan data. For each stage `s` (where `s = 0` is the leaf stage, `s = num_stages - 1` is the root):

1. **Determine input count.** For `s = 0`: `num_partials`. For `s > 0`: $\lceil N_{s-1} / K \rceil$, where $N_{s-1}$ is the input count of the previous stage.

2. **Resolve offset list.** For `s = 0`: use `initial_offset_list`. For `s > 0`: generate a contiguous offset list `(0, 1, ..., input_count - 1)`.

3. **Select kernel tier.** Apply the backend's hardware-specific crossover heuristic to determine whether to use the register-based or local-memory-based aggregate kernel. This is a rendering concern — the plan does not prescribe it.

4. **Dispatch the aggregation kernel.** Upload the offset list (backend-native mechanism), bind buffers, and dispatch.

5. **Conditionally dispatch the clip kernel.** If `threshold_schedule[s]` is not `None`, dispatch `clip_intermediate_grad` with the pre-computed threshold $T_j$. If `None`, skip (diagnostic tree — no clipping at this stage).

6. **Insert synchronization.** The backend inserts its native synchronization mechanism between stages: `cl.Event` wait-list (OpenCL), `vkCmdPipelineBarrier` (Vulkan), or implicit sequencing (CPU).

7. **Manage intermediate buffers.** The renderer allocates intermediate result buffers for each stage's output. Buffer sizing is derived from `elements_per_partial` and the stage's output count. A ping-pong buffer scheme (as in the current `PingPongManager`) is a natural renderer optimization.

### Relationship to `ReductionTreeNode`

The `ReductionTreeNode` (ADR-002) embeds a `ReductionTreePlan` alongside its standard node fields:

```python
@dataclass(frozen=True)
class ReductionTreeNode:
    node_id: str
    depends_on: FrozenSet[str]
    reduction_plan: ReductionTreePlan
    destination_buffer: str   # Logical name of the final output buffer
```

The `ReductionTreeNode` is a plan node with dependency edges; the `ReductionTreePlan` is its internal computational structure. This nesting mirrors the `KernelDispatchNode` / `KernelContract` relationship — the node owns its DAG position and dependencies; the embedded structure owns the computation's parameters.

---

## Consequences

### Positive

- **Single computation of shared policy.** The Quadratic Scaling Policy thresholds, fan-in resolution, and stage count are computed once by the Policy tier and carried as immutable data. No renderer reimplements this logic. Backend divergence in threshold computation is structurally prevented.
- **Full plan-time validation.** All reduction tree invariants — threshold monotonicity, safety bounds, stage count consistency, fan-in validity — are verifiable at plan-construction time without any backend. This satisfies CONTRACT.md Article 1.4.
- **Backend rendering freedom.** Kernel tier selection, intermediate buffer management, offset list upload, and inter-stage synchronization are Orchestration-tier concerns (ADR-001 §Three-tier jurisdictional model). Each backend applies its own hardware heuristics without constraint.
- **Compact representation.** The plan carries only non-derivable data: the initial offset list (placement-dependent), the threshold schedule (policy-dependent), and the tree parameters. Intermediate contiguous offset lists (trivially derivable) are not stored.
- **Inspectable policy execution trace.** The threshold schedule tuple is a complete, ordered record of the stabilization policy's decisions for this tree. It can be logged, compared across training steps, and validated against expected monotonicity — all without executing any reduction.
- **Diagnostic/stabilized distinction preserved.** The `tree_variant` field and the `threshold_schedule` jointly preserve the CONCEPT.md distinction between Node 14 (diagnostic) and Nodes 15/20 (stabilized) as inspectable plan data.

### Negative

- **Renderer must still derive per-stage data.** Intermediate offset lists and kernel tier selection require per-stage computation at render time. Both are trivial — contiguous iota sequences and a single integer comparison — but they are computations nonetheless. The alternative (Option A) would eliminate them at the cost of violating the abstraction boundary.
- **Uniform fan-in assumption.** The representation assumes uniform `K` across all stages, matching `StabilizationPolicy.plan_uniform_reduction_tree()`'s current contract. A future policy producing non-uniform fan-ins would require extending `ReductionTreePlan` — per CONCEPT.md §1, this would be a formal vocabulary extension, not an ad-hoc workaround.
- **`tree_variant` redundancy.** The `tree_variant` field is derivable from `threshold_schedule`. Carrying both technically violates the DRY principle. The justification (validation hook, renderer dispatch optimization, architectural correspondence) is documented in the Analysis section.

### Migration implications

Per ADR-002's migration path:

1. **Define `ReductionTreePlan`** as a frozen dataclass in the shared layer (e.g., `execution_plan.py`). No existing code is modified.
2. **Add plan-construction logic** to the `ExecutionPlanBuilder`: for each reduction tree (Nodes 14, 15, 20), call `StabilizationPolicy.plan_uniform_reduction_tree()` to resolve `K` and `num_stages`, compute the threshold schedule, and construct a `ReductionTreePlan`.
3. **Write plan-level tests** (ADR-016, layer 1) that validate:
   - Threshold schedule monotonicity for stabilized trees.
   - Safety bound compliance ($T_j \leq \text{FP\_FORMAT\_MAX} / K$).
   - Stage count consistency ($K^{\text{num\_stages}} \geq N$).
   - All-`None` thresholds for diagnostic trees.
   - Initial offset list length equals `num_partials`.
4. **OpenCL renderer** (Phase 3 of ADR-001 migration) interprets `ReductionTreePlan` by replacing the current `_execute_reduction_pipeline` function in `graph_recipes.py`. The renderer iterates over stages, applies its register/local crossover heuristic (currently `max_reg_agg = 16`), and dispatches via `AggregationManager`.
5. **CPU and Vulkan renderers** (Phase 4) consume the same `ReductionTreePlan` with their native dispatch models.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; shared correctness requirement; three-tier jurisdictional model (Policy / Orchestration / Execution)
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `ReductionTreeNode` definition; atomicity principle; Complexity Ceiling Constraint
- [CONCEPT.md](../CONCEPT.md) — §2 Recursive Tiered Aggregation Engine; §3 Gradient Stabilization (Quadratic Scaling Policy, safety clamping); §1 Architectural Elegance Feedback
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Group 5 Parallel Processing & Reduction Primitives; `offset_list` Indirection Contract
- [ADR_PLAN.md](../ADR_PLAN.md) — ADR-003 problem statement and narrowing analysis
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — plan-level buffer lifetime annotations; two-tier buffer scope formalization
