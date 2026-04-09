# ADR-005: Node 16 Opacity in the Plan

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002, ADR-003  
**Blocks:** —

---

## Context

CONCEPT.md designates Node 16 (`stabilize_and_reduce_grad_hidden_activations`) as a **Specialized Kernel** — a self-contained reduction engine that internally executes a multi-stage `log_K(M)` reduction with per-stage `sum-then-clip` stabilization, applying the host-provided Quadratic Scaling Policy. This immediately raises a plan-representation question: should Node 16's internal multi-stage structure be visible in the execution plan, analogous to how `ReductionTreeNode` (ADR-003) exposes the Recursive Clip-Aggregation Engine's stages?

The question is sharpened by the contrast with Nodes 14, 15, and 20. Those nodes implement the *same mathematical operation* — a `log_K(N)` reduction with optional per-stage clipping — but are represented as `ReductionTreeNode`s whose internal structure (fan-in, stage count, threshold schedule, initial offset lists) is carried in a `ReductionTreePlan` and rendered by the backend's Orchestration tier. Node 16 performs the same reduction mathematics internally within a single kernel dispatch. The plan must decide which representation this kernel receives.

### The structural precondition: Node 13

Node 16's specialization is architecturally justified by its upstream dependency on Node 13 (`gather_and_permute_grad_h`), the canonical **Item Synchronization Point**. Node 13 reads from the scattered `Clipped_PARTIALS_Grad_H_AoS` collection and writes a single, contiguous `Permuted_Grad_H_SoA` buffer. This guarantee of contiguous input fundamentally changes the Orchestration/Execution tier boundary:

- **Host-orchestrated trees (Nodes 14, 15, 20):** Input partials are scattered across a collection buffer at placement-dependent offsets. The host must compute non-trivial offset lists, upload them to the device, allocate intermediate buffers, select kernel tiers, and synchronize between stages. These are Orchestration-tier responsibilities that require host participation at every stage.
- **Node 16:** Input is pre-gathered and contiguous. There are no scattered offsets to compute, no intermediate buffers to allocate cross-stage, and no kernel tier selection to make — the kernel's work-group size determines its own fan-in from `min(policy_max_k, get_local_size(0))`. The inter-stage logistics that necessitate host-driven orchestration for Nodes 14/15/20 simply do not exist.

### The three justifications from CONCEPT.md

CONCEPT.md §Kernel Inventory provides three justifications for Node 16's specialization, each of which maps to a plan-representation consequence:

1. **Algorithmic Fidelity.** The `Grad_H` vector is an intermediate error signal whose internal directionality must be preserved through a deliberate two-phase stabilization strategy: Phase I (Safety) at Node 11's leaf-level clip, Phase II (Fidelity) inside Node 16's staged reduction. The generic Recursive Clip-Aggregation Engine would subject this sensitive signal to a cascade of non-linear re-normalizations at the Orchestration tier's granularity, potentially distorting the error information. Node 16's internal reduction — operating on already-sanitized partials — minimizes directional distortion by keeping the entire reduction within a single computational context.

2. **Performance.** A host-orchestrated multi-stage loop over a contiguous buffer introduces per-stage dispatch overhead (kernel launch latency, event synchronization, buffer rebinding) that is entirely unnecessary when the kernel can manage its own internal stages using work-group synchronization barriers. Node 16 eliminates this overhead by collapsing the stage loop into intra-kernel control flow.

3. **Architectural Coherence.** The Recursive Clip-Aggregation Engine is a scatter-gather primitive: it uses an indirection table (`offset_list`) to sum partials at arbitrary memory locations. Applying this primitive to a pre-gathered, contiguous buffer is a logical contradiction — the indirection becomes a trivial identity mapping, and the host-driven logistics serve no purpose. Node 16 resolves this by operating directly on the contiguous layout that Node 13 guarantees.

---

## Decision Drivers

1. **ADR-001 (Three-tier jurisdictional model — Policy/Orchestration/Execution).** The plan boundary separates Policy from Orchestration. The kernel contract boundary (CONTRACT.md) separates Orchestration from Execution. The three-tier model explicitly accommodates tier collapse: when the host need not participate in inter-stage logistics, the Orchestration tier collapses into the Execution tier. ADR-001 §Three-tier jurisdictional model documents Node 16 as the canonical example of this collapse.

2. **ADR-001 (Plan boundary stops at the kernel's public interface).** ADR-001's design principle states: "Plan nodes describe kernel identity, logical buffer bindings, tile decomposition, dependency edges, and policy parameters. Internal kernel algorithms (e.g., Node 16's multi-stage reduction) are never expressed in the plan." This is a direct statement that Node 16's internals are plan-opaque.

3. **ADR-002 (Closed node type taxonomy — `KernelDispatchNode`).** ADR-002's node type taxonomy assigns Node 16 to `KernelDispatchNode`, noting: "Node 16 appears here despite its internal multi-stage reduction because ADR-001's plan boundary stops at the kernel's public interface — the Orchestration and Execution tiers collapse into a single kernel dispatch." The taxonomy is closed; Node 16 is already classified.

4. **ADR-003 (Kernel tier selection is an Orchestration-tier concern).** ADR-003 formally places the register-vs-local crossover heuristic in the Orchestration tier for `ReductionTreeNode`s. Node 16's internal kernel tier decisions go further — they belong to the *Execution* tier, since the kernel itself determines `K_plan = min(policy_max_k, get_local_size(0))` at runtime. This reinforces the opacity: if even the Orchestration tier cannot prescribe Node 16's internal kernel tier, the Policy tier (the plan) certainly cannot prescribe its stage structure.

5. **CONTRACT.md (Behavioral Invariants specify the internal algorithm).** The kernel contract in `kernels.cl.h` specifies Node 16's complete internal algorithm as a `Behavioral Invariants` block — the two-phase execution model (Pre-computation Phase → Per-Stage Execution), the fan-in derivation, and the per-stage threshold formula. This is an Execution-tier specification. The plan need not (and must not) duplicate it.

6. **CONCEPT.md §1 (Architectural Elegance Feedback).** Node 16's opacity is not a special case or an exception. It is a natural consequence of the tier model applied to a kernel whose upstream dependency (Node 13) eliminates the need for host-driven orchestration. If a future optimization were to require exposing Node 16's internal stages to the plan — e.g., for a backend that cannot support intra-kernel global barriers — the response would be to formalize this as a new plan representation (perhaps a `ReductionTreeNode` variant), not to silently break the opacity.

---

## Options Considered

### Option A: Expose Node 16's internal stages as a `ReductionTreeNode`

Treat Node 16 the same as Nodes 14, 15, and 20: represent it as a `ReductionTreeNode` with a `ReductionTreePlan` carrying the fan-in, stage count, threshold schedule, and initial offset list. The backend renderer drives the multi-stage loop.

**Advantages:**
- Uniform representation. All reductions in the plan use the same node type and rendering path.
- The plan carries a complete, inspectable record of Node 16's reduction parameters.
- Backends that cannot support intra-kernel global barriers (hypothetical) could render the tree externally.

**Disadvantages:**
- **Contradicts ADR-001's plan boundary principle.** The plan boundary stops at the kernel's public interface. Exposing Node 16's internal stages places the plan inside the kernel, violating the separation.
- **Destroys the Orchestration/Execution tier collapse.** Node 16's specialization *is* the collapse. Representing it as a `ReductionTreeNode` would re-introduce the Orchestration tier for a kernel that has demonstrated it does not need one — re-introducing per-stage dispatch overhead, inter-stage synchronization, and intermediate buffer allocation for a contiguous input that requires none of these.
- **Misapplies the scatter-gather primitive.** The `ReductionTreePlan`'s `initial_offset_list` exists because Nodes 14/15/20 reduce scattered partials. Node 16's input is contiguous — the offset list would be a trivial identity `(0, 1, ..., N-1)`, carrying zero information.
- **Defeats the algorithmic fidelity justification.** CONCEPT.md's Phase II (Fidelity) rationale for Node 16 is that the entire reduction operates within a single computational context, minimizing directional distortion. A host-driven multi-stage rendering fragments this context.
- **Fan-in is runtime-determined, not plan-determined.** Node 16 derives its fan-in at kernel launch time from `min(policy_max_k, get_local_size(0))`. The `ReductionTreePlan` prescribes fan-in as a Policy-tier value (`fan_in: int`). Carrying a fan-in in the plan that the kernel will override at runtime is misleading at best, incorrect at worst.

### Option B: Represent Node 16 as a single `KernelDispatchNode` with policy parameters (opacity)

The plan specifies Node 16 as a single `KernelDispatchNode`. Policy parameters (`T_algorithmic`, `λ`, `fp_max`, `policy_max_k`, `epsilon`) are carried in the node's `scalar_params` dict. The kernel's internal multi-stage reduction is an Execution-tier concern, documented in the kernel contract's `Behavioral Invariants` block in `kernels.cl.h`.

**Advantages:**
- **Respects the plan boundary.** The plan describes what it can: kernel identity, buffer bindings, dependency edges, and the Policy tier's constraint parameters. It does not describe what it cannot: the kernel's internal algorithm.
- **Preserves the tier collapse.** The Orchestration tier collapses into the Execution tier, as observed in the existing code and formalized in ADR-001.
- **Policy parameters are carried faithfully.** `T_algorithmic`, `λ`, `fp_max`, and `policy_max_k` are the shared layer's complete and sufficient output. The kernel contract specifies exactly how these are consumed (Behavioral Invariants §1–2 in `kernels.cl.h`). The plan carries the Policy tier's decision; the kernel contract specifies the Execution tier's algorithm; no Orchestration tier is needed.
- **Plan-time validation is complete.** The shared layer validates Node 16's `KernelContract` at plan-construction time: buffer shapes (the contiguous SoA buffer from Node 13), scalar parameter ranges (`policy_max_k >= 1`, `T_algorithmic >= 0`, `λ >= 0`, `fp_max > 0`), and dependency correctness (depends on `"item_sync_barrier"` / Node 13). This is the same validation applied to every other `KernelDispatchNode`. No additional reduction-specific validation is needed in the plan because the reduction invariants (stage count consistency, threshold monotonicity) are internal kernel concerns enforced by the kernel itself.
- **Compact representation.** Node 16 is a single plan node with a scalar parameter dict. No interim structures, no threshold schedules, no offset lists.

**Disadvantages:**
- **Reduction internals are opaque.** The plan does not carry Node 16's stage count, per-stage thresholds, or fan-in. These are not inspectable from the plan alone — they are specified in the kernel contract and resolved at kernel launch time.
- **Backend cannot override internal stages.** A hypothetical backend that cannot execute intra-kernel global barriers would not be able to render Node 16 from the plan. Per CONCEPT.md §1, this would trigger Architectural Elegance Feedback — the representation would be extended, not worked around.

---

## Analysis

### Why the asymmetry with `ReductionTreeNode` is correct

At first glance, representing one set of reductions (Nodes 14, 15, 20) as multi-stage `ReductionTreeNode`s and another (Node 16) as a single `KernelDispatchNode` appears inconsistent. The asymmetry is not incidental — it reflects a fundamental structural difference in where the Orchestration/Execution boundary falls.

**Nodes 14, 15, 20 — host orchestration is mandatory:**

| Orchestration responsibility | Why the host must participate |
| :--- | :--- |
| Initial offset list computation | Input partials are at placement-dependent, scattered offsets in the collection buffer. Only the shared layer knows the placement strategy. |
| Intermediate buffer allocation | Each stage produces intermediate results that the next stage consumes. The host allocates these buffers using its native allocator (e.g., `cl.Buffer`, `VkDeviceMemory`). |
| Kernel tier selection | The register-vs-local crossover depends on hardware-specific heuristics that differ per backend. |
| Inter-stage synchronization | Between stages, the host inserts its native synchronization primitive (event chain, pipeline barrier, implicit sequencing). |
| Offset list upload | Each stage's offset list must be transferred to device-accessible memory via the backend's native mechanism. |

Every one of these is an Orchestration-tier concern that requires host-side code. The `ReductionTreePlan` exists to carry the Policy tier's output (threshold schedule, fan-in, initial offsets) *across* the plan boundary to the Orchestration tier that will consume it.

**Node 16 — host orchestration is eliminated:**

| Orchestration responsibility | Why the host need not participate |
| :--- | :--- |
| Offset computation | Input is contiguous (Node 13 guarantee). No scatter-gather indirection needed. |
| Intermediate buffer allocation | The kernel uses work-group local memory (`update_buffer_LOCAL_reduction_tile`), allocated once at dispatch time via the standard `__local` mechanism. |
| Kernel tier selection | The kernel determines its own effective fan-in at runtime: `K_plan = min(policy_max_k, get_local_size(0))`. |
| Inter-stage synchronization | The kernel uses `barrier(CLK_LOCAL_MEM_FENCE)` (OpenCL), `groupMemoryBarrier()` (GLSL), or equivalent intra-work-group primitives. |
| Offset list upload | No offset list exists. |

With every Orchestration-tier responsibility eliminated, there is nothing for the plan to prescribe beyond the kernel's public interface. The plan boundary at the kernel's public interface is not an approximation for Node 16 — it is exact.

### The instructive contrast with ADR-003

ADR-003 defines the `ReductionTreePlan` as a frozen dataclass carrying the Policy tier's pre-computed output: fan-in `K`, stage count, initial offset list, tree variant, and threshold schedule. The motivating principle is that these values are **non-trivially computed by the shared layer** and must cross the plan boundary as data.

For Node 16, the analogous values — fan-in, stage count, and threshold schedule — are **not pre-computed by the shared layer**. They are resolved at kernel launch time by the kernel itself:

- **Fan-in:** `K_plan = min(policy_max_k, get_local_size(0))` — depends on the runtime work-group size, which the backend determines at dispatch time.
- **Stage count:** `num_stages = ceil(log(total_modules_count) / log(K_plan))` — depends on `K_plan`, which is runtime-determined.
- **Per-stage threshold:** `T_j = min(T_algorithmic + λ·j², fp_max / K_actual)` — depends on `K_actual` (the actual fan-in at each stage), which may differ from `K_plan` at the final stage.

The shared layer's contribution is limited to the Policy-tier constraint parameters: `T_algorithmic`, `λ`, `fp_max`, and `policy_max_k`. These are exactly the scalar parameters carried in the `KernelDispatchNode`. The `ReductionTreePlan` pattern does not apply because the specific values it would carry are not known until kernel execution — they are Execution-tier computations, not Policy-tier computations.

### Plan-time validation scope

Although Node 16's internal reduction is opaque, the plan builder still validates everything that *is* plan-level:

- **Buffer shape consistency:** The source buffer `Permuted_Grad_H_SoA` has shape `(total_batch_count × padded_hidden_count, padded_total_modules_count)`. This is validated against the upstream Node 13's output shape.
- **Destination buffer shape:** `Summed_Grad_H` has shape `(total_batch_count × padded_hidden_count)`. This is validated against downstream consumers (Nodes 17, 18).
- **Scalar parameter validity:** `policy_max_k >= 1`, `T_algorithmic >= 0`, `λ >= 0`, `fp_max > 0`, `epsilon > 0`, `total_modules_count >= 1`.
- **Dependency correctness:** Node 16 depends on `"item_sync_barrier"` (Node 13), ensuring the contiguous input is available.
- **`KernelContract` validation:** The contract's calculability proofs confirm that all buffer shapes are derivable from the scalar parameters (CONTRACT.md Article 1.4).

Internal reduction correctness — threshold monotonicity, safety ceiling compliance, stage count consistency — is governed by the kernel contract's `Behavioral Invariants` specification. The host's role is limited to supplying valid policy parameters; the kernel's contract guarantees correct internal behavior given valid inputs.

### The `policy_max_k` synthesis

A distinctive feature of Node 16's interface is the `src_scalar_NATURAL_policy_max_k` parameter, whose kernel contract documentation (`kernels.cl.h`) explicitly states:

> "The Host is contractually obligated to compute this value by synthesizing three distinct constraints and taking their minimum: (a) the user's desired reduction policy, (b) the physical hardware limits of the target device, (c) the absolute mathematical safety limit required to prevent signal annihilation."

This synthesis is a Policy-tier computation — it consumes user configuration, hardware profile data, and mathematical safety constraints — and its result is carried as a scalar parameter in the plan. The kernel does not receive a `min_threshold` parameter; instead, `policy_max_k` encodes the safety constraint implicitly. This is a deliberate architectural choice documented in the kernel contract: "This architectural choice delegates the responsibility for preventing signal annihilation to the Host, allowing the kernel to remain a more focused and efficient computational unit."

The `StabilizationPolicy` module in the shared layer is the sole authority for this synthesis. It is the same policy module that computes `ReductionTreePlan` parameters for Nodes 14/15/20. The difference is only in what crosses the plan boundary:

- **Nodes 14/15/20:** Policy outputs a complete threshold schedule and fan-in (`ReductionTreePlan`).
- **Node 16:** Policy outputs a single constraint parameter (`policy_max_k`), and the kernel internally applies the threshold formula using the other policy parameters (`T_algorithmic`, `λ`, `fp_max`).

---

## Decision

**Option B: Node 16 is a single `KernelDispatchNode` with policy parameters.**

The plan specifies Node 16 as a `KernelDispatchNode` with the following structure:

```python
KernelDispatchNode(
    node_id="stabilize_and_reduce_grad_hidden",
    depends_on=frozenset({"item_sync_barrier"}),
    kernel_identity="stabilize_and_reduce_grad_hidden_activations",
    buffer_bindings={
        "update_buffer_LOCAL_reduction_tile": "__local__",
        "src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa": "permuted_grad_h_soa",
        "dest_buffer_GLOBAL_summed_grad_hidden_activations": "summed_grad_hidden",
    },
    scalar_params={
        "src_scalar_REAL_fp_max": ...,                          # FP_FORMAT_MAX
        "src_scalar_REAL_policy_t_algorithmic": ...,            # T_algorithmic
        "src_scalar_REAL_policy_lambda": ...,                   # λ
        "src_scalar_NATURAL_policy_max_k": ...,                 # Synthesized K upper bound
        "src_scalar_REAL_epsilon": ...,                         # Numerical stability epsilon
        "src_scalar_NATURAL_total_batch_count": ...,            # N (batch size)
        "src_scalar_NATURAL_padded_hidden_count": ...,          # Padded hidden dimension
        "src_scalar_NATURAL_total_modules_count": ...,          # M (module count, unpadded)
        "src_scalar_NATURAL_padded_total_modules_count": ...,   # Padded module count
    },
    tile_count=1,
    placement_strategy=None,
    contract=...,  # KernelContract for Node 16
)
```

The kernel's internal multi-stage reduction — fan-in derivation, stage count computation, per-stage threshold calculation, intra-work-group synchronization — is an Execution-tier concern specified by the `Behavioral Invariants` block in the kernel contract (`kernels.cl.h`). The plan does not express, prescribe, or constrain this internal algorithm.

### The tier distribution for Node 16

| Tier | Responsibility | Owner |
| :--- | :--- | :--- |
| **Policy** | Compute `policy_max_k` (synthesizing user intent, hardware limits, mathematical safety). Supply `T_algorithmic`, `λ`, `fp_max`, `epsilon`. Validate buffer shapes and scalar parameter ranges. | Shared layer (plan builder, `StabilizationPolicy`) |
| **Orchestration** | Dispatch the single kernel. Determine work-group size. Bind buffers. Allocate local memory. Insert dependency-edge synchronization. | Backend renderer |
| **Execution** | Derive `K_plan = min(policy_max_k, get_local_size(0))`. Compute stage count. Execute `log_K(M)` reduction with per-stage `sum-then-clip` using the Quadratic Scaling Policy formula. | The kernel itself |

The Orchestration tier for Node 16 is minimal — a single kernel dispatch — because Node 13's contigity guarantee eliminates all inter-stage host logistics. This tier collapse defines Node 16's characteristic opacity.

### Invariant: the plan boundary is exact at the public interface

For Node 16, the plan carries everything the host contributes (policy parameters, buffer bindings, dependency edges) and nothing the kernel determines internally (fan-in, stage count, per-stage thresholds). This is not an under-specification — it is the *precise* boundary. The kernel contract's `Behavioral Invariants` specification completes the picture by documenting the Execution-tier algorithm that is invisible to the plan. Together, the plan and the kernel contract form a complete specification of Node 16's behavior across all three tiers.

---

## Consequences

### Positive

- **Faithful representation of the tier model.** Node 16's opacity in the plan directly reflects the Orchestration/Execution tier collapse documented in ADR-001. The plan does not pretend to control what the kernel manages internally.
- **Eliminates structural waste.** No threshold schedule, offset list, or stage descriptor is carried for a kernel that derives all of these at runtime. The plan is both smaller and more accurate.
- **Unified `KernelDispatchNode` rendering.** The backend renders Node 16 using the same `KernelDispatchNode` rendering path as Nodes 4–11, 17–19, 21, 24, and 25. No special-case rendering logic is needed for Node 16's internal complexity.
- **Policy parameters are validatable.** The shared layer validates `policy_max_k`, `T_algorithmic`, `λ`, and `fp_max` at plan-construction time. The kernel contract guarantees that valid inputs produce correct internal behavior. The two-tier validation is complete without either tier knowing the other's internals.
- **ADR-003 contrast is instructive.** The co-existence of `ReductionTreeNode` (plan-visible stages for Nodes 14/15/20) and `KernelDispatchNode` (plan-opaque stages for Node 16) serves as a teaching example: the plan exposes exactly as much internal structure as the Orchestration tier needs and no more. Where the host must drive the stage loop, the plan carries the stage data. Where the kernel drives its own loop, the plan carries only the constraint parameters.
- **Preserved kernel contract fidelity.** The `Behavioral Invariants` specification in `kernels.cl.h` is the authoritative record of Node 16's internal algorithm. This specification is language-neutral in substance (it defines the mathematical formula, not OpenCL-specific syntax), supporting faithful reimplementation across GLSL and C backends per ADR-013.

### Negative

- **Internal reduction is not plan-inspectable.** Unlike `ReductionTreeNode`s, Node 16's stage count, per-stage thresholds, and actual fan-in cannot be extracted from the plan. They are recoverable from the kernel contract's `Behavioral Invariants` given the scalar parameters, but this requires interpreting the contract specification — it is not machine-readable plan data.
- **Backend cannot override internal stages.** If a future backend cannot support intra-kernel global barriers (e.g., a hypothetical WASM backend with no work-group synchronization), it cannot render Node 16 from a `KernelDispatchNode`. Per CONCEPT.md §1 (Architectural Elegance Feedback), this would trigger a formal extension — likely a new `ReductionTreeNode` variant for the `Grad_H` path — not an ad-hoc workaround. The extension would replace Node 16's `KernelDispatchNode` with a `ReductionTreeNode` carrying a `ReductionTreePlan`, effectively re-introducing the Orchestration tier for that backend. This is a clean evolution path, not a limitation.
- **Apparent asymmetry with Nodes 14/15/20.** The fact that some reductions are plan-visible and others are plan-opaque requires explanation. This ADR provides that explanation: the asymmetry is architecturally principled, following from the presence or absence of host-driven orchestration, but it must be understood by anyone reading the plan.

### Relationship to ADR-016 (Test Strategy)

Node 16's opacity affects the test strategy (ADR-016):

- **Layer 1 (Plan tests):** Node 16 is tested as a `KernelDispatchNode` — buffer shape consistency, scalar parameter validity, dependency correctness, and `KernelContract` validation. No reduction-specific invariants (threshold monotonicity, stage count consistency) are testable at the plan level for Node 16 because these are Execution-tier concerns.
- **Layer 2 (Backend unit tests):** Each backend validates that its dispatch of Node 16 produces numerically correct output — the summed `Grad_H` matches a reference computation. This implicitly validates the kernel's internal reduction correctness.
- **Layer 3 (Cross-backend integration tests):** Node 16's output is included in the numerical-equivalence comparisons across backends. Since the kernel internally determines its fan-in from `min(policy_max_k, get_local_size(0))`, and work-group sizes may differ across backends, the exact reduction order may differ — but the result must be equivalent within precision-aware tolerances.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — Plan boundary at the kernel's public interface; three-tier jurisdictional model; tier collapse for specialized kernels
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode` taxonomy; Node 16 classification; closed taxonomy constraint
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `ReductionTreePlan` for host-orchestrated Nodes 14/15/20; kernel tier selection placed in Orchestration tier; parametric header pattern
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — Stride-based parametric specification; reinforcement of the two-tier buffer scope
- [CONCEPT.md](../CONCEPT.md) — §Kernel Inventory (Node 16 specialization justification, three-part rationale); §1 Architectural Elegance Feedback; §2 Recursive Clip-Aggregation Engine; §3 Gradient Stabilization (Quadratic Scaling Policy)
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; kernel contract parameter grammar
- [kernels.cl.h](../kernels/kernels.cl.h) — Node 16 kernel contract: `Behavioral Invariants` specification (Pre-computation Phase, Per-Stage Execution), `policy_max_k` synthesis obligation
- [ADR_PLAN.md](../ADR_PLAN.md) — ADR-005 problem statement and resolution summary
