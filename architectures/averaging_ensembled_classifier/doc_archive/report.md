# Integrated Recommendation: Host-Prescribed Threshold Schedule for Node 16

## 1. Problem Statement

Node 16's contract (§0–§2 of its Behavioral Invariants) specifies a three-phase internal algorithm: the kernel computes a reduction tree topology and threshold schedule from policy parameters (`T_algorithmic`, `λ`, `policy_max_k`), then executes it. However, GPU backends implement a "work-group per row" pre-accumulation strategy that collapses M input elements to P = workgroup_size intermediates *before* the staged tree begins. This creates three contradictions:

1. **Stage count mismatch.** The contract computes `num_stages = ⌈log(M)/log(K)⌉` but the staged tree operates on only P values, requiring `⌈log(P)/log(K)⌉` stages.
2. **Group composition divergence.** Strided thread assignment means stage 0's K-way groups contain elements drawn from across the abstract tree's partition, rendering per-group policy thresholds structurally non-equivalent.
3. **Policy elision.** The pre-accumulation receives only a safety clip (`fp_max / P`), silently discarding the Quadratic Scaling Policy thresholds intended for the deepest tree stages.

These contradictions arise because the CONCEPT doc acknowledges the pre-accumulation as an "Execution-tier concern" (§3.4) while the contract declares the staged model as "contractually bound internal logic" — an incompatibility between approximate and exact conformance requirements.

## 2. Decision

Move threshold schedule computation from the Execution tier (inside the kernel) to the Orchestration tier (host-side rendering). The kernel receives a pre-computed threshold schedule and applies it. The Policy tier specifies the Quadratic Scaling Policy parameters in the plan; the Orchestration tier renders a backend-appropriate schedule accounting for the dispatch topology; the kernel executes the schedule without independent policy computation.

## 3. Rationale

| Principle | Alignment |
|:----------|:----------|
| **§1 (Architectural Elegance Feedback)** | The pre-accumulation was an optimization that violated the contract's staged model. Principle 1's directive — suspend, formalize, reify — is followed: the pre-accumulation becomes a first-class parameter in the rendering contract, and the schedule is reified through a revised kernel interface. |
| **§3 (Dumb Kernels)** | Node 16 is currently the smartest kernel in the system: it computes tree topology, stage counts, and threshold schedules internally. This change makes it a parametric executor — it receives a schedule and applies it. |
| **§5 (Three-Tier Jurisdictional Model)** | Policy computation (threshold formulas) moves to where all other policy computation lives: the shared Policy tier and per-backend Orchestration tier. The kernel retains sole control of execution geometry (thread assignment, binary tree structure, local memory usage). |
| **Consistency** | Nodes 14/15/20 already receive host-computed thresholds (`clipping_threshold_t_j`). Node 16 was the sole exception. This elimination of the exception unifies the reduction engine's threshold injection pattern. |
| **Cross-backend transparency** | The structural difference between CPU (P = M, full-depth tree) and GPU (P < M, shallower tree with pre-accumulation) becomes visible in the Orchestration tier's rendering output — inspectable, testable, and tolerance-bounded — rather than hidden inside opaque kernel internals. |

## 4. CONCEPT.md Amendments

### 4.1. §3.4 — Execution-Tier Internal Amplification

**Replace** the paragraph beginning "GPU backends (OpenCL, Vulkan) implement Node 16…" with:

> **Execution-Tier Internal Amplification.** GPU backends (OpenCL, Vulkan) implement Node 16 using a "work-group per row" strategy where each thread first accumulates `⌈total_modules/workgroup_size⌉` elements before the staged reduction begins. The Orchestration tier accounts for this pre-accumulation when rendering Node 16's threshold schedule: it computes a pre-accumulation threshold from `compute_fp_format_max` and the dispatch workgroup size, and computes the staged reduction schedule over the `min(total_modules_count, workgroup_size)` post-accumulation intermediates. The kernel applies these host-prescribed thresholds without independent policy computation, consistent with the threshold injection pattern used by the generic reduction engine (Nodes 14/15/20).

### 4.2. §5 — Three-Tier Jurisdictional Model

**Replace** the sentence:

> Node 16's internal multi-stage reduction exemplifies the Orchestration/Execution tier collapse: when the upstream Item Synchronization Point (Node 13) guarantees contiguous input, the host-driven inter-stage logistics that necessitate Orchestration-tier participation for Nodes 14/15/20 are eliminated, and the kernel manages its own reduction internally.

**With:**

> Node 16 receives its threshold schedule from the Orchestration tier, consistent with the threshold injection pattern used by the generic reduction engine (Nodes 14/15/20). The kernel's specialization lies in its optimized single-launch reduction over contiguous SoA data — guaranteed by the upstream Node 13 synchronization point — and its row-wise reduction topology, not in independent policy computation. The Orchestration tier adapts the Policy tier's Quadratic Scaling Policy parameters to the backend's dispatch topology (e.g., accounting for GPU workgroup-size-limited pre-accumulation), then prescribes the concrete schedule.

### 4.3. Node 16 Description (Kernel & Synchronization Contracts Section)

**Replace** the description of Node 16 with:

> - **(16) `stabilize_and_reduce_grad_hidden_activations`**: **[Specialized Kernel]** A parametric reduction engine that applies a host-prescribed stabilization schedule to the `Grad_H` vector.
>   - **Contract:** Receives a pre-computed threshold schedule and pre-accumulation threshold from the Orchestration tier. Executes a multi-stage `sum-then-clip` reduction over the contiguous SoA buffer produced by Node (13), applying the prescribed threshold at each clip stage. The schedule encodes the **Quadratic Scaling Policy**, rendered by the Orchestration tier for the target backend's reduction topology.
>   - **Justification for Specialization:** This kernel works in synergy with the mandatory leaf-level clipping from Node (11) to provide a complete, two-stage stabilization strategy. A generic engine is unsuitable because:
>     1. **Algorithmic Fidelity:** The `Grad_H` vector is an intermediate error signal whose internal directionality must be preserved. Node (11) provides Phase I (Safety) coarse clipping; this kernel provides Phase II (Fidelity) by carefully aggregating the sanitized vectors with staged `sum-then-clip` using the prescribed schedule.
>     2. **Performance:** It is optimized for the contiguous data block guaranteed by the **`(13) Item Synchronization Point`**, avoiding the latency and overhead of a host-driven recursive engine.
>     3. **Architectural Coherence:** It resolves the logical conflict of applying a scatter-gather primitive to a pre-gathered, contiguous buffer.
>     4. **Pattern Consistency:** It receives host-prescribed thresholds in the same manner as the generic reduction engine (Nodes 14/15/20), eliminating the former jurisdictional exception where this kernel computed its own policy schedule.

### 4.4. §11 — Reduction Planning & Rendering

**Append** to point 4 ("Reduction Planning & Rendering"):

> For Node 16, the Orchestration tier additionally renders a per-stage threshold schedule and pre-accumulation threshold, adapted to the backend's dispatch topology. The CPU backend renders the abstract `⌈log_K(M)⌉`-stage schedule directly (no pre-accumulation). GPU backends render a `⌈log_K(P)⌉`-stage schedule over `P = min(M, workgroup_size)` post-accumulation intermediates, with a pre-accumulation threshold of `compute_fp_format_max / K`. The rendered schedule is passed to the kernel as a buffer parameter, paralleling the per-stage threshold scalars passed to the generic `reduce_k_fan_in_and_clip` kernels.

## 5. Revised Kernel Interface

### 5.1. Parameters Removed

| Parameter | Reason |
|:----------|:-------|
| `src_scalar_REAL_fp_max` | Safety ceilings are now pre-computed into the threshold schedule by the host. |
| `src_scalar_REAL_policy_t_algorithmic` | Policy computation moved to the Orchestration tier. |
| `src_scalar_REAL_policy_lambda` | Policy computation moved to the Orchestration tier. |
| `src_scalar_NATURAL_policy_max_k` | Tree topology computation moved to the Orchestration tier. |

### 5.2. Parameters Added

| Parameter | Type | Role |
|:----------|:-----|:-----|
| `src_buffer_GLOBAL_CONST_clipping_threshold_per_stage` | `COMPUTE_TYPE[num_reduction_stages]` | Pre-computed threshold schedule, one entry per staged clip point, indexed in execution order (0 = first clip after pre-accumulation, `num - 1` = root). |
| `src_scalar_NATURAL_num_reduction_stages` | `uint` | Number of clip stages in the staged reduction. Zero when `total_modules_count ≤ 1`. |
| `src_scalar_REAL_clipping_threshold_t_pre` | `COMPUTE_TYPE` | Clip threshold for each thread's pre-accumulation phase. Set to `compute_fp_format_max / K` by the Orchestration tier when pre-accumulation occurs; set to `compute_fp_format_max` when it does not. |

### 5.3. Full Revised Declaration

```c
/**
 * @brief (Node 16) Specialized Kernel: Reduces the permuted Grad_H buffer
 *        using a host-prescribed, multi-stage, numerically-stable reduction.
 * @kernel_contract
 *        - Holistic Constraints: "This kernel performs a complete, row-wise
 *          reduction on the monolithic, contiguous SoA buffer produced by the
 *          upstream Item Synchronization Point (Node 13). The stabilization
 *          schedule is prescribed by the Orchestration tier; the kernel applies
 *          it without independent policy computation."
 *        - Behavioral Invariants: "Precision Boundary Conversion: storage-role
 *          inputs widened via load_storage(); reduction accumulation in
 *          COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE.
 *          The kernel's behavior is contractually bound to:
 *
 *          1. **Pre-accumulation Phase:**
 *             Each thread accumulates
 *             ceil(total_modules_count / get_local_size(0)) elements from its
 *             assigned row. Each thread's accumulator is clipped to
 *             src_scalar_REAL_clipping_threshold_t_pre after accumulation.
 *             When total_modules_count <= get_local_size(0), each thread loads
 *             at most one element and the clip is a no-op (the host sets the
 *             threshold sufficiently high).
 *
 *          2. **Staged Reduction Phase:**
 *             num_reduction_stages rounds of work-group-parallel tree reduction.
 *             After each round's summation, the per-element result is clipped
 *             to src_buffer_GLOBAL_CONST_clipping_threshold_per_stage[s]
 *             (where s indexes the round from 0 to num_reduction_stages - 1).
 *             The kernel determines the internal binary tree structure and
 *             fan-in distribution among stages.
 *             Thread 0 writes the final single-element result to the
 *             destination buffer.
 *
 *          When num_reduction_stages is 0, the kernel bypasses the Staged
 *          Reduction Phase entirely: pre-accumulation produces at most one
 *          value per row, and thread 0 writes it directly."
 *        - Synchronization Model: "Specialized Reduction Kernel / Global
 *          Barrier"
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Architectural Justification: "This kernel is a specialized,
 *          single-launch reduction engine optimized for the contiguous SoA
 *          buffer produced by the upstream Node 13 synchronization point.
 *          The Orchestration tier prescribes the threshold schedule, and the
 *          kernel applies it — consistent with the threshold injection pattern
 *          used by the generic reduction engine (Nodes 14/15/20). The kernel
 *          retains sole control of its internal execution geometry (thread
 *          assignment, pre-accumulation fan-in, binary tree structure), while
 *          the Host retains sole control of stabilization policy through the
 *          prescribed schedule."
 */
__kernel void stabilize_and_reduce_grad_hidden_activations(
    /**
     * @param update_buffer_LOCAL_reduction_tile Work-group exclusive memory
     *        for high-bandwidth parallel reduction.
     *        - Tensor Shape: (get_local_size(0))
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute" (LOCAL scratch)
     *        - Calculability Proof: [Implicit from work-group dispatch]
     *        - Validation Preconditions: Host shall allocate local memory equal
     *          to the work-group size in dimension 0 multiplied by
     *          sizeof(COMPUTE_TYPE).
     */
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,

    /**
     * @param src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa
     *        The pre-gathered, contiguous input data in SoA layout.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count,
     *          src_scalar_NATURAL_padded_total_modules_count)
     *        - Padding Contract: {
     *            dim[0] ("total_batch_count * hidden_count" →
     *              "total_batch_count * padded_hidden_count"):
     *              {Type: CACHE, Formula: "128-byte alignment on
     *              hidden_count stride"},
     *            dim[1] ("total_modules_count" →
     *              "padded_total_modules_count"):
     *              {Type: CACHE, Formula: "128-byte alignment"}
     *          }
     *        - Precision Role: "storage"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count,
     *          src_scalar_NATURAL_padded_total_modules_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [(src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count) *
     *          src_scalar_NATURAL_padded_total_modules_count *
     *          sizeof(STORAGE_TYPE)] bytes. This buffer must be fully
     *          populated by Node 13 before dispatch.
     */
    __global const STORAGE_TYPE
        *src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa,

    /**
     * @param dest_buffer_GLOBAL_summed_grad_hidden_activations The
     *        destination for the final, summed hidden layer gradient vector.
     *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count,
     *          src_scalar_NATURAL_padded_hidden_count]
     *        - Validation Preconditions: Host shall allocate exactly
     *          [(src_scalar_NATURAL_total_batch_count *
     *          src_scalar_NATURAL_padded_hidden_count) *
     *          sizeof(COMPUTE_TYPE)] bytes.
     */
    __global COMPUTE_TYPE
        *dest_buffer_GLOBAL_summed_grad_hidden_activations,

    /**
     * @param src_buffer_GLOBAL_CONST_clipping_threshold_per_stage
     *        Host-prescribed threshold schedule for the staged reduction.
     *        Entry s (0-indexed) is the clip threshold applied after staged
     *        reduction round s. Index 0 is the first clip after
     *        pre-accumulation (leaf); index num_reduction_stages - 1 is the
     *        final clip (root). Each entry is the min of the Quadratic
     *        Scaling Policy threshold and the safety ceiling for that stage,
     *        pre-computed by the Orchestration tier.
     *        - Tensor Shape:
     *          (src_scalar_NATURAL_num_reduction_stages)
     *        - Padding Contract: {Type: NONE}
     *        - Precision Role: "compute"
     *        - Calculability Proof:
     *          [src_scalar_NATURAL_num_reduction_stages]
     *        - Validation Preconditions: [1] Host shall allocate exactly
     *          [src_scalar_NATURAL_num_reduction_stages *
     *          sizeof(COMPUTE_TYPE)] bytes. [2] When
     *          num_reduction_stages == 0, the Host MAY pass a minimal stub
     *          buffer. [3] All entries must be positive real numbers.
     *          [4] The schedule must be monotonically non-increasing
     *          (schedule[0] >= schedule[1] >= ... >= schedule[num-1]),
     *          reflecting the Quadratic Scaling Policy's funnel property.
     */
    __global const COMPUTE_TYPE
        *src_buffer_GLOBAL_CONST_clipping_threshold_per_stage,

    /**
     * @param src_scalar_NATURAL_num_reduction_stages Number of clip stages
     *        in the staged reduction. Zero when total_modules_count <= 1.
     *        - Validation Preconditions: [1] Must be 0 when
     *          total_modules_count <= 1. [2] Must be >= 1 when
     *          total_modules_count > 1.
     */
    uint src_scalar_NATURAL_num_reduction_stages,

    /**
     * @param src_scalar_REAL_clipping_threshold_t_pre Clip threshold for
     *        each thread's pre-accumulation phase. Set to
     *        compute_fp_format_max / K by the Orchestration tier when
     *        pre-accumulation occurs (total_modules_count >
     *        dispatch workgroup size); set to compute_fp_format_max when
     *        each thread loads at most one element.
     *        - Validation Preconditions: Must be a positive real number.
     */
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold_t_pre,

    /**
     * @param src_scalar_REAL_epsilon Small constant to prevent division by
     *        zero during norm calculation.
     *        - Validation Preconditions: Must be a small, positive real
     *          number (e.g., 1e-6).
     */
    COMPUTE_TYPE src_scalar_REAL_epsilon,

    uint src_scalar_NATURAL_total_batch_count,
    uint src_scalar_NATURAL_padded_hidden_count,
    uint src_scalar_NATURAL_total_modules_count,
    uint src_scalar_NATURAL_padded_total_modules_count);
```

## 6. Orchestration-Tier Rendering Contract

The Orchestration tier renders Node 16's threshold schedule from the Policy tier's Quadratic Scaling Policy parameters, adapted to the backend's dispatch topology. The following algorithm is the recommended rendering procedure.

### 6.1. Inputs

| Source | Parameter | Symbol |
|:-------|:----------|:-------|
| Policy tier | Algorithmic threshold | `T_alg` |
| Policy tier | Scale parameter | `λ` |
| Policy tier | Maximum fan-in | `max_fan_in` |
| Policy tier | Compute format maximum | `fp_max` |
| Policy tier | Epsilon | `ε` |
| Orchestration tier | Total module count | `M` |
| Orchestration tier | Dispatch workgroup size | `W` |

### 6.2. Algorithm

```
1.  P ← min(M, W)                          // active threads after pre-accumulation
2.  K ← min(max_fan_in, P)                 // reduction fan-in per clip stage
3.  K ← max(K, 2)                          // minimum valid fan-in

4.  IF P ≤ 1:
        num_reduction_stages ← 0
        pre_accum_threshold  ← fp_max       // no pre-accumulation, no clip
        threshold_schedule   ← []            // empty buffer
        RETURN

5.  num_reduction_stages ← ⌈log(P) / log(K)⌉

6.  IF M > P:                               // pre-accumulation occurs
        pre_accum_threshold ← fp_max / K    // ensures K-way first-stage sum ≤ fp_max
    ELSE:
        pre_accum_threshold ← fp_max        // each thread loads ≤ 1 element

7.  FOR s IN 0 .. num_reduction_stages - 1:
        j ← num_reduction_stages - 1 - s    // leaf = highest j, root = j=0
        T_policy ← T_alg + λ × j²
        T_safety ← fp_max / K
        threshold_schedule[s] ← min(T_policy, T_safety)
```

**Key properties:**

- **CPU backend:** `W = M` (no workgroup limitation), so `P = M`, `pre_accum_threshold = fp_max` (step 6 ELSE branch), and the schedule spans the full `⌈log_K(M)⌉` abstract tree. The CPU rendering is identical to the former internal computation.

- **GPU backend:** `P = W < M`, yielding a shallower tree (`⌈log_K(P)⌉ < ⌈log_K(M)⌉` stages). The Quadratic Scaling Policy's funnel opens less widely — the leaf threshold is `T_alg + λ × (num_reduction_stages - 1)²` versus the abstract `T_alg + λ × (⌈log_K(M)⌉ - 1)²`. This is the *explicit, inspectable* consequence of pre-accumulation: the GPU's shallower tree produces a narrower policy funnel, which is more conservative (clips earlier). Users who require parity across backends should increase `λ` proportionally to restore the funnel width, or use the CPU backend as the precision reference.

- **Pre-accumulation threshold derivation:** When pre-accumulation occurs, each thread's accumulator enters the first K-way summation. The bound `K × pre_accum_threshold ≤ fp_max` prevents overflow at the first clip stage. The recommended value `fp_max / K` is the tightest sufficient bound. An alternative conservative value `fp_max / P` (the current contract's Data Ingress Safety Invariant) is also valid and provides additional headroom.

- **Monotonicity:** The `j²` term ensures `threshold_schedule[0] ≥ threshold_schedule[1] ≥ … ≥ threshold_schedule[num-1]` when `T_safety` does not dominate. When `T_safety < T_policy` at some stage, the schedule flattens at `T_safety` for those stages. The monotonicity validation precondition in §5.3 catches any host miscalculation.

### 6.3. Alternative: Abstract-Depth Anchoring

As a future extension, an Orchestration tier may choose to anchor `j` values at the abstract tree depth (`⌈log_K(M)⌉`) rather than the physical tree depth. This widens the funnel on GPU backends to match the CPU's schedule at the cost of a looser pre-accumulation threshold. The rendering algorithm becomes:

```
    abstract_depth ← ⌈log(M) / log(K)⌉
    stage_offset   ← abstract_depth - num_reduction_stages
    FOR s IN 0 .. num_reduction_stages - 1:
        j ← abstract_depth - 1 - s          // anchored to abstract depth
        threshold_schedule[s] ← min(T_alg + λ × j², fp_max / K)
```

This is a valid Orchestration-tier rendering choice. The kernel's interface and contract are unchanged — only the schedule values differ.

## 7. Lexicon Amendment (CONTRACT.md Article 8)

### 7.1. §4.0 — Context Modifiers

**Add one suffix entry:**

| Type | Term | Function |
|:-----|:-----|:---------|
| Suffix | `_per_stage` | Denotes a per-stage parameterization of a scalar quantity, providing one value per reduction tree stage rather than a single global scalar. |

This follows the existing `_per_item` and `_per_chunk` suffix conventions. The constituent term `stage` is already defined in §5.0.

## 8. Impact Assessment

### 8.1. Faithful Oracle

The oracle's `_staged_reduce_node16` currently computes thresholds internally using the same formula the kernel uses. Under this recommendation:

- The oracle should compute the schedule using the §6.2 algorithm with `W = M` (CPU-equivalent rendering), producing the same thresholds as before.
- No behavioral change for the oracle. The internal computation was already correct for the CPU case.
- Cross-backend oracle comparison should use the CPU rendering as the reference, with GPU results checked within a tolerance bound derived from the funnel width difference.

### 8.2. Validation Scenarios

| Scenario | Impact |
|:---------|:-------|
| **The Cross-Backend Arbiter** | The structural difference between CPU and GPU schedules is now *explicit* in the rendered plans. Tolerance bounds for Node 16 parity can be derived from the known funnel width difference: `Δ_max = λ × ((⌈log_K(M)⌉ - 1)² - (⌈log_K(P)⌉ - 1)²)`. |
| **The Rodeo** | The pre-accumulation threshold is now a host-prescribed parameter, directly testable. The test can verify that `pre_accum_threshold × K ≤ fp_max` for the dispatched configuration. |
| **The Colossus** | No change to reduction correctness. The schedule is computed once at plan construction time; no additional per-batch overhead. |
| All others | No impact. The policy parameters (`T_algorithmic`, `λ`) are unchanged. The schedule is computed at plan construction, not per-batch. |

### 8.3. Existing ADR Chain

- **ADR-005** (if it mandates workgroup size opacity): The Orchestration tier already knows the workgroup size for dispatch geometry. This recommendation uses the same value for threshold computation — no new information crosses the tier boundary.
- **ADR-019** (K-fan-in reduction primitive): Unaffected. The generic `reduce_k_fan_in_and_clip` kernel already receives host-computed thresholds.
- **ADR-020, ADR-023, ADR-025** (precision model): Unaffected. The threshold buffer uses `COMPUTE_TYPE`, consistent with all other threshold parameters.

### 8.4. Interface Surface Change

| Metric | Before | After | Net |
|:-------|:-------|:------|:----|
| Scalar parameters | 11 | 8 | −3 |
| Buffer parameters | 3 | 4 | +1 |
| Total parameters | 14 | 12 | −2 |
| Policy logic in kernel | ~20 lines | 0 lines | Eliminated |

The kernel becomes simpler. The Orchestration tier gains ~15 lines of schedule computation — logic that already exists in the host for Nodes 14/15/20 and can be shared.
