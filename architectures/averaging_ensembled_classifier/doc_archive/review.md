# Gradient Flow Review

## TL;DR

The mathematical backpropagation chain is **correct**. The five parameter-gradient paths and the intermediate Grad_H signal all trace valid derivative chains from loss to parameter update. The precision boundary conversions are consistently applied, and the multi-stage clipping strategy is coherently structured. I found **no correctness bugs in the kernel contracts**, but I identified several architectural observations — one critical **orchestration-tier constraint** that is implicit but not explicitly documented, and a few design tensions worth surfacing.

---

## 1. Verified: The Mathematical Chain

I traced every gradient from its source at the loss to its destination at `adam_update`. The decomposition is:

```
L = Σ_m L_m(softmax(H @ W_m / T_m + B_m), y)
H = ReLU(X @ W_s + B_s)
```

| Parameter | Gradient Formula | Production Node(s) | Correct? |
|:---|:---|:---|:---|
| **W_module** | `Σ_b (error_b ⊗ h_b)` | Node 8 (batch-reduced per tile) | ✓ |
| **B_module** | `Σ_b error_b` | Node 8 (batch-reduced per tile) | ✓ |
| **T_m** | `Σ_{b,c} ∂L/∂T_m` | Node 10 (batch-reduced per tile) | ✓ |
| **Grad_H** | `Σ_c (error_bc · W_m[h][c])` per sample | Node 9 (per-sample, per-tile) | ✓ |
| **W_shared** | `Σ_b (x_b ⊗ (Grad_H_b ⊙ relu'(h_b)))` | Node 17 (per batch chunk) | ✓ |
| **B_shared** | `Σ_b (Grad_H_b ⊙ relu'(h_b))` | Node 18 (per batch chunk) | ✓ |

### Key correctness observations:

**ReLU derivative placement is correct.** Node 9 computes `dL/dH_post_relu` — i.e., the gradient with respect to the hidden activations *after* ReLU —without applying the ReLU derivative. The derivative is deferred to Nodes 17/18, which recompute `relu'(h) = (h > 0)` from stored activations. This is mathematically equivalent to applying it before the module summation (by linearity of `Σ_m`), and the architecture takes advantage of this to avoid passing the ReLU mask through the Grad_H reduction path.

**Class-chunk summation in Node 13 is correct.** Each tile from Node 9 contributes `Σ_{c ∈ chunk} error_bc · W_m[h][c]`. Node 13 sums over `num_class_chunks`, reconstructing the full `Σ_c` sum per (module, sample, hidden) before Node 16 reduces over modules.

**ZERO_REQUIRED initialization on Node 8's output is correct and necessary.** Each tile writes only `classes_per_chunk` positions within the `padded_total_output_class_count` width. The zeros in non-active positions contribute nothing to Node 11's L2 norm (`0² = 0`) and nothing to the downstream reduction sum. The sparse non-overlapping write pattern plus element-wise summation correctly reconstructs the full gradient matrix.

**Normalization order is correct.** Node 21 divides by `effective_batch_size = Σ(sample_mask)`, converting batch-wide sums to averages. This occurs after all reductions are complete, yielding the correct mean gradient regardless of which components were batch-reduced internally (Nodes 8/10) vs. reduced by the engine (Nodes 15/20).

---

## 2. Verified: Precision Flow

I traced the precision role of every buffer through each gradient path. The pattern is consistent:

```
Gradient production → STORAGE
                ↓ (load_storage → COMPUTE)
Node 11 clip   → STORAGE
                ↓ (load_storage → COMPUTE)
Node 13/15 reduction tree → COMPUTE (at each interior stage)
                ↓
Node 16/21/24  → COMPUTE/ACCUM_TYPE as appropriate
```

**Summary table of precision transitions for Grad_H:**

| Stage | Input Role | Arithmetic | Output Role | Quantization? |
|:---|:---|:---|:---|:---|
| Node 9 produce | — | COMPUTE | storage | ① |
| Node 11 clip | storage → compute | COMPUTE | storage | ② |
| Node 13 gather | storage → compute | COMPUTE | storage | ③ |
| Node 16 reduce | storage → compute | COMPUTE | compute | — |

Three storage round-trips (①②③) before the value reaches stable COMPUTE precision. For FP32/FP16, this is well-tolerated. For **FP8 E4M3** (3-bit mantissa, ~12.5% relative error per round), the cumulative error through three quantization steps is a concern — I expand on this in §5.2 below.

**Adam's ACCUM_TYPE usage is correct.** The EMA updates (`β₁m + (1−β₁)g`) use `ACCUM_TYPE = max(COMPUTE_TYPE, STATE_TYPE)`, preserving state precision. The bias correction and delta computation use `COMPUTE_TYPE` (transformative, not accumulative). The parameter subtraction `θ ← θ − δ` uses `ACCUM_TYPE` to preserve θ's existing precision bits during the subtraction. This is the correct pattern: widening for accumulation, standard precision for transformation.

---

## 3. Verified: The Stabilization Pipeline

The gradient stabilization operates as a coherent multi-stage funnel:

```
              Node 11:  Partial-Group-Wise clip per tile (T_pre)
                 │
                 ├── Grad_ModW/ModB/Temps → Node 15: Component-Wise
                 │                           reduction with Quadratic
                 │                           Scaling Policy per stage
                 │
                 └── Grad_H → Node 13 (sum over class chunks, amplification = C)
                               → Node 16: Internal multi-stage reduction
                                  with Quadratic Scaling Policy
                                  (safety ceiling accounts for C×T_pre)

              Node 17/18 → Node 19: Partial-Group-Wise clip per batch chunk
                             → Node 20: Component-Wise reduction
                                with Quadratic Scaling Policy
```

**Pre-summation amplification is correctly accounted for.** The CONCEPT.md specifies `T_pre ≤ COMPUTE_FP_FORMAT_MAX / num_class_chunks`, and Node 16's safety ceiling formula uses `A_j = num_class_chunks`. The Data Ingress Safety Invariant in Node 16's contract (`clip to fp_max / F_total` before first reduction stage) handles the execution-tier amplification from workgroup-level accumulation. This is a complete overflow-prevention chain.

**The threshold schedule is correctly linearized.** Node 16's contract resolves the apparent circularity by mandating a fixed two-phase execution: first compute `K_plan = min(policy_max_k, workgroup_size)` and `num_stages`, then execute each stage with pre-determined thresholds. No circular dependency exists.

---

## 4. Identified Concerns

### 4.1 CRITICAL (Orchestration-Tier): Module-Chunk Grouping in Reduction Trees

**The kernel contracts are correct, but they contain an implicit, undocumented orchestration constraint.**

When `num_module_chunks > 1`, the collection buffers for Grad_ModW, Grad_ModB, and Grad_Temps have `partial_width = modules_per_chunk × ...` The reduction engine sums these partials element-wise. Tiles from the **same** `module_chunk` but different `class_chunk` values write non-overlapping class positions — summing them correctly fills the class dimension. But tiles from **different** `module_chunk` values write gradient values for **different physical parameters** at the **same positional indices** within their `modules_per_chunk`-wide slice.

If the Policy tier constructs an offset list that mixes tiles from different module_chunks into a single reduction node, the element-wise sum **conflates gradients for different parameters**. This produces silent, catastrophic corruption.

**The constraint:** For Grad_ModW, Grad_ModB, and Grad_Temps reductions (Node 15), each reduction tree must contain tiles exclusively from a **single `module_chunk`**. The offset lists must encode this grouping.

This constraint is implicit in the Indirection Contract's design ("the Host Orchestrator constructs the offset list"), but it is not stated as an explicit invariant anywhere in the CONCEPT.md, CONTRACT.md, or kernel headers. I'd recommend adding a documented invariant:

> **Module-Chunk Isolation Invariant:** For parameter-gradient reduction trees (Nodes 15, 20), offset lists for the `Grad_ModW`, `Grad_ModB`, and `Grad_Temps` reduction trees SHALL contain only tile offsets belonging to a single `module_chunk` group. Cross-module-chunk reduction is architecturally prohibited because it conflates gradients for disjoint parameter sets.

(Grad_H and Grad_S reductions are not affected — Grad_H is permuted to a contiguous SoA buffer indexed by module before Node 16 operates, and shared-layer gradients have no module decomposition.)

### 4.2 MODERATE: Scale Imbalance in Node 11's Joint L2 Norm

Node 11 computes a single L2 norm over four gradient components that exist at **different reduction stages**:

| Component | Batch-Reduced? | Element Count (per tile) |
|:---|:---|:---|
| Grad_ModW | **Yes** (summed over batch) | M_pc × H × C_chunk |
| Grad_ModB | **Yes** | M_pc × C_chunk |
| Grad_Temps | **Yes** | M_pc |
| Grad_H | **No** (per-sample) | M_pc × B × H |

The batch-summed components' magnitudes scale as O(B), while per-sample Grad_H elements are O(1) in batch size. The squared L2 norms scale as:

- `‖Grad_ModW‖² ~ M·H·C_chunk · B² · σ²`
- `‖Grad_H‖² ~ M·B·H · C_chunk² · σ²`

For large B, `‖Grad_ModW‖²` dominates (ratio ≈ B/C_chunk), meaning the effective clipping on Grad_H becomes `T_pre · ‖Grad_H‖/‖total‖` — potentially far below `T_pre`. Conversely, increasing `num_class_chunks` (reducing C_chunk) shifts dominance toward Grad_H.

**Impact:** The effective per-component clipping strength varies with batch size and class-chunk count, creating an implicit hyperparameter coupling. Users tuning `T_algorithmic` may observe qualitatively different clipping behavior when changing batch size.

**This is a documented design choice** (CONCEPT.md §3.5: "Component-Wise/Partial-Group-Wise combination"). I flag it here for completeness because the scale dependency on batch size is a consequence that isn't explicitly called out. It may merit a note in the validation scenario for "The Scientist's Repeater" or a new scenario that varies batch size while monitoring per-component effective clip ratios.

### 4.3 MODERATE: FP8 Cumulative Quantization Error for Grad_H

As traced in §2, Grad_H passes through **three** storage round-trips before reaching stable COMPUTE precision at Node 16's output. For FP8 E4M3 with a 3-bit mantissa:

- Each `store_storage` → `load_storage` cycle introduces up to 1 ULP of relative error
- E4M3 ULP at the scale of typical gradient values (say ~0.01) is 2⁻³ × 2⁻⁷ = 2⁻¹⁰ ≈ 0.001, which is ~10% relative error per round-trip
- Three round-trips don't simply multiply (each operates on the already-quantized value), but the cumulative effect is a ~3-ULP maximum error band

**For E4M3, this is within the acknowledged "stochastic sparsification" envelope** — the CONCEPT.md already notes that the FP8 quantization floor silently zeroes small contributions. The three round-trips widen this effective floor.

**No action required**, but worth noting that Grad_ModW/ModB/Temps experience only **two** round-trips (Node 8 → Node 11 → reduction engine), while Grad_H experiences **three** (extra Node 13 step). This asymmetry means Grad_H has slightly worse quantization fidelity than the parameter gradients in FP8 configurations.

### 4.4 MODERATE: Batch-Chunk Scope Asymmetry Between Nodes 8 and 9

Node 8 accepts `batch_chunk_offset` / `batch_chunk_count` parameters, enabling partial-batch processing. Node 9's Holistic Constraint explicitly prohibits batch chunking ("processes the complete batch dimension in a single dispatch") because Node 13 requires a monolithic collection buffer.

These two kernels are both within the same "Computational Tile" (Phase I) and both feed into Node 11. If the host sets Node 8's `batch_chunk_count < total_batch_count`:

1. Node 8 produces a **partial** batch sum of Grad_ModW
2. Node 9 produces Grad_H for the **full** batch
3. Node 11 clips them jointly — but the partial-batch Grad_ModW has different magnitude than it would with the full batch

**This is not incorrect for a single dispatch** (Node 11 clips whatever it receives), but it means the relative scale between Grad_ModW and Grad_H within the joint norm changes with `batch_chunk_count`, compounding the concern from §4.2.

**Pragmatic resolution:** In standard operation, the host likely sets `batch_chunk_count = total_batch_count` for the Computational Tile dispatch, and uses the "Accumulate via Recompute" streaming model (Model A) only when memory constraints force it. When Model A is active, the tile decomposition would need to be reorganized (potentially expanding `total_tile_count` to include the batch dimension), which changes how Node 11 operates. The contracts support the standard case correctly; the streaming variant likely requires orchestration-tier adaptations beyond what the current contracts explicitly describe.

### 4.5 LOW: `T_pre` Restrictiveness Under Large `num_class_chunks`

The safety constraint `T_pre ≤ COMPUTE_FP_FORMAT_MAX / num_class_chunks` can become limiting:

| Compute Type | FP_MAX | num_class_chunks | Max T_pre |
|:---|:---|:---|:---|
| FP32 | 3.4e38 | 100 | 3.4e36 |
| FP16 | 65504 | 100 | 655 |
| FP16 | 65504 | 1000 | 65.5 |

For FP16 compute with many class chunks, `T_pre ≈ 65` is a tight ceiling that may clip normal-magnitude gradients. This is a natural consequence of the Primacy of Memory Strategy — FP16 compute trades dynamic range for bandwidth — and the Quadratic Scaling Policy accommodates it gracefully. But users targeting very large class counts with FP16 compute should be aware of this implicit constraint.

---

## 5. Minor Observations

**5.1 Temperature gradient's negligible influence on Node 11's joint norm.** `Grad_Temps` contributes `modules_per_chunk` elements — typically 1 to ~8 — to a norm dominated by thousands of Grad_ModW and Grad_H elements. The temperature gradient is effectively "along for the ride," clipped by whatever scale factor the other components dictate. This is benign (temperature gradients that are enormous relative to weight gradients would indicate a pathological configuration), but it means per-item threshold overrides (CONCEPT.md §3.5 extension) wouldn't provide meaningful independent temperature-gradient control through Node 11.

**5.2 Node 5's sparsity optimization is explicitly not applied to Nodes 17/18.** The contract for Node 17 states: "The sparsity-predicated approach used by Node 5 is architecturally valid here but is not applied." This is a deliberate simplicity-over-performance choice for the streaming path. A future optimization could pass `hidden_mask` to Nodes 17/18 to skip zero-activation contributions, but the current approach of recomputing `relu' = (h > 0)` from stored activations is functionally equivalent and keeps the `StreamingLoopNode` body's parameter count minimal.

**5.3 Self-consistency under FP8 quantization.** The forward pass (Nodes 4→5→6/7) and backward pass both read hidden activations from the **same storage-precision buffer**. If FP8 quantization zeros a small positive activation, both the forward logit computation and the backward ReLU mask recomputation see zero. The system computes correct gradients **with respect to the quantized model**, not the theoretical unquantized model. This is the right behavior — the architecture is self-consistent.

**5.4 End-to-end path completeness.** All five parameter groups reach `adam_update` through well-defined paths:

```
ModW:   8 → 11 → 15 → 21 → 22 → 24         ✓
ModB:   8 → 11 → 15 → 21 → 22 → 24         ✓
Temps:  10 → 11 → 15 → 21 → 22 → 24 → 25   ✓
W_s:    17 → 19 → 20 → 21 → 22 → 24        ✓
B_s:    18 → 19 → 20 → 21 → 22 → 24        ✓
Grad_H: 9 → 11 → 13 → 16 → {17, 18}        ✓ (consumed, not updated)
```

No parameter group is missing a reduction, normalization, synchronization, or update step.
