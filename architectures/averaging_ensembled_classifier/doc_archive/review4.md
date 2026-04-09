# Principled Recommendations for `kernels.cl.h`

Each recommendation is grounded in a specific Contract article, CONCEPT principle, or cross-document consistency requirement. They are ordered by severity: mandatory fixes first, then defensive strengthening, then documentation fidelity.

---

## R1 — Fix: Padding Contract SHALL Violation on Node 11 Destination Buffer

**Grounding:** CONTRACT Article 3.1 — *"When a buffer has multiple independently padded dimensions, the Padding Contract field **SHALL** use a per-dimension dictionary format."*

**Location:** Node 11 (`clip_partial_gradients`), parameter `dest_buffer_GLOBAL_clipped_partial_grad_weights_module`

**Issue:** This buffer's Calculability Proof resolves to four dimensions: `(total_tile_count, modules_per_chunk, padded_hidden_count, padded_total_output_class_count)`. Dimensions 2 and 3 carry independent padding motivations (CACHE and SIMD respectively). The current declaration uses the flat format `{Type: NONE}`. Its *source* counterpart in the same kernel correctly uses the per-dimension dictionary.

**Recommendation:**

```
- Padding Contract: {
    dim[0] ("total_tile_count"): {Type: NONE},
    dim[1] ("modules_per_chunk"): {Type: NONE},
    dim[2] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
    dim[3] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
  }
```

**Risk if unaddressed:** Article 3.1 uses SHALL — this is a binding contract violation. Any automated contract validator would reject the parameter specification.

---

## R2 — Strengthen: Explicit Padding Zero-Fill Invariant for Gradient-Producing Kernels

**Grounding:** CONTRACT Article 4.2 (Behavioral Invariants) + CONCEPT Principle 2 (Primacy of Memory Strategy — SIMD-aligned layouts are first-class) + correctness of downstream L2 norm computation.

**Location:** Nodes 8, 9, 10, 13, 17, 18 — any kernel whose output buffer has a `padded_*` dimension consumed by an L2 norm computation (Nodes 11, 19, 16).

**Issue:** Node 11 computes a single L2 norm across `padded_hidden_count × padded_total_output_class_count` elements per tile. It receives *only* `padded_hidden_count` and `padded_total_output_class_count` — not the unpadded logical counts. Therefore, it iterates over padding positions. For the norm to be correct, padding positions must be exactly zero.

The current architecture establishes this guarantee through two mechanisms:
- **Node 8's weight/bias gradients:** `ZERO_REQUIRED_ADDITIVE` initialization + kernel writes only `classes_per_chunk` positions → non-written positions stay zero. ✓ *Guaranteed by Initialization Contract.*
- **Node 9's hidden gradient:** No Initialization Contract (defaults to `NONE`) + implicit assumption that zero-padded state-role weights propagate zero gradients for hidden indices ≥ `hidden_count`. ⚠️ *Guaranteed only by an unstated assumption chain.*

This implicit chain is: host zeros weight padding → weight[h≥hidden_count][c] = 0 → grad_H[h≥hidden_count] = Σ(W^T × error) = 0. If any upstream buffer's padding invariant is ever violated, the L2 norm silently includes garbage, producing incorrect clipping.

**Recommendation:** Add a **Behavioral Invariant** clause to every gradient-producing kernel that writes to a buffer consumed by an L2 norm computation:

> *"Padding Zero-Fill: For padded dimensions where the logical extent (`hidden_count`, `total_output_class_count`) is less than the padded extent (`padded_hidden_count`, `padded_total_output_class_count`), the kernel SHALL write zero for all positions at indices ≥ the logical extent. This guarantees that downstream L2 norm computations over the full padded extent are mathematically equivalent to norms over the logical extent."*

This transforms an implicit assumption chain into a kernel-level contract obligation.

**Risk if unaddressed:** A future kernel implementation that short-circuits at `hidden_count` without zero-filling padding (a natural performance optimization) would silently corrupt every downstream L2 norm. The defect would manifest as subtly incorrect gradient clipping — extremely difficult to diagnose.

---

## R3 — Strengthen: Add Initialization Contract to Node 9 Output

**Grounding:** CONTRACT Article 3.1.1 (Initialization Contract Specification) + R2's analysis.

**Location:** Node 9 (`backprop_error_to_hidden_chunk`), parameter `dest_buffer_GLOBAL_partial_grad_hidden_activations_aos`

**Issue:** This buffer's Initialization Contract defaults to `NONE`, which asserts: *"The producing kernel(s) guarantee that all positions read by downstream consumers are written before consumption."* However, downstream consumer Node 11 reads the full `padded_hidden_count` extent per (module, sample) pair. If the kernel implementation iterates only to `hidden_count` (a natural optimization given the parameter is available), padding positions are uninitialized.

Unlike Node 8's buffers — which have `ZERO_REQUIRED_ADDITIVE` providing a host-side zero guarantee independent of kernel behavior — Node 9's output relies entirely on the kernel writing zeros for padding positions.

**Recommendation:** Two valid approaches, choose based on implementation strategy:

**(A) If the kernel guarantees zero-fill for padding (R2's Behavioral Invariant):**
Keep `NONE` but add the R2 Behavioral Invariant clause. The contract then correctly asserts the kernel writes all consumed positions.

**(B) If the kernel does NOT zero-fill padding positions:**
Add `Initialization Contract: {Type: ZERO_REQUIRED}`. This shifts the zero-fill responsibility to the host, making the kernel implementation free to iterate only over `hidden_count`.

Option (B) is the more defensive choice — it decouples correctness from kernel implementation details and follows the precedent set by Node 8's buffers and Node 13's destination buffer.

**Risk if unaddressed:** Identical to R2 — silent norm corruption from uninitialized padding values. The risk is concrete and immediate, not hypothetical.

---

## R4 — Improve: Scalar `total_tile_count` Calculability Proofs for Consistency

**Grounding:** CONTRACT Article 1.4 (Axiom of Collaborative Interface Verifiability) — *"A kernel's public interface constitutes a closed logical system for verification."*

**Location:** Nodes 7, 8, 9, 10, 11 — all receive `src_scalar_NATURAL_total_tile_count` without a `@param` commentary block.

**Issue:** Node 6 (`compute_probs_loss_cce_chunk`) is the only kernel that provides a Calculability Proof for the `total_tile_count` scalar:

> *Calculability Proof: `[ceil(total_modules_count / modules_per_chunk) × num_class_chunks]`*

Nodes 7 through 11 use the identical parameter with the identical semantic meaning (the total number of tiles in the flattened module × class grid), but provide no Calculability Proof. A contract validator applying Axiom 1.4 cannot verify the scalar's derivation for these kernels without out-of-band knowledge.

**Recommendation:** Add a `@param` commentary block with the Calculability Proof to `src_scalar_NATURAL_total_tile_count` in Nodes 7, 8, 9, 10, and 11. Since the formula is identical across all kernels, this is a pure documentation consistency lift:

```c
/**
 * @param src_scalar_NATURAL_total_tile_count Total tiles in the (module_chunk, class_chunk) grid.
 *        - Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count / src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
 */
```

**Risk if unaddressed:** No correctness risk. Reduces the machine-verifiability of the interface specification — a validator must special-case scalars without proofs.

---

## R5 — Document: Pre-Summation Amplification Constraint on Node 11's Threshold

**Grounding:** CONCEPT §3.4 (Pre-Summation Amplification Factor) — *"`T_pre ≤ COMPUTE_FP_FORMAT_MAX / num_class_chunks`"*

**Location:** Node 11 (`clip_partial_gradients`), parameter `src_scalar_REAL_clipping_threshold_t_pre`

**Issue:** The CONCEPT establishes a non-obvious upper bound on the threshold passed to Node 11: the Grad_H component, after clipping by Node 11, is summed across `num_class_chunks` tiles by Node 13's implicit reduction. If `T_pre × num_class_chunks > FP_MAX`, Node 13's output overflows.

Node 11's current Validation Preconditions state only: *"Must be a positive real number."* While the amplification constraint is a Policy-tier responsibility (the Host Orchestrator computes `T_pre`), the kernel's interface already includes `num_class_chunks` — all terms for the constraint are present.

**Recommendation:** Add a Validation Precondition or a Performance Note cross-referencing the CONCEPT's amplification formula:

```
- Validation Preconditions: [1] Must be a positive real number.
  [2] The Host MUST ensure this value satisfies the Pre-Summation
  Amplification constraint: (src_scalar_REAL_clipping_threshold_t_pre *
  src_scalar_NATURAL_num_class_chunks) <= COMPUTE_FP_FORMAT_MAX,
  because the downstream Node 13 sums num_class_chunks clipped tiles
  per element (see CONCEPT.md §3.4).
```

This preserves the closed-system property (Axiom 1.4) — all terms are kernel parameters — and makes the constraint discoverable from the kernel specification alone.

**Risk if unaddressed:** A Policy-tier maintainer modifying the threshold computation could introduce post-Node-13 overflow without any kernel-level warning. The overflow would manifest as `Inf` in Node 16's input — caught by safety clipping but wasteful of gradient information.

---

## R6 — Document: FP8 Re-Quantization in Clip-to-Storage Paths

**Grounding:** CONCEPT Principle 2 (Primacy of Memory Strategy) + CONTRACT Article 4.2 (Behavioral Invariants: Precision Boundary Conversion).

**Location:** Node 11 (`clip_partial_gradients`) and Node 19 (`clip_shared_gradients_chunk`), specifically the interaction between the clipping operation and the `store_storage()` narrowing.

**Issue:** Under FP8 storage configurations, these kernels execute a **load → compute → clip → store** cycle. The clipping operation scales gradients in COMPUTE_TYPE (FP32), then `store_storage()` narrows back to FP8. This narrowing introduces quantization error that is *not present* in FP16 or FP32 configurations. Specifically:

1. Two gradients that were distinguishable pre-clip at FP32 may round to the same FP8 value post-clip.
2. Small gradient components preserved by clipping (below the threshold) may round to zero if they fall below FP8's quantization floor (2⁻⁹ ≈ 0.00195 for E4M3).

The current Behavioral Invariants for both kernels state: *"storage-role inputs widened via load_storage(); storage-role outputs narrowed via store_storage()"* — correct but silent about the quantization implications.

**Recommendation:** Add an FP8-specific note to the Behavioral Invariants of Nodes 11 and 19:

> *"FP8 Quantization Note: When STORAGE_TYPE is FP8, the clip-then-store sequence introduces re-quantization error. Gradient components scaled below the FP8 quantization floor (2⁻⁹ for E4M3, 2⁻¹⁶ for E5M2) may round to zero, effectively zeroing a subset of the gradient. This is an accepted consequence of the Primacy of Memory Strategy — the architecture trades gradient fidelity for 4× bandwidth compression. The Quadratic Scaling Policy's threshold schedule accounts for this by maintaining gradients well above the quantization floor."*

**Risk if unaddressed:** No correctness risk — the architecture explicitly accepts FP8 quantization. The recommendation is about documentation fidelity: a reader of the kernel contract should understand the full precision consequences of the Precision Boundary Conversion without needing to cross-reference the CONCEPT's FP8 discussion.

---

## R7 — Polish: Descriptive Padding Contract Types for Single-Padded Buffers

**Grounding:** CONTRACT Article 3.1 (Padding Contract Specification) — documentation quality.

**Location:** Multiple buffers across Nodes 8, 9, 10, 11, 17, 18, 19 that have a single padded dimension and use `{Type: NONE}`.

**Issue:** Article 3.1 defines `NONE` as: *"No independent padding strategy is applied to this buffer."* For buffers with `padded_*` parameters in their Tensor Shape, `NONE` is technically correct — the padding is determined by the host-computed `padded_*` scalar. However, the *motivation* for the padding (CACHE? SIMD?) is lost.

For example, Node 8's `dest_buffer_GLOBAL_partial_grad_biases_module` has shape `(total_tile_count, modules_per_chunk, padded_total_output_class_count)`. The `padded_total_output_class_count` dimension is SIMD-motivated, but the Padding Contract says `{Type: NONE}`.

Meanwhile, structurally similar buffers with *two* padded dimensions use the per-dimension dictionary and clearly state `{Type: SIMD}` / `{Type: CACHE}` for each dimension.

**Recommendation:** For single-padded-dimension buffers, upgrade from flat `{Type: NONE}` to a descriptive type that names the motivation:

```
- Padding Contract: {Type: SIMD, Formula: "SIMD_WIDTH alignment via padded_total_output_class_count"}
```

Or, using the per-dimension format even for a single padded dimension (which is valid but not required by Article 3.1):

```
- Padding Contract: {
    dim[2] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
  }
```

This creates uniform documentation style across all padded dimensions regardless of whether the buffer has one or multiple padded dimensions.

**Risk if unaddressed:** No correctness risk. The host correctly computes `padded_*` scalars regardless. The only cost is documentation asymmetry: a reader can immediately see why a dimension is padded in multi-padded buffers but must infer the reason in single-padded buffers.

---

## R8 — Future-Proof: Node 24 Bias Correction Scalars in STATE_TYPE

**Grounding:** Node 24's *own* Behavioral Invariants footnote — *"A future revision may accept these scalars in STATE_TYPE for full consistency with the state role's unbounded-training-stability guarantee."*

**Location:** Node 24 (`adam_update`), parameters `src_scalar_REAL_beta1_pow_t` and `src_scalar_REAL_beta2_pow_t`

**Issue:** These scalars are computed in FP64 by the host but narrowed to `COMPUTE_TYPE` at the kernel interface. Under `mixed_f32_f64_state()` (FP32 compute, FP64 state), the Behavioral Invariants note that `beta1^t` values below ~1.4e-45 round to FP32 zero. At t ≈ 100K with β₁ = 0.999, `beta1^t ≈ e^{-100} ≈ 3.7e-44`, which is below FP32's minimum normal (1.18e-38) but above the minimum subnormal (1.4e-45). The bias correction factor `1/(1 - beta1^t)` is effectively 1.0 at this scale, so the loss is negligible.

However, the architecture explicitly states *"Optimizer implementations must avoid precision erosion across unbounded training steps"* (CONCEPT §6, Architectural Hierarchy example mandate). For extremely long training runs (t > 150K), `beta1^t` enters FP32 subnormal territory, and the bias correction factor deviates from 1.0 by less than FP32 epsilon — still safe. But the principle of unbounded stability suggests the interface should eventually match the precision model.

**Recommendation:** This is a tracked future extension, not an immediate fix. The recommendation is to:

1. **Retain** the current Behavioral Invariants footnote documenting the precision ceiling.
2. **Add** a cross-reference to the CONCEPT's state-precision accumulation design (§11) noting that the interface typing of these scalars is the last remaining non-state-precision bottleneck in the Adam path.
3. **When the extension is implemented:** Change the parameter types from `COMPUTE_TYPE` to `STATE_TYPE`, add corresponding `load_state()` widening in the kernel, and update the host to pass the FP64 values without narrowing.

**Risk if unaddressed:** Negligible for practical training horizons (< 10⁶ steps). The recommendation is about architectural completeness rather than correctness — ensuring the "unbounded stability" guarantee is formally satisfied end-to-end.

---

## Summary Table

| # | Severity | Article/Principle | Location | Nature |
|---|---|---|---|---|
| R1 | **Mandatory** | Article 3.1 SHALL | Node 11 dest padding | Format violation |
| R2 | **Strong** | Article 4.2 + correctness | Nodes 8–18 outputs | Missing invariant |
| R3 | **Strong** | Article 3.1.1 + correctness | Node 9 output | Missing initialization |
| R4 | Moderate | Axiom 1.4 | Nodes 7–11 scalars | Consistency |
| R5 | Moderate | CONCEPT §3.4 | Node 11 threshold | Cross-reference |
| R6 | Moderate | Principle 2 | Nodes 11, 19 | FP8 awareness |
| R7 | Low | Article 3.1 | Multiple buffers | Documentation |
| R8 | Low | CONCEPT §11 | Node 24 scalars | Future-proofing |

R1 through R3 affect verifiability or correctness and should be addressed in the next revision. R4 through R6 improve the self-containedness of the specification. R7 and R8 are pure polish.
