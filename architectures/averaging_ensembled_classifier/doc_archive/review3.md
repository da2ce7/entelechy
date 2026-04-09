# Principled Recommendations for Revision 9

These recommendations are organized around six architectural principles drawn from the system's own guiding philosophy. Each recommendation traces to a specific article, axiom, or CONCEPT mandate.

---

## Principle A: Every CONCEPT Behavioral Mandate Must Be Contract-Enforceable

**Rationale (Article 1.4, Axiom of Interface Verifiability):** If a behavioral guarantee exists in the CONCEPT but not in the kernel's `@kernel_contract`, then no implementation can be validated against it. The contract must be the _sufficient_ specification.

### A-1. Formalize Node 7's Numerical Stability Guarantee

**Priority:** High — this is the sole substantive compliance gap.

CONCEPT mandates:
> _"Performs the complete, temperature-aware, numerically stable Sigmoid calculation internally (two-branch form: positive/negative logit split to avoid exp() overflow)"_

Node 6 correctly formalizes its counterpart:
> _"The implementation is a fused, indivisible unit for numerically stable Softmax calculation."_

Node 7's Behavioral Invariants omit this. A backend implementor could write a naive `1/(1+exp(-x))` and pass the current contract.

**Recommended addition** to Node 7's `Behavioral Invariants`, prepended before the existing Precision Boundary Conversion text:

```
"The implementation shall employ a numerically stable Sigmoid computation
(two-branch form: for positive logits, σ(x) = 1/(1+exp(-x)); for negative
logits, σ(x) = exp(x)/(1+exp(x))) to prevent exp() overflow. BCE loss
shall use a numerically stable formulation that avoids log(0) (e.g.,
max(x,0) - x*y + log(1+exp(-|x|))). Precision Boundary Conversion: ..."
```

This pins both the sigmoid _and_ the loss to stable formulations, matching the specificity of Node 6's softmax contract.

### A-2. Make Node 11's Atomic-Group-Clip Semantics Explicit

**Priority:** Medium — currently inferrable but not machine-verifiable.

CONCEPT mandates:
> _"It computes a **single L2 norm** over the logical concatenation of all input gradient buffers... a single scaling factor is then applied **uniformly** to all four constituent buffers."_

Node 11's contract communicates this through inference (four src/dest pairs, single-tile scope, two-pass algorithm), but never makes the atomicity explicit. An implementation that computed four independent norms would technically satisfy the current contract text.

**Recommended addition** to Node 11's `Behavioral Invariants`, after the existing two-pass description:

```
"[3] The L2 norm is computed over the logical concatenation of all four
input gradient buffers (weights, biases, temperatures, hidden activations).
A single derived scaling factor is applied uniformly to all four output
buffers. Independent per-buffer norms are a contract violation."
```

This mirrors the explicit phrasing of Node 19's existing invariant: _"The L2 norm is computed over the concatenated vector of both weight and bias gradients."_

---

## Principle B: Data-Flow Traceability Across the DAG Should Be Maximized

**Rationale (Architectural Hierarchy §1, Article 1.1):** When a human reader traces data through the DAG, buffer name continuity across producer/consumer boundaries reduces cognitive load and audit error. The naming discontinuity between Node 13's output and Node 16's input creates a traceability gap.

### B-1. Restore the `clipped_` Prefix on Node 16's Input Buffer

**Priority:** Low — no functional impact, but improves readability.

Node 13 outputs:
```
dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa
```

Node 16 consumes:
```
src_buffer_GLOBAL_grad_hidden_activations_permuted_soa
```

The `clipped_` prefix documents a meaningful provenance: this data has undergone Node 11's stability clipping. Dropping it in Node 16 forces the reader to reverse-lookup the BufferDescriptor to confirm the buffer's history.

**Recommended rename** in Node 16:
```
src_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa
```

This restores the Article 8 §4.0 `clipped_` modifier's semantic continuity across the DAG edge.

### B-2. Add Calculability Proof for `total_tile_count`

**Priority:** Low — currently satisfies closedness but not explicitness.

The scalar `src_scalar_NATURAL_total_tile_count` appears in 7 kernels as a bounds-check parameter, but its derivation is never stated as a Calculability Proof in any kernel's commentary. The proof is:

```
total_tile_count = ceil(total_modules_count / modules_per_chunk) × num_class_chunks
```

All constituent terms are present in every kernel that declares `total_tile_count`, satisfying Article 1.4's closedness requirement. But the proof is implicit—a reader must reconstruct it from the `grid_mod_cls` placement strategy and the chunking model.

**Recommendation:** Add a Calculability Proof to `src_scalar_NATURAL_total_tile_count` on its first occurrence (Node 6), with a cross-reference on subsequent kernels:

```
- Calculability Proof: [ceil(src_scalar_NATURAL_total_modules_count /
  src_scalar_NATURAL_modules_per_chunk) * src_scalar_NATURAL_num_class_chunks]
```

---

## Principle C: Defensive Documentation Should Be Symmetric Across Parallel Constructs

**Rationale (Principle §1, Architectural Elegance Feedback):** When two constructs share identical structure but only one carries a defensive annotation, the omission on the other invites implementation divergence without detection.

### C-1. Add Subnormal Rounding Note to E5M2 Store Path

**Priority:** Low — the limitation is identical but undocumented.

The E4M3 `store_storage_fp8` path carries an explicit limitation note:

```c
// NOTE: Subnormal rounding omits shifted-out bits from sticky calculation.
// Max error: 1 ULP of FP8 subnormal (2^-9). Below quantization floor; no fix required.
```

The E5M2 path has the identical pattern (subnormal mantissa right-shift losing sticky bits) but no corresponding note. For E5M2, the max error is 1 ULP of FP8 subnormal (2⁻¹⁶), which is even smaller.

**Recommendation:** Add an analogous note to the E5M2 subnormal handling block:

```c
// NOTE: Subnormal rounding omits shifted-out bits from sticky calculation.
// Max error: 1 ULP of FP8 E5M2 subnormal (2^-16). Below quantization floor; no fix required.
```

### C-2. Unify Behavioral Invariant Style for Aggregate Variants

**Priority:** Low — a documentation clarity improvement.

The storage-entry `aggregate_register_reduce` specifies:
> _"Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE."_

The compute-entry `aggregate_register_reduce_from_compute` specifies:
> _"All buffers are compute-role; no precision boundary conversion is required. All arithmetic exclusively in COMPUTE_TYPE."_

The same asymmetry exists for both the `aggregate_local_reduce` and `reduce_k_fan_in_and_clip` pairs. The compute-entry variants **correctly** omit Precision Boundary Conversion, but they don't affirmatively state the accumulation path. While not required by the contract, adding a single phrase to each compute-entry variant improves auditable equivalence:

```
"All buffers are compute-role; no precision boundary conversion is required.
Reduction accumulation in COMPUTE_TYPE; compute-role output written directly."
```

This makes the behavioral symmetry immediately verifiable without cross-referencing the storage-entry partner.

---

## Principle D: The Initialization Contract Vocabulary Should Precisely Capture Intent

**Rationale (Article 3.1.1):** The `ZERO_REQUIRED_ADDITIVE` token conflates two structurally distinct patterns under the same label. This doesn't cause correctness issues today, but obscures the host's obligation for future maintainers.

### D-1. Consider Distinguishing "Zero-Padded Disjoint Write" from "Additive Streaming"

**Priority:** Informational — a vocabulary evolution recommendation for a future CONTRACT revision, not an immediate action.

Node 8's `dest_buffer_GLOBAL_partial_grad_weights_module` uses `ZERO_REQUIRED_ADDITIVE`. The buffer serves two simultaneous purposes:

1. **Additive streaming** across batch chunks (multiple dispatches accumulate to the same spatial positions) — the canonical `ZERO_REQUIRED_ADDITIVE` use case.
2. **Zero-padded disjoint write** across class chunks (each class chunk writes to non-overlapping columns within the `padded_total_output_class_count` stride; padding positions must be zero for Node 11's norm computation).

Both are correctly handled by `ZERO_REQUIRED_ADDITIVE`, but a future reader seeing only the token cannot distinguish "this needs zero-init because of accumulation" from "this needs zero-init because of sparse write coverage." If the chunking model evolves (e.g., class-chunk accumulation within a single dispatch), the distinction becomes safety-critical.

**No immediate action required.** If a future revision introduces more complex streaming patterns, consider a `ZERO_REQUIRED_SPARSE` token for the disjoint-write case, or expanding Article 3.1.1 commentary to capture the dual-purpose nature.

---

## Principle E: Exemplary Patterns Should Be Generalized as Vocabulary

**Rationale (Principle §1, Architectural Elegance Feedback):** When a pattern emerges organically in a specific kernel's contract, it should be evaluated for promotion to the canonical vocabulary.

### E-1. Promote Node 6's "Loss Write Predicate" Pattern

**Priority:** Medium — improves the contract vocabulary's expressiveness.

Node 6 introduces a precise behavioral pattern in its `Behavioral Invariants`:

> _"Loss Write Predicate: The kernel writes to dest\_buffer\_GLOBAL\_final\_loss[module][sample] only when the target class index falls within the current tile's class chunk range..."_

This is an instance of a general pattern: **conditional write predicated on tile-scope membership**. It's distinct from the Placement Contract (which governs _where_ to write) and the Initialization Contract (which governs _pre-state_). It governs _whether_ to write.

**Recommendation for CONTRACT Article 4.3:** Add a canonical term:

| Term | Definition |
|:-----|:-----------|
| `Conditional Writer` | A kernel that writes to a destination buffer only when a data-dependent predicate (specified in Behavioral Invariants) is satisfied for the current work item. Positions not satisfying the predicate remain at their initialization state. |

This would allow Node 6's Synchronization Model to be expressed more concisely:

```
- Synchronization Model: "Partial Renderer for probabilities. Conditional Writer for loss (predicate: target class index ∈ [class_chunk_offset, class_chunk_offset + classes_per_chunk))."
```

### E-2. Promote Node 9's "Intentional Absence" Documentation Pattern

**Priority:** Low — a best-practice recommendation.

Node 9's Holistic Constraints contain a defensive note:

> _"Batch-chunking parameters (batch\_chunk\_offset, batch\_chunk\_count) are intentionally absent because the downstream Item Synchronization Point (Node 13) requires a monolithic collection buffer."_

This is excellent practice — it documents _why_ a parameter is missing, not just what's present. Other kernels could benefit from this pattern. For example, Node 16 could note why it receives no `offset_list` (because the upstream Node 13 guarantees contiguous input, making indirection unnecessary).

**Recommendation:** Establish a convention in a future CONTRACT article addendum that `Holistic Constraints` SHOULD document intentional parameter absences when the absence is a deliberate architectural choice rather than a default. This transforms the `Holistic Constraints` key from a "last resort" (Article 4.2) to also serve as a "deliberate omission record."

---

## Principle F: Prepare for Architectural Evolution

**Rationale (Principle §1, Architectural Elegance Feedback; §4, Backend-Neutral Plan Model):** Recommendations that reduce friction for anticipated growth paths.

### F-1. Document the ACCUM_TYPE Coverage Matrix

**Priority:** Low — the implementation is correct, but the coverage isn't immediately auditable.

The `ACCUM_TYPE` typedef chain in the header handles 9 (STATE × COMPUTE) combinations through 3 `#if` branches. The inline comments explain individual branches, but the complete coverage matrix is not documented. For an auditor verifying that no combination falls through to an incorrect branch, tracing the logic requires mental simulation.

**Recommendation:** Add a coverage matrix comment above the `ACCUM_TYPE` section:

```c
// ACCUM_TYPE coverage matrix (S=STATE, C=COMPUTE):
//   S=double, C=double → else branch:  ACCUM=double (COMPUTE) ✓
//   S=double, C=float  → branch 1:     ACCUM=double (STATE)   ✓
//   S=double, C=half   → branch 1:     ACCUM=double (STATE)   ✓
//   S=float,  C=double → else branch:  ACCUM=double (COMPUTE) ✓
//   S=float,  C=float  → else branch:  ACCUM=float  (COMPUTE) ✓
//   S=float,  C=half   → branch 2:     ACCUM=float  (STATE)   ✓
//   S=half,   C=double → else branch:  ACCUM=double (COMPUTE) ✓
//   S=half,   C=float  → else branch:  ACCUM=float  (COMPUTE) ✓
//   S=half,   C=half   → else branch:  ACCUM=half   (COMPUTE) ✓
```

This costs nothing at compile time and makes the correctness proof self-contained.

### F-2. Anticipate FP8 Variant Proliferation

**Priority:** Informational — a forward-looking architectural note.

The current kernel set has 4 precision-variant pairs (2 aggregate × 2 entry types, 2 reduce × 2 entry types). FP8 storage introduces a third entry type possibility: uchar-entry (raw byte access without `load_storage()` abstraction). The current architecture handles this elegantly via `load_storage_fp8()` called inside the existing `load_storage()` dispatch, so no third variant is needed today.

However, if future performance optimization reveals that the FP8 LUT decode is a bottleneck for large reductions (256 lookups per partial per element), a specialized FP8-entry variant could bypass the generic `load_storage()` dispatch and use direct table access with vectorized reads. Per Principle §1 (Architectural Elegance Feedback):

**Recommendation:** No action needed now. But document in the ADR chain that the current 2-variant model (storage-entry, compute-entry) was evaluated for FP8 adequacy and found sufficient. If profiling reveals the LUT dispatch to be a bottleneck, the response is to formalize an `fp8-entry` variant through ADR, not to add ad-hoc specialization.

