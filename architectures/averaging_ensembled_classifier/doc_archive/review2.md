# Principled Recommendations for `kernels.cl.h` Revision 8

These recommendations are organized by urgency and framed against the architecture's own principles. Each identifies the specific contract article or architectural mandate it serves.

---

## Tier 1: Concrete Fixes

### R-1. Rename `fan_in_K` → `fan_in`

**Principle served:** Article 8 §1.0 (Canonical Lexicon mandate)

**Location:** `reduce_k_fan_in_and_clip` and `reduce_k_fan_in_and_clip_from_compute`, parameter `src_scalar_NATURAL_fan_in_K`

**Problem:** The `[ContextAndUsage]` component `fan_in_K` contains the suffix `_K` which is not defined anywhere in the Lexicon. The §5.0 entry for `fan_in` already defines it as *"the K in K-fan-in"* — the `_K` is semantically redundant and constitutes an unlexiconed term.

**Fix:** Rename to `src_scalar_NATURAL_fan_in` in both kernels. No information loss; the K-semantics are inherent in the Lexicon definition.

---

## Tier 2: Strengthen Existing Contracts

### R-2. Make Node 8's accumulative write pattern explicit

**Principle served:** Article 1.4 (Axiom of Collaborative Interface Verifiability): the interface should be a *closed logical system* for verification; implicit behavioral patterns undermine this.

**Location:** Node 8 `calculate_module_param_grads_chunk`, destination buffers `dest_buffer_GLOBAL_partial_grad_weights_module` and `dest_buffer_GLOBAL_partial_grad_biases_module`

**Observation:** Node 8 is unique among the gradient-producing kernels: it receives `batch_chunk_offset`/`batch_chunk_count` parameters while Nodes 9 and 10 do not. This is the *Accumulate via Recompute* model from CONCEPT §11: the Host Orchestrator dispatches Node 8 multiple times with different batch chunk ranges against the *same* `flat_tile_index`, and the kernel *adds* each chunk's contribution into the collection buffer. The `ZERO_REQUIRED` initialization ensures the first dispatch starts from zero.

This critical behavior — that the kernel performs **additive accumulation** rather than overwriting — is currently expressed only through the *intersection* of three separate contract features: `{Initialization Contract: ZERO_REQUIRED}`, `{Idempotency: "Associatively Non-Idempotent"}`, and the presence of batch-chunk parameters. No single artifact says "this kernel accumulates."

**Recommendation:** Extend the Initialization Contract vocabulary (Article 3.1.1) with a new type token:

| Type Token              | Definition |
|:------------------------|:-----------|
| `ZERO_REQUIRED_ADDITIVE` | Host must zero-fill the buffer before the *first* dispatch of a streaming series. The producing kernel adds to existing values on each dispatch; downstream consumers read only after the complete series. |

Apply this to Node 8's two destination buffers. This makes the accumulation pattern a *first-class contract artifact* rather than an inference the reader must assemble from three clues.

If a vocabulary extension is premature, the minimum fix is adding a note to the Behavioral Invariants:
> *"Accumulative Write Pattern: When the batch dimension is streamed across multiple dispatches sharing the same `flat_tile_index`, each dispatch ADDs its contribution to the ZERO_REQUIRED-initialized destination. The buffer content is only valid after all batch chunks have been dispatched."*

---

### R-3. Sharpen Node 16's Data Ingress Safety Invariant variable definition

**Principle served:** Article 4.2 Behavioral Invariants (precision of mandatory implementation rules); Principle 1 (Architectural Elegance Feedback — fuzzy terminology is a design smell)

**Location:** Node 16 `stabilize_and_reduce_grad_hidden_activations`, Behavioral Invariant §0 (Data Ingress Safety Invariant)

**Problem:** The invariant introduces `F_total` as *"the number of accumulators that the first reduction stage will sum"* and then uses it in the formula `src_scalar_REAL_fp_max / F_total`. While technically correct, `F_total` is defined in terms of an implementation detail ("each thread accumulates...") that conflates the Execution tier with the Contractual tier.

**Recommendation:** Rewrite the invariant to stay at the contract level:

> *"If the implementation partitions the `total_modules_count` input elements across `P` independent pre-reduction accumulators (where `P` is an Execution-tier concern, e.g., the work-group size), each accumulator MUST be clipped to `src_scalar_REAL_fp_max / P` before entering the first staged reduction. This ensures the first stage's sum of `P` accumulators cannot exceed `fp_max`, regardless of the value of `P`."*

This preserves backend neutrality (Principle 4) while being precise.

---

### R-4. Add explicit Calculability Proof for Node 19 write offset bounds

**Principle served:** Article 1.4 (Axiom of Collaborative Interface Verifiability) — specifically §1.4.1 (*"All parameters used in each Calculability Proof must exist in the interface"*)

**Location:** Node 19 `clip_shared_gradients_chunk`, parameters `dest_scalar_NATURAL_weights_write_offset` and `dest_scalar_NATURAL_biases_write_offset`

**Problem:** The commentary delegates responsibility to the host (*"The Host is responsible for providing a valid `dest_scalar_NATURAL_weights_write_offset`..."*) but provides no Calculability Proof or Validation Precondition that a verifier could check. Since all terms for the bound check *are* present in the interface, the proof should be explicit:

**Recommendation:** Add to `dest_buffer_GLOBAL_clipped_partial_grad_weights_shared`:

```
- Validation Preconditions: [1] (dest_scalar_NATURAL_weights_write_offset +
  src_scalar_NATURAL_weights_parameter_count) <=
  (src_scalar_NATURAL_num_batch_chunks * src_scalar_NATURAL_weights_parameter_count).
```

And correspondingly for the biases buffer:
```
- Validation Preconditions: [1] (dest_scalar_NATURAL_biases_write_offset +
  src_scalar_NATURAL_biases_parameter_count) <=
  (src_scalar_NATURAL_num_batch_chunks * src_scalar_NATURAL_biases_parameter_count).
```

This closes the verifiability loop without adding new parameters.

---

### R-5. Formalize the "Conditional" Precision Role for type-punned target buffers

**Principle served:** Article 2.1 Note (Axiom of Semantic Uniqueness — precision role determined by C type), Article 3 (Precision Role is **mandatory** for all floating-point buffer parameters)

**Location:** Nodes 8, 9, 10 — `src_buffer_GLOBAL_targets` declared as `__global const void*`

**Problem:** These buffers declare their Precision Role as `"Conditional — storage-role when src_scalar_FLAG_problem_type == BCE; integer-typed (exempt) when CCE."` This is a clear and accurate description, but `"Conditional"` is not one of the three canonical precision roles (`"storage"`, `"compute"`, `"state"`), nor is it listed as an exemption category. The current contract article offers only the `"exempt (integer bitmask)"` precedent for non-floating buffers, but no formal category for FLAG-dependent type duality.

**Recommendation:** Add to Article 3 a recognized fourth role annotation:

| Precision Role | Definition |
|:---|:---|
| `"flag-conditional"` | The buffer's element type depends on a FLAG scalar parameter's value. The commentary block MUST enumerate each flag value and its corresponding interpretation (role+type). Host-side validation operates on the runtime flag value. |

This elevates the `void*` / FLAG duality from ad-hoc commentary into a recognized contract pattern, giving the `KernelContract` frozen dataclass a formal category to carry. The existing `"Conditional — ..."` text becomes the mandatory commentary expansion of this role.

---

## Tier 3: Structural Recommendations for Future Revisions

### R-6. Separate "what" from "why" in Node 16's Behavioral Invariants

**Principle served:** Article 1.1 (Axiom of Jurisdictional Separation — syntactic vs. semantic); Article 4.2 (Behavioral Invariants define *rules*, not rationale)

**Observation:** Node 16 carries two distinct artifacts that are currently co-located:
- The **mandatory implementation formula** (§0–§2 of the invariant — the pre-computation phase, per-stage threshold logic)
- The **architectural justification** explaining why this isn't a jurisdictional violation or circular dependency

The justification text currently lives in the `Architectural Justification` key (correctly), but the invariant itself mixes "what the kernel MUST do" with commentary like *"This linearizes the problem"*. As the system evolves, other specialized kernels may face similar jurisdictional questions.

**Recommendation:** In the next contract revision, draw a firmer line: `Behavioral Invariants` contains ONLY machine-verifiable obligations (expressed as formulas and pseudocode). The `Architectural Justification` key carries *all* reasoning about why those invariants are sound. This already holds for most kernels; Node 16 should converge.

---

### R-7. Consider `COMPUTE_TYPE` vs `STATE_TYPE` for Adam bias correction terms

**Principle served:** CONCEPT §2 (Primacy of Memory Strategy — precision roles serve distinct optimization targets); CONCEPT §11 Item 7 (*"the host computes beta1\*\*t and beta2\*\*t bias correction terms in high precision (FP64)"*)

**Observation:** The kernel parameters `src_scalar_REAL_beta1_pow_t` and `src_scalar_REAL_beta2_pow_t` are `COMPUTE_TYPE` scalars. The host computes these in FP64 and then narrows to `COMPUTE_TYPE` at the interface boundary. Under `PrecisionConfig.mixed_f32_f64_state()` (FP32 compute, FP64 state), this means:

1. Host computes `0.999^100000` in FP64: ≈ 2.07×10⁻⁴⁴ (exact within FP64)
2. Passes as FP32: rounds to 0.0 (below FP32 min subnormal ~1.4×10⁻⁴⁵) **or** to a subnormal with reduced precision
3. Kernel widens back to FP64 for the bias correction division: the FP64 bits are lost

The kernel's own contract correctly documents this: *"The widening preserves only COMPUTE_TYPE precision for these constants."* At t ≈ 100K, `beta1^t` is so small that `1 / (1 - beta1^t) ≈ 1.0`, so the precision loss in the bias correction divisor is negligible in practice.

**Recommendation:** This is currently sound for all practical training horizons, and the contract correctly documents the precision ceiling. However, for full consistency with the architecture's principle that the state role enables *"unbounded training stability"*, a future revision could:
- Accept `beta_pow_t` scalars in `STATE_TYPE` (adding two `STATE_TYPE` scalar parameters)
- Compute bias correction in `ACCUM_TYPE`

This would require an Article 2.2 grammar extension (STATE_TYPE scalars don't have a `[NumberType]` token yet — `REAL_` maps to COMPUTE_TYPE). The benefit is marginal for realistic training horizons (≤10⁶ steps), so this is a documentation-now, implement-later recommendation.

---

### R-8. Note the logits buffer's uninitialized padding semantics

**Principle served:** Article 3.1.1 (Initialization Contract completeness)

**Location:** Node 5 `render_logits_chunk`, dest `dest_buffer_GLOBAL_logits`

**Observation:** The logits buffer has `padded_total_output_class_count` as its innermost stride, but Node 5 writes only `class_chunk_count` classes per dispatch (within `total_output_class_count`). Padding positions beyond `total_output_class_count` are never written. The Initialization Contract is implicitly `{Type: NONE}` — correct, since downstream consumers (Nodes 6, 7, 10) all receive `total_output_class_count` and iterate only within the valid range.

**Recommendation:** While no initialization is needed (the unwritten padding is never consumed), the *reason* it isn't needed is non-obvious. Add to the logits buffer's Validation Preconditions:

> *"Padding positions beyond `total_output_class_count` within each `padded_total_output_class_count` stride are architecturally unwritten. Downstream consumers (Nodes 6, 7, 10) are contractually bounded by `total_output_class_count` and DO NOT access padding."*

This creates an explicit cross-kernel safety statement that a verifier can check.

---

### R-9. Per-dimension Padding Contract specification (future roadmap)

**Principle served:** Article 3.1 (already notes *"A future revision may extend the Padding Contract to per-dimension specifications"*)

**Observation:** Several buffers have two independently padded dimensions with different motivations. For example, Node 9's `src_buffer_GLOBAL_CONST_weights_module` has SIMD padding on the class dimension and cache-line padding implicit in the hidden dimension (via `padded_hidden_count`). The current contract's `Type` field can only declare one primary motivation.

**Recommendation:** When the next contract revision occurs, extend the Padding Contract to a per-dimension dictionary:

```
- Padding Contract: {
    dim[0] ("total_modules_count"): {Type: NONE},
    dim[1] ("hidden_count" → "padded_hidden_count"): {Type: CACHE, Formula: "128-byte alignment"},
    dim[2] ("total_output_class_count" → "padded_total_output_class_count"): {Type: SIMD, Formula: "SIMD_WIDTH alignment"}
  }
```

This eliminates the *"primary padding motivation"* ambiguity while staying within the existing vocabulary. It's a natural evolution of the `padded_*` scalar convention that already carries the padding values — this extension would carry the padding *rationale* per dimension.

---

## Summary Table

| ID | Severity | Article/Principle | Action |
|:---|:---|:---|:---|
| **R-1** | Fix | Art. 8 §1.0 | Rename `fan_in_K` → `fan_in` |
| **R-2** | Strengthen | Art. 1.4, 3.1.1 | Formalize Node 8's accumulative write pattern |
| **R-3** | Strengthen | Art. 4.2 | Sharpen Node 16's `F_total` definition |
| **R-4** | Strengthen | Art. 1.4, 1.4.1 | Add Calculability Proof for Node 19 write offsets |
| **R-5** | Strengthen | Art. 2.1, 3 | Formalize `"flag-conditional"` precision role |
| **R-6** | Evolve | Art. 1.1, 4.2 | Separate invariant formulas from justification prose in Node 16 |
| **R-7** | Evolve | CONCEPT §2, §11 | Consider STATE_TYPE for Adam bias correction terms |
| **R-8** | Evolve | Art. 3.1.1 | Document logits padding non-consumption guarantee |
| **R-9** | Evolve | Art. 3.1 (noted) | Per-dimension Padding Contract specification |
