# Principled Recommendations for `kernels.cl.h` Revision 9

Each recommendation is grounded in a specific governing principle from the architecture's own documents. They are ordered by architectural severity — closures to axiom-level guarantees first, consistency refinements last.

---

## R1: Restore Interface Completeness for Padding Zero-Fill Invariants

**Governing Principle:** Axiom of Collaborative Interface Verifiability (CONTRACT Article 1.4) — *"A kernel's public interface constitutes a closed logical system for verification."*

**Finding:** Nodes 17 and 18 declare **Padding Zero-Fill** behavioral invariants referencing logical-extent quantities (`input_count`, `hidden_count`) that are absent from their parameter lists. The invariant language uses mandatory "SHALL" phrasing:

> *"…the kernel SHALL write zero for all positions at indices >= the logical extent."*

Yet the kernel has no parameter to determine where the logical extent ends. The zero-fill is actually an emergent consequence of upstream invariants (zero-padded host input for the input dimension; zero-valued `summed_grad_h` at padding positions from Node 16 for the hidden dimension). This creates a behavioral claim the kernel cannot independently verify or enforce from its own interface — violating Article 1.4's closed-system property.

**Contrast:** Phase I kernels (Nodes 5, 8, 9) consistently receive *both* logical and padded extent parameters (e.g., `hidden_count` and `padded_hidden_count`, `total_output_class_count` and `padded_total_output_class_count`), enabling them to implement zero-fill by explicit boundary check. The asymmetry is unprincipled.

**Recommendation:** Add the missing logical-extent scalars:

| Kernel | Add Parameter | Purpose |
|:---|:---|:---|
| Node 17 (`backprop_shared_weights_chunk`) | `src_scalar_NATURAL_input_count` | Enables explicit zero-fill for input padding positions |
| Node 17 (`backprop_shared_weights_chunk`) | `src_scalar_NATURAL_hidden_count` | Enables explicit zero-fill for hidden padding positions |
| Node 18 (`backprop_shared_biases_chunk`) | `src_scalar_NATURAL_hidden_count` | Enables explicit zero-fill for hidden padding positions |

This aligns Phase III with the Phase I pattern, closes the interface verification loop, and — per Article 1.4.1 — gives the device an explicit boundary parameter for its assurance fallback. The redundancy with upstream invariants is precisely the kind of defensive redundancy the axiom blesses.

**Severity:** Axiom-level. The kernel's behavioral claim exceeds what its interface can prove.

---

## R2: Normalize Inactive CONDITIONAL Buffer Language

**Governing Principle:** CONTRACT Article 3.3 (Conditional Buffer Contract) and CONCEPT §9 (Multi-Backend Architecture) — backends must receive valid buffer descriptors.

**Finding:** Node 11's `src_buffer_GLOBAL_CONST_clipping_threshold_per_item` permits:

> *"the Host MAY pass a **NULL pointer** for this argument."*

Every other `[CONDITIONAL]` buffer across the header uses the Article 3.3 prescribed language:

> *"the Host MAY pass a **minimal stub buffer**."*

NULL global pointers are not universally safe. Vulkan requires valid descriptor bindings for all bound resources. OpenCL behavior for NULL `__global` pointers is implementation-defined. The "minimal stub buffer" convention (a single-element allocation) guarantees a valid device pointer exists across all three backends.

**Recommendation:** Replace "the Host MAY pass a NULL pointer for this argument" with "the Host MAY pass a minimal stub buffer" in Node 11's `src_buffer_GLOBAL_CONST_clipping_threshold_per_item` commentary, matching the five other CONDITIONAL buffers.

**Severity:** Backend-portability risk. Currently safe for OpenCL-only but would fail during Vulkan backend integration.

---

## R3: Include Controlling Flag in CONDITIONAL Annotations

**Governing Principle:** CONTRACT Article 3.3 — *"A `[CONDITIONAL]` annotation in the `@param` description line, **identifying the controlling flag**."*

**Finding:** All six `[CONDITIONAL]` buffer parameters use a bare annotation without naming the controlling flag:

```
// Current (all instances):
@param dest_buffer_GLOBAL_hidden_mask [CONDITIONAL] Derivative mask...

// Article 3.3 form:
@param dest_buffer_GLOBAL_hidden_mask [CONDITIONAL on dest_scalar_FLAG_produce_hidden_mask] Derivative mask...
```

The controlling flag IS identified in the Validation Preconditions block in every case, so the information is present but not at the mandated location (the `@param` description line). This violates the letter of Article 3.3 while satisfying its spirit.

**Recommendation:** Amend the six `@param` lines:

| Kernel | Parameter | Annotate With |
|:---|:---|:---|
| Node 4 | `dest_buffer_GLOBAL_hidden_mask` | `[CONDITIONAL on dest_scalar_FLAG_produce_hidden_mask]` |
| Node 5 | `src_buffer_GLOBAL_hidden_mask` | `[CONDITIONAL on src_scalar_FLAG_use_explicit_hidden_mask]` |
| Node 11 | `src_buffer_GLOBAL_CONST_clipping_threshold_per_item` | `[CONDITIONAL on src_scalar_FLAG_use_per_item_norm]` |
| Node 17 | `src_buffer_GLOBAL_hidden_mask` | `[CONDITIONAL on src_scalar_FLAG_use_explicit_hidden_mask]` |
| Node 18 | `src_buffer_GLOBAL_hidden_mask` | `[CONDITIONAL on src_scalar_FLAG_use_explicit_hidden_mask]` |

**Severity:** Documentation compliance. No functional impact, but strengthens machine-parseability of the contract for automated validation tooling.

---

## R4: Harmonize Padding Contract Across Buffer Provenance Chains

**Governing Principle:** Internal consistency of the buffer lifecycle model (CONCEPT §8) — a buffer's Padding Contract should be consistent at every point where it appears in the DAG.

**Finding:** The `clipped_partial_grad_hidden_activations_aos` buffer flows through Nodes 9 → 11 → 13. Its Padding Contract is declared consistently at every point *except* Node 13's input:

| Node | Role | Inner Dimension Padding Contract |
|:---|:---|:---|
| Node 9 (output) | `dest_` | `{Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}` |
| Node 11 (input) | `src_` | `{Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}` |
| Node 11 (output) | `dest_` | `{Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}` |
| **Node 13 (input)** | **`src_`** | **`{Type: NONE}`** |

Under Article 3.1, `{Type: NONE}` is technically valid because the Tensor Shape incorporates `padded_hidden_count`, which already embeds the CACHE alignment. The two interpretations are functionally equivalent. However, the asymmetry creates documentation friction when cross-referencing buffer flows and could mislead an implementer into believing the buffer has no alignment guarantees at that consumption point.

**Recommendation:** Change Node 13's `src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos` Padding Contract from `{Type: NONE}` to `{Type: CACHE, Formula: "128-byte alignment via padded_hidden_count"}`, matching its provenance chain.

**Severity:** Documentation consistency. No functional impact.

---

## R5: Formalize the Upstream Zero-Fill Dependency Model

**Governing Principle:** Architectural Elegance Feedback (CONCEPT Principle §1) — *"When emergent efficiency gains contradict current constraints... formalize the pattern as a documented architectural primitive."*

**Finding:** Several kernels rely on upstream invariants to guarantee zero values at padding positions without declaring this dependency explicitly. The most prominent instances:

1. **Node 17** relies on `summed_grad_h` (from Node 16) being zero at hidden padding positions, which in turn relies on Node 13's SoA output being `ZERO_REQUIRED`-initialized at padding module positions.
2. **Node 4**'s hidden activation output at padding positions depends on state-role weight/bias buffers having zero padding — an invariant maintained by host initialization and never explicitly documented in any kernel contract.
3. **Node 16** relies on its Node 13 input having zero at padding module positions (from the `ZERO_REQUIRED` initialization), propagating zeros through its reduction.

These dependencies form a **zero-propagation chain** that spans multiple nodes but is never documented as a first-class architectural invariant. Each kernel's zero-fill claim appears locally justified, but the chain's correctness depends on global reasoning that exists only in the implementer's mind.

**Recommendation:** Introduce a brief **§3.5 "Zero-Propagation Invariant"** clause in the CONCEPT.md document (or a new ADR), formally establishing:

> *"Padding positions in state-role parameter buffers (weights, biases) are host-initialized to zero and maintained at zero by the optimizer (which operates only on logical-extent positions). This invariant propagates through the forward pass (Nodes 4→5) and backward pass (Nodes 16→17→18), ensuring that padding positions in compute-role and storage-role intermediate buffers carry zero values without requiring per-kernel boundary checks at every consumption point."*

This makes the implicit dependency chain explicit, auditable, and centralized — rather than requiring each auditor to reconstruct it from scattered kernel contracts.

**Severity:** Architectural hygiene. The system is currently correct but the correctness argument is implicit and fragile under maintenance.

---

## R6: Unify `Idempotency` Classification Criteria

**Governing Principle:** Semantic Uniqueness (CONTRACT Axiom 1.2) and the Kernel Contract Model (CONTRACT Article 4.2) — the three idempotency categories should have deterministic selection criteria.

**Finding:** The `Idempotency` classification across kernels is consistent *in practice* but the selection criteria are undocumented. The implied convention appears to be:

| Classification | Applied To | Implied Criterion |
|:---|:---|:---|
| `Strictly Idempotent` | Nodes 4, 5, 6, 7, 11, 19, 21 | Deterministic read→compute→write; same inputs always produce same outputs; no in-place mutation of persistent state |
| `Associatively Non-Idempotent` | Nodes 8, 9, 10, 13, all aggregate/reduce | Contains parallel floating-point reduction whose result depends on work-group decomposition (FP non-associativity) |
| `Fundamentally Non-Idempotent (Stateful)` | Nodes 24, 25 | Modifies persistent state in-place; re-execution corrupts model parameters |

This convention is principled but not self-evident. A reasonable reader might argue that Node 11's local-memory L2 norm reduction makes it "Associatively Non-Idempotent" (it contains the same FP-non-associative reduction that classifies Nodes 8–10). The distinguishing criterion appears to be whether the kernel *writes to a shared accumulation buffer* versus *writing to its own dedicated output slot* — but this isn't stated.

**Recommendation:** Add a brief clarification to Article 4.2's `Idempotency` key definition, specifying the selection criteria:

> - **`Strictly Idempotent`**: The kernel's output is solely a deterministic function of its inputs. Re-execution with identical inputs and execution geometry overwrites the output with identical values. Internal FP reductions do not disqualify: the criterion is output-level determinism for fixed inputs and fixed dispatch geometry, not bit-exact reproducibility across geometries.
> - **`Associatively Non-Idempotent`**: The kernel accumulates results into a shared buffer whose final value depends on the composition of multiple dispatches, OR its output is a reduction whose result is input-order-sensitive in a composition context.
> - **`Fundamentally Non-Idempotent (Stateful)`**: The kernel modifies persistent model state in-place. Re-execution corrupts the model.

This makes the classification machine-enforceable and removes the ambiguity around reduction-containing-but-overwrite-writing kernels.

**Severity:** Documentation clarity. Prevents misclassification as the kernel count grows.

---

## R7: Explicit `Initialization Contract` for Node 6's Conditional Loss Write

**Governing Principle:** CONTRACT Article 3.1.1 and the Conditional Writer behavioral vocabulary (Article 4.3).

**Finding:** Node 6's `dest_buffer_GLOBAL_final_loss` correctly declares `Initialization Contract: {Type: ZERO_REQUIRED}` and the Behavioral Invariants correctly explain the conditional write predicate:

> *"Loss Write Predicate: The kernel writes to `dest_buffer_GLOBAL_final_loss[module][sample]` only when the target class index for the sample falls within the current tile's class chunk range."*

The Synchronization Model correctly identifies this as `Conditional Writer`. This is exemplary — the finding here is purely about raising the pattern as a candidate for broader application.

Node 8's `dest_buffer_GLOBAL_partial_grad_weights_module` uses `ZERO_REQUIRED_ADDITIVE` and documents that "Each tile writes only `classes_per_chunk` positions within the `padded_total_output_class_count`-wide innermost dimension." However, Node 8 is NOT classified as a `Conditional Writer` in its Synchronization Model — it's a `Dual Partial Renderer`. The distinction is subtle: Node 8 writes to a *deterministic* subset of positions (the class-chunk slice), not a *data-dependent* subset. So `Conditional Writer` doesn't apply in the Article 4.3 sense. This is correct.

**Recommendation:** No change required. The current classification is precise. This note serves as validation that the `Conditional Writer` vocabulary term is applied at the correct boundary (data-dependent predicate, not structural-slice-based writing), and should be documented as a clarifying example in Article 4.3 if ambiguity arises in future kernel development.

**Severity:** None (positive finding). Included for completeness.

---

## R8: Add `src_scalar_NATURAL_input_count` to Node 4's Interface

**Governing Principle:** Consistency of the closed-system property across the DAG; the Phase I precedent established by Nodes 5, 8, and 9.

**Finding:** Node 4 (`forward_pass`) receives `padded_input_count` but not `input_count`. Its Behavioral Invariants do not claim explicit padding zero-fill (correctly — the output shape is `padded_hidden_count`, not `padded_input_count`). However, the kernel iterates over the input dimension to compute the dot product `W*x + b`, and the iteration bound is implicitly `padded_input_count`. If the host ensures input padding positions are zero, the result is correct. But the kernel has no parameter to distinguish "iterate over all padded positions" from "iterate only over logical positions" — the former wastes computation, the latter requires `input_count`.

Every other kernel that iterates over a dimension where logical and padded extents may differ receives both parameters:

| Kernel | Dimension | Has Logical | Has Padded |
|:---|:---|:---|:---|
| Node 5 | hidden | `hidden_count` ✓ | `padded_hidden_count` ✓ |
| Node 8 | hidden | `hidden_count` ✓ | `padded_hidden_count` ✓ |
| Node 8 | class | `total_output_class_count` ✓ | `padded_total_output_class_count` ✓ |
| Node 9 | hidden | `hidden_count` ✓ | `padded_hidden_count` ✓ |
| **Node 4** | **input** | **absent** | `padded_input_count` ✓ |

**Recommendation:** Add `src_scalar_NATURAL_input_count` to Node 4. This enables the implementation to iterate only over logical input positions (a performance optimization for large padding margins), and aligns Node 4 with the universal Phase I pattern where both logical and padded extents are provided.

**Severity:** Consistency and performance opportunity. Not a correctness issue (zero-padded input produces correct results), but breaks the otherwise universal pattern.

---

## Summary Matrix

| # | Title | Principle | Severity | Action |
|:--|:------|:----------|:---------|:-------|
| R1 | Restore interface completeness (Nodes 17, 18) | Axiom 1.4 | **Axiom-level** | Add 3 scalar parameters |
| R2 | Normalize inactive-buffer language (Node 11) | Article 3.3 / Backend neutrality | **Portability risk** | 1 wording change |
| R3 | Include controlling flag in CONDITIONAL annotations | Article 3.3 | **Compliance** | 5 annotation edits |
| R4 | Harmonize padding contract (Node 13 input) | Buffer lifecycle consistency | **Consistency** | 1 Padding Contract edit |
| R5 | Formalize zero-propagation invariant | Principle §1 | **Architectural hygiene** | New CONCEPT.md clause or ADR |
| R6 | Unify idempotency classification criteria | Article 4.2 | **Documentation clarity** | Expand Article 4.2 definition |
| R7 | Validate Conditional Writer boundary | Article 4.3 | **None (positive)** | No change |
| R8 | Add `input_count` to Node 4 | Pattern consistency | **Consistency** | 1 scalar parameter |

**Total code changes:** 4 new scalar parameters, 1 wording fix, 5 annotation edits, 1 Padding Contract edit.
**Total document changes:** 1 new CONCEPT.md clause (or ADR), 1 Article 4.2 expansion.
