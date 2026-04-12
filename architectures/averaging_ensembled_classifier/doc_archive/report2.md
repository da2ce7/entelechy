## Principled Recommendations for the Zeroing Policy

### Governing Rationale

The Zero-Propagation Theorem (CONCEPT.md §3.6) claims to be "the emergent property of correct per-kernel invariant assignment." This makes it a **compositional proof**: its soundness rests on every link in the chain carrying an explicit, declared invariant. An undeclared invariant that happens to hold is a correctness accident — it survives until an implementation change breaks the unstated assumption. The following recommendations close the gap between the theorem's proof sketch and the kernel contracts it cites.

All recommendations are grounded in the architecture's own principles:

- **Principle §1 (Architectural Elegance Feedback):** Emergent properties that depend on unstated assumptions are the architectural equivalent of "hidden logic." They must be formalized.
- **Principle §6 (Architectural Hierarchy):** The Conceptual layer (theorem) claims properties that the Contractual layer (kernel headers) must enforce. Undeclared invariants violate the strict hierarchy.
- **CONTRACT Article 4.2:** The `Behavioral Invariants` key exists precisely to declare rules governing internal implementation. Omission is not neutrality — it is an unfalsifiable claim of "no constraints exist."

---

### Recommendation 1: Declare Padding Zero-Preservation on Node 11

**Severity:** Contract gap — theorem cites undeclared property.

**Node:** `clip_partial_gradients`

**Action:** Add to Behavioral Invariants:

> `Padding Zero-Preservation: The uniform-scaling algorithm applies a single multiplicative factor derived from the joint L2 norm to all positions in all four output buffers. No per-element additive term exists. Zero-valued positions in the input (established by Node 9's Padding Zero-Establishment for Grad_H, and by Node 8's Padding Zero-Preservation for Grad_ModW and Grad_ModB) are mapped to zero in the output for all finite scaling factors. This is a mathematical consequence of the single-scale-factor design, not an active zeroing step.`

**Rationale:** Node 11 is the sole intermediary between the zero-establishment kernels (8, 9) and the zero-consuming stages (13, 15). The theorem's chain breaks here without a declared invariant. The mathematical argument is trivial (`s × 0 = 0`), but the *declaration* is what makes the proof composable.

---

### Recommendation 2: Declare Padding Zero-Preservation on Node 13

**Severity:** Contract gap — theorem explicitly names this invariant but the contract omits it.

**Node:** `gather_and_permute_grad_hidden_activations`

**Action:** Add to Behavioral Invariants:

> `Padding Zero-Preservation: The kernel writes only to destination positions corresponding to tiles within the logical (module, batch, hidden) domain. For the module dimension: positions at indices ≥ total_modules_count within the padded_total_modules_count stride are never written; the ZERO_REQUIRED initialization of the destination buffer is the primary guarantor of zeros at these positions. For the hidden dimension: the input's zero-valued padding positions (indices ≥ hidden_count, established by Node 9 and preserved by Node 11) are read, summed across class chunks (sum of zeros = zero), and stored via store_storage() — preserving zero at the corresponding output positions.`

**Rationale:** The CONCEPT.md proof sketch states: *"Node 13's ZERO_REQUIRED output initialization ensures padded module positions carry zero (host is the guarantor; kernel applies Padding Zero-Preservation)."* The contract must contain what the theorem claims it contains.

---

### Recommendation 3: Declare Padding Zero-Preservation on Node 19

**Severity:** Contract gap — identical structural role to Node 11.

**Node:** `clip_shared_gradients_chunk`

**Action:** Add to Behavioral Invariants:

> `Padding Zero-Preservation: Identical mathematical property to Node 11. The uniform-scaling algorithm applies a single multiplicative factor to all positions in both output buffers. Zero-valued positions established by Node 17's Padding Zero-Establishment (for weights: indices ≥ input_count × hidden_count) and Node 18's Padding Zero-Establishment (for biases: indices ≥ hidden_count) are mapped to zero in the output for all finite scaling factors.`

**Rationale:** Nodes 17 and 18 declare Padding Zero-Establishment on their output buffers. Node 19 consumes those buffers and produces the collection consumed by Node 20's reduction engine, which reads the full `partial_width` (including padding). The preservation link must be declared.

---

### Recommendation 4: Document Emergent Zero Propagation on Node 4

**Severity:** Documentation gap — the theorem relies on a three-party invariant that no single contract declares.

**Node:** `forward_pass`

**Action:** Add to Behavioral Invariants:

> `Padding Zero Propagation (Emergent): Hidden-dimension padding positions (indices ≥ padded_hidden_count's logical extent) in dest_buffer_GLOBAL_hidden_activations and dest_buffer_GLOBAL_hidden_mask carry zero when all three conditions hold: (a) input-dimension padding in src_buffer_GLOBAL_input is zero, (b) hidden-dimension padding in weights is zero, (c) hidden-dimension padding in biases is zero. This is a mathematical consequence of the affine transform (W·x + b) and ReLU(0) = 0, not an active zeroing step. The kernel does not distinguish padding from logical positions — it receives only padded dimensions (padded_input_count, padded_hidden_count). The host initialization and optimizer (Node 24) are the co-guarantors of the three preconditions.`

**Rationale:** This kernel is the genesis of hidden-dimension zeros that propagate through the entire backward pass (Nodes 5, 8, 9, 11, 13, 16, 17, 18). The emergent property is real and correct, but a future optimization (e.g., a bias initialization strategy that doesn't zero-pad, or a weight pruning scheme) could silently violate the unstated precondition. Making the dependency explicit converts a fragile accident into a verifiable contract.

---

### Recommendation 5: Correct the Zero-Propagation Theorem's Node 24 Proof Mechanism

**Severity:** Logical inconsistency — the theorem's claimed mechanism ("operates only on logical-extent positions") does not match the actual mechanism.

**Node:** `adam_update` and the theorem itself.

**Action (Two-Part):**

**(A) Amend the theorem's proof in CONCEPT.md §3.6, item 1:**

Replace:
> *"Optimizer kernels (Nodes 24, 25) operate only on logical-extent positions, applying Padding Zero-Preservation — they cannot corrupt host-established zeros."*

With:
> *"Optimizer kernels (Nodes 24, 25) process the full padded extent. At padding positions, the gradient is zero (guaranteed by the upstream chain), and moment vectors are host-initialized to zero. By the Adam recurrence, zero gradient and zero prior moments produce zero updates: m₁ ← β₁·0 + (1−β₁)·0 = 0, m₂ ← β₂·0 + (1−β₂)·0 = 0, Δw = 0. Parameters at padding positions remain at their host-initialized zero. This maintains state-role padding by induction over training steps, not by positional exclusion."*

**(B) Add to Node 24's Behavioral Invariants:**

> `Padding Zero-Preservation (Inductive): The kernel processes all positions in [parameter_offset, parameter_offset + parameter_count). At padding positions where the gradient is zero and moment vectors are zero (both host-initialized), the Adam recurrence produces zero moment updates and zero parameter change. This preserves padding zeros by mathematical induction over training steps — the kernel does not distinguish padding from logical positions.`

**Rationale:** The current theorem claims a mechanism that doesn't exist (positional exclusion) to explain a property that does hold (zero preservation). This is the most dangerous class of documentation error: a correct conclusion with an incorrect proof. If someone later reads the proof and "optimizes" Node 24 to skip padding positions (trusting the theorem's claim that it already does), they would introduce a behavioral change that happens to be redundant — but their reasoning was wrong. A future non-Adam optimizer with a non-zero update at zero gradient (e.g., weight decay) would then silently violate the theorem. Recording the actual mechanism makes the dependency explicit.

---

### Recommendation 6: Add a Formal Padding Invariant Column to CONTRACT Article 4.2

**Severity:** Structural improvement — prevents future omissions.

**Action:** Extend the `Behavioral Invariants` key's recognized vocabulary with a rule:

> Every kernel whose output buffer has a padded dimension consumed by a downstream kernel at the full padded extent **SHALL** declare exactly one of: `Padding Zero-Establishment`, `Padding Zero-Preservation`, or `Padding Zero Propagation (Emergent)` with its preconditions. Omission is not equivalent to "no padding concern" — it is a contract violation if the Zero-Propagation Theorem's proof chain traverses that kernel.

Additionally, consider a summary table (parallel to the one in the review) as a mandatory appendix to the theorem in CONCEPT.md, mapping each node to its declared invariant. This makes the compositional proof auditable at a glance.

**Rationale:** The current vocabulary defines the invariant types but does not mandate their assignment. Four kernels (11, 13, 19, plus the emergent case of 4) passed review with no declared padding behavior. A mandate converts the theorem from "emergent property of hopefully-correct assignments" to "emergent property of verifiably-complete assignments."

---

### Recommendation 7: Verify Node 16's Interaction with Module-Dimension Padding

**Severity:** Low — likely correct but unverified by the current proof.

**Node:** `stabilize_and_reduce_grad_hidden_activations`

**Action:** Confirm and document that Node 16 reads `total_modules_count` columns (not `padded_total_modules_count`) from the SoA buffer produced by Node 13. The contract specifies `src_scalar_NATURAL_total_modules_count` and `src_scalar_NATURAL_padded_total_modules_count` as separate parameters — the kernel must use the logical count for its reduction width and the padded count for its row stride calculaton only. If this is the case, padding columns are never read and no invariant is needed. If the kernel reads the full padded width, it relies on Node 13's `ZERO_REQUIRED` initialization (via Recommendation 2's preservation invariant) and should declare `Padding Zero-Preservation` on its own output.

**Rationale:** Node 16's output (`summed_grad_hidden_activations`) feeds Nodes 17 and 18, which use it as a compute-role vector with no padding. The compute-role output has tensor shape `(total_batch_count × padded_hidden_count)` with no module dimension — the module dimension was reduced away. The question is whether the reduction was over `total_modules_count` or `padded_total_modules_count` elements. The answer determines whether zero-padded module positions contribute (correctly, as zero) or are excluded. Either mechanism is correct; the contract should state which one applies.

---

### Summary of Actions

| # | Node(s) | Action | Type |
|---|---------|--------|------|
| 1 | 11 | Declare `Padding Zero-Preservation` | Contract amendment |
| 2 | 13 | Declare `Padding Zero-Preservation` | Contract amendment |
| 3 | 19 | Declare `Padding Zero-Preservation` | Contract amendment |
| 4 | 4 | Document `Padding Zero Propagation (Emergent)` | Contract amendment |
| 5 | 24 + Theorem | Correct proof mechanism + declare inductive preservation | CONCEPT + Contract |
| 6 | CONTRACT Art. 4.2 | Mandate padding invariant for padded-output kernels | Structural rule |
| 7 | 16 | Verify and document module-dimension reduction bound | Audit action |

No recommendation changes runtime behavior. All recommendations formalize properties that already hold, closing the gap between the theorem's claimed proof and the contracts it rests upon.