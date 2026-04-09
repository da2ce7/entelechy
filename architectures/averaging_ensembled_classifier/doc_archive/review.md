# Principled Recommendations for CONTRACT.md Revision 8

## Framing Principle

These recommendations follow CONCEPT.md §1 (Architectural Elegance Feedback): the kernel header's parameter inventory has revealed vocabulary and structural patterns that the Contract's formal abstractions do not yet subsume. Per the directive, we **suspend ad-hoc workarounds** and instead **formalize each pattern as a first-class primitive** through Contract amendments.

Each recommendation identifies the pressure source, the principle it invokes, and the minimal amendment required.

---

## Recommendation 1: Establish Core Dimension Primitives in Article 8 §2.0

**Pressure Source:** Three foundational dimensional terms — `batch`, `hidden`, and `tile` — appear in 20+ scalar parameters across nearly every kernel but have no §2.0 entry. The Lexicon defines compound forms (`hidden_activations`, `hidden_mask`, `batch_chunk`, `flat_tile`) but not the standalone dimension identifiers from which they are composed.

**Principle Invoked:** Article 8 §1.0 Mandate — *"Usage of any term not defined herein is a violation."*

**Amendment:** Add to Article 8 §2.0, Group 6 (Control, Scoping & Dimensionality Primitives):

| Term | Definition |
|:---|:---|
| `batch` | The primary sample dimension of a training batch. Used as a dimensional qualifier in counts, offsets, and chunk decompositions. |
| `hidden` | The dimensionality of the intermediate (hidden) representation layer. Used as a dimensional qualifier in counts and padding parameters. |
| `tile` | A discrete, indivisible unit of parallel work produced by the flattening of a logical problem grid. Distinguished from `flat_tile` (§3.0) which denotes the specific decomposition strategy; `tile` is the resulting work unit itself. |

**Rationale:** These are irreducible dimensional primitives — they cannot be derived from existing entries. Every `total_batch_count`, `padded_hidden_count`, and `total_tile_count` in the system builds on them. Their absence forces a reading where every usage is technically a violation, despite being the most natural terms in the architecture's vocabulary.

---

## Recommendation 2: Register ADR-031 Flag Identifiers in Article 8 §6.0

**Pressure Source:** The mask strategy introduced by ADR-031 and CONCEPT §11 creates two new FLAG parameters (`dest_scalar_FLAG_produce_hidden_mask` on Node 4; `src_scalar_FLAG_use_explicit_hidden_mask` on Nodes 5, 17, 18) whose identifiers are not registered in §6.0's closed flag vocabulary.

**Principle Invoked:** Article 8 §1.0 Mandate, reinforced by §6.0's implicit closed-set semantics.

**Amendment:** Add to Article 8 §6.0:

| Term | Definition |
|:---|:---|
| `produce_hidden_mask` | Controls whether the forward pass kernel writes the ReLU derivative mask to the `hidden_mask` buffer. When 0, mask writes are skipped and the Host MAY pass a minimal stub buffer. |
| `use_explicit_hidden_mask` | Selects the source for the ReLU derivative in consuming kernels. When 0, the mask is derived internally from stored activations (`activation > 0`). When 1, the mask is read from an explicit `hidden_mask` buffer. |

**Rationale:** These flags embody a Policy-tier decision (mask strategy selection based on `PrecisionConfig`) that crosses the host-device boundary. Their registration makes the flag vocabulary exhaustive — a property §6.0 implicitly guarantees but currently doesn't deliver for the mask strategy.

---

## Recommendation 3: Add Transitional State Prefixes to Article 8 §4.0

**Pressure Source:** Two context modifiers used in parameter names — `intermediate_` and `write_` — appear in the kernel header but are absent from §4.0's prefix/suffix inventory.

- `intermediate_` appears in `update_buffer_GLOBAL_intermediate_grad` (Node 15b/20b's `clip_intermediate_grad`).
- `write_` appears in `dest_scalar_NATURAL_weights_write_offset` and `dest_scalar_NATURAL_biases_write_offset` (Node 19).

**Principle Invoked:** Lexicon completeness (§1.0 Mandate) and semantic precision (Article 1.2 — Axiom of Semantic Uniqueness). These modifiers carry meaning distinct from existing entries.

**Amendment:** Add to Article 8 §4.0:

| Type | Term | Function |
|:---|:---|:---|
| Prefix | `intermediate_` | Denotes a buffer at a transitional reduction stage — past `partial_` (raw/clipped) but before `summed_` (fully reduced). Applicable to buffers that are the output of one aggregation stage and the input to a subsequent clip or aggregation stage. |
| Prefix | `write_` | Pertaining to the computed write position within a destination buffer. Distinct from `out_` (which marks buffer affinity) in that `write_` qualifies an address offset calculated by the host for placement within a collection buffer. |

**Rationale:** `intermediate_` fills a genuine semantic gap between `partial_` and `summed_` in the reduction pipeline. `write_` distinguishes host-calculated placement offsets from the generic `out_` qualifier, which the existing Lexicon uses for buffer identity rather than address arithmetic.

---

## Recommendation 4: Add Structural Layout Suffix for Flattened Representations

**Pressure Source:** The K-fan-in reduction kernels use `src_buffer_GLOBAL_CONST_offset_list_flat` where `_flat` is not registered in §5.0's Specialized Layout vocabulary.

**Principle Invoked:** §5.0 establishes a closed set of layout suffixes (`_simd_major`, `_aos`, `_soa`, `_permuted`) — `_flat` is an unlisted sixth entry.

**Amendment:** Add to Article 8 §5.0 (Specialized Layout):

| Domain | Term | Type | Definition |
|:---|:---|:---|:---|
| Specialized Layout | `_flat` | Suffix | A logically multi-dimensional structure (e.g., node × K offset pairs) serialized into a contiguous 1D representation. The flattening convention (e.g., row-major) follows Article 1.3. |

**Rationale:** The K-fan-in offset list is logically `(node_count, fan_in_K)` but physically a flat array. The `_flat` suffix communicates this structural choice to both human readers and the host validation layer.

---

## Recommendation 5: Register `output` as a Utility Primitive or Standardize on `partial`

**Pressure Source:** `dest_buffer_GLOBAL_stage_output` in the K-fan-in reduction kernels uses `output` as a noun — but the Lexicon only defines `out_` (a prefix in §4.0). The term `stage_output` denotes the result of a single reduction stage, which is semantically a `partial` result in the tree.

**Principle Invoked:** Lexicon precision — avoid unregistered synonyms that could diverge from canonical meanings.

**Amendment (Option A):** Add to Article 8 §2.0, Group 7 (Utility Primitives):

| Term | Definition |
|:---|:---|
| `output` | The result product of a computational stage or kernel. When used as a buffer name component, denotes the primary write destination for the kernel's computed result. |

**Amendment (Option B):** Rename to `dest_buffer_GLOBAL_stage_partial` in both the header and all backend implementations, reflecting that a reduction stage's output is architecturally a `partial` until the tree terminates. This tightens adherence to the existing vocabulary but requires a header and implementation update.

**Recommendation:** Option B

---

## Recommendation 6: Formalize Pluralization Convention

**Pressure Source:** The Lexicon universally defines singular forms (`module`, `class_chunk`, `output_class`, `weight`, `bias`) while the header universally uses plurals in dimensional contexts (`modules_per_chunk`, `num_class_chunks`, `total_modules_count`, `classes_per_chunk`, `weights_parameter_count`, `biases_parameter_count`). This systemic pattern affects every kernel but has no governing rule.

**Principle Invoked:** Lexicon completeness — a convention that applies to every parameter name should be formally stated.

**Amendment:** Add to Article 8 §1.0 immediately after the Mandate paragraph:

> **§1.3 Pluralization Rule.** The Canonical Lexicon defines terms in their singular form. Plural forms of defined terms are implicitly valid when they appear as dimensional components — specifically when combined with `num_`, `total_`, `_count`, `_per_*`, or analogous cardinality modifiers. The plural is the natural morphological form for expressing "how many of this entity." Singular and plural forms carry no semantic distinction beyond grammatical number; they refer to the same Lexicon entry.

**Rationale:** This converts an implicit convention into an explicit rule, eliminating the class of spurious violations where `biases_parameter_count` would fail because only `biases` (singular `bias`) is defined. The rule is natural, unsurprising, and already universally practiced.

---

## Recommendation 7: Register `Kernel Bifurcation` as a Recognized `@kernel_contract` Key

**Pressure Source:** Article 4.2 defines exactly five recognized keys for the `@kernel_contract` block. CONTRACT §7.0's *Exception: Architecturally-Mandated Kernel Bifurcation* requires that the distinction *"is documented in the kernel's `@kernel_contract` block"* — but provides no key to document it with. The header resolves this by introducing an unrecognized `Kernel Bifurcation` key on Nodes 6 and 7.

**Principle Invoked:** Internal consistency of the Contract document. A mandatory documentation requirement must have a valid syntactic vehicle.

**Amendment:** Add to Article 4.2's key table:

| Key | Definition | Status |
|:---|:---|:---|
| **`Kernel Bifurcation`** | Documents that this kernel is one half of a CONCEPT.md Principle 3(B) bifurcated pair. The value SHALL include: (1) an explicit reference to Principle 3(B), (2) identification of the peer kernel, and (3) a summary of the structural incompatibility (differing buffer types, shapes, or DAG edges) that prevents unification under a single interface with a FLAG parameter. This key is applicable only when the §7.0 Exception criteria are met. | **Optional** |

**Rationale:** This closes a self-referential gap in the Contract. The §7.0 Exception creates a documentation obligation that the §4.2 key vocabulary cannot fulfil. The amendment makes the obligation actionable.

---

## Recommendation 8: Clarify Conditional Buffer Semantics with a `@param` Convention

**Pressure Source:** Several kernels (Nodes 4, 5, 17, 18) have conditionally-accessed buffers (`hidden_mask`) whose validity depends on a FLAG parameter. The header handles this well with `[CONDITIONAL]` annotations and detailed Validation Preconditions, but this pattern has no formal recognition in Article 3's Parameter Commentary Contract.

**Principle Invoked:** Architectural Elegance Feedback (§1) — a recurring pattern should be formalized rather than left as ad-hoc commentary.

**Amendment:** Add to Article 3 as §3.3 (Conditional Buffer Contract):

> **§3.3 Conditional Buffer Contract.** When a buffer parameter's access is gated by a FLAG scalar (the *controlling flag*), the parameter's commentary block SHALL include:
>
> 1. A `[CONDITIONAL]` annotation in the `@param` description line, identifying the controlling flag.
> 2. Validation Preconditions specifying: (a) the flag value under which the buffer is accessed, (b) the full allocation and bounds requirements when active, and (c) the permissible stub-buffer behavior when inactive (*"the Host MAY pass a minimal stub buffer"*).
>
> The `[CONDITIONAL]` annotation is a documentation convention; it does not introduce a new flow prefix or memory scope. The buffer's `[Flow]` prefix (`src_` or `dest_`) reflects its role when active.

**Rationale:** This elevates a pattern used by 4 kernels × 2 flags = 8 conditional buffer instances from implicit convention to explicit contract. It provides the host validation layer with a formal basis for understanding when buffer allocation can be elided.

---

## Recommendation 9: Add Reduction Engine Topology Terms to Article 8 §5.0

**Pressure Source:** The K-fan-in kernels introduce `fan_in`, `node`, and `stage` as parameter name components (`src_scalar_NATURAL_fan_in_K`, `src_scalar_NATURAL_node_count`). These terms are correctly defined in §5.0 under "Reduction Engine" — however, I note for completeness that their placement in the §5.0 table is sound and no amendment is needed.

**Status: No action required.** This recommendation is included for traceability — to confirm that the K-fan-in kernel vocabulary was reviewed and found covered.

---

## Recommendation 10: Add `input` Dimension Primitive to Article 8 §2.0 Group 6

**Pressure Source:** `padded_input_count` appears in Nodes 4, 5, 17, and 19. The Lexicon defines `input` in Group 1 as *"The initial, untransformed data set for a complete computation"* — a data role, not a dimensional primitive. When `input` appears in `padded_input_count`, it functions as a dimension identifier (the feature count of the input layer), not a data role.

**Principle Invoked:** Semantic precision (Article 1.2) — a term should not silently serve double duty across semantic categories.

**Amendment:** Add a dimensional usage note to Group 1's `input` entry, or add a separate Group 6 entry:

| Term | Definition |
|:---|:---|
| `input` *(Group 6 dimensional usage)* | When used as a dimensional qualifier (e.g., `padded_input_count`, `input_count`), denotes the feature dimensionality of the primary input data. This dimensional usage is distinct from the Group 1 data-role usage, which denotes the data buffer itself. |

**Rationale:** The distinction between "the input buffer" and "the input dimension" is architecturally meaningful — `padded_input_count` is a dimension scalar, not a buffer reference. A brief note acknowledging this dual usage prevents future confusion.

---

## Summary of Amendments

| # | Target | Type | Scope |
|:---|:---|:---|:---|
| R1 | Article 8 §2.0 Group 6 | 3 new entries | `batch`, `hidden`, `tile` |
| R2 | Article 8 §6.0 | 2 new entries | `produce_hidden_mask`, `use_explicit_hidden_mask` |
| R3 | Article 8 §4.0 | 2 new entries | `intermediate_`, `write_` |
| R4 | Article 8 §5.0 | 1 new entry | `_flat` |
| R5 | Article 8 §2.0 Group 7 | 1 new entry | `output` |
| R6 | Article 8 §1.0 | 1 new subsection | Pluralization rule (§1.3) |
| R7 | Article 4.2 | 1 new key | `Kernel Bifurcation` |
| R8 | Article 3 | 1 new subsection | Conditional Buffer Contract (§3.3) |
| R9 | — | No action | Reduction Engine terms confirmed covered |
| R10 | Article 8 §2.0 Group 1/6 | 1 note/entry | `input` dimensional usage |

**Total Contract changes:** 11 new Lexicon entries, 2 new Article subsections, 1 new `@kernel_contract` key. Zero header changes required — the header is ahead of the Contract, not behind it.
