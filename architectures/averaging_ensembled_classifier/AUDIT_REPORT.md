## **Cross-Document Audit Report: Kernel Header × System Contract × Architectural Concept**

**Scope:** `kernels.cl.h` (Rev.), `CONTRACT.md` (Rev. 6), `CONCEPT.md` (Rev. 7)
**Date:** 10 March 2026

---

### **Findings Summary**

| ID    | Severity              | Category                      | Status |
| :---- | :-------------------- | :---------------------------- | :----- |
| A1    | Contract Violation    | Missing Build-Time Symbols    | Resolved |
| A2    | Contract Violation    | Forbidden Lexical Term        | Resolved |
| A3    | Contract Violation    | Memory Scope Token Mismatch   | Resolved |
| A4    | Contract Violation    | Duplicate Lexicon Definition  | Resolved |
| B1    | Cross-Doc Inaccuracy  | Buffer Naming Inconsistency   | Resolved |
| B2    | Cross-Doc Inaccuracy  | Wrong Node Reference          | Resolved |
| B3    | Cross-Doc Inaccuracy  | Incomplete Kernel List        | Resolved |
| B4    | Cross-Doc Inaccuracy  | Missing DAG Edge              | Resolved |
| B5    | Cross-Doc Inaccuracy  | Incorrect DAG Edges           | Resolved |
| C1    | Naming Inconsistency  | Spurious `_count` Suffixes    | Open   |
| D1    | Editorial             | List Numbering Artifact       | Open   |
| D2    | Editorial             | Unexplained Node Gap          | Open   |
| E1    | Informational         | Stale Policy Name Reference   | Open   |

---

### **Category A: Contract Violations (Header ↔ CONTRACT)**

#### A1. Undocumented Mandatory Build-Time Symbols

**Affected Documents:** `kernels.cl.h` §Preamble, `CONTRACT.md` Article 6

The kernel header enforces `SCALAR_IS_HALF` and `NUMERICAL_STABILITY_EPSILON` as mandatory build-time symbols via `#error` directives:

```c
// kernels.cl.h, lines 40–43
#ifndef SCALAR_IS_HALF
#error "System Contract Violation: SCALAR_IS_HALF must be defined by the host build system."
#endif
```

```c
// kernels.cl.h, lines 80–82
#ifndef NUMERICAL_STABILITY_EPSILON
#error "System Contract Violation: NUMERICAL_STABILITY_EPSILON must be defined by the host build system."
#endif
```

CONTRACT Article 6 ("Mandatory Build-Time Symbols") enumerates only `SCALAR_TYPE`, `SIMD_WIDTH`, and `C_TILE_SIZE`. The two enforced symbols have no contractual backing. The header's enforcement is therefore ungrounded — either the header enforcement is unauthorized, or the CONTRACT is incomplete.

**Resolution:** Add `SCALAR_IS_HALF` and `NUMERICAL_STABILITY_EPSILON` to CONTRACT Article 6.

---

#### A2. Forbidden Lexical Terms in Kernel Names

**Affected Documents:** `kernels.cl.h` (kernel declarations), `CONTRACT.md` Article 8 §7.0, `CONCEPT.md` Principle 3

CONTRACT Article 8 §7.0 explicitly forbids `cce` and `bce`:

> | Term | Reason | Replacement |
> | :--- | :--- | :--- |
> | `cce`/`bce` | Problem-specific type in name. | Use generic terms (`loss`, `targets`); type is handled by a `FLAG` param. |

The kernel header declares two kernels that violate this rule:

- `compute_probs_loss_cce_chunk` (Node 6)
- `compute_probs_loss_bce_chunk` (Node 7)

However, CONCEPT Principle 3(B) architecturally mandates separate kernels when _"divergence is complex or imposes conflicting memory patterns."_ Nodes 6 and 7 are fundamentally incompatible:

| Property               | Node 6 (CCE)                            | Node 7 (BCE)                               |
| :---------------------- | :-------------------------------------- | :----------------------------------------- |
| Target tensor type      | `__global const int*`                   | `__global const SCALAR_TYPE*`              |
| Target tensor shape     | `(total_batch_count)`                   | `(total_batch_count, padded_output_class)` |
| Loss output strategy    | Scatter-write to monolithic `final_loss`| Partial renderer to `partial_loss`         |
| Loss aggregation needed | No                                      | Yes (via Node 14)                          |

These differ in type signature, memory layout, and downstream DAG topology. Unification via a `FLAG` parameter would violate Principle 3's modularity intent and produce a "smart" kernel with complex internal branching over incompatible memory patterns. The §7.0 prohibition was drafted assuming a single-kernel-with-flag model and does not account for cases where separate kernels are the architecturally correct strategy.

**Resolution:** Add a scoped exception to CONTRACT §7.0 for kernel names where separate kernels are required by CONCEPT Principle 3(B), or adopt an alternative naming scheme (e.g., `_softmax_path` / `_sigmoid_path`).

---

#### A3. `GLOBAL_` vs `GLOBAL_CONST_` Semantic Mismatch

**Affected Documents:** `kernels.cl.h` (all kernels), `CONTRACT.md` Article 2.1.1

CONTRACT Article 2.1.1 defines:

> - **`GLOBAL_`**: Standard read/write `__global` device memory.
> - **`GLOBAL_CONST_`**: Read-only `__global` device memory (qualified as `__global const`).

By this literal definition, any `__global const SCALAR_TYPE*` parameter should carry the `GLOBAL_CONST_` scope token. In practice, the header systematically uses bare `GLOBAL_` for read-only transient pipeline data while reserving `GLOBAL_CONST_` exclusively for persistent learnable model state:

| Actual Usage                                       | Scope Token Used | C Qualifier          |
| :------------------------------------------------- | :--------------- | :------------------- |
| Learnable weights, biases, temperatures             | `GLOBAL_CONST_`  | `__global const`     |
| Transient inputs, masks, partials, activations      | `GLOBAL_`        | `__global const`     |
| Writable destinations                               | `GLOBAL_`        | `__global`           |

Representative examples of the discrepancy:

- `src_buffer_GLOBAL_input` — declared `__global const SCALAR_TYPE*`
- `src_buffer_GLOBAL_sample_mask` — declared `__global const SCALAR_TYPE*`
- `src_buffer_GLOBAL_partial_probs` — declared `__global const SCALAR_TYPE*`
- `src_buffer_GLOBAL_partial_collection` — declared `__global const SCALAR_TYPE*`

The implemented convention draws a _semantic_ distinction (persistent model state vs. transient data flow) rather than the _syntactic_ distinction (read-only vs. read-write) that the CONTRACT defines. This convention is arguably more useful, but it directly contradicts the published definition.

**Resolution:** Amend CONTRACT Article 2.1.1 to formalize the actual semantic distinction, e.g.:

> - **`GLOBAL_`**: Standard `__global` device memory for pipeline data (may carry `const` qualifier when used as a source).
> - **`GLOBAL_CONST_`**: Read-only `__global` device memory holding persistent model state that is invariant for the duration of a kernel dispatch.

---

#### A4. Duplicate `final_` Prefix Definition

**Affected Documents:** `CONTRACT.md` Article 8 §4.0

The Context Modifiers table in §4.0 defines the `final_` prefix twice:

> | Prefix | `final_` | Denotes a fully reduced, final result. |
> | Prefix | `final_` | Denotes a fully processed, normalized result ready for consumption by a final state-modifying kernel (e.g., optimizer). |

This violates the spirit of Article 1.2 (Axiom of Semantic Uniqueness) and creates ambiguity — the first definition is broader, the second is stricter.

**Resolution:** Remove the first, broader entry and retain only the precise definition, or merge them into a single entry.

---

### **Category B: Cross-Document Inconsistencies (Header ↔ CONCEPT)**

#### B1. Node 18: `final_` vs `summed_` Buffer Naming

**Affected Documents:** `kernels.cl.h` (Nodes 17, 18), `CONTRACT.md` Article 8 §4.0

Node 17 (`backprop_shared_weights_chunk`) names its upstream gradient input:
```
src_buffer_GLOBAL_summed_grad_hidden_activations
```

Node 18 (`backprop_shared_biases_chunk`) names the same logical buffer:
```
src_buffer_GLOBAL_final_grad_hidden_activations
```

Per CONTRACT §4.0:
- `summed_` = "a buffer whose elements are the result of a batch-wide reduction (summation)"
- `final_` = "a fully processed, normalized result ready for consumption by a final state-modifying kernel"

The Node 16 output is a summed reduction result that has _not_ been normalized by Node 21. It is consumed by Nodes 17 and 18 as an intermediate, not as a final gradient. Node 18's use of `final_` is semantically incorrect.

**Resolution:** Rename Node 18's parameter to `src_buffer_GLOBAL_summed_grad_hidden_activations` for consistency with Node 17.

---

#### B2. Node 24 `adam_update` — Wrong Source Node Reference

**Affected Documents:** `kernels.cl.h` (Node 24)

The `@param` commentary for `src_buffer_GLOBAL_final_grad` states:

> "The buffer containing the final, normalized, batch-averaged gradients from **Node 20**."

Per the CONCEPT DAG, the data flow is:

```
Node 15/20 (Summed Grad) → Node 21 (normalize_gradients) → Node 22 (Barrier) → Node 24 (adam_update)
```

The immediate producer of `final_grad` buffers is Node 21, not Node 20.

**Resolution:** Change the reference from "Node 20" to "Node 21."

---

#### B3. Incomplete Placement Contract Kernel List

**Affected Documents:** `CONCEPT.md` §"The Placement Contract"

The CONCEPT states:

> "This contract applies to kernels: **(7), (8), (9), (10), (17), (18)**."

Examining the header, the following additional kernels declare `Placement Contract` keys on their output parameters:

- **Node 6** (`compute_probs_loss_cce_chunk`): `dest_buffer_GLOBAL_partial_probs` carries `Placement Contract: grid_mod_cls(...)`.
- **Node 11** (`clip_partial_gradients`): All four `dest_buffer_GLOBAL_clipped_partial_*` outputs carry `Placement Contract: grid_mod_cls(...)`.

**Resolution:** Expand the list to: **(6), (7), (8), (9), (10), (11), (17), (18)**.

---

#### B4. Missing `hidden_mask` Edge in CONCEPT DAG

**Affected Documents:** `CONCEPT.md` (Mermaid DAG), `kernels.cl.h` (Node 5)

The header's `render_logits_chunk` (Node 5) accepts `src_buffer_GLOBAL_hidden_mask` as an input parameter. The CONCEPT DAG only shows:

```mermaid
hidden_i --> K5["(5) render_logits_chunk"]
```

There is no edge from the `hidden_mask` data node (produced by Node 4 alongside `hidden_activations`) into Node 5. The mask is required for the kernel to correctly zero out padded hidden dimensions during the linear transform.

**Resolution:** Add a `hidden_mask` data node as an output of Node 4, and draw an edge from it into Node 5 (and into Nodes 8, 9, 10 if applicable).

---

#### B5. Incorrect Data Flow Edges for Nodes 17 and 18 in CONCEPT DAG

**Affected Documents:** `CONCEPT.md` (Mermaid DAG), `kernels.cl.h` (Nodes 17, 18)

The DAG currently shows:

```
P_Shared & hidden_i --> K17
Input_i --- K17
Input_i --- K18
hidden_i --> K18
```

Two errors are present:

**1. `P_Shared --> K17` is spurious.** Node 17's header signature does not consume shared weight parameters. The weight gradient computation ($\nabla W = X^T \cdot \nabla H$) requires the _input data_ and the _upstream gradient_, not the current weight values. The `P_Shared` edge should be removed from K17.

**2. `Input_i --- K18` is spurious.** Node 18's header signature does not accept an input buffer. The bias gradient computation ($\nabla b = \sum \nabla H$) requires only `hidden_activations`, `summed_grad_H`, and `sample_mask`. The `Input_i` edge should be removed from K18.

**Resolution:** Correct the DAG edges to:

```
Input_i & hidden_i --> K17
hidden_i --> K18
Summed_Grad_H -. "slice" .-> K17 & K18
```

---

### **Category C: Internal Naming Inconsistencies (within Header)**

#### C1. Node 13 — Spurious `_count` Suffixes on Dimensional Parameters

**Affected Documents:** `kernels.cl.h` (Node 13)

Node 13 (`gather_and_permute_grad_hidden_activations`) uses uniquely suffixed parameter names:

| Node 13 Name                                       | Equivalent in All Other Kernels            |
| :------------------------------------------------- | :----------------------------------------- |
| `src_scalar_NATURAL_modules_per_chunk_count`        | `src_scalar_NATURAL_modules_per_chunk`      |
| `src_scalar_NATURAL_num_module_chunks_count`         | `src_scalar_NATURAL_num_module_chunks`       |
| `src_scalar_NATURAL_num_class_chunks_count`          | `src_scalar_NATURAL_num_class_chunks`        |

Per CONTRACT Article 8 §4.0, the `_count` suffix denotes _"the number of elements to process, relative to a corresponding `_offset`."_ These parameters are not element counts relative to an offset; they are partition counts. Their semantic meaning is identical to their counterparts in Nodes 8, 9, 10, and 11. The trailing `_count` is redundant and creates a naming inconsistency across the kernel set.

**Resolution:** Remove the trailing `_count` from these three parameters in Node 13 to align with all other kernels.

---

### **Category D: CONCEPT-Internal Issues**

#### D1. Section 3.1 List Numbering Artifact

**Affected Documents:** `CONCEPT.md` §3.1

The "Methods of Gradient Clipping" subsection numbers its items 3, 4, 5 instead of 1, 2, 3:

> 3. **`Full-Group-Wise Clipping`**
> 4. **`Partial-Group-Wise Clipping`**
> 5. **`Component-Wise Clipping`**

This is a markdown artifact from a prior revision where these list items continued from an earlier numbered list.

**Resolution:** Renumber to 1, 2, 3.

---

#### D2. Unexplained Node 12 Gap

**Affected Documents:** `CONCEPT.md` (DAG, Kernel Contracts section)

Node numbers proceed: ..., 10, 11, 13, 14, ... The Node 12 identifier is never used or acknowledged. The CONCEPT's Model A streaming description references _"Node 12 permutation"_ in passing, suggesting it was a historical node that was superseded by Node 13's current design.

**Resolution:** Either add a brief note (e.g., _"Node 12: Reserved / superseded by Node 13"_) for traceability, or update the Model A reference to cite Node 13.

---

### **Category E: Informational**

#### E1. Stale Policy Name in `clip_intermediate_grad`

**Affected Documents:** `kernels.cl.h` (Node 15b/20b)

The `src_scalar_REAL_clipping_threshold_t_j` parameter's Calculability Proof reads:

> "Host-side calculation based on the active stabilization policy (e.g., **Normalized Log-Space Quadratic Scaling**)"

The CONCEPT and Node 16's Behavioral Invariants consistently define the policy as **"Quadratic Scaling Policy"** with the formula $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$. The "Normalized Log-Space" qualifier does not appear anywhere else in the document set and is likely a stale reference from a prior revision.

**Resolution:** Update the Calculability Proof to reference "Quadratic Scaling Policy."

---

### **Positive Observations**

1. **Node 16's Architectural Justification is well-aligned.** The kernel's extensive `Behavioral Invariants` and `Architectural Justification` blocks correctly mirror the CONCEPT's two-phase stabilization strategy (Phase I = Node 11 safety gate, Phase II = Node 16 staged reduction). The pre-computation/per-stage execution model cleanly resolves the jurisdictional separation concern raised by the "No Silent Monoliths" principle.

2. **The Indirection Contract is consistently implemented.** Both `aggregate_register_reduce` and `aggregate_local_reduce` faithfully implement the `partial_collection` + `offset_list` pattern described in the CONCEPT's Recursive Tiered Aggregation Engine, with no semantic drift.

3. **The `clip_partial_gradients` (Node 11) and `clip_shared_gradients_chunk` (Node 19) bifurcation** correctly implements the CONCEPT's dual clipping architecture for tiled module gradients and streamed shared-layer gradients, with appropriate interfaces for each streaming model.

4. **Article 1.4 (Axiom of Collaborative Interface Verifiability)** is well-enforced across the header. Calculability Proofs are self-contained, and Validation Preconditions reference only interface-visible parameters.
