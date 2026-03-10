# Remediation Plan: Cross-Document Inaccuracies (Category B)

**Source:** `AUDIT_REPORT.md` (10 March 2026)
**Scope:** Findings B1, B2, B3, B4, B5 — all classified **Cross-Doc Inaccuracy**
**Date:** 10 March 2026

---

## Preamble

This plan addresses the five Category B findings from the cross-document audit. Each finding represents a factual inconsistency between the kernel header (`kernels.cl.h`) and the architectural concept document (`CONCEPT.md`). Unlike Category A (contract violations resolved by amending `CONTRACT.md`), Category B requires edits across both the CONCEPT document and the kernel header — the source of truth for any given finding depends on which document correctly reflects the intended architecture.

Per CONCEPT.md §1 (Architectural Elegance Feedback), when an inconsistency is discovered, the resolution must formalize the *correct* pattern — not paper over the divergence. Each finding below identifies which document holds the ground truth and which must be amended.

### Ordering & Dependencies

| Finding | Depends On | Document(s) Modified                  | Complexity |
| :------ | :--------- | :------------------------------------ | :--------- |
| B1      | —          | `kernels.cl.h`                        | Trivial    |
| B2      | —          | `kernels.cl.h`                        | Trivial    |
| B3      | —          | `CONCEPT.md`                          | Trivial    |
| B4      | —          | `CONCEPT.md`                          | Moderate   |
| B5      | B4 (advisory) | `CONCEPT.md`                       | Moderate   |

B1 and B2 are isolated header commentary corrections — they can be applied in any order and have zero risk of side-effects. B3 is a text expansion in the CONCEPT. B4 and B5 both modify the Mermaid DAG in CONCEPT.md; B5 builds on the data nodes introduced by B4, so B4 should be applied first. However, they are not strictly dependent — B5's corrections to the K17/K18 edges are valid independent of B4.

**Risk profile:** All changes are documentary. No kernel signatures, no implementation code, and no host-side integration logic are affected. The changes are confined to `@param` commentary text in the header and Mermaid diagram syntax plus prose in the CONCEPT.

---

## B1: Node 18 — `final_` vs `summed_` Buffer Naming

### Finding Summary

Node 17 (`backprop_shared_weights_chunk`) names its upstream gradient input `src_buffer_GLOBAL_summed_grad_hidden_activations`. Node 18 (`backprop_shared_biases_chunk`) names the identical logical buffer `src_buffer_GLOBAL_final_grad_hidden_activations`. Both kernels consume the same Node 16 output — the batch-wide summed (but not yet normalized) hidden layer gradient.

Per CONTRACT §4.0:
- `summed_` = "a buffer whose elements are the result of a batch-wide reduction (summation)"
- `final_` = "a fully processed, normalized result ready for consumption by a final state-modifying kernel"

The Node 16 output has been reduced but **not** normalized by Node 21. It is an intermediate consumed by Nodes 17 and 18, not a terminal gradient. Node 18's use of `final_` is semantically incorrect under the CONTRACT's own definitions.

### Recommendation

**Rename Node 18's parameter from `final_` to `summed_` to align with Node 17 and the CONTRACT lexicon.**

The ground truth is Node 17's naming plus the CONTRACT definition. Node 18's naming is the error. This is a straightforward case: the buffer is a summed reduction result consumed as an intermediate, which is exactly the `summed_` semantic.

### Scope of Change

| Document        | Location                         | Action                                                                                                       |
| :-------------- | :------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| `kernels.cl.h`  | Node 18 parameter declaration    | Rename `src_buffer_GLOBAL_final_grad_hidden_activations` → `src_buffer_GLOBAL_summed_grad_hidden_activations` |
| `kernels.cl.h`  | Node 18 `@param` commentary      | Update the parameter name in the doc comment to match                                                        |

### Downstream Impact

This rename affects the kernel signature. Any host-side code that sets this kernel argument by name (`kernel_signatures/phase_2_learn_D_backprop.py` and its callers) must be updated to use the new parameter name. Grep for `final_grad_hidden_activations` across the `src/` and `tests/` trees to identify all references.

**Note:** The associated scalar `src_scalar_NATURAL_final_grad_hidden_total_element_count` in Node 18 should also be evaluated for consistency — it describes the element count of the `summed_` buffer. However, since the same scalar name is used in Node 17 under the `summed_` buffer, and the scalar is a dimensionality parameter (not a buffer reference), this may be acceptable as-is. The recommendation is to survey both Nodes 17 and 18's full parameter lists and ensure the dimensional scalar names are aligned.

### Verification

1. Confirm Node 17 and Node 18 use identical naming for their upstream gradient parameter.
2. Confirm no `final_grad_hidden` reference remains in the Node 18 block of `kernels.cl.h`.
3. Run the kernel signature test suite to confirm host-side bindings still resolve.

---

## B2: Node 24 `adam_update` — Wrong Source Node Reference

### Finding Summary

The `@param` commentary for `src_buffer_GLOBAL_final_grad` in Node 24 (`adam_update`) states:

> "The buffer containing the final, normalized, batch-averaged gradients from **Node 20**."

The actual data flow per the CONCEPT DAG is:

```
Nodes 15/20 (Summed Grad) → Node 21 (normalize_gradients) → Node 22 (Barrier) → Node 24 (adam_update)
```

The immediate producer of `final_grad` buffers is **Node 21** (`normalize_gradients`), not Node 20. Node 20 produces `Summed_Grad` (the pre-normalization aggregate), which is then divided by the batch count N in Node 21 to produce the `Final_Grad` consumed by Node 24.

### Recommendation

**Change the reference from "Node 20" to "Node 21."**

The ground truth is the CONCEPT DAG and the `normalize_gradients` kernel contract. The header commentary simply has an off-by-one node reference. This is an editorial correction with no architectural implications.

### Scope of Change

| Document        | Location                            | Action                                                                     |
| :-------------- | :---------------------------------- | :------------------------------------------------------------------------- |
| `kernels.cl.h`  | Node 24 `@param src_buffer_GLOBAL_final_grad` | Change "from Node 20" → "from Node 21" in the description text |

### Verification

1. Confirm the `@param` block for `src_buffer_GLOBAL_final_grad` references Node 21.
2. Cross-reference with the CONCEPT DAG: the edge from `K21` to `B22` to `K24` is the canonical path.

---

## B3: Incomplete Placement Contract Kernel List

### Finding Summary

The CONCEPT states:

> "This contract applies to kernels: **(7), (8), (9), (10), (17), (18)**."

Inspection of the kernel header reveals two additional kernels whose output parameters carry `Placement Contract` annotations:

- **Node 6** (`compute_probs_loss_cce_chunk`): `dest_buffer_GLOBAL_partial_probs` carries `Placement Contract: grid_mod_cls(...)`.
- **Node 11** (`clip_partial_gradients`): All four `dest_buffer_GLOBAL_clipped_partial_*` outputs carry `Placement Contract: grid_mod_cls(...)`.

The kernel header is the ground truth here — it is the authoritative record of which kernels implement the Placement Contract. The CONCEPT's list is simply incomplete.

### Recommendation

**Expand the CONCEPT's Placement Contract kernel list to include Nodes 6 and 11.**

The updated sentence should read:

> "This contract applies to kernels: **(6), (7), (8), (9), (10), (11), (17), (18)**."

### Rationale for Inclusion

**Node 6 (`compute_probs_loss_cce_chunk`):** This kernel is a Partial Renderer for probabilities — it writes to a tile-indexed slot within the `partial_probs` collection buffer. Its placement semantics are identical to Node 7 (which is already listed). The omission of Node 6 appears to be an oversight, likely because the original list was drafted when only the BCE path was considered a "partial renderer" for both probs and loss, while the CCE path's scatter-write for loss (which is *not* placement-contract-governed) may have caused the entire kernel to be passed over.

**Node 11 (`clip_partial_gradients`):** This kernel reads partials at a tile index and writes clipped partials at the same tile index. It preserves the tile placement topology established by Nodes 8, 9, and 10. Its outputs are consumed by the reduction engine (Nodes 13, 15) which depend on correct placement for indirection-driven aggregation. Including it in the list is necessary for completeness.

### Scope of Change

| Document      | Location                             | Action                                                        |
| :------------ | :----------------------------------- | :------------------------------------------------------------ |
| `CONCEPT.md`  | §"The Placement Contract" paragraph  | Replace `(7), (8), (9), (10), (17), (18)` with `(6), (7), (8), (9), (10), (11), (17), (18)` |

### Verification

1. Confirm the expanded list in the CONCEPT matches every kernel in `kernels.cl.h` that has at least one `Placement Contract:` annotation on an output parameter.
2. Confirm no kernel *not* in the list carries a `Placement Contract` annotation. (Nodes 17 and 18 use `linear_batch(...)` rather than `grid_mod_cls(...)` — they are already listed and their different placement function is expected.)

---

## B4: Missing `hidden_mask` Edge in CONCEPT DAG

### Finding Summary

Node 5 (`render_logits_chunk`) accepts `src_buffer_GLOBAL_hidden_mask` as an input parameter in the kernel header. Node 4 (`forward_pass`) produces `dest_buffer_GLOBAL_hidden_mask` as an output. The CONCEPT DAG shows only:

```mermaid
hidden_i --> K5["(5) render_logits_chunk"]
```

There is no `hidden_mask` data node, and no edge from Node 4's mask output to Node 5's mask input. The mask is required for Node 5 to correctly zero out contributions from padded hidden dimensions during the linear transform.

### Recommendation

**Add `hidden_mask` as an explicit data node output of Node 4, and draw an edge from it into Node 5.**

The kernel header is the ground truth — Node 5's signature unambiguously requires the mask. The CONCEPT DAG must be updated to reflect this data dependency.

### Proposed DAG Changes

In the Act Process subgraph, immediately after the `hidden_i` data node:

1. Add a new data node: `hidden_mask["Hidden Mask 'i'"]:::data`
2. Update Node 4's output edge: `K4 --> hidden_i & hidden_mask` (or as two separate edges, depending on Mermaid style preferences)
3. Add an input edge to Node 5: `hidden_mask --> K5`

### Additional Consideration: Nodes 8, 9, 10

The audit report raises the question of whether `hidden_mask` also flows into Nodes 8, 9, and 10. A review of the kernel header shows:

- **Node 8** (`calculate_module_param_grads_chunk`): Does **not** accept `hidden_mask`.
- **Node 9** (`backprop_error_to_hidden_chunk`): Does **not** accept `hidden_mask`.
- **Node 10** (`calculate_chunk_temp_gradients`): Does **not** accept `hidden_mask`.

Therefore, the `hidden_mask` edge should flow **only** to Node 5. The audit report's parenthetical suggestion of "Nodes 8, 9, 10 if applicable" can be dismissed — it is not applicable.

### Scope of Change

| Document      | Location                              | Action                                                                                     |
| :------------ | :------------------------------------ | :----------------------------------------------------------------------------------------- |
| `CONCEPT.md`  | Mermaid DAG, Act Process subgraph     | Add `hidden_mask` data node as output of K4                                                |
| `CONCEPT.md`  | Mermaid DAG, Act Process subgraph     | Add edge `hidden_mask --> K5`                                                              |

### Proposed Mermaid Syntax

Replace:

```mermaid
K4["(4) forward_pass"]:::kernel --> hidden_i["Hidden Activations 'i'"]:::data
P_Shared & P_Module & SampleMask --> K4
hidden_i --> K5["(5) render_logits_chunk"]:::kernel
```

With:

```mermaid
K4["(4) forward_pass"]:::kernel --> hidden_i["Hidden Activations 'i'"]:::data
K4 --> hidden_mask["Hidden Mask 'i'"]:::data
P_Shared & P_Module & SampleMask --> K4
hidden_i & hidden_mask --> K5["(5) render_logits_chunk"]:::kernel
```

### Verification

1. Confirm `hidden_mask` appears as a data node in the DAG with an edge from K4 and an edge into K5.
2. Confirm no edge from `hidden_mask` into K8, K9, or K10 (they do not consume it).
3. Visually render the Mermaid diagram and confirm the added node does not break layout or legibility.

---

## B5: Incorrect Data Flow Edges for Nodes 17 and 18

### Finding Summary

The CONCEPT DAG currently shows:

```mermaid
Input_i["Recomputed Input 'i'"]:::data --- K17["(17) backprop_shared_weights"]:::kernel
Input_i --- K18["(18) backprop_shared_biases"]:::kernel
P_Shared & hidden_i --> K17
hidden_i --> K18
Summed_Grad_H -. "slice" .-> K17 & K18
```

Two edges are incorrect when compared to the kernel header signatures:

**Error 1: `P_Shared --> K17` is spurious.**
Node 17's kernel signature (`backprop_shared_weights_chunk`) does not consume the shared weight parameters as an input. The weight gradient computation ($\nabla W = X^T \cdot \nabla H$) requires the _input data_ and the _upstream gradient_, not the current weight values. The `P_Shared` edge must be removed from K17.

**Error 2: `Input_i --- K18` is spurious.**
Node 18's kernel signature (`backprop_shared_biases_chunk`) does not accept an input data buffer. The bias gradient computation ($\nabla b = \sum \nabla H$) requires only the `summed_grad_hidden_activations`, `hidden_activations`, and `sample_mask`. The `Input_i` edge must be removed from K18.

### Recommendation

**Correct the DAG edges to match the kernel signatures.**

The kernel header is the ground truth for data flow — it defines the actual parameters each kernel accepts. The CONCEPT DAG must reflect the true interface, not a schematic approximation.

### Proposed DAG Changes

Replace the current Phase III Streaming Loop edges:

```mermaid
Input_i["Recomputed Input 'i'"]:::data --- K17["(17) backprop_shared_weights"]:::kernel
Input_i --- K18["(18) backprop_shared_biases"]:::kernel
P_Shared & hidden_i --> K17
hidden_i --> K18
Summed_Grad_H -. "slice" .-> K17 & K18
```

With:

```mermaid
Input_i["Recomputed Input 'i'"]:::data --> K17["(17) backprop_shared_weights"]:::kernel
hidden_i --> K17
hidden_i --> K18["(18) backprop_shared_biases"]:::kernel
Summed_Grad_H -. "slice" .-> K17 & K18
```

### Summary of Edge Changes

| Edge               | Before  | After   | Reason                                                      |
| :----------------- | :------ | :------ | :---------------------------------------------------------- |
| `P_Shared → K17`   | Present | Removed | Node 17 does not consume weights; it computes weight *gradients* from input data × upstream grad |
| `Input_i → K17`    | Present | Present | Correct — Node 17 uses input data for $\nabla W = X^T \cdot \nabla H$ |
| `Input_i → K18`    | Present | Removed | Node 18 does not consume input data; bias grads are $\nabla b = \sum \nabla H$ |
| `hidden_i → K17`   | Present | Present | Correct — Node 17 requires hidden activations                |
| `hidden_i → K18`   | Present | Present | Correct — Node 18 requires hidden activations                |
| `Summed_Grad_H → K17/K18` | Present | Present | Correct — both kernels consume the upstream gradient slice   |

### Additional Note on CONCEPT Prose

The CONCEPT §"Learn Phase Kernels" contract descriptions for Nodes 17 and 18 should also be reviewed for consistency:

- **Node 17** currently reads: _"Consumes a slice of the `Summed_Grad_H` and a **chunk** of recomputed `hidden_i`."_ This should be amended to mention `Input_i` as well: _"Consumes `Input_i`, a slice of `Summed_Grad_H`, and a **chunk** of recomputed `hidden_i`."_
- **Node 18** currently reads: _"Consumes a slice of the `Summed_Grad_H` and a **chunk** of recomputed `hidden_i`."_ This is correct — no input data is needed.

### Scope of Change

| Document      | Location                                        | Action                                                |
| :------------ | :---------------------------------------------- | :---------------------------------------------------- |
| `CONCEPT.md`  | Mermaid DAG, Phase III Streaming Loop            | Remove `P_Shared --> K17` edge                        |
| `CONCEPT.md`  | Mermaid DAG, Phase III Streaming Loop            | Remove `Input_i --- K18` edge                         |
| `CONCEPT.md`  | Mermaid DAG, Phase III Streaming Loop            | Change `Input_i --- K17` from undirected to directed (`-->`) |
| `CONCEPT.md`  | §"Learn Phase Kernels", Node 17 contract text    | Add `Input_i` to the list of consumed inputs          |

### Verification

1. Confirm K17 has exactly three input edges: `Input_i`, `hidden_i`, and `Summed_Grad_H` (slice).
2. Confirm K18 has exactly two input edges: `hidden_i` and `Summed_Grad_H` (slice).
3. Confirm `P_Shared` has no edge into K17 (it flows only into K24_Shared for the optimizer update).
4. Confirm `Input_i` has no edge into K18.
5. Visually render the Mermaid diagram and confirm the Streaming Loop subgraph is legible.

---

## Implementation Checklist

| #  | Finding | Action                                                                          | Status      |
| :- | :------ | :------------------------------------------------------------------------------ | :---------- |
| 1  | B1      | Rename Node 18 `final_grad_hidden_activations` → `summed_grad_hidden_activations` in `kernels.cl.h` | Completed |
| 2  | B1      | Update host-side kernel signature bindings to match                             | N/A (positional) |
| 3  | B2      | Change "Node 20" → "Node 21" in Node 24 `@param` commentary                   | Completed   |
| 4  | B3      | Expand Placement Contract kernel list in `CONCEPT.md` to include (6) and (11)  | Completed   |
| 5  | B4      | Add `hidden_mask` data node and edges to CONCEPT DAG                           | Completed   |
| 6  | B5      | Remove `P_Shared → K17` edge from CONCEPT DAG                                 | Completed   |
| 7  | B5      | Remove `Input_i → K18` edge from CONCEPT DAG                                  | Completed   |
| 8  | B5      | Change `Input_i --- K17` to directed `Input_i --> K17`                         | Completed   |
| 9  | B5      | Update Node 17 contract prose to mention `Input_i`                             | Completed   |

### Post-Remediation Audit Gate

Once all items are marked complete, a follow-up audit pass shall confirm:

- The AUDIT_REPORT.md findings B1–B5 can each be marked **Resolved**.
- No new cross-document inconsistencies were introduced by the amendments.
- The CONCEPT.md Mermaid DAG renders correctly and all edges match the kernel header signatures.
- The CONCEPT.md revision number is incremented (Rev. 7 → Rev. 8).
