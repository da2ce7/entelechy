# Remediation Plan: Naming, Editorial & Informational Findings (Categories C, D, E)

**Source:** `AUDIT_REPORT.md` (10 March 2026)
**Scope:** Findings C1, D1, D2, E1 — classified **Naming Inconsistency**, **Editorial**, and **Informational**
**Date:** 10 March 2026

---

## Preamble

This plan addresses the four remaining open findings from the cross-document audit. Unlike Categories A and B — which required formal CONTRACT amendments and cross-document DAG corrections — these findings are lower-severity consistency and editorial issues. They carry no architectural risk, but resolving them strengthens the naming coherence and documentary clarity that the system's own axioms demand (Article 1.2: Axiom of Semantic Uniqueness; Article 1.4: Axiom of Collaborative Interface Verifiability).

All changes are documentary. C1 is the only finding that touches a kernel signature (and therefore has a downstream host-side impact). D1, D2, and E1 are confined to prose edits.

### Ordering & Dependencies

| Finding | Depends On | Document(s) Modified                                              | Complexity |
| :------ | :--------- | :---------------------------------------------------------------- | :--------- |
| D1      | —          | `CONCEPT.md`                                                      | Trivial    |
| D2      | —          | `CONCEPT.md`                                                      | Trivial    |
| E1      | —          | `kernels.cl.h`                                                    | Trivial    |
| C1      | —          | `kernels.cl.h`, `phase_2_learn_B_processing.py`, test files       | Moderate   |

D1, D2, and E1 are independent single-line edits with zero side-effects. C1 is a coordinated rename across the kernel header and its host-side binding layer — it should be applied last so that the simpler items establish the editorial precedent.

**Risk profile:** D1, D2, and E1 are zero-risk prose corrections. C1 requires a synchronized rename across the kernel header, the Python kernel signature dataclass, and any tests that reference the affected parameter names; a grep-verified sweep is mandatory.

---

## D1: Section 3.1 List Numbering Artifact

### Finding Summary

The "Methods of Gradient Clipping" subsection in CONCEPT.md §3.1 numbers its items 3, 4, 5 instead of 1, 2, 3. This is a markdown artifact from a prior revision where these list items continued from a preceding numbered list.

### Recommendation

**Renumber the list items to 1, 2, 3.**

The list is self-contained within §3.1 and has no inbound cross-references that depend on the current numbering. The items describe the three methods of gradient clipping (`Full-Group-Wise`, `Partial-Group-Wise`, `Component-Wise`) and should begin at 1 for clarity.

### Scope of Change

| Document      | Location                   | Action                                         |
| :------------ | :------------------------- | :--------------------------------------------- |
| `CONCEPT.md`  | §3.1 (approx. lines 89–101) | Change `3.` → `1.`, `4.` → `2.`, `5.` → `3.` |

### Verification

After the edit, confirm that §3.1 contains a consecutively-numbered list starting at 1 with three items.

---

## D2: Unexplained Node 12 Gap

### Finding Summary

Node numbers in the CONCEPT DAG proceed: ..., 10, 11, 13, 14, ... Node 12 is never defined or acknowledged. The Model A streaming description (CONCEPT.md line 179) references _"Node 12 permutation"_ in passing, indicating that Node 12 was a historical identifier superseded by the current Node 13 (`gather_and_permute_grad_hidden_activations`).

### Recommendation

**Add a brief traceability note in the CONCEPT, and update the stale Model A reference.**

Two edits:

1. **Insert a reservation note** in the DAG section or kernel contracts section — e.g., at the point where Node 13 is introduced — stating that Node 12 was retired and its functionality subsumed by Node 13. This preserves historical traceability without re-numbering the entire DAG (which would be a high-risk, high-churn change with no architectural benefit).

2. **Update the Model A reference** (line 179) from "Node 12 permutation" to "Node 13 permutation" so the prose matches the actual kernel set.

### Scope of Change

| Document      | Location                                  | Action                                                                                                 |
| :------------ | :---------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| `CONCEPT.md`  | Model A description (line 179)            | Change "Node 12 permutation" → "Node 13 permutation"                                                  |
| `CONCEPT.md`  | DAG or Kernel Contracts section near Node 13 | Insert: _"Note: Node 12 is reserved (retired). Its permutation functionality was subsumed by Node 13."_ |

### Verification

1. Confirm no remaining reference to "Node 12" exists in the CONCEPT without an accompanying explanation.
2. Confirm the Model A streaming description references Node 13.

---

## E1: Stale Policy Name in `clip_intermediate_grad`

### Finding Summary

The `src_scalar_REAL_clipping_threshold_t_j` parameter's Calculability Proof in Node 15b/20b (`clip_intermediate_grad`) reads:

> _"Host-side calculation based on the active stabilization policy (e.g., **Normalized Log-Space Quadratic Scaling**)"_

The CONCEPT and all other kernel documentation consistently use the name **"Quadratic Scaling Policy"** with the formula $T_j = T_{\text{algorithmic}} + \lambda \cdot j^2$. The "Normalized Log-Space" qualifier does not appear anywhere else in the document set and is a stale reference from a prior revision.

### Recommendation

**Replace "Normalized Log-Space Quadratic Scaling" with "Quadratic Scaling Policy" to match the CONCEPT's canonical name.**

The CONCEPT is the authoritative source for policy names — it uses "Quadratic Scaling Policy" in four separate locations (§3.3, §3.4, §3.5, and the Node 16 kernel contract). The header should mirror this exactly.

### Scope of Change

| Document        | Location                                                     | Action                                                                          |
| :-------------- | :----------------------------------------------------------- | :------------------------------------------------------------------------------ |
| `kernels.cl.h`  | Node 15b/20b, `@param src_scalar_REAL_clipping_threshold_t_j` (line 1018) | Change `Normalized Log-Space Quadratic Scaling` → `Quadratic Scaling Policy` |

### Verification

1. Grep for "Normalized Log-Space" across the full document set — confirm zero remaining occurrences.
2. Confirm the Calculability Proof now reads: _"Host-side calculation based on the active stabilization policy (e.g., Quadratic Scaling Policy)"_.

---

## C1: Node 13 — Spurious `_count` Suffixes on Dimensional Parameters

### Finding Summary

Node 13 (`gather_and_permute_grad_hidden_activations`) appends a `_count` suffix to three dimensional parameters that are named without it in every other kernel:

| Node 13 Name                                       | Equivalent in All Other Kernels            |
| :------------------------------------------------- | :----------------------------------------- |
| `src_scalar_NATURAL_modules_per_chunk_count`        | `src_scalar_NATURAL_modules_per_chunk`      |
| `src_scalar_NATURAL_num_module_chunks_count`         | `src_scalar_NATURAL_num_module_chunks`       |
| `src_scalar_NATURAL_num_class_chunks_count`          | `src_scalar_NATURAL_num_class_chunks`        |

Per CONTRACT Article 8 §4.0, the `_count` suffix denotes _"the number of elements to process, relative to a corresponding `_offset`."_ These parameters are partition counts (number of chunks), not element counts relative to an offset. Their semantic meaning is identical to their counterparts in Nodes 7, 8, 9, 10, and 11.

### Recommendation

**Remove the trailing `_count` from all three parameters in Node 13 to align with the rest of the kernel set.**

Rationale:
- The `_count` suffix has a defined semantic in the CONTRACT lexicon that does not apply here.
- All other kernels using these exact partition dimensions omit `_count`.
- The inconsistency creates a false signal that Node 13's parameters have different semantics from their counterparts elsewhere.

### Scope of Change

This is a **signature-breaking rename** — the kernel's parameter names are part of its public interface, and the host-side Python binding layer mirrors them exactly. A coordinated rename is required across all layers.

| Document / File                                    | Location                            | Action                                                                   |
| :------------------------------------------------- | :---------------------------------- | :----------------------------------------------------------------------- |
| `kernels.cl.h`                                     | Node 13 parameter declarations (lines 887–889) | Rename the three parameters (drop `_count` suffix)            |
| `kernels.cl.h`                                     | Node 13 `@param` commentary (Tensor Shape, Calculability Proof, Validation Preconditions) | Update all references in doc comments |
| `src/kernel_signatures/phase_2_learn_B_processing.py` | Dataclass fields (lines 235–237) and all internal references | Rename the three fields                  |
| `src/kernel_signatures/tests/test_phase_2_learn_B_processing.py` | All test call sites constructing Node 13 args | Rename keyword arguments                     |
| `src/kernel_signatures/tests/test_header_contract_conformance.py` | Node 13 conformance test call site | Rename keyword arguments                     |

### Pre-Implementation Sweep

Before applying the rename, run:

```bash
grep -rn "modules_per_chunk_count\|num_module_chunks_count\|num_class_chunks_count" \
  architectures/averaging_ensembled_classifier/
```

This will produce the exhaustive list of all references. Every match must be updated. Any match in a file not listed above indicates an additional dependency that must be addressed.

### Downstream Impact

- **Kernel signature tests** (`test_phase_2_learn_B_processing.py`, `test_header_contract_conformance.py`): keyword argument names must be updated.
- **Host-side orchestrator** (`execution_plan.py`, `batch_processor.py`, `graph_recipes.py`): if these files construct Node 13 argument dictionaries by name, they must be updated. The grep sweep will identify any such references.
- **No runtime behavioral change.** The renamed parameters are scalar `uint` values passed by position in the OpenCL ABI — the rename only affects the human-readable name and the Python binding layer's named-argument interface.

### Verification

1. Run the pre-implementation grep sweep — confirm zero remaining `_count`-suffixed references for these three parameters.
2. Run `pytest` on the kernel signature test suite to confirm all bindings resolve.
3. Confirm Node 13's parameter names now exactly match the equivalent parameters in Nodes 7, 8, 9, 10, and 11.

---

## Implementation Checklist

| #  | Finding | Action                                                                            | Status    |
| :- | :------ | :-------------------------------------------------------------------------------- | :-------- |
| 1  | D1      | Renumber §3.1 list items (3,4,5 → 1,2,3) in `CONCEPT.md`                        | Completed |
| 2  | D2      | Update "Node 12 permutation" → "Node 13 permutation" in `CONCEPT.md` Model A     | Completed |
| 3  | D2      | Insert Node 12 reservation note near Node 13 in `CONCEPT.md`                     | Completed |
| 4  | E1      | Update stale policy name in `kernels.cl.h` Node 15b/20b                          | Completed |
| 5  | C1      | Rename Node 13 parameters in `kernels.cl.h` (header + doc comments)              | Completed |
| 6  | C1      | Rename Node 13 fields in `phase_2_learn_B_processing.py`                         | Completed |
| 7  | C1      | Rename Node 13 arguments in all test files                                        | Completed |
| 8  | C1      | Run grep sweep to confirm zero remaining `_count`-suffixed references             | Completed |

### Post-Remediation Audit Gate

Once all items are marked complete:

- Findings C1, D1, D2, and E1 in `AUDIT_REPORT.md` can each be marked **Resolved**.
- The full test suite shall pass with no regressions.
- No new naming inconsistencies shall have been introduced by the edits.
