# Audit Remediation Plan: Contract Violations (Category A)

**Source:** `AUDIT_REPORT.md` (10 March 2026)
**Scope:** Findings A1, A2, A3, A4 — all classified **Contract Violation**
**Date:** 10 March 2026

---

## Preamble

This plan addresses the four Category A findings from the cross-document audit. Each finding represents a material inconsistency between the kernel header (`kernels.cl.h`) and the governing system contract (`CONTRACT.md`). Per CONCEPT.md §1 (Architectural Elegance Feedback), the resolution path is to evolve the formal abstractions to subsume the discovered patterns — not to patch the implementation around them.

The remediation targets are exclusively documentary. No kernel implementation code changes are required for A1, A3, or A4. A2 requires either a CONTRACT amendment or a kernel rename — the recommendation below addresses both options.

### Ordering & Dependencies

| Finding | Depends On | Document(s) Modified       |
| :------ | :--------- | :------------------------- |
| A4      | —          | `CONTRACT.md`              |
| A1      | —          | `CONTRACT.md`              |
| A3      | —          | `CONTRACT.md`              |
| A2      | A3 (advisory) | `CONTRACT.md`, possibly `kernels.cl.h` + host signatures |

A4 is listed first because it is the simplest and establishes the editorial precedent (deduplicate before amending). A1 and A3 are independent CONTRACT amendments. A2 is the only finding with a design decision component and should be resolved last once the lexicon (§7.0) and memory scope (Article 2.1.1) definitions have been stabilised.

---

## A4: Duplicate `final_` Prefix Definition

### Finding Summary

CONTRACT Article 8 §4.0 defines the `final_` prefix twice in the Context Modifiers table:

1. _"Denotes a fully reduced, final result."_
2. _"Denotes a fully processed, normalized result ready for consumption by a final state-modifying kernel (e.g., optimizer)."_

This violates Article 1.2 (Axiom of Semantic Uniqueness) and creates ambiguity — the first definition is broader than the second.

### Recommendation

**Remove the first (broader) entry. Retain only the second (precise) definition.**

Rationale: The second definition is the one that accurately describes every actual usage of `final_` in the kernel header. Buffers carrying the `final_` prefix (e.g., `final_grad`, `final_loss`, `final_probs`) are always outputs of a completed reduction-and-normalization pipeline, consumed by a terminal kernel (the optimizer or D2H copy). The broader definition ("a fully reduced, final result") fails to distinguish `final_` from `summed_`, which denotes a reduced-but-not-yet-normalized result.

### Scope of Change

| Document       | Section          | Action                                                 |
| :------------- | :--------------- | :----------------------------------------------------- |
| `CONTRACT.md`  | Article 8 §4.0  | Delete the row containing the broader `final_` definition. |

### Verification

After the edit, confirm that exactly one row in the §4.0 table defines `final_`, and that the retained definition reads:

> _"Denotes a fully processed, normalized result ready for consumption by a final state-modifying kernel (e.g., optimizer)."_

---

## A1: Undocumented Mandatory Build-Time Symbols

### Finding Summary

The kernel header enforces `SCALAR_IS_HALF` and `NUMERICAL_STABILITY_EPSILON` as mandatory build-time symbols via `#error` directives (lines 40–43 and 80–82). CONTRACT Article 6 ("Mandatory Build-Time Symbols") enumerates only `SCALAR_TYPE`, `SIMD_WIDTH`, and `C_TILE_SIZE`. The two enforced symbols have no contractual backing.

### Recommendation

**Add both symbols to CONTRACT Article 6.**

Both symbols are architecturally necessary:

- **`SCALAR_IS_HALF`** — The C preprocessor cannot perform type comparison (`#if SCALAR_TYPE == half` is undefined behaviour — type tokens evaluate to `0`). The header's own commentary (lines 36–39) explains this. The symbol is the only correct mechanism for conditional FP16 extension enablement, making it a mandatory companion to `SCALAR_TYPE`. Its domain is `{0, 1}` (integer flag).

- **`NUMERICAL_STABILITY_EPSILON`** — Used throughout the kernel set as the epsilon floor in `max(denominator, ε)` guards. Its value is precision-dependent (e.g., `1e-7` for FP32, `1e-4` for FP16) and must be injected by the host build system alongside the precision choice. It cannot be derived from `SCALAR_TYPE` at preprocessor time.

### Scope of Change

| Document       | Section    | Action                                                                                                   |
| :------------- | :--------- | :------------------------------------------------------------------------------------------------------- |
| `CONTRACT.md`  | Article 6  | Add `SCALAR_IS_HALF` with definition: _"Integer flag (`0` or `1`) indicating whether `SCALAR_TYPE` is `half`. Required because the C preprocessor cannot perform type-name comparison."_ |
| `CONTRACT.md`  | Article 6  | Add `NUMERICAL_STABILITY_EPSILON` with definition: _"The minimum epsilon value used for numerical stability guards (e.g., division-by-zero prevention). Its value is precision-dependent and must be consistent with `SCALAR_TYPE`."_ |

### Verification

After the edit:

1. Confirm Article 6 enumerates exactly five symbols: `SCALAR_TYPE`, `SIMD_WIDTH`, `C_TILE_SIZE`, `SCALAR_IS_HALF`, `NUMERICAL_STABILITY_EPSILON`.
2. Confirm every `#error "System Contract Violation: ..."` directive in `kernels.cl.h` references a symbol that appears in Article 6 (except `LOCAL_MEM_BANK_PADDING`, which is governed by Article 5).

---

## A3: `GLOBAL_` vs `GLOBAL_CONST_` Semantic Mismatch

### Finding Summary

CONTRACT Article 2.1.1 defines the memory scope tokens as a **syntactic** distinction:

- `GLOBAL_` = read/write `__global`
- `GLOBAL_CONST_` = read-only `__global const`

The kernel header systematically implements a **semantic** distinction:

- `GLOBAL_` = pipeline data (transient buffers, may carry `const` qualifier when used as source)
- `GLOBAL_CONST_` = persistent model state (learnable parameters, invariant for the duration of a dispatch)

This means many read-only transient buffers (`input`, `sample_mask`, `partial_probs`, `partial_collection`, etc.) carry `GLOBAL_` despite being `__global const` in their C declarations.

### Recommendation

**Amend CONTRACT Article 2.1.1 to formalise the semantic distinction actually implemented.**

Rationale: The implemented convention is architecturally superior. It encodes a meaningful, load-bearing distinction — "Is this buffer part of the persistent learnable model, or is it transient pipeline data?" — directly into the parameter name. This supports the Axiom of Jurisdictional Separation (Article 1.1): the scope token conveys *semantic category*, while the C qualifier conveys *access pattern*. Reverting the header to match the current CONTRACT definition would destroy this useful signal and require renaming dozens of parameters across the entire kernel set for no functional benefit.

### Proposed Definitions

Replace the current Article 2.1.1 definitions with:

> **2.1.1. Memory Scope Token Definitions**
>
> - **`GLOBAL_`**: Standard `__global` device memory for pipeline data. Buffers in this scope represent transient, per-dispatch data flowing through the computational DAG (inputs, activations, masks, partials, intermediates). They may carry the `const` qualifier in the C declaration when used as a read-only source.
> - **`LOCAL_`**: Work-group exclusive `__local` memory.
> - **`GLOBAL_CONST_`**: Read-only `__global` device memory holding persistent model state (learnable parameters: weights, biases, temperatures) that is invariant for the duration of a kernel dispatch.
> - **`DEVICE_CONST_`**: The hardware-specific, read-only `__constant` address space.

### Scope of Change

| Document       | Section         | Action                                                     |
| :------------- | :-------------- | :--------------------------------------------------------- |
| `CONTRACT.md`  | Article 2.1.1   | Replace the `GLOBAL_` and `GLOBAL_CONST_` definitions with the amended text above. `LOCAL_` and `DEVICE_CONST_` are unchanged. |

### Verification

After the edit:

1. Audit every `GLOBAL_CONST_` parameter in `kernels.cl.h` — confirm each refers to a learnable parameter buffer (`weights`, `biases`, `temperatures`, `parameters`, `m1`, `m2`).
2. Audit every `GLOBAL_` parameter with a `__global const` C qualifier — confirm each refers to transient pipeline data (`input`, `sample_mask`, `hidden_mask`, `partial_*`, `clipped_*`, `summed_*`, `offset_list`, etc.).
3. Confirm no parameter violates the new definitions.

---

## A2: Forbidden Lexical Terms in Kernel Names

### Finding Summary

CONTRACT Article 8 §7.0 forbids `cce` and `bce` in kernel names, prescribing use of generic terms with a `FLAG` parameter. The kernel header declares:

- `compute_probs_loss_cce_chunk` (Node 6)
- `compute_probs_loss_bce_chunk` (Node 7)

However, CONCEPT Principle 3(B) architecturally mandates separate kernels when _"divergence is complex or imposes conflicting memory patterns."_ The audit demonstrates that Nodes 6 and 7 diverge in type signature, memory layout, and downstream DAG topology — making unification via a FLAG parameter architecturally inappropriate.

### Analysis

This is a genuine conflict between two valid authorities:

1. **CONTRACT §7.0** — drafted assuming all CCE/BCE divergence could be handled by a single kernel with a flag. This assumption is false for the loss computation path.
2. **CONCEPT Principle 3(B)** — the higher-authority document explicitly permits (and in this case, requires) separate kernels for incompatible memory patterns.

Per CONCEPT §6 (Architectural Hierarchy), the Conceptual layer has sovereign authority. The Contractual layer translates conceptual mandates into interface requirements. When a CONTRACT rule contradicts a CONCEPT principle, the CONTRACT must be amended.

### Recommendation

**Amend CONTRACT §7.0 to add a scoped exception for architecturally-mandated kernel bifurcation, and retain the current kernel names.**

This is preferred over a rename to alternative suffixes (e.g., `_softmax_path` / `_sigmoid_path`) for three reasons:

1. **Clarity.** `cce` and `bce` are the universally recognised, unambiguous identifiers for these loss functions. Alternative names like `_softmax_path` are less precise — softmax is the activation function, not the loss; and sigmoid is shared by both binary cross-entropy and other use cases.

2. **Traceability.** The kernel names directly mirror the CONCEPT's node descriptions ("compute_probs_loss_cce" / "compute_probs_loss_bce"), maintaining bidirectional traceability between the three document tiers.

3. **Principled exception.** The §7.0 prohibition's *rationale* is to prevent problem-specific coupling in kernel names when a FLAG can provide generic dispatch. When that rationale does not apply — because the kernels are structurally incompatible and cannot share an interface — the prohibition should not apply either.

### Proposed §7.0 Amendment

Add the following note after the existing forbidden terms table:

> **Exception: Architecturally-Mandated Kernel Bifurcation.**
> When CONCEPT.md Principle 3(B) requires separate kernels due to incompatible type signatures, memory layouts, or downstream DAG topologies, those kernels may use otherwise-forbidden terms to distinguish the variant. This exception applies only when:
>
> 1. The kernel pair cannot share a unified interface (differing buffer types, shapes, or DAG edges).
> 2. The distinction is documented in the kernel's `@kernel_contract` block with an explicit reference to Principle 3(B).
> 3. No `FLAG` parameter could eliminate the interface divergence without producing a "smart kernel" with complex internal branching over incompatible memory patterns.
>
> **Current applicants:** `compute_probs_loss_cce_chunk` (Node 6), `compute_probs_loss_bce_chunk` (Node 7).

### Scope of Change

| Document       | Section        | Action                                                                      |
| :------------- | :------------- | :-------------------------------------------------------------------------- |
| `CONTRACT.md`  | Article 8 §7.0 | Add the scoped exception text after the forbidden terms table.              |
| `kernels.cl.h` | Nodes 6 & 7   | *(Optional but recommended)* Add a note to each kernel's `@kernel_contract` block referencing Principle 3(B) and citing this exception. |

### Verification

After the edit:

1. Confirm §7.0 retains the general prohibition on `cce`/`bce` (the default rule is unchanged).
2. Confirm the exception clause lists its three qualifying conditions.
3. Confirm Nodes 6 and 7 in `kernels.cl.h` satisfy all three conditions (they do — see the divergence table in the audit report).
4. If the optional `@kernel_contract` annotations are added, confirm they reference "CONCEPT.md Principle 3(B)" and "CONTRACT §7.0 Exception."

---

## Implementation Checklist

| #  | Finding | Action                                                          | Status    |
| :- | :------ | :-------------------------------------------------------------- | :-------- |
| 1  | A4      | Remove duplicate `final_` row from CONTRACT §4.0                | Completed |
| 2  | A1      | Add `SCALAR_IS_HALF` to CONTRACT Article 6                      | Completed |
| 3  | A1      | Add `NUMERICAL_STABILITY_EPSILON` to CONTRACT Article 6         | Completed |
| 4  | A3      | Amend `GLOBAL_` / `GLOBAL_CONST_` definitions in Article 2.1.1 | Completed |
| 5  | A2      | Add scoped exception to CONTRACT §7.0                           | Completed |
| 6  | A2      | Annotate Nodes 6 & 7 `@kernel_contract` blocks                 | Completed |

### Post-Remediation Audit Gate

Once all items are marked complete, a follow-up audit pass shall confirm:

- The AUDIT_REPORT.md findings A1–A4 can each be marked **Resolved**.
- No new contract violations were introduced by the amendments.
- The CONTRACT revision number is incremented (Rev. 6 → Rev. 7).
