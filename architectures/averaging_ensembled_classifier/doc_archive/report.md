# Principled Recommendation: Formalizing the Padding-Zero Jurisdiction

## Diagnosis

The root cause is that a single invariant name — **"Padding Zero-Fill"** — conflates two jurisdictionally distinct obligations. It is used identically on Node 8 (where `ZERO_REQUIRED_ADDITIVE` makes the host the primary guarantor) and Node 9 (where `Initialization Contract: NONE` makes the kernel the *sole* guarantor). This violates the architecture's own principle of jurisdictional separation.

The ambiguity is further compounded by CONCEPT §3.6, which describes a system-level *emergent property* (the zero-propagation chain) in the same language as a per-kernel *obligation*, making it unclear whether any given kernel can rely on upstream zeros or must establish them independently.

---

## Governing Principles

Three architectural principles converge on a single resolution:

1. **Primacy of Memory Strategy** — A kernel actively writing zeros to positions the host has already zeroed is wasted bandwidth. The architecture should never mandate redundant writes.

2. **Modular, "Dumb" Kernels** — A kernel's padding obligation should be derivable from its own contract block, not from reasoning about what the host did or didn't do to the buffer before dispatch. The Initialization Contract already encodes this information per-buffer; the kernel's Behavioral Invariant should reference it, not duplicate it.

3. **Axiom of Jurisdictional Separation** — The host and the kernel have distinct, non-overlapping responsibilities. When both claim to guarantee the same property via independent mechanisms, the contract has failed to assign jurisdiction.

---

## Recommendation

### 1. Replace "Padding Zero-Fill" with two distinguished invariants

The current single invariant is retired. In its place, the `Behavioral Invariants` vocabulary (CONTRACT Article 4.2) recognizes two primitives:

| Invariant | Meaning | Applies When |
|:---|:---|:---|
| **`Padding Zero-Establishment`** | The kernel SHALL actively write zero to all positions at indices ≥ the logical extent, within its write footprint. The kernel is the **sole guarantor** that these positions contain zero. | The destination buffer's `Initialization Contract` is `NONE`, **and** downstream consumers read the full padded extent. |
| **`Padding Zero-Preservation`** | The kernel SHALL NOT write non-zero values to positions at indices ≥ the logical extent. The host initialization is the **primary guarantor**; the kernel's obligation is non-corruption. | The destination buffer's `Initialization Contract` is `ZERO_REQUIRED` or `ZERO_REQUIRED_ADDITIVE`. |

A third case requires no kernel invariant at all:

| Pattern | Meaning | Applies When |
|:---|:---|:---|
| **No padding obligation** | The kernel makes no claims about padding positions. Downstream consumers are contractually bounded by the logical extent and SHALL NOT access padding. | The destination buffer's `Initialization Contract` is `NONE`, **and** downstream consumers declare bounded access in their own Validation Preconditions. |

This is the existing Node 5 logits pattern, already correctly expressed: *"Padding positions beyond `total_output_class_count`... are architecturally unwritten. Downstream consumers... are contractually bounded by `total_output_class_count` and DO NOT access padding."*

### 2. Derive each kernel's invariant from the buffer's Initialization Contract

The assignment is mechanical — determined by the destination buffer's existing contract, not by editorial judgment:

| Node | Buffer | Init Contract | Invariant | Reasoning |
|:---|:---|:---|:---|:---|
| **8** | `partial_grad_weights_module` | `ZERO_REQUIRED_ADDITIVE` | **Preservation** | Host zeros before first dispatch. The kernel accumulates into `[class_chunk_offset, class_chunk_offset + classes_per_chunk)`. Padding zone `[total_output_class_count, padded)` is never touched by any tile. Host init is the sole zero source. |
| **8** | `partial_grad_biases_module` | `ZERO_REQUIRED_ADDITIVE` | **Preservation** | Same analysis as weight gradients. |
| **9** | `partial_grad_hidden_activations_aos` | `NONE` | **Establishment** | No host zeroing. Node 11 reads the full padded extent for L2 norm. The kernel is the sole guarantor. |
| **17** | `partial_grad_weights_shared` | `NONE` (implicit) | **Establishment** | No host zeroing. Node 19 reads the full padded extent for L2 norm. |
| **18** | `partial_grad_biases_shared` | `NONE` (implicit) | **Establishment** | Same reasoning. |
| **13** | `clipped_grad_hidden_...permuted_soa` | `ZERO_REQUIRED` | **Preservation** | Host zeros. The kernel writes only to positions within `[0, total_modules_count)` per row; padding `[total_modules_count, padded_total_modules_count)` is never touched. |

### 3. Recharacterize CONCEPT §3.6 as a derived theorem

The Zero-Propagation Invariant (§3.6) currently reads as if it is an independent mechanism. It should be reframed as a **system-level theorem** that *follows from* the per-kernel invariants:

> **Zero-Propagation Theorem.** *Given that* (a) the host initializes state-role padding to zero, (b) the optimizer updates only logical-extent positions (Preservation), (c) each kernel in the forward and backward chains correctly applies its declared padding invariant (Establishment or Preservation as assigned above), *then* padding positions carry zero values at every point in the DAG where a downstream consumer reads the full padded extent.
>
> This theorem is not an independent guarantee — it is the emergent property of correct per-kernel invariant assignment. Any modification to a kernel's write pattern or a buffer's Initialization Contract must be verified against this theorem by confirming the invariant assignment remains sound.

This eliminates the current confusion where a reader might believe §3.6 *independently* guarantees zeros, even if a specific kernel's invariant is ambiguous or missing.

### 4. Bandwidth consequence

Under the current contract, an implementer of Node 8 might interpret "SHALL write zero" as requiring active zero-writes to the padding zone on every tile dispatch — `padded_total_output_class_count - total_output_class_count` elements per row, per tile, per streaming dispatch. Under **Preservation**, the kernel is explicitly *prohibited* from touching those positions (it has no business writing there), and the bandwidth cost is zero.

For Node 9, the Establishment invariant costs exactly what it does today — the kernel must write those zeros. But this cost is now clearly justified: there is no other guarantor.

This is consistent with the Primacy of Memory Strategy: zero writes occur exactly once, at the jurisdictionally correct tier.

---

## Summary

| Layer | Change |
|:---|:---|
| **CONTRACT Article 4.2** | Add `Padding Zero-Establishment` and `Padding Zero-Preservation` to the Behavioral Invariants vocabulary; retire the undifferentiated `Padding Zero-Fill`. |
| **CONCEPT §3.6** | Reframe as a derived theorem contingent on per-kernel invariant correctness, not an independent system-level mechanism. |
| **kernels.cl.h** | Replace each kernel's `"Padding Zero-Fill: ..."` invariant with the appropriate distinguished invariant, mechanically derived from the destination buffer's Initialization Contract. |

The result is that every padding zero in the system has exactly one jurisdictional owner, identifiable from the buffer's own contract block, with no ambiguity about whether the host or the kernel is the load-bearing guarantor.