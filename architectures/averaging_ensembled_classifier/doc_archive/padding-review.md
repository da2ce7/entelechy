# Padding Review: `kernels.cl.h` (Revision 8)

---

## Finding 1: `weights_module` Padding Type Contradiction

**Severity: Bug**

The same physical `MODEL_STATE` buffer `src_buffer_GLOBAL_CONST_weights_module` is declared with contradictory Padding Contract types across consuming kernels:

| Kernel | Node | Declared Padding Type |
|:---|:---|:---|
| `render_logits_chunk` | 5 | `{Type: SIMD, Formula: "output_class_count padded for SIMD/Cache alignment"}` |
| `backprop_error_to_hidden_chunk` | 9 | `{Type: CACHE, Formula: "output_class_count padded for alignment"}` |

This is a single device-side allocation. The host computes one `padded_total_output_class_count` value satisfying all downstream constraints. The two kernels receive the same physical buffer with the same physical padding — the contract must agree.

The primary architectural driver is `SIMD`: the `C_TILE_SIZE` / `SIMD_WIDTH` tiling strategy determines the fundamental access granularity for the class dimension across both forward and backward passes. Cache-line alignment is a secondary consequence that the host satisfies simultaneously through the same scalar computation.

**Resolution:** Unify both declarations to `SIMD` with an explicit Formula acknowledging the joint constraint:

```
- Padding Contract: {Type: SIMD, Formula: "output_class_count padded to SIMD_WIDTH; host ensures cache-line alignment is jointly satisfied"}
```

Apply this identically to every kernel that declares `src_buffer_GLOBAL_CONST_weights_module` (Nodes 5 and 9). Verify consistency on all other buffers whose innermost dimension is `padded_total_output_class_count`:

| Buffer | Nodes | Current Type | Action |
|:---|:---|:---|:---|
| `src_buffer_GLOBAL_CONST_weights_module` | 5, 9 | SIMD / CACHE | Unify to SIMD |
| `src_buffer_GLOBAL_CONST_biases_module` | 5, 8 | SIMD / SIMD | Already consistent — verify Formula matches |
| `dest_buffer_GLOBAL_logits` | 5 (produced), 6, 7, 10 (consumed) | CACHE | Verify consistency across all consumers |

---

## Finding 2: `{Type: NONE}` on Buffers Whose Shapes Contain `padded_*` Dimensions

**Severity: Clarity**

Twelve buffer parameters declare `{Type: NONE}` while their Tensor Shape includes `padded_*` dimension parameters — the physical, in-memory padded cardinality of a logical dimension. CONTRACT.md Article 3.1 defines `NONE` as *"No padding is required or applied."* But padding **is** applied — it is encoded in the `padded_*` dimension values inherited from upstream layout constraints.

The affected buffers are not originators of padding — the host performs no independent padding calculation for them. They **reuse** pre-computed `padded_*` scalars that were derived to satisfy the constraints of upstream buffers (weights, activations):

**Node 8 — destinations:**
- `dest_buffer_GLOBAL_partial_grad_weights_module` — shape includes `padded_hidden_count × padded_total_output_class_count`
- `dest_buffer_GLOBAL_partial_grad_biases_module` — shape includes `padded_total_output_class_count`

**Node 9 — destination:**
- `dest_buffer_GLOBAL_partial_grad_hidden_activations_aos` — shape includes `padded_hidden_count`

**Node 11 — all four `src_` inputs and all four `dest_` outputs:**
- `*_partial_grad_weights_module` — `padded_hidden_count × padded_total_output_class_count`
- `*_partial_grad_biases_module` — `padded_total_output_class_count`
- `*_partial_grad_hidden_activations_aos` — `padded_hidden_count`

(The temps buffers in Node 11 correctly use `NONE` — no `padded_*` dimension in their shape.)

**Node 13 — source:**
- `src_buffer_GLOBAL_clipped_partial_grad_hidden_activations_aos` — `padded_hidden_count`

**Node 17 — destination:**
- `dest_buffer_GLOBAL_partial_grad_weights_shared` — `padded_input_count × padded_hidden_count`

**Node 18 — destination:**
- `dest_buffer_GLOBAL_partial_grad_biases_shared` — `padded_hidden_count`

Buffers that correctly remain `NONE` with no ambiguity:
- `partial_grad_temps` — shape `(total_tile_count, modules_per_chunk)`, no `padded_*` dimension
- `partial_probs` — shape uses `classes_per_chunk`, not `padded_*`
- `partial_loss` — no `padded_*` dimension
- `offset_list` — integer metadata, no `padded_*` dimension
- All scalar-shaped buffers (`temps`, `targets`, `sample_mask`)

**Why `INHERITED` was considered and rejected:** An `INHERITED` type with a Formula identifying the originating buffer would provide provenance traceability. However, `padded_*` scalars are not inherited from a single source — they are **synthesized** from the union of constraints across every buffer sharing that dimension. The binding constraint changes with `PrecisionConfig`: for `PrecisionConfig.float32()` the SIMD constraint may dominate `padded_hidden_count`; for `PrecisionConfig.fp8_e4m3()` the CACHE constraint (128 bytes ÷ 1 byte = 128 elements) dominates instead. An `INHERITED` Formula pointing to one buffer would create false precision — the reader would believe the padding originates from that specific buffer, when it originates from a host-side `lcm()` over all role-specific alignment requirements.

**Resolution:** Refine the `NONE` definition in CONTRACT.md Article 3.1:

> | Type Token | Definition |
> |:---|:---|
> | `NONE` | No independent padding strategy is applied to this buffer. The buffer's allocation dimensions may incorporate padding from `padded_*` dimension parameters, which are host-computed scalars satisfying the union of alignment constraints across all buffers sharing those dimensions. When `padded_*` parameters appear in the Tensor Shape, the padding is fully determined by those parameter values; no additional buffer-specific padding calculation is required. |

This correctly distinguishes **originators** (`CACHE`, `SIMD`, `BANK_CONFLICT_AVOIDANCE`) from **consumers** (`NONE`) of padding strategy, while the Tensor Shape's `padded_*` prefixes (syntactic jurisdiction) encode the *fact* of padding and the Padding Contract (semantic jurisdiction) encodes whether the host must *compute* a padding strategy for this buffer. No redundancy under Axiom 1.2.

---

## Finding 3: Node 13 Output — Uninitialized Padded Trailing Region

**Severity: Defensive**

**The buffer:**
```
dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa
- Tensor Shape: (total_batch_count × padded_hidden_count, padded_total_modules_count)
- Padding Contract: {Type: CACHE}
- Initialization Contract: omitted → implicit {Type: NONE}
```

Node 13 iterates over `total_modules_count` sources per row, summing across class chunks and writing one value per module position per row. Positions `[total_modules_count, padded_total_modules_count)` in each row are never written. The implicit `{Type: NONE}` initialization means those positions contain arbitrary data from allocation.

Node 16 receives both `total_modules_count` and `padded_total_modules_count`. Its Behavioral Invariants prescribe iterating over the logical count. However, Node 16's "Data Ingress Safety Invariant" describes a work-group-per-row strategy where each thread accumulates `ceil(total_modules_count / workgroup_size)` elements. A natural GPU implementation assigns threads by stride:

With `total_modules_count = 100`, `padded_total_modules_count = 128`, `workgroup_size = 64`:
- Thread 36 processes indices 36, 100 — out of logical bounds, must guard
- Thread 63 processes indices 63, 127 — in the padded region

An implementation checking `index < total_modules_count` is safe. An implementation checking `index < padded_total_modules_count` (relying on zeros in the pad region) reads uninitialized memory. The architecture should not constrain the Execution tier's loop bounds to favour one branch predicate over another — this is inconsistent with the jurisdictional model (§5) where kernel internals are an Execution-tier concern.

**Resolution:** Add `ZERO_REQUIRED` to Node 13's output:

```
@param dest_buffer_GLOBAL_clipped_grad_hidden_activations_permuted_soa
       ...
       - Initialization Contract: {Type: ZERO_REQUIRED}
```

The cost is one `clEnqueueFillBuffer` / `memset` on a buffer of size `(batch × padded_hidden) × padded_modules × sizeof(STORAGE_TYPE)`. For typical dimensions this is negligible. The benefit is that Node 16 implementations are free to use either guard condition — any thread that reads a padded position gets zero, which is the identity element for summation and harmless under the L2 norm clip.

Node 13's contract remains unchanged in what it writes — only the host's pre-dispatch obligation is added.

---

## Finding 4: Multi-Dimensional Padding Expressed as Single Type

**Severity: Structural (non-urgent)**

Several buffers have two independently padded dimensions, each padded for a different reason:

| Buffer | Dim 1 | Padded For | Dim 2 | Padded For |
|:---|:---|:---|:---|:---|
| `weights_module` (Node 5) | `padded_hidden_count` | SIMD + CACHE | `padded_total_output_class_count` | SIMD |
| `logits` (Node 5) | row stride across batch×module | CACHE | `padded_total_output_class_count` | SIMD |
| `partial_grad_weights_shared` (Node 17) | `padded_input_count` | CACHE | `padded_hidden_count` | SIMD + CACHE |

The Padding Contract field accepts one `{Type}` value, which can describe only one padding motivation. Current contracts handle this by declaring the dominant or innermost dimension's padding and mentioning the secondary one informally in the Formula.

**Assessment:** This causes no correctness issues. The host computes each `padded_*` scalar once, satisfying the union of all constraints, and every buffer gets the correct allocation size regardless of which single type the Padding Contract names. The Tensor Shape's `padded_*` prefixes already tell the reader which dimensions are padded — the single `Type` field tells them *why this buffer exists with padding at all*.

**Resolution:** Document the limitation and the future extension point in CONTRACT.md Article 3.1:

> *"When a buffer has multiple independently padded dimensions, the `Type` field declares the primary padding motivation for the buffer's characteristic access pattern. Secondary padding on other dimensions is expressed through `padded_*` parameters in the Tensor Shape and may be documented in the `Formula` field. A future revision may extend the Padding Contract to per-dimension specifications."*

If a future revision needs per-dimension contracts (e.g., for a validator that verifies alignment independently per dimension), the extension schema is:

```
- Padding Contract: {
    padded_hidden_count: {Type: CACHE, Formula: "Row stride to 128-byte alignment"},
    padded_total_output_class_count: {Type: SIMD, Formula: "Padded to SIMD_WIDTH"}
  }
```

This adds complexity with no current consumer, so deferral is appropriate.

---

## Finding 5: Mixed-Precision LCM Constraint on Shared `padded_*` Scalars

**Severity: Documentation**

A single `padded_hidden_count` scalar must satisfy alignment constraints from buffers of different precision roles. The binding constraint changes with `PrecisionConfig`:

| Config | Storage Element Size | CACHE Requirement | SIMD Requirement | Likely Binding Constraint |
|:---|:---|:---|:---|:---|
| `float32()` | 4 bytes | multiple of 32 | multiple of SIMD_WIDTH | SIMD |
| `mixed_f16_f32()` | 2 bytes | multiple of 64 | multiple of SIMD_WIDTH | Depends on SIMD_WIDTH |
| `fp8_e4m3()` | 1 byte | multiple of 128 | multiple of SIMD_WIDTH | CACHE |

Each kernel contract correctly states its own alignment requirement. The gap is that the **synthesis rule** — the host must compute `lcm(all role-specific requirements)` — is implicit knowledge not documented anywhere.

**Resolution:** Add a paragraph to CONCEPT.md §11 (Host Orchestrator & Execution Policies), point 1 (Memory Assessment & Chunk Definition):

> *"**Padded dimension synthesis.** When a padded dimension scalar (e.g., `padded_hidden_count`, `padded_total_output_class_count`) appears in the Tensor Shape of buffers with different precision roles, the Host Orchestrator computes its value as the least common multiple of all role-specific alignment requirements across every buffer that uses the scalar. For example, `padded_hidden_count` must simultaneously satisfy `padded_hidden_count × sizeof(STORAGE_TYPE) ≡ 0 (mod 128)` for CACHE-padded storage-role buffers, `padded_hidden_count ≡ 0 (mod SIMD_WIDTH)` for SIMD-padded state-role buffers, and any analogous constraint for compute-role buffers. The resulting LCM is configuration-dependent: under `PrecisionConfig.fp8_e4m3()`, the CACHE constraint (128 elements) typically dominates; under `PrecisionConfig.float32()`, the SIMD constraint may dominate. This synthesis is a Policy-tier concern; kernel contracts declare their individual requirements and are not aware of the cross-buffer union."*

This places the obligation precisely where it belongs — in the Policy tier's planning logic — and makes the implicit rule explicit without polluting individual kernel contracts with cross-buffer coordination details they should not know about.

---

## Summary

| # | Severity | Issue | Resolution | Target Document |
|:---|:---|:---|:---|:---|
| 1 | **Bug** | `weights_module`: SIMD (Node 5) vs CACHE (Node 9) | Unify to `{Type: SIMD}` with joint Formula | `kernels.cl.h` Nodes 5, 9 |
| 2 | **Clarity** | 12 buffers: `{Type: NONE}` with `padded_*` dimensions | Refine `NONE` definition | `CONTRACT.md` Article 3.1 |
| 3 | **Defensive** | Node 13 output: padded trailing region uninitialized | Add `{Type: ZERO_REQUIRED}` initialization | `kernels.cl.h` Node 13 dest |
| 4 | **Structural** | Multi-dimensional padding single type | Document limitation and future extension point | `CONTRACT.md` Article 3.1 |
| 5 | **Documentation** | Mixed-precision LCM on shared `padded_*` scalars | Add padded dimension synthesis paragraph | `CONCEPT.md` §11 point 1 |
