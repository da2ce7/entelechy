# ADR-026: Precision-Typed Reduction Kernel Variants

**Status:** PROPOSED
**Date:** 2026-04-03
**Deciders:** —
**Triggered by:** Type mismatch discovered in multi-stage reductions and BCE loss leaf-stage reductions under mixed-precision configurations
**Depends on:** ADR-020 (Three-Role Precision Model), ADR-021, ADR-007 (Kernel Signature Contract), ADR-019 (K-Fan-In Reduction)
**Constrains:** All reduction kernels in `kernels/`

---

## Context

Working through mixed-precision configurations reveals a type mismatch wider than originally identified. The mismatch manifests in **two independent scenarios**:

**Scenario A — Multi-Stage Interior Stages:**
Stage 0's `COMPUTE_TYPE` output is fed to stage ≥ 1, whose parameter declares `STORAGE_TYPE*`. Under `PrecisionConfig.mixed_f16_f32()`, stage 1 applies `vload_half` to FP32 bit patterns.

**Scenario B — BCE Loss Leaf-Stage Reduction:**
Node 7 (`compute_probs_loss_bce_chunk`) writes `dest_buffer_GLOBAL_partial_loss` as `COMPUTE_TYPE*` (precision role `"compute"`). When the Orchestration tier feeds this collection into any current reduction kernel — whether `aggregate_register_reduce`, `aggregate_local_reduce`, or `reduce_k_fan_in_and_clip` — the kernel's input parameter declares `STORAGE_TYPE*` and applies `load_storage()`. Same reinterpretation failure.

Both scenarios are invisible when all precision roles share a type (the common `float32()` configuration), which explains their latency.

| Reduction Target | Source Buffer Role | Single-Stage | Multi-Stage Stage 0 | Multi-Stage Stage ≥ 1 |
|:---|:---|:---|:---|:---|
| Probs (CCE/BCE) | `storage` | ✓ | ✓ | ✗ **(A)** |
| BCE Loss | `compute` | ✗ **(B)** | ✗ **(B)** | ✗ **(A)** |
| All Gradient Paths | `storage` | ✓ | ✓ | ✗ **(A)** |

✓ = type-safe with existing kernels. ✗ = type mismatch.

---

## Decision Drivers

1. **Principle §1 (Architectural Elegance Feedback).** Per AGENTS.md, optimization pressure that violates core principles indicates incomplete architectural modeling. The type mismatch triggers the formal feedback cycle: suspend ad-hoc workarounds, formalize the pattern, and reify through revised contracts.

2. **The Primacy of Memory Strategy.** Leaf-stage data lives in STORAGE_TYPE (narrow, bandwidth-efficient); once widened at the precision boundary, intermediate data stays in COMPUTE_TYPE for the remainder of the pipeline. The reduction kernel set must respect both roles.

3. **OpenCL type constraints are hard.** The C type signature is enforced by the OpenCL compiler. Variant kernels with different parameter types are the only correct solution — no void* casting or narrowing hacks.

4. **No plan model changes required.** The Policy tier's `ReductionTreeNode` already delegates kernel selection to the backend renderer. Variant selection is an Orchestration-tier concern.

---

## Decision

Every reduction kernel exists in two **precision variants**, differentiated solely by the precision role of the source buffer. The Orchestration tier selects the variant based on the source buffer's `precision_role` from its `BufferDescriptor`.

### §1: Variant Taxonomy

| Variant | Input Role | Load Mechanism | When Used |
|:---|:---|:---|:---|
| **Storage-entry** (existing) | `"storage"` | `load_storage()` | Leaf stage reading from a STORAGE_TYPE partial collection |
| **Compute-entry** (new) | `"compute"` | Direct `COMPUTE_TYPE` read | Interior stages reading prior COMPUTE_TYPE output; leaf stage reading from a COMPUTE_TYPE collection (e.g., BCE loss) |

Both variants share identical output type (`COMPUTE_TYPE`), identical algorithm, identical scalar parameters, and identical clipping logic. The divergence is exactly one buffer's C type and its load path.

### §2: Naming Convention

**Suffix: `_from_compute`**

Applied to all three reduction kernel families:

| Existing (Storage-Entry) | New (Compute-Entry) |
|:---|:---|
| `reduce_k_fan_in_and_clip` | `reduce_k_fan_in_and_clip_from_compute` |
| `aggregate_register_reduce` | `aggregate_register_reduce_from_compute` |
| `aggregate_local_reduce` | `aggregate_local_reduce_from_compute` |

The existing kernels are **unchanged** — no rename, no contract modification. The `_from_compute` suffix is self-documenting within the precision model's vocabulary and follows the existing descriptive-suffix pattern (`_soa`, `_aos`, `_simd_major`, `_permuted`).

### §3: Orchestration Tier Selection Rule

The Orchestration tier selects the variant based on two orthogonal criteria resolved by simple dispatch logic:

```
let source_role = buffer_descriptor.precision_role of the input collection

# Single-stage tree
if num_stages == 1:
    if source_role == "storage":
        dispatch aggregate_{tier}_reduce
    else:  # source_role == "compute"
        dispatch aggregate_{tier}_reduce_from_compute

# Multi-stage tree
if num_stages > 1:
    for stage in 0..num_stages-1:
        if stage == 0 and source_role == "storage":
            dispatch reduce_k_fan_in_and_clip
        else:
            dispatch reduce_k_fan_in_and_clip_from_compute
```

The `source_role` for stage 0 is the precision role of the original partial collection buffer (known to the Policy tier via `BufferDescriptor`). For stages ≥ 1, the source is always the prior stage's `COMPUTE_TYPE` output — the dispatch is unconditionally `_from_compute`.

### §4: Kernel Contract — `reduce_k_fan_in_and_clip_from_compute`

```c
/**
 * @brief (Node 14, 15a, 20a — interior stages and compute-role leaf stages)
 *        Compute-entry variant of reduce_k_fan_in_and_clip. Identical algorithm,
 *        but reads COMPUTE_TYPE intermediates rather than STORAGE_TYPE partials.
 * @kernel_contract
 *        - Holistic Constraints: "Each work-group processes one reduction node.
 *          The kernel reads K partials per node from the source buffer via an
 *          offset list, sums them, optionally clips the result per-node, and
 *          writes one output vector of partial_width elements. Supports absent
 *          partials via sentinel offset 0xFFFFFFFF for the tail node."
 *        - Behavioral Invariants: "When clipping_threshold > 0, per-node L2
 *          clip is applied: scale = threshold / (norm + epsilon). When
 *          clipping_threshold == 0, clip is bypassed (diagnostic mode).
 *          Epsilon prevents division by zero. All buffers are compute-role;
 *          no precision boundary conversion is required. All arithmetic
 *          exclusively in COMPUTE_TYPE."
 *        - Idempotency: "Associatively Non-Idempotent"
 *        - Synchronization Model: "Reduction Engine Stage"
 *        - Precision Variant: "Compute-entry variant of reduce_k_fan_in_and_clip.
 *          Used for interior stages of multi-stage reduction trees (where the
 *          source is a prior stage's COMPUTE_TYPE output) and for leaf stages
 *          whose source collection is natively COMPUTE_TYPE (e.g., BCE loss
 *          partials from Node 7)."
 */
__kernel void reduce_k_fan_in_and_clip_from_compute(
    __local COMPUTE_TYPE *update_buffer_LOCAL_reduction_tile,
    __global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,
    __global const uint *src_buffer_GLOBAL_CONST_offset_list_flat,
    __global COMPUTE_TYPE *dest_buffer_GLOBAL_stage_output,
    uint src_scalar_NATURAL_fan_in_K,
    uint src_scalar_NATURAL_node_count,
    uint src_scalar_NATURAL_partial_width,
    COMPUTE_TYPE src_scalar_REAL_clipping_threshold,
    COMPUTE_TYPE src_scalar_REAL_epsilon);
```

**Delta from the storage-entry variant:**

| Aspect | Storage-Entry | Compute-Entry |
|:---|:---|:---|
| `src_buffer_GLOBAL_partial_collection` C type | `__global const STORAGE_TYPE *` | `__global const COMPUTE_TYPE *` |
| Precision Role of that buffer | `"storage"` | `"compute"` |
| Behavioral Invariant | `Precision Boundary Conversion` required | Not required (all compute-role) |
| Internal load | `load_storage(buf, idx)` | `buf[idx]` (direct COMPUTE_TYPE read) |
| Everything else | Identical | Identical |

### §5: Aggregate Kernel Variants

`aggregate_register_reduce_from_compute` and `aggregate_local_reduce_from_compute` follow the identical pattern. Each differs from its parent kernel in exactly one parameter:

```c
// Storage-entry (existing):
__global const STORAGE_TYPE *src_buffer_GLOBAL_partial_collection,
// Precision Role: "storage"

// Compute-entry (new):
__global const COMPUTE_TYPE *src_buffer_GLOBAL_partial_collection,
// Precision Role: "compute"
```

Behavioral Invariants drop the `Precision Boundary Conversion` clause. All other parameters, shapes, validation preconditions, and algorithmic behavior are unchanged.

### §6: Implementation — Shared Algorithm

At the OpenCL source level, the algorithm is shared via a macro-parameterized expansion:

```c
#define REDUCE_K_BODY(LOAD_FN, collection, offset_list, dest, ...)    \
    /* ... identical algorithm for sum, clip, write ... */             \
    /* Each element access: val = LOAD_FN(collection, offset + j) */

__kernel void KERNEL_ATTR reduce_k_fan_in_and_clip(
    __local COMPUTE_TYPE *local_tile,
    __global const STORAGE_TYPE *collection,  /* storage-entry */
    __global const uint *offsets,
    __global COMPUTE_TYPE *dest, ...) {
    REDUCE_K_BODY(load_storage, collection, offsets, dest, ...)
}

__kernel void KERNEL_ATTR reduce_k_fan_in_and_clip_from_compute(
    __local COMPUTE_TYPE *local_tile,
    __global const COMPUTE_TYPE *collection,  /* compute-entry */
    __global const uint *offsets,
    __global COMPUTE_TYPE *dest, ...) {
    REDUCE_K_BODY(/* identity load */, collection, offsets, dest, ...)
}
```

This is an **Execution-tier concern** — the macro is internal to the kernel source file and invisible to the contract, the plan model, and the host. The two entry points remain separate kernel functions with distinct, verifiable `KernelContract` dataclasses.

---

## Consequences

### Positive

1. **Type-safe mixed-precision reductions.** All scenarios (A) and (B) are resolved. The Orchestration tier dispatches the correct variant based on buffer role.

2. **No plan model changes.** The `ReductionTreeNode` delegates kernel selection to the backend renderer. Variant selection is transparent to the Policy tier.

3. **No Policy-tier changes.** The existing `BufferDescriptor.precision_role` provides all information the Orchestration tier needs.

4. **Compiler optimization for equal-type configurations.** When `STORAGE_TYPE == COMPUTE_TYPE`, both variants compile to identical machine code. The identity `load_storage` path is eliminated by the compiler.

5. **Formal pattern for future variants.** The `Precision Variant` contract key (CONTRACT.md Article 4.2) documents the relationship between variant pairs, making the pattern inspectable and discoverable.

### Negative

1. **Three new kernels.** The kernel set grows from N to N+3. Build time increases marginally.

2. **Backend renderers must implement selection.** Each backend's Orchestration-tier code must query `buffer_descriptor.precision_role` and dispatch accordingly. This is straightforward dispatch logic.

### Cross-Cutting Amendments

**CONCEPT.md — Recursive Clip-Aggregation Engine:**

For Multi-Stage Trees:
> The tree's leaf stage (stage 0) uses `reduce_k_fan_in_and_clip` when the source collection is storage-role, or `reduce_k_fan_in_and_clip_from_compute` when the source collection is compute-role (e.g., BCE loss partials). All interior stages (stage ≥ 1) use `reduce_k_fan_in_and_clip_from_compute`, reading the prior stage's COMPUTE_TYPE output.

For Single-Stage Trees:
> The tree uses `aggregate_stage_j` (storage-entry or compute-entry variant, depending on the source buffer's precision role), optionally followed by `clip_stage_j`.

**CONTRACT.md — Article 4.2:**

Add `Precision Variant` as a recognized key in the `@kernel_contract` block:

| Key | Definition | Status |
|:---|:---|:---|
| **`Precision Variant`** | Declares this kernel as a precision-typed variant of a named base kernel, documenting the specific buffer role divergence and the selection criterion. | Optional |

---

## Verification

This design is validated by the existing **"Alchemist" scenario** (Mixed-Precision Fidelity Validation):

**Test 1:** Execute a multi-stage gradient reduction under `PrecisionConfig.mixed_f16_f32()` and confirm that:
1. Stage 0 reads FP16 partials via `load_storage()` (storage-entry kernel)
2. Stages ≥ 1 read FP32 intermediates directly (compute-entry kernel)
3. Final summed gradient matches the FP32 reference within tolerance

**Test 2:** Execute a BCE loss diagnostic reduction under the same mixed configuration and confirm the single-stage aggregate correctly sums the COMPUTE_TYPE loss partials.

Both sub-tests produce garbage under the current (unfixed) kernel set, providing a clear pass/fail signal.
