# ADR-021: Precision-Role Migration of the Kernel Reference Corpus

**Status:** ACCEPTED
**Date:** 2026-04-02
**Deciders:** —
**Triggered by:** ADR-020 (Three-Role Precision Model) §4.4 and §3.6
**Depends on:** ADR-020, ADR-013 (Kernel Source Strategy), ADR-007 (Kernel Signature Contract)
**Constrains:** All files under `kernels/`

---

## Context

ADR-020 established the three-role precision model as a first-class architectural primitive and mandated (§4.4) that all kernel sources migrate from the retired single-symbol `SCALAR_TYPE` to the role-appropriate symbols `STORAGE_TYPE`, `COMPUTE_TYPE`, and `STATE_TYPE`. ADR-020 §3.6 retired `SCALAR_TYPE` and `SCALAR_IS_HALF` from the binding build-time symbol set defined in CONTRACT.md Article 6.

The `kernels/` directory contains two distinct artifact categories that are both affected:

1. **`kernels.cl.h` — the algorithmic specification document.** Per ADR-013, this file is the language-neutral reference that all backend implementations are verified against. Its `@kernel_contract` blocks and `@param` annotations are the source of truth for `KernelContract` and `BufferParamSpec` objects in the plan-layer. Under ADR-020, every buffer `@param` block must gain a mandatory `Precision Role` commentary key (CONTRACT.md Article 3, amended by ADR-020 §3.2), and every buffer type declaration must change from `SCALAR_TYPE*` to the appropriate role type symbol. Additionally, the header's OpenCL build-time symbol checks and the host/C++ stub definitions must be updated to reflect the new three-symbol scheme. Precision boundary load/store abstraction declarations must be added to the header.

2. **`phase_*.cl.c` — the OpenCL implementation files.** These files implement the kernels declared in `kernels.cl.h`. Their `SCALAR_TYPE` references in local variables, function body arithmetic, and kernel parameter lists must be migrated to match the role-typed declarations. This ADR does not change any algorithmic logic — it changes only the type symbols used at data boundaries.

This ADR formally specifies the per-kernel precision role assignment for every buffer in the specification corpus, derives the `@kernel_contract` `Behavioral Invariants` additions required by CONTRACT.md Article 4.2 (amended by ADR-020 §3.5), and documents the required changes to the header's build-symbol machinery.

---

## Decision Drivers

1. **ADR-020 §4.4 is binding.** The migration is not optional. The requirement flows from the Three-Role Precision Model (a CONCEPT.md-level decision) through the CONTRACT.md Article 6 replacement. `kernels.cl.h` as an authority-document annex cannot diverge from CONTRACT.md's vocabulary.

2. **Role assignment must be exhaustive and consistent.** Every buffer in every kernel gets exactly one precision role. The role follows from the buffer's semantic function (ADR-020 §4.2 table) with no ambiguity and no remainder.

3. **`kernels.cl.h` is the specification; the `.cl.c` files are derived.** Role assignments are decided here for the specification document. The `.cl.c` files must match the specification and are updated accordingly.

4. **No kernel logic changes.** This migration changes type declarations at data boundaries. Algorithmic correctness is preserved because the precision boundary load/store abstractions emit the same bit patterns as the prior direct `SCALAR_TYPE` accesses when roles are equal-type (the current only-deployed configuration). When roles differ, the abstractions insert the correct widening/narrowing conversions that were previously architecturally inexpressible.

5. **Transitional alias.** Consistent with ADR-020 §3.6, the build system may provide `SCALAR_TYPE` as an alias for `COMPUTE_TYPE` and `SCALAR_IS_HALF` as an alias for `COMPUTE_TYPE_IS_HALF` during the migration window. This alias allows continuous build validity while the `.cl.c` files are updated incrementally. The alias is removed when the migration of all files is complete.

---

## Role Assignment

### Governing Principle (ADR-020 §4.2)

| Buffer classification | Precision role |
|:---|:---|
| Forward-pass activations, gradient partials, logit intermediates, backpropagated error signals | `"storage"` |
| Reduction tree accumulation outputs, final loss values, normalized gradient vectors | `"compute"` |
| Learnable parameters (weights, biases, temperatures), optimizer moment vectors (`m1`, `m2`) | `"state"` |

Local memory (`__local`) scratch buffers used for intra-workgroup arithmetic reduction hold intermediate sums in the arithmetic type. They are declared `COMPUTE_TYPE`.

`int`-typed buffers (e.g., `targets` for CCE) are not role-dependent — they carry integer labels, not floating-point data. They do not receive a `Precision Role` commentary key and their declarations are unchanged.

---

### §1: `kernels.cl.h` — Specification Document Changes

#### §1.1: Build-Time Symbol Section

The OpenCL-environment mandatory-symbol block currently checks `SCALAR_TYPE`, `SCALAR_IS_HALF`, `SIMD_WIDTH`, `C_TILE_SIZE`, and `NUMERICAL_STABILITY_EPSILON`. It is replaced by:

```c
// --- Mandatory Build-Time Symbols (Article 6, amended by ADR-020 §3.6) ---

#ifndef STORAGE_TYPE
#error "System Contract Violation: STORAGE_TYPE must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE
#error "System Contract Violation: COMPUTE_TYPE must be defined by the host build system."
#endif
#ifndef STATE_TYPE
#error "System Contract Violation: STATE_TYPE must be defined by the host build system."
#endif
#ifndef STORAGE_TYPE_IS_HALF
#error "System Contract Violation: STORAGE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef COMPUTE_TYPE_IS_HALF
#error "System Contract Violation: COMPUTE_TYPE_IS_HALF must be defined by the host build system."
#endif
#ifndef SIMD_WIDTH
#error "System Contract Violation: SIMD_WIDTH must be defined by the host build system."
#endif
#ifndef C_TILE_SIZE
#error "System Contract Violation: C_TILE_SIZE must be defined by the host build system."
#endif
#ifndef NUMERICAL_STABILITY_EPSILON
#error "System Contract Violation: NUMERICAL_STABILITY_EPSILON must be defined by the host build system."
#endif
```

The `cl_khr_fp16` extension guard moves from `SCALAR_IS_HALF` to `STORAGE_TYPE_IS_HALF`. When `STORAGE_TYPE_IS_HALF == 1`, `vload_half`/`vstore_half` semantics activate for storage-role buffers. A separate guard on `COMPUTE_TYPE_IS_HALF` enables half-precision extension support for compute-role arithmetic if needed.

The `SCALAR_ZERO` definition, which required a half/float conditional, is replaced by `COMPUTE_ZERO` derived from `COMPUTE_TYPE_IS_HALF`, since zero-initialization of scratch space uses the arithmetic type.

#### §1.2: Host/C++ Stub Section

The host-mode stub definitions that currently define `SCALAR_TYPE`, `SCALAR_ZERO`, and `SCALAR_IS_HALF` at compile time are replaced with stubs for `STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`, `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`, and `COMPUTE_ZERO`. These stubs default all three role types to `float` and the `_IS_HALF` flags to `0` for host-mode analysis builds, matching the prior behavior for the uniform FP32 configuration.

#### §1.3: Precision Boundary Load/Store Abstractions

The following inline function declarations are added immediately after the mandatory-symbol checks, within the `#ifdef __OPENCL_VERSION__` guard:

```c
// --- Precision Boundary Abstractions (ADR-020 §4.4) ---
// These abstractions are the sole mechanism for crossing precision boundaries.
// When STORAGE_TYPE == COMPUTE_TYPE, the compiler eliminates these as identity
// operations. No #ifdef on type equality is used anywhere in the kernel sources.

static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx);

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val);

static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx);

static inline void store_state(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val);

static inline void store_state_update(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val);
```

Implementations of these abstractions reside in each backend's `kernel_sources/` directory per ADR-013. `kernels.cl.h` provides declarations only.

#### §1.4: Per-Kernel Buffer Role Declarations

The following table specifies the `Precision Role` commentary value and the corresponding C type symbol for every buffer parameter across all kernel declarations in `kernels.cl.h`. Every buffer `@param` block gains a `Precision Role` line positioned after `Padding Contract` and before `Calculability Proof`.

##### Phase 1 Act — `forward_pass`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_simd_tile` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_input` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_sample_mask` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_CONST_weights_shared_simd_major` | `"state"` | `STATE_TYPE` |
| `src_buffer_GLOBAL_CONST_biases_shared` | `"state"` | `STATE_TYPE` |
| `dest_buffer_GLOBAL_hidden_activations` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_hidden_mask` | `"storage"` | `STORAGE_TYPE` |

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: storage-role inputs loaded via load_storage(); state-role inputs loaded via load_state(); storage-role outputs narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."`

##### Phase 1 Act — `render_logits_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `src_buffer_GLOBAL_hidden_activations` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_hidden_mask` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_CONST_weights_module` | `"state"` | `STATE_TYPE` |
| `src_buffer_GLOBAL_CONST_biases_module` | `"state"` | `STATE_TYPE` |
| `dest_buffer_GLOBAL_logits` | `"storage"` | `STORAGE_TYPE` |

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; logit output narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."`

##### Phase 1 Act — `compute_probs_loss_cce_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `src_buffer_GLOBAL_logits` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_CONST_temps` | `"state"` | `STATE_TYPE` |
| `src_buffer_GLOBAL_targets` | *(integer, no role)* | `int` |
| `src_buffer_GLOBAL_sample_mask` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_probs` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_final_loss` | `"compute"` | `COMPUTE_TYPE` |

`dest_buffer_GLOBAL_final_loss` carries precision role `"compute"` because it is a direct arithmetic output of the Softmax/log-sum-exp reduction. This is the first compute-role write-destination in the DAG.

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: storage-role and state-role inputs widened to COMPUTE_TYPE upon load; partial_probs narrowed via store_storage(); final_loss written directly in COMPUTE_TYPE (no narrowing). All arithmetic exclusively in COMPUTE_TYPE."`

##### Phase 1 Act — `compute_probs_loss_bce_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `src_buffer_GLOBAL_logits` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_CONST_temps` | `"state"` | `STATE_TYPE` |
| `src_buffer_GLOBAL_targets` | *(float mask, storage-role)* | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_sample_mask` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_probs` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_loss` | `"compute"` | `COMPUTE_TYPE` |

Note: for BCE, `targets` is a multi-hot float tensor and is classified `"storage"`. The CCE `targets` is an integer class-index tensor and is typed `int` — it has no precision role.

**Required `Behavioral Invariants` addition:** Same pattern as CCE chunk; partial_loss written in `COMPUTE_TYPE`.

##### Phase 2 Learn A — `calculate_module_param_grads_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_reduction_tile` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_hidden_activations` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_partial_probs` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_grad_weights_module` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_grad_biases_module` | `"storage"` | `STORAGE_TYPE` |

Partial gradient buffers are `"storage"` because they are intermediate transient DAG data written by one kernel and consumed by the reduction engine. They carry gradient partial sums before normalization — bandwidth-bound transient data.

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: storage-role inputs widened via load_storage(); partial gradient outputs narrowed via store_storage(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE."`

##### Phase 2 Learn A — `backprop_error_to_hidden_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_reduction_tile` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_partial_probs` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_CONST_weights_module` | `"state"` | `STATE_TYPE` |
| `dest_buffer_GLOBAL_partial_grad_hidden_activations` | `"storage"` | `STORAGE_TYPE` |

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: storage-role and state-role inputs widened upon load; storage-role output narrowed via store_storage(). All arithmetic exclusively in COMPUTE_TYPE."`

##### Phase 2 Learn A — `calculate_chunk_temp_gradients`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_reduction_tile` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_partial_probs` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_logits` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_CONST_temps` | `"state"` | `STATE_TYPE` |
| `dest_buffer_GLOBAL_partial_grad_temps` | `"storage"` | `STORAGE_TYPE` |

##### Phase 2 Learn B — `clip_partial_gradients`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_GLOBAL_partial_grads` | `"storage"` | `STORAGE_TYPE` |

Pure in-place elementwise clip on transient gradient partials — storage-role throughout.

##### Phase 2 Learn B — `gather_and_permute_grad_hidden_activations`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `src_buffer_GLOBAL_partial_grad_hidden_activations` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_gathered_grad_hidden_activations` | `"storage"` | `STORAGE_TYPE` |

Pure gather/permute of transient gradient partials.

##### Phase 2 Learn C — `aggregate_register_reduce` / `aggregate_local_reduce`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_*` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_*` (partial inputs) | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_*` (reduction output) | `"compute"` | `COMPUTE_TYPE` |

The reduction engine's output buffers (final accumulated gradient sums) are `"compute"` because they are the product of the reduction tree arithmetic. They feed directly into `normalize_gradients`, which operates in `COMPUTE_TYPE` throughout.

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: storage-role inputs widened via load_storage(); reduction accumulation in COMPUTE_TYPE; compute-role output written directly in COMPUTE_TYPE."`

##### Phase 2 Learn C — `clip_intermediate_grad` / `stabilize_and_reduce_grad_hidden_activations` / `reduce_k_fan_in_and_clip`

All intermediate reduction and clipping kernels in the reduction engine:

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_*` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_*` (partial inputs, pre-final-reduction) | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_*` (intermediate reduction outputs) | `"storage"` or `"compute"` | see note |

Intermediate reduction outputs that feed another reduction stage are still `"storage"` (transient inter-stage data). The *final* reduction output that exits the reduction tree into the normalization phase is `"compute"`.

##### Phase 2 Learn D — `backprop_shared_weights_chunk` / `backprop_shared_biases_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_LOCAL_reduction_tile` | compute (LOCAL scratch) | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_grad_hidden_activations` | `"compute"` | `COMPUTE_TYPE` |
| `src_buffer_GLOBAL_input` | `"storage"` | `STORAGE_TYPE` |
| `src_buffer_GLOBAL_hidden_mask` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_grad_weights_shared` | `"storage"` | `STORAGE_TYPE` |
| `dest_buffer_GLOBAL_partial_grad_biases_shared` | `"storage"` | `STORAGE_TYPE` |

Note: `grad_hidden_activations` arrives from the reduction engine as `"compute"` and is consumed directly in COMPUTE_TYPE arithmetic without a widening conversion needed.

##### Phase 2 Learn D — `clip_shared_gradients_chunk`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_GLOBAL_partial_grad_shared` | `"storage"` | `STORAGE_TYPE` |

In-place clip on transient gradient partials.

##### Phase 3 Update — `normalize_gradients`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_GLOBAL_summed_gradient` | `"compute"` | `COMPUTE_TYPE` |

Receives the final reduction output in `COMPUTE_TYPE` and divides in place. All operations remain in `COMPUTE_TYPE`.

##### Phase 3 Update — `adam_update`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `src_buffer_GLOBAL_normalized_gradient` | `"compute"` | `COMPUTE_TYPE` |
| `update_buffer_GLOBAL_m1` | `"state"` | `STATE_TYPE` |
| `update_buffer_GLOBAL_m2` | `"state"` | `STATE_TYPE` |
| `update_buffer_GLOBAL_weights_or_biases` | `"state"` | `STATE_TYPE` |

Adam's EMA update reads and writes optimizer moments in `STATE_TYPE` via `load_state`/`store_state_update`, and reads the normalized gradient in `COMPUTE_TYPE` (already correct arithmetic type). The parameter update is applied to the `"state"` weight/bias buffer via `store_state_update`.

**Required `Behavioral Invariants` addition:** `"Precision Boundary Conversion: state-role moment and parameter buffers accessed via load_state()/store_state_update(); gradient consumed directly in COMPUTE_TYPE. EMA arithmetic exclusively in COMPUTE_TYPE."`

##### Phase 3 Update — `clamp_temperatures`

| Buffer parameter | Precision Role | C type |
|:---|:---|:---|
| `update_buffer_GLOBAL_temps` | `"state"` | `STATE_TYPE` |

Temperature parameters are persistent learnable state — `"state"` role. In-place clamp reads via `load_state`, writes via `store_state_update`.

---

### §2: `.cl.c` Implementation File Changes

Each implementation file is updated independently as follows:

1. Every kernel parameter declaration changes from `SCALAR_TYPE` to the role-appropriate type per the §1.4 table.
2. Every `__local SCALAR_TYPE` scratch buffer declaration changes to `__local COMPUTE_TYPE`.
3. Every local variable that holds an intermediate arithmetic value changes from `SCALAR_TYPE` to `COMPUTE_TYPE`.
4. Direct array loads from storage-role buffers (`buf[idx]`) are replaced with `load_storage(buf, idx)`.
5. Direct array loads from state-role buffers are replaced with `load_state(buf, idx)`.
6. Direct array stores to storage-role buffers are replaced with `store_storage(buf, idx, val)`.
7. Direct array stores to state-role buffers are replaced with `store_state_update(buf, idx, val)`.
8. Loads from compute-role buffers and `__local COMPUTE_TYPE` buffers are direct (`buf[idx]`) — no abstraction needed since the value is already in the arithmetic type.
9. The literal `SCALAR_ZERO` is replaced with `COMPUTE_ZERO` in local variable initializations.

No algorithmic expressions, loop bounds, index calculations, barrier calls, or work-item dispatch semantics are changed.

---

### §3: Transitional Alias Protocol

During the migration window (in-progress state where some `.cl.c` files have been migrated and others have not), the Meson build system provides:

```meson
# Transitional aliases — removed when migration completes.
'-DSCALAR_TYPE=COMPUTE_TYPE',
'-DSCALAR_IS_HALF=COMPUTE_TYPE_IS_HALF',
'-DSCALAR_ZERO=COMPUTE_ZERO',
```

These aliases allow unmigrated files to compile without error. The `kernels.cl.h` mandatory-symbol block does **not** check for `SCALAR_TYPE` after the header update — the check section reflects the post-migration symbol set. Unmigrated `.cl.c` files compile only because the build system injects the alias; the contract document does not endorse the retired symbol.

The transitional alias block is removed in the same commit that migrates the last `.cl.c` file.

---

### §4: Padding Contract Byte-Size Corrections

Every buffer `@param` block in `kernels.cl.h` that currently references `sizeof(SCALAR_TYPE)` in its `Validation Preconditions` text is updated to reference `sizeof(ROLE_TYPE)` where `ROLE_TYPE` is the concrete symbol for that buffer's precision role (e.g., `sizeof(STORAGE_TYPE)`, `sizeof(STATE_TYPE)`, `sizeof(COMPUTE_TYPE)`).

This is a documentation-correctness fix: the byte allocation formula must use the element size of the buffer's actual type, which now varies by role under mixed-precision configurations.

---

### §5: Verification Obligations

Per CONTRACT.md Article 1.4 (as extended by ADR-020 §3.5), the host's proof obligation for every kernel with `Precision Boundary Conversion` in its `Behavioral Invariants` extends to:

1. **Type symbol consistency:** the `BufferParamSpec.precision_role` field in the Python `KernelContract` matches the C type symbol in the kernel signature (`STORAGE_TYPE` ↔ `"storage"`, etc.).
2. **Invariant declaration:** every kernel that accesses non-`"compute"`-role buffers explicitly declares `Precision Boundary Conversion` in its `@kernel_contract` `Behavioral Invariants`.
3. **Allocation size:** all host-side buffer allocation size formulas use the role-appropriate dtype's `itemsize` from the active `PrecisionConfig`.

These obligations are verified by the plan-construction validation layer (ADR-007/ADR-009) and exercised by Tier 2 tests across all `PrecisionConfig` factory configurations per ADR-020 §4.6.

---

## Consequences

### Positive

- **`kernels.cl.h` becomes specification-complete under the three-role model.** Every buffer parameter carries a `Precision Role` key; every kernel with non-compute-role buffers declares `Precision Boundary Conversion`; the build-symbol section reflects the authoritative Article 6 vocabulary. The file fulfills its ADR-013 designation as the sole specification document.
- **Mixed-precision deployable immediately.** Once the migration is complete and the transitional alias removed, `PrecisionConfig.mixed_f16_f32()` is usable end-to-end with zero kernel source changes.
- **Uniform configurations are zero overhead.** When all roles are `float32`, the precision boundary abstractions are identity functions. The compiler eliminates them. No performance regression for the currently-deployed configuration.
- **FP8 storage path is structurally ready.** When the FP8 preconditions from ADR-020 §5 are met, FP8 as `STORAGE_TYPE` requires only a new `load_storage`/`store_storage` implementation — no kernel source changes beyond the abstraction layer.

### Negative

- **Scope.** `kernels.cl.h` is a large file with ~30 kernel declarations. Adding `Precision Role` to every buffer parameter and updating every type declaration is mechanical but non-trivial in line count.
- **`@param` allocation preconditions become role-specific.** The `Validation Preconditions` byte-size formulas must be updated on every buffer to replace `sizeof(SCALAR_TYPE)` with the role-specific expression. This is a correctness requirement, not optional.
- **Transitional alias must be actively removed.** Without a disciplined removal commit, the alias persists and the migration is never formally closed. The completion criterion is: `SCALAR_TYPE` appears in no kernel source file (`.cl.c`, `.comp`, `.inc`) and the alias is absent from all Meson build files.

### Migration Completion Criterion

The migration is complete when:
1. `grep -r SCALAR_TYPE architectures/averaging_ensembled_classifier/kernels/` returns no results.
2. `grep -r SCALAR_TYPE architectures/averaging_ensembled_classifier/src/backends/` returns no results.
3. The transitional alias block is absent from all Meson build configuration files.
4. Tier 2 tests pass for `PrecisionConfig.float32()` and `PrecisionConfig.mixed_f16_f32()`.

---

## References

- [ADR-020](ADR-020-mixed-precision.md) §4.4 — Kernel Source Strategy constraint (the triggering mandate)
- [ADR-020](ADR-020-mixed-precision.md) §3.6 — Article 6 build-symbol replacement
- [ADR-020](ADR-020-mixed-precision.md) §3.2 — `Precision Role` commentary key
- [ADR-020](ADR-020-mixed-precision.md) §3.5 — `Precision Boundary Conversion` behavioral invariant
- [ADR-020](ADR-020-mixed-precision.md) §4.2 — Buffer role assignment table (governing principle for §1.4 above)
- [ADR-013](ADR-013-kernel-source-strategy.md) — `kernels.cl.h` as the language-neutral specification document
- [ADR-007](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract`/`BufferParamSpec` verification
- [ADR-014](ADR-014-build-system-integration.md) — build-system provisioning of precision symbols (§3 above)
- [ADR-016](ADR-016-test-strategy.md) — Tier 2 test obligations across precision configurations
