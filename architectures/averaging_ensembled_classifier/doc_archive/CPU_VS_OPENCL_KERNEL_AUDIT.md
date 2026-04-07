# CPU vs OpenCL Kernel Audit Report

**Date:** 2026-03-29
**Scope:** All 18 kernel functions across 6 phase files
**OpenCL sources:** `kernels/*.cl.c`
**CPU sources:** `src/backends/cpu/kernel_sources/*.c`
**Authoritative spec:** `kernels/kernels.cl.h` (ADR-013)

---

## Kernel-by-Kernel Equivalence

| Node | OpenCL Kernel | CPU Task Function | Verdict |
|------|--------------|-------------------|---------|
| 4 | `forward_pass` | `task_forward_pass` | Equivalent (see Finding 3) |
| 5 | `render_logits_chunk` | `task_render_logits` | Equivalent |
| 6 | `compute_probs_loss_cce_chunk` | `task_cce_probs_loss` | Equivalent |
| 7 | `compute_probs_loss_bce_chunk` | `task_bce_probs_loss` | **Divergence** (Finding 1) |
| 8 | `calculate_module_param_grads_chunk` | `task_module_param_grads` | Equivalent (see Finding 4) |
| 9 | `backprop_error_to_hidden_chunk` | `task_backprop_to_hidden` | Equivalent |
| 10 | `calculate_chunk_temp_gradients` | `task_temp_gradients` | Equivalent |
| 11 | `clip_partial_gradients` | `task_clip_partial_grads` | Equivalent |
| 13 | `gather_and_permute_grad_hidden_activations` | `task_gather_permute_grad_h` | Equivalent |
| 14/15a/20a | `aggregate_register_reduce` / `aggregate_local_reduce` / `reduce_k_fan_in_and_clip` | `execute_reduction_tree` | **Resolved** (Finding 2, ADR-019) |
| 15b/20b | `clip_intermediate_grad` | `task_clip_intermediate` | Equivalent |
| 16 | `stabilize_and_reduce_grad_hidden_activations` | `task_stabilize_reduce_grad_h` | Equivalent |
| 17 | `backprop_shared_weights_chunk` | `task_backprop_shared_weights` | Equivalent |
| 18 | `backprop_shared_biases_chunk` | `task_backprop_shared_biases` | Equivalent |
| 19 | `clip_shared_gradients_chunk` | `task_clip_shared_grads` | Equivalent |
| 21 | `normalize_gradients` | `task_normalize_gradients` | Equivalent |
| 24 | `adam_update` | `task_adam_update` | Equivalent |
| 25 | `clamp_temperatures` | `task_clamp_temperatures` | Equivalent |

---

## Finding 1: BCE Loss Epsilon Handling — BUG (Contract Violation)

### Description

The CPU and OpenCL BCE kernels (Node 7) use different numerical stability strategies for the `log` arguments in the binary cross-entropy loss computation.

**OpenCL** (`kernels/phase_1_act.cl.c`, lines 386–389) uses **clamping**:

```c
const SCALAR_TYPE term1 = target_val * MATH_FN log(fmax(prob, (SCALAR_TYPE)NUMERICAL_STABILITY_EPSILON));
const SCALAR_TYPE term2 = (1.0f - target_val) * MATH_FN log(fmax(1.0f - prob, (SCALAR_TYPE)NUMERICAL_STABILITY_EPSILON));
```

**CPU** (`src/backends/cpu/kernel_sources/phase_2_learn_A_production.c`, lines 159–160) uses **additive shift**:

```c
float log_p   = logf(prob + NUMERICAL_STABILITY_EPSILON);
float log_1mp = logf(1.0f - prob + NUMERICAL_STABILITY_EPSILON);
```

### Analysis

- `fmax(p, eps)` clamps the argument to a minimum of `eps`. For `p > eps`, the value passes through unmodified.
- `p + eps` always shifts the argument upward by `eps`, introducing a systematic bias across all probability values.

The `fmax` approach is the one specified in `kernels.cl.h` (the authoritative contract under ADR-013). The CCE kernel (Node 6) consistently uses `fmax` in both backends. The CPU BCE kernel is the only kernel that deviates.

### Impact

For typical probabilities in `(0.01, 0.99)`, the divergence is negligible. For extreme probabilities near 0 or 1 (where this epsilon handling is designed to matter), the results differ slightly. More importantly, this is a contract violation: the CPU must implement the same algorithm as specified in the OpenCL reference.

### Verdict

**Bug.** Fix the CPU kernel to use `fmaxf`:

```c
float log_p   = logf(fmaxf(prob, NUMERICAL_STABILITY_EPSILON));
float log_1mp = logf(fmaxf(1.0f - prob, NUMERICAL_STABILITY_EPSILON));
```

---

## Finding 2: Reduction Tree Clip Granularity — RESOLVED (ADR-019)

### Description

The OpenCL and CPU backends originally implemented multi-stage reduction trees with fundamentally different architectures, leading to different mathematical results when `num_stages > 1` and `tree_variant == "sum_and_clip"`.

### Original Problem

The OpenCL renderer dispatched `aggregate_register_reduce` or `aggregate_local_reduce` at each stage. These kernels reduce **all** current partials into a **single** output vector — they contain no K-fan-in primitive. The renderer's loop computed `output_N = ceil(current_N / K)` and constructed contiguous offsets for subsequent stages, but the aggregate kernel already produced a single output, not `output_N` outputs. For `num_stages > 1`, subsequent stages referenced uninitialized memory (silent data corruption).

The CPU's `execute_reduction_tree` correctly implemented genuine multi-stage K-fan-in with per-node L2 clip via `task_reduce_and_clip_node`.

### Resolution

**ADR-019** (K-Fan-In Reduction Kernel Primitive) resolved this by:

1. **New kernel:** `reduce_k_fan_in_and_clip` added to `kernels/phase_2_learn_C_reduction.cl.c` and specified in `kernels/kernels.cl.h`. One work-group per reduction node. Each node gathers K partials via a flat offset list (with sentinel `0xFFFFFFFF` for absent tail-node partials), sums them, computes a per-node L2 norm, and conditionally clips. Negative threshold bypasses clip for diagnostic trees.

2. **Renderer split:** `_render_reduction_tree` in `src/backends/opencl/renderer.py` now dispatches:
   - **Single-stage trees** (`num_stages == 1`): existing `aggregate_*` + `clip_intermediate_grad` path (unchanged, correct, optimal).
   - **Multi-stage trees** (`num_stages > 1`): `reduce_k_fan_in_and_clip` at each stage with sentinel-padded flat offset lists and ping-pong intermediate buffers.

3. **Binding:** `ReduceKFanInAndClipBinding` in `src/backends/opencl/kernel_bindings/binding_phase_2_learn_C.py`, wired via `set_reduction_bindings(k_fan_in=...)`.

4. **Python contract:** `reduce_k_fan_in_and_clip_contract` in `src/shared/kernel_contracts/phase_2_learn_C_reduction.py`.

The OpenCL `reduce_k_fan_in_and_clip` is the direct GPU counterpart of the CPU's `task_reduce_and_clip_node`. Both backends now produce mathematically identical results for multi-stage reduction trees.

### Verdict

**Resolved.** No remaining divergence. Tier 3 cross-backend parity tests for multi-stage reduction trees are unblocked.

---

## Finding 3: Forward Pass Hidden Mask Encoding — HARMLESS

### Description

The OpenCL `forward_pass` kernel writes exactly `0.0` or `1.0` to the hidden mask via `select((SCALAR_TYPE)0.0f, (SCALAR_TYPE)1.0f, activation > SCALAR_ZERO)`.

The CPU `task_forward_pass` writes `mask_val` (the sample mask value, guaranteed to be `1.0f` for valid samples) via `simd_select(v_zero, v_mask, accum)`, where `v_mask = simd_set1(mask_val)`.

Additionally, the CPU multiplies activations by `mask_val` (`activated = simd_mul(activated, v_mask)`), which is a redundant multiply-by-1.0 for valid samples. The OpenCL does not perform this multiplication.

### Analysis

The `kernels.cl.h` contract for `dest_buffer_GLOBAL_hidden_mask` states: *"Derived mask from ReLU operation (1 if activation > 0, else 0)."* The sample mask contract guarantees values are either 0 or 1.

Under the binary mask contract, writing `mask_val` (which is `1.0f`) is identical to writing `1.0f`. The only downstream consumer is `render_logits_chunk`, which reads `h_mask > 0.5f` — a threshold comparison that treats any value > 0.5 as "active".

The redundant multiply is a no-op for valid samples (multiply by 1.0) and padded samples already return early before reaching this code.

### Verdict

**Not a bug.** The CPU implementation is correct under the contract. No action needed. The `simd_select(v_zero, v_mask, accum)` idiom is a natural SIMD pattern. The multiply is harmless dead work.

---

## Finding 4: Vestigial `batch_chunk_offset`/`batch_chunk_count` in Node 8 — CONTRACT INCONSISTENCY

### Description

Both the OpenCL `calculate_module_param_grads_chunk` and the CPU `task_module_param_grads` accept `batch_chunk_offset` and `batch_chunk_count` parameters but neither uses them. Both iterate over `total_batch_count` via strided or sequential loops.

### Analysis

The `kernels.cl.h` contract for Node 8 explicitly references these params in validation preconditions: *"The batch access slice must be within bounds, as proven by: (batch_chunk_offset + batch_chunk_count) <= total_batch_count."* The OpenCL binding (`src/backends/opencl/kernel_bindings/phase_2_learn_A_production.py`) passes them through to the kernel.

However:
- The OpenCL kernel's batch loop uses `lid` and `lsize` striding over `total_batch_count`, not the chunk subset.
- The CPU kernel's batch loop iterates `0..total_batch_count`, not the chunk subset.
- The host always passes `(0, total_batch_count)`, making the precondition trivially satisfied.

This was likely designed for a future batch-streamed module gradient computation that was never implemented. The contract, OpenCL binding, and CPU struct all carry the parameters, but no implementation reads them.

### Verdict

**Cosmetic inconsistency.** Not a correctness issue — the host always passes values that make the precondition vacuously true. Should be cleaned up in a future contract revision to remove the misleading parameters and preconditions, or the implementations should be updated to actually use batch chunking.

---

## Finding 5: CPU BCE Supports Optional Loss Skip — HARMLESS ASYMMETRY

### Description

The CPU BCE kernel (`task_bce_probs_loss`) guards loss writes with `if (a->partial_loss != NULL)`, allowing the caller to skip loss computation by passing a NULL pointer. The OpenCL BCE kernel unconditionally writes to `dest_buffer_GLOBAL_partial_loss`.

### Analysis

The BCE strategy always allocates a `partial_loss` buffer. This is confirmed by:
- `test_integration_dag_orchestration.py`: *"With BCE, the host must plan a reduction tree for partial_loss"* and asserts `"partial_loss" in layouts`.
- `test_execution_plan_strategy.py`: CCE strategy pops `partial_loss_out_ref`; BCE strategy keeps it.

In all actual code paths, `partial_loss` is always non-NULL for BCE. The CPU's NULL guard is purely defensive — the OpenCL buffer system guarantees the buffer exists when bound.

### Verdict

**Not a bug.** The CPU has a superset of the OpenCL's capability. The NULL guard is harmless defensive coding. No action needed.

---

## Finding 6: `AGG_MODE_AVERAGE` Never Exercised — LATENT CAPABILITY

### Description

The OpenCL `aggregate_register_reduce` and `aggregate_local_reduce` kernels support `AGG_MODE_AVERAGE` (divides the sum by the number of partials). The CPU `execute_reduction_tree` always sums and never averages.

### Analysis

Every invocation in the entire codebase passes `operation_type=np.uint32(0)` (`AGG_MODE_SUM`):
- `src/backends/opencl/compute_patterns.py` — always SUM
- All benchmark and test files — always SUM
- The plan builder's `_build_reduction_tree` does not expose an average option

The kernel contract in `src/shared/kernel_contracts/phase_2_learn_C_reduction.py` documents it as a behavioral invariant (both SUM and AVERAGE are contractually valid), but no host path selects AVERAGE.

### Verdict

**Non-issue.** Latent, never-exercised capability in the OpenCL aggregate kernels. The CPU not supporting it is acceptable because no host code path requests it. If average-mode aggregation is ever needed, it would be a new feature requiring implementation in both backends.

---

## Bonus Finding: Tier 2 Reference Fixture Epsilon Placement Mismatch

### Description

The numpy reference implementation for L2-norm clipping (`tests/tier2/fixtures/analytical.py`, `ref_clip_l2_norm`) places epsilon **inside** the `sqrt`:

```python
norm = np.sqrt(np.sum(grads * grads) + epsilon)
```

Both the OpenCL and CPU kernels place epsilon **outside** the `sqrt`:

```c
norm = sqrt(sum_sq);
scale = threshold / (norm + epsilon);
```

### Analysis

These are mathematically different:
- Reference: `threshold / sqrt(||g||² + ε)`
- Kernels: `threshold / (sqrt(||g||²) + ε)` = `threshold / (||g|| + ε)`

For large norms (where clipping fires), the difference is negligible because `ε` is dominated by `||g||`. For small norms near zero (where clipping doesn't fire), the comparison `norm > threshold` produces the same boolean result either way.

However, the reference fixture doesn't match the kernels' contract. If Tier 2 tolerance tables are very tight, this could cause spurious test failures for edge-case gradient magnitudes near the clipping threshold.

### Verdict

**Test fixture bug.** The reference should match the kernel contract. Fix:

```python
norm = np.sqrt(np.sum(grads * grads))
if norm > threshold:
    return grads * (threshold / (norm + epsilon))
return grads.copy()
```

---

## Summary

| # | Finding | Severity | Action Required |
|---|---------|----------|----------------|
| 1 | BCE epsilon: `prob + eps` vs `fmax(prob, eps)` | **Bug** | Fix CPU to use `fmaxf` |
| 2 | Reduction tree: per-node clip vs global clip | **Resolved** | None (ADR-019 implemented) |
| 3 | Hidden mask: writes `mask_val` vs `1.0f` | Harmless | None |
| 4 | Vestigial `batch_chunk_offset/count` in Node 8 | Cosmetic | Clean up in future contract revision |
| 5 | CPU BCE `partial_loss` NULL guard | Harmless | None |
| 6 | `AGG_MODE_AVERAGE` never used | Latent | None |
| Bonus | Reference fixture epsilon placement | **Test bug** | Fix `ref_clip_l2_norm` |

### ADR-017 Phase Implications

- **Finding 1** blocks Phase 3 (CPU Backend) Tier 2 gate if BCE tolerance tables are tight enough to detect the divergence. Fix before gate evaluation.
- **Finding 2** ~~blocks Phase 6 (Legacy Removal) Tier 3 parity gate for large models.~~ **Resolved** by ADR-019. Tier 3 cross-backend parity tests for multi-stage reduction trees are unblocked.
- **Bonus finding** affects Phase 4 (Test Harness) — the reference fixture should be corrected before Tier 2 tests are treated as authoritative.
