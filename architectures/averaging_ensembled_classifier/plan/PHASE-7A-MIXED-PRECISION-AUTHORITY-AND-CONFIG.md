# Phase 7A: Mixed Precision — Authority Documents & Shared Configuration

**Status: ✅ COMPLETED**  
**Phase:** 7A of 7  
**Objective:** Amend the authority documents (CONCEPT.md, CONTRACT.md), redesign `PrecisionConfig` with three roles, and cascade the role-aware types through the shared configuration, plan, and host layers. No kernel sources or backend-native code are touched. The phase ends when all existing tests pass against the new `PrecisionConfig` interface and the `compute_fp_format_max` rename is live throughout.  
**Governing ADRs:** ADR-020 (authority amendments §§2–3; design-layer §4.1–4.3, 4.7), ADR-022 (§§1–6, §§3–4, §9.1–9.2)  
**Rollback gate:** All tests that passed before Phase 7A must pass after. The new `PrecisionConfig.float32()` factory must be a drop-in replacement at the call level (same method name; field names differ internally). Failing any existing Tier 2 test is a blocking regression.  
**Dependencies:** Phase 6 (legacy removal) complete, or the repository is in the post-Phase-6 state.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 7A.1: Amend CONCEPT.md](#step-7a1-amend-conceptmd)
   - [Step 7A.2: Amend CONTRACT.md](#step-7a2-amend-contractmd)
   - [Step 7A.3: Redesign `PrecisionConfig`](#step-7a3-redesign-precisionconfig)
   - [Step 7A.4: Rename `StabilizationPolicy.fp_format_max`](#step-7a4-rename-stabilizationpolicyfp_format_max)
   - [Step 7A.5: Update `ModelSpec` padding properties](#step-7a5-update-modelspec-padding-properties)
   - [Step 7A.6: Add `precision_role` to `BufferDescriptor`](#step-7a6-add-precision_role-to-bufferdescriptor)
   - [Step 7A.7: Add `precision_role` to `BufferParamSpec`](#step-7a7-add-precision_role-to-bufferparamspec)
   - [Step 7A.8: Fix `LocalMemorySpec` `sizeof` type references](#step-7a8-fix-localmemoryspec-sizeof-type-references)
   - [Step 7A.9: Update `main_orchestrator.py`](#step-7a9-update-main_orchestratorpy)
   - [Step 7A.10: Update test fixtures](#step-7a10-update-test-fixtures)
   - [Step 7A.11: Validate rollback gate](#step-7a11-validate-rollback-gate)
4. [Field-Rename Reference](#4-field-rename-reference)
5. [Migration Order Rationale](#5-migration-order-rationale)
6. [Risk Register](#6-risk-register)

---

## 1. Scope & Constraints

### In scope

- Amending `CONCEPT.md` with the six precision-role additions specified in ADR-020 §§2.1–2.6.
- Amending `CONTRACT.md` with the four precision-role additions specified in ADR-020 §§3.1–3.6.
- Replacing the single-axis `PrecisionConfig` dataclass (`numpy_dtype`, `fp_format_max`, `epsilon`) with the three-role design (`storage_dtype`, `compute_dtype`, `state_dtype`, `storage_fp_format_max`, `compute_fp_format_max`, `compute_epsilon`) plus three factory classmethods: `float32()`, `float16()`, `mixed_f16_f32()`.
- Renaming `StabilizationPolicy.fp_format_max` → `compute_fp_format_max` and updating its docstring.
- Updating `ModelSpec`'s three cache-line-aligned padding properties to use `storage_dtype.itemsize`. Removing the deprecated `SCALAR_NP_TYPE` and `SCALAR_C_TYPE_NAME` properties.
- Adding the mandatory `precision_role: Literal["storage", "compute", "state"]` field to `BufferDescriptor`.
- Adding the mandatory `precision_role: Literal["storage", "compute", "state"]` field to every `BufferParamSpec` across all six kernel-contract modules.
- Replacing `sizeof(SCALAR_TYPE)` with `sizeof(COMPUTE_TYPE)` in every `LocalMemorySpec` string across the six kernel-contract modules.
- Updating `main_orchestrator.py` to construct `StabilizationPolicy` from `PrecisionConfig.compute_fp_format_max` directly, removing the `np.finfo` re-derivation.
- Updating test fixtures in `test_stabilization_policy.py` and `test_reduction_logic.py` to use the `compute_fp_format_max` keyword.

### Out of scope

- Any change to `kernels/kernels.cl.h` or `kernels/phase_*.cl.c` — deferred to Phase 7B.
- Any change to `src/backends/opencl/type_mapping.py` — deferred to Phase 7B.
- Any change to `src/shared/plan_builder.py` `BufferDescriptor` construction call sites — deferred to Phase 7B (depends on Phase 7A's `BufferDescriptor` field being added, but the wiring into the plan builder requires the kernel-contract `precision_role` fields which are established here; the _populating_ of the role argument at every call-site is 7B work).
- Any change to CPU or Vulkan backend files — deferred to Phase 7C.
- `PrecisionConfig.mixed_f16_f32()` multi-configuration test coverage — deferred to Phase 7C (the kernel infrastructure to support it is not yet in place).
- FP8 `PrecisionConfig` factories — explicitly out-of-scope pending ADR-020 §5 preconditions.

### Key constraint: breaking API change is centralised

`PrecisionConfig`'s three-field redesign is the root change for the entire mixed-precision migration. All other implications cascade from it. The **old fields (`numpy_dtype`, `fp_format_max`, `epsilon`) are removed outright — no shim properties**. ADR-022 §1 states this explicitly. Every consumer in `src/` is updated in this phase or a subsequent one. Partial migration where some consumers still reference `precision.numpy_dtype` will produce `AttributeError` at runtime — this is a self-revealing failure mode that tests will catch.

The `BufferDescriptor.precision_role` and `BufferParamSpec.precision_role` fields are added here as mandatory fields. Construction sites in `plan_builder.py` and the Vulkan renderer that don't yet supply the role argument will fail with `TypeError` — again self-revealing. The plan builder call-site plumbing is done in Phase 7B once the kernel-contract `precision_role` tables are live.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/shared/precision_config.py` | `PrecisionConfig` frozen dataclass: fields `numpy_dtype`, `fp_format_max`, `epsilon`. Two factories: `float32()`, `float16()`. |
| `src/shared/stabilization_policy.py` | `StabilizationPolicy` field: `fp_format_max: float`. Used in Quadratic Scaling Policy safety ceiling. |
| `src/shared/model_spec.py` | `padded_input_dim`, `padded_class_dim`, `padded_module_dim` use `np.dtype(self.precision.numpy_dtype).itemsize`. Properties `SCALAR_NP_TYPE` and `SCALAR_C_TYPE_NAME` delegate to `precision.numpy_dtype`. |
| `src/shared/buffer_lifecycle.py` | `BufferDescriptor` frozen dataclass: no `precision_role` field. |
| `src/shared/kernel_contracts/phase_*.py` | Six files; `BufferParamSpec` objects have no `precision_role` field. `LocalMemorySpec` objects contain `sizeof(SCALAR_TYPE)`. |
| `src/main_orchestrator.py` | Constructs `StabilizationPolicy(fp_format_max=float(np.finfo(model_spec.precision.numpy_dtype).max), ...)`. |
| `src/tests/test_stabilization_policy.py` | Constructs `StabilizationPolicy(..., fp_format_max=...)`. |
| `src/tests/test_reduction_logic.py` | Constructs `StabilizationPolicy(..., fp_format_max=...)`. |

---

## 3. Task Breakdown

---

### Step 7A.1: Amend `CONCEPT.md`

**Governing authority:** ADR-020 §§2.1–2.6  
**File:** `CONCEPT.md` (architecture root)

Six discrete amendments. Each must be inserted precisely at the target section to preserve document authority. Work through them in section order.

#### Amendment 2.1 — Append to Principle §2 (Primacy of Memory Strategy)

Append the following paragraph at the end of the §2 body:

> **Storage precision is a first-class bandwidth lever.** The system decomposes numeric precision into three independent roles — storage, compute, and state — reflecting the distinct optimization targets of bandwidth, arithmetic fidelity, and long-term stability. Narrowing the storage format reduces buffer sizes and transfer bandwidth proportionally (FP16 halves FP32's footprint; FP8 halves again) without constraining the arithmetic precision used for computation or the precision of persistent optimizer state. This decoupling is a direct expression of the Primacy of Memory Strategy: data is stored in the narrowest format that preserves sufficient information for the downstream operation, and widened to the arithmetic format only at the point of computation.

#### Amendment 2.2 — Replace in §3.4 (Orthogonal Enforcement of Numerical Safety)

Replace the safety ceiling formula block:

> **`T_safety_j = FP_FORMAT_MAX / K_j`**  
> **`FP_FORMAT_MAX`**: A conservative, high value representing the maximum representable number for the current precision...

with:

> **`T_safety_j = COMPUTE_FP_FORMAT_MAX / K_j`**  
> **`COMPUTE_FP_FORMAT_MAX`**: The maximum representable value of the **compute precision** format (e.g., `~3.4e38` for FP32 compute, `~6.5e4` for FP16 compute). The safety ceiling reflects the arithmetic precision that performs the summation, not the storage precision of the values being summed. In a configuration where all precision roles share a format, `COMPUTE_FP_FORMAT_MAX` equals that format's maximum.

#### Amendment 2.3 — Append to §7 (`KernelContract` description)

Append to the `KernelContract` description:

> Each buffer parameter in a `KernelContract` carries a `precision_role` declaration — one of `"storage"`, `"compute"`, or `"state"` — that identifies the buffer's element type under the active precision configuration. When a kernel reads a buffer whose precision role differs from `"compute"`, the kernel widens the value to compute precision before any arithmetic. When writing, the kernel narrows from compute precision to the buffer's role precision. These conversions are declared in the contract's `Behavioral Invariants` as a mandatory `Precision Boundary Conversion` invariant. The conversion pattern is structurally identical regardless of whether the role types happen to be equal — the kernel source has one codepath, and the compiler eliminates identity conversions.

#### Amendment 2.4 — Extend `BufferDescriptor` sentence in §8 (Buffer Lifecycle)

Extend the `BufferDescriptor` introductory sentence:

> Every device-side buffer's `BufferDescriptor` carries a `precision_role` field — one of `"storage"`, `"compute"`, or `"state"` — that determines the buffer's element type and allocation size under the active `PrecisionConfig`. The `element_size_bytes` and `size_bytes` fields derive from the role-appropriate dtype. A storage-role activation buffer allocated at FP16 is half the size of a compute-role accumulation buffer at FP32, enabling the bandwidth benefits of narrow formats without affecting the fidelity of arithmetic operations.

#### Amendment 2.5 — Replace in §11 (Host Orchestrator, Item 7)

Replace item 7:

> "Precision is uniformly configured through a `PrecisionConfig` frozen dataclass carrying `numpy_dtype`, `fp_format_max`, and `epsilon`."

with:

> Precision is configured through a `PrecisionConfig` frozen dataclass carrying three independent role assignments: `storage_dtype` (bandwidth optimization), `compute_dtype` (arithmetic fidelity), and `state_dtype` (optimizer stability), along with derived constants `storage_fp_format_max`, `compute_fp_format_max`, and `compute_epsilon`. A configuration where all three roles share a type (e.g., `PrecisionConfig.float32()`) is a parameterization — not a distinct mode. The host's existing practice of computing Adam bias correction terms in FP64 is recognized as an instance of state-precision independence; the three-role model formalizes this pattern as a first-class concern rather than an ad-hoc exception.

#### Amendment 2.6 — Add Validation Scenario: The Alchemist

Add to the *Validation Scenarios* section:

> **Scenario: The Alchemist (Mixed-Precision Fidelity Validation)**
>
> - **Description:** The same training task is executed in three precision configurations: `PrecisionConfig.float32()`, `PrecisionConfig.float16()`, and `PrecisionConfig.mixed_f16_f32()`. All three use the same `PrecisionConfig` with three roles — there is no mode switch.
> - **Validation Focus:** Confirms that the mixed configuration produces loss curves and final parameters tracking the FP32 baseline within tolerance, while achieving memory footprint comparable to the all-FP16 configuration. Validates that buffer allocation sizes reflect `storage_dtype`, that accumulation uses `compute_dtype`, that the stabilization policy's safety ceiling derives from `compute_fp_format_max`, and that optimizer state maintains `state_dtype` fidelity.
> - **Key Insight:** Proves the three-role decomposition achieves its stated goal: narrow-storage bandwidth without narrow-compute fidelity loss.

**Verification:** After editing, confirm the amended sections are internally consistent and that all cross-references within CONCEPT.md that mention `FP_FORMAT_MAX`, `numpy_dtype`, or `SCALAR_TYPE` (in a host-layer context) have been updated.

---

### Step 7A.2: Amend `CONTRACT.md`

**Governing authority:** ADR-020 §§3.1–3.6  
**File:** `CONTRACT.md` (architecture root)

Four amendments in section order.

#### Amendment 3.1 — Note to Article 2.1 (Buffer Name Grammar)

The buffer name grammar itself is unchanged. Append the following note to Article 2.1:

> A buffer's precision role is determined by its C type declaration (`STORAGE_TYPE`, `COMPUTE_TYPE`, or `STATE_TYPE`) and formally specified in the parameter's `Precision Role` commentary key (Article 3). It is not encoded in the buffer name's `[ContextAndUsage]` component.
>
> **Rationale (Article 1.2 — Axiom of Semantic Uniqueness):** The C declaration encodes the precision role via the type symbol. Duplicating this in the name would violate the prohibition on cross-jurisdictional redundancy.

#### Amendment 3.2 — New mandatory commentary key in Article 3

Add a new row to the mandatory key table, positioned after `Padding Contract` and before `Calculability Proof`:

| Key | Definition | Status |
|:---|:---|:---|
| **`Precision Role`** | One of `"storage"`, `"compute"`, or `"state"`. Declares which precision-role dtype from the active `PrecisionConfig` governs this buffer's element type and allocation size. | **Mandatory** for all buffer parameters |

#### Amendment 3.3 — Article 3 padding note (role-typed element sizes)

Add a note to the `Padding Contract` key definition:

> Under the three-role precision model, padding byte counts use the element size of the buffer's actual `precision_role` type: `sizeof(STORAGE_TYPE)`, `sizeof(COMPUTE_TYPE)`, or `sizeof(STATE_TYPE)` as appropriate. Using a different role's `sizeof` in a `Padding Contract` expression is a contract violation.

#### Amendment 3.4 — New behavioral invariant in Article 4.2

Add `Precision Boundary Conversion` to the list of recognized `Behavioral Invariants`:

> **`Precision Boundary Conversion`** — Required for any kernel that accesses buffers whose `precision_role` is `"storage"` or `"state"`. Declares that storage-role and state-role inputs are widened to `COMPUTE_TYPE` upon load, and that outputs to storage-role or state-role buffers are narrowed from `COMPUTE_TYPE` upon store. All intermediate arithmetic is exclusively `COMPUTE_TYPE`. Kernels that access only `"compute"`-role and integer buffers do not require this invariant.

#### Amendment 3.5 — New `PrecisionConfig` contract paragraph in Article 5

In the host-side configuration section, replace the paragraph that describes `PrecisionConfig` as a single-dtype object with:

> `PrecisionConfig` is a frozen dataclass with three independent dtype fields — `storage_dtype`, `compute_dtype`, `state_dtype` — and three derived scalar constants: `storage_fp_format_max`, `compute_fp_format_max`, and `compute_epsilon`. The invariant `storage_dtype.itemsize ≤ compute_dtype.itemsize` and `storage_dtype.itemsize ≤ state_dtype.itemsize` is enforced by `__post_init__`. Three factory classmethods are defined: `float32()` (all FP32), `float16()` (all FP16), `mixed_f16_f32()` (FP16 storage, FP32 compute, FP32 state). The retired fields `numpy_dtype`, `fp_format_max`, and `epsilon` do not exist in this type.

#### Amendment 3.6 — Article 6 build-time symbol replacement

Replace the current `SCALAR_TYPE`-centred build symbol block with the three-role symbol set. The new mandatory build-time symbols are:

| Symbol | Type | Meaning |
|:---|:---|:---|
| `STORAGE_TYPE` | OpenCL/C type name | Element type for storage-role buffers |
| `COMPUTE_TYPE` | OpenCL/C type name | Element type for arithmetic and compute-role buffers |
| `STATE_TYPE` | OpenCL/C type name | Element type for state-role buffers |
| `STORAGE_TYPE_IS_HALF` | `int` (0 or 1) | 1 when `STORAGE_TYPE == half`; gates `cl_khr_fp16` extension and `vload_half`/`vstore_half` |
| `COMPUTE_TYPE_IS_HALF` | `int` (0 or 1) | 1 when `COMPUTE_TYPE == half`; enables FP16 arithmetic extension if required |
| `SIMD_WIDTH` | `int` | Hardware SIMD lane count from `HardwareProfile` |
| `C_TILE_SIZE` | `int` | Column tile size for the module-chunking strategy |
| `NUMERICAL_STABILITY_EPSILON` | float literal | Epsilon for numerical stability guards; derived from `compute_epsilon` |

The symbols `SCALAR_TYPE` and `SCALAR_IS_HALF` are **retired**. They must not appear in any kernel source file after Phase 7B migration is complete (transitional aliases exist during Phase 7B; see Phase 7B plan §3).

---

### Step 7A.3: Redesign `PrecisionConfig`

**Governing authority:** ADR-020 §4.1; ADR-022 §1  
**File:** `src/shared/precision_config.py`

Replace the entire file body (keeping imports). The new dataclass:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype

    storage_fp_format_max: float
    compute_fp_format_max: float
    compute_epsilon: float

    def __post_init__(self) -> None:
        assert self.storage_dtype.itemsize <= self.compute_dtype.itemsize, (
            "storage precision must not be wider than compute precision"
        )
        assert self.storage_dtype.itemsize <= self.state_dtype.itemsize, (
            "storage precision must not be wider than state precision"
        )

    @classmethod
    def float32(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float32),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=float(np.finfo(np.float32).max),
            compute_fp_format_max=float(np.finfo(np.float32).max),
            compute_epsilon=float(np.finfo(np.float32).eps),
        )

    @classmethod
    def float16(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float16),
            storage_fp_format_max=float(np.finfo(np.float16).max),
            compute_fp_format_max=float(np.finfo(np.float16).max),
            compute_epsilon=float(np.finfo(np.float16).eps),
        )

    @classmethod
    def mixed_f16_f32(cls) -> "PrecisionConfig":
        return cls(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float32),
            state_dtype=np.dtype(np.float32),
            storage_fp_format_max=float(np.finfo(np.float16).max),
            compute_fp_format_max=float(np.finfo(np.float32).max),
            compute_epsilon=float(np.finfo(np.float32).eps),
        )
```

**Postcondition checks:**
- `PrecisionConfig.float32()` produces the same `compute_fp_format_max` and `compute_epsilon` values that the old `float32()` produced as `fp_format_max` and `epsilon`.
- `PrecisionConfig.float16()` likewise.
- No field named `numpy_dtype`, `fp_format_max`, or `epsilon` exists on the new dataclass.
- The `__post_init__` assertion fires for a hypothetical `storage_dtype=float32, compute_dtype=float16` construction, confirming the invariant is enforced.

---

### Step 7A.4: Rename `StabilizationPolicy.fp_format_max`

**Governing authority:** ADR-020 §2.2; ADR-022 §3  
**File:** `src/shared/stabilization_policy.py`

- Rename field `fp_format_max: float` → `compute_fp_format_max: float`.
- Update the field docstring to read: *"The maximum representable value of the compute precision format. Governs overflow safety during reduction tree summation. Under the three-role model, the safety ceiling is bounded by arithmetic precision, not storage precision."*
- The formula using this field (`T_safety_j = compute_fp_format_max / K_j`) is unchanged structurally. Confirm the field reference in the computation is updated to the new name.

There are no other callers of this field beyond `main_orchestrator.py` and the two test files updated in Steps 7A.9–7A.10.

---

### Step 7A.5: Update `ModelSpec` padding properties

**Governing authority:** ADR-020 §3.3; ADR-022 §2  
**File:** `src/shared/model_spec.py`

#### §5.1: Cache-line-aligned padding properties

Three properties currently compute byte-level alignment using `np.dtype(self.precision.numpy_dtype).itemsize`. Replace with `self.precision.storage_dtype.itemsize` in each:

| Property | Change |
|:---|:---|
| `padded_input_dim` | `np.dtype(self.precision.numpy_dtype).itemsize` → `self.precision.storage_dtype.itemsize` |
| `padded_class_dim` | Same |
| `padded_module_dim` | Same |

`padded_hidden_dim` operates in element-count space (dividing by `simd_width`), never touching `itemsize` — **leave unchanged**.

#### §5.2: Remove deprecated properties

Delete `SCALAR_NP_TYPE` and `SCALAR_C_TYPE_NAME` entirely. These properties have no valid role under the three-role model and no `precision_role`-aware replacement can be expressed as a single type. Any surviving caller in the source tree will fail with `AttributeError` at runtime (treat as a self-revealing regression).

Before removing, grep for all callers:

```bash
grep -rn "SCALAR_NP_TYPE\|SCALAR_C_TYPE_NAME" \
  architectures/averaging_ensembled_classifier/src/ \
  architectures/averaging_ensembled_classifier/tests/
```

If callers exist outside the files explicitly updated in this phase, add them to the risk register and plan their migration before committing this step.

---

### Step 7A.6: Add `precision_role` to `BufferDescriptor`

**Governing authority:** ADR-020 §4.2; ADR-022 §5  
**File:** `src/shared/buffer_lifecycle.py`

Add field after the existing `role: BufferRole` field:

```python
precision_role: Literal["storage", "compute", "state"]
```

This requires `from typing import Literal` if not already imported.

**Important:** This is a mandatory field with no default. Every existing `BufferDescriptor(...)` call site will produce a `TypeError` until it supplies `precision_role=...`. The plan builder call sites are updated in Phase 7B. To prevent a total construction failure of the existing test suite, two options exist:

**Option A (preferred):** Set a temporary default `precision_role: Literal["storage", "compute", "state"] = "compute"` in this step, then remove the default in Phase 7B when all call sites are updated. This allows the existing tests to continue running without modification.

**Option B:** Update all `BufferDescriptor` construction sites in the same commit. This is only feasible if `plan_builder.py` and `renderer.py` are also within scope of this step — they are in 7B, so Option A is the correct choice here.

Apply **Option A**: add the field with a temporary `= "compute"` default. Document this as a transitional default that will be removed in Step 7B.6.

---

### Step 7A.7: Add `precision_role` to `BufferParamSpec`

**Governing authority:** ADR-020 §4.3; ADR-022 §6.1  
**File:** `src/shared/kernel_contracts/phase_*.py` (6 files)

Add field to `BufferParamSpec`:

```python
precision_role: Literal["storage", "compute", "state"]
```

As with `BufferDescriptor`, this field is mandatory. Every existing `BufferParamSpec(...)` construction in the six kernel-contract files must be updated in the same commit to supply the correct role. The per-kernel role assignment tables are given in ADR-021 §1.4, reproduced in the reference table below.

Apply the role values to every `BufferParamSpec` in each contract file. All six files are updated in this step. Do **not** defer any file to Phase 7B — a partially-populated contract corpus is invalid.

Use a temporary `= "compute"` default on `BufferParamSpec.precision_role` (matching the approach for `BufferDescriptor`) so that the contract files can be migrated incrementally without Python import failures if a file is read during partial execution.

The role reference tables (from ADR-021 §1.4) are:

**`phase_1_act.py`**
| Buffer parameter | `precision_role` |
|:---|:---|
| `update_buffer_LOCAL_simd_tile` | `"compute"` |
| `src_buffer_GLOBAL_input` | `"storage"` |
| `src_buffer_GLOBAL_sample_mask` | `"storage"` |
| `src_buffer_GLOBAL_CONST_weights_shared_simd_major` | `"state"` |
| `src_buffer_GLOBAL_CONST_biases_shared` | `"state"` |
| `dest_buffer_GLOBAL_hidden_activations` | `"storage"` |
| `dest_buffer_GLOBAL_hidden_mask` | `"storage"` |

**`phase_1_act.py` (render_logits_chunk)**
| Buffer parameter | `precision_role` |
|:---|:---|
| `src_buffer_GLOBAL_hidden_activations` | `"storage"` |
| `src_buffer_GLOBAL_hidden_mask` | `"storage"` |
| `src_buffer_GLOBAL_CONST_weights_module` | `"state"` |
| `src_buffer_GLOBAL_CONST_biases_module` | `"state"` |
| `dest_buffer_GLOBAL_logits` | `"storage"` |

**`phase_1_act.py` (compute_probs_loss_cce_chunk)**
| Buffer parameter | `precision_role` |
|:---|:---|
| `src_buffer_GLOBAL_logits` | `"storage"` |
| `src_buffer_GLOBAL_CONST_temps` | `"state"` |
| `src_buffer_GLOBAL_targets` | *(integer — omit field or set a sentinel)* |
| `src_buffer_GLOBAL_sample_mask` | `"storage"` |
| `dest_buffer_GLOBAL_partial_probs` | `"storage"` |
| `dest_buffer_GLOBAL_final_loss` | `"compute"` |

> Note on `targets` (CCE): `targets` is `int`-typed and carries no floating-point precision role. If `BufferParamSpec` is used uniformly for all parameters, set `precision_role = None` with `Optional[Literal[...]]` typing, or use a separate `IntBufferParamSpec` type. The simplest approach for this migration is `Optional` with `None` for integer buffers. Confirm the plan-validation layer handles `None` gracefully (skip precision-role matching for integer params).

For BCE's `targets` (float multi-hot tensor): role is `"storage"`.

**`phase_2_learn_A_production.py`**
| Buffer parameter | `precision_role` |
|:---|:---|
| `update_buffer_LOCAL_reduction_tile` | `"compute"` |
| `src_buffer_GLOBAL_hidden_activations` | `"storage"` |
| `src_buffer_GLOBAL_partial_probs` | `"storage"` |
| `dest_buffer_GLOBAL_partial_grad_weights_module` | `"storage"` |
| `dest_buffer_GLOBAL_partial_grad_biases_module` | `"storage"` |
| *(backprop_error_to_hidden_chunk `src_buffer_GLOBAL_partial_probs`)* | `"storage"` |
| *(backprop_error_to_hidden_chunk `src_buffer_GLOBAL_CONST_weights_module`)* | `"state"` |
| *(backprop_error_to_hidden_chunk `dest_buffer_GLOBAL_partial_grad_hidden_activations`)* | `"storage"` |
| *(calculate_chunk_temp_gradients all grad inputs)* | `"storage"` |
| *(calculate_chunk_temp_gradients `src_buffer_GLOBAL_CONST_temps`)* | `"state"` |
| *(calculate_chunk_temp_gradients `dest_buffer_GLOBAL_partial_grad_temps`)* | `"storage"` |

**`phase_2_learn_B_processing.py`**
| Buffer parameter | `precision_role` |
|:---|:---|
| `update_buffer_GLOBAL_partial_grads` (clip) | `"storage"` |
| `src_buffer_GLOBAL_partial_grad_hidden_activations` (gather) | `"storage"` |
| `dest_buffer_GLOBAL_gathered_grad_hidden_activations` (gather) | `"storage"` |

**`phase_2_learn_C_reduction.py`**
| Buffer parameter | `precision_role` |
|:---|:---|
| `update_buffer_LOCAL_*` scratch | `"compute"` |
| `src_buffer_GLOBAL_*` (partial inputs, pre-final) | `"storage"` |
| `dest_buffer_GLOBAL_*` (final reduction output) | `"compute"` |
| intermediate reduction output (not final) | `"storage"` |

**`phase_2_learn_D_backprop.py`**
| Buffer parameter | `precision_role` |
|:---|:---|
| `update_buffer_LOCAL_reduction_tile` | `"compute"` |
| `src_buffer_GLOBAL_grad_hidden_activations` | `"compute"` |
| `src_buffer_GLOBAL_input` | `"storage"` |
| `src_buffer_GLOBAL_hidden_mask` | `"storage"` |
| `dest_buffer_GLOBAL_partial_grad_weights_shared` | `"storage"` |
| `dest_buffer_GLOBAL_partial_grad_biases_shared` | `"storage"` |
| `update_buffer_GLOBAL_partial_grad_shared` (clip) | `"storage"` |

**`phase_3_update.py`**
| Buffer parameter | `precision_role` |
|:---|:---|
| `update_buffer_GLOBAL_summed_gradient` (normalize) | `"compute"` |
| `src_buffer_GLOBAL_normalized_gradient` (adam) | `"compute"` |
| `update_buffer_GLOBAL_m1` | `"state"` |
| `update_buffer_GLOBAL_m2` | `"state"` |
| `update_buffer_GLOBAL_weights_or_biases` | `"state"` |
| `update_buffer_GLOBAL_temps` (clamp) | `"state"` |

---

### Step 7A.8: Fix `LocalMemorySpec` `sizeof` type references

**Governing authority:** ADR-021 §1.4 (local scratch invariant); ADR-022 §6.2  
**Files:** `src/shared/kernel_contracts/phase_*.py` (6 files)

Every `LocalMemorySpec` string that contains `sizeof(SCALAR_TYPE)` is updated to `sizeof(COMPUTE_TYPE)`. Counts per file (from ADR-022 §6.2):

| File | `sizeof(SCALAR_TYPE)` occurrences to replace |
|:---|:---|
| `phase_1_act.py` | 1 |
| `phase_2_learn_A_production.py` | 1 |
| `phase_2_learn_B_processing.py` | 2 |
| `phase_2_learn_C_reduction.py` | 4 |
| `phase_2_learn_D_backprop.py` | 3 |
| `phase_3_update.py` | 0 — no change needed |

Verify with grep after the edit:

```bash
grep -rn "sizeof(SCALAR_TYPE)" \
  architectures/averaging_ensembled_classifier/src/shared/kernel_contracts/
```

Expected: zero results.

---

### Step 7A.9: Update `main_orchestrator.py`

**Governing authority:** ADR-022 §4  
**File:** `src/main_orchestrator.py`

Replace the `StabilizationPolicy` construction block. Current form:

```python
fp_max = float(np.finfo(model_spec.precision.numpy_dtype).max)
self.stabilization_policy = StabilizationPolicy(
    t_algorithmic=hyperparams.stabilization.max_grad_norm,
    lambda_=hyperparams.stabilization.lambda_,
    fp_format_max=fp_max,
)
```

New form:

```python
self.stabilization_policy = StabilizationPolicy(
    t_algorithmic=hyperparams.stabilization.max_grad_norm,
    lambda_=hyperparams.stabilization.lambda_,
    compute_fp_format_max=model_spec.precision.compute_fp_format_max,
)
```

Remove the `fp_max` local variable and any `np.finfo` import that becomes unused after this change.

---

### Step 7A.10: Update test fixtures

**Governing authority:** ADR-022 §9.1, §9.2  
**Files:** `src/tests/test_stabilization_policy.py`, `src/tests/test_reduction_logic.py`

In both files, replace every keyword argument `fp_format_max=` with `compute_fp_format_max=` in `StabilizationPolicy(...)` construction calls. The values passed (e.g. `3.4e38` for FP32) are unchanged — only the keyword name changes.

Also update any test that constructs a `PrecisionConfig` directly to use the new field names. If any test uses `PrecisionConfig(numpy_dtype=..., fp_format_max=..., epsilon=...)`, replace with either the factory method or the three-field constructor.

---

### Step 7A.11: Validate rollback gate

Run the full test suite:

```bash
cd architectures/averaging_ensembled_classifier
python -m pytest src/tests/ -x -q 2>&1 | tee /tmp/phase7a_tests.txt
grep -E "passed|failed|error" /tmp/phase7a_tests.txt
```

All tests that passed before Phase 7A must pass. No new failures are acceptable. Investigate any failure before proceeding to Phase 7B.

---

## 4. Field-Rename Reference

| Old field / symbol | New field / symbol | Location |
|:---|:---|:---|
| `PrecisionConfig.numpy_dtype` | `storage_dtype`, `compute_dtype`, `state_dtype` | `precision_config.py` |
| `PrecisionConfig.fp_format_max` | `storage_fp_format_max`, `compute_fp_format_max` | `precision_config.py` |
| `PrecisionConfig.epsilon` | `compute_epsilon` | `precision_config.py` |
| `StabilizationPolicy.fp_format_max` | `compute_fp_format_max` | `stabilization_policy.py` |
| `ModelSpec.SCALAR_NP_TYPE` | *(removed)* | `model_spec.py` |
| `ModelSpec.SCALAR_C_TYPE_NAME` | *(removed)* | `model_spec.py` |
| `BufferDescriptor.(missing)` | `precision_role` (transitional default `"compute"`) | `buffer_lifecycle.py` |
| `BufferParamSpec.(missing)` | `precision_role` (per ADR-021 §1.4 tables) | `kernel_contracts/phase_*.py` |
| `sizeof(SCALAR_TYPE)` in `LocalMemorySpec` | `sizeof(COMPUTE_TYPE)` | `kernel_contracts/phase_*.py` |

---

## 5. Migration Order Rationale

Steps are ordered to satisfy intra-phase dependencies:

1. **CONCEPT.md / CONTRACT.md** first — establishes sovereign vocabulary before any code change.
2. **`PrecisionConfig`** (7A.3) — root change; every downstream step depends on the new interface.
3. **`StabilizationPolicy`** (7A.4) — no callers beyond `main_orchestrator.py` and tests; can proceed immediately.
4. **`ModelSpec`** (7A.5) — depends on `PrecisionConfig` new field names; no plan-layer dependency.
5. **`BufferDescriptor`** (7A.6) — foundational for 7A.7 `precision_role` vocabulary and Phase 7B call sites; transitional default prevents test failures.
6. **`BufferParamSpec` + `LocalMemorySpec`** (7A.7–7A.8) — depends on `BufferParamSpec` field addition being available for population; all six contract files updated together.
7. **`main_orchestrator.py`** (7A.9) — depends on `PrecisionConfig` (7A.3) and `StabilizationPolicy` (7A.4).
8. **Tests** (7A.10) — depends on `PrecisionConfig` (7A.3) and `StabilizationPolicy` (7A.4).
9. **Gate** (7A.11) — final.

---

## 6. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| Undiscovered callers of `precision.numpy_dtype` outside explicitly enumerated files | Medium | Run `grep -rn "precision\.numpy_dtype\|\.fp_format_max\|\.epsilon" src/` before committing Step 7A.3; fix all hits. |
| Undiscovered callers of `SCALAR_NP_TYPE` or `SCALAR_C_TYPE_NAME` | Low | Run grep per Step 7A.5 instruction before deleting. |
| `BufferDescriptor` transitional default `"compute"` masks a real allocation bug | Low | The default is explicitly documented and removed in Phase 7B Step 7B.6. Tests will fail with `TypeError` if 7B.6 is missed. |
| Integer `targets` buffer in CCE contract needing `Optional` precision role | Medium | Decide `Optional[Literal["storage","compute","state"]]` vs separate type in Step 7A.7. Document the decision. The plan-validation layer must be updated to skip precision-role match for `None`. |
| `__post_init__` assertion breaks a valid existing `PrecisionConfig` construction | Very low | The only constructions are via `float32()` / `float16()` factories — both satisfy the assertion. No direct field construction exists today. |
