# Phase 7B: Mixed Precision — Kernel Specification & OpenCL Pipeline

**Status: 📋 PLANNED**  
**Phase:** 7B of 7  
**Prerequisite:** Phase 7A complete and rollback gate passed.  
**Objective:** Migrate the kernel specification document (`kernels.cl.h`) and all OpenCL implementation files (`phase_*.cl.c`) to the three-role symbol set. Implement the precision boundary abstraction bodies in `kernels.cl.h`. Update the OpenCL type-mapping and compiler-flag generation. Wire the `precision_role`-aware `BufferDescriptor` construction into the plan builder and Vulkan renderer. Inject transitional aliases for unmigrated files. The phase ends when `PrecisionConfig.float32()` produces identical Tier 2 test results to those produced before Phase 7A, with no transitional alias removed yet.  
**Governing ADRs:** ADR-021 (kernel spec migration §§1–5), ADR-023 §1 (OpenCL boundary implementations), ADR-022 §§7.1, 8 (host OpenCL type mapping; plan builder)  
**Rollback gate:** All Tier 2 tests pass against `PrecisionConfig.float32()`. Bit-for-bit identical arithmetic outcomes to pre-migration baseline. `grep -r SCALAR_TYPE kernels/` returns only the transitional alias injection in Meson (no occurrence in `.cl.c` or `.cl.h` spec sections). Compilation succeeds for all three builddir variants (`builddir`, `builddir-asan`, `builddir-vulkan`) that include the OpenCL backend.  
**Dependencies:** Phase 7A complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 7B.1: Update `kernels.cl.h` build-symbol check section](#step-7b1-update-kernelsclh-build-symbol-check-section)
   - [Step 7B.2: Update `kernels.cl.h` host/C++ stub section](#step-7b2-update-kernelsclh-hostc-stub-section)
   - [Step 7B.3: Add precision boundary abstraction declarations and OpenCL bodies](#step-7b3-add-precision-boundary-abstraction-declarations-and-opencl-bodies)
   - [Step 7B.4: Annotate every `@param` block with `Precision Role`](#step-7b4-annotate-every-param-block-with-precision-role)
   - [Step 7B.5: Add `Behavioral Invariants` `Precision Boundary Conversion` entries](#step-7b5-add-behavioral-invariants-precision-boundary-conversion-entries)
   - [Step 7B.6: Correct `Padding Contract` `sizeof` type references in `kernels.cl.h`](#step-7b6-correct-padding-contract-sizeof-type-references-in-kernelsclh)
   - [Step 7B.7: Migrate `phase_1_act.cl.c`](#step-7b7-migrate-phase_1_actclc)
   - [Step 7B.8: Migrate `phase_2_learn_A_production.cl.c`](#step-7b8-migrate-phase_2_learn_a_productionclc)
   - [Step 7B.9: Migrate `phase_2_learn_B_processing.cl.c`](#step-7b9-migrate-phase_2_learn_b_processingclc)
   - [Step 7B.10: Migrate `phase_2_learn_C_reduction.cl.c`](#step-7b10-migrate-phase_2_learn_c_reductionclc)
   - [Step 7B.11: Migrate `phase_2_learn_D_backprop.cl.c`](#step-7b11-migrate-phase_2_learn_d_backpropclc)
   - [Step 7B.12: Migrate `phase_3_update.cl.c`](#step-7b12-migrate-phase_3_updateclc)
   - [Step 7B.13: Inject transitional aliases into Meson build](#step-7b13-inject-transitional-aliases-into-meson-build)
   - [Step 7B.14: Update `src/backends/opencl/type_mapping.py`](#step-7b14-update-srcbackendsopencltype_mappingpy)
   - [Step 7B.15: Wire `precision_role` into `plan_builder.py` and Vulkan `renderer.py`](#step-7b15-wire-precision_role-into-plan_builderpy-and-vulkan-rendererpy)
   - [Step 7B.16: Remove transitional default from `BufferDescriptor` and `BufferParamSpec`](#step-7b16-remove-transitional-default-from-bufferdescriptor-and-bufferparamspec)
   - [Step 7B.17: Rebuild and validate rollback gate](#step-7b17-rebuild-and-validate-rollback-gate)
4. [`.cl.c` Migration Checklist](#4-clc-migration-checklist)
5. [Transitional Alias Protocol Details](#5-transitional-alias-protocol-details)
6. [Risk Register](#6-risk-register)

---

## 1. Scope & Constraints

### In scope

- Updating the mandatory build-symbol check block in `kernels.cl.h` to the new three-symbol set (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`, `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`) and removing the `SCALAR_TYPE`/`SCALAR_IS_HALF` checks.
- Updating the host/C++ stub definitions in `kernels.cl.h` to define all three role-type symbols, defaulting to `float`/`0` for analysis builds.
- Adding inline precision boundary function declarations (signatures only) in `kernels.cl.h` within the `__OPENCL_VERSION__` guard, and providing the full OpenCL implementations for all four functions (ADR-023 §1).
- Adding a `Precision Role` commentary key to every float buffer `@param` block in every `@kernel_contract` declaration in `kernels.cl.h`. Integer `targets` parameters receive no `Precision Role` key.
- Adding `"Precision Boundary Conversion"` to the `Behavioral Invariants` of every kernel that accesses non-compute-role buffers (per ADR-021 §1.4).
- Replacing `sizeof(SCALAR_TYPE)` with `sizeof(ROLE_TYPE)` in every buffer parameter's `Validation Preconditions` / `Padding Contract` text in `kernels.cl.h`.
- Migrating all six `phase_*.cl.c` files: parameter type declarations, local variable types, load/store site replacements (per ADR-021 §2).
- Injecting transitional `SCALAR_TYPE`/`SCALAR_IS_HALF`/`SCALAR_ZERO` aliases into `meson.build` build option flag lists.
- Updating `src/backends/opencl/type_mapping.py` `build_compiler_flags()` to generate the three-role symbol set plus transitional aliases.
- Updating all `BufferDescriptor(...)` construction sites in `src/shared/plan_builder.py` to supply `precision_role=` from the corresponding `BufferParamSpec.precision_role`.
- Updating `BufferDescriptor(...)` construction sites in `src/backends/vulkan/renderer.py` (ping/pong reduction buffers) to use `precision_role="compute"` and derive `element_size_bytes` from `compute_dtype.itemsize`.
- Removing the transitional `= "compute"` default from `BufferDescriptor.precision_role` and `BufferParamSpec.precision_role` once all construction sites supply the field.
- Rebuilding the CPU shared library (`ninja -C builddir`) and verifying the Tier 2 test suite.

### Out of scope

- CPU `kernel_sources/` changes (`cpu_precision.h`, `cpu_kernels.h`, `.inc` files) — deferred to Phase 7C.
- Vulkan `kernel_sources/` shader changes (`common.glsl`, `.comp` files) — deferred to Phase 7C.
- Removal of the transitional alias block from Meson — deferred to Phase 7C (removal gate: all `.cl.c` _and_ `.inc` files migrated; that's not complete until 7C).
- `PrecisionConfig.float16()` and `PrecisionConfig.mixed_f16_f32()` multi-configuration Tier 2 tests — deferred to Phase 7C.
- `src/backends/opencl/kernel_bindings/` — only the type-mapping layer changes here; the binding objects themselves use `KernelContract` objects which already have `precision_role` from Phase 7A.

### Key constraint: one codepath, no `#ifdef` on type equality

Every precision boundary crossing in the migrated `.cl.c` files must use the `load_storage()` / `store_storage()` / `load_state()` / `store_state_update()` abstractions. No `#if STORAGE_TYPE == COMPUTE_TYPE` guards are introduced. The compiler eliminates identity conversions for the uniform FP32 case. This is the ADR-020 core contract.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `kernels/kernels.cl.h` | Build-symbol checks: `SCALAR_TYPE`, `SCALAR_IS_HALF`, `SIMD_WIDTH`, `C_TILE_SIZE`, `NUMERICAL_STABILITY_EPSILON`. Host stubs define `SCALAR_TYPE=float`, `SCALAR_IS_HALF=0`. No precision boundary function declarations. All `@param` blocks lack `Precision Role` key. |
| `kernels/phase_*.cl.c` (6 files) | All kernel parameters, local variables, and load/store sites use `SCALAR_TYPE`. |
| `src/backends/opencl/type_mapping.py` | `build_compiler_flags()` generates `-DSCALAR_TYPE=...` and `-DSCALAR_IS_HALF=...`. |
| `src/shared/plan_builder.py` | `BufferDescriptor(...)` calls lack `precision_role=` argument (currently uses transitional default `"compute"` from Phase 7A). |
| `src/backends/vulkan/renderer.py` | Ping/pong reduction buffer descriptors hardcode `element_size_bytes=4` and lack `precision_role=`. |
| `src/shared/buffer_lifecycle.py` | `BufferDescriptor.precision_role` has transitional default `= "compute"` (set in Phase 7A). |
| `src/shared/kernel_contracts/phase_*.py` | `BufferParamSpec.precision_role` populated (set in Phase 7A). |

---

## 3. Task Breakdown

---

### Step 7B.1: Update `kernels.cl.h` build-symbol check section

**Governing authority:** ADR-021 §1.1  
**File:** `kernels/kernels.cl.h`

Locate the mandatory-symbol check block (currently checking `SCALAR_TYPE`, `SCALAR_IS_HALF`, etc.). Replace it entirely with:

```c
// --- Mandatory Build-Time Symbols (CONTRACT.md Article 6, amended by ADR-020 §3.6) ---

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

Immediately after, update the `cl_khr_fp16` extension guard: change `SCALAR_IS_HALF` to `STORAGE_TYPE_IS_HALF`:

```c
#if STORAGE_TYPE_IS_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
#if COMPUTE_TYPE_IS_HALF
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif
```

Replace the `SCALAR_ZERO` definition with `COMPUTE_ZERO`:

```c
#if COMPUTE_TYPE_IS_HALF
#define COMPUTE_ZERO ((COMPUTE_TYPE)0.0h)
#else
#define COMPUTE_ZERO ((COMPUTE_TYPE)0.0f)
#endif
```

**Verification:** After the edit, `grep SCALAR_TYPE kernels/kernels.cl.h` must return zero results in the mandatory-symbol section. (`SCALAR_TYPE` may still appear as a comment referencing the transitional alias elsewhere in the file — acceptable until Phase 7C removal.)

---

### Step 7B.2: Update `kernels.cl.h` host/C++ stub section

**Governing authority:** ADR-021 §1.2  
**File:** `kernels/kernels.cl.h`

Locate the host-mode (`#ifndef __OPENCL_VERSION__`) stub definitions. Replace the `SCALAR_TYPE`, `SCALAR_ZERO`, and `SCALAR_IS_HALF` stub definitions with:

```c
#ifndef STORAGE_TYPE
#define STORAGE_TYPE float
#endif
#ifndef COMPUTE_TYPE
#define COMPUTE_TYPE float
#endif
#ifndef STATE_TYPE
#define STATE_TYPE float
#endif
#ifndef STORAGE_TYPE_IS_HALF
#define STORAGE_TYPE_IS_HALF 0
#endif
#ifndef COMPUTE_TYPE_IS_HALF
#define COMPUTE_TYPE_IS_HALF 0
#endif
#ifndef COMPUTE_ZERO
#define COMPUTE_ZERO 0.0f
#endif
```

These stubs default all three role types to `float` for host-mode analysis builds (cppcheck, IDE tooling), matching the prior uniform-FP32 behavior.

---

### Step 7B.3: Add precision boundary abstraction declarations and OpenCL bodies

**Governing authority:** ADR-021 §1.3; ADR-023 §1  
**File:** `kernels/kernels.cl.h`

Inside the `#ifdef __OPENCL_VERSION__` guard, immediately after the mandatory-symbol checks and extension guards, add the following block:

```c
// --- Precision Boundary Abstractions (ADR-020 §4.4, ADR-023 §1) ------
// These are the sole mechanism for crossing precision role boundaries.
// When STORAGE_TYPE == COMPUTE_TYPE, these compile to identity casts
// that any OpenCL compiler eliminates. No #ifdef on type equality is
// used anywhere in the kernel sources.

static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx)
{
#if STORAGE_TYPE_IS_HALF
    return (COMPUTE_TYPE)vload_half(idx, (__global const half *)buf);
#else
    return (COMPUTE_TYPE)buf[idx];
#endif
}

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
#if STORAGE_TYPE_IS_HALF
    vstore_half((half)val, idx, (__global half *)buf);
#else
    buf[idx] = (STORAGE_TYPE)val;
#endif
}

static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx)
{
    return (COMPUTE_TYPE)buf[idx];
}

static inline void store_state(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}

static inline void store_state_update(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val)
{
    buf[idx] = (STATE_TYPE)val;
}
// --- End Precision Boundary Abstractions ---------------------------------
```

**Rationale for `STORAGE_TYPE_IS_HALF` gate in `load_storage`/`store_storage`:** OpenCL's `vload_half`/`vstore_half` are the semantically-correct mechanism for packed FP16 global memory access with the `cl_khr_fp16` extension. Direct cast `(float)((half *)buf)[idx]` is not portably guaranteed across drivers. The flag controls this gate; it is already defined before this section. In the FP32 baseline case (`STORAGE_TYPE_IS_HALF == 0`), both branches compile to `(float)buf[idx]` — an identity cast the compiler eliminates.

**Note on `store_state` vs. `store_state_update`:** Both are identical OpenCL implementations. The semantic distinction (fresh write vs. EMA in-place update) is preserved for future specialization by an optimization pass. The separation has zero runtime cost.

---

### Step 7B.4: Annotate every `@param` block with `Precision Role`

**Governing authority:** ADR-021 §1.4; CONTRACT.md Article 3 (amended Phase 7A)  
**File:** `kernels/kernels.cl.h`

For every float-typed buffer parameter in every `@kernel_contract` block:

1. Add a `Precision Role:` line to the parameter's commentary block, positioned after `Padding Contract:` and before `Calculability Proof:`.
2. Set the value per the role assignment tables in ADR-021 §1.4 (also reproduced in Phase 7A Step 7A.7 above).
3. Change the C type declaration in the kernel signature from `SCALAR_TYPE` to the role-appropriate symbol:
   - `"storage"` → `STORAGE_TYPE`
   - `"compute"` → `COMPUTE_TYPE`
   - `"state"` → `STATE_TYPE`
   - integer / `int` buffers → unchanged `int`

Work kernel by kernel. The complete kernel list by file:

| `.cl.c` file | Kernels to annotate |
|:---|:---|
| `phase_1_act.cl.c` | `forward_pass`, `render_logits_chunk`, `compute_probs_loss_cce_chunk`, `compute_probs_loss_bce_chunk` |
| `phase_2_learn_A_production.cl.c` | `calculate_module_param_grads_chunk`, `backprop_error_to_hidden_chunk`, `calculate_chunk_temp_gradients` |
| `phase_2_learn_B_processing.cl.c` | `clip_partial_gradients`, `gather_and_permute_grad_hidden_activations` |
| `phase_2_learn_C_reduction.cl.c` | `aggregate_register_reduce`, `aggregate_local_reduce`, `clip_intermediate_grad`, `stabilize_and_reduce_grad_hidden_activations`, `reduce_k_fan_in_and_clip` |
| `phase_2_learn_D_backprop.cl.c` | `backprop_shared_weights_chunk`, `backprop_shared_biases_chunk`, `clip_shared_gradients_chunk` |
| `phase_3_update.cl.c` | `normalize_gradients`, `adam_update`, `clamp_temperatures` |

For `__local` scratch parameters: change from `__local SCALAR_TYPE *` to `__local COMPUTE_TYPE *`. The `Precision Role` annotation for local parameters is `"compute"` (per ADR-021 §1.4 governing principle).

**Verification after this step:**

```bash
grep -c "Precision Role:" kernels/kernels.cl.h
```

Count must equal the number of float-typed buffer parameters across all kernels (local + global, excluding `int` parameters).

---

### Step 7B.5: Add `Behavioral Invariants` `Precision Boundary Conversion` entries

**Governing authority:** ADR-021 §1.4; CONTRACT.md Article 4.2 (amended Phase 7A)  
**File:** `kernels/kernels.cl.h`

For every kernel that accesses at least one storage-role or state-role buffer, add `"Precision Boundary Conversion"` to the `@kernel_contract`'s `Behavioral Invariants` list. Use the specific text from ADR-021 §1.4 for each kernel.

Kernels that access **only** compute-role and integer buffers do not require this invariant. From the role tables: `normalize_gradients` (all `"compute"`) does not require it. All other kernels do.

---

### Step 7B.6: Correct `Padding Contract` `sizeof` type references in `kernels.cl.h`

**Governing authority:** ADR-021 §4  
**File:** `kernels/kernels.cl.h`

Every buffer `@param` block `Validation Preconditions` or `Padding Contract` expression that references `sizeof(SCALAR_TYPE)` is updated to reference the role-appropriate size expression:

| Buffer's `precision_role` | Replace `sizeof(SCALAR_TYPE)` with |
|:---|:---|
| `"storage"` | `sizeof(STORAGE_TYPE)` |
| `"compute"` | `sizeof(COMPUTE_TYPE)` |
| `"state"` | `sizeof(STATE_TYPE)` |

Verify:

```bash
grep -n "sizeof(SCALAR_TYPE)" kernels/kernels.cl.h
```

Expected: zero results.

---

### Step 7B.7: Migrate `phase_1_act.cl.c`

**Governing authority:** ADR-021 §2  
**File:** `kernels/phase_1_act.cl.c`

Apply the following mechanical transformations in order. Do not change any algorithmic expression, loop bound, barrier call, or index calculation.

**Kernels covered:** `forward_pass`, `render_logits_chunk`, `compute_probs_loss_cce_chunk`, `compute_probs_loss_bce_chunk`

1. **Parameter type declarations:** Replace `SCALAR_TYPE` with the role-appropriate symbol per ADR-021 §1.4 tables. Every `__global SCALAR_TYPE*` → `__global STORAGE_TYPE*`, `__global STATE_TYPE*`, or `__global COMPUTE_TYPE*` as applicable. `__local SCALAR_TYPE` → `__local COMPUTE_TYPE`.

2. **Local variable declarations:** Every `SCALAR_TYPE` local variable that holds an intermediate arithmetic result → `COMPUTE_TYPE`.

3. **Load sites from storage-role buffers:**
   - `buf[idx]` → `load_storage(buf, idx)` for every direct read from a `STORAGE_TYPE*` parameter.

4. **Load sites from state-role buffers:**
   - `buf[idx]` → `load_state(buf, idx)` for every direct read from a `STATE_TYPE*` parameter.

5. **Store sites to storage-role buffers:**
   - `buf[idx] = val` → `store_storage(buf, idx, val)` for every direct write to a `STORAGE_TYPE*` parameter.

6. **`SCALAR_ZERO` → `COMPUTE_ZERO`** in all local variable initializations.

7. **No changes** to: load/store of `int`-typed buffers (`targets` in CCE), loop bounds, `get_global_id`/`get_local_id` calls, `barrier()` calls, index arithmetic, reduction logic, Softmax computation expressions.

**Post-edit check for this file:**

```bash
grep -n "SCALAR_TYPE\|SCALAR_ZERO\|SCALAR_IS_HALF" kernels/phase_1_act.cl.c
```

Expected: zero results (the transitional alias from Meson means the file still compiles if any were missed, but there should be none).

---

### Step 7B.8: Migrate `phase_2_learn_A_production.cl.c`

**File:** `kernels/phase_2_learn_A_production.cl.c`

Apply the same six-step transformation as Step 7B.7. Kernels: `calculate_module_param_grads_chunk`, `backprop_error_to_hidden_chunk`, `calculate_chunk_temp_gradients`.

Key role assignments for this file:
- `hidden_activations`, `partial_probs`, `partial_grad_*` buffers → `"storage"` → `load_storage`/`store_storage`
- `weights_module`, `biases_module`, `temps` → `"state"` → `load_state`
- `__local` reduction tile → `COMPUTE_TYPE`

No stores to state-role buffers in this file (partial gradient outputs go to storage-role buffers). `store_state_update` is not used here.

---

### Step 7B.9: Migrate `phase_2_learn_B_processing.cl.c`

**File:** `kernels/phase_2_learn_B_processing.cl.c`

Kernels: `clip_partial_gradients`, `gather_and_permute_grad_hidden_activations`.

Both kernels operate entirely on storage-role transient gradient partials. All `SCALAR_TYPE*` parameters → `STORAGE_TYPE*`. All in-place reads/writes use `load_storage`/`store_storage`. No state-role or compute-role buffers in this file.

---

### Step 7B.10: Migrate `phase_2_learn_C_reduction.cl.c`

**File:** `kernels/phase_2_learn_C_reduction.cl.c`

Kernels: `aggregate_register_reduce`, `aggregate_local_reduce`, `clip_intermediate_grad`, `stabilize_and_reduce_grad_hidden_activations`, `reduce_k_fan_in_and_clip`.

Key distinctions:
- Input (partial sum) buffers → `"storage"` → `load_storage`
- Final reduction output buffers (that exit the reduction tree) → `"compute"` → direct write `buf[idx] = val` (no narrowing — value is already `COMPUTE_TYPE`)
- Intermediate reduction output buffers (inter-stage, not final) → `"storage"` → `store_storage`
- `__local` scratch → `COMPUTE_TYPE`

**Careful handling required:** Identify which `dest` buffer in each kernel is the "final reduction output" versus an "intermediate stage output." Consult the per-kernel role tables in ADR-021 §1.4. The final-output buffer's store must be a direct assignment (`COMPUTE_TYPE dest_buf[idx] = accum`); intermediate outputs use `store_storage`.

---

### Step 7B.11: Migrate `phase_2_learn_D_backprop.cl.c`

**File:** `kernels/phase_2_learn_D_backprop.cl.c`

Kernels: `backprop_shared_weights_chunk`, `backprop_shared_biases_chunk`, `clip_shared_gradients_chunk`.

Key role assignments:
- `grad_hidden_activations` input → `"compute"` (arrives from reduction engine) → direct read `buf[idx]`
- `input`, `hidden_mask` → `"storage"` → `load_storage`
- `partial_grad_weights_shared`, `partial_grad_biases_shared` → `"storage"` → `store_storage`
- `partial_grad_shared` (clip, in-place) → `"storage"` → `load_storage` + `store_storage`

---

### Step 7B.12: Migrate `phase_3_update.cl.c`

**File:** `kernels/phase_3_update.cl.c`

Kernels: `normalize_gradients`, `adam_update`, `clamp_temperatures`.

Key role assignments:
- `normalize_gradients`: `summed_gradient` is `"compute"` — in-place divide, direct read/write (`buf[idx]`, `buf[idx] = val`).
- `adam_update`: `normalized_gradient` → `"compute"` → direct read. `m1`, `m2`, `weights_or_biases` → `"state"` → `load_state` + `store_state_update`.
- `clamp_temperatures`: `temps` → `"state"` → `load_state` + `store_state_update`.

After this step, run the full `.cl.c` migration verification:

```bash
grep -r "SCALAR_TYPE\|SCALAR_ZERO\|SCALAR_IS_HALF" kernels/
```

Expected: zero results in any `.cl.c` file. Results in `kernels.cl.h` (if present in transitional-alias comments) are acceptable.

---

### Step 7B.13: Inject transitional aliases into Meson build

**Governing authority:** ADR-021 §3  
**File:** `meson.build` (architecture-level, or the relevant `src/backends/opencl/meson.build` that generates kernel build flags)

Add the transitional alias block to the OpenCL compiler flag generation. This block is injected alongside the flags already supplied to the OpenCL kernel compilation target. It ensures that any file not yet migrated (or any test fixture that still supplies `SCALAR_TYPE` via a custom build path) continues to compile.

```meson
# Transitional aliases (ADR-021 §3) — removed when Phase 7C kernel migration is complete.
'-DSCALAR_TYPE=COMPUTE_TYPE',
'-DSCALAR_IS_HALF=COMPUTE_TYPE_IS_HALF',
'-DSCALAR_ZERO=COMPUTE_ZERO',
```

**Important:** These aliases are defined in Meson as build-flag strings passed to the OpenCL runtime/compiler, not as C preprocessor `#define` lines in headers. The exact injection site depends on how the OpenCL kernel source is compiled — either via the `src/backends/opencl/type_mapping.py` `build_compiler_flags()` return value (updated in Step 7B.14), or directly in `meson.build` for native builds.

Per ADR-022 §7.1, the alias injection for runtime OpenCL compilation happens in `build_compiler_flags()`. For any static analysis / cppcheck build targets in `meson.build`, the flags are also added there.

---

### Step 7B.14: Update `src/backends/opencl/type_mapping.py`

**Governing authority:** ADR-022 §7.1  
**File:** `src/backends/opencl/type_mapping.py`

Replace `build_compiler_flags()` entirely. The new function signature and body:

```python
def build_compiler_flags(
    precision: PrecisionConfig,
    hardware: HardwareProfile,
    c_tile_size: int,
) -> list[str]:
    storage_cl = _dtype_to_cl_type(precision.storage_dtype)
    compute_cl = _dtype_to_cl_type(precision.compute_dtype)
    state_cl   = _dtype_to_cl_type(precision.state_dtype)
    storage_is_half = 1 if precision.storage_dtype == np.dtype(np.float16) else 0
    compute_is_half = 1 if precision.compute_dtype == np.dtype(np.float16) else 0
    eps = _epsilon_literal(precision.compute_epsilon, compute_is_half)
    flags = [
        f"-DSTORAGE_TYPE={storage_cl}",
        f"-DCOMPUTE_TYPE={compute_cl}",
        f"-DSTATE_TYPE={state_cl}",
        f"-DSTORAGE_TYPE_IS_HALF={storage_is_half}",
        f"-DCOMPUTE_TYPE_IS_HALF={compute_is_half}",
        f"-DSIMD_WIDTH={hardware.simd_width}",
        f"-DC_TILE_SIZE={c_tile_size}",
        f"-DNUMERICAL_STABILITY_EPSILON={eps}",
        "-DLOCAL_MEM_BANK_PADDING=1",
    ]
    # Transitional aliases (ADR-021 §3) — removed in Phase 7C when all
    # kernel source files are migrated.
    flags += [
        "-DSCALAR_TYPE=COMPUTE_TYPE",
        "-DSCALAR_IS_HALF=COMPUTE_TYPE_IS_HALF",
        "-DSCALAR_ZERO=COMPUTE_ZERO",
    ]
    return flags
```

Replace the internal helper `numpy_dtype_to_cl_type_name(precision)` (which accepted a full `PrecisionConfig`) with `_dtype_to_cl_type(dtype: np.dtype) -> str` (which accepts a bare `np.dtype`). Update all callers within the module.

Update `_epsilon_literal` to accept `compute_epsilon: float` and `compute_is_half: int` directly rather than deriving them from a single-precision config.

---

### Step 7B.15: Wire `precision_role` into `plan_builder.py` and Vulkan `renderer.py`

**Governing authority:** ADR-022 §8  
**Files:** `src/shared/plan_builder.py`, `src/backends/vulkan/renderer.py`

#### `plan_builder.py`

Every `BufferDescriptor(...)` construction site must now supply `precision_role=` explicitly. The value comes from the corresponding `BufferParamSpec.precision_role` (populated in Phase 7A Step 7A.7). Update each construction site:

```python
# Before (using transitional default):
BufferDescriptor(
    name=param.name,
    role=...,
    element_size_bytes=precision.storage_dtype.itemsize,  # (already may vary)
    ...
)

# After:
BufferDescriptor(
    name=param.name,
    role=...,
    precision_role=param.precision_role,
    element_size_bytes=_element_size_for_role(param.precision_role, precision),
    ...
)
```

Where `_element_size_for_role` is an inline helper (or a simple `match` expression):

```python
def _element_size_for_role(
    precision_role: str, precision: PrecisionConfig
) -> int:
    if precision_role == "storage":
        return precision.storage_dtype.itemsize
    elif precision_role == "state":
        return precision.state_dtype.itemsize
    else:  # "compute"
        return precision.compute_dtype.itemsize
```

#### `src/backends/vulkan/renderer.py`

The reduction ping/pong buffer descriptors (currently hardcoding `element_size_bytes=4`) are updated:

```python
# Ping/pong buffers: reduction tree outputs → "compute" role
BufferDescriptor(
    ...
    precision_role="compute",
    element_size_bytes=precision.compute_dtype.itemsize,
    ...
)
```

The offset/index descriptor (carrying `int` elements) is unchanged.

---

### Step 7B.16: Remove transitional default from `BufferDescriptor` and `BufferParamSpec`

**Files:** `src/shared/buffer_lifecycle.py`, `src/shared/kernel_contracts/` (all `BufferParamSpec` definitions across six files if a default was placed on the dataclass field rather than each instance)

The `= "compute"` temporary defaults added in Phase 7A Step 7A.6 and 7A.7 are now removed, making `precision_role` a mandatory field with no default. This confirms that every construction site in Steps 7B.15 and 7A.7 is complete.

After removing the defaults, run:

```bash
python -c "from src.shared.buffer_lifecycle import BufferDescriptor; BufferDescriptor()"
```

This should raise `TypeError: missing required keyword argument 'precision_role'`, confirming the default is gone.

---

### Step 7B.17: Rebuild and validate rollback gate

Per the AGENTS.md build discipline, any edit to `src/backends/cpu/kernel_sources/` requires a rebuild. Phase 7B does **not** touch those files. However, the Meson build system generates `_build_config.py` from flags, and the OpenCL type-mapping change in Step 7B.14 may affect the build configuration. Rebuild to confirm:

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir 2>&1 | tee /tmp/phase7b_build.txt
grep -E "error:|warning:" /tmp/phase7b_build.txt | head -30
```

Then run the full test suite:

```bash
python -m pytest src/tests/ -x -q 2>&1 | tee /tmp/phase7b_tests.txt
grep -E "passed|failed|error" /tmp/phase7b_tests.txt
```

Migration completion check for Phase 7B:

```bash
# No SCALAR_TYPE in any .cl.c file
grep -r "SCALAR_TYPE" kernels/*.cl.c && echo "FAIL: unreplaced SCALAR_TYPE found" || echo "OK"

# No sizeof(SCALAR_TYPE) in kernels.cl.h
grep "sizeof(SCALAR_TYPE)" kernels/kernels.cl.h && echo "FAIL" || echo "OK"
```

---

## 4. `.cl.c` Migration Checklist

Use this checklist for each of the six files (Steps 7B.7–7B.12):

- [ ] All kernel parameter declarations: `SCALAR_TYPE` → role-appropriate symbol
- [ ] All `__local SCALAR_TYPE` → `__local COMPUTE_TYPE`
- [ ] All local intermediate variable declarations: `SCALAR_TYPE` → `COMPUTE_TYPE`
- [ ] All reads from `STORAGE_TYPE*` params: `buf[idx]` → `load_storage(buf, idx)`
- [ ] All reads from `STATE_TYPE*` params: `buf[idx]` → `load_state(buf, idx)`
- [ ] All writes to `STORAGE_TYPE*` params: `buf[idx] = val` → `store_storage(buf, idx, val)`
- [ ] All writes to `STATE_TYPE*` params: `buf[idx] = val` → `store_state_update(buf, idx, val)` (EMA updates) or `store_state(buf, idx, val)` (plain writes)
- [ ] All `SCALAR_ZERO` → `COMPUTE_ZERO`
- [ ] No algorithmic expressions, loop bounds, index calculations, or barrier calls changed
- [ ] Post-edit `grep SCALAR_TYPE <file>` returns zero results

---

## 5. Transitional Alias Protocol Details

The aliases injected in Step 7B.13 and 7B.14:

```
-DSCALAR_TYPE=COMPUTE_TYPE
-DSCALAR_IS_HALF=COMPUTE_TYPE_IS_HALF
-DSCALAR_ZERO=COMPUTE_ZERO
```

**What they do:** Allow any remaining source file that still uses `SCALAR_TYPE` (in a `.cl.c` file, `.inc` file, or CPU kernel source that hasn't been migrated yet) to continue compiling. `SCALAR_TYPE` expands to the compute-role type name, which is the semantically correct fallback.

**When they are removed:** Phase 7C Step 7C.13, after all `.inc` template files and Vulkan `.comp` shaders are migrated. The removal commit's gate is: `grep -r SCALAR_TYPE architectures/averaging_ensembled_classifier/kernels/ src/backends/` returns zero results in non-comment code.

**The aliases must not be relied on as a permanent substitute.** Their presence in the build is a migration progress indicator. Any file that still uses `SCALAR_TYPE` after Phase 7B shows as migration debt carried into Phase 7C.

---

## 6. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| A `.cl.c` file has an indirect `SCALAR_TYPE` reference (macro concatenation, stringify) that grep doesn't catch | Low | Run a test build with transitional aliases removed temporarily; compilation failure reveals missed occurrences. |
| `_element_size_for_role` computes wrong size for a buffer, causing allocation mismatch | Medium | Add an assertion in `BufferDescriptor.__post_init__` that `size_bytes % element_size_bytes == 0`. Tier 2 tests will catch allocation size regressions. |
| Reduction final-vs-intermediate output misclassification in Step 7B.10 | Medium | Cross-reference the role tables in ADR-021 §1.4 per kernel. If in doubt, the DAG diagram in DESIGN.md §5 shows which nodes consume reduction outputs — final outputs exit the reduction tree into normalize/adam. |
| OpenCL runtime rejects `(COMPUTE_TYPE)buf[idx]` cast when `STORAGE_TYPE_IS_HALF == 0` and both are `float` | Very low | `(float)float_val` is a valid no-op cast in OpenCL C. All major drivers accept it. Verified by the FP32 baseline Tier 2 run. |
| Transitional alias causes the `kernels.cl.h` mandatory-symbol `#error` to not fire when `SCALAR_TYPE` is missing | N/A | The mandatory-symbol block checks `STORAGE_TYPE`, not `SCALAR_TYPE`. The alias can't inject `STORAGE_TYPE` — that must be supplied explicitly. |
| `plan_builder.py` has `BufferDescriptor` construction sites not imported via `KernelContract`/`BufferParamSpec` (e.g., hand-constructed descriptors for reduction scratch) | Medium | Audit with `grep -n "BufferDescriptor(" src/shared/plan_builder.py`. Every hit must supply `precision_role=`. |
