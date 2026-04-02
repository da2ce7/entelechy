# Phase 7C: Mixed Precision — CPU & Vulkan Backends, Alias Removal, Multi-Config Tests

**Status: 📋 PLANNED**  
**Phase:** 7C of 7  
**Prerequisite:** Phase 7B complete and rollback gate passed.  
**Objective:** Migrate the CPU backend's native kernel infrastructure and the Vulkan backend's GLSL shader library to the three-role precision model. Complete the migration by removing the transitional `SCALAR_TYPE` aliases. Add multi-configuration Tier 2 test coverage for `PrecisionConfig.float16()` and `PrecisionConfig.mixed_f16_f32()`. The phase ends when the migration completion criteria from ADR-021 §5 and ADR-023 §6 are all satisfied and "The Alchemist" validation scenario passes.  
**Governing ADRs:** ADR-023 §§2–6 (CPU and Vulkan backend-native code; build system), ADR-022 §§7.2–7.3 (type-mapping modules), ADR-021 §3 (alias removal gate), ADR-020 §4.6, §2.6 (multi-config tests; Alchemist scenario)  
**Rollback gate:** All Tier 2 tests pass for all three `PrecisionConfig` factories. `PrecisionConfig.float32()` results are bit-for-bit identical to the Phase 7B baseline. `grep -r SCALAR_TYPE architectures/averaging_ensembled_classifier/` returns zero results in non-comment code. `grep -r simd_load_real src/backends/cpu/` returns zero results. `grep -r REAL_T src/backends/cpu/kernel_sources/` returns zero results (outside transitional alias comments).  
**Dependencies:** Phase 7B complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown — CPU Backend](#3-task-breakdown--cpu-backend)
   - [Step 7C.1: Update `cpu_precision.h` — role-split load/store macro system](#step-7c1-update-cpu_precisionh--role-split-loadstore-macro-system)
   - [Step 7C.2: Extend `cpu_kernels.h` `DECLARE_PRECISION_STRUCTS`](#step-7c2-extend-cpu_kernelsh-declare_precision_structs)
   - [Step 7C.3: Add `cpu_compute_t` type constant](#step-7c3-add-cpu_compute_t-type-constant)
   - [Step 7C.4: Update `cpu_kernels.c` struct instantiations](#step-7c4-update-cpu_kernelsc-struct-instantiations)
   - [Step 7C.5: Migrate all `.inc` template files](#step-7c5-migrate-all-inc-template-files)
   - [Step 7C.6: Update `src/backends/cpu/type_mapping.py`](#step-7c6-update-srcbackendscputype_mappingpy)
   - [Step 7C.7: Update CPU FFI layer (`_ffi_types.py`, `_dispatch_table.py`)](#step-7c7-update-cpu-ffi-layer-_ffi_typespy-_dispatch_tablepy)
   - [Step 7C.8: Rebuild CPU shared library](#step-7c8-rebuild-cpu-shared-library)
4. [Task Breakdown — Vulkan Backend](#4-task-breakdown--vulkan-backend)
   - [Step 7C.9: Update `common.glsl`](#step-7c9-update-commonglsl)
   - [Step 7C.10: Migrate all `.comp` shaders](#step-7c10-migrate-all-comp-shaders)
   - [Step 7C.11: Update `src/backends/vulkan/type_mapping.py`](#step-7c11-update-srcbackendsvulkantype_mappingpy)
   - [Step 7C.12: Update Vulkan `meson.build` for multi-variant SPIR-V compilation](#step-7c12-update-vulkan-mesonbuild-for-multi-variant-spir-v-compilation)
   - [Step 7C.13: Update `_pipeline_cache.py` for variant selection](#step-7c13-update-_pipeline_cachepy-for-variant-selection)
5. [Task Breakdown — Final Alias Removal & Verification](#5-task-breakdown--final-alias-removal--verification)
   - [Step 7C.14: Remove transitional `SCALAR_TYPE` aliases from Meson and `type_mapping.py`](#step-7c14-remove-transitional-scalar_type-aliases-from-meson-and-type_mappingpy)
   - [Step 7C.15: Add multi-configuration Tier 2 test coverage](#step-7c15-add-multi-configuration-tier-2-test-coverage)
   - [Step 7C.16: Full migration completion verification](#step-7c16-full-migration-completion-verification)
6. [`.inc` Template Migration Checklist](#6-inc-template-migration-checklist)
7. [`.comp` Shader Migration Checklist](#7-comp-shader-migration-checklist)
8. [Cross-Backend Summary](#8-cross-backend-summary)
9. [Risk Register](#9-risk-register)

---

## 1. Scope & Constraints

### In scope

**CPU backend:**
- `src/backends/cpu/kernel_sources/cpu_precision.h`: Replace single-axis `simd_load_real`/`simd_store_real` macros with two-axis `simd_load_storage`/`simd_store_storage` and `simd_load_state`/`simd_store_state` families; introduce `STORAGE_SUFFIX`/`STATE_SUFFIX` macro axes; retire `PREC_SIZEOF_REAL` in favour of `PREC_SIZEOF_STORAGE` and `PREC_SIZEOF_STATE`.
- `src/backends/cpu/kernel_sources/cpu_kernels.h`: Extend `DECLARE_PRECISION_STRUCTS(SUFFIX, REAL_T)` to `DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)`; retype all struct float buffer fields to the role-appropriate type per ADR-021 §1.4; add new instantiations `s32x32`, `s16x16`, `s16x32`; retire the `fp64` instantiation.
- `src/backends/cpu/kernel_sources/cpu_kernels.c`: Update struct instantiation `#define`/`#include` blocks to use the new two-axis scheme.
- `src/backends/cpu/kernel_sources/phase_*.inc` (6 files): Replace every `simd_load_real`/`scalar_load_real`/`simd_store_real`/`scalar_store_real` call with the role-appropriate equivalent.
- `src/backends/cpu/type_mapping.py`: Replace `get_numpy_dtype(precision)` with `get_storage_dtype`, `get_compute_dtype`, `get_state_dtype`.
- `src/backends/cpu/_ffi_types.py`: Update struct suffix references from `fp32`/`fp16` to `s32x32`/`s16x16`/`s16x32`.
- `src/backends/cpu/_dispatch_table.py`: Update dispatch-suffix selection to use `storage_dtype` + `state_dtype` jointly.
- `src/backends/cpu/meson.build`: Rebuild trigger ensured (see Step 7C.8).

**Vulkan backend:**
- `src/backends/vulkan/kernel_sources/common.glsl`: Add precision-role macro definitions (`STORAGE_FLOAT`, `STATE_FLOAT`), conditional FP16 extension guard, `_compute_scratch` rename, and precision boundary load/store helpers.
- `src/backends/vulkan/kernel_sources/*.comp` (18 files with storage/state role buffers): Update buffer layout element type declarations to `STORAGE_FLOAT`/`STATE_FLOAT`; wrap load/store sites through `read_storage()`/`write_storage()`/`read_state()`/`write_state()` helpers.
- `src/backends/vulkan/type_mapping.py`: Replace `get_numpy_dtype(precision)` and `get_specialization_scalar_is_half(precision)` with role-split equivalents.
- `src/backends/vulkan/meson.build`: Add multi-variant `glslc` compilation producing `_fp32`, `_s16fp32`, `_fp16` SPIR-V artifacts per storage/state-role-bearing shader.
- `src/backends/vulkan/_pipeline_cache.py`: Update SPIR-V path selection to use `PrecisionConfig.storage_dtype` + `PrecisionConfig.state_dtype`.

**Shared finalization:**
- Remove transitional `SCALAR_TYPE`/`SCALAR_IS_HALF`/`SCALAR_ZERO` alias flags from `src/backends/opencl/type_mapping.py` and any Meson build flag lists.
- Add Tier 2 test parameterization over `PrecisionConfig.float16()` and `PrecisionConfig.mixed_f16_f32()`.

### Out of scope

- Any further changes to `kernels/kernels.cl.h` or `kernels/phase_*.cl.c` — completed in Phase 7B.
- Any changes to `src/shared/` Python modules — completed in Phase 7A/7B.
- FP8 `PrecisionConfig` factories — gated by ADR-020 §5 preconditions; the abstraction layer established here structurally supports FP8 storage but no factory or tests are added in this phase.
- Vulkan `_pipeline_cache.py` internal implementation details beyond the SPIR-V variant path selection (e.g. pipeline recreation logic, descriptor set management) — those are pre-existing implementation concerns.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/backends/cpu/kernel_sources/cpu_precision.h` | Single-axis `simd_load_real`/`simd_store_real` keyed to `REAL_T`/`PRECISION_SUFFIX`. `PREC_SIZEOF_REAL` macro. |
| `src/backends/cpu/kernel_sources/cpu_kernels.h` | `DECLARE_PRECISION_STRUCTS(SUFFIX, REAL_T)` — 2-parameter macro. Instantiations: `fp32`/`float`, `fp16`/`_Float16`, `fp64`/`double`. All struct float fields use `REAL_T`. |
| `src/backends/cpu/kernel_sources/cpu_kernels.c` | Three instantiation blocks (fp32, fp16, fp64), each `#define REAL_T ... #define PRECISION_SUFFIX ... #include phase_*.inc #undef ...`. |
| `src/backends/cpu/kernel_sources/phase_*.inc` | All float loads/stores use `simd_load_real(buf)` / `scalar_load_real(buf, idx)` / `simd_store_real(buf, val)` / `scalar_store_real(buf, idx, val)`. |
| `src/backends/cpu/type_mapping.py` | `get_numpy_dtype(precision: PrecisionConfig) -> np.dtype` returns `precision.numpy_dtype`. |
| `src/backends/cpu/_ffi_types.py` | Struct suffixes `_fp32`, `_fp16` in ctypes bindings. |
| `src/backends/cpu/_dispatch_table.py` | Dispatches on a single dtype to select struct suffix. |
| `src/backends/vulkan/kernel_sources/common.glsl` | No `STORAGE_FLOAT`/`STATE_FLOAT` macros. Workgroup scratch named `_cross_subgroup_scratch`. No precision boundary helpers. |
| `src/backends/vulkan/kernel_sources/*.comp` | All SSBOs declared as `float data[]`. No `read_storage()`/`write_storage()` wrappers. |
| `src/backends/vulkan/type_mapping.py` | `get_numpy_dtype(precision)`, `get_element_size(precision)`, `get_specialization_scalar_is_half(precision)` — single-axis. |
| `src/backends/vulkan/meson.build` | Single `glslc` invocation per `.comp` file, no precision variants. |
| `src/backends/vulkan/_pipeline_cache.py` | SPIR-V paths are single per kernel name. |

---

## 3. Task Breakdown — CPU Backend

---

### Step 7C.1: Update `cpu_precision.h` — role-split load/store macro system

**Governing authority:** ADR-023 §2.2  
**File:** `src/backends/cpu/kernel_sources/cpu_precision.h`

#### 7C.1.1: Retire `simd_load_real` family and `PREC_SIZEOF_REAL`

Remove (or `#undef` at the appropriate location) the existing single-axis macro definitions:
- `simd_load_real`, `simd_store_real`, `scalar_load_real`, `scalar_store_real`
- `PREC_SIZEOF_REAL`

#### 7C.1.2: Add two-axis dispatch macros

Add separate macro families for each precision role. The pattern mirrors the existing single-axis design but parameterises across two suffix axes — `STORAGE_SUFFIX` and `STATE_SUFFIX`:

```c
// Storage-role load/store — keyed on STORAGE_SUFFIX and STORAGE_T
#undef simd_load_storage
#undef simd_store_storage
#undef scalar_load_storage
#undef scalar_store_storage
#define simd_load_storage    _PREC_CAT2(simd_load_storage_,  STORAGE_SUFFIX)
#define simd_store_storage   _PREC_CAT2(simd_store_storage_, STORAGE_SUFFIX)
#define scalar_load_storage  _PREC_CAT2(scalar_load_storage_,  STORAGE_SUFFIX)
#define scalar_store_storage _PREC_CAT2(scalar_store_storage_, STORAGE_SUFFIX)

// State-role load/store — keyed on STATE_SUFFIX and STATE_T
#undef simd_load_state
#undef simd_store_state
#undef scalar_load_state
#undef scalar_store_state
#define simd_load_state      _PREC_CAT2(simd_load_state_,  STATE_SUFFIX)
#define simd_store_state     _PREC_CAT2(simd_store_state_, STATE_SUFFIX)
#define scalar_load_state    _PREC_CAT2(scalar_load_state_,  STATE_SUFFIX)
#define scalar_store_state   _PREC_CAT2(scalar_store_state_, STATE_SUFFIX)

// Role-specific element sizes
#define PREC_SIZEOF_STORAGE sizeof(STORAGE_T)
#define PREC_SIZEOF_STATE   sizeof(STATE_T)
```

The actual function implementations (e.g., `simd_load_storage_fp32`, `simd_load_storage_fp16`) follow the existing implementation pattern for the `simd_load_real_fp32` / `simd_load_real_fp16` functions — only the function name changes. The function bodies (SIMD load from `STORAGE_T*`, widen to `float`, etc.) are identical in structure.

**Key invariant:** When `STATE_SUFFIX == STORAGE_SUFFIX`, `simd_load_state_fp32` and `simd_load_storage_fp32` resolve to identical function definitions. The compiler inlines them identically. Zero overhead for the uniform configuration.

#### 7C.1.3: Backward-compatibility alias (transitional)

Add transitional aliases to ease migration of `.inc` files before they are updated:

```c
// Transitional aliases — removed in the same commit that migrates all .inc files.
#define simd_load_real    simd_load_storage
#define simd_store_real   simd_store_storage
#define scalar_load_real  scalar_load_storage
#define scalar_store_real scalar_store_storage
#define PREC_SIZEOF_REAL  PREC_SIZEOF_STORAGE
```

These aliases are removed at the end of Step 7C.5 when all `.inc` files are migrated.

---

### Step 7C.2: Extend `cpu_kernels.h` `DECLARE_PRECISION_STRUCTS`

**Governing authority:** ADR-023 §2.3, §2.4  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.h`

#### 7C.2.1: New macro signature

Change `DECLARE_PRECISION_STRUCTS(SUFFIX, REAL_T)` to:

```c
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)
```

#### 7C.2.2: Retype all struct buffer fields

Inside the macro expansion, every float buffer pointer field is retyped per the role assignment tables in ADR-021 §1.4. The key column mapping (from ADR-023 §2.3):

| Buffer role | Old C pointer type | New C pointer type |
|:---|:---|:---|
| Storage-role input | `const REAL_T*` | `const STORAGE_T*` |
| Storage-role output | `REAL_T*` | `STORAGE_T*` |
| Compute-role input | `const REAL_T*` | `const float*` |
| Compute-role output | `REAL_T*` | `float*` |
| State-role read-only | `const REAL_T*` | `const STATE_T*` |
| State-role read-write | `STATE_T*` | `STATE_T*` |

Concrete examples from ADR-023 §2.3 (verify against the actual struct definitions in `cpu_kernels.h`):

| Struct | Field | Old type | New type |
|:---|:---|:---|:---|
| `ForwardPassArgs` | `input` | `const REAL_T*` | `const STORAGE_T*` |
| `ForwardPassArgs` | `weights_shared_simd_major` | `const REAL_T*` | `const STATE_T*` |
| `ForwardPassArgs` | `biases_shared` | `const REAL_T*` | `const STATE_T*` |
| `ForwardPassArgs` | `hidden_activations` | `REAL_T*` | `STORAGE_T*` |
| `CceChunkArgs` | `temps` | `const REAL_T*` | `const STATE_T*` |
| `CceChunkArgs` | `final_loss` | `REAL_T*` | `float*` |
| `AdamUpdateArgs` | `normalized_gradient` | `const REAL_T*` | `const float*` |
| `AdamUpdateArgs` | `m1`, `m2` | `REAL_T*` | `STATE_T*` |
| `AdamUpdateArgs` | `weights_or_biases` | `REAL_T*` | `STATE_T*` |
| Reduction output struct fields | output | `REAL_T*` | `float*` |

Apply the full retyping to every struct — the table above shows representative examples only.

#### 7C.2.3: Update the three instantiations

Replace the old instantiation calls (`DECLARE_PRECISION_STRUCTS(fp32, float)`, etc.) with:

```c
DECLARE_PRECISION_STRUCTS(s32x32, float, float)
DECLARE_PRECISION_STRUCTS(s16x16, _Float16, _Float16)
DECLARE_PRECISION_STRUCTS(s16x32, _Float16, float)   // mixed: FP16 storage, FP32 state
```

Remove `DECLARE_PRECISION_STRUCTS(fp64, double)` — no `PrecisionConfig` factory exists for FP64.

#### 7C.2.4: Transitional backward-compat alias

Add transitional name aliases for the old `fp32` / `fp16` suffix names during the FFI migration window:

```c
// Transitional aliases — removed when _ffi_types.py and _dispatch_table.py are updated.
#define ForwardPassArgs_fp32  ForwardPassArgs_s32x32
#define AdamUpdateArgs_fp32   AdamUpdateArgs_s32x32
// ... (one alias per struct type per old suffix)
#define ForwardPassArgs_fp16  ForwardPassArgs_s16x16
// etc.
```

These aliases are removed in Step 7C.7 when the Python FFI layer is updated.

---

### Step 7C.3: Add `cpu_compute_t` type constant

**Governing authority:** ADR-023 §2.1  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.h`

Add at the top of `cpu_kernels.h`, near the other type declarations:

```c
/* COMPUTE_TYPE is invariant on the CPU backend: always float.
 * Per ADR-023 §2.1: CPU arithmetic always executes at FP32 precision.
 * STORAGE_T and STATE_T vary by PrecisionConfig; float does not. */
typedef float cpu_compute_t;
```

This is a documentation artifact — it formalises the existing "compute is always float" convention as an architectural constant. No `.inc` code changes are needed from this step alone.

---

### Step 7C.4: Update `cpu_kernels.c` struct instantiations

**Governing authority:** ADR-023 §4.2  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.c`

Each instantiation block currently defines `REAL_T` and `PRECISION_SUFFIX` before including `cpu_precision.h` and then the phase `.inc` files. Replace with the two-axis pattern.

**Before (example — fp32 instantiation):**
```c
#define REAL_T           float
#define PRECISION_SUFFIX fp32
#include "cpu_precision.h"
#include "phase_1_act.inc"
// ...
#undef REAL_T
#undef PRECISION_SUFFIX
```

**After (example — s32x32 instantiation):**
```c
#define STORAGE_T       float
#define STORAGE_SUFFIX  fp32
#define STATE_T         float
#define STATE_SUFFIX    fp32
#include "cpu_precision.h"
#include "phase_1_act.inc"
// ...
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
```

**Three instantiation blocks:** `s32x32` (float/float), `s16x16` (_Float16/_Float16), `s16x32` (_Float16/float).

Remove the `fp64` (double/double) instantiation block entirely.

During the `.inc` migration window (before Step 7C.5 is complete), the backward-compat aliases in `cpu_precision.h` (Step 7C.1.3) allow the old `simd_load_real` calls in unmigrated `.inc` files to continue resolving to `simd_load_storage` (keyed to `STORAGE_SUFFIX`), which is the correct semantic fallback.

---

### Step 7C.5: Migrate all `.inc` template files

**Governing authority:** ADR-023 §2.5  
**Files:** `src/backends/cpu/kernel_sources/phase_*.inc` (6 files)

Apply the following mechanical transformation to each file. No algorithmic changes.

For each `.inc` file:

1. All reads from storage-role buffer pointers:
   - `simd_load_real(buf)` → `simd_load_storage(buf)`
   - `scalar_load_real(buf, idx)` → `scalar_load_storage(buf, idx)`

2. All writes to storage-role buffer pointers:
   - `simd_store_real(buf, val)` → `simd_store_storage(buf, val)`
   - `scalar_store_real(buf, idx, val)` → `scalar_store_storage(buf, idx, val)`

3. All reads from state-role buffer pointers:
   - `simd_load_real(buf)` → `simd_load_state(buf)`
   - `scalar_load_real(buf, idx)` → `scalar_load_state(buf, idx)`

4. All writes to state-role buffer pointers (EMA updates in Adam):
   - `simd_store_real(buf, val)` → `simd_store_state(buf, val)`
   - `scalar_store_real(buf, idx, val)` → `scalar_store_state(buf, idx, val)`

5. All reads/writes to compute-role buffers (`float*`) and stack scratch (`float` arrays):
   - Direct dereference `buf[idx]` — **unchanged**, no wrapper.

6. `PREC_SIZEOF_REAL` → `PREC_SIZEOF_STORAGE` (for storage-role buffer padding calculations).

The role of each buffer pointer in each `.inc` file is determined by the corresponding struct field type from `cpu_kernels.h` (updated in Step 7C.2):
- `const STORAGE_T*` / `STORAGE_T*` fields → use `storage` family
- `const STATE_T*` / `STATE_T*` fields → use `state` family
- `const float*` / `float*` fields → direct access

After completing all six files, remove the backward-compat aliases added in Step 7C.1.3.

Post-migration check:
```bash
grep -rn "simd_load_real\|simd_store_real\|scalar_load_real\|scalar_store_real\|PREC_SIZEOF_REAL" \
  src/backends/cpu/kernel_sources/
```
Expected: zero results (outside any transitional-alias comment).

---

### Step 7C.6: Update `src/backends/cpu/type_mapping.py`

**Governing authority:** ADR-022 §7.2  
**File:** `src/backends/cpu/type_mapping.py`

Replace `get_numpy_dtype(precision: PrecisionConfig) -> np.dtype` with three role-specific functions:

```python
def get_storage_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.storage_dtype

def get_compute_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.compute_dtype

def get_state_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.state_dtype
```

Update all callers within `src/backends/cpu/` that used `get_numpy_dtype(precision)` to call the role-appropriate function. For buffer allocators that previously accepted a single `dtype`, separate allocator instances are constructed per precision role when the plan requires heterogeneous allocation (i.e., when `storage_dtype != state_dtype`).

---

### Step 7C.7: Update CPU FFI layer (`_ffi_types.py`, `_dispatch_table.py`)

**Governing authority:** ADR-023 §4.3  
**Files:** `src/backends/cpu/_ffi_types.py`, `src/backends/cpu/_dispatch_table.py`

#### `_ffi_types.py`

Update all `ctypes.Structure` class references that embed a precision suffix. Old suffixes `_fp32` and `_fp16` are replaced with `_s32x32` and `_s16x16` respectively. Add `_s16x32` structs for the mixed configuration. Field types in the ctypes structures must mirror the updated `DECLARE_PRECISION_STRUCTS` layout from Step 7C.2 — `ctypes.c_float` for compute-role fields, `ctypes.c_float` for state-role fields in the s32x32 and s16x32 variants, `ctypes.c_uint16` (with appropriate `half`-compatible interpretation) for `_Float16` state-role fields in s16x16.

Remove the transitional name aliases added to `cpu_kernels.h` in Step 7C.2.4 once this file is updated.

#### `_dispatch_table.py`

The dispatch table currently selects a struct suffix based on a single dtype. Update to select based on the `(storage_dtype, state_dtype)` pair:

```python
def _select_suffix(precision: PrecisionConfig) -> str:
    s = precision.storage_dtype
    t = precision.state_dtype
    if s == np.dtype(np.float32) and t == np.dtype(np.float32):
        return "s32x32"
    elif s == np.dtype(np.float16) and t == np.dtype(np.float16):
        return "s16x16"
    elif s == np.dtype(np.float16) and t == np.dtype(np.float32):
        return "s16x32"
    else:
        raise ValueError(f"No CPU kernel instantiation for storage={s}, state={t}")
```

---

### Step 7C.8: Rebuild CPU shared library

Per the AGENTS.md build discipline, any edit to `src/backends/cpu/kernel_sources/` requires a rebuild:

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir 2>&1 | tee /tmp/phase7c_cpu_build.txt
grep -E "error:" /tmp/phase7c_cpu_build.txt | head -20
```

Also rebuild the ASAN variant if used in CI:

```bash
ninja -C builddir-asan 2>&1 | tee /tmp/phase7c_asan_build.txt
grep -E "error:" /tmp/phase7c_asan_build.txt | head -20
```

Fix all compilation errors before proceeding to the Vulkan steps.

---

## 4. Task Breakdown — Vulkan Backend

---

### Step 7C.9: Update `common.glsl`

**Governing authority:** ADR-023 §3.2  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Four additions in order:

#### 7C.9.1: Conditional FP16 extension guard

Add before the existing `#extension` declarations (or at the top of the file after the `#version` directive):

```glsl
// Conditionally enable FP16 extension when a role type requires it.
// Injected by glslc -DENABLE_FP16_EXTENSION=1 when any role is float16_t.
#ifdef ENABLE_FP16_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
```

#### 7C.9.2: Precision-role type macros

Add immediately after the specialization constant declarations:

```glsl
// ── Precision-Role Type Macros (ADR-023 §3.2) ─────────────────────────
// STORAGE_FLOAT and STATE_FLOAT are injected by the build system via
// glslc -D flags. Default to float (FP32) when not injected.
// When STORAGE_FLOAT == float, WIDEN_STORAGE/NARROW_STORAGE are identity
// operations eliminated by the SPIR-V compiler.
#ifndef STORAGE_FLOAT
#define STORAGE_FLOAT float
#endif
#ifndef STATE_FLOAT
#define STATE_FLOAT float
#endif

#define WIDEN_STORAGE(x)   float(x)
#define NARROW_STORAGE(x)  STORAGE_FLOAT(x)
#define WIDEN_STATE(x)     float(x)
#define NARROW_STATE(x)    STATE_FLOAT(x)
```

#### 7C.9.3: Rename `_cross_subgroup_scratch` → `_compute_scratch`

Find all occurrences of `_cross_subgroup_scratch` in `common.glsl` and rename to `_compute_scratch`. Update the declaration comment:

```glsl
// Compute-role scratch for cross-subgroup bridge.
// Always float (COMPUTE_TYPE = float on Vulkan backend, invariant per ADR-023 §3.3).
shared float _compute_scratch[32];
```

Verify no other `.comp` file references `_cross_subgroup_scratch` directly (the array is defined in `common.glsl` and should only be used through the `workgroup_reduce_add`/`workgroup_reduce_max` functions defined there):

```bash
grep -rn "_cross_subgroup_scratch" src/backends/vulkan/kernel_sources/
```

Expected: zero results after the rename (the helpers using it are in `common.glsl` itself).

#### 7C.9.4: Precision boundary helper functions

Add after the existing reduction helper functions in `common.glsl`:

```glsl
// ── Precision Boundary Helpers (ADR-023 §3.2.4) ───────────────────────
// Value-semantics wrappers that cross the storage/state → compute boundary.
// When STORAGE_FLOAT == float, these are identity operations. The SPIR-V
// compiler eliminates them. One codepath — no #ifdef on type equality.

float read_storage(STORAGE_FLOAT val) { return WIDEN_STORAGE(val); }
STORAGE_FLOAT write_storage(float val) { return NARROW_STORAGE(val); }

float read_state(STATE_FLOAT val) { return WIDEN_STATE(val); }
STATE_FLOAT write_state(float val) { return NARROW_STATE(val); }
```

---

### Step 7C.10: Migrate all `.comp` shaders

**Governing authority:** ADR-023 §3.4  
**Files:** `src/backends/vulkan/kernel_sources/*.comp` (all files with storage/state-role buffer bindings)

The shader list and which need changes (from the `kernel_sources/` directory listing):

| Shader | Has storage/state-role buffers? | Change needed |
|:---|:---|:---|
| `forward_pass.comp` | Yes | Yes |
| `render_logits_chunk.comp` | Yes | Yes |
| `compute_probs_loss_cce_chunk.comp` | Yes | Yes |
| `compute_probs_loss_bce_chunk.comp` | Yes | Yes |
| `calculate_module_param_grads_chunk.comp` | Yes | Yes |
| `backprop_error_to_hidden_chunk.comp` | Yes | Yes |
| `calculate_chunk_temp_gradients.comp` | Yes | Yes |
| `clip_partial_gradients.comp` | Yes (storage only) | Yes |
| `gather_and_permute_grad_hidden_activations.comp` | Yes (storage only) | Yes |
| `aggregate_partials.comp` | Yes | Yes |
| `clip_intermediate_grad.comp` | Yes | Yes |
| `stabilize_and_reduce_grad_hidden_activations.comp` | Yes | Yes |
| `backprop_shared_weights_chunk.comp` | Yes | Yes |
| `backprop_shared_biases_chunk.comp` | Yes | Yes |
| `clip_shared_gradients_chunk.comp` | Yes (storage only) | Yes |
| `adam_update.comp` | Yes (state) | Yes |
| `clamp_temperatures.comp` | Yes (state) | Yes |
| `normalize_gradients.comp` | No (all compute-role) | **No change needed** |

For every shader that needs changes, apply the following three-step transformation:

**1. Buffer layout declarations:** Replace `float data[]` in SSBO binding blocks with `STORAGE_FLOAT data[]` (storage-role) or `STATE_FLOAT data[]` (state-role). Compute-role and integer SSBO bindings keep `float data[]` or `uint`/`int` as appropriate.

Example (`forward_pass.comp`):
```glsl
// Before:
layout(set = 0, binding = 0) readonly buffer InputBuf   { float data[]; } src_input;
layout(set = 0, binding = 2) readonly buffer WeightsBuf { float data[]; } src_weights;
layout(set = 0, binding = 4) buffer HiddenBuf           { float data[]; } dest_hidden;

// After:
layout(set = 0, binding = 0) readonly buffer InputBuf   { STORAGE_FLOAT data[]; } src_input;
layout(set = 0, binding = 2) readonly buffer WeightsBuf { STATE_FLOAT   data[]; } src_weights;
layout(set = 0, binding = 4) buffer HiddenBuf           { STORAGE_FLOAT data[]; } dest_hidden;
```

**2. Load sites:** Every direct array read from a storage-role SSBO wraps through `read_storage()`; state-role reads through `read_state()`.

```glsl
// Before:
float mask_val = src_sample_mask.data[b_idx];
float bias_val = src_biases.data[h_idx];

// After:
float mask_val = read_storage(src_sample_mask.data[b_idx]);
float bias_val = read_state(src_biases.data[h_idx]);
```

**3. Store sites:** Every write to a storage-role SSBO wraps through `write_storage()`; state-role writes through `write_state()`.

```glsl
// Before:
dest_hidden.data[out_idx] = accum;
src_weights.data[w_idx] = new_weight;  // (adam_update: in-place)

// After:
dest_hidden.data[out_idx] = write_storage(accum);
src_weights.data[w_idx]   = write_state(new_weight);
```

**4. No changes to:** `shared float` scratch arrays, push constants (always `float`), integer buffers (`uint`/`int`), subgroup operations, loop bounds, index arithmetic.

Buffer role assignments per shader follow ADR-021 §1.4 tables exactly — the same role tables used in Phase 7A and 7B.

---

### Step 7C.11: Update `src/backends/vulkan/type_mapping.py`

**Governing authority:** ADR-022 §7.3  
**File:** `src/backends/vulkan/type_mapping.py`

Replace the single-axis functions with role-split equivalents:

```python
def get_storage_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.storage_dtype

def get_compute_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.compute_dtype

def get_state_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.state_dtype

def get_storage_element_size(precision: PrecisionConfig) -> int:
    return precision.storage_dtype.itemsize

def get_storage_is_half(precision: PrecisionConfig) -> int:
    return 1 if precision.storage_dtype == np.dtype(np.float16) else 0

def get_compute_is_half(precision: PrecisionConfig) -> int:
    return 1 if precision.compute_dtype == np.dtype(np.float16) else 0
```

Remove `get_numpy_dtype(precision)`, `get_element_size(precision)`, and `get_specialization_scalar_is_half(precision)`. Update all callers within `src/backends/vulkan/`.

---

### Step 7C.12: Update Vulkan `meson.build` for multi-variant SPIR-V compilation

**Governing authority:** ADR-023 §4.1  
**File:** `src/backends/vulkan/meson.build`

The current `meson.build` calls `glslc` once per `.comp` file. Update to produce three SPIR-V variants for every shader that has storage-role or state-role typed buffer bindings (all shaders except `normalize_gradients.comp`) using the following precision flag sets:

| Variant suffix | `glslc` flags |
|:---|:---|
| `_fp32` | *(no additional flags — `STORAGE_FLOAT` and `STATE_FLOAT` default to `float`)* |
| `_s16fp32` | `-DSTORAGE_FLOAT=float16_t -DSTATE_FLOAT=float -DENABLE_FP16_EXTENSION=1` |
| `_fp16` | `-DSTORAGE_FLOAT=float16_t -DSTATE_FLOAT=float16_t -DENABLE_FP16_EXTENSION=1` |

For `normalize_gradients.comp` (all compute-role buffers): only the `_fp32` variant is built (or equivalently: no precision flags — the single build produces correct output for all `PrecisionConfig` factories).

The SPIR-V output file naming convention:
- `forward_pass_fp32.spv`
- `forward_pass_s16fp32.spv`
- `forward_pass_fp16.spv`

Implement this in Meson using a `foreach` loop over the variant list:

```meson
precision_variants = [
  ['_fp32',    []],
  ['_s16fp32', ['-DSTORAGE_FLOAT=float16_t', '-DSTATE_FLOAT=float', '-DENABLE_FP16_EXTENSION=1']],
  ['_fp16',    ['-DSTORAGE_FLOAT=float16_t', '-DSTATE_FLOAT=float16_t', '-DENABLE_FP16_EXTENSION=1']],
]

role_bearing_shaders = [
  'forward_pass',
  'render_logits_chunk',
  # ... (all shaders except normalize_gradients)
]

foreach variant : precision_variants
  suffix = variant[0]
  extra_flags = variant[1]
  foreach shader_name : role_bearing_shaders
    custom_target(
      shader_name + suffix + '.spv',
      input  : shader_name + '.comp',
      output : shader_name + suffix + '.spv',
      command: ['glslc', '--target-env=vulkan1.2',
                '-I', meson.current_source_dir(),
                extra_flags,
                '@INPUT@', '-o', '@OUTPUT@'],
      install: false,
    )
  endforeach
endforeach

# normalize_gradients: single variant only
custom_target('normalize_gradients_fp32.spv', ...)
```

---

### Step 7C.13: Update `_pipeline_cache.py` for variant selection

**Governing authority:** ADR-023 §4.1  
**File:** `src/backends/vulkan/_pipeline_cache.py`

The pipeline cache currently loads a single SPIR-V path per kernel name. Update the path-resolution logic to select the variant based on `PrecisionConfig`:

```python
def _spv_variant_suffix(precision: PrecisionConfig) -> str:
    s = precision.storage_dtype
    t = precision.state_dtype
    if s == np.dtype(np.float32) and t == np.dtype(np.float32):
        return "_fp32"
    elif s == np.dtype(np.float16) and t == np.dtype(np.float32):
        return "_s16fp32"
    elif s == np.dtype(np.float16) and t == np.dtype(np.float16):
        return "_fp16"
    else:
        raise ValueError(f"No Vulkan SPIR-V variant for storage={s}, state={t}")
```

For kernels whose SPIR-V has no precision variant (i.e., `normalize_gradients`), the suffix is always `"_fp32"` regardless of `PrecisionConfig`. Maintain an explicit set or predicate for this case.

---

## 5. Task Breakdown — Final Alias Removal & Verification

---

### Step 7C.14: Remove transitional `SCALAR_TYPE` aliases from Meson and `type_mapping.py`

**Governing authority:** ADR-021 §3 (completion criterion)  
**Files:** `src/backends/opencl/type_mapping.py`, and any `meson.build` that injects the aliases

Remove the transitional alias block:

```python
# REMOVE THIS BLOCK:
flags += [
    "-DSCALAR_TYPE=COMPUTE_TYPE",
    "-DSCALAR_IS_HALF=COMPUTE_TYPE_IS_HALF",
    "-DSCALAR_ZERO=COMPUTE_ZERO",
]
```

**Gate condition for removal (all must pass):**

```bash
# 1. No SCALAR_TYPE in any OpenCL kernel source
grep -r "SCALAR_TYPE" architectures/averaging_ensembled_classifier/kernels/ && echo "FAIL" || echo "OK"

# 2. No SCALAR_TYPE in any backend source directory
grep -r "SCALAR_TYPE" architectures/averaging_ensembled_classifier/src/backends/ && echo "FAIL" || echo "OK"

# 3. No simd_load_real in CPU kernel sources
grep -r "simd_load_real\|simd_store_real" \
  architectures/averaging_ensembled_classifier/src/backends/cpu/kernel_sources/ && echo "FAIL" || echo "OK"

# 4. No REAL_T in CPU kernel sources (outside comments)
grep -r "\bREAL_T\b" \
  architectures/averaging_ensembled_classifier/src/backends/cpu/kernel_sources/ && echo "FAIL" || echo "OK"
```

All four commands must print `OK`. Do not remove until all gates pass — the aliases protect unmigrated files from compilation failures.

---

### Step 7C.15: Add multi-configuration Tier 2 test coverage

**Governing authority:** ADR-020 §4.6, §2.6; ADR-022 §9.3; ADR-023 §6 item 4  

Extend Tier 2 tests to parameterise over all three `PrecisionConfig` factories. Concretely:

#### Tests to extend

Any test that:
- Constructs a `PrecisionConfig` instance, or
- Constructs a `ModelSpec` with a precision argument, or
- Runs an end-to-end training/inference step

...must be parameterised to run with `PrecisionConfig.float32()`, `PrecisionConfig.float16()`, and `PrecisionConfig.mixed_f16_f32()`.

#### Numerical tolerance strategy

- For tests that compare numeric values (loss, parameter norms, gradient magnitudes): use `compute_dtype`-level tolerance, not `storage_dtype`-level tolerance.
  - `PrecisionConfig.float32()`: tolerance derived from `np.finfo(np.float32).eps`
  - `PrecisionConfig.float16()`: tolerance derived from `np.finfo(np.float16).eps`
  - `PrecisionConfig.mixed_f16_f32()`: tolerance derived from `np.finfo(np.float32).eps` (compute role is FP32)

#### "The Alchemist" scenario (ADR-020 §2.6)

Create a dedicated test function (e.g., `test_alchemist_mixed_precision_fidelity`) that:
1. Runs the same training task (same model, same data, same hyperparams) with `float32()` and `mixed_f16_f32()`.
2. Asserts that the mixed-precision run's final loss value is within FP32-compute tolerance of the FP32 baseline.
3. Asserts that storage buffer allocation sizes in the mixed-precision plan are half those of the FP32 plan (FP16 storage = 2 bytes vs FP32 = 4 bytes per element).
4. Asserts that the `StabilizationPolicy` safety ceiling in the mixed-precision plan is derived from `compute_fp_format_max` (FP32 max ≈ 3.4e38), not from `storage_fp_format_max` (FP16 max ≈ 65504).

#### Vulkan variant coverage

For the Vulkan backend's Tier 2 tests, assert that the pipeline cache loads distinct SPIR-V paths for `PrecisionConfig.float32()` vs `PrecisionConfig.mixed_f16_f32()`:

```python
fp32_path = pipeline_cache.get_spv_path("forward_pass", PrecisionConfig.float32())
mixed_path = pipeline_cache.get_spv_path("forward_pass", PrecisionConfig.mixed_f16_f32())
assert fp32_path != mixed_path
assert fp32_path.endswith("_fp32.spv")
assert mixed_path.endswith("_s16fp32.spv")
```

---

### Step 7C.16: Full migration completion verification

Run the complete migration completion criterion from ADR-021 §5 and ADR-023 §6 item 3:

```bash
# ADR-021 §5 criteria
echo "=== Kernel source migration ==="
grep -r SCALAR_TYPE architectures/averaging_ensembled_classifier/kernels/ \
  && echo "FAIL: SCALAR_TYPE in kernels/" || echo "OK: kernels/ clean"

grep -r SCALAR_TYPE architectures/averaging_ensembled_classifier/src/backends/ \
  && echo "FAIL: SCALAR_TYPE in backends/" || echo "OK: backends/ clean"

# ADR-023 §6 item 3 criteria
echo "=== CPU backend migration ==="
grep -r simd_load_real \
  architectures/averaging_ensembled_classifier/src/backends/cpu/ \
  && echo "FAIL: simd_load_real present" || echo "OK"

grep -r "\bREAL_T\b" \
  architectures/averaging_ensembled_classifier/src/backends/cpu/kernel_sources/ \
  && echo "FAIL: REAL_T present" || echo "OK"

grep -r "\bfp64\b" \
  architectures/averaging_ensembled_classifier/src/backends/cpu/ \
  && echo "FAIL: fp64 variant present" || echo "OK (check comments)"

echo "=== Vulkan backend migration ==="
# No unresolved 'float data[]' for storage/state-role SSBOs
# (This check requires knowledge of which shaders have role-typed SSBOs;
#  manual review or a test assertion on the pipeline cache variant paths
#  is more reliable than grep here.)

echo "=== Full test suite ==="
cd architectures/averaging_ensembled_classifier
python -m pytest src/tests/ -q 2>&1 | tee /tmp/phase7c_final_tests.txt
grep -E "passed|failed|error" /tmp/phase7c_final_tests.txt
```

All criteria must be `OK` or `passed`. Any failure is a blocker before the phase is marked complete.

---

## 6. `.inc` Template Migration Checklist

For each of the 6 `.inc` files:

- [ ] All reads from `STORAGE_T*` struct fields → `simd_load_storage` / `scalar_load_storage`
- [ ] All writes to `STORAGE_T*` struct fields → `simd_store_storage` / `scalar_store_storage`
- [ ] All reads from `STATE_T*` struct fields → `simd_load_state` / `scalar_load_state`
- [ ] All writes to `STATE_T*` struct fields → `simd_store_state` / `scalar_store_state`
- [ ] All reads/writes to `float*` (compute-role) fields → direct dereference, no macro
- [ ] All stack `float` scratch arrays → direct access, unchanged
- [ ] `PREC_SIZEOF_REAL` → `PREC_SIZEOF_STORAGE` where used in padding calculations
- [ ] Post-edit: `grep simd_load_real <file>` returns zero results
- [ ] Post-edit: `grep REAL_T <file>` returns zero results

---

## 7. `.comp` Shader Migration Checklist

For each of the 17 shaders requiring changes (all except `normalize_gradients.comp`):

- [ ] Storage-role SSBO bindings: `float data[]` → `STORAGE_FLOAT data[]`
- [ ] State-role SSBO bindings: `float data[]` → `STATE_FLOAT data[]`
- [ ] Compute-role SSBO bindings: `float data[]` — **unchanged**
- [ ] All reads from storage-role SSBOs wrapped in `read_storage()`
- [ ] All reads from state-role SSBOs wrapped in `read_state()`
- [ ] All writes to storage-role SSBOs wrapped in `write_storage()`
- [ ] All writes to state-role SSBOs wrapped in `write_state()`
- [ ] `shared float` scratch arrays — **unchanged**
- [ ] Push constants — **unchanged**
- [ ] Integer (`uint`/`int`) SSBO bindings — **unchanged**
- [ ] Post-edit: `glslc --target-env=vulkan1.2 <file>` compiles without error for all three variant flag sets

---

## 8. Cross-Backend Summary

| Backend | Precision boundary mechanism | `COMPUTE_TYPE` | Build-time parameterisation |
|:---|:---|:---|:---|
| **OpenCL** | `load_storage()`/`store_storage()`/`load_state()`/`store_state_update()` inline functions in `kernels.cl.h`; `vload_half`/`vstore_half` when `STORAGE_TYPE_IS_HALF` | `COMPUTE_TYPE` via `-D` flag (typically `float`) | `-DSTORAGE_TYPE`, `-DCOMPUTE_TYPE`, `-DSTATE_TYPE`, `-DSTORAGE_TYPE_IS_HALF`, `-DCOMPUTE_TYPE_IS_HALF` |
| **CPU** | `simd_load_storage()`/`simd_store_storage()`/`simd_load_state()`/`simd_store_state()` macros via `cpu_precision.h`; two-axis `STORAGE_SUFFIX`/`STATE_SUFFIX` dispatch; `cpu_compute_t = float` invariant | `float` — fixed CPU backend constant | `DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)` per variant in `cpu_kernels.h` |
| **Vulkan** | `read_storage()`/`write_storage()`/`read_state()`/`write_state()` GLSL helpers in `common.glsl`; `STORAGE_FLOAT`/`STATE_FLOAT` macros via `glslc -D`; `float16_t` SSBO elements under FP16 configurations | `float` — fixed Vulkan backend constant | Three SPIR-V variants per storage/state-role-bearing shader: `_fp32`, `_s16fp32`, `_fp16` |

The "one codepath" invariant (ADR-020 §2.3) holds in all three backends: conversion helpers are always syntactically present; the compiler/SPIR-V optimizer eliminates them when types are equal.

---

## 9. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| `_Float16` unavailable on target CPU build toolchain | Low | `cpu_kernels.h` already uses `_Float16`; the s16x16/s16x32 instantiations only compile if the toolchain supports it. Add a Meson `cc.has_type('_Float16')` check and conditional compilation guard. |
| `GL_EXT_shader_explicit_arithmetic_types_float16` not supported by the target Vulkan driver | Medium | Gate the `_s16fp32` and `_fp16` Vulkan SPIR-V artifacts behind a runtime capability check in `context.py`. Fall back to `_fp32` variant if FP16 storage extension is absent. |
| GLSL `float16_t` in a buffer layout member not accepted by the installed `glslc` version | Low | Verify with `glslc --version`; require at minimum `glslc` from SDK version 1.2.182. Add to the CI prerequisites check. |
| CPU struct ABI breaks Python `ctypes` binding silently (field byte offsets shift) | Medium | Add an assertion in `_ffi_types.py` that `ctypes.sizeof(ForwardPassArgs_s32x32) == <expected>`. Run before and after to confirm. |
| `.inc` file has a SIMD path that uses `REAL_T` directly (e.g., `simd_vec_of_real`) rather than through the load/store macros | Medium | Audit with `grep -n "REAL_T" src/backends/cpu/kernel_sources/*.inc` before Step 7C.5. Any direct `REAL_T` usage (not inside a load/store macro implementation) must be role-typed explicitly. |
| Transitional `simd_load_real` aliases in `cpu_precision.h` are removed before all `.inc` files are migrated | Low | Remove aliases only at the end of Step 7C.5 after the post-migration grep confirms zero `simd_load_real` occurrences. |
| "The Alchemist" test fails due to FP16 storage rounding affecting convergence at narrow model widths | Medium | The mixed-precision configuration's test tolerance is derived from `compute_dtype` (FP32). The loss comparison is a convergence-tracking assertion, not bit-exact. If the test fails, investigate whether the storage→compute widening is correctly applied at every load site (a missed `load_storage()` call can trigger this). |
| `normalize_gradients.comp` is accidentally treated as role-bearing and gets spurious `STORAGE_FLOAT` annotations | Low | The role table in ADR-021 §1.4 explicitly lists its only buffer as `"compute"`. The `.comp` shader migration checklist marks it as "no change needed". |
