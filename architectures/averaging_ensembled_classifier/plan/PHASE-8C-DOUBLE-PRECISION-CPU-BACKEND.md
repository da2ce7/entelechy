# Phase 8C: Double Precision — CPU Backend Three-Axis Migration

**Status: ✅ COMPLETED**  
**Phase:** 8C of 8  
**Prerequisite:** Phase 8B complete and rollback gate passed.  
**Objective:** Migrate the CPU backend from the two-axis precision scheme (`STORAGE_T × STATE_T`, suffix `s{s}x{x}`) to the three-axis scheme (`STORAGE_T × COMPUTE_T × STATE_T`, suffix `s{s}c{c}x{x}`). Remove the `cpu_compute_t = float` invariant. Extend `DECLARE_PRECISION_STRUCTS` to accept three type parameters. Instantiate all 11 valid precision combinations. Add scalar FP64 load/store variants to `cpu_precision.h`. Update the FFI layer and dispatch table. Rebuild the shared library. The phase ends when the CPU library compiles cleanly with all 11 instantiations and existing FP32/FP16-pathway tests pass.  
**Governing ADR:** ADR-024 §4  
**Rollback gate:** All Tier 2 tests that exercise the CPU backend pass for `PrecisionConfig.float32()`, `.float16()`, `.mixed_f16_f32()`. `PrecisionConfig.float32()` results are bit-for-bit identical to the Phase 8B baseline. The CPU shared library (`libcpu_kernels.so`) exports all 11 suffixed symbol sets. `grep -r "cpu_compute_t" src/backends/cpu/kernel_sources/` returns zero results.  
**Dependencies:** Phase 8B complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 8C.1: Update `cpu_precision.h` — add FP64 scalar load/store and COMPUTE_SUFFIX axis](#step-8c1-update-cpu_precisionh--add-fp64-scalar-loadstore-and-compute_suffix-axis)
   - [Step 8C.2: Remove `cpu_compute_t` typedef from `cpu_kernels.h`](#step-8c2-remove-cpu_compute_t-typedef-from-cpu_kernelsh)
   - [Step 8C.3: Extend `DECLARE_PRECISION_STRUCTS` to three types](#step-8c3-extend-declare_precision_structs-to-three-types)
   - [Step 8C.4: Update `cpu_kernels.c` — all 11 instantiation blocks](#step-8c4-update-cpu_kernelsc--all-11-instantiation-blocks)
   - [Step 8C.5: Migrate `.inc` template files to use `COMPUTE_T`](#step-8c5-migrate-inc-template-files-to-use-compute_t)
   - [Step 8C.6: Update `cpu_simd.h` — FP64 SIMD note](#step-8c6-update-cpu_simdh--fp64-simd-note)
   - [Step 8C.7: Update `src/backends/cpu/type_mapping.py`](#step-8c7-update-srcbackendscputype_mappingpy)
   - [Step 8C.8: Update `src/backends/cpu/_ffi_types.py`](#step-8c8-update-srcbackendscpu_ffi_typespy)
   - [Step 8C.9: Update `src/backends/cpu/_dispatch_table.py`](#step-8c9-update-srcbackendscpu_dispatch_tablepy)
   - [Step 8C.10: Rebuild CPU shared library](#step-8c10-rebuild-cpu-shared-library)
   - [Step 8C.11: Validate rollback gate](#step-8c11-validate-rollback-gate)
4. [Three-Axis Suffix Enumeration](#4-three-axis-suffix-enumeration)
5. [`.inc` Template Migration Checklist](#5-inc-template-migration-checklist)
6. [Risk Register](#6-risk-register)

---

## 1. Scope & Constraints

### In scope

- **`cpu_precision.h`:** Add a third axis `COMPUTE_SUFFIX` to the macro dispatch system. Add scalar-only FP64 load/store functions for state and storage roles: `scalar_load_state_fp64`, `scalar_store_state_fp64`, `scalar_load_storage_fp64`, `scalar_store_storage_fp64`. No SIMD FP64 variants (see rationale below). Add `PREC_SIZEOF_COMPUTE` macro.
- **`cpu_kernels.h`:** Remove `typedef float cpu_compute_t`. Extend `DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)` to `DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, COMPUTE_T, STATE_T)`. Retype all struct fields that were previously `float` (compute-role) to `COMPUTE_T`. Update header commentary.
- **`cpu_kernels.c`:** Replace the three existing instantiation blocks (s16x16, s32x32, s16x32) with 11 instantiations using the three-axis suffix `s{s}c{c}x{x}`. The three existing configurations become `s16c32x16`, `s32c32x32`, `s16c32x32`.
- **`.inc` templates (6 files):** Replace all direct `float` usage for compute-role variables with `COMPUTE_T`. Replace `float` scratch arrays with `COMPUTE_T` scratch arrays. Update function return types from `float` to `COMPUTE_T` where applicable.
- **`type_mapping.py`:** No changes needed (already role-split).
- **`_ffi_types.py`:** Update `PRECISION_SUFFIXES` to the 11 three-axis suffixes. Add `c_double`/`POINTER(c_double)` entries to `PRECISION_C_TYPES` for variants where `COMPUTE_T = double` or `STATE_T = double`. Extend layout verification.
- **`_dispatch_table.py`:** Update `_select_suffix()` to derive from the `(storage_dtype, compute_dtype, state_dtype)` triple.
- **CPU Meson build:** Rebuild.

### Out of scope

- SIMD FP64 load/store functions. ADR-024 §4.3 rationale: the Adam kernel's element-wise access pattern does not benefit from FP64 SIMD at AVX2 width (4 × `double`). Scalar-only is the initial implementation. AVX-512 (8 × `double`) variants may be added later.
- Vulkan backend changes — deferred to Phase 8D.
- OpenCL `.cl.c` file changes — already type-generic via build symbols (Phase 8B).
- FP64 Tier 2 test execution on CPU — deferred to Phase 8E (test coverage). The CPU library must compile and export all symbols; tests exercising FP64 kernel correctness are Phase 8E.

### Key constraint: backward-compatible suffix renaming

Existing suffixes (`s16x16`, `s32x32`, `s16x32`) are renamed to three-axis equivalents (`s16c32x16`, `s32c32x32`, `s16c32x32`). All references in `_ffi_types.py`, `_dispatch_table.py`, and any test code must be updated atomically in this phase to prevent broken symbol resolution.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/backends/cpu/kernel_sources/cpu_precision.h` | Two-axis dispatch: `STORAGE_SUFFIX` / `STATE_SUFFIX`. `simd_load_storage`, `simd_store_storage`, `simd_load_state`, `simd_store_state`. `PREC_SIZEOF_STORAGE`, `PREC_SIZEOF_STATE`. FP32 and FP16 implementations only (`_fp32`, `_fp16` suffixes on the underlying `simd_load_real_*` functions). |
| `src/backends/cpu/kernel_sources/cpu_kernels.h` | `typedef float cpu_compute_t`. `DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)` — 3-parameter macro. All compute-role struct fields are `float*` or `float`. Three instantiations: `s16x16`, `s32x32`, `s16x32`. |
| `src/backends/cpu/kernel_sources/cpu_kernels.c` | Three instantiation blocks defining `STORAGE_T`, `STORAGE_SUFFIX`, `STATE_T`, `STATE_SUFFIX`, `PRECISION_SUFFIX`. |
| `src/backends/cpu/kernel_sources/phase_*.inc` (6 files) | All compute-role local variables and scratch are `float`. Arithmetic uses `float`. |
| `src/backends/cpu/_ffi_types.py` | `PRECISION_SUFFIXES = ("s16x16", "s32x32", "s16x32")`. `PRECISION_C_TYPES` maps suffixes to `(c_storage, c_storage_p, c_state, c_state_p)` — no compute-role type parameter. Compute fields use `c_float` / `POINTER(c_float)` unconditionally. |
| `src/backends/cpu/_dispatch_table.py` | `_select_suffix()` dispatches on `(storage_dtype, state_dtype)` only. |

---

## 3. Task Breakdown

---

### Step 8C.1: Update `cpu_precision.h` — add FP64 scalar load/store and COMPUTE_SUFFIX axis

**Governing authority:** ADR-024 §4.3  
**File:** `src/backends/cpu/kernel_sources/cpu_precision.h`

#### 8C.1.1: Add `COMPUTE_SUFFIX` macro axis

The including `.c` file will now define `COMPUTE_SUFFIX` (e.g., `fp32`, `fp64`) alongside `STORAGE_SUFFIX` and `STATE_SUFFIX`. Add:

```c
#undef PREC_SIZEOF_COMPUTE
#define PREC_SIZEOF_COMPUTE ((uint)sizeof(COMPUTE_T))
```

No `simd_load_compute` / `simd_store_compute` macros — compute-role buffers are accessed directly via `COMPUTE_T*` dereference, consistent with the existing design.

#### 8C.1.2: Add scalar FP64 load/store function implementations

Add scalar-only FP64 functions that widen `double↔float` for cross-role access:

```c
/* ── FP64 scalar load/store (ADR-024 §4.3) ── */

/* State-role FP64: load double from state buffer, return as COMPUTE_T */
static inline float scalar_load_real_fp64_to_f32(const double *ptr) {
    return (float)(*ptr);
}
static inline void scalar_store_real_f32_to_fp64(double *ptr, float val) {
    *ptr = (double)val;
}

/* Identity: when COMPUTE_T = double */
static inline double scalar_load_real_fp64(const double *ptr) {
    return *ptr;
}
static inline void scalar_store_real_fp64(double *ptr, double val) {
    *ptr = val;
}
```

The macro dispatch selects the correct function based on `STATE_SUFFIX`. When `STATE_SUFFIX = fp64`:

```c
#define scalar_load_state(ptr, idx)  scalar_load_real_fp64((ptr) + (idx))
#define scalar_store_state(ptr, idx, val) scalar_store_real_fp64((ptr) + (idx), (val))
```

The return type is `double`, which is then implicitly narrowed to `COMPUTE_T` if `COMPUTE_T = float`. This matches the C-style cast semantics from ADR-024 §3.2.

**Alternative (simpler):** Since the macro expansion produces `_PREC_CAT2(scalar_load_real, STATE_SUFFIX)`, and STATE_SUFFIX for FP64 will be `fp64`, the existing dispatcher automatically resolves to `scalar_load_real_fp64`. Add the `fp64`-suffixed implementations alongside the existing `fp32` and `fp16` implementations.

Similarly for `STORAGE_SUFFIX = fp64`:

```c
#define scalar_load_storage(ptr, idx)  scalar_load_real_fp64((ptr) + (idx))
// ... etc.
```

#### 8C.1.3: SIMD FP64 — placeholder error macros

For SIMD paths with FP64, add compile-time errors to prevent accidental use:

```c
/* No SIMD FP64 variants (ADR-024 §4.3 — deferred to AVX-512 initiative) */
static inline simd_float simd_load_real_fp64(const double *ptr) {
    (void)ptr;
    /* This function should never be called — FP64 buffers use scalar access */
    __builtin_unreachable();
}
```

Or simply do not define `simd_load_real_fp64` and let the compiler error if a SIMD path is reached for FP64 buffers. The `.inc` templates must use scalar paths for FP64 buffers — see Step 8C.5.

---

### Step 8C.2: Remove `cpu_compute_t` typedef from `cpu_kernels.h`

**Governing authority:** ADR-024 §4.2  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.h`

Remove:
```c
typedef float cpu_compute_t;
```

And update the header commentary:

```c
/* COMPUTE_T is now a per-instantiation parameter (ADR-024 §4.2).
 * The prior cpu_compute_t typedef is removed.
 * Each precision variant defines COMPUTE_T via the instantiation block
 * in cpu_kernels.c. */
```

---

### Step 8C.3: Extend `DECLARE_PRECISION_STRUCTS` to three types

**Governing authority:** ADR-024 §4.1  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.h`

Change the macro signature from:
```c
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)
```
to:
```c
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, COMPUTE_T, STATE_T)
```

Within the macro body, replace every `float` that represents a compute-role field with `COMPUTE_T`. Specifically:

| Struct field | Current type | New type | Role |
|:---|:---|:---|:---|
| `float* summed_grad` (normalize) | `float*` | `COMPUTE_T*` | compute |
| `const float* normalized_gradient` (adam) | `const float*` | `const COMPUTE_T*` | compute |
| `float* final_loss` (loss kernels) | `float*` | `COMPUTE_T*` | compute |
| `float* output` (misc compute outputs) | `float*` | `COMPUTE_T*` | compute |

Fields that are `uint`, `int`, or plain scalars used for dimensions/offsets remain unchanged.

**Verification:** After the macro change, a `grep -n "float\|cpu_compute_t" cpu_kernels.h` (excluding comments and `_Float16`) should show zero compute-role `float` references inside the `DECLARE_PRECISION_STRUCTS` macro body. Only `STORAGE_T`, `COMPUTE_T`, `STATE_T`, and integer types should appear.

---

### Step 8C.4: Update `cpu_kernels.c` — all 11 instantiation blocks

**Governing authority:** ADR-024 §4.1  
**File:** `src/backends/cpu/kernel_sources/cpu_kernels.c`

Replace the three existing instantiation blocks with 11. Each block defines `STORAGE_T`, `STORAGE_SUFFIX`, `COMPUTE_T`, `COMPUTE_SUFFIX`, `STATE_T`, `STATE_SUFFIX`, and `PRECISION_SUFFIX`, then includes `cpu_precision.h` and all six `.inc` files.

```c
/* ================================================================
 * All valid s{storage}c{compute}x{state} combinations (ADR-024 §4.1)
 * ================================================================ */

/* --- storage=16, compute=32 --- */

/* s16c32x16: FP16 storage, FP32 compute, FP16 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T       _Float16
#define STORAGE_SUFFIX  fp16
#define COMPUTE_T       float
#define COMPUTE_SUFFIX  fp32
#define STATE_T         _Float16
#define STATE_SUFFIX    fp16
#define PRECISION_SUFFIX s16c32x16
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c32x32: FP16 storage, FP32 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T       _Float16
#define STORAGE_SUFFIX  fp16
#define COMPUTE_T       float
#define COMPUTE_SUFFIX  fp32
#define STATE_T         float
#define STATE_SUFFIX    fp32
#define PRECISION_SUFFIX s16c32x32
#include "cpu_precision.h"
/* ... all six .inc files ... */
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c32x64: FP16 storage, FP32 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T       _Float16
#define STORAGE_SUFFIX  fp16
#define COMPUTE_T       float
#define COMPUTE_SUFFIX  fp32
#define STATE_T         double
#define STATE_SUFFIX    fp64
#define PRECISION_SUFFIX s16c32x64
#include "cpu_precision.h"
/* ... all six .inc files ... */
/* ... #undef block ... */
#pragma GCC diagnostic pop

/* --- storage=16, compute=64 --- */

/* s16c64x16: FP16 storage, FP64 compute, FP16 state */
/* s16c64x32: FP16 storage, FP64 compute, FP32 state */
/* s16c64x64: FP16 storage, FP64 compute, FP64 state */
/* (same pattern with COMPUTE_T=double, COMPUTE_SUFFIX=fp64) */

/* --- storage=32, compute=32 --- */

/* s32c32x32: FP32 storage, FP32 compute, FP32 state (reference — was s32x32) */
#define STORAGE_T       float
#define STORAGE_SUFFIX  fp32
#define COMPUTE_T       float
#define COMPUTE_SUFFIX  fp32
#define STATE_T         float
#define STATE_SUFFIX    fp32
#define PRECISION_SUFFIX s32c32x32
#include "cpu_precision.h"
/* ... all six .inc files ... */
/* ... #undef block ... */

/* s32c32x64: FP32 storage, FP32 compute, FP64 state */
/* s32c64x32: FP32 storage, FP64 compute, FP32 state */
/* s32c64x64: FP32 storage, FP64 compute, FP64 state */

/* --- storage=64 --- */

/* s64c64x64: FP64 storage, FP64 compute, FP64 state (uniform FP64 reference) */
#define STORAGE_T       double
#define STORAGE_SUFFIX  fp64
#define COMPUTE_T       double
#define COMPUTE_SUFFIX  fp64
#define STATE_T         double
#define STATE_SUFFIX    fp64
#define PRECISION_SUFFIX s64c64x64
#include "cpu_precision.h"
/* ... all six .inc files ... */
/* ... #undef block ... */
```

The full enumeration follows the table from ADR-024 §4.1 (11 valid combinations).

---

### Step 8C.5: Migrate `.inc` template files to use `COMPUTE_T`

**Governing authority:** ADR-024 §4.2  
**Files:** All 6 `phase_*.inc` files

For each `.inc` file, replace every `float` that represents a compute-role variable or scratch with `COMPUTE_T`:

1. **Local variables:** `float accum = 0.0f;` → `COMPUTE_T accum = (COMPUTE_T)0.0;`
2. **Scratch arrays:** `float scratch[SIMD_WIDTH];` → `COMPUTE_T scratch[SIMD_WIDTH];`
3. **Function return types:** If any helper functions are defined inline, their return types change from `float` to `COMPUTE_T`.
4. **Literal constants:** `0.0f` → `(COMPUTE_T)0.0` in compute-role contexts. For FP64, `0.0` without suffix is a `double` literal. Using `(COMPUTE_T)0.0` works for both FP32 (cast narrows, no-op for float) and FP64 (identity).
5. **Math library functions:** `expf()` → use `COMPUTE_T`-appropriate function. For FP32: `expf()`. For FP64: `exp()`. Strategy: use a macro `COMPUTE_EXP`, `COMPUTE_LOG`, etc., defined by `cpu_precision.h` based on `COMPUTE_SUFFIX`.

#### 8C.5.1: Compute-role math function macros

Add to `cpu_precision.h`:

```c
/* Compute-role math function dispatch (ADR-024 §4.2) */
#undef COMPUTE_EXP
#undef COMPUTE_LOG
#undef COMPUTE_SQRT
#undef COMPUTE_FABS
#undef COMPUTE_FMAX
#undef COMPUTE_FMIN

#if defined(COMPUTE_SUFFIX) && (COMPUTE_SUFFIX == fp64)
/* Workaround: token comparison doesn't work; use COMPUTE_T sizeof */
#endif

/* Use _Generic for type-safe dispatch (C11) */
#define COMPUTE_EXP(x)   _Generic((x), float: expf, double: exp)(x)
#define COMPUTE_LOG(x)    _Generic((x), float: logf, double: log)(x)
#define COMPUTE_SQRT(x)   _Generic((x), float: sqrtf, double: sqrt)(x)
#define COMPUTE_FABS(x)   _Generic((x), float: fabsf, double: fabs)(x)
#define COMPUTE_FMAX(x,y) _Generic((x), float: fmaxf, double: fmax)(x,y)
#define COMPUTE_FMIN(x,y) _Generic((x), float: fminf, double: fmin)(x,y)
```

This uses C11 `_Generic` to dispatch to the correct math function based on the type of `COMPUTE_T`. When `COMPUTE_T = float`, `_Generic` selects `expf`; when `COMPUTE_T = double`, it selects `exp`. Both GCC and Clang support C11 `_Generic`.

#### 8C.5.2: SIMD vs scalar path gating for FP64

The existing `.inc` files use SIMD paths for the main loops. When `COMPUTE_T = double`, the SIMD intrinsics (`simd_float`, `simd_load`, etc.) are `float`-typed and cannot hold `double` values. Two approaches:

**Approach A (recommended for initial FP64 support):** When `COMPUTE_T = double`, the `.inc` files fall through to the scalar tail path for all elements (effectively: `SIMD_WIDTH_COMPUTE = 1` for FP64). Gate the SIMD vectorized loop on `sizeof(COMPUTE_T) == sizeof(float)`:

```c
#if sizeof(COMPUTE_T) == sizeof(float)
    /* SIMD vectorized loop (existing code) */
    for (uint i = 0; i < count; i += SIMD_WIDTH) { ... }
#endif
    /* Scalar tail / FP64 full loop */
    for (uint i = simd_end; i < count; i++) { ... }
```

**Approach B (future — deferred):** Define `simd_double` as `__m256d` (AVX2) or `__m512d` (AVX-512) and implement FP64 SIMD paths.

For Phase 8C, use Approach A. The scalar path is already present in every `.inc` file as the "tail" loop. The change is: when `COMPUTE_T = double`, the SIMD-width loop processes zero elements and falls through entirely to scalar.

---

### Step 8C.6: Update `cpu_simd.h` — FP64 SIMD note

**Governing authority:** ADR-024 §6.1  
**File:** `src/backends/cpu/kernel_sources/cpu_simd.h`

No structural changes. Add a commentary note:

```c
/* FP64 SIMD is not implemented in this initial iteration (ADR-024 §4.3).
 * The simd_float / simd_int types are FP32-specific.
 * When COMPUTE_T = double, kernel templates use scalar paths exclusively.
 * Future: simd_double typedef and FP64 SIMD intrinsics (AVX-512). */
```

---

### Step 8C.7: Update `src/backends/cpu/type_mapping.py`

**Governing authority:** ADR-024 §4  
**File:** `src/backends/cpu/type_mapping.py`

No changes needed — the existing `get_storage_dtype`, `get_compute_dtype`, `get_state_dtype` functions return the dtype directly from `PrecisionConfig`. They naturally return `np.float64` for FP64 configs.

Verify no other functions in this file assume FP32-only.

---

### Step 8C.8: Update `src/backends/cpu/_ffi_types.py`

**Governing authority:** ADR-024 §4.1  
**File:** `src/backends/cpu/_ffi_types.py`

#### 8C.8.1: Update `PRECISION_SUFFIXES`

Replace:
```python
PRECISION_SUFFIXES = ("s16x16", "s32x32", "s16x32")
```
with:
```python
PRECISION_SUFFIXES = (
    "s16c32x16", "s16c32x32", "s16c32x64",
    "s16c64x16", "s16c64x32", "s16c64x64",
    "s32c32x32", "s32c32x64",
    "s32c64x32", "s32c64x64",
    "s64c64x64",
)
```

#### 8C.8.2: Update `PRECISION_C_TYPES`

Extend the mapping to include a compute-role type parameter:

```python
# (c_storage, c_storage_p, c_compute, c_compute_p, c_state, c_state_p) per suffix.
PRECISION_C_TYPES: dict[str, tuple[type, type, type, type, type, type]] = {
    "s16c32x16": (c_uint16, POINTER(c_uint16), c_float, c_float_p, c_uint16, POINTER(c_uint16)),
    "s16c32x32": (c_uint16, POINTER(c_uint16), c_float, c_float_p, c_float, POINTER(c_float)),
    "s16c32x64": (c_uint16, POINTER(c_uint16), c_float, c_float_p, c_double, POINTER(c_double)),
    "s16c64x16": (c_uint16, POINTER(c_uint16), c_double, POINTER(c_double), c_uint16, POINTER(c_uint16)),
    "s16c64x32": (c_uint16, POINTER(c_uint16), c_double, POINTER(c_double), c_float, POINTER(c_float)),
    "s16c64x64": (c_uint16, POINTER(c_uint16), c_double, POINTER(c_double), c_double, POINTER(c_double)),
    "s32c32x32": (c_float, c_float_p, c_float, c_float_p, c_float, POINTER(c_float)),
    "s32c32x64": (c_float, c_float_p, c_float, c_float_p, c_double, POINTER(c_double)),
    "s32c64x32": (c_float, c_float_p, c_double, POINTER(c_double), c_float, POINTER(c_float)),
    "s32c64x64": (c_float, c_float_p, c_double, POINTER(c_double), c_double, POINTER(c_double)),
    "s64c64x64": (c_double, POINTER(c_double), c_double, POINTER(c_double), c_double, POINTER(c_double)),
}
```

#### 8C.8.3: Update `make_precision_types()`

Update `make_precision_types()` to unpack six types instead of four:

```python
c_storage, c_storage_p, c_compute, c_compute_p, c_state, c_state_p = PRECISION_C_TYPES[suffix]
```

And use `c_compute_p` for compute-role struct fields where previously `c_float_p` was hardcoded.

---

### Step 8C.9: Update `src/backends/cpu/_dispatch_table.py`

**Governing authority:** ADR-024 §4.1  
**File:** `src/backends/cpu/_dispatch_table.py`

Update `_select_suffix()` to derive the three-axis suffix from the full `PrecisionConfig`:

```python
def _select_suffix(precision: PrecisionConfig) -> str:
    """Map (storage, compute, state) dtypes to a CPU kernel suffix."""
    _SUFFIX_MAP = {
        (np.float16, np.float32, np.float16): "s16c32x16",
        (np.float16, np.float32, np.float32): "s16c32x32",
        (np.float16, np.float32, np.float64): "s16c32x64",
        (np.float16, np.float64, np.float16): "s16c64x16",
        (np.float16, np.float64, np.float32): "s16c64x32",
        (np.float16, np.float64, np.float64): "s16c64x64",
        (np.float32, np.float32, np.float32): "s32c32x32",
        (np.float32, np.float32, np.float64): "s32c32x64",
        (np.float32, np.float64, np.float32): "s32c64x32",
        (np.float32, np.float64, np.float64): "s32c64x64",
        (np.float64, np.float64, np.float64): "s64c64x64",
    }
    key = (
        precision.storage_dtype.type,
        precision.compute_dtype.type,
        precision.state_dtype.type,
    )
    suffix = _SUFFIX_MAP.get(key)
    if suffix is None:
        raise ValueError(
            f"No CPU kernel instantiation for "
            f"storage={precision.storage_dtype}, "
            f"compute={precision.compute_dtype}, "
            f"state={precision.state_dtype}"
        )
    return suffix
```

---

### Step 8C.10: Rebuild CPU shared library

Per AGENTS.md build discipline:

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir 2>&1 | tee /tmp/phase8c_cpu_build.txt
grep -E "error:" /tmp/phase8c_cpu_build.txt | head -20
```

Also rebuild the ASAN variant:

```bash
ninja -C builddir-asan 2>&1 | tee /tmp/phase8c_asan_build.txt
grep -E "error:" /tmp/phase8c_asan_build.txt | head -20
```

Fix all compilation errors before proceeding.

---

### Step 8C.11: Validate rollback gate

```bash
cd architectures/averaging_ensembled_classifier
python -m pytest src/tests/ -q 2>&1 | tee /tmp/phase8c_tests.txt
grep -E "passed|failed|error" /tmp/phase8c_tests.txt

# Verify all 11 suffixes are exported
nm -D builddir/src/libcpu_kernels.so | grep "task_forward_pass_" | sort
# Expected: 11 entries (one per suffix)

# Verify cpu_compute_t is removed
grep -r "cpu_compute_t" src/backends/cpu/kernel_sources/ && echo "FAIL" || echo "OK"
```

---

## 4. Three-Axis Suffix Enumeration

Complete table from ADR-024 §4.1:

| Suffix | `STORAGE_T` | `COMPUTE_T` | `STATE_T` | Primary Use Case |
|:---|:---|:---|:---|:---|
| `s16c32x16` | `_Float16` | `float` | `_Float16` | `PrecisionConfig.float16()` on CPU |
| `s16c32x32` | `_Float16` | `float` | `float` | `PrecisionConfig.mixed_f16_f32()` |
| `s16c32x64` | `_Float16` | `float` | `double` | `PrecisionConfig.mixed_f16_f64_state()` |
| `s16c64x16` | `_Float16` | `double` | `_Float16` | FP64 compute validation, FP16 state |
| `s16c64x32` | `_Float16` | `double` | `float` | FP64 compute validation, FP32 state |
| `s16c64x64` | `_Float16` | `double` | `double` | FP64 compute + stability, FP16 bandwidth |
| `s32c32x32` | `float` | `float` | `float` | `PrecisionConfig.float32()` (reference) |
| `s32c32x64` | `float` | `float` | `double` | `PrecisionConfig.mixed_f32_f64_state()` |
| `s32c64x32` | `float` | `double` | `float` | FP64 compute validation |
| `s32c64x64` | `float` | `double` | `double` | `PrecisionConfig.mixed_f32_f64()` |
| `s64c64x64` | `double` | `double` | `double` | `PrecisionConfig.float64()` (reference) |

---

## 5. `.inc` Template Migration Checklist

For each of the 6 `.inc` files:

- [ ] All `float` local variables for compute-role values → `COMPUTE_T`
- [ ] All `float` scratch arrays → `COMPUTE_T`
- [ ] All `0.0f` in compute-role contexts → `(COMPUTE_T)0.0`
- [ ] All `expf` → `COMPUTE_EXP`, `logf` → `COMPUTE_LOG`, etc.
- [ ] SIMD vectorized loop gated on `sizeof(COMPUTE_T) == sizeof(float)`
- [ ] Scalar tail loop uses `COMPUTE_T` types
- [ ] Post-edit: `grep -n "\bfloat\b" <file>` shows zero compute-role `float` references (only struct type casts and parameter declarations remain as `float` when STORAGE_T/STATE_T happen to be float)
- [ ] Post-edit: `grep -n "expf\|logf\|sqrtf\|fabsf\|fmaxf\|fminf" <file>` shows zero results in compute-role code paths

---

## 6. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| Compile time increases significantly with 11 instantiations | Low | Each instantiation compiles six `.inc` files. The `.inc` files are template-included; modern compilers handle this efficiently. If needed, parallelize via Meson `threads:` option. Measure before and after. |
| FP64 scalar-only path is orders of magnitude slower than SIMD FP32 for the same data size | Expected | This is a known tradeoff (ADR-024 Consequences §negative). FP64 is for validation and stability, not throughput. Document performance characteristics in tests. |
| `_Float16 → double` conversions in `s16c64x64` trigger compiler warnings or undefined behaviour | Low | GCC and Clang correctly chain `_Float16 → float → double` or `_Float16 → double` directly. The `-Wfloat-conversion` diagnostic suppression (already present for FP16 blocks) covers this. |
| `_Generic` not supported by the CPU backend's target compiler | Very Low | The project already requires C11 (`_Float16` is C23, but GCC supports it as an extension with C11). `_Generic` is C11 standard. |
| Existing tests that hard-reference `s32x32` / `s16x16` / `s16x32` suffixes break | High | Atomic rename: all suffix references must be updated in the same commit (Steps 8C.4, 8C.8, 8C.9). Run the full test suite before committing. |
| `c_double` struct field alignment differs from `c_float`, causing ctypes layout mismatch | Medium | Add explicit layout verification assertions (already done via `_verify_layouts()` in `_ffi_types.py`). Test with the `get_struct_size_*` exports from the C library for every new suffix. |
