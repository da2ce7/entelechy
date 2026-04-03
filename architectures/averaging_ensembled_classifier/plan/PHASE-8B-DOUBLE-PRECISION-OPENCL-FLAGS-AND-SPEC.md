# Phase 8B: Double Precision — OpenCL Kernel Specification & Compiler Flags

**Status: NOT STARTED**  
**Phase:** 8B of 8  
**Prerequisite:** Phase 8A complete and rollback gate passed.  
**Objective:** Extend the OpenCL kernel specification (`kernels.cl.h`) and all OpenCL implementation files (`phase_*.cl.c`) with FP64 support. Add the `cl_khr_fp64` extension guard. Emit `_IS_DOUBLE` flags from the OpenCL type-mapping layer. Extend `COMPUTE_ZERO` for FP64. Update the precision boundary abstraction functions (`load_state`, `store_state`) to handle FP64↔FP32 narrowing/widening. The phase ends when `PrecisionConfig.float32()` produces identical Tier 2 test results (no regression), and the OpenCL compilation succeeds when invoked with FP64 flags.  
**Governing ADR:** ADR-024 §§2–3  
**Rollback gate:** All Tier 2 tests pass against `PrecisionConfig.float32()`. Compilation succeeds for all builddir variants that include the OpenCL backend. `grep -r "COMPUTE_TYPE_IS_DOUBLE\|STATE_TYPE_IS_DOUBLE" kernels/kernels.cl.h` returns the expected guard locations.  
**Dependencies:** Phase 8A complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 8B.1: Add `_IS_DOUBLE` build-symbol checks to `kernels.cl.h`](#step-8b1-add-_is_double-build-symbol-checks-to-kernelsclh)
   - [Step 8B.2: Add `cl_khr_fp64` extension guard](#step-8b2-add-cl_khr_fp64-extension-guard)
   - [Step 8B.3: Extend `COMPUTE_ZERO` macro for FP64](#step-8b3-extend-compute_zero-macro-for-fp64)
   - [Step 8B.4: Update precision boundary functions for FP64](#step-8b4-update-precision-boundary-functions-for-fp64)
   - [Step 8B.5: Update host stub definitions for FP64](#step-8b5-update-host-stub-definitions-for-fp64)
   - [Step 8B.6: Update `src/backends/opencl/type_mapping.py`](#step-8b6-update-srcbackendsopencltype_mappingpy)
   - [Step 8B.7: Rebuild and validate rollback gate](#step-8b7-rebuild-and-validate-rollback-gate)
4. [Risk Register](#4-risk-register)

---

## 1. Scope & Constraints

### In scope

- Adding mandatory build-symbol checks for `COMPUTE_TYPE_IS_DOUBLE` and `STATE_TYPE_IS_DOUBLE` in `kernels.cl.h`.
- Adding the `cl_khr_fp64` extension pragma, conditionally enabled when `COMPUTE_TYPE_IS_DOUBLE == 1` or `STATE_TYPE_IS_DOUBLE == 1`.
- Extending the `COMPUTE_ZERO` macro to handle `COMPUTE_TYPE = double` (`#define COMPUTE_ZERO 0.0`).
- Updating precision boundary abstraction implementations (`load_storage`, `store_storage`, `load_state`, `store_state_update`) in `kernels.cl.h` — these already use C-style casts via the role-type symbols; the casts naturally handle `double↔float` narrowing/widening. Verify and add explicit commentary.
- Updating the host/C++ stub definitions in `kernels.cl.h` to define `COMPUTE_TYPE_IS_DOUBLE=0` and `STATE_TYPE_IS_DOUBLE=0` for analysis builds.
- Updating `src/backends/opencl/type_mapping.py` `build_compiler_flags()` to emit:
  - `-DCOMPUTE_TYPE_IS_DOUBLE=1/0`
  - `-DSTATE_TYPE_IS_DOUBLE=1/0`
  - `-DCOMPUTE_TYPE=double` / `-DSTATE_TYPE=double` when FP64 roles active
  - `cl_khr_fp64` capability check note
- Updating `_dtype_to_cl_type()` to map `np.float64` → `"double"`.

### Out of scope

- No changes to `phase_*.cl.c` files. The existing OpenCL implementations use `COMPUTE_TYPE`, `STORAGE_TYPE`, `STATE_TYPE`, and precision boundary abstractions — these symbols resolve correctly to `double` when the build flags specify it. The implementations are type-generic by construction.
- CPU backend changes — deferred to Phase 8C.
- Vulkan backend changes — deferred to Phase 8D.
- Runtime `cl_khr_fp64` device capability check — deferred to Phase 8C/8D (the check belongs in the backend's discovery/capability layer).

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `kernels/kernels.cl.h` | Mandatory-symbol checks: `STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`, `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`, `SIMD_WIDTH`, `C_TILE_SIZE`, `NUMERICAL_STABILITY_EPSILON`. FP16 extension guard on `STORAGE_TYPE_IS_HALF` and `COMPUTE_TYPE_IS_HALF`. `COMPUTE_ZERO` macro handles `COMPUTE_TYPE_IS_HALF` and default (`float`). Host stubs define `COMPUTE_TYPE=float`, `COMPUTE_TYPE_IS_HALF=0`. Precision boundary functions use C-style casts via role-type symbols. |
| `src/backends/opencl/type_mapping.py` | `build_compiler_flags()` emits `STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`, `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`. `_dtype_to_cl_type()` maps `np.float32`→`"float"`, `np.float16`→`"half"`. No `np.float64` mapping. |

---

## 3. Task Breakdown

---

### Step 8B.1: Add `_IS_DOUBLE` build-symbol checks to `kernels.cl.h`

**Governing authority:** ADR-024 §2  
**File:** `kernels/kernels.cl.h`

Add two new mandatory-symbol checks immediately after the existing `COMPUTE_TYPE_IS_HALF` check:

```c
#ifndef COMPUTE_TYPE_IS_DOUBLE
#error "System Contract Violation: COMPUTE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif
#ifndef STATE_TYPE_IS_DOUBLE
#error "System Contract Violation: STATE_TYPE_IS_DOUBLE must be defined by the host build system."
#endif
```

---

### Step 8B.2: Add `cl_khr_fp64` extension guard

**Governing authority:** ADR-024 §3.1  
**File:** `kernels/kernels.cl.h`

After the existing `cl_khr_fp16` extension blocks, add:

```c
// Enable FP64 extension if using double precision in compute or state roles.
#if COMPUTE_TYPE_IS_DOUBLE || STATE_TYPE_IS_DOUBLE
#if !defined(cl_khr_fp64)
#error "FP64 extension (cl_khr_fp64) required for double precision but not supported by device"
#endif
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
```

---

### Step 8B.3: Extend `COMPUTE_ZERO` macro for FP64

**Governing authority:** ADR-024 §3.3  
**File:** `kernels/kernels.cl.h`

Replace the current `COMPUTE_ZERO` definition with a three-way guard:

```c
#if COMPUTE_TYPE_IS_DOUBLE
    #define COMPUTE_ZERO 0.0
#elif COMPUTE_TYPE_IS_HALF
    #define COMPUTE_ZERO ((COMPUTE_TYPE)0.0h)
#else
    #define COMPUTE_ZERO ((COMPUTE_TYPE)0.0f)
#endif
```

The mutual-exclusivity guarantee (ADR-024 §2 — `_IS_HALF` and `_IS_DOUBLE` cannot both be 1 for the same role) ensures that exactly one branch is taken.

---

### Step 8B.4: Update precision boundary functions for FP64

**Governing authority:** ADR-024 §3.2  
**File:** `kernels/kernels.cl.h`

The existing precision boundary functions (`load_storage`, `store_storage`, `load_state`, `store_state_update`) already use C-style casts via the role-type symbols:

```c
static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx) {
    return (COMPUTE_TYPE)buf[idx];
}
```

When `STATE_TYPE = double` and `COMPUTE_TYPE = float`, this becomes `(float)buf[idx]` — a valid double→float narrowing cast. When `STATE_TYPE = double` and `COMPUTE_TYPE = double`, it becomes `(double)buf[idx]` — an identity cast eliminated by the compiler.

**No code change is required** to the function bodies. Add a commentary block documenting the FP64 narrowing/widening semantics:

```c
// FP64 Precision Boundary Notes (ADR-024 §3.2):
// When STATE_TYPE = double and COMPUTE_TYPE = float:
//   load_state: double → float narrowing (precision loss accepted;
//               the value is about to enter lower-precision arithmetic)
//   store_state_update: float → double widening (no precision loss;
//                       the narrower compute value preserves all its bits)
// When STATE_TYPE = double and COMPUTE_TYPE = double:
//   Both casts are identity operations, eliminated by the compiler.
```

---

### Step 8B.5: Update host stub definitions for FP64

**Governing authority:** ADR-024 §2  
**File:** `kernels/kernels.cl.h`

In the host/C++ stub section (the `#else` branch for non-OpenCL analysis builds), add:

```c
#ifndef COMPUTE_TYPE_IS_DOUBLE
#define COMPUTE_TYPE_IS_DOUBLE 0
#endif
#ifndef STATE_TYPE_IS_DOUBLE
#define STATE_TYPE_IS_DOUBLE 0
#endif
```

---

### Step 8B.6: Update `src/backends/opencl/type_mapping.py`

**Governing authority:** ADR-024 §2, §3  
**File:** `src/backends/opencl/type_mapping.py`

#### 8B.6.1: Extend `_dtype_to_cl_type()`

Add `np.float64` → `"double"` to the mapping:

```python
mapping = {
    np.dtype(np.float32): "float",
    np.dtype(np.float16): "half",
    np.dtype(np.float64): "double",
}
```

#### 8B.6.2: Extend `build_compiler_flags()`

Add the two new `_IS_DOUBLE` flags:

```python
compute_is_double = 1 if precision.compute_dtype == np.dtype(np.float64) else 0
state_is_double = 1 if precision.state_dtype == np.dtype(np.float64) else 0

flags += [
    f"-DCOMPUTE_TYPE_IS_DOUBLE={compute_is_double}",
    f"-DSTATE_TYPE_IS_DOUBLE={state_is_double}",
]
```

#### 8B.6.3: Update `_epsilon_literal()` for FP64

Add FP64 handling — double literals have no suffix in OpenCL C:

```python
def _epsilon_literal(epsilon: float, is_half: int, is_double: int) -> str:
    if is_half:
        return str(epsilon)
    if is_double:
        return str(epsilon)  # No suffix for double literals in OpenCL C
    return f"{epsilon}f"
```

Update the call site to pass the new `is_double` parameter.

---

### Step 8B.7: Rebuild and validate rollback gate

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir 2>&1 | tee /tmp/phase8b_build.txt
grep -E "error:" /tmp/phase8b_build.txt | head -20
python -m pytest src/tests/ -q 2>&1 | tee /tmp/phase8b_tests.txt
grep -E "passed|failed|error" /tmp/phase8b_tests.txt
```

All existing tests must pass. No regressions. The build must succeed with the default (FP32) flags. FP64-specific compilation (exercised by the test that constructs a `PrecisionConfig.float64()` and generates flags) must produce valid flag strings.

---

## 4. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| OpenCL device used in CI lacks `cl_khr_fp64` support | Medium | FP64 OpenCL tests will be skipped on devices without the extension. Add a `pytest.mark.skipif` decorator checking device FP64 capability. The flag generation itself is always testable (Python-only, no device needed). |
| Existing OpenCL `.cl.c` files use `float` literals (e.g., `0.0f`) that should be `COMPUTE_ZERO` | Low | Audit all `.cl.c` files for hardcoded `float` literals that participate in compute-role arithmetic. Any found should use `COMPUTE_ZERO` or `(COMPUTE_TYPE)` casts. This is an existing code quality issue, not introduced by FP64 support. |
| `COMPUTE_TYPE_IS_DOUBLE` and `COMPUTE_TYPE_IS_HALF` build flags supplied simultaneously as both 1 | Very Low | The type-mapping function derives these flags from `compute_dtype`; it is impossible for a single dtype to be both `float16` and `float64`. Add a static assertion in `build_compiler_flags()` as defense-in-depth. |
