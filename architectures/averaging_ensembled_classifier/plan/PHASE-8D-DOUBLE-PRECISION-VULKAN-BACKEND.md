# Phase 8D: Double Precision — Vulkan Backend & SPIR-V Variants

**Status: ✅ COMPLETED**  
**Phase:** 8D of 8  
**Prerequisite:** Phase 8C complete and rollback gate passed.  
**Objective:** Extend the Vulkan backend's GLSL shader library, `common.glsl`, Meson SPIR-V compilation, and pipeline cache to support FP64 roles. Add the `GL_EXT_shader_explicit_arithmetic_types_float64` extension guard. Parameterize `COMPUTE_FLOAT` (previously hardcoded to `float`), completing the three-axis scheme. Compile all 11 valid precision SPIR-V variants per role-bearing shader. Add `supports_float64()` capability check. The phase ends when the Vulkan SPIR-V build produces all variant artifacts and existing FP32-pathway tests pass.  
**Governing ADR:** ADR-024 §5  
**Rollback gate:** All Tier 2 tests pass for `PrecisionConfig.float32()`. The Vulkan `builddir-vulkan` build succeeds. SPIR-V artifacts for all 11 variants exist in the output directory. `PrecisionConfig.float32()` pipeline paths are unchanged from Phase 8C baseline.  
**Dependencies:** Phase 8C complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 8D.1: Update `common.glsl` — parameterize `COMPUTE_FLOAT`](#step-8d1-update-commonglsl--parameterize-compute_float)
   - [Step 8D.2: Add FP64 extension guard to `common.glsl`](#step-8d2-add-fp64-extension-guard-to-commonglsl)
   - [Step 8D.3: Update workgroup scratch type](#step-8d3-update-workgroup-scratch-type)
   - [Step 8D.4: Update precision boundary helpers for three-axis scheme](#step-8d4-update-precision-boundary-helpers-for-three-axis-scheme)
   - [Step 8D.5: Update reduction helper functions](#step-8d5-update-reduction-helper-functions)
   - [Step 8D.6: Migrate `.comp` shaders for `COMPUTE_FLOAT` parameterization](#step-8d6-migrate-comp-shaders-for-compute_float-parameterization)
   - [Step 8D.7: Add `supports_float64()` capability check](#step-8d7-add-supports_float64-capability-check)
   - [Step 8D.8: Update `src/backends/vulkan/type_mapping.py`](#step-8d8-update-srcbackendsvulkantype_mappingpy)
   - [Step 8D.9: Update Vulkan `meson.build` — 11-variant SPIR-V compilation](#step-8d9-update-vulkan-mesonbuild--11-variant-spirv-compilation)
   - [Step 8D.10: Update `_pipeline_cache.py` — three-axis variant selection](#step-8d10-update-_pipeline_cachepy--three-axis-variant-selection)
   - [Step 8D.11: Rebuild and validate rollback gate](#step-8d11-rebuild-and-validate-rollback-gate)
4. [SPIR-V Variant Enumeration](#4-spirv-variant-enumeration)
5. [`.comp` Shader Migration Checklist](#5-comp-shader-migration-checklist)
6. [Risk Register](#6-risk-register)

---

## 1. Scope & Constraints

### In scope

- **`common.glsl`:** Add `COMPUTE_FLOAT` macro (defaulting to `float`). Update `WIDEN_STORAGE`, `WIDEN_STATE` to cast to `COMPUTE_FLOAT` (not hardcoded `float`). Add `GL_EXT_shader_explicit_arithmetic_types_float64` conditional extension guard. Change `_compute_scratch` from `shared float[32]` to `shared COMPUTE_FLOAT[32]`. Update `workgroup_reduce_add`/`workgroup_reduce_max` to use `COMPUTE_FLOAT` return type and scratch type. Update precision boundary helpers to use `COMPUTE_FLOAT`.
- **`.comp` shaders (17 role-bearing files):** Replace any hardcoded `float` for compute-role local variables, accumulation, and push constant arithmetic with `COMPUTE_FLOAT`. Push constants themselves remain `float`-typed unless the shader's compute-role precision dictates otherwise (push constants are typically small scalars — they can be cast at the usage site).
- **`type_mapping.py`:** Add `get_compute_is_double()` and `get_state_is_double()`.
- **Vulkan `meson.build`:** Replace the 3 precision variants with 11 (all valid `STORAGE_FLOAT × COMPUTE_FLOAT × STATE_FLOAT` combinations).
- **`_pipeline_cache.py`:** Update `_spv_variant_suffix()` to derive from the `(storage, compute, state)` triple.
- **Capability check:** Add `supports_float64()` to the Vulkan context or capabilities module.

### Out of scope

- Changes to `normalize_gradients.comp` — it uses compute-role buffers only and will be compiled once per `COMPUTE_FLOAT` value (not explicitly per storage/state variant).
- Vulkan descriptor set management or pipeline recreation logic — unchanged.
- Runtime FP64 performance benchmarking — deferred to Phase 8E.
- OpenCL or CPU backend changes — completed in 8B/8C.

### Key constraint: ADR-023 §3.3 invariant removal

ADR-023 §3.3 established "`COMPUTE_TYPE = float` on Vulkan backend, invariant." ADR-024 §5 removes this invariant. `COMPUTE_FLOAT` is now parameterized. All Vulkan code that assumed `float` for compute-role values must be updated.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/backends/vulkan/kernel_sources/common.glsl` | `STORAGE_FLOAT` and `STATE_FLOAT` macros (defaulting to `float`). No `COMPUTE_FLOAT` macro. `WIDEN_STORAGE(x)` casts to `float(x)`, not parameterized compute type. `_compute_scratch` is `shared float[32]`. `workgroup_reduce_add` returns `float`. Precision boundary helpers return/accept `float`. |
| `src/backends/vulkan/kernel_sources/*.comp` (17 role-bearing) | Use `read_storage()` / `write_storage()` / `read_state()` / `write_state()` with `float` compute-role semantics. Local `float` accumulators. |
| `src/backends/vulkan/kernel_sources/meson.build` | Three precision variants: `_fp32`, `_s16fp32`, `_fp16`. STORAGE_FLOAT and STATE_FLOAT flags only. |
| `src/backends/vulkan/_pipeline_cache.py` | `_spv_variant_suffix()` dispatches on `(storage_dtype, state_dtype)` only. |
| `src/backends/vulkan/type_mapping.py` | `get_storage_is_half()` and `get_state_is_half()`. No `_is_double` functions. |
| `src/backends/vulkan/context.py` | Vulkan device context; no `supports_float64()` check. |

---

## 3. Task Breakdown

---

### Step 8D.1: Update `common.glsl` — parameterize `COMPUTE_FLOAT`

**Governing authority:** ADR-024 §5.1  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Add the `COMPUTE_FLOAT` macro alongside the existing `STORAGE_FLOAT` and `STATE_FLOAT`:

```glsl
#ifndef COMPUTE_FLOAT
#define COMPUTE_FLOAT float
#endif
```

Update `WIDEN_STORAGE` and `WIDEN_STATE` to cast to `COMPUTE_FLOAT` instead of hardcoded `float`:

```glsl
#define WIDEN_STORAGE(x)   COMPUTE_FLOAT(x)
#define NARROW_STORAGE(x)  STORAGE_FLOAT(x)
#define WIDEN_STATE(x)     COMPUTE_FLOAT(x)
#define NARROW_STATE(x)    STATE_FLOAT(x)
```

---

### Step 8D.2: Add FP64 extension guard to `common.glsl`

**Governing authority:** ADR-024 §5.3  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Add after the FP16 extension guard:

```glsl
#ifdef ENABLE_FP64_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
#endif
```

The Meson build injects `-DENABLE_FP64_EXTENSION=1` when any role uses `double`.

---

### Step 8D.3: Update workgroup scratch type

**Governing authority:** ADR-024 §5.2  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Change:
```glsl
shared float _compute_scratch[32];
```
to:
```glsl
shared COMPUTE_FLOAT _compute_scratch[32];
```

Update the comment:
```glsl
// Compute-role scratch for cross-subgroup bridge.
// Type is COMPUTE_FLOAT, parameterized by the active precision configuration.
```

---

### Step 8D.4: Update precision boundary helpers for three-axis scheme

**Governing authority:** ADR-024 §5.1  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Update the precision boundary helper functions to use `COMPUTE_FLOAT` instead of hardcoded `float`:

```glsl
COMPUTE_FLOAT read_storage(STORAGE_FLOAT val) { return WIDEN_STORAGE(val); }
STORAGE_FLOAT write_storage(COMPUTE_FLOAT val) { return NARROW_STORAGE(val); }

COMPUTE_FLOAT read_state(STATE_FLOAT val) { return WIDEN_STATE(val); }
STATE_FLOAT write_state(COMPUTE_FLOAT val) { return NARROW_STATE(val); }
```

---

### Step 8D.5: Update reduction helper functions

**Governing authority:** ADR-024 §5.2  
**File:** `src/backends/vulkan/kernel_sources/common.glsl`

Update `workgroup_reduce_add` and `workgroup_reduce_max` signatures and internal types from `float` to `COMPUTE_FLOAT`:

```glsl
COMPUTE_FLOAT workgroup_reduce_add(COMPUTE_FLOAT value) {
    COMPUTE_FLOAT subgroup_sum = subgroupAdd(value);
    if (subgroupElect()) {
        _compute_scratch[gl_SubgroupID] = subgroup_sum;
    }
    barrier();
    COMPUTE_FLOAT total = COMPUTE_FLOAT(0.0);
    if (gl_SubgroupID == 0) {
        COMPUTE_FLOAT val = (gl_SubgroupInvocationID < gl_NumSubgroups)
                    ? _compute_scratch[gl_SubgroupInvocationID] : COMPUTE_FLOAT(0.0);
        total = subgroupAdd(val);
    }
    if (gl_SubgroupID == 0 && subgroupElect()) {
        _compute_scratch[0] = total;
    }
    barrier();
    return _compute_scratch[0];
}
```

Similarly update `workgroup_reduce_max`.

**Key concern:** `subgroupAdd(double)` and `subgroupMax(double)` require `VK_KHR_shader_subgroup_arithmetic` to support FP64. Verify the extension spec. If FP64 subgroup operations are not guaranteed, a fallback using shared memory atomics or sequential reduction may be needed for FP64 variants. Investigate at implementation time; document the requirement.

---

### Step 8D.6: Migrate `.comp` shaders for `COMPUTE_FLOAT` parameterization

**Governing authority:** ADR-024 §5.1  
**Files:** All 17 role-bearing `.comp` shaders, plus `normalize_gradients.comp`

For each `.comp` file:

1. **Local compute-role variables:** `float accum = 0.0;` → `COMPUTE_FLOAT accum = COMPUTE_FLOAT(0.0);`
2. **Intermediate arithmetic results:** All `float` temporaries in compute paths → `COMPUTE_FLOAT`.
3. **Function calls:** `workgroup_reduce_add(float)` → `workgroup_reduce_add(COMPUTE_FLOAT)` (signature change in `common.glsl` handles this).
4. **Push constant scalar usage:** Push constants that are `float` and used in compute-role arithmetic should be cast: `COMPUTE_FLOAT(push.learning_rate)`.
5. **`NUMERICAL_STABILITY_EPSILON`:** The constant in `common.glsl` is currently `1e-7` (FP32). For FP64, this should be `1e-15`. Parameterize: inject via `glslc -DNUMERICAL_STABILITY_EPSILON=1e-15` for FP64 compute variants, or use a conditional define in `common.glsl`:

```glsl
#ifndef NUMERICAL_STABILITY_EPSILON
#if defined(ENABLE_FP64_EXTENSION) && COMPUTE_FLOAT == double
const COMPUTE_FLOAT NUMERICAL_STABILITY_EPSILON = 1e-15;
#else
const COMPUTE_FLOAT NUMERICAL_STABILITY_EPSILON = 1e-7;
#endif
#endif
```

Note: GLSL preprocessor cannot compare type macros directly. Use the build-system approach: inject `-DNUMERICAL_STABILITY_EPSILON=1e-15` for FP64 compute variants.

---

### Step 8D.7: Add `supports_float64()` capability check

**Governing authority:** ADR-024 §5.4  
**File:** `src/backends/vulkan/context.py` or a dedicated `capabilities.py`

Add:
```python
def supports_float64(self) -> bool:
    """Check if the Vulkan device supports FP64 shader operations."""
    return bool(self.physical_device_features.shaderFloat64)
```

Plan construction for a `PrecisionConfig` with FP64 roles must check this and fail fast:

```python
if any dtype is float64 and not context.supports_float64():
    raise RuntimeError(
        "PrecisionConfig requires FP64 but Vulkan device does not support shaderFloat64"
    )
```

---

### Step 8D.8: Update `src/backends/vulkan/type_mapping.py`

**Governing authority:** ADR-024 §5.1, §5.6  
**File:** `src/backends/vulkan/type_mapping.py`

Add FP64 helper functions:

```python
def get_compute_is_double(precision: PrecisionConfig) -> int:
    return 1 if precision.compute_dtype == np.dtype(np.float64) else 0

def get_state_is_double(precision: PrecisionConfig) -> int:
    return 1 if precision.state_dtype == np.dtype(np.float64) else 0

def get_storage_is_double(precision: PrecisionConfig) -> int:
    return 1 if precision.storage_dtype == np.dtype(np.float64) else 0
```

These are consumed by the Meson build flag generation and the pipeline cache.

---

### Step 8D.9: Update Vulkan `meson.build` — 11-variant SPIR-V compilation

**Governing authority:** ADR-024 §5.5  
**File:** `src/backends/vulkan/kernel_sources/meson.build`

Replace the 3 precision variant definitions with the 11 valid combinations:

```meson
# Precision variant definitions: [suffix, extra glslc flags]
# Three-axis: STORAGE_FLOAT × COMPUTE_FLOAT × STATE_FLOAT
precision_variants = [
    # storage=16, compute=32
    ['_s16c32x16', ['-DSTORAGE_FLOAT=float16_t', '-DCOMPUTE_FLOAT=float', '-DSTATE_FLOAT=float16_t', '-DENABLE_FP16_EXTENSION=1']],
    ['_s16c32x32', ['-DSTORAGE_FLOAT=float16_t', '-DCOMPUTE_FLOAT=float', '-DSTATE_FLOAT=float', '-DENABLE_FP16_EXTENSION=1']],
    ['_s16c32x64', ['-DSTORAGE_FLOAT=float16_t', '-DCOMPUTE_FLOAT=float', '-DSTATE_FLOAT=double', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1']],
    # storage=16, compute=64
    ['_s16c64x16', ['-DSTORAGE_FLOAT=float16_t', '-DCOMPUTE_FLOAT=double', '-DSTATE_FLOAT=float16_t', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1']],
    ['_s16c64x32', ['-DSTORAGE_FLOAT=float16_t', '-DCOMPUTE_FLOAT=double', '-DSTATE_FLOAT=float', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1']],
    ['_s16c64x64', ['-DSTORAGE_FLOAT=float16_t', '-DCOMPUTE_FLOAT=double', '-DSTATE_FLOAT=double', '-DENABLE_FP16_EXTENSION=1', '-DENABLE_FP64_EXTENSION=1']],
    # storage=32, compute=32
    ['_s32c32x32', []],
    ['_s32c32x64', ['-DSTATE_FLOAT=double', '-DENABLE_FP64_EXTENSION=1']],
    # storage=32, compute=64
    ['_s32c64x32', ['-DCOMPUTE_FLOAT=double', '-DENABLE_FP64_EXTENSION=1']],
    ['_s32c64x64', ['-DCOMPUTE_FLOAT=double', '-DSTATE_FLOAT=double', '-DENABLE_FP64_EXTENSION=1']],
    # storage=64
    ['_s64c64x64', ['-DSTORAGE_FLOAT=double', '-DCOMPUTE_FLOAT=double', '-DSTATE_FLOAT=double', '-DENABLE_FP64_EXTENSION=1']],
]
```

For FP64 compute variants, also inject the appropriate epsilon:
```meson
    # Add to FP64-compute variants:
    '-DNUMERICAL_STABILITY_EPSILON=1e-15',
```

For `normalize_gradients.comp` (compute-only): compile two variants — `_c32` (COMPUTE_FLOAT=float, default) and `_c64` (COMPUTE_FLOAT=double), since it operates entirely in compute space.

The `foreach` loop structure already established in Phase 7C is preserved; only the variant list expands.

---

### Step 8D.10: Update `_pipeline_cache.py` — three-axis variant selection

**Governing authority:** ADR-024 §5.5  
**File:** `src/backends/vulkan/_pipeline_cache.py`

Replace `_spv_variant_suffix()` with:

```python
def _spv_variant_suffix(precision: PrecisionConfig) -> str:
    """Map a PrecisionConfig to the SPIR-V variant suffix (ADR-024 §5.5)."""
    _SUFFIX_MAP = {
        (np.float16, np.float32, np.float16): "_s16c32x16",
        (np.float16, np.float32, np.float32): "_s16c32x32",
        (np.float16, np.float32, np.float64): "_s16c32x64",
        (np.float16, np.float64, np.float16): "_s16c64x16",
        (np.float16, np.float64, np.float32): "_s16c64x32",
        (np.float16, np.float64, np.float64): "_s16c64x64",
        (np.float32, np.float32, np.float32): "_s32c32x32",
        (np.float32, np.float32, np.float64): "_s32c32x64",
        (np.float32, np.float64, np.float32): "_s32c64x32",
        (np.float32, np.float64, np.float64): "_s32c64x64",
        (np.float64, np.float64, np.float64): "_s64c64x64",
    }
    key = (
        precision.storage_dtype.type,
        precision.compute_dtype.type,
        precision.state_dtype.type,
    )
    suffix = _SUFFIX_MAP.get(key)
    if suffix is None:
        raise ValueError(
            f"No Vulkan SPIR-V variant for storage={precision.storage_dtype}, "
            f"compute={precision.compute_dtype}, state={precision.state_dtype}"
        )
    return suffix
```

For `_COMPUTE_ONLY_SHADERS`, select the suffix based on `compute_dtype` only:
```python
def _compute_only_suffix(precision: PrecisionConfig) -> str:
    if precision.compute_dtype == np.dtype(np.float64):
        return "_c64"
    return "_c32"
```

---

### Step 8D.11: Rebuild and validate rollback gate

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir-vulkan 2>&1 | tee /tmp/phase8d_vulkan_build.txt
grep -E "error:" /tmp/phase8d_vulkan_build.txt | head -20

# Verify all 11 variant SPV files exist for a representative shader
ls builddir-vulkan/src/backends/vulkan/kernel_sources/forward_pass_*.spv | wc -l
# Expected: 11

python -m pytest src/tests/ -q 2>&1 | tee /tmp/phase8d_tests.txt
grep -E "passed|failed|error" /tmp/phase8d_tests.txt
```

---

## 4. SPIR-V Variant Enumeration

| STORAGE | COMPUTE | STATE | Suffix | FP16 ext | FP64 ext |
|:---|:---|:---|:---|:---|:---|
| `float16_t` | `float` | `float16_t` | `_s16c32x16` | yes | no |
| `float16_t` | `float` | `float` | `_s16c32x32` | yes | no |
| `float16_t` | `float` | `double` | `_s16c32x64` | yes | yes |
| `float16_t` | `double` | `float16_t` | `_s16c64x16` | yes | yes |
| `float16_t` | `double` | `float` | `_s16c64x32` | yes | yes |
| `float16_t` | `double` | `double` | `_s16c64x64` | yes | yes |
| `float` | `float` | `float` | `_s32c32x32` | no | no |
| `float` | `float` | `double` | `_s32c32x64` | no | yes |
| `float` | `double` | `float` | `_s32c64x32` | no | yes |
| `float` | `double` | `double` | `_s32c64x64` | no | yes |
| `double` | `double` | `double` | `_s64c64x64` | no | yes |

Total: **11 variants per role-bearing shader**, **2 variants for compute-only shaders**.

---

## 5. `.comp` Shader Migration Checklist

For each of the 17 role-bearing shaders:

- [ ] All `float` local variables for compute-role values → `COMPUTE_FLOAT`
- [ ] All `0.0` / `0.0f` in compute-role contexts → `COMPUTE_FLOAT(0.0)`
- [ ] `workgroup_reduce_add`/`workgroup_reduce_max` return types are already `COMPUTE_FLOAT` (from `common.glsl` update)
- [ ] Push constant scalars cast to `COMPUTE_FLOAT` at usage site
- [ ] `shared float` scratch arrays for compute use → `shared COMPUTE_FLOAT`
- [ ] Post-edit: `glslc --target-env=vulkan1.1 <file>` compiles without error for all 11 variant flag sets (or at minimum: `_s32c32x32` and `_s64c64x64`)

For `normalize_gradients.comp`:
- [ ] All `float` → `COMPUTE_FLOAT`
- [ ] Compiled with `_c32` and `_c64` flags only

---

## 6. Risk Register

| Risk | Likelihood | Mitigation |
|:---|:---|:---|
| `subgroupAdd(double)` / `subgroupMax(double)` not supported by target Vulkan drivers | Medium | Check `VK_KHR_shader_subgroup_arithmetic` capability for FP64. If unsupported, implement a shared-memory sequential reduction fallback for FP64 workgroups. Gate the subgroup path on `COMPUTE_FLOAT != double`. |
| `double` SSBO access alignment requirements differ from `float` | Low | SPIR-V `OpTypeFloat 64` has 8-byte natural alignment. Buffer allocations must respect this. Verify via `VkPhysicalDeviceLimits::minStorageBufferOffsetAlignment`. |
| `glslc` version does not support `double` in SSBO layouts or compute-role positions | Low | Require `glslc` from Vulkan SDK ≥ 1.2.182. Add version check to Meson. |
| SPIR-V artifact count (11 × 18 = 198 files + 2 for normalize) impacts build time significantly | Medium | `glslc` compilation is fast (~100ms per file). 200 compilations ≈ 20 seconds. If unacceptable, parallelize via Meson's default build parallelism. Measure before optimizing. |
| `NUMERICAL_STABILITY_EPSILON` injection via `-D` conflicts with the `const` declaration in `common.glsl` | Medium | Gate the `const` declaration with `#ifndef NUMERICAL_STABILITY_EPSILON` so the `-D` flag takes precedence. Already the recommended pattern; verify it compiles. |
| Backward compatibility: `_fp32` / `_s16fp32` / `_fp16` SPIR-V filenames change to `_s32c32x32` etc., breaking runtime path resolution | High | Atomic rename in `_pipeline_cache.py` and `meson.build` in the same commit. Alternatively, produce both old and new filenames during a transition period (symlinks or dual custom_targets). Recommend atomic approach for cleanliness. |
