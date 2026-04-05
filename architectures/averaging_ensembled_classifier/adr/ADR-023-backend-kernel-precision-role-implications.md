# ADR-023: Precision-Role Implications for Backend-Native Kernel Code

**Status:** ACCEPTED
**Date:** 2026-04-02
**Deciders:** —
**Triggered by:** ADR-020 (Three-Role Precision Model), ADR-021 (Kernel Spec Migration), ADR-022 (Host Code Implications)
**Depends on:** ADR-013 (Kernel Source Strategy), ADR-014 (Build System Integration), ADR-015 (Python/Native Interop), ADR-020, ADR-021, ADR-022
**Constrains:**
  - `kernels/kernels.cl.h` (OpenCL precision boundary implementations)
  - `src/backends/cpu/kernel_sources/` (all files)
  - `src/backends/vulkan/kernel_sources/` (all files)
  - `src/backends/vulkan/meson.build`, `src/backends/cpu/meson.build`

---

## Context

ADR-021 established the specification-level migration of `kernels.cl.h` and the OpenCL `.cl.c` implementation files. It declared the precision boundary load/store abstractions (`load_storage`, `store_storage`, `load_state`, `store_state_update`) as inline function signatures in `kernels.cl.h` but explicitly deferred their *implementations* to each backend's `kernel_sources/` directory (ADR-021 §1.3):

> "Implementations of these abstractions reside in each backend's `kernel_sources/` directory per ADR-013."

ADR-022 addressed the full host-code cascade: `PrecisionConfig`, `BufferDescriptor`, `BufferParamSpec`, backend type-mapping modules, and the plan builder. Neither ADR addressed the backend-native kernel code comprehensively. That gap is closed here.

The three backend kernel implementations sit at different positions relative to the three-role model:

**OpenCL (`kernels/kernels.cl.h`, `kernels/phase_*.cl.c`)**: ADR-021 completed the specification changes. The missing work is the inline function *bodies* for the four precision boundary abstractions. These bodies live in `kernels.cl.h` inside the `#ifdef __OPENCL_VERSION__` guard — the OpenCL backend has no separate `kernel_sources/` directory (ADR-013: OpenCL sources reside at the architecture-root `kernels/`).

**CPU (`src/backends/cpu/kernel_sources/`)**: `cpu_precision.h` already embodies the insight that "Computation always uses float (FP32) internally; the storage type only affects buffer pointers" — this is the three-role model's `COMPUTE_TYPE = float` partially instantiated. What it lacks is the state/storage distinction: `cpu_kernels.h`'s `DECLARE_PRECISION_STRUCTS(SUFFIX, REAL_T)` generates structs where *all* float buffers share a single `REAL_T`, conflating storage-role transient buffers (`STORAGE_TYPE`) with persistent optimizer state (`STATE_TYPE`). The `.inc` template files use `simd_load_real`/`simd_store_real` — role-unaware names — for all float buffer accesses. These are incomplete instantiations of the three-role model that require extension.

**Vulkan (`src/backends/vulkan/kernel_sources/`)**: The `.comp` shaders currently hardcode `float` for all SSBO buffer element types and for `common.glsl`'s workgroup reduction scratch. No precision-role concept exists in the Vulkan kernel sources at all. This backend requires the most structural work, complicated by a fundamental GLSL constraint: SSBO element types are declared in-source as built-in type names (`float`, `float16_t`); they cannot be aliased at compile time by `-D` preprocessor substitution the way OpenCL and C can alias `SCALAR_TYPE=float`. A GLSL-specific strategy for role-typed buffer layouts must be selected.

---

## Decision Drivers

1. **ADR-021 §1.3 deferred to this ADR.** The precision boundary abstraction implementations are an explicitly scoped deliverable. This ADR fulfills that deferral for all three backends.

2. **CPU backend partially anticipated the three-role model.** The "compute is always float" convention is an implicit `COMPUTE_TYPE = float` commitment. Formalizing this as the architectural fact it is (rather than a convention) and extending it to cover the storage/state split is the natural completion of that path.

3. **GLSL cannot parameterise SSBO element types via `-D` flags.** The precision-role mechanism that works in OpenCL (`-DSTORAGE_TYPE=half`) and C (`-DSTORAGE_T=_Float16`) has no direct equivalent at the GLSL `layout(buffer)` declaration level. The strategy for Vulkan must be selected from available GLSL techniques rather than assumed to mirror the other backends.

4. **ADR-020 §2.3 — one codepath, no `#ifdef` on type equality.** The three-role model's core contract is that the kernel source has a single codepath; the compiler eliminates identity conversions when role types are equal. What is forbidden is `#ifdef MIXED_PRECISION_ENABLED` gating around a second implementation. What is required is that typing boundaries are always expressed through the same abstraction, regardless of whether the types happen to be the same.

5. **The CPU ABI is a public interface boundary (ADR-015).** Changes to `cpu_kernels.h` struct field types affect the Python `ctypes` bindings in the CPU backend's FFI layer. The struct field type changes must be named precisely enough for the binding layer to update in lockstep.

6. **Build system variants are a build concern, not a source concern.** Backends that require multi-variant compilation (Vulkan SPIR-V under different precision flags; CPU struct instantiations) handle this through their `meson.build` and headers. The kernel source files themselves express the role model once, without mode flags.

---

## §1: OpenCL — Precision Boundary Abstraction Implementations

ADR-021 §1.3 declared the following inline function signatures in `kernels.cl.h`, within the `#ifdef __OPENCL_VERSION__` guard, without providing bodies. The bodies are specified here.

```c
// --- Precision Boundary Implementations (OpenCL backend, kernels.cl.h) ---

static inline COMPUTE_TYPE load_storage(
    __global const STORAGE_TYPE *buf, size_t idx) {
#if STORAGE_TYPE_IS_HALF
    return (COMPUTE_TYPE)vload_half(idx, (__global const half *)buf);
#else
    return (COMPUTE_TYPE)buf[idx];
#endif
}

static inline void store_storage(
    __global STORAGE_TYPE *buf, size_t idx, COMPUTE_TYPE val) {
#if STORAGE_TYPE_IS_HALF
    vstore_half((half)val, idx, (__global half *)buf);
#else
    buf[idx] = (STORAGE_TYPE)val;
#endif
}

static inline COMPUTE_TYPE load_state(
    __global const STATE_TYPE *buf, size_t idx) {
    return (COMPUTE_TYPE)buf[idx];
}

static inline void store_state(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) {
    buf[idx] = (STATE_TYPE)val;
}

static inline void store_state_update(
    __global STATE_TYPE *buf, size_t idx, COMPUTE_TYPE val) {
    buf[idx] = (STATE_TYPE)val;
}
```

### §1.1: Rationale for `STORAGE_TYPE_IS_HALF` conditional in `load_storage`/`store_storage`

OpenCL's `vload_half`/`vstore_half` builtins require the `cl_khr_fp16` extension and are the correct mechanism for loading FP16 data from global memory into a widened arithmetic register. Direct assignment `(float)((half *)buf)[idx]` is not guaranteed portable across drivers without the extension active. The `STORAGE_TYPE_IS_HALF` flag (injected by `build_compiler_flags` per ADR-022 §7.1) controls this gate. The flag encodes storage type, not compute type. In the uniform FP32 configuration, `STORAGE_TYPE_IS_HALF == 0` and both branches compile to `(float)buf[idx]` — a cast that the compiler eliminates as an identity.

### §1.2: `load_state` / `store_state` have no `_IS_HALF` conditional

For the deployed precision configurations (`PrecisionConfig.float32()`, `PrecisionConfig.mixed_f16_f32()`), `STATE_TYPE` is `float`. The direct cast `(COMPUTE_TYPE)buf[idx]` is an identity. If a future configuration introduces `STATE_TYPE = half`, the cast produces the correct widening. No `_IS_HALF` flag is required because OpenCL's implicit conversion between any numeric type via C-style cast is well-defined for the scalar case (no `vload_half` needed for state buffers, which are accessed scalar-element-wise by Adam, not as packed SIMD streams).

### §1.3: `store_state` vs. `store_state_update`

Both are identical in the OpenCL implementation. The distinction is preserved for semantic clarity: `store_state` writes a fresh state value; `store_state_update` signals an EMA in-place mutation (the Adam case). A future backend (or a future non-temporal-store optimization pass per CONCEPT.md §1) may choose to specialize the variant. The separation costs nothing and preserves architectural expressiveness.

### §1.4: No abstraction for compute-role memory

Loads from `__global COMPUTE_TYPE *` and `__local COMPUTE_TYPE` scratch are direct pointer dereferences (`buf[idx]`). No abstraction is used or declared. Compute-role buffers are already in the arithmetic type; there is no boundary to cross.

---

## §2: CPU Backend — Three-Role Architecture

### §2.1: `COMPUTE_TYPE = float` as a first-class CPU backend constant

The CPU backend's existing practice — "Computation always uses float (FP32) internally" (from `cpu_kernels.h` comment, in force since ADR-008) — is the three-role model's `COMPUTE_TYPE = float` instantiated as a CPU backend invariant. This is not an approximation; it is exact. On the CPU backend, the compute precision role is always `float`. This fact is now declared as an architectural constant, not an implementation convention:

```c
/* cpu_kernels.h (amended) */
/* COMPUTE_TYPE is invariant on the CPU backend: always float.
 * Per ADR-023 §2.1: CPU arithmetic always executes at FP32 precision.
 * STORAGE_T and STATE_T vary by PrecisionConfig; float does not. */
typedef float cpu_compute_t;
```

This `typedef` is a documentation artifact. All existing arithmetic in `.inc` template files is `float` — no renaming of arithmetic expressions is needed.

### §2.2: `cpu_precision.h` — Role-Split Load/Store Abstractions

`cpu_precision.h` currently provides four macro-dispatched SIMD/scalar load/store operations keyed to a single type axis (`REAL_T` / `PRECISION_SUFFIX`):

```c
simd_load_real(ptr)      simd_store_real(ptr, val)
scalar_load_real(ptr)    scalar_store_real(ptr, val)
```

These names are retired. Their replacements are split by precision role:

```c
/* Storage-role — keyed off STORAGE_T / STORAGE_SUFFIX */
simd_load_storage(ptr)      simd_store_storage(ptr, val)
scalar_load_storage(ptr)    scalar_store_storage(ptr, val)

/* State-role — keyed off STATE_T / STATE_SUFFIX */
simd_load_state(ptr)        simd_store_state(ptr, val)
scalar_load_state(ptr)      scalar_store_state(ptr, val)
```

The macro dispatch pattern in `cpu_precision.h` is extended to two independent suffix axes:

```c
/* Storage role: STORAGE_SUFFIX and STORAGE_T must be defined by the .c includer */
#undef simd_load_storage
#undef simd_store_storage
#undef scalar_load_storage
#undef scalar_store_storage
#define simd_load_storage    _PREC_CAT2(simd_load_storage,  STORAGE_SUFFIX)
#define simd_store_storage   _PREC_CAT2(simd_store_storage, STORAGE_SUFFIX)
#define scalar_load_storage  _PREC_CAT2(scalar_load_storage,  STORAGE_SUFFIX)
#define scalar_store_storage _PREC_CAT2(scalar_store_storage, STORAGE_SUFFIX)

/* State role: STATE_SUFFIX and STATE_T must be defined by the .c includer */
#undef simd_load_state
#undef simd_store_state
#undef scalar_load_state
#undef scalar_store_state
#define simd_load_state      _PREC_CAT2(simd_load_state,  STATE_SUFFIX)
#define simd_store_state     _PREC_CAT2(simd_store_state, STATE_SUFFIX)
#define scalar_load_state    _PREC_CAT2(scalar_load_state,  STATE_SUFFIX)
#define scalar_store_state   _PREC_CAT2(scalar_store_state, STATE_SUFFIX)
```

`PREC_SIZEOF_STORAGE` and `PREC_SIZEOF_STATE` replace `PREC_SIZEOF_REAL`, giving callers the element size for each role independently.

The prior names (`simd_load_real`, etc.) are removed. Every `.inc` template file that calls them is updated to the role-appropriate name per the role assignment tables in ADR-021 §1.4. No algorithmic change — only the name of the load/store macro changes.

**Key invariant:** When `STATE_SUFFIX == STORAGE_SUFFIX` (uniform FP16 configuration), `simd_load_state_fp16` resolves to the same function as `simd_load_storage_fp16`. The compiler sees identical code at both call sites and may inline them identically. The role distinction costs nothing at runtime.

### §2.3: `cpu_kernels.h` — `DECLARE_PRECISION_STRUCTS` Extension

`DECLARE_PRECISION_STRUCTS(SUFFIX, REAL_T)` is extended to three parameters:

```c
#define DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)
```

Every float buffer pointer field in each generated struct is re-typed to the role-appropriate type from ADR-021 §1.4:

| Buffer role | C pointer type in struct |
|:---|:---|
| Storage-role input | `const STORAGE_T*` |
| Storage-role output | `STORAGE_T*` |
| Compute-role input | `const float*` |
| Compute-role output | `float*` |
| State-role read-only | `const STATE_T*` |
| State-role read-write | `STATE_T*` |

Integer buffers (`const int* targets` for CCE) are unchanged.

**Affected struct fields — concrete examples:**

| Struct | Field | Old type | New type | Role |
|:---|:---|:---|:---|:---|
| `ForwardPassArgs` | `input` | `const REAL_T*` | `const STORAGE_T*` | storage |
| `ForwardPassArgs` | `weights_shared_simd_major` | `const REAL_T*` | `const STATE_T*` | state |
| `ForwardPassArgs` | `biases_shared` | `const REAL_T*` | `const STATE_T*` | state |
| `ForwardPassArgs` | `hidden_activations` | `REAL_T*` | `STORAGE_T*` | storage |
| `CceChunkArgs` | `temps` | `const REAL_T*` | `const STATE_T*` | state |
| `CceChunkArgs` | `final_loss` | `REAL_T*` | `float*` | compute |
| `AdamUpdateArgs` | `normalized_gradient` | `const REAL_T*` | `const float*` | compute |
| `AdamUpdateArgs` | `m1`, `m2` | `REAL_T*` | `STATE_T*` | state |
| `AdamUpdateArgs` | `weights_or_biases` | `REAL_T*` | `STATE_T*` | state |
| Reduction output buffers | output | `REAL_T*` | `float*` | compute |

This covers every struct declared by `DECLARE_PRECISION_STRUCTS`. The full retyping is mechanical following the role table; only the representative examples are listed here.

### §2.4: Suffix naming scheme

The prior single-suffix scheme (`fp16`, `fp32`, `fp64`) encoded a single precision axis. It is replaced by a two-axis scheme `s{storage}x{state}`:

| Configuration | `SUFFIX` arg | `STORAGE_T` | `STATE_T` |
|:---|:---|:---|:---|
| All FP32 | `s32x32` | `float` | `float` |
| All FP16 | `s16x16` | `_Float16` | `_Float16` |
| Mixed: FP16 storage / FP32 state | `s16x32` | `_Float16` | `float` |

The `fp64` variant (present under ADR-008 for completeness but never a deployed production configuration) is retired. `PrecisionConfig` defines no FP64 factory. The `s16x32` variant replaces it as the third instantiation.

A transitional backward-compatibility alias is provided during the migration window:

```c
/* Transitional alias — removed when all FFI callers migrate to s32x32 naming */
#define DECLARE_PRECISION_STRUCTS_fp32 DECLARE_PRECISION_STRUCTS(s32x32, float, float)
```

The alias is removed in the same commit that updates the Python `ctypes` bindings in the CPU backend's FFI layer (ADR-015 scope).

### §2.5: `.inc` template file updates

Each `.inc` template file currently calls `simd_load_real(buf)` or `scalar_load_real(buf, idx)` for all float buffer accesses indiscriminately.

The required update for every `.inc` file follows the role assignment tables in ADR-021 §1.4 directly:

1. All reads from storage-role buffer pointers → `simd_load_storage(buf)` / `scalar_load_storage(buf, idx)`
2. All writes to storage-role buffer pointers → `simd_store_storage(buf, val)` / `scalar_store_storage(buf, idx, val)`
3. All reads from state-role buffer pointers → `simd_load_state(buf)` / `scalar_load_state(buf, idx)`
4. All writes to state-role buffer pointers → `simd_store_state(buf, val)` / `scalar_store_state(buf, idx, val)`
5. All reads/writes to compute-role buffers (`float*`) → direct dereference (`buf[idx]`), no abstraction
6. `__local`-equivalent stack scratch (`float` arrays) → direct access, unchanged

No algorithmic expression, loop bound, index calculation, reduction tree structure, or SIMD vector arithmetic changes. Only the macro name at load/store boundaries changes.

---

## §3: Vulkan Backend — Precision-Role Typed Buffer Layouts

### §3.1: The GLSL buffer-type parameterisation problem

OpenCL and C backends receive precision type symbols via `-D` preprocessor flags at compile time (`-DSTORAGE_TYPE=half`, `-DSTORAGE_T=_Float16`). The type name is then used in function signatures and pointer declarations. This works because both languages allow preprocessor-aliased type names everywhere, including in pointer declarations.

GLSL does not support preprocessor-aliased type names at the `layout(buffer)` block member position. The following is not valid GLSL:

```glsl
// INVALID — STORAGE_TYPE is not a built-in GLSL type name
layout(set = 0, binding = 0) buffer Buf { STORAGE_TYPE data[]; } b;
```

The three viable strategies are:

| Strategy | Mechanism | Assessment |
|:---|:---|:---|
| **A: Preprocessor macro + glslc `-D`** | Define `STORAGE_TYPE` as a GLSL type alias via `-DSTORAGE_TYPE=float16_t` passed to `glslc`. GLSL preprocessor substitution applies to buffer layout member types. | Valid. `glslc` supports `-D`. Requires `GL_EXT_shader_explicit_arithmetic_types_float16` enabled in source. Produces clean single-source shaders. |
| **B: `uint16_t` reinterpret** | Declare storage-role SSBOs as `{ uint16_t data[]; }`. Use `unpackHalf2x16` / `packHalf2x16` for FP16 load/store. No extension required. | Valid but indirect. Destroys the "identity conversion when types equal" property in the FP32 case. `unpackHalf2x16` returns a `vec2` requiring explicit lane selection. Not idiomatic. |
| **C: Separate shader source files per precision** | Maintain `forward_pass_fp32.comp` and `forward_pass_fp16.comp`. | Precise code duplication that the architecture's principles reject. Violates "one codepath" mandate (ADR-020 §2.3). |

**Decision: Strategy A.** The `glslc` preprocessor accepts `-D` defines. GLSL's preprocessor substitution applies to built-in type names in buffer member positions because the substitution occurs in the token stream before parsing. The approach:

1. `common.glsl` defines `STORAGE_TYPE`, `NARROW_STORAGE()`, `WIDEN_STORAGE()`, `STATE_TYPE`, `NARROW_STATE()`, `WIDEN_STATE()` macros, defaulting to `float` when no define is injected.
2. Each `.comp` shader uses `STORAGE_TYPE` and `STATE_TYPE` in its buffer layout declarations and the conversion macros at load/store sites.
3. The Meson build passes `-DSTORAGE_TYPE=float16_t -DSTATE_TYPE=float` (etc.) to `glslc` per precision variant.
4. `GL_EXT_shader_explicit_arithmetic_types_float16` is conditionally enabled in `common.glsl` when `float16_t` is in use.

### §3.2: `common.glsl` changes

Four additions are required.

#### §3.2.1: Precision-role macro definitions

The following block is added immediately after the specialization constant declarations:

```glsl
// ── Precision-Role Type Macros (ADR-023 §3.2) ─────────────────────────
// STORAGE_TYPE and STATE_TYPE are injected by the build system via
// glslc -D flags. Default to float (FP32) when not injected, matching
// PrecisionConfig.float32() and maintaining the uniform-FP32 baseline.
//
// WIDEN_*() converts a role-precision value to float (COMPUTE_TYPE, always
// float in the Vulkan backend). NARROW_*() converts float to role precision.
// When STORAGE_TYPE == float, both are identity functions eliminated by
// the SPIR-V compiler. No #ifdef on type equality anywhere in shaders.

#ifndef STORAGE_TYPE
#define STORAGE_TYPE float
#endif
#ifndef STATE_TYPE
#define STATE_TYPE float
#endif

#if defined(float16_t)
// float16_t is a keyword, not a macro — test via a known companion macro.
#endif

#define WIDEN_STORAGE(x)   float(x)
#define NARROW_STORAGE(x)  STORAGE_TYPE(x)
#define WIDEN_STATE(x)     float(x)
#define NARROW_STATE(x)    STATE_TYPE(x)
```

Note: `float(float_val)` is a valid no-op cast in GLSL. When `STORAGE_TYPE = float`, `NARROW_STORAGE(x)` expands to `float(x)`, an identity. The SPIR-V compiler emits no conversion instruction.

#### §3.2.2: Conditional FP16 extension enablement

The following guard is added before the macro definitions:

```glsl
// Conditionally enable FP16 extension when a role type requires it.
// Injected by glslc -DENABLE_FP16_EXTENSION=1 when any role is float16_t.
#ifdef ENABLE_FP16_EXTENSION
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif
```

The Meson build injects `-DENABLE_FP16_EXTENSION=1` for any precision variant where `STORAGE_TYPE=float16_t` or `STATE_TYPE=float16_t`.

#### §3.2.3: Compute-role scratch declared explicitly

The workgroup reduction scratch array is renamed and annotated to make its compute-role nature explicit:

```glsl
// Compute-role scratch for cross-subgroup bridge.
// Always float (COMPUTE_TYPE = float on Vulkan backend, invariant per ADR-023 §3.3).
shared float _compute_scratch[32];
```

`_cross_subgroup_scratch` is renamed to `_compute_scratch`. All callsites in `workgroup_reduce_add` and `workgroup_reduce_max` update accordingly. No behavioural change; the rename makes the compute-role assignment visible in the source.

#### §3.2.4: Precision boundary load/store helpers

The following helper functions are added to `common.glsl` after the reduction helpers:

```glsl
// ── Precision Boundary Helpers (ADR-023 §3.2.4) ───────────────────────
// Inline load/store wrappers that cross the storage/state → compute boundary.
// When STORAGE_TYPE == float, these are identity operations. The SPIR-V
// compiler eliminates them. One codepath — no #ifdef on type equality.

float read_storage(STORAGE_TYPE val) { return WIDEN_STORAGE(val); }
STORAGE_TYPE write_storage(float val) { return NARROW_STORAGE(val); }

float read_state(STATE_TYPE val) { return WIDEN_STATE(val); }
STATE_TYPE write_state(float val) { return NARROW_STATE(val); }
```

These helpers have value-semantics (taking the element value, not an array index). Each `.comp` shader passes the element value obtained from its SSBO binding through these helpers at every precision boundary crossing.

### §3.3: `COMPUTE_TYPE = float` as a Vulkan backend constant

As on the CPU backend, the Vulkan backend commits to `COMPUTE_TYPE = float` as an invariant. The `common.glsl` `workgroup_reduce_add`/`workgroup_reduce_max` helpers return `float`. All arithmetic throughout every `.comp` shader executes at `float` precision. This is not a constraint — it is the correct instantiation: the Vulkan backend's subgroup operations (`subgroupAdd`, `subgroupMax`) operate natively at `float` precision on all Vulkan 1.2+ implementations.

### §3.4: `.comp` shader changes

Every `.comp` shader that declares storage-role or state-role buffer bindings is updated as follows:

1. **Buffer layout declarations:** replace `float data[]` with `STORAGE_TYPE data[]` (for storage-role buffers) or `STATE_TYPE data[]` (for state-role buffers). Compute-role buffer bindings keep `float data[]`.

   ```glsl
   // Before (forward_pass.comp):
   layout(set = 0, binding = 0) readonly buffer InputBuf   { float data[]; } src_input;
   layout(set = 0, binding = 2) readonly buffer WeightsBuf { float data[]; } src_weights;

   // After:
   layout(set = 0, binding = 0) readonly buffer InputBuf   { STORAGE_TYPE data[]; } src_input;
   layout(set = 0, binding = 2) readonly buffer WeightsBuf { STATE_TYPE data[]; }   src_weights;
   ```

2. **Load sites:** every read from a storage-role buffer wraps the element through `read_storage()`; state-role reads through `read_state()`.

   ```glsl
   // Before:
   float mask_val = src_sample_mask.data[b_idx];
   float accum    = src_biases.data[h_idx];

   // After:
   float mask_val = read_storage(src_sample_mask.data[b_idx]);
   float accum    = read_state(src_biases.data[h_idx]);
   ```

3. **Store sites:** every write to a storage-role buffer wraps the value through `write_storage()`; state-role writes through `write_state()`.

   ```glsl
   // Before:
   dest_hidden.data[out_idx] = accum;

   // After:
   dest_hidden.data[out_idx] = write_storage(accum);
   ```

4. **`shared float` scratch arrays** are unchanged — they are compute-role, always `float`.

5. **Push constants** are unchanged — they carry scalar arithmetic values (learning rate, beta, epsilon) at compute precision, always `float`.

6. **Integer buffers** (`uint`, `int`) are unchanged.

Shaders that contain only compute-role and integer buffer bindings (`normalize_gradients.comp`) require no changes.

The role assignment for each buffer in each `.comp` shader follows the tables in ADR-021 §1.4 exactly.

---

## §4: Build System Implications

### §4.1: Vulkan SPIR-V — multi-variant compilation

The Meson build in `src/backends/vulkan/meson.build` currently calls `glslc` once per `.comp` file. Under this ADR the precision-variant flags are injected. Three precision combinations correspond to the three `PrecisionConfig` factories:

| Variant suffix | `STORAGE_TYPE` | `STATE_TYPE` | `ENABLE_FP16_EXTENSION` |
|:---|:---|:---|:---|
| `_fp32` | `float` | `float` | (not set) |
| `_s16fp32` | `float16_t` | `float` | `1` |
| `_fp16` | `float16_t` | `float16_t` | `1` |

For shaders that contain no storage-role or state-role typed buffers, only the `_fp32` variant is built (no `STORAGE_TYPE`/`STATE_TYPE` substitution occurs; single SPIR-V is sufficient).

The Vulkan backend's `_pipeline_cache.py` is updated to select the correct SPIR-V variant path based on `PrecisionConfig.storage_dtype` and `PrecisionConfig.state_dtype` at pipeline creation time. This is an extension of the existing pipeline cache parameterisation; the file path convention (`kernel_name_fp32.spv`, `kernel_name_s16fp32.spv`, `kernel_name_fp16.spv`) is the only new artifact.

### §4.2: CPU — `DECLARE_PRECISION_STRUCTS` instantiation changes

The three instantiations in `cpu_kernels.h` change as follows:

| Old instantiation | New instantiation |
|:---|:---|
| `DECLARE_PRECISION_STRUCTS(fp32, float)` | `DECLARE_PRECISION_STRUCTS(s32x32, float, float)` |
| `DECLARE_PRECISION_STRUCTS(fp16, _Float16)` | `DECLARE_PRECISION_STRUCTS(s16x16, _Float16, _Float16)` |
| `DECLARE_PRECISION_STRUCTS(fp64, double)` | `DECLARE_PRECISION_STRUCTS(s16x32, _Float16, float)` |

The `fp64` instantiation is retired; it had no deployed use case and `PrecisionConfig` defines no FP64 factory. The `s16x32` instantiation (FP16 storage, FP32 state) replaces it as the canonical mixed-precision CPU configuration.

The `.c` file that instantiates the `.inc` templates currently defines `REAL_T` and `PRECISION_SUFFIX` before including `cpu_precision.h`. It must now define both pairs:

```c
// instantiation for s16x32 (mixed FP16 storage / FP32 state):
#define STORAGE_T       _Float16
#define STORAGE_SUFFIX  fp16
#define STATE_T         float
#define STATE_SUFFIX    fp32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
```

When `STORAGE_T == STATE_T`, `simd_load_storage_fp32` and `simd_load_state_fp32` resolve to the same function, which is inlined identically. No redundancy in the emitted code.

### §4.3: CPU — `fp64` variant retirement and FFI impact

The Python FFI layer (ADR-015) currently binds against `CceChunkArgs_fp64`, `AdamUpdateArgs_fp64`, etc. These bindings are updated to their `s32x32`, `s16x16`, `s16x32` equivalents. The Python dispatch table in `src/backends/cpu/_dispatch_table.py` is updated to select the struct suffix based on `PrecisionConfig.storage_dtype` and `PrecisionConfig.state_dtype`. This is ADR-015 scope but is flagged here as the coupling point.

---

## §5: Cross-Backend Summary

| Backend | Precision boundary mechanism | COMPUTE_TYPE | Build-time parameterisation |
|:---|:---|:---|:---|
| OpenCL | Inline `load_storage()`/`store_storage()`/`load_state()`/`store_state_update()` in `kernels.cl.h`; `vload_half`/`vstore_half` when `STORAGE_TYPE_IS_HALF` | `COMPUTE_TYPE` (typically `float`) via `-D` | `-DSTORAGE_TYPE`, `-DCOMPUTE_TYPE`, `-DSTATE_TYPE`, `-DSTORAGE_TYPE_IS_HALF` per ADR-022 §7.1 |
| CPU | `simd_load_storage()`/`simd_load_state()` macros via `cpu_precision.h`; two-axis `STORAGE_SUFFIX`/`STATE_SUFFIX` dispatch; `cpu_compute_t = float` invariant | `float` — fixed CPU backend constant | `DECLARE_PRECISION_STRUCTS(SUFFIX, STORAGE_T, STATE_T)` per precision variant in `cpu_kernels.h` |
| Vulkan | `read_storage()`/`write_storage()`/`read_state()`/`write_state()` helpers in `common.glsl`; `STORAGE_TYPE`/`STATE_TYPE` macros injected via `glslc -D`; `float16_t` SSBO elements under FP16 configs | `float` — fixed Vulkan backend constant | Three SPIR-V variants per storage/state-role-bearing shader (`_fp32`, `_s16fp32`, `_fp16`) |

The "one codepath" invariant holds in all three backends: the conversion helpers are always present at the code level; the compiler/SPIR-V optimizer eliminates them when types are equal.

---

## §6: Verification Obligations

Per ADR-021 §5 (extended here to cover all backends), the following obligations apply across all three backends:

1. **Abstraction completeness:** Every load from a storage-role or state-role buffer passes through the role-appropriate abstraction. `grep -n 'STORAGE_TYPE\b\|STATE_TYPE\b\|STORAGE_TYPE\b\|STATE_TYPE\b' <source>` must not appear as a raw dereference on the left of `=` without the helper wrapper.

2. **Uniform-FP32 baseline invariance:** All three `PrecisionConfig.float32()` test runs must produce results bit-for-bit identical to the prior single-axis FP32 results. The precision boundary helpers with equal types must be genuinely zero-overhead in the compiled output.

3. **Migration completion criterion (extends ADR-021 §5's criterion to all backends):**
   - `grep -r simd_load_real architectures/averaging_ensembled_classifier/src/backends/cpu/` returns no results.
   - `grep -r REAL_T architectures/averaging_ensembled_classifier/src/backends/cpu/` returns no results (outside the transitional alias comment).
   - `grep -r 'float data\[\]' architectures/averaging_ensembled_classifier/src/backends/vulkan/kernel_sources/` returns no results for any storage-role or state-role buffer binding.
   - `grep -r fp64 architectures/averaging_ensembled_classifier/src/backends/cpu/` returns no results in non-comment code.

4. **Multi-configuration Tier 2 coverage:** Per ADR-020 §4.6, Tier 2 tests run against `PrecisionConfig.float32()`, `PrecisionConfig.float16()`, and `PrecisionConfig.mixed_f16_f32()` for each backend. The Vulkan backend's pipeline cache must select distinct SPIR-V variants for the FP32 and mixed configurations (verified by asserting different kernel file paths are loaded).

---

## Consequences

### Positive

- **ADR-021 §1.3 deferral fulfilled.** The precision boundary abstractions declared in the kernel specification document now have concrete implementations in all three backends.
- **CPU backend's existing three-role intuition is formalized.** The existing "compute is always float" practice becomes an architectural declaration. The storage/state split eliminates the last ambiguity in the CPU kernel struct ABI.
- **Vulkan FP16 storage is unblocked.** The `STORAGE_TYPE`/`STATE_TYPE` macro pattern in `common.glsl` and the multi-variant `glslc` compilation provide a GLSL-native mechanism for precision-role typed buffer binding. `PrecisionConfig.mixed_f16_f32()` can drive a Vulkan backend end-to-end.
- **Role assignment is visible in source.** Every buffer access — across all three backends — uses a named abstraction that encodes its role. Future readers can determine the precision role of any buffer from the access pattern alone, without consulting external tables.

### Negative

- **Vulkan SPIR-V artifact count increases.** Up to three SPIR-V variants are compiled per shader with storage or state role buffers. This increases build time and artifact footprint. For the initial supported factory set (three configurations), this doubles or triples the Vulkan artifact count for affected shaders.
- **CPU struct ABI is a breaking change.** `ForwardPassArgs_fp32`, `AdamUpdateArgs_fp32`, etc. have different field types after the `DECLARE_PRECISION_STRUCTS` extension. The Python FFI layer (`_dispatch_table.py`, `_ffi_types.py`) must be updated in lockstep. Any external consumer of the `libcpu_kernels` C ABI will observe field layout changes (pointer sizes are unchanged; the types differ).
- **`fp64` CPU variant is retired without replacement.** Code that explicitly constructed a `PrecisionConfig` equivalent to FP64 (outside of the three factory configurations) will no longer find a corresponding struct instantiation. This is acceptable given that FP64 was never a deployed production configuration and has no `PrecisionConfig` factory.
- **`_cross_subgroup_scratch` rename in `common.glsl`.** Any existing shader code that references `_cross_subgroup_scratch` by name (e.g., in inline comments or debug symbols) must be updated. The rename is a correctness-free documentation improvement.

### Not in scope

- Changes to `kernels/kernels.cl.h` specification content — covered by ADR-021.
- Changes to `phase_*.cl.c` type declarations — covered by ADR-021 §2.
- Changes to `PrecisionConfig`, `BufferDescriptor`, `StabilizationPolicy`, or any Python host module — covered by ADR-022.
- Changes to `CONCEPT.md` or `CONTRACT.md` authority text — covered by ADR-020 §§2–3.
- FP8 `STORAGE_TYPE=float8_e4m3` or equivalent — gated by ADR-020 §5 preconditions; the abstraction layer defined here structurally supports it but no `PrecisionConfig` factory exists.
- `_pipeline_cache.py` implementation details for SPIR-V variant selection — flagged in §4.1 but scoped to the Vulkan backend implementation sprint.

---

## References

- [ADR-013](ADR-013-kernel-source-strategy.md) — Kernel source layout (OpenCL at architecture root; Vulkan/CPU under `src/backends/<name>/kernel_sources/`)
- [ADR-014](ADR-014-build-system-integration.md) — Build targets for Vulkan SPIR-V and CPU shared library; extended by §4
- [ADR-015](ADR-015-python-native-backend-interop.md) — CPU FFI ABI; coupling point for §2.4 transitional alias and §4.3
- [ADR-020](ADR-020-mixed-precision.md) §2.3 — One-codepath mandate (no `#ifdef` on type equality); governs §3.1 strategy selection
- [ADR-020](ADR-020-mixed-precision.md) §4.4 — Kernel source strategy constraint; root mandate for precision boundary abstractions
- [ADR-020](ADR-020-mixed-precision.md) §4.5 — Build system implications; extended by §4.1
- [ADR-020](ADR-020-mixed-precision.md) §4.6 — Test strategy; extended by §6 item 4
- [ADR-021](ADR-021-kernels-precision-role-migration.md) §1.3 — Precision boundary abstraction declarations (this ADR provides the implementations)
- [ADR-021](ADR-021-kernels-precision-role-migration.md) §1.4 — Per-kernel role assignment tables (governing authority for §2.3 and §3.4)
- [ADR-021](ADR-021-kernels-precision-role-migration.md) §5 — Verification obligations (extended by §6)
- [ADR-022](ADR-022-host-code-precision-role-implications.md) §7.1 — OpenCL `build_compiler_flags()` symbol set (prerequisite for §1)
- [ADR-022](ADR-022-host-code-precision-role-implications.md) §7.2 — CPU type mapping (prerequisite for §2.2)
- [ADR-022](ADR-022-host-code-precision-role-implications.md) §7.3 — Vulkan type mapping (prerequisite for §3, §4.1)
