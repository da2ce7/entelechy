# ADR-008: Precision Configuration

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-006 (constrains)  
**Blocks:** ADR-012

---

## Context

`PrecisionContext` is currently an abstract base class in `arch_primitives.py` with two abstract properties — `SCALAR_NP_TYPE` (a numpy dtype) and `SCALAR_C_TYPE_NAME` (a string like `"float"` or `"half"` for the OpenCL compiler). Two concrete subclasses, `Float32Context` and `Float16Context`, provide fixed implementations. The type hierarchy permeates the system through multiple inheritance:

- `ModelSpec` inherits from `PrecisionContext`, yielding `Float32ModelSpec` and `Float16ModelSpec`.
- `DiscoveredArchConstants` inherits from `PrecisionContext`, yielding `Float32DiscoveredArchConstants` and `Float16DiscoveredArchConstants`.
- `ComputeEnvironment` inherits from `PrecisionContext`, yielding `Float32ComputeEnvironment` and `Float16ComputeEnvironment`.

This design creates three problems for the multi-backend refactoring:

### 1. Backend-specific field in the shared layer

`SCALAR_C_TYPE_NAME` returns `"float"` or `"half"` — OpenCL C type names. The property's own docstring states it is "for the OpenCL compiler." This field is meaningless for Vulkan (which uses SPIR-V type IDs and `VkFormat` enums) and for the CPU backend (which uses `float` or `_Float16` / compiler-emulated FP16). Placing a backend-specific property on a shared-layer abstraction violates ADR-001's plan boundary: the shared layer must contain no backend-specific types or logic.

### 2. Inheritance-based precision propagation

Precision propagates through class hierarchy composition — `ModelSpec` *is-a* `PrecisionContext`, `DiscoveredArchConstants` *is-a* `PrecisionContext`. This creates two subclass variants of every consuming type. Adding a new precision (e.g., BFloat16) requires adding a third subclass of `ModelSpec`, `DiscoveredArchConstants`, and `ComputeEnvironment`. The class count grows as `O(precisions × consuming_types)`.

ADR-006 (ACCEPTED) already breaks this pattern for hardware constants: `HardwareProfile` is a standalone frozen dataclass with no `PrecisionContext` inheritance. The same decoupling is needed for `ModelSpec` and all remaining consumers.

### 3. Coupling between precision and hardware discovery

`DiscoveredArchConstants` inherits from both `PrecisionContext` and `abc.ABC`, making it impossible to represent hardware capabilities independently of precision. ADR-006 eliminates this coupling for hardware constants. This ADR eliminates it for precision itself — making `PrecisionContext` a standalone, self-contained dataclass that is composed with (not inherited by) its consumers.

### What the shared layer actually consumes from precision

Tracing through the codebase, precision information is consumed in three distinct roles:

| Consumer | What it needs | How it uses it |
| :--- | :--- | :--- |
| `ModelSpec.padded_input_dim`, `padded_class_dim`, `padded_module_dim` | `numpy_dtype.itemsize` (byte width) | Cache-line-aligned row stride: `padded_row_bytes = ceil(dim × itemsize / cache_line_bytes) × cache_line_bytes` |
| `StabilizationPolicy.__init__` | `fp_format_max` (float) | Safety ceiling: `T_safety_j = fp_format_max / K_j` (CONCEPT.md §3.4) |
| `StabilizationPolicy.__init__` | `min_threshold` (indirectly precision-derived) | Signal annihilation floor |
| Plan builder → kernel scalar params | `epsilon` (float) | Build-time symbol `NUMERICAL_STABILITY_EPSILON` (CONTRACT.md Article 6) |
| `MemoryLayout` / `ParameterSpace` | `numpy_dtype` | Buffer byte-size: `total_bytes = element_count × numpy_dtype.itemsize` |
| `KernelContract` local memory specs (ADR-007) | `numpy_dtype.itemsize` | Symbolic size formulas: `SIMD_WIDTH * (SIMD_WIDTH + 1) * scalar_bytes` |

No shared-layer consumer needs `SCALAR_C_TYPE_NAME`. Every shared-layer consumer needs either the numpy dtype (for byte arithmetic) or a precision-derived scalar constant (`fp_format_max`, `epsilon`).

### What backends need from precision

Each backend maps the shared-layer `numpy_dtype` to its native type system at render time:

| Backend | Mapping from `numpy_dtype` | Mechanism |
| :--- | :--- | :--- |
| OpenCL | `np.float32` → `"float"`, `np.float16` → `"half"` | `-D SCALAR_TYPE=float` compiler flag; `-D SCALAR_IS_HALF=0\|1` |
| Vulkan | `np.float32` → `float` (GLSL), `np.float16` → `float16_t` (GLSL) | Specialization constants; `VkFormat` selection for buffer views |
| CPU | `np.float32` → `float` (C), `np.float16` → `_Float16` or emulated | Compile-time `#define` or template instantiation |

These mappings are one-to-one and deterministic from `numpy_dtype`. No backend needs additional shared-layer assistance to derive its native type — the numpy dtype is sufficient.

### The `Float32Context` / `Float16Context` hierarchy

The current design uses two concrete classes with hardcoded property returns. Under the new model, a single `PrecisionConfig` frozen dataclass parameterized by `numpy_dtype` replaces the entire hierarchy. The precision-derived constants (`fp_format_max`, `epsilon`) are computed at construction time from the dtype, not encoded in class methods.

---

## Decision Drivers

1. **ADR-001 (Plan boundary — shared layer produces backend-neutral data).** The `PrecisionConfig` is consumed at plan-construction time by `ModelSpec`, `StabilizationPolicy`, and the plan builder. It must contain no backend-specific types, no backend-specific strings, and no backend-discriminated logic. `SCALAR_C_TYPE_NAME` violates this directly.

2. **ADR-001 (Three-tier jurisdictional model — Policy tier is invariant).** `PrecisionConfig` carries Policy-tier constants (`fp_format_max`, `epsilon`, `numpy_dtype`). The mapping from `numpy_dtype` to backend-native types is an Orchestration-tier concern. The boundary must separate these roles.

3. **ADR-006 (HardwareProfile is precision-decoupled).** ADR-006 establishes that `HardwareProfile` does not inherit from `PrecisionContext` — the two are separate inputs to the plan builder. This ADR must produce a `PrecisionConfig` that is composable alongside `HardwareProfile` without inheritance coupling.

4. **ADR-006 (`simd_width` is precision-dependent).** The backend constructs `HardwareProfile` for the target precision — FP16 may have twice the SIMD width of FP32. The `PrecisionConfig` must provide enough information for a plan-time cross-check, but must not embed SIMD width itself (that is a hardware constant, not a precision constant).

5. **CONTRACT.md Article 6 (Mandatory Build-Time Symbols).** `SCALAR_TYPE`, `SCALAR_IS_HALF`, and `NUMERICAL_STABILITY_EPSILON` are build-time symbols the host must provide. The shared layer must carry the precision-derived values (`epsilon`, `numpy_dtype`) from which the backend can compute these symbols. The symbols themselves are backend-specific rendering concerns.

6. **CONCEPT.md §3.4 (Safety ceiling calculations).** `FP_FORMAT_MAX` is a precision-specific constant used by `StabilizationPolicy` to compute `T_safety_j = FP_FORMAT_MAX / K_j`. This must be a first-class field on the precision configuration, not hidden in a subclass property.

7. **CONCEPT.md §1 (Architectural Elegance Feedback).** If a new precision format (e.g., BFloat16) creates tension with the configuration's field set, the response is to extend the dataclass — not to add a new subclass hierarchy.

---

## Options Considered

### Option A: Frozen dataclass with `numpy_dtype` as the canonical precision identifier

Replace the `PrecisionContext` ABC and its two subclasses with a single frozen dataclass. The `numpy_dtype` field (e.g., `numpy.float32`, `numpy.float16`) is the canonical identifier from which all other precision properties are derived. Precision-derived constants (`fp_format_max`, `epsilon`) are explicit fields, computed at construction time. No backend-specific fields.

```python
@dataclass(frozen=True)
class PrecisionConfig:
    numpy_dtype: type          # e.g., numpy.float32, numpy.float16
    fp_format_max: float       # e.g., 3.4028235e+38 (FP32), 65504.0 (FP16)
    epsilon: float             # e.g., 1e-7 (FP32), 1e-3 (FP16)
```

Consumers access `precision_config.numpy_dtype` for byte arithmetic and `precision_config.fp_format_max` / `precision_config.epsilon` for numerical policy. Backends map `numpy_dtype` to their native type system at render time.

`ModelSpec` receives `PrecisionConfig` as a composed field rather than inheriting from it:

```python
@dataclass(frozen=True)
class ModelSpec:
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    simd_width: int
    cache_line_bytes: int
    precision: PrecisionConfig
```

**Advantages:**
- **No backend-specific fields.** `SCALAR_C_TYPE_NAME` is eliminated from the shared layer entirely. The shared layer's type vocabulary is `numpy.dtype` exclusively.
- **No class hierarchy.** `Float32Context`, `Float16Context`, `Float32ModelSpec`, `Float16ModelSpec`, `Float32DiscoveredArchConstants`, etc. are all eliminated. Adding BFloat16 requires constructing a new `PrecisionConfig(numpy_dtype=numpy.bfloat16, ...)` instance — no new classes.
- **Explicit precision-derived constants.** `fp_format_max` and `epsilon` are visible, documented fields — not hidden behind abstract property resolution. A reader of `StabilizationPolicy.__init__` can see that it receives `fp_format_max` from the precision config without tracing through an inheritance chain.
- **Composable with `HardwareProfile`.** Both are independent frozen dataclasses passed to the plan builder. No diamond inheritance, no MRO complications.
- **Testable.** Any test can construct `PrecisionConfig(numpy_dtype=numpy.float32, fp_format_max=3.4e38, epsilon=1e-7)` directly. No abstract classes to subclass, no mixins to compose.

**Disadvantages:**
- **Construction responsibility.** The caller must provide correct `fp_format_max` and `epsilon` values for the given `numpy_dtype`. An incorrect `fp_format_max` for FP16 (e.g., `3.4e38` instead of `65504.0`) would produce silently wrong safety ceilings.
- **No compile-time precision-type safety via the Python type system.** The current `Float32ModelSpec` / `Float16ModelSpec` distinction lets `@overload` signatures guarantee type-level precision matching (as in `OpenCLContextManager.build_and_discover`). Under this option, `ModelSpec` is a single type — a function receiving `ModelSpec` cannot statically distinguish FP32 from FP16 via the Python type checker.
- **Largest refactoring scope.** Every module that currently inherits from `PrecisionContext` (`ModelSpec`, `DiscoveredArchConstants`, `ComputeEnvironment`, every concrete subclass) must be restructured.

### Option B: Frozen dataclass with retained `bit_width` field for backend mapping convenience

Same as Option A, but with an additional `bit_width: int` field (16 or 32) to simplify backend type mapping. Backends would use `bit_width` instead of inspecting `numpy_dtype.itemsize * 8`.

```python
@dataclass(frozen=True)
class PrecisionConfig:
    numpy_dtype: type
    bit_width: int             # 16 or 32
    fp_format_max: float
    epsilon: float
```

**Advantages:**
- All advantages of Option A.
- `bit_width` is a simple integer that backends can match on without importing numpy: `if precision.bit_width == 16: scalar_type = "half"`.

**Disadvantages:**
- All disadvantages of Option A.
- **Redundant field.** `bit_width` is derivable from `numpy_dtype`: `numpy.dtype(numpy_dtype).itemsize * 8`. Storing it as a separate field creates a consistency obligation — the caller must ensure `bit_width` matches `numpy_dtype.itemsize * 8`. A mismatch would be a silent contract violation.
- **Violates ADR-001's data minimalism precedent.** ADR-006 deliberately excluded `c_tile_size` from `HardwareProfile` because it was derivable from `simd_width`. The same principle applies here: derivable values do not become profile fields.

### Option C: Retain ABC hierarchy with backend-neutral interface

Keep `PrecisionContext` as an ABC but remove `SCALAR_C_TYPE_NAME`. Retain `SCALAR_NP_TYPE` (renamed to `numpy_dtype`). Add `fp_format_max` and `epsilon` as abstract properties. Keep `Float32Context` and `Float16Context` as concrete implementations.

```python
class PrecisionContext(abc.ABC):
    @property
    @abc.abstractmethod
    def numpy_dtype(self) -> type: ...
    
    @property
    @abc.abstractmethod
    def fp_format_max(self) -> float: ...
    
    @property
    @abc.abstractmethod
    def epsilon(self) -> float: ...

class Float32Context(PrecisionContext):
    @property
    def numpy_dtype(self) -> type: return numpy.float32
    @property
    def fp_format_max(self) -> float: return 3.4028235e+38
    @property
    def epsilon(self) -> float: return 1e-7

class Float16Context(PrecisionContext):
    @property
    def numpy_dtype(self) -> type: return numpy.float16
    @property
    def fp_format_max(self) -> float: return 65504.0
    @property
    def epsilon(self) -> float: return 1e-3
```

`ModelSpec` continues to inherit from `PrecisionContext`; `Float32ModelSpec` and `Float16ModelSpec` continue to exist.

**Advantages:**
- **Compile-time precision-type safety.** `Float32ModelSpec` and `Float16ModelSpec` remain as distinct types. The `@overload` pattern in `OpenCLContextManager.build_and_discover()` continues to work — the type checker can statically verify that an FP32 environment receives an FP32 model spec.
- **No construction-time consistency risk.** `Float32Context` always returns correct FP32 constants from its properties. There is no opportunity for a caller to misconfigure `fp_format_max` for the wrong dtype.
- **Smallest refactoring scope.** Only `SCALAR_C_TYPE_NAME` is removed. The inheritance chains remain structurally intact.

**Disadvantages:**
- **Preserves the inheritance chain.** `ModelSpec` is still a `PrecisionContext`. `DiscoveredArchConstants` is still a `PrecisionContext`. Every consuming type still needs precision-specific subclasses. Adding BFloat16 still requires `BFloat16Context`, `BFloat16ModelSpec`, `BFloat16DiscoveredArchConstants`, etc.
- **ADR-006 pattern divergence.** ADR-006 explicitly chose composition over inheritance for `HardwareProfile`. Keeping inheritance for `PrecisionContext` creates an inconsistency in the system's design vocabulary. The two inputs to the plan builder would use different composition patterns — one is an independent dataclass, the other is mixed into consumer types via inheritance.
- **Cannot be inspected as pure data.** An ABC with abstract properties is an interface, not a data structure. The plan builder cannot inspect `PrecisionContext` as frozen data — it must call properties, which are method dispatches. This contradicts ADR-001's plan-as-data-structure principle.
- **`DiscoveredArchConstants` coupling persists.** Even with `SCALAR_C_TYPE_NAME` removed, `DiscoveredArchConstants` still inherits from `PrecisionContext`. ADR-006 requires that `HardwareProfile` has no `PrecisionContext` parent — but the migration from `DiscoveredArchConstants` to `HardwareProfile` is more complex when precision inheritance remains.

---

## Analysis

### Eliminating Option B

Option B is eliminated for the same reason ADR-006 excluded `c_tile_size` from `HardwareProfile`: derivable values do not become fields on a canonical configuration object. `bit_width` is `numpy.dtype(numpy_dtype).itemsize * 8` — a trivial, deterministic derivation. Storing it as a field creates a redundancy that must be kept consistent. If a caller constructs `PrecisionConfig(numpy_dtype=numpy.float16, bit_width=32, ...)`, the system has contradictory precision information with no structural mechanism to detect the conflict.

The convenience argument ("backends can match on `bit_width` without importing numpy") is unpersuasive. Backends already import numpy for buffer creation and data marshalling. A backend that needs the bit width can derive it locally: `scalar_bytes = numpy.dtype(precision.numpy_dtype).itemsize`. This is a one-line derivation, not a computational burden.

### Eliminating Option C

Option C preserves the inheritance-based precision propagation that is the root cause of the problems this ADR addresses. While removing `SCALAR_C_TYPE_NAME` fixes the backend-specificity issue, the class hierarchy remains:

- `Float32ModelSpec(ModelSpec, Float32Context)` — a diamond inheriting from two ABCs.
- `Float32DiscoveredArchConstants(DiscoveredArchConstants, Float32Context)` — another diamond.
- Every new precision adds one subclass per consuming type.

ADR-006 established a clear precedent: `HardwareProfile` is a standalone frozen dataclass, not an ABC hierarchy. The rationale in ADR-006 applies equally here: *precision-derived constants are data, not behavior*. `fp_format_max = 65504.0` is a number, not a method. Hiding it behind abstract property dispatch adds conceptual weight with no functional benefit.

Furthermore, Option C contradicts ADR-001's plan-as-data-structure principle. The execution plan must be an inspectable data object. If the plan references a `PrecisionContext` with abstract properties, it references an interface — plan inspection requires method dispatch, not field reads. Option A's frozen dataclass is pure data: `precision.fp_format_max` is a field read, not a property call.

The compile-time type-safety advantage of Option C — that `Float32ModelSpec` and `Float16ModelSpec` are distinct types for `@overload` — is real but narrowly applicable. The `@overload` pattern in `OpenCLContextManager.build_and_discover()` is itself a backend-specific API surface. Under ADR-001's refactoring, each backend has its own factory that constructs the environment. The shared plan builder receives `ModelSpec` and `PrecisionConfig` — it does not need to distinguish FP32 from FP16 at the type level, because it processes them identically. Runtime validation (verifying that `precision.numpy_dtype` is a supported type) subsumes what the type checker provided.

### Choosing Option A

Option A replaces the inheritance hierarchy with a single frozen dataclass — the same pattern ADR-006 applied to hardware constants. The `PrecisionConfig` is pure data: three fields, no methods, no properties, no abstract dispatch. It composes cleanly with `HardwareProfile` as a separate input to the plan builder, fulfilling ADR-006's expectation.

The construction-responsibility concern (that a caller might provide incorrect `fp_format_max` for a given `numpy_dtype`) is addressed by a factory function that centralizes the derivation:

```python
def make_precision_config(numpy_dtype: type) -> PrecisionConfig:
    """Construct a PrecisionConfig with correct derived constants for the given dtype."""
    info = numpy.finfo(numpy_dtype)
    return PrecisionConfig(
        numpy_dtype=numpy_dtype,
        fp_format_max=float(info.max),
        epsilon=_DTYPE_EPSILON_MAP[numpy_dtype],
    )
```

The factory is a convenience, not a mandate. Test code may construct `PrecisionConfig` directly with hand-chosen values (e.g., a synthetic `fp_format_max` to exercise safety-ceiling edge cases). Production code uses the factory for correctness.

---

## Decision

**Option A: Frozen dataclass with `numpy_dtype` as the canonical precision identifier.**

`PrecisionConfig` replaces the `PrecisionContext` ABC, `Float32Context`, `Float16Context`, and all precision-specific subclass variants across the system.

### The canonical dataclass

```python
@dataclass(frozen=True)
class PrecisionConfig:
    numpy_dtype: type
    fp_format_max: float
    epsilon: float
```

| Field | Type | Policy-tier role | Consumed by |
| :--- | :--- | :--- | :--- |
| `numpy_dtype` | `type` (numpy dtype class) | Canonical precision identifier; byte-size derivation via `numpy.dtype(numpy_dtype).itemsize` | `ModelSpec` (padding arithmetic), `MemoryLayout` (byte-size calculation), `KernelContract` local memory specs (ADR-007), backend renderers (native type mapping) |
| `fp_format_max` | `float` | Maximum finite representable value for the format; safety ceiling input | `StabilizationPolicy` (`T_safety_j = fp_format_max / K_j`), `StabilizationPolicy.get_specialized_reduction_policy_k` (math safety limit) |
| `epsilon` | `float` | Numerical stability guard; build-time symbol value | Plan builder → `NUMERICAL_STABILITY_EPSILON` kernel scalar parameter (CONTRACT.md Article 6) |

### Standard precision configurations

Two precision configurations are defined as canonical constants:

| Configuration | `numpy_dtype` | `fp_format_max` | `epsilon` | Source |
| :--- | :--- | :--- | :--- | :--- |
| FP32 | `numpy.float32` | `3.4028235e+38` | `1e-7` | `numpy.finfo(numpy.float32).max`; established convention |
| FP16 | `numpy.float16` | `65504.0` | `1e-3` | `numpy.finfo(numpy.float16).max`; widened for FP16 stability |

The `epsilon` values are not derived mechanically from `numpy.finfo().eps` (which is the machine epsilon — the smallest representable difference from 1.0). They are *operational* epsilon values — chosen for numerical stability in the specific context of division-by-zero prevention and gradient norm guards. FP16's epsilon (`1e-3`) is deliberately larger than FP16 machine epsilon (`~9.77e-4`) to provide a comfortable margin above the representability threshold.

### Factory function

```python
_DTYPE_EPSILON_MAP: Dict[type, float] = {
    numpy.float32: 1e-7,
    numpy.float16: 1e-3,
}

def make_precision_config(numpy_dtype: type) -> PrecisionConfig:
    if numpy_dtype not in _DTYPE_EPSILON_MAP:
        raise ValueError(
            f"Unsupported precision dtype: {numpy_dtype}. "
            f"Supported: {set(_DTYPE_EPSILON_MAP.keys())}"
        )
    info = numpy.finfo(numpy_dtype)
    return PrecisionConfig(
        numpy_dtype=numpy_dtype,
        fp_format_max=float(info.max),
        epsilon=_DTYPE_EPSILON_MAP[numpy_dtype],
    )
```

The factory validates that the requested dtype is a supported precision and derives `fp_format_max` from `numpy.finfo`. The `epsilon` is looked up from a curated map — not derived from `finfo.eps` — because operational epsilon is a system-level policy choice, not a mechanical property of the format.

Adding a new precision (e.g., BFloat16) requires:
1. Adding an entry to `_DTYPE_EPSILON_MAP` with the appropriate operational epsilon.
2. Verifying that `numpy.finfo(numpy.bfloat16).max` returns the correct format maximum.
3. No new classes, no new subclass variants, no modifications to `ModelSpec` or `StabilizationPolicy`.

### Consumer migration

**`ModelSpec`**: Changes from inheriting `PrecisionContext` to composing `PrecisionConfig`:

```python
@dataclass(frozen=True)
class ModelSpec:
    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    simd_width: int
    cache_line_bytes: int
    precision: PrecisionConfig

    @property
    def padded_input_dim(self) -> int:
        item_size = numpy.dtype(self.precision.numpy_dtype).itemsize
        row_bytes = self.input_dim * item_size
        padded_row_bytes = _pad_to_multiple(row_bytes, self.cache_line_bytes)
        return padded_row_bytes // item_size
    # ... padded_class_dim, padded_module_dim follow the same pattern
```

`Float32ModelSpec` and `Float16ModelSpec` are eliminated. A single `ModelSpec` class parameterized by `precision: PrecisionConfig` replaces the hierarchy.

**`StabilizationPolicy`**: Already receives `fp_format_max` as an explicit constructor parameter. No change needed — the orchestrator passes `precision_config.fp_format_max` when constructing the policy. The `StabilizationPolicy` does not depend on `PrecisionConfig` directly; it depends on the scalar value.

**`DiscoveredArchConstants`**: Eliminated by ADR-006. The hardware-constant role is served by `HardwareProfile`; the precision role is served by `PrecisionConfig`. The `Float32DiscoveredArchConstants` / `Float16DiscoveredArchConstants` classes and their `PrecisionContext` inheritance chain are fully dissolved.

**`ComputeEnvironment`**: Becomes backend-specific. Each backend defines its own environment type that composes a `HardwareProfile`, a `PrecisionConfig`, and its native resources (OpenCL: `cl.Context` + `cl.CommandQueue` + `cl.Program`; Vulkan: `VkDevice` + `VkQueue` + pipelines; CPU: thread pool + compiled library).

**Plan builder**: Receives `PrecisionConfig` as a separate input alongside `HardwareProfile` and `ModelSpec`:

```python
plan = build_execution_plan(
    model_spec=model_spec,              # ModelSpec (composes PrecisionConfig)
    hardware_profile=hardware_profile,  # HardwareProfile (independent)
    stabilization_policy=policy,
    batch_params=batch_params,
)
```

`model_spec.precision` provides `numpy_dtype`, `fp_format_max`, and `epsilon` to the plan builder. `hardware_profile` provides `simd_width`, `cache_line_bytes`, `max_reduce_fan_in`, `max_local_mem_bytes`, and `global_mem_bytes`. These are orthogonal inputs.

### Backend type-mapping contract

Each backend maps `PrecisionConfig.numpy_dtype` to its native type vocabulary at render time. The mapping is a backend-owned function — it does not appear in the shared layer.

| Backend | Render-time mapping | Mechanism |
| :--- | :--- | :--- |
| OpenCL | `numpy.float32` → `{SCALAR_TYPE: "float", SCALAR_IS_HALF: 0}` | `-D` flags to `clBuildProgram` |
|        | `numpy.float16` → `{SCALAR_TYPE: "half", SCALAR_IS_HALF: 1}` | |
| Vulkan | `numpy.float32` → `float` (GLSL) | Shader source selection or specialization |
|        | `numpy.float16` → `float16_t` (GLSL, `VK_KHR_16bit_storage`) | Required extension enablement |
| CPU | `numpy.float32` → `float` (C) | Compile-time typedef or template |
|     | `numpy.float16` → `_Float16` (C23 / compiler extension) or emulated | Platform-specific; may be unsupported |

The mapping from `numpy_dtype` to backend-native types is deterministic and one-directional. Each backend implements it as a simple lookup table or match statement. This is an Orchestration-tier concern — it crosses the plan boundary in the render direction only.

### Precision–hardware cross-check

ADR-006 notes that `simd_width` is precision-dependent but the `HardwareProfile` is not precision-typed. The plan builder performs a cross-check at plan-construction time:

```python
def _validate_precision_hardware_consistency(
    precision: PrecisionConfig,
    profile: HardwareProfile,
) -> None:
    scalar_bytes = numpy.dtype(precision.numpy_dtype).itemsize
    if profile.simd_width < 1:
        raise ValueError("HardwareProfile.simd_width must be >= 1")
    # The SIMD width must produce valid SIMD-aligned dimensions.
    # A width that doesn't evenly tile the scalar byte width is suspect
    # but not necessarily invalid (some hardware reports SIMD width in
    # elements regardless of type). This check flags the most common
    # misconfiguration: supplying an FP32 profile for an FP16 plan.
```

This is a lightweight sanity check, not a formal proof. The primary correctness obligation remains on the backend: it must construct the `HardwareProfile` for the target precision, querying the hardware's capabilities for that specific type width.

### CPU FP16 unsupported-precision handling

The CPU backend may not support FP16 — hardware-accelerated FP16 SIMD is not universally available (it requires AVX-512 FP16 on x86, or ARMv8.2-A FP16 on ARM). When a backend cannot support the requested precision, it must signal this at profile-construction time — before the plan builder runs.

The mechanism is straightforward: the backend's discovery function validates the requested precision against the hardware's capabilities and raises `UnsupportedPrecisionError` if it cannot be fulfilled. The orchestrator catches this at the top level, before any plan construction occurs.

```python
class UnsupportedPrecisionError(Exception):
    """Raised when a backend cannot support the requested precision."""
    pass
```

This is a backend-level concern, not a shared-layer concern. The shared layer's `PrecisionConfig` is valid for any supported numpy floating dtype. The decision of which precisions a specific backend on specific hardware can execute belongs to the backend's discovery module.

---

## Consequences

### Positive

- **Backend-neutral precision representation.** `PrecisionConfig` contains no OpenCL type names, no Vulkan format enums, no C type strings. The shared layer's precision vocabulary is `numpy.dtype` exclusively. Each backend's native type mapping is a render-time concern.

- **No class hierarchy.** The `PrecisionContext` ABC, `Float32Context`, `Float16Context`, `Float32ModelSpec`, `Float16ModelSpec`, `Float32DiscoveredArchConstants`, `Float16DiscoveredArchConstants`, `Float32ComputeEnvironment`, and `Float16ComputeEnvironment` are all eliminated — nine classes replaced by one frozen dataclass. Adding BFloat16 support requires one line in `_DTYPE_EPSILON_MAP`, not six new class definitions.

- **Testable in isolation.** Any test can construct `PrecisionConfig(numpy_dtype=numpy.float32, fp_format_max=3.4e38, epsilon=1e-7)` without importing any backend types or hardware discovery modules. Edge-case testing with synthetic values (e.g., `fp_format_max=1000.0` to stress-test safety ceiling logic, or `epsilon=0.5` to verify threshold flooring) is trivial.

- **Consistent design vocabulary with ADR-006.** Both `HardwareProfile` and `PrecisionConfig` are frozen dataclasses composed into (not inherited by) their consumers. The plan builder receives both as independent inputs. The system's two configuration surfaces — hardware capabilities and precision properties — follow the same structural pattern.

- **Explicit, auditable constants.** `fp_format_max` and `epsilon` are visible fields on the dataclass, not hidden behind abstract property dispatch chains. A reader examining `StabilizationPolicy(fp_format_max=precision_config.fp_format_max, ...)` can trace the value directly to its source without navigating inheritance hierarchies.

### Negative

- **Construction-time consistency obligation.** A caller constructing `PrecisionConfig` directly (not via the factory) must ensure `fp_format_max` matches `numpy_dtype`. A `PrecisionConfig(numpy_dtype=numpy.float16, fp_format_max=3.4e38, ...)` would silently produce wrong safety ceilings. This is mitigated by the factory function for production use and by review discipline for test code.

- **Loss of Python type-checker precision discrimination.** Functions that previously accepted `Float32ModelSpec` vs `Float16ModelSpec` as distinct types lose this static distinction under a single `ModelSpec`. The `@overload` pattern in `OpenCLContextManager.build_and_discover()` is no longer expressible via the type system. This is acceptable because: (a) the `@overload` pattern was itself backend-specific (it exists only in the OpenCL context manager), and (b) runtime validation (`assert isinstance(model_spec.precision.numpy_dtype, ...)`) provides equivalent safety where needed.

- **Refactoring scope.** Every module that inherits from `PrecisionContext` must be restructured: `model_spec.py`, `cl_context_manager.py`, `arch_primitives.py`, and all test files that construct precision-specific instances. This is a mechanically large but conceptually simple change — replacing inheritance with composition at each site.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure; shared layer produces backend-neutral data; three-tier jurisdictional model
- [ADR-006: Hardware Profile](ADR-006-hardware-profile.md) — `HardwareProfile` as standalone frozen dataclass; precision-decoupled; `simd_width` is precision-dependent but profile is not precision-typed; plan builder receives hardware profile and precision as separate inputs
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — local memory size formulas are symbolic to avoid coupling to precision configuration; `scalar_bytes` derived at bind time
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `shared/precision_config.py` directory placement; `PrecisionContext` dissolved per this ADR
- [CONTRACT.md](../CONTRACT.md) — Article 6 mandates `SCALAR_TYPE`, `SCALAR_IS_HALF`, `NUMERICAL_STABILITY_EPSILON` as build-time symbols
- [CONCEPT.md](../CONCEPT.md) — §3.4 safety ceiling calculation using `FP_FORMAT_MAX`
- [CPU_BACKEND.md](../CPU_BACKEND.md) — compile-time ISA detection; FP16 may be unsupported
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — specialization constants for build-time symbols; GLSL type vocabulary
