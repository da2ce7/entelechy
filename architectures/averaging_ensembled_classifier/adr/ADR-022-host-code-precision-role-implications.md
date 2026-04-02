# ADR-022: Host Code Implications of the Three-Role Precision Model

**Status:** PROPOSED
**Date:** 2026-04-02
**Deciders:** —
**Triggered by:** ADR-020 (Three-Role Precision Model) §4 and ADR-021 (Kernel Precision-Role Migration) §5
**Depends on:** ADR-020, ADR-021, ADR-007, ADR-008, ADR-009, ADR-014, ADR-015, ADR-016
**Constrains:** All Python source files under `src/`

---

## Context

ADR-020 established the three-role precision model as a sovereign-authority primitive and specified the architectural implications at the design layer (§4). ADR-021 specified the complete migration of the kernel source corpus. Neither ADR provided an exhaustive enumeration of the host-code artifacts that must change, nor the exact scope of each change. This ADR provides that enumeration.

The host orchestration layer is the consumer of `PrecisionConfig` and the producer of everything downstream: `StabilizationPolicy` initialisation, build-time compiler flag generation, buffer allocation sizing, kernel contract validation, and test fixtures. Under the former single-axis `SCALAR_TYPE` model, a single `numpy_dtype` flowed everywhere unchanged. Under the three-role model, three independent dtypes flow to the right destinations — storage buffers, arithmetic boundaries, and persistent state — and must not be conflated.

The affected artifacts fall into five categories:

1. **Shared configuration** — `PrecisionConfig`, `ModelSpec`, `StabilizationPolicy`
2. **Shared plan layer** — `BufferDescriptor`, `BufferParamSpec` / `KernelContract`, `Plan builder`
3. **Backend type mappings** — OpenCL, CPU, Vulkan
4. **Backend buffer allocation sites** — Vulkan renderer, CPU allocator
5. **Test fixtures** — `test_stabilization_policy.py`, `test_reduction_logic.py`

---

## Decision Drivers

1. **ADR-020 §4 is binding.** The design-layer implications listed there constitute required changes, not suggestions. This ADR translates those headings into change-level specifics for every affected source file.

2. **CONCEPT.md §6 — Authority hierarchy.** All implications must be derived from the conceptual and contractual amendments in ADR-020 §§2–3, not independently invented. Every change in this ADR cites the ADR-020 or ADR-021 provision that requires it.

3. **No ambiguity at boundary crossings.** Every site in the host code that was previously parameterised by a single `numpy_dtype` must now make an explicit role choice: storage, compute, or state. Implicit fallback to "the" precision type must be eliminated.

4. **Breaking change is centralised.** `PrecisionConfig`'s three-field redesign (ADR-020 §4.1) is the root change. All other implications cascade from it. Isolating the cascade in one ADR allows the migration to proceed with a clear dependency order.

---

## Decision

The following changes are required across the host source tree. They are specified in dependency order: each section's changes are not blocked by later sections.

---

### §1: `src/shared/precision_config.py` (ADR-020 §4.1)

The current three-field design (`numpy_dtype`, `fp_format_max`, `epsilon`) is replaced entirely. The new design is:

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
        assert self.storage_dtype.itemsize <= self.compute_dtype.itemsize
        assert self.storage_dtype.itemsize <= self.state_dtype.itemsize
        assert self.compute_fp_format_max >= self.storage_fp_format_max
```

Three factory classmethods replace the existing two:

| Factory | `storage_dtype` | `compute_dtype` | `state_dtype` |
|:---|:---|:---|:---|
| `PrecisionConfig.float32()` | `float32` | `float32` | `float32` |
| `PrecisionConfig.float16()` | `float16` | `float16` | `float16` |
| `PrecisionConfig.mixed_f16_f32()` | `float16` | `float32` | `float32` |

The retired fields `numpy_dtype`, `fp_format_max`, and `epsilon` are removed. No compatibility shim is provided — all consumers are updated in the same migration. `ModelSpec` factories delegate to the new `PrecisionConfig` factories unchanged (`PrecisionConfig.float32()` etc. continue to exist by name).

---

### §2: `src/shared/model_spec.py` (ADR-020 §4.2, §3.3)

#### §2.1: Padding calculation dtype selections

The four padding properties use a single `self.precision.numpy_dtype` today. Under the three-role model, each uses the role-appropriate dtype:

| Property | Padding dimension role | Dtype to use |
|:---|:---|:---|
| `padded_hidden_dim` | SIMD alignment in element-count (not byte-count) | No change — alignment is in `COMPUTE_TYPE` elements; the `simd_width` is already in compute-type elements |
| `padded_input_dim` | Cache-line alignment of input activation row bytes | `storage_dtype` — inputs are transient DAG data |
| `padded_class_dim` | Cache-line alignment of logit/output row bytes | `storage_dtype` — logits are transient DAG data |
| `padded_module_dim` | Cache-line alignment of module-aggregation row bytes | `storage_dtype` — per-module transient aggregations |

In effect: the three cache-line-aligned properties change from `np.dtype(self.precision.numpy_dtype).itemsize` to `self.precision.storage_dtype.itemsize`.

`padded_hidden_dim` divides by `simd_width` in the element domain and never touches `itemsize`, so it requires no change.

#### §2.2: Deprecated backward-compat properties

`ModelSpec.SCALAR_NP_TYPE` and `ModelSpec.SCALAR_C_TYPE_NAME` delegated to `precision.numpy_dtype`. Under the three-role model the concept of a single "scalar type" does not exist. These properties are removed.

Any caller that still uses `model_spec.SCALAR_NP_TYPE` or `model_spec.SCALAR_C_TYPE_NAME` must be identified and migrated to explicitly choose the role-appropriate dtype before this removal completes.

---

### §3: `src/shared/stabilization_policy.py` (ADR-020 §2.2, §4.7)

The Quadratic Scaling Policy safety ceiling is $T_{\text{safety}_j} = \text{COMPUTE\_FP\_FORMAT\_MAX} / K_j$ (CONCEPT.md §3.4 as amended by ADR-020 §2.2). The `StabilizationPolicy` dataclass field `fp_format_max` is renamed to `compute_fp_format_max` to reflect its contractual role.

The field docstring is updated to read:

> The maximum representable value of the **compute precision** format. Governs overflow safety during reduction tree summation. Under the three-role model, the safety ceiling is bounded by arithmetic precision, not storage precision.

No other algorithmic changes. The formula using this field is unchanged in structure; only the field name and its semantic scope become explicit.

---

### §4: `src/main_orchestrator.py` (ADR-020 §2.5, §4.7)

`StabilizationPolicy` is currently constructed as:

```python
fp_max = float(np.finfo(model_spec.precision.numpy_dtype).max)
self.stabilization_policy = StabilizationPolicy(
    t_algorithmic=hyperparams.stabilization.max_grad_norm,
    lambda_=hyperparams.stabilization.lambda_,
    fp_format_max=fp_max,
)
```

Under the three-role model, `fp_max` is already computed and stored by `PrecisionConfig` as `compute_fp_format_max`. The `np.finfo` call is removed. The construction becomes:

```python
self.stabilization_policy = StabilizationPolicy(
    t_algorithmic=hyperparams.stabilization.max_grad_norm,
    lambda_=hyperparams.stabilization.lambda_,
    compute_fp_format_max=model_spec.precision.compute_fp_format_max,
)
```

This eliminates the ad-hoc re-derivation of a value already authorised in `PrecisionConfig`, and ensures the correct precision role (compute) is used even under mixed-precision configurations.

---

### §5: `src/shared/buffer_lifecycle.py` (ADR-020 §4.2)

`BufferDescriptor` gains a mandatory `precision_role` field:

```python
precision_role: Literal["storage", "compute", "state"]
```

Positioned after the existing `role: BufferRole` field. The `element_size_bytes` and `size_bytes` fields continue to be computed values, but their derivation at all call sites must use the role-appropriate dtype's `itemsize` from the active `PrecisionConfig` (see §8 below for plan builder call sites).

`Literal` requires `from typing import Literal`.

---

### §6: `src/shared/kernel_contracts/` — `BufferParamSpec` and `LocalMemorySpec` (ADR-020 §4.3, ADR-021 §1.3)

#### §6.1: `BufferParamSpec`

Every `BufferParamSpec` in all six kernel contract modules gains:

```python
precision_role: Literal["storage", "compute", "state"]
```

The role assigned to each buffer parameter is given by the per-kernel role assignment tables in ADR-021 §1.4. This field is mandatory. Plan-construction validation (ADR-007 contract binding) must verify that the `precision_role` of each bound buffer's `BufferDescriptor` matches the corresponding `BufferParamSpec.precision_role`.

#### §6.2: `LocalMemorySpec` — `sizeof(SCALAR_TYPE)` → `sizeof(COMPUTE_TYPE)`

Per ADR-021 §1.4, `__local` scratch buffers are always `COMPUTE_TYPE`. All `sizeof(SCALAR_TYPE)` occurrences in `LocalMemorySpec` strings have been updated to `sizeof(COMPUTE_TYPE)`.

---

### §7: Backend Type Mappings

#### §7.1: `src/backends/opencl/type_mapping.py` (ADR-020 §4.5, §3.6)

`build_compiler_flags()` generates the full Article 6 symbol set:

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
    return flags
```

`numpy_dtype_to_cl_type_name(precision)` is replaced by the internal `_dtype_to_cl_type(dtype)` helper that accepts an `np.dtype` directly, removing the coupling to a single-dtype `PrecisionConfig`.

The transitional `SCALAR_TYPE` alias block (previously present during Phase 7B migration) has been removed — all kernel signatures now use the three-role precision model exclusively.

#### §7.2: `src/backends/cpu/type_mapping.py` (ADR-020 §4.2)

The single `get_numpy_dtype(precision)` function is replaced by three role-specific functions:

```python
def get_storage_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.storage_dtype

def get_compute_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.compute_dtype

def get_state_dtype(precision: PrecisionConfig) -> np.dtype:
    return precision.state_dtype
```

Callers that previously used `get_numpy_dtype(precision)` must be updated to call the role-appropriate function. For the CPU backend's buffer allocator, which accepts a single `dtype` at construction time, a separate allocator instance is constructed per precision role (storage, compute, state) when the plan requires heterogeneous allocation.

#### §7.3: `src/backends/vulkan/type_mapping.py` (ADR-020 §4.2)

`get_numpy_dtype(precision)` and `get_element_size(precision)` are replaced by role-specific equivalents matching the CPU pattern in §7.2. `get_specialization_scalar_is_half(precision)` splits into:

```python
def get_storage_is_half(precision: PrecisionConfig) -> int:
    return 1 if precision.storage_dtype == np.dtype(np.float16) else 0

def get_compute_is_half(precision: PrecisionConfig) -> int:
    return 1 if precision.compute_dtype == np.dtype(np.float16) else 0
```

---

### §8: Plan Builder — `BufferDescriptor` Construction (ADR-020 §4.2)

All `BufferDescriptor` construction sites in `src/shared/plan_builder.py` and `src/backends/vulkan/renderer.py` must:

1. Supply the new mandatory `precision_role` field with the appropriate role string per ADR-021 §1.4.
2. Derive `element_size_bytes` from the role-appropriate dtype: `precision.storage_dtype.itemsize`, `precision.compute_dtype.itemsize`, or `precision.state_dtype.itemsize` accordingly.

The Vulkan renderer's reduction ping/pong/offset buffer descriptors (`src/backends/vulkan/renderer.py` lines ~358–389) currently hardcode `element_size_bytes=4`. These buffers are reduction tree intermediate buffers:

- Ping/pong buffers initially hold `"storage"`-role partial sums; after the final reduction stage they hold `"compute"`-role accumulated gradients. Until the final stage, `storage_dtype.itemsize` drives sizing. The final output buffer is `"compute"`. Given that the renderer constructs a single pair of buffers and ping-pongs across stages, the conservative correct choice is `compute_dtype.itemsize` so that the arithmetic-output final value is never truncated. This matches the role assignment for reduction tree output buffers in ADR-021 §1.4 (`"compute"`).
- The offset descriptor carries `int` elements (not floating-point); it is unchanged.

---

### §9: Test Fixtures (ADR-020 §4.6)

#### §9.1: `src/tests/test_stabilization_policy.py`

All `StabilizationPolicy(fp_format_max=...)` constructions are updated to `StabilizationPolicy(compute_fp_format_max=...)`.

#### §9.2: `src/tests/test_reduction_logic.py`

All `StabilizationPolicy(..., fp_format_max=...)` constructions are updated. The values (`3.4e38` for FP32 compute) are unchanged — they were already compute-precision values; the rename makes the role explicit.

#### §9.3: New multi-configuration coverage

Per ADR-020 §4.6, Tier 2 tests must execute against all active `PrecisionConfig` factory configurations, including at least one configuration where roles diverge. Concretely:

- `PrecisionConfig.float32()` — existing coverage
- `PrecisionConfig.float16()` — existing coverage (FP16 tests)
- `PrecisionConfig.mixed_f16_f32()` — **new: must be added**

Any test that parameterises over a `PrecisionConfig` instance must include `PrecisionConfig.mixed_f16_f32()` as a test case. Numerical tolerance for mixed-precision comparison is derived from `compute_dtype` (FP32 in `mixed_f16_f32`), not from `storage_dtype`.

---

## Migration Order

The following order respects the intra-host dependency structure:

1. **§1 — `PrecisionConfig`** — root change; all others depend on its new interface.
2. **§3 — `StabilizationPolicy`** — no callers beyond §4; can proceed immediately after §1.
3. **§2 — `ModelSpec`** — depends on §1; no plan-layer dependency.
4. **§5 — `BufferDescriptor`** — foundational for §6 validation and §8 call sites.
5. **§6 — Kernel contracts** — depends on §5 for `precision_role` vocabulary.
6. **§7 — Backend type mappings** — each backend is independent.
7. **§8 — Plan builder and Vulkan renderer** — depends on §5, §6, §7.
8. **§4 — `main_orchestrator.py`** — depends on §1 and §3.
9. **§9 — Tests** — depends on §§1, 3.

---

## Consequences

### Positive

- **Host code fully expresses the three-role model.** Every site that previously collapsed precision to a single type becomes either an explicit role choice or a contract-validated role annotation. There are no implicit "use the precision" sites remaining.
- **`mixed_f16_f32` is immediately usable end-to-end.** Once §§1–8 are complete, calling `PrecisionConfig.mixed_f16_f32()` at the top of a training run will correctly route FP16 to storage buffer allocations, FP32 to `StabilizationPolicy`'s safety ceiling, FP32 to OpenCL compiler flags for the arithmetic symbols, and FP32 to optimizer state buffers — with no additional code changes.
- **Stability ceiling is correct under mixed precision.** The current code computes `fp_max = np.finfo(precision.numpy_dtype).max` and passes it as the safety ceiling. For a `mixed_f16_f32` configuration, this would have derived FP16's maximum (~65504) — causing grossly conservative clip thresholds on a FP32-compute reduction tree. The §3 and §4 changes make it impossible to supply storage precision where compute precision is required.
- **Buffer allocation is role-accurate.** Under mixed precision, storage buffers are smaller (FP16 = 2 bytes/element) and accumulation/state buffers are wider (FP32 = 4 bytes/element). The §5 and §8 changes ensure that correct byte counts flow to every allocation call.
- **`LocalMemorySpec` sizes match the actual kernel type.** The `sizeof(SCALAR_TYPE)` expressions were accurate under the uniform model but become incorrect once `SCALAR_TYPE` is aliased to `COMPUTE_TYPE` and local scratch is declared `COMPUTE_TYPE`. The §6.2 change ensures consistency immediately on alias introduction.

### Negative

- **`PrecisionConfig` is a breaking API change.** All construction sites (`ModelSpec` factories, test fixtures, hypothetical user-constructed instances) must be updated. There is no deprecation path — the old fields are absent. The migration must be coordinated across all of §§2–9 in a single propagation pass.
- **`BufferDescriptor` and `BufferParamSpec` gain a mandatory field.** Any construction site not yet enumerated here will produce a runtime `TypeError` (missing keyword argument). This is a self-revealing failure mode, but it requires that test coverage be broad enough to exercise every construction path.
- **Backend type mapping signatures change.** Any external code (beyond the backends in `src/`) that calls `numpy_dtype_to_cl_type_name(precision)`, `get_numpy_dtype(precision)`, `get_element_size(precision)`, or `get_specialization_scalar_is_half(precision)` will break. These are internal APIs per ADR-001's three-tier boundary, so no external callers are expected — but this should be verified before the migration commit.

### Not in scope

- Changes to `kernels/` source files — covered by ADR-021.
- Changes to `CONCEPT.md` or `CONTRACT.md` authority text — covered by ADR-020 §§2–3.
- Changes to `meson.build` build configurations for the three-symbol kernel compilation — covered by ADR-020 §4.5 and ADR-014.
- FP8 `PrecisionConfig` factories — gated by ADR-020 §5 preconditions.

---

## References

- [ADR-020](ADR-020-mixed-precision.md) §4.1 — `PrecisionConfig` redesign (supersedes ADR-008); root mandate for §1
- [ADR-020](ADR-020-mixed-precision.md) §4.2 — `BufferDescriptor.precision_role` and role-aware sizing; mandate for §§5, 8
- [ADR-020](ADR-020-mixed-precision.md) §4.3 — `KernelContract`/`BufferParamSpec` `precision_role`; mandate for §6.1
- [ADR-020](ADR-020-mixed-precision.md) §4.4 — Kernel source strategy (constrains §6.2)
- [ADR-020](ADR-020-mixed-precision.md) §4.5 — Build system implications; mandate for §7.1
- [ADR-020](ADR-020-mixed-precision.md) §4.6 — Test strategy; mandate for §9.3
- [ADR-020](ADR-020-mixed-precision.md) §4.7 — Safety ceiling uses `compute_fp_format_max`; mandate for §§3, 4
- [ADR-020](ADR-020-mixed-precision.md) §2.2 — CONCEPT.md safety ceiling amendment (root authority for §§3, 4)
- [ADR-020](ADR-020-mixed-precision.md) §3.3 — Padding in role-type elements (mandate for §2.1)
- [ADR-021](ADR-021-kernels-precision-role-migration.md) §1.4 — Per-kernel role assignment tables (needed for §§6.1, 8)
- [ADR-021](ADR-021-kernels-precision-role-migration.md) §3 — Transitional alias protocol (mandate for §7.1 alias block)
- [ADR-007](ADR-007-kernel-signature-contract-binding-split.md) — KernelContract/KernelBinding split (extended by §6.1)
- [ADR-008](ADR-008-precision-configuration.md) — Superseded by ADR-020 §4.1; superseded design implemented in `precision_config.py`
- [ADR-009](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` (extended by §5)
- [ADR-014](ADR-014-build-system-integration.md) — Build system (constrained; see §7.1)
- [ADR-015](ADR-015-python-native-backend-interop.md) — Python/native interop boundary (context for §7.2 CPU type mapping)
- [ADR-016](ADR-016-test-strategy.md) — Test strategy (extended by §9)
