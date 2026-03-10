# ADR-006: Hardware Profile

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-003 (constrains), ADR-004 (constrains), ADR-005 (constrains)  
**Blocks:** ADR-008, ADR-012

---

## Context

Plan construction requires hardware constants to compute tiling, padding, reduction tree depth, and Node 16's `policy_max_k`. These constants are currently embedded in `DiscoveredArchConstants` in `cl_context_manager.py` — a frozen dataclass populated exclusively through OpenCL device queries. This creates two problems for the multi-backend refactoring:

1. **Backend coupling.** `DiscoveredArchConstants` lives inside the OpenCL context manager. Every module that consumes hardware constants — `ModelSpec`, `ParameterSpace`, `MemoryLayout`, `StabilizationPolicy`, and the plan builder — transitively depends on `cl_context_manager.py`.

2. **Precision coupling.** `DiscoveredArchConstants` inherits from `PrecisionContext`, bundling hardware constants with precision-specific type information (`SCALAR_NP_TYPE`, `SCALAR_C_TYPE_NAME`). This makes it impossible to represent a device's hardware capabilities independently of the chosen precision — even though constants like `cache_line_bytes` and `max_work_group_size` are precision-independent.

### Current `DiscoveredArchConstants` fields

| Field                                  | Source                                      | Consumers                                             |
| :------------------------------------- | :------------------------------------------ | :---------------------------------------------------- |
| `simd_width`                           | `device.preferred_vector_width_float/half`   | `ModelSpec` (padding), plan builder (tile count)      |
| `global_mem_cacheline_size`            | `device.global_mem_cacheline_size`           | `ModelSpec` (row stride padding)                      |
| `local_mem_size_bytes`                 | `device.local_mem_size`                      | Host validation of local memory allocations           |
| `optimal_workgroup_size_1d_reduction`  | `min(256, device.max_work_group_size)`       | `StabilizationPolicy` (`hardware_max_fan_in`)         |
| `optimal_square_tile_dim`              | `int(sqrt(max_work_group_size)) & ~1`        | Kernel build constant `C_TILE_SIZE`                   |
| `optimal_rectangular_tile_dim1`        | Hard-coded `16`                              | Kernel build constant for GEMM-like tiling            |

These field names embed OpenCL-specific heuristics. `optimal_workgroup_size_1d_reduction = min(256, max_work_group_size)` applies a GPU-tuned cap that may not be appropriate for Vulkan's subgroup-optimized reduction or CPU's cache-aware sequential accumulation. `optimal_square_tile_dim = sqrt(max_work_group_size)` is a GPU-specific heuristic — the Vulkan and CPU backends both derive tile size from subgroup/SIMD width instead (`VULKAN_BACKEND.md`: `.tile_size = device_props.subgroup_size`; `CPU_BACKEND.md`: `#define C_TILE_SIZE SIMD_WIDTH`).

### How the three backends discover hardware

- **OpenCL:** Runtime queries via `cl.device_info` — `preferred_vector_width_*`, `global_mem_cacheline_size`, `local_mem_size`, `max_work_group_size`, `global_mem_size`.
- **Vulkan:** `vkGetPhysicalDeviceProperties2` — `VkPhysicalDeviceSubgroupProperties.subgroupSize`, `maxComputeSharedMemorySize`, `maxComputeWorkGroupSize[0]`, `VkPhysicalDeviceMemoryProperties`.
- **CPU:** Compile-time ISA flags (`__AVX512F__` → `simd_width=16`, `__AVX2__` → 8, `__SSE2__` → 4, `__ARM_NEON` → 4, scalar fallback → 1). `CACHE_LINE_BYTES=64` is a platform constant. Memory is queried from the OS at runtime (`sysconf(_SC_PHYS_PAGES)`, etc.).

### What the shared layer actually consumes

Tracing through the consumers:

| Shared-layer consumer                                        | Constant needed        | How it's used                                         |
| :----------------------------------------------------------- | :--------------------- | :---------------------------------------------------- |
| `ModelSpec.__init__`                                          | `simd_width`           | `padded_hidden_dim` alignment                         |
| `ModelSpec.__init__`                                          | `cache_line_bytes`     | `padded_input_dim`, `padded_class_dim` row stride     |
| `StabilizationPolicy.plan_uniform_reduction_tree`            | `hardware_max_fan_in`  | Upper bound on uniform fan-in K                       |
| `StabilizationPolicy.get_specialized_reduction_policy_k`     | `hardware_max_fan_in`  | Upper bound on Node 16's `policy_max_k`               |
| `StreamingLoopPlan` construction (ADR-004)                    | `global_mem_bytes`     | Chunk count for Phase III                             |
| `MemoryLayout`                                                | (via `ModelSpec`)      | Padded dimensions drive byte-size calculations        |

Notably, `max_local_mem_bytes` is absent from this list. The shared layer does not consume it for Policy-tier derivations — it is used only by backend renderers for dispatch validation (e.g., verifying that local memory allocations fit) and by the plan builder for feasibility checks on local-memory-requiring dispatch nodes.

### The central design tension

The shared layer needs a value it calls `hardware_max_fan_in` — the maximum fan-in a single reduction dispatch can safely execute. The derivation of this value is fundamentally different per backend:

- **OpenCL:** `min(256, device.max_work_group_size)` — a GPU-tuned cap on the work-group size limit.
- **Vulkan:** `maxComputeWorkGroupSize[0]` — the hardware workgroup limit, potentially refined by subgroup size considerations.
- **CPU:** A function of L1 cache capacity and partial element size — structurally unrelated to work-group size, because the CPU has no work-groups.

These are not three instances of the same formula with different inputs. They are three different derivations reflecting three physically distinct constraints. The profile must bridge this divergence without introducing backend-discriminated logic in the shared layer.

---

## Decision Drivers

1. **ADR-001 (Plan boundary — shared layer produces backend-neutral data).** The `HardwareProfile` is consumed entirely by Policy-tier code. It must contain no backend-specific types and no backend-discriminator logic. The shared layer must not branch on which backend populated the profile.

2. **ADR-001 (Three-tier jurisdictional model — Policy tier is invariant).** The Policy tier is the only tier invariant across all node types and all backends. Every constant in the `HardwareProfile` must have a well-defined Policy-tier *role*, independent of its hardware origin.

3. **ADR-003 (Kernel tier selection is Orchestration-tier).** The register-vs-local crossover threshold is a backend-specific heuristic. The `HardwareProfile` must not prescribe it. The profile carries data from which each backend may derive its own crossover — but the crossover itself is not a profile field.

4. **CONCEPT.md §1 (Architectural Elegance Feedback).** The constant set is a formal vocabulary. If a new backend requires a hardware constant not in the profile, the response is to extend the profile with a formally documented field — not to add ad-hoc fields or bypass the profile.

5. **ADR-005 (`policy_max_k` synthesis requires hardware limits).** `StabilizationPolicy.get_specialized_reduction_policy_k(user_policy_k, hardware_max_fan_in)` resolves three constraints: user intent, hardware maximum, and mathematical safety. The profile must carry sufficient data for this synthesis.

6. **ADR-004 (`global_mem_bytes` determines streaming chunk count).** The Phase III streaming loop's chunk count is determined by host memory assessment. The profile must carry total device-accessible memory.

---

## Options Considered

### Option A: Raw hardware measurements with backend-category discriminator

The `HardwareProfile` carries raw, uninterpreted hardware measurements exactly as reported by each backend's native API. A `backend_category` enum field (`GPU_OPENCL`, `GPU_VULKAN`, `CPU`) allows the shared layer to apply category-specific derivation formulas when computing Policy-tier values.

```python
class BackendCategory(enum.Enum):
    GPU_OPENCL = "gpu_opencl"
    GPU_VULKAN = "gpu_vulkan"
    CPU = "cpu"

@dataclass(frozen=True)
class HardwareProfile:
    backend_category: BackendCategory
    simd_width: int
    cache_line_bytes: int
    max_work_group_size: Optional[int]   # None for CPU
    max_local_mem_bytes: Optional[int]   # None for CPU
    global_mem_bytes: int
```

The shared layer would derive `hardware_max_fan_in` with category-branching logic:

```python
if profile.backend_category == BackendCategory.CPU:
    hardware_max_fan_in = _derive_cpu_fan_in(profile)
else:
    hardware_max_fan_in = profile.max_work_group_size
```

**Advantages:**
- Maximum transparency. Every field is a direct, auditable hardware measurement with no reinterpretation.
- Fields carry their literal hardware meaning — `max_work_group_size` always means the actual maximum work-group size.

**Disadvantages:**
- **Violates ADR-001's backend-neutrality.** The shared layer must branch on `backend_category` to derive Policy-tier values. This is backend-discriminator logic in the shared codebase — the exact coupling the refactoring eliminates.
- **Growing case analysis.** Each new backend category adds a branch to every shared-layer derivation. Adding a SYCL or Metal backend requires modifying the shared layer's derivation logic, not just implementing a new `PlanRenderer`.
- **`max_work_group_size` is Optional for CPU.** The CPU backend has no work-groups. This introduces `None`-handling in the shared layer for a field that the shared layer always needs (it derives `hardware_max_fan_in` from it). The shared layer must guard against `None` or fall back to a default — a type-level contradiction for a universally-consumed value.

### Option B: Semantic-contract constants — uniform interface with hardware-origin naming

The `HardwareProfile` carries constants with hardware-origin names but with uniform *semantic contracts* that each backend maps its native measurements into. No backend discriminator; no branching in the shared layer.

```python
@dataclass(frozen=True)
class HardwareProfile:
    simd_width: int
    cache_line_bytes: int
    max_work_group_size: int
    max_local_mem_bytes: Optional[int]
    global_mem_bytes: int
```

Each backend maps its native measurements into the uniform vocabulary:

| Field                 | OpenCL                                   | Vulkan                          | CPU                           |
| :-------------------- | :--------------------------------------- | :------------------------------ | :---------------------------- |
| `simd_width`          | `preferred_vector_width_*`               | `subgroupSize`                  | Compile-time ISA flag         |
| `cache_line_bytes`    | `global_mem_cacheline_size`              | Assumed 64 (or queried)         | Platform constant (64)        |
| `max_work_group_size` | `min(256, device.max_work_group_size)`   | `maxComputeWorkGroupSize[0]`    | Thread pool size              |
| `max_local_mem_bytes` | `local_mem_size`                         | `maxComputeSharedMemorySize`    | `None`                        |
| `global_mem_bytes`    | `global_mem_size`                        | Memory heap queries             | `sysconf` queries             |

The shared layer derives `hardware_max_fan_in = max_work_group_size` uniformly. The CPU backend sets `max_work_group_size` to its own appropriate limit.

**Advantages:**
- No backend discriminator. The shared layer consumes the profile uniformly with zero branching.
- Universal constants (`simd_width`, `cache_line_bytes`, `global_mem_bytes`) map naturally to all backends.
- `max_local_mem_bytes` is Optional, handling the CPU case cleanly.

**Disadvantages:**
- **Semantic overloading of `max_work_group_size`.** For GPUs, this is a genuine hardware limit on work-group size. For CPU, it would be repurposed to mean "thread pool size" or "max sequential reduction batch" — a forced reinterpretation that obscures the field's actual meaning.
- **CPU's fan-in limit is unrelated to work-group size.** The CPU's maximum efficient fan-in for reduction is determined by L1 cache capacity and partial element size — fundamentally different from GPU work-group limits. Mapping this to `max_work_group_size` creates a semantic gap: the number stored in the field is not a work-group size in any meaningful sense for the CPU backend.
- **Validation confusion.** A test checking `max_work_group_size` would have different interpretations depending on which backend produced the profile, even though the shared layer is supposed to be backend-neutral. The field's name promises more specificity than its actual contract delivers.

### Option C: Policy-input profile — constants named for their plan-construction role

The `HardwareProfile` carries constants whose names describe their *consumption* in the Policy tier, not their hardware origin. Each backend fills these fields from its native measurements through whatever derivation is appropriate. The naming contract is defined by what the shared layer needs, not by what the hardware reports.

```python
@dataclass(frozen=True)
class HardwareProfile:
    simd_width: int
    cache_line_bytes: int
    max_reduce_fan_in: int
    max_local_mem_bytes: Optional[int]
    global_mem_bytes: int
```

Each backend computes `max_reduce_fan_in` through its native logic:

| Backend  | `max_reduce_fan_in` derivation                                                                     |
| :------- | :------------------------------------------------------------------------------------------------- |
| OpenCL   | `min(256, device.max_work_group_size)` — the current `optimal_workgroup_size_1d_reduction`         |
| Vulkan   | `maxComputeWorkGroupSize[0]` (or subgroup-optimized limit)                                         |
| CPU      | L1-cache-derived: `l1_data_cache_bytes / (elements_per_partial × scalar_byte_width)`, capped       |

The shared layer uses `max_reduce_fan_in` directly as the `hardware_max_fan_in` parameter to `StabilizationPolicy`.

**Advantages:**
- **Clean naming.** `max_reduce_fan_in` says exactly what the Policy tier cares about: the maximum fan-in for a single reduction dispatch. No semantic gap between the field name and its Policy-tier role.
- **No backend discrimination in the shared layer.** Each backend computes its own `max_reduce_fan_in` from its native capabilities. The shared layer never asks "which backend are you?"
- **Each backend applies its own hardware-specific heuristics.** The OpenCL `min(256, ...)` cap stays in the OpenCL backend. The CPU's cache-based derivation stays in the CPU backend. These are backend-tuning decisions, not Policy-tier decisions.
- **The profile matches its consumption site.** `StabilizationPolicy` calls `plan_uniform_reduction_tree(num_partials, max_reduce_fan_in)` — the profile field name matches the formal parameter name at the call site.

**Disadvantages:**
- **Pushes a Policy-tier derivation into backend code.** The `max_reduce_fan_in` value is consumed by Policy-tier `StabilizationPolicy`. Moving the derivation into each backend means the contract (the derivation formula) is implemented three times instead of once.
- **Risk of cross-backend divergence.** If the fan-in limit semantics evolve (e.g., a future ADR refines the definition), each backend's derivation must be updated independently.

---

## Analysis

### Eliminating Option A

Option A is eliminated. The `backend_category` discriminator reintroduces backend-specific logic into the shared layer — the exact coupling that ADR-001's refactoring is designed to eliminate. Every shared-layer consumer that derives a Policy-tier value from raw measurements would contain a `match backend_category:` branch. Adding a new backend would require modifying the shared layer's derivation logic, violating the principle that new backends require only a new `PlanRenderer`, not shared-layer changes.

Furthermore, `max_work_group_size: Optional[int]` for CPU forces `None`-handling at every consumption site. The field is consumed universally (it feeds `hardware_max_fan_in`), but it is sometimes absent. This is a type-level contradiction — a universally-consumed value should not be optionally present. The alternative of providing a fake sentinel value (e.g., `sys.maxsize` for CPU) disguises the semantic mismatch behind an arbitrary constant.

### Choosing between Option B and Option C

The decision between B and C turns on a precise question: **where does the `hardware_max_fan_in` derivation belong?**

The derivation across backends reveals that it is not a single formula applied to raw data — it is a fundamentally different computation per backend:

- **OpenCL:** `min(256, device.max_work_group_size)` — a GPU-tuned cap on the hardware limit.
- **Vulkan:** `maxComputeWorkGroupSize[0]` — direct hardware limit, potentially with subgroup-size considerations.
- **CPU:** A function of L1 cache capacity, partial element size, and scalar byte width — structurally unrelated to work-group size.

If this were a single formula with different inputs, it would belong in the shared layer. But it is three *different formulas* reflecting three *physically distinct constraints*. Placing all three in the shared layer requires either backend-discriminated branching (Option A, eliminated) or semantic overloading of a hardware-origin field name (Option B).

Option B forces the CPU backend to express its L1-cache-derived fan-in limit as a "work-group size" — a concept that does not exist in the CPU execution model. `CPU_BACKEND.md` §Threading Model explicitly maps GPU work-groups to single-thread task executions ("one thread owns the whole 'work-group'"). The CPU's fan-in limit for sequential accumulation inside a task has no relationship to this mapping. Encoding it as `max_work_group_size` creates a field whose documented meaning ("maximum cooperating items in one dispatch") disagrees with its actual content for one of the three backends.

**Option C avoids this because `max_reduce_fan_in` means exactly one thing across all backends:** the maximum number of input partials that a single reduction dispatch can efficiently process. The derivation is backend-specific because the physical mechanism that limits fan-in is backend-specific:

| Backend  | Physical limiting mechanism                                                                                    |
| :------- | :------------------------------------------------------------------------------------------------------------- |
| OpenCL   | Work-group size — all items must synchronize via local-memory barriers within one work-group                   |
| Vulkan   | Workgroup size — same constraint, via subgroup/shared-memory barriers                                          |
| CPU      | L1/L2 cache capacity — all partials must remain cache-resident for efficient sequential accumulation           |

These are genuinely different physical constraints that happen to serve the same Policy-tier role. The profile contracts the role, not the mechanism.

### The "Policy-tier derivation in backend code" concern

Option C's principal disadvantage — pushing a Policy-tier derivation into backend code — merits scrutiny.

ADR-003 establishes a precedent that resolves this cleanly. ADR-003 determines that kernel tier selection (register vs. local) is an **Orchestration-tier** concern, even though the fan-in `K` that informs it is a **Policy-tier** value. The principle: **a computation's tier is determined by what it *needs to know*, not by what *consumes its result*.** The `max_reduce_fan_in` derivation needs to know:

- L1 cache capacity and line size (CPU),
- `maxComputeWorkGroupSize` and subgroup properties (Vulkan),
- `device.max_work_group_size` plus a GPU-tuning heuristic (OpenCL).

These are *backend-intrinsic facts*, not model-level or policy-level facts. The derivation belongs in the backend for the same reason kernel tier selection belongs in the backend: it requires knowledge that only the backend possesses.

What the Policy tier needs is the *result* — a single integer representing the maximum safe fan-in. The `HardwareProfile` is the channel that transports this result across the plan boundary. This parallels the pattern ADR-003 uses for the `ReductionTreePlan`: the plan carries pre-computed threshold schedules (Policy-tier output) without exposing how the Quadratic Scaling Policy computes them. Here the direction is reversed — the backend carries a pre-computed fan-in limit without exposing the hardware-specific derivation — but the boundary discipline is the same: pure data crosses the boundary, derivation logic stays on the side that possesses the requisite knowledge.

### Cross-backend divergence risk

The risk that backends diverge on `max_reduce_fan_in` semantics is real but bounded. The field's contract is testable: for any backend, `max_reduce_fan_in` must be ≥ 2 (a fan-in < 2 is nonsensical for reduction), must produce a valid reduction tree when passed to `StabilizationPolicy.plan_uniform_reduction_tree()`, and must not exceed what the backend can actually execute. These invariants can be validated at profile-construction time, independent of which backend produced the profile.

---

## Decision

**Option C: Policy-input profile — constants named for their plan-construction role.**

The `HardwareProfile` is a frozen dataclass in the shared layer that carries hardware-informed constants named for their Policy-tier consumption role. Each backend fills this profile from its native hardware discovery mechanism. The shared layer consumes the profile without backend-specific branching.

### The canonical constant set

```python
@dataclass(frozen=True)
class HardwareProfile:
    simd_width: int
    cache_line_bytes: int
    max_reduce_fan_in: int
    max_local_mem_bytes: Optional[int]
    global_mem_bytes: int
```

| Field                | Type            | Policy-tier role                                                                                                                           | Consumed by                                                                                                 |
| :------------------- | :-------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- |
| `simd_width`         | `int`           | SIMD-aligned padding for hidden dimension; tile count calculation                                                                          | `ModelSpec`, plan builder                                                                                   |
| `cache_line_bytes`   | `int`           | Cache-line-aligned row stride padding                                                                                                      | `ModelSpec`                                                                                                 |
| `max_reduce_fan_in`  | `int`           | Upper bound on uniform reduction fan-in K                                                                                                  | `StabilizationPolicy.plan_uniform_reduction_tree()`, `StabilizationPolicy.get_specialized_reduction_policy_k()` |
| `max_local_mem_bytes`| `Optional[int]` | `None` for backends without addressable work-group-local memory. Carried for plan-time validation of local-memory dispatch node feasibility | Plan builder (validation only)                                                                              |
| `global_mem_bytes`   | `int`           | Streaming loop chunk count (Phase III); host memory budget assessment                                                                      | `StreamingLoopPlan` construction                                                                            |

### Backend population contracts

Each backend must fill `HardwareProfile` from its native hardware discovery mechanism. The field contracts specify what each value must represent; the derivation is backend-owned.

**OpenCL:**

```python
HardwareProfile(
    simd_width=device.preferred_vector_width_float,  # or _half for FP16
    cache_line_bytes=device.global_mem_cacheline_size or 64,
    max_reduce_fan_in=min(256, device.max_work_group_size),
    max_local_mem_bytes=device.local_mem_size,
    global_mem_bytes=device.global_mem_size,
)
```

**Vulkan:**

```python
HardwareProfile(
    simd_width=subgroup_properties.subgroupSize,
    cache_line_bytes=64,  # no direct Vulkan query; platform assumption
    max_reduce_fan_in=device_properties.limits.maxComputeWorkGroupSize[0],
    max_local_mem_bytes=device_properties.limits.maxComputeSharedMemorySize,
    global_mem_bytes=sum(
        heap.size for heap in memory_properties.memoryHeaps
        if heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    ),
)
```

**CPU:**

```python
HardwareProfile(
    simd_width=COMPILED_SIMD_WIDTH,       # from compiled shared library's ISA detection
    cache_line_bytes=64,                  # platform constant
    max_reduce_fan_in=l1_cache_limit,     # L1 data cache / (elements_per_partial * sizeof(scalar))
    max_local_mem_bytes=None,             # CPU has no work-group-local memory
    global_mem_bytes=os_physical_memory,  # sysconf(_SC_PHYS_PAGES) * page_size
)
```

### `max_reduce_fan_in` derivation contract

Each backend's derivation of `max_reduce_fan_in` must satisfy these invariants:

1. **Lower bound:** `max_reduce_fan_in >= 2`. A fan-in of 1 produces an infinite-depth tree; a fan-in of 0 is undefined.
2. **Hardware fidelity:** The value must represent a fan-in that the backend can actually execute in a single reduction dispatch without correctness failure. Over-reporting causes correctness violations; under-reporting causes unnecessary tree depth.
3. **Stability across invocations:** For the same device and precision, the value must be deterministic.

The shared layer validates invariant (1) at profile-construction time. Invariants (2) and (3) are backend responsibilities — the profile contract documents them; backend tests verify them.

### Precision interaction

The current `DiscoveredArchConstants` inherits from `PrecisionContext`, coupling hardware constants to precision type information. `HardwareProfile` does **not** inherit from `PrecisionContext`. The two concerns are orthogonal:

- Hardware constants like `cache_line_bytes`, `max_reduce_fan_in`, and `global_mem_bytes` are precision-independent.
- `simd_width` is precision-dependent — FP16 may have twice the SIMD width of FP32 on the same device.

The backend produces a precision-appropriate `simd_width` when constructing the profile. When the system operates at a specific precision, the backend queries the hardware for that precision's SIMD width and fills the profile accordingly. The plan builder receives both a `HardwareProfile` and a `PrecisionContext` (ADR-008) as separate inputs:

```python
plan = build_execution_plan(
    model_spec=model_spec,              # includes PrecisionContext
    hardware_profile=hardware_profile,  # precision-appropriate simd_width
    stabilization_policy=policy,
    batch_params=batch_params,
)
```

This decoupling means a test harness can construct a `HardwareProfile(simd_width=4, cache_line_bytes=64, max_reduce_fan_in=256, max_local_mem_bytes=None, global_mem_bytes=8*1024**3)` without importing any precision or backend types — enabling pure shared-layer unit tests with synthetic hardware profiles that exercise edge cases (e.g., `max_reduce_fan_in=2` for maximum tree depth, or `simd_width=1` for scalar fallback).

### Derived constants — not in the profile

The following constants are deliberately excluded from `HardwareProfile`:

| Constant                            | Why excluded                                                                                                                                                                                                        | Where it lives                                                        |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-------------------------------------------------------------------- |
| `c_tile_size`                       | Convergent across backends: Vulkan = `subgroupSize`, CPU = `SIMD_WIDTH`, OpenCL = backend-specific heuristic. The plan builder derives a default from `simd_width`; backends may override at kernel compilation time | Plan builder (from `simd_width`) or backend kernel build options      |
| `optimal_rectangular_tile_dim1`     | Currently hard-coded to `16`. A tuning constant for GEMM-like kernel dispatch, not a hardware measurement. This is an Orchestration-tier concern                                                                    | Backend kernel compilation options                                    |
| Register/local crossover threshold  | Orchestration-tier concern per ADR-003. Each backend applies its own heuristic (OpenCL: `16`; Vulkan: subgroup-size-based; CPU: no distinction)                                                                     | Backend renderer                                                      |

### Upstream constraint satisfaction

**ADR-003 (`hardware_max_fan_in` for reduction trees):** Satisfied. `StabilizationPolicy.plan_uniform_reduction_tree(num_partials, profile.max_reduce_fan_in)` receives the fan-in limit directly from the profile. The register/local crossover threshold that ADR-003 identifies as an Orchestration-tier concern is correctly excluded from the profile.

**ADR-004 (`global_mem_bytes` for streaming chunk count; `simd_width` for padding):** Satisfied. `global_mem_bytes` is a direct profile field. `simd_width` feeds `ModelSpec`'s padded dimensions, which in turn feed `MemoryLayout` shapes, which in turn feed `ScratchBufferSpec` sizes in the `StreamingLoopPlan`.

**ADR-005 (`policy_max_k` synthesis):** Satisfied. `StabilizationPolicy.get_specialized_reduction_policy_k(user_policy_k, profile.max_reduce_fan_in)` uses `max_reduce_fan_in` as the hardware limit in its three-constraint resolution (user intent, hardware maximum, mathematical safety).

---

## Consequences

### Positive

- **Backend-neutral shared layer.** No `pyopencl`, `vulkan`, or platform-specific imports in the shared layer. The `HardwareProfile` is a plain frozen dataclass with primitive fields. `ModelSpec`, `StabilizationPolicy`, and the plan builder consume it without knowing which backend produced it.

- **Testable in isolation.** Any test can construct a `HardwareProfile` with hand-chosen values. No device required, no backend required. This enables comprehensive plan-builder unit tests with synthetic hardware profiles that exercise edge cases (e.g., `max_reduce_fan_in=2` for maximum tree depth, or `simd_width=1` for scalar fallback).

- **Clean migration from `DiscoveredArchConstants`.** The OpenCL backend's profile-construction code is a direct refactoring of `OpenCLContextManager.build_and_discover()`. The mapping is:

  | `DiscoveredArchConstants` field          | `HardwareProfile` field  |
  | :--------------------------------------- | :----------------------- |
  | `simd_width`                             | `simd_width`             |
  | `global_mem_cacheline_size`              | `cache_line_bytes`       |
  | `optimal_workgroup_size_1d_reduction`    | `max_reduce_fan_in`      |
  | `local_mem_size_bytes`                   | `max_local_mem_bytes`    |
  | (not present)                            | `global_mem_bytes`       |

- **Extensible.** When a future backend requires an additional hardware constant for Policy-tier plan construction, the profile is extended with a new field via a formal ADR update, per CONCEPT.md §1 (Architectural Elegance Feedback).

### Negative

- **`max_reduce_fan_in` derivation is per-backend.** Three backends implement three derivations. If the semantic contract of "max reduction fan-in" evolves, all three must be updated. This is mitigated by the testable invariants (≥ 2, hardware-faithful, deterministic) and by integration tests that verify plan correctness with each backend's profile.

- **`max_local_mem_bytes: Optional[int]` adds a nullability concern.** Consumers must handle `None`. However, the only shared-layer consumer is plan-time validation (checking that local-memory-using dispatch nodes are feasible), where `None` naturally means "local memory is not available — local-memory kernels cannot be used." Backend renderers that always have local memory (OpenCL, Vulkan) can assert non-`None` in their own code.

- **`simd_width` is precision-dependent, but the profile is not precision-typed.** The profile does not statically enforce that its `simd_width` matches the precision context. This is a runtime correctness obligation on the backend: the profile must be constructed for the target precision. A mismatch (e.g., supplying FP32 `simd_width` when building an FP16 plan) would produce incorrect padding. This is detectable by cross-checking `simd_width` against `PrecisionContext.SCALAR_NP_TYPE().itemsize` at plan-construction time.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure; shared layer produces backend-neutral data; three-tier jurisdictional model
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `hardware_max_fan_in` requirement; kernel-tier crossover as Orchestration-tier concern
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `global_mem_bytes` for chunk count; padding inputs for scratch buffer sizing
- [ADR-005: Node 16 Opacity](ADR-005-node-16-opacity-in-the-plan.md) — `policy_max_k` synthesis from hardware profile data
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — decoupled precision context; interaction with hardware capabilities
- [CPU_BACKEND.md](../CPU_BACKEND.md) — SIMD compile-time detection, cache line constants, threading model
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — subgroup size discovery, specialization constants, memory queries
