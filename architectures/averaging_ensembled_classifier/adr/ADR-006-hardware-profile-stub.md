# ADR-006: Hardware Profile

**Status:** STUB (NARROWED — Shared `HardwareProfile` dataclass)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-003 (constrains), ADR-004 (constrains), ADR-005 (constrains)  
**Blocks:** ADR-008, ADR-012

---

## Context

Plan construction requires hardware constants — `simd_width`, `c_tile_size`, `max_local_mem`, `cache_line_bytes` — to compute tiling, padding, reduction tree depth, and Node 16's `policy_max_k`. These must be available in the shared layer without importing any backend-specific types. Each backend fills the profile via its native discovery mechanism:

- **OpenCL:** Runtime discovery via `cl.device_info`.
- **Vulkan:** `VkPhysicalDeviceSubgroupProperties` and memory queries.
- **CPU:** Compile-time constants (`__AVX512F__` → `simd_width=16`, etc.) reported by the compiled shared library.

`ParameterSpace`, `MemoryLayout`, `StabilizationPolicy`, and the plan builder consume this dataclass. These modules require zero backend-specific changes once `DiscoveredArchConstants` is replaced.

---

## Narrowed Direction

A shared `HardwareProfile` frozen dataclass populated by each backend.

---

## Remaining Decision

The canonical constant set. A minimal proposal:

| Constant             | Type  | Source (OpenCL)          | Source (Vulkan)                        | Source (CPU)             |
| :------------------- | :---- | :----------------------- | :------------------------------------- | :----------------------- |
| `simd_width`         | `int` | `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` | `subgroupSize`        | Compile-time ISA flag    |
| `c_tile_size`        | `int` | Derived from simd_width  | Derived from subgroup size             | Derived from simd_width  |
| `max_local_mem`      | `int` | `CL_DEVICE_LOCAL_MEM_SIZE` | `maxComputeSharedMemorySize`         | N/A (no local memory)   |
| `max_work_group_size`| `int` | `CL_DEVICE_MAX_WORK_GROUP_SIZE` | `maxComputeWorkGroupSize[0]`  | Thread pool size         |
| `cache_line_bytes`   | `int` | `CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE` | Assumed 64           | Platform-specific        |
| `global_mem_bytes`   | `int` | `CL_DEVICE_GLOBAL_MEM_SIZE` | `VkPhysicalDeviceMemoryProperties` | `sysconf(_SC_PHYS_PAGES)` etc. |

---

## Upstream Constraints

**ADR-003:** `StabilizationPolicy.plan_uniform_reduction_tree()` requires a `hardware_max_fan_in` parameter — the maximum fan-in the device can safely execute in a single aggregation dispatch. This is derived from `max_work_group_size` (GPU backends) or a CPU-specific limit (L1 cache capacity / partial element size). The `HardwareProfile` must either supply this directly or carry sufficient data for `StabilizationPolicy` to derive it. The derivation is Policy-tier logic and should remain in the shared layer. Additionally, the backend rendering contract requires raw hardware data (local memory size, SIMD width) from which each backend derives its own register/local crossover threshold — an Orchestration-tier concern the profile need not prescribe.

**ADR-004:** `StreamingLoopPlan`'s `IterationDimension.chunk_count` for Phase III is determined by the host memory assessment, which consumes `global_mem_bytes`. `ScratchBufferSpec` entries declare per-iteration scratch buffer sizes derived from `MemoryLayout` shapes, which depend on padding computed from `simd_width` and `c_tile_size`. The `HardwareProfile` is a transitive input to streaming loop plan construction.

**ADR-005:** Node 16's `policy_max_k` scalar parameter is synthesized by `StabilizationPolicy` from user intent, hardware limits (from `HardwareProfile.max_work_group_size`), and mathematical safety constraints. The profile must carry sufficient data for this synthesis.

---

## Tensions

- The CPU backend's SIMD width is a compile-time constant. The `HardwareProfile` must accept pre-determined values without requiring a "discovery" phase.
- `max_local_mem` is meaningless for the CPU backend. The profile must tolerate `None` or sentinel values for inapplicable constants.
- The `hardware_max_fan_in` derivation is Policy-tier logic (it feeds directly into the tree plan). It should live in `StabilizationPolicy`, not in any backend, even though its inputs come from backend-supplied hardware data.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure; shared layer produces backend-neutral data
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `hardware_max_fan_in` requirement; kernel-tier crossover as Orchestration-tier concern
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `global_mem_bytes` for chunk count; padding inputs for scratch buffer sizing
- [ADR-005: Node 16 Opacity](ADR-005-node-16-opacity-in-the-plan.md) — `policy_max_k` synthesis from hardware profile data
