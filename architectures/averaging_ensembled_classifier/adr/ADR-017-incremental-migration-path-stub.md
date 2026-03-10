# ADR-017: Incremental Migration Path

**Status:** STUB (NARROWED — backend abstraction boundary resolved by ADR-001; kernel source locations resolved by ADR-013; build system integration resolved by ADR-014; CPU FFI mechanism resolved by ADR-015; test strategy resolved by ADR-016 with tier-based rollback gates now concrete; phase ordering and feature-flag strategy remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
**Blocks:** —

---

## Context

The averaging ensembled classifier architecture is being migrated from a monolithic Python+PyOpenCL implementation to a multi-backend, formally-specified system. The migration must be incremental — existing PyOpenCL functionality must remain operational at each step, with new infrastructure (memory layout, execution plan, CPU backend, Vulkan backend) layered in behind feature gates.

The resolved ADRs establish the technical constraints for each migration phase:

| ADR | Key constraint on migration |
| :--- | :--- |
| ADR-001 | Backend abstraction boundary — backends expose a uniform dispatch interface behind the plan model |
| ADR-007 | `KernelContract`/`KernelBinding` split — host-side validation distinct from per-backend compilation |
| ADR-009 | `BufferContract` lifecycle — all buffer allocations, transfers, and releases are plan-controlled |
| ADR-012 | Module factoring — services dissolution into stateless plan primitives |
| ADR-013 | Kernel source strategy — `kernels.cl.h` as shared spec; per-backend source directories under `src/backends/` |
| ADR-014 | Build system — Meson with `meson-python`; `_build_config.py` manifest; `feature` options with `auto` default; conditional `subdir()` delegation |
| ADR-015 | CPU FFI — ctypes with struct layout verification; `_ffi_types.py` for struct definitions; dispatch table mapping `kernel_name` → `(task_fn_ptr, args_struct_class)`; `_verify_layouts()` at library load |
| ADR-016 | Test strategy — layered pytest framework (Tier 1/2/3); CPU oracle for Tier 3 parity; analytical + numpy fixtures for Tier 2; per-kernel tolerance tables; `_build_config.py`-driven skip logic |

ADR-016 (ACCEPTED) resolves the test strategy and provides concrete rollback gate definitions for each migration phase. The tier structure is:

- **Tier 1** — host-side plan correctness; always runs; no backend required.
- **Tier 2** — per-backend kernel correctness against analytical/numpy fixtures; runs per enabled backend.
- **Tier 3** — cross-backend parity with CPU as reference oracle; runs when ≥ 2 backends are available.

---

## Phase Structure (tentative)

The migration is organized into six phases. Each phase introduces a specific capability and has a rollback gate defined by ADR-016's tier outcomes.

| Phase | Deliverable | ADR constraints | Rollback gate (ADR-016) |
| :--- | :--- | :--- | :--- |
| **1** | Host-side plan model (`ExecutionPlan`, `MemoryLayout`, `KernelContract`) | ADR-001, ADR-007, ADR-009, ADR-012 | Tier 1 green |
| **2** | PyOpenCL backend adapter — existing kernels wrapped in plan-model dispatch | ADR-001, ADR-013 | Tier 1 green + OpenCL Tier 2 green |
| **3** | CPU backend — native C library + ctypes FFI | ADR-013, ADR-014, ADR-015 | Tier 1 green + CPU Tier 2 green |
| **4** | Test harness — Tier 1/2/3 test suites operational | ADR-016 | Full Tier 1/2/3 framework operational; all enabled tiers green |
| **5** | Vulkan backend — GLSL compute shaders + vulkan-python dispatch | ADR-001, ADR-013, ADR-014 | Tier 1 green + Vulkan Tier 2 green + Tier 3 parity (Vulkan-vs-oracle) green |
| **6** | Legacy PyOpenCL removal — unified dispatch through plan model only | All | Tier 3 parity green across all enabled backends for the full kernel inventory at both FP32 and supported FP16 configurations |

### Phase 3 detail (informed by ADR-015 and ADR-016)

Phase 3 is the first phase that introduces compiled native code and an FFI boundary. ADR-015's decisions define the concrete deliverables:

- **`libcpu_backend.so`** compiled via Meson `shared_library()` (ADR-014)
- **`_ffi_types.py`** — ctypes `Structure` subclasses for each kernel's argument struct, generated or hand-maintained to mirror the C headers
- **Layout verification** — `_verify_layouts()` callable invoked at `cdll.LoadLibrary` time; asserts Python-side struct `sizeof` matches C-side `get_struct_size_*` return values
- **Dispatch table** — `dict[str, tuple[ctypes.CFUNCTYPE, type[ctypes.Structure]]]` mapping kernel names to their C entry points and argument types
- **Thread pool** — `pool_create` / `pool_destroy` lifecycle managed by the CPU backend adapter; `pool_dispatch_and_wait` as the single blocking dispatch primitive

Phase 3 completion gate (per ADR-016): layout verification passes and all 22 CPU Tier 2 kernel tests produce numerically correct results against analytical/numpy fixtures within the per-kernel tolerance table.

---

## Decision Required

### Phase sequencing strategy

- **(A) Strict sequential.** Each phase must fully complete and pass its rollback gate before the next begins. Simplest to reason about; slowest to deliver. No partial overlap.

- **(B) Overlapping phases with dependency ordering.** Phases may overlap where their deliverables are independent. E.g., Phase 3 (CPU backend) and Phase 5 (Vulkan backend) could proceed in parallel once Phase 2 is stable. Phase 4 (test harness) can begin immediately — ADR-016 is decided — potentially overlapping with Phase 3.

- **(C) Feature-flag gated.** All phases proceed in parallel behind `_build_config.py` feature flags (ADR-014). Each backend is independently toggleable. Integration testing gates promotion of each feature flag from `auto` to `enabled`. **Favored direction** — ADR-014's `auto`/`enabled`/`disabled` feature options already provide the gating mechanism; this option leverages existing infrastructure.

### Rollback gate formalization

- **(D) Tier-based gates.** Each phase's rollback gate is defined in terms of ADR-016's test tiers, as specified in the phase table above. Phase 1 requires Tier 1 green; Phase 3 requires Tier 1 + CPU Tier 2 green; Phase 5 requires Tier 1 + Vulkan Tier 2 + Tier 3 parity green; Phase 6 requires Tier 3 parity green across all enabled backends. **ADR-016's tier definitions and per-kernel tolerance tables provide the concrete criteria.**

- **(E) Metric-based gates.** In addition to tier-based gates, define quantitative thresholds (e.g., maximum allowable precision deviation per kernel beyond ADR-016's tolerance table, performance regression bounds). ADR-016's tolerance tables provide the baseline; this option extends them with performance and resource-consumption metrics.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Phase 3 FFI struct drift during development | Medium | High — silent corruption | Layout verification (ADR-015) catches size-level drift at load time; CPU Tier 2 (ADR-016) catches field reorderings behaviorally |
| Phase 5 Vulkan descriptor set complexity delays | Medium | Medium — Vulkan backend delayed | Phase 5 independent of Phases 3/4 under Option B/C |
| Legacy PyOpenCL removal (Phase 6) reveals undocumented behavior | Medium | High — correctness regression | ADR-016 Tier 3 parity tests must cover full kernel inventory before Phase 6 |
| CI backend availability non-determinism | Low | Medium — gate flicker | ADR-014's `auto` default + ADR-016's `_build_config`-driven skip logic degrade gracefully; pin `enabled` in critical CI for stability |

---

## Tensions

- Option A (strict sequential) provides the strongest correctness guarantees but is incompatible with parallel development across backends. Option C is operationally efficient but requires robust per-backend isolation — ADR-014's feature flags provide this.
- Phase 4 (test harness) is no longer blocked — ADR-016 is decided. The harness can be built in parallel with Phases 2/3 and serve as a validation tool during their development.
- Phase 6 (legacy removal) is the highest-risk phase. The cross-backend parity threshold depends on ADR-008 precision tolerances and ADR-016's Tier 3 definitions — both are now resolved. The formal gate is: Tier 3 parity green across all enabled backends for the full 22-kernel inventory.
- The `auto` default for `backend_cpu` and `backend_vulkan` (ADR-014) means CI machines may non-deterministically gain or lose backends if their toolchains change. ADR-016's skip logic handles this gracefully (tiers are skipped, not failed, when a backend is absent), but rollback gate definitions must distinguish "skipped tier" from "failed tier."

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — uniform dispatch interface behind the plan model
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — host-side validation distinct from per-backend compilation
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` precision-derived tolerances consumed by ADR-016's per-kernel tolerance tables
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — plan-controlled buffer allocations, transfers, releases
- [ADR-012: Module Factoring and Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — stateless plan primitives replacing service objects
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — shared spec in `kernels.cl.h`; per-backend source directories; cross-backend fidelity model
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — Meson with `meson-python`; `_build_config.py` manifest; `feature` options with `auto` default; conditional `subdir()` delegation
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop.md) — ctypes FFI for CPU backend; struct layout verification; `_ffi_types.py`; dispatch table; thread pool lifecycle
- [ADR-016: Test Strategy](ADR-016-test-strategy.md) — layered pytest framework (Tier 1/2/3); CPU oracle model; analytical + numpy fixtures; per-kernel tolerance tables; rollback gate definitions per migration phase
