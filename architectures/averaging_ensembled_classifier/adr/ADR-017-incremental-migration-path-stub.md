# ADR-017: Incremental Migration Path

**Status:** STUB (NARROWED — backend abstraction boundary resolved by ADR-001; kernel source locations resolved by ADR-013; build system integration resolved by ADR-014; CPU FFI mechanism resolved by ADR-015; phase ordering, rollback gates, and feature-flag strategy remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-016  
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

ADR-016 (STUB) will resolve the test strategy. Until ADR-016 is decided, the rollback gate criteria for each migration phase cannot be fully specified — rollback gates depend on which test tiers pass and what constitutes a phase-completion signal.

---

## Phase Structure (tentative)

The migration is organized into six phases. Each phase introduces a specific capability and has a rollback gate (to be formalized after ADR-016).

| Phase | Deliverable | ADR constraints | Open decisions |
| :--- | :--- | :--- | :--- |
| **1** | Host-side plan model (`ExecutionPlan`, `MemoryLayout`, `KernelContract`) | ADR-001, ADR-007, ADR-009, ADR-012 | Rollback gate criteria (ADR-016) |
| **2** | PyOpenCL backend adapter — existing kernels wrapped in plan-model dispatch | ADR-001, ADR-013 | Degree of refactoring vs. thin wrapper |
| **3** | CPU backend — native C library + ctypes FFI | ADR-013, ADR-014, ADR-015 | Rollback gate criteria (ADR-016) |
| **4** | Test harness — Tier 1/2/3 test suites operational | ADR-016 | **Blocked on ADR-016** |
| **5** | Vulkan backend — GLSL compute shaders + vulkan-python dispatch | ADR-001, ADR-013, ADR-014 | Vulkan descriptor set strategy; rollback gate criteria |
| **6** | Legacy PyOpenCL removal — unified dispatch through plan model only | All | Cross-backend parity threshold for legacy removal |

### Phase 3 detail (informed by ADR-015)

Phase 3 is the first phase that introduces compiled native code and an FFI boundary. ADR-015's decisions define the concrete deliverables:

- **`libcpu_backend.so`** compiled via Meson `shared_library()` (ADR-014)
- **`_ffi_types.py`** — ctypes `Structure` subclasses for each kernel's argument struct, generated or hand-maintained to mirror the C headers
- **Layout verification** — `_verify_layouts()` callable invoked at `cdll.LoadLibrary` time; asserts Python-side struct `sizeof` matches C-side `get_struct_size_*` return values
- **Dispatch table** — `dict[str, tuple[ctypes.CFUNCTYPE, type[ctypes.Structure]]]` mapping kernel names to their C entry points and argument types
- **Thread pool** — `pool_create` / `pool_destroy` lifecycle managed by the CPU backend adapter; `pool_dispatch_and_wait` as the single blocking dispatch primitive

Phase 3 completion gate requires, at minimum, that layout verification passes and a representative subset of kernels produce numerically correct results when dispatched through the ctypes FFI. The formal gate criteria depend on ADR-016's test tier definitions.

---

## Decision Required

### Phase sequencing strategy

- **(A) Strict sequential.** Each phase must fully complete and pass its rollback gate before the next begins. Simplest to reason about; slowest to deliver. No partial overlap.

- **(B) Overlapping phases with dependency ordering.** Phases may overlap where their deliverables are independent. E.g., Phase 3 (CPU backend) and Phase 5 (Vulkan backend) could proceed in parallel once Phase 2 is stable. Phase 4 (test harness) can begin as soon as ADR-016 is decided, potentially overlapping with Phase 3.

- **(C) Feature-flag gated.** All phases proceed in parallel behind `_build_config.py` feature flags (ADR-014). Each backend is independently toggleable. Integration testing gates promotion of each feature flag from `auto` to `enabled`. **Favored direction** — ADR-014's `auto`/`enabled`/`disabled` feature options already provide the gating mechanism; this option leverages existing infrastructure.

### Rollback gate formalization

- **(D) Tier-based gates.** Each phase's rollback gate is defined in terms of ADR-016's test tiers: Phase 1 requires Tier 1 green; Phase 3 requires Tier 1 + CPU Tier 2 green; Phase 5 requires Tier 1 + Vulkan Tier 2 green; Phase 6 requires Tier 3 parity green across all enabled backends. **Blocked on ADR-016.**

- **(E) Metric-based gates.** In addition to tier-based gates, define quantitative thresholds (e.g., maximum allowable precision deviation per kernel, performance regression bounds). **Blocked on ADR-016 and ADR-008 tolerance tables.**

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| Phase 3 FFI struct drift during development | Medium | High — silent corruption | Layout verification (ADR-015) catches size-level drift at load time; Tier 2 catches field reorderings behaviorally |
| Phase 5 Vulkan descriptor set complexity delays | Medium | Medium — Vulkan backend delayed | Phase 5 independent of Phases 3/4 under Option B/C |
| ADR-016 delayed — rollback gates undefined | Low | High — phases proceed without formal gates | Informal gates (manual numerical spot-checks) until ADR-016 is decided |
| Legacy PyOpenCL removal (Phase 6) reveals undocumented behavior | Medium | High — correctness regression | Tier 3 parity tests must cover full kernel inventory before Phase 6 |

---

## Tensions

- Option A (strict sequential) provides the strongest correctness guarantees but is incompatible with parallel development across backends. Option C is operationally efficient but requires robust per-backend isolation — ADR-014's feature flags provide this.
- Phase 4 (test harness) is gated on ADR-016, but Phase 3 needs at minimum informal test coverage to validate the FFI layer. ADR-015's layout verification provides a mechanical pre-test gate, reducing the risk of entering Phase 3 without formal Tier 2 tests.
- Phase 6 (legacy removal) is the highest-risk phase. The cross-backend parity threshold depends on ADR-008 precision tolerances and ADR-016's Tier 3 definitions — both must be fully resolved before Phase 6 can be gated.
- The `auto` default for `backend_cpu` and `backend_vulkan` (ADR-014) means CI machines may non-deterministically gain or lose backends if their toolchains change. This interacts with rollback gate definitions — a phase that was "green" may become "yellow" if a backend disappears from the build manifest.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — uniform dispatch interface behind the plan model
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — host-side validation distinct from per-backend compilation
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — plan-controlled buffer allocations, transfers, releases
- [ADR-012: Module Factoring and Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — stateless plan primitives replacing service objects
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — shared spec in `kernels.cl.h`; per-backend source directories; cross-backend fidelity model
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — Meson with `meson-python`; `_build_config.py` manifest; `feature` options with `auto` default; conditional `subdir()` delegation
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop.md) — ctypes FFI for CPU backend; struct layout verification; `_ffi_types.py`; dispatch table; thread pool lifecycle
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — (STUB) test tier definitions; rollback gate criteria dependency
