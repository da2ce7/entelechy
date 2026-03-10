# ADR-017: Incremental Migration Path

**Status:** STUB (NARROWED — kernel source locations, directory layout, and per-phase source creation sequence resolved by ADR-013; build system structure, conditional enablement, and per-phase build evolution resolved by ADR-014; phase ordering, rollback gates, and feature-flag strategy remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-015, ADR-016  
**Blocks:** —

---

## Context

The multi-backend refactoring replaces a monolithic OpenCL-only implementation with the layered architecture described by ADRs 001–014. This ADR defines the incremental migration sequence: the order in which components are extracted, backends are introduced, and the legacy code path is retired.

ADR-013 (ACCEPTED) resolves kernel source locations and provides concrete per-phase migration implications (ADR-013 §Consequences → Migration implications):

- **Phase 0 (Foundation):** `kernels/` stays in place, unchanged. `kernels.cl.h` is formally designated as the algorithmic specification. No file moves.
- **Phase 2 (OpenCL Renderer):** The OpenCL backend loads sources from architecture-root `kernels/` directly.
- **Phase 3 (CPU Backend):** `src/backends/cpu/kernel_sources/` is created with C implementations developed against `kernels.cl.h` as algorithmic reference. `cpu_simd.h` and `cpu_kernels.h` written per CPU_BACKEND.md. Tier 2 + Tier 3 tests (ADR-016) validate.
- **Phase 4 (Vulkan Backend):** `src/backends/vulkan/kernel_sources/` is created with GLSL compute shaders. `common.glsl` provides shared declarations. Meson compiles `*.comp` → `*.spv`. Tier 3 parity tests validate against OpenCL + CPU.

ADR-014 (ACCEPTED) resolves the build system evolution per phase:

- **Phase 0:** Create `meson.options` with `backend_vulkan` and `backend_cpu` options. Create `src/_build_config.py.in` template. Refactor `meson.build` to `subdir()` delegation. Create skeleton `meson.build` files in `src/shared/`, `src/backends/opencl/`.
- **Phase 2:** OpenCL backend's `meson.build` installs kernel binding Python files. `install_data` for `kernels/` already in top-level `meson.build`. `_build_config.py` reflects `BACKEND_OPENCL = True`.
- **Phase 3:** Create `src/backends/cpu/meson.build` with `shared_library('cpu_kernels', ...)`. `_build_config.py` gains `BACKEND_CPU = True`. Tier 2 + Tier 3 tests compare CPU vs. OpenCL.
- **Phase 4:** Create `src/backends/vulkan/meson.build` with `custom_target` SPIR-V compilation. `_build_config.py` gains `BACKEND_VULKAN = True`. Tier 3 tests compare Vulkan vs. OpenCL/CPU.

ADR-014's `auto` default for backend options means each phase's build artifacts are automatically picked up when their source files and dependencies exist — no manual option toggling required during migration.

ADR-015 (pending) determines how the CPU shared library is loaded at runtime. ADR-016 (pending) determines the test tiers executed at each phase gate.

---

## Decision Required

### Phase structure

Six phases are envisioned. ADR-013 constrains Phases 0, 2, 3, and 4. ADR-014 constrains the build system evolution within each phase. The remaining open decisions are:

| Phase | Constrained by | Open decisions |
| :--- | :--- | :--- |
| **0 — Foundation** | ADR-013 (no file moves), ADR-014 (`meson.options`, `subdir()` refactor, `_build_config.py.in`) | Rollback strategy; feature flag for legacy/new code paths |
| **1 — Shared layer extraction** | ADR-012 (target structure) | Extraction order for `src/shared/` modules; backward-compatibility shims |
| **2 — OpenCL Renderer** | ADR-013 (direct `kernels/` ref), ADR-014 (kernel `install_data`, `_build_config`) | `PlanRenderer` interface freeze gate; legacy path deprecation timeline |
| **3 — CPU Backend** | ADR-013 (source layout), ADR-014 (`shared_library`, ISA flags) | CPU Tier 2 fixture generation; FFI mechanism (ADR-015); CI hardware requirements |
| **4 — Vulkan Backend** | ADR-013 (GLSL layout), ADR-014 (`custom_target` SPIR-V) | Vulkan SDK version pinning; CI GPU requirements |
| **5 — Legacy retirement** | — | Cutover criteria; deprecation warnings; removal timeline |

### Options

- **(A) Strict sequential gating.** Each phase must pass a defined acceptance gate (Tier 1–3 tests from ADR-016) before the next phase begins. Slower but lower risk.

- **(B) Overlapping phases.** Phases 3 and 4 (CPU + Vulkan) proceed in parallel once Phase 2 is stable. Faster but requires careful coordination of shared-layer changes. ADR-014's independent `subdir()` per backend supports this — the CPU and Vulkan `meson.build` files do not interact.

- **(C) Feature-flag coexistence.** Legacy and new code paths coexist behind runtime feature flags throughout all phases. Enables gradual rollout per-user/per-environment but increases code maintenance burden.

Options A and C are combinable (sequential phases with feature-flag coexistence within each phase).

---

## Risk Assessment

- **Partial migration stall.** If Phase 2 (OpenCL Renderer) destabilizes the existing training loop, the legacy path must remain available. Feature flags (Option C) mitigate this.
- **Cross-phase specification drift.** A kernel algorithm change during Phase 3 or 4 must propagate to `kernels.cl.h` first (CONCEPT.md §1 Architectural Elegance Feedback), then to all in-progress backend implementations.
- **Test infrastructure dependency.** ADR-016's Tier 3 tests must be operational before Phase 3 can be accepted — CPU correctness is validated by parity against OpenCL. ADR-014's `_build_config.py` provides the test framework with the enabled-backend manifest needed for skip/run logic.

---

## Tensions

- Phase 0 is largely formalization (designating `kernels.cl.h` per ADR-013, creating `meson.options` and `_build_config.py.in` per ADR-014), but it establishes the foundational invariants that later phases depend on. Rushing Phase 0 risks under-specifying the `kernels/` directory's dual role.
- Phases 3 and 4 have independent source trees (ADR-013) and independent `meson.build` files (ADR-014), supporting Option B (overlapping phases). However, Tier 3 tests for Vulkan would benefit from CPU as oracle (ADR-016 favored direction), creating a soft dependency of Phase 4 on Phase 3.
- The OpenCL backend's lack of a `kernel_sources/` subdirectory (ADR-013 — it references `kernels/` directly) simplifies Phase 2 but means Phase 2 and Phase 0 are tightly coupled — any reorganization of `kernels/` during Phase 0 immediately affects the OpenCL backend.

---

## References

- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` + `src/backends/<name>/` directory structure defining the extraction target
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — per-phase migration implications (§Consequences); `kernels/` dual role; per-backend `kernel_sources/` locations
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — per-phase build system evolution; `meson.options`; `subdir()` delegation; `_build_config.py` manifest; `shared_library` for CPU; `custom_target` for Vulkan SPIR-V
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop-stub.md) — CPU shared library loading mechanism required by Phase 3
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — Tier 1/2/3 test structure; phase acceptance gates; CPU as oracle candidate
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback (formalize first, implement second)
- [CPU_BACKEND.md](../CPU_BACKEND.md) — CPU kernel source specifications referenced by Phase 3
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — Vulkan shader specifications referenced by Phase 4
