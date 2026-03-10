# ADR-017: Incremental Migration Path

**Status:** STUB (NARROWED — kernel source locations, directory layout, and per-phase source creation sequence resolved by ADR-013; phase ordering, rollback gates, and feature-flag strategy remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-014, ADR-015, ADR-016  
**Blocks:** —

---

## Context

The multi-backend refactoring replaces a monolithic OpenCL-only implementation with the layered architecture described by ADRs 001–013. This ADR defines the incremental migration sequence: the order in which components are extracted, backends are introduced, and the legacy code path is retired.

ADR-013 (ACCEPTED) resolves kernel source locations and provides concrete per-phase migration implications (ADR-013 §Consequences → Migration implications), reproduced here as constraints:

- **Phase 0 (Foundation):** `kernels/` stays in place, unchanged. `kernels.cl.h` is formally designated as the algorithmic specification. No file moves.
- **Phase 2 (OpenCL Renderer):** The OpenCL backend loads sources from architecture-root `kernels/` directly. ADR-014 provides the path as a build constant.
- **Phase 3 (CPU Backend):** `src/backends/cpu/kernel_sources/` is created with C implementations developed against `kernels.cl.h` as algorithmic reference. `cpu_simd.h` and `cpu_kernels.h` written per CPU_BACKEND.md. Tier 2 + Tier 3 tests (ADR-016) validate.
- **Phase 4 (Vulkan Backend):** `src/backends/vulkan/kernel_sources/` is created with GLSL compute shaders. `common.glsl` provides shared declarations. Meson compiles `*.comp` → `*.spv`. Tier 3 parity tests validate against OpenCL + CPU.

ADR-014 (pending) determines the Meson build system structure that gates backend enablement. ADR-015 (pending) determines how the CPU shared library is loaded at runtime. ADR-016 (pending) determines the test tiers executed at each phase gate.

---

## Decision Required

### Phase structure

Six phases are envisioned. ADR-013 constrains Phases 0, 2, 3, and 4. The remaining open decisions are:

| Phase | ADR-013 constrained? | Open decisions |
| :--- | :--- | :--- |
| **0 — Foundation** | Yes (no file moves, `kernels.cl.h` designation) | Rollback strategy; feature flag for legacy/new code paths |
| **1 — Shared layer extraction** | No | Extraction order for `src/shared/` modules; backward-compatibility shims |
| **2 — OpenCL Renderer** | Yes (direct `kernels/` reference) | `PlanRenderer` interface freeze gate; legacy path deprecation timeline |
| **3 — CPU Backend** | Yes (source layout, `cpu_kernels.h`, Tier 2+3 tests) | CPU Tier 2 fixture generation; CI hardware requirements |
| **4 — Vulkan Backend** | Yes (source layout, SPIR-V compilation, Tier 3) | Vulkan SDK version pinning; CI GPU requirements |
| **5 — Legacy retirement** | No | Cutover criteria; deprecation warnings; removal timeline |

### Options

- **(A) Strict sequential gating.** Each phase must pass a defined acceptance gate (Tier 1–3 tests from ADR-016) before the next phase begins. Slower but lower risk.

- **(B) Overlapping phases.** Phases 3 and 4 (CPU + Vulkan) proceed in parallel once Phase 2 is stable. Faster but requires careful coordination of shared-layer changes.

- **(C) Feature-flag coexistence.** Legacy and new code paths coexist behind runtime feature flags throughout all phases. Enables gradual rollout per-user/per-environment but increases code maintenance burden.

Options A and C are combinable (sequential phases with feature-flag coexistence within each phase).

---

## Risk Assessment

- **Partial migration stall.** If Phase 2 (OpenCL Renderer) destabilizes the existing training loop, the legacy path must remain available. Feature flags (Option C) mitigate this.
- **Cross-phase specification drift.** A kernel algorithm change during Phase 3 or 4 must propagate to `kernels.cl.h` first (CONCEPT.md §1 Architectural Elegance Feedback), then to all in-progress backend implementations.
- **Build system bootstrapping.** ADR-014 must be resolved before Phase 2 can begin — the OpenCL backend needs the `kernels/` path as a build constant.
- **Test infrastructure dependency.** ADR-016's Tier 3 tests must be operational before Phase 3 can be accepted — CPU correctness is validated by parity against OpenCL.

---

## Tensions

- Phase 0 is largely formalization (designating `kernels.cl.h`, no file moves per ADR-013), but it establishes the foundational invariants that later phases depend on. Rushing Phase 0 risks under-specifying the `kernels/` directory's dual role.
- Phases 3 and 4 have independent source trees (ADR-013: `src/backends/cpu/kernel_sources/` and `src/backends/vulkan/kernel_sources/`) and could theoretically proceed in parallel (Option B), but Tier 3 tests for Vulkan would benefit from CPU as oracle (ADR-016 Option C), creating a soft dependency.
- The OpenCL backend's lack of a `kernel_sources/` subdirectory (ADR-013 — it references `kernels/` directly) simplifies Phase 2 but means Phase 2 and Phase 0 are tightly coupled — any reorganization of `kernels/` during Phase 0 immediately affects the OpenCL backend.

---

## References

- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` + `src/backends/<name>/` directory structure defining the extraction target
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — per-phase migration implications (§Consequences); `kernels/` dual role; per-backend `kernel_sources/` locations
- [ADR-014: Build System Integration](ADR-014-build-system-integration-stub.md) — Meson build targets, conditional backend enablement, `kernels/` path configuration
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop-stub.md) — CPU shared library loading mechanism required by Phase 3
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — Tier 1/2/3 test structure; phase acceptance gates; CPU as oracle candidate
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback (formalize first, implement second)
- [CPU_BACKEND.md](../CPU_BACKEND.md) — CPU kernel source specifications referenced by Phase 3
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — Vulkan shader specifications referenced by Phase 4
