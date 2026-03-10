# ADR-017: Incremental Migration Path

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** All preceding ADRs  
**Blocks:** —

---

## Context

The refactoring must proceed incrementally while keeping the existing OpenCL backend operational at every intermediate commit. This ADR defines the phased migration strategy.

---

## Decision Required

A 6-phase plan is proposed. The phases are ordered by the ADR dependency graph — each phase resolves one or more ADRs and produces a testable intermediate state.

### Phase 0: Foundation (No behavioral change)
- Extract `HardwareProfile` from current OpenCL device queries (ADR-006).
- Extract `PrecisionContext` shared fields (ADR-008).
- Extract `KernelContract` from existing `KernelSignature` classes (ADR-007, shared half).
- All existing tests pass unchanged — the OpenCL backend still uses its current code paths.

### Phase 1: Plan Model (New shared layer, unused)
- Implement plan node types (ADR-002): `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode`, `BarrierNode`, `RetrievalNode`.
- Implement plan builder: `HardwareProfile` + `ModelSpec` + `PrecisionContext` → immutable plan DAG.
- Implement `BufferHandle` plan-level tokens (ADR-009).
- Add Tier 1 shared-layer tests for plan construction and contract validation.
- The OpenCL backend is **not yet wired** to the plan model — it still uses the old code path.

### Phase 2: OpenCL Renderer (Dual code path)
- Implement OpenCL `PlanRenderer`: consumes plan DAG, produces dispatch sequences.
- Implement OpenCL `KernelBinding`s (ADR-007, backend half).
- Implement reduction tree rendering (ADR-003) and streaming loop rendering (ADR-004).
- Wire the new renderer alongside the old code path, selectable by flag.
- Add Tier 2 per-backend tests for OpenCL renderer.
- **Validation gate:** New renderer produces bit-identical results to old code path for all existing test cases.

### Phase 3: CPU Backend (Second backend)
- Implement CPU `PlanRenderer` with C kernel implementations (ADR-013, ADR-015).
- Build system integration for CPU backend (ADR-014).
- Add Tier 2 tests for CPU backend.
- Add Tier 3 cross-backend tests: OpenCL vs. CPU within tolerance (ADR-016).
- **Validation gate:** Cross-backend oracle tests pass.

### Phase 4: Old Code Path Removal
- Remove the old (non-plan-based) OpenCL code path.
- Complete module factoring (ADR-012): physical directory split into `shared/` and `backends/`.
- Dissolve remaining Services layer code.
- All tests run through the plan-based renderer.

### Phase 5: Vulkan Backend (Third backend)
- Implement Vulkan `PlanRenderer`.
- GLSL compute shaders + SPIR-V compilation (ADR-013, ADR-014).
- Extend Tier 2 and Tier 3 tests for Vulkan.
- **Validation gate:** Three-way cross-backend oracle tests pass.

---

## Risk Assessment

| Risk | Mitigation |
| :--- | :---------- |
| Phase 2 dual code path divergence | Bit-identical validation gate; old path removed in Phase 4 |
| CPU floating-point divergence (no local memory, different rounding) | Per-precision tolerance model in test strategy (ADR-016) |
| Vulkan SDK availability in CI | Tier 3 tests optional; Tier 2 uses mock or software Vulkan (SwiftShader) |
| Migration stalls mid-phase | Each phase is independently valuable; Phase 2 alone improves testability |

---

## Tensions

- The phases are ordered by the ADR dependency graph, but some work within a phase can be parallelized (e.g., Phase 0's three extractions are independent).
- Phase 2's "dual code path" period must be kept short to avoid maintenance burden. The validation gate provides a clear criterion for Phase 4.
- Phase 5 (Vulkan) may be deferred indefinitely if the CPU backend satisfies the multi-backend validation goal. The architecture must not assume Vulkan will be implemented.

---

## References

- All preceding ADRs — this ADR synthesizes the migration order from the full dependency graph
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — foundational constraint for all phases
