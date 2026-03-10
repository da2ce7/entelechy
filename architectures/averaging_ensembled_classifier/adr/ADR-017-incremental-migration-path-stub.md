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
- Extract `HardwareProfile` frozen dataclass from current OpenCL device queries (ADR-006).
- Replace `PrecisionContext` ABC hierarchy with `PrecisionConfig` frozen dataclass (ADR-008): dissolve `Float32Context`/`Float16Context`/`Float32ModelSpec`/`Float16ModelSpec` etc.; `ModelSpec` composes `PrecisionConfig` as a field; `make_precision_config()` factory provides safe construction.
- Extract `KernelContract` from existing `KernelSignature` classes (ADR-007, shared half).
- All existing tests pass unchanged — the OpenCL backend still uses its current code paths, with a thin adapter mapping the new `PrecisionConfig` to the legacy `-D SCALAR_TYPE=...` build flags.

### Phase 1: Plan Model (New shared layer, unused)
- Implement plan node types (ADR-002): `KernelDispatchNode`, `ReductionTreeNode`, `StreamingLoopNode`, `BarrierNode`, `RetrievalNode` (with `logical_shape` field per ADR-010).
- Implement `RetrievalFuture` Protocol in the shared layer (ADR-010) — `shared/retrieval_future.py`.
- Implement plan builder: `HardwareProfile` + `ModelSpec` (composing `PrecisionConfig`) → immutable plan DAG.
- Implement `BufferHandle` plan-level tokens and `BufferDescriptor` lifecycle annotations (ADR-009).
- Add Tier 1 shared-layer tests for plan construction, contract validation, buffer lifecycle invariants (ADR-016 targets 19–26), and `RetrievalNode` logical-shape derivation.
- The OpenCL backend is **not yet wired** to the plan model — it still uses the old code path.

### Phase 2: OpenCL Renderer (Dual code path)
- Implement OpenCL `PlanRenderer`: consumes plan DAG, produces dispatch sequences. `render()` returns `Dict[str, RetrievalFuture]`.
- Implement `_OpenCLRetrievalFuture` (ADR-010) — absorbs `HostView`'s functionality: pre-allocated numpy host buffer, `cl.enqueue_copy`, `cl.Event` completion, and padding-stripping via numpy slice in `.result()`. The `.release()` method signals that the host has consumed the data and the renderer may reclaim the host-side allocation and underlying `BATCH_OUTPUT` device buffer.
- Implement OpenCL `KernelBinding`s (ADR-007, backend half) — including the OpenCL-specific `numpy_dtype` → `{SCALAR_TYPE, SCALAR_IS_HALF}` mapping (ADR-008's backend type-mapping contract).
- Implement reduction tree rendering (ADR-003) and streaming loop rendering (ADR-004).
- Wire the new renderer alongside the old code path, selectable by flag.
- Add Tier 2 per-backend tests for OpenCL renderer, including `RetrievalFuture` Protocol conformance (ADR-016 targets 27–34).
- **Validation gate:** New renderer produces bit-identical results to old code path for all existing test cases.

### Phase 3: CPU Backend (Second backend)
- Implement CPU `PlanRenderer` with C kernel implementations (ADR-013, ADR-015).
- Implement `_CPURetrievalFuture` (ADR-010) — zero-copy, zero-wait: `.wait()` is a no-op; `.result()` returns a numpy view over the compute buffer sliced to `logical_shape`; `.release()` drops the reference.
- Implement CPU precision type-mapping (`numpy_dtype` → `float`/`_Float16` or emulated) and `UnsupportedPrecisionError` for hardware without FP16 SIMD (ADR-008).
- Build system integration for CPU backend (ADR-014).
- Add Tier 2 tests for CPU backend, including CPU-specific zero-copy verification (ADR-016 target 33).
- Add Tier 3 cross-backend tests: OpenCL vs. CPU within tolerance (ADR-016).
- **Validation gate:** Cross-backend oracle tests pass.

### Phase 4: Old Code Path Removal
- Remove the old (non-plan-based) OpenCL code path.
- Complete module factoring (ADR-012): physical directory split into `shared/` and `backends/`.
- Dissolve remaining Services layer code — `arch_primitives.py` (old `PrecisionContext` host), `cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py` (including `HostView`, now fully replaced by `_OpenCLRetrievalFuture`).
- All tests run through the plan-based renderer.

### Phase 5: Vulkan Backend (Third backend)
- Implement Vulkan `PlanRenderer`.
- Implement `_VulkanRetrievalFuture` (ADR-010) — wraps `VkFence` + pre-allocated staging buffer. `.wait()` calls `vkWaitForFences`. `.result()` constructs a numpy array from the mapped staging pointer and slices to `logical_shape`. `.release()` marks the staging buffer as reclaimable.
- GLSL compute shaders + SPIR-V compilation (ADR-013, ADR-014).
- Implement Vulkan precision type-mapping (`numpy_dtype` → GLSL `float`/`float16_t`, specialization constants).
- Extend Tier 2 and Tier 3 tests for Vulkan.
- **Validation gate:** Three-way cross-backend oracle tests pass.

---

## Risk Assessment

| Risk | Mitigation |
| :--- | :---------- |
| Phase 2 dual code path divergence | Bit-identical validation gate; old path removed in Phase 4 |
| CPU floating-point divergence (no local memory, different rounding) | Per-precision tolerance model in test strategy (ADR-016) |
| CPU FP16 unsupported on target hardware | `UnsupportedPrecisionError` at discovery time (ADR-008); tests skip gracefully |
| Vulkan SDK availability in CI | Tier 3 tests optional; Tier 2 uses mock or software Vulkan (SwiftShader) |
| Migration stalls mid-phase | Each phase is independently valuable; Phase 2 alone improves testability |
| `release()` lifecycle bugs (memory leak or use-after-free) | Tier 2 tests include Protocol conformance and lifecycle verification (ADR-016 targets 27–34); defensive `__del__` fallback considered |

---

## Tensions

- The phases are ordered by the ADR dependency graph, but some work within a phase can be parallelized (e.g., Phase 0's three extractions are independent).
- Phase 2's "dual code path" period must be kept short to avoid maintenance burden. The validation gate provides a clear criterion for Phase 4.
- Phase 5 (Vulkan) may be deferred indefinitely if the CPU backend satisfies the multi-backend validation goal. The architecture must not assume Vulkan will be implemented.
- ADR-010's `release()` lifecycle introduces a new correctness concern per phase: each backend's `RetrievalFuture` must correctly retain `BATCH_OUTPUT` memory until `release()`. This must be verified in each phase's Tier 2 tests before proceeding.

---

## References

- All preceding ADRs — this ADR synthesizes the migration order from the full dependency graph
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — foundational constraint for all phases
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` frozen dataclass; backend type-mapping contract; `UnsupportedPrecisionError`
- [ADR-010: D2H Transfer & Phase Sync Points](ADR-010-d2h-transfer-and-phase-sync-points.md) — `RetrievalFuture` Protocol; `HostView` dissolution; per-backend implementation; `release()` lifecycle
