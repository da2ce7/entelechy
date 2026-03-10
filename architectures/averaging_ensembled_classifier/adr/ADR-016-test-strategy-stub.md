# ADR-016: Test Strategy

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-013, ADR-014  
**Blocks:** ADR-017

---

## Context

The current test suite validates a single OpenCL backend. The multi-backend refactoring requires a test strategy that verifies three backends (OpenCL, Vulkan, CPU) plus the shared Policy tier.

ADR-012 (ACCEPTED) establishes the `shared/` vs. `backends/<name>/` boundary with a mechanically enforceable import invariant: no module in `shared/` may import from any `backends/` sub-package. This boundary creates a natural three-tier test structure:

- **Tier 1 — Shared (pure Python).** Unit tests for `shared/` modules: `PlanBuilder` graph construction, `BufferLayout` calculations, `StabilizationPolicy` threshold logic, `KernelContract` validation, precision configuration. These require no GPU hardware and import only from `shared/`.
- **Tier 2 — Per-backend.** Integration tests for each `backends/<name>/` backend: buffer allocation, kernel dispatch, `PlanRenderer` execution of known plan graphs, D2H transfer fidelity. These require backend-specific hardware or emulation.
- **Tier 3 — Cross-backend parity.** Oracle-based fidelity tests: execute the same `ExecutionPlan` on two or more backends and compare outputs within configured precision tolerances (ADR-008). Validates that all backends implement the same algorithmic semantics.

ADR-012's import boundary makes Tier 1 tests trivially isolatable — they run against `shared/` with no hardware dependencies. ADR-013 (pending) will determine kernel source organization, which affects how Tier 2 tests locate and load kernel implementations. ADR-014 (pending) will determine build system integration, which affects how Tier 2 tests for Vulkan (SPIR-V) and CPU (shared library) are provisioned.

---

## Decision Required

- **(A) Layered strategy (Tier 1 / 2 / 3).** Tests are organized into three tiers as described above, with Tier 1 as the mandatory CI baseline, Tier 2 conditional on hardware availability, and Tier 3 as nightly/release-gate checks. **Favored direction.**

- **(B) Parameterized-only.** A single test suite parameterized over backends. Simpler to maintain but conflates shared-tier testing with backend-specific testing, making CI hardware requirements mandatory for all tests.

- **(C) Reference oracle (single source of truth).** One backend (e.g., CPU) designated the reference implementation; other backends are validated against it. Subsumable into Option A as a Tier 3 oracle selection policy.

---

## Concrete Test Targets

The following test targets derive from accepted upstream ADRs. This inventory is preliminary; ADR-013 and ADR-014 may add build-related test targets.

### From ADR-003 (Reduction Tree)

1. Single-chunk, single-level reduction (trivial plan)
2. Multi-chunk, single-level reduction (chunk boundary handling)
3. Multi-level reduction (recursive tree)
4. Per-classifier output fidelity (Ensemble invariant: final output = mean of classifier outputs)
5. Edge case: `n_classifiers = 1` (degenerate ensemble)

### From ADR-004 (Streaming Loop)

6. Single-batch streaming execution
7. Multi-batch streaming with buffer reuse across iterations
8. Streaming plan graph structure matches expected DAG shape
9. Early-exit / partial-epoch termination (if supported)

### From ADR-005 (Node 16 Opacity)

10. Node 16 output buffer correctness under opacity-via-reduction
11. Node 16 plan representation matches expected structure
12. Buffer lifecycle for Node 16 intermediate results

### From ADR-009 (Buffer Lifecycle)

13. `BufferLayout` slot offset calculations (deterministic, no overlap)
14. Buffer aliasing correctness (non-overlapping lifetimes verified)
15. Peak memory calculation matches sum of max-concurrent allocations
16. `BackendMemoryInfo` round-trip (allocate → populate → read-back)
17. Layout stability under plan mutation (idempotent re-layout)

### From ADR-010 (D2H Transfer & Phase Sync)

18. D2H transfer at phase boundaries returns correct host-side values
19. Sync-point placement in plan graph matches phase transitions
20. Transfer size matches `BufferLayout` declaration
21. Async transfer completion (if backend supports non-blocking D2H)
22. D2H under streaming: correct buffer selected per iteration

### From ADR-011 (CCE/BCE Strategy Delegation)

23. Strategy B (Nodes 6/7): CCE and BCE use distinct kernel names
24. Strategy B: CCE Node 6 produces `int*` targets for Node 7
25. Strategy B: BCE Node 6 produces `SCALAR_TYPE*` targets for Node 7
26. Strategy A (Nodes 8/9/10): unified kernel dispatched with `FLAG_problem_type = 0` (CCE) and `= 1` (BCE)
27. Strategy A: FLAG value propagated correctly through `ScalarParamSpec`
28. Mixed plan: CCE Nodes 6/7 (Strategy B) + Nodes 8/9/10 (Strategy A) in single plan
29. Per-backend rendering: OpenCL `#define FLAG`, Vulkan specialization constant, CPU function argument

### From ADR-012 (Module Factoring)

30. Import boundary invariant: no module in `shared/` imports from `backends/`
31. `shared/` modules importable without any backend installed
32. Backend discovery: `backends/<name>/` correctly registered and selectable at runtime
33. Module relocation: all public symbols accessible from new paths

### Cross-Backend Parity (Tier 3)

34. Identical `ExecutionPlan` produces numerically equivalent outputs (within ADR-008 tolerances) across all enabled backends
35. Buffer contents match at every phase sync point across backends
36. Precision configuration (ADR-008) applied identically: FP32 and FP16 modes
37. Reduction tree produces identical final ensemble output across backends
38. Streaming loop buffer reuse yields identical results across backends
39. Strategy A FLAG dispatch produces identical outputs across backends
40. Strategy B kernel selection produces identical outputs across backends
41. Node 16 opacity-via-reduction produces identical outputs across backends
42. D2H transfer values match across backends at every sync point

---

## Tensions

- Tier 2 tests for Vulkan and CPU require build artifacts (SPIR-V modules, C shared library) that depend on ADR-014's build system.
- Tier 3 cross-backend tests require at least two backends to be built and hardware-available — this may limit CI environments.
- The 42-target inventory above covers *functional* correctness; performance benchmarks and regression tests are a separate concern (potentially ADR-016 addendum or a distinct ADR).
- Option C (reference oracle) is not exclusive — it can serve as the Tier 3 truth source within Option A's layered structure.

---

## References

- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md)
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md)
- [ADR-005: Node 16 Opacity in the Plan](ADR-005-node-16-opacity-in-the-plan.md)
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — FP32/FP16 tolerance definitions
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md)
- [ADR-010: D2H Transfer & Phase Sync Points](ADR-010-d2h-transfer-and-phase-sync-points.md)
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md)
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — import boundary invariant; `shared/` vs. `backends/` structure
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy-stub.md) — kernel source file locations
- [ADR-014: Build System Integration](ADR-014-build-system-integration-stub.md) — backend build artifact provisioning
