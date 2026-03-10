# ADR-016: Test Strategy for Multi-Backend Correctness

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-013, ADR-014  
**Blocks:** —

---

## Context

With three backends producing numerical results for the same plan, a cross-backend correctness strategy is essential. The existing test suite validates OpenCL results against known values. The multi-backend world needs:

1. **Shared-layer tests** — plan construction, contract validation, memory layout, parameter space: backend-independent, fast.
2. **Per-backend integration tests** — each backend's renderer produces correct results for reference plans.
3. **Cross-backend oracle tests** — the same plan, run on two backends, produces results within tolerance.

---

## Decision Required

- **(A) Layered strategy (favored).** Three tiers:
  - **Tier 1 (Shared):** Pure-Python unit tests for plan builder, contracts, policies. No backend required. Fast CI.
  - **Tier 2 (Per-backend):** Each backend runs reference plans and checks results against golden values (numpy reference implementation).
  - **Tier 3 (Cross-backend):** If ≥2 backends available, run the same plan on each and compare results within `atol`/`rtol`. Optional in CI (requires multiple backends).

- **(B) Parameterized-only.** All tests are parameterized by backend. Simpler test structure but no explicit cross-backend comparison — drift detected only by golden-value divergence.

- **(C) Reference oracle.** A pure-numpy reference implementation serves as the single source of truth. All backends are compared against it. More expensive to maintain but eliminates cross-backend comparison.

---

## Concrete Test Targets from Decided ADRs

The following test categories are derived from decided ADRs:

### From ADR-003 (Reduction Tree):
1. Tree depth computation from `num_elements` and `fan_in`
2. Elements-per-partial derivation from `policy_max_k` and padding
3. Threshold schedule correctness (monotonically non-increasing, terminal threshold ≤ `fp_max`)
4. Offset-list boundary correctness for each stage
5. Intermediate buffer shapes (elements × partials per stage)
6. Multi-stage numerical fidelity vs. single-pass reference
7. Rendering contract compliance: ping-pong allocation, barrier insertion, offset-list upload

### From ADR-004 (Streaming Loop):
8. Chunk-count derivation from `total_iterations` and `chunk_size`
9. Stride table correctness (constant parameters have stride 0; per-iteration parameters stride correctly)
10. Scratch buffer spec sizing matches `MemoryLayout` shapes
11. Collection buffer accumulation fidelity across iterations
12. Last-chunk handling when `total_iterations % chunk_size != 0`
13. Body node contract validation within streaming loop context
14. Rendering contract compliance: per-iteration dispatch, scratch allocation/deallocation

### From ADR-005 (Node 16):
15. Node 16 scalar parameters received correctly via `KernelDispatchNode.scalar_params`
16. Node 13 contiguity guarantee: `destination_buffer` of Node 13 equals `source_buffer` of Node 16
17. Policy parameter synthesis: `policy_max_k`, threshold schedule, `epsilon` computed correctly from `HardwareProfile` + `ModelSpec`
18. Numerical fidelity: Node 16 output matches reference numpy stabilized reduction

### From ADR-009 (Buffer Lifecycle):
19. Single-producer invariant: no buffer has two producing nodes
20. Coverage: every consumed buffer has a producing node or is `MODEL_STATE`/`BATCH_INPUT`
21. Shape consistency: producer and consumer shapes agree for every buffer
22. Topological ordering: `producing_node` precedes `last_consumer` in the DAG
23. Name and handle uniqueness across all descriptors
24. `size_bytes` correctness: `prod(padded_shape) * element_size_bytes`
25. Role classification consistency: `MODEL_STATE` ↔ learnable parameters; `BATCH_INPUT` ↔ host-uploaded data
26. Memory footprint estimation: plan-level + renderer-internal estimate fits within `HardwareProfile` available memory

### From ADR-010 (D2H Transfer & Phase Sync Points):
27. `RetrievalFuture` Protocol conformance: each backend's implementation passes `isinstance(future, RetrievalFuture)`
28. `result()` returns a numpy array with shape matching `RetrievalNode.logical_shape`
29. `result()` returns the same array on repeated calls (before `release()`)
30. `wait()` is idempotent — multiple calls do not raise
31. `release()` can be called after `result()` without error
32. Padding-stripping correctness: `result()` shape equals logical shape, not padded shape, when `padded_shape != logical_shape`
33. CPU zero-copy: `wait()` returns immediately; `result()` is a view over the compute buffer (no copy)
34. dtype consistency: `result().dtype` matches `PrecisionConfig.numpy_dtype`

---

## Tensions

- Tier 3 tests require multiple backends in CI, which may be impractical (e.g., no GPU in CI for OpenCL/Vulkan). Solution: Tier 3 runs locally or in GPU-enabled CI; Tier 1+2 run universally.
- Floating-point tolerances differ between backends (especially FP16). The tolerance model must be per-precision, not global.
- The CPU backend serves as a natural reference oracle (option C), reducing the need for a separate numpy implementation. But this creates a circular dependency: the CPU backend must be correct *first*.
- ADR-010's `release()` lifecycle introduces a new class of correctness tests: verifying that the renderer does not reclaim `BATCH_OUTPUT` memory before `release()` is called. These tests may require backend-specific introspection (e.g., checking that a `cl.Buffer` is still allocated) and belong in Tier 2.

---

## References

- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — test targets 1–7
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — test targets 8–14
- [ADR-005: Node 16 Opacity](ADR-005-node-16-opacity-in-the-plan.md) — test targets 15–18
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — test targets 19–26
- [ADR-010: D2H Transfer & Phase Sync Points](ADR-010-d2h-transfer-and-phase-sync-points.md) — test targets 27–34
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy-stub.md) — cross-language fidelity
- [ADR-014: Build System Integration](ADR-014-build-system-integration-stub.md) — CI backend availability
