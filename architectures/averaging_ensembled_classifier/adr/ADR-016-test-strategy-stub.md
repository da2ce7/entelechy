# ADR-016: Test Strategy

**Status:** STUB (NARROWED — kernel source locations and cross-backend fidelity model resolved by ADR-013; test tier structure and tooling remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-014  
**Blocks:** ADR-017

---

## Context

The multi-backend architecture (OpenCL, Vulkan, CPU) requires a layered test strategy that addresses:

1. **Host-side plan construction.** Verify that `ExecutionPlan` construction, `KernelContract` validation, `BufferContract` lifecycle, and `MemoryLayout` placement are correct independently of any backend.
2. **Per-backend kernel correctness.** Verify that each backend's compiled kernels produce numerically correct results for a known input/output pair.
3. **Cross-backend parity.** Verify that all backends produce equivalent outputs (within ADR-008 precision tolerances) for the same `ExecutionPlan`.

ADR-013 (ACCEPTED) resolves kernel source locations and establishes a **three-tier specification hierarchy** that directly structures the parity assurance model:

| Tier | Artifact | Verification mechanism |
| :--- | :--- | :--- |
| **Interface** | `KernelContract` (ADR-007) | Plan-construction-time validation — type/shape/padding/placement checks |
| **Algorithm** | `kernels.cl.h` in `kernels/` | Code review against algorithmic reference during development |
| **Implementation** | Per-backend sources in `src/backends/<name>/kernel_sources/` | Cross-backend oracle tests (this ADR, Tier 3) |

ADR-013 identifies three complementary fidelity mechanisms: *specification review*, *KernelContract interface tests*, and *cross-backend parity tests*. This stub must define how those mechanisms are realized as executable tests.

The kernel inventory is now known (ADR-013 §Decision, file table): ~20 kernels across 6 phase files, with Strategy A/B variants (ADR-011) expanding the effective test matrix. Each kernel has a `KernelContract` (ADR-007) that declares its interface and a reference algorithm in `kernels.cl.h`.

ADR-014 (pending) determines how backends are conditionally enabled at build time, which constrains which test tiers can execute in a given environment.

---

## Decision Required

### Tier 1 — Host-side plan correctness (no backend required)

Validate `ExecutionPlan` construction, node topological ordering, `KernelContract` parameter calculability proofs, `BufferContract` lifecycle (ADR-009), and `MemoryLayout` placement. These are pure Python tests with no hardware dependency.

### Tier 2 — Per-backend kernel correctness (single backend required)

For each enabled backend, dispatch individual kernels via `KernelBinding` against known input/output fixtures and verify numerical correctness within ADR-008 precision tolerances. The kernel sources are located per ADR-013:
- **OpenCL:** `kernels/*.cl.c` (architecture-root, no `kernel_sources/` subdirectory)
- **Vulkan:** `src/backends/vulkan/kernel_sources/*.comp`
- **CPU:** `src/backends/cpu/kernel_sources/*.c`

### Tier 3 — Cross-backend parity (two or more backends required)

Execute identical `ExecutionPlan` instances on every enabled backend and compare outputs. ADR-013 identifies the CPU backend as a natural oracle candidate (deterministic, bit-reproducible in non-SIMD mode). Tolerances governed by ADR-008 precision configuration.

### Options

- **(A) Layered pytest framework.** pytest markers/fixtures for each tier; `--backend` flags control which tiers execute. Tier 1 always runs. Tier 2 runs if any backend is available. Tier 3 runs if ≥ 2 backends are available.

- **(B) Parameterized test matrix.** Single test body parameterized across backends × kernels × strategies. Scales combinatorially but risks slow test suites and complex skip logic.

- **(C) CPU reference oracle.** Designate the CPU backend (deterministic, no GPU required) as the golden reference and run all parity comparisons against it. Simplifies Tier 3 to pairwise comparisons instead of all-vs-all. **Favored direction** per ADR-013's identification of CPU as natural oracle.

These are combinable — A provides the test framework structure; C defines the parity comparison strategy within Tier 3.

---

## Test Target Inventory (partial)

The kernel inventory from ADR-013 gives a concrete test surface per backend:

| Phase file | Kernels | Strategy variants | Tier 2 tests per backend |
| :--- | :--- | :--- | :--- |
| `phase_1_act` | forward_pass, compute_hidden_mask, compute_probs_loss_{cce,bce} | B (problem_type) | 4 |
| `phase_2_learn_A_production` | calculate_module_param_grads, calculate_ensemble_error_grads, calculate_module_grads | A (FLAG) | 3 × 2 = 6 |
| `phase_2_learn_B_processing` | Nodes 11, 13 | — | ~2 |
| `phase_2_learn_C_reduction` | Aggregation + clip kernels | — | ~3 |
| `phase_2_learn_D_backprop` | Nodes 16, 17, 18, 19 | — | ~4 |
| `phase_3_update` | Nodes 21, 24, 25 | — | ~3 |

This yields ~22 Tier 2 test cases per backend, ~22 × (backends − 1) Tier 3 comparisons under Option C.

---

## Tensions

- Tier 3 tests require at least two compiled backends in the CI environment, which depends on ADR-014's conditional backend enablement mechanism.
- The Strategy A FLAG variants (ADR-011) create combinatorial pressure — each Strategy A kernel needs two test paths. Option B addresses this but may over-parameterize.
- Precision tolerances (ADR-008) vary by kernel; Tier 3 comparisons need per-kernel tolerance tables rather than a single global epsilon.
- Using CPU as oracle (Option C) assumes CPU correctness is established independently (Tier 2). Circular dependency risk if CPU Tier 2 fixtures are derived from CPU execution.

---

## References

- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` interface specification; validation at plan-construction time
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — per-kernel tolerance configuration for numerical comparison
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferContract` lifecycle tested at Tier 1
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — Strategy A/B variants expanding test matrix
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — three-tier specification hierarchy; kernel source locations; cross-backend fidelity model; CPU as natural oracle candidate
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — conditional backend enablement constraining testable configurations
