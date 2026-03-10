# ADR-016: Test Strategy

**Status:** STUB (NARROWED — kernel source locations and cross-backend fidelity model resolved by ADR-013; conditional backend enablement and build manifest resolved by ADR-014; CPU FFI marshalling discipline and layout verification resolved by ADR-015; test tier structure, tooling, and fixture strategy remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
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

ADR-014 (ACCEPTED) resolves conditional backend enablement and provides a generated `_build_config.py` module that declares `BACKEND_OPENCL`, `BACKEND_VULKAN`, and `BACKEND_CPU` as booleans. This module is the authoritative source for determining which backends are available at runtime and, consequently, which test tiers can execute in a given environment:

- **Tier 1** (host-side plan correctness) — always runs; no backend required.
- **Tier 2** (per-backend kernel correctness) — runs for each backend where `_build_config.BACKEND_<NAME>` is `True`.
- **Tier 3** (cross-backend parity) — runs when two or more backends have `True` values. ADR-014's `auto` default for `backend_vulkan` and `backend_cpu` means CI environments get the tiers their hardware supports with no manual configuration.

ADR-015 (ACCEPTED) resolves the CPU backend's FFI mechanism as ctypes with struct layout verification. This has direct implications for the test strategy:

- **Layout verification as a pre-test gate.** ADR-015's `_verify_layouts()` function asserts that Python-side ctypes struct definitions match the C-side struct sizes at library load time. If layout drift exists, the CPU backend fails to initialize — the test suite never reaches Tier 2 with a corrupted FFI layer. This converts the most dangerous FFI failure mode (silent struct corruption) into a deterministic startup error that precedes test execution.
- **Tier 2 as implicit FFI validation.** ADR-015 notes that struct field reordering within the same type roster (e.g., swapping two `uint` fields) is not caught by size-based layout verification. Tier 2 per-kernel correctness tests are the behavioral safety net: a reordered struct produces numerically wrong results, which Tier 2 detects. This makes Tier 2 tests load-bearing for FFI correctness — not just algorithmic correctness.
- **CPU as oracle viability.** ADR-015's fine-grained dispatch pattern (Python walks the plan DAG, calls `pool_dispatch_and_wait` per node) means CPU execution is maximally inspectable. Combined with the deterministic, single-threaded-per-task execution model (CPU_BACKEND.md), the CPU backend remains the natural Tier 3 oracle candidate identified by ADR-013.

The kernel inventory is known (ADR-013 §Decision, file table): ~20 kernels across 6 phase files, with Strategy A/B variants (ADR-011) expanding the effective test matrix. Each kernel has a `KernelContract` (ADR-007) that declares its interface and a reference algorithm in `kernels.cl.h`.

---

## Decision Required

### Test framework structure

- **(A) Layered pytest framework.** pytest markers/fixtures for each tier; `--backend` flags control which tiers execute. Tier 1 always runs. Tier 2 runs for each backend where `_build_config.BACKEND_<NAME>` is `True`. Tier 3 runs if ≥ 2 backends are available. Skip logic reads `_build_config` at collection time.

- **(B) Parameterized test matrix.** Single test body parameterized across backends × kernels × strategies. Scales combinatorially but risks slow test suites and complex skip logic.

### Parity comparison strategy

- **(C) CPU reference oracle.** Designate the CPU backend (deterministic, no GPU required) as the golden reference and run all parity comparisons against it. Simplifies Tier 3 to pairwise comparisons instead of all-vs-all. **Favored direction** per ADR-013's identification of CPU as natural oracle, reinforced by ADR-015's fine-grained inspectable dispatch.

Options A and C are combinable — A provides the framework structure; C defines the parity comparison strategy within Tier 3.

### CPU Tier 2 fixture generation

- **(D) Independent golden fixtures.** Generate expected outputs from a trusted reference implementation (e.g., a manually-verified numpy computation for each kernel). CPU Tier 2 compares against these fixtures. Avoids circular dependency where CPU-generated outputs validate CPU correctness.

- **(E) Analytical fixtures.** For kernels with closed-form solutions (e.g., ReLU, element-wise scaling, Adam update with known inputs), derive expected outputs analytically. For kernels without closed-form solutions (e.g., tiled matmul), use numpy reference implementations with known-correct logic.

Option E is favored — it provides independently verifiable expected values for CPU Tier 2 without depending on any backend's execution.

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

- The Strategy A FLAG variants (ADR-011) create combinatorial pressure — each Strategy A kernel needs two test paths. Option B addresses this but may over-parameterize.
- Precision tolerances (ADR-008) vary by kernel; Tier 3 comparisons need per-kernel tolerance tables rather than a single global epsilon.
- Using CPU as oracle (Option C) assumes CPU correctness is established independently (Tier 2). Option E mitigates the circular dependency risk by using analytically/numpy-derived fixtures for CPU Tier 2.
- CI environments without GPU hardware can still run Tier 1 + CPU Tier 2 (ADR-014's `backend_cpu` auto-enables if a C compiler is present). Tier 3 requires at least one GPU backend, gating full parity testing to GPU-equipped CI.
- ADR-015's layout verification catches size-level struct drift at startup, but field reorderings within the same type roster are caught only by Tier 2 behavioral tests. The test strategy must treat Tier 2 CPU tests as load-bearing for FFI correctness, not merely algorithmic correctness.

---

## References

- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` interface specification; validation at plan-construction time
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — per-kernel tolerance configuration for numerical comparison
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferContract` lifecycle tested at Tier 1
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — Strategy A/B variants expanding test matrix
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — three-tier specification hierarchy; kernel source locations; cross-backend fidelity model; CPU as natural oracle candidate
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — `_build_config.py` manifest for enabled backend discovery; conditional backend enablement via Meson `feature` options; `auto` default enabling zero-configuration CI
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop.md) — ctypes FFI for CPU backend; layout verification as pre-test gate; Tier 2 as implicit FFI validation; CPU oracle viability via fine-grained inspectable dispatch
