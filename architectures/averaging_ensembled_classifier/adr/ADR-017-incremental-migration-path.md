# ADR-017: Incremental Migration Path

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-014, ADR-015, ADR-016  
**Blocks:** ADR-018

---

## Context

The averaging ensembled classifier architecture is being migrated from a monolithic Python+PyOpenCL implementation to a multi-backend, formally-specified system. The migration must be incremental — existing PyOpenCL functionality must remain operational at each step, with new infrastructure (memory layout, execution plan, CPU backend, Vulkan backend) layered in behind feature gates.

The resolved ADRs establish the technical constraints for each migration phase:

| ADR | Key constraint on migration |
| :--- | :--- |
| ADR-001 | Backend abstraction boundary — backends expose a uniform dispatch interface behind the plan model; three-tier jurisdictional model (Policy / Orchestration / Execution) |
| ADR-007 | `KernelContract`/`KernelBinding` split — host-side validation distinct from per-backend compilation |
| ADR-009 | `BufferContract` lifecycle — all buffer allocations, transfers, and releases are plan-controlled |
| ADR-012 | Module factoring — services dissolution into stateless plan primitives; `src/shared/` + `src/backends/<name>/` directory structure |
| ADR-013 | Kernel source strategy — `kernels.cl.h` as shared spec; per-backend source directories under `src/backends/`; ~22 kernels across 6 phase files |
| ADR-014 | Build system — Meson with `meson-python`; `_build_config.py` manifest; `feature` options with `auto` default; conditional `subdir()` delegation |
| ADR-015 | CPU FFI — ctypes with struct layout verification; `_ffi_types.py` for struct definitions; dispatch table mapping `kernel_name` → `(task_fn_ptr, args_struct_class)`; `_verify_layouts()` at library load |
| ADR-016 | Test strategy — layered pytest framework (Tier 1/2/3); CPU oracle for Tier 3 parity; analytical + numpy fixtures for Tier 2; per-kernel tolerance tables; `_build_config.py`-driven skip logic |

ADR-016 (ACCEPTED) resolves the test strategy and provides concrete rollback gate definitions for each migration phase. The tier structure is:

- **Tier 1** — host-side plan correctness; always runs; no backend required.
- **Tier 2** — per-backend kernel correctness against analytical/numpy fixtures; runs per enabled backend.
- **Tier 3** — cross-backend parity with CPU as reference oracle; runs when ≥ 2 backends are available.

ADR-014 (ACCEPTED) resolves the build system and provides the gating mechanism. Each backend is controlled by a Meson `feature` option (`backend_opencl`, `backend_vulkan`, `backend_cpu`) with `auto` as the default. The generated `_build_config.py` module declares `BACKEND_OPENCL`, `BACKEND_VULKAN`, and `BACKEND_CPU` as booleans. ADR-016's skip logic reads these booleans at test collection time.

### Open questions resolved by this ADR

1. **Phase sequencing strategy.** Must phases execute strictly sequentially, or can independent phases overlap? If overlapping, what mechanism gates parallel work?
2. **Rollback gate formalization.** How are rollback gates defined — purely in terms of ADR-016's test tiers, or extended with quantitative performance and resource-consumption thresholds?

---

## Decision Drivers

1. **CONCEPT.md §1 (Architectural Elegance Feedback).** Optimization pressure that violates core principles is signal for incomplete architectural modeling. If a migration shortcut creates tension with the plan model, the response is to extend the architecture — not to bypass the migration discipline. This principle applies to the migration process itself: if phase overlap reveals a structural conflict, it is a signal to refine phase boundaries.

2. **CONCEPT.md §5 (Unified Execution Model).** All workflows follow Act/Learn sequencing regardless of backend. The migration must preserve this invariant at every phase boundary — no intermediate state may break the Act/Learn contract.

3. **ADR-001 (Backend Abstraction Boundary — plan-as-data-structure).** The shared orchestration layer produces a backend-neutral execution plan; each backend renders it natively. Migration phases must respect this boundary: shared-layer changes (Phases 0–1) are independent of backend-layer changes (Phases 2–5).

4. **ADR-014 (`_build_config.py` — `auto`/`enabled`/`disabled` feature options).** The build system already provides per-backend feature flags with `auto` as default. This is an existing gating mechanism that migration can leverage directly — each backend's availability is independently toggleable without affecting others.

5. **ADR-016 (Test strategy — layered tiers with concrete gates).** ADR-016's Consequences section explicitly defines the rollback gate for each migration phase in terms of tier outcomes. These definitions are concrete and testable. The question is whether they are sufficient or require extension.

6. **ADR-012 (Module factoring — `src/shared/` + `src/backends/<name>/`).** The directory structure split provides physical isolation between the shared layer and each backend. This isolation enables parallel development: changes to `src/backends/cpu/` cannot affect `src/backends/vulkan/`, and neither can affect `src/shared/`.

7. **ADR-015 (CPU FFI — layout verification as pre-gate).** ADR-015's `_verify_layouts()` converts struct size drift into a deterministic startup error that precedes test execution. This provides a fast-fail mechanism within Phase 3 that catches the most dangerous FFI failure mode before Tier 2 tests run.

8. **ADR-013 (Kernel source strategy — `kernels.cl.h` as algorithmic spec).** All backends implement from the same algorithmic specification. Backend development is independently verifiable against this specification, supporting parallel backend development.

9. **Practical development velocity.** The migration involves multiple independent workstreams (shared-layer refactoring, CPU backend implementation, Vulkan backend implementation, test harness construction). Strict sequential ordering would serialize inherently parallel work, significantly extending the migration timeline.

---

## Options Considered

### Phase sequencing strategy

#### Option A: Strict sequential

Each phase must fully complete and pass its rollback gate before the next begins. No partial overlap.

**Advantages:**
- Simplest to reason about — at any point, exactly one phase is in progress.
- Each phase builds on a fully validated foundation.
- No risk of cross-phase interference.

**Disadvantages:**
- Serializes inherently parallel work. Phase 3 (CPU backend) and Phase 5 (Vulkan backend) have no technical dependency — they target different directories (`src/backends/cpu/`, `src/backends/vulkan/`), different build targets, and different test markers.
- Phase 4 (test harness) is unblocked now — ADR-016 is decided — but cannot begin until Phases 1–3 complete under this option. The harness would be most valuable *during* Phases 2–3, not after.
- Estimated timeline impact: 2–3× longer than overlapping options for a multi-contributor project.

#### Option B: Overlapping phases with dependency ordering

Phases may overlap where their deliverables are independent. Dependency ordering is maintained manually: Phase 3 (CPU backend) requires Phase 1 (plan model) to be stable; Phase 5 (Vulkan backend) requires Phase 1 to be stable; Phase 4 (test harness) can begin immediately.

**Advantages:**
- Respects technical dependencies while enabling parallelism.
- Phase 4 (test harness) begins immediately, providing validation infrastructure during backend development.
- CPU and Vulkan backends can proceed in parallel once the shared layer is stable.

**Disadvantages:**
- Requires manual coordination to determine when a dependency is "stable enough" to begin dependent work. "Stable" is ambiguous without a formal gate definition.
- No mechanical enforcement of dependency ordering — relies on developer discipline.

#### Option C: Feature-flag gated

All phases proceed in parallel behind `_build_config.py` feature flags (ADR-014). Each backend is independently toggleable. Integration testing gates promotion of each feature flag from `auto` to `enabled`. Each workstream has its own independent gate defined by ADR-016's tier outcomes.

**Advantages:**
- Leverages ADR-014's existing `auto`/`enabled`/`disabled` feature options — no new gating infrastructure required.
- Maximum development velocity — all independent workstreams proceed simultaneously.
- Mechanical enforcement: a backend's feature flag remains `auto` (probe-based) until its tier gate passes, at which point it can be promoted to `enabled` in CI. A failing gate prevents promotion — no ambiguity about "stable enough."
- Graceful degradation: if a backend's development stalls, the rest of the system continues to function with that backend at `auto` (or `disabled`). No blocking dependency chain.
- Aligns with ADR-016's skip logic: tiers are skipped (not failed) when a backend is absent, so an incomplete backend does not cause CI failures.

**Disadvantages:**
- Requires that the shared layer (Phases 0–1) is stable before backend phases can make meaningful progress. If the plan model's API is in flux, backend renderers built against it will need rework. This is mitigated by the fact that the plan model's design is constrained by ADRs 001–011, which are all accepted.
- Parallel development across backends requires clear module boundaries — but ADR-012's physical directory split already provides this.
- Risk of integration-time surprises when multiple independently-developed backends first interact in Tier 3 testing. This is the intended purpose of Tier 3.

### Rollback gate formalization

#### Option D: Tier-based gates

Each phase's rollback gate is defined in terms of ADR-016's test tiers. Phase 1 requires Tier 1 green; Phase 3 requires Tier 1 + CPU Tier 2 green; Phase 5 requires Tier 1 + Vulkan Tier 2 + Tier 3 parity green; Phase 6 requires Tier 3 parity green across all enabled backends. ADR-016's tier definitions and per-kernel tolerance tables provide the concrete criteria.

**Advantages:**
- Gates are testable and unambiguous — "green" means all tests in the tier pass.
- Directly reuses ADR-016's existing infrastructure. No new tooling required.
- Per-kernel tolerance tables (ADR-016) provide precision-aware comparison that accounts for expected numerical divergence.
- Consistent vocabulary: "Tier 1 green" has a single meaning across all documentation.

**Disadvantages:**
- Does not capture performance regression. A backend that produces correct results but is 10× slower than the legacy implementation would pass its tier gate.
- Does not capture resource consumption. A backend that allocates 10× more device memory would pass its tier gate.

#### Option E: Metric-based gates

In addition to tier-based gates, define quantitative thresholds for performance and resource consumption. E.g., maximum allowable latency regression per kernel, maximum device memory overhead vs. the legacy implementation, minimum throughput for a canonical workload.

**Advantages:**
- Catches performance and resource regressions that correctness-only gates miss.
- Provides quantitative targets for backend optimization.

**Disadvantages:**
- Performance baselines are environment-dependent — a threshold that passes on a high-end GPU may fail on integrated graphics. ADR-014's `auto` default means CI environments are heterogeneous.
- The legacy PyOpenCL implementation does not have established performance baselines. Creating them is a separate effort that precedes this ADR's scope.
- Premature optimization targets conflict with CONCEPT.md §1: if a performance threshold forces an optimization that violates plan-model contracts, the response should be to evolve the architecture — not to compromise correctness for speed.
- Increases gate complexity and maintenance burden. Each metric requires collection infrastructure, baseline management, and threshold tuning.

---

## Analysis

### Choosing Option C (feature-flag gated)

Option A is eliminated on practical grounds. The migration involves at least four independent workstreams: shared-layer refactoring, CPU backend, Vulkan backend, and test harness. Serializing these imposes a 2–3× timeline penalty with no correctness benefit — the workstreams operate on physically isolated directory trees (ADR-012) and independently toggleable feature flags (ADR-014).

Option B is subsumed by Option C. Option B's "overlapping with dependency ordering" is exactly what Option C provides, but with a mechanical enforcement mechanism (feature flags + tier gates) instead of manual coordination. The distinction is not philosophical but operational: under Option B, "Phase 1 is stable enough to start Phase 3" is a judgment call. Under Option C, "Phase 1's Tier 1 gate is green" is a CI observable.

Option C leverages three existing architectural primitives:

1. **ADR-014's feature flags.** `backend_cpu` and `backend_vulkan` are Meson `feature` options with `auto` default. A backend under development naturally starts at `auto` — it is available when its toolchain is present, absent otherwise. Promotion to `enabled` (the backend *must* be present; build fails if it cannot be found) is a deliberate act that follows gate passage.

2. **ADR-016's tier-based skip logic.** The `pytest_collection_modifyitems` hook reads `_build_config.py` and skips tiers whose backends are absent. An incomplete backend does not cause test failures — its tiers are simply not collected. This means a developer working on the CPU backend can run `pytest -m "tier1 or (tier2 and cpu)"` without Vulkan and OpenCL tiers interfering.

3. **ADR-012's physical isolation.** `src/shared/`, `src/backends/opencl/`, `src/backends/cpu/`, and `src/backends/vulkan/` are distinct directory trees. Changes to one backend cannot affect another's source or build artifacts.

The primary risk — shared-layer API instability affecting parallel backend work — is mitigated by the fact that the plan model's interface is heavily constrained by ADRs 001 through 011. The node types (ADR-002), reduction tree representation (ADR-003), streaming loop representation (ADR-004), buffer lifecycle (ADR-009), and kernel contract/binding split (ADR-007) are all accepted decisions. The shared-layer API is not a moving target; it is a concrete specification that backend developers can code against.

### Choosing Option D (tier-based gates)

Option D is chosen over Option E. The decisive factors:

**Correctness is the primary migration objective.** The migration's purpose is to move from a monolithic PyOpenCL implementation to a multi-backend, formally-specified system *without breaking correctness at any step*. Performance optimization is a subsequent concern, not a migration gate. CONCEPT.md §1 explicitly warns against optimization pressure that violates architectural contracts — making performance a gate condition creates exactly this pressure.

**Performance baselines do not exist.** The current implementation has no established performance benchmarks with defined thresholds. Creating a baseline suite, running it across representative hardware, stabilizing the measurements, and setting meaningful thresholds is a separate engineering effort. Blocking the migration on this effort is not justified.

**ADR-016's tiers are concrete and sufficient.** The tier definitions provide unambiguous pass/fail criteria:
- Tier 1: plan construction correctness — structural validation with no numerical component.
- Tier 2: per-kernel numerical correctness within per-kernel tolerance tables — catches algorithmic and FFI errors.
- Tier 3: cross-backend parity within tolerance — catches implementation divergence.

These criteria catch the failure modes that matter during migration: broken contracts, incorrect computation, and cross-backend inconsistency. Performance regressions are observable through benchmarks run *after* a phase's tier gate passes — they become optimization work items, not gate blockers.

**Option E can be added later without rework.** If performance baselines are established post-migration, adding metric-based gates to specific phases (e.g., "Phase 6 legacy removal requires no more than 10% throughput regression") is an additive change. The tier-based gates remain valid; metric gates layer on top. This aligns with CONCEPT.md §1: formalize when the need is concrete, not speculatively.

---

## Decision

**Option C (feature-flag gated phase sequencing) + Option D (tier-based rollback gates).**

All migration phases proceed in parallel behind ADR-014's `_build_config.py` feature flags. Each phase has a rollback gate defined by ADR-016's test tier outcomes. Phase completion is a CI-observable event: the gate passes in CI, the phase is marked complete, and the feature flag can be promoted from `auto` to `enabled`.

### Phase structure

The migration is organized into seven phases (Phase 0 through Phase 6). Phases 0–1 establish the shared layer. Phases 2–5 are independent backend and infrastructure workstreams that may proceed in parallel once their shared-layer dependencies are met. Phase 6 is the terminal phase that removes the legacy code path.

#### Phase 0: Foundation

**Objective:** Create the directory structure and move existing modules without behavioral change. No new functionality is introduced.

**Deliverables:**
- Create `src/shared/` and `src/backends/opencl/` directory skeletons (ADR-012).
- Move backend-neutral modules (`model_spec.py`, `parameter_space.py`, `memory_layout.py`, `stabilization_policy.py`, `workload_primitives.py`) into `src/shared/` (ADR-012).
- Extract `HardwareProfile` → `src/shared/hardware_profile.py` (ADR-006).
- Extract `PrecisionConfig` → `src/shared/precision_config.py` (ADR-008).
- Extract `KernelContract` hierarchy → `src/shared/kernel_contracts/` (ADR-007).
- Extract `ProblemTypeStrategy` → `src/shared/problem_type_strategy.py` (ADR-011).
- Move OpenCL-specific modules into `src/backends/opencl/` with thin shim adapters (ADR-012).
- Create `meson.options` with `backend_vulkan` and `backend_cpu` options (ADR-014).
- Create `src/_build_config.py.in` template (ADR-014).
- Designate `kernels.cl.h` as the algorithmic specification (ADR-013).
- All existing tests pass with updated import paths.

**Rollback gate:** All existing tests green. No tier gate — Tier 1 tests do not yet exist (they are a Phase 1 deliverable). The gate is the existing test suite: every test that passed before Phase 0 must pass after.

**Dependencies:** None. Phase 0 is the starting point.

#### Phase 1: Host-side plan model

**Objective:** Implement the shared orchestration layer's plan data structures. The plan model is usable but not yet consumed by any backend renderer.

**Deliverables:**
- Implement plan node types as frozen dataclasses → `src/shared/plan_types.py` (ADR-002).
- Implement `BufferHandle`, `BufferRole`, `BufferDescriptor` → `src/shared/buffer_lifecycle.py` (ADR-009).
- Implement `ReductionTreePlan` → `src/shared/reduction_tree_plan.py` (ADR-003).
- Implement `StreamingLoopPlan` → `src/shared/streaming_loop_plan.py` (ADR-004).
- Implement `RetrievalFuture` Protocol → `src/shared/retrieval_future.py` (ADR-010).
- Implement plan builder → `src/shared/plan_builder.py`.
- Write Tier 1 tests: plan construction, contract validation, buffer lifecycle, reduction tree plan, streaming loop plan, strategy delegation, memory layout, precision config (ADR-016).

**Rollback gate:** Tier 1 green — all plan construction, contract validation, buffer lifecycle, and strategy delegation tests pass.

**Dependencies:** Phase 0 (directory structure must exist).

#### Phase 2: PyOpenCL backend adapter

**Objective:** Wrap the existing PyOpenCL kernel dispatch in the plan-model's `PlanRenderer` interface. The legacy code path is preserved alongside the new renderer; both produce identical results.

**Deliverables:**
- Implement OpenCL `PlanRenderer` → `src/backends/opencl/renderer.py` (ADR-001).
- Implement `_OpenCLRetrievalFuture` → `src/backends/opencl/retrieval.py` (ADR-010).
- Implement OpenCL `KernelBinding`s → `src/backends/opencl/kernel_bindings/` (ADR-007).
- Implement OpenCL buffer allocator, discovery, type mapping, context → `src/backends/opencl/` (ADR-012).
- Configure OpenCL backend to load source files from `kernels/` (ADR-013).
- Write OpenCL Tier 2 tests (ADR-016).

**Rollback gate:** Tier 1 green + OpenCL Tier 2 green — existing kernel correctness preserved under the new plan-model dispatch.

**Dependencies:** Phase 1 (plan model must be stable enough to render against). Under Option C, this means Phase 1's Tier 1 gate must be green.

#### Phase 3: CPU backend

**Objective:** Implement the CPU backend as a native C library with ctypes FFI, providing a deterministic, inspectable execution path.

**Deliverables:**
- Create `src/backends/cpu/kernel_sources/` with C implementations of all ~22 kernels, developed against `kernels.cl.h` as the algorithmic reference (ADR-013).
- Implement `cpu_simd.h` ISA detection cascade (CPU_BACKEND.md).
- Implement `cpu_kernels.h` ABI surface with `task_<kernel_name>` function signatures and `get_struct_size_*` verification functions (ADR-015).
- Build `libcpu_kernels.so` via Meson `shared_library()` (ADR-014).
- Create `src/backends/cpu/_ffi_types.py` with ctypes struct definitions (ADR-015).
- Implement library loading via `importlib.resources` + `ctypes.CDLL` (ADR-015).
- Implement layout verification `_verify_layouts()` at library load time (ADR-015).
- Implement dispatch table mapping `kernel_name` → `(task_fn_ptr, args_struct_class)` (ADR-015).
- Implement `CPUPlanRenderer` with `pool_dispatch_and_wait` as the blocking dispatch primitive (ADR-001, ADR-015).
- Create `src/backends/cpu/meson.build` with `shared_library` target (ADR-014).
- `_build_config.py` gains `BACKEND_CPU = True`.
- Write CPU Tier 2 tests (ADR-016).

**Rollback gate:** Tier 1 green + CPU Tier 2 green — layout verification passes and all 22 CPU Tier 2 kernel tests produce numerically correct results against analytical/numpy fixtures within the per-kernel tolerance table.

**Dependencies:** Phase 1 (plan model). Independent of Phase 2 (OpenCL adapter).

#### Phase 4: Test harness

**Objective:** Bring the full Tier 1/2/3 test framework to operational status. This phase can begin immediately — ADR-016 is decided — and serves as validation infrastructure during the development of Phases 2, 3, and 5.

**Deliverables:**
- Implement shared test infrastructure: tolerance configuration, fixture factories, backend fixture lifecycle (ADR-016).
- Implement `tests/conftest.py` with `_build_config`-driven skip logic and `pytest_collection_modifyitems` hook (ADR-016).
- Implement `tests/tier1/`, `tests/tier2/`, `tests/tier3/` directory structure with per-tier conftest and test files (ADR-016).
- Implement analytical fixtures for closed-form kernels (ADR-016).
- Implement numpy reference implementations for complex kernels (ADR-016).
- Implement Tier 3 oracle selection logic and multi-backend fixture setup (ADR-016).
- Implement `--all-pairs` optional mode for supplementary GPU-vs-GPU comparison (ADR-016).
- Validate the framework against whatever backends are currently available.

**Rollback gate:** Full Tier 1/2/3 framework operational; all enabled tiers green. This gate is satisfied incrementally — the framework reports correct results for whatever backends exist at the time.

**Dependencies:** ADR-016 (decided). Phase 4 can begin in parallel with all other phases. It consumes infrastructure from Phases 1–3 as that infrastructure becomes available.

#### Phase 5: Vulkan backend

**Objective:** Implement the Vulkan backend with GLSL compute shaders, SPIR-V compilation, and vulkan-python dispatch.

**Deliverables:**
- Create `src/backends/vulkan/kernel_sources/` with GLSL compute shaders, developed against `kernels.cl.h` as the algorithmic reference (ADR-013).
- Implement `common.glsl` shared specialization constant declarations (ADR-013).
- Implement SPIR-V compilation via Meson `custom_target` (`glslc --target-env=vulkan1.1`) (ADR-014).
- Implement Vulkan `PlanRenderer` with command buffer recording, pipeline barriers, and descriptor set management (ADR-001, VULKAN_BACKEND.md).
- Implement single-dispatch parallelism: `vkCmdDispatch(N, 1, 1)` for N-tile kernels (VULKAN_BACKEND.md).
- Create `src/backends/vulkan/meson.build` with `custom_target` SPIR-V compilation rules (ADR-014).
- `_build_config.py` gains `BACKEND_VULKAN = True`.
- Write Vulkan Tier 2 tests (ADR-016).

**Rollback gate:** Tier 1 green + Vulkan Tier 2 green + Tier 3 parity (Vulkan-vs-oracle) green.

**Dependencies:** Phase 1 (plan model). Independent of Phases 2, 3, and 4.

#### Phase 6: Legacy PyOpenCL removal

**Objective:** Remove the legacy code path. The system operates exclusively through the plan-model dispatch. All backends are exercised through their `PlanRenderer` implementations.

**Deliverables:**
- Delete dissolved modules: `arch_primitives.py`, `cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`, `execution_plan.py`, `kernel_signatures/`, `graph_recipes.py`, `batch_processor.py` (ADR-012).
- Remove thin shim adapters introduced in Phase 0.
- Retire legacy test infrastructure — Tier 1/2/3 framework is the sole test infrastructure (ADR-016).
- Update `main_orchestrator.py` to use plan-model dispatch exclusively.
- The physical directory structure is now in its final state (ADR-012).

**Rollback gate:** Tier 3 parity green across all enabled backends for the full 22-kernel inventory at both FP32 and supported FP16 configurations. This is the most stringent gate — it requires that every backend produces equivalent results for every kernel at every supported precision before the legacy fallback is removed.

**Dependencies:** All preceding phases. Phase 6 is the terminal phase and cannot begin until Phases 2, 3, 4, and 5 have passed their respective gates.

#### User-Facing API Workstream (Phase 4-adjacent)

**Objective:** Implement the `WorkTicket` / `LearnHandle` / `Engine` user-facing API surface on top of the plan model, enabling both Sequential and Event-Triggered execution modes (CONCEPT.md §4, §5). This workstream develops in parallel with Phase 4 (Test Harness), consuming the plan builder and `PlanRenderer` interfaces as they stabilize.

**Deliverables:**
- Implement `WorkTicket` (stateful ticket with `PENDING → ACT_COMPLETE → RESOLVED → CONSUMED` lifecycle) → `src/shared/work_ticket.py` (ADR-018).
- Implement `LearnHandle` (Learn-phase future wrapping `RetrievalFuture`) → `src/shared/work_ticket.py` (ADR-018).
- Implement `Engine` (user-facing entry point: `submit()`, `train_batch()`) → `src/shared/engine.py` (ADR-018).
- Implement Act-only and Learn-only plan construction modes in the plan builder (ADR-018 Choice 3A + 4A: explicit batch, recompute).
- Feature-flag gate: the ticket API is gated behind its own `_build_config.py` flag, independently toggleable.
- Write Tier 1 tests for ticket lifecycle, plan-splitting correctness, and Engine convenience methods.
- Write integration tests exercising the ticket API against each available backend's `PlanRenderer`.

**Rollback gate:** Tier 1 green for ticket lifecycle tests + integration tests green against at least one backend renderer.

**Dependencies:** Phase 1 (plan model must be stable). Independent of Phases 2, 3, and 5. Cross-pollinates with Phase 4: ticket API provides realistic plan builder usage; test harness validates ticket plan-construction patterns.

**Risk:** If Phase 1's plan builder API undergoes significant revision, the ticket workstream must pause until the interface re-stabilizes. This risk is mitigated by the heavy constraint ADRs 001–010 place on the plan builder's shape.

### Phase dependency graph

```
Phase 0 (Foundation)
  │
  ▼
Phase 1 (Plan Model) ──────── Tier 1 gate
  │
  ├──▶ Phase 2 (OpenCL Adapter) ─── Tier 1 + OpenCL Tier 2 gate
  │
  ├──▶ Phase 3 (CPU Backend) ────── Tier 1 + CPU Tier 2 gate
  │
  ├──▶ Phase 5 (Vulkan Backend) ─── Tier 1 + Vulkan Tier 2 + Tier 3 gate
  │
  ├──▶ User-Facing API ─────────── Tier 1 (ticket) + integration green
  │    [Phase 4-adjacent; ADR-018]
  │
  │    Phase 4 (Test Harness) ────── All enabled tiers green
  │    [can start in parallel with Phase 0]
  │
  ▼
Phase 6 (Legacy Removal) ───── Tier 3 parity green, all backends, FP32 + FP16
  [requires Phases 2, 3, 4, 5 complete]
```

Phases 2, 3, and 5 are independent workstreams that fan out from Phase 1. The User-Facing API workstream also fans out from Phase 1 and cross-pollinates with Phase 4. Phase 4 has no hard dependency — it begins immediately and grows its coverage as backend infrastructure becomes available. Phase 6 is a join point that requires all preceding phases to have passed their gates.

### Feature-flag lifecycle

Each backend follows a three-stage lifecycle through ADR-014's feature flag system:

| Stage | Feature flag value | Meaning |
| :--- | :--- | :--- |
| **Development** | `auto` (default) | Backend is available when its toolchain is detected; absent otherwise. CI machines without the toolchain skip the backend's tiers. |
| **Validated** | `enabled` (explicit) | Backend's tier gate has passed. CI is configured to require the backend — a build failure occurs if the toolchain is missing. This is a deliberate promotion. |
| **Mandatory** | `enabled` (enforced) | Phase 6 complete. The backend is a required component. `auto` is no longer appropriate — the system depends on the backend's presence. |

The transition from `auto` to `enabled` is the operational definition of "phase complete" for Phases 2, 3, and 5. It is a deliberate act in the CI configuration, triggered by the tier gate passing consistently.

### Rollback protocol

If a phase's tier gate fails after previously passing (regression), the rollback procedure is:

1. **Revert the feature flag** for the affected backend from `enabled` back to `auto`. This immediately removes the backend from CI's required-pass set, restoring CI stability.
2. **Diagnose** the regression using the tier test output. Tier 2 failures indicate per-kernel correctness issues; Tier 3 failures indicate cross-backend divergence.
3. **Fix-forward or revert** the offending change. The tier test output provides diagnostic specificity: which kernel, which backend, which tolerance was exceeded.
4. **Re-promote** the feature flag to `enabled` once the gate is green again.

The `auto` default ensures graceful degradation — a backend under repair does not block development on other backends or the shared layer.

### Distinguishing "skipped tier" from "failed tier"

ADR-016's skip logic marks tiers as *skipped* (not *failed*) when a backend is absent. The rollback gate definitions must interpret these correctly:

- **Skipped** (backend absent): The tier's precondition is not met. The gate is *not applicable* for that backend in this environment. This is not a failure.
- **Failed** (backend present, test fails): The tier's precondition is met but the test produces incorrect results. This is a gate failure.

A phase's gate is satisfied when: all applicable tiers are green (no failures), and the required backends are present (not skipped). The CI environment must have the necessary backends installed for a gate to be evaluable. ADR-014's `enabled` flag enforces this — a backend set to `enabled` fails the build if its toolchain is missing, ensuring the tier is not inadvertently skipped.

---

## Consequences

### Positive

- **Maximum development velocity.** Independent workstreams (CPU backend, Vulkan backend, test harness) proceed in parallel without blocking each other. The shared layer (Phases 0–1) is the only sequential bottleneck, and its scope is bounded by the already-decided ADRs.

- **Mechanical enforcement via existing infrastructure.** Feature flags (ADR-014), tier-based skip logic (ADR-016), and physical directory isolation (ADR-012) together provide a robust gating mechanism. No new tooling or infrastructure is required — the migration rides on architectural primitives that are already decided and (partially) implemented.

- **Graceful degradation at every phase.** A stalled backend does not block the rest of the system. The `auto` default means the stalled backend simply disappears from CI until its issues are resolved. Other backends continue to develop and validate independently.

- **Unambiguous gate criteria.** "Tier 1 green + CPU Tier 2 green" is a CI-observable, binary condition. There is no ambiguity about whether a phase is "complete enough" — the gate either passes or it does not.

- **Rollback is cheap.** Reverting a feature flag from `enabled` to `auto` is a one-line change that immediately restores CI stability. The tier test infrastructure provides specific diagnostic output for regression analysis.

- **Phase 4 (test harness) as a concurrent enabler.** The test harness begins immediately and provides validation infrastructure throughout the migration. Backend developers can run tier tests against their work-in-progress backends from the earliest stages, catching issues during development rather than at phase boundaries.

- **Consistent with cross-ADR expectations.** ADRs 002, 012, 013, 014, 015, and 016 all contain "Migration implications" sections keyed to this ADR's phase structure. The phase definitions here are consistent with those sections.

### Negative

- **Shared-layer API stability is load-bearing.** Option C's parallel workstreams are viable only if the plan model's API (Phase 1) is stable. If the plan model requires significant revision after Phases 2/3/5 begin, all backend renderers must adapt. This risk is mitigated by the extensive prior ADR work (001–011) that constrains the plan model's design, but it is not eliminated.

- **Integration-time surprises.** Backends developed in parallel may encounter unexpected interaction effects when they first meet in Tier 3 testing. A numerical divergence that both backends independently tolerate (within their Tier 2 tolerances) may exceed the Tier 3 cross-backend tolerance. This is by design — Tier 3 exists precisely to catch these effects — but the diagnostic work occurs later in the migration rather than earlier.

- **No performance gates.** Option D's tier-based gates do not catch performance regressions. A backend that is 10× slower than the legacy implementation passes its gate. Performance optimization is deferred to post-migration work. This is a deliberate trade-off: correctness-first migration, with performance benchmarking as a follow-on effort.

- **CI environment requirements grow.** Phase 6 requires all backends to be present and passing. CI machines must have the C compiler toolchain (CPU backend), the Vulkan SDK with `glslc` (Vulkan backend), and an OpenCL-capable device (OpenCL backend). This is a higher infrastructure bar than the current OpenCL-only requirement. ADR-014's `auto` default mitigates this during development, but Phase 6's gate requires the full environment.

- **Phase 0 import churn.** Every existing import path changes when modules move to `src/shared/` and `src/backends/opencl/`. This is a one-time mechanical cost (ADR-012) but affects all test files, the main orchestrator, and any internal tooling. It must be the first change, and it must be atomic — partial migration of import paths creates an inconsistent codebase.

### Migration implications for other ADRs

This decision formalizes the phase structure that ADRs 002, 012, 013, 014, 015, and 016 reference in their "Migration implications" sections. The phase definitions here are authoritative:

| ADR | Phase reference | Alignment |
| :--- | :--- | :--- |
| ADR-001 | Phases 0–4 in Consequences | Consistent — ADR-001's four steps map to Phases 0, 1, 2, and 3–5 here |
| ADR-002 | Phase 2 in Migration implications | Consistent — node type definitions are Phase 1; OpenCL renderer is Phase 2 |
| ADR-012 | Phases 0, 1, 2, 4 in Migration implications | Consistent — directory structure (Phase 0), plan model (Phase 1), OpenCL renderer (Phase 2), old code removal (Phase 6) |
| ADR-013 | Phases 0, 2, 3, 4 in Migration implications | Consistent — kernel source locations are Phase-invariant; per-backend sources are created in their respective phases |
| ADR-014 | Phases 0, 2, 3, 4 in Migration implications | Consistent — `meson.options` is Phase 0; per-backend `meson.build` files are created in their respective phases |
| ADR-015 | Phase 3 in Migration implications | Consistent — CPU FFI is entirely within Phase 3 |
| ADR-016 | All phases in Migration implications | Consistent — tier definitions map directly to rollback gates defined here |
| ADR-018 | User-Facing API workstream (Phase 4-adjacent) | Consistent — `WorkTicket`/`Engine` develop behind feature flag, consuming plan builder post-Phase 1; cross-pollinates with Phase 4 test harness |

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; three-tier jurisdictional model (Policy / Orchestration / Execution); `PlanRenderer` interface
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — typed DAG nodes; plan-construction-time validation; Phase 2 migration implications
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — `ReductionTreePlan` frozen dataclass; stage counts and fan-in
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — `StreamingLoopPlan` chunk counts and parameter strides
- [ADR-006: Hardware Profile](ADR-006-hardware-profile.md) — `HardwareProfile` extraction from OpenCL context manager
- [ADR-007: Kernel Signature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` / `KernelBinding` separation; host-side validation
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig` frozen dataclass; precision-derived tolerances
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` lifetime annotations; plan-controlled allocation
- [ADR-010: D2H Transfer & Phase Sync Points](ADR-010-d2h-transfer-and-phase-sync-points.md) — `RetrievalFuture` Protocol; named synchronization points
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — `ProblemTypeStrategy` extraction; Strategy A/B test matrix expansion
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` + `src/backends/<name>/` directory structure; services dissolution; physical isolation
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — `kernels.cl.h` as algorithmic specification; per-backend source directories; ~22 kernel inventory
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — Meson with `meson-python`; `_build_config.py` manifest; `auto`/`enabled`/`disabled` feature options; conditional `subdir()` delegation
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop.md) — ctypes FFI; `_verify_layouts()` pre-gate; dispatch table; `pool_dispatch_and_wait`
- [ADR-016: Test Strategy](ADR-016-test-strategy.md) — layered pytest framework; Tier 1/2/3 definitions; CPU oracle; analytical + numpy fixtures; per-kernel tolerance tables; rollback gate definitions
- [ADR-018: User-Facing API](ADR-018-user-facing-api.md) — `WorkTicket`/`LearnHandle`/`Engine` user-facing surface; parallel workstream (Phase 4-adjacent); feature-flag gated development
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback; §5 Unified Execution Model
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability
- [CPU_BACKEND.md](../CPU_BACKEND.md) — `cpu_simd.h` ISA detection; `task_<kernel_name>` function signatures; `pool_dispatch_and_wait` threading model
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — command buffer recording; single-dispatch parallelism; SPIR-V compilation
