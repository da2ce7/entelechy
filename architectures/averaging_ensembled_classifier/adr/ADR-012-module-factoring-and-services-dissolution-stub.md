# ADR-012: Module Factoring & Services Dissolution

**Status:** STUB (NARROWED — Two-layer physical split; Services dissolved into Policy/Orchestration)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-009  
**Blocks:** ADR-013, ADR-014, ADR-015

---

## Context

The current `src/` directory is a flat namespace where shared-layer logic, backend-specific dispatch, and infrastructure services co-exist. The multi-backend refactoring requires a physical split that mirrors the jurisdictional model:

- **Shared layer** (Policy tier): `ParameterSpace`, `MemoryLayout`, `StabilizationPolicy`, `PrecisionContext`, plan builder, `KernelContract`s, `HardwareProfile`, `ModelSpec`.
- **Backend layer** (Orchestration tier): One sub-package per backend — `backends/opencl/`, `backends/vulkan/`, `backends/cpu/`. Each contains a `PlanRenderer`, `KernelBinding`s, device discovery, hardware-profile construction, and buffer management.

The existing Services layer (`cl_context_manager.py`, `compute_patterns.py`, `launcher_infra.py`) is dissolved:
- **Context management** → backend-specific (each backend's renderer creates and owns its execution context).
- **Hardware discovery** → backend-specific. ADR-006 (ACCEPTED) establishes `HardwareProfile` as a shared-layer frozen dataclass; each backend *constructs* it from its native discovery mechanism.
- **Compute patterns** → split between shared (tiling, padding helpers as pure math) and backend-specific (dispatch pattern translation).
- **Launcher infrastructure** → shared orchestration layer that calls the selected backend's renderer.

---

## Narrowed Direction

Two-layer physical split:
1. **`shared/`** — Everything that is backend-neutral.
2. **`backends/<name>/`** — Everything backend-specific.

One `orchestrator.py` at the top level assembles `shared` + selected `backend`.

---

## Remaining Decision

The concrete directory structure. A proposal:

```
src/
├── shared/
│   ├── __init__.py
│   ├── model_spec.py
│   ├── parameter_space.py
│   ├── memory_layout.py
│   ├── precision_context.py         # ADR-008
│   ├── stabilization_policy.py
│   ├── hardware_profile.py          # ADR-006 — frozen dataclass only
│   ├── plan_builder.py              # Consumes above, produces plan DAG
│   ├── plan_types.py                # Node dataclasses from ADR-002
│   ├── kernel_contracts/            # ADR-007 — KernelContract frozen dataclasses
│   │   ├── __init__.py              # Exports KernelContractBlock registry
│   │   ├── phase_1_act.py
│   │   ├── phase_2_learn_A_production.py
│   │   ├── ...
│   │   └── phase_3_update.py
│   └── buffer_handles.py            # ADR-009 — BufferHandle definitions
├── backends/
│   ├── __init__.py
│   ├── opencl/
│   │   ├── __init__.py
│   │   ├── renderer.py              # OpenCL PlanRenderer
│   │   ├── context.py               # cl.Context + queue management
│   │   ├── discovery.py             # Populates HardwareProfile from cl.device_info
│   │   ├── kernel_bindings/         # ADR-007 — OpenCL KernelBinding implementations
│   │   │   ├── __init__.py
│   │   │   ├── phase_1_act.py       # Injects flat_tile_index, marshals cl.Buffer args
│   │   │   └── ...
│   │   └── buffer_allocator.py
│   ├── vulkan/
│   │   ├── discovery.py             # Populates HardwareProfile from VkPhysicalDevice
│   │   ├── kernel_bindings/         # Push constants + descriptor sets
│   │   └── ...
│   └── cpu/
│       ├── discovery.py             # Populates HardwareProfile from ISA flags + OS queries
│       ├── kernel_bindings/         # C function arg struct marshalling
│       └── ...
└── orchestrator.py                   # Top-level assembly
```

The `shared/kernel_contracts/` directory maps one-to-one with `KernelContract` frozen dataclasses (ADR-007). Each file constructs `KernelContract` instances with `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, and `PlacementContract` entries per the decided schema. Each `backends/<name>/kernel_bindings/` directory contains the corresponding `KernelBinding` implementations that translate validated contracts to native dispatch format.

---

## Upstream Confirmations

**ADR-003 (ACCEPTED):** `ReductionTreePlan` is a shared-layer dataclass → `shared/plan_types.py`. Rendering (ping-pong dispatch, offset-list upload) is backend-specific → `backends/<name>/renderer.py`.

**ADR-004 (ACCEPTED):** `StreamingLoopNode` lives in shared plan types. Streaming loop execution is backend-specific.

**ADR-006 (ACCEPTED):** `HardwareProfile` is a shared-layer type (`shared/hardware_profile.py`); its construction is backend-specific (`backends/<name>/discovery.py`). The current `DiscoveredArchConstants` + `PrecisionContext` inheritance hierarchy is dissolved: `HardwareProfile` has no `PrecisionContext` parent; `PrecisionContext` is refactored separately per ADR-008.

**ADR-007 (ACCEPTED):** The `KernelContract` / `KernelBinding` split maps directly to `shared/kernel_contracts/` and `backends/<name>/kernel_bindings/`. `KernelContract` is a frozen dataclass carrying `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, and `PlacementContract` entries. `KernelBinding` is Orchestration-tier code that accepts a validated `KernelContract` and translates it to native dispatch format. The existing `kernel_signatures/` sub-package dissolves: each signature becomes a `KernelContract` (shared) + one `KernelBinding` per backend.

**ADR-009:** `BufferHandle` is shared. Physical allocation is backend-specific.

---

## Tensions

- Utility code that is *currently* shared but may evolve backend-specific variants (e.g., `arch_primitives.py` with `c_tile_extent`) needs a clear home. Proposed: `shared/` for the abstract primitive; backends import and specialize.

---

## References

- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md)
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md)
- [ADR-006: Hardware Profile](ADR-006-hardware-profile.md)
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md)
- [ADR-008: Precision Configuration](ADR-008-precision-configuration-stub.md)
- [ADR-009: Buffer Lifecycle](ADR-009-buffer-lifecycle-in-the-plan-model-stub.md)
