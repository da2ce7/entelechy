# ADR-015: Python ↔ Native Backend Interop

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-014  
**Blocks:** ADR-017

---

## Context

The CPU backend (ADR-012, `src/backends/cpu/`) compiles C kernel implementations into a shared library. The Vulkan backend may also wrap native Vulkan SDK calls. Both require a Python ↔ native code Foreign Function Interface (FFI) mechanism.

ADR-012 (ACCEPTED) places each backend's Python-facing code in `src/backends/<name>/`, so the FFI binding module resides alongside the backend's `PlanRenderer` implementation. The `PlanRenderer` interface (ADR-001) abstracts the dispatch mechanism — callers in `src/shared/` never see which FFI library is in use. This means the FFI choice is an internal implementation detail of each backend, not a cross-cutting concern.

ADR-014 (pending) determines how the CPU shared library is built and what ABI it exports. The interop layer must be compatible with those build artifacts.

---

## Decision Required

- **(A) cffi (ABI mode or API mode).** Well-established, generates C extension modules or can load shared libraries at runtime. Good fit for a C ABI surface. Requires a build step in API mode.

- **(B) nanobind / pybind11.** More appropriate for C++ interfaces. Produces Python extension modules with direct CPython API integration. Higher-fidelity type mapping but heavier build dependency.

- **(C) ctypes (stdlib).** Zero external dependency. Runtime loading of shared libraries via `ctypes.cdll`. Simpler but more brittle (no compile-time type checking). Adequate if the C ABI surface is narrow.

- **(D) Per-backend choice.** Since ADR-012 confines each backend's internals behind the `PlanRenderer` interface, each backend can independently choose the most natural FFI mechanism: OpenCL uses PyOpenCL (already decided), Vulkan might use vulkan-python or cffi, CPU uses cffi or ctypes. No forced uniformity.

---

## Tensions

- Option D is the most architecturally consistent with ADR-012's isolation boundary — the FFI mechanism is an internal concern of `src/backends/<name>/`, invisible to `src/shared/`. However, it means multiple FFI libraries may appear in the dependency tree.
- The CPU ABI surface size (determined by ADR-013's kernel enumeration and ADR-014's build output) affects whether ctypes is viable or cffi's stronger typing is warranted.
- Whatever mechanism is chosen must be correctly packaged in the `pip install` workflow (ADR-014 build system integration).

---

## References

- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — backend isolation boundary; `PlanRenderer` as FFI abstraction point
- [ADR-014: Build System Integration](ADR-014-build-system-integration-stub.md) — shared library build output and ABI contract
- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — `PlanRenderer` interface
