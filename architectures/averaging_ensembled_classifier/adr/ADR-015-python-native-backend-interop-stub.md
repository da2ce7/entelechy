# ADR-015: Python ↔ Native Backend Interop

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-012, ADR-014  
**Blocks:** —

---

## Context

The CPU backend (and potentially Vulkan) requires calling compiled native code from Python. The OpenCL backend uses PyOpenCL, which handles interop transparently. For other backends, an explicit interop mechanism is needed.

---

## Decision Required

- **(A) cffi.** ABI-level foreign function interface. No compilation step at import time. Requires maintaining C header declarations in Python. Well-suited for a stable, narrow C ABI.

- **(B) nanobind (or pybind11).** C++ binding layer. Richer type support, automatic numpy ↔ C++ array conversion. Requires compilation during build. Better developer ergonomics but heavier build dependency.

- **(C) ctypes + vulkan bindings.** Python's built-in ctypes for CPU; a Vulkan Python wrapper (e.g., `vulkan` PyPI package) for Vulkan. Minimal dependencies but verbose and error-prone.

- **(D) Per-backend choice.** Each backend uses whatever interop is most natural: PyOpenCL for OpenCL, cffi for CPU, a Vulkan Python wrapper for Vulkan. The `PlanRenderer` interface (Python-level) abstracts the mechanism.

---

## Tensions

- Option D is the most pragmatic but means three different interop stacks to maintain.
- The `PlanRenderer` Python interface (ADR-001) already abstracts the backend, so the interop mechanism is an implementation detail of each backend package. This argues for Option D.
- Performance matters for the CPU backend — kernel dispatch overhead should be minimal. cffi and nanobind are both low-overhead; ctypes is measurably slower for high-frequency calls.

---

## References

- [ADR-012: Module Factoring](ADR-012-module-factoring-and-services-dissolution-stub.md) — backend package structure
- [ADR-014: Build System Integration](ADR-014-build-system-integration-stub.md) — shared library output requirements
