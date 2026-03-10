# ADR-014: Build System Integration

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-013  
**Blocks:** ADR-015, ADR-016

---

## Context

The current Meson build system compiles OpenCL kernels and links the Python package. The multi-backend refactoring adds two new compilation targets:

- **Vulkan:** GLSL → SPIR-V compilation (via `glslangValidator` or `glslc`), Vulkan SDK dependency.
- **CPU:** C/C++ compilation of kernel implementations into a shared library, potentially with SIMD intrinsics (`-mavx2`, `-mavx512f`).

Each backend may be optional — a user might build with only OpenCL support, or only CPU support.

ADR-012 (ACCEPTED) establishes the physical directory structure: `src/shared/` (pure Python, no compilation artifacts), `src/backends/opencl/` (PyOpenCL-based, runtime kernel compilation), `src/backends/vulkan/` (SPIR-V build artifacts + Vulkan Python bindings), and `src/backends/cpu/` (compiled C shared library + Python FFI layer). The build system must produce per-backend artifacts aligned with this structure.

ADR-013 (pending) will determine where kernel source files live within the `backends/<name>/` tree. The build system must compile those sources into the correct output format per backend.

---

## Decision Required

- **(A) Conditional backend targets.** The existing `meson.build` gains conditional sections: `if get_option('backend_vulkan')` enables SPIR-V compilation and Vulkan library linking. `if get_option('backend_cpu')` enables C kernel compilation. The Python package discovers available backends at import time based on which artifacts are present.

- **(B) Optional Meson subproject per backend.** Each backend is a Meson subproject (`subprojects/backend_opencl/`, etc.) that can be independently enabled. The top-level project aggregates available backends. More modular but more complex.

---

## Tensions

- The CPU backend's C library needs to export a C ABI that the Python interop layer (ADR-015) can call. This constrains the build to produce a shared library with a stable symbol interface.
- SPIR-V compilation is a build-time step (offline compilation) whereas OpenCL kernels are currently compiled at runtime. This asymmetry affects the build system but not the plan model.
- `pyproject.toml` integration: the Meson build must produce artifacts that `pip install` can package correctly with optional backend dependencies.
- ADR-012's `src/shared/` is pure Python — it requires no compilation, only packaging. The build system must handle the asymmetry between `shared/` (package-only) and `backends/` (compile + package).

---

## References

- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/`, `src/backends/opencl/`, `src/backends/vulkan/`, `src/backends/cpu/` directory structure; build target delineation
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy-stub.md) — kernel source file locations within `backends/<name>/`
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop-stub.md) — C ABI requirements for CPU backend
