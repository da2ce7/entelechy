# ADR-014: Build System Integration

**Status:** STUB (NARROWED — source layout resolved by ADR-013; build target structure and conditional backend enablement remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
**Blocks:** ADR-015, ADR-016

---

## Context

The current Meson build system compiles OpenCL kernels and links the Python package. The multi-backend refactoring adds two new compilation targets:

- **Vulkan:** GLSL → SPIR-V offline compilation (via `glslangValidator` or `glslc`). Source files are in `src/backends/vulkan/kernel_sources/*.comp` per ADR-013.
- **CPU:** C compilation of kernel implementations into a shared library with ISA-specific SIMD intrinsics (`-mavx2`, `-mavx512f`, `-msse2`, NEON). Source files are in `src/backends/cpu/kernel_sources/*.c` per ADR-013.

Each backend may be optional — a user might build with only OpenCL support, or only CPU support.

ADR-012 (ACCEPTED) establishes the physical directory structure: `src/shared/` (pure Python, no compilation artifacts), `src/backends/opencl/` (PyOpenCL-based, runtime kernel compilation), `src/backends/vulkan/` (SPIR-V build artifacts + Vulkan Python bindings), and `src/backends/cpu/` (compiled C shared library + Python FFI layer).

ADR-013 (ACCEPTED) establishes the kernel source layout:

- **OpenCL** — sources remain at the architecture-root `kernels/` directory. The OpenCL backend references them directly for runtime compilation; no build-time compilation step is needed. The build system must communicate the `kernels/` path to the OpenCL backend as a configuration constant.
- **Vulkan** — GLSL compute shaders in `src/backends/vulkan/kernel_sources/` must be compiled to SPIR-V at build time. The compiled `*.spv` modules are embedded in or shipped alongside the Python package.
- **CPU** — C kernel implementations in `src/backends/cpu/kernel_sources/` (with `cpu_simd.h` SIMD abstraction and `cpu_kernels.h` common declarations) must be compiled into a shared library exporting a stable C ABI.

---

## Decision Required

Two open questions remain:

1. **Conditional backend enablement.** How are backends selectively enabled in the build?

   - **(A) Conditional Meson targets.** The existing `meson.build` gains conditional sections: `if get_option('backend_vulkan')` enables SPIR-V compilation and Vulkan SDK dependency. `if get_option('backend_cpu')` enables C kernel compilation with ISA-flagged variants. The Python package discovers available backends at import time based on which artifacts are present. **Favored direction.**
   - **(B) Optional Meson subproject per backend.** Each backend is a Meson subproject (`subprojects/backend_opencl/`, etc.) that can be independently enabled. The top-level project aggregates available backends. More modular but more complex.

2. **SPIR-V artifact packaging.** How are pre-compiled SPIR-V modules shipped with the Python package? Options include embedding as package data, generating during `pip install`, or requiring a pre-built cache.

---

## Tensions

- The CPU backend's C library needs to export a C ABI that the Python interop layer (ADR-015) can call. This constrains the build to produce a shared library with a stable symbol interface — one entry point per kernel following the `task_<kernel_name>` convention established in CPU_BACKEND.md and formalized by ADR-013.
- SPIR-V compilation is a build-time step whereas OpenCL kernels are compiled at runtime. The build system must handle this asymmetry: Vulkan requires `glslc`/`glslangValidator` at build time; OpenCL requires only the `kernels/` source directory at runtime.
- `pyproject.toml` integration: the Meson build must produce artifacts that `pip install` can package correctly with optional backend dependencies.
- ADR-012's `src/shared/` is pure Python — it requires no compilation, only packaging. The build system must handle the asymmetry between `shared/` (package-only) and `backends/` (compile + package).
- The CPU backend's `cpu_simd.h` (ADR-013) uses compile-time ISA detection (`__AVX512F__`, `__AVX2__`, `__SSE2__`, `__ARM_NEON`). The build system may need to produce multiple ISA-specialized shared libraries or a single fat binary with runtime dispatch.

---

## References

- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/`, `src/backends/opencl/`, `src/backends/vulkan/`, `src/backends/cpu/` directory structure; build target delineation
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — OpenCL sources at architecture-root `kernels/`; Vulkan GLSL in `src/backends/vulkan/kernel_sources/`; CPU C in `src/backends/cpu/kernel_sources/`; build system path requirements
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop-stub.md) — C ABI requirements for CPU backend
