# ADR-013: Kernel Source Strategy

**Status:** STUB (NARROWED — KernelContract as language-neutral interface specification; algorithmic reference and source organization remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-012  
**Blocks:** ADR-014, ADR-016

---

## Context

The current system has a single kernel source set: OpenCL C files (`*.cl.c`) with a shared header (`kernels.cl.h`). The multi-backend refactoring introduces three kernel languages:

- **OpenCL C** — existing, well-tested.
- **GLSL (Vulkan compute shaders)** — compiled to SPIR-V.
- **C/C++ (CPU backend)** — scalar or SIMD-intrinsic implementations.

With ADR-007 (ACCEPTED), the `KernelContract` frozen dataclass is the authoritative language-neutral *interface* specification for every kernel. `BufferParamSpec` entries define buffer shapes and padding contracts, `ScalarParamSpec` entries define scalar parameters with number types, `LocalMemorySpec` entries define shared memory requirements with size formulas, and `PlacementContract` defines the abstract placement strategy. The kernel *implementations* are necessarily language-specific.

---

## Decision Required

ADR-007's acceptance resolves the interface specification question but leaves three open questions:

1. **Algorithmic specification.** Where is the reference implementation of each kernel's *algorithm* documented? The `KernelContract` specifies the *interface* (parameters, shapes, placement), not the *computation* (forward pass matrix multiply, softmax reduction, gradient accumulation). The current carrier is `kernels.cl.h` — its commentary blocks document the algorithmic intent. A language-neutral successor or companion artifact may be needed.

2. **Source file organization.** Physical layout of `*.cl.c`, `*.comp` (GLSL), and `*.c`/`*.cpp` files within the ADR-012 directory structure. Each backend's `kernel_bindings/` directory (ADR-012) contains dispatch marshalling code, but the kernel *source files* (the actual compute implementations) also need a home.

3. **`kernels.cl.h` evolution.** Whether the shared header becomes a multi-language specification artifact or remains OpenCL-specific with the `KernelContract` assuming the cross-language specification role.

### Options

- **(A) `KernelContract` as interface spec + `kernels.cl.h` as algorithmic reference.** The `KernelContract` (ADR-007) is the machine-readable, language-neutral interface specification. `kernels.cl.h` is retained as a human-readable algorithmic reference and OpenCL compile-time constant definition file. Each backend implements kernels against the `KernelContract`; fidelity is verified by cross-backend test oracles (ADR-016). **Favored direction.**

- **(B) Independent kernel sets.** Each backend maintains its own kernel source tree with no formal shared specification beyond the `KernelContract` interface. Algorithmic documentation is duplicated per-language. Risk: semantic drift in algorithmic intent.

---

## Tensions

- The `KernelContract` specifies *what* (interface), not *how* (algorithm). Algorithmic specification still requires a human-readable reference — `kernels.cl.h` is the current carrier, but its role as a cross-language artifact needs clarification.
- Vulkan compute shaders use GLSL with different buffer binding semantics (`layout(set=, binding=)`) and no pointer arithmetic. Source organization must accommodate this without conflating interface (contract) with implementation (source).
- For the CPU backend, the "kernel" is a normal C function. The specification's value is in documenting the algorithm, not the dispatch mechanics.
- This ADR is blocked by ADR-012's concrete directory structure decision — source file locations depend on the physical module layout.

---

## References

- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` as the language-neutral interface specification
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution-stub.md) — physical directory structure for shared and backend-specific files
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — cross-backend fidelity verification
