# ADR-013: Kernel Source Strategy

**Status:** STUB (NARROWED — KernelContract as language-neutral interface specification; algorithmic reference and source organization remain open)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
**Blocks:** ADR-014, ADR-016

---

## Context

The current system has a single kernel source set: OpenCL C files (`*.cl.c`) with a shared header (`kernels.cl.h`). The multi-backend refactoring introduces three kernel languages: **OpenCL C**, **GLSL** (Vulkan compute shaders, compiled to SPIR-V), and **C/C++** (CPU backend, scalar or SIMD-intrinsic implementations).

ADR-007 (ACCEPTED) establishes the `KernelContract` frozen dataclass as the authoritative language-neutral *interface* specification for every kernel — `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, and `PlacementContract` entries fully describe the kernel's public interface. The kernel *implementations* are necessarily language-specific.

ADR-011 (ACCEPTED) establishes the mixed CCE/BCE strategy: Strategy B (separate `kernel_name`s) for Nodes 6/7, Strategy A (unified kernel with `FLAG_problem_type` scalar) for Nodes 8/9/10. This directly impacts source organization — Strategy B kernels require distinct source implementations per language; Strategy A kernels contain an internal branch.

ADR-012 (ACCEPTED) establishes the concrete physical directory structure: `shared/kernel_contracts/` contains the frozen `KernelContract` instances, and each `backends/<name>/kernel_bindings/` contains per-backend dispatch marshalling code. The kernel *source files* (the actual compute implementations) also need a home within this structure.

---

## Decision Required

Three open questions remain:

1. **Algorithmic specification.** Where is the reference implementation of each kernel's *algorithm* documented? The `KernelContract` specifies *interface*, not *computation*. The current carrier is `kernels.cl.h` — a language-neutral successor or companion artifact may be needed.

2. **Source file organization.** Physical layout of `*.cl.c`, `*.comp` (GLSL), and `*.c`/`*.cpp` files within ADR-012's `backends/<name>/` tree. Options include placing source files alongside `kernel_bindings/`, in a sibling `kernel_sources/` directory, or in a top-level `kernels/` directory outside `src/`.

3. **`kernels.cl.h` evolution.** Whether the shared header becomes a multi-language specification artifact or remains OpenCL-specific, with the `KernelContract` assuming the cross-language specification role.

### Options

- **(A) `KernelContract` as interface spec + `kernels.cl.h` as algorithmic reference.** `kernels.cl.h` is retained as a human-readable algorithmic reference and OpenCL compile-time constant definition file. Each backend implements kernels against the `KernelContract`; cross-backend fidelity is verified by test oracles (ADR-016). **Favored direction.**

- **(B) Independent kernel sets.** Each backend maintains its own kernel source tree with no formal shared algorithmic specification beyond the `KernelContract` interface. Risk: semantic drift in algorithmic intent.

---

## Tensions

- The `KernelContract` specifies *what* (interface), not *how* (algorithm). Algorithmic specification still requires a human-readable reference.
- Vulkan compute shaders use GLSL with different buffer binding semantics (`layout(set=, binding=)`) and no pointer arithmetic.
- For Strategy A kernels (Nodes 8/9/10), GLSL's handling of the FLAG parameter differs from OpenCL C — specialization constants (compile-time) vs. push constants (runtime). This is a binding concern (ADR-007, ADR-011), but the source file must accommodate it.
- For Strategy B kernels (Nodes 6/7), source implementations are genuinely different per language (Softmax vs. Sigmoid, `int*` vs. `SCALAR_TYPE*` targets). These share no source code.
- The existing `kernels/` directory at the architecture root (containing `kernels.cl.h` and Phase `*.cl.c` files) must be reconciled with the new `backends/<name>/` structure.

---

## References

- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` as language-neutral interface specification
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — mixed Strategy B (Nodes 6/7) / Strategy A (Nodes 8/9/10); backend rendering responsibilities for FLAG delivery
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `shared/kernel_contracts/` and `backends/<name>/kernel_bindings/` directory structure
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — cross-backend fidelity verification
