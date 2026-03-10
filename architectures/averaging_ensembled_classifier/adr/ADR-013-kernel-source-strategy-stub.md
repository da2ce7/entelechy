# ADR-013: Kernel Source Strategy

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-007  
**Blocks:** ADR-014, ADR-016

---

## Context

The current system has a single kernel source set: OpenCL C files (`*.cl.c`) with a shared header (`kernels.cl.h`). The multi-backend refactoring introduces three kernel languages:

- **OpenCL C** — existing, well-tested.
- **GLSL (Vulkan compute shaders)** — compiled to SPIR-V.
- **C/C++ (CPU backend)** — scalar or SIMD-intrinsic implementations.

The kernel *contracts* are shared (ADR-007), but the kernel *implementations* are necessarily language-specific. The question is how to organize and maintain cross-language fidelity.

---

## Decision Required

- **(A) Shared header as reference specification.** `kernels.cl.h` (or a language-neutral successor) serves as the *specification* — parameter names, buffer contracts, algorithmic pseudocode. Each backend implements against this specification. Fidelity is verified by cross-backend test oracles (ADR-016). This is the favored direction.

- **(B) Independent kernel sets.** Each backend maintains its own kernel source tree with no formal shared specification. Fidelity is verified purely by end-to-end tests. Risk: semantic drift across backends.

---

## Tensions

- If Option A is chosen, the "reference specification" must be maintained as a first-class artifact. It cannot simply be the OpenCL source — it must be abstraction-level documentation embedded in `KernelContract` definitions or companion specification files.
- Vulkan compute shaders use GLSL with different buffer binding semantics (`layout(set=, binding=)`) and no pointer arithmetic. The specification must be abstract enough to accommodate this.
- For the CPU backend, the "kernel" is a normal C function. The specification's value is in documenting the algorithm, not the dispatch mechanics.

---

## References

- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split-stub.md) — shared contracts as the natural home for reference specifications
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — cross-backend fidelity verification
