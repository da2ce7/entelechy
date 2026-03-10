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

With ADR-011 (ACCEPTED), the CCE/BCE divergence is expressed through a mixed strategy:
- **Strategy B (Nodes 6/7):** Separate kernel names — `compute_probs_loss_cce_chunk` and `compute_probs_loss_bce_chunk` — with distinct source implementations per backend language. Each backend must provide separate source files (or separate entry points) for these two kernels.
- **Strategy A (Nodes 8/9/10):** Unified kernel names with a `src_scalar_FLAG_problem_type` scalar. A single source file per backend language implements the branch internally.

This mixed strategy directly impacts kernel source organization: Strategy B kernels have a one-kernel-name-to-one-source-implementation mapping per language; Strategy A kernels are language-specific in how the FLAG-based branch is realized (OpenCL C `if/else`, GLSL `if/else` with push constant or specialization constant, C function with branch or separate function pointers).

---

## Decision Required

ADR-007's acceptance resolves the interface specification question and ADR-011's acceptance resolves the CCE/BCE delegation model, but three open questions remain:

1. **Algorithmic specification.** Where is the reference implementation of each kernel's *algorithm* documented? The `KernelContract` specifies the *interface* (parameters, shapes, placement), not the *computation* (forward pass matrix multiply, softmax reduction, gradient accumulation). The current carrier is `kernels.cl.h` — its commentary blocks document the algorithmic intent. A language-neutral successor or companion artifact may be needed.

2. **Source file organization.** Physical layout of `*.cl.c`, `*.comp` (GLSL), and `*.c`/`*.cpp` files within the ADR-012 directory structure. Each backend's `kernel_bindings/` directory (ADR-012) contains dispatch marshalling code, but the kernel *source files* (the actual compute implementations) also need a home. For Strategy B kernels (Nodes 6/7), each language has two separate source implementations; for Strategy A kernels (Nodes 8/9/10), each language has one unified source with an internal branch. The directory layout must make this distinction navigable.

3. **`kernels.cl.h` evolution.** Whether the shared header becomes a multi-language specification artifact or remains OpenCL-specific with the `KernelContract` assuming the cross-language specification role. The `Kernel Bifurcation` annotations already present in `kernels.cl.h` for Nodes 6/7 (referencing CONCEPT.md Principle 3(B)) are complemented by ADR-011's formal decision — the header should reference the ADR for the complete rationale.

### Options

- **(A) `KernelContract` as interface spec + `kernels.cl.h` as algorithmic reference.** The `KernelContract` (ADR-007) is the machine-readable, language-neutral interface specification. `kernels.cl.h` is retained as a human-readable algorithmic reference and OpenCL compile-time constant definition file. Each backend implements kernels against the `KernelContract`; fidelity is verified by cross-backend test oracles (ADR-016). For Strategy B kernels (Nodes 6/7), the algorithmic reference documents both variants; for Strategy A kernels (Nodes 8/9/10), it documents the unified kernel with its FLAG branch. **Favored direction.**

- **(B) Independent kernel sets.** Each backend maintains its own kernel source tree with no formal shared specification beyond the `KernelContract` interface. Algorithmic documentation is duplicated per-language. Risk: semantic drift in algorithmic intent — particularly for Strategy A kernels where the FLAG branch semantics must be identical across languages.

---

## Tensions

- The `KernelContract` specifies *what* (interface), not *how* (algorithm). Algorithmic specification still requires a human-readable reference — `kernels.cl.h` is the current carrier, but its role as a cross-language artifact needs clarification.
- Vulkan compute shaders use GLSL with different buffer binding semantics (`layout(set=, binding=)`) and no pointer arithmetic. Source organization must accommodate this without conflating interface (contract) with implementation (source).
- For the CPU backend, the "kernel" is a normal C function. The specification's value is in documenting the algorithm, not the dispatch mechanics.
- For Strategy A kernels (Nodes 8/9/10), GLSL's handling of the FLAG parameter may differ from OpenCL C — a Vulkan backend could choose specialization constants (compile-time branching, potentially compiled out) or push constants (runtime branching). This is a binding concern (ADR-007, ADR-011), but the source file must be written to accommodate the chosen mechanism.
- For Strategy B kernels (Nodes 6/7), the structural differences (Softmax vs. Sigmoid, `int*` vs. `SCALAR_TYPE*` targets, scatter-write vs. partial-render for loss) mean each backend language has genuinely different source implementations. These are not template variations — they share no source code.
- This ADR is blocked by ADR-012's concrete directory structure decision — source file locations depend on the physical module layout.

---

## References

- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` as the language-neutral interface specification
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — mixed Strategy B (Nodes 6/7) / Strategy A (Nodes 8/9/10); backend rendering responsibilities for FLAG delivery
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution-stub.md) — physical directory structure for shared and backend-specific files
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — cross-backend fidelity verification
