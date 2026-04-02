# ADR-013: Kernel Source Strategy

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-007, ADR-011, ADR-012  
**Blocks:** ADR-014, ADR-016

---

## Context

The current system has a single kernel source set: six OpenCL C implementation files (`*.cl.c`) and a shared header (`kernels.cl.h`), all residing in a top-level `kernels/` directory at the architecture root:

```
kernels/
├── kernels.cl.h
├── phase_1_act.cl.c
├── phase_2_learn_A_production.cl.c
├── phase_2_learn_B_processing.cl.c
├── phase_2_learn_C_reduction.cl.c
├── phase_2_learn_D_backprop.cl.c
└── phase_3_update.cl.c
```

The multi-backend refactoring introduces three kernel languages:

- **OpenCL C** — `*.cl.c` files compiled at runtime by the OpenCL driver via `clBuildProgram`. Scalars delivered through positional `clSetKernelArg`; tile index via host-provided `flat_tile_index`; local memory via `cl.LocalMemory()`.
- **GLSL** — `*.comp` Vulkan compute shaders, compiled offline to SPIR-V via `glslangValidator` or `glslc`. Buffers bound through descriptor sets; scalars through push constants; tile index implicit via `gl_WorkGroupID`; local memory via `shared` qualifier at compile time.
- **C/C++** — CPU backend kernel implementations, compiled into a shared library. Scalars passed as typed C function parameters; tile index via `task_index` in the thread pool; "local memory" is stack/`alloca` allocation; SIMD through a `cpu_simd.h` abstraction layer.

Three upstream ADRs constrain this decision:

**ADR-007 (ACCEPTED)** establishes the `KernelContract` frozen dataclass as the authoritative language-neutral *interface* specification for every kernel — `BufferParamSpec`, `ScalarParamSpec`, `LocalMemorySpec`, and `PlacementContract` entries fully describe the kernel's public interface. The kernel *implementations* are necessarily language-specific. The `KernelContract` specifies *what* (interface); the source files specify *how* (algorithm + language-native implementation).

**ADR-011 (ACCEPTED)** establishes the mixed CCE/BCE strategy: Strategy B (separate `kernel_name`s) for Nodes 6/7, Strategy A (unified kernel with `FLAG_problem_type` scalar) for Nodes 8/9/10. This directly impacts source organization — Strategy B kernels require distinct source implementations per backend language; Strategy A kernels contain an internal branch governed by a flag parameter (OpenCL scalar, Vulkan push constant/specialization constant, CPU function parameter).

**ADR-012 (ACCEPTED)** establishes the concrete physical directory structure: `src/shared/kernel_contracts/` contains the frozen `KernelContract` instances, and each `src/backends/<name>/kernel_bindings/` contains per-backend dispatch marshalling code. ADR-012 explicitly deferred kernel *source file* placement to this ADR: "Kernel source file organization is deferred to ADR-013."

### The three open questions

1. **Algorithmic specification.** Where is the reference implementation of each kernel's *algorithm* documented? The `KernelContract` specifies *interface*, not *computation*. The current carrier is `kernels.cl.h` — its role, evolution, and relationship to per-backend source files needs formal resolution.

2. **Source file organization.** Physical layout of `*.cl.c`, `*.comp`, and `*.c`/`*.cpp` files within ADR-012's directory structure. Where do kernel source files live relative to `src/backends/<name>/`, and how are they organized within each backend?

3. **Cross-backend algorithmic fidelity.** How is semantic equivalence between the three language implementations maintained? The `KernelContract` guarantees interface equivalence; a separate mechanism is needed to guarantee *algorithmic* equivalence.

### The current `kernels.cl.h` dual role

`kernels.cl.h` currently serves two distinct functions:

**Function A — Compile-time infrastructure.** Platform detection (`__OPENCL_VERSION__`), extension enablement (`cl_khr_fp16`), mandatory build-time symbols (CONTRACT.md Article 6: `STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`, `SIMD_WIDTH`, `C_TILE_SIZE`, `NUMERICAL_STABILITY_EPSILON`), architectural constants (Article 5: `LOCAL_MEM_BANK_PADDING`), and math configuration (`USE_FAST_MATH`). This function is inherently OpenCL-specific — GLSL uses specialization constants, C uses `#define` from the build system.

**Function B — Algorithmic specification.** The `@kernel_contract` blocks, `@param` commentary with `Tensor Shape`, `Padding Contract`, `Calculability Proof`, and `Validation Preconditions` — these document the kernel's complete interface and algorithmic intent in a language-neutral notation embedded in C-style comments. This function is language-neutral in *substance*: the mathematical operations (Softmax, ReLU, L2 norm, tiled matmul, Adam update), data access patterns (scatter-write, placement-governed partial rendering), and synchronization roles (Streamable, Partial Renderer, Global Barrier) apply identically to all backends.

ADR-007 extracted Function B's *interface* aspects into `KernelContract` frozen dataclasses. But Function B also specifies *algorithm*: the comment blocks describe the computational strategy (e.g., "A tiled matrix-vector multiplication using local memory to broadcast input slices"), the mathematical formula (e.g., "dL/dLogit = prob - target for CCE; prob - target[class] for BCE"), and the reduction structure (e.g., "two-stage intra-workgroup reduction"). These algorithmic specifications have no representation in the `KernelContract` — they are currently documented only in `kernels.cl.h`'s comment blocks and in the OpenCL C implementations themselves.

### Per-backend implementation divergence

The three backend languages are not trivially translatable. Structural differences affect how the same algorithm is expressed:

| Aspect | OpenCL C | GLSL (Vulkan) | C (CPU) |
| :--- | :--- | :--- | :--- |
| Coordinate model | `get_global_id(d)`, `get_local_id(d)`, `get_group_id(d)` | `gl_GlobalInvocationID`, `gl_LocalInvocationID`, `gl_WorkGroupID` | `task_index` + explicit indexing |
| Local memory | `__local` parameter, runtime-sized via `cl.LocalMemory()` | `shared` qualifier, compile-time sized (specialization constants) | Stack allocation, no synchronization needed (one thread per workgroup) |
| Synchronization | `barrier(CLK_LOCAL_MEM_FENCE)` | `barrier()`, `memoryBarrierShared()` | Not applicable (single-threaded per task) |
| Address spaces | `__global`, `__local`, `__constant`, `__private` | `buffer`, `shared`, `uniform` (descriptor binding semantics) | Flat address space, `const` qualifier |
| Type casting | C-style casts, `void*` type-punning for Strategy A targets | No `void*` — requires typed buffer bindings or reinterpret via `uint` | Standard C casts |
| Tile index | Host scalar `flat_tile_index` per dispatch | `gl_WorkGroupID.x` (implicit) | `task_index` function parameter |
| SIMD | Implicit (warp/wavefront) | Implicit (subgroup) | Explicit via `cpu_simd.h` intrinsics |
| Build-time constants | `-D` preprocessor flags | Specialization constants | `-D` preprocessor flags or `#define` in header |

**Strategy B (Nodes 6/7) divergence.** CCE and BCE loss kernels are structurally different implementations per language. GLSL cannot use `void*` type-punning, so Strategy B's separate-kernel approach naturally produces separate `.comp` shader files. CPU implementations use separate C functions. The divergence is genuine — no shared source code is possible between CCE and BCE loss for any single backend, nor across backends for either variant.

**Strategy A (Nodes 8/9/10) divergence.** The `FLAG_problem_type` branch in OpenCL is a runtime `if`. In Vulkan, it can be a specialization constant (compile-time branch elimination) or a push constant (runtime branch). In CPU, it's a C `if`. The branch content is the same 4-line `d_loss_d_logit` calculation, but the surrounding code (coordinate mapping, memory access patterns, SIMD vectorization) differs structurally across backends.

**Node 16 (specialized kernel) divergence.** The CPU backend document describes this kernel as using explicit SIMD intrinsics for the internal reduction, with no `barrier()` calls (one thread owns the whole workgroup). The Vulkan backend uses subgroup operations (`subgroupAdd`). The OpenCL implementation uses local memory and `barrier(CLK_LOCAL_MEM_FENCE)`. All three implement the same mathematical reduction with the same stabilization policy, but the implementation structure is entirely backend-native.

### What must be decided

Given this divergence, the system needs:
1. A canonical location for the algorithmic specification that all three backends implement.
2. A physical directory layout for kernel source files, compatible with ADR-012's `src/backends/<name>/` structure.
3. A fidelity mechanism ensuring all backends implement equivalent algorithms.

---

## Decision Drivers

1. **ADR-001 (Backend Abstraction Boundary).** The shared layer contains no backend-specific types or logic. Kernel source files are backend-specific artifacts — they contain `__kernel`, `layout(set=)`, or `#include <immintrin.h>`. Source files must live in the backend layer, not the shared layer.

2. **ADR-007 (`KernelContract` as interface specification).** The `KernelContract` specifies the kernel's public interface: parameter names, types, shapes, padding, placement. It does not and should not specify the algorithm. This ADR must address the gap between interface specification (ADR-007) and algorithmic specification (currently `kernels.cl.h`).

3. **ADR-012 (Physical directory structure).** The established structure is `src/shared/` + `src/backends/<name>/`. Kernel source files are backend artifacts. ADR-012 explicitly deferred their placement to this ADR and established the container (`backends/<name>/`) without dictating internal layout.

4. **ADR-011 (CCE/BCE strategy).** Strategy B kernels (Nodes 6/7) are inherently separate source files per variant per backend. Strategy A kernels (Nodes 8/9/10) are single source files per backend with an internal branch. The source organization must accommodate both patterns.

5. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability).** The `@kernel_contract` blocks and `@param` annotations in `kernels.cl.h` are the authoritative interface specification. Under the new model, the `KernelContract` frozen dataclasses carry this role. The source files implement, but do not redefine, the interface.

6. **CONCEPT.md §1 (Architectural Elegance Feedback).** If an optimization in one backend's source code reveals an algorithmic improvement applicable to all backends, the response is to update the algorithmic reference first, then propagate to all implementations — not to diverge silently.

7. **ADR-014 (Build system integration, pending).** The build system must locate kernel source files to compile them. A predictable directory layout with a consistent naming convention simplifies build targets. GLSL files must be found for SPIR-V compilation; C files for shared library linking; OpenCL files for runtime loading or embedding.

8. **ADR-016 (Test strategy, pending).** Cross-backend parity tests (Tier 3) verify that all backends produce equivalent outputs. The test strategy needs to know where kernel sources are located and how they map to the kernel contract inventory.

---

## Options Considered

### Option A: `kernels.cl.h` as algorithmic reference + per-backend `kernel_sources/` directories

Retain `kernels.cl.h` as the human-readable, language-neutral algorithmic reference document. It continues to house the `@kernel_contract` blocks, `@param` annotations, and algorithmic strategy descriptions (the content that ADR-007's `KernelContract` does not capture). Each backend has a `kernel_sources/` directory containing its language-native implementations. Cross-backend fidelity is verified by test oracles (ADR-016 Tier 3).

```
kernels/                           # Architecture-root: algorithmic reference
├── kernels.cl.h                   # Retained as specification document
├── phase_1_act.cl.c              # OpenCL reference implementations (historical)
├── phase_2_learn_A_production.cl.c
└── ...

src/
├── shared/
│   └── kernel_contracts/          # KernelContract frozen dataclasses (ADR-007)
└── backends/
    ├── opencl/
    │   ├── kernel_bindings/       # KernelBinding dispatch code (ADR-007)
    │   └── kernel_sources/        # *.cl.c files (may be the same as kernels/)
    ├── vulkan/
    │   ├── kernel_bindings/
    │   └── kernel_sources/        # *.comp GLSL files
    └── cpu/
        ├── kernel_bindings/
        └── kernel_sources/        # *.c files + cpu_simd.h
```

**Advantages:**
- `kernels.cl.h` is a well-established document whose notation (`@kernel_contract`, `@param`, `Tensor Shape`, `Calculability Proof`) is already referenced by every upstream ADR and CONTRACT.md itself. Retaining it preserves continuity and avoids a disruptive reformatting.
- The algorithmic descriptions (e.g., "tiled matrix-vector multiplication," "two-stage intra-workgroup reduction") are genuinely language-neutral in intent even though they use C comment syntax. A developer porting to GLSL or C can read the `.cl.h` and understand what the kernel must compute.
- Each backend's `kernel_sources/` directory is self-contained — it has everything needed to compile that backend's kernels. No cross-backend source dependencies.
- The `kernels/` directory at the architecture root becomes a specification artifact, not a compiled source artifact. Its `*.cl.c` files serve as the historical/reference OpenCL implementations.

**Disadvantages:**
- **Dual identity for `kernels.cl.h`.** The file still contains OpenCL-specific compile infrastructure (platform detection, extension enablement, macro definitions). A reader must mentally separate "specification" content (comments, `@param` blocks) from "OpenCL infrastructure" content (`#ifdef __OPENCL_VERSION__`, `#pragma OPENCL EXTENSION`). The file cannot be consumed by GLSL or C compilers.
- **Two copies of OpenCL source.** If `kernels/*.cl.c` is retained as the reference and the OpenCL backend also needs `*.cl.c` files in `backends/opencl/kernel_sources/`, the files are either duplicated (violating DRY) or the backend must reference the architecture-root `kernels/` directory (creating a cross-boundary dependency).
- **`kernels.cl.h` accumulates concerns.** As the algorithmic specification grows (new kernel variants, new backends revealing algorithmic nuances), the file becomes a monolithic specification document using C preprocessor syntax — an awkward format for language-neutral specification.

### Option B: Independent kernel sets per backend — no shared algorithmic specification

Each backend maintains its own complete kernel source tree with its own header/specification file. There is no cross-backend algorithmic reference beyond the `KernelContract` interface specification. Each backend's source files document their own algorithmic approach.

```
src/backends/
├── opencl/
│   └── kernel_sources/
│       ├── kernels.cl.h           # OpenCL-specific header + spec
│       ├── phase_1_act.cl.c
│       └── ...
├── vulkan/
│   └── kernel_sources/
│       ├── common.glsl            # Vulkan-specific shared declarations
│       ├── forward_pass.comp
│       └── ...
└── cpu/
    └── kernel_sources/
        ├── cpu_simd.h             # SIMD abstraction
        ├── forward_pass.c
        └── ...
```

**Advantages:**
- Maximum backend autonomy. Each backend organizes its sources in the most natural way for its language. GLSL can use `#include` for shared utility functions; C can use its own header hierarchy; OpenCL keeps its existing structure.
- No dual-identity problem for any file. Each source file is exactly what it appears to be: a compilable implementation in its target language.
- No cross-backend source dependencies. Each backend is fully self-contained.

**Disadvantages:**
- **No shared algorithmic specification.** The `KernelContract` specifies interface but not algorithm. Without a shared reference, each backend independently documents its computational strategy. Algorithmic intent can silently diverge — one backend might implement a numerically different Softmax formula, a different reduction tree structure, or a subtly different gradient calculation. This drift is discoverable only through test failures, not through specification review.
- **Specification duplication.** The algorithmic documentation (mathematical formulas, reduction strategy descriptions, clipping behavior, idempotency invariants) must be maintained in three places. Changes to algorithmic intent require updating three backend-specific documents.
- **Loss of the existing specification investment.** `kernels.cl.h` contains 1,487 lines of rigorously structured interface and algorithmic specification. Fragmenting this across three backends discards a single source of truth that has been validated and referenced by every upstream ADR.

### Option C: Extract algorithmic specification into a language-neutral document; `kernels.cl.h` becomes OpenCL-only infrastructure; per-backend `kernel_sources/`

Factor `kernels.cl.h` into two artifacts:

1. **`ALGORITHM.md`** (new) — A language-neutral specification document at the architecture root that describes each kernel's computational strategy, mathematical formula, reduction structure, and behavioral invariants. This is the authoritative *algorithmic* reference, complementing the `KernelContract` *interface* specification (ADR-007). It uses mathematical notation and pseudocode, not any specific programming language.

2. **`kernels.cl.h`** (reduced) — Retains only OpenCL-specific compile infrastructure: platform detection, extension enablement, build-time symbol enforcement, macro definitions, and host/C++ mode stubs. Moves into `src/backends/opencl/kernel_sources/` as an OpenCL-internal artifact.

Each backend has a `kernel_sources/` directory containing its language-native implementations, all implementing the algorithms described in `ALGORITHM.md`.

```
ALGORITHM.md                       # NEW: language-neutral algorithmic specification

src/
├── shared/
│   └── kernel_contracts/          # KernelContract frozen dataclasses (ADR-007)
└── backends/
    ├── opencl/
    │   ├── kernel_bindings/
    │   └── kernel_sources/
    │       ├── kernels.cl.h       # OpenCL compile infrastructure only
    │       ├── phase_1_act.cl.c
    │       └── ...
    ├── vulkan/
    │   ├── kernel_bindings/
    │   └── kernel_sources/
    │       ├── common.glsl        # Vulkan shared declarations
    │       ├── forward_pass.comp
    │       └── ...
    └── cpu/
        ├── kernel_bindings/
        └── kernel_sources/
            ├── cpu_simd.h
            ├── forward_pass.c
            └── ...
```

**Advantages:**
- Clean separation of three concerns: *interface* (`KernelContract`, ADR-007), *algorithm* (`ALGORITHM.md`), and *implementation* (per-backend `kernel_sources/`). Each concern has a single authoritative location.
- `ALGORITHM.md` is fully language-neutral — no C preprocessor syntax, no `__OPENCL_VERSION__` guards, no platform-specific types. It can be read by any backend implementor without mental filtering.
- `kernels.cl.h` loses its dual identity and becomes a clean OpenCL-specific artifact, living inside the OpenCL backend where it belongs per ADR-001's plan boundary.
- Each backend's `kernel_sources/` is self-contained and independently compilable.
- Cross-backend algorithmic fidelity is anchored in a single reference document that code reviewers, new backend implementors, and cross-backend parity tests (ADR-016) can consult.

**Disadvantages:**
- **Migration effort.** Extracting algorithmic specifications from `kernels.cl.h`'s 1,487 lines into a separate Markdown document is substantial. The `@param` blocks, algorithmic strategy descriptions, and behavioral invariants must be reformatted from C comment syntax to language-neutral prose/pseudocode.
- **Specification drift risk.** Two specification documents now exist for overlapping concerns: `KernelContract` specifies interface; `ALGORITHM.md` specifies algorithm. If a kernel's algorithm changes, both must be updated. The risk is mitigated by their distinct scopes (interface vs. computation) and by cross-backend parity tests (ADR-016 Tier 3).
- **New document to maintain.** `ALGORITHM.md` is a new artifact in the architecture's documentation hierarchy. Its maintenance burden is real — every algorithmic change must be reflected in it.

### Option D: `kernels.cl.h` as algorithmic reference (relocated into shared); per-backend `kernel_sources/` with OpenCL sources as canonical implementations

Keep `kernels.cl.h` as the authoritative algorithmic specification but relocate it outside any single backend — into a new `kernels/` directory under the architecture root, interpreted as a specification artifact rather than a compiled source. The existing `*.cl.c` files remain alongside it as the canonical *reference implementations*. Each backend has a `kernel_sources/` directory containing its language-native implementations.

```
kernels/                           # Architecture-root: specification + reference impl
├── kernels.cl.h                   # Algorithmic specification (OpenCL notation)
├── phase_1_act.cl.c              # Reference OpenCL implementations
├── phase_2_learn_A_production.cl.c
├── phase_2_learn_B_processing.cl.c
├── phase_2_learn_C_reduction.cl.c
├── phase_2_learn_D_backprop.cl.c
└── phase_3_update.cl.c

src/
├── shared/
│   └── kernel_contracts/          # KernelContract frozen dataclasses (ADR-007)
└── backends/
    ├── opencl/
    │   ├── kernel_bindings/
    │   └── kernel_sources/        # Symlink or build ref → kernels/*.cl.c
    ├── vulkan/
    │   ├── kernel_bindings/
    │   └── kernel_sources/        # *.comp GLSL files
    └── cpu/
        ├── kernel_bindings/
        └── kernel_sources/        # *.c files + cpu_simd.h
```

**Advantages:**
- `kernels.cl.h` remains the single, proven specification artifact. No new document type is introduced. Every upstream ADR reference to `kernels.cl.h` remains valid without amendment.
- The OpenCL implementations serve as executable reference implementations — they are compilable, testable, and human-readable. A new backend implementor reads the `.cl.h` specification and the `.cl.c` implementation side-by-side.
- The `kernels/` directory at the architecture root is already established in the current codebase. No structural relocation is needed for the reference artifact.
- The OpenCL backend can reference the architecture-root `kernels/` sources directly (via build system path), avoiding file duplication.

**Disadvantages:**
- **OpenCL notation as universal specification.** `kernels.cl.h` uses OpenCL C syntax and idioms (`__kernel void`, `__global`, `get_global_id`). While the algorithmic descriptions in comments are language-neutral, the function signatures and parameter declarations are OpenCL-specific. A Vulkan or CPU implementor must mentally translate OpenCL concepts to their own language, which is manageable for experienced developers but adds friction.
- **`kernels.cl.h`'s dual infrastructure/specification role persists.** The file still contains `#ifdef __OPENCL_VERSION__` guards, `#pragma OPENCL EXTENSION` directives, and host/C++ mode stubs. These are interleaved with the specification content.
- **Cross-boundary reference.** The OpenCL backend's `kernel_sources/` referencing `kernels/` at the architecture root creates a dependency that crosses the `src/backends/` boundary. While this is a source-file dependency (not a Python import), it complicates the build system's source enumeration.
- **Specification evolution is constrained by OpenCL syntax.** If an algorithm benefits from a description that doesn't map cleanly to OpenCL C (e.g., subgroup operations for Vulkan, SIMD intrinsics for CPU), the specification must either use generic pseudocode in comments (already the case for some descriptions) or favor OpenCL's idiom.

---

## Analysis

### Eliminating Option B

Option B is eliminated because it provides no mechanism for maintaining algorithmic equivalence across backends. The `KernelContract` specifies *what* — the interface — but not *how* — the algorithm. Without a shared algorithmic reference:

- A Vulkan implementor writing `forward_pass.comp` must reverse-engineer the algorithmic intent from the OpenCL `phase_1_act.cl.c` implementation (or independently derive it from CONCEPT.md's high-level descriptions). There is no intermediate document bridging the gap between conceptual description and implementation detail.
- Algorithmic drift is silent and insidious. One backend might compute Softmax with a different numerical stability technique, or apply gradient clipping with a different norm calculation, or use a subtly different tiled reduction structure. These differences are semantic, not syntactic — they produce different numerical results but satisfy the same `KernelContract` interface.
- Discovery of drift depends entirely on ADR-016 Tier 3 cross-backend parity tests. While these tests are necessary regardless, relying on them as the *sole* mechanism for algorithmic consistency is a testing-as-specification antipattern. The tests verify *behavior*; a specification artifact documents *intent*.

The analysis parallels ADR-007's elimination of Option B (superset parameter list): just as backend-specific parameters should not pollute the shared contract, backend-specific implementation details should not be the only carrier of algorithmic specification. Both create coupling without providing the right abstraction.

### Choosing between Options A, C, and D

Options A, C, and D all provide a shared algorithmic reference. They differ in the artifact's form and location:

| Criterion | Option A | Option C | Option D |
| :--- | :--- | :--- | :--- |
| Specification artifact | `kernels.cl.h` (dual-role, architecture root) | `ALGORITHM.md` (new, architecture root) | `kernels.cl.h` (dual-role, architecture root) |
| OpenCL infrastructure | Mixed into `kernels.cl.h` | Separated into `backends/opencl/kernel_sources/kernels.cl.h` | Mixed into `kernels.cl.h` |
| OpenCL source location | `kernels/` + `backends/opencl/kernel_sources/` (duplication risk) | `backends/opencl/kernel_sources/` | `kernels/` (referenced by OpenCL backend) |
| Language neutrality | OpenCL C notation with language-neutral comments | Fully language-neutral (Markdown + pseudocode) | OpenCL C notation with language-neutral comments |
| Existing reference stability | All ADR references to `kernels.cl.h` remain valid | References must be amended (specification split) | All ADR references to `kernels.cl.h` remain valid |
| Cross-boundary dependency | Potential duplication or cross-boundary reference | None — clean separation | Cross-boundary build reference for OpenCL |

**Option A vs. Option D.** These are nearly identical — both retain `kernels.cl.h` as the algorithmic reference in the architecture-root `kernels/` directory. Option A implies OpenCL sources may be duplicated into `backends/opencl/kernel_sources/`; Option D explicitly resolves this by having the OpenCL backend reference the architecture-root sources. Option D formalizes what Option A leaves ambiguous.

**Option C vs. Option D.** The critical trade-off:

Option C achieves the cleanest separation — three concerns, three artifact types, no dual identities. `ALGORITHM.md` would be a first-class language-neutral specification, and `kernels.cl.h` would shed its specification burden and become a pure OpenCL internal.

However, Option C introduces a substantial migration cost and, critically, creates a *new* specification artifact (`ALGORITHM.md`) that must be maintained alongside the `KernelContract` hierarchy. The algorithmic specification content currently in `kernels.cl.h` is extensive (1,487 lines of structured `@param` blocks, `@kernel_contract` annotations, and inline algorithmic descriptions). Extracting and reformatting this into Markdown would be a significant effort that produces a document with an ongoing maintenance burden — every algorithmic change must be reflected in both `ALGORITHM.md` and the implementing source files.

Option D avoids this migration cost by recognizing a pragmatic reality: `kernels.cl.h` already *is* the algorithmic specification, and it does this job well. The algorithmic descriptions in its comment blocks are language-neutral in substance even though they use C comment syntax. The OpenCL-specific compile infrastructure, while present, is cleanly delimited by `#ifdef` guards and is easily distinguished from the specification content by any developer familiar with the codebase.

The cross-boundary build reference concern (Option D's main disadvantage) is manageable: the `kernels/` directory is a specification artifact at the architecture root, not a module in the `src/` Python package. Build systems routinely reference source files outside their immediate package. ADR-014 can express this as a Meson source reference.

### The `kernels/` directory evolution

Under the chosen approach, the architecture-root `kernels/` directory evolves from its current role (sole source tree) to a new dual role:

1. **Specification artifact.** `kernels.cl.h` is the authoritative algorithmic specification. Its `@kernel_contract` blocks and `@param` annotations define the algorithm that all backends implement. Developers adding a new backend read this file to understand what each kernel must compute.

2. **OpenCL reference implementation.** The `*.cl.c` files are the canonical OpenCL implementations, directly consumed by the OpenCL backend's build/runtime compilation process. They are not duplicated into `backends/opencl/kernel_sources/`.

The `kernels/` directory is *not* part of the `src/` Python package tree. It is a specification and source artifact at the architecture level, alongside `CONCEPT.md`, `CONTRACT.md`, and the `adr/` directory. This placement is consistent with STRUCTURE.md §2 (Architectural Autonomy) — the architecture's specification artifacts live at its root.

### The three-tier specification model

This decision completes a three-tier specification hierarchy:

| Tier | Artifact | Scope | ADR |
| :--- | :--- | :--- | :--- |
| **Interface** | `KernelContract` frozen dataclasses in `src/shared/kernel_contracts/` | *What* — parameter names, types, shapes, padding, placement, calculability proofs, validation preconditions | ADR-007 |
| **Algorithm** | `kernels.cl.h` in `kernels/` | *How conceptually* — computational strategy, mathematical formula, reduction structure, behavioral invariants, numerical stability approach | This ADR |
| **Implementation** | Per-backend source files in `src/backends/<name>/kernel_sources/` | *How concretely* — language-native code implementing the algorithm using the backend's execution model | This ADR |

The `KernelContract` (Tier 1) is machine-readable and validated at plan-construction time. `kernels.cl.h` (Tier 2) is human-readable and reviewed during development. The per-backend implementations (Tier 3) are machine-compiled and verified by cross-backend parity tests (ADR-016 Tier 3).

### The `kernels.cl.h` clarification

Under this model, `kernels.cl.h`'s Function A (OpenCL compile infrastructure) and Function B (algorithmic specification) coexist intentionally. The file serves as the algorithmic specification *because* it is a compilable, executable artifact — the specification is verified every time the OpenCL backend compiles and runs it. A separate `ALGORITHM.md` (Option C) would be purely documentary and inherently at risk of specification-implementation drift.

The existing `#ifdef __OPENCL_VERSION__` / `#else` pattern already cleanly separates the two functions:
- Inside `#ifdef __OPENCL_VERSION__`: OpenCL-specific compile infrastructure (platform checks, extensions, mandatory symbol enforcement).
- Inside `#else`: Host/C++ mode stubs that make the file parseable by non-OpenCL compilers, enabling IDE analysis and host-side tests.
- Outside both guards: The `@kernel_contract` blocks, `@param` annotations, and inline algorithmic descriptions — which are in C-style comments and thus language-neutral in practice.

This structure means `kernels.cl.h` is already factored into specification (comments) and infrastructure (preprocessor). The factoring is implicit (comment vs. code) rather than physical (separate files), but it is effective and well-established.

### Cross-backend fidelity assurance

Algorithmic fidelity across backends is ensured through three complementary mechanisms:

1. **Specification review.** New backend implementations are developed with `kernels.cl.h` as the algorithmic reference. Code review verifies that each backend's source file implements the same algorithm described in the specification.

2. **`KernelContract` interface tests (ADR-016 Tier 1).** The `KernelContract` validates that all backends expose the same interface — same parameter names, same tensor shapes, same padding contracts, same placement strategies. Interface violations are caught at plan-construction time.

3. **Cross-backend parity tests (ADR-016 Tier 3).** Oracle-based tests execute the same `ExecutionPlan` on two or more backends and compare outputs within configured precision tolerances (ADR-008). This is the definitive behavioral equivalence verification — it catches algorithmic divergence that specification review might miss.

The combination of specification-level (review), interface-level (`KernelContract`), and behavioral-level (parity tests) verification provides defense-in-depth against algorithmic drift. No single mechanism is sufficient alone; together they cover specification intent, structural correctness, and numerical equivalence.

---

## Decision

**Option D: `kernels.cl.h` as algorithmic reference in the architecture-root `kernels/` directory; per-backend `kernel_sources/` directories for language-native implementations; OpenCL backend references architecture-root sources directly.**

### Kernel source directory layout

```
averaging_ensembled_classifier/
├── kernels/                                   # Specification + OpenCL reference implementation
│   ├── kernels.cl.h                           # Algorithmic specification + OpenCL compile infra
│   ├── phase_1_act.cl.c                       # OpenCL: Nodes 4, 5, 6, 7
│   ├── phase_2_learn_A_production.cl.c        # OpenCL: Nodes 8, 9, 10
│   ├── phase_2_learn_B_processing.cl.c        # OpenCL: Node 11, 13
│   ├── phase_2_learn_C_reduction.cl.c         # OpenCL: Aggregation + clip kernels
│   ├── phase_2_learn_D_backprop.cl.c          # OpenCL: Nodes 16, 17, 18, 19
│   └── phase_3_update.cl.c                    # OpenCL: Nodes 21, 24, 25
│
├── src/
│   ├── shared/
│   │   └── kernel_contracts/                  # KernelContract dataclasses (ADR-007)
│   │       ├── __init__.py
│   │       ├── phase_1_act.py
│   │       ├── phase_2_learn_A_production.py
│   │       ├── phase_2_learn_B_processing.py
│   │       ├── phase_2_learn_C_reduction.py
│   │       ├── phase_2_learn_D_backprop.py
│   │       └── phase_3_update.py
│   │
│   └── backends/
│       ├── opencl/
│       │   ├── kernel_bindings/               # OpenCL KernelBinding dispatch code
│       │   │   └── ...
│       │   └── (no kernel_sources/ — references kernels/ directly)
│       │
│       ├── vulkan/
│       │   ├── kernel_bindings/               # Vulkan KernelBinding dispatch code
│       │   └── kernel_sources/                # GLSL compute shaders
│       │       ├── common.glsl                # Shared declarations, specialization constants
│       │       ├── forward_pass.comp          # Node 4
│       │       ├── compute_hidden_mask.comp   # Node 5
│       │       ├── compute_probs_loss_cce.comp  # Node 6 (Strategy B)
│       │       ├── compute_probs_loss_bce.comp  # Node 7 (Strategy B)
│       │       ├── calculate_module_param_grads.comp  # Node 8 (Strategy A — FLAG)
│       │       └── ...
│       │
│       └── cpu/
│           ├── kernel_bindings/               # CPU KernelBinding dispatch code
│           └── kernel_sources/                # C implementations
│               ├── cpu_simd.h                 # SIMD abstraction (ISA-independent)
│               ├── cpu_kernels.h              # Common declarations, struct defs
│               ├── forward_pass.c             # Node 4
│               ├── compute_hidden_mask.c      # Node 5
│               ├── compute_probs_loss_cce.c   # Node 6 (Strategy B)
│               ├── compute_probs_loss_bce.c   # Node 7 (Strategy B)
│               ├── calculate_module_param_grads.c  # Node 8 (Strategy A — FLAG)
│               └── ...
```

### OpenCL backend: direct reference to `kernels/`

The OpenCL backend does **not** have a `kernel_sources/` subdirectory. It consumes the architecture-root `kernels/` directory directly:

- **Runtime compilation path (current).** The OpenCL backend's `context.py` reads `kernels.cl.h` and the `*.cl.c` files from the architecture-root `kernels/` directory and passes them to `cl.Program(ctx, source)` for runtime compilation. The build system (ADR-014) provides the path to the `kernels/` directory as a configuration constant.

- **Build system path specification.** ADR-014 defines how the Meson build system communicates the `kernels/` directory location to the OpenCL backend. This is a single path constant — not a file-by-file enumeration. The OpenCL backend discovers its source files at startup by scanning the directory.

This eliminates file duplication. The OpenCL `*.cl.c` files exist in exactly one location (`kernels/`), and that location also serves as the algorithmic reference.

### Vulkan backend: `kernel_sources/` with GLSL compute shaders

The Vulkan backend's `kernel_sources/` directory contains GLSL compute shaders (`*.comp`) compiled to SPIR-V at build time. The file naming convention follows the `KernelContract`'s `kernel_name` field:

| kernel_name (from KernelContract) | GLSL file | Notes |
| :--- | :--- | :--- |
| `forward_pass` | `forward_pass.comp` | Uses `shared` memory for tiling, `gl_WorkGroupID` for sample/block index |
| `compute_hidden_mask` | `compute_hidden_mask.comp` | — |
| `compute_probs_loss_cce_chunk` | `compute_probs_loss_cce.comp` | Strategy B — separate file from BCE |
| `compute_probs_loss_bce_chunk` | `compute_probs_loss_bce.comp` | Strategy B — separate file from CCE |
| `calculate_module_param_grads_chunk` | `calculate_module_param_grads.comp` | Strategy A — `FLAG_problem_type` via specialization constant |
| `backprop_error_to_hidden_chunk` | `backprop_error_to_hidden.comp` | Strategy A |
| `calculate_chunk_temp_gradients` | `calculate_chunk_temp_gradients.comp` | Strategy A |
| `clip_partial_gradients_global_norm` | `clip_partial_gradients.comp` | — |
| `gather_and_permute_grad_h` | `gather_and_permute_grad_h.comp` | Global barrier |
| `aggregate_register_reduce` | `aggregate_register_reduce.comp` | Reduction engine |
| `aggregate_local_reduce` | `aggregate_local_reduce.comp` | Reduction engine |
| `clip_intermediate_grad` | `clip_intermediate_grad.comp` | Reduction engine |
| `stabilize_and_reduce_grad_hidden_activations` | `stabilize_reduce_grad_h.comp` | Specialized — uses subgroup operations |
| `backprop_shared_weights_chunk` | `backprop_shared_weights.comp` | Streaming |
| `backprop_shared_biases_chunk` | `backprop_shared_biases.comp` | Streaming |
| `clip_shared_gradients_chunk` | `clip_shared_gradients.comp` | Streaming |
| `normalize_gradients` | `normalize_gradients.comp` | — |
| `adam_update` | `adam_update.comp` | — |
| `clamp_temperatures` | `clamp_temperatures.comp` | — |

A `common.glsl` file provides shared declarations: specialization constant definitions (mapping to CONTRACT.md Article 6 build-time symbols), common buffer layouts, and utility functions.

The Vulkan backend's GLSL shaders are compiled to SPIR-V at build time by the Meson build system (ADR-014). The compiled `*.spv` modules are embedded in or shipped alongside the Python package.

### CPU backend: `kernel_sources/` with C implementations

The CPU backend's `kernel_sources/` directory contains C implementations compiled into a shared library at build time. File organization parallels the Vulkan pattern — one file per kernel or per closely related kernel group:

| kernel_name (from KernelContract) | C file | Notes |
| :--- | :--- | :--- |
| `forward_pass` | `forward_pass.c` | SIMD vectorized via `cpu_simd.h`, tiled matmul |
| `compute_probs_loss_cce_chunk` | `compute_probs_loss_cce.c` | Strategy B |
| `compute_probs_loss_bce_chunk` | `compute_probs_loss_bce.c` | Strategy B |
| `calculate_module_param_grads_chunk` | `calculate_module_param_grads.c` | Strategy A — `problem_type` C function parameter |
| `stabilize_and_reduce_grad_hidden_activations` | `stabilize_reduce_grad_h.c` | No barriers — single-threaded per task |
| `adam_update` | `adam_update.c` | SIMD vectorized |
| (etc.) | (etc.) | |

Two shared headers:
- **`cpu_simd.h`** — Compile-time SIMD dispatch layer providing ISA-independent intrinsic abstractions (`simd_load`, `simd_fmadd`, `simd_reduce_add`) as described in CPU_BACKEND.md.
- **`cpu_kernels.h`** — Common type definitions, `TaskBatch` argument structs, and the `pool_dispatch_and_wait` function prototype. This header defines the C ABI surface that ADR-015 (Python ↔ native interop) bridges.

Each C kernel function follows the `task_<kernel_name>(void* args, uint task_index, uint thread_id)` signature expected by the thread pool, with a typed argument struct per kernel.

### Source file naming convention

A consistent naming convention across backends simplifies navigation and build system integration:

- **Kernel identity maps to file name.** The `KernelContract.kernel_name` field (e.g., `forward_pass`, `compute_probs_loss_cce_chunk`) directly determines the source file name in each backend, with backend-appropriate extensions (`.cl.c`, `.comp`, `.c`). Phase-grouped files (OpenCL's current `phase_*.cl.c` pattern) are an OpenCL-specific organizational choice; Vulkan and CPU use per-kernel files for finer-grained SPIR-V module and object file granularity.

- **Closely related kernels may share a file.** A backend may group closely related kernels (e.g., all aggregation tier kernels) into a single file where the grouping aids readability. The mapping from `kernel_name` to file is documented in each backend's `kernel_sources/README.md` or equivalent.

- **Strategy B variants are separate files.** Per ADR-011, Nodes 6 and 7 have genuinely different implementations. Each gets its own source file in every backend.

- **Strategy A kernels are single files.** Per ADR-011, Nodes 8, 9, and 10 have a parametric branch. Each is a single source file per backend with the flag-governed branch internal.

### Fidelity assurance model

Cross-backend algorithmic fidelity is established through a three-layer model:

1. **`kernels.cl.h` as algorithmic reference (specification layer).** Developers implementing kernels for any backend consult `kernels.cl.h` for algorithmic intent: mathematical formulas, computational strategies, reduction structures, numerical stability techniques, and behavioral invariants. Code review verifies that new implementations faithfully translate the specification. The specification is *always consulted*, never bypassed.

2. **`KernelContract` as interface specification (structural layer, ADR-007).** The `KernelContract` ensures all backends implement the same interface: identical parameter names, identical tensor shapes, identical padding contracts, identical placement strategies. Plan-time validation catches interface divergence before any dispatch occurs.

3. **Cross-backend parity tests (behavioral layer, ADR-016 Tier 3).** Oracle-based tests execute the same `ExecutionPlan` on multiple backends and compare outputs within ADR-008 precision tolerances. These tests catch numerical divergence that specification review might miss. The CPU backend serves as a natural oracle candidate due to its deterministic, single-threaded-per-task execution model — it is the easiest backend to debug and verify manually.

When a parity test failure occurs, the diagnostic path is:
1. Check the `KernelContract` — is the interface identical? If not, fix the contract/binding.
2. Check the source implementation against `kernels.cl.h`'s algorithmic specification — is the algorithm faithful? If not, fix the implementation.
3. If both are correct, investigate numerical precision differences (ADR-008 tolerance configuration) or hardware-specific behavior (denormals, rounding modes).

---

## Consequences

### Positive

- **Single algorithmic specification.** `kernels.cl.h` is the one-and-only location where each kernel's algorithm is described. No specification duplication across backends. New backend implementors have a single, proven reference to consult.

- **No new specification artifact.** Unlike Option C's `ALGORITHM.md`, this decision introduces no new document type. The existing `kernels.cl.h` continues to serve the role it already fills, avoiding migration effort and ongoing maintenance burden for a parallel specification.

- **No OpenCL source duplication.** The OpenCL backend references the architecture-root `kernels/` directory directly. The `*.cl.c` files exist in exactly one location. No symlinks, no file copies, no build-time duplication step.

- **Backend self-containment.** Each non-OpenCL backend's `kernel_sources/` directory is fully self-contained — it has everything needed to compile that backend's kernels without referencing another backend's sources. Only the OpenCL backend crosses the boundary to the architecture-root `kernels/` directory, which is a specification artifact, not another backend.

- **Three-tier specification hierarchy completed.** Interface (`KernelContract`), Algorithm (`kernels.cl.h`), Implementation (per-backend sources) — each tier has a clear scope, a single authoritative location, and a distinct ADR governing it. The hierarchy is navigable: start with the interface (ADR-007), understand the algorithm (this ADR), examine the implementation (backend sources).

- **Existing ADR reference stability.** Every upstream ADR referencing `kernels.cl.h` (ADR-005, ADR-007, ADR-008, ADR-011, ADR-012) remains valid without amendment. The file's location and role are unchanged — only its formal dual identity is explicitly acknowledged.

- **Build system clarity for ADR-014.** The Meson build system has a predictable source layout: OpenCL sources at `kernels/`, Vulkan GLSL sources at `src/backends/vulkan/kernel_sources/`, CPU C sources at `src/backends/cpu/kernel_sources/`. Each backend's build target has a single source directory to enumerate.

- **Test strategy clarity for ADR-016.** Cross-backend parity tests (Tier 3) need only know the `KernelContract` inventory and the plan builder's output — they do not need to locate or parse source files. Per-backend tests (Tier 2) know exactly where their backend's sources are.

### Negative

- **`kernels.cl.h` dual identity persists.** The file serves as both algorithmic specification and OpenCL compile infrastructure. A reader must distinguish specification-content (comments) from infrastructure (preprocessor directives). This is the status quo, not a new burden, and the `#ifdef` structure provides clear internal boundaries.

- **OpenCL-centric specification notation.** The algorithmic specification uses OpenCL C function signatures and parameter names as its notation. A Vulkan or CPU developer must translate `__global const STORAGE_TYPE*` to `layout(binding=N) readonly buffer` or `const float*` conceptually. This translation is straightforward for experienced systems programmers and is documented in the dispatch-mapping tables in VULKAN_BACKEND.md and CPU_BACKEND.md.

- **Cross-boundary build reference for OpenCL.** The OpenCL backend references source files outside the `src/backends/` tree. This is a build-system concern (ADR-014), not a Python import concern — the `src/shared/` import boundary invariant is unaffected. The Meson build system handles cross-directory source references natively.

- **Algorithmic evolution requires `kernels.cl.h` update.** If a new backend reveals an algorithmic improvement (e.g., a more numerically stable Softmax formulation), the improvement must be reflected in `kernels.cl.h` before propagating to all backends. Per CONCEPT.md §1 (Architectural Elegance Feedback), this is the intended workflow: formalize first, implement second. The specification update ensures all backends benefit from the improvement, not just the discovering backend.

### Migration implications

Per ADR-017's phasing:

**Phase 0 (Foundation):**
- The `kernels/` directory remains in its current location, unchanged.
- `kernels.cl.h` is formally designated as the algorithmic specification in this ADR.
- No physical file moves are needed for the OpenCL source files.

**Phase 2 (OpenCL Renderer):**
- The OpenCL backend's `context.py` is configured to load source files from the architecture-root `kernels/` directory.
- The build system (ADR-014) provides the `kernels/` path as a build constant.

**Phase 3 (CPU Backend):**
- `src/backends/cpu/kernel_sources/` is created with C implementations.
- `cpu_simd.h` and `cpu_kernels.h` are written per CPU_BACKEND.md's specifications.
- Each C kernel file is developed with `kernels.cl.h` as the algorithmic reference.
- Per-backend tests (ADR-016 Tier 2) validate each CPU kernel independently.
- Cross-backend parity tests (ADR-016 Tier 3) validate CPU vs. OpenCL numerical equivalence.

**Phase 4 (Vulkan Backend):**
- `src/backends/vulkan/kernel_sources/` is created with GLSL compute shaders.
- `common.glsl` provides shared specialization constant declarations.
- The Meson build system (ADR-014) compiles `*.comp` to `*.spv` at build time.
- Each GLSL shader is developed with `kernels.cl.h` as the algorithmic reference.
- Cross-backend parity tests (ADR-016 Tier 3) validate Vulkan vs. OpenCL/CPU numerical equivalence.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; three-tier jurisdictional model (Policy / Orchestration / Execution); no backend-specific types in the shared layer
- [ADR-005: Node 16 Opacity in the Plan](ADR-005-node-16-opacity-in-the-plan.md) — behavioral invariants specification in `kernels.cl.h` as language-neutral algorithmic reference; cross-backend reimplementation
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract` as interface specification (Tier 1); `KernelBinding` as backend-specific dispatch; abstract placement keys
- [ADR-011: CCE/BCE Strategy Delegation](ADR-011-cce-bce-strategy-delegation.md) — Strategy B (Nodes 6/7, separate kernel names) and Strategy A (Nodes 8/9/10, FLAG parameter); per-backend rendering responsibilities
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` + `src/backends/<name>/` directory structure; kernel source placement deferred to this ADR
- [ADR-014: Build System Integration](ADR-014-build-system-integration.md) — SPIR-V compilation targets, C shared library compilation, OpenCL kernel source path configuration
- [ADR-016: Test Strategy](ADR-016-test-strategy.md) — Tier 3 cross-backend parity tests as behavioral fidelity verification
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback (formalize first, implement second); §3 Modular Dumb Kernels
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 5 Architectural Constants; Article 6 Mandatory Build-Time Symbols
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — GLSL shader structure, specialization constants, push constants, descriptor set strategy
- [CPU_BACKEND.md](../CPU_BACKEND.md) — `cpu_simd.h` abstraction layer, `task_*` function signatures, `pool_dispatch_and_wait` threading model
