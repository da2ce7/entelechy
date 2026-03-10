# ADR-014: Build System Integration

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-012, ADR-013  
**Blocks:** ADR-015, ADR-016

---

## Context

The current Meson build system compiles the architecture as a single subproject with two artifact classes: Python source files (installed via `py.install_sources`) and OpenCL kernel files (installed as package data via `install_data`). The OpenCL backend requires no build-time compilation of kernels — they are compiled at runtime by the OpenCL driver via `clBuildProgram`. This simplicity is a consequence of the single-backend architecture.

The multi-backend refactoring introduces two new backends with build-time compilation requirements:

- **Vulkan:** GLSL compute shaders (`*.comp`) in `src/backends/vulkan/kernel_sources/` must be compiled to SPIR-V (`*.spv`) at build time via `glslc` (from the Vulkan SDK) or `glslangValidator`. The compiled modules are shipped as package data alongside the Python package. SPIR-V compilation is a mandatory pre-deployment step — there is no runtime GLSL compilation path.

- **CPU:** C kernel implementations in `src/backends/cpu/kernel_sources/` must be compiled into a shared library (`.so` / `.dylib` / `.dll`) exporting a stable C ABI. The shared library uses ISA-specific SIMD intrinsics (`-mavx512f`, `-mavx2`, `-msse2`, NEON flags) selected at compile time via `cpu_simd.h`'s `#ifdef` cascade (CPU_BACKEND.md). The Python backend loads this library at runtime via FFI (ADR-015).

The **OpenCL** backend requires no build-time kernel compilation. Its sources remain at the architecture-root `kernels/` directory (ADR-013). The build system must communicate this path to the OpenCL backend as a configuration constant, and install the `kernels/` contents as package data so they are discoverable at runtime.

### The current build system

The root `meson.build` is a conductor manifest that delegates to architecture subprojects:

```meson
project('ml_engine_framework', version: '0.1.0',
        default_options: ['warning_level=2'],
        subproject_dir: 'architectures')
subproject('averaging_ensembled_classifier')
```

The architecture's `meson.build` declares `'c'` as its language (for OpenCL kernel syntax checking), finds the Python installation, enumerates all Python source files and OpenCL kernel files, and installs them:

```meson
project('averaging_ensembled_classifier-sub', 'c', version: '0.1.0')
py = import('python').find_installation()
py.install_sources(py_sources, subdir: 'averaging_ensembled_classifier')
install_data(cl_kernels, install_dir: py.get_install_dir() / '...' / 'kernels')
```

The architecture's `pyproject.toml` uses `meson-python` as the build backend, enabling `pip install -e .` to invoke Meson automatically.

### Upstream ADR constraints

**ADR-012 (ACCEPTED)** establishes the physical directory structure: `src/shared/` (pure Python, no compilation artifacts), `src/backends/opencl/` (PyOpenCL-based, runtime kernel compilation), `src/backends/vulkan/` (SPIR-V build artifacts + Vulkan Python bindings), and `src/backends/cpu/` (compiled C shared library + Python FFI layer). The build system must handle the asymmetry between `shared/` (package-only) and `backends/` (mixed: package-only for OpenCL, compile + package for Vulkan and CPU).

**ADR-013 (ACCEPTED)** establishes the kernel source layout:

- **OpenCL** — sources remain at the architecture-root `kernels/` directory. The OpenCL backend references them directly for runtime compilation; no build-time compilation step is needed. The build system must install the `kernels/` contents and provide the path as a discoverable constant.
- **Vulkan** — GLSL compute shaders in `src/backends/vulkan/kernel_sources/` must be compiled to SPIR-V at build time. The compiled `*.spv` modules are installed alongside the Python package.
- **CPU** — C implementations in `src/backends/cpu/kernel_sources/` (with `cpu_simd.h` SIMD abstraction and `cpu_kernels.h` common declarations) must be compiled into a shared library exporting the `task_<kernel_name>` C ABI surface.

### The three build-time compilation targets

| Backend | Source location | Build action | Output artifact | Installation |
| :--- | :--- | :--- | :--- | :--- |
| OpenCL | `kernels/*.cl.c`, `kernels/kernels.cl.h` | None (runtime compilation) | None | Install as package data |
| Vulkan | `src/backends/vulkan/kernel_sources/*.comp` | `glslc` → SPIR-V | `*.spv` modules | Install as package data alongside backend Python |
| CPU | `src/backends/cpu/kernel_sources/*.c` | C compiler with ISA flags | `libcpu_kernels.so` | Install as extension module alongside backend Python |

### The open questions

Two design questions must be resolved:

1. **Conditional backend enablement.** How are backends selectively enabled in the build? Not every build environment has the Vulkan SDK, and the CPU backend's SIMD intrinsics require a C compiler with ISA support. The build must degrade gracefully when external dependencies are absent.

2. **SPIR-V artifact packaging.** How are pre-compiled SPIR-V modules shipped with the Python package? They must be discoverable by the Vulkan backend's `context.py` at runtime, and must survive both editable installs (`pip install -e .`) and standard installs (`pip install .`).

---

## Decision Drivers

1. **ADR-012 (Physical directory structure).** The build targets mirror the `src/shared/` + `src/backends/<name>/` split. `shared/` is pure Python. Each backend under `backends/` may have distinct compilation requirements. The build system must reflect this structure without introducing cross-backend build dependencies.

2. **ADR-013 (Kernel source locations).** The build system's source enumeration must align with ADR-013's concrete file layout: OpenCL at `kernels/`, Vulkan at `src/backends/vulkan/kernel_sources/`, CPU at `src/backends/cpu/kernel_sources/`. The OpenCL backend's cross-boundary reference to the architecture-root `kernels/` directory must be expressed cleanly in Meson.

3. **STRUCTURE.md §2 (Architectural Autonomy).** The architecture is a self-contained subproject. Its build system must function both standalone (`meson setup builddir` from the architecture root) and as a subproject of the framework conductor. External tool dependencies (Vulkan SDK, ISA-capable C compiler) are the architecture's concern, not the framework's.

4. **STRUCTURE.md §4 (Build-Time System Guarantees).** A successful build must guarantee that all enabled backends have valid, deployable artifacts. If the Vulkan backend is enabled, every `*.comp` file must compile to valid SPIR-V. If the CPU backend is enabled, the shared library must link and export the expected symbols.

5. **CONCEPT.md §1 (Architectural Elegance Feedback).** If a new backend introduces compilation requirements that cannot be expressed within the existing build target model, the build system must evolve its abstractions — not introduce ad-hoc shell scripts or post-install hooks.

6. **ADR-001 (Backend Abstraction Boundary).** The build system must not conflate backend artifacts. Each backend's build products are installed into that backend's package namespace. The `shared/` package never contains compiled artifacts.

7. **`pyproject.toml` integration.** The Meson build must produce artifacts that `meson-python` can package correctly for `pip install`. Optional backend dependencies (Vulkan SDK, C compiler) must be expressible as optional build requirements.

8. **ADR-015 (Python ↔ Native Interop, ACCEPTED).** The CPU backend's shared library must export a stable C ABI that the FFI layer can load. The build system defines the library name, symbol visibility, and installation path that ADR-015's ctypes interop mechanism depends on.

9. **ADR-016 (Test Strategy, pending).** Conditional backend enablement determines which test tiers can execute. The build system must produce a discoverable manifest of enabled backends so the test framework can skip unavailable tiers.

---

## Options Considered

### Option A: Meson build options with conditional target blocks

The architecture's `meson.build` gains Meson build options (`-Dbackend_vulkan=enabled`, `-Dbackend_cpu=enabled`) using Meson's `feature` option type with `auto` as the default. When set to `auto`, Meson probes for the required external dependency (Vulkan SDK / C compiler ISA support) and enables the backend if available. When `enabled`, the backend is mandatory and the build fails if dependencies are missing. When `disabled`, the backend is unconditionally skipped.

Each backend's compilation targets are guarded by `if backend_option.allowed()` conditionals within the single `meson.build`:

```meson
# meson_options.txt
option('backend_vulkan', type: 'feature', value: 'auto',
       description: 'Build Vulkan SPIR-V compute shaders')
option('backend_cpu', type: 'feature', value: 'auto',
       description: 'Build CPU SIMD kernel shared library')
```

```meson
# In meson.build:
backend_vulkan = get_option('backend_vulkan')
backend_cpu = get_option('backend_cpu')

# Probe dependencies
if backend_vulkan.allowed()
  glslc = find_program('glslc', required: backend_vulkan)
  if glslc.found()
    # ... compile *.comp → *.spv
  endif
endif

if backend_cpu.allowed()
  cc = meson.get_compiler('c')
  # ... compile CPU kernel sources → shared library
endif
```

The OpenCL backend is always enabled (no build-time compilation, only Python + package data).

**Advantages:**
- Uses Meson's native build option system with `auto`/`enabled`/`disabled` semantics. The `auto` default means a standard `meson setup` probes the environment and enables what it can — zero-configuration for most users.
- Single `meson.build` keeps the build logic co-located with the source enumeration. The conditional blocks are visually parallel, making the per-backend build requirements easy to compare.
- `meson-python` handles the resulting artifacts naturally — Python sources go through `py.install_sources`, compiled artifacts through `install_data` (SPIR-V) and `shared_library` (CPU).
- Conditional enablement directly solves the CI/CD problem: environments without Vulkan SDK build with `-Dbackend_vulkan=disabled`, and tests skip Vulkan-dependent tiers.

**Disadvantages:**
- A single `meson.build` grows in complexity as backends are added. Three backends' worth of source enumeration, dependency probing, and install rules in one file could exceed ~200 lines.
- Backend build logic is not physically co-located with the backend's source code. The Vulkan SPIR-V compilation rules live in the architecture's `meson.build`, not in `src/backends/vulkan/`.

### Option B: Per-backend Meson subprojects

Each backend is a Meson subproject with its own `meson.build` under a `subprojects/` directory (or under `src/backends/<name>/`). The architecture's top-level `meson.build` conditionally includes each backend subproject.

```
meson.build                              # Architecture conductor
src/backends/opencl/meson.build          # OpenCL: install Python + package data
src/backends/vulkan/meson.build          # Vulkan: SPIR-V compilation + install
src/backends/cpu/meson.build             # CPU: C shared library + install
```

**Advantages:**
- Maximum modularity. Each backend's build logic is co-located with its source code.
- Adding a new backend means adding a new `meson.build` in the backend's directory, with no changes to the architecture's top-level manifest.
- Backends can declare their own dependencies (e.g., Vulkan SDK) without polluting the architecture's dependency set.

**Disadvantages:**
- Meson's subproject mechanism is designed for external dependencies, not internal package-level modularization. Using `subdir()` instead of `subproject()` would be more appropriate, but `subdir()` shares the same project context and cannot conditionally include/exclude cleanly.
- `meson-python` expects a single `meson.build` at the project root to drive installation. Splitting the build across multiple files is supported (via `subdir()`) but requires careful coordination of install prefixes to ensure all artifacts land in the correct package namespace.
- Three `meson.build` files plus the top-level one create more places to look when debugging build issues. The indirection adds cognitive overhead for a project with only three backends.

### Option C: Top-level `meson.build` with `subdir()` delegation

A hybrid: the architecture's `meson.build` remains the single entry point but delegates backend-specific logic to child `meson.build` files via Meson's `subdir()` directive. Each backend directory contains a `meson.build` that is conditionally included:

```meson
# Top-level meson.build
subdir('src/shared')       # Pure Python install
subdir('src/backends/opencl')  # Always: Python + kernels/ package data

if backend_vulkan_opt.allowed() and glslc.found()
  subdir('src/backends/vulkan')
endif

if backend_cpu_opt.allowed()
  subdir('src/backends/cpu')
endif
```

**Advantages:**
- Build logic is co-located with source code (each backend has its own `meson.build`) while maintaining a single project context.
- The top-level `meson.build` reads as a concise build plan: "install shared, always install OpenCL, conditionally install Vulkan and CPU."
- `subdir()` shares the project's Python installation finder, compiler detection, and option state — no redundant setup per backend.
- Conditional inclusion via `subdir()` is idiomatic Meson for optional components within a single project.

**Disadvantages:**
- `subdir()` is unconditional in Meson — the conditional must wrap the `subdir()` call, not live inside the subdirectory's `meson.build`. This means the top-level file must perform dependency probing before delegating.
- The child `meson.build` files execute in the parent project's context. Variable names must be coordinated to avoid collisions (e.g., each backend's source list must use distinct variable names).

---

## Analysis

### Eliminating Option B

Option B is eliminated because Meson's subproject mechanism is architecturally mismatched for this use case. Meson subprojects (`subproject()`) are designed for external dependencies with independent version constraints and build isolation. The backends are not external — they are integral components of a single Python package, sharing the same `pyproject.toml`, the same `py.install_sources` namespace, and the same install prefix.

Using `subproject()` would require each backend to declare its own `project()` call, independently find the Python installation, and coordinate install paths to land artifacts in the correct package namespace (`averaging_ensembled_classifier/backends/<name>/`). This coordination overhead contradicts ADR-012's decision that the backends are sub-packages of a single architecture, not autonomous projects.

The analogy to ADR-012's elimination of Option B (flat namespace without `shared/` boundary) applies: Option B here distributes build authority without a clear coordination mechanism, making it harder to enforce invariants that span backends (e.g., "all enabled backends must install into the same package namespace").

### Choosing between Options A and C

Options A and C differ in a single dimension: whether backend build logic lives in the top-level `meson.build` (Option A) or in per-backend `meson.build` files included via `subdir()` (Option C).

The key consideration is the anticipated complexity per backend:

- **OpenCL:** ~5 lines (enumerate kernel files, `install_data`).
- **Vulkan:** ~15–25 lines (enumerate `*.comp` sources, find `glslc`, generate `*.spv` via `custom_target` per shader, `install_data` for SPIR-V modules).
- **CPU:** ~20–30 lines (enumerate `*.c` sources, detect ISA flags, build `shared_library` with appropriate `-m` flags, install alongside Python package).

The total backend-specific build logic is ~40–60 lines. Combined with shared Python installation (~20 lines) and option/dependency probing (~15 lines), the full `meson.build` would be ~75–95 lines under Option A. This is well within the manageable range for a single file.

However, Option C offers a structural advantage that aligns with ADR-012's principle of co-location: the CPU backend's `meson.build` lives next to `cpu_simd.h` and `cpu_kernels.h`, the Vulkan backend's `meson.build` lives next to `*.comp` shaders. When a developer adds a new kernel source file, they edit the `meson.build` in the same directory — not a top-level file that also handles two other backends.

Furthermore, Option C naturally scales. If a fourth backend is introduced (e.g., Metal Shading Language for Apple platforms), it adds a new `src/backends/metal/meson.build` and a single `subdir('src/backends/metal')` conditional in the top-level file. Under Option A, the top-level file grows by another 20–30 lines of interleaved source enumeration, dependency probing, and install rules.

The decisive factor is ADR-012's structural parallel: just as the Python package structure separates `shared/` from `backends/<name>/`, the build system should separate shared build logic from per-backend build logic. Option C achieves this with `subdir()` delegation, mirroring the package structure in the build system.

### OpenCL kernel path configuration

ADR-013 requires the build system to communicate the `kernels/` directory path to the OpenCL backend. Three mechanisms are available:

1. **Build-time generated Python module.** The Meson build generates a small Python file (e.g., `_build_config.py`) containing the absolute or package-relative path to the kernel directory. The OpenCL backend's `context.py` imports this module.

2. **Package data with `importlib.resources`.** The kernel files are installed as package data, and the backend discovers them at runtime via `importlib.resources.files('averaging_ensembled_classifier') / 'kernels'`. No build-time path injection needed.

3. **Environment variable.** A `KERNEL_SOURCE_DIR` environment variable, set by the build system or the user. Fragile and non-portable.

Option 2 is the most Pythonic and works correctly for both editable and standard installs. The `meson-python` backend handles package data installation natively, and `importlib.resources` (Python 3.9+, well within the `>=3.11` requirement) provides a reliable discovery mechanism. The build system simply ensures the kernel files are installed in the correct package subdirectory.

### ISA dispatch strategy for the CPU backend

The CPU backend's `cpu_simd.h` uses compile-time ISA detection (`#ifdef __AVX512F__`, etc.). This means the compiled shared library targets a single ISA tier. Two strategies exist:

1. **Single library, host ISA.** Compile once with the build host's native ISA flags (e.g., `-march=native`). The library runs optimally on the build host but may fail on machines with lesser ISA support. Simple and sufficient for development and single-target deployment.

2. **Multiple libraries, runtime selection.** Compile multiple variants (`libcpu_kernels_avx512.so`, `libcpu_kernels_avx2.so`, `libcpu_kernels_sse2.so`) with different `-m` flags. The Python FFI layer (ADR-015) probes CPU features at runtime and loads the appropriate library. Maximizes portability at the cost of build complexity and disk space.

Strategy 1 is chosen for the initial implementation. The CPU backend is primarily a development and debugging tool (ADR-016 identifies it as the natural oracle for cross-backend parity tests) and a deployment option for environments without GPU hardware. The `-march=native` default is overridable via the Meson build option `cpu_isa_flags` for cross-compilation or specific ISA targeting. Strategy 2 is a documented future extension that can be implemented by adding multiple `shared_library` targets when distribution requirements demand it.

---

## Decision

**Option C: Top-level `meson.build` with `subdir()` delegation to per-backend build files, Meson `feature` options for conditional enablement, `importlib.resources` for OpenCL kernel path discovery, and single-ISA CPU library with configurable flags.**

### Build option definitions

A `meson.options` file at the architecture root defines the build options:

```meson
# meson.options

option('backend_vulkan', type: 'feature', value: 'auto',
       description: 'Build Vulkan SPIR-V compute shaders (requires glslc)')

option('backend_cpu', type: 'feature', value: 'auto',
       description: 'Build CPU SIMD kernel shared library')

option('cpu_isa_flags', type: 'array', value: [],
       description: 'C compiler ISA flags for CPU backend (e.g., [\'-mavx2\']. Empty = -march=native)')
```

The `backend_vulkan` and `backend_cpu` options use Meson's `feature` type, which accepts three values:
- **`auto` (default):** Probe for dependencies; enable if available, skip if not. Zero-configuration for most users.
- **`enabled`:** Mandatory; the build fails if dependencies are missing.
- **`disabled`:** Unconditionally skip.

The OpenCL backend has no build option — it is always enabled because it requires no build-time compilation. Its Python sources and kernel data files are always installed.

### Top-level `meson.build` structure

The architecture's `meson.build` is the single entry point. It performs dependency probing, then delegates to per-backend `meson.build` files via `subdir()`:

```meson
project('averaging_ensembled_classifier-sub', 'c', version: '0.1.0')

py = import('python').find_installation()

# ── Build option retrieval ──────────────────────────────────────────
backend_vulkan_opt = get_option('backend_vulkan')
backend_cpu_opt    = get_option('backend_cpu')

# ── Dependency probing ──────────────────────────────────────────────
# Vulkan: requires glslc (Vulkan SDK ships it)
glslc = find_program('glslc', required: backend_vulkan_opt)

# CPU: requires the already-declared C compiler (project language 'c').
# ISA capability is verified at compile time via cpu_simd.h's #ifdef cascade.

# ── Shared layer (pure Python — no compilation) ────────────────────
subdir('src/shared')

# ── OpenCL backend (always enabled — runtime compilation only) ─────
subdir('src/backends/opencl')

# ── OpenCL kernel sources (architecture-root specification artifact) ─
cl_kernel_data = files(
  'kernels/kernels.cl.h',
  'kernels/phase_1_act.cl.c',
  'kernels/phase_2_learn_A_production.cl.c',
  'kernels/phase_2_learn_B_processing.cl.c',
  'kernels/phase_2_learn_C_reduction.cl.c',
  'kernels/phase_2_learn_D_backprop.cl.c',
  'kernels/phase_3_update.cl.c',
)
install_data(
  cl_kernel_data,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'kernels',
)

# ── Vulkan backend (conditional) ───────────────────────────────────
if glslc.found()
  subdir('src/backends/vulkan')
endif

# ── CPU backend (conditional) ──────────────────────────────────────
if backend_cpu_opt.allowed()
  subdir('src/backends/cpu')
endif

# ── Build manifest (enabled backends discovery for runtime + tests) ─
build_config_data = configuration_data()
build_config_data.set('BACKEND_OPENCL', 'True')
build_config_data.set('BACKEND_VULKAN', glslc.found() ? 'True' : 'False')
build_config_data.set('BACKEND_CPU', backend_cpu_opt.allowed() ? 'True' : 'False')

configure_file(
  input: 'src/_build_config.py.in',
  output: '_build_config.py',
  configuration: build_config_data,
  install: true,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier',
)

# ── Top-level package init ─────────────────────────────────────────
py.install_sources(
  files('src/__init__.py', 'src/orchestrator.py'),
  subdir: 'averaging_ensembled_classifier',
)
```

### Per-backend `meson.build` files

**`src/shared/meson.build`** — Pure Python installation:

```meson
py.install_sources(
  files(
    '__init__.py',
    'model_spec.py',
    'parameter_space.py',
    'memory_layout.py',
    'precision_config.py',
    'hardware_profile.py',
    'stabilization_policy.py',
    'plan_builder.py',
    'plan_types.py',
    'buffer_lifecycle.py',
    'retrieval_future.py',
    'problem_type_strategy.py',
    'workload_primitives.py',
  ),
  subdir: 'averaging_ensembled_classifier' / 'shared',
)

py.install_sources(
  files(
    'kernel_contracts/__init__.py',
    'kernel_contracts/phase_1_act.py',
    'kernel_contracts/phase_2_learn_A_production.py',
    'kernel_contracts/phase_2_learn_B_processing.py',
    'kernel_contracts/phase_2_learn_C_reduction.py',
    'kernel_contracts/phase_2_learn_D_backprop.py',
    'kernel_contracts/phase_3_update.py',
  ),
  subdir: 'averaging_ensembled_classifier' / 'shared' / 'kernel_contracts',
)
```

**`src/backends/opencl/meson.build`** — Python package installation (no compilation):

```meson
py.install_sources(
  files(
    '__init__.py',
    'renderer.py',
    'context.py',
    'discovery.py',
    'type_mapping.py',
    'buffer_allocator.py',
    'retrieval.py',
  ),
  subdir: 'averaging_ensembled_classifier' / 'backends' / 'opencl',
)

py.install_sources(
  files('kernel_bindings/__init__.py'),
  # ... per-phase kernel binding files
  subdir: 'averaging_ensembled_classifier' / 'backends' / 'opencl' / 'kernel_bindings',
)
```

**`src/backends/vulkan/meson.build`** — SPIR-V compilation + Python installation:

```meson
# ── Python source installation ─────────────────────────────────────
py.install_sources(
  files(
    '__init__.py',
    'renderer.py',
    'context.py',
    'discovery.py',
    'type_mapping.py',
    'buffer_allocator.py',
    'retrieval.py',
  ),
  subdir: 'averaging_ensembled_classifier' / 'backends' / 'vulkan',
)

# ── SPIR-V compilation ────────────────────────────────────────────
# Each .comp shader is compiled to .spv via glslc.
# glslc was found by the top-level meson.build and is in scope via subdir().

vulkan_shader_dir = 'kernel_sources'
vulkan_shaders = [
  'forward_pass',
  'compute_hidden_mask',
  'compute_probs_loss_cce',
  'compute_probs_loss_bce',
  'calculate_module_param_grads',
  'backprop_error_to_hidden',
  'calculate_chunk_temp_gradients',
  'clip_partial_gradients',
  'gather_and_permute_grad_h',
  'aggregate_register_reduce',
  'aggregate_local_reduce',
  'clip_intermediate_grad',
  'stabilize_reduce_grad_h',
  'backprop_shared_weights',
  'backprop_shared_biases',
  'clip_shared_gradients',
  'normalize_gradients',
  'adam_update',
  'clamp_temperatures',
]

spv_targets = []
foreach shader_name : vulkan_shaders
  comp_file = files(vulkan_shader_dir / shader_name + '.comp')
  spv_target = custom_target(
    'spv_' + shader_name,
    input: comp_file,
    output: shader_name + '.spv',
    command: [glslc, '--target-env=vulkan1.1', '-O', '@INPUT@', '-o', '@OUTPUT@'],
    install: true,
    install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'vulkan' / 'spirv',
  )
  spv_targets += spv_target
endforeach
```

The `--target-env=vulkan1.1` flag pins the SPIR-V target to Vulkan 1.1, which provides subgroup operations (`VK_KHR_shader_subgroup`) required by Node 16's reduction kernel (VULKAN_BACKEND.md). The `-O` flag enables `glslc`'s standard optimization pass. These defaults are appropriate for the architecture's compute shader workload.

**`src/backends/cpu/meson.build`** — C shared library compilation:

```meson
# ── Python source installation ─────────────────────────────────────
py.install_sources(
  files(
    '__init__.py',
    'renderer.py',
    'discovery.py',
    'type_mapping.py',
    'buffer_allocator.py',
    'retrieval.py',
  ),
  subdir: 'averaging_ensembled_classifier' / 'backends' / 'cpu',
)

# ── CPU kernel shared library ──────────────────────────────────────
cpu_kernel_sources = files(
  'kernel_sources/forward_pass.c',
  'kernel_sources/compute_hidden_mask.c',
  'kernel_sources/compute_probs_loss_cce.c',
  'kernel_sources/compute_probs_loss_bce.c',
  'kernel_sources/calculate_module_param_grads.c',
  'kernel_sources/backprop_error_to_hidden.c',
  'kernel_sources/calculate_chunk_temp_gradients.c',
  'kernel_sources/clip_partial_gradients.c',
  'kernel_sources/gather_and_permute_grad_h.c',
  'kernel_sources/aggregate_register_reduce.c',
  'kernel_sources/aggregate_local_reduce.c',
  'kernel_sources/clip_intermediate_grad.c',
  'kernel_sources/stabilize_reduce_grad_h.c',
  'kernel_sources/backprop_shared_weights.c',
  'kernel_sources/backprop_shared_biases.c',
  'kernel_sources/clip_shared_gradients.c',
  'kernel_sources/normalize_gradients.c',
  'kernel_sources/adam_update.c',
  'kernel_sources/clamp_temperatures.c',
  'kernel_sources/thread_pool.c',
)

cpu_kernel_includes = include_directories('kernel_sources')

# ISA flags: user-provided or default to -march=native
cpu_isa_flags = get_option('cpu_isa_flags')
if cpu_isa_flags.length() == 0
  cpu_isa_flags = ['-march=native']
endif

cc = meson.get_compiler('c')

# Thread library (pthreads on POSIX, Windows threads via C11)
thread_dep = dependency('threads')
math_dep = cc.find_library('m', required: false)

cpu_kernels_lib = shared_library(
  'cpu_kernels',
  cpu_kernel_sources,
  include_directories: cpu_kernel_includes,
  c_args: cpu_isa_flags + ['-fvisibility=hidden', '-DCPU_KERNELS_EXPORT'],
  dependencies: [thread_dep, math_dep],
  gnu_symbol_visibility: 'hidden',
  install: true,
  install_dir: py.get_install_dir() / 'averaging_ensembled_classifier' / 'backends' / 'cpu',
)
```

The shared library uses hidden symbol visibility by default, with explicit export macros (`CPU_KERNELS_EXPORT`) on the public `task_<kernel_name>` entry points and `pool_dispatch_and_wait`. This ensures the ABI surface is exactly the set of functions the FFI layer (ADR-015) expects, preventing accidental symbol leakage.

### Build configuration manifest

A template file `src/_build_config.py.in` is processed by Meson's `configure_file()` to produce a Python module discoverable at runtime:

```python
# _build_config.py.in — processed by Meson configure_file()
"""Build-time configuration for averaging_ensembled_classifier.

Generated by the Meson build system. Do not edit manually.
"""

BACKEND_OPENCL: bool = @BACKEND_OPENCL@
BACKEND_VULKAN: bool = @BACKEND_VULKAN@
BACKEND_CPU: bool = @BACKEND_CPU@
```

This module serves two consumers:
1. **`orchestrator.py`** — queries `_build_config` to determine which backends are available for selection at runtime.
2. **Test framework (ADR-016)** — queries `_build_config` to determine which test tiers can execute. Tier 2 tests for a backend are skipped if that backend was not built. Tier 3 tests require at least two backends with `True` values.

### OpenCL kernel path discovery

The OpenCL backend discovers its kernel sources at runtime via `importlib.resources`:

```python
# src/backends/opencl/context.py
import importlib.resources

def _kernel_source_dir() -> Path:
    """Resolve the installed kernel source directory."""
    return importlib.resources.files('averaging_ensembled_classifier') / 'kernels'
```

This works for both standard and editable installs. In editable mode, `importlib.resources` resolves to the source tree's `kernels/` directory. In standard installs, it resolves to the installed package data directory. No build-time path injection is needed.

### SPIR-V module discovery

The Vulkan backend discovers compiled SPIR-V modules analogously:

```python
# src/backends/vulkan/context.py
import importlib.resources

def _spirv_dir() -> Path:
    """Resolve the installed SPIR-V module directory."""
    return importlib.resources.files(
        'averaging_ensembled_classifier.backends.vulkan'
    ) / 'spirv'
```

The `spirv/` subdirectory contains the `*.spv` files installed by the Vulkan backend's `custom_target` rules.

### CPU shared library discovery

The CPU backend discovers the compiled shared library via the same mechanism:

```python
# src/backends/cpu/renderer.py
import importlib.resources

def _cpu_kernels_lib_path() -> Path:
    """Resolve the CPU kernels shared library."""
    backend_dir = importlib.resources.files(
        'averaging_ensembled_classifier.backends.cpu'
    )
    # Platform-specific library name
    import sys
    if sys.platform == 'win32':
        return backend_dir / 'cpu_kernels.dll'
    elif sys.platform == 'darwin':
        return backend_dir / 'libcpu_kernels.dylib'
    else:
        return backend_dir / 'libcpu_kernels.so'
```

### Dependency declarations in `pyproject.toml`

The architecture's `pyproject.toml` is extended with optional dependency groups for backends that require external Python packages:

```toml
[project.optional-dependencies]
vulkan = ["vulkan-python>=0.2.0"]
cpu = []  # No additional Python deps — FFI is stdlib ctypes (ADR-015)
dev = [
    "pytest",
    "pytest-benchmark",
    "black",
    "isort",
    "flake8",
    "mypy",
]
```

The core `dependencies` list retains `numpy`, `pyopencl`, and `scikit-learn` — these are required for the always-enabled OpenCL backend.

### The complete build file tree

```
averaging_ensembled_classifier/
├── meson.build                              # Architecture conductor: options, probing, subdir() delegation
├── meson.options                            # backend_vulkan, backend_cpu, cpu_isa_flags
├── pyproject.toml                           # meson-python build backend + optional deps
├── kernels/                                 # OpenCL sources — installed as package data
│   ├── kernels.cl.h
│   └── *.cl.c
├── src/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── _build_config.py.in                  # Template: enabled backend manifest
│   ├── shared/
│   │   ├── meson.build                      # Pure Python install
│   │   └── *.py
│   └── backends/
│       ├── opencl/
│       │   ├── meson.build                  # Python install + kernel_bindings
│       │   └── *.py
│       ├── vulkan/
│       │   ├── meson.build                  # Python install + SPIR-V compilation
│       │   ├── *.py
│       │   └── kernel_sources/
│       │       └── *.comp
│       └── cpu/
│           ├── meson.build                  # Python install + C shared library
│           ├── *.py
│           └── kernel_sources/
│               ├── cpu_simd.h
│               ├── cpu_kernels.h
│               └── *.c
└── builddir/                                # Build output (not in source tree)
    └── ...
```

---

## Consequences

### Positive

- **Zero-configuration default.** `meson setup builddir` with no options probes the environment and enables available backends automatically. A developer with only PyOpenCL gets the OpenCL backend. A developer with the Vulkan SDK also gets SPIR-V compilation. A developer with a modern C compiler also gets the CPU backend. No manual flags required.

- **Conditional enablement is explicit and debuggable.** `meson configure builddir` shows the effective state of `backend_vulkan` and `backend_cpu`. Build logs clearly indicate which backends were probed, which dependencies were found, and which `subdir()` paths were entered.

- **Build logic co-location.** Each backend's `meson.build` lives alongside its source code. A developer adding a new Vulkan shader edits `src/backends/vulkan/meson.build` in the same directory. A developer adding a new CPU kernel edits `src/backends/cpu/meson.build` in the same directory. The top-level `meson.build` rarely changes after initial setup.

- **`meson-python` compatibility.** All installation rules use Meson's native `py.install_sources`, `install_data`, and `shared_library` with `install: true`. The `meson-python` build backend packages these artifacts correctly for both `pip install .` and `pip install -e .`.

- **Build manifest enables runtime and test discovery.** The generated `_build_config.py` module provides a single, reliable source of truth for which backends are available. The `orchestrator.py` and test framework (ADR-016) consume this module directly, eliminating runtime probing heuristics.

- **Predictable artifact layout.** Every backend's build products land in a predictable package namespace: `averaging_ensembled_classifier/backends/<name>/`. SPIR-V modules go to `.../vulkan/spirv/`. The CPU shared library goes to `.../cpu/libcpu_kernels.so`. OpenCL kernel sources go to `.../kernels/`. Runtime discovery via `importlib.resources` is straightforward and portable.

- **ISA flexibility without build complexity.** The `cpu_isa_flags` option defaults to `['-march=native']` for development, and can be overridden to `['-mavx2']`, `['-msse2']`, etc. for targeted deployment. The single-library strategy keeps the build simple while providing the override mechanism for when multi-ISA shipping becomes necessary.

- **STRUCTURE.md compliance.** The build system respects Architectural Autonomy (§2) — the architecture's build is self-contained with no framework-level changes. It provides Build-Time System Guarantees (§4) — a successful build ensures all enabled SPIR-V shaders compile and the CPU library links. Declarative Interaction (§3) — `meson.options` are the sole configuration surface.

### Negative

- **`subdir()` shares project context.** Variables defined in one `subdir()`'s `meson.build` are visible in subsequent `subdir()` calls. Backend build files must avoid variable name collisions (e.g., both Vulkan and CPU defining a `kernel_sources` variable). This is mitigated by using backend-prefixed variable names (`vulkan_shaders`, `cpu_kernel_sources`).

- **Dependency probing in top-level file.** Conditional `subdir()` inclusion requires the top-level `meson.build` to probe for `glslc` before delegating to the Vulkan `subdir()`. This means the probing logic is separated from the backend that uses it. The separation is minimal (one `find_program` call) and the result is passed implicitly via `subdir()` context sharing.

- **Single-ISA CPU library limits portability.** The compiled CPU library targets one ISA tier. Users deploying to heterogeneous hardware must either build with the lowest common ISA (e.g., `-msse2`) or provide multiple builds. This is an acceptable trade-off for the initial implementation; multi-ISA dispatch is documented as a future extension and can be added by introducing additional `shared_library` targets with distinct names and a runtime selection layer in ADR-015's FFI mechanism.

- **`_build_config.py` is generated.** The build configuration module does not exist in the source tree — it is generated during `meson setup`. Editable installs with `meson-python` handle this correctly (the generated file is placed in the build directory and made importable), but developers must run at least `meson setup` before the module is available. This is consistent with the existing workflow documented in STRUCTURE.md §4.

- **Vulkan SDK is an external dependency.** The `glslc` compiler is not installable via `pip` — it requires the Vulkan SDK (LunarG or system packages). This adds an environment setup step for developers targeting the Vulkan backend. The `auto` default mitigates this: developers without the Vulkan SDK simply don't get the Vulkan backend, with no build failure.

### Migration implications

Per ADR-017's phasing:

**Phase 0 (Foundation):**
- Create `meson.options` with `backend_vulkan` and `backend_cpu` options.
- Create `src/_build_config.py.in` template.
- Refactor the architecture's `meson.build` to use `subdir()` delegation.
- Create skeleton `meson.build` files in `src/shared/`, `src/backends/opencl/`.
- The existing OpenCL-only build continues to work — `src/shared/` and `src/backends/opencl/` contain the migrated Python source files from ADR-012's Phase 0.

**Phase 2 (OpenCL Renderer):**
- The OpenCL backend's `meson.build` installs kernel binding Python files.
- `install_data` for `kernels/` is already in the top-level `meson.build`.
- The generated `_build_config.py` reflects `BACKEND_OPENCL = True`.

**Phase 3 (CPU Backend):**
- Create `src/backends/cpu/meson.build` with `shared_library` target.
- Create `src/backends/cpu/kernel_sources/` with C implementations.
- `_build_config.py` gains `BACKEND_CPU = True`.
- Tier 2 tests execute for the CPU backend; Tier 3 tests compare CPU vs. OpenCL.

**Phase 4 (Vulkan Backend):**
- Create `src/backends/vulkan/meson.build` with `custom_target` SPIR-V compilation rules.
- Create `src/backends/vulkan/kernel_sources/` with GLSL compute shaders.
- `_build_config.py` gains `BACKEND_VULKAN = True`.
- Tier 3 tests compare Vulkan vs. OpenCL/CPU.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; backend isolation; `PlanRenderer` interface
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `src/shared/` + `src/backends/<name>/` directory structure; pure Python shared layer; per-backend compilation requirements
- [ADR-013: Kernel Source Strategy](ADR-013-kernel-source-strategy.md) — kernel source locations (`kernels/`, `src/backends/vulkan/kernel_sources/`, `src/backends/cpu/kernel_sources/`); OpenCL cross-boundary reference; SPIR-V at build time; `task_<kernel_name>` C ABI
- [ADR-015: Python ↔ Native Backend Interop](ADR-015-python-native-backend-interop.md) — CPU shared library loading mechanism; C ABI surface consumed by FFI layer; ctypes as CPU backend FFI
- [ADR-016: Test Strategy](ADR-016-test-strategy-stub.md) — Tier 1/2/3 test structure; conditional test execution based on enabled backends
- [ADR-017: Incremental Migration Path](ADR-017-incremental-migration-path-stub.md) — per-phase build system evolution
- [CONCEPT.md](../CONCEPT.md) — §1 Architectural Elegance Feedback (formalize build abstractions, don't hack around them)
- [CONTRACT.md](../CONTRACT.md) — Article 6 Mandatory Build-Time Symbols (`SCALAR_TYPE`, `SIMD_WIDTH`, `C_TILE_SIZE`, `NUMERICAL_STABILITY_EPSILON`)
- [STRUCTURE.md](../../STRUCTURE.md) — §2 Architectural Autonomy; §3 Declarative Interaction; §4 Build-Time System Guarantees
- [CPU_BACKEND.md](../CPU_BACKEND.md) — `cpu_simd.h` ISA detection cascade; `task_<kernel_name>` function signature; thread pool; `cpu_kernels.h` ABI surface
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — SPIR-V compilation requirement; specialization constants; `--target-env=vulkan1.1`
