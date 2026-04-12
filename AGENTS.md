(from architectures/averaging_ensembled_classifier/CONCEPT.md)

#### 1. **Architectural Elegance Feedback**

Optimization pressure that violates these core principles shall be interpreted an useful signal that indicates incomplete architectural modeling, thus is no-justification for exceptions. The system must evolve its formal abstractions to subsume valid optimizations as first-class primitives, never compromise its contracts.

> **When emergent efficiency gains contradict current constraints:**
>
> 1. **Suspend implementation** of the optimization
> 2. **Formalize the pattern** as a documented architectural primitive
> 3. **Reify the optimization** through revised contracts & DAG extensions
>
> _Example:_ Discovering a kernel fusion opportunity that violates interface verifiability triggers:
>
> - Halting ad-hoc fusion attempts
> - Modeling fused operation in Design Document as new DAG node type
> - Defining strict fusion contracts in Kernel Headers
> - Implementing through host-controlled parameter switches, not hidden logic

#### 2. Development Build Discipline (CPU Backend)

When working in this repository, CPU backend native code is loaded from Meson build outputs during development. If any file under `architectures/averaging_ensembled_classifier/src/backends/cpu/kernel_sources/` is edited, rebuild the shared library before testing:

```bash
cd architectures/averaging_ensembled_classifier
ninja -C builddir
```

Do not assume an editable Python reinstall will recompile the native CPU library.

## Replacing a File

When a file needs significant rewriting, avoid accumulating many in-place edits — they are error-prone. Instead:

1. Read the file.
2. `rm` the file.
3. Recreate the file with the full updated content.

## Running Tests

When running tests or lints, tee to a temp file (`/tmp/...`) and then grep that
file after the tests have completed. Rerunning a test to get a different grep is wasteful.
