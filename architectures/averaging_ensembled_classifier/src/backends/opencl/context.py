# src/backends/opencl/context.py
"""OpenCL context and kernel compilation utilities."""

from dataclasses import dataclass

import pyopencl as cl


@dataclass(frozen=True)
class OpenCLContext:
    """Holds the OpenCL context, queue, device, and compiled program.

    Created by the renderer at initialization. This is the renderer's
    context object.
    """
    context: cl.Context
    queue: cl.CommandQueue
    device: cl.Device
    program: cl.Program


# =========================================================================
# === Kernel compilation utilities
# =========================================================================

# Ordered list of kernel source files. Header first, then phase source files.
_KERNEL_FILES = (
    "kernels.cl.h",
    "phase_1_act.cl.c",
    "phase_2_learn_A_production.cl.c",
    "phase_2_learn_B_processing.cl.c",
    "phase_2_learn_C_reduction.cl.c",
    "phase_2_learn_D_backprop.cl.c",
    "phase_3_update.cl.c",
)


def load_and_compile_kernels(
    context: cl.Context,
    device: cl.Device,
    compiler_flags: list[str],
) -> cl.Program:
    """Load kernel sources from the architecture's kernels/ package and compile.

    Sources are loaded via importlib.resources so the package can be
    installed or run from a source tree.
    """
    import importlib.resources
    package = importlib.resources.files("averaging_ensembled_classifier.kernels")
    sources: list[str] = []
    for fname in _KERNEL_FILES:
        sources.append(package.joinpath(fname).read_text())

    program = cl.Program(context, "\n".join(sources))
    program.build(options=" ".join(compiler_flags), devices=[device])
    return program


def load_and_compile_kernels_from_path(
    context: cl.Context,
    device: cl.Device,
    compiler_flags: list[str],
    kernel_dir: str,
) -> cl.Program:
    """Load kernel sources from a filesystem path and compile.

    This is the fallback for development trees where the kernels package
    may not be importable as a Python package.
    """
    from pathlib import Path
    kernel_dir_path = Path(kernel_dir)
    sources: list[str] = []
    for fname in _KERNEL_FILES:
        sources.append((kernel_dir_path / fname).read_text())

    program = cl.Program(context, "\n".join(sources))
    program.build(options=" ".join(compiler_flags), devices=[device])
    return program
