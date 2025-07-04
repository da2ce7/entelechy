# cl_context_manager.py

"""
A Self-Configuring Manager for the OpenCL Compute Environment.

This module provides the definitive "Artisan" of the system: the OpenCLContextManager.
Its sole and sacred jurisdiction is to act as the bridge between the abstract,
logical world of our Python code and the concrete, physically-constrained world
of the GPU hardware.

It performs an act of architectural alchemy:
1.  It GATHERS the raw materials: scattered kernel files from the filesystem.
2.  It COMMUNES with the hardware: discovering its physical truths through introspection.
3.  It FORGES a contract: translating Python-side choices into C-level build flags.
4.  It ASSEMBLES the final artifact: a single, immutable, and verifiably type-safe
    ComputeEnvironment, ready to be passed to the rest of the system as the
    unquestionable source of truth for the runtime context.
"""

import os
import abc
from dataclasses import dataclass
from typing import Dict, List, Type
import math

import pyopencl as cl

from .arch_primitives import PrecisionContext, Float32Context, Float16Context


# =========================================================================
# === The Canonical Data Structures of the Compute Environment          ===
# =========================================================================


@dataclass(frozen=True)
class CLBundle:
    """A simple, immutable vessel for the core, untyped OpenCL resources."""

    context: cl.Context
    queue: cl.CommandQueue
    program: cl.Program


@dataclass(frozen=True)
class DiscoveredArchConstants(PrecisionContext, abc.ABC):
    """
    An *abstract contract* for the physical truths of the hardware, discovered
    for a specific precision.

    This class codifies a fundamental architectural mandate: a physical fact
    like `simd_width` is meaningless without the logical context of the data type
    it operates upon. By inheriting from `PrecisionContext`, we make it
    impossible to represent the invalid state of having hardware constants
    divorced from their precision, enforcing correctness by design.
    """

    simd_width: int
    optimal_tile_size: int
    global_mem_cacheline_size: int
    local_mem_size_bytes: int


@dataclass(frozen=True)
class Float32DiscoveredArchConstants(DiscoveredArchConstants, Float32Context):
    """The concrete realization of the FP32 hardware contract."""

    pass


@dataclass(frozen=True)
class Float16DiscoveredArchConstants(DiscoveredArchConstants, Float16Context):
    """The concrete realization of the FP16 hardware contract."""

    pass


@dataclass(frozen=True)
class ComputeEnvironment(PrecisionContext, abc.ABC):
    """
    An *abstract contract* for the complete, canonical runtime environment.

    This is the final, assembled artifact passed to the application.
    It synthesizes the logical choice of precision with the discovered physical
    hardware truths and the compiled device program into a single, cohesive unit.
    """

    cl_bundle: CLBundle
    arch_consts: DiscoveredArchConstants


@dataclass(frozen=True)
class Float32ComputeEnvironment(ComputeEnvironment, Float32Context):
    """The concrete, type-safe runtime environment for FP32 operations."""

    # This type hint is a powerful contract: an FP32 environment MUST
    # contain a set of constants discovered for FP32.
    arch_consts: Float32DiscoveredArchConstants


@dataclass(frozen=True)
class Float16ComputeEnvironment(ComputeEnvironment, Float16Context):
    """The concrete, type-safe runtime environment for FP16 operations."""

    arch_consts: Float16DiscoveredArchConstants


# =========================================================================
# === The Artisan Class: The OpenCL Context Manager                     ===
# =========================================================================


class OpenCLContextManager:
    """The Artisan that builds the complete, self-configured OpenCL environment."""

    def __init__(self, kernel_source_dir: str):
        """
        Initializes the manager with the path to the raw materials.

        Args:
            kernel_source_dir: Path to the directory containing .cl.h and .cl.c files.
        """
        if not os.path.isdir(kernel_source_dir):
            raise FileNotFoundError(f"Kernel source directory does not exist: {kernel_source_dir}")
        self.kernel_source_dir = kernel_source_dir

    def _find_kernel_files(self) -> List[str]:
        """Gathers all kernel source files, the raw clay for our program."""
        all_files = [os.path.join(path, name) for path, _, files in os.walk(self.kernel_source_dir) for name in files]
        headers = sorted([f for f in all_files if f.endswith(".cl.h")])
        sources = sorted([f for f in all_files if f.endswith(".cl.c")])
        if not headers and not sources:
            raise FileNotFoundError(f"No kernel files (.cl.h, .cl.c) found in '{self.kernel_source_dir}'")
        return headers + sources

    def _load_and_concatenate_sources(self, kernel_files: List[str]) -> str:
        """Reads and consolidates all source files into a single compilation unit."""
        full_source = ""
        for fname in kernel_files:
            with open(fname, "r") as f:
                full_source += f.read() + "\n\n"
        return full_source

    def build_and_discover(self, precision: str = "float32") -> ComputeEnvironment:
        """
        The primary factory method. It orchestrates the entire bootstrapping
        process, from hardware discovery to program compilation, returning the
        final, immutable compute environment.
        """
        # --- Phase A: Architectural Strategy Selection ---
        # Based on the user's high-level intent (`precision`), we select the
        # full suite of concrete classes that will embody this strategy.
        arch_consts_class: Type[DiscoveredArchConstants]
        env_class: Type[ComputeEnvironment]

        if precision == "float32":
            arch_consts_class = Float32DiscoveredArchConstants
            env_class = Float32ComputeEnvironment
        elif precision == "float16":
            arch_consts_class = Float16DiscoveredArchConstants
            env_class = Float16ComputeEnvironment
        else:
            raise ValueError(f"Unsupported precision '{precision}'. Choose 'float32' or 'float16'.")

        try:
            ctx = cl.create_some_context(interactive=False)
        except cl.RuntimeError as e:
            # Propagate failure with a clear, actionable message.
            raise RuntimeError(
                f"FATAL: Could not create OpenCL context. Is an OpenCL-capable GPU installed and drivers up to date? ({e})"
            )

        device = ctx.devices[0]

        # --- Phase B: Device Introspection ---
        # We commune with the hardware to discover its physical truths, making
        # the implicit runtime properties an explicit part of our context.
        simd_width_val = (
            device.preferred_vector_width_float if precision == "float32" else device.preferred_vector_width_half
        )
        discovered_consts = arch_consts_class(
            simd_width=simd_width_val or 4,
            optimal_tile_size=int(math.sqrt(device.max_work_group_size)) & ~1,
            global_mem_cacheline_size=device.global_mem_cacheline_size or 64,
            local_mem_size_bytes=device.local_mem_size,
        )

        # --- Phase C: Source Aggregation and Program Forging ---
        # We gather our source code and forge it into a device-specific program,
        # injecting our discovered truths as C-level preprocessor macros. This
        # is the critical bridge from Python context to device contract.
        kernel_files = self._find_kernel_files()
        full_source = self._load_and_concatenate_sources(kernel_files)

        options = ["-cl-std=CL1.2"]
        options.append(f"-D SCALAR_TYPE={discovered_consts.SCALAR_C_TYPE_NAME}")
        options.append(f"-D SIMD_WIDTH={discovered_consts.simd_width}")
        options.append(f"-D C_TILE_SIZE={discovered_consts.optimal_tile_size}")
        options.append(f"-D LOCAL_MEM_BANK_PADDING=1")
        if precision == "float16":
            options.append("-cl-fp32-correctly-rounded-divide-sqrt")
            options.append("-D cl_khr_fp16")

        try:
            program = cl.Program(ctx, full_source).build(options=options)
        except cl.Error as e:
            # Provide maximum transparency upon failure, upholding the principle
            # that build-time validation prevents runtime chaos.
            log_header = "\n" + "=" * 80 + "\n--- KERNEL BUILD FAILED ---\n" + "=" * 80
            log_details = ""
            if hasattr(e, "device_logs"):
                log_details = "\n\n".join(
                    [f"Device: {dev.name}\n--- Build Log ---\n{log}" for dev, log in e.device_logs]
                )
            else:
                log_details = f"An unexpected OpenCL error occurred: {e}"

            full_error = "\n".join([log_header, log_details, "=" * 80])
            raise RuntimeError(full_error) from e

        # --- Phase D: Final Assembly ---
        # All individual resources are bundled into the final, immutable artifact.
        # This ComputeEnvironment is now the sole source of truth for the runtime.
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        cl_bundle = CLBundle(context=ctx, queue=queue, program=program)
        final_environment = env_class(cl_bundle=cl_bundle, arch_consts=discovered_consts)

        return final_environment
