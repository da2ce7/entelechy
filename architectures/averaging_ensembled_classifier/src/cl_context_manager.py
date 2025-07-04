# cl_context_manager.py

"""
A Self-Configuring Manager for the OpenCL Compute Environment.

This module provides the `OpenCLContextManager` class, whose sole responsibility
is to prepare a complete, ready-to-use OpenCL compute environment.

This manager embodies the principle of "intelligent self-configuration":
1.  It automatically discovers and concatenates all kernel source files from
    a specified directory.
2.  It performs device introspection at runtime to discover optimal hardware
    parameters (e.g., SIMD width, optimal tile sizes).
3.  It returns a single, comprehensive `ComputeEnvironment` object that bundles
    the CL resources (context, queue, program) with the discovered hardware
    constants, providing the application with a clean, pre-built foundation.
"""

import os
import abc
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
from typing import Type
import math
import shutil

import pyopencl as cl

from .arch_primitives import PrecisionContext, Float32Context, Float16Context


# ====== The Data Structures for the Compute Environment ======
# These dataclasses define the structure of the object that this manager produces.


@dataclass(frozen=True)
class CLBundle:
    """A simple, immutable container for the core OpenCL resources."""

    context: cl.Context
    queue: cl.CommandQueue
    program: cl.Program


@dataclass(frozen=True)
class DiscoveredArchConstants:
    """Holds hardware parameters discovered through device introspection."""

    simd_width: int
    optimal_tile_size: int
    global_mem_cacheline_size: int
    local_mem_size_bytes: int


@dataclass(frozen=True)
class ComputeEnvironment(PrecisionContext, abc.ABC):
    """
    An *abstract* contract for a runtime environment.

    By inheriting from PrecisionContext, this class is now abstract. Any attempt
    to instantiate it directly will result in a TypeError, as it does not
    implement the abstract properties `SCALAR_NP_TYPE` and `SCALAR_C_TYPE_NAME`.
    This enforces the architectural rule that every compute environment MUST
    have a defined precision.
    """

    cl_bundle: CLBundle
    arch_consts: DiscoveredArchConstants


@dataclass(frozen=True)
class Float32ComputeEnvironment(ComputeEnvironment, Float32Context):
    """
    A concrete, instantiable runtime environment for FP32.
    It fulfills the data contract of ComputeEnvironment and the precision
    contract of Float32Context, making it a valid, complete object.
    """

    pass


@dataclass(frozen=True)
class Float16ComputeEnvironment(ComputeEnvironment, Float16Context):
    """A concrete, instantiable runtime environment for FP16."""

    pass


# ====== The Manager Class ======


class OpenCLContextManager:
    """Builds a complete, self-configured, and precision-aware OpenCL environment."""

    def __init__(self, kernel_source_dir: str):
        """
        Initializes the manager with the path to the kernel source code.

        Args:
            kernel_source_dir: Path to the directory containing .cl.h and .cl.c files.
        """
        if not os.path.isdir(kernel_source_dir):
            raise FileNotFoundError(f"Kernel source directory does not exist: {kernel_source_dir}")
        self.kernel_source_dir = kernel_source_dir

    def _find_kernel_files(self) -> List[str]:
        """Recursively finds all .cl.h and .cl.c files in the source directory."""
        all_files = [os.path.join(path, name) for path, _, files in os.walk(self.kernel_source_dir) for name in files]
        headers = sorted([f for f in all_files if f.endswith(".cl.h")])
        sources = sorted([f for f in all_files if f.endswith(".cl.c")])
        if not headers and not sources:
            raise FileNotFoundError(f"No kernel files (.cl.h, .cl.c) found in '{self.kernel_source_dir}'")
        return headers + sources

    def _load_and_concatenate_sources(self, kernel_files: List[str]) -> str:
        """Reads all specified kernel files and concatenates them into a single string."""
        full_source = "/* Auto-concatenated OpenCL Source */\n\n"
        print("INFO: Loading kernel sources...")
        for fname in kernel_files:
            with open(fname, "r") as f:
                print(f"      - Appending {fname}")
                full_source += f.read() + "\n\n"
        return full_source

    def build_and_discover(self, precision: str = "float32") -> ComputeEnvironment:
        """
        The primary factory method. Creates the context, discovers hardware
        parameters, builds the program for a specific precision, and returns the
        complete, concrete compute environment.

        Args:
            precision: The target precision ("float32" or "float16").

        Returns:
            A concrete subclass of ComputeEnvironment (e.g., Float32ComputeEnvironment).
        """
        print(f"\n--- Building and Discovering OpenCL Environment for Precision: {precision.upper()} ---")

        # Step A: Select the Precision Context and Environment Class
        # This is the core of the factory pattern.
        context_mixin: PrecisionContext
        env_class: Type[ComputeEnvironment]

        if precision == "float32":
            context_mixin = Float32Context()
            env_class = Float32ComputeEnvironment
        elif precision == "float16":
            context_mixin = Float16Context()
            env_class = Float16ComputeEnvironment
        else:
            raise ValueError(f"Unsupported precision '{precision}'. Choose 'float32' or 'float16'.")

        try:
            ctx = cl.create_some_context(interactive=False)
        except cl.RuntimeError as e:
            print(
                f"FATAL: Could not create OpenCL context. Is an OpenCL-capable GPU installed and drivers up to date? ({e})"
            )
            raise

        device = ctx.devices[0]
        print(f"INFO: Using device: {device.name} ({device.vendor})")

        # Step B: Device Introspection
        print("INFO: Discovering hardware parameters...")
        simd_width = (
            device.preferred_vector_width_float if precision == "float32" else device.preferred_vector_width_half
        )
        discovered_consts = DiscoveredArchConstants(
            simd_width=simd_width or 4,
            optimal_tile_size=int(math.sqrt(device.max_work_group_size)) & ~1,
            global_mem_cacheline_size=device.global_mem_cacheline_size or 64,
            local_mem_size_bytes=device.local_mem_size,
        )
        print(f"      - Discovered Constants: {discovered_consts}")

        # Step C: Link Python Type System to C Preprocessor and Compile
        kernel_files = self._find_kernel_files()
        full_source = self._load_and_concatenate_sources(kernel_files)

        options = ["-cl-std=CL1.2"]
        # Use the selected context to set the C-level type.
        options.append(f"-D SCALAR_TYPE={context_mixin.SCALAR_C_TYPE_NAME}")
        options.append(f"-D SIMD_WIDTH={discovered_consts.simd_width}")
        options.append(f"-D C_TILE_SIZE={discovered_consts.optimal_tile_size}")
        options.append(f"-D LOCAL_MEM_BANK_PADDING=1")
        if precision == "float16":
            # FP16 is an optional extension that must be explicitly enabled.
            options.append("-cl-fp32-correctly-rounded-divide-sqrt")  # Often good practice
            options.append("-D cl_khr_fp16")

        print(f"INFO: Compiling with options: {' '.join(options)}")
        try:
            program = cl.Program(ctx, full_source).build(options=options)
            print("INFO: Kernel compilation successful.")
        except cl.Error as e:
            # (Detailed error logging as before)
            log_header = "\n" + "=" * 80 + "\n--- KERNEL BUILD FAILED ---\n" + "=" * 80
            if hasattr(e, "device_logs"):
                log_details = "\n\n".join(
                    [f"Device: {dev.name}\n--- Build Log ---\n{log}" for dev, log in e.device_logs]
                )
                print(log_header, log_details, "=" * 80, sep="\n")
            else:
                print(log_header, f"An unexpected OpenCL error occurred: {e}", "=" * 80, sep="\n")
            raise

        # Step D: Assemble and Return the Final, Concrete Environment
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        cl_bundle = CLBundle(context=ctx, queue=queue, program=program)

        # Instantiate the correct subclass (e.g., Float32ComputeEnvironment).
        final_environment = env_class(cl_bundle=cl_bundle, arch_consts=discovered_consts)

        return final_environment
