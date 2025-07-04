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
from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import math
import shutil

import pyopencl as cl


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
class ComputeEnvironment:
    """The complete, ready-to-use compute environment, including the CLBundle
    and the discovered hardware parameters."""

    cl_bundle: CLBundle
    arch_consts: DiscoveredArchConstants


# ====== The Manager Class ======


class OpenCLContextManager:
    """Builds a complete, self-configured OpenCL environment."""

    def __init__(self, kernel_source_dir: str):
        """
        Initializes the manager with the path to the kernel source code.

        Args:
            kernel_source_dir: The a path to the directory containing all
                               .cl.h and .cl.c files.
        """
        if not os.path.isdir(kernel_source_dir):
            raise FileNotFoundError(f"The specified kernel source directory does not exist: {kernel_source_dir}")
        self.kernel_source_dir = kernel_source_dir

    def _find_kernel_files(self) -> List[str]:
        """Recursively finds all .cl.h and .cl.c files in the source directory."""
        all_files = [os.path.join(path, name) for path, _, files in os.walk(self.kernel_source_dir) for name in files]
        # Robustness: Process headers first to ensure definitions are available before use.
        headers = sorted([f for f in all_files if f.endswith(".cl.h")])
        sources = sorted([f for f in all_files if f.endswith(".cl.c")])
        if not (headers or sources):
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

    def build_and_discover(self) -> ComputeEnvironment:
        """
        The primary public method. Creates the context, discovers hardware
        parameters, builds the program, and returns the complete environment.
        """
        print("\n--- Building and Discovering OpenCL Environment ---")
        try:
            ctx = cl.create_some_context(interactive=False)
        except cl.RuntimeError as e:
            print(
                f"FATAL: Could not create OpenCL context. Is an OpenCL-capable GPU installed and drivers up to date? ({e})"
            )
            raise

        device = ctx.devices[0]
        print(f"INFO: Using device: {device.name} ({device.vendor})")

        # --- 1. Device Introspection ---
        print("INFO: Discovering hardware parameters through introspection...")
        simd_width = device.preferred_vector_width_float or 4
        max_wg_size = device.max_work_group_size
        optimal_tile_size = int(math.sqrt(max_wg_size)) & ~1
        cacheline_bytes = device.global_mem_cacheline_size or 64
        local_mem_bytes = device.local_mem_size

        discovered_consts = DiscoveredArchConstants(
            simd_width=simd_width,
            optimal_tile_size=optimal_tile_size,
            global_mem_cacheline_size=cacheline_bytes,
            local_mem_size_bytes=local_mem_bytes,
        )
        print(f"      - Discovered Constants: {discovered_consts}")

        # --- 2. Program Compilation ---
        kernel_files = self._find_kernel_files()
        full_source = self._load_and_concatenate_sources(kernel_files)

        # Build compiler options from discovered constants
        options = ["-cl-std=CL1.2"]
        options.append(f"-D SIMD_WIDTH={discovered_consts.simd_width}")
        options.append(f"-D C_TILE_SIZE={discovered_consts.optimal_tile_size}")
        options.append(f"-D LOCAL_MEM_BANK_PADDING=1")  # This could also be a configurable static define

        print(f"INFO: Compiling with options: {' '.join(options)}")
        try:
            # Attempt to build the OpenCL program with the provided options.
            program = cl.Program(ctx, full_source).build(options=options)
            print("INFO: Kernel compilation successful.")

        # Here, we catch the broader cl.Error for static type checker compatibility (e.g., mypy).
        # mypy may not be able to resolve the full inheritance chain and might flag
        # a direct catch of cl.BuildError as an undefined attribute of the 'cl' module.
        except cl.Error as e:
            print("\n" + "=" * 80 + "\n--- KERNEL BUILD FAILED ---\n" + "=" * 80)

            # At runtime, we inspect the caught exception to see if it has the specific
            # 'device_logs' attribute. This is a robust way to check if the error is,
            # in fact, the more detailed cl.BuildError, without making static assumptions.
            if hasattr(e, 'device_logs'):
                # If it is a BuildError, we can safely access 'device_logs'.
                # This provides the detailed, device-specific compiler output which is
                # essential for debugging kernel code.
                log = "\n\n".join([f"Device: {dev.name}\n--- Build Log --- \n{log}" for dev, log in e.device_logs])
                print(log)
            else:
                # If it's another type of cl.Error (e.g., a runtime error not related
                # to compilation), it won't have 'device_logs'. In this case, we print a
                # general error message to avoid an `AttributeError`.
                print(f"An unexpected OpenCL error occurred: {e}")

            print("=" * 80)
            raise

        # --- 3. Assemble and Return the Final Environment ---
        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        cl_bundle = CLBundle(context=ctx, queue=queue, program=program)

        return ComputeEnvironment(cl_bundle=cl_bundle, arch_consts=discovered_consts)


if __name__ == "__main__":
    print("--- OpenCL Context Manager: Demonstration ---")
    DUMMY_KERNEL_DIR = "./temp_kernels_demo"
    try:
        os.makedirs(DUMMY_KERNEL_DIR, exist_ok=True)
        with open(os.path.join(DUMMY_KERNEL_DIR, "contract.cl.h"), "w") as f:
            f.write("/* Header */\n")
        with open(os.path.join(DUMMY_KERNEL_DIR, "kernel.cl.c"), "w") as f:
            f.write("__kernel void test_kernel() {}\n")

        manager = OpenCLContextManager(kernel_source_dir=DUMMY_KERNEL_DIR)

        print("\nAttempting to build and discover...")
        # compute_env = manager.build_and_discover() # This line would be active in a real run
        print("\n(Simulating successful build and discovery for demonstration.)")

    except (cl.RuntimeError, FileNotFoundError) as e:
        print(f"\nCaught expected error during demonstration: {e}")
        print("This is normal if OpenCL drivers are not installed on this machine.")
    finally:
        if os.path.exists(DUMMY_KERNEL_DIR):
            shutil.rmtree(DUMMY_KERNEL_DIR)

    print("\n--- Context Manager demonstration complete. ---")
