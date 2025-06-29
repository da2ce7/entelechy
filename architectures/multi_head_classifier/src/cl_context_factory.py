# cl_context_factory.py

"""
A Dedicated Factory for OpenCL Environment Creation.

This module provides a single class, `OpenCLContextFactory`, whose sole
responsibility is to abstract away all the boilerplate of finding kernel files,
setting compile-time constants, building the OpenCL program, and cleanly
handling any build errors.

This enforces the Single Responsibility Principle by separating the concerns of
hardware/compiler setup from the high-level application logic of the main
orchestrator.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Union

import pyopencl as cl


@dataclass(frozen=True)
class CLBundle:
    """A simple, immutable container for a ready-to-use OpenCL environment."""

    context: cl.Context
    queue: cl.CommandQueue
    program: cl.Program


class OpenCLContextFactory:
    """Builds a complete, compiled OpenCL environment from source files and definitions."""

    def __init__(self, kernel_files: List[str], defines: Dict[str, Union[str, int, float]]):
        """
        Initializes the factory with the list of kernel source files and a
        dictionary of preprocessor definitions.

        Args:
            kernel_files: A list of file paths to the .cl.h and .cl.c files.
            defines: A dictionary of key-value pairs to be passed as -D flags
                     to the OpenCL compiler (e.g., {'SCALAR_TYPE': 'float'}).
        """
        self.kernel_files = kernel_files
        self.defines = defines

    def _load_and_concatenate_sources(self) -> str:
        """Reads all specified kernel files and concatenates them into a single string."""
        full_source = "/* Auto-concatenated OpenCL Source */\n\n"
        print("INFO: Loading kernel sources...")
        for fname in self.kernel_files:
            if not os.path.exists(fname):
                raise FileNotFoundError(f"Kernel source file not found: {fname}")
            with open(fname, "r") as f:
                print(f"      - Appending {fname}")
                full_source += f.read() + "\n\n"
        return full_source

    def _format_compiler_options(self) -> List[str]:
        """Formats the preprocessor definitions into a list of compiler flags."""
        options = ["-cl-std=CL1.2"]
        for key, value in self.defines.items():
            options.append(f"-D {key}={value}")
            # Special handling for half-precision floating point support
            if key == "SCALAR_TYPE" and value == "half":
                options.append("-cl-khr-fp16")
        return options

    def build(self) -> CLBundle:
        """
        Creates the context, builds the program, and returns a CLBundle.

        This method encapsulates the entire setup and compilation process, including
        robust error handling for kernel build failures.

        Returns:
            A CLBundle object containing the a usable context, queue, and program.

        Raises:
            cl.BuildError: If the OpenCL kernel compilation fails, this error is
                           caught, logged cleanly, and then re-raised.
            FileNotFoundError: If a specified kernel source file cannot be found.
        """
        print("\n--- Building OpenCL Environment ---")
        ctx = cl.create_some_context(interactive=False)
        device = ctx.devices[0]
        print(f"INFO: Using device: {device.name} ({device.vendor})")

        queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
        full_source = self._load_and_concatenate_sources()
        options = self._format_compiler_options()
        print(f"INFO: Compiling with options: {' '.join(options)}")

        try:
            program = cl.Program(ctx, full_source).build(options=options)
            print("INFO: Kernel compilation successful.")
            return CLBundle(context=ctx, queue=queue, program=program)
        except cl.BuildError as e:
            # This is the critical error handling block. It makes build failures
            # readable and easy to debug.
            print("\n" + "=" * 80)
            print("--- KERNEL BUILD FAILED ---")
            print("=" * 80)
            log = "\n\n".join([f"Device: {dev.name}\n--- Build Log ---\n{log}" for dev, log in e.device_logs])
            print(log)
            print("=" * 80)
            raise  # Re-raise the exception to halt the application
