# cl_context_manager.py

"""
The Jurisdictional Mandate of the OpenCL Context Manager.

This module provides the definitive "Artisan" of the system: the
`OpenCLContextManager`. Its sole and sacred jurisdiction is to act as the
bridge between the abstract, logical world of our Python code and the concrete,
physically-constrained world of the GPU hardware.

It performs an act of architectural alchemy:
1.  It GATHERS the raw materials: scattered kernel files from the filesystem.
2.  It COMMUNES with the hardware: discovering its physical truths through
    programmatic introspection.
3.  It FORGES a contract: translating Python-side choices into C-level build
    flags, turning abstract constants into concrete device-side macros.
4.  It ASSEMBLES the final artifact: a single, immutable, and verifiably
    type-safe `ComputeEnvironment`, which is then passed to the rest of the
    system as the unquestionable source of truth for the runtime context.
"""

import os
import abc
import math
from dataclasses import dataclass
from typing import Dict, List, Type, overload

import pyopencl as cl

from ...arch_primitives import PrecisionContext, Float32Context, Float16Context


# =========================================================================
# === The Canonical Data Structures of the Compute Environment
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

    # --- Core Hardware Properties ---

    simd_width: int
    """
    The preferred vector width for the chosen scalar type (e.g., float, half).
    This value is the source of truth for the `SIMD_WIDTH` macro and is used
    to define SIMD-aware memory padding and grid dimensions.
    """

    global_mem_cacheline_size: int
    """
    The size of a global memory cache line in bytes. This is the source of
    truth for fulfilling any `Padding Contract` of type `CACHE`.
    """

    local_mem_size_bytes: int
    """
    The total available local memory per compute unit in bytes. Used by the host
    for validation and planning of local memory allocations.
    """

    # --- Purpose-Driven Tiling & Work-Group Constants ---

    optimal_workgroup_size_1d_reduction: int
    """
    The optimal work-group size (number of work-items) for 1D reduction kernels
    (e.g., `aggregate_local_reduce`). This is the canonical size for any kernel
    whose primary purpose is a parallel reduction over a single dimension.
    """

    optimal_square_tile_dim: int
    """
    The optimal side length (in elements) for a square tile used in 2D algorithms.
    This is the source of truth for the `C_TILE_SIZE` macro, primarily serving
    kernels like `transpose_chunk`.
    """

    optimal_rectangular_tile_dim1: int
    """
    The optimal work-group size for dimension 1 of a rectangular tiling scheme,
    typically used in GEMM-like computation kernels (e.g., `backprop_shared_weights_chunk`).
    This provides a distinct tuning parameter for non-square 2D workloads.
    """


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
    It represents the final, assembled artifact passed to the application.
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
# === The Artisan Class: The OpenCL Context Manager
# =========================================================================


class OpenCLContextManager:
    """The Artisan that builds the complete, self-configured OpenCL environment."""

    def __init__(self, kernel_source_dir: str):
        """Initializes the manager with the path to its raw materials."""
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

    # --- THE @OVERLOAD DECORATOR FORGES THE UNBREAKABLE CONTRACT ---
    # This first signature is a promise: "If you give me the Float32Context CLASS,
    # I GUARANTEE I will return a Float32ComputeEnvironment INSTANCE."
    @overload
    def build_and_discover(self, precision_context_class: Type[Float32Context]) -> Float32ComputeEnvironment: ...

    # This second signature makes the same promise for FP16.
    @overload
    def build_and_discover(self, precision_context_class: Type[Float16Context]) -> Float16ComputeEnvironment: ...

    def build_and_discover(self, precision_context_class: Type[PrecisionContext]) -> ComputeEnvironment:
        """
        The primary factory method. It orchestrates the entire bootstrapping
        process, returning a final, immutable, and statically-verified
        compute environment whose concrete type matches the input class.
        """
        try:
            ctx = cl.create_some_context(interactive=False)
        except cl.RuntimeError as e:
            raise RuntimeError(f"FATAL: Could not create OpenCL context: {e}")
        device = ctx.devices[0]
        kernel_files = self._find_kernel_files()
        full_source = self._load_and_concatenate_sources(kernel_files)

        if precision_context_class is Float32Context:
            arch_consts_fp32 = Float32DiscoveredArchConstants(
                simd_width=(device.preferred_vector_width_float or 4),
                global_mem_cacheline_size=device.global_mem_cacheline_size or 64,
                local_mem_size_bytes=device.local_mem_size,
                optimal_workgroup_size_1d_reduction=min(256, device.max_work_group_size),
                optimal_square_tile_dim=int(math.sqrt(device.max_work_group_size)) & ~1,
                optimal_rectangular_tile_dim1=16,
            )
            options = self._get_build_options(arch_consts_fp32)
            try:
                program = cl.Program(ctx, full_source).build(options=options)
            except cl.Error as e:
                self._handle_build_error(e)
            queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
            cl_bundle = CLBundle(context=ctx, queue=queue, program=program)
            return Float32ComputeEnvironment(cl_bundle=cl_bundle, arch_consts=arch_consts_fp32)

        elif precision_context_class is Float16Context:
            # By using a distinct variable name (`arch_consts_fp16`), we create a
            # lexical firewall. This prevents MyPy from creating a problematic
            # Union type and guarantees that the type within this branch is pure.
            arch_consts_fp16 = Float16DiscoveredArchConstants(
                simd_width=(device.preferred_vector_width_half or 4),
                global_mem_cacheline_size=device.global_mem_cacheline_size or 64,
                local_mem_size_bytes=device.local_mem_size,
                optimal_workgroup_size_1d_reduction=min(256, device.max_work_group_size),
                optimal_square_tile_dim=int(math.sqrt(device.max_work_group_size)) & ~1,
                optimal_rectangular_tile_dim1=16,
            )
            options = self._get_build_options(arch_consts_fp16)
            try:
                program = cl.Program(ctx, full_source).build(options=options)
            except cl.Error as e:
                self._handle_build_error(e)
            queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE)
            cl_bundle = CLBundle(context=ctx, queue=queue, program=program)
            # This call is now guaranteed to be type-safe.
            return Float16ComputeEnvironment(cl_bundle=cl_bundle, arch_consts=arch_consts_fp16)
        else:
            raise TypeError(f"Unsupported precision_context_class: {precision_context_class.__name__}")

    def _get_build_options(self, arch_consts: DiscoveredArchConstants) -> List[str]:
        """A helper to centralize the assembly of C-level build flags."""
        options = ["-cl-std=CL1.2"]
        options.append(f"-D SCALAR_TYPE={arch_consts.SCALAR_C_TYPE_NAME}")
        options.append(f"-D SIMD_WIDTH={arch_consts.simd_width}")
        options.append(f"-D C_TILE_SIZE={arch_consts.optimal_square_tile_dim}")
        options.append(f"-D LOCAL_MEM_BANK_PADDING=1")
        if isinstance(arch_consts, Float16DiscoveredArchConstants):
            options.append("-D SCALAR_IS_HALF=1")
            options.append("-D NUMERICAL_STABILITY_EPSILON=9.77e-04h")
            options.extend(["-cl-fp32-correctly-rounded-divide-sqrt", "-D cl_khr_fp16"])
        else:
            options.append("-D SCALAR_IS_HALF=0")
            options.append("-D NUMERICAL_STABILITY_EPSILON=1e-7f")
        return options

    def _handle_build_error(self, e: cl.Error):
        """Provides maximum transparency upon failure, upholding the principle that
        build-time validation prevents runtime chaos."""
        log_header = "\n" + "=" * 80 + "\n--- KERNEL BUILD FAILED ---\n" + "=" * 80
        log_details = ""
        if hasattr(e, "device_logs"):
            log_details = "\n\n".join([f"Device: {dev.name}\n--- Build Log ---\n{log}" for dev, log in e.device_logs])
        else:
            log_details = f"An unexpected OpenCL error occurred: {e}"
        full_error = "\n".join([log_header, log_details, "=" * 80])
        raise RuntimeError(full_error) from e


# =========================================================================
# === Phase 2A: Lightweight context for the new PlanRenderer path
# =========================================================================


@dataclass(frozen=True)
class OpenCLContext:
    """Holds the OpenCL context, queue, device, and compiled program.

    Created by the renderer at initialization. This is the new renderer's
    context object — independent of the legacy ComputeEnvironment.
    """
    context: cl.Context
    queue: cl.CommandQueue
    device: cl.Device
    program: cl.Program
