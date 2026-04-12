"""CPU kernel shared library loader + layout verification (ADR-014, ADR-015).

Discovers and loads libcpu_kernels via importlib.resources, sets ctypes
function signatures, and verifies struct layout parity for all precisions.

On Linux and macOS, if no pre-built library is found, the loader
invokes JIT compilation via _compiler.py (gcc/clang + -march=native).

Set ``AEC_CPU_BUILDDIR`` to an absolute path to force loading from a
specific Meson builddir (e.g. ``builddir-asan`` for sanitizer builds).
"""
from __future__ import annotations

import ctypes
import importlib.resources
import logging
import os
import pathlib
import platform
from ctypes import CFUNCTYPE, c_size_t, c_uint32, c_void_p

from . import _ffi_types as ffi

logger = logging.getLogger(__name__)

_TASK_FUNCTION_NAMES = [
    "task_forward_pass",
    "task_render_logits",
    "task_cce_probs_loss",
    "task_bce_probs_loss",
    "task_module_param_grads",
    "task_backprop_to_hidden",
    "task_temp_gradients",
    "task_clip_partial_grads",
    "task_gather_permute_grad_hidden_activations",
    "task_stabilize_reduce_grad_hidden_activations",
    "task_clip_intermediate",
    "task_backprop_shared_weights",
    "task_backprop_shared_biases",
    "task_clip_shared_grads",
    "task_normalize_gradients",
    "task_adam_update",
    "task_clamp_temperatures",
]

# The canonical task function type for pool_dispatch_and_wait callbacks
TASK_FUNC_TYPE = CFUNCTYPE(None, c_void_p, c_uint32, c_uint32)


def _library_filename() -> str:
    """Resolve the platform-specific shared library filename."""
    system = platform.system()
    if system == "Linux":
        return "libcpu_kernels.so"
    elif system == "Darwin":
        return "libcpu_kernels.dylib"
    elif system == "Windows":
        return "cpu_kernels.dll"
    raise RuntimeError(f"Unsupported platform for CPU backend: {system}")


def load_cpu_library() -> ctypes.CDLL:
    """Load the CPU kernel shared library.

    Discovery order:
    1. importlib.resources (installed package)
    2. AEC_CPU_BUILDDIR override (debug/sanitizer builds)
    3. JIT compilation via system compiler (Linux/macOS only)
    4. Meson builddir adjacent to the source tree (development)

    Sets ctypes function signatures and runs layout verification.
    Raises RuntimeError on load failure or layout mismatch.
    """
    lib_name = _library_filename()
    lib_path = _find_library(lib_name)
    lib = ctypes.CDLL(str(lib_path))

    _set_function_signatures(lib)
    _verify_layouts(lib)
    return lib


def _find_library(lib_name: str) -> pathlib.Path:
    """Locate the shared library file.

    1. importlib.resources (works for pip-installed package)
    2. AEC_CPU_BUILDDIR override (debug/sanitizer builds)
    3. JIT compilation via system compiler (Linux/macOS)
    4. Meson builddir paths relative to the source tree
    """
    # --- Strategy 1: importlib.resources ---
    try:
        package_files = importlib.resources.files(
            "averaging_ensembled_classifier.backends.cpu"
        )
        lib_resource = package_files / lib_name
        with importlib.resources.as_file(lib_resource) as resolved:
            if resolved.exists():
                return resolved
    except Exception:
        pass

    _this_dir = pathlib.Path(__file__).resolve().parent
    _arch_root = _this_dir.parent.parent.parent  # src/backends/cpu -> arch root

    # --- Strategy 2: AEC_CPU_BUILDDIR override ---
    # Explicit builddir selection (e.g. builddir-asan for debug/sanitizer).
    _override = os.environ.get("AEC_CPU_BUILDDIR")
    if _override:
        override_dir = pathlib.Path(_override)
        if not override_dir.is_absolute():
            override_dir = _arch_root / override_dir
        override_lib = override_dir / "src" / "backends" / "cpu" / lib_name
        if override_lib.exists():
            return override_lib

    # --- Strategy 3: JIT compilation (Linux/macOS) ---
    from ._compiler import try_compile_library

    jit_path = try_compile_library()
    if jit_path is not None:
        return jit_path

    # --- Strategy 4: Meson builddir discovery (development fallback) ---
    _candidates: list[pathlib.Path] = [
        _arch_root / "builddir" / "src" / "backends" / "cpu" / lib_name,
    ]
    # meson-python editable builds use build/cp*/ directories
    build_dir = _arch_root / "build"
    if build_dir.is_dir():
        for child in build_dir.iterdir():
            if child.is_dir() and child.name.startswith("cp"):
                _candidates.append(
                    child / "src" / "backends" / "cpu" / lib_name
                )

    for candidate in _candidates:
        if candidate.exists():
            return candidate

    searched = [str(c) for c in _candidates]
    raise RuntimeError(
        f"Could not find {lib_name}. Searched:\n" +
        "\n".join(f"  - {p}" for p in searched) +
        "\nJIT compilation also failed (see log for details)."
    )


def _set_function_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for all exported C functions."""
    # Thread pool lifecycle (precision-agnostic)
    lib.pool_create.argtypes = [c_uint32]
    lib.pool_create.restype = c_void_p

    lib.pool_destroy.argtypes = [c_void_p]
    lib.pool_destroy.restype = None

    lib.pool_dispatch_and_wait.argtypes = [c_void_p, c_void_p,
                                           c_void_p, c_uint32]
    lib.pool_dispatch_and_wait.restype = None

    # SIMD width query (precision-agnostic)
    lib.get_simd_width.argtypes = []
    lib.get_simd_width.restype = c_uint32

    # Per-precision exports: task functions, reduction engine, layout getters
    for suffix in ffi.ALL_PRECISION_SUFFIXES:
        # Reduction engine — storage-entry variant
        fn = getattr(lib, f"execute_reduction_tree_{suffix}")
        fn.argtypes = [c_void_p, c_void_p]
        fn.restype = None

        # ADR-026: Reduction engine — compute-entry variant
        fn = getattr(lib, f"execute_reduction_tree_from_compute_{suffix}")
        fn.argtypes = [c_void_p, c_void_p]
        fn.restype = None

        # Layout verification functions
        for getter_name, _ in ffi.PRECISION_LAYOUT_CHECKS[suffix]:
            fn = getattr(lib, getter_name)
            fn.argtypes = []
            fn.restype = c_size_t

        # Task functions — uniform signature: (void*, uint, uint) → void
        for name in _TASK_FUNCTION_NAMES:
            fn = getattr(lib, f"{name}_{suffix}")
            fn.argtypes = [c_void_p, c_uint32, c_uint32]
            fn.restype = None


def _verify_layouts(lib: ctypes.CDLL) -> None:
    """Assert Python struct sizes match C struct sizes for all precisions (ADR-015).

    Called at library load time. Raises RuntimeError on any mismatch,
    preventing the CPU backend from initializing with a corrupted FFI layer.
    """
    mismatches: list[str] = []
    for suffix in ffi.ALL_PRECISION_SUFFIXES:
        for c_getter_name, py_struct_cls in ffi.PRECISION_LAYOUT_CHECKS[suffix]:
            c_size = getattr(lib, c_getter_name)()
            py_size = ctypes.sizeof(py_struct_cls)
            if c_size != py_size:
                mismatches.append(
                    f"  {py_struct_cls.__name__}: C={c_size}, Python={py_size}"
                )

    if mismatches:
        detail = "\n".join(mismatches)
        raise RuntimeError(
            f"CPU kernel library struct layout mismatch:\n{detail}\n"
            f"Update _ffi_types.py to match cpu_kernels.h."
        )
