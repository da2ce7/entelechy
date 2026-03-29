"""CPU kernel shared library loader + layout verification (ADR-014, ADR-015).

Discovers and loads libcpu_kernels via importlib.resources, sets ctypes
function signatures, and verifies struct layout parity.
"""
from __future__ import annotations

import ctypes
import importlib.resources
import pathlib
import platform
from ctypes import CFUNCTYPE, c_size_t, c_uint32, c_void_p

from . import _ffi_types as ffi

_STRUCT_SIZE_FUNCTIONS = [name for name, _ in ffi.LAYOUT_CHECKS]

_TASK_FUNCTION_NAMES = [
    "task_forward_pass",
    "task_render_logits",
    "task_cce_probs_loss",
    "task_bce_probs_loss",
    "task_module_param_grads",
    "task_backprop_to_hidden",
    "task_temp_gradients",
    "task_clip_partial_grads",
    "task_gather_permute_grad_h",
    "task_stabilize_reduce_grad_h",
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
    2. Meson builddir adjacent to the source tree (development)

    Development note: when path (2) is selected, C source edits are picked up
    only after rebuilding the Meson target (for example: ``ninja -C builddir``).
    Reinstalling the Python package alone may leave a stale native library.

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
    2. Meson builddir paths relative to the source tree
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

    # --- Strategy 2: Meson builddir discovery ---
    # Walk up from this file to find the architecture root, then search
    # known builddir locations for the compiled library.
    _this_dir = pathlib.Path(__file__).resolve().parent
    _arch_root = _this_dir.parent.parent.parent  # src/backends/cpu -> arch root
    _candidates = [
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
        "\n".join(f"  - {p}" for p in searched)
    )


def _set_function_signatures(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for all exported C functions."""
    # Thread pool lifecycle
    lib.pool_create.argtypes = [c_uint32]
    lib.pool_create.restype = c_void_p

    lib.pool_destroy.argtypes = [c_void_p]
    lib.pool_destroy.restype = None

    lib.pool_dispatch_and_wait.argtypes = [c_void_p, c_void_p,
                                           c_void_p, c_uint32]
    lib.pool_dispatch_and_wait.restype = None

    # Reduction engine
    lib.execute_reduction_tree.argtypes = [c_void_p, c_void_p]
    lib.execute_reduction_tree.restype = None

    # SIMD width query
    lib.get_simd_width.argtypes = []
    lib.get_simd_width.restype = c_uint32

    # Layout verification functions
    for name in _STRUCT_SIZE_FUNCTIONS:
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = c_size_t

    # Task functions — uniform signature: (void*, uint, uint) → void
    for name in _TASK_FUNCTION_NAMES:
        fn = getattr(lib, name)
        fn.argtypes = [c_void_p, c_uint32, c_uint32]
        fn.restype = None


def _verify_layouts(lib: ctypes.CDLL) -> None:
    """Assert Python struct sizes match C struct sizes (ADR-015).

    Called at library load time. Raises RuntimeError on any mismatch,
    preventing the CPU backend from initializing with a corrupted FFI layer.
    """
    mismatches: list[str] = []
    for c_getter_name, py_struct_cls in ffi.LAYOUT_CHECKS:
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
