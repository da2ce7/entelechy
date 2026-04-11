"""JIT compiler for the CPU kernel shared library (Linux & macOS).

When no pre-built library is found, this module compiles the C kernel
sources on the fly using the system C compiler.  The result is cached
in a per-version directory under the platform cache path so subsequent
imports skip compilation.

Windows is excluded — the pre-built .dll is shipped with the package.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

_KERNEL_SOURCES_DIR = pathlib.Path(__file__).resolve().parent / "kernel_sources"

# Source files that form the compilation unit (order matters: threads first).
_C_SOURCES = [
    _KERNEL_SOURCES_DIR / "cpu_threads.c",
    _KERNEL_SOURCES_DIR / "cpu_kernels.c",
]

# All files whose content contributes to the cache key.
_HASHABLE_FILES = sorted(_KERNEL_SOURCES_DIR.iterdir())


def _source_hash() -> str:
    """SHA-256 of all kernel source files — cache invalidation key."""
    h = hashlib.sha256()
    for p in _HASHABLE_FILES:
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _cache_dir() -> pathlib.Path:
    """Per-version cache directory for compiled libraries.

    Linux:  ~/.cache/aec_cpu_kernels/<version_hash>/
    macOS:  ~/Library/Caches/aec_cpu_kernels/<version_hash>/
    """
    version_hash = _source_hash()

    if platform.system() == "Darwin":
        base = pathlib.Path.home() / "Library" / "Caches"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = pathlib.Path(xdg) if xdg else pathlib.Path.home() / ".cache"

    return base / "aec_cpu_kernels" / version_hash


def _lib_filename() -> str:
    if platform.system() == "Darwin":
        return "libcpu_kernels.dylib"
    return "libcpu_kernels.so"


def _find_compiler() -> str:
    """Find a usable C compiler, preferring CC env var."""
    cc = os.environ.get("CC")
    if cc and shutil.which(cc):
        return cc
    for candidate in ("cc", "gcc", "clang"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "No C compiler found. Install gcc or clang, or set the CC "
        "environment variable."
    )


def _get_march_flags(cc: str) -> list[str]:
    """Determine the best -march flag for the detected compiler."""
    # Check if the compiler supports -march=native
    try:
        result = subprocess.run(
            [cc, "-march=native", "-x", "c", "-E", "-"],
            input=b"",
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return ["-march=native"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def _compile_library() -> pathlib.Path:
    """Compile the kernel sources into a shared library and cache the result."""
    cache = _cache_dir()
    lib_name = _lib_filename()
    cached_lib = cache / lib_name

    if cached_lib.exists():
        logger.debug("Using cached CPU kernel library: %s", cached_lib)
        return cached_lib

    cc = _find_compiler()
    march_flags = _get_march_flags(cc)
    system = platform.system()

    logger.info(
        "Compiling CPU kernel library with %s (this happens once per "
        "source version)...", cc
    )

    # Platform-specific shared library flags
    if system == "Darwin":
        shared_flags = ["-dynamiclib"]
    else:
        shared_flags = ["-shared"]

    # Link flags
    link_flags = ["-lm", "-lpthread"]

    # Compile in a temp directory, atomically move into cache
    with tempfile.TemporaryDirectory(prefix="aec_cpu_build_") as tmpdir:
        tmp_lib = pathlib.Path(tmpdir) / lib_name

        cmd = [
            cc,
            *shared_flags,
            "-fPIC",
            "-O2",
            "-DCPU_KERNELS_BUILDING",
            "-fvisibility=hidden",
            *march_flags,
            f"-I{_KERNEL_SOURCES_DIR}",
            *[str(s) for s in _C_SOURCES],
            "-o", str(tmp_lib),
            *link_flags,
        ]

        logger.debug("Compile command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"CPU kernel compilation failed (exit {result.returncode}):\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr}"
            )

        if result.stderr:
            logger.debug("Compiler warnings:\n%s", result.stderr)

        # Atomic install: write to temp in cache dir, then rename
        cache.mkdir(parents=True, exist_ok=True)
        staging = cached_lib.with_suffix(".tmp")
        shutil.copy2(tmp_lib, staging)
        staging.rename(cached_lib)

    logger.info("Cached compiled library at %s", cached_lib)
    return cached_lib


def try_compile_library() -> pathlib.Path | None:
    """Attempt JIT compilation; return path on success, None on failure.

    This is the entry point called by _loader.py.  It never raises —
    compilation failures are logged and return None so the loader can
    produce a clear "library not found" error.
    """
    if platform.system() == "Windows":
        return None

    if not _KERNEL_SOURCES_DIR.is_dir():
        logger.debug("Kernel sources not found at %s — skipping JIT", _KERNEL_SOURCES_DIR)
        return None

    try:
        return _compile_library()
    except Exception:
        logger.warning(
            "JIT compilation of CPU kernels failed; falling back to "
            "pre-built library discovery.",
            exc_info=True,
        )
        return None
