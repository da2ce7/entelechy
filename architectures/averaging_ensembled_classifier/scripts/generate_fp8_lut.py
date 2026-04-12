#!/usr/bin/env python3
"""Generate FP8 lookup tables for OpenCL and CPU backends (ADR-025 §6.1).

Usage:
    # Generate OpenCL header:
    python scripts/generate_fp8_lut.py --backend opencl \\
        --output kernels/fp8_lut.gen.h

    # Generate CPU header:
    python scripts/generate_fp8_lut.py --backend cpu \\
        --output src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h

Note: The Vulkan backend uses arithmetic conversion (bit manipulation + exp2())
instead of lookup tables.  See Phase 9D Technical Notes for rationale.  Do NOT
add a --backend vulkan option.

Regenerate whenever ml_dtypes version changes.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import ml_dtypes
import numpy as np

# ---------------------------------------------------------------------------
# E5M2 max finite value — used as the defensive fallback for ±infinity
# entries in the LUT.  The store path saturates to this value, so infinity
# bit patterns should never be loaded; the fallback prevents silent
# corruption if a consumer reads one regardless.
# ---------------------------------------------------------------------------
_E5M2_MAX_FINITE: float = 57344.0


# ---------------------------------------------------------------------------
# LUT generation
# ---------------------------------------------------------------------------

def _generate_e4m3_lut() -> list[float]:
    """Generate E4M3 → float32 lookup table (256 entries).

    Each index i ∈ [0, 255] is reinterpreted as a float8_e4m3fn bit pattern
    and converted to its exact float32 representation.  NaN patterns (0x7F,
    0xFF) map to Python ``float('nan')``.
    """
    values: list[float] = []
    for i in range(256):
        arr = np.array([i], dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        values.append(float(arr[0]))
    return values


def _generate_e5m2_lut() -> list[float]:
    """Generate E5M2 → float32 lookup table (256 entries).

    Each index i ∈ [0, 255] is reinterpreted as a float8_e5m2 bit pattern
    and converted to its exact float32 representation.  Infinity and NaN
    patterns (0x7C–0x7F, 0xFC–0xFF) map to the corresponding Python special
    float values.
    """
    values: list[float] = []
    for i in range(256):
        arr = np.array([i], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        values.append(float(arr[0]))
    return values


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_tables(e4m3: list[float], e5m2: list[float]) -> None:
    """Validate LUT correctness against known reference values.

    Assertions cover:
      - E4M3: zero, ±1.0, ±max (448), NaN patterns
      - E5M2: zero, ±1.0, ±max finite (57344), ±inf, NaN patterns
    """
    # --- E4M3 positive values ---
    assert e4m3[0x00] == 0.0, \
        f"E4M3[0x00]: expected 0.0, got {e4m3[0x00]}"
    assert abs(e4m3[0x38] - 1.0) < 1e-6, \
        f"E4M3[0x38]: expected 1.0, got {e4m3[0x38]}"
    assert abs(e4m3[0x7E] - 448.0) < 1e-6, \
        f"E4M3[0x7E]: expected 448.0 (max), got {e4m3[0x7E]}"

    # --- E4M3 NaN patterns ---
    assert math.isnan(e4m3[0x7F]), \
        f"E4M3[0x7F]: expected NaN, got {e4m3[0x7F]}"
    assert math.isnan(e4m3[0xFF]), \
        f"E4M3[0xFF]: expected NaN, got {e4m3[0xFF]}"

    # --- E4M3 negative values (copysign verifies the sign of −0.0) ---
    assert e4m3[0x80] == 0.0 and math.copysign(1.0, e4m3[0x80]) == -1.0, \
        f"E4M3[0x80]: expected -0.0, got {e4m3[0x80]}"
    assert abs(e4m3[0xB8] - (-1.0)) < 1e-6, \
        f"E4M3[0xB8]: expected -1.0, got {e4m3[0xB8]}"
    assert abs(e4m3[0xFE] - (-448.0)) < 1e-6, \
        f"E4M3[0xFE]: expected -448.0 (min), got {e4m3[0xFE]}"

    # --- E5M2 positive values ---
    assert e5m2[0x00] == 0.0, \
        f"E5M2[0x00]: expected 0.0, got {e5m2[0x00]}"
    assert abs(e5m2[0x3C] - 1.0) < 1e-6, \
        f"E5M2[0x3C]: expected 1.0, got {e5m2[0x3C]}"
    assert abs(e5m2[0x7B] - 57344.0) < 1e-6, \
        f"E5M2[0x7B]: expected 57344.0 (max finite), got {e5m2[0x7B]}"

    # --- E5M2 special values ---
    assert math.isinf(e5m2[0x7C]) and e5m2[0x7C] > 0, \
        f"E5M2[0x7C]: expected +inf, got {e5m2[0x7C]}"
    assert math.isnan(e5m2[0x7D]), \
        f"E5M2[0x7D]: expected NaN, got {e5m2[0x7D]}"
    assert math.isinf(e5m2[0xFC]) and e5m2[0xFC] < 0, \
        f"E5M2[0xFC]: expected -inf, got {e5m2[0xFC]}"
    assert math.isnan(e5m2[0xFD]), \
        f"E5M2[0xFD]: expected NaN, got {e5m2[0xFD]}"

    # --- E5M2 negative values ---
    assert e5m2[0x80] == 0.0 and math.copysign(1.0, e5m2[0x80]) == -1.0, \
        f"E5M2[0x80]: expected -0.0, got {e5m2[0x80]}"
    assert abs(e5m2[0xBC] - (-1.0)) < 1e-6, \
        f"E5M2[0xBC]: expected -1.0, got {e5m2[0xBC]}"
    assert abs(e5m2[0xFB] - (-57344.0)) < 1e-6, \
        f"E5M2[0xFB]: expected -57344.0 (min finite), got {e5m2[0xFB]}"

    print("LUT validation passed.")
    print("  Note: LUT generation maps inf → ±57344.0f, NaN → 0.0f for C")
    print("  compatibility.  The store path never writes non-finite patterns,")
    print("  so these entries should never be loaded at runtime.")


# ---------------------------------------------------------------------------
# C / OpenCL formatting
# ---------------------------------------------------------------------------

def _format_c_float(v: float) -> str:
    """Format a Python float as a C float literal.

    Special-value mapping (defensive; the store path never writes these):
      - NaN  → ``0.0f``         (safe zero — no parameter update)
      - ±inf → ``±57344.0f``    (E5M2 max finite — saturation fallback)
    """
    if math.isnan(v):
        return "0.0f"
    if math.isinf(v):
        sign = "-" if v < 0 else ""
        return f"{sign}{_E5M2_MAX_FINITE:.1f}f"
    return f"{v:.10e}f"


def _format_c_array(
    name: str,
    values: list[float],
    *,
    is_opencl: bool,
) -> str:
    """Format a 256-entry lookup table as a C/OpenCL constant array."""
    qualifier = "__constant" if is_opencl else "static const"
    rows: list[str] = []
    for i in range(0, 256, 8):
        row = ", ".join(_format_c_float(v) for v in values[i : i + 8])
        rows.append(f"    {row}")
    body = ",\n".join(rows)  # Join rows with commas between (none trailing)
    return f"{qualifier} float {name}[256] = {{\n{body}\n}};"


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------

def _get_ml_dtypes_version() -> str:
    """Return the installed ml_dtypes version string."""
    try:
        return ml_dtypes.__version__
    except AttributeError:
        return "unknown (pre-0.2.0)"


def _generate_header(
    e4m3_values: list[float],
    e5m2_values: list[float],
    *,
    is_opencl: bool,
    output_path: Path,
) -> None:
    """Generate a complete C/OpenCL header file with both lookup tables."""
    guard = "FP8_LUT_OPENCL_GEN_H" if is_opencl else "FP8_LUT_CPU_GEN_H"
    array_prefix = "" if is_opencl else "cpu_"
    backend_name = "opencl" if is_opencl else "cpu"
    ml_version = _get_ml_dtypes_version()

    header = f"""\
/* Auto-generated by scripts/generate_fp8_lut.py — DO NOT EDIT
 * ml_dtypes version: {ml_version}
 * Regenerate with: python scripts/generate_fp8_lut.py --backend {backend_name} --output {output_path}
 */

#ifndef {guard}
#define {guard}

/* E4M3: sign(1) + exp(4) + mantissa(3), bias=7, max=448, no inf
 * Note: float8_e4m3fn has 2 NaN patterns (0x7F, 0xFF) and 254 finite values.
 * The store path never writes NaN patterns; the LUT maps them to 0.0f defensively.
 *
 * Index derivation examples:
 *   0x00 = 0b00000000 → sign=0, exp=0, mant=0 → subnormal 0 × 2^(-6) = 0.0
 *   0x38 = 0b00111000 → sign=0, exp=7, mant=0 → 2^(7-7) × 1.0 = 1.0
 *   0x3C = 0b00111100 → sign=0, exp=7, mant=4 → 2^(7-7) × 1.5 = 1.5
 *   0x7E = 0b01111110 → sign=0, exp=15, mant=6 → 2^(15-7) × 1.75 = 448.0 (max)
 *   0x7F = 0b01111111 → NaN (mapped to 0.0f in LUT)
 *   0x80 = 0b10000000 → sign=1, exp=0, mant=0 → -0.0
 *   0xB8 = 0b10111000 → sign=1, exp=7, mant=0 → -1.0
 *   0xFE = 0b11111110 → sign=1, exp=15, mant=6 → -448.0 (min)
 *   0xFF = 0b11111111 → NaN (mapped to 0.0f in LUT)
 */
{_format_c_array(f"{array_prefix}fp8_e4m3_to_float_lut", e4m3_values, is_opencl=is_opencl)}

/* E5M2: sign(1) + exp(5) + mantissa(2), bias=15, max=57344, IEEE-like inf/NaN
 *
 * Unlike E4M3fn, E5M2 follows IEEE conventions for exponent 0x1F (all ones):
 *   mantissa=0 → ±infinity, mantissa≠0 → NaN.
 * This means 8 bit patterns are non-finite: 0x7C–0x7F (+inf/NaN), 0xFC–0xFF (−inf/NaN).
 * The store path never writes these patterns (it saturates to 0x7B/0xFB),
 * but the LUT includes them for completeness — callers should not rely on
 * loading non-finite values from FP8 buffers.
 *
 * Index derivation examples:
 *   0x00 = 0b00000000 → sign=0, exp=0, mant=0 → subnormal 0 × 2^(-14) = 0.0
 *   0x3C = 0b00111100 → sign=0, exp=15, mant=0 → 2^(15-15) × 1.0 = 1.0
 *   0x7B = 0b01111011 → sign=0, exp=30, mant=3 → 2^(30-15) × 1.75 = 57344.0 (max finite)
 *   0x7C = 0b01111100 → sign=0, exp=31, mant=0 → +infinity
 *   0x7D = 0b01111101 → sign=0, exp=31, mant=1 → NaN
 *   0x80 = 0b10000000 → sign=1, exp=0, mant=0 → -0.0
 *   0xBC = 0b10111100 → sign=1, exp=15, mant=0 → -1.0
 *   0xFB = 0b11111011 → sign=1, exp=30, mant=3 → -57344.0 (min finite)
 */
{_format_c_array(f"{array_prefix}fp8_e5m2_to_float_lut", e5m2_values, is_opencl=is_opencl)}

#endif /* {guard} */
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header)
    print(f"Generated: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FP8 lookup tables for OpenCL and CPU backends.",
    )
    parser.add_argument(
        "--backend",
        choices=["opencl", "cpu"],
        required=True,
        help="Target backend: opencl or cpu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for generated header",
    )
    args = parser.parse_args()

    # --- Generate raw tables from ml_dtypes ---
    e4m3 = _generate_e4m3_lut()
    e5m2 = _generate_e5m2_lut()

    # --- Validate against known reference values ---
    _validate_tables(e4m3, e5m2)

    # --- Emit backend-specific header ---
    _generate_header(
        e4m3,
        e5m2,
        is_opencl=(args.backend == "opencl"),
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
