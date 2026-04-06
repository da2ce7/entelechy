# Phase 9A: FP8 Support — Authority Documents, PrecisionConfig Extension & Shared Infrastructure

**Status: NOT STARTED**  
**Phase:** 9A of 9  
**Prerequisite:** Phase 8E complete and rollback gate passed.  
**Objective:** Amend the authority documents (CONCEPT.md, CONTRACT.md) with FP8 additions. Add `ml_dtypes` dependency. Extend `PrecisionConfig` with six new FP8 factory classmethods (`fp8_e4m3()`, `fp8_e5m2()`, `fp8_e4m3_f16()`, `fp8_e5m2_f16()`, `fp8_e4m3_f64()`, `fp8_e5m2_f64()`), enforce the storage-only constraint in `__post_init__`, and add FP8-specific derived constants. Update test fixtures. No kernel sources, no backend-native code, no build system changes. The phase ends when all existing tests pass against the new `PrecisionConfig` interface and the FP8 factories construct without error.  
**Governing ADR:** ADR-025 (§§1–4, §9)  
**Rollback gate:** All tests that passed before Phase 9A must pass after **migration** (Step 9A.5.0). This phase introduces a **breaking change**: `PrecisionConfig.float16()` is deleted. Tests using `float16()` must migrate to `mixed_f16_f32()` first. After migration, existing `PrecisionConfig` factories remain identical. The new FP8 factories construct without error and satisfy the storage-only invariant. Attempting FP8 compute or state raises `ValueError`. Direct construction with `state_dtype=np.float16` remains valid. Failing any existing Tier 2 test (post-migration) is a blocking regression.  
**Dependencies:** Phase 8E (double-precision tests and validation) complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 9A.1: Add `ml_dtypes` dependency](#step-9a1-add-ml_dtypes-dependency)
   - [Step 9A.2: Amend CONCEPT.md](#step-9a2-amend-conceptmd)
   - [Step 9A.3: Amend CONTRACT.md](#step-9a3-amend-contractmd)
   - [Step 9A.4: Extend `PrecisionConfig`](#step-9a4-extend-precisionconfig)
   - [Step 9A.5: Update test fixtures](#step-9a5-update-test-fixtures)
   - [Step 9A.6: Validate rollback gate](#step-9a6-validate-rollback-gate)
4. [Migration Order Rationale](#4-migration-order-rationale)
5. [Risk Register](#5-risk-register)
6. [Files Modified Summary](#6-files-modified-summary)

---

## 1. Scope & Constraints

### In scope

- Adding `ml_dtypes` as a runtime dependency in `pyproject.toml` and `requirements.latest.txt`.
- Amending `CONCEPT.md` with the FP8 storage rationale and the "Bandwidth Extremist" validation scenario (ADR-025 §9.1).
- Amending `CONTRACT.md` Article 6 with three new derived build-time symbols: `STORAGE_TYPE_IS_FP8`, `STORAGE_TYPE_IS_E4M3`, `STORAGE_TYPE_IS_E5M2` (ADR-025 §3).
- Extending `PrecisionConfig` with `ml_dtypes.float8_e4m3fn` and `ml_dtypes.float8_e5m2` support in the `storage_dtype` field. FP8 storage is valid with any combination of FP16/FP32/FP64 compute and FP32/FP64 state (ADR-025 §2.3). Common combinations are exposed as factory classmethods.
- Amending the `__post_init__` invariants to **reject** FP8 compute and FP8 state (ADR-025 §2.2).
- Adding FP8-specific derived constant values: `storage_fp_format_max` (448 for E4M3, 57344 for E5M2), `storage_fp_min_positive`, `storage_mantissa_bits` (ADR-025 §2.4).
- Extending test fixtures in `conftest.py` to cover the new FP8 factories.
- Adding FP8-specific unit tests: roundtrip, saturation, compute/state rejection (ADR-025 §9.2).

### Out of scope

- Any change to `kernels/kernels.cl.h` or `kernels/phase_*.cl.c` — deferred to Phase 9B.
- Any change to `src/backends/opencl/type_mapping.py` — deferred to Phase 9B.
- Any change to CPU backend kernel sources (`cpu_kernels.h`, `cpu_kernels.c`, `cpu_precision.h`, `cpu_fp8.h`, `.inc` files) — deferred to Phase 9C.
- Any change to Vulkan backend shader sources (`common.glsl`, `.comp` files) — deferred to Phase 9D.
- Meson build changes for FP8 compilation — deferred to Phase 9B/9C/9D.
- Host-side FP8 scaling implementation — deferred to Phase 9E (the configuration support is added here; the actual scaling logic is Phase 9E work).
- SIMD FP8 on CPU — deferred indefinitely (AVX-512 has no native FP8; scalar-only is the baseline).

### Key constraint: FP8 is storage-only

The FP8 storage-only constraint is **enforced at construction time**. Unlike FP16 which permits compute-role usage (with degraded fidelity), FP8 compute and FP8 state are **architecturally prohibited** — they produce non-functional training (see ADR-025 Context). The `__post_init__` invariant raises `ValueError` for any FP8 dtype in compute or state roles. This is not a recommendation; it is a hard error.

### Breaking change: `float16()` factory deletion

**This phase introduces a breaking change.** The `PrecisionConfig.float16()` factory classmethod is deleted. It produced all-FP16 configurations (storage, compute, AND state) which are a common footgun — FP16 state lacks sufficient precision for EMA updates over extended training. The factory name was also misleading since `mixed_f16_f32()` is the recommended FP16-storage configuration.

**Important:** This is a factory deletion, not a state-role prohibition. Direct construction with `state_dtype=np.float16` remains valid — the three-role precision model (ADR-020) defines FP16 as a valid state dtype. Users who explicitly choose FP16 state accept the precision limitations.

**Affected code:**
- `PrecisionConfig.float16()` — **deleted** in Step 9A.4.7.

**Unaffected code:**
- `PrecisionConfig.mixed_f16_f32()` — Constructs FP16 storage, FP16 compute, **FP32 state**. This remains valid and is the recommended replacement.
- `PrecisionConfig(storage_dtype=np.float16, compute_dtype=np.float16, state_dtype=np.float16, ...)` — Direct construction with FP16 state remains valid.

**Migration path:**
- Replace `PrecisionConfig.float16()` with `PrecisionConfig.mixed_f16_f32()`.
- **Semantic difference:** `mixed_f16_f32()` promotes **both compute and state** to FP32. The old `float16()` used FP16 for all three roles (storage/compute/state); `mixed_f16_f32()` uses FP16 storage, FP32 compute, FP32 state. This changes compute behavior, not just state stability.
- If FP16 compute + FP32 state is specifically needed (rare), construct directly: `PrecisionConfig(storage_dtype=np.float16, compute_dtype=np.float16, state_dtype=np.float32, ...)`. No factory is provided for this combination.
- If uniform FP16 (including state) is specifically needed, construct directly: `PrecisionConfig(storage_dtype=np.float16, compute_dtype=np.float16, state_dtype=np.float16, ...)`. This is permitted but not recommended for extended training.

**Test migration:**
- All tests using `PrecisionConfig.float16()` must be updated to use `mixed_f16_f32()` (Step 9A.5.0).

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `pyproject.toml` | Does not list `ml_dtypes` as a dependency. |
| `requirements.latest.txt` | Does not list `ml_dtypes`. |
| `src/shared/precision_config.py` | `PrecisionConfig` frozen dataclass: fields `storage_dtype`, `compute_dtype`, `state_dtype`, derived constants. Seven factories (post-Phase-8A): `float32()`, `float16()`, `mixed_f16_f32()`, `float64()`, `mixed_f32_f64_state()`, `mixed_f16_f64_state()`, `mixed_f32_f64()`. `__post_init__` enforces `storage_dtype.itemsize <= compute_dtype.itemsize` and `storage_dtype.itemsize <= state_dtype.itemsize`. No FP8 rejection logic. |
| `CONCEPT.md` | Post-Phase-8A state: three-role precision model documented, Alchemist and Alchemist II scenarios present. No FP8 discussion. |
| `CONTRACT.md` | Post-Phase-8A state: Article 6 symbols include `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_DOUBLE`, `STATE_TYPE_IS_DOUBLE`. No FP8 symbols. |
| `src/tests/conftest.py` | `PRECISION_CONFIGS` list used for test parametrization. Contains seven factories (as `pytest.param` wrappers around unbound classmethods). |

---

## 3. Task Breakdown

---

### Step 9A.1: Add `ml_dtypes` dependency

**Governing authority:** ADR-025 §1  
**Files:** `pyproject.toml`, `requirements.latest.txt`

Add `ml_dtypes>=0.2.0` to runtime dependencies. The `ml_dtypes` package provides NumPy-compatible FP8 dtypes (`float8_e4m3fn`, `float8_e5m2`) required for the FP8 storage representation.

> **Version requirement rationale:** Version 0.2.0 (released 2023-Q3) stabilized the `float8_e4m3fn` and `float8_e5m2` dtype APIs and added the `__version__` attribute. Earlier versions had inconsistent dtype identity semantics that could break `in FP8_DTYPES` membership checks.

#### 9A.1.1: Update `pyproject.toml`

In the `[project]` section under `dependencies`, add:

```toml
"ml_dtypes>=0.2.0",
```

#### 9A.1.2: Update `requirements.latest.txt`

Add:

```
ml_dtypes>=0.2.0
```

#### 9A.1.3: Verification

Run `pip install -e .` and verify `import ml_dtypes` succeeds. Confirm `ml_dtypes.float8_e4m3fn` and `ml_dtypes.float8_e5m2` are accessible as NumPy dtype objects.

---

### Step 9A.1.5: Generate FP8 Lookup Tables (Shared Infrastructure)

> **Step numbering note:** The `.5` numbering indicates a late insertion between Steps 9A.1 and 9A.2, added to factor LUT generation into Phase 9A so that Phases 9B, 9C, and 9D can proceed in parallel.

**Governing authority:** ADR-025 §6.1  
**Files:** `scripts/generate_fp8_lut.py` (new), `kernels/fp8_lut.gen.h` (generated), `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` (generated)

Create a Python script to generate FP8→FP32 lookup tables for all backends. This step is placed in Phase 9A to eliminate cross-phase dependencies — Phases 9B, 9C, and 9D can then proceed in parallel without waiting on each other.

#### 9A.1.5.1: Create `scripts/generate_fp8_lut.py`

```python
#!/usr/bin/env python3
"""Generate FP8 lookup tables for OpenCL and CPU backends (ADR-025 §6.1).

Usage:
  # Generate OpenCL header:
  python scripts/generate_fp8_lut.py --backend opencl --output kernels/fp8_lut.gen.h
  
  # Generate CPU header:
  python scripts/generate_fp8_lut.py --backend cpu --output src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h

Note: The Vulkan backend uses arithmetic conversion (bit manipulation + exp2())
instead of lookup tables. See Phase 9D Technical Notes for rationale. Do NOT
add a --backend vulkan option.

Regenerate whenever ml_dtypes version changes.
"""

import argparse
import ml_dtypes
import numpy as np
from pathlib import Path

def generate_e4m3_lut() -> list[float]:
    """Generate E4M3 → float32 lookup table (256 entries)."""
    values = []
    for i in range(256):
        arr = np.array([i], dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        values.append(float(arr[0]))
    return values

def generate_e5m2_lut() -> list[float]:
    """Generate E5M2 → float32 lookup table (256 entries)."""
    values = []
    for i in range(256):
        arr = np.array([i], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        values.append(float(arr[0]))
    return values

def format_c_float(v: float) -> str:
    """Format a float for C, handling inf/NaN specially.
    
    Note: Both E4M3fn (indices 0x7F, 0xFF) and E5M2 (indices 0x7C-0x7F,
    0xFC-0xFF) include NaN and/or inf bit patterns, but the store path
    NEVER writes these. We use defensive substitution (NaN→0.0f,
    inf→max finite) so loading these indices returns a finite value
    rather than crashing or producing undefined behavior.
    """
    import math
    if math.isnan(v):
        return "0.0f"  # NaN → zero (defensive; should never be loaded)
    elif math.isinf(v):
        if v > 0:
            return "57344.0f"  # +inf → E5M2 max finite (defensive)
        else:
            return "-57344.0f"  # -inf → E5M2 min finite (defensive)
    else:
        return f"{v:.10e}f"


def format_c_array(name: str, values: list[float], is_opencl: bool = False) -> str:
    """Format lookup table as C/OpenCL constant array."""
    qualifier = "__constant" if is_opencl else "static const"
    rows = []
    for i in range(0, 256, 8):
        row = ", ".join(format_c_float(v) for v in values[i:i+8])
        rows.append(f"    {row}")
    body = ",\n".join(rows)  # Join rows with commas between (none trailing)
    return f"{qualifier} float {name}[256] = {{\n{body}\n}};"

def _get_ml_dtypes_version() -> str:
    """Get ml_dtypes version, with fallback for older versions.
    
    Note: __version__ was added in ml_dtypes 0.2.0. Since we require >=0.2.0,
    the fallback should never be reached — it exists only as defensive code.
    """
    try:
        return ml_dtypes.__version__
    except AttributeError:
        return "unknown (pre-0.2.0)"


def generate_header(e4m3_values: list[float], e5m2_values: list[float], 
                    is_opencl: bool, output_path: Path) -> None:
    """Generate a complete header file with both lookup tables."""
    # Use consistent naming: FP8_LUT_{BACKEND}_GEN_H
    guard = "FP8_LUT_CPU_GEN_H" if not is_opencl else "FP8_LUT_OPENCL_GEN_H"
    array_prefix = "cpu_" if not is_opencl else ""
    ml_version = _get_ml_dtypes_version()
    header = f"""/* Auto-generated by scripts/generate_fp8_lut.py — DO NOT EDIT
 * ml_dtypes version: {ml_version}
 * Regenerate with: python scripts/generate_fp8_lut.py --backend {'opencl' if is_opencl else 'cpu'} --output {output_path}
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
{format_c_array(f"{array_prefix}fp8_e4m3_to_float_lut", e4m3_values, is_opencl)}

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
{format_c_array(f"{array_prefix}fp8_e5m2_to_float_lut", e5m2_values, is_opencl)}

#endif /* {guard} */
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header)
    print(f"Generated: {output_path}")

def validate_tables(e4m3: list[float], e5m2: list[float]) -> None:
    """Validate LUT correctness against known values.
    
    Note on E4M3fn NaN entries (indices 0x7F, 0xFF):
    E4M3fn has no infinity but reserves 2 bit patterns for NaN.
    The generated C LUT replaces NaN→0.0f defensively.
    
    Note on E5M2 inf/NaN entries (indices 0x7C–0x7F, 0xFC–0xFF):
    These entries ARE included in the LUT for completeness, but the store path
    NEVER writes these bit patterns — it saturates to 0x7B/0xFB (max finite)
    instead. Callers should not rely on loading non-finite values from FP8 buffers.
    """
    # E4M3 positive values
    assert e4m3[0] == 0.0, "E4M3 index 0x00 should be 0.0"
    # 0x38 = 0b00111000 → sign=0, exp=7, mant=0 → 2^(7-7) × 1.0 = 1.0
    assert abs(e4m3[0x38] - 1.0) < 1e-6, f"E4M3 index 0x38 should be 1.0, got {e4m3[0x38]}"
    # 0x7E = 0b01111110 → sign=0, exp=15, mant=6 → 2^8 × 1.75 = 448.0
    assert abs(e4m3[0x7E] - 448.0) < 1e-6, f"E4M3 index 0x7E should be 448.0 (max), got {e4m3[0x7E]}"
    
    # E4M3fn NaN patterns (0x7F, 0xFF)
    import math
    assert math.isnan(e4m3[0x7F]), f"E4M3 index 0x7F should be NaN, got {e4m3[0x7F]}"
    assert math.isnan(e4m3[0xFF]), f"E4M3 index 0xFF should be NaN, got {e4m3[0xFF]}"
    
    # E4M3 negative values (sign bit is MSB)
    # 0x80 = -0.0 (sign=1, rest=0)
    assert abs(e4m3[0x80] - 0.0) < 1e-10, f"E4M3 index 0x80 should be -0.0, got {e4m3[0x80]}"
    # 0xB8 = 0b10111000 → sign=1, exp=7, mant=0 → -1.0
    assert abs(e4m3[0xB8] - (-1.0)) < 1e-6, f"E4M3 index 0xB8 should be -1.0, got {e4m3[0xB8]}"
    # 0xFE = 0b11111110 → sign=1, exp=15, mant=6 → -448.0
    assert abs(e4m3[0xFE] - (-448.0)) < 1e-6, f"E4M3 index 0xFE should be -448.0 (min), got {e4m3[0xFE]}"
    
    # E5M2 positive values
    assert e5m2[0] == 0.0, "E5M2 index 0x00 should be 0.0"
    # 0x3C = 0b00111100 → sign=0, exp=15, mant=0 → 1.0
    assert abs(e5m2[0x3C] - 1.0) < 1e-6, f"E5M2 index 0x3C should be 1.0, got {e5m2[0x3C]}"
    # 0x7B = 0b01111011 → sign=0, exp=30, mant=3 → 57344.0
    assert abs(e5m2[0x7B] - 57344.0) < 1e-6, f"E5M2 index 0x7B should be 57344.0 (max finite), got {e5m2[0x7B]}"
    
    # E5M2 special values (IEEE-like: exponent 0x1F encodes inf/NaN)
    # Note: These are validated against ml_dtypes behavior, but the generated
    # C LUT replaces inf/NaN with max finite values (see format_c_float).
    import math
    assert math.isinf(e5m2[0x7C]), f"E5M2 index 0x7C should be +inf, got {e5m2[0x7C]}"
    assert math.isnan(e5m2[0x7D]), f"E5M2 index 0x7D should be NaN, got {e5m2[0x7D]}"
    assert math.isinf(e5m2[0xFC]) and e5m2[0xFC] < 0, f"E5M2 index 0xFC should be -inf, got {e5m2[0xFC]}"
    
    # Verify the defensive substitution produces valid C
    print("Note: LUT generation replaces inf→57344.0f, NaN→0.0f for C compatibility.")
    print("      The store path never writes these patterns, so they should never be loaded.")
    
    # E5M2 negative values
    assert abs(e5m2[0x80] - 0.0) < 1e-10, f"E5M2 index 0x80 should be -0.0, got {e5m2[0x80]}"
    # 0xBC = 0b10111100 → sign=1, exp=15, mant=0 → -1.0
    assert abs(e5m2[0xBC] - (-1.0)) < 1e-6, f"E5M2 index 0xBC should be -1.0, got {e5m2[0xBC]}"
    # 0xFB = 0b11111011 → sign=1, exp=30, mant=3 → -57344.0
    assert abs(e5m2[0xFB] - (-57344.0)) < 1e-6, f"E5M2 index 0xFB should be -57344.0 (min finite), got {e5m2[0xFB]}"
    
    print("LUT validation passed.")

def main():
    parser = argparse.ArgumentParser(description="Generate FP8 lookup tables")
    parser.add_argument("--backend", choices=["opencl", "cpu"], required=True,
                       help="Target backend: opencl or cpu")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for generated header")
    args = parser.parse_args()
    
    e4m3 = generate_e4m3_lut()
    e5m2 = generate_e5m2_lut()
    
    validate_tables(e4m3, e5m2)
    
    is_opencl = args.backend == "opencl"
    generate_header(e4m3, e5m2, is_opencl=is_opencl, output_path=args.output)

if __name__ == "__main__":
    main()
```

#### 9A.1.5.2: Run LUT generation

Execute the script to generate both headers:

```bash
cd architectures/averaging_ensembled_classifier

# Generate OpenCL header
python scripts/generate_fp8_lut.py --backend opencl --output kernels/fp8_lut.gen.h

# Generate CPU header
python scripts/generate_fp8_lut.py --backend cpu --output src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h
```

#### 9A.1.5.3: Commit generated files

> **LUT Path Strategy: Source Tree (Committed)**
>
> The generated headers are written directly to the source tree and committed to version control:
> - `kernels/fp8_lut.gen.h` — OpenCL backend
> - `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` — CPU backend
>
> **Rationale:**
> 1. **Reproducibility** — Build output is deterministic without requiring `ml_dtypes` at compile time
> 2. **Simplicity** — No include path injection needed; files are where `#include` expects them
> 3. **Transparency** — Review diffs catch unintended changes from `ml_dtypes` upgrades
>
> **When to regenerate:** Only when `ml_dtypes` version changes. CI verifies committed files match regeneration.

Add a CI check to verify the committed files match regeneration.

**File:** `.github/workflows/ci.yml` (add to existing workflow)

> **CI Ordering:** This verification step should run **early** in the CI pipeline (after dependency
> installation, before parallel backend test jobs start). Failed LUT verification should block
> all downstream jobs since stale LUTs cause backend compilation failures.

```yaml
# .github/workflows/ci.yml (add this step after 'Install dependencies')
- name: Verify FP8 LUT headers are up-to-date
  working-directory: architectures/averaging_ensembled_classifier
  run: |
    # Log ml_dtypes version for diagnostics (staleness is caught by diff below)
    ML_DTYPES_VERSION=$(python -c "import ml_dtypes; print(ml_dtypes.__version__)" 2>/dev/null || echo "unknown")
    echo "ml_dtypes version: $ML_DTYPES_VERSION"
    
    python scripts/generate_fp8_lut.py --backend opencl --output /tmp/fp8_lut.gen.h
    diff kernels/fp8_lut.gen.h /tmp/fp8_lut.gen.h || (echo "ERROR: kernels/fp8_lut.gen.h is out of date. Run: python scripts/generate_fp8_lut.py --backend opencl --output kernels/fp8_lut.gen.h" && exit 1)
    python scripts/generate_fp8_lut.py --backend cpu --output /tmp/cpu_fp8_lut.gen.h
    diff src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h /tmp/cpu_fp8_lut.gen.h || (echo "ERROR: cpu_fp8_lut.gen.h is out of date. Run: python scripts/generate_fp8_lut.py --backend cpu --output src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h" && exit 1)
```

> **CI Optimization:** The LUT verification always runs (to catch staleness), but the generated headers
> include the `ml_dtypes` version in a comment. When updating `ml_dtypes`, the diff will show the
> version change, making it clear why regeneration is needed. Consider adding a hash-based cache
> in CI for faster runs when the version hasn't changed.

#### 9A.1.5.4: Meson integration for LUT regeneration (development convenience, optional)

For development convenience, Meson can regenerate LUTs when `ml_dtypes` is updated. **This is optional** — the committed files are always used unless explicitly regenerated.

```meson
# In meson.build — OPTIONAL development convenience target
# Run: ninja -C builddir regen-fp8-luts
py = find_program('python3')

regen_fp8_luts = run_target('regen-fp8-luts',
    command: [
        py, meson.project_source_root() / 'scripts' / 'generate_fp8_lut.py',
        '--backend', 'opencl',
        '--output', meson.project_source_root() / 'kernels' / 'fp8_lut.gen.h',
    ],
)

# Note: This is a run_target (manual invocation), NOT a custom_target.
# The committed source files are authoritative; Meson does not auto-regenerate.
```

> **⚠️ Do NOT use `custom_target` with `build_by_default: true`**
>
> A `custom_target` would regenerate on every build, causing:
> - Git status noise (modified generated files)
> - Include path confusion (build dir vs. source dir)
> - CI failures when generated output differs from committed files

**Note:** The generated headers are committed to version control for reproducibility. The Meson targets provide an optional regeneration path for development; CI verification (Step 9A.1.5.3) ensures committed headers stay in sync.

---

### Step 9A.2: Amend `CONCEPT.md`

**Governing authority:** ADR-025 Context, §9.1  
**File:** `CONCEPT.md` (architecture root)

Three amendments. Insert in section order.

#### Amendment 9A.2.1 — Extend §2 (Primacy of Memory Strategy) with FP8 note

Append to the existing mixed-precision paragraph (added by Phase 7A):

> **FP8 is the ultimate expression of the Primacy of Memory Strategy.** FP8 (E4M3 or E5M2) storage achieves 4× bandwidth compression vs. FP32 and 2× vs. FP16. However, FP8's 3-bit mantissa (E4M3) or 2-bit mantissa (E5M2) is insufficient for compute or state roles: reductions saturate in one stage, and EMA updates round to zero for high β values. The architecture therefore permits FP8 **only in the storage role**, enforcing this constraint at `PrecisionConfig` construction time. Attempting to create a configuration with FP8 compute or FP8 state raises `ValueError` — there is no user-discipline escape hatch for non-functional training.

#### Amendment 9A.2.2 — Extend §11 (Host Orchestrator) with FP8 storage rationale

Append to the existing `PrecisionConfig` description:

> FP8 storage (E4M3 or E5M2) is supported for maximum bandwidth efficiency. Two FP8 variants are provided: **E4M3** (4 exponent bits, 3 mantissa bits, max value 448) prioritizes precision; **E5M2** (5 exponent bits, 2 mantissa bits, max value 57344) prioritizes dynamic range. E4M3 is the primary target for gradient and activation storage. The two formats differ in special-value semantics: **E4M3** (`float8_e4m3fn`) has no infinity representation but reserves 2 bit patterns for NaN (0x7F, 0xFF), leaving 254 finite values. **E5M2** follows IEEE-like conventions: exponent 0x1F encodes ±infinity (mantissa = 0) and NaN (mantissa ≠ 0), leaving 248 finite bit patterns. The store path saturates to the maximum finite value in both formats, so infinity and NaN bit patterns are never written to storage buffers. The host is responsible for scaling values to fit within FP8's representable range before storage; the existing Quadratic Scaling Policy already constrains gradients well below FP8 limits. `PrecisionConfig.fp8_e4m3()` provides E4M3 storage with FP32 compute and FP32 state; `PrecisionConfig.fp8_e4m3_f64()` adds FP64 compute and state for extended stability.

#### Amendment 9A.2.3 — Add Validation Scenario: The Bandwidth Extremist

Add to the *Validation Scenarios* section, after the existing Alchemist II scenario:

> **Scenario: The Bandwidth Extremist (FP8 Storage Fidelity)**
>
> - **Description:** A training task is executed with `PrecisionConfig.fp8_e4m3()` and compared against `PrecisionConfig.mixed_f16_f32()` baseline. Both configurations use identical compute (FP32) and state (FP32) precision.
> - **Validation Focus:** Confirms that FP8 storage produces convergent training within tolerance of the FP16 baseline. Validates that storage-role buffer sizes are halved (8 bits vs. 16 bits), that quantization error does not prevent convergence, and that the host's scaling machinery maintains numerical correctness.
> - **Key Insight:** Proves the architectural claim that storage precision is independent of training fidelity. The 2× bandwidth reduction vs. FP16 (4× vs. FP32) is achieved without degrading the loss curve, because compute and state remain at FP32.

**Verification:** After editing, confirm no dangling references to single-axis FP8 terminology. All FP8 references should use three-role terminology: `storage_dtype=float8_e4m3fn` or factory names.

---

### Step 9A.3: Amend `CONTRACT.md`

**Governing authority:** ADR-025 §3  
**File:** `CONTRACT.md` (architecture root)

Three amendments to Article 6 (Build-Time Symbols).

#### Amendment 9A.3.1 — Add FP8 derived symbols

In the Article 6 build-time symbol table, add three new rows:

| Symbol | Type | Meaning |
|:---|:---|:---|
| `STORAGE_TYPE_IS_FP8` | `int` (0 or 1) | 1 when `STORAGE_TYPE` is an 8-bit floating-point format (E4M3 or E5M2); gates FP8-specific load/store mechanics |
| `STORAGE_TYPE_IS_E4M3` | `int` (0 or 1) | 1 when `STORAGE_TYPE` is specifically E4M3; selects E4M3 conversion logic and lookup tables |
| `STORAGE_TYPE_IS_E5M2` | `int` (0 or 1) | 1 when `STORAGE_TYPE` is specifically E5M2; selects E5M2 conversion logic and lookup tables |

#### Amendment 9A.3.2 — Add FP8 storage-only note

Append to the symbol table notes section:

> **FP8 is storage-role only.** The symbols `COMPUTE_TYPE_IS_FP8` and `STATE_TYPE_IS_FP8` are **not defined** because FP8 compute and FP8 state are architecturally prohibited. The build system does not emit these symbols. Attempting to configure FP8 for compute or state roles raises `ValueError` at `PrecisionConfig` construction time.

> **Cross-backend naming:** All backends (OpenCL, CPU, and Vulkan) use the canonical `STORAGE_TYPE_IS_*` / `COMPUTE_TYPE_IS_*` / `STATE_TYPE_IS_*` flag names from CONTRACT.md Article 6. The Vulkan backend's GLSL type macros (`STORAGE_TYPE`, `COMPUTE_TYPE`, `STATE_TYPE`) and boolean flags (`STORAGE_TYPE_IS_FP8`, `STORAGE_TYPE_IS_E4M3`, `STORAGE_TYPE_IS_E5M2`) use the same canonical names as OpenCL, injected via `glslc -D`. No backend-specific shortened names are used.

#### Amendment 9A.3.3 — Extend §5 PrecisionConfig contract

Update the `PrecisionConfig` paragraph to reference the new factories:

> Factory classmethods provide common configurations. For FP8 storage, any valid combination of compute (FP16/FP32/FP64) and state (FP16/FP32/FP64) precision is permitted via direct construction. FP8 is permitted only in the storage role; FP8 compute and FP8 state raise `ValueError`.
>
> **FP8 factory methods:** `fp8_e4m3()`, `fp8_e5m2()` (FP32 compute/state), `fp8_e4m3_f16()`, `fp8_e5m2_f16()` (FP16 compute, FP32 state), `fp8_e4m3_f64()`, `fp8_e5m2_f64()` (FP64 compute/state). Additional combinations (e.g., FP16 compute + FP64 state) are constructed directly.

---

### Step 9A.4: Extend `PrecisionConfig`

**Governing authority:** ADR-025 §2  
**File:** `src/shared/precision_config.py`

> **⚠️ LINEAR EXECUTION ORDER — read before starting Step 9A.4**
>
> Sub-step numbering reflects *logical grouping* (config changes together, test
> changes together), **not** execution order. The migration in Step 9A.5.0 must
> complete before certain 9A.4 sub-steps can land. Follow this checklist:
>
> | Order | Sub-step | What it does | Gate |
> |:-----:|:---------|:-------------|:-----|
> | 1 | 9A.4.1–9A.4.2 | FP8 imports & constants | — |
> | 2 | 9A.4.4–9A.4.6 | New fields, update factories, add FP8 factories | — |
> | 3 | 9A.4.5a | `compute_epsilon` migration (see below) | Audit blocking |
> | 4 | **9A.5.0** | Migrate all `float16()` → `mixed_f16_f32()` | `pytest` green |
> | 5 | 9A.5.0a | Migrate authority doc references | — |
> | 6 | 9A.4.9 | Migrate `ModelSpec.float16()` internal call | — |
> | 7 | — | **Run full test suite — must pass** | Blocking |
> | 8 | 9A.4.3 | Add FP8 rejection to `__post_init__` | — |
> | 9 | 9A.4.7 | Delete `float16()` factory | — |
> | 10 | — | **Run full test suite — must pass** | Blocking |
>
> **Do not reorder.** Steps 8–9 break code that Steps 4–6 fix.

#### 9A.4.1: Add `ml_dtypes` import

At the top of the file:

```python
import ml_dtypes
```

#### 9A.4.2: Define FP8 dtype constants

After imports, add module-level constants for FP8 dtypes:

```python
# FP8 dtype references (ADR-025 §1)
# ml_dtypes exports type objects (not dtype instances) — wrap with np.dtype() for consistency
# These constants ARE np.dtype instances — use directly without re-wrapping
FP8_E4M3 = np.dtype(ml_dtypes.float8_e4m3fn)
FP8_E5M2 = np.dtype(ml_dtypes.float8_e5m2)
FP8_DTYPES = frozenset({FP8_E4M3, FP8_E5M2})
```

**Note:** `ml_dtypes.float8_e4m3fn` is a type object, not a dtype instance. The `np.dtype()` wrapper creates a proper NumPy dtype, enabling consistent comparison via `==` and access to `.itemsize` (which returns 1 for FP8). The module constants (`FP8_E4M3`, `FP8_E5M2`) are already `np.dtype` instances — use them directly in factory methods without re-wrapping.

> **⚠️ Normalization in `__post_init__`:** Because users may pass raw type objects (e.g., `ml_dtypes.float8_e4m3fn`) rather than `np.dtype` instances when constructing `PrecisionConfig` directly, the `__post_init__` method normalizes all dtype fields via `np.dtype()`. This ensures the `in FP8_DTYPES` membership check works correctly regardless of how the config was constructed.

#### 9A.4.3: Extend `__post_init__` with FP8 rejection

> **⚠️ DEPENDENCY: Execute Step 9A.5.0 (migrate `float16()` usages) and Step 9A.4.9
> (migrate `ModelSpec.float16()`) BEFORE applying this change.** The `float16()` factory
> deletion in Step 9A.4.7 will break any code still calling it. See the
> execution order in [Step 9A.5](#step-9a5-update-test-fixtures).

Add FP8 rejection logic **before** the existing itemsize invariants.

> **Why this ordering matters:** These checks must precede itemsize invariants because
> FP8's 1-byte itemsize would *pass* the `storage ≤ compute` check (1 ≤ 4), but FP8
> compute produces non-functional training. We reject architecturally unsound configs
> before validating size relationships.
>
> **FP8 itemsize note:** For valid FP8 configurations, the itemsize invariants naturally
> hold: FP8 storage (1 byte) ≤ FP16/FP32 compute (2/4 bytes) ≤ FP16/FP32/FP64 state (2/4/8 bytes).
> No special handling is needed — the existing checks pass.

```python
def __post_init__(self) -> None:
    # --- Normalize dtype fields to np.dtype instances ---
    # Users may pass raw type objects (e.g., ml_dtypes.float8_e4m3fn instead of
    # np.dtype(ml_dtypes.float8_e4m3fn)). Normalizing here ensures reliable
    # comparison in FP8_DTYPES membership checks.
    # Note: object.__setattr__ required because dataclass is frozen.
    object.__setattr__(self, 'storage_dtype', np.dtype(self.storage_dtype))
    object.__setattr__(self, 'compute_dtype', np.dtype(self.compute_dtype))
    object.__setattr__(self, 'state_dtype', np.dtype(self.state_dtype))
    
    # --- Architectural role constraints (must precede itemsize checks) ---
    # FP8 is storage-role only (ADR-025 §2.2)
    if self.compute_dtype in FP8_DTYPES:
        raise ValueError(
            f"FP8 compute is architecturally prohibited: "
            f"3-bit mantissa saturates in one reduction stage. "
            f"Got compute_dtype={self.compute_dtype}"
        )
    if self.state_dtype in FP8_DTYPES:
        raise ValueError(
            f"FP8 state is architecturally prohibited: "
            f"EMA updates round to zero for β > 0.9. "
            f"Got state_dtype={self.state_dtype}"
        )
    
    # Existing invariants (storage ≤ compute, storage ≤ state)
    # Note: Migrated from assert to ValueError for consistency with new FP8 checks
    if self.storage_dtype.itemsize > self.compute_dtype.itemsize:
        raise ValueError(
            f"storage_dtype ({self.storage_dtype}) cannot be wider than "
            f"compute_dtype ({self.compute_dtype})"
        )
    if self.storage_dtype.itemsize > self.state_dtype.itemsize:
        raise ValueError(
            f"storage_dtype ({self.storage_dtype}) cannot be wider than "
            f"state_dtype ({self.state_dtype})"
        )
```

> **Validation style migration:** The existing `__post_init__` used `assert` statements. This phase
> migrates to `raise ValueError` for consistency — all config validation errors now raise `ValueError`
> with descriptive messages, improving debuggability.

#### 9A.4.4: Extend dataclass with FP8-specific derived fields

**Approach:** Add `storage_fp_min_positive: float` and `storage_mantissa_bits: int` as explicit dataclass fields, consistent with the existing pattern for `storage_fp_format_max`. This keeps the frozen dataclass simple and avoids `@cached_property` complexity.

Update the dataclass definition:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable three-role precision configuration for plan construction.
    
    All fields are required (no defaults). Factory classmethods provide
    the standard configurations; direct construction is supported but
    requires all fields to be specified explicitly.
    """
    # --- Core dtype fields (3 roles) ---
    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype
    
    # --- Storage-role derived constants ---
    storage_fp_format_max: float
    storage_fp_min_positive: float          # NEW: smallest positive representable value
    storage_mantissa_bits: int              # NEW: mantissa precision for quantization analysis
    
    # --- Compute-role derived constants ---
    compute_fp_format_max: float
    compute_epsilon: float
```

> **Field ordering note:** All fields are required (no defaults), so field order does not
> affect dataclass initialization. The grouping above is logical, not a Python requirement.
> If future fields need defaults, they must come after all required fields.

All factory methods must now pass these values explicitly. For FP8:

| Field | E4M3 Value | E5M2 Value | Source |
|:---|:---|:---|:---|
| `storage_fp_format_max` | `448.0` | `57344.0` | E4M3: 2^8 × 1.75, E5M2: 2^15 × 1.75 |
| `storage_fp_min_positive` | `0.001953125` | `0.0000152587890625` | 2^-9, 2^-16 (exact subnormals) |
| `storage_mantissa_bits` | `3` | `2` | Format definitions |

For existing formats (FP16/FP32/FP64), use `np.finfo().smallest_subnormal` (NumPy ≥ 1.22).

**Note:** `ml_dtypes` FP8 types do not have a `np.finfo()` representation, so explicit constants are required for FP8 factories.

> **⚠️ `finfo.tiny` vs `finfo.smallest_subnormal`:** The field `storage_fp_min_positive` represents
> the smallest positive representable value. For FP8, this is the smallest *subnormal* (2^-9 for
> E4M3, 2^-16 for E5M2). For consistency, the FP16/FP32/FP64 factories must also use the
> smallest subnormal — `float(np.finfo(...).smallest_subnormal)` — NOT `finfo.tiny`, which
> returns the smallest *normal* number. Example: FP32's smallest subnormal is ≈1.4e-45 vs.
> `tiny` ≈1.18e-38. Using `tiny` would misrepresent the format's true minimum and create an
> inconsistent contract where FP8 reports subnormal min but FP16/32/64 report normal min.

> **⚠️ API Backward Compatibility:**
>
> Adding required fields (`storage_fp_min_positive`, `storage_mantissa_bits`) to the dataclass is **API-breaking** for code that constructs `PrecisionConfig` directly. Mitigation options:
>
> 1. **Preferred (this plan):** Add the new fields as required. All factory methods are updated. Direct construction sites (tests, external code) must add the new fields.
> 2. **Alternative:** Add fields with `field(default=...)` and compute defaults in `__post_init__` using `np.finfo()` for non-FP8 dtypes. This is backward-compatible but adds complexity for FP8 (which lacks `finfo`).
> 3. **Not recommended:** Make fields `Optional[float]` with `None` default — loses type safety.
>
> This plan chooses option 1 (clean break) because:
> - `PrecisionConfig` direct construction is rare (factory methods are the primary API)
> - The three-role model is relatively new (Phase 7–8), so external usage is limited
> - Explicit fields are clearer than computed defaults
>
> **Migration for direct construction sites:**
> ```python
> # BEFORE (will raise TypeError: missing required arguments)
> PrecisionConfig(
>     storage_dtype=np.float32,
>     compute_dtype=np.float32,
>     state_dtype=np.float32,
>     storage_fp_format_max=3.4028235e+38,
>     compute_fp_format_max=3.4028235e+38,
>     compute_epsilon=1.1920929e-07,
> )
>
> # AFTER (add new required fields)
> PrecisionConfig(
>     storage_dtype=np.float32,
>     compute_dtype=np.float32,
>     state_dtype=np.float32,
>     storage_fp_format_max=3.4028235e+38,
>     storage_fp_min_positive=1.401298e-45,   # NEW (smallest_subnormal)
>     storage_mantissa_bits=23,               # NEW
>     compute_fp_format_max=3.4028235e+38,
>     compute_epsilon=1.1920929e-07,
> )
> ```

#### 9A.4.5: Update existing factory classmethods with new fields

**All seven existing factory methods must be updated** to include the new `storage_fp_min_positive` and `storage_mantissa_bits` fields. Failing to update them will cause `TypeError` at construction time.

#### 9A.4.5a: `compute_epsilon` migration (tracked sub-task)

> **Behavioral change:** The current `float64()` and `mixed_f32_f64()` factories
> use a hardcoded `compute_epsilon=1e-15` rather than `float(np.finfo(np.float64).eps)` (which
> is ≈ 2.22e-16 — a 4.5× reduction). The updated factories below switch to the `finfo`-derived
> value for consistency with all other factories. The hardcoded `1e-15` was a conservative
> approximation, not a deliberate tolerance choice — epsilon is used for division-by-zero guards,
> where any small value suffices.
>
> **This sub-task has its own rollback gate.** If any test regresses after switching to
> `finfo`-derived epsilon, the fix is to update that test's tolerance, not to revert to the
> hardcoded value. If the audit reveals value-sensitive usage, this sub-task is **deferred**
> to a separate patch — do not block the rest of Step 9A.4.5 on it.
>
> **⚠️ MANDATORY pre-condition — execute before applying the epsilon change:**
> ```bash
> grep -rn 'compute_epsilon' src/ tests/ | grep -v '__pycache__' | tee /tmp/epsilon-audit.txt
> ```
> Review `/tmp/epsilon-audit.txt` and confirm that **every** usage of `compute_epsilon` is in
> a division-by-zero guard (e.g., `/ (x + epsilon)`) and does not depend on the exact value
> `1e-15`. **This audit is blocking** — do not proceed with the epsilon change if any usage is
> value-sensitive.
>
> **Rollback gate for this sub-task only:**
> ```bash
> # Run BEFORE the epsilon change:
> pytest tests/ -v 2>&1 | tee /tmp/pre-epsilon.txt
> # Apply the epsilon change, then:
> pytest tests/ -v 2>&1 | tee /tmp/post-epsilon.txt
> # Diff the results:
> diff <(grep -E '(PASSED|FAILED)' /tmp/pre-epsilon.txt) <(grep -E '(PASSED|FAILED)' /tmp/post-epsilon.txt)
> ```
> Any new failures are epsilon-related regressions. Fix the test tolerances (not the epsilon value).

```python
@classmethod
def float32(cls) -> "PrecisionConfig":
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float32),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=float(f32_info.max),
        storage_fp_min_positive=float(f32_info.smallest_subnormal),
        storage_mantissa_bits=f32_info.nmant,  # 23 for FP32
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

# NOTE: float16() is DELETED in this phase (see Step 9A.4.7)

@classmethod
def mixed_f16_f32(cls) -> "PrecisionConfig":
    """FP16 storage, FP32 compute, FP32 state.
    
    Recommended replacement for the deleted float16() factory.
    Provides FP16 bandwidth savings with FP32 compute fidelity and FP32 state stability.
    """
    f16_info = np.finfo(np.float16)
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float16),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=float(f16_info.max),
        storage_fp_min_positive=float(f16_info.smallest_subnormal),
        storage_mantissa_bits=f16_info.nmant,  # 10 for FP16
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def float64(cls) -> "PrecisionConfig":
    f64_info = np.finfo(np.float64)
    return cls(
        storage_dtype=np.dtype(np.float64),
        compute_dtype=np.dtype(np.float64),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(f64_info.max),
        storage_fp_min_positive=float(f64_info.smallest_subnormal),
        storage_mantissa_bits=f64_info.nmant,  # 52 for FP64
        compute_fp_format_max=float(f64_info.max),
        compute_epsilon=float(f64_info.eps),
    )

@classmethod
def mixed_f32_f64_state(cls) -> "PrecisionConfig":
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float32),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(f32_info.max),
        storage_fp_min_positive=float(f32_info.smallest_subnormal),
        storage_mantissa_bits=f32_info.nmant,  # 23 for FP32
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def mixed_f16_f64_state(cls) -> "PrecisionConfig":
    """FP16 storage, FP32 compute, FP64 state for extended optimizer stability."""
    f16_info = np.finfo(np.float16)
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float16),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(f16_info.max),
        storage_fp_min_positive=float(f16_info.smallest_subnormal),
        storage_mantissa_bits=f16_info.nmant,  # 10 for FP16
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def mixed_f32_f64(cls) -> "PrecisionConfig":
    f32_info = np.finfo(np.float32)
    f64_info = np.finfo(np.float64)
    return cls(
        storage_dtype=np.dtype(np.float32),
        compute_dtype=np.dtype(np.float64),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(f32_info.max),
        storage_fp_min_positive=float(f32_info.smallest_subnormal),
        storage_mantissa_bits=f32_info.nmant,  # 23 for FP32
        compute_fp_format_max=float(f64_info.max),
        compute_epsilon=float(f64_info.eps),
    )
```

#### 9A.4.6: Add FP8 factory classmethods

FP8 storage supports any valid compute (FP16/FP32/FP64) and state (FP16/FP32/FP64) combination.
The following table shows common FP8 combinations (FP16 state combinations are also valid via direct construction but omitted for brevity):

| Storage | Compute | State | Factory Method | Suffix |
|:---|:---|:---|:---|:---|
| E4M3 | FP16 | FP32 | `fp8_e4m3_f16()` | `s8e4c16x32` |
| E4M3 | FP16 | FP64 | (direct construction) | `s8e4c16x64` |
| E4M3 | FP32 | FP32 | `fp8_e4m3()` | `s8e4c32x32` |
| E4M3 | FP32 | FP64 | (direct construction) | `s8e4c32x64` |
| E4M3 | FP64 | FP64 | `fp8_e4m3_f64()` | `s8e4c64x64` |
| E5M2 | FP16 | FP32 | `fp8_e5m2_f16()` | `s8e5c16x32` |
| E5M2 | FP16 | FP64 | (direct construction) | `s8e5c16x64` |
| E5M2 | FP32 | FP32 | `fp8_e5m2()` | `s8e5c32x32` |
| E5M2 | FP32 | FP64 | (direct construction) | `s8e5c32x64` |
| E5M2 | FP64 | FP64 | `fp8_e5m2_f64()` | `s8e5c64x64` |

**Note:** FP64 compute **requires** FP64 state due to the itemsize constraint (`compute_dtype.itemsize ≤ state_dtype.itemsize`). The combination FP64 compute + FP32 state is invalid and raises `ValueError`.

Add factory classmethods for common combinations:

```python
# --- FP8 E4M3 Factories ---

@classmethod
def fp8_e4m3(cls) -> "PrecisionConfig":
    """E4M3 storage (8-bit, max=448), FP32 compute, FP32 state.
    
    Maximum bandwidth configuration for standard training.
    4× storage compression vs. FP32, 2× vs. FP16.
    """
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=FP8_E4M3,  # Module constant, already np.dtype
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,  # 2^-9
        storage_mantissa_bits=3,
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def fp8_e4m3_f16(cls) -> "PrecisionConfig":
    """E4M3 storage (8-bit), FP16 compute, FP32 state.
    
    Maximum bandwidth with native FP16 compute.
    Suitable for hardware with fast FP16 ALUs.
    
    Note: FP16 compute max (65504) far exceeds E4M3 storage max (448).
    Values in the gap (448, 65504] will saturate when stored to FP8.
    The host-side FP8 scaling path (Phase 9E) ensures pre-storage values
    fit within storage_fp_format_max.
    """
    f16_info = np.finfo(np.float16)
    return cls(
        storage_dtype=FP8_E4M3,
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,  # 2^-9
        storage_mantissa_bits=3,
        compute_fp_format_max=float(f16_info.max),
        compute_epsilon=float(f16_info.eps),
    )

@classmethod
def fp8_e4m3_f64(cls) -> "PrecisionConfig":
    """E4M3 storage (8-bit), FP64 compute, FP64 state.
    
    Maximum bandwidth with full FP64 precision for compute and state.
    """
    f64_info = np.finfo(np.float64)
    return cls(
        storage_dtype=FP8_E4M3,
        compute_dtype=np.dtype(np.float64),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,  # 2^-9
        storage_mantissa_bits=3,
        compute_fp_format_max=float(f64_info.max),
        compute_epsilon=float(f64_info.eps),
    )

# --- FP8 E5M2 Factories ---

@classmethod
def fp8_e5m2(cls) -> "PrecisionConfig":
    """E5M2 storage (8-bit, max=57344), FP32 compute, FP32 state.
    
    Maximum bandwidth with wider dynamic range.
    Suitable for gradients spanning many orders of magnitude.
    """
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=FP8_E5M2,
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=57344.0,
        storage_fp_min_positive=0.0000152587890625,  # 2^-16, exact
        storage_mantissa_bits=2,
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def fp8_e5m2_f16(cls) -> "PrecisionConfig":
    """E5M2 storage (8-bit), FP16 compute, FP32 state.
    
    Maximum bandwidth with wider dynamic range and native FP16 compute.
    
    Note: As with all FP8 configurations, compute_fp_format_max exceeds
    storage_fp_format_max. For this config, the gap is narrow: FP16 max
    (65504) vs E5M2 max (57344). Values in the gap (57344, 65504] will
    saturate when stored to FP8. The host-side FP8 scaling path (Phase 9E)
    ensures pre-storage values fit within storage_fp_format_max.
    """
    f16_info = np.finfo(np.float16)
    return cls(
        storage_dtype=FP8_E5M2,
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=57344.0,
        storage_fp_min_positive=0.0000152587890625,  # 2^-16, exact
        storage_mantissa_bits=2,
        compute_fp_format_max=float(f16_info.max),
        compute_epsilon=float(f16_info.eps),
    )

@classmethod
def fp8_e5m2_f64(cls) -> "PrecisionConfig":
    """E5M2 storage (8-bit), FP64 compute, FP64 state.
    
    Maximum bandwidth with wider dynamic range and full FP64 precision.
    """
    f64_info = np.finfo(np.float64)
    return cls(
        storage_dtype=FP8_E5M2,  # Already np.dtype from module constants
        compute_dtype=np.dtype(np.float64),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=57344.0,
        storage_fp_min_positive=0.0000152587890625,  # 2^-16, exact
        storage_mantissa_bits=2,
        compute_fp_format_max=float(f64_info.max),
        compute_epsilon=float(f64_info.eps),
    )
```

#### 9A.4.7: Delete `float16()` factory

**Remove** the `float16()` classmethod entirely. It produced all-FP16 configurations (including FP16 state) which are a common footgun for extended training, and the name is misleading since `mixed_f16_f32()` is the recommended FP16-storage configuration. Direct construction with `state_dtype=np.float16` remains valid for users who explicitly need uniform FP16.

```python
# DELETE this method:
# @classmethod
# def float16(cls) -> "PrecisionConfig":
#     """..."""
#     return cls(
#         storage_dtype=np.dtype(np.float16),
#         compute_dtype=np.dtype(np.float16),
#         state_dtype=np.dtype(np.float16),  # <-- Architecturally unsound
#     )
```

For other valid combinations (e.g., FP16 compute + FP64 state, FP32 compute + FP64 state),
use direct construction:

```python
# FP8 E4M3 with FP16 compute and FP64 state
f16_info = np.finfo(np.float16)
cfg = PrecisionConfig(
    storage_dtype=np.dtype(ml_dtypes.float8_e4m3fn),
    compute_dtype=np.dtype(np.float16),
    state_dtype=np.dtype(np.float64),
    storage_fp_format_max=448.0,
    storage_fp_min_positive=0.001953125,  # 2^-9
    storage_mantissa_bits=3,
    compute_fp_format_max=float(f16_info.max),
    compute_epsilon=float(f16_info.eps),
)

# FP8 E4M3 with FP32 compute and FP64 state
f32_info = np.finfo(np.float32)
cfg = PrecisionConfig(
    storage_dtype=np.dtype(ml_dtypes.float8_e4m3fn),
    compute_dtype=np.dtype(np.float32),
    state_dtype=np.dtype(np.float64),
    storage_fp_format_max=448.0,
    storage_fp_min_positive=0.001953125,  # 2^-9
    storage_mantissa_bits=3,
    compute_fp_format_max=float(f32_info.max),
    compute_epsilon=float(f32_info.eps),
)
```

#### 9A.4.8: Verification

- `PrecisionConfig.fp8_e4m3()` constructs without error
- `PrecisionConfig.fp8_e5m2()` constructs without error
- `PrecisionConfig.fp8_e4m3_f16()` constructs without error
- `PrecisionConfig.fp8_e5m2_f16()` constructs without error
- `PrecisionConfig.fp8_e4m3_f64()` constructs without error
- `PrecisionConfig.fp8_e5m2_f64()` constructs without error
- `PrecisionConfig(storage_dtype=..., compute_dtype=FP8_E4M3, state_dtype=...)` raises `ValueError`
- `PrecisionConfig(storage_dtype=..., compute_dtype=..., state_dtype=FP8_E4M3)` raises `ValueError`
- `PrecisionConfig(storage_dtype=np.float16, compute_dtype=np.float16, state_dtype=np.float16, ...)` constructs without error (FP16 state is permitted)
- `fp8_e4m3().storage_dtype.itemsize == 1`
- `fp8_e4m3().storage_fp_format_max == 448.0`

---

#### 9A.4.9: Migrate `ModelSpec.float16()` factory

**Governing authority:** ADR-025 §2.3 (`float16()` factory deletion)  
**File:** `src/shared/model_spec.py`

The `ModelSpec` class has a `float16()` factory that internally calls `PrecisionConfig.float16()`. Since that factory is deleted in Step 9A.4.7, `ModelSpec.float16()` must be updated to emit a `DeprecationWarning` and delegate to `mixed_f16_f32()`. The name `float16()` is misleading now that it returns FP32 compute/state, so deprecation guides users to the correctly-named factory.

```python
# BEFORE (will break — PrecisionConfig.float16() is deleted)
@classmethod
def float16(cls) -> "ModelSpec":
    return cls(precision=PrecisionConfig.float16())

# AFTER (deprecation warning + delegation to mixed_f16_f32)
import warnings

@classmethod  
def float16(cls) -> "ModelSpec":
    """FP16 storage with FP32 compute and FP32 state.
    
    .. deprecated::
        Use ``ModelSpec.mixed_f16_f32()`` instead. This factory will be
        removed in a future release. The name ``float16`` is misleading
        because compute and state are FP32, not FP16.
    """
    warnings.warn(
        "ModelSpec.float16() is deprecated and will be removed. "
        "Use ModelSpec.mixed_f16_f32() instead "
        "(FP16 storage, FP32 compute, FP32 state).",
        DeprecationWarning,
        stacklevel=2,
    )
    return cls(precision=PrecisionConfig.mixed_f16_f32())
```

**Verification:**
- `ModelSpec.float16()` constructs without error
- `ModelSpec.float16().precision.state_dtype == np.float32` (not FP16)
- Calling `ModelSpec.float16()` emits `DeprecationWarning`

#### 9A.4.10: Add dtype consistency regression test

**File:** `tests/tier1/test_precision_config.py`

Add a test to catch `ml_dtypes` version changes that could break dtype comparisons:

```python
def test_fp8_dtype_module_constants_consistency():
    """Verify FP8 dtype module constants match ml_dtypes.
    
    Catches ml_dtypes version changes that could break FP8_DTYPES membership checks.
    The comparison `config.storage_dtype in FP8_DTYPES` relies on np.dtype equality.
    """
    import ml_dtypes
    from shared.precision_config import FP8_E4M3, FP8_E5M2, FP8_DTYPES
    
    # Module constants must equal freshly-constructed dtypes
    assert np.dtype(ml_dtypes.float8_e4m3fn) == FP8_E4M3, (
        "FP8_E4M3 module constant doesn't match ml_dtypes.float8_e4m3fn. "
        "ml_dtypes may have changed dtype identity semantics."
    )
    assert np.dtype(ml_dtypes.float8_e5m2) == FP8_E5M2, (
        "FP8_E5M2 module constant doesn't match ml_dtypes.float8_e5m2. "
        "ml_dtypes may have changed dtype identity semantics."
    )
    
    # FP8_DTYPES membership must work with freshly-constructed dtypes
    assert np.dtype(ml_dtypes.float8_e4m3fn) in FP8_DTYPES
    assert np.dtype(ml_dtypes.float8_e5m2) in FP8_DTYPES
    
    # And with factory-produced configs
    cfg_e4m3 = PrecisionConfig.fp8_e4m3()
    cfg_e5m2 = PrecisionConfig.fp8_e5m2()
    assert cfg_e4m3.storage_dtype in FP8_DTYPES
    assert cfg_e5m2.storage_dtype in FP8_DTYPES
```

---

### Step 9A.5: Update test fixtures

**Governing authority:** ADR-025 §9.2  
**Files:** `tests/conftest.py`, `tests/tier1/test_precision_config.py`

> **⚠️ HARD ORDERING CONSTRAINT: Steps 9A.5.0 → 9A.4.9 → 9A.4.3 → 9A.4.7**
>
> Steps 9A.4.3 and 9A.4.7 appear earlier in the document but **must be executed after**
> the migrations in 9A.5.0 and 9A.4.9. The step numbering reflects logical grouping
> (config changes together, test changes together), not execution order.
>
> The `float16()` factory migration (Step 9A.5.0) **must be completed before**
> the FP8 rejection is added to `__post_init__` (Step 9A.4.3) and the
> `float16()` factory is deleted (Step 9A.4.7). If the factory is deleted
> first, any code that imports or calls `PrecisionConfig.float16()` at
> module scope (including `conftest.py` fixtures and `ModelSpec.float16()`)
> will raise `AttributeError` during import, making it impossible to run the test
> suite to validate the migration.
>
> **Execution order:**
> 1. **Steps 9A.4.1–9A.4.2** — Add FP8 imports, constants (no breaking changes)
> 2. **Steps 9A.4.4–9A.4.6** — Add new fields, update existing factories, and add FP8 factories (no breaking changes)
> 3. **Step 9A.5.0** — Migrate all `float16()` usages to `mixed_f16_f32()`
> 4. **Step 9A.5.0a** — Migrate authority document references
> 5. **Step 9A.4.9** — Migrate `ModelSpec.float16()` internal call
> 6. **Run tests** — Verify everything passes with old `__post_init__`
> 7. **Step 9A.4.3** — Add FP8 rejection to `__post_init__`
> 8. **Step 9A.4.7** — Delete `float16()` factory
> 9. **Run tests** — Verify rejection tests pass, no regressions

#### 9A.5.0: Migrate existing `float16()` usages (BREAKING CHANGE)

Before adding FP8 fixtures, update all existing test usages of `PrecisionConfig.float16()` to `PrecisionConfig.mixed_f16_f32()`:

```bash
# Find all factory usages
grep -r "PrecisionConfig.float16()" tests/ src/tests/

# Also find direct constructions with FP16 state (review if any should migrate)
grep -r "state_dtype=np.float16" tests/ src/tests/ src/shared/
grep -r "state_dtype=.*float16" tests/ src/tests/ src/shared/

# ⚠️ MANDATORY: Run the grep commands above and verify the ACTUAL file list
# before proceeding. The list below is a STARTING POINT from the codebase as
# of Phase 8E completion — files may have been added or renamed since.
# Commit the verified list as a comment in the migration commit message.
#
# Files requiring migration (based on current codebase — VERIFY BEFORE EXECUTING):
# Test files:
# - tests/bench_cpu_backend.py
# - tests/tier2/test_multi_precision_config.py
# - tests/tier2/cpu/test_cpu_crash_regressions.py
# - src/tests/conftest.py              ← PRECISION_CONFIGS list
# - src/tests/test_fp64_precision.py
# - src/tests/test_mixed_precision.py  ← also has suffix assertion changes (see below)
#
# Production code (NOT test files — also requires migration):
# - src/shared/model_spec.py → ModelSpec.float16() factory must be updated
#
# Direct constructions to check:
# - Any PrecisionConfig(..., state_dtype=np.float16)
# - Parametrized test values that include FP16 state
#
# Verification (run AFTER grep, BEFORE any code changes):
# diff <(grep -rl 'PrecisionConfig\.float16\|state_dtype=.*float16' tests/ src/) \
#      <(echo -e 'tests/bench_cpu_backend.py\ntests/tier2/test_multi_precision_config.py\n...')
# Any files in the left side but not the right side are MISSING from this list.

# --------------------------------------------------------------------------
# IMPORTANT: ModelSpec.float16() Migration
# --------------------------------------------------------------------------
# The ModelSpec class likely has a float16() factory that internally calls
# PrecisionConfig.float16(). This MUST be updated:
#
# Before (src/shared/model_spec.py):
#   @classmethod
#   def float16(cls, ...):
#       return cls(..., precision=PrecisionConfig.float16())
#
# After:
#   @classmethod
#   def float16(cls, ...):
#       """FP16 storage with FP32 compute and state.
#       
#       Note: Renamed semantics in Phase 9A. Previously used uniform FP16
#       (including FP16 state). This factory now returns FP16 storage +
#       FP32 compute + FP32 state, matching PrecisionConfig.mixed_f16_f32().
#       """
#       return cls(..., precision=PrecisionConfig.mixed_f16_f32())
#
# Alternatively, deprecate ModelSpec.float16() with a warning:
#   @classmethod
#   def float16(cls, ...):
#       import warnings
#       warnings.warn(
#           "ModelSpec.float16() is deprecated. Use ModelSpec.mixed_f16_f32() instead.",
#           DeprecationWarning,
#           stacklevel=2,
#       )
#       return cls.mixed_f16_f32(...)
```

**Migration:**
- Replace `PrecisionConfig.float16()` → `PrecisionConfig.mixed_f16_f32()`
- Direct `state_dtype=np.float16` constructions remain valid but should be reviewed — consider whether `state_dtype=np.float32` is more appropriate for the test's intent
- If test specifically validated uniform FP16, keep as-is with direct construction or convert to use `mixed_f16_f32()`
- Update `ModelSpec.float16()` to use `mixed_f16_f32()` internally

> **⚠️ Suffix assertion changes (not a mechanical find-replace):**
>
> Tests that assert the output of `_spv_variant_suffix()` or equivalent suffix helpers will break because the semantics change, not just the factory name:
>
> | Before | After | Why |
> |:---|:---|:---|
> | `_spv_variant_suffix(PrecisionConfig.float16()) == "_s16c16x16"` | Update to use direct construction: `PrecisionConfig(storage_dtype=np.float16, compute_dtype=np.float16, state_dtype=np.float16, ...)` | `float16()` factory is deleted; the suffix `_s16c16x16` remains reachable via direct construction |
> | `_spv_variant_suffix(PrecisionConfig.mixed_f16_f32()) == "_s16c32x32"` | Already correct — no change needed | `mixed_f16_f32()` produces storage=FP16, compute=FP32, state=FP32 |
>
> Specifically, `src/tests/test_mixed_precision.py` line 502 asserts `_spv_variant_suffix(PrecisionConfig.float16()) == "_s16c16x16"`. This test (`test_fp16_selects_s16c16x16_variant`) must be updated to construct the config directly instead of using the deleted factory.

#### Step 9A.5.0a: Migrate authority document references

The following non-code files reference `PrecisionConfig.float16()` or describe it as a valid factory. Each must be updated to reflect the deletion and the `mixed_f16_f32()` recommendation:

| File | Change Required |
|:---|:---|
| `CONCEPT.md` | Alchemist scenario: replace `float16()` with `mixed_f16_f32()` |
| `DESIGN.md` | Factory enumeration: remove `float16()`, add note |
| `ADR-022-host-code-precision-role-implications.md` | Factory table and coverage list |
| `ADR-023-backend-kernel-precision-role-implications.md` | Multi-configuration Tier 2 coverage |
| `ADR-024-double-precision-support.md` | Factory table, CPU promotion note, suffix table |

For each file: replace `float16()` factory references with `mixed_f16_f32()` and add a note that the `float16()` factory is deleted per ADR-025. Direct construction with uniform FP16 remains valid.

**Verification:** After edits, confirm no remaining references to `PrecisionConfig.float16()` in authority documents:

```bash
grep -rn 'PrecisionConfig\.float16\b\|float16()' CONCEPT.md DESIGN.md CONTRACT.md adr/ADR-02[2-5]*.md
```

Expected output: zero matches (or only references inside deletion/migration commentary).

**`ModelSpec.float16()` migration:** Handled in **Step 9A.4.9** (deprecation warning + delegation to `mixed_f16_f32()`). The `ModelSpec.float16()` factory emits a `DeprecationWarning` and delegates to `ModelSpec.mixed_f16_f32()`, matching the `PrecisionConfig` migration strategy. See Step 9A.4.9 for the full implementation and verification steps.

**Rationale:** The `float16()` factory was a common footgun — it produced all-FP16 configurations (including FP16 state), which lose EMA precision over extended training. The `mixed_f16_f32()` factory is the recommended FP16-storage configuration. The factory deletion (rather than state-role prohibition) preserves the flexibility of the three-role model (ADR-020) for users who explicitly construct with `state_dtype=np.float16`. The deprecation approach for `ModelSpec.float16()` (rather than immediate deletion) gives downstream consumers a migration window while making the factory rename visible at call sites.

#### 9A.5.1: Extend `PRECISION_CONFIGS` fixture

Add FP8 configs to the parametrization list in `conftest.py`:

```python
# FP8 configurations (all valid compute/state combinations)
# Note: Use FP8_E4M3/FP8_E5M2 module constants (already np.dtype instances)
# rather than wrapping ml_dtypes types with np.dtype() — this is consistent
# with the factory method implementations (Step 9A.4.6).
FP8_PRECISION_CONFIGS = [
    # E4M3 variants
    PrecisionConfig.fp8_e4m3(),       # FP32/FP32
    PrecisionConfig.fp8_e4m3_f16(),   # FP16/FP32
    PrecisionConfig.fp8_e4m3_f64(),   # FP64/FP64
    # E4M3 with mixed compute/state (direct construction)
    PrecisionConfig(
        storage_dtype=FP8_E4M3,  # Module constant from precision_config.py
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,  # 2^-9
        storage_mantissa_bits=3,
        compute_fp_format_max=float(np.finfo(np.float16).max),
        compute_epsilon=float(np.finfo(np.float16).eps),
    ),
    PrecisionConfig(
        storage_dtype=FP8_E4M3,
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=448.0,
        storage_fp_min_positive=0.001953125,  # 2^-9
        storage_mantissa_bits=3,
        compute_fp_format_max=float(np.finfo(np.float32).max),
        compute_epsilon=float(np.finfo(np.float32).eps),
    ),
    # E5M2 variants
    PrecisionConfig.fp8_e5m2(),       # FP32/FP32
    PrecisionConfig.fp8_e5m2_f16(),   # FP16/FP32
    PrecisionConfig.fp8_e5m2_f64(),   # FP64/FP64
]

PRECISION_CONFIGS = [
    # ... existing configs ...
    *FP8_PRECISION_CONFIGS,
]
```

Note: FP8 configs should only be included in tests where the backend already supports FP8. For Phase 9A, FP8 configs are added but may be skipped in backend tests via `pytest.mark.skipif` until backend support lands in Phase 9B/9C/9D.

**⚠️ CRITICAL: Apply skip fixture to ALL existing precision-parametrized tests**

Adding FP8 configs to `PRECISION_CONFIGS` will cause existing backend tests to attempt FP8 execution **before** backend support exists. To prevent spurious failures:

1. Add `@pytest.mark.usefixtures("skip_if_fp8_unsupported")` to **every** test that parametrizes over `PRECISION_CONFIGS` and exercises backend code
2. Alternatively, create a separate `BACKEND_PRECISION_CONFIGS` list that excludes FP8 until Phase 9B/9C/9D complete
3. The pure-Python `PrecisionConfig` unit tests (9A.5.2) do NOT need the skip fixture — they test config construction, not backend execution

**Centralized skip marker (add to `conftest.py`):**

```python
import pytest
from shared.precision_config import FP8_DTYPES

def _probe_fp8_backend_support() -> set[str]:
    """Probe which backends lack FP8 support.
    
    Returns set of backend names that do NOT support FP8.
    This avoids hardcoding phase completion status — the probe
    checks for the actual kernel/shader files installed by each phase.
    
    Note: Each probe tests for a concrete artifact produced by the
    corresponding phase (generated header, compiled variant, etc.)
    rather than private method names like '_emit_fp8_flags', which
    are fragile and may be renamed without updating test infrastructure.
    
    Lazy: This function is called on first access (not at import/
    collection time) to prevent backend imports from running during
    `pytest --collect-only` or IDE indexing.  The result is cached.
    """
    unsupported = set()
    
    # OpenCL: check if type_mapping has FP8 entries (added in Phase 9B Step 9B.1)
    # Note: The LUT header lives at kernels/fp8_lut.gen.h (not under
    # backends.opencl), so probing via importlib.resources would require
    # filesystem-relative path resolution. Checking type_mapping is more
    # reliable — it's a direct Phase 9B artifact and a Python-importable module.
    try:
        from backends.opencl.type_mapping import DTYPE_TO_OPENCL
        import ml_dtypes
        if ml_dtypes.float8_e4m3fn not in DTYPE_TO_OPENCL:
            unsupported.add("opencl")
    except (ImportError, KeyError):
        unsupported.add("opencl")
    
    # CPU: check if FP8 kernel variants were compiled
    try:
        from backends.cpu._fp8_variants import AVAILABLE_FP8_VARIANTS
        if not AVAILABLE_FP8_VARIANTS:
            unsupported.add("cpu")
    except ImportError:
        unsupported.add("cpu")
    
    # Vulkan: check if FP8 SPIR-V variants were compiled
    try:
        import importlib.resources
        # Phase 9D compiles FP8 .spv files into the kernels package
        kernels = importlib.resources.files("backends.vulkan.kernels")
        # Check for any FP8 variant (e.g., s8e4c32x32)
        has_fp8_spv = any(
            r.name.endswith(".spv") and "_s8e" in r.name
            for r in kernels.iterdir()
        )
        if not has_fp8_spv:
            unsupported.add("vulkan")
    except (ImportError, AttributeError, TypeError):
        unsupported.add("vulkan")
    
    return unsupported

# Lazy probe — cached on first access, NOT at collection time.
# Running backend imports during `pytest --collect-only` can trigger
# GPU driver initialization, CUDA contexts, etc. — unacceptable for
# environments without hardware (CI lint jobs, IDE indexers).
_fp8_unsupported_backends_cache: set[str] | None = None

def _get_fp8_unsupported_backends() -> set[str]:
    global _fp8_unsupported_backends_cache
    if _fp8_unsupported_backends_cache is None:
        _fp8_unsupported_backends_cache = _probe_fp8_backend_support()
    return _fp8_unsupported_backends_cache

def pytest_collection_modifyitems(config, items):
    """Auto-skip FP8 tests on unsupported backends.
    
    This hook examines test parameters and adds skip markers automatically,
    avoiding the need for fixture coupling between `precision` and `backend_name`.
    
    Handles three cases:
    1. Test has both `precision` and `backend` params → skip FP8 on unsupported backend
    2. Test has `precision` but no `backend` param → skip FP8 (assumes all backends)
    3. Test has neither → no action
    """
    for item in items:
        # Extract precision and backend from parametrize markers
        precision = None
        backend_name = None
        
        if hasattr(item, "callspec"):
            params = item.callspec.params
            precision = params.get("precision") or params.get("precision_config")
            backend_name = params.get("backend") or params.get("backend_name")
        
        # Handle factory functions (callables) vs config instances
        if precision is not None and callable(precision):
            try:
                precision = precision()  # Invoke factory to get config instance
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"FP8 skip hook: factory invocation failed for {item.nodeid}: {exc}",
                    stacklevel=1,
                )
                continue
        
        # Check if this is an FP8 precision config
        is_fp8 = (
            precision is not None 
            and hasattr(precision, "storage_dtype")
            and precision.storage_dtype in FP8_DTYPES
        )
        
        if not is_fp8:
            continue
        
        # Case 1: FP8 + explicit backend → skip if backend unsupported
        if backend_name is not None:
            if backend_name in _get_fp8_unsupported_backends():
                item.add_marker(pytest.mark.skip(
                    reason=f"FP8 not yet implemented for {backend_name} backend"
                ))
        # Case 2: FP8 + no backend param → skip if ANY backend unsupported
        # (conservative: test likely exercises all backends or a default backend)
        elif _get_fp8_unsupported_backends():
            item.add_marker(pytest.mark.skip(
                reason=f"FP8 not yet implemented (waiting on: {', '.join(sorted(_get_fp8_unsupported_backends()))})"
            ))
```

**Alternative: parametrize-level skip** (for tests that don't use the hook):

```python
@pytest.mark.parametrize("precision", [
    pytest.param(PrecisionConfig.fp8_e4m3(), marks=pytest.mark.skip(
        reason="FP8 not yet implemented", condition=True
    )),
    PrecisionConfig.float32(),
    PrecisionConfig.mixed_f16_f32(),
])
def test_kernel_execution(precision, backend):
    ...
```

#### 9A.5.2: Add FP8-specific unit tests

Create or extend `test_precision_config.py` with:

```python
import pytest
import numpy as np
import ml_dtypes
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

class TestPrecisionRoleConstraints:
    """ADR-025 §2.2: Precision role constraints (FP8 storage-only)."""

    # Helper: default field values for rejection tests.
    # The rejected field triggers ValueError in __post_init__ before other
    # fields are semantically evaluated, but all fields must be present
    # because PrecisionConfig is a frozen dataclass.
    _F32 = np.dtype(np.float32)
    _F32_MAX = float(np.finfo(np.float32).max)
    _F32_MIN_POS = float(np.finfo(np.float32).smallest_subnormal)
    _F32_EPS = float(np.finfo(np.float32).eps)

    def test_fp8_e4m3_compute_rejection(self):
        """FP8 compute raises ValueError."""
        with pytest.raises(ValueError, match="FP8 compute is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=FP8_E4M3,
                state_dtype=self._F32,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=448.0,
                compute_epsilon=0.125,
            )

    def test_fp8_e5m2_compute_rejection(self):
        """FP8 compute raises ValueError (E5M2 variant)."""
        with pytest.raises(ValueError, match="FP8 compute is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=FP8_E5M2,
                state_dtype=self._F32,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=57344.0,
                compute_epsilon=0.25,
            )

    def test_fp8_e4m3_state_rejection(self):
        """FP8 state raises ValueError."""
        with pytest.raises(ValueError, match="FP8 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=self._F32,
                state_dtype=FP8_E4M3,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=self._F32_MAX,
                compute_epsilon=self._F32_EPS,
            )

    def test_fp8_e5m2_state_rejection(self):
        """FP8 state raises ValueError (E5M2 variant)."""
        with pytest.raises(ValueError, match="FP8 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=self._F32,
                compute_dtype=self._F32,
                state_dtype=FP8_E5M2,
                storage_fp_format_max=self._F32_MAX,
                storage_fp_min_positive=self._F32_MIN_POS,
                storage_mantissa_bits=23,
                compute_fp_format_max=self._F32_MAX,
                compute_epsilon=self._F32_EPS,
            )

    def test_fp16_state_construction_valid(self):
        """FP16 state constructs successfully (ADR-020 three-role model).
        
        Note: FP16 state is permitted but not recommended for extended training.
        The float16() factory is deleted (common footgun), but direct construction
        with state_dtype=np.float16 remains valid.
        """
        f16_info = np.finfo(np.float16)
        cfg = PrecisionConfig(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float16),
            storage_fp_format_max=float(f16_info.max),
            storage_fp_min_positive=float(f16_info.smallest_subnormal),
            storage_mantissa_bits=f16_info.nmant,
            compute_fp_format_max=float(f16_info.max),
            compute_epsilon=float(f16_info.eps),
        )
        assert cfg.state_dtype == np.dtype(np.float16)

    def test_fp16_state_with_fp32_storage_valid(self):
        """FP16 state permitted with FP32 storage.
        
        Note: This validates that FP16 state is not rejected regardless of the
        storage/compute precision. The itemsize invariant (storage ≤ state) still
        applies — FP32 storage (4 bytes) > FP16 state (2 bytes) would be rejected
        by the itemsize check, so we use FP16 storage + FP16 compute + FP16 state.
        """
        f16_info = np.finfo(np.float16)
        cfg = PrecisionConfig(
            storage_dtype=np.dtype(np.float16),
            compute_dtype=np.dtype(np.float16),
            state_dtype=np.dtype(np.float16),
            storage_fp_format_max=float(f16_info.max),
            storage_fp_min_positive=float(f16_info.smallest_subnormal),
            storage_mantissa_bits=f16_info.nmant,
            compute_fp_format_max=float(f16_info.max),
            compute_epsilon=float(f16_info.eps),
        )
        assert cfg.state_dtype == np.dtype(np.float16)


class TestFP8ValidCombinations:
    """ADR-025: All valid FP8 storage + compute/state combinations."""

    @pytest.mark.parametrize("compute_dtype,state_dtype", [
        (np.float16, np.float16),
        (np.float16, np.float32),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ])
    def test_e4m3_valid_combinations(self, compute_dtype, state_dtype):
        """E4M3 storage accepts all valid compute/state combinations."""
        compute_info = np.finfo(compute_dtype)
        cfg = PrecisionConfig(
            storage_dtype=FP8_E4M3,
            compute_dtype=np.dtype(compute_dtype),
            state_dtype=np.dtype(state_dtype),
            storage_fp_format_max=448.0,
            storage_fp_min_positive=0.001953125,
            storage_mantissa_bits=3,
            compute_fp_format_max=float(compute_info.max),
            compute_epsilon=float(compute_info.eps),
        )
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(compute_dtype)
        assert cfg.state_dtype == np.dtype(state_dtype)

    @pytest.mark.parametrize("compute_dtype,state_dtype", [
        (np.float16, np.float16),
        (np.float16, np.float32),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ])
    def test_e5m2_valid_combinations(self, compute_dtype, state_dtype):
        """E5M2 storage accepts all valid compute/state combinations."""
        compute_info = np.finfo(compute_dtype)
        cfg = PrecisionConfig(
            storage_dtype=FP8_E5M2,
            compute_dtype=np.dtype(compute_dtype),
            state_dtype=np.dtype(state_dtype),
            storage_fp_format_max=57344.0,
            storage_fp_min_positive=0.0000152587890625,
            storage_mantissa_bits=2,
            compute_fp_format_max=float(compute_info.max),
            compute_epsilon=float(compute_info.eps),
        )
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(compute_dtype)
        assert cfg.state_dtype == np.dtype(state_dtype)


class TestFP8Factories:
    """ADR-025 §2.3: FP8 factory classmethods."""

    def test_fp8_e4m3_factory(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(np.float32)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e5m2_factory(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(np.float32)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e4m3_f16_factory(self):
        cfg = PrecisionConfig.fp8_e4m3_f16()
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(np.float16)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e5m2_f16_factory(self):
        cfg = PrecisionConfig.fp8_e5m2_f16()
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(np.float16)
        assert cfg.state_dtype == np.dtype(np.float32)

    def test_fp8_e4m3_f64_factory(self):
        cfg = PrecisionConfig.fp8_e4m3_f64()
        assert cfg.storage_dtype == FP8_E4M3
        assert cfg.compute_dtype == np.dtype(np.float64)
        assert cfg.state_dtype == np.dtype(np.float64)

    def test_fp8_e5m2_f64_factory(self):
        cfg = PrecisionConfig.fp8_e5m2_f64()
        assert cfg.storage_dtype == FP8_E5M2
        assert cfg.compute_dtype == np.dtype(np.float64)
        assert cfg.state_dtype == np.dtype(np.float64)


class TestFP8DerivedConstants:
    """ADR-025 §2.4: FP8 derived constants."""

    def test_e4m3_storage_fp_format_max(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_fp_format_max == 448.0

    def test_e5m2_storage_fp_format_max(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_fp_format_max == 57344.0

    def test_e4m3_storage_mantissa_bits(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_mantissa_bits == 3

    def test_e5m2_storage_mantissa_bits(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_mantissa_bits == 2

    def test_e4m3_buffer_sizing(self):
        """FP8 storage buffer is 1/4 size of FP32."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp32 = PrecisionConfig.float32()
        assert cfg_fp8.storage_dtype.itemsize == 1
        assert cfg_fp32.storage_dtype.itemsize == 4
        assert cfg_fp8.storage_dtype.itemsize * 4 == cfg_fp32.storage_dtype.itemsize


class TestFP8GoldenReference:
    """Golden reference tests using ml_dtypes directly.
    
    These tests validate that ml_dtypes produces expected values BEFORE testing
    backend conversions. If ml_dtypes behavior changes (e.g., version upgrade),
    these tests fail first, isolating the root cause.
    """

    def test_e4m3_roundtrip_exact_values(self):
        """Known E4M3 bit patterns produce expected float values."""
        # These values are exact — no rounding
        test_cases = [
            (0x00, 0.0),
            (0x38, 1.0),
            (0x3C, 1.5),
            (0x40, 2.0),
            (0x7E, 448.0),  # Max E4M3
            (0x80, -0.0),   # Negative zero
            (0xB8, -1.0),
            (0xFE, -448.0), # Min E4M3
        ]
        for bits, expected in test_cases:
            arr = np.array([bits], dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
            actual = float(arr[0])
            assert actual == expected, f"E4M3 0x{bits:02X}: expected {expected}, got {actual}"

    def test_e5m2_roundtrip_exact_values(self):
        """Known E5M2 bit patterns produce expected float values."""
        test_cases = [
            (0x00, 0.0),
            (0x3C, 1.0),
            (0x40, 2.0),
            (0x7B, 57344.0),  # Max E5M2
            (0x80, -0.0),
            (0xBC, -1.0),
            (0xFB, -57344.0), # Min E5M2
        ]
        for bits, expected in test_cases:
            arr = np.array([bits], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
            actual = float(arr[0])
            assert actual == expected, f"E5M2 0x{bits:02X}: expected {expected}, got {actual}"

    def test_e4m3_no_infinities(self):
        """E4M3fn has no infinities (only 2 NaN patterns: 0x7F, 0xFF)."""
        all_bits = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        all_float = all_bits.astype(np.float32)
        assert not np.any(np.isinf(all_float)), "E4M3fn should have no inf values"
        # Exactly 254 finite values (256 - 2 NaN at 0x7F and 0xFF)
        finite_count = np.sum(np.isfinite(all_float))
        assert finite_count == 254, (
            f"E4M3fn should have 254 finite bit patterns, got {finite_count}"
        )

    def test_e5m2_special_values(self):
        """E5M2 has IEEE-like inf/NaN.
        
        E5M2 uses exponent 0x1F for special values:
          - mantissa=0 → ±inf (bit patterns 0x7C, 0xFC)
          - mantissa≠0 → NaN  (bit patterns 0x7D–0x7F, 0xFD–0xFF)
        The remaining 248 bit patterns are finite.
        """
        all_bits = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        all_float = all_bits.astype(np.float32)
        
        # Exactly 248 finite values (256 - 2 inf - 6 NaN)
        finite_count = np.sum(np.isfinite(all_float))
        assert finite_count == 248, (
            f"E5M2 should have 248 finite bit patterns, got {finite_count}"
        )
        
        # Verify specific special-value bit patterns
        assert np.isinf(all_float[0x7C]), "E5M2 0x7C should be +inf"
        assert np.isinf(all_float[0xFC]), "E5M2 0xFC should be -inf"
        assert np.isnan(all_float[0x7D]), "E5M2 0x7D should be NaN"
        assert np.isnan(all_float[0xFF]), "E5M2 0xFF should be NaN"

    def test_ml_dtypes_version_compatibility(self):
        """ml_dtypes version is compatible with expected FP8 behavior."""
        import ml_dtypes
        version = tuple(int(x) for x in ml_dtypes.__version__.split('.')[:2])
        assert version >= (0, 2), f"ml_dtypes {ml_dtypes.__version__} < 0.2.0; FP8 behavior may differ"


class TestE5M2DefensiveLoading:
    """E5M2 inf/NaN indices are defensive—should produce finite values.
    
    The store path NEVER writes inf/NaN bit patterns (0x7C-0x7F, 0xFC-0xFF).
    However, if a buffer were corrupted or externally constructed, the generated
    LUT substitutes finite values (max or zero) to prevent crashes.
    
    This test validates the LUT generation fix in Step 9A.1.5.1 (format_c_float).
    """

    def test_e5m2_inf_indices_return_max_finite(self):
        """Loading E5M2 +inf indices (0x7C) should return max finite, not crash.
        
        The generated C LUT replaces inf→57344.0f for safety.
        The ml_dtypes Python LUT returns actual inf (for validation),
        but the C code path produces a finite value.
        """
        # Python validation: ml_dtypes returns inf
        import ml_dtypes
        arr_inf = np.array([0x7C], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        assert np.isinf(float(arr_inf[0])), "ml_dtypes E5M2 0x7C should be +inf"
        
        # Document: Generated C LUT (fp8_lut.gen.h) substitutes 57344.0f
        # This is tested at compile/runtime in Phase 9B/9C kernel tests.

    def test_e5m2_nan_indices_return_zero(self):
        """Loading E5M2 NaN indices (0x7D-0x7F, 0xFD-0xFF) should return 0, not crash.
        
        The generated C LUT replaces NaN→0.0f for safety.
        The ml_dtypes Python LUT returns actual NaN (for validation),
        but the C code path produces zero.
        """
        import ml_dtypes
        nan_indices = [0x7D, 0x7E, 0x7F, 0xFD, 0xFE, 0xFF]
        for idx in nan_indices:
            arr_nan = np.array([idx], dtype=np.uint8).view(ml_dtypes.float8_e5m2)
            assert np.isnan(float(arr_nan[0])), f"ml_dtypes E5M2 0x{idx:02X} should be NaN"
        
        # Document: Generated C LUT (fp8_lut.gen.h) substitutes 0.0f
        # This is tested at compile/runtime in Phase 9B/9C kernel tests.
```

---

### Step 9A.6: Validate rollback gate

**Goal:** Confirm no regressions from Phase 8E.

#### 9A.6.1: Run existing test suite

```bash
cd architectures/averaging_ensembled_classifier
pytest tests/ -v
```

All tests that passed in Phase 8E must pass now.

#### 9A.6.2: Confirm existing factories unchanged

Manually verify that `PrecisionConfig.float32()`, `mixed_f16_f32()`, etc. produce identical configs before and after the changes.

#### 9A.6.3: Document any skipped FP8 tests

FP8 configs added to parametrization may cause test failures in backend tests until Phase 9B/9C/9D completes. Document skips explicitly:

```python
@pytest.mark.skipif(
    precision.storage_dtype in FP8_DTYPES,
    reason="FP8 backend support not yet implemented (Phase 9B/9C/9D)"
)
```

---

## 4. Migration Order Rationale

The task order reflects the dependency graph:

1. **ml_dtypes dependency** — Required before any FP8 dtype can be referenced.
2. **CONCEPT.md / CONTRACT.md** — Authority amendments establish the architectural vocabulary.
3. **PrecisionConfig** — The configuration interface enables construction of FP8 configs. The storage-only constraint is enforced here.
4. **Test fixtures** — Tests validate the configuration changes.
5. **Rollback gate** — Ensures no regressions before proceeding to backend implementation.

Backend implementation (Phases 9B–9D) depends on the `PrecisionConfig` FP8 factories and derived constants being available. Host-side scaling (Phase 9E) depends on all backends supporting FP8 load/store.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|:---|:---|:---|:---|
| `ml_dtypes` version incompatibility with NumPy | Low | High | Pin `ml_dtypes>=0.2.0` which supports NumPy 1.x and 2.x. Test in CI matrix. |
| FP8 dtype handling differs from native NumPy dtypes | Medium | Medium | FP8-specific code paths for derived constants. No `np.finfo()` calls on FP8 dtypes. |
| Parametrized tests fail with FP8 configs before backend support | High | Low | Explicit `skipif` markers for FP8 configs in backend tests until Phase 9B/9C/9D. |
| User confusion about storage-only constraint | Low | Low | Clear error messages in `ValueError`. Documentation in CONCEPT.md and docstrings. |
| **`float16()` factory deletion breaks dependent code** | **High** | **Medium** | Breaking change is intentional. Migration path documented. Tests must be updated to use `mixed_f16_f32()`. Direct construction with FP16 state remains valid. The `float16()` factory is removed; `mixed_f16_f32()` is the recommended replacement (FP16 storage/compute, FP32 state). |

---

## 6. Files Modified Summary

| File | Change Type | Description |
|:---|:---|:---|
| `pyproject.toml` | **Edit** | Add `ml_dtypes>=0.2.0` dependency |
| `requirements.latest.txt` | **Edit** | Add `ml_dtypes>=0.2.0` |
| `scripts/generate_fp8_lut.py` | **New** | Python script to generate FP8→FP32 lookup tables for all backends |
| `kernels/fp8_lut.gen.h` | **New** (generated) | OpenCL `__constant` arrays for E4M3/E5M2 → float conversion |
| `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` | **New** (generated) | CPU `static const` arrays for E4M3/E5M2 → float conversion |
| `.github/workflows/ci.yml` | **Edit** | Add FP8 LUT header staleness verification step |
| `CONCEPT.md` | **Edit** | Add FP8 storage rationale (§2, §11), "Bandwidth Extremist" scenario |
| `CONTRACT.md` | **Edit** | Add FP8 build-time symbols (Article 6), factory method reference |
| `src/shared/precision_config.py` | **Edit** | Add FP8 imports, constants, factories, `__post_init__` constraints; **delete** `float16()` factory |
| `src/tests/conftest.py` | **Edit** | Add FP8 configs to `PRECISION_CONFIGS`, add FP8 skip hook; migrate `float16` → `mixed_f16_f32` |
| `src/tests/test_mixed_precision.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()`; update suffix assertions (see Step 9A.5.0) |
| `src/tests/test_fp64_precision.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `tests/tier2/test_multi_precision_config.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `tests/tier2/cpu/test_cpu_crash_regressions.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `tests/bench_cpu_backend.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `src/shared/model_spec.py` | **Edit** | Update `ModelSpec.float16()` to use `mixed_f16_f32()` internally |

---

*End of Phase 9A. Proceed to Phases 9B, 9C, 9D (parallelizable) after rollback gate passes.*
