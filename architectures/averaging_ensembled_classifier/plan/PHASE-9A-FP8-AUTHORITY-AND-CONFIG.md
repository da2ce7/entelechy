# Phase 9A: FP8 Support — Authority Documents, PrecisionConfig Extension & Shared Infrastructure

**Status: NOT STARTED**  
**Phase:** 9A of 9  
**Prerequisite:** Phase 8E complete and rollback gate passed.  
**Objective:** Amend the authority documents (CONCEPT.md, CONTRACT.md) with FP8 additions. Add `ml_dtypes` dependency. Extend `PrecisionConfig` with six new FP8 factory classmethods (`fp8_e4m3()`, `fp8_e5m2()`, `fp8_e4m3_f16()`, `fp8_e5m2_f16()`, `fp8_e4m3_f64()`, `fp8_e5m2_f64()`), enforce the storage-only constraint in `__post_init__`, and add FP8-specific derived constants. Update test fixtures. No kernel sources, no backend-native code, no build system changes. The phase ends when all existing tests pass against the new `PrecisionConfig` interface and the FP8 factories construct without error.  
**Governing ADR:** ADR-025 (§§1–4, §9)  
**Rollback gate:** All tests that passed before Phase 9A must pass after **migration** (Step 9A.5.0). This phase introduces a **breaking change**: `PrecisionConfig.float16()` is deleted and FP16 state is prohibited. Tests using `float16()` must migrate to `mixed_f16_f32()` first. After migration, existing `PrecisionConfig` factories remain identical. The new FP8 factories construct without error and satisfy the storage-only invariant. Attempting FP8 compute or state raises `ValueError`. Failing any existing Tier 2 test (post-migration) is a blocking regression.  
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
- Adding FP8-specific derived constant values: `storage_fp_format_max` (448 for E4M3, 57344 for E5M2), `storage_fp_format_min_subnormal`, `storage_mantissa_bits` (ADR-025 §2.4).
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

### Breaking change: FP16 state prohibition

**This phase introduces a breaking change.** The `__post_init__` invariant now also rejects FP16 in the `state_dtype` role. FP16 state lacks sufficient precision for EMA updates over extended training — the same precision erosion that disqualifies FP8 state applies (to a lesser degree) to FP16 state.

**Affected code:**
- `PrecisionConfig.float16()` — Currently constructs with all-FP16 (storage, compute, AND state). **This factory is deleted in this phase.**
- Any direct `PrecisionConfig(..., state_dtype=np.float16)` construction — **raises `ValueError`.**

**Unaffected code:**
- `PrecisionConfig.mixed_f16_f32()` — Constructs FP16 storage, FP16 compute, **FP32 state**. This remains valid and is the recommended replacement.

**Migration path:**
- Replace `PrecisionConfig.float16()` with `PrecisionConfig.mixed_f16_f32()`.
- The semantic difference: `mixed_f16_f32()` uses FP32 state for EMA stability, which is what users actually need.
- The `float16()` factory is **deleted** in Step 9A.4.7; it was architecturally unsound.

**Test migration:**
- All tests using `PrecisionConfig.float16()` must be updated to use `mixed_f16_f32()` (Step 9A.5.0).
- The test `test_fp16_state_rejection` validates the new constraint.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `pyproject.toml` | Does not list `ml_dtypes` as a dependency. |
| `requirements.latest.txt` | Does not list `ml_dtypes`. |
| `src/shared/precision_config.py` | `PrecisionConfig` frozen dataclass: fields `storage_dtype`, `compute_dtype`, `state_dtype`, derived constants. Seven factories (post-Phase-8A): `float32()`, `float16()`, `mixed_f16_f32()`, `float64()`, `mixed_f32_f64_state()`, `mixed_f16_f64_state()`, `mixed_f32_f64()`. `__post_init__` enforces `storage_dtype.itemsize <= compute_dtype.itemsize` and `storage_dtype.itemsize <= state_dtype.itemsize`. No FP8 rejection logic. |
| `CONCEPT.md` | Post-Phase-8A state: three-role precision model documented, Alchemist and Alchemist II scenarios present. No FP8 discussion. |
| `CONTRACT.md` | Post-Phase-8A state: Article 6 symbols include `STORAGE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_HALF`, `COMPUTE_TYPE_IS_DOUBLE`, `STATE_TYPE_IS_DOUBLE`. No FP8 symbols. |
| `src/tests/conftest.py` | `PRECISION_CONFIGS` list used for test parametrization. Contains seven factories. |

---

## 3. Task Breakdown

---

### Step 9A.1: Add `ml_dtypes` dependency

**Governing authority:** ADR-025 §1  
**Files:** `pyproject.toml`, `requirements.latest.txt`

Add `ml_dtypes>=0.2.0` to runtime dependencies. The `ml_dtypes` package provides NumPy-compatible FP8 dtypes (`float8_e4m3fn`, `float8_e5m2`) required for the FP8 storage representation.

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

**Governing authority:** ADR-025 §6.1  
**Files:** `scripts/generate_fp8_lut.py` (new), `kernels/fp8_lut.gen.h` (generated), `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` (generated)

Create a Python script to generate FP8→FP32 lookup tables for all backends. This step is placed in Phase 9A to eliminate cross-phase dependencies — Phases 9B, 9C, and 9D can then proceed in parallel without waiting on each other.

#### 9A.1.5.1: Create `scripts/generate_fp8_lut.py`

```python
#!/usr/bin/env python3
"""Generate FP8 lookup tables for all backends (ADR-025 §6.1).

Usage:
  # Generate OpenCL header:
  python scripts/generate_fp8_lut.py --backend opencl --output kernels/fp8_lut.gen.h
  
  # Generate CPU header:
  python scripts/generate_fp8_lut.py --backend cpu --output src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h

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

def format_c_array(name: str, values: list[float], is_opencl: bool = False) -> str:
    """Format lookup table as C/OpenCL constant array."""
    qualifier = "__constant" if is_opencl else "static const"
    lines = [f"{qualifier} float {name}[256] = {{"]
    for i in range(0, 256, 8):
        row = ", ".join(f"{v:.10e}f" for v in values[i:i+8])
        lines.append(f"    {row},")
    lines[-1] = lines[-1].rstrip(",")  # Remove trailing comma from last row
    lines.append("};")
    return "\n".join(lines)

def generate_header(e4m3_values: list[float], e5m2_values: list[float], 
                    is_opencl: bool, output_path: Path) -> None:
    """Generate a complete header file with both lookup tables."""
    guard = "CPU_FP8_LUT_GEN_H" if not is_opencl else "FP8_LUT_OPENCL_GEN_H"
    array_prefix = "cpu_" if not is_opencl else ""
    header = f"""/* Auto-generated by scripts/generate_fp8_lut.py — DO NOT EDIT
 * ml_dtypes version: {ml_dtypes.__version__}
 * Regenerate with: python scripts/generate_fp8_lut.py --backend {'opencl' if is_opencl else 'cpu'} --output {output_path}
 */

#ifndef {guard}
#define {guard}

/* E4M3: sign(1) + exp(4) + mantissa(3), bias=7, max=448, no inf/nan
 * 
 * Index derivation examples:
 *   0x00 = 0b00000000 → sign=0, exp=0, mant=0 → subnormal 0 × 2^(-6) = 0.0
 *   0x38 = 0b00111000 → sign=0, exp=7, mant=0 → 2^(7-7) × 1.0 = 1.0
 *   0x3C = 0b00111100 → sign=0, exp=7, mant=4 → 2^(7-7) × 1.5 = 1.5
 *   0x7E = 0b01111110 → sign=0, exp=15, mant=6 → 2^(15-7) × 1.75 = 448.0 (max)
 *   0x80 = 0b10000000 → sign=1, exp=0, mant=0 → -0.0
 *   0xB8 = 0b10111000 → sign=1, exp=7, mant=0 → -1.0
 *   0xFE = 0b11111110 → sign=1, exp=15, mant=6 → -448.0 (min)
 */
{format_c_array(f"{array_prefix}fp8_e4m3_to_float_lut", e4m3_values, is_opencl)}

/* E5M2: sign(1) + exp(5) + mantissa(2), bias=15, max=57344, no inf/nan
 *
 * Index derivation examples:
 *   0x00 = 0b00000000 → sign=0, exp=0, mant=0 → subnormal 0 × 2^(-14) = 0.0
 *   0x3C = 0b00111100 → sign=0, exp=15, mant=0 → 2^(15-15) × 1.0 = 1.0
 *   0x7B = 0b01111011 → sign=0, exp=30, mant=3 → 2^(30-15) × 1.75 = 57344.0 (max)
 *   0x80 = 0b10000000 → sign=1, exp=0, mant=0 → -0.0
 *   0xBC = 0b10111100 → sign=1, exp=15, mant=0 → -1.0
 *   0xFB = 0b11111011 → sign=1, exp=30, mant=3 → -57344.0 (min)
 */
{format_c_array(f"{array_prefix}fp8_e5m2_to_float_lut", e5m2_values, is_opencl)}

#endif /* {guard} */
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header)
    print(f"Generated: {output_path}")

def validate_tables(e4m3: list[float], e5m2: list[float]) -> None:
    """Validate LUT correctness against known values."""
    # E4M3 positive values
    assert e4m3[0] == 0.0, "E4M3 index 0x00 should be 0.0"
    # 0x38 = 0b00111000 → sign=0, exp=7, mant=0 → 2^(7-7) × 1.0 = 1.0
    assert abs(e4m3[0x38] - 1.0) < 1e-6, f"E4M3 index 0x38 should be 1.0, got {e4m3[0x38]}"
    # 0x7E = 0b01111110 → sign=0, exp=15, mant=6 → 2^8 × 1.75 = 448.0
    assert abs(e4m3[0x7E] - 448.0) < 1e-6, f"E4M3 index 0x7E should be 448.0 (max), got {e4m3[0x7E]}"
    
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
    assert abs(e5m2[0x7B] - 57344.0) < 1e-6, f"E5M2 index 0x7B should be 57344.0 (max), got {e5m2[0x7B]}"
    
    # E5M2 negative values
    assert abs(e5m2[0x80] - 0.0) < 1e-10, f"E5M2 index 0x80 should be -0.0, got {e5m2[0x80]}"
    # 0xBC = 0b10111100 → sign=1, exp=15, mant=0 → -1.0
    assert abs(e5m2[0xBC] - (-1.0)) < 1e-6, f"E5M2 index 0xBC should be -1.0, got {e5m2[0xBC]}"
    # 0xFB = 0b11111011 → sign=1, exp=30, mant=3 → -57344.0
    assert abs(e5m2[0xFB] - (-57344.0)) < 1e-6, f"E5M2 index 0xFB should be -57344.0 (min), got {e5m2[0xFB]}"
    
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

Add a CI check to verify the committed files match regeneration:

```yaml
# .github/workflows/ci.yml (excerpt)
- name: Verify FP8 LUT headers are up-to-date
  run: |
    python scripts/generate_fp8_lut.py --backend opencl --output /tmp/fp8_lut.gen.h
    diff kernels/fp8_lut.gen.h /tmp/fp8_lut.gen.h
    python scripts/generate_fp8_lut.py --backend cpu --output /tmp/cpu_fp8_lut.gen.h
    diff src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h /tmp/cpu_fp8_lut.gen.h
```

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

> FP8 storage (E4M3 or E5M2) is supported for maximum bandwidth efficiency. Two FP8 variants are provided: **E4M3** (4 exponent bits, 3 mantissa bits, max value 448) prioritizes precision; **E5M2** (5 exponent bits, 2 mantissa bits, max value 57344) prioritizes dynamic range. E4M3 is the primary target for gradient and activation storage. Neither format supports infinity or NaN — all bit patterns represent finite values. The host is responsible for scaling values to fit within FP8's representable range before storage; the existing Quadratic Scaling Policy already constrains gradients well below FP8 limits. `PrecisionConfig.fp8_e4m3()` provides E4M3 storage with FP32 compute and FP32 state; `PrecisionConfig.fp8_e4m3_f64()` adds FP64 compute and state for extended stability.

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

#### Amendment 9A.3.3 — Extend §5 PrecisionConfig contract

Update the `PrecisionConfig` paragraph to reference the new factories:

> Factory classmethods provide common configurations. For FP8 storage, any valid combination of compute (FP16/FP32/FP64) and state (FP32/FP64) precision is permitted via direct construction. FP8 is permitted only in the storage role; FP8 compute and FP8 state raise `ValueError`. FP16 state is also prohibited (EMA updates require FP32+ precision for stability).
>
> **FP8 factory methods:** `fp8_e4m3()`, `fp8_e5m2()` (FP32 compute/state), `fp8_e4m3_f16()`, `fp8_e5m2_f16()` (FP16 compute, FP32 state), `fp8_e4m3_f64()`, `fp8_e5m2_f64()` (FP64 compute/state). Additional combinations (e.g., FP16 compute + FP64 state) are constructed directly.

---

### Step 9A.4: Extend `PrecisionConfig`

**Governing authority:** ADR-025 §2  
**File:** `src/shared/precision_config.py`

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

#### 9A.4.3: Extend `__post_init__` with FP8 rejection

Add FP8 and FP16 state rejection logic **before** the existing itemsize invariants.

> **Why this ordering matters:** These checks must precede itemsize invariants because
> FP8's 1-byte itemsize would *pass* the `storage ≤ compute` check (1 ≤ 4), but FP8
> compute produces non-functional training. We reject architecturally unsound configs
> before validating size relationships.
>
> **FP8 itemsize note:** For valid FP8 configurations, the itemsize invariants naturally
> hold: FP8 storage (1 byte) ≤ FP32 compute (4 bytes) ≤ FP32/FP64 state (4/8 bytes).
> No special handling is needed — the existing checks pass.

```python
def __post_init__(self) -> None:
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
    
    # FP16 state is also prohibited (EMA precision erosion)
    if self.state_dtype == np.float16:
        raise ValueError(
            f"FP16 state is architecturally prohibited: "
            f"EMA updates lose precision over extended training. "
            f"Use FP32 or FP64 for state_dtype."
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

**Approach:** Add `storage_fp_format_min_subnormal: float` and `storage_mantissa_bits: int` as explicit dataclass fields, consistent with the existing pattern for `storage_fp_format_max`. This keeps the frozen dataclass simple and avoids `@cached_property` complexity.

Update the dataclass definition:

```python
@dataclass(frozen=True)
class PrecisionConfig:
    """Immutable three-role precision configuration for plan construction."""
    storage_dtype: np.dtype
    compute_dtype: np.dtype
    state_dtype: np.dtype

    storage_fp_format_max: float
    storage_fp_format_min_subnormal: float  # NEW: smallest positive representable value
    storage_mantissa_bits: int              # NEW: mantissa precision for quantization analysis
    compute_fp_format_max: float
    compute_epsilon: float
```

All factory methods must now pass these values explicitly. For FP8:

| Field | E4M3 Value | E5M2 Value | Source |
|:---|:---|:---|:---|
| `storage_fp_format_max` | `448.0` | `57344.0` | E4M3: 2^8 × 1.75, E5M2: 2^15 × 1.75 |
| `storage_fp_format_min_subnormal` | `0.001953125` | `0.0000152587890625` | 2^-9, 2^-16 (exact) |
| `storage_mantissa_bits` | `3` | `2` | Format definitions |

For existing formats (FP16/FP32/FP64), use `np.finfo()`.

**Note:** `ml_dtypes` FP8 types do not have a `np.finfo()` representation, so explicit constants are required for FP8 factories.

#### 9A.4.5: Update existing factory classmethods with new fields

**All seven existing factory methods must be updated** to include the new `storage_fp_format_min_subnormal` and `storage_mantissa_bits` fields. Failing to update them will cause `TypeError` at construction time.

```python
@classmethod
def float32(cls) -> "PrecisionConfig":
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float32),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=float(f32_info.max),
        storage_fp_format_min_subnormal=float(f32_info.tiny),
        storage_mantissa_bits=f32_info.nmant,  # 23 for FP32
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

# NOTE: float16() is DELETED in this phase (see Step 9A.4.7)

@classmethod
def mixed_f16_f32(cls) -> "PrecisionConfig":
    f16_info = np.finfo(np.float16)
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float16),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=float(f16_info.max),
        storage_fp_format_min_subnormal=float(f16_info.tiny),
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
        storage_fp_format_min_subnormal=float(f64_info.tiny),
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
        storage_fp_format_min_subnormal=float(f32_info.tiny),
        storage_mantissa_bits=f32_info.nmant,  # 23 for FP32
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def mixed_f16_f64_state(cls) -> "PrecisionConfig":
    f16_info = np.finfo(np.float16)
    f32_info = np.finfo(np.float32)
    return cls(
        storage_dtype=np.dtype(np.float16),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
        storage_fp_format_max=float(f16_info.max),
        storage_fp_format_min_subnormal=float(f16_info.tiny),
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
        storage_fp_format_min_subnormal=float(f32_info.tiny),
        storage_mantissa_bits=f32_info.nmant,  # 23 for FP32
        compute_fp_format_max=float(f64_info.max),
        compute_epsilon=float(f64_info.eps),
    )
```

#### 9A.4.6: Add FP8 factory classmethods

FP8 storage supports any valid compute (FP16/FP32/FP64) and state (FP32/FP64) combination.
The following table shows valid FP8 combinations:

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
        storage_fp_format_min_subnormal=0.001953125,  # 2^-9
        storage_mantissa_bits=3,
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def fp8_e4m3_f16(cls) -> "PrecisionConfig":
    """E4M3 storage (8-bit), FP16 compute, FP32 state.
    
    Maximum bandwidth with native FP16 compute.
    Suitable for hardware with fast FP16 ALUs.
    """
    f16_info = np.finfo(np.float16)
    return cls(
        storage_dtype=FP8_E4M3,
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=448.0,
        storage_fp_format_min_subnormal=0.001953125,  # 2^-9
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
        storage_fp_format_min_subnormal=0.001953125,  # 2^-9
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
        storage_fp_format_min_subnormal=0.0000152587890625,  # 2^-16, exact
        storage_mantissa_bits=2,
        compute_fp_format_max=float(f32_info.max),
        compute_epsilon=float(f32_info.eps),
    )

@classmethod
def fp8_e5m2_f16(cls) -> "PrecisionConfig":
    """E5M2 storage (8-bit), FP16 compute, FP32 state.
    
    Maximum bandwidth with wider dynamic range and native FP16 compute.
    """
    f16_info = np.finfo(np.float16)
    return cls(
        storage_dtype=FP8_E5M2,
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=57344.0,
        storage_fp_format_min_subnormal=0.0000152587890625,  # 2^-16, exact
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
        storage_fp_format_min_subnormal=0.0000152587890625,  # 2^-16, exact
        storage_mantissa_bits=2,
        compute_fp_format_max=float(f64_info.max),
        compute_epsilon=float(f64_info.eps),
    )
```

#### 9A.4.7: Delete `float16()` factory

**Remove** the `float16()` classmethod entirely. It produced architecturally unsound configurations (FP16 state). Users should use `mixed_f16_f32()` instead.

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
cfg = PrecisionConfig(
    storage_dtype=np.dtype(ml_dtypes.float8_e4m3fn),
    compute_dtype=np.dtype(np.float16),
    state_dtype=np.dtype(np.float64),
)

# FP8 E4M3 with FP32 compute and FP64 state  
cfg = PrecisionConfig(
    storage_dtype=np.dtype(ml_dtypes.float8_e4m3fn),
    compute_dtype=np.dtype(np.float32),
    state_dtype=np.dtype(np.float64),
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
- `PrecisionConfig(storage_dtype=..., compute_dtype=..., state_dtype=np.float16)` raises `ValueError`
- `fp8_e4m3().storage_dtype.itemsize == 1`
- `fp8_e4m3().storage_fp_format_max == 448.0`

---

### Step 9A.5: Update test fixtures

**Governing authority:** ADR-025 §9.2  
**Files:** `src/tests/conftest.py`, `src/tests/test_precision_config.py` (new or existing)

#### 9A.5.0: Migrate existing `float16()` usages (BREAKING CHANGE)

Before adding FP8 fixtures, update all existing test usages of `PrecisionConfig.float16()` to `PrecisionConfig.mixed_f16_f32()`:

```bash
# Find all factory usages
grep -r "PrecisionConfig.float16()" tests/ src/tests/

# Also find direct constructions with FP16 state (these will also break)
grep -r "state_dtype=np.float16" tests/ src/tests/ src/shared/
grep -r "state_dtype=.*float16" tests/ src/tests/ src/shared/

# Files requiring migration (based on current codebase):
# Test files:
# - tests/bench_cpu_backend.py
# - tests/tier2/test_multi_precision_config.py
# - tests/tier1/test_precision_config.py
# - tests/tier1/conftest.py
# - tests/tier2/cpu/test_cpu_crash_regressions.py
# - src/tests/test_fp64_precision.py
# - src/tests/test_mixed_precision.py
#
# Production code (NOT test files — also requires migration):
# - src/shared/model_spec.py → ModelSpec.float16() factory must be updated
#
# Direct constructions to check:
# - Any PrecisionConfig(..., state_dtype=np.float16)
# - Parametrized test values that include FP16 state
```

**Migration:**
- Replace `PrecisionConfig.float16()` → `PrecisionConfig.mixed_f16_f32()`
- Replace direct `state_dtype=np.float16` constructions with `state_dtype=np.float32`
- If test specifically validated uniform FP16, convert to rejection test
- Update `ModelSpec.float16()` to use `mixed_f16_f32()` internally

**`ModelSpec.float16()` migration (in `src/shared/model_spec.py`):**

```python
# BEFORE (architecturally unsound — FP16 state causes EMA drift)
@classmethod
def float16(cls) -> "ModelSpec":
    return cls(precision=PrecisionConfig.float16())

# AFTER (FP16 storage/compute, FP32 state for stability)
@classmethod  
def float16(cls) -> "ModelSpec":
    """FP16 storage and compute with FP32 state.
    
    Note: This factory now uses mixed_f16_f32() internally.
    Pure FP16 (including state) was architecturally unsound and is no longer supported.
    See ADR-025 for rationale.
    """
    return cls(precision=PrecisionConfig.mixed_f16_f32())
```

> **Alternative:** Delete `ModelSpec.float16()` entirely and require explicit `ModelSpec(precision=PrecisionConfig.mixed_f16_f32())`. This is more verbose but avoids the confusing "float16" name that implies uniform FP16.

**Rationale:** FP16 state was always architecturally unsound — EMA updates lose precision over extended training. The `float16()` factory existed for API completeness but produced unstable training for long runs. The three-role model (ADR-020) should have prohibited this from the start.

#### 9A.5.1: Extend `PRECISION_CONFIGS` fixture

Add FP8 configs to the parametrization list in `conftest.py`:

```python
# FP8 configurations (all valid compute/state combinations)
FP8_PRECISION_CONFIGS = [
    # E4M3 variants
    PrecisionConfig.fp8_e4m3(),       # FP32/FP32
    PrecisionConfig.fp8_e4m3_f16(),   # FP16/FP32
    PrecisionConfig.fp8_e4m3_f64(),   # FP64/FP64
    # E4M3 with mixed compute/state (direct construction)
    PrecisionConfig(
        storage_dtype=np.dtype(ml_dtypes.float8_e4m3fn),
        compute_dtype=np.dtype(np.float16),
        state_dtype=np.dtype(np.float64),
    ),
    PrecisionConfig(
        storage_dtype=np.dtype(ml_dtypes.float8_e4m3fn),
        compute_dtype=np.dtype(np.float32),
        state_dtype=np.dtype(np.float64),
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

# Track FP8 backend support — update as phases complete
_FP8_UNSUPPORTED_BACKENDS = {
    "opencl",   # Remove after Phase 9B
    "cpu",      # Remove after Phase 9C  
    "vulkan",   # Remove after Phase 9D
}

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
            if backend_name in _FP8_UNSUPPORTED_BACKENDS:
                item.add_marker(pytest.mark.skip(
                    reason=f"FP8 not yet implemented for {backend_name} backend"
                ))
        # Case 2: FP8 + no backend param → skip if ANY backend unsupported
        # (conservative: test likely exercises all backends or a default backend)
        elif _FP8_UNSUPPORTED_BACKENDS:
            item.add_marker(pytest.mark.skip(
                reason=f"FP8 not yet implemented (waiting on: {', '.join(sorted(_FP8_UNSUPPORTED_BACKENDS))})"
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
    """ADR-025 §2.2: Precision role constraints (FP8 storage-only, FP16 state prohibited)."""

    def test_fp8_e4m3_compute_rejection(self):
        """FP8 compute raises ValueError."""
        with pytest.raises(ValueError, match="FP8 compute is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float32),
                compute_dtype=np.dtype(FP8_E4M3),
                state_dtype=np.dtype(np.float32),
            )

    def test_fp8_e5m2_compute_rejection(self):
        """FP8 compute raises ValueError (E5M2 variant)."""
        with pytest.raises(ValueError, match="FP8 compute is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float32),
                compute_dtype=np.dtype(FP8_E5M2),
                state_dtype=np.dtype(np.float32),
            )

    def test_fp8_e4m3_state_rejection(self):
        """FP8 state raises ValueError."""
        with pytest.raises(ValueError, match="FP8 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float32),
                compute_dtype=np.dtype(np.float32),
                state_dtype=np.dtype(FP8_E4M3),
            )

    def test_fp8_e5m2_state_rejection(self):
        """FP8 state raises ValueError (E5M2 variant)."""
        with pytest.raises(ValueError, match="FP8 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float32),
                compute_dtype=np.dtype(np.float32),
                state_dtype=np.dtype(FP8_E5M2),
            )

    def test_fp16_state_rejection(self):
        """FP16 state raises ValueError (EMA precision erosion).
        
        Note: This constraint applies to ALL configurations, not just FP8.
        FP16 state lacks sufficient precision for EMA updates over extended training.
        """
        with pytest.raises(ValueError, match="FP16 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float16),
                compute_dtype=np.dtype(np.float16),
                state_dtype=np.dtype(np.float16),
            )

    def test_fp16_state_rejection_with_fp32_storage(self):
        """FP16 state rejected even with FP32 storage.
        
        Note: This test complements test_fp16_state_rejection() by verifying
        the constraint applies regardless of storage/compute precision — the
        FP16 state prohibition is universal, not FP8-specific.
        """
        with pytest.raises(ValueError, match="FP16 state is architecturally prohibited"):
            PrecisionConfig(
                storage_dtype=np.dtype(np.float32),
                compute_dtype=np.dtype(np.float32),
                state_dtype=np.dtype(np.float16),
            )


class TestFP8ValidCombinations:
    """ADR-025: All valid FP8 storage + compute/state combinations."""

    @pytest.mark.parametrize("compute_dtype,state_dtype", [
        (np.float16, np.float32),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ])
    def test_e4m3_valid_combinations(self, compute_dtype, state_dtype):
        """E4M3 storage accepts all valid compute/state combinations."""
        cfg = PrecisionConfig(
            storage_dtype=np.dtype(FP8_E4M3),
            compute_dtype=np.dtype(compute_dtype),
            state_dtype=np.dtype(state_dtype),
        )
        assert cfg.storage_dtype == np.dtype(FP8_E4M3)
        assert cfg.compute_dtype == np.dtype(compute_dtype)
        assert cfg.state_dtype == np.dtype(state_dtype)

    @pytest.mark.parametrize("compute_dtype,state_dtype", [
        (np.float16, np.float32),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.float64, np.float64),
    ])
    def test_e5m2_valid_combinations(self, compute_dtype, state_dtype):
        """E5M2 storage accepts all valid compute/state combinations."""
        cfg = PrecisionConfig(
            storage_dtype=np.dtype(FP8_E5M2),
            compute_dtype=np.dtype(compute_dtype),
            state_dtype=np.dtype(state_dtype),
        )
        assert cfg.storage_dtype == np.dtype(FP8_E5M2)
        assert cfg.compute_dtype == np.dtype(compute_dtype)
        assert cfg.state_dtype == np.dtype(state_dtype)


class TestFP8Factories:
    """ADR-025 §2.3: FP8 factory classmethods."""

    def test_fp8_e4m3_factory(self):
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_dtype == np.dtype(FP8_E4M3)
        assert cfg.compute_dtype == np.dtype(np.float32)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e5m2_factory(self):
        cfg = PrecisionConfig.fp8_e5m2()
        assert cfg.storage_dtype == np.dtype(FP8_E5M2)
        assert cfg.compute_dtype == np.dtype(np.float32)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e4m3_f16_factory(self):
        cfg = PrecisionConfig.fp8_e4m3_f16()
        assert cfg.storage_dtype == np.dtype(FP8_E4M3)
        assert cfg.compute_dtype == np.dtype(np.float16)
        assert cfg.state_dtype == np.dtype(np.float32)
        assert cfg.storage_dtype.itemsize == 1

    def test_fp8_e5m2_f16_factory(self):
        cfg = PrecisionConfig.fp8_e5m2_f16()
        assert cfg.storage_dtype == np.dtype(FP8_E5M2)
        assert cfg.compute_dtype == np.dtype(np.float16)
        assert cfg.state_dtype == np.dtype(np.float32)

    def test_fp8_e4m3_f64_factory(self):
        cfg = PrecisionConfig.fp8_e4m3_f64()
        assert cfg.storage_dtype == np.dtype(FP8_E4M3)
        assert cfg.compute_dtype == np.dtype(np.float64)
        assert cfg.state_dtype == np.dtype(np.float64)

    def test_fp8_e5m2_f64_factory(self):
        cfg = PrecisionConfig.fp8_e5m2_f64()
        assert cfg.storage_dtype == np.dtype(FP8_E5M2)
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

    def test_e4m3_all_256_patterns_finite(self):
        """All 256 E4M3 bit patterns produce finite float values (no NaN/inf)."""
        all_bits = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)
        all_float = all_bits.astype(np.float32)
        assert np.all(np.isfinite(all_float)), "E4M3 should have no NaN/inf values"

    def test_e5m2_all_256_patterns_finite(self):
        """All 256 E5M2 bit patterns produce finite float values (no NaN/inf)."""
        all_bits = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e5m2)
        all_float = all_bits.astype(np.float32)
        assert np.all(np.isfinite(all_float)), "E5M2 should have no NaN/inf values"

    def test_ml_dtypes_version_compatibility(self):
        """ml_dtypes version is compatible with expected FP8 behavior."""
        import ml_dtypes
        version = tuple(int(x) for x in ml_dtypes.__version__.split('.')[:2])
        assert version >= (0, 2), f"ml_dtypes {ml_dtypes.__version__} < 0.2.0; FP8 behavior may differ"
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
| **FP16 state prohibition breaks `float16()` factory** | **High** | **Medium** | Breaking change is intentional. Migration path documented. Tests must be updated to use `mixed_f16_f32()`. The `float16()` factory is removed; equivalent functionality via `mixed_f16_f32()` (FP16 storage/compute, FP32 state). |

---

## 6. Files Modified Summary

| File | Change Type | Description |
|:---|:---|:---|
| `pyproject.toml` | **Edit** | Add `ml_dtypes>=0.2.0` dependency |
| `requirements.latest.txt` | **Edit** | Add `ml_dtypes>=0.2.0` |
| `scripts/generate_fp8_lut.py` | **New** | Python script to generate FP8→FP32 lookup tables for all backends |
| `kernels/fp8_lut.gen.h` | **New** (generated) | OpenCL `__constant` arrays for E4M3/E5M2 → float conversion |
| `src/backends/cpu/kernel_sources/cpu_fp8_lut.gen.h` | **New** (generated) | CPU `static const` arrays for E4M3/E5M2 → float conversion |
| `CONCEPT.md` | **Edit** | Add FP8 storage rationale (§2, §11), "Bandwidth Extremist" scenario |
| `CONTRACT.md` | **Edit** | Add FP8 build-time symbols (Article 6), factory method reference |
| `src/shared/precision_config.py` | **Edit** | Add FP8 imports, constants, factories, `__post_init__` constraints; **delete** `float16()` factory |
| `src/tests/conftest.py` | **Edit** | Add FP8 configs to `PRECISION_CONFIGS`, add `skip_if_fp8_unsupported` fixture |
| `src/tests/test_precision_config.py` | **Edit/New** | Add FP8 constraint tests, factory tests, derived constant tests |
| `tests/bench_cpu_backend.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `tests/tier2/test_multi_precision_config.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `tests/tier1/test_precision_config.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `tests/tier1/conftest.py` | **Edit** | Migrate `float16()` → `mixed_f16_f32()` |
| `src/shared/model_spec.py` | **Edit** | Update `ModelSpec.float16()` to use `mixed_f16_f32()` internally |
