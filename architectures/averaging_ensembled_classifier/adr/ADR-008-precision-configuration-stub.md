# ADR-008: Precision Configuration

**Status:** STUB (NARROWED — Backend resolves native types independently)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
**Blocks:** —

---

## Context

`PrecisionContext` currently carries `SCALAR_C_TYPE_NAME` (explicitly "for the OpenCL compiler") and serves as a base class for `DiscoveredArchConstants`, coupling precision to hardware discovery. ADR-006 (ACCEPTED) decouples these concerns: `HardwareProfile` is a standalone frozen dataclass with no `PrecisionContext` inheritance. The plan builder receives both as separate inputs:

```python
plan = build_execution_plan(
    model_spec=model_spec,              # includes PrecisionContext
    hardware_profile=hardware_profile,  # precision-appropriate simd_width
    stabilization_policy=policy,
    batch_params=batch_params,
)
```

Under the plan model, the shared layer works exclusively with numpy types (`numpy.float16`, `numpy.float32`). Backend renderers map to their native type systems (`float`/`half` for OpenCL, `VK_FORMAT_*` for Vulkan, `_Float16` or emulated for CPU).

The following values remain in the shared layer as they are precision-derived but backend-neutral:
- `FP_FORMAT_MAX`: Used by `StabilizationPolicy` for safety ceiling calculations (CONCEPT.md §3.4).
- `NUMERICAL_STABILITY_EPSILON`: Used by plan builder for kernel scalar parameters.
- `numpy_dtype`: Used by `MemoryLayout` for byte-size calculations.

The current `SCALAR_C_TYPE_NAME` property is backend-specific (OpenCL-only) and must not remain in the shared `PrecisionContext`.

---

## Narrowed Direction

`PrecisionContext` becomes a shared-layer frozen dataclass carrying `numpy_dtype`, `fp_format_max`, and `epsilon`. Backends derive everything else from `numpy_dtype` (`float16` → `half` for OpenCL, `VK_FORMAT_R16_SFLOAT` for Vulkan, `_Float16` or emulated for CPU).

ADR-006 establishes that `HardwareProfile.simd_width` is precision-dependent (FP16 may have twice the SIMD width of FP32). The backend must construct the `HardwareProfile` for the target precision. A cross-check between `simd_width` and `PrecisionContext.numpy_dtype.itemsize` at plan-construction time can detect mismatches.

---

## Remaining Decision

Does `PrecisionContext` retain a backend-neutral `bit_width` field that backends map from, or does each backend independently determine its native type from the numpy dtype? This is a minor design question — the practical direction is clear.

---

## Tensions

- CPU FP16 is not universally hardware-accelerated. The CPU backend may need to declare FP16 as unsupported, which the shared layer must handle gracefully (e.g., by refusing to construct a plan for an unsupported precision).
- The current `Float32Context` / `Float16Context` class hierarchy uses abstract properties. Under the new model, a single `PrecisionContext` frozen dataclass parameterized by `numpy_dtype` may replace the hierarchy entirely.

---

## References

- [ADR-006: Hardware Profile](ADR-006-hardware-profile.md) — `HardwareProfile` is precision-decoupled; `simd_width` is precision-appropriate; plan builder receives both separately
- [CONCEPT.md](../CONCEPT.md) — §3.4 Safety ceiling calculations using `FP_FORMAT_MAX`
