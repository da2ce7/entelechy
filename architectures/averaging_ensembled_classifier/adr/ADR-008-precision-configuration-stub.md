# ADR-008: Precision Configuration

**Status:** STUB (NARROWED — Backend resolves native types independently)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-006  
**Blocks:** —

---

## Context

`PrecisionContext` currently carries `SCALAR_C_TYPE_NAME` (explicitly "for the OpenCL compiler"). Under the plan model, the shared layer works exclusively with numpy types (`numpy.float16`, `numpy.float32`). Backend renderers map to their native type systems.

The following values remain in the shared layer as they are precision-derived but backend-neutral:
- `FP_FORMAT_MAX`: Used by `StabilizationPolicy` for safety ceiling calculations (CONCEPT.md §3.4).
- `NUMERICAL_STABILITY_EPSILON`: Used by plan builder for kernel scalar parameters.
- `numpy_dtype`: Used by `MemoryLayout` for byte-size calculations.

---

## Narrowed Direction

`PrecisionContext` carries `numpy_dtype`, `fp_format_max`, and `epsilon`. Backends derive everything else from `numpy_dtype` (`float16` → `half` for OpenCL, `VK_FORMAT_R16_SFLOAT` for Vulkan, `_Float16` or emulated for CPU).

---

## Remaining Decision

Does `PrecisionContext` retain a backend-neutral `bit_width` field that backends map from, or does each backend independently determine its type from the numpy dtype? This is a minor design question — the practical direction is clear.

---

## Tensions

- CPU FP16 is not universally hardware-accelerated. The CPU backend may need to declare FP16 as unsupported, which the shared layer must handle gracefully (e.g., by refusing to construct a plan for an unsupported precision).

---

## References

- [ADR-006: Hardware Profile](ADR-006-hardware-profile-stub.md) — `HardwareProfile` supplies hardware data; precision interacts with hardware capability
- [CONCEPT.md](../CONCEPT.md) — §3.4 Safety ceiling calculations using `FP_FORMAT_MAX`
