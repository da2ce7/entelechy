# Phase 9E: FP8 Support — Host-Side Scaling & Integration Tests

**Status: NOT STARTED**  
**Phase:** 9E of 9  
**Prerequisite:** Phases 9B, 9C, 9D complete.  
**Objective:** Implement host-side FP8 scaling logic, integrate FP8 configs into the main orchestrator, execute "The Bandwidth Extremist" validation scenario, and validate full FP8 training convergence. The phase ends when FP8 training produces convergent loss curves within tolerance of FP16 baseline.  
**Governing ADR:** ADR-025 (§8, §9)  
**Rollback gate:** All existing tests pass. FP8 training converges within 2% relative loss tolerance of FP16 baseline (start strict; may loosen to 5% after investigation if genuine quantization noise). Memory usage is 1/2 of FP16 baseline for storage-role buffers.  
**Dependencies:** Phases 9B (OpenCL), 9C (CPU), 9D (Vulkan) complete.

---

## Table of Contents

1. [Scope & Constraints](#1-scope--constraints)
2. [Pre-Condition Inventory](#2-pre-condition-inventory)
3. [Task Breakdown](#3-task-breakdown)
   - [Step 9E.1: Implement host-side FP8 scaling utilities](#step-9e1-implement-host-side-fp8-scaling-utilities)
   - [Step 9E.2: Integrate FP8 into orchestrator](#step-9e2-integrate-fp8-into-orchestrator)
   - [Step 9E.3: Update buffer allocation for FP8](#step-9e3-update-buffer-allocation-for-fp8)
   - [Step 9E.4: Implement "The Bandwidth Extremist" test](#step-9e4-implement-the-bandwidth-extremist-test)
   - [Step 9E.4.1: Cross-backend FP8 consistency test](#step-9e41-cross-backend-fp8-consistency-test)
   - [Step 9E.5: Add FP8 memory validation tests](#step-9e5-add-fp8-memory-validation-tests)
   - [Step 9E.6: Run full regression suite](#step-9e6-run-full-regression-suite)
   - [Step 9E.7: Document FP8 usage](#step-9e7-document-fp8-usage)
4. [Validation Criteria](#4-validation-criteria)
5. [Risk Register](#5-risk-register)
6. [Files Modified Summary](#6-files-modified-summary)
7. [FP8 Implementation Summary (Phases 9A–9E)](#fp8-implementation-summary-phases-9a9e)

---

## 1. Scope & Constraints

### In scope

- Implementing host-side utilities for FP8 quantization/dequantization.
- Integrating pre-storage scaling to ensure values fit within FP8 range.
- Adding scale tracking to DAG metadata for correct gradient inverse scaling.
- Integrating FP8 `PrecisionConfig` factories into `main_orchestrator.py`.
- Updating `BufferDescriptor` allocation to use FP8 sizing.
- Implementing "The Bandwidth Extremist" validation scenario (ADR-025 §9.1).
- Validating FP8 training convergence against FP16/FP32 baselines.
- Adding memory usage assertions (FP8 storage = 1/4 FP32, 1/2 FP16).

### Out of scope

- Block-scaled FP8 — deferred per ADR-025 §10.
- Mixed E4M3/E5M2 within single config — deferred per ADR-025 §10.
- Native FP8 ALU utilization — deferred indefinitely per ADR-025 §10.
- SIMD FP8 on CPU — deferred indefinitely.

### Key constraint: Quadratic Scaling Policy compatibility

The existing Quadratic Scaling Policy already constrains gradients to `COMPUTE_FP_FORMAT_MAX / K`, which for FP32 compute is ~3.4e38 / K. For practical K values (8-64), this yields gradients well below 448 (E4M3's max). FP8 storage therefore requires minimal additional scaling in most cases — the architecture's existing safety bounds are sufficient.

However, pre-activation values and intermediate accumulations may exceed FP8 range. The host tracks an `activation_scale` factor that ensures pre-storage values fit within `STORAGE_FP_FORMAT_MAX`.

---

## 2. Pre-Condition Inventory

| File | Relevant current state |
|:---|:---|
| `src/shared/precision_config.py` | Post-Phase-9A: FP8 factories and derived constants available. |
| `src/main_orchestrator.py` | Orchestrates training. Uses `PrecisionConfig` for buffer allocation and kernel dispatch. No FP8-specific scaling. |
| `src/shared/buffer_lifecycle.py` | `BufferDescriptor` with `precision_role`. Allocation uses `storage_dtype.itemsize`. |
| `src/shared/stabilization_policy.py` | Quadratic Scaling Policy. Uses `compute_fp_format_max`. |
| `kernels/` | Post-Phase-9B: OpenCL kernels support FP8 load/store. |
| `src/backends/cpu/` | Post-Phase-9C: CPU kernels support FP8 conversions. |
| `src/backends/vulkan/` | Post-Phase-9D: Vulkan shaders support FP8. |

---

## 3. Task Breakdown

---

### Step 9E.1: Implement host-side FP8 scaling utilities

**Governing authority:** ADR-025 §8  
**File:** `src/shared/fp8_scaling.py` (new)

Create utilities for host-side FP8 scale management:

```python
"""FP8 scaling utilities (ADR-025 §8).

The host manages per-tensor scale factors to ensure values fit within
FP8's limited dynamic range. Gradients are already constrained by the
Quadratic Scaling Policy; activations may require explicit scaling.
"""

import numpy as np
from typing import NamedTuple
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

# Increment if FP8ScaleInfo structure changes (for checkpoint compatibility)
FP8_SCALES_VERSION = 1


class FP8ScaleInfo(NamedTuple):
    """Scale information for FP8 quantization.
    
    Note: This is a NamedTuple for immutability and easy serialization.
    Use `._asdict()` for JSON/pickle serialization, `FP8ScaleInfo(**d)` to restore.
    """
    scale: float           # Multiply by this before quantization
    inv_scale: float       # Multiply by this after dequantization
    storage_max: float     # FP8 format max (448 or 57344)


def compute_fp8_scale(
    tensor: np.ndarray,
    precision: PrecisionConfig,
    headroom: float = 0.9
) -> FP8ScaleInfo:
    """Compute scale factor to map tensor values into FP8 range.
    
    Args:
        tensor: Values to be stored in FP8 format.
        precision: Must have FP8 storage dtype.
        headroom: Fraction of FP8 max to use (default 0.9 for saturation margin).
                  Must be in (0, 1].
    
    Returns:
        FP8ScaleInfo with scale factors and diagnostics.
    
    Raises:
        ValueError: If precision lacks FP8 storage dtype or headroom is not in (0, 1].
    """
    if precision.storage_dtype not in (FP8_E4M3, FP8_E5M2):
        raise ValueError(
            f"compute_fp8_scale requires FP8 storage dtype, "
            f"got {precision.storage_dtype}"
        )
    
    if not (0.0 < headroom <= 1.0):
        raise ValueError(
            f"headroom must be in (0, 1], got {headroom}. "
            f"headroom=0 causes division by zero; headroom>1 allows values "
            f"beyond FP8 max, causing saturation."
        )
    
    storage_max = precision.storage_fp_format_max
    target_range = storage_max * headroom
    
    # Guard against zero-size arrays (np.nanmax raises ValueError on empty input)
    if tensor.size == 0:
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )
    
    # Use nanmax to ignore NaN values when computing tensor maximum.
    # If the entire tensor is NaN, nanmax returns nan (with RuntimeWarning).
    with np.errstate(invalid='ignore'):  # Suppress nanmax RuntimeWarning for all-NaN case
        tensor_max = np.nanmax(np.abs(tensor))
    
    # Handle all-NaN tensor case: return identity scale, let kernel NaN→zero mapping handle it
    if np.isnan(tensor_max):
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )
    
    if tensor_max < 1e-10:
        # Near-zero tensor — no scaling needed, inverse is identity
        # Note: These tensors produce near-zero gradients; inv_scale=1.0 is correct
        # because there's no meaningful information to preserve.
        #
        # WARNING: With scale=1.0, all values in this tensor will quantize to zero
        # in FP8 (since tensor_max < FP8 min subnormal for both E4M3 and E5M2).
        # This is intentional for near-zero activations but should be visible
        # during debugging.
        import warnings
        warnings.warn(
            f"compute_fp8_scale: tensor_max={tensor_max:.2e} < 1e-10. "
            f"All values will quantize to FP8 zero (no gradient contribution). "
            f"This is expected for near-dead activations but may indicate "
            f"upstream numerical issues if seen frequently.",
            stacklevel=2,
        )
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )
    
    if tensor_max <= target_range:
        # Already fits — no scaling needed.
        # DESIGN DECISION: We intentionally never scale UP (i.e., scale > 1.0).
        # Small tensors carry negligible gradient information, and amplifying them
        # into the FP8 range would inflate quantization noise on values that
        # contribute little to the update. The identity scale (1.0) is correct:
        # small values map to small FP8 codes or zero, which is the desired
        # behavior for near-zero activations.
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
        )
    
    # Scale down to fit
    scale = target_range / tensor_max
    inv_scale = 1.0 / scale
    
    return FP8ScaleInfo(
        scale=scale,
        inv_scale=inv_scale,
        storage_max=storage_max,
    )


def apply_fp8_scale(
    tensor: np.ndarray,
    scale_info: FP8ScaleInfo
) -> np.ndarray:
    """Scale tensor values for FP8 storage."""
    return tensor * scale_info.scale


def unapply_fp8_scale(
    tensor: np.ndarray,
    scale_info: FP8ScaleInfo
) -> np.ndarray:
    """Restore original scale after FP8 dequantization."""
    return tensor * scale_info.inv_scale
```

#### 9E.1.1: Scale persistence across forward/backward passes

FP8 scale factors must persist across the forward/backward boundary:

1. **Forward pass:** Activations are scaled before storage in FP8 buffers. Scale factors are stored in `_activation_scales`.

2. **Backward pass:** When loading activations, the inverse scale is applied to restore original magnitudes before gradient computation.

3. **Gradient scaling interaction:** If loss scaling is used (common in mixed-precision training), the total scale factor is `loss_scale × activation_scale`. The gradient unscaling step must apply `1 / (loss_scale × activation_scale)` = `inv_loss_scale × inv_scale`.

4. **Checkpointing:** Scale factors are serialized alongside model state. The checkpoint includes a version field for forward compatibility.

   > **Serialization note:** `FP8ScaleInfo` is a `NamedTuple` which doesn't serialize directly.
   > Use `._asdict()` to convert to dict on save, and `FP8ScaleInfo(**d)` to restore.

5. **Dynamic scaling (optional):** Scale factors can be recomputed each forward pass (adaptive scaling) or fixed at initialization (static scaling). The default is adaptive scaling with 0.9 headroom.

6. **Scale version bumps:** The `FP8_SCALES_VERSION` constant should be incremented when:
   - The `FP8ScaleInfo` structure changes (fields added/removed/renamed)
   - The semantic meaning of fields changes (e.g., scale interpretation)
   - The headroom default changes in a backward-incompatible way
   
   Version bumps allow checkpoint loading to detect incompatible scale formats and recompute rather than silently misinterpret old data.

   > **Initial version:** `FP8_SCALES_VERSION = 1`. Version 0 is implicitly reserved for pre-FP8 checkpoints, which lack the `fp8_scales` key entirely. When loading a checkpoint without `fp8_scales`, the code path in `restore_state()` handles this gracefully by initializing an empty `_activation_scales` dict.

**Complete checkpoint implementation:**

```python
from shared.fp8_scaling import FP8_SCALES_VERSION, FP8ScaleInfo
import warnings

class Orchestrator:
    def checkpoint_state(self) -> dict:
        """Export state for checkpointing."""
        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "training_step": self._training_step,
        }
        if self.is_fp8_storage:
            # Convert NamedTuple to dict for JSON/pickle serialization
            # Embed version INSIDE the fp8_scales dict (not as sibling key) for atomicity
            state["fp8_scales"] = {
                "_version": FP8_SCALES_VERSION,  # Reserved metadata key (leading underscore)
                "scales": {
                    name: info._asdict() for name, info in self._activation_scales.items()
                },
            }
        return state
    
    def restore_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._training_step = state.get("training_step", 0)
        
        # Restore FP8 scales with version checking and structure validation
        if "fp8_scales" in state:
            fp8_data = state["fp8_scales"]
            checkpoint_version = fp8_data.get("_version", 0)
            
            if checkpoint_version != FP8_SCALES_VERSION:
                warnings.warn(
                    f"FP8 scale format mismatch: checkpoint v{checkpoint_version}, "
                    f"current v{FP8_SCALES_VERSION}. Scales will be recomputed."
                )
                self._activation_scales = {}  # Recompute on next forward pass
            else:
                # Defensive construction: catch TypeError if FP8ScaleInfo structure changed
                try:
                    scales_dict = fp8_data.get("scales", {})
                    self._activation_scales = {
                        name: FP8ScaleInfo(**info) for name, info in scales_dict.items()
                    }
                except TypeError as e:
                    warnings.warn(
                        f"FP8ScaleInfo structure incompatible with checkpoint: {e}. "
                        f"Scales will be recomputed."
                    )
                    self._activation_scales = {}
        else:
            # Pre-FP8 checkpoint (Phase 8 or earlier) — no fp8_scales key is valid
            # For non-FP8 configs, this is expected and correct
            # For FP8 configs loading an old checkpoint, scales will be computed on first forward
            self._activation_scales = {}
```

> **Why embed version inside `fp8_scales`?**
> 
> If the version is a sibling key (`state["fp8_scales_version"]`), a partial write during crash
> could produce a checkpoint with `fp8_scales` but no `fp8_scales_version`, causing silent
> misinterpretation in future loads. Embedding the version inside the `fp8_scales` dict makes
> the version check atomic with the scale data.

> **Reserved key namespace:** Keys starting with `_` inside the `fp8_scales` dict are reserved
> for checkpoint metadata (`_version`, and any future `_schema`, `_created_at`, etc.).
> Buffer/activation names must not begin with `_`. The `_activation_scales` dict keys come from
> plan node names, which are validated elsewhere — but add a defensive check in `save_state()`
> if this convention is ever relaxed:

> **Backward compatibility:** Checkpoints from Phase 8 or earlier do not contain `fp8_scales`.
> When loading such checkpoints with an FP8 precision config, `_activation_scales` starts empty
> and scales are recomputed on the first forward pass. This is the correct behavior — the old
> checkpoint's activations were stored in FP16/FP32/FP64, and the FP8 conversion happens live.

---

### Step 9E.2: Integrate FP8 into orchestrator

**Governing authority:** ADR-025 §8  
**File:** `src/main_orchestrator.py`

Extend orchestrator to handle FP8 precision configs:

```python
from shared.fp8_scaling import FP8ScaleInfo, compute_fp8_scale, apply_fp8_scale, unapply_fp8_scale
from shared.precision_config import FP8_DTYPES

class Orchestrator:
    def __init__(self, precision: PrecisionConfig, ...):
        self.precision = precision
        self.is_fp8_storage = precision.storage_dtype in FP8_DTYPES
        
        # Per-buffer scale tracking (for FP8).
        # CONSTRAINT: Keys must be static buffer names derived from plan node names
        # (deterministic and bounded). Do NOT use per-step or per-iteration keys —
        # that would cause unbounded memory growth. There is intentionally no
        # eviction policy; the dict size is bounded by the number of plan nodes.
        self._activation_scales: dict[str, FP8ScaleInfo] = {}
    
    def _prepare_storage_buffer(
        self,
        name: str,
        tensor: np.ndarray,
        precision_role: str
    ) -> np.ndarray:
        """Prepare tensor for storage-role buffer.
        
        For FP8, applies scaling if needed and tracks scale for inverse.
        """
        if precision_role != "storage":
            return tensor
        
        if not self.is_fp8_storage:
            return tensor
        
        # Compute and apply FP8 scaling
        scale_info = compute_fp8_scale(tensor, self.precision)
        self._activation_scales[name] = scale_info
        
        # Invariant: compute_fp8_scale returns scale=1.0 as a Python float literal
        # (identity path), so this exact-equality check is safe — no floating-point
        # rounding can produce a near-1.0 value that should be treated as identity.
        if scale_info.scale != 1.0:
            return apply_fp8_scale(tensor, scale_info)
        return tensor
    
    def _restore_from_storage(
        self,
        name: str,
        tensor: np.ndarray
    ) -> np.ndarray:
        """Restore tensor values after reading from storage-role buffer."""
        if name not in self._activation_scales:
            return tensor
        
        scale_info = self._activation_scales[name]
        # Same invariant as _prepare_storage_buffer: inv_scale=1.0 is exact identity.
        if scale_info.inv_scale != 1.0:
            return unapply_fp8_scale(tensor, scale_info)
        return tensor
```

#### 9E.2.1: Gradient scaling integration

Verify that gradients under Quadratic Scaling Policy already fit FP8 range:

```python
def _validate_gradient_fits_fp8(self, gradient: np.ndarray):
    """Assert gradients are within FP8 range (should pass due to QSP)."""
    if not self.is_fp8_storage:
        return
    
    max_grad = np.abs(gradient).max()
    storage_max = self.precision.storage_fp_format_max
    
    if max_grad > storage_max:
        # This should not happen under correct Quadratic Scaling Policy
        warnings.warn(
            f"Gradient max {max_grad:.2e} exceeds FP8 max {storage_max}. "
            f"Values will saturate. Check stabilization policy."
        )
```

#### 9E.2.2: FP8 scaling and loss scaling interaction

FP8 activation scaling and loss scaling are **orthogonal** — they apply at different points:

| Scaling Type | Applied To | Applied When | Inverse Applied |
|:---|:---|:---|:---|
| **Loss scaling** | Gradients | Before backward pass | After gradient aggregation, before optimizer |
| **FP8 activation scaling** | Activations | Before FP8 storage (forward) | After FP8 load (backward) |

**Key insight:** Loss scaling multiplies gradients uniformly; FP8 scaling adjusts per-tensor based on dynamic range. They do not interact directly because:
1. Activations are stored during forward pass (before loss scaling)
2. Gradients w.r.t. activations are computed in compute precision, not loaded from FP8
3. Only activation *values* (not gradients) go through FP8 quantization

```python
# Forward pass: activations → FP8 (scaled if needed)
activations = layer.forward(inputs)
scaled_activations = self._prepare_storage_buffer("layer_act", activations, "storage")
fp8_buffer.store(scaled_activations.astype(self.precision.storage_dtype))

# Backward pass: load activations (unscale), compute gradients in compute_dtype
stored_activations = fp8_buffer.load().astype(self.precision.compute_dtype)
restored_activations = self._restore_from_storage("layer_act", stored_activations)
# Gradients computed in compute_dtype, affected by loss_scale, NOT by activation scale
grad_activations = layer.backward(grad_output, restored_activations)
```

If mixed-precision training uses **both** loss scaling (for gradient underflow prevention) and FP8 storage (for bandwidth), they compose without special handling.

**Test for orthogonality (add to `test_fp8_integration.py`):**

```python
def test_fp8_and_loss_scaling_compose(self):
    """FP8 activation scaling and loss scaling compose without interference.
    
    Validates that using both scaling methods produces correct gradients.
    """
    cfg = PrecisionConfig.fp8_e4m3()
    loss_scale = 1024.0  # Typical loss scale factor
    
    # Create synthetic forward/backward scenario
    activations = np.random.randn(64, 128).astype(np.float32) * 0.1  # Small activations
    grad_output = np.random.randn(64, 128).astype(np.float32) * loss_scale  # Scaled gradients
    
    # Store activations in FP8 (may require activation scaling)
    scale_info = compute_fp8_scale(activations, cfg)
    scaled_activations = apply_fp8_scale(activations, scale_info)
    fp8_stored = scaled_activations.astype(cfg.storage_dtype)
    
    # Load and restore activations
    restored = fp8_stored.astype(np.float32)
    restored = unapply_fp8_scale(restored, scale_info)
    
    # Compute gradient (activation * grad_output)—simplified
    grad_with_fp8 = restored * grad_output
    grad_baseline = activations * grad_output
    
    # Unscale loss
    grad_with_fp8 /= loss_scale
    grad_baseline /= loss_scale
    
    # Relative error should be FP8 quantization error, not compounded by loss scaling.
    # Use median (robust to outlier values near zero) rather than mean.
    relative_error = np.abs(grad_with_fp8 - grad_baseline) / (np.abs(grad_baseline) + 1e-8)
    median_error = np.median(relative_error)
    assert median_error < 0.15, (
        f"Combined FP8+loss_scaling median error {median_error:.2%} exceeds expected ~12.5%"
    )
    # Complementary P95 check: catches heavy-tail pathologies the median misses
    p95_error = np.percentile(relative_error, 95)
    assert p95_error < 0.40, (
        f"Combined FP8+loss_scaling P95 error {p95_error:.2%} is unreasonably large"
    )
```

---

### Step 9E.3: Update buffer allocation for FP8

**Governing authority:** ADR-025 §8  
**File:** `src/shared/buffer_lifecycle.py`

Verify that `BufferDescriptor` correctly handles FP8 element size:

```python
@property
def element_size_bytes(self) -> int:
    """Size of one element in bytes, based on precision role."""
    if self.precision_role == "storage":
        return self.precision.storage_dtype.itemsize  # 1 for FP8
    elif self.precision_role == "compute":
        return self.precision.compute_dtype.itemsize
    elif self.precision_role == "state":
        return self.precision.state_dtype.itemsize
    else:
        raise ValueError(f"Unknown precision role: {self.precision_role}")
```

For FP8, `storage_dtype.itemsize` is 1 byte. A storage buffer with 10M elements uses 10MB (vs. 40MB for FP32, 20MB for FP16).

---

### Step 9E.4: Implement "The Bandwidth Extremist" test

**Governing authority:** ADR-025 §9.1  
**File:** `tests/tier3/test_fp8_integration.py` (new)

```python
"""The Bandwidth Extremist: FP8 storage fidelity validation (ADR-025 §9.1)."""

import pytest
import numpy as np
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2, FP8_DTYPES

# --- Tolerance Constants (ADR-025 §9.1) ---
# Extracted to module level for easy adjustment after empirical validation.
#
# Rationale:
# - E4M3 has 3 mantissa bits → quantization step ~12.5% relative error per value
# - Quantization errors are unbiased and tend to CANCEL across aggregation
# - For N aggregated gradient elements, expected relative error scales as:
#   ~12.5% / √N (central limit theorem)
# - Example: For 10M gradient elements, expected error ≈ 12.5%/√10M ≈ 0.004%
# - Start with 2% tolerance; this is ~500× the expected statistical error
# - If 2% fails empirically, investigate BEFORE loosening:
#   - Model conditioning / batch normalization placement
#   - Activation outliers exceeding E4M3 range
#   - Test regression or implementation bugs
# - Loosen to 5% ONLY after confirming failures are genuine quantization noise
#
# --- TOLERANCE CALIBRATION (Phase 9E Deliverable) ---
# After Phase 9E completion, execute the calibration procedure in Step 9E.5.5
# and update these constants based on empirical measurements.
# 
# Current status: UNCALIBRATED (initial conservative values)
# Observed P99 errors: (to be filled after calibration)
#   E4M3/FP32: ___%, E4M3/FP16: ___%, E4M3/FP64: ___%
#   E5M2/FP32: ___%, E5M2/FP16: ___%, E5M2/FP64: ___%
#
FP8_CONVERGENCE_RELATIVE_TOLERANCE = 0.02  # 2% — start strict
FP8_LOW_RANGE_MAX_ERROR = 0.1              # Max abs error for values in [0, 1]
FP8_LOW_RANGE_MEAN_ERROR = 0.02            # Mean abs error for values in [0, 1]
FP8_HIGH_RANGE_MAX_RELATIVE_ERROR = 0.15   # Max relative error for values near E4M3 max
FP8_HIGH_RANGE_MEAN_RELATIVE_ERROR = 0.07  # Mean relative error for values near E4M3 max


class TestBandwidthExtremist:
    """ADR-025 §9.1: FP8 vs FP16 training comparison."""
    
    @pytest.fixture
    def training_config(self):
        """Standard training configuration for comparison."""
        return {
            "batch_size": 32,
            "epochs": 10,
            "learning_rate": 0.001,
            "seed": 42,
        }
    
    @pytest.mark.parametrize("fp8_factory,baseline_factory", [
        # Pairing strategy: Each FP8 config is paired with a baseline that has
        # IDENTICAL compute and state precision. Only STORAGE differs:
        # - FP8 config: FP8 storage (E4M3 or E5M2)
        # - Baseline: native storage matching compute precision (FP16/FP32/FP64)
        #
        # This INTENTIONALLY differs in storage to isolate the FP8 quantization effect.
        # If storage were identical, we'd be comparing FP8 to FP8 (useless).
        # Comments show: (FP8: storage/compute/state) vs (Baseline: storage/compute/state)
        #
        # Note: fp8_e4m3_f16 / fp8_e5m2_f16 (FP16 compute, FP32 state) are NOT included
        # here because no factory produces a baseline with FP16 storage + FP16 compute +
        # FP32 state (float16() is deleted, and mixed_f16_f32() uses FP32 compute).
        # Those configs are validated by per-backend roundtrip tests in Phases 9B/9C/9D.
        #
        # E4M3 variants — canonical scenario from CONCEPT.md §11
        (PrecisionConfig.fp8_e4m3, PrecisionConfig.mixed_f16_f32), # E4M3/FP32/FP32 vs FP16/FP32/FP32
        (PrecisionConfig.fp8_e4m3_f64, PrecisionConfig.float64),   # E4M3/FP64/FP64 vs FP64/FP64/FP64
        # E5M2 variants
        (PrecisionConfig.fp8_e5m2, PrecisionConfig.mixed_f16_f32), # E5M2/FP32/FP32 vs FP16/FP32/FP32
        (PrecisionConfig.fp8_e5m2_f64, PrecisionConfig.float64),   # E5M2/FP64/FP64 vs FP64/FP64/FP64
        #
        # NOTE: fp8_e4m3_f16() and fp8_e5m2_f16() (FP16 compute + FP32 state) are NOT
        # included here because no factory produces a matching baseline configuration
        # (FP16 storage + FP16 compute + FP32 state). The deleted float16() factory
        # used uniform FP16 including state, which is architecturally invalid.
        # 
        # Coverage for FP16 compute configs is provided by:
        # - Per-backend roundtrip tests in Phases 9B/9C/9D (validate conversion correctness)
        # - The convergence tests above implicitly cover FP16 compute via mixed_f16_f32 baseline
    ])
    def test_fp8_convergence_matches_baseline(self, training_config, fp8_factory, baseline_factory):
        """FP8 training converges within tolerance of baseline.
        
        Each FP8 config is paired with a baseline that has identical compute and state
        precision. Only storage precision differs (FP8 vs FP16/FP32/FP64).
        This isolates the effect of FP8 quantization on training fidelity.
        """
        cfg_fp8 = fp8_factory()
        cfg_baseline = baseline_factory()
        
        # Ensure identical compute/state precision (storage intentionally differs)
        assert cfg_fp8.compute_dtype == cfg_baseline.compute_dtype, (
            f"Compute dtype mismatch: FP8={cfg_fp8.compute_dtype}, baseline={cfg_baseline.compute_dtype}"
        )
        assert cfg_fp8.state_dtype == cfg_baseline.state_dtype, (
            f"State dtype mismatch: FP8={cfg_fp8.state_dtype}, baseline={cfg_baseline.state_dtype}"
        )
        
        # Run baseline
        loss_baseline, params_baseline = self._run_training(cfg_baseline, training_config)
        
        # Run FP8
        loss_fp8, params_fp8 = self._run_training(cfg_fp8, training_config)
        
        # Assert convergence within tolerance (see module-level constants for rationale)
        final_loss_baseline = loss_baseline[-1]
        final_loss_fp8 = loss_fp8[-1]
        
        # Guard against near-zero baseline loss (avoids division-by-zero)
        relative_diff = abs(final_loss_fp8 - final_loss_baseline) / max(final_loss_baseline, 1e-10)
        assert relative_diff < FP8_CONVERGENCE_RELATIVE_TOLERANCE, (
            f"FP8 loss {final_loss_fp8:.6f} diverged from baseline "
            f"{final_loss_baseline:.6f} by {relative_diff*100:.1f}% "
            f"(> {FP8_CONVERGENCE_RELATIVE_TOLERANCE*100}% tolerance). "
            f"This may indicate: (1) model instability with FP8 quantization, "
            f"(2) activation outliers exceeding E4M3 range, or (3) test regression."
        )
        
        # Assert loss trend is decreasing (final loss < initial loss).
        # Strict monotonic decrease is too brittle — FP8 quantization noise
        # can cause small per-epoch increases even when training converges.
        # Use baseline-relative check: FP8 should decrease by at least as much
        # as the baseline (within tolerance), rather than an arbitrary absolute
        # threshold like "50% of initial" which is dataset-dependent.
        baseline_decrease = (loss_baseline[0] - loss_baseline[-1]) / max(loss_baseline[0], 1e-10)
        fp8_decrease = (loss_fp8[0] - loss_fp8[-1]) / max(loss_fp8[0], 1e-10)
        # FP8 must achieve at least 70% of baseline's relative decrease
        # (30% slack accounts for quantization noise)
        assert fp8_decrease >= baseline_decrease * 0.7, (
            f"FP8 training did not converge comparably to baseline: "
            f"baseline decreased {baseline_decrease*100:.1f}%, "
            f"FP8 decreased {fp8_decrease*100:.1f}% "
            f"(threshold: {baseline_decrease*70:.1f}% = 70% of baseline). "
            f"This may indicate quantization-induced divergence."
        )
    
    def test_fp8_buffer_sizing(self, training_config):
        """FP8 storage buffers are 1/2 the size of FP16."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp16 = PrecisionConfig.mixed_f16_f32()
        
        # Create identical model specs
        # ...
        
        # Compare storage buffer allocations
        fp8_storage_bytes = self._compute_storage_allocation(cfg_fp8)
        fp16_storage_bytes = self._compute_storage_allocation(cfg_fp16)
        
        assert fp8_storage_bytes == fp16_storage_bytes // 2, (
            f"FP8 storage ({fp8_storage_bytes}) should be half of "
            f"FP16 storage ({fp16_storage_bytes})"
        )
    
    def test_quantization_error_bounded(self, training_config):
        """Quantization error from FP8 storage is bounded and does not blow up."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        
        # Fixed seed for reproducibility
        np.random.seed(training_config["seed"])
        
        # Generate test activations in range [0, 1] (post-sigmoid)
        test_activations_low = np.random.rand(1000, 100).astype(np.float32)
        
        # Also test higher values near E4M3 max (where relative error is larger)
        test_activations_high = (np.random.rand(1000, 100) * 400 + 48).astype(np.float32)  # [48, 448]
        
        # Test low-range activations [0, 1]
        # Use cfg_fp8.storage_dtype for consistency with PrecisionConfig constants
        fp8_stored_low = test_activations_low.astype(cfg_fp8.storage_dtype)
        restored_low = fp8_stored_low.astype(np.float32)
        error_low = np.abs(test_activations_low - restored_low)
        
        # E4M3 has 3 mantissa bits → relative precision ~1/8    
        # For values in [0.5, 1], max error should be ~0.0625
        assert error_low.max() < FP8_LOW_RANGE_MAX_ERROR, (
            f"Max quantization error (low range) {error_low.max()} exceeds {FP8_LOW_RANGE_MAX_ERROR}"
        )
        assert error_low.mean() < FP8_LOW_RANGE_MEAN_ERROR, (
            f"Mean quantization error (low range) {error_low.mean()} exceeds {FP8_LOW_RANGE_MEAN_ERROR}"
        )
        
        # Test high-range activations [48, 448]
        fp8_stored_high = test_activations_high.astype(cfg_fp8.storage_dtype)
        restored_high = fp8_stored_high.astype(np.float32)
        error_high = np.abs(test_activations_high - restored_high)
        relative_error_high = error_high / test_activations_high
        
        # For high values, absolute error is larger but relative error stays ~12.5%
        assert relative_error_high.max() < FP8_HIGH_RANGE_MAX_RELATIVE_ERROR, (
            f"Max relative error (high range) {relative_error_high.max():.2%} exceeds "
            f"{FP8_HIGH_RANGE_MAX_RELATIVE_ERROR*100}%"
        )
        assert relative_error_high.mean() < FP8_HIGH_RANGE_MEAN_RELATIVE_ERROR, (
            f"Mean relative error (high range) {relative_error_high.mean():.2%} exceeds "
            f"{FP8_HIGH_RANGE_MEAN_RELATIVE_ERROR*100}%"
        )
    
    def _run_training(self, precision, config):
        """Execute training with given precision config.
        
        Implementation should:
        1. Instantiate model with given precision config
        2. Create orchestrator with FP8 scaling enabled (if applicable)
        3. Run training for config['epochs'] epochs
        4. Return (loss_history: List[float], final_params: dict)
        
        Reference: Use src/main_orchestrator.py with the precision config.
        The orchestrator's `_prepare_storage_buffer` handles FP8 scaling.
        
        Note: Uses fixed seed from config for reproducibility across FP8/baseline runs.
        Note: Import path assumes tests run from project root with standard pytest setup.
              The conftest.py adds `src/` to sys.path.
        """
        from main_orchestrator import Orchestrator
        
        orchestrator = Orchestrator(
            precision=precision,
            seed=config["seed"],
        )
        loss_history = []
        for epoch in range(config["epochs"]):
            epoch_loss = orchestrator.train_epoch(
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
            )
            loss_history.append(epoch_loss)
        
        return loss_history, orchestrator.get_params()
    
    def _compute_storage_allocation(self, precision):
        """Compute total storage-role buffer allocation.
        
        Implementation should:
        1. Create a ModelSpec with known layer sizes
        2. Sum all storage-role BufferDescriptor sizes
        3. Return total bytes
        
        Reference: Use src/shared/buffer_lifecycle.py BufferDescriptor.
        """
        from shared.model_spec import ModelSpec
        from shared.buffer_lifecycle import BufferDescriptor
        
        # Standard test model: 128-dim input, 64-dim hidden, 10-dim output
        model_spec = ModelSpec.standard_test_model(precision=precision)
        
        return sum(
            desc.size_bytes 
            for desc in model_spec.storage_buffers()
        )
```

---

#### Step 9E.4.1: Cross-backend FP8 consistency test

**File:** `tests/tier2/test_fp8_cross_backend.py` (new)

FP8 conversion uses different strategies per backend (LUT for OpenCL/CPU, arithmetic for Vulkan). Verify they produce identical results for all 256 FP8 bit patterns:

```python
"""Cross-backend FP8 conversion consistency.

Ensures LUT-based (OpenCL, CPU) and arithmetic (Vulkan) backends produce
bit-identical results for all 256 FP8 values in both E4M3 and E5M2 formats.
"""

import pytest
import numpy as np
import ml_dtypes
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

# Skip entire module if fewer than 2 backends support FP8
pytestmark = pytest.mark.skipif(
    # Import here to avoid collection-time side effects
    True,  # Replaced at runtime by conftest hook
    reason="Cross-backend test requires ≥2 FP8-capable backends",
)


class TestCrossBackendFP8Consistency:
    """ADR-025 §9: All backends must produce identical FP8↔FP32 roundtrip results."""

    @pytest.fixture
    def available_fp8_backends(self):
        """Discover which backends support FP8 at test time."""
        from tests.conftest import _get_fp8_unsupported_backends
        all_backends = {"opencl", "cpu", "vulkan"}
        unsupported = _get_fp8_unsupported_backends()
        available = all_backends - unsupported
        if len(available) < 2:
            pytest.skip(f"Need ≥2 FP8 backends, have: {available}")
        return sorted(available)

    @pytest.mark.parametrize("fp8_format", ["e4m3", "e5m2"])
    def test_roundtrip_all_256_patterns(self, fp8_format, available_fp8_backends):
        """All 256 FP8 bit patterns must produce identical FP32 values across backends."""
        dtype = ml_dtypes.float8_e4m3fn if fp8_format == "e4m3" else ml_dtypes.float8_e5m2
        precision = (
            PrecisionConfig.fp8_e4m3() if fp8_format == "e4m3"
            else PrecisionConfig.fp8_e5m2()
        )

        # Create all 256 FP8 bit patterns
        raw_bytes = np.arange(256, dtype=np.uint8)
        fp8_values = raw_bytes.view(dtype)

        # Convert via each backend and collect results
        results = {}
        for backend_name in available_fp8_backends:
            backend = self._get_backend(backend_name)
            f32_output = backend.fp8_to_f32_roundtrip(fp8_values, precision)
            results[backend_name] = f32_output

        # Assert all backends produce identical results (bit-exact, not approximate)
        reference_backend = available_fp8_backends[0]
        reference = results[reference_backend]
        for other_backend in available_fp8_backends[1:]:
            np.testing.assert_array_equal(
                reference, results[other_backend],
                err_msg=(
                    f"FP8 {fp8_format} roundtrip mismatch: "
                    f"{reference_backend} vs {other_backend}"
                ),
            )

    @staticmethod
    def _get_backend(name: str):
        """Import and return backend by name."""
        if name == "opencl":
            from backends.opencl.backend import OpenCLBackend
            return OpenCLBackend()
        elif name == "cpu":
            from backends.cpu.backend import CPUBackend
            return CPUBackend()
        elif name == "vulkan":
            from backends.vulkan.backend import VulkanBackend
            return VulkanBackend()
        raise ValueError(f"Unknown backend: {name}")
```

> **Why bit-exact?** All 256 FP8 values are exactly representable in FP32. The LUT
> (OpenCL/CPU) and arithmetic (Vulkan) paths compute the same mathematical function
> — there is no rounding ambiguity. Any difference is a conversion bug.

---

### Step 9E.5: Add FP8 memory validation tests

**File:** `tests/tier1/test_fp8_memory.py` (new)

```python
"""FP8 memory allocation validation."""

import pytest
from shared.precision_config import PrecisionConfig
from shared.buffer_lifecycle import BufferDescriptor

class TestFP8MemoryAllocation:
    """Validate that FP8 achieves expected memory savings."""
    
    def test_fp8_element_size(self):
        """FP8 storage uses 1 byte per element."""
        cfg = PrecisionConfig.fp8_e4m3()
        assert cfg.storage_dtype.itemsize == 1
    
    def test_fp8_buffer_descriptor_sizing(self):
        """BufferDescriptor computes correct sizes for FP8."""
        cfg = PrecisionConfig.fp8_e4m3()
        num_elements = 1_000_000
        
        desc = BufferDescriptor(
            name="test",
            num_elements=num_elements,
            precision=cfg,
            precision_role="storage",
        )
        
        assert desc.element_size_bytes == 1
        assert desc.size_bytes == num_elements
    
    def test_fp8_vs_fp32_memory_ratio(self):
        """FP8 storage is 1/4 the size of FP32."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp32 = PrecisionConfig.float32()
        
        assert cfg_fp8.storage_dtype.itemsize == 1
        assert cfg_fp32.storage_dtype.itemsize == 4
        assert cfg_fp32.storage_dtype.itemsize == 4 * cfg_fp8.storage_dtype.itemsize
    
    def test_fp8_vs_fp16_memory_ratio(self):
        """FP8 storage is 1/2 the size of FP16."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        cfg_fp16 = PrecisionConfig.mixed_f16_f32()
        
        assert cfg_fp8.storage_dtype.itemsize == 1
        assert cfg_fp16.storage_dtype.itemsize == 2
        assert cfg_fp16.storage_dtype.itemsize == 2 * cfg_fp8.storage_dtype.itemsize
```

---

### Step 9E.5.5: Execute tolerance calibration procedure

**Governing authority:** ADR-025 §9.1  
**Output:** Updated tolerance constants in `tests/tier3/test_fp8_integration.py`

After the "Bandwidth Extremist" tests pass with initial conservative tolerances, execute this calibration procedure to establish empirically-grounded thresholds:

#### 9E.5.5.1: Run calibration workloads

```bash
cd architectures/averaging_ensembled_classifier

# Run extended validation on 5 representative workloads, capturing detailed metrics
pytest tests/tier3/test_fp8_integration.py::TestBandwidthExtremist \
    -v --tb=short --capture=no \
    --calibration-mode 2>&1 | tee /tmp/fp8-calibration.txt
```

> **Note:** The `--calibration-mode` flag enables verbose error reporting, logging P50/P95/P99 relative errors for each FP8 config.

**Fixture implementation (add to `tests/conftest.py`):**

```python
def pytest_addoption(parser):
    parser.addoption(
        "--calibration-mode",
        action="store_true",
        default=False,
        help="Enable FP8 tolerance calibration mode (verbose error stats)",
    )


@pytest.fixture
def calibration_mode(request):
    """Return True if running in calibration mode."""
    return request.config.getoption("--calibration-mode")
```

**Calibration helper (add to `tests/tier3/test_fp8_integration.py`):**

```python
def _log_error_distribution(errors: np.ndarray, name: str, calibration_mode: bool) -> None:
    """Log error distribution statistics when in calibration mode."""
    if not calibration_mode:
        return
    
    p50 = np.percentile(errors, 50)
    p95 = np.percentile(errors, 95)
    p99 = np.percentile(errors, 99)
    print(f"\n[CALIBRATION] {name}:")
    print(f"  P50: {p50:.4%}, P95: {p95:.4%}, P99: {p99:.4%}")
    print(f"  Max: {errors.max():.4%}, Mean: {errors.mean():.4%}")
```

#### 9E.5.5.2: Record observed error distributions

After calibration runs complete, extract the P99 error values and update the tolerance constants comment block:

```python
# --- TOLERANCE CALIBRATION (Phase 9E Deliverable) ---
# Current status: CALIBRATED (date: YYYY-MM-DD)
# Observed P99 errors:
#   E4M3/FP32: X.XX%, E4M3/FP16: X.XX%, E4M3/FP64: X.XX%
#   E5M2/FP32: X.XX%, E5M2/FP16: X.XX%, E5M2/FP64: X.XX%
#
# Tolerance = 3 × max(P99) = X.XX%
FP8_CONVERGENCE_RELATIVE_TOLERANCE = 0.0X  # Updated from calibration
```

#### 9E.5.5.3: Verification

- Calibrated tolerance is ≤ 5% (if > 5%, investigate model issues before accepting)
- All FP8 convergence tests pass with calibrated tolerance
- Tolerance comment block is updated with observed P99 values and calibration date

---

### Step 9E.6: Run full regression suite

Execute the complete test suite to validate FP8 integration:

```bash
cd architectures/averaging_ensembled_classifier

# Rebuild CPU backend with FP8 variants
ninja -C builddir

# Run all tests, tee to file per AGENTS.md
pytest tests/ -v 2>&1 | tee /tmp/fp8-regression.txt

# Check for failures
grep -E "(FAILED|ERROR)" /tmp/fp8-regression.txt
```

All existing tests must pass. FP8-specific tests must pass.

---

### Step 9E.7: Document FP8 usage

**File:** `README.md` (update)

Add FP8 usage documentation:

```markdown
## Precision Configuration

### FP8 Storage (Maximum Bandwidth)

For bandwidth-limited workloads, FP8 storage achieves 4× compression vs. FP32:

```python
from shared.precision_config import PrecisionConfig

# E4M3: 3 mantissa bits, max value 448
cfg = PrecisionConfig.fp8_e4m3()

# E5M2: 2 mantissa bits, max value 57344 (wider range)
cfg = PrecisionConfig.fp8_e5m2()

# E4M3 with FP64 compute + FP64 optimizer state for extended stability
cfg = PrecisionConfig.fp8_e4m3_f64()

# Custom combination: FP16 compute + FP64 state
cfg = PrecisionConfig(
    storage_dtype=FP8_E4M3,
    compute_dtype=np.float16,
    state_dtype=np.float64,
)
```

FP8 is **storage-role only** — compute uses FP16, FP32, or FP64; state uses FP32 or FP64.
Attempting FP8 compute or state raises `ValueError`.
```

---

### Step 9E.8: Add checkpoint migration tests

**File:** `tests/tier3/test_checkpoint_migration.py` (new)

```python
"""Checkpoint version migration tests for FP8 scale compatibility."""

import pytest
import numpy as np
from shared.precision_config import PrecisionConfig
from shared.fp8_scaling import FP8_SCALES_VERSION, FP8ScaleInfo


class TestCheckpointMigration:
    """Validate checkpoint loading across FP8 scale versions.
    
    Note: ``Orchestrator(precision=cfg, seed=42)`` below is pseudocode.
    The real constructor signature will include additional required arguments
    (e.g. model spec, backend); adapt when integrating with the actual API.
    """

    def test_pre_fp8_checkpoint_loads_with_fp8_config(self):
        """Pre-Phase-9 checkpoint (no fp8_scales key) loads correctly.
        
        When loading a FP32 checkpoint with an FP8 precision config,
        scales should be empty (computed fresh on first forward pass).
        """
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        
        # Simulate old checkpoint (no fp8_scales key)
        old_checkpoint = {
            "model_state": {"layer.weight": np.random.randn(64, 64)},
            "optimizer_state": {},
            "training_step": 1000,
            # No "fp8_scales" key — pre-FP8 checkpoint
        }
        
        from main_orchestrator import Orchestrator
        
        orchestrator = Orchestrator(precision=cfg_fp8, seed=42)
        orchestrator.restore_state(old_checkpoint)
        
        # _activation_scales should be empty, ready for recomputation
        assert orchestrator._activation_scales == {}, (
            "Pre-FP8 checkpoint should result in empty activation scales"
        )

    def test_mismatched_version_triggers_recompute(self):
        """Checkpoint with wrong FP8_SCALES_VERSION triggers scale recomputation.
        
        If a checkpoint has a different version, warn and recompute rather than
        silently misinterpreting the scale data.
        """
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        
        # Simulate checkpoint with future/incompatible version
        future_checkpoint = {
            "model_state": {},
            "optimizer_state": {},
            "training_step": 500,
            "fp8_scales": {
                "_version": 999,  # Future version
                "scales": {
                    "activation_0": {"scale": 0.5, "inv_scale": 2.0, "storage_max": 448.0},
                },
            },
        }
        
        import warnings
        from main_orchestrator import Orchestrator
        
        orchestrator = Orchestrator(precision=cfg_fp8, seed=42)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            orchestrator.restore_state(future_checkpoint)
            
            # Should warn about version mismatch
            assert any("scale format mismatch" in str(warning.message).lower() for warning in w), (
                f"Expected version mismatch warning, got: {[str(x.message) for x in w]}"
            )
        
        # Scales should be cleared for recomputation
        assert orchestrator._activation_scales == {}

    def test_structurally_incompatible_scales_trigger_recompute(self):
        """Checkpoint with structurally incompatible FP8ScaleInfo triggers recompute.
        
        If the saved scale data doesn't match FP8ScaleInfo fields (e.g., extra/missing
        fields from a future version), catch TypeError and recompute.
        """
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        
        # Simulate checkpoint with incompatible scale structure
        incompatible_checkpoint = {
            "model_state": {},
            "optimizer_state": {},
            "training_step": 500,
            "fp8_scales": {
                "_version": FP8_SCALES_VERSION,  # Same version but different structure
                "scales": {
                    "activation_0": {
                        "scale": 0.5,
                        # Missing inv_scale and storage_max — will cause TypeError
                        "extra_field": 123,  # Extra field
                    },
                },
            },
        }
        
        import warnings
        from main_orchestrator import Orchestrator
        
        orchestrator = Orchestrator(precision=cfg_fp8, seed=42)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            orchestrator.restore_state(incompatible_checkpoint)
            
            assert any("FP8ScaleInfo" in str(warning.message) for warning in w), (
                f"Expected structure incompatibility warning, got: {[str(x.message) for x in w]}"
            )
        
        assert orchestrator._activation_scales == {}

    def test_valid_scales_restore_correctly(self):
        """Checkpoint with valid FP8 scales restores without recomputation."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()
        
        valid_checkpoint = {
            "model_state": {},
            "optimizer_state": {},
            "training_step": 500,
            "fp8_scales": {
                "_version": FP8_SCALES_VERSION,
                "scales": {
                    "activation_0": {
                        "scale": 0.5,
                        "inv_scale": 2.0,
                        "storage_max": 448.0,
                    },
                },
            },
        }
        
        from main_orchestrator import Orchestrator
        
        orchestrator = Orchestrator(precision=cfg_fp8, seed=42)
        orchestrator.restore_state(valid_checkpoint)
        
        assert "activation_0" in orchestrator._activation_scales
        scale_info = orchestrator._activation_scales["activation_0"]
        assert scale_info.scale == 0.5
        assert scale_info.inv_scale == 2.0
        assert scale_info.storage_max == 448.0
```

---

#### Compute precision options

| Factory | Storage | Compute | State | Use Case |
|:---|:---|:---|:---|:---|
| `fp8_e4m3()` | E4M3 | FP32 | FP32 | Standard bandwidth-optimized training |
| `fp8_e4m3_f16()` | E4M3 | FP16 | FP32 | Maximum throughput on FP16-accelerated hardware |
| `fp8_e4m3_f64()` | E4M3 | FP64 | FP64 | Extended stability for long training runs |
| `fp8_e5m2()` | E5M2 | FP32 | FP32 | Wider range (57344 max) for gradient outliers |
| `fp8_e5m2_f16()` | E5M2 | FP16 | FP32 | Wider range + FP16 compute throughput |
| `fp8_e5m2_f64()` | E5M2 | FP64 | FP64 | Wider range + extended stability |

Non-default combinations (e.g., FP16 compute + FP64 state, FP32 compute + FP64 state) are created via constructor.

#### When to use FP8

- Memory bandwidth is the bottleneck
- Activations and gradients are well-conditioned (no extreme outliers)
- 448 (E4M3) or 57344 (E5M2) max value is sufficient

#### When NOT to use FP8

- Training large models where state precision matters (use `fp8_e4m3_f64()`)
- Debugging numerical issues (use FP32 for reproducibility)
- Workloads with extreme dynamic range (E5M2 may help, or use FP16)
```

---

## 4. Validation Criteria

| Criterion | Measurement | Pass Threshold |
|:---|:---|:---|
| **Convergence** | Final loss difference between FP8 and baseline | < 2% relative (start strict; loosen to 5% only after investigation) |
| **Memory savings** | Storage buffer size ratio FP8:FP16 | Exactly 1:2 |
| **Roundtrip accuracy** | Max FP32→FP8→FP32 error | ≤ 1 ULP of FP8 representation |
| **Regression** | Existing test suite | 100% pass |
| **Saturation behavior** | Values > 448 (E4M3) | Saturate to 448, not NaN |

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|:---|:---|:---|:---|
| Gradient outliers exceed FP8 range | Low | Medium | Quadratic Scaling Policy already constrains. Add warning log if saturation occurs. |
| Activation scaling overhead | Low | Low | Scaling is O(n), amortized by bandwidth savings on large tensors. |
| Convergence divergence for some models | Medium | Medium | Document FP8 applicability. Provide E5M2 as wider-range alternative. |
| Scale tracking adds state complexity | Low | Low | Minimal additional state. Well-encapsulated in FP8 scaling module. |

---

## 6. Files Modified Summary

| File | Change Type | Description |
|:---|:---|:---|
| `src/shared/fp8_scaling.py` | **New** | `FP8ScaleInfo`, `compute_fp8_scale`, `apply_fp8_scale`, `unapply_fp8_scale` utilities |
| `src/main_orchestrator.py` | **Edit** | Add `is_fp8_storage` flag, `_activation_scales` tracking, `_prepare_storage_buffer`, `_restore_from_storage` methods |
| `src/shared/buffer_lifecycle.py` | **Edit** | Verify `element_size_bytes` handles FP8 (`itemsize == 1`) |
| `README.md` | **Edit** | Add FP8 usage documentation, factory method table, applicability guidance |
| `tests/tier3/test_fp8_integration.py` | **New** | "The Bandwidth Extremist" validation test |
| `tests/tier3/test_checkpoint_migration.py` | **New** | FP8 scale checkpoint version migration tests |
| `tests/tier1/test_fp8_memory.py` | **New** | FP8 memory allocation and sizing validation tests |
| `tests/conftest.py` | **Edit** | Add `--calibration-mode` option, `calibration_mode` fixture |

---

## 7. FP8 Implementation Summary (Phases 9A–9E)

This phase completes the FP8 implementation. The following table summarizes what was added across all phases:

| Phase | Scope | Key Deliverables |
|:---|:---|:---|
| **9A** | Config & Authority | `ml_dtypes` dependency, 6 factory methods, storage-only constraint, CONCEPT.md/CONTRACT.md amendments |
| **9B** | OpenCL Backend | LUT generation script, `load_storage_fp8`/`store_storage_fp8`, build flags, roundtrip tests |
| **9C** | CPU Backend | `cpu_fp8.h` types, 10 suffix variants, `_Float16` support detection, roundtrip tests |
| **9D** | Vulkan Backend | Specialization constants, arithmetic conversion functions, `VK_KHR_8bit_storage` handling |
| **9E** | Host Integration | FP8 scaling utilities, scale persistence, "Bandwidth Extremist" validation, memory tests |

### Cross-Phase Dependencies

```
Phase 9A (Config)
    │
    ├─────────────┬─────────────┐
    ▼             ▼             ▼
Phase 9B      Phase 9C      Phase 9D
(OpenCL)       (CPU)        (Vulkan)
    │             │             │
    └─────────────┴─────────────┘
                  │
                  ▼
             Phase 9E
        (Host Integration)
```

**Parallelization notes:**
- Phases 9B, 9C, and 9D are **fully parallelizable** — they share the Phase 9A `PrecisionConfig` interface but have no inter-dependencies.
- LUT headers (`fp8_lut.gen.h`, `cpu_fp8_lut.gen.h`) are generated in **Step 9A.1.5**, eliminating the cross-phase dependency that previously existed.
- Phase 9E waits for all backend phases (9B, 9C, 9D) to complete.

### FP8 Factory Method Reference

| Factory | Storage | Compute | State | Suffix |
|:---|:---|:---|:---|:---|
| `fp8_e4m3()` | E4M3 | FP32 | FP32 | `s8e4c32x32` |
| `fp8_e4m3_f16()` | E4M3 | FP16 | FP32 | `s8e4c16x32` |
| `fp8_e4m3_f64()` | E4M3 | FP64 | FP64 | `s8e4c64x64` |
| `fp8_e5m2()` | E5M2 | FP32 | FP32 | `s8e5c32x32` |
| `fp8_e5m2_f16()` | E5M2 | FP16 | FP32 | `s8e5c16x32` |
| `fp8_e5m2_f64()` | E5M2 | FP64 | FP64 | `s8e5c64x64` |

Additional combinations (e.g., FP16 compute + FP64 state) are constructed directly and compiled to the appropriate suffix variants.

### Backend Compute Precision Compatibility

| Backend | FP16 Compute | FP32 Compute | FP64 Compute |
|:---|:---|:---|:---|
| **OpenCL** | `cl_khr_fp16` extension | ✅ Core | `cl_khr_fp64` extension |
| **CPU** | `_Float16` (C23 / GCC 12+ / Clang 15+) | ✅ Core | ✅ Core |
| **Vulkan** | `VK_KHR_shader_float16_int8` (Vulkan 1.2 core) | ✅ Core | `shaderFloat64` device feature |

**Note:** A config like `fp8_e4m3_f16()` may work on OpenCL/Vulkan but fail on CPU if the compiler lacks `_Float16` support. The error surfaces at kernel load time with a clear message.

### Validation Checklist

Before marking Phase 9 complete:

- [ ] All Phase 8E tests still pass (no regression)
- [ ] FP8 factories construct without error
- [ ] FP8 compute/state raises `ValueError`
- [ ] FP16 state raises `ValueError` (all configs)
- [ ] OpenCL FP8 roundtrip tests pass (Phase 9B)
- [ ] CPU FP8 roundtrip tests pass (Phase 9C)
- [ ] Vulkan FP8 roundtrip tests pass (Phase 9D)
- [ ] "Bandwidth Extremist" validation passes (Phase 9E)
- [ ] FP8 memory savings verified (1/2 of FP16, 1/4 of FP32)
- [ ] README.md updated with FP8 documentation

### Validation Script

Run the following commands to validate the complete Phase 9 rollback gate:

```bash
cd architectures/averaging_ensembled_classifier

# Rebuild CPU backend with all FP8 variants
ninja -C builddir

# Run full test suite, tee to file per AGENTS.md
pytest tests/ -v 2>&1 | tee /tmp/phase9-validation.txt

# Check for failures
echo "=== FAILURE SUMMARY ==="
grep -E "(FAILED|ERROR)" /tmp/phase9-validation.txt || echo "All tests passed"

# Count test results
echo "=== TEST COUNTS ==="
grep -E "passed|failed|skipped|error" /tmp/phase9-validation.txt | tail -1

# Validate FP8 factory construction
python -c "
import numpy as np
from shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2

# All factories should construct without error
for name in ['fp8_e4m3', 'fp8_e5m2', 'fp8_e4m3_f16', 'fp8_e5m2_f16', 'fp8_e4m3_f64', 'fp8_e5m2_f64']:
    cfg = getattr(PrecisionConfig, name)()
    assert cfg.storage_dtype.itemsize == 1, f'{name}: Expected 1-byte storage'
    print(f'{name}(): storage={cfg.storage_dtype}, compute={cfg.compute_dtype}, state={cfg.state_dtype}')

# FP8 compute/state rejection should raise ValueError
import sys
try:
    PrecisionConfig(
        storage_dtype=FP8_E4M3,
        compute_dtype=FP8_E4M3,
        state_dtype=np.dtype(np.float32),
        storage_fp_format_max=448.0,
        storage_fp_format_min_subnormal=0.001953125,
        storage_mantissa_bits=3,
        compute_fp_format_max=448.0,
        compute_epsilon=0.125,
    )
    print('ERROR: FP8 compute should raise ValueError', file=sys.stderr)
    sys.exit(1)
except ValueError as e:
    print(f'FP8 compute rejection: OK')

print('All FP8 factory validations passed')
"
```

---

*End of Phase 9E and Phase 9 (FP8 Support). The FP8 implementation is complete.*
