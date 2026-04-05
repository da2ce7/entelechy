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
    """Scale information for FP8 quantization."""
    scale: float           # Multiply by this before quantization
    inv_scale: float       # Multiply by this after dequantization
    storage_max: float     # FP8 format max (448 or 57344)
    clipped_count: int     # Number of values that required saturation


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
    
    Returns:
        FP8ScaleInfo with scale factors and diagnostics.
    """
    assert precision.storage_dtype in (FP8_E4M3, FP8_E5M2)
    
    storage_max = precision.storage_fp_format_max
    target_range = storage_max * headroom
    
    tensor_max = np.abs(tensor).max()
    
    if tensor_max < 1e-10:
        # Near-zero tensor — no scaling needed, inverse is identity
        # Note: These tensors produce near-zero gradients; inv_scale=1.0 is correct
        # because there's no meaningful information to preserve.
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
            clipped_count=0
        )
    
    if tensor_max <= target_range:
        # Already fits — no scaling needed
        return FP8ScaleInfo(
            scale=1.0,
            inv_scale=1.0,
            storage_max=storage_max,
            clipped_count=0
        )
    
    # Scale down to fit
    scale = target_range / tensor_max
    inv_scale = 1.0 / scale
    
    return FP8ScaleInfo(
        scale=scale,
        inv_scale=inv_scale,
        storage_max=storage_max,
        clipped_count=0
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
            state["fp8_scales"] = {
                name: info._asdict() for name, info in self._activation_scales.items()
            }
            state["fp8_scales_version"] = FP8_SCALES_VERSION
        return state
    
    def restore_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._training_step = state.get("training_step", 0)
        
        # Restore FP8 scales with version checking
        if "fp8_scales" in state:
            if state.get("fp8_scales_version", 0) != FP8_SCALES_VERSION:
                warnings.warn(
                    f"FP8 scale format mismatch: checkpoint v{state.get('fp8_scales_version', 0)}, "
                    f"current v{FP8_SCALES_VERSION}. Scales will be recomputed."
                )
                self._activation_scales = {}  # Recompute on next forward pass
            else:
                self._activation_scales = {
                    name: FP8ScaleInfo(**info) for name, info in state["fp8_scales"].items()
                }
```
```

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
        
        # Per-buffer scale tracking (for FP8)
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
    
    # Relative error should be FP8 quantization error, not compounded by loss scaling
    relative_error = np.abs(grad_with_fp8 - grad_baseline) / (np.abs(grad_baseline) + 1e-8)
    assert relative_error.mean() < 0.15, (
        f"Combined FP8+loss_scaling error {relative_error.mean():.2%} exceeds expected ~12.5%"
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
from shared.precision_config import PrecisionConfig

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
        # IDENTICAL compute and state precision. The ONLY difference is storage:
        # - FP8 config: FP8 storage (E4M3 or E5M2)
        # - Baseline: native storage (FP32 or FP64)
        #
        # This isolates the FP8 quantization effect. Comments show:
        # (FP8: storage/compute/state) vs (Baseline: storage/compute/state)
        #
        # E4M3 variants
        (PrecisionConfig.fp8_e4m3, PrecisionConfig.float32),           # E4M3/FP32/FP32 vs FP32/FP32/FP32
        (PrecisionConfig.fp8_e4m3_f16, PrecisionConfig.mixed_f16_f32), # E4M3/FP16/FP32 vs FP16/FP16/FP32
        (PrecisionConfig.fp8_e4m3_f64, PrecisionConfig.float64),       # E4M3/FP64/FP64 vs FP64/FP64/FP64
        # E5M2 variants
        (PrecisionConfig.fp8_e5m2, PrecisionConfig.float32),           # E5M2/FP32/FP32 vs FP32/FP32/FP32
        (PrecisionConfig.fp8_e5m2_f16, PrecisionConfig.mixed_f16_f32), # E5M2/FP16/FP32 vs FP16/FP16/FP32
        (PrecisionConfig.fp8_e5m2_f64, PrecisionConfig.float64),       # E5M2/FP64/FP64 vs FP64/FP64/FP64
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
        
        relative_diff = abs(final_loss_fp8 - final_loss_baseline) / final_loss_baseline
        assert relative_diff < FP8_CONVERGENCE_RELATIVE_TOLERANCE, (
            f"FP8 loss {final_loss_fp8:.6f} diverged from baseline "
            f"{final_loss_baseline:.6f} by {relative_diff*100:.1f}% "
            f"(> {FP8_CONVERGENCE_RELATIVE_TOLERANCE*100}% tolerance). "
            f"This may indicate: (1) model instability with FP8 quantization, "
            f"(2) activation outliers exceeding E4M3 range, or (3) test regression."
        )
        
        # Assert loss curve is monotonically decreasing (training is working)
        assert all(loss_fp8[i] >= loss_fp8[i+1] for i in range(len(loss_fp8)-1)), \
            "FP8 training loss is not monotonically decreasing"
    
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
        
        # Generate test activations in range [0, 1] (post-sigmoid)
        test_activations_low = np.random.rand(1000, 100).astype(np.float32)
        
        # Also test higher values near E4M3 max (where relative error is larger)
        test_activations_high = (np.random.rand(1000, 100) * 400 + 48).astype(np.float32)  # [48, 448]
        
        import ml_dtypes
        
        # Test low-range activations [0, 1]
        fp8_stored_low = test_activations_low.astype(ml_dtypes.float8_e4m3fn)
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
        fp8_stored_high = test_activations_high.astype(ml_dtypes.float8_e4m3fn)
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
        
        **SKELETON** — Placeholder for Step 9E.4 implementation.
        
        Estimated effort: ~40 LOC, ~30 minutes
        Complexity: Low — orchestrator integration only
        
        Implementation should:
        1. Instantiate model with given precision config
        2. Create orchestrator with FP8 scaling enabled (if applicable)
        3. Run training for config['epochs'] epochs
        4. Return (loss_history: List[float], final_params: dict)
        
        Reference: Use src/main_orchestrator.py with the precision config.
        The orchestrator's `_prepare_storage_buffer` handles FP8 scaling.
        """
        raise NotImplementedError("SKELETON: Implement in Step 9E.4")
    
    def _compute_storage_allocation(self, precision):
        """Compute total storage-role buffer allocation.
        
        **SKELETON** — Placeholder for Step 9E.4 implementation.
        
        Estimated effort: ~15 LOC, ~15 minutes
        Complexity: Trivial — buffer size summation
        
        Implementation should:
        1. Create a ModelSpec with known layer sizes
        2. Sum all storage-role BufferDescriptor sizes via:
           `sum(desc.size_bytes for desc in model_spec.storage_buffers())`
        3. Return total bytes
        
        Reference: Use src/shared/buffer_lifecycle.py BufferDescriptor.
        """
        raise NotImplementedError("SKELETON: Implement in Step 9E.4")
```

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
| `tests/tier1/test_fp8_memory.py` | **New** | FP8 memory allocation and sizing validation tests |

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
