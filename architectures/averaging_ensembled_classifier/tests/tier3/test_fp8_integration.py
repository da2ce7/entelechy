# tests/tier3/test_fp8_integration.py
"""The Bandwidth Extremist: FP8 storage fidelity validation (ADR-025 §9.1)."""

import pytest
import numpy as np

from src.shared.precision_config import PrecisionConfig, FP8_E4M3, FP8_E5M2, FP8_DTYPES
from src.shared.fp8_scaling import (
    compute_fp8_scale,
    apply_fp8_scale,
    unapply_fp8_scale,
)

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
        # E4M3 variants — canonical scenario from CONCEPT.md §11
        (PrecisionConfig.fp8_e4m3, PrecisionConfig.mixed_f16_f32),  # E4M3/FP32/FP32 vs FP16/FP32/FP32
        (PrecisionConfig.fp8_e4m3_f64, PrecisionConfig.float64),    # E4M3/FP64/FP64 vs FP64/FP64/FP64
        # E5M2 variants
        (PrecisionConfig.fp8_e5m2, PrecisionConfig.mixed_f16_f32),  # E5M2/FP32/FP32 vs FP16/FP32/FP32
        (PrecisionConfig.fp8_e5m2_f64, PrecisionConfig.float64),    # E5M2/FP64/FP64 vs FP64/FP64/FP64
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

        # Assert loss trend is decreasing
        baseline_decrease = (loss_baseline[0] - loss_baseline[-1]) / max(loss_baseline[0], 1e-10)
        fp8_decrease = (loss_fp8[0] - loss_fp8[-1]) / max(loss_fp8[0], 1e-10)
        # FP8 must achieve at least 70% of baseline's relative decrease
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

        # Compare storage element sizes directly
        fp8_storage_bytes = cfg_fp8.storage_dtype.itemsize
        fp16_storage_bytes = cfg_fp16.storage_dtype.itemsize

        assert fp8_storage_bytes == fp16_storage_bytes // 2, (
            f"FP8 storage ({fp8_storage_bytes}) should be half of "
            f"FP16 storage ({fp16_storage_bytes})"
        )

    def test_quantization_error_bounded(self, training_config, calibration_mode):
        """Quantization error from FP8 storage is bounded and does not blow up."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()

        # Fixed seed for reproducibility
        np.random.seed(training_config["seed"])

        # Generate test activations in range [0, 1] (post-sigmoid)
        test_activations_low = np.random.rand(1000, 100).astype(np.float32)

        # Also test higher values near E4M3 max (where relative error is larger)
        test_activations_high = (np.random.rand(1000, 100) * 400 + 48).astype(np.float32)  # [48, 448]

        # Test low-range activations [0, 1]
        fp8_stored_low = test_activations_low.astype(cfg_fp8.storage_dtype)
        restored_low = fp8_stored_low.astype(np.float32)
        error_low = np.abs(test_activations_low - restored_low)

        _log_error_distribution(error_low.ravel(), "Low-range abs error", calibration_mode)

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

        _log_error_distribution(relative_error_high.ravel(), "High-range relative error", calibration_mode)

        assert relative_error_high.max() < FP8_HIGH_RANGE_MAX_RELATIVE_ERROR, (
            f"Max relative error (high range) {relative_error_high.max():.2%} exceeds "
            f"{FP8_HIGH_RANGE_MAX_RELATIVE_ERROR*100}%"
        )
        assert relative_error_high.mean() < FP8_HIGH_RANGE_MEAN_RELATIVE_ERROR, (
            f"Mean relative error (high range) {relative_error_high.mean():.2%} exceeds "
            f"{FP8_HIGH_RANGE_MEAN_RELATIVE_ERROR*100}%"
        )

    def test_fp8_and_loss_scaling_compose(self):
        """FP8 activation scaling and loss scaling compose without interference.

        Validates that using both scaling methods produces correct gradients.
        """
        cfg = PrecisionConfig.fp8_e4m3()
        loss_scale = 1024.0  # Typical loss scale factor

        # Create synthetic forward/backward scenario
        np.random.seed(42)
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

    def _run_training(self, precision, config):
        """Execute training with given precision config.

        Returns (loss_history: List[float], final_params: dict).

        Note: Uses fixed seed from config for reproducibility across FP8/baseline runs.
        Note: Import path assumes tests run from project root with standard pytest setup.
              The conftest.py adds `src/` to sys.path.
        """
        from src.main_orchestrator import TrainingOrchestrator, TrainingHyperparams, StabilizationConfig
        from src.shared.model_spec import ModelSpec
        from src.shared.hardware_profile import HardwareProfile
        from src.backends.cpu.renderer import CPUPlanRenderer
        from src.shared.problem_type_strategy import PlanCceStrategy

        np.random.seed(config["seed"])

        spec = ModelSpec(
            precision=precision,
            input_dim=4,
            hidden_dim=32,
            output_classes=3,
            num_modules=8,
            simd_width=4,
            cache_line_bytes=64,
        )
        hw = HardwareProfile(
            simd_width=4,
            cache_line_bytes=64,
            max_reduce_fan_in=256,
            max_local_mem_bytes=65536,
            global_mem_bytes=4 * 1024**3,
        )
        hyperparams = TrainingHyperparams(
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-7,
            temp_min=0.1,
            temp_max=10.0,
            stabilization=StabilizationConfig(max_grad_norm=1.0, lambda_=1.0),
        )

        renderer = CPUPlanRenderer()
        orchestrator = TrainingOrchestrator(
            model_spec=spec,
            hardware=hw,
            renderer=renderer,
            hyperparams=hyperparams,
            problem_type_name="CCE",
            batch_size=config["batch_size"],
        )

        # Generate synthetic data
        X_train = np.random.randn(config["batch_size"], 4).astype(np.float32)
        y_train = np.random.randint(0, 3, size=config["batch_size"]).astype(np.int32)

        loss_history = []
        for epoch in range(config["epochs"]):
            # Use the orchestrator's train method for a single epoch
            probs = orchestrator.train(X_train, y_train)
            # Compute loss from probabilities
            if probs.size > 0:
                # Cross-entropy loss
                eps = 1e-10
                log_probs = np.log(np.clip(probs, eps, 1.0))
                loss = -np.mean(log_probs[np.arange(len(y_train)), y_train])
                loss_history.append(float(loss))

        params = {}  # Placeholder — actual param extraction depends on orchestrator API
        return loss_history, params
