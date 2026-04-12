# tests/tier3/test_checkpoint_migration.py
# pyright: reportPrivateUsage=false
"""Checkpoint version migration tests for FP8 scale compatibility."""

import warnings

from src.shared.precision_config import PrecisionConfig
from src.shared.fp8_scaling import FP8_SCALES_VERSION, FP8ScaleInfo
from src.main_orchestrator import (
    TrainingOrchestrator,
    TrainingHyperparams,
    StabilizationConfig,
)
from src.shared.model_spec import ModelSpec
from src.shared.hardware_profile import HardwareProfile
from src.backends.cpu.renderer import CPUPlanRenderer


def _make_orchestrator(precision: PrecisionConfig) -> TrainingOrchestrator:
    """Create a minimal orchestrator for checkpoint tests."""
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
        epochs=1,
        learning_rate=0.001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-7,
        temp_min=0.1,
        temp_max=10.0,
        stabilization=StabilizationConfig(max_grad_norm=1.0, lambda_=1.0),
    )
    renderer = CPUPlanRenderer()
    return TrainingOrchestrator(
        model_spec=spec,
        hardware=hw,
        renderer=renderer,
        hyperparams=hyperparams,
        problem_type_name="CCE",
        batch_size=32,
    )


class TestCheckpointMigration:
    """Validate checkpoint loading across FP8 scale versions."""

    def test_pre_fp8_checkpoint_loads_with_fp8_config(self):
        """Pre-Phase-9 checkpoint (no fp8_scales key) loads correctly.

        When loading a FP32 checkpoint with an FP8 precision config,
        scales should be empty (computed fresh on first forward pass).
        """
        cfg_fp8 = PrecisionConfig.fp8_e4m3()

        # Simulate old checkpoint (no fp8_scales key)
        old_checkpoint = {
            "training_step": 1000,
            # No "fp8_scales" key — pre-FP8 checkpoint
        }

        orchestrator = _make_orchestrator(cfg_fp8)
        orchestrator.restore_state(old_checkpoint)

        # _activation_scales should be empty, ready for recomputation
        assert orchestrator._activation_scales == {}, (
            "Pre-FP8 checkpoint should result in empty activation scales"
        )

    def test_mismatched_version_triggers_recompute(self):
        """Checkpoint with wrong FP8_SCALES_VERSION triggers scale recomputation."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()

        # Simulate checkpoint with future/incompatible version
        future_checkpoint = {
            "training_step": 500,
            "fp8_scales": {
                "_version": 999,  # Future version
                "scales": {
                    "activation_0": {"scale": 0.5, "inv_scale": 2.0, "storage_max": 448.0},
                },
            },
        }

        orchestrator = _make_orchestrator(cfg_fp8)

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
        """Checkpoint with structurally incompatible FP8ScaleInfo triggers recompute."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()

        # Simulate checkpoint with incompatible scale structure
        incompatible_checkpoint = {
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

        orchestrator = _make_orchestrator(cfg_fp8)

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

        orchestrator = _make_orchestrator(cfg_fp8)
        orchestrator.restore_state(valid_checkpoint)

        assert "activation_0" in orchestrator._activation_scales
        scale_info = orchestrator._activation_scales["activation_0"]
        assert scale_info.scale == 0.5
        assert scale_info.inv_scale == 2.0
        assert scale_info.storage_max == 448.0

    def test_checkpoint_roundtrip(self):
        """Save and restore checkpoint preserves FP8 scales."""
        cfg_fp8 = PrecisionConfig.fp8_e4m3()

        orchestrator = _make_orchestrator(cfg_fp8)
        # Manually set scales
        orchestrator._activation_scales = {
            "act_0": FP8ScaleInfo(scale=0.8, inv_scale=1.25, storage_max=448.0),
            "act_1": FP8ScaleInfo(scale=1.0, inv_scale=1.0, storage_max=448.0),
        }

        # Save
        state = orchestrator.checkpoint_state()

        # Restore into fresh orchestrator
        orchestrator2 = _make_orchestrator(cfg_fp8)
        orchestrator2.restore_state(state)

        assert len(orchestrator2._activation_scales) == 2
        assert orchestrator2._activation_scales["act_0"].scale == 0.8
        assert orchestrator2._activation_scales["act_1"].scale == 1.0

    def test_non_fp8_checkpoint_has_no_fp8_scales(self):
        """Non-FP8 orchestrator checkpoint omits fp8_scales key."""
        cfg_fp32 = PrecisionConfig.float32()

        orchestrator = _make_orchestrator(cfg_fp32)
        state = orchestrator.checkpoint_state()

        assert "fp8_scales" not in state
