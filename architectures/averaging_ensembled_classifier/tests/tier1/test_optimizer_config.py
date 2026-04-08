# tests/tier1/test_optimizer_config.py
"""Unit tests for OptimizerConfig (ADR-029)."""
import pytest

from src.shared.optimizer_config import OptimizerConfig


@pytest.mark.tier1
class TestOptimizerConfigDefaults:
    """Verify default values match the previously hardcoded constants."""

    def test_default_learning_rate(self):
        cfg = OptimizerConfig()
        assert cfg.learning_rate == 0.001

    def test_default_beta1(self):
        cfg = OptimizerConfig()
        assert cfg.beta1 == 0.9

    def test_default_beta2(self):
        cfg = OptimizerConfig()
        assert cfg.beta2 == 0.999

    def test_default_epsilon_is_none(self):
        cfg = OptimizerConfig()
        assert cfg.epsilon is None

    def test_frozen(self):
        cfg = OptimizerConfig()
        with pytest.raises(AttributeError):
            cfg.learning_rate = 0.01  # type: ignore[misc]


@pytest.mark.tier1
class TestOptimizerConfigValidation:
    """Verify __post_init__ rejects invalid values."""

    def test_negative_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=-0.001)

    def test_zero_learning_rate(self):
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=0.0)

    def test_beta1_negative(self):
        with pytest.raises(ValueError, match="beta1 must be in"):
            OptimizerConfig(beta1=-0.1)

    def test_beta1_one(self):
        with pytest.raises(ValueError, match="beta1 must be in"):
            OptimizerConfig(beta1=1.0)

    def test_beta2_negative(self):
        with pytest.raises(ValueError, match="beta2 must be in"):
            OptimizerConfig(beta2=-0.1)

    def test_beta2_one(self):
        with pytest.raises(ValueError, match="beta2 must be in"):
            OptimizerConfig(beta2=1.0)

    def test_epsilon_zero(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            OptimizerConfig(epsilon=0.0)

    def test_epsilon_negative(self):
        with pytest.raises(ValueError, match="epsilon must be positive"):
            OptimizerConfig(epsilon=-1e-8)

    def test_valid_custom_values(self):
        """Accepts valid non-default values without raising."""
        cfg = OptimizerConfig(
            learning_rate=0.1,
            beta1=0.95,
            beta2=0.9999,
            epsilon=1e-7,
        )
        assert cfg.learning_rate == 0.1
        assert cfg.beta1 == 0.95
        assert cfg.beta2 == 0.9999
        assert cfg.epsilon == 1e-7

    def test_beta_zero_allowed(self):
        """beta=0 is valid (disables momentum)."""
        cfg = OptimizerConfig(beta1=0.0, beta2=0.0)
        assert cfg.beta1 == 0.0
        assert cfg.beta2 == 0.0


@pytest.mark.tier1
class TestResolveEpsilon:
    """Verify resolve_epsilon() returns correct epsilon values."""

    def test_none_epsilon_uses_precision_default(self):
        """When epsilon is None, delegate to PrecisionConfig.compute_epsilon."""
        from src.shared.model_spec import ModelSpec

        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=8, output_classes=3,
            num_modules=4, simd_width=4, cache_line_bytes=64,
        )
        cfg = OptimizerConfig()  # epsilon=None
        resolved = cfg.resolve_epsilon(spec.precision)
        assert resolved == spec.precision.compute_epsilon

    def test_explicit_epsilon_overrides_precision(self):
        """When epsilon is set, the precision default is ignored."""
        from src.shared.model_spec import ModelSpec

        spec = ModelSpec.float32(
            input_dim=4, hidden_dim=8, output_classes=3,
            num_modules=4, simd_width=4, cache_line_bytes=64,
        )
        cfg = OptimizerConfig(epsilon=1e-7)
        resolved = cfg.resolve_epsilon(spec.precision)
        assert resolved == 1e-7
