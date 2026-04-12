"""Oracle D: NumPy Precision Oracle — multi-precision convergence reference.

Pure NumPy implementation of the mathematical model with faithful
three-role precision simulation at every storage/compute/state boundary.

Authority: precision-limited convergence rate.  Answers the question:
"Given only the mathematical model and the precision configuration,
what convergence trajectory should we expect?"

Comparison against engine backends isolates precision effects (expected)
from decomposition bugs (unexpected).  When the engine's clipping is
inactive, loss curves should track within precision-derived tolerance.
When clipping activates, Oracle D provides the "unclipped, precision-only"
baseline — divergence from the engine is expected and informative.

Dependencies: numpy only.  Zero torch.  Zero shared code with Oracles
A/B/C.  Optional: ml_dtypes for FP8 simulation (falls back to a software
scalar quantizer matching kernels.cl.h's store_storage_fp8 logic).
Optional: scipy for ``make_independent_bce`` (a scipy-free alternative
``make_independent_bce_no_scipy`` is provided).

Precision boundary modeling:
  - Hidden activation storage (H): always modeled (dominant effect).
  - Gradient storage round-trip: modeled by default (significant for FP8).
    Configurable hops (default 1; set ``gradient_storage_hops=2`` to
    model the engine's write→clip→write→reduce pipeline more faithfully).
  - Logit storage round-trip: optional (secondary effect, default off).
  - Probability storage round-trip: optional (tertiary effect, default off).
  - Hidden mask strategy: configurable (``"explicit"`` or ``"recompute"``).
    ``"explicit"`` captures the compute-precision derivative truth before
    activation storage narrowing.  ``"recompute"`` derives the mask from
    stored activations, exposing FP8 quantization-floor effects on
    gradient gating.

  Note: the engine applies multiple storage round-trips through the
  tiled pipeline (write partial → read → clip → write → read → reduce).
  The oracle applies ``gradient_storage_hops`` round-trips per boundary,
  capturing the dominant quantization effect without replicating
  node-level decomposition.

Reference: doc_archive/Oracle.md §Option D: NumPy Precision Oracle
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ── Optional FP8 support ──────────────────────────────────────────────

_ML_DTYPES_AVAILABLE: bool = False
try:
    import ml_dtypes  # type: ignore[import-untyped]

    _ML_DTYPES_AVAILABLE = True  # pyright: ignore[reportConstantRedefinition]
except ImportError:
    pass


# =====================================================================
# Configuration & Data Types
# =====================================================================


@dataclass(frozen=True)
class NumpyPrecisionSpec:
    """Three-role precision specification for NumPy simulation.

    Maps directly to the architecture's ``PrecisionConfig``:

    ============= ==========================================
    Role          Purpose
    ============= ==========================================
    storage_dtype Bandwidth optimization (narrowest format)
    compute_dtype Arithmetic fidelity
    state_dtype   Long-term optimizer stability
    ============= ==========================================

    For FP8 configurations, ``storage_dtype`` is the computational
    container (``float32``).  The actual FP8 quantization is applied by
    ``_fp8_roundtrip`` keyed on ``fp8_variant``.  This models the
    information loss faithfully without requiring numpy to natively
    support 8-bit floats.
    """

    storage_dtype: type | np.dtype
    compute_dtype: type | np.dtype
    state_dtype: type | np.dtype
    fp8_variant: str | None = None  # "e4m3" or "e5m2"

    def __post_init__(self) -> None:
        if self.fp8_variant is not None and self.fp8_variant not in (
            "e4m3",
            "e5m2",
        ):
            msg = f"fp8_variant must be 'e4m3' or 'e5m2', got {self.fp8_variant!r}"
            raise ValueError(msg)
        if not self.is_fp8:
            s = np.dtype(self.storage_dtype).itemsize
            c = np.dtype(self.compute_dtype).itemsize
            st = np.dtype(self.state_dtype).itemsize
            if s > c:
                msg = (
                    f"storage itemsize ({s}) must be <= compute itemsize ({c})"
                )
                raise ValueError(msg)
            if s > st:
                msg = (
                    f"storage itemsize ({s}) must be <= state itemsize ({st})"
                )
                raise ValueError(msg)

    # ── Derived properties ──

    @property
    def is_fp8(self) -> bool:
        """True when storage uses 8-bit floating-point quantization."""
        return self.fp8_variant is not None

    @property
    def accum_dtype(self) -> np.dtype:
        """``ACCUM_TYPE = max(compute, state)`` by element size."""
        s = np.dtype(self.state_dtype)
        c = np.dtype(self.compute_dtype)
        return s if s.itemsize > c.itemsize else c

    # ── Factories mirroring PrecisionConfig ──

    @classmethod
    def float32(cls) -> NumpyPrecisionSpec:
        return cls(np.float32, np.float32, np.float32)

    @classmethod
    def float64(cls) -> NumpyPrecisionSpec:
        return cls(np.float64, np.float64, np.float64)

    @classmethod
    def mixed_f16_f32(cls) -> NumpyPrecisionSpec:
        return cls(np.float16, np.float32, np.float32)

    @classmethod
    def mixed_f32_f64_state(cls) -> NumpyPrecisionSpec:
        return cls(np.float32, np.float32, np.float64)

    @classmethod
    def mixed_f16_f64_state(cls) -> NumpyPrecisionSpec:
        return cls(np.float16, np.float32, np.float64)

    @classmethod
    def mixed_f32_f64(cls) -> NumpyPrecisionSpec:
        return cls(np.float32, np.float64, np.float64)

    @classmethod
    def fp8_e4m3(cls) -> NumpyPrecisionSpec:
        """E4M3 storage, FP32 compute, FP32 state."""
        return cls(np.float32, np.float32, np.float32, fp8_variant="e4m3")

    @classmethod
    def fp8_e5m2(cls) -> NumpyPrecisionSpec:
        """E5M2 storage, FP32 compute, FP32 state."""
        return cls(np.float32, np.float32, np.float32, fp8_variant="e5m2")

    @classmethod
    def fp8_e4m3_f16(cls) -> NumpyPrecisionSpec:
        """E4M3 storage, FP16 compute, FP32 state."""
        return cls(np.float32, np.float16, np.float32, fp8_variant="e4m3")

    @classmethod
    def fp8_e5m2_f16(cls) -> NumpyPrecisionSpec:
        """E5M2 storage, FP16 compute, FP32 state."""
        return cls(np.float32, np.float16, np.float32, fp8_variant="e5m2")

    @classmethod
    def fp8_e4m3_f64(cls) -> NumpyPrecisionSpec:
        """E4M3 storage, FP64 compute, FP64 state."""
        return cls(np.float32, np.float64, np.float64, fp8_variant="e4m3")

    @classmethod
    def fp8_e5m2_f64(cls) -> NumpyPrecisionSpec:
        """E5M2 storage, FP64 compute, FP64 state."""
        return cls(np.float32, np.float64, np.float64, fp8_variant="e5m2")


@dataclass(frozen=True)
class OracleDConfig:
    """Model configuration for Oracle D.

    Contains only the mathematical model's hyperparameters.  No tiling,
    no clipping thresholds, no reduction tree parameters — those are
    engine-specific decomposition artifacts.
    """

    input_dim: int
    hidden_dim: int
    output_classes: int
    num_modules: int
    mode: str  # "CCE" or "BCE"
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    temp_min: float = 0.1
    temp_max: float = 10.0
    normalize_epsilon: float = 1e-7
    """Epsilon for Node 21 gradient normalization (division by
    effective_batch_size).  Distinct from ``epsilon`` which is the
    Adam optimizer epsilon.  Matches the engine's
    ``src_scalar_REAL_epsilon`` on the ``normalize_gradients`` kernel."""

    def __post_init__(self) -> None:
        if self.mode not in ("CCE", "BCE"):
            msg = f"mode must be 'CCE' or 'BCE', got {self.mode!r}"
            raise ValueError(msg)


@dataclass
class ConvergenceTrace:
    """Diagnostics recorded during a multi-step training run.

    Locally defined — no dependency on Oracle C's ConvergenceTrace.
    Interface-compatible for test interop.
    """

    loss_history: list[float] = field(default_factory=lambda: [])
    grad_norm_history: dict[str, list[float]] = field(
        default_factory=lambda: {},
    )
    param_norm_history: dict[str, list[float]] = field(
        default_factory=lambda: {},
    )

    @property
    def num_recorded(self) -> int:
        return len(self.loss_history)

    @property
    def initial_loss(self) -> float:
        return self.loss_history[0] if self.loss_history else float("nan")

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def loss_reduction_ratio(self) -> float:
        """``final / initial``.  Values < 1 indicate improvement."""
        if len(self.loss_history) < 2 or self.loss_history[0] == 0.0:
            return float("nan")
        return self.loss_history[-1] / self.loss_history[0]

    @property
    def converged(self) -> bool:
        """Loss decreased overall with no NaN/Inf."""
        if len(self.loss_history) < 2:
            return False
        return self.is_stable and self.loss_history[-1] < self.loss_history[0]

    @property
    def is_stable(self) -> bool:
        """No NaN or Inf in any recorded quantity."""
        for v in self.loss_history:
            if not math.isfinite(v):
                return False
        for norms in self.param_norm_history.values():
            for v in norms:
                if not math.isfinite(v):
                    return False
        for norms in self.grad_norm_history.values():
            for v in norms:
                if not math.isfinite(v):
                    return False
        return True

    @property
    def loss_monotonicity_violations(self) -> int:
        count = 0
        for i in range(1, len(self.loss_history)):
            if self.loss_history[i] > self.loss_history[i - 1]:
                count += 1
        return count


# =====================================================================
# Module-Level Helpers
# =====================================================================


def _stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid avoiding exp overflow for large ``|x|``."""
    pos = x >= 0
    exp_neg = np.exp(-np.abs(x))
    return np.where(
        pos,
        1.0 / (1.0 + exp_neg),
        exp_neg / (1.0 + exp_neg),
    )


# ── Software FP8 Quantization ────────────────────────────────────────
#
# Matches kernels.cl.h store_storage_fp8 + load_storage_fp8 logic:
# round-to-nearest-even, saturation to max finite, NaN → 0.
# Used when ml_dtypes is not installed.  Scalar implementation —
# performance is not a concern for oracle-sized problems.


def _fp8e4m3_roundtrip_scalar(val: float) -> float:
    """Quantize to FP8 E4M3fn and back.  Returns a ``float``."""
    if math.isnan(val):
        return 0.0

    MAX_VAL = 448.0
    MIN_SUB_HALF = 2.0**-10  # min_subnormal(2^-9) * 0.5

    sign = -1.0 if val < 0.0 else 1.0
    aval = abs(val)

    if aval >= MAX_VAL:
        return sign * MAX_VAL
    if aval < MIN_SUB_HALF:
        return 0.0

    # Extract FP32 bits (little-endian)
    (fbits,) = struct.unpack("<I", struct.pack("<f", np.float32(aval)))
    exp32 = ((fbits >> 23) & 0xFF) - 127  # unbiased
    mant32 = fbits & 0x7FFFFF  # 23-bit mantissa

    exp8 = exp32 + 7  # E4M3 bias = 7

    # Subnormal handling
    # NOTE: Subnormal rounding omits shifted-out bits from sticky calculation.
    # Max error: 1 ULP of FP8 subnormal (2^-9). Below quantization floor;
    # no fix required.
    if exp8 <= 0:
        shift = 1 - exp8
        if shift >= 24:
            return 0.0
        mant32 = (mant32 | 0x800000) >> shift
        exp8 = 0
    elif exp8 > 15:
        return sign * MAX_VAL

    # Round 23-bit mantissa to 3 bits (round-to-nearest-even)
    round_bit = (mant32 >> 19) & 1
    sticky = mant32 & ((1 << 19) - 1)
    mant8 = mant32 >> 20

    if round_bit and (sticky or (mant8 & 1)):
        mant8 += 1

    if mant8 >= 8:
        mant8 = 0
        exp8 += 1
        if exp8 > 15:
            return sign * MAX_VAL

    # E4M3fn: exp=15 mant>=7 is NaN — clamp to max finite
    if exp8 == 15 and mant8 >= 7:
        return sign * MAX_VAL

    # Reconstruct float
    if exp8 == 0:
        result = (mant8 / 8.0) * 2.0 ** (1 - 7)
    else:
        result = (1.0 + mant8 / 8.0) * 2.0 ** (exp8 - 7)
    return sign * result


def _fp8e5m2_roundtrip_scalar(val: float) -> float:
    """Quantize to FP8 E5M2 and back.  Returns a ``float``."""
    if math.isnan(val):
        return 0.0

    MAX_VAL = 57344.0
    MIN_SUB_HALF = 2.0**-17  # min_subnormal(2^-16) * 0.5

    sign = -1.0 if val < 0.0 else 1.0
    aval = abs(val)

    if aval >= MAX_VAL:
        return sign * MAX_VAL
    if aval < MIN_SUB_HALF:
        return 0.0

    (fbits,) = struct.unpack("<I", struct.pack("<f", np.float32(aval)))
    exp32 = ((fbits >> 23) & 0xFF) - 127
    mant32 = fbits & 0x7FFFFF

    exp8 = exp32 + 15  # E5M2 bias = 15

    if exp8 <= 0:
        shift = 1 - exp8
        if shift >= 24:
            return 0.0
        mant32 = (mant32 | 0x800000) >> shift
        exp8 = 0
    elif exp8 >= 31:
        return sign * MAX_VAL

    # Round 23-bit mantissa to 2 bits (round-to-nearest-even)
    round_bit = (mant32 >> 20) & 1
    sticky = mant32 & ((1 << 20) - 1)
    mant8 = mant32 >> 21

    if round_bit and (sticky or (mant8 & 1)):
        mant8 += 1

    if mant8 >= 4:
        mant8 = 0
        exp8 += 1
        if exp8 >= 31:
            return sign * MAX_VAL

    if exp8 == 0:
        result = (mant8 / 4.0) * 2.0 ** (1 - 15)
    else:
        result = (1.0 + mant8 / 4.0) * 2.0 ** (exp8 - 15)
    return sign * result


def _software_fp8_roundtrip(
    x: np.ndarray,
    variant: str,
) -> np.ndarray:
    """Scalar-loop FP8 round-trip.  Fallback when ``ml_dtypes`` is absent."""
    func = (
        _fp8e4m3_roundtrip_scalar
        if variant == "e4m3"
        else _fp8e5m2_roundtrip_scalar
    )
    vfunc = np.vectorize(func, otypes=[np.float32])
    return vfunc(np.asarray(x, dtype=np.float32))


# =====================================================================
# Oracle D
# =====================================================================


class NumPyPrecisionOracle:
    """Pure NumPy multi-precision convergence reference.

    Implements the mathematical model::

        Input → ReLU(X @ W_shared.T + b_shared)
              → per-module (H @ W_module[m] + b_module[m])
              → temperature scaling (logits / T[m])
              → softmax (CCE) or sigmoid (BCE)
              → loss

    with faithful three-role precision simulation at every boundary.

    Parameters initialize to zero.  Use ``with_xavier_init()`` to create
    an oracle with standard random initialization that breaks weight
    symmetry and enables shared-layer gradient flow. Alternatively, call
    ``load_state()`` with externally-prepared parameters.

    Parameters
    ----------
    config : OracleDConfig
        Model hyperparameters.
    precision : NumpyPrecisionSpec
        Three-role precision configuration.
    max_grad_norm : float or None
        Optional simple global L2 gradient clip (safety net).  Not
        equivalent to the engine's staged Quadratic Scaling Policy.
    clip_epsilon : float
        Small constant to prevent division by zero during gradient
        clipping.  Distinct from ``config.epsilon`` (the Adam epsilon)
        and ``config.normalize_epsilon`` (the normalization epsilon).
    mask_strategy : str
        Hidden mask derivation strategy.  One of:

        - ``"explicit"`` (default): Captures the compute-precision ReLU
          derivative truth *before* activation storage narrowing,
          matching the engine's ``produce_hidden_mask`` / explicit mode.
          Preserves correct gradient gating for activations that were
          positive at compute precision but quantized to zero by narrow
          storage (especially FP8).
        - ``"recompute"``: Derives the mask from stored activations
          (``mask = stored_activation > 0``), matching the engine's
          recompute mode.  Activations quantized below the storage
          format's minimum positive subnormal produce mask=0, blocking
          gradient flow through those units.

        The mask strategy affects only the backward pass (gradient
        gating).  Forward-pass probabilities and loss are identical
        under both strategies.
    model_gradient_storage : bool
        Apply storage round-trip to computed gradients, modeling the
        engine's STORAGE_TYPE partial gradient buffers.
    gradient_storage_hops : int
        Number of sequential storage round-trips applied to gradients
        when ``model_gradient_storage`` is True.  Default 1 captures
        the dominant quantization effect.  Set to 2 to more faithfully
        model the engine's multi-hop pipeline (write partial → read →
        clip → write clipped → read → reduce), which applies two
        storage round-trips before gradients enter compute-role
        reduction buffers.
    model_logit_storage : bool
        Apply storage round-trip to logits.  Secondary effect; the
        hidden-activation boundary dominates.
    model_probability_storage : bool
        Apply storage round-trip to probabilities before backward-pass
        gradient computation, modeling the engine's STORAGE_TYPE
        ``partial_probs`` collection buffer read by Nodes 8, 9, and 10.
        Tertiary effect — hidden-activation storage dominates.  For FP8
        E4M3 (3-bit mantissa), softmax outputs near ``1/K`` for large
        ``K`` experience up to ~12.5% relative quantization error,
        which may produce detectable precision gaps.
    """

    _PARAM_NAMES = ("W_shared", "b_shared", "W_module", "b_module", "temps")

    def __init__(
        self,
        config: OracleDConfig,
        precision: NumpyPrecisionSpec,
        max_grad_norm: float | None = None,
        clip_epsilon: float = 1e-7,
        mask_strategy: str = "explicit",
        model_gradient_storage: bool = True,
        gradient_storage_hops: int = 1,
        model_logit_storage: bool = False,
        model_probability_storage: bool = False,
    ) -> None:
        if mask_strategy not in ("explicit", "recompute"):
            msg = (
                f"mask_strategy must be 'explicit' or 'recompute', "
                f"got {mask_strategy!r}"
            )
            raise ValueError(msg)
        if gradient_storage_hops < 1:
            msg = (
                f"gradient_storage_hops must be >= 1, "
                f"got {gradient_storage_hops}"
            )
            raise ValueError(msg)

        self.config = config
        self.prec = precision
        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.mask_strategy = mask_strategy
        self.model_gradient_storage = model_gradient_storage
        self.gradient_storage_hops = gradient_storage_hops
        self.model_logit_storage = model_logit_storage
        self.model_probability_storage = model_probability_storage

        S = np.dtype(precision.state_dtype)

        # ── Learnable parameters at state precision ──
        self.W_shared = np.zeros(
            (config.hidden_dim, config.input_dim),
            dtype=S,
        )
        self.b_shared = np.zeros(config.hidden_dim, dtype=S)
        self.W_module = np.zeros(
            (config.num_modules, config.hidden_dim, config.output_classes),
            dtype=S,
        )
        self.b_module = np.zeros(
            (config.num_modules, config.output_classes),
            dtype=S,
        )
        self.temps = np.ones(config.num_modules, dtype=S)

        # ── Adam state at state precision ──
        self.m1: dict[str, np.ndarray] = {
            n: np.zeros_like(p) for n, p in self._named_params()
        }
        self.m2: dict[str, np.ndarray] = {
            n: np.zeros_like(p) for n, p in self._named_params()
        }
        self.t: int = 0

        # ── Diagnostics ──
        # Cumulative across all step() calls on this oracle instance.
        # Distinct from ConvergenceTrace.loss_history which is per-run
        # with optional subsampling via record_every.
        self.cumulative_loss_history: list[float] = []
        self.last_loss: float = float("nan")
        self.last_grad_norms: dict[str, float] = {}
        self.final_grads: dict[str, np.ndarray] = {}

    # =================================================================
    # Factory constructors
    # =================================================================

    @classmethod
    def with_xavier_init(
        cls,
        config: OracleDConfig,
        precision: NumpyPrecisionSpec,
        seed: int = 42,
        **kwargs: Any,
    ) -> NumPyPrecisionOracle:
        """Create an oracle with Xavier/Glorot uniform weight initialization.

        Breaks weight symmetry to enable shared-layer gradient flow.
        Biases remain zero and temperatures remain 1.0 (standard practice).

        Parameters
        ----------
        config : OracleDConfig
            Model hyperparameters.
        precision : NumpyPrecisionSpec
            Three-role precision configuration.
        seed : int
            Random seed for reproducible initialization.
        **kwargs
            Forwarded to the ``NumPyPrecisionOracle`` constructor
            (e.g., ``max_grad_norm``, ``mask_strategy``,
            ``gradient_storage_hops``).

        Returns
        -------
        NumPyPrecisionOracle
            Initialized oracle ready for training.
        """
        oracle = cls(config, precision, **kwargs)
        rng = np.random.default_rng(seed)
        S = np.dtype(precision.state_dtype)

        # Xavier/Glorot uniform: U[-limit, +limit]
        # where limit = sqrt(6 / (fan_in + fan_out))
        limit_s = np.sqrt(6.0 / (config.input_dim + config.hidden_dim))
        oracle.W_shared[...] = rng.uniform(
            -limit_s, limit_s, oracle.W_shared.shape,
        ).astype(S)

        limit_m = np.sqrt(6.0 / (config.hidden_dim + config.output_classes))
        oracle.W_module[...] = rng.uniform(
            -limit_m, limit_m, oracle.W_module.shape,
        ).astype(S)

        return oracle

    # =================================================================
    # Parameter access
    # =================================================================

    def _named_params(self) -> list[tuple[str, np.ndarray]]:
        return [
            ("W_shared", self.W_shared),
            ("b_shared", self.b_shared),
            ("W_module", self.W_module),
            ("b_module", self.b_module),
            ("temps", self.temps),
        ]

    # =================================================================
    # State serialization
    # =================================================================

    def export_state(self) -> dict[str, np.ndarray]:
        """Deep copy of all learnable state (parameters + Adam moments)."""
        state: dict[str, np.ndarray] = {}
        for name, param in self._named_params():
            state[name] = param.copy()
            state[f"m1_{name}"] = self.m1[name].copy()
            state[f"m2_{name}"] = self.m2[name].copy()
        state["_step"] = np.array(self.t, dtype=np.int64)
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        """Load state, casting to the oracle's precision roles."""
        for name, param in self._named_params():
            param[...] = np.asarray(state[name], dtype=param.dtype)
            self.m1[name][...] = np.asarray(
                state[f"m1_{name}"],
                dtype=self.m1[name].dtype,
            )
            self.m2[name][...] = np.asarray(
                state[f"m2_{name}"],
                dtype=self.m2[name].dtype,
            )
        self.t = int(state["_step"])

    # =================================================================
    # Public API
    # =================================================================

    def step(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Execute one full training step (forward + backward + Adam).

        Parameters
        ----------
        X : ndarray, shape (batch, input_dim)
            Input data.
        targets : ndarray
            CCE: ``(batch,)`` integer class indices.
            BCE: ``(batch, output_classes)`` binary targets.
        sample_mask : ndarray or None, shape (batch,)
            Boolean mask.  ``True`` = valid, ``False`` = padding/invalid.
            ``None`` treats all samples as valid.

        Returns
        -------
        probs : ndarray, shape (num_modules, batch, output_classes)
            Probabilities at compute precision.  Masked samples receive
            uniform (CCE) or 0.5 (BCE) probabilities from zeroed logits.

        Notes
        -----
        The returned loss (available via ``self.last_loss``) is the
        **total scalar loss** summed across all modules and all valid
        samples, consistent with the engine's batch-level loss
        aggregation.  For per-sample or per-module loss, divide by the
        appropriate factor externally.
        """
        C = np.dtype(self.prec.compute_dtype)
        B = X.shape[0]

        # ── Setup ────────────────────────────────────────────
        if sample_mask is None:
            mask_arr = np.ones(B, dtype=bool)
        else:
            mask_arr = np.asarray(sample_mask, dtype=bool)
        effective_N = float(np.sum(mask_arr))
        mask_f = mask_arr.astype(C)

        # === FORWARD =============================================

        X_c, H_c, hidden_mask, logits, T_c, probs, loss = self._forward(
            X, mask_f, targets,
        )
        self.last_loss = loss
        self.cumulative_loss_history.append(loss)

        # === BACKWARD ============================================

        # Probability storage round-trip for backward pass.
        # In the engine, Nodes 8/9/10 read partial_probs from storage.
        # The loss was already computed from compute-precision probs
        # (Nodes 6/7 produce loss and probs in the same kernel, before
        # writing probs to storage), so loss is unaffected.
        probs_bwd = (
            self._storage_roundtrip(probs)
            if self.model_probability_storage
            else probs
        )

        # dL/d(scaled_logits)
        d_scaled = self._d_loss_d_scaled(probs_bwd, targets, mask_f)

        # Temperature gradient:
        #   dL/dT[m] = -1/T² · Σ_{b,c} d_scaled[m,b,c] · logits[m,b,c]
        grad_temps = (
            np.einsum("mbc,mbc->m", d_scaled, logits)
            * (-1.0 / (T_c**2))
        ).astype(C)

        # dL/d(logits) = d_scaled / T  (temperature chain rule)
        d_logits = (d_scaled / T_c[:, None, None]).astype(C)

        # Module parameter gradients
        grad_W_m = np.einsum("bh,mbc->mhc", H_c, d_logits).astype(C)
        grad_b_m = d_logits.sum(axis=1).astype(C)

        # Backprop to hidden (sum across modules)
        W_m_c = self._load_state(self.W_module)
        grad_H = np.einsum("mbc,mhc->bh", d_logits, W_m_c).astype(C)

        # ReLU backward: gated by hidden_mask (strategy-dependent).
        # Under "explicit", mask was captured at compute precision before
        # storage narrowing — positive activations quantized to zero by
        # storage still pass gradient.  Under "recompute", mask is derived
        # from stored activations — those units are gated off.
        grad_H_act = (grad_H * hidden_mask).astype(C)

        # Shared parameter gradients
        grad_W_s = (grad_H_act.T @ X_c).astype(C)
        grad_b_s = grad_H_act.sum(axis=0).astype(C)

        grads: dict[str, np.ndarray] = {
            "W_shared": grad_W_s,
            "b_shared": grad_b_s,
            "W_module": grad_W_m,
            "b_module": grad_b_m,
            "temps": grad_temps,
        }

        # === POST-PROCESSING =====================================

        # Gradient storage round-trips.  The engine's tiled pipeline
        # applies multiple storage boundaries (write partial → read →
        # clip → write clipped → read → reduce).  gradient_storage_hops
        # controls accuracy: 1 captures the dominant effect, 2 matches
        # the engine's write→clip→write→reduce pipeline.
        if self.model_gradient_storage:
            for _hop in range(self.gradient_storage_hops):
                grads = {
                    k: self._storage_roundtrip(v) for k, v in grads.items()
                }

        # Optional global clip (simple safety net)
        if self.max_grad_norm is not None:
            self._global_clip(grads)

        # Record raw gradient norms (before normalization)
        self.last_grad_norms = {
            k: float(np.linalg.norm(v)) for k, v in grads.items()
        }

        # Node 21: normalize by effective batch size.
        # Uses config.normalize_epsilon (distinct from Adam's epsilon).
        cfg = self.config
        eps_norm = np.asarray(cfg.normalize_epsilon, dtype=C)
        N_c = np.asarray(effective_N, dtype=C)
        denom = N_c + eps_norm
        for k in grads:
            grads[k] = (grads[k] / denom).astype(C)
        self.final_grads = grads

        # === OPTIMIZER ===========================================

        self.t += 1
        for name, param in self._named_params():
            self._adam_update(name, param, grads[name])

        # Node 25: temperature clamp
        np.clip(self.temps, cfg.temp_min, cfg.temp_max, out=self.temps)

        return probs

    def predict(
        self,
        X: np.ndarray,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Forward pass only (no learning).

        Returns
        -------
        probs : ndarray, shape (num_modules, batch, output_classes)
        """
        C = np.dtype(self.prec.compute_dtype)
        B = X.shape[0]

        if sample_mask is None:
            mask_f = np.ones(B, dtype=C)
        else:
            mask_f = np.asarray(sample_mask, dtype=bool).astype(C)

        _, _, _, _, _, probs, _ = self._forward(X, mask_f)
        return probs

    def train_n_steps(
        self,
        X: np.ndarray,
        targets: np.ndarray,
        n: int,
        sample_mask: np.ndarray | None = None,
        record_every: int = 1,
    ) -> ConvergenceTrace:
        """Train for *n* steps on the same batch, recording diagnostics.

        The same ``(X, targets)`` are presented every step — this tests
        the system's ability to overfit a fixed dataset.
        """
        trace = ConvergenceTrace()
        for name in self._PARAM_NAMES:
            trace.grad_norm_history[name] = []
            trace.param_norm_history[name] = []

        for step_idx in range(n):
            self.step(X, targets, sample_mask)

            if step_idx % record_every == 0 or step_idx == n - 1:
                trace.loss_history.append(self.last_loss)
                for name, param in self._named_params():
                    trace.grad_norm_history[name].append(
                        self.last_grad_norms.get(name, 0.0),
                    )
                    trace.param_norm_history[name].append(
                        float(np.linalg.norm(param)),
                    )
        return trace

    # =================================================================
    # Internal: forward pass
    # =================================================================

    def _forward(
        self,
        X: np.ndarray,
        mask_f: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,  # X_c
        np.ndarray,  # H_c
        np.ndarray,  # hidden_mask
        np.ndarray,  # logits
        np.ndarray,  # T_c
        np.ndarray,  # probs
        float,  # loss
    ]:
        """Shared forward pass: input → hidden → logits → probs (± loss).

        Returns ``(X_c, H_c, hidden_mask, logits, T_c, probs, loss)``.
        When ``targets`` is ``None``, ``loss`` is ``nan`` and only
        probabilities are computed.

        The ``hidden_mask`` is derived according to ``self.mask_strategy``:

        - ``"explicit"``: ``(H_compute > 0)`` before storage narrowing.
        - ``"recompute"``: ``(H_stored > 0)`` after storage narrowing.

        Both produce identical masks when storage precision preserves
        all positive activations (FP16/FP32/FP64).  They diverge when
        storage quantization zeroes near-zero positive activations (FP8).
        """
        C = np.dtype(self.prec.compute_dtype)

        # Node 4: shared layer
        X_c = self._storage_roundtrip(np.asarray(X, dtype=C))
        W_s_c = self._load_state(self.W_shared)
        b_s_c = self._load_state(self.b_shared)
        pre_relu = (X_c @ W_s_c.T + b_s_c).astype(C)
        H_compute = np.maximum(pre_relu, np.zeros(1, dtype=C))

        # ★ Key precision boundary: activation storage
        H_c = self._storage_roundtrip(H_compute)

        # Hidden mask: strategy-dependent derivation
        if self.mask_strategy == "explicit":
            # Compute-precision derivative truth captured before storage
            # narrowing.  Preserves gradient gating for activations that
            # were positive at compute precision but quantized to zero
            # by narrow storage formats (FP8 quantization floor).
            # The mask values (0.0/1.0) are exactly representable in all
            # floating-point formats, so the storage round-trip of the
            # mask buffer itself is lossless — we skip it.
            hidden_mask = (H_compute > 0).astype(C)
        else:
            # Derived from stored activations, matching the engine's
            # recompute codepath: mask = (load_storage(h) > 0).
            # For FP8 E4M3, activations in (0, 2^-9) round to zero,
            # producing mask=0 and blocking gradient flow.
            hidden_mask = (H_c > 0).astype(C)

        # Node 5: module logits
        W_m_c = self._load_state(self.W_module)  # (M, H, Cl)
        b_m_c = self._load_state(self.b_module)  # (M, Cl)
        T_c = self._load_state(self.temps)  # (M,)
        logits: np.ndarray = (
            np.einsum("bh,mhc->mbc", H_c, W_m_c) + b_m_c[:, None, :]
        ).astype(C)
        # Sample masking (Node 5 behavior)
        logits = (logits * mask_f[None, :, None]).astype(C)
        if self.model_logit_storage:
            logits = self._storage_roundtrip(logits)

        # Nodes 6/7: temperature scaling → probs + loss
        scaled = (logits / T_c[:, None, None]).astype(C)

        if targets is not None:
            probs, loss = self._probs_and_loss(scaled, targets, mask_f)
        else:
            probs = self._compute_probs(scaled)
            loss = float("nan")

        return X_c, H_c, hidden_mask, logits, T_c, probs, loss

    # =================================================================
    # Internal: forward helpers
    # =================================================================

    def _compute_probs(self, scaled: np.ndarray) -> np.ndarray:
        """Compute probabilities from temperature-scaled logits.

        Used by ``predict`` path (no loss computation needed).
        """
        C = np.dtype(self.prec.compute_dtype)

        if self.config.mode == "CCE":
            max_s = scaled.max(axis=-1, keepdims=True)
            exp_s = np.exp(scaled - max_s)
            return (exp_s / exp_s.sum(axis=-1, keepdims=True)).astype(C)
        return _stable_sigmoid(scaled).astype(C)

    def _probs_and_loss(
        self,
        scaled: np.ndarray,
        targets: np.ndarray,
        mask_f: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Compute probabilities and loss from temperature-scaled logits.

        Parameters
        ----------
        scaled : (M, B, C_out) at compute dtype
        targets : CCE (B,) int or BCE (B, C_out) float
        mask_f : (B,) float mask (1.0 valid, 0.0 masked)

        Returns
        -------
        (probs, loss) where probs is (M, B, C_out) at compute dtype.
        Loss is the total scalar sum across all modules and valid samples.
        """
        C = np.dtype(self.prec.compute_dtype)
        B = scaled.shape[1]

        if self.config.mode == "CCE":
            # Numerically stable softmax via log-sum-exp
            max_s = scaled.max(axis=-1, keepdims=True)
            shifted = scaled - max_s
            exp_s = np.exp(shifted)
            sum_exp = exp_s.sum(axis=-1, keepdims=True)
            probs = (exp_s / sum_exp).astype(C)

            # Log-softmax for loss (more stable than log(probs))
            log_probs = shifted - np.log(sum_exp)
            target_lp = log_probs[
                :, np.arange(B), np.asarray(targets, dtype=int),
            ]  # (M, B)
            loss = float(-(target_lp * mask_f[None, :]).sum())
            return probs, loss

        # BCE
        probs = _stable_sigmoid(scaled).astype(C)
        targets_bc = np.broadcast_to(
            np.asarray(targets, dtype=C)[None, :, :],
            scaled.shape,
        )
        # Stable BCE: max(x,0) - x*y + log(1 + exp(-|x|))
        loss_elem = (
            np.maximum(scaled, 0)
            - scaled * targets_bc
            + np.log1p(np.exp(-np.abs(scaled)))
        )
        loss = float((loss_elem * mask_f[None, :, None]).sum())
        return probs, loss

    def _d_loss_d_scaled(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        mask_f: np.ndarray,
    ) -> np.ndarray:
        """``dL/d(scaled_logits)`` with sample masking applied.

        CCE: ``probs - one_hot(targets)``
        BCE: ``probs - targets``
        """
        C = np.dtype(self.prec.compute_dtype)
        B = probs.shape[1]

        if self.config.mode == "CCE":
            one_hot = np.zeros(
                (B, self.config.output_classes),
                dtype=C,
            )
            one_hot[np.arange(B), np.asarray(targets, dtype=int)] = 1.0
            d_scaled = probs - one_hot[None, :, :]  # broadcast across M
        else:
            targets_bc = np.broadcast_to(
                np.asarray(targets, dtype=C)[None, :, :],
                probs.shape,
            )
            d_scaled = probs - targets_bc

        # Zero masked samples' contributions
        d_scaled = d_scaled * mask_f[None, :, None]
        return d_scaled.astype(C)

    # =================================================================
    # Internal: gradient clipping
    # =================================================================

    def _global_clip(self, grads: dict[str, np.ndarray]) -> None:
        """Simple global L2 clip — not the engine's staged policy."""
        assert self.max_grad_norm is not None
        C = np.dtype(self.prec.compute_dtype)
        all_flat = np.concatenate(
            [g.astype(C).ravel() for g in grads.values()],
        )
        global_norm = float(np.sqrt(np.sum(all_flat * all_flat)))
        if global_norm > self.max_grad_norm:
            scale = np.asarray(
                self.max_grad_norm / (global_norm + self.clip_epsilon),
                dtype=C,
            )
            for k in grads:
                grads[k] = (grads[k].astype(C) * scale).astype(C)

    # =================================================================
    # Internal: optimizer
    # =================================================================

    def _adam_update(
        self,
        name: str,
        param: np.ndarray,
        grad: np.ndarray,
    ) -> None:
        """State-precision-accumulation-aware Adam.

        EMA in ``ACCUM_TYPE = max(compute, state)``.
        Bias correction and update delta in ``COMPUTE_TYPE``.
        Parameter subtraction accumulative in ``ACCUM_TYPE``.

        Hyperparameters are first narrowed to ``COMPUTE_TYPE`` and then
        widened to ``ACCUM_TYPE``, faithfully modeling the engine's
        interface where scalar hyperparameters are received as
        ``COMPUTE_TYPE`` and widened via ``widen_to_accum()``.
        """
        cfg = self.config
        A = np.dtype(self.prec.accum_dtype)
        C = np.dtype(self.prec.compute_dtype)

        # Host-computed bias correction (Python float64)
        beta1_pow_t = cfg.beta1 ** self.t
        beta2_pow_t = cfg.beta2 ** self.t

        # Load moments at ACCUM_TYPE (preserves state fidelity)
        m1 = self.m1[name].astype(A)
        m2 = self.m2[name].astype(A)
        g = grad.astype(A)

        # Model the COMPUTE_TYPE bottleneck on hyperparameters:
        # The engine receives β₁, β₂ as COMPUTE_TYPE scalars and widens
        # them to ACCUM_TYPE.  The narrowing to C loses precision for
        # configurations where C < A (e.g., mixed_f32_f64_state).
        b1 = np.asarray(cfg.beta1, dtype=C).astype(A)
        b2 = np.asarray(cfg.beta2, dtype=C).astype(A)
        one_a = np.asarray(1.0, dtype=A)  # 1.0 is exact in all formats

        # EMA in ACCUM_TYPE (accumulative)
        m1_new = b1 * m1 + (one_a - b1) * g
        m2_new = b2 * m2 + (one_a - b2) * (g * g)

        # Store moments at state precision
        self.m1[name] = m1_new.astype(np.dtype(self.prec.state_dtype))
        self.m2[name] = m2_new.astype(np.dtype(self.prec.state_dtype))

        # Bias correction in COMPUTE_TYPE (transformative)
        m1_c = m1_new.astype(C)
        m2_c = m2_new.astype(C)

        # lr and eps are also received as COMPUTE_TYPE scalars in the
        # engine.  They are used in transformative (not accumulative)
        # operations, so no widening to ACCUM_TYPE is needed.
        lr = np.asarray(cfg.learning_rate, dtype=C)
        eps = np.asarray(cfg.epsilon, dtype=C)

        # Bias correction terms: host-computed in FP64, narrowed to
        # COMPUTE_TYPE at the interface boundary (matching Node 24).
        bc1 = np.asarray(1.0 - beta1_pow_t, dtype=C)
        bc2 = np.asarray(1.0 - beta2_pow_t, dtype=C)

        m1_hat = m1_c / bc1
        m2_hat = m2_c / bc2
        update = lr * m1_hat / (np.sqrt(m2_hat) + eps)

        # Parameter update in ACCUM_TYPE (accumulative on state-role)
        param_a = param.astype(A)
        param_a -= update.astype(A)
        param[...] = param_a.astype(np.dtype(self.prec.state_dtype))

    # =================================================================
    # Internal: precision boundary helpers
    # =================================================================

    def _storage_roundtrip(self, x: np.ndarray) -> np.ndarray:
        """``compute → storage → compute``.  The key precision-loss point."""
        if self.prec.is_fp8:
            return self._fp8_roundtrip(x)
        sd = np.dtype(self.prec.storage_dtype)
        cd = np.dtype(self.prec.compute_dtype)
        if sd == cd:
            return x
        return x.astype(sd).astype(cd)

    def _load_state(self, x: np.ndarray) -> np.ndarray:
        """``state → compute`` for forward/backward arithmetic."""
        return x.astype(np.dtype(self.prec.compute_dtype))

    def _fp8_roundtrip(self, x: np.ndarray) -> np.ndarray:
        """FP8 quantize-dequantize.  ``ml_dtypes`` preferred; scalar fallback."""
        assert self.prec.fp8_variant is not None
        cd = np.dtype(self.prec.compute_dtype)

        if _ML_DTYPES_AVAILABLE:
            fp8_type = {
                "e4m3": ml_dtypes.float8_e4m3fn,  # pyright: ignore[reportPossiblyUnboundVariable]
                "e5m2": ml_dtypes.float8_e5m2,  # pyright: ignore[reportPossiblyUnboundVariable]
            }[self.prec.fp8_variant]
            return (
                x.astype(np.float32)
                .astype(fp8_type)
                .astype(np.float32)
                .astype(cd)
            )

        result = _software_fp8_roundtrip(x, self.prec.fp8_variant)
        return result.astype(cd)

    # =================================================================
    # Test data generators
    # =================================================================

    @staticmethod
    def make_separable_clusters_cce(
        num_classes: int,
        input_dim: int,
        num_samples_per_class: int = 50,
        separation: float = 3.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Well-separated Gaussian clusters for CCE convergence tests.

        Returns ``(X, targets)`` where ``X`` is ``(N, input_dim)``
        float64 and ``targets`` is ``(N,)`` int64.
        """
        rng = np.random.default_rng(seed)
        centers = rng.standard_normal((num_classes, input_dim)) * separation

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for c in range(num_classes):
            noise = rng.standard_normal(
                (num_samples_per_class, input_dim),
            ) * 0.5
            x_parts.append(centers[c][None, :] + noise)
            y_parts.append(np.full(num_samples_per_class, c, dtype=np.int64))

        X = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        perm = rng.permutation(X.shape[0])
        return X[perm], y[perm]

    @staticmethod
    def make_independent_bce(
        num_classes: int,
        input_dim: int,
        num_samples: int = 200,
        active_probability: float = 0.3,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Independent binary labels for BCE convergence tests.

        Each class is separated by a random hyperplane.

        Requires ``scipy``.  For a scipy-free alternative, use
        ``make_independent_bce_no_scipy``.

        Returns ``(X, targets)`` where ``X`` is ``(N, input_dim)``
        float64 and ``targets`` is ``(N, num_classes)`` float64 binary.
        """
        try:
            from scipy.stats import norm as _norm_dist  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        except ImportError:
            msg = (
                "make_independent_bce requires scipy. "
                "Install scipy or use make_independent_bce_no_scipy "
                "for a scipy-free alternative."
            )
            raise ImportError(msg) from None

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((num_samples, input_dim))

        normals = rng.standard_normal((num_classes, input_dim))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        # Bias to achieve approximate active_probability
        bias = _norm_dist.ppf(1.0 - active_probability)  # pyright: ignore[reportUnknownMemberType]
        projections = X @ normals.T  # (N, num_classes)
        targets = (projections > bias).astype(np.float64)
        return X, targets

    @staticmethod
    def make_independent_bce_no_scipy(
        num_classes: int,
        input_dim: int,
        num_samples: int = 200,
        active_probability: float = 0.3,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """BCE test data without scipy dependency.

        Uses a per-class quantile threshold to approximate the target
        activation rate.  Statistically equivalent to
        ``make_independent_bce`` for large ``num_samples``, but uses
        empirical quantiles rather than the theoretical Gaussian ppf.
        """
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((num_samples, input_dim))
        normals = rng.standard_normal((num_classes, input_dim))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        projections = X @ normals.T
        # Approximate threshold: sort projections and take quantile
        thresholds = np.quantile(
            projections,
            1.0 - active_probability,
            axis=0,
        )
        targets = (projections > thresholds[None, :]).astype(np.float64)
        return X, targets
