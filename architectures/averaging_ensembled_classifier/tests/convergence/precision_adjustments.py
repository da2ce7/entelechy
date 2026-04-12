# tests/convergence/precision_adjustments.py
"""Precision-specific convergence criteria adjustments (ADR-028, ADR-033)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.precision_config import PrecisionConfig
from tests.oracle.precision_oracle import NumpyPrecisionSpec

from .criteria import ConvergenceCriteria


@dataclass(frozen=True)
class PrecisionAdjustment:
    accuracy_delta: float
    loss_delta: float
    epoch_multiplier: float


PRECISION_ADJUSTMENTS: dict[str, PrecisionAdjustment] = {
    "float16": PrecisionAdjustment(accuracy_delta=0.000000, loss_delta=0.000297, epoch_multiplier=1.0000),
    "float32": PrecisionAdjustment(accuracy_delta=0.000000, loss_delta=0.000000, epoch_multiplier=1.0000),
    "float64": PrecisionAdjustment(accuracy_delta=0.000000, loss_delta=-0.000000, epoch_multiplier=1.0000),
    "float8_e4m3fn": PrecisionAdjustment(accuracy_delta=0.003889, loss_delta=0.002664, epoch_multiplier=1.1111),
}


def adjust_criteria_for_precision(
    base: ConvergenceCriteria,
    precision: PrecisionConfig,
) -> ConvergenceCriteria:
    """Return a new ConvergenceCriteria with precision-specific adjustments applied."""
    key = precision.storage_dtype.name
    adjustments = PRECISION_ADJUSTMENTS.get(key)
    if adjustments is None:
        return base
    return ConvergenceCriteria(
        problem_id=base.problem_id,
        mode=base.mode,
        accuracy_threshold=base.accuracy_threshold + adjustments.accuracy_delta,
        loss_threshold=base.loss_threshold + adjustments.loss_delta,
        epoch_budget=int(base.epoch_budget * adjustments.epoch_multiplier),
        warmup_epochs=base.warmup_epochs,
        monotonicity_tolerance=base.monotonicity_tolerance,
        nan_inf_allowed=base.nan_inf_allowed,
    )


# ── Oracle D (NumpyPrecisionSpec) mapping ────────────────────────────

# Maps NumpyPrecisionSpec storage dtype to the adjustment key.
# FP8 variants use the fp8_variant field instead.
_ORACLE_D_DTYPE_KEY: dict[str, str] = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
}


def adjust_criteria_for_oracle_d_precision(
    base: ConvergenceCriteria,
    precision: NumpyPrecisionSpec,
) -> ConvergenceCriteria:
    """Return adjusted criteria for Oracle D's precision spec.

    Maps ``NumpyPrecisionSpec`` to the same adjustment table used for
    engine ``PrecisionConfig``.  FP8 variants use the fp8_e4m3fn key.
    """
    if precision.is_fp8:
        key = "float8_e4m3fn"
    else:
        key = _ORACLE_D_DTYPE_KEY.get(np.dtype(precision.storage_dtype).name, "float32")

    adjustments = PRECISION_ADJUSTMENTS.get(key)
    if adjustments is None:
        return base
    return ConvergenceCriteria(
        problem_id=base.problem_id,
        mode=base.mode,
        accuracy_threshold=base.accuracy_threshold + adjustments.accuracy_delta,
        loss_threshold=base.loss_threshold + adjustments.loss_delta,
        epoch_budget=int(base.epoch_budget * adjustments.epoch_multiplier),
        warmup_epochs=base.warmup_epochs,
        monotonicity_tolerance=base.monotonicity_tolerance,
        nan_inf_allowed=base.nan_inf_allowed,
    )


# ── Phase 2: Oracle-D-derived precision adjustments ──────────────────


def derive_precision_adjustment(
    *,
    problem_module: object,
    target_precision: NumpyPrecisionSpec,
    reference_precision: NumpyPrecisionSpec | None = None,
    seed: int = 42,
) -> PrecisionAdjustment:
    """Derive a precision adjustment by running Oracle D at two precisions.

    Runs Oracle D at both the reference (default FP32) and the target
    precision on the same problem, then computes the trajectory delta:

    - ``accuracy_delta = target.final_accuracy - reference.final_accuracy``
    - ``loss_delta = target.final_loss - reference.final_loss``
    - ``epoch_multiplier = target.epochs_to_threshold / reference.epochs_to_threshold``
      (capped at 2.0 when either doesn't reach the threshold)

    Parameters
    ----------
    problem_module
        A problem module (e.g. ``problems.iris``) with ``load()``,
        ``BASELINE``, ``FULL_CRITERIA``, and ``MODE`` attributes.
    target_precision
        The precision configuration to derive adjustments for.
    reference_precision
        Baseline precision (defaults to FP32).
    seed
        Weight initialization seed for reproducibility.
    """
    from .oracle_baseline import run_oracle_d_baseline

    if reference_precision is None:
        reference_precision = NumpyPrecisionSpec.float32()

    load_fn = getattr(problem_module, "load")
    baseline = getattr(problem_module, "BASELINE")
    criteria = getattr(problem_module, "FULL_CRITERIA")
    mode = getattr(problem_module, "MODE")

    X, y = load_fn()

    input_dim = int(X.shape[1])
    if mode == "CCE":
        output_classes = int(np.max(y) + 1)
    else:
        output_classes = 1 if y.ndim == 1 else int(y.shape[1])

    common_kwargs = dict(
        input_dim=input_dim,
        hidden_dim=baseline.hidden_size,
        output_classes=output_classes,
        num_modules=8,
        mode=mode,
        X=X, y=y,
        epochs=criteria.epoch_budget,
        batch_size=baseline.batch_size,
        learning_rate=baseline.learning_rate,
        beta1=baseline.beta1,
        beta2=baseline.beta2,
        epsilon=baseline.epsilon,
        seed=seed,
    )

    ref_history = run_oracle_d_baseline(precision=reference_precision, **common_kwargs)
    tgt_history = run_oracle_d_baseline(precision=target_precision, **common_kwargs)

    accuracy_delta = tgt_history.final_accuracy - ref_history.final_accuracy
    loss_delta = tgt_history.final_loss - ref_history.final_loss

    ref_epochs = ref_history.epochs_to_threshold(criteria.accuracy_threshold)
    tgt_epochs = tgt_history.epochs_to_threshold(criteria.accuracy_threshold)

    if ref_epochs <= 0 or tgt_epochs <= 0:
        epoch_multiplier = 2.0
    else:
        epoch_multiplier = tgt_epochs / ref_epochs

    return PrecisionAdjustment(
        accuracy_delta=accuracy_delta,
        loss_delta=loss_delta,
        epoch_multiplier=epoch_multiplier,
    )


# Default precisions to derive adjustments for (keyed by the storage
# dtype name used in PRECISION_ADJUSTMENTS).
_DERIVATION_PRECISIONS: dict[str, NumpyPrecisionSpec] = {
    "float16": NumpyPrecisionSpec.mixed_f16_f32(),
    "float64": NumpyPrecisionSpec.float64(),
    "float8_e4m3fn": NumpyPrecisionSpec.fp8_e4m3(),
}


def derive_all_adjustments(
    problem_modules: list[object] | None = None,
) -> dict[str, PrecisionAdjustment]:
    """Derive precision adjustments across the standard problem x precision matrix.

    For each precision config in ``_DERIVATION_PRECISIONS``, runs
    ``derive_precision_adjustment`` on every problem module and averages
    the deltas.  FP32 is always the identity adjustment.

    Parameters
    ----------
    problem_modules
        List of problem modules.  Defaults to the three standard problems
        (iris, xor, breast_cancer).

    Returns
    -------
    dict[str, PrecisionAdjustment]
        Adjustment table keyed by dtype name, suitable for replacing
        ``PRECISION_ADJUSTMENTS``.
    """
    if problem_modules is None:
        from .problems import breast_cancer, iris, xor
        problem_modules = [iris, xor, breast_cancer]

    result: dict[str, PrecisionAdjustment] = {
        "float32": PrecisionAdjustment(
            accuracy_delta=0.0, loss_delta=0.0, epoch_multiplier=1.0,
        ),
    }

    for key, prec in _DERIVATION_PRECISIONS.items():
        acc_deltas: list[float] = []
        loss_deltas: list[float] = []
        epoch_mults: list[float] = []

        for mod in problem_modules:
            adj = derive_precision_adjustment(
                problem_module=mod,
                target_precision=prec,
            )
            acc_deltas.append(adj.accuracy_delta)
            loss_deltas.append(adj.loss_delta)
            epoch_mults.append(adj.epoch_multiplier)

        result[key] = PrecisionAdjustment(
            accuracy_delta=float(np.mean(acc_deltas)),
            loss_delta=float(np.mean(loss_deltas)),
            epoch_multiplier=float(np.mean(epoch_mults)),
        )

    return result
