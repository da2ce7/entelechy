"""
The Public API for the Averaging Ensembled Classifier Architecture.

This file defines the package's public API surface. It exports only the
high-level types required by an external client to configure and run the
architecture via the plan-model dispatch path.

Phase 6: All legacy shim re-exports removed. The plan-model infrastructure
(shared layer + backend renderers) is the sole execution mechanism.
"""

# --- Orchestration ---
from .main_orchestrator import TrainingOrchestrator, TrainingHyperparams, StabilizationConfig

# --- Model Configuration ---
from .shared.model_spec import ModelSpec
from .shared.parameter_space import ParameterSpace
from .shared.precision_config import PrecisionConfig
from .shared.hardware_profile import HardwareProfile
from .shared.stabilization_policy import StabilizationPolicy

# --- Plan-Model Primitives ---
from .shared.plan_builder import build_act_plan, build_learn_plan
from .shared.plan_renderer import PlanRenderer
from .shared.plan_types import ExecutionPlan


__all__ = [
    # Orchestration
    "TrainingOrchestrator",
    "TrainingHyperparams",
    "StabilizationConfig",
    # Configuration
    "ModelSpec",
    "ParameterSpace",
    "PrecisionConfig",
    "HardwareProfile",
    "StabilizationPolicy",
    # Plan Model
    "build_act_plan",
    "build_learn_plan",
    "PlanRenderer",
    "ExecutionPlan",
]