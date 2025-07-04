"""
The Public API for the Averaging Ensembled Classifier Architecture.

Architectural Mandate:
This file is the canonical entry point for the `averaging_ensembled_classifier`
Python package. Its presence marks the `src` directory as a package, enabling
robust, explicit relative imports for all internal modules.

Its primary purpose is to define the package's public Application Programming
Interface (API). It exports only the high-level classes required by an external
client (such as a test suite or a user-facing script) to configure and run the
architecture.

Internal implementation details (e.g., `BatchProcessor`, `graph_recipes`,
`launcher_infra`, `kernel_signatures`) are intentionally omitted from this
public contract, enforcing a strict separation of concerns and a stable
interface for the consumer.
"""

# --- High-Level Orchestration & Configuration Primitives ---
# These are the primary "nouns" a user interacts with.

# The main entry point for running the architecture.
from .main_orchestrator import TrainingOrchestrator, TrainingHyperparams, StabilizationConfig

# The primitive that defines the model's static, logical shape.
from .model_spec import ModelSpec

# The manifest that defines the model's learnable parameter space.
from .parameter_space import ParameterSpace

# The object that implements the system's gradient stabilization policy.
from .stabilization_policy import StabilizationPolicy


# --- Public API Contract (`__all__`) ---
# This list explicitly defines all symbols that are considered part of the
# public, stable API of this package. Consumers should only rely on these names.
__all__ = [
    # Core Orchestration
    "TrainingOrchestrator",
    # Configuration & Specification Objects
    "ModelSpec",
    "ParameterSpace",
    "TrainingHyperparams",
    "StabilizationConfig",
    "StabilizationPolicy",
]