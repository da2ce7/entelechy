# kernel_signatures/__init__.py

"""
The Public API for the Contractual Kernel Launch Layer.

This package provides the collection of all concrete `KernelSignature` classes,
each one a direct, executable embodiment of a kernel contract defined in the
`kernels.cl.h` file.

This `__init__.py` file serves as the sole entry point to the package,
exporting all available signature classes into a single, unified namespace.
This a) decouples the high-level `HostOrchestrator` from the internal, modular
file structure of this package, and b) explicitly codifies the public API
of this layer.

Architectural Rectification Note:
As part of the iterative design process, general-purpose utility kernel
signatures (`TransposeChunkSignature`, `IdentityCopySignature`) have been
relocated to a dedicated `utility_signatures.py` module. This file imports
them and presents them as part of the unified API, maintaining a clean
separation of concerns between domain-specific and general-purpose contracts.
"""

# --- Phase 1: Act (Forward Pass & Loss) ---
from .phase_1_act import (
    ForwardPassSignature,
    RenderLogitsChunkSignature,
    ComputeProbsLossCceChunkSignature,
    ComputeProbsLossBceChunkSignature,
)

# --- Phase 2, Set A: Learn (Initial Gradient Production) ---
from .phase_2_learn_A_production import (
    CalculateModuleParamGradsCceSignature,
    CalculateModuleParamGradsBceSignature,
    BackpropErrorToHiddenChunkCceSignature,
    BackpropErrorToHiddenChunkBceSignature,
    CalculateChunkTempGradientsCceSignature,
    CalculateChunkTempGradientsBceSignature,
)

# --- Phase 2, Set B: Learn (Gradient Processing & Permutation) ---
from .phase_2_learn_B_processing import (
    ClipPartialGradientsGlobalNormSignature,
    ClipPartialGradientsPerItemNormSignature,
    GradientHandles,
    GatherAndPermuteGradHiddenActivationsSignature,
)

# --- Phase 2, Set C: Learn (Aggregation & Reduction) ---
from .phase_2_learn_C_reduction import (
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
    ClipIntermediateGradSignature,
    StabilizeAndReduceGradHiddenActivationsSignature,
)

# --- Phase 2, Set D: Learn (Shared Layer Backpropagation) ---
from .phase_2_learn_D_backprop import (
    BackpropSharedWeightsChunkSignature,
    BackpropSharedBiasesChunkSignature,
    SharedGradientHandles,
    ClipSharedGradientsChunkSignature,
)

# --- Phase 3: Update (Normalization & Finalization) ---
from .phase_3_update import (
    NormalizeGradientsSignature,
    AdamParameterGroup,
    AdamUpdateSignature,
    ClampTemperaturesSignature,
)

# --- Phase 2B: New KernelBinding adapters (coexist with legacy Signatures) ---
from .base import KernelBinding
from .dispatch_table import build_dispatch_table

# --- Public API Contract (`__all__`) ---
# This list explicitly defines all symbols that are considered part of the
# public, stable API of this package. Consumers should only rely on these names.
__all__ = [
    # Phase 1: Act
    "ForwardPassSignature",
    "RenderLogitsChunkSignature",
    "ComputeProbsLossCceChunkSignature",
    "ComputeProbsLossBceChunkSignature",
    # Phase 2A: Gradient Production
    "CalculateModuleParamGradsCceSignature",
    "CalculateModuleParamGradsBceSignature",
    "BackpropErrorToHiddenChunkCceSignature",
    "BackpropErrorToHiddenChunkBceSignature",
    "CalculateChunkTempGradientsCceSignature",
    "CalculateChunkTempGradientsBceSignature",
    # Phase 2B: Gradient Processing
    "ClipPartialGradientsGlobalNormSignature",
    "ClipPartialGradientsPerItemNormSignature",
    "GradientHandles",
    "GatherAndPermuteGradHiddenActivationsSignature",
    # Phase 2C: Reduction
    "AggregateRegisterReduceSignature",
    "AggregateLocalReduceSignature",
    "ClipIntermediateGradSignature",
    "StabilizeAndReduceGradHiddenActivationsSignature",
    # Phase 2D: Shared Backprop
    "BackpropSharedWeightsChunkSignature",
    "BackpropSharedBiasesChunkSignature",
    "SharedGradientHandles",
    "ClipSharedGradientsChunkSignature",
    # Phase 3: Update
    "NormalizeGradientsSignature",
    "AdamParameterGroup",
    "AdamUpdateSignature",
    "ClampTemperaturesSignature",
    # New KernelBinding adapters (Phase 2B)
    "KernelBinding",
    "build_dispatch_table",
]
