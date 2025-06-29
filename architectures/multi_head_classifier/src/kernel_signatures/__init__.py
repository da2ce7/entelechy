# kernel_signatures/__init__.py

"""
The Public API for the Contractual Kernel Launch Layer.

This package provides the collection of all concrete `KernelSignature` classes,
each one a direct, executable embodiment of a kernel contract defined in the
`kernels.cl.h` file.

This `__init__.py` file serves as the sole entry point to the package,
exporting all available signature classes into a single, unified namespace.
This decouples the high-level `HostOrchestrator` from the internal, modular
file structure of this package, ensuring a clean and stable interface.

The `__all__` variable explicitly defines the public contract of this package,
listing every `KernelSignature` that a consumer is permitted to import and use.
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
    TransposeChunkSignature,
    GatherAndPermuteGradHSignature,
)

# --- Phase 2, Set C: Learn (Aggregation & Reduction) ---
from .phase_2_learn_C_reduction import (
    AggregateIdentitySignature,
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
    ReduceGradHOverModulesSignature,
)

# --- Phase 2, Set D: Learn (Shared Layer Backpropagation) ---
from .phase_2_learn_D_backprop import (
    BackpropSharedWeightsChunkSignature,
    BackpropSharedBiasesChunkSignature,
)

# --- Phase 3: Update (Normalization & Finalization) ---
from .phase_3_update import (
    NormalizeGradientsSignature,
    AdamUpdateSignature,
    AdamParameterGroup,
    ClampTemperaturesSignature,
)

# --- Public API Contract (`__all__`) ---
# This list explicitly defines all symbols that are considered part of the
# public, stable API of this package. Consumers should only rely on these names.
__all__ = [
    # Phase 1
    "ForwardPassSignature",
    "RenderLogitsChunkSignature",
    "ComputeProbsLossCceChunkSignature",
    "ComputeProbsLossBceChunkSignature",
    # Phase 2A
    "CalculateModuleParamGradsCceSignature",
    "CalculateModuleParamGradsBceSignature",
    "BackpropErrorToHiddenChunkCceSignature",
    "BackpropErrorToHiddenChunkBceSignature",
    "CalculateChunkTempGradientsCceSignature",
    "CalculateChunkTempGradientsBceSignature",
    # Phase 2B
    "ClipPartialGradientsGlobalNormSignature",
    "ClipPartialGradientsPerItemNormSignature",
    "GradientHandles",
    "TransposeChunkSignature",
    "GatherAndPermuteGradHSignature",
    # Phase 2C
    "AggregateIdentitySignature",
    "AggregateRegisterReduceSignature",
    "AggregateLocalReduceSignature",
    "ReduceGradHOverModulesSignature",
    # Phase 2D
    "BackpropSharedWeightsChunkSignature",
    "BackpropSharedBiasesChunkSignature",
    # Phase 3
    "NormalizeGradientsSignature",
    "AdamUpdateSignature",
    "AdamParameterGroup",
    "ClampTemperaturesSignature",
]
# kernel_signatures/__init__.py

"""
The Public API for the Contractual Kernel Launch Layer.

This package provides the collection of all concrete `KernelSignature` classes,
each one a direct, executable embodiment of a kernel contract defined in the
`kernels.cl.h` file.

This `__init__.py` file serves as the sole entry point to the package,
exporting all available signature classes into a single, unified namespace.
This decouples the high-level `HostOrchestrator` from the internal, modular
file structure of this package, ensuring a clean and stable interface.
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
# CORRECTED: TransposeChunkSignature has been moved to utility_kernels.
from .phase_2_learn_B_processing import (
    ClipPartialGradientsGlobalNormSignature,
    ClipPartialGradientsPerItemNormSignature,
    GradientHandles,
    GatherAndPermuteGradHSignature,
)

# --- Phase 2, Set C: Learn (Aggregation & Reduction) ---
from .phase_2_learn_C_reduction import (
    AggregateIdentitySignature,
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
    ReduceGradHOverModulesSignature,
)

# --- Phase 2, Set D: Learn (Shared Layer Backpropagation) ---
from .phase_2_learn_D_backprop import (
    BackpropSharedWeightsChunkSignature,
    BackpropSharedBiasesChunkSignature,
)

# --- Phase 3: Update (Normalization & Finalization) ---
from .phase_3_update import (
    NormalizeGradientsSignature,
    AdamUpdateSignature,
    AdamParameterGroup,
    ClampTemperaturesSignature,
)

# --- NEW: General Purpose Utility Kernels ---
from .utility_kernels import (
    TransposeChunkSignature,
)


# --- Public API Contract (`__all__`) ---
# This list explicitly defines all symbols that are considered part of the
# public, stable API of this package. The contents remain the same, but the
# source of `TransposeChunkSignature` is now correctly resolved.
__all__ = [
    # Phase 1
    "ForwardPassSignature",
    "RenderLogitsChunkSignature",
    "ComputeProbsLossCceChunkSignature",
    "ComputeProbsLossBceChunkSignature",
    # Phase 2A
    "CalculateModuleParamGradsCceSignature",
    "CalculateModuleParamGradsBceSignature",
    "BackpropErrorToHiddenChunkCceSignature",
    "BackpropErrorToHiddenChunkBceSignature",
    "CalculateChunkTempGradientsCceSignature",
    "CalculateChunkTempGradientsBceSignature",
    # Phase 2B
    "ClipPartialGradientsGlobalNormSignature",
    "ClipPartialGradientsPerItemNormSignature",
    "GradientHandles",
    "GatherAndPermuteGradHSignature",
    # Phase 2C
    "AggregateIdentitySignature",
    "AggregateRegisterReduceSignature",
    "AggregateLocalReduceSignature",
    "ReduceGradHOverModulesSignature",
    # Phase 2D
    "BackpropSharedWeightsChunkSignature",
    "BackpropSharedBiasesChunkSignature",
    # Phase 3
    "NormalizeGradientsSignature",
    "AdamUpdateSignature",
    "AdamParameterGroup",
    "ClampTemperaturesSignature",
    # Utilities
    "TransposeChunkSignature",
]
