# stabilization_policy.py

"""
This module constitutes the sole and authoritative host-side implementation of
the System's Dynamic Gradient Stabilization Strategy.

Jurisdictional Mandate:
This module's jurisdiction is to translate the high-level stabilization policy
(defined by user intent and hardware constraints) into the low-level, primitive
scalar values required by the device-side kernel contracts. It is the definitive,
verifiable bridge between the host's strategic orchestration and the device's
tactical execution.

Architectural Role:
This module provides the `StabilizationPolicy` object, a formal architectural
entity. This object is the host-side computational engine that implements the
stabilization logic specified for Nodes 15, 16, and 20 in the Architectural
Concept and Kernel Contracts. Its existence enables the separation of concerns,
ensuring the Host Orchestrator is decoupled from the nuanced mathematics of
stabilization.

Adherence to the interfaces provided herein is mandatory for any host-side
logic that governs gradient reduction.
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StabilizationPolicy:
    """
    An immutable policy object that embodies the contract between user intent
    and hardware reality for gradient stabilization.

    Contractual Role:
    This class serves as an immutable configuration manifest and a provider of
    pure, stateless calculation methods. It is instantiated by the Host
    Orchestrator with the high-level policy parameters and provides the
    necessary functions to compute the exact, primitive values required for
    kernel dispatch, thereby fulfilling the system's stabilization contract.
    """

    # --- Algorithmic & System Parameters ---

    # WHY: This parameter represents the user's declared algorithmic objective.
    # It is the final, target L2 norm to which the fully aggregated gradient
    # at the root of the reduction tree (j=0) must adhere.
    t_algorithmic: float

    # WHY: This parameter governs the curvature of the stabilization policy
    # funnel. It dictates the rate at which clipping thresholds are relaxed
    # for earlier, more granular stages of the reduction tree.
    lambda_: float

    # WHY: This parameter represents a non-negotiable physical system boundary.
    # The maximum representable value of the compute precision format. Governs
    # overflow safety during reduction tree summation. Under the three-role model,
    # the safety ceiling is bounded by arithmetic precision, not storage precision.
    # At FP64 compute precision, the safety ceiling (~1.8e+308 / K_j) makes
    # overflow a non-practical concern for any realistic fan-in K. The
    # stabilization machinery remains active (it costs nothing when not
    # triggered) but will never fire under FP64 compute.
    compute_fp_format_max: float

    # WHY: This parameter is a mandatory system-level safeguard. It establishes
    # a floor for all computed thresholds, preventing signal annihilation that
    # could occur if the policy (e.g., a very small t_algorithmic) otherwise
    # produced a pathologically small threshold.
    min_threshold: float = 1.0

    # =========================================================================
    # === API for Specialized Reduction Kernels (Consumed by Node 16)       ===
    # =========================================================================

    def get_specialized_reduction_policy_k(self, user_policy_k: int, hardware_max_fan_in: int) -> int:
        """
        Function: Pre-flight Contract Synthesizer for Specialized Reducers.

        Architectural Mandate:
        The kernel contract for `stabilize_and_reduce_grad_hidden_activations`
        (Node 16) mandates that the host shall provide a single, pre-sanitized
        integer, `src_scalar_NATURAL_policy_max_k`. This method is the sole,
        authoritative implementation of the synthesis required to produce that
        value.

        It resolves three distinct constraints into a single, primitive integer
        that the device kernel can accept with absolute trust:
          A. User Intent: The user's desired K for a specific policy.
          B. Hardware Reality: The physical work-group limits of the device.
          C. Mathematical Safety: The absolute fan-in limit required to
             prevent signal annihilation.
        """
        # Constraint C: Mathematical Safety.
        # WHY: This calculates the absolute maximum fan-in (K) permitted by the
        # system's fundamental `min_threshold`. It guarantees that the hardware
        # safety ceiling (`fp_format_max / K`) can never fall below this
        # non-negotiable floor, thus preventing signal annihilation.
        math_safety_k_limit = self.compute_fp_format_max / self.min_threshold if self.min_threshold > 0 else float("inf")

        # Synthesis and Finalization.
        # WHY: The final value is contractually obligated to be the most
        # restrictive (minimum) of the three constraints. This safely encodes
        # all high-level policy into a primitive value suitable for dispatch.
        final_k = min(user_policy_k, hardware_max_fan_in, math_safety_k_limit)

        # WHY: A fan-in of less than 2 is mathematically nonsensical for a reduction.
        # This enforces a sane lower bound on the final return value.
        return max(2, int(final_k))

    # =========================================================================
    # === API for Generic Reduction Trees (Consumed by Nodes 15 & 20)       ===
    # =========================================================================

    def plan_uniform_reduction_tree(self, num_partials: int, hardware_max_fan_in: int) -> Tuple[int, int]:
        """
        Function: Strategic Reduction Planner.

        Architectural Mandate:
        This method devises a uniform, globally-valid reduction plan for a
        host-driven reduction tree. It resolves the paradox of needing `num_stages`
        to find the safest `K`, and needing `K` to find `num_stages`. It does so
        by first calculating a candidate plan, then finding the mathematical fan-in
        limit (`math_limit`) for the most restrictive stage (the leaves), and
        finally synthesizing a final, safe `K` that is valid for the entire tree.
        """
        if num_partials <= 1:
            return (1, 0)

        # Stage 1: Candidate Plan Formulation.
        # WHY: A candidate `K` is determined by hardware limits to derive a
        # plausible number of stages, `num_stages`, which is required to
        # identify the most mathematically restrictive point in the plan.
        k_candidate = max(2, min(hardware_max_fan_in, num_partials))
        num_stages = math.ceil(math.log(num_partials) / math.log(k_candidate))

        # Stage 2: Worst-Case Constraint Analysis.
        # WHY: The leaf stage (`j = num_stages - 1`) is the most restrictive due
        # to the `j^2` policy term. We must calculate the theoretical maximum
        # fan-in that this single stage can tolerate.
        leaf_stage_j = num_stages - 1
        math_limit = self._get_max_fan_in_for_stage_math_only(leaf_stage_j)

        # Stage 3: Synthesis of Final, Globally-Valid Plan.
        # WHY: The final, safe `K` must respect both the hardware limits and
        # the single most restrictive mathematical constraint of the entire tree.
        safe_k = max(2, int(min(k_candidate, math_limit)))

        # WHY: With the definitive `safe_k`, the final number of stages is
        # recalculated for the Host Orchestrator to execute.
        if safe_k <= 1:
            num_stages = num_partials - 1 if num_partials > 1 else 0
        else:
            num_stages = math.ceil(math.log(num_partials) / math.log(safe_k))
        return (safe_k, num_stages)

    def get_threshold_for_generic_stage(self, stage_j: int, runtime_fan_in_k: int) -> float:
        """
        Function: Tactical Threshold Synthesizer.

        Architectural Mandate:
        This is the tactical, runtime counterpart to the strategic planner.
        Its purpose is to compute the precise, final scalar threshold value for
        a *specific* stage `j` of a generic reduction tree, providing the
        `src_scalar_REAL_clipping_threshold_t_j` value required by the
        `clip_intermediate_grad` kernel at that point in the DAG.
        """
        return self._get_final_threshold(stage_j, runtime_fan_in_k)

    # =========================================================================
    # === API for Leaf-Level Clipping (Consumed by Nodes 11 & 19)           ===
    # =========================================================================

    def get_leaf_safety_threshold(self) -> float:
        """
        Function: Leaf-Level Safety Gate.

        Architectural Mandate:
        The clipping performed at the leaf level (Nodes 11 & 19) serves a
        distinct architectural purpose from the main reduction engine. It is a
        coarse, pre-emptive safety primitive, not a policy-driven shaping tool.
        Its sole function is to prevent pathologically large raw gradients from
        entering the reduction pipeline. Therefore, this function provides a
        simple, high, and robust safety threshold that is independent of the
        quadratic policy.

        Pre-summation Amplification (CONCEPT §3.4):
        Callers must account for any downstream summation that occurs before
        the values enter a reduction kernel. For Node 11, the clipped gradients
        pass through Node 13, which sums across `num_class_chunks`. The caller
        should divide this threshold by `num_class_chunks` to ensure the
        post-summation magnitude stays within safe bounds for Node 16.
        """
        # WHY: A 10% safety margin below the absolute hardware maximum provides
        # a robust buffer against unforeseen floating-point edge cases without
        # being overly restrictive for a coarse safety gate.
        return self.compute_fp_format_max * 0.9

    # =========================================================================
    # === Internal Calculation Primitives                                   ===
    # =========================================================================

    def _get_max_fan_in_for_stage_math_only(self, stage_j: int) -> float:
        """Calculates the theoretical upper bound on fan-in K for a given stage."""
        # Calculate the policy-defined threshold for stage j.
        policy_threshold = self.t_algorithmic + self.lambda_ * (stage_j**2)

        # Enforce the system-wide minimum threshold to produce a floor.
        effective_floor = max(policy_threshold, self.min_threshold)

        if effective_floor <= 0:
            return float("inf")

        # Calculate the K-limit imposed by the absolute minimum threshold.
        absolute_k_max = self.compute_fp_format_max / self.min_threshold if self.min_threshold > 0 else float("inf")
        # Calculate the K-limit imposed by the policy at this stage.
        policy_k_max = self.compute_fp_format_max / effective_floor

        # The true limit is the more restrictive of the two.
        return min(absolute_k_max, policy_k_max)

    def _get_final_threshold(self, stage_j: int, runtime_fan_in_k: int) -> float:
        """Synthesizes policy and safety limits into a final, authoritative threshold."""
        # Defensively validate the runtime K against the stage's math limit.
        math_limit = self._get_max_fan_in_for_stage_math_only(stage_j)
        validated_k = max(2, min(runtime_fan_in_k, math_limit))

        # Calculate the canonical hardware safety ceiling for this fan-in.
        safety_ceiling = self.compute_fp_format_max / validated_k

        # For safety-only policies, the hardware ceiling is the only constraint.
        if self.t_algorithmic <= 0:
            return safety_ceiling

        # Recalculate the policy threshold and enforce the minimum floor.
        policy_threshold = self.t_algorithmic + self.lambda_ * (stage_j**2)
        floored_policy = max(policy_threshold, self.min_threshold)

        # The final, authoritative value respects the policy, but is bound by safety.
        # This is the canonical clamp that embodies the system's core principle.
        return min(floored_policy, safety_ceiling)
