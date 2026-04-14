# src/shared/stabilization_policy.py
"""Gradient stabilization policy (CONCEPT.md §3.3–3.4).

Implements the Quadratic Scaling Policy and orthogonal numerical safety
enforcement for the log_K(N) reduction engine.

Two-step threshold formula (per stage)
──────────────────────────────────────
::

    1. policy_threshold  = T_algorithmic + λ · j²
    2. final_threshold   = min(policy_threshold, compute_fp_format_max / K)

where *j* is the stage index (0 = root, higher values = earlier/leaf
stages) and *K* is the uniform reduction fan-in.

K resolution (CONCEPT.md §2)
─────────────────────────────
::

    K = min(K_hw, K_policy)

When ``policy_reduce_fan_in`` is ``None`` (the default), K_hw is
binding.  K is further capped at ``num_partials`` and clamped to ≥ 2
for any actual reduction.

Node 16 specialisation
──────────────────────
The Grad_H reduction kernel (Node 16) uses a work-group-per-row
strategy with optional pre-accumulation.  ``render_node16_schedule``
produces a backend-adapted schedule accounting for the dispatch
topology (CONCEPT.md §3.4, §11 item 4).

Authoritative sources
─────────────────────
CONCEPT.md §2    — Reduction Batch Size resolution
CONCEPT.md §3.3  — Quadratic Scaling Policy
CONCEPT.md §3.4  — Orthogonal numerical safety enforcement
CONCEPT.md §11   — Node 16 schedule rendering
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Node16Schedule",
    "ReductionTopology",
    "StabilizationPolicy",
]


# ═════════════════════════════════════════════════════════════════════
# Result types
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReductionTopology:
    """Uniform reduction tree geometry returned by
    :meth:`StabilizationPolicy.plan_reduction_tree`.

    Parameters
    ----------
    fan_in:
        K — partials consumed per output node per stage.
        1 when ``num_stages == 0`` (Identity tier bypass).
    num_stages:
        ⌈log_K(N)⌉.  0 signals the Identity tier
        (single partial, no reduction dispatch).
    """

    fan_in: int
    num_stages: int


@dataclass(frozen=True)
class Node16Schedule:
    """Rendered threshold schedule for the specialised Grad_H
    reduction kernel (Node 16).

    Produced by :meth:`StabilizationPolicy.render_node16_schedule`.

    Parameters
    ----------
    num_stages:
        Number of staged reduction rounds after pre-accumulation.
        0 when post-accumulation intermediates ≤ 1.
    pre_accumulation_threshold:
        Clip threshold for each thread's serial pre-accumulation
        result.  ``compute_fp_format_max`` when no pre-accumulation
        occurs (total_modules ≤ workgroup_size).
    stage_thresholds:
        Per-stage clip thresholds.  Index 0 = first clip after
        pre-accumulation (leaf); last index = final clip (root).
        Length equals ``num_stages``.  Empty when
        ``num_stages == 0``.
    """

    num_stages: int
    pre_accumulation_threshold: float
    stage_thresholds: tuple[float, ...]


# ═════════════════════════════════════════════════════════════════════
# StabilizationPolicy
# ═════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StabilizationPolicy:
    """Immutable gradient stabilization policy.

    Encodes the user's algorithmic intent (``t_algorithmic``, ``λ``)
    and the hardware's arithmetic ceiling (``compute_fp_format_max``)
    into a unified configuration that produces per-stage clipping
    thresholds for the reduction engine.

    Parameters
    ----------
    t_algorithmic:
        Target L2 norm at the root of the reduction tree (j = 0).
        The Quadratic Scaling Policy's anchor point.  When 0,
        every stage clips to zero — effectively suppressing all
        gradients.
    lambda_:
        Funnel curvature coefficient.  Larger values produce more
        permissive early-stage (leaf) thresholds.  When 0, the
        policy degrades to a fixed ceiling at ``t_algorithmic``
        for all stages (CONCEPT.md §3.3).
    compute_fp_format_max:
        Maximum finite value of the compute-precision format
        (``PrecisionConfig.compute_fp_format_max``).  Source of
        the orthogonal safety ceiling ``cfm / K`` that prevents
        summation-induced overflow.
    policy_reduce_fan_in:
        K_policy — user-configured stabilisation granularity
        (CONCEPT.md §2).  Lower values produce more intermediate
        clip stages and finer-grained gradient direction
        preservation.  ``None`` (default) makes the hardware
        ceiling K_hw binding.
    """

    t_algorithmic: float
    lambda_: float
    compute_fp_format_max: float
    policy_reduce_fan_in: int | None = None

    def __post_init__(self) -> None:
        if self.t_algorithmic < 0:
            raise ValueError(
                f"t_algorithmic must be ≥ 0, got {self.t_algorithmic}"
            )
        if self.lambda_ < 0:
            raise ValueError(
                f"lambda_ must be ≥ 0, got {self.lambda_}"
            )
        if self.compute_fp_format_max <= 0:
            raise ValueError(
                f"compute_fp_format_max must be > 0, "
                f"got {self.compute_fp_format_max}"
            )
        if (
            self.policy_reduce_fan_in is not None
            and self.policy_reduce_fan_in < 2
        ):
            raise ValueError(
                f"policy_reduce_fan_in must be ≥ 2 or None, "
                f"got {self.policy_reduce_fan_in}"
            )

    # ── K resolution (CONCEPT.md §2) ─────────────────────────────

    def resolve_fan_in(self, hardware_max_fan_in: int) -> int:
        """Resolve the Reduction Batch Size K.

        ``K = min(K_hw, K_policy)``, clamped to ≥ 2.  When
        ``policy_reduce_fan_in`` is ``None``, ``K = K_hw``.

        This method does **not** cap K at ``num_partials`` — that
        is the caller's responsibility (handled by
        :meth:`plan_reduction_tree`).
        """
        k_policy = (
            self.policy_reduce_fan_in
            if self.policy_reduce_fan_in is not None
            else hardware_max_fan_in
        )
        return max(2, min(hardware_max_fan_in, k_policy))

    # ── Tree planning ─────────────────────────────────────────────

    def plan_reduction_tree(
        self,
        num_partials: int,
        hardware_max_fan_in: int,
    ) -> ReductionTopology:
        """Plan a uniform log_K(N) reduction tree.

        Returns the fan-in K and stage count for *num_partials*
        input partials.

        When ``num_partials ≤ 1``, returns the Identity tier
        (``fan_in=1``, ``num_stages=0``): no reduction dispatch,
        the backend references the single partial directly.

        K is ``min(resolve_fan_in(hardware_max_fan_in), num_partials)``
        — never exceeding the number of partials available.
        """
        if num_partials <= 1:
            return ReductionTopology(fan_in=1, num_stages=0)

        k = min(self.resolve_fan_in(hardware_max_fan_in), num_partials)
        k = max(2, k)  # defense-in-depth; resolve_fan_in guarantees ≥ 2
        num_stages = math.ceil(math.log(num_partials) / math.log(k))
        return ReductionTopology(fan_in=k, num_stages=num_stages)

    # ── Per-stage thresholds ──────────────────────────────────────

    def stage_threshold(self, stage_j: int, fan_in: int) -> float:
        """Clipping threshold for reduction stage *j*.

        Implements the canonical two-step formula (CONCEPT.md §3.3–3.4)::

            policy = T_algorithmic + λ · j²
            safety = compute_fp_format_max / K
            result = min(policy, safety)

        Parameters
        ----------
        stage_j:
            Stage index relative to root.  ``j = 0`` is the root
            (most restrictive); higher values are earlier stages
            (leaves, more permissive).
        fan_in:
            Reduction fan-in K at this stage.
        """
        policy = self.t_algorithmic + self.lambda_ * (stage_j ** 2)
        safety = self.compute_fp_format_max / max(1, fan_in)
        return min(policy, safety)

    def render_schedule(
        self,
        num_stages: int,
        fan_in: int,
    ) -> tuple[float, ...]:
        """Threshold schedule for all stages of a reduction tree.

        Returns a tuple of length *num_stages*.  Index 0 is the leaf
        stage (``j = num_stages − 1``, most permissive); the last
        index is the root (``j = 0``, most restrictive).

        This ordering matches ``ReductionTreePlan`` convention:
        "Index 0 is the leaf stage; index ``num_stages - 1`` is
        the root."
        """
        return tuple(
            self.stage_threshold(num_stages - 1 - s, fan_in)
            for s in range(num_stages)
        )

    # ── Safety ceilings (CONCEPT.md §3.4) ─────────────────────────

    def safety_ceiling(self, divisor: int) -> float:
        """``compute_fp_format_max / divisor``.

        The orthogonal safety bound preventing summation-induced
        overflow.  Used for pre-reduction leaf clipping (Nodes 11,
        19) where *divisor* encodes the downstream amplification
        factor or first-stage fan-in.

        Examples::

            # Node 11: downstream Node 13 sums num_class_chunks tiles
            t_11 = policy.safety_ceiling(num_class_chunks)

            # Node 19: downstream Node 20 first stage sums K partials
            t_19 = policy.safety_ceiling(K_first_stage)
        """
        return self.compute_fp_format_max / max(1, divisor)

    def leaf_safety_ceiling(
        self,
        num_partials: int,
        hardware_max_fan_in: int,
        *,
        amplification: int = 1,
    ) -> float:
        """Safety ceiling for pre-reduction leaf clipping.

        Computes ``compute_fp_format_max / (K × A)`` where K is the
        first-stage fan-in of the downstream reduction tree and A is
        the pre-summation amplification factor.

        When ``num_partials ≤ 1`` (Identity tier bypass), K = 1.

        Parameters
        ----------
        num_partials:
            Number of partials entering the downstream reduction tree.
        hardware_max_fan_in:
            ``HardwareProfile.max_reduce_fan_in``.
        amplification:
            Pre-summation amplification factor.  E.g.
            ``num_class_chunks`` for the Node 11 → Node 13 path
            where Node 13 sums across class chunks before Node 16.
            Defaults to 1 (no upstream summation).
        """
        topo = self.plan_reduction_tree(num_partials, hardware_max_fan_in)
        return self.safety_ceiling(max(1, topo.fan_in) * max(1, amplification))

    # ── Node 16 specialisation ────────────────────────────────────

    def render_node16_schedule(
        self,
        total_modules: int,
        workgroup_size: int,
        hardware_max_fan_in: int,
    ) -> Node16Schedule:
        """Render the threshold schedule for the Grad_H reducer.

        Adapts the Quadratic Scaling Policy to the backend's dispatch
        topology, accounting for GPU work-group-limited
        pre-accumulation (CONCEPT.md §3.4, §11 item 4).

        The staged reduction tree operates over
        ``P = min(total_modules, workgroup_size)`` post-accumulation
        intermediates — not over ``total_modules`` directly.

        Pre-accumulation threshold derivation (CONCEPT.md §3.4)::

            When total_modules > workgroup_size, each thread serially
            accumulates ⌈M/W⌉ elements.  The first clip stage then
            sums K of these accumulators.  For non-overflow:
                K × T_pre ≤ cfm  →  T_pre = cfm / K

            When total_modules ≤ workgroup_size, each thread loads
            at most one element (no pre-accumulation):
                T_pre = cfm

        Parameters
        ----------
        total_modules:
            M — number of modules to reduce across.
        workgroup_size:
            W — dispatch workgroup size.  For the CPU backend,
            pass ``total_modules`` (no pre-accumulation).  For
            GPU backends, pass the actual workgroup size.
        hardware_max_fan_in:
            ``HardwareProfile.max_reduce_fan_in``.
        """
        cfm = self.compute_fp_format_max
        p = min(total_modules, workgroup_size)

        if p <= 1:
            return Node16Schedule(
                num_stages=0,
                pre_accumulation_threshold=cfm,
                stage_thresholds=(),
            )

        k = min(self.resolve_fan_in(hardware_max_fan_in), p)
        k = max(2, k)
        num_stages = math.ceil(math.log(p) / math.log(k))

        t_pre = cfm / k if total_modules > p else cfm

        schedule = self.render_schedule(num_stages, k)

        return Node16Schedule(
            num_stages=num_stages,
            pre_accumulation_threshold=t_pre,
            stage_thresholds=schedule,
        )
