# tests/tier1/test_reduction_tree_plan.py
"""ReductionTreePlan structure, threshold schedule, and invariants."""
import math
from typing import Literal

import pytest

from src.shared.buffer_lifecycle import BufferHandle
from src.shared.reduction_tree_plan import ReductionTreePlan
from src.shared.stabilization_policy import StabilizationPolicy


def _make_tree(
    num_partials: int, fan_in: int, variant: Literal["sum", "sum_and_clip"] = "sum_and_clip",
    fp_max: float = 3.4028235e+38,
) -> ReductionTreePlan:
    num_stages = max(1, math.ceil(math.log(num_partials) / math.log(fan_in))) if num_partials > 1 else 1
    policy = StabilizationPolicy(t_algorithmic=1.0, lambda_=0.1, compute_fp_format_max=fp_max)
    if variant == "sum_and_clip":
        schedule = tuple(
            policy.get_threshold_for_generic_stage(j, fan_in)
            for j in reversed(range(num_stages))
        )
    else:
        schedule = tuple(None for _ in range(num_stages))
    return ReductionTreePlan(
        num_partials=num_partials, fan_in_K=fan_in, num_stages=num_stages,
        elements_per_partial=1, initial_offset_list=tuple(range(num_partials)),
        tree_variant=variant, threshold_schedule=schedule,
        partial_width=1, source_buffer=BufferHandle(0),
        destination_buffer=BufferHandle(1),
    )


class TestReductionTreeStructure:
    def test_small_tree(self):
        tree = _make_tree(4, 2)
        assert tree.num_stages == 2
        assert len(tree.threshold_schedule) == 2

    def test_large_fan_in(self):
        tree = _make_tree(65536, 256)
        expected = math.ceil(math.log(65536) / math.log(256))
        assert tree.num_stages == expected  # 3

    def test_offset_list_length(self):
        tree = _make_tree(8, 2)
        assert len(tree.initial_offset_list) == 8


class TestThresholdSchedule:
    def test_sum_only_all_none(self):
        tree = _make_tree(4, 2, variant="sum")
        assert all(t is None for t in tree.threshold_schedule)

    def test_schedule_length_matches_stages(self):
        tree = _make_tree(16, 4)
        assert len(tree.threshold_schedule) == tree.num_stages

    def test_threshold_monotonicity_positive_lambda(self):
        """Leaf thresholds >= root thresholds for lambda > 0."""
        tree = _make_tree(64, 4)
        schedule = [t for t in tree.threshold_schedule if t is not None]
        assert len(schedule) > 1
        # leaf→root ordering: first is leaf, last is root
        assert schedule[0] >= schedule[-1]

    def test_fp16_safety_clamping(self):
        """No threshold exceeds FP16 max / K."""
        fp16_max = 65504.0
        tree = _make_tree(16, 4, fp_max=fp16_max)
        safety = fp16_max / tree.fan_in_K
        for t in tree.threshold_schedule:
            if t is not None:
                assert t <= safety + 1e-6


class TestFrozenImmutability:
    def test_frozen(self):
        tree = _make_tree(4, 2)
        with pytest.raises(AttributeError):
            tree.fan_in_K = 99  # type: ignore[misc]
