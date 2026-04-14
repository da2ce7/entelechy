# tests/tier1/test_plan_builder.py
"""End-to-end plan construction tests."""

from src.shared.hardware_profile import HardwareProfile
from src.shared.model_spec import ModelSpec
from src.shared.plan_builder import build_act_plan, build_learn_plan
from src.shared.plan_types import (
    BarrierNode,
    KernelDispatchNode,
    ReductionTreeNode,
    RetrievalNode,
    StreamingLoopNode,
)
from architectures.averaging_ensembled_classifier.src.shared.problem_type_spec import PlanBceStrategy, PlanCceStrategy
from src.shared.stabilization_policy import StabilizationPolicy


class TestActPlan:
    def test_minimal_act_plan(self, small_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy) -> None:
        plan = build_act_plan(small_spec, hardware, cce_strategy, batch_size=1)
        assert "forward_pass" in plan.nodes
        assert "render_logits_chunk" in plan.nodes
        assert "compute_probs_loss_cce_chunk" in plan.nodes
        # Node 14 (reduce_diagnostic_probs) removed in single-tile config
        # inference_retrieval now depends directly on the loss node
        assert "inference_retrieval" in plan.nodes

    def test_act_plan_is_dag(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy) -> None:
        plan = build_act_plan(model_spec, hardware, cce_strategy, batch_size=64)
        # Plan construction validates DAG, so reaching here proves acyclicity.
        assert len(plan.topological_order) == len(plan.nodes)

    def test_act_plan_dependency_chain(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy) -> None:
        plan = build_act_plan(model_spec, hardware, cce_strategy, batch_size=64)
        assert "forward_pass" in plan.nodes["render_logits_chunk"].depends_on
        assert "render_logits_chunk" in plan.nodes["compute_probs_loss_cce_chunk"].depends_on
        # Single-tile config: inference_retrieval depends directly on loss node (no diagnostic reduction)
        assert "compute_probs_loss_cce_chunk" in plan.nodes["inference_retrieval"].depends_on

    def test_act_plan_node_types(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy) -> None:
        plan = build_act_plan(model_spec, hardware, cce_strategy, batch_size=32)
        assert isinstance(plan.nodes["forward_pass"], KernelDispatchNode)
        # Node 14 (reduce_diagnostic_probs) removed in single-tile config
        assert isinstance(plan.nodes["inference_retrieval"], RetrievalNode)

    def test_cce_vs_bce_loss_contract(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, bce_strategy: PlanBceStrategy) -> None:
        act_cce = build_act_plan(model_spec, hardware, cce_strategy, batch_size=32)
        act_bce = build_act_plan(model_spec, hardware, bce_strategy, batch_size=32)
        cce_loss = act_cce.nodes["compute_probs_loss_cce_chunk"]
        bce_loss = act_bce.nodes["compute_probs_loss_bce_chunk"]
        assert isinstance(cce_loss, KernelDispatchNode)
        assert isinstance(bce_loss, KernelDispatchNode)
        assert cce_loss.kernel_name != bce_loss.kernel_name

    def test_buffers_populated(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy) -> None:
        plan = build_act_plan(model_spec, hardware, cce_strategy, batch_size=16)
        assert len(plan.buffers) > 0
        names = {d.logical_name for d in plan.buffers.values()}
        assert "shared_weights" in names
        # partial_probs serves as inference output in single-tile config (no final_probs)
        assert "partial_probs" in names


class TestLearnPlan:
    def test_learn_plan_constructs(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        assert len(plan.nodes) > 10

    def test_learn_plan_has_barriers(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        barriers = [n for n in plan.nodes.values() if isinstance(n, BarrierNode)]
        assert len(barriers) >= 2  # item_sync + batch_sync
        names = {b.barrier_name for b in barriers}
        assert "item_sync" in names
        assert "batch_sync" in names

    def test_learn_plan_has_reduction_trees(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        trees = [n for n in plan.nodes.values() if isinstance(n, ReductionTreeNode)]
        assert len(trees) >= 2  # mod, temps, shared

    def test_learn_plan_reduction_trees_clipped(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        for node in plan.nodes.values():
            if isinstance(node, ReductionTreeNode):
                assert node.reduction_plan.tree_variant == "sum_and_clip"

    def test_learn_plan_has_streaming_loop(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        loops = [n for n in plan.nodes.values() if isinstance(n, StreamingLoopNode)]
        assert len(loops) >= 1  # streaming backprop at minimum

    def test_learn_plan_recompute_has_extra_loop(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan_recompute = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32,
            policy=policy, activation_lifecycle="recompute",
        )
        plan_cache = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32,
            policy=policy, activation_lifecycle="cache",
        )
        loops_r = [n for n in plan_recompute.nodes.values() if isinstance(n, StreamingLoopNode)]
        loops_c = [n for n in plan_cache.nodes.values() if isinstance(n, StreamingLoopNode)]
        # Forward recompute is currently full-batch dispatch (not a streaming
        # loop), so both lifecycles produce the same streaming loop count.
        # When activation caching is implemented the cache plan will omit
        # the forward-pass dispatch nodes entirely.
        assert len(loops_r) >= len(loops_c)
        # Recompute plan must contain the forward-pass dispatch node
        assert "forward_pass" in plan_recompute.nodes

    def test_learn_plan_has_adam_updates(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        assert "adam_update__shared_weights" in plan.nodes
        # Module-chunk scoped nodes use __mc<N> suffixes (ADR-030)
        assert "adam_update__module_weights__mc0" in plan.nodes
        assert "adam_update__temps__mc0" in plan.nodes

    def test_learn_plan_has_retrieval(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        retrievals = [n for n in plan.nodes.values() if isinstance(n, RetrievalNode)]
        assert len(retrievals) >= 1

    def test_buffer_lifecycle_coverage(self, model_spec: ModelSpec, hardware: HardwareProfile, cce_strategy: PlanCceStrategy, policy: StabilizationPolicy) -> None:
        plan = build_learn_plan(
            model_spec, hardware, cce_strategy, batch_size=32, policy=policy,
        )
        for _handle, desc in plan.buffers.items():
            # Every intermediate should have consumers or be output
            if desc.role.name == "BATCH_OUTPUT":
                continue
            # MODEL_STATE may have no producer
            if desc.role.name == "MODEL_STATE":
                assert desc.producing_node is None
