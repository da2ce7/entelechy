# tests/tier1/test_plan_types.py
"""Node type construction, validation, and ExecutionPlan invariants."""
import pytest

from src.shared.buffer_lifecycle import BufferDescriptor, BufferHandle, BufferRole
from src.shared.hardware_profile import HardwareProfile
from src.shared.kernel_contracts import forward_pass
from src.shared.plan_types import (
    BarrierNode,
    ExecutionPlan,
    KernelDispatchNode,
    PlanNode,
    PlanValidationError,
    RetrievalNode,
    StreamingLoopNode,
)
from src.shared.precision_config import PrecisionConfig
from src.shared.streaming_loop_plan import (
    IterationDimension,
    StreamingLoopPlan,
)


def _buf(handle_id: int, name: str, role: BufferRole = BufferRole.BATCH_INTERMEDIATE):
    h = BufferHandle(handle_id)
    return h, BufferDescriptor(
        handle=h, logical_name=name, padded_shape=(1,),
        element_size_bytes=4, size_bytes=4, role=role,
        precision_role="compute",
        producing_node=None, consumers=frozenset(), last_consumer=None,
    )


class TestNodeConstruction:
    def test_kernel_dispatch_frozen(self):
        h, _ = _buf(0, "b")
        node = KernelDispatchNode(
            node_id="n1", depends_on=frozenset(), kernel_name="test",
            contract=forward_pass, buffer_bindings={"b": h},
            scalar_params={"x": 1}, tile_count=1, local_work_size=None,
            placement_strategy=None,
        )
        assert node.node_id == "n1"
        with pytest.raises(AttributeError):
            node.node_id = "changed"  # type: ignore[misc]

    def test_barrier_node(self):
        node = BarrierNode(node_id="b1", depends_on=frozenset({"a"}), barrier_name="sync")
        assert node.barrier_name == "sync"

    def test_retrieval_node(self):
        h, _ = _buf(0, "out")
        node = RetrievalNode(
            node_id="r1", depends_on=frozenset(), source_buffer=h,
            logical_shape=(10,), event_name="ev",
        )
        assert node.logical_shape == (10,)


class TestExecutionPlanValidation:
    def _make_plan(self, nodes: dict[str, PlanNode], buffers: dict[BufferHandle, BufferDescriptor], topo: tuple[str, ...]) -> ExecutionPlan:
        return ExecutionPlan(
            nodes=nodes, buffers=buffers, topological_order=topo,
            precision=PrecisionConfig.float32(),
            hardware=HardwareProfile(
                simd_width=16, cache_line_bytes=64,
                max_reduce_fan_in=256, max_local_mem_bytes=65536,
                global_mem_bytes=4 * 1024**3,
            ),
        )

    def test_valid_simple_plan(self):
        h, bd = _buf(0, "out", BufferRole.BATCH_OUTPUT)
        n1 = KernelDispatchNode(
            node_id="a", depends_on=frozenset(), kernel_name="k",
            contract=forward_pass, buffer_bindings={"out": h},
            scalar_params={}, tile_count=1, local_work_size=None,
            placement_strategy=None,
        )
        n2 = RetrievalNode(
            node_id="b", depends_on=frozenset({"a"}), source_buffer=h,
            logical_shape=(1,), event_name="ev",
        )
        plan = self._make_plan(
            {"a": n1, "b": n2}, {h: bd}, ("a", "b"),
        )
        assert len(plan.nodes) == 2

    def test_missing_dependency_raises(self):
        h, bd = _buf(0, "out")
        n1 = KernelDispatchNode(
            node_id="a", depends_on=frozenset({"nonexistent"}),
            kernel_name="k", contract=forward_pass,
            buffer_bindings={"out": h}, scalar_params={}, tile_count=1,
            local_work_size=None, placement_strategy=None,
        )
        with pytest.raises(PlanValidationError, match="nonexistent"):
            self._make_plan({"a": n1}, {h: bd}, ("a",))

    def test_cycle_detected(self):
        h, bd = _buf(0, "out")
        n1 = BarrierNode(node_id="a", depends_on=frozenset({"b"}), barrier_name="s1")
        n2 = BarrierNode(node_id="b", depends_on=frozenset({"a"}), barrier_name="s2")
        with pytest.raises(PlanValidationError, match="cycle"):
            self._make_plan({"a": n1, "b": n2}, {h: bd}, ("a", "b"))

    def test_topo_order_inconsistency(self):
        h, bd = _buf(0, "out")
        n1 = BarrierNode(node_id="a", depends_on=frozenset(), barrier_name="s1")
        n2 = BarrierNode(node_id="b", depends_on=frozenset({"a"}), barrier_name="s2")
        with pytest.raises(PlanValidationError, match="inconsistency"):
            self._make_plan({"a": n1, "b": n2}, {h: bd}, ("b", "a"))

    def test_missing_buffer_raises(self):
        h_missing = BufferHandle(99)
        n1 = KernelDispatchNode(
            node_id="a", depends_on=frozenset(), kernel_name="k",
            contract=forward_pass,
            buffer_bindings={"out": h_missing}, scalar_params={},
            tile_count=1, local_work_size=None, placement_strategy=None,
        )
        with pytest.raises(PlanValidationError, match="BufferHandle"):
            self._make_plan({"a": n1}, {}, ("a",))

    def test_streaming_body_must_be_dispatch_nodes(self):
        h, bd = _buf(0, "b")
        barrier = BarrierNode(node_id="inner", depends_on=frozenset(), barrier_name="s")
        loop_plan = StreamingLoopPlan(
            iteration=IterationDimension(total_extent=1, chunk_count=1, chunk_size=1),
            body=("inner",),
            parameter_strides=(),
            scratch_buffers=(),
            constant_scalars={},
        )
        outer = StreamingLoopNode(
            node_id="loop", depends_on=frozenset(), streaming_plan=loop_plan,
        )
        with pytest.raises(PlanValidationError, match="KernelDispatchNode"):
            self._make_plan(
                {"inner": barrier, "loop": outer}, {h: bd}, ("inner", "loop"),
            )
