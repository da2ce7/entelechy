# tests/tier1/test_kernel_contracts.py
"""Contract field completeness and validation."""
import pytest

from src.shared.kernel_contracts import KernelContract, BufferParam, ScalarParam, KERNEL_REGISTRY


ALL_CONTRACTS: list[tuple[str, KernelContract]] = list(KERNEL_REGISTRY.items())
CANONICAL_STRATEGIES = {
    "grid_mod_cls",
    "grid_mod_cls_batch",
    "linear_batch",
    "linear_generic",
}


class TestRegistryPopulated:
    def test_registry_nonempty(self):
        assert len(KERNEL_REGISTRY) > 0

    def test_registry_keys_match_names(self):
        for key, contract in KERNEL_REGISTRY.items():
            assert key == contract.kernel_name


class TestContractFields:
    @pytest.mark.parametrize(
        "name,contract",
        ALL_CONTRACTS,
        ids=[n for n, _ in ALL_CONTRACTS],
    )
    def test_has_buffers(self, name: str, contract: KernelContract) -> None:
        bufs = [p for p in contract.params if isinstance(p, BufferParam)]
        assert len(bufs) > 0, f"{name} has no buffer params"

    @pytest.mark.parametrize(
        "name,contract",
        ALL_CONTRACTS,
        ids=[n for n, _ in ALL_CONTRACTS],
    )
    def test_name_nonempty(self, name: str, contract: KernelContract) -> None:
        assert contract.kernel_name

    @pytest.mark.parametrize(
        "name,contract",
        ALL_CONTRACTS,
        ids=[n for n, _ in ALL_CONTRACTS],
    )
    def test_idempotency_valid(self, name: str, contract: KernelContract) -> None:
        assert contract.idempotency in {
            "Strictly Idempotent",
            "Associatively Non-Idempotent",
            "Fundamentally Non-Idempotent (Stateful)",
        }

    @pytest.mark.parametrize(
        "name,contract",
        ALL_CONTRACTS,
        ids=[n for n, _ in ALL_CONTRACTS],
    )
    def test_placement_strategy_canonical(self, name: str, contract: KernelContract) -> None:
        for p in contract.params:
            if isinstance(p, BufferParam) and p.placement is not None:
                assert p.placement.strategy in CANONICAL_STRATEGIES, (
                    f"{name}.{p.name} has non-canonical placement "
                    f"{p.placement.strategy}"
                )

    @pytest.mark.parametrize(
        "name,contract",
        ALL_CONTRACTS,
        ids=[n for n, _ in ALL_CONTRACTS],
    )
    def test_params_are_typed(self, name: str, contract: KernelContract) -> None:
        for p in contract.params:
            assert isinstance(p, (BufferParam, ScalarParam)), (
                f"{name}: param {p!r} is not BufferParam or ScalarParam"
            )
