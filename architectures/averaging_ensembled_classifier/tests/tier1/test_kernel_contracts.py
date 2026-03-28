# tests/tier1/test_kernel_contracts.py
"""Contract field completeness and validation."""
import importlib

import pytest

from src.shared.kernel_contracts import KernelContract


PHASE_MODULES = [
    "src.shared.kernel_contracts.phase_1_act",
    "src.shared.kernel_contracts.phase_2_learn_A_production",
    "src.shared.kernel_contracts.phase_2_learn_B_processing",
    "src.shared.kernel_contracts.phase_2_learn_C_reduction",
    "src.shared.kernel_contracts.phase_2_learn_D_backprop",
    "src.shared.kernel_contracts.phase_3_update",
]


def _collect_contracts() -> list[tuple[str, str, KernelContract]]:
    """Collect all KernelContract instances from per-phase modules."""
    contracts: list[tuple[str, str, KernelContract]] = []
    for mod_name in PHASE_MODULES:
        mod = importlib.import_module(mod_name)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, KernelContract):
                contracts.append((mod_name, attr_name, obj))
    return contracts


ALL_CONTRACTS = _collect_contracts()
CANONICAL_STRATEGIES = {"grid_mod_cls", "linear_batch", "linear_generic"}


class TestAllContractsPopulated:
    def test_at_least_one_per_phase(self):
        for mod_name in PHASE_MODULES:
            phase_contracts = [c for m, _, c in ALL_CONTRACTS if m == mod_name]
            assert len(phase_contracts) >= 1, f"No contracts in {mod_name}"


class TestContractFields:
    @pytest.mark.parametrize(
        "mod,name,contract",
        ALL_CONTRACTS,
        ids=[f"{n}" for _, n, _ in ALL_CONTRACTS],
    )
    def test_has_buffer_params(self, mod: str, name: str, contract: KernelContract) -> None:
        assert len(contract.buffer_params) > 0, f"{name} has no buffer_params"

    @pytest.mark.parametrize(
        "mod,name,contract",
        ALL_CONTRACTS,
        ids=[f"{n}" for _, n, _ in ALL_CONTRACTS],
    )
    def test_kernel_name_nonempty(self, mod: str, name: str, contract: KernelContract) -> None:
        assert contract.kernel_name

    @pytest.mark.parametrize(
        "mod,name,contract",
        ALL_CONTRACTS,
        ids=[f"{n}" for _, n, _ in ALL_CONTRACTS],
    )
    def test_placement_strategy_canonical(self, mod: str, name: str, contract: KernelContract) -> None:
        if contract.placement is not None:
            assert contract.placement.strategy in CANONICAL_STRATEGIES, (
                f"{name} has non-canonical placement {contract.placement.strategy}"
            )


class TestCalculabilityProofClosure:
    @pytest.mark.parametrize(
        "mod,name,contract",
        ALL_CONTRACTS,
        ids=[f"{n}" for _, n, _ in ALL_CONTRACTS],
    )
    def test_proof_references_exist(self, mod: str, name: str, contract: KernelContract) -> None:
        """All parameter names in calculability_proof exist within the contract."""
        all_param_names: set[str] = set()
        for bp in contract.buffer_params:
            all_param_names.add(bp.name)
        for sp in contract.scalar_params:
            all_param_names.add(sp.name)
        for bp in contract.buffer_params:
            for ref in bp.calculability_proof:
                assert ref in all_param_names, (
                    f"{name}: calculability_proof ref {ref!r} not found "
                    f"in param names {all_param_names}"
                )
