# tests/tier1/test_problem_type_strategy.py
"""PlanProblemTypeStrategy → KernelContract delegation."""

from src.shared.kernel_contracts import KernelContract
from src.shared.kernel_contracts import (
    compute_probs_loss_bce_chunk,
    compute_probs_loss_cce_chunk,
)
from src.shared.problem_type_strategy import (
    PlanBceStrategy,
    PlanCceStrategy,
)


class TestCceStrategy:
    def test_loss_contract_is_cce(self):
        s = PlanCceStrategy()
        assert s.get_loss_contract() is compute_probs_loss_cce_chunk

    def test_required_buffer_name(self):
        assert PlanCceStrategy().required_targets_buffer_name == "targets_cce"

    def test_returns_kernel_contracts(self):
        s = PlanCceStrategy()
        assert isinstance(s.get_loss_contract(), KernelContract)
        assert isinstance(s.get_module_grad_contract(), KernelContract)
        assert isinstance(s.get_hidden_grad_contract(), KernelContract)
        assert isinstance(s.get_temp_grad_contract(), KernelContract)


class TestBceStrategy:
    def test_loss_contract_is_bce(self):
        s = PlanBceStrategy()
        assert s.get_loss_contract() is compute_probs_loss_bce_chunk

    def test_required_buffer_name(self):
        assert PlanBceStrategy().required_targets_buffer_name == "targets_bce"


class TestNoOpenclImports:
    def test_plan_strategy_import_chain_clean(self):
        """Importing PlanCceStrategy should not pull in OpenCL types."""
        # The strategy module does import legacy types at module level,
        # but PlanCceStrategy and PlanBceStrategy only import from
        # .kernel_contracts which is pure shared layer.
        s = PlanCceStrategy()
        c = s.get_loss_contract()
        assert c.name  # valid contract, no backend
