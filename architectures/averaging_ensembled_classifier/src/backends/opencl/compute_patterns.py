# compute_patterns.py

"""
A Module of Tactical, Reusable Parallel Computing Primitives.

Jurisdictional Mandate:
This module provides the definitive, low-level components for executing the
system's core parallel patterns. Its jurisdiction is strictly tactical; it
provides the "engine room" primitives that are commanded by the high-level,
stateless "recipes" found in `graph_recipes.py`.

Architectural Role:
This module embodies the principle of implementation purity. The former,
stateful `ReductionTreeExecutor` has been dissolved, its logic elevated to a
more appropriate strategic layer. What remains are the pure, stateless tools and
the immutable data contracts they consume. This revision completes the system's
adherence to a strict separation of concerns, ensuring that the "how" of a
parallel reduction is fully decoupled from the "why" of the overall algorithm.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pyopencl as cl

# --- Foundational Imports from Sibling Architectural Modules ---
from .launcher_infra import BufferHandle, BufferManager, KernelExecutor, KernelSignature
from .kernel_bindings import (
    AggregateRegisterReduceSignature,
    AggregateLocalReduceSignature,
)
from .context import DiscoveredArchConstants


@dataclass(frozen=True)
class ReductionPlan:
    """
    A pure configuration primitive that carries the strategic fan-in (`k`)
    for a reduction tree.

    Contractual Role:
    This object is a message, not an actor. It serves as an immutable data
    contract, carrying a single piece of strategic information from the
    high-level `ExecutionPlan` down to the recipes that will execute it.
    """

    k: int


class AggregationManager:
    """
    A stateless, tactical tool that executes a SINGLE stage of a reduction.

    Contractual Role:
    This class is the 'dumb' but powerful workhorse of the reduction hierarchy.
    Its sole and sacred responsibility is to select the correct kernel tier
    (register-based vs. local-memory-based) and dispatch it against a given set
    of partials. It operates exclusively via the "Indirection Contract," using
    an `offset_list` to gather scattered data, thereby upholding the system's
    "Primacy of Memory Strategy" by avoiding intermediate memory copies. It has
    no knowledge of the overall reduction tree.
    """

    def __init__(self, ex: KernelExecutor, bm: BufferManager, arch_consts: DiscoveredArchConstants):
        """Initializes the manager via dependency injection of core system services."""
        self.ex = ex
        self.bm = bm
        self.arch_consts = arch_consts
        # WHY: This is a hardware-informed heuristic. It defines the crossover
        # point where a register-based reduction becomes less efficient than
        # one that utilizes local memory, providing the basis for the tier selection.
        self.max_reg_agg = 16

    def execute_stage(
        self,
        queue: cl.CommandQueue,
        collection_ref: BufferHandle,
        offset_list_ref: BufferHandle,
        num_partials_to_reduce: int,
        elements_per_partial: int,
        destination_ref: BufferHandle,
        wait_for: List[cl.Event],
    ) -> cl.Event:
        """
        Function: Execute a single, atomic stage of a reduction.

        Architectural Mandate:
        This method is the physical execution of the manager's core contract.
        Its primary purpose is to act as an intelligent dispatcher, inspecting
        the workload size (`num_partials_to_reduce`) and selecting the most
        performant kernel tier for that specific task. This embodies the
        principle of using the right tool for the job at the lowest possible level.
        """
        # WHY: This is a defensive check of the contract. A reduction of one is a
        # logical contradiction. The higher-level recipes are responsible for
        # handling this case (as a direct memory copy), ensuring this manager
        # is only ever invoked for a true aggregation task where N > 1.
        if num_partials_to_reduce <= 1:
            raise ValueError("AggregationManager.execute_stage should only be called for N > 1 partials.")

        sig: KernelSignature  # The final, chosen 'artisan' for the job.

        # WHY: This is the core intelligence of the AggregationManager. It selects
        # the optimal physical implementation based on the problem size.
        # - For a small number of partials, a register-based reduction is faster,
        #   avoiding the latency of local memory synchronization.
        # - For a larger number, using local memory is essential to avoid
        #   register pressure and ensure scalability.
        if num_partials_to_reduce <= self.max_reg_agg:
            # WHY: This is the moment of commitment. The manager's job is to
            # assemble the correct, self-sufficient contractual object — the
            # `KernelSignature`. This object possesses all knowledge required
            # to marshal its own arguments and derive its execution grid.
            sig = AggregateRegisterReduceSignature(
                _buffer_mgr=self.bm,
                _arch_consts=self.arch_consts,
                partial_collection_ref=collection_ref,
                partial_offset_list_ref=offset_list_ref,
                dest_ref=destination_ref,
                partial_offset_list_count=np.uint32(num_partials_to_reduce),
                partial_width=np.uint32(elements_per_partial),
                operation_type=np.uint32(0),  # AGG_MODE_SUM
            )
        else:
            sig = AggregateLocalReduceSignature(
                _buffer_mgr=self.bm,
                _arch_consts=self.arch_consts,
                partial_collection_ref=collection_ref,
                partial_offset_list_ref=offset_list_ref,
                dest_ref=destination_ref,
                partial_offset_list_count=np.uint32(num_partials_to_reduce),
                partial_width=np.uint32(elements_per_partial),
                operation_type=np.uint32(0),  # AGG_MODE_SUM
            )

        # WHY: The final act is delegation. The manager passes the fully-formed
        # contract to the pure `KernelExecutor`, which handles the final,
        # non-negotiable act of dispatch. This maintains a perfect separation
        # of concerns.
        return self.ex.launch(queue, sig, wait_for=wait_for)
