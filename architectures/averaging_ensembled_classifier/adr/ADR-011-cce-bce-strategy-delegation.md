# ADR-011: CCE/BCE Strategy Delegation

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002, ADR-007  
**Blocks:** —

---

## Context

CONCEPT.md §3 (Modular, "Dumb" Kernels) explicitly permits two strategies for expressing CCE/BCE divergence — the fundamental branching point where single-label (Categorical Cross-Entropy) and multi-label (Binary Cross-Entropy) classification paths diverge in the computational DAG:

> - **(A) Single Kernel with Host-Injected Flag:** A unified kernel uses a `FLAG__` scalar to toggle paths, provided the divergence is manageable.
> - **(B) Separate Kernels:** Distinct kernels are expected when divergence is complex or imposes conflicting memory patterns.

The current implementation uses **both strategies simultaneously** across different kernel categories, and the plan model established by ADR-001, ADR-002, and ADR-007 requires a formal position on how this divergence is expressed in the backend-neutral execution plan.

### The current implementation state

A survey of the kernel inventory reveals a principled split between Strategy A and Strategy B usage:

| Kernel (Node) | Strategy | Mechanism | Rationale |
| :--- | :--- | :--- | :--- |
| `compute_probs_loss_cce_chunk` (6) | **B — Separate** | Distinct kernel name | Incompatible type signatures (`int*` vs. `SCALAR_TYPE*` targets), different output topologies (scatter-write vs. partial-render for loss), different math (Softmax vs. Sigmoid) |
| `compute_probs_loss_bce_chunk` (7) | **B — Separate** | Distinct kernel name | (Same — counterpart to Node 6) |
| `calculate_module_param_grads_chunk` (8) | **A — Flag** | `src_scalar_FLAG_problem_type` | Branch is a 4-line `d_loss_d_logit` divergence; shared memory layout, shared reduction, shared placement |
| `backprop_error_to_hidden_chunk` (9) | **A — Flag** | `src_scalar_FLAG_problem_type` | Identical structure and justification to Node 8 |
| `calculate_chunk_temp_gradients` (10) | **A — Flag** | `src_scalar_FLAG_problem_type` | Identical structure and justification to Node 8 |

The host-side Python code in `execution_plan.py` mirrors this through the `ProblemTypeStrategy` hierarchy: `CceStrategy` and `BceStrategy` each produce distinct `KernelSignature` subclasses per node — e.g., `CalculateModuleParamGradsCceSignature` vs. `CalculateModuleParamGradsBceSignature`. These signature variants set the `problem_type` flag and bind the correct targets buffer type.

### Why a formal position is needed

ADR-001 establishes that the shared orchestration layer produces a backend-neutral execution plan. ADR-002 defines the `KernelDispatchNode` as the fundamental unit of work in this plan. ADR-007 defines the `KernelContract` as the validation data carried by each node, including `kernel_name` and `scalar_params`. The plan must express CCE/BCE divergence using these established primitives — but the question of *how* it does so has three distinct concerns:

1. **Plan-level expression.** How does the plan represent the choice between CCE and BCE for a given module? Through different `kernel_name` values (Strategy B)? Through a `ScalarParamSpec` carrying a `problem_type` flag (Strategy A)? Or a combination?

2. **Backend rendering.** How does each backend translate the plan-level expression into native dispatch? OpenCL currently passes `PROBLEM_TYPE_CCE = 0` or `PROBLEM_TYPE_BCE = 1` as a positional scalar. Vulkan would use specialization constants or pipeline variants. CPU would use separate C function pointers or a branch on the flag.

3. **Host orchestration.** How does the `ProblemTypeStrategy` hierarchy map onto the plan builder? The current `get_loss_signature()` / `get_module_grad_signature()` / etc. factory methods produce strategy-specific signatures that must be translated into plan nodes.

Without a formal resolution, these three concerns risk ad-hoc coupling between the plan vocabulary and backend-specific dispatch mechanisms.

### The structural divergence between Node 6/7 and Nodes 8/9/10

The reason the current implementation uses different strategies for different kernels is not arbitrary — it reflects a genuine architectural divergence:

**Nodes 6 and 7 (loss computation) require Strategy B** because:

- **Type-incompatible interfaces.** CCE targets are `int*` (class indices); BCE targets are `SCALAR_TYPE*` (per-class floats). The `src_buffer_GLOBAL_targets` parameter has fundamentally different tensor shapes, padding contracts, and calculability proofs between the two kernels. A single `KernelContract` cannot express both without type-punning the buffer specification — violating CONTRACT.md Article 1.4 (Collaborative Interface Verifiability), which requires the contract to be a closed logical system for validation.
- **Topologically distinct DAG outputs.** CCE loss is a scalar per (module, sample) pair, written via direct scatter-write to `dest_buffer_GLOBAL_final_loss` — no reduction required. BCE loss is a partial sum per (module, sample, class_chunk), written to `dest_buffer_GLOBAL_partial_loss` — requiring subsequent aggregation through the Recursive Clip-Aggregation Engine (Node 14). This changes the downstream DAG structure: the BCE path has a reduction sub-tree that the CCE path lacks entirely.
- **Different math.** Softmax + log-loss (CCE) vs. Sigmoid + binary log-loss (BCE). The internal computation structure is sufficiently different that a flag-based branch would create a complex, non-"dumb" kernel.

**Nodes 8, 9, and 10 (gradient production) use Strategy A** because:

- **Type-compatible interfaces via punned pointer.** Both strategies consume the `src_buffer_GLOBAL_targets` as a `void*` pointer, cast internally based on the `src_scalar_FLAG_problem_type` flag. The remaining buffer parameters, tensor shapes, padding contracts, and placement strategies are identical between CCE and BCE.
- **Identical DAG topology.** Both strategies produce identically-shaped partial gradient outputs through the same placement contract. The downstream reduction trees (Nodes 14, 15, 20) are structurally identical regardless of problem type.
- **Minimal branch.** The CCE/BCE divergence is confined to a 4-line `d_loss_d_logit` calculation — the rest of each kernel (coordinate mapping, batch reduction, local memory management, placement-governed write) is shared.

### The `void*` type-punning question

The use of `void*` for `src_buffer_GLOBAL_targets` in Nodes 8, 9, and 10 creates a tension with CONTRACT.md Article 1.4 (Collaborative Interface Verifiability). The contract for these kernels cannot express a single, closed calculability proof for the targets buffer — its tensor shape depends on the problem type flag. The current kernel headers acknowledge this explicitly:

> `Tensor Shape: Varies based on problem type flag.`
> `Calculability Proof: Dependent on problem type flag.`

This is a documented exception, not a silent violation. The host's `KernelSignature` hierarchy resolves this by producing strategy-specific signature classes (`CalculateModuleParamGradsCceSignature`, `CalculateModuleParamGradsBceSignature`) that each validate the targets buffer against its known type. The type-punning occurs at the device interface, but the host-side validation is type-safe.

Under the plan model, the `KernelContract`'s `BufferParamSpec` for the targets parameter must either:
- Carry a conditional tensor shape (violating the "closed logical system" property), or
- Be paired with a `ScalarParamSpec` for `problem_type` that the plan-time validator uses to select the correct shape expectation.

This ADR formalizes the latter approach.

---

## Decision Drivers

1. **CONCEPT.md §3 (Modular, "Dumb" Kernels).** Both Strategy A and Strategy B are explicitly permitted. The choice is governed by the nature of the divergence — complex or type-incompatible divergence mandates Strategy B; manageable branching permits Strategy A. The system must not arbitrarily impose one strategy where the other is architecturally appropriate.

2. **ADR-001 (Backend Abstraction Boundary).** The plan conveys intent; the renderer conveys mechanism. The plan-level expression of CCE/BCE divergence must be backend-neutral. The mechanism by which each backend implements the divergence — runtime flag, specialization constant, separate function pointer — is a binding concern.

3. **ADR-002 (`KernelDispatchNode`).** Each node carries a `kernel_name`, `contract: KernelContract`, `scalar_params`, and `depends_on` edges. The CCE/BCE divergence must be expressible through these existing fields without extending the node taxonomy.

4. **ADR-007 (`KernelContract` / `KernelBinding` split).** The contract carries backend-neutral validation data; the binding translates to native dispatch. The `problem_type` flag is an abstract scalar parameter in the contract; its delivery mechanism (positional arg, specialization constant, function selection) is a binding concern.

5. **CONTRACT.md Article 1.4 (Collaborative Interface Verifiability).** Validation preconditions and calculability proofs must be expressible within the contract's closed parameter manifest. For Strategy B kernels, each variant has its own self-consistent contract. For Strategy A kernels, the flag parameter gates the interpretation of conditional specifications.

6. **CONCEPT.md §1 (Architectural Elegance Feedback).** If a future backend discovers that Strategy A kernels (Nodes 8/9/10) would benefit from being split into separate kernel variants — e.g., because a Vulkan backend achieves better pipeline specialization with separate SPIR-V modules — the response is to introduce the split as a formalized binding-level concern, not to retroactively split the plan-level kernel identity. Conversely, if a future optimization discovers a way to unify Nodes 6 and 7 into a single kernel with a flag, the same process applies: formalize first, implement second.

---

## Options Considered

### Option A: Normalize to Strategy B throughout — separate `kernel_name` for all CCE/BCE variants

Require all CCE/BCE divergent kernels to be expressed as separate `kernel_name` values in the plan. Nodes 8, 9, and 10 would each split into two plan-level identities: e.g., `calculate_module_param_grads_chunk_cce` and `calculate_module_param_grads_chunk_bce`.

**Advantages:**
- Every `KernelContract` has a fully self-consistent parameter manifest. No conditional tensor shapes, no flag-gated calculability proofs. Article 1.4's "closed logical system" is satisfied unconditionally.
- The plan validator can check buffer shapes without consulting a problem type flag.
- Backend renderers have maximum freedom — Vulkan can use separate pipeline variants, CPU can use separate function pointers, OpenCL can keep or discard the unified kernel as it wishes.

**Disadvantages:**
- **Contradicts the current kernel implementation.** Nodes 8, 9, and 10 are each a single kernel function in `kernels.cl.h` and their `.cl.c` implementations, unified by a `FLAG_problem_type` branch. Splitting them at the plan level would either: (a) require maintaining duplicate `.cl.c` files with 4 lines of difference, violating DRY; or (b) require the OpenCL binding to map two plan-level names (`_cce`, `_bce`) back to the same physical kernel with different flag values — reintroducing Strategy A at the binding level and gaining nothing.
- **Misrepresents the kernel's nature.** CONCEPT.md §3 explicitly states Strategy A is valid "provided the divergence is manageable." The 4-line branch in Nodes 8/9/10 is the paradigmatic case of manageable divergence. Forcing Strategy B here violates the architectural principle that the *nature of the divergence* should determine the strategy, not a desire for plan-level uniformity.
- **Doubles the plan's kernel inventory for these nodes** without semantic benefit. The plan would contain twice as many `KernelDispatchNode` definitions for Nodes 8/9/10, with identical buffer shapes (except the conditional targets tensor), identical placement strategies, identical tile counts, and identical dependency structures.

### Option B: Normalize to Strategy A throughout — single `kernel_name` with flag for all CCE/BCE variants

Express all CCE/BCE divergence through a single `kernel_name` per node with a `src_scalar_FLAG_problem_type` parameter. Nodes 6 and 7 would be unified into a single `compute_probs_loss_chunk` kernel.

**Advantages:**
- Uniform plan-level representation. Every CCE/BCE divergent node uses the same mechanism.
- Reduces the plan's kernel inventory.

**Disadvantages:**
- **Violates CONCEPT.md §3 Strategy B mandate for Nodes 6/7.** The divergence between CCE and BCE loss computation is not "manageable" — it involves incompatible type signatures (`int*` vs. `SCALAR_TYPE*`), incompatible output topologies (scatter-write vs. partial-render), incompatible DAG structures (no reduction vs. reduction tree), and fundamentally different mathematics (Softmax vs. Sigmoid). A unified kernel would be precisely the kind of "smart," complex-branching kernel that CONCEPT.md §3 prohibits.
- **Violates CONTRACT.md Article 1.4.** A single `KernelContract` for a unified loss kernel cannot express closed calculability proofs for both target buffer types simultaneously. The targets buffer's tensor shape, element type, and padding contract are all flag-dependent.
- **Topological impossibility.** The CCE and BCE paths produce structurally different DAG sub-trees. CCE's loss output (`dest_buffer_GLOBAL_final_loss`) is a final result with no downstream reduction. BCE's loss output (`dest_buffer_GLOBAL_partial_loss`) feeds into Node 14's reduction tree. A single kernel cannot produce both output topologies — the plan would need conditional downstream edges, which violates ADR-002's typed DAG model.

### Option C: Mixed strategy — Strategy B where divergence is structural, Strategy A where divergence is parametric

Preserve the current split: separate `kernel_name` values for Nodes 6/7 (where divergence is structural), `src_scalar_FLAG_problem_type` for Nodes 8/9/10 (where divergence is parametric). The plan builder uses the `ProblemTypeStrategy` hierarchy to select the correct kernel names and flag values at plan-construction time.

**Advantages:**
- **Faithful to the actual computational structure.** The plan representation mirrors the genuine architectural divergence: Nodes 6/7 are structurally different kernels with different interfaces, outputs, and downstream effects; Nodes 8/9/10 are the same kernel with a parametric branch.
- **Honors CONCEPT.md §3.** Strategy B is used exactly where the principle mandates it (complex/incompatible divergence) and Strategy A is used exactly where the principle permits it (manageable divergence). The decision is driven by the nature of the computation, not by a desire for representational uniformity.
- **Minimal plan vocabulary impact.** Uses only existing ADR-002 and ADR-007 primitives: `KernelDispatchNode.contract.kernel_name` distinguishes Strategy B variants; `KernelContract.scalar_params` with `ScalarParamSpec(problem_type, FLAG)` expresses Strategy A. No new node types, no new contract fields, no plan vocabulary extensions.
- **Backend rendering flexibility preserved.** For Strategy B kernels (Nodes 6/7): each backend compiles/dispatches the named kernel variant natively. For Strategy A kernels (Nodes 8/9/10): OpenCL passes the flag as a positional scalar; Vulkan can use a specialization constant or push constant; CPU can branch on the flag in a single C function or use separate function pointers. The binding (ADR-007) makes this choice invisible to the plan.
- **Type-safe host-side validation.** The `ProblemTypeStrategy` hierarchy continues to produce strategy-specific `KernelSignature` classes that validate buffer types before dispatch. Under the plan model, the plan builder constructs strategy-specific `KernelContract` instances for Nodes 6/7 (each with its own closed parameter manifest) and a shared `KernelContract` with a FLAG parameter for Nodes 8/9/10 (with the flag-gated conditional validation documented in the contract).

**Disadvantages:**
- **Conditional validation for Nodes 8/9/10.** The `KernelContract` for these nodes cannot express a single, unconditional calculability proof for the targets buffer. The plan-time validator must interpret the `problem_type` flag to select the correct tensor shape expectation. This is a minor complexity cost — the flag is a well-defined `{0, 1}` domain — but it means these contracts are not fully "closed" in the Article 1.4 sense without the flag's value.
- **Mixed representation.** A reader examining the plan must understand that CCE/BCE divergence is expressed through two different mechanisms depending on the kernel category. This requires documentation (provided by this ADR) to prevent confusion.

---

## Analysis

### Eliminating Option A

Option A imposes representational uniformity at the cost of architectural fidelity. Splitting Nodes 8/9/10 into separate plan-level identities misrepresents their computational reality — they are a single kernel with a parametric branch, not two fundamentally different algorithms. The split creates phantom complexity: two `KernelContract` definitions per node that differ only in a targets buffer type annotation and a missing flag parameter, while the underlying kernel code, dispatch grid, tile decomposition, placement strategy, and reduction tree structure remain identical.

Furthermore, the OpenCL binding would need to map both plan-level names back to the same physical kernel function, reconstructing the flag internally. This is the same mechanism as Strategy A but hidden inside the binding — an extra layer of indirection with no validation benefit.

### Eliminating Option B

Option B is eliminated on three independent grounds:

1. **Topological impossibility.** CCE and BCE loss paths produce structurally different DAG sub-trees. A single plan node cannot conditionally introduce or omit downstream reduction nodes — the DAG structure must be statically determined at plan-construction time (ADR-002).

2. **Contract impossibility.** A single `KernelContract` for a unified loss kernel would require the targets buffer's element type to be flag-dependent (`int` for CCE, `SCALAR_TYPE` for BCE). CONTRACT.md Article 1.4 requires that calculability proofs reference parameters in the contract's closed manifest — a proof that says "if FLAG=0, shape is (B); if FLAG=1, shape is (B, padded_C)" is not a closed system.

3. **Principle violation.** CONCEPT.md §3 explicitly reserves Strategy B for "complex or conflicting memory patterns." The CCE/BCE loss divergence is the canonical example cited by the principle itself.

### Choosing Option C

Option C is the only option that respects both the architectural principles (CONCEPT.md §3's strategy criteria) and the formal constraints (CONTRACT.md Article 1.4's verifiability requirements, ADR-002's static DAG structure). It uses the right tool for each situation:

- **Strategy B where the divergence is structural** — different interfaces, different output topologies, different downstream DAG structure. Each variant gets its own `KernelContract` with a self-consistent, closed parameter manifest.

- **Strategy A where the divergence is parametric** — same interface (modulo a type-punned buffer), same output shapes, same downstream DAG structure, same placement strategy. A single `KernelContract` with a FLAG parameter expresses the divergence within the existing ADR-007 vocabulary.

The conditional validation concern (Nodes 8/9/10's targets buffer) is addressed through the `ProblemTypeStrategy` hierarchy, which constructs separate `KernelSignature` classes per strategy — each performing type-safe validation before constructing the `KernelContract`. The flag's value is known at plan-construction time (it is a property of the module's Operating Mode, not a runtime variable), so the plan builder validates the correct tensor shape unconditionally for the chosen strategy.

---

## Decision

**Option C: Mixed strategy — Strategy B for structural divergence, Strategy A for parametric divergence.**

The plan model expresses CCE/BCE divergence through two complementary mechanisms, both already defined by ADR-002 and ADR-007:

### Strategy B expression (Nodes 6 and 7 — loss computation)

The `KernelDispatchNode` for the loss computation step carries a `kernel_name` that distinguishes the CCE and BCE variants:

- **CCE module:** `kernel_name = "compute_probs_loss_cce_chunk"`. The contract specifies `src_buffer_GLOBAL_targets` with tensor shape `(total_batch_count)` and element type `int`. The loss output `dest_buffer_GLOBAL_final_loss` has placement semantics of a direct scatter-write — no downstream reduction node for loss.

- **BCE module:** `kernel_name = "compute_probs_loss_bce_chunk"`. The contract specifies `src_buffer_GLOBAL_targets` with tensor shape `(total_batch_count, padded_total_output_class_count)` and element type `SCALAR_TYPE`. The loss output `dest_buffer_GLOBAL_partial_loss` is governed by a `PlacementContract` and requires downstream aggregation via the Recursive Clip-Aggregation Engine (Node 14).

Each variant has its own `KernelContract` with a fully self-consistent, closed parameter manifest. Plan-time validation verifies each contract independently. The plan builder (via `ProblemTypeStrategy.get_loss_signature()`) selects the correct variant at plan-construction time based on the module's Operating Mode.

**DAG impact.** The CCE and BCE paths produce structurally different plan sub-trees between Nodes 6/7 and Node 14:
- CCE: Node 6 → (probabilities feed downstream; loss is final) → Node 14 aggregates only probabilities.
- BCE: Node 7 → (probabilities and partial loss feed downstream) → Node 14 aggregates both probabilities and partial loss.

This structural difference is expressed naturally in the plan's dependency edges — the BCE plan contains additional `ReductionTreeNode` inputs to Node 14 for the loss path, while the CCE plan does not. No conditional logic in the plan representation is required; the plan builder simply constructs the appropriate sub-tree for the chosen strategy.

### Strategy A expression (Nodes 8, 9, and 10 — gradient production)

The `KernelDispatchNode` for each gradient production kernel carries:

- `kernel_name`: The single kernel identity (e.g., `"calculate_module_param_grads_chunk"`).
- `scalar_params`: Includes a `ScalarParamSpec` with `param_name = "src_scalar_FLAG_problem_type"`, `number_type = "FLAG"`, and value `0` (CCE) or `1` (BCE).

The `KernelContract` for these kernels documents the conditional targets buffer specification:

```
BufferParamSpec(
    param_name="src_buffer_GLOBAL_targets",
    flow="src",
    memory_scope="GLOBAL",
    tensor_shape=("CONDITIONAL: see problem_type flag",),
    padding_contract={"Type": "CONDITIONAL", "Note": "Depends on problem_type flag"},
    calculability_proof=("src_scalar_FLAG_problem_type",),
    validation_preconditions=(
        "Host is contractually obligated to provide the correct target buffer "
        "whose layout, type, and total size correspond to the value of "
        "src_scalar_FLAG_problem_type. CCE (0): int* with shape "
        "(total_batch_count). BCE (1): SCALAR_TYPE* with shape "
        "(total_batch_count, padded_total_output_class_count).",
    ),
)
```

Plan-time validation for these nodes operates in two stages:
1. The strategy-specific `KernelSignature` class validates the targets buffer against its known type (closed validation within the strategy).
2. The `KernelContract` records the conditional specification as documentation for backend binding implementors.

This approach preserves the contract's role as a verification artifact while honestly documenting the flag-dependent semantics that exist in the actual kernel interface.

### Backend rendering responsibilities

Each backend's `KernelBinding` (ADR-007) translates both mechanisms:

| Backend | Strategy B (Nodes 6/7) | Strategy A (Nodes 8/9/10) |
| :--- | :--- | :--- |
| **OpenCL** | Dispatch the named kernel. Different kernel objects for `compute_probs_loss_cce_chunk` and `compute_probs_loss_bce_chunk`. | Pass `src_scalar_FLAG_problem_type` as a positional scalar argument (current approach). |
| **Vulkan** | Select the pre-compiled pipeline variant by `kernel_name`. Different SPIR-V modules for CCE and BCE loss. | Deliver the flag as a specialization constant at pipeline creation time, or as a push constant at dispatch time. The renderer may alternatively pre-compile two pipeline variants and select by flag value — this is a binding-internal optimization. |
| **CPU** | Dispatch to separate C function implementations: `compute_probs_loss_cce_chunk_cpu()` vs. `compute_probs_loss_bce_chunk_cpu()`. | Pass the flag as a C function parameter. The CPU binding may alternatively dispatch to separate functions (`calculate_module_param_grads_chunk_cce_cpu` / `_bce_cpu`) as a binding-internal optimization — the plan does not prescribe this. |

In all cases, the plan-level representation is identical across backends. The rendering differences are confined to the `KernelBinding` implementations, which are backend-internal per ADR-007.

### Host orchestration mapping

The existing `ProblemTypeStrategy` hierarchy (`CceStrategy`, `BceStrategy`) maps naturally onto the plan builder:

1. **At plan-construction time**, the plan builder queries `ExecutionPlan.problem_type` (a `ProblemTypeStrategy` instance) for each module.
2. **For Nodes 6/7**, `strategy.get_loss_signature()` produces the strategy-specific signature, which yields the strategy-specific `KernelContract` with the correct `kernel_name` and fully typed targets buffer specification.
3. **For Nodes 8/9/10**, `strategy.get_module_grad_signature()` (and equivalents) produces the strategy-specific signature, which yields a `KernelContract` with the shared `kernel_name` and the `FLAG_problem_type` scalar set to the appropriate value.
4. **For the DAG structure**, the plan builder conditionally includes the BCE loss reduction sub-tree in Node 14's inputs based on the strategy — `CceStrategy` omits it, `BceStrategy` includes it.

This mapping requires no changes to the `ProblemTypeStrategy` interface. The existing factory methods already produce the correct signatures; the plan builder simply extracts `KernelContract` data from them.

---

## Consequences

### Positive

- **Architecturally faithful.** The plan representation mirrors the genuine computational structure: structural divergence is expressed through kernel identity (Strategy B), parametric divergence through scalar parameters (Strategy A). Neither case is forced into the other's idiom.

- **No plan vocabulary extensions.** CCE/BCE divergence is fully expressible through existing ADR-002 node types (`KernelDispatchNode`, `ReductionTreeNode`) and existing ADR-007 contract fields (`kernel_name`, `ScalarParamSpec`). No new node types, no new contract fields, no new plan primitives.

- **Backend rendering freedom.** Each backend can implement Strategy A in its native idiom — runtime flag, specialization constant, or separate function dispatch — without plan-level impact. Strategy B similarly gives each backend full freedom to handle named kernel variants natively.

- **Single validation path.** Strategy B contracts are each self-consistent and closed. Strategy A contracts are validated through the `ProblemTypeStrategy` hierarchy, which resolves the flag value at plan-construction time and validates buffer types against the known strategy. No validation logic is duplicated across backends.

- **DAG correctness.** The structural difference between CCE and BCE loss paths (presence or absence of the loss reduction sub-tree in Node 14) is expressed through the plan's dependency edges, not through conditional logic. The plan is a static DAG for any given module configuration.

- **Extensibility.** If a future Operating Mode is introduced (e.g., a focal loss variant), it can be added as either a new Strategy B kernel (if structurally divergent) or a new flag value (if parametrically divergent), following the same decision criteria established here. The `ProblemTypeStrategy` hierarchy is explicitly designed for this extension (Open/Closed Principle).

### Negative

- **Mixed representation requires documentation.** A reader examining the plan must understand that CCE/BCE divergence uses two mechanisms depending on the kernel category. This ADR serves as that documentation. The plan builder's code should include comments referencing this ADR at the points where each mechanism is applied.

- **Conditional contract for Nodes 8/9/10.** The `KernelContract` for gradient production kernels has a flag-gated targets buffer specification that is not a fully closed logical system in the strictest Article 1.4 sense. This is an honest representation of the kernel's actual interface (which uses `void*` type-punning) and is mitigated by the strategy-specific host-side validation.

- **Kernel header documentation burden.** The `Kernel Bifurcation` annotation already present in `kernels.cl.h` for Nodes 6/7 (referencing CONTRACT.md §7.0) should be complemented by a corresponding `Strategy Delegation` annotation for Nodes 8/9/10 referencing this ADR. This is a documentation-only change.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan conveys intent; renderer conveys mechanism; three-tier jurisdictional model (Policy / Orchestration / Execution)
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode` with `contract: KernelContract`; static DAG structure; closed node taxonomy
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract.kernel_name` and `ScalarParamSpec` as the plan-level expression vocabulary; `KernelBinding` as the backend-specific rendering mechanism
- [CONCEPT.md](../CONCEPT.md) — §3 CCE/BCE strategy options (Strategy A and Strategy B); §1 Architectural Elegance Feedback
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 2.3 `FLAG_` number type taxonomy; Article 3.2 Placement Contract specification
