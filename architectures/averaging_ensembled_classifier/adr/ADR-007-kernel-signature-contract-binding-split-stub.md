# ADR-007: KernelSignature Contract/Binding Split

**Status:** STUB (NARROWED — Split into shared `KernelContract` + backend-specific `KernelBinding`)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-001, ADR-002, ADR-003 (constrains), ADR-004 (constrains)  
**Blocks:** ADR-011, ADR-012, ADR-013

---

## Context

The `KernelSignature` classes currently produce OpenCL argument lists and return `cl.Event`. ADR-001 demands that plan construction validates kernel contracts without touching backend types. CONTRACT.md Article 1.4 (Collaborative Interface Verifiability) requires that all validation occurs pre-dispatch — this validation is inherently shared.

---

## Narrowed Direction

Split into two components:

- **`KernelContract`** (shared): Encodes shapes, padding contracts, calculability proofs, validation preconditions, and placement strategies from CONTRACT.md. Used by the plan builder to validate node correctness at plan-construction time.
- **`KernelBinding`** (backend-specific): Translates a validated contract into native dispatch arguments — positional `clSetKernelArg` calls (OpenCL), push constant structs + descriptor sets (Vulkan), typed arg structs (CPU).

---

## Remaining Decision

How does the `KernelContract` handle parameters that exist in some backends but not others (the `flat_tile_index` divergence)?

- **(A) Abstract placement key.** The contract specifies "this kernel is a Partial Renderer with N tiles using placement strategy `grid_mod_cls`." The *mechanism* — host scalar, `gl_WorkGroupID.x`, `task_index` — is a binding concern. The contract's calculability proof references the abstract placement key, not a specific parameter name.

- **(B) Superset parameter list with backend annotations.** The contract lists all parameters including `flat_tile_index`, annotated as `[OpenCL-only]`. Risks: pollutes the shared contract with backend concerns; must be extended for every new backend.

Option A is strongly favored. The Placement Contract (CONTRACT.md Article 3.2) already defines strategies abstractly (`grid_mod_cls`, `linear_batch`). The contract specifies the *strategy and key domain*; the binding specifies *how the key is communicated*.

---

## Upstream Constraints

**ADR-003:** Establishes the concrete precedent for the jurisdictional split. The `ReductionTreePlan` carries Policy-tier output (thresholds, fan-in, offset lists) as pure data; the renderer interprets it through backend-native dispatch. The Contract/Binding split applies the same principle to individual kernel invocations.

**ADR-004:** `StreamingLoopNode` embeds `KernelDispatchNode` templates as its `body_nodes`, and each body node carries its own `KernelContract` reference for plan-time validation. The Contract/Binding split governs both top-level `KernelDispatchNode`s and child nodes inside `StreamingLoopNode` bodies. The stride table's `ParameterStride.param_name` fields reference abstract parameter names, confirming Option A as the only viable approach.

**ADR-005:** Node 16's `KernelContract` validates buffer shapes and scalar parameter ranges at plan-construction time, consistent with the same Contract/Binding split applied to every other `KernelDispatchNode`.

---

## Tensions

- CONTRACT.md Article 1.4.1 demands that all calculability proof terms exist in the interface. If `flat_tile_index` is abstracted away, the proof must reference the abstract placement key. This requires a CONTRACT.md amendment — specifically, adding a "Backend Binding" section to Article 3.2.3 that acknowledges the mechanism is backend-specific while the strategy is universal.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure; shared validation
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode` carries `contract: KernelContract`
- [ADR-003: Reduction Tree Plan Representation](ADR-003-reduction-tree-plan-representation.md) — jurisdictional split precedent
- [ADR-004: Streaming Loop Plan Representation](ADR-004-streaming-loop-plan-representation.md) — body node contract validation; abstract parameter names
- [ADR-005: Node 16 Opacity](ADR-005-node-16-opacity-in-the-plan.md) — plan-time contract validation for opaque specialized kernels
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability; Article 3.2 Placement Contract
