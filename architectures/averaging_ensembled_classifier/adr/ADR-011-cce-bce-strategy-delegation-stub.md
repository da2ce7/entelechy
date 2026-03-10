# ADR-011: CCE/BCE Strategy Delegation

**Status:** STUB (DECIDED — Resolved by ADR-001; confirmed by ADR-002 and ADR-007)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** —  
**Blocks:** —

---

## Context

CONCEPT.md §3 explicitly permits both Strategy A (single kernel with runtime flag) and Strategy B (separate kernels) for CCE/BCE divergence. The plan model requires a position on how this divergence is expressed.

---

## Decision

Resolved by ADR-001's jurisdictional principle, confirmed by ADR-002's `KernelDispatchNode` design and ADR-007's `KernelContract` data structure.

**The plan conveys intent; the renderer conveys mechanism.**

The `KernelDispatchNode` (ADR-002) carries a `KernelContract` (ADR-007) that expresses CCE/BCE divergence through two complementary mechanisms:

- **`kernel_name`** — Distinguishes CCE and BCE kernel variants at the plan level (e.g., `compute_probs_loss_cce_chunk` vs. `compute_probs_loss_bce_chunk`). This is Strategy B: separate kernels, separate contracts.
- **`ScalarParamSpec` with `problem_type`** — Carries a runtime flag within a single unified kernel's contract. This is Strategy A: single kernel, scalar branch.

The backend's `KernelBinding` (ADR-007) translates whichever mechanism the plan uses into backend-native dispatch:

- **OpenCL:** Single kernel with a runtime flag passed as a positional scalar argument (current approach).
- **Vulkan:** Pre-compiled pipeline variants via specialization constants, selected by `kernel_name` or pushed as a constant.
- **CPU:** Separate C functions dispatched by `kernel_name`, or a single function with a branch on the flag.

The `KernelContract`'s abstract vocabulary (`kernel_name`, `scalar_params`) is sufficient to express both strategies without leaking backend dispatch details into the plan. No remaining decision.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan conveys intent; renderer conveys mechanism
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode` with `contract: KernelContract`
- [ADR-007: KernelSignature Contract/Binding Split](ADR-007-kernel-signature-contract-binding-split.md) — `KernelContract.kernel_name` and `ScalarParamSpec` as the plan-level expression mechanism
- [CONCEPT.md](../CONCEPT.md) — §3 CCE/BCE strategy options
