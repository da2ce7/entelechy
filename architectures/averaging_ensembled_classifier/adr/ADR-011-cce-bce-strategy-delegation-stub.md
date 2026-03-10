# ADR-011: CCE/BCE Strategy Delegation

**Status:** STUB (DECIDED — Resolved by ADR-001; confirmed by ADR-002 node design)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-002, ADR-007  
**Blocks:** —

---

## Context

CONCEPT.md §3 explicitly permits both Strategy A (single kernel with runtime flag) and Strategy B (separate kernels) for CCE/BCE divergence. The plan model requires a position on how this divergence is expressed.

---

## Decision

Resolved by ADR-001 and confirmed by ADR-002's `KernelDispatchNode` design.

The `KernelDispatchNode` for Nodes 6/7 (ADR-002) carries the `problem_type` in its `scalar_params` or via distinct `kernel_identity` values. The backend decides the dispatch mechanism:

- Single kernel with a runtime flag (OpenCL's current approach).
- Pre-compiled pipeline variants via specialization constants (Vulkan).
- Separate C functions (CPU).

**The plan conveys intent; the renderer conveys mechanism.** No remaining decision.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan conveys intent; renderer conveys mechanism
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `KernelDispatchNode` with `scalar_params` and `kernel_identity`
- [CONCEPT.md](../CONCEPT.md) — §3 CCE/BCE strategy options
