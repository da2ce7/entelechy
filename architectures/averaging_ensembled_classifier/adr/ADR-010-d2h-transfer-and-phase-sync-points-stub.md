# ADR-010: D2H Transfer & Phase Sync Points

**Status:** STUB (OPEN)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-002, ADR-009  
**Blocks:** —

---

## Context

ADR-002 establishes `RetrievalNode` as one of the five canonical plan node types, with canonical instances `"inference_retrieval"` (`inference_event`) and `"final_batch_retrieval"` (`final_batch_event`). The `RetrievalNode` specifies the source buffer, expected shape, and the named event it signals — but the mechanism by which the renderer communicates host-side availability is left to this ADR. The backends diverge:

- **OpenCL:** `cl.enqueue_copy` → numpy array, signaled via `cl.Event`.
- **Vulkan:** `vkCmdCopyBuffer` to staging → `vkWaitForFences` → `memcpy` from mapped pointer.
- **CPU:** Direct pointer access — the buffer *is* host memory. Zero-cost.

---

## Decision Required

What does the renderer return for retrieval points?

- **(A) Synchronous `retrieve_results()` method.** The renderer exposes a blocking call that returns a numpy array. Simple, but forces synchronous blocking even where the host could overlap computation with the Learn phase trigger decision.

- **(B) `Future`-like handle.** The renderer returns a lightweight handle with `.wait()` and `.result()` methods. OpenCL wraps its event+buffer. Vulkan wraps its fence+staging pointer. CPU wraps an already-resolved value. This preserves temporal decoupling for Event-Triggered Execution Mode (CONCEPT.md §5).

---

## Tensions

- CONCEPT.md §4 requires that `inference_event` and `final_batch_event` be expressible as named synchronization points. A Future-like handle naturally represents "this named point has been reached."
- The Event-Triggered Execution Mode requires the host to read Act-phase results *before* deciding when to trigger Learn. A blocking API (Option A) forces the host to commit to retrieving results at a specific point. A Future (Option B) allows deferred retrieval.
- Option A is simpler; Option B is more faithful to the architecture's event-driven scenarios.

---

## References

- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `RetrievalNode` definition; named synchronization points
- [ADR-009: Buffer Lifecycle](ADR-009-buffer-lifecycle-in-the-plan-model-stub.md) — plan-level buffer lifetime annotations
- [CONCEPT.md](../CONCEPT.md) — §4 Event-Triggered Execution Mode; §5 Validation Scenarios
