# ADR-010: D2H Transfer & Phase Sync Points

**Status:** STUB (NARROWED — `Future`-like handle; renderer signals via backend-native mechanism)  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-002, ADR-009  
**Blocks:** —

---

## Context

ADR-002 establishes `RetrievalNode` as one of the five canonical plan node types, with canonical instances `"inference_retrieval"` (`inference_event`) and `"final_batch_retrieval"` (`final_batch_event`). The `RetrievalNode` specifies the source buffer, expected shape, and the named event it signals — but the mechanism by which the renderer communicates host-side availability is left to this ADR.

ADR-009 (ACCEPTED) provides the upstream inputs this ADR requires:

- Each `RetrievalNode`'s source buffer has a `BufferDescriptor` in the plan's buffer namespace, with `role = BATCH_OUTPUT`. The descriptor carries `padded_shape`, `element_size_bytes`, and lifetime annotations (`producing_node`, `last_consumer`). The `last_consumer` for a `BATCH_OUTPUT` buffer is the `RetrievalNode`'s `node_id` — the renderer must keep the physical allocation live until the host has consumed the data.
- The `BufferHandle` token is backend-neutral. The renderer maps it to physical memory (`cl.Buffer`, `VkBuffer` offset, host pointer) via its internal handle-to-physical map. The D2H transfer reads from this physical allocation.

The backends diverge in transfer mechanism:

- **OpenCL:** `cl.enqueue_copy` → numpy array, signaled via `cl.Event`.
- **Vulkan:** `vkCmdCopyBuffer` to staging → `vkWaitForFences` → `memcpy` from mapped pointer.
- **CPU:** Direct pointer access — the buffer *is* host memory. Zero-cost.

---

## Narrowed Direction

`Future`-like handle returned by the renderer for each `RetrievalNode`. The host receives a lightweight object with `.wait()` and `.result()` methods. Each backend wraps its native mechanism:

- **OpenCL:** `cl.Event` + pre-allocated numpy host buffer.
- **Vulkan:** `VkFence` + staging buffer mapped pointer.
- **CPU:** Already-resolved value (`.wait()` is a no-op; `.result()` returns the pointer cast to numpy).

---

## Remaining Decision

The exact shared-layer protocol type (`RetrievalFuture` protocol or ABC) and whether `.result()` returns a numpy array, a memoryview, or a typed buffer reference. Key constraints:

- The return type must be consumable by the host orchestrator without backend imports.
- For `BATCH_OUTPUT` buffers, the host needs the unpadded logical shape (derivable from `BufferDescriptor.padded_shape` and the original `MemoryLayout`'s logical shape, or carried directly on the `RetrievalNode`).
- The `HostView` padding-stripping logic currently in `launcher_infra.py` must be preserved — either in the shared layer's `RetrievalFuture` contract or as a utility the host calls on the raw result.

---

## Upstream Confirmations

**ADR-002 (ACCEPTED):** `RetrievalNode` carries `node_id`, `depends_on`, source buffer name, expected shape, and named event. The renderer must signal host-side availability at this point.

**ADR-009 (ACCEPTED):** The source buffer's `BufferDescriptor` has `role = BATCH_OUTPUT`, `last_consumer` = the `RetrievalNode`'s `node_id`. Physical memory must remain allocated until the host has consumed the retrieved data. The `BufferHandle` → physical mapping is entirely renderer-internal.

---

## Tensions

- CONCEPT.md §4 requires that `inference_event` and `final_batch_event` be expressible as named synchronization points. A Future-like handle naturally represents "this named point has been reached."
- The Event-Triggered Execution Mode requires the host to read Act-phase results *before* deciding when to trigger Learn. A blocking API forces the host to commit to retrieving results at a specific point. A Future allows deferred retrieval.
- The CPU backend's zero-cost retrieval means the Future must gracefully degenerate to an already-resolved handle with negligible overhead.

---

## References

- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `RetrievalNode` definition; named synchronization points
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` with `BATCH_OUTPUT` role; `last_consumer` semantics; `BufferHandle` as shared-layer token
- [CONCEPT.md](../CONCEPT.md) — §4 Asynchronous Host Interaction; §5 Unified Execution Model
