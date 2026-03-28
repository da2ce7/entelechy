# ADR-010: D2H Transfer & Phase Sync Points

**Status:** ACCEPTED  
**Date:** 2026-03-10  
**Deciders:** —  
**Supersedes:** —  
**Blocked by:** ADR-002, ADR-009  
**Blocks:** ADR-018

---

## Context

ADR-002 establishes `RetrievalNode` as one of the five canonical plan node types. Each `RetrievalNode` carries a `node_id`, `depends_on` set, source buffer name, expected shape, and named event identifier. The two canonical instances are `"inference_retrieval"` (signaling `inference_event` — Act-phase completion with `Final Probs` available on the host) and `"final_batch_retrieval"` (signaling `final_batch_event` — Learn-phase completion with all parameter updates committed). ADR-002 specifies that the renderer "must signal host-side availability at this point" but defers the signaling mechanism to this ADR.

ADR-009 provides the upstream buffer contract:

- Each `RetrievalNode`'s source buffer has a `BufferDescriptor` with `role = BATCH_OUTPUT`. The descriptor carries `padded_shape`, `element_size_bytes`, and lifetime annotations (`producing_node`, `last_consumer`). The `last_consumer` for a `BATCH_OUTPUT` buffer is the `RetrievalNode`'s `node_id` — the renderer must keep the physical allocation live until the host has consumed the data.
- The `BufferHandle` token is backend-neutral. The renderer maps it to physical memory (`cl.Buffer`, `VkBuffer` offset, host pointer) via its internal handle-to-physical map. The D2H transfer reads from this physical allocation.

### The current state

The existing OpenCL implementation couples D2H transfer into two mechanisms:

1. **`HostView`** (`launcher_infra.py`): A class that pre-allocates a padded numpy host buffer, enqueues a non-blocking `cl.enqueue_copy` from `cl.Buffer` to this host buffer, and returns a `cl.Event`. The `get()` method returns a numpy slice of the host buffer, stripping padding to yield the logical shape.

2. **`BatchProcessor.run()`**: Returns `Tuple[cl.Event, cl.Event, HostView]` — the `final_learn_event`, the `inference_event`, and the `HostView` containing `Final Probs`. The caller blocks on `inference_event` completion and then calls `final_probs_view.get()` to obtain the unpadded numpy array.

Both mechanisms are OpenCL-specific: `HostView.enqueue_read()` takes `cl.CommandQueue` and `cl.Buffer`; the returned `cl.Event` is an OpenCL primitive.

### Backend divergence

The three target backends have fundamentally different D2H transfer mechanics:

| Aspect | OpenCL | Vulkan | CPU |
| :--- | :--- | :--- | :--- |
| Transfer mechanism | `cl.enqueue_copy(queue, host_arr, cl_buf)` | `vkCmdCopyBuffer` to staging → `vkWaitForFences` → read mapped pointer | Direct pointer access — buffer *is* host memory |
| Completion signal | `cl.Event` (per-enqueue token) | `VkFence` (associated with command buffer submission) | Function return (synchronous) |
| Host buffer allocation | Caller pre-allocates numpy array | Renderer maps staging buffer via `VMA_ALLOCATION_CREATE_MAPPED_BIT` | Zero-cost numpy view over existing allocation |
| Transfer cost | DMA copy, ~µs–ms depending on size | DMA copy + staging buffer, ~µs–ms | Zero (no copy) |
| Staging requirement | None (direct device→host) | Mandatory staging buffer in host-visible memory | None |

The CPU backend's zero-cost case is structurally significant: the computational buffer IS host-accessible memory. No data movement occurs. The transferred "result" is a numpy view over the original allocation. Any shared-layer protocol must gracefully degenerate to this zero-copy case without imposing a mandatory copy.

### The design question

The plan's `RetrievalNode` specifies *what* data to retrieve and *when* (after which dependencies). The open question is the shared-layer protocol by which the renderer communicates host-side data availability to the host orchestrator — the type that crosses the plan boundary from the Orchestration tier back to the host.

### What varies across backends

Only the completion-signaling mechanism and the physical transfer path differ:

- **OpenCL:** `cl.Event.wait()` then read from pre-copied numpy array.
- **Vulkan:** `vkWaitForFences(fence)` then `memcpy` from mapped staging pointer to numpy array.
- **CPU:** No wait, no copy — direct numpy view over the compute buffer.

The shared layer needs a uniform interface over these three completion models.

---

## Decision Drivers

1. **ADR-001 (Plan as data structure):** The plan boundary is a pure data structure. No `cl.Event`, `VkFence`, or backend-specific synchronization type may appear in the shared layer. The return type from the renderer must be backend-neutral.

2. **ADR-001 (Policy / Orchestration boundary):** The shared layer is the Policy tier. It determines *what* data is retrieved (the `RetrievalNode`'s source buffer and expected shape) and *when* (the dependency edges). The Orchestration tier (renderer) determines *how* — the transfer mechanism, staging strategy, and completion signaling. The shared-layer protocol type must capture Policy-tier invariants (data shape, completion semantics) without constraining Orchestration-tier implementation.

3. **ADR-002 (RetrievalNode semantics):** The `RetrievalNode` "marks a host-observable completion point. Its `node_id` corresponds to the named event from CONCEPT.md §4." The return type must faithfully represent "this named point has been (or will be) reached."

4. **ADR-009 (BufferDescriptor with `BATCH_OUTPUT` role):** The source buffer's `BufferDescriptor` carries `padded_shape` and `role = BATCH_OUTPUT`. The `last_consumer` is the `RetrievalNode`'s `node_id`. Physical memory must remain allocated until the host has consumed the retrieved data. The protocol must define when "consumed" occurs — i.e., when the renderer may reclaim the physical allocation.

5. **CONCEPT.md §4 (Asynchronous Host Interaction):** The `inference_event` and `final_batch_event` are named synchronization points. The Event-Triggered Execution Mode requires the host to read Act-phase results *before* deciding when to trigger Learn. The protocol must support deferred, non-blocking retrieval — the host receives a handle immediately and consumes the data later.

6. **CONCEPT.md §5 (Unified Execution Model):** Both Sequential and Event-Triggered modes must use the same retrieval protocol. The only difference is *when* the host consults the result — immediately (Sequential) or after an external event (Event-Triggered). The protocol must not assume either pattern.

7. **CPU backend zero-copy:** The CPU backend's buffer IS host memory. The protocol must degenerate to an already-resolved, zero-overhead handle. Imposing a mandatory copy would violate CONCEPT.md §2 (Primacy of Memory Strategy) — the CPU backend's "fastest possible memory tier" is direct access.

8. **Backend allocation freedom (ADR-009):** The renderer's physical allocation strategy is unconstrained. The protocol must not prescribe staging buffer lifetime, host buffer allocation, or transfer scheduling. These are Orchestration-tier concerns.

---

## Options Considered

### Option A: `RetrievalFuture` Protocol with padding-aware numpy result

The renderer returns a `RetrievalFuture` for each `RetrievalNode`. The `RetrievalFuture` is a `typing.Protocol` (structural type) with methods for waiting, result access, and lifecycle management.

- `.wait()` — Blocks until the data is host-accessible. Idempotent.
- `.result()` → `numpy.ndarray` — Returns the retrieved data as a numpy array with logical (unpadded) shape. Implicitly calls `.wait()` if the transfer has not yet completed.
- `.release()` — Signals the host has finished consuming the result; the renderer may reclaim physical memory.

The renderer owns padding-stripping. The `RetrievalNode` carries the `logical_shape` (unpadded); the corresponding `BufferDescriptor` carries `padded_shape`. The renderer has both values and slices the transfer buffer to the logical shape before storing it as the future's result.

```python
from typing import Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RetrievalFuture(Protocol):

    @property
    def node_id(self) -> str: ...

    def wait(self) -> None: ...

    def result(self) -> NDArray[np.floating]: ...

    def release(self) -> None: ...
```

Each backend implements:

- **OpenCL:** Wraps `cl.Event` + pre-allocated numpy host buffer. `wait()` calls `event.wait()`. `result()` returns a numpy slice to `logical_shape`. `release()` marks the host buffer as reclaimable.
- **Vulkan:** Wraps `VkFence` + staging buffer mapped pointer. `wait()` calls `vkWaitForFences`. `result()` constructs a numpy array from the mapped pointer and slices it. `release()` marks the staging buffer as reclaimable.
- **CPU:** Wraps a direct numpy view over the compute buffer. `wait()` is a no-op. `result()` returns the pre-constructed view. `release()` decrements a reference count on the underlying allocation.

**Advantages:**
- The host orchestrator receives a clean numpy array with no padding, no backend imports, and no staging-buffer awareness.
- Protocols are structural (duck-typed). Backends need not inherit from a shared base class — they implement the method signatures.
- `release()` gives the renderer explicit lifecycle control: the physical memory backing the `BATCH_OUTPUT` buffer (and any staging buffer) is retained until the host explicitly signals consumption is complete.
- The CPU backend's zero-copy path is naturally supported: `result()` returns a view, `wait()` is a no-op, `release()` is a buffer lifecycle marker.
- `runtime_checkable` allows `isinstance` checks for defensive validation.

**Disadvantages:**
- The renderer must incorporate padding-stripping logic — a minor amount of shared-layer knowledge (the logical-to-padded shape relationship) enters the Orchestration tier. But the `RetrievalNode` carries both shapes, so the renderer has the information without importing shared-layer modules.
- `release()` is an explicit lifecycle management call. If the host forgets to call it, the renderer cannot reclaim memory. This is a correctness burden on the host orchestrator.

### Option B: `RetrievalFuture` Protocol returning raw padded array, shared-layer stripping

Same as Option A, but `.result()` returns the raw, padded numpy array. The shared layer provides a `strip_padding(arr, logical_shape)` utility. The host orchestrator calls `strip_padding(future.result(), node.logical_shape)` to obtain the clean array.

```python
@runtime_checkable
class RetrievalFuture(Protocol):

    @property
    def node_id(self) -> str: ...

    def wait(self) -> None: ...

    def result(self) -> NDArray[np.floating]: ...  # padded shape

    def release(self) -> None: ...


def strip_padding(
    arr: NDArray, logical_shape: tuple[int, ...]
) -> NDArray:
    """Slice a padded array to its logical shape. Pure function."""
    slicing = tuple(slice(0, dim) for dim in logical_shape)
    return arr[slicing]
```

**Advantages:**
- The renderer is simpler — it returns the transfer result without shape manipulation. No padding-stripping logic in the Orchestration tier.
- Padding-stripping is a pure function in the shared layer — testable in isolation, no backend dependency.
- The host sees the padded array if it needs it (e.g., for debugging physical memory layout).

**Disadvantages:**
- Every host call site must remember to call `strip_padding()`. Missing the call produces a subtly wrong (padded) array — a bug that may not manifest until the array is consumed by downstream logic expecting the logical shape.
- The `RetrievalFuture` contract is incomplete: `.result()` returns data in a representation that is not directly consumable by the host orchestrator without an additional transformation. The future's result is not "the result" — it is an intermediate form.
- For the CPU backend, the padded array may expose padding elements that contain stale or uninitialized data. Returning this to the host violates the principle of least surprise.

### Option C: Abstract Base Class `RetrievalFuture` with `HostBuffer` return type

An abstract base class with a `HostBuffer` wrapper type that carries both the raw data and metadata for interpretation.

```python
from abc import ABC, abstractmethod


class HostBuffer:
    """Carries retrieved data with metadata for interpretation."""

    def __init__(
        self,
        raw: NDArray,
        padded_shape: tuple[int, ...],
        logical_shape: tuple[int, ...],
        dtype: np.dtype,
    ):
        self._raw = raw
        self.padded_shape = padded_shape
        self.logical_shape = logical_shape
        self.dtype = dtype

    def to_numpy(self) -> NDArray:
        """Return unpadded numpy array."""
        slicing = tuple(slice(0, dim) for dim in self.logical_shape)
        return self._raw[slicing]


class RetrievalFuture(ABC):

    @property
    @abstractmethod
    def node_id(self) -> str: ...

    @abstractmethod
    def wait(self) -> None: ...

    @abstractmethod
    def result(self) -> HostBuffer: ...

    @abstractmethod
    def release(self) -> None: ...
```

**Advantages:**
- `HostBuffer` is self-describing: carries its own shape metadata. The host can inspect padding if needed and extract the logical array via `.to_numpy()`.
- ABC enforces that all backends implement the required methods — missing methods are caught at class definition time, not at call time.
- The `HostBuffer` type provides a natural place for future extensions (e.g., dtype verification, provenance tracking).

**Disadvantages:**
- Introduces two new types (`HostBuffer` + `RetrievalFuture` ABC) where Option A introduces one (Protocol). `HostBuffer` is a thin wrapper around a numpy array with trivial logic — the wrapper adds conceptual overhead without proportional value.
- ABC requires backends to inherit from the shared-layer base class. This creates a coupling direction (backend → shared layer) that is stronger than Protocol's structural typing. If the shared layer changes the ABC's method signatures, all backends must update.
- The two-step `.result().to_numpy()` pattern means the host must remember the second call. Using `.result()` directly returns a `HostBuffer`, not a numpy array — a type error that is caught but still an ergonomic burden.
- For the CPU backend, constructing a `HostBuffer` around an already-correct numpy view is pure overhead.

---

## Analysis

### Eliminating Option B

Option B pushes padding-stripping to the host orchestrator. The `RetrievalFuture` returns a padded array — an intermediate representation that requires a mandatory post-processing step before the host can consume it. This makes the abstraction incomplete.

The analysis parallels ADR-009's elimination of its Option B (renderer-owned lifetimes), where deferred computation "forfeits the shared layer's ability to validate" at construction time. Here, the concern is not deferred validation but **deferred correctness**: a `RetrievalFuture` whose `.result()` returns an array in a representation the host cannot directly use is not delivering "the result" — it is delivering a staging artifact.

The current `HostView` pattern performs padding-stripping internally — `get()` slices to `real_shape`. Option B regresses from this design by splitting an operation that the existing code correctly encapsulates into two separate steps across two layers.

The decisive argument is the failure mode. If a host call site omits `strip_padding()`:

- For 1D buffers where `padded_shape == logical_shape`, the omission is invisible — tests pass, behavior is correct. The bug lies dormant.
- For 2D buffers with SIMD padding (`padded_width > logical_width`), the result array silently contains extra columns with stale or uninitialized data. This is a correctness failure that manifests at *consumption* time, not retrieval time, and cannot be caught by CONTRACT.md Article 1.4's pre-dispatch validation.

The padding-stripping operation is trivial (one numpy slice) and requires only information already available on the `RetrievalNode` (`logical_shape`) and `BufferDescriptor` (`padded_shape`). The renderer has both values. Making it the renderer's responsibility ensures every backend produces the same clean result type — without trusting every host call site to perform the conversion.

### Eliminating Option C

Option C introduces `HostBuffer` as an intermediate type between the renderer and the host. The type wraps a numpy array with shape metadata that the host already possesses from the `RetrievalNode` and `BufferDescriptor`. The wrapper carries no information that is not already available in the plan data structure. It exists solely to bundle these values for convenience — but the `RetrievalFuture` itself already represents "the result of this `RetrievalNode`," making the bundling redundant.

The two-step `.result().to_numpy()` pattern is strictly worse than Option A's `.result()` → numpy. Every call site adds one method call, one intermediate type, and one additional opportunity for misuse (using the `HostBuffer` directly instead of extracting the array). This mirrors Option B's incompleteness problem, merely relocating the mandatory post-processing from a free function to a method call.

The ABC inheritance requirement creates a stronger coupling than Protocol's structural typing. Protocols align with Python's existing duck-typing idiom and the project's data-oriented architecture (ADR-001: "plan boundary is a pure data structure"). An ABC at the plan boundary introduces a class hierarchy where a structural contract suffices. If the shared layer adds or removes a method on the ABC, every backend implementation must update its class definition — a cascading change that Protocol's structural compliance avoids.

For the CPU backend, constructing a `HostBuffer` around an already-correct numpy view imposes allocation and initialization overhead for a wrapper object whose metadata the caller already knows and whose `.to_numpy()` method performs the same trivial slice that Option A performs once, inside the renderer.

### Choosing Option A

Option A provides the cleanest mapping from the `RetrievalNode` plan data to a host-consumable result:

**The renderer receives** a `RetrievalNode` with:
- Source buffer name → resolved to `BufferHandle` → resolved to physical memory.
- `logical_shape` (unpadded) and `padded_shape` (from the corresponding `BufferDescriptor`).
- Dependency edges (upstream completions required before transfer begins).

**The renderer produces** a `RetrievalFuture` that:
- Wraps the backend-native completion mechanism (`cl.Event`, `VkFence`, synchronous return).
- Pre-allocates a host buffer (OpenCL, Vulkan) or creates a view (CPU).
- On `.result()`, returns an unpadded numpy array — the final, consumable form.

**The host receives** a `RetrievalFuture` that:
- Has no backend imports on its interface (Protocol is structural).
- Returns numpy (the project's host-side numerical lingua franca).
- Supports deferred consumption (`.wait()` + `.result()` at the host's discretion).
- Explicitly manages buffer lifecycle via `.release()`.

This boundary satisfies all decision drivers:

1. **ADR-001:** `RetrievalFuture` is a Protocol — no backend types in the shared layer.
2. **ADR-001 (Policy/Orchestration boundary):** Policy decides what to retrieve (plan nodes); Orchestration decides how (transfer, staging, completion signaling). The Protocol is the interface between them.
3. **ADR-002:** The `RetrievalNode`'s named event is represented by `RetrievalFuture.node_id`.
4. **ADR-009:** `BATCH_OUTPUT` buffer memory is retained until `release()`. The `last_consumer` semantics from ADR-009 are enforced: the `RetrievalNode` is the last consumer, and `release()` signals that the last consumer has finished.
5. **CONCEPT.md §4:** Deferred retrieval is natural — the host receives the future immediately and calls `.result()` when ready.
6. **CONCEPT.md §5:** Both Sequential and Event-Triggered modes use the same protocol. Sequential mode calls `.result()` immediately; Event-Triggered mode stores the future and calls `.result()` after the external trigger.
7. **CPU zero-copy:** `.wait()` is a no-op, `.result()` returns a pre-constructed numpy view.
8. **Backend allocation freedom:** Physical transfer path, staging buffers, and host allocation are Orchestration-tier concerns. The Protocol prescribes none of them.

### The `release()` contract and buffer lifecycle

ADR-009 establishes that the `BufferDescriptor` for a `BATCH_OUTPUT` buffer has `last_consumer` = the `RetrievalNode`'s `node_id`. The rendering contract (ADR-009 §Backend rendering contract, item 5) states: "The renderer must not reclaim or reuse a buffer's physical memory before its `last_consumer` node has completed execution."

For `BATCH_OUTPUT` buffers, "completed execution" of the `RetrievalNode` is ambiguous — the D2H transfer may complete (the device-side operation finishes) before the host has consumed the result (the host-side read of the numpy array). The `release()` method resolves this ambiguity: the renderer must retain the physical allocation (and any staging buffer) until the host calls `release()`. This extends ADR-009's `last_consumer` semantics from "last DAG node has completed" to "last DAG node has completed AND the host has signaled consumption."

This extension is necessary because:

- The Vulkan backend's staging buffer must remain mapped until the host has read from it. Reclaiming or reusing the staging buffer before the host reads would corrupt the data.
- The OpenCL backend's pre-allocated numpy host array references a copy of the device data. Once the `cl.Event` completes, the numpy array is independent of the device buffer — but the host still needs the numpy array to remain valid. `release()` signals that the host is done with the numpy array, allowing the renderer to reclaim or reuse the host-side allocation.
- The CPU backend's numpy view references the original compute buffer directly. The buffer cannot be reused for the next batch until the host has finished reading. `release()` signals this completion.

Without `release()`, the renderer would have no programmatic signal for when to reclaim `BATCH_OUTPUT` memory. It would either retain memory indefinitely (leak) or reclaim it speculatively (use-after-free risk). `release()` makes the lifecycle contract explicit and enforceable.

### Padding-stripping in the renderer

The `RetrievalNode` carries `logical_shape` (the unpadded shape the host expects) and the corresponding `BufferDescriptor` carries `padded_shape`. The renderer performs padding-stripping as part of the retrieval:

- **OpenCL:** `cl.enqueue_copy` transfers the full padded buffer to a pre-allocated numpy array. `.result()` returns a numpy slice of this array to `logical_shape`.
- **Vulkan:** `vkCmdCopyBuffer` copies the full padded region to staging. After fence completion, `.result()` constructs a numpy array from the mapped pointer and slices it to `logical_shape`.
- **CPU:** `.result()` constructs a numpy view over the compute buffer with the logical shape — zero-copy if `padded_shape == logical_shape`, or a view-slice if padding exists.

This operation is trivial (one numpy slice) and uses only information available on the `RetrievalNode` and `BufferDescriptor` — no shared-layer module imports required. Making it the renderer's responsibility ensures every backend produces the same clean result type, replicating the existing `HostView.get()` pattern inside each backend's `RetrievalFuture` implementation.

---

## Decision

**Option A: `RetrievalFuture` Protocol with padding-aware numpy result.**

The renderer returns a `RetrievalFuture` for each `RetrievalNode` in the plan. The `RetrievalFuture` is a `typing.Protocol` (structural type) with methods for waiting, result access, and lifecycle management. The result is a numpy array with the logical (unpadded) shape.

### Data structures

#### `RetrievalNode` (refinement of ADR-002)

ADR-002 defines the `RetrievalNode` as one of the five canonical plan node types. This ADR specifies the fields required for D2H transfer:

```python
from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class RetrievalNode:
    """A host-accessible result extraction point in the execution plan.

    Specifies the source buffer to retrieve, the expected shape of the
    result, and the named event that this retrieval represents.

    Pure data. Constructed by the plan builder. Consumed by the renderer.
    """

    node_id: str
    """Unique identifier for this plan node.
    Corresponds to CONCEPT.md §4 named events:
    'inference_retrieval' or 'final_batch_retrieval'."""

    depends_on: FrozenSet[str]
    """node_ids of upstream nodes that must complete before
    the D2H transfer begins. The renderer resolves these to
    its native synchronization mechanism."""

    source_buffer: str
    """Logical name of the buffer to retrieve from device.
    Resolved to a BufferHandle via the plan's buffer namespace.
    The corresponding BufferDescriptor has role = BATCH_OUTPUT."""

    logical_shape: Tuple[int, ...]
    """Unpadded shape of the result the host expects.
    The renderer slices the padded transfer buffer to this shape.
    Derived from the original MemoryLayout's logical dimensions."""

    event_name: str
    """The named synchronization event this retrieval represents.
    One of the canonical event names from CONCEPT.md §4:
    'inference_event' or 'final_batch_event'.
    Carried on the RetrievalFuture as node_id for host identification."""
```

#### `RetrievalFuture` Protocol

```python
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class RetrievalFuture(Protocol):
    """Backend-neutral handle for a pending D2H retrieval.

    Returned by the renderer for each RetrievalNode in the plan.
    The host orchestrator receives this handle immediately after
    plan submission and consumes the result at its discretion.

    Lifecycle:
        1. Renderer creates the future during plan rendering.
        2. Host calls wait() or result() when ready to consume.
        3. Host calls release() when finished with the data.
        4. Renderer may reclaim physical memory after release().

    Thread safety:
        wait() and result() are safe to call from any thread.
        release() must be called exactly once.
    """

    @property
    def node_id(self) -> str:
        """The RetrievalNode's node_id that produced this future.

        Identifies which named synchronization point this future
        represents: 'inference_retrieval' or 'final_batch_retrieval'.
        The host uses this to route the result to the correct consumer.
        """
        ...

    def wait(self) -> None:
        """Block until the retrieved data is host-accessible.

        Idempotent. After the first completion, subsequent calls
        return immediately. The underlying mechanism is backend-specific:
        - OpenCL: cl.Event.wait()
        - Vulkan: vkWaitForFences(fence)
        - CPU: no-op (data is always host-accessible)
        """
        ...

    def result(self) -> NDArray[np.floating]:
        """Return the retrieved data as a numpy array.

        The array has the logical (unpadded) shape specified by the
        RetrievalNode's logical_shape field. If the transfer has not
        completed, this method blocks until it has (equivalent to
        calling wait() first).

        The returned array may be a view over renderer-internal memory
        (particularly for the CPU backend). The host must treat it as
        read-only. The array remains valid until release() is called.

        Returns:
            numpy.ndarray with shape == RetrievalNode.logical_shape
            and dtype matching the plan's PrecisionConfig.numpy_dtype.
        """
        ...

    def release(self) -> None:
        """Signal that the host has finished consuming the result.

        After this call, the renderer may reclaim physical memory
        backing both the source BATCH_OUTPUT buffer and any
        backend-specific staging memory (Vulkan staging buffer,
        OpenCL host-side numpy allocation). The numpy array
        previously returned by result() becomes invalid — accessing
        it after release() has undefined behavior.

        Must be called exactly once per future. Calling release()
        before result() discards the retrieval. Calling release()
        more than once has undefined behavior.

        This method bridges ADR-009's last_consumer semantics:
        the RetrievalNode is the last_consumer of the BATCH_OUTPUT
        buffer, and release() signals that the last consumer has
        truly finished.
        """
        ...
```

### Renderer rendering contract

When the renderer encounters a `RetrievalNode` during plan rendering, it performs the following steps:

1. **Resolve the source buffer.** Look up `source_buffer` in the plan's buffer namespace to obtain the `BufferHandle`, then resolve the handle to physical memory via the internal handle-to-physical map. Verify the corresponding `BufferDescriptor` has `role = BATCH_OUTPUT`.

2. **Respect dependency edges.** The D2H transfer must not begin until all nodes in `depends_on` have completed execution. The renderer uses its native synchronization mechanism to enforce this:
   - **OpenCL:** Add upstream `cl.Event`s to the `wait_for` list of the `cl.enqueue_copy` call.
   - **Vulkan:** Ensure the `vkCmdCopyBuffer` command is recorded *after* a pipeline barrier that depends on all upstream dispatches. Alternatively, place the copy in a separate command buffer submitted after the compute command buffer signals a semaphore.
   - **CPU:** Dependencies are satisfied by the synchronous `pool_dispatch_and_wait` execution model — by the time the renderer reaches the `RetrievalNode`, all upstream nodes have completed.

3. **Initiate the D2H transfer.** Execute the backend-native transfer:
   - **OpenCL:** Pre-allocate a numpy host array with `padded_shape`. Call `cl.enqueue_copy(queue, host_arr, cl_buffer, wait_for=upstream_events)` to obtain a `cl.Event`.
   - **Vulkan:** Record `vkCmdCopyBuffer` from the device-local buffer to a pre-allocated, host-visible staging buffer. The staging buffer is allocated with `VMA_ALLOCATION_CREATE_MAPPED_BIT` for zero-copy host access after fence signaling. The copy size is `prod(padded_shape) * element_size_bytes`.
   - **CPU:** No transfer. The compute buffer is already host-accessible.

4. **Construct the `RetrievalFuture`.** Wrap the backend-native completion state in an object satisfying the `RetrievalFuture` protocol:
   - **OpenCL:** `OpenCLRetrievalFuture(node_id, event, host_arr, logical_shape)`.
   - **Vulkan:** `_VulkanRetrievalFuture(node_id, fence, staging_ptr, padded_shape, logical_shape, dtype)`.
   - **CPU:** `_CPURetrievalFuture(node_id, compute_buffer_ptr, logical_shape, dtype)`.

5. **Return the future.** The renderer's `render()` method returns a mapping from `RetrievalNode.node_id` to `RetrievalFuture`:

```python
from typing import Dict

def render(self, plan: "ExecutionPlan") -> Dict[str, RetrievalFuture]:
    """Render the execution plan on the backend.

    Dispatches all plan nodes using the backend's native execution
    model and returns a future for each RetrievalNode.

    Returns:
        Mapping from RetrievalNode.node_id to RetrievalFuture.
        The host orchestrator uses node_id to identify which
        result each future carries.
    """
    ...
```

### Backend implementation sketches

The following illustrates how each backend wraps its native mechanism. These are implementation-level details — the shared layer sees only the `RetrievalFuture` protocol.

#### OpenCL

```python
class OpenCLRetrievalFuture:
    """OpenCL implementation of RetrievalFuture."""

    def __init__(
        self,
        node_id: str,
        event: cl.Event,
        host_buffer: np.ndarray,
        logical_shape: tuple[int, ...],
    ):
        self._node_id = node_id
        self._event = event
        self._host_buffer = host_buffer
        self._logical_shape = logical_shape
        self._result: NDArray | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        self._event.wait()

    def result(self) -> NDArray[np.floating]:
        self.wait()
        if self._result is None:
            slicing = tuple(slice(0, d) for d in self._logical_shape)
            self._result = self._host_buffer[slicing]
        return self._result

    def release(self) -> None:
        self._host_buffer = None  # type: ignore[assignment]
        self._result = None
        self._event = None  # type: ignore[assignment]
```

#### Vulkan

```python
class _VulkanRetrievalFuture:
    """Vulkan implementation of RetrievalFuture."""

    def __init__(
        self,
        node_id: str,
        device: "VkDevice",
        fence: "VkFence",
        staging_mapped_ptr: "ctypes.c_void_p",
        padded_shape: tuple[int, ...],
        logical_shape: tuple[int, ...],
        dtype: np.dtype,
    ):
        self._node_id = node_id
        self._device = device
        self._fence = fence
        self._staging_ptr = staging_mapped_ptr
        self._padded_shape = padded_shape
        self._logical_shape = logical_shape
        self._dtype = dtype
        self._result: NDArray | None = None

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        vkWaitForFences(self._device, 1, [self._fence], True, UINT64_MAX)

    def result(self) -> NDArray[np.floating]:
        self.wait()
        if self._result is None:
            import ctypes
            total_bytes = int(np.prod(self._padded_shape)) * self._dtype.itemsize
            raw = np.frombuffer(
                (ctypes.c_char * total_bytes).from_address(self._staging_ptr),
                dtype=self._dtype,
            ).reshape(self._padded_shape)
            slicing = tuple(slice(0, d) for d in self._logical_shape)
            self._result = raw[slicing].copy()  # copy out of mapped memory
        return self._result

    def release(self) -> None:
        self._result = None
        self._staging_ptr = None  # type: ignore[assignment]
        self._fence = None  # type: ignore[assignment]
```

#### CPU

```python
class _CPURetrievalFuture:
    """CPU implementation of RetrievalFuture. Zero-copy, zero-wait."""

    def __init__(
        self,
        node_id: str,
        buffer_array: np.ndarray,
        logical_shape: tuple[int, ...],
    ):
        self._node_id = node_id
        self._buffer = buffer_array
        self._logical_shape = logical_shape
        slicing = tuple(slice(0, d) for d in logical_shape)
        self._result = buffer_array[slicing]

    @property
    def node_id(self) -> str:
        return self._node_id

    def wait(self) -> None:
        pass  # CPU execution is synchronous; data is already available.

    def result(self) -> NDArray[np.floating]:
        return self._result

    def release(self) -> None:
        self._result = None  # type: ignore[assignment]
        self._buffer = None  # type: ignore[assignment]
```

### Host orchestrator consumption pattern

The host orchestrator uses the returned futures in both execution modes:

**Sequential Execution Mode:**

```python
futures = renderer.render(plan)

# Act-phase results — available after diagnostic aggregation (Node 14)
inference_future = futures["inference_retrieval"]
final_probs = inference_future.result()  # blocks until ready
# ... use final_probs for logging, display, etc. ...
inference_future.release()

# Learn-phase completion — available after all updates (Nodes 24, 25)
batch_future = futures["final_batch_retrieval"]
batch_future.wait()  # blocks until parameter updates are committed
batch_future.release()
```

**Event-Triggered Execution Mode:**

```python
# Phase 1: Submit Act plan, obtain inference future
act_futures = renderer.render(act_plan)
inference_future = act_futures["inference_retrieval"]

# Phase 2: Host reads inference results when ready
final_probs = inference_future.result()  # blocks until Act completes
# ... present predictions to user, await ground truth ...

# Phase 3: Ground truth arrives — submit Learn plan
learn_futures = renderer.render(learn_plan)

# Phase 4: Cleanup — release inference data after Learn submission
inference_future.release()

# Phase 5: Await Learn completion
batch_future = learn_futures["final_batch_retrieval"]
batch_future.wait()
batch_future.release()
```

### Relationship to `HostView`

The existing `HostView` class in `launcher_infra.py` is functionally equivalent to `OpenCLRetrievalFuture` — it pre-allocates a numpy host buffer, enqueues a non-blocking copy with `cl.Event`, and provides `get()` to slice to the logical shape. Under this ADR:

- `HostView` is **not renamed or preserved** in the shared layer. It is an OpenCL-specific implementation detail.
- The OpenCL renderer's `OpenCLRetrievalFuture` absorbs `HostView`'s functionality: host buffer allocation, `enqueue_copy`, and `get()`-style slicing.
- The shared layer's `HostView` import is replaced by the `RetrievalFuture` Protocol. The host orchestrator depends only on the Protocol.

### Relationship to the `ExecutionPlan`

The `RetrievalNode` is already part of the `ExecutionPlan`'s node tuple (ADR-002). This ADR adds no new fields to `ExecutionPlan`. The `RetrievalFuture` is a *return type* from the renderer, not a plan-level data structure — it exists only at render time, never in the plan.

```python
@dataclass(frozen=True)
class ExecutionPlan:
    # ... existing fields from ADR-002 (nodes, dependency edges) ...
    # ... buffer_descriptors from ADR-009 ...

    # RetrievalNodes are part of the nodes tuple.
    # No additional fields needed for D2H transfer.
```

---

## Consequences

### Positive

- **Single host-facing type.** The `RetrievalFuture` Protocol is the sole type through which D2H retrieval results cross the plan boundary. The host orchestrator imports one Protocol from the shared layer; all backend-specific transfer mechanics are hidden behind it. No `cl.Event`, `VkFence`, or staging buffer type appears in host code.

- **Padding-stripping is encapsulated.** The renderer strips padding before surfacing the result. The host receives a numpy array with the logical shape and never handles padded data. This replicates the existing `HostView.get()` design — correctly and consistently — across all three backends.

- **Explicit buffer lifecycle.** The `release()` method gives the renderer a programmatic signal for when `BATCH_OUTPUT` physical memory may be reclaimed. This closes the gap in ADR-009's `last_consumer` semantics: the `RetrievalNode`'s DAG completion signals transfer completion, but `release()` signals host consumption completion. The renderer retains memory between these two events; the host controls the duration.

- **CPU zero-copy preserved.** The CPU backend's `RetrievalFuture` is an already-resolved handle: `wait()` is a no-op, `result()` returns a pre-constructed numpy view, `release()` drops the reference. No copy or transfer occurs. This satisfies CONCEPT.md §2 (Primacy of Memory Strategy).

- **Deferred consumption naturally supported.** The Future pattern decouples "data transfer initiated" from "data consumed by host." In Event-Triggered Mode, the host can store the `inference_retrieval` future, attend to external events, and call `.result()` when ready. In Sequential Mode, the host calls `.result()` immediately. Both patterns use identical code paths through the same Protocol.

- **Structural typing alignment.** `typing.Protocol` with `@runtime_checkable` provides the strongest typing guarantee available without inheritance: static type checkers verify the interface at import time; `isinstance` checks provide runtime validation where needed. Backends comply by implementing the method signatures — no base class import or class hierarchy required.

- **Consistent architectural pattern.** The Protocol-based retrieval future follows the project's data-oriented architecture: the plan carries pure data (`RetrievalNode`), the renderer produces a structural type (`RetrievalFuture`), and the host consumes it via a Protocol interface. This mirrors the plan → render → consume flow established by ADR-001.

### Negative

- **`release()` is a manual lifecycle call.** If the host forgets to call `release()`, the renderer cannot reclaim `BATCH_OUTPUT` memory. For long-running training loops, this manifests as a memory leak. Mitigation: the host orchestrator should call `release()` in a `finally` block or use a context manager wrapper. A future ADR could introduce a `RetrievalFuture.__del__` fallback that issues a warning and calls `release()`, but this must not be relied upon for correctness.

- **Renderer incorporates padding-stripping.** Each backend's `RetrievalFuture` implementation includes one numpy slice operation — a trivial amount of shared-layer knowledge (the logical-to-padded shape relationship) in the Orchestration tier. This is justified by the analysis: the alternative (pushing stripping to the host) produces a strictly worse interface. The `RetrievalNode` carries `logical_shape`, making the information available without cross-tier imports.

- **Protocol runtime checking limitations.** `@runtime_checkable` Protocols verify only that methods exist — not their signatures or return types. A backend that implements `result()` returning an integer would pass an `isinstance` check. Mitigation: static type checkers (mypy, pyright) catch signature mismatches at analysis time. Runtime validation in the host orchestrator can check `isinstance(result, np.ndarray)` defensively.

- **One copy unavoidable on GPU backends.** OpenCL and Vulkan must copy data from device to host memory — this is inherent to discrete GPU architectures, not a consequence of the protocol. The protocol does not introduce any *additional* copies beyond the minimum required by the hardware. For the Vulkan backend, `.result()` performs a second copy from mapped staging memory into a numpy-owned buffer. This ensures the returned array is independent of the staging buffer's lifecycle and avoids aliasing hazards when the staging buffer is recycled for the next batch.

### Migration implications

Per ADR-002's migration path and ADR-012's structure:

1. **Define the Protocol** (`RetrievalFuture`) in the shared layer (`shared/retrieval_future.py`). Define the `RetrievalNode` frozen dataclass alongside the other node types in `shared/plan_types.py`. These are additive definitions — no existing code is modified.

2. **Add `logical_shape` to `RetrievalNode`.** The plan builder computes the logical (unpadded) shape from `MemoryLayout` and stores it on the `RetrievalNode`. The `padded_shape` remains on the `BufferDescriptor` (ADR-009). The renderer uses both.

3. **OpenCL renderer** (Phase 3 of ADR-001 migration) implements `OpenCLRetrievalFuture`, absorbing `HostView`'s functionality. The renderer's `render()` method returns `Dict[str, RetrievalFuture]`. The host orchestrator replaces `HostView` usage with the Protocol.

4. **CPU renderer** (Phase 4) implements `_CPURetrievalFuture` as a zero-cost wrapper. The synchronous execution model means the future is already-resolved at construction time.

5. **Vulkan renderer** (Phase 5) implements `_VulkanRetrievalFuture` wrapping `VkFence` + staging buffer. The renderer pre-allocates the staging buffer during plan rendering (size from `BufferDescriptor.size_bytes`) and records `vkCmdCopyBuffer` into the command buffer.

6. **Write Protocol conformance tests** (ADR-016, layer 1) that validate:
   - `result()` returns a numpy array with shape matching `RetrievalNode.logical_shape`.
   - `result()` returns the same array on repeated calls (before `release()`).
   - `wait()` is idempotent.
   - `release()` can be called after `result()` without error.
   - For the CPU backend: `wait()` returns immediately; `result()` is a view (no copy).

7. **`HostView` deprecation.** Once the OpenCL renderer implements `OpenCLRetrievalFuture`, the `HostView` class in `launcher_infra.py` becomes the renderer's internal concern — it may be kept as a private helper or dissolved entirely. The shared-layer import of `HostView` is removed.

---

## References

- [ADR-001: Backend Abstraction Boundary](ADR-001-backend-abstraction-boundary.md) — plan-as-data-structure principle; Policy/Orchestration boundary; three-tier jurisdictional model
- [ADR-002: Plan Node Types & Synchronization Structure](ADR-002-plan-node-types-and-synchronization-structure.md) — `RetrievalNode` definition; named synchronization points (`inference_event`, `final_batch_event`); renderer signals host-side availability
- [ADR-009: Buffer Lifecycle in the Plan Model](ADR-009-buffer-lifecycle-in-the-plan-model.md) — `BufferDescriptor` with `BATCH_OUTPUT` role; `last_consumer` semantics; `BufferHandle` as shared-layer token; backend rendering contract
- [ADR-008: Precision Configuration](ADR-008-precision-configuration.md) — `PrecisionConfig.numpy_dtype` governs the dtype of the returned numpy array
- [ADR-012: Module Factoring & Services Dissolution](ADR-012-module-factoring-and-services-dissolution.md) — `shared/` directory structure where `RetrievalFuture` and `RetrievalNode` live
- [CONCEPT.md](../CONCEPT.md) — §4 Asynchronous Host Interaction (`inference_event`, `final_batch_event`); §5 Unified Execution Model (Sequential vs. Event-Triggered); §2 Primacy of Memory Strategy (CPU zero-copy)
- [CONTRACT.md](../CONTRACT.md) — Article 1.4 Collaborative Interface Verifiability (pre-dispatch validation in the shared layer)
- [VULKAN_BACKEND.md](../VULKAN_BACKEND.md) — Command buffer strategy (Act/Learn split); staging buffer allocation (`VMA_ALLOCATION_CREATE_MAPPED_BIT`); `VkFence` for host synchronization
- [CPU_BACKEND.md](../CPU_BACKEND.md) — `pool_dispatch_and_wait` synchronous model; host-accessible memory (zero-copy D2H)
- [ADR-018: User-Facing API](ADR-018-user-facing-api.md) — `WorkTicket` consumes `RetrievalFuture` internally; `get_prediction()` maps to `RetrievalFuture.result()` + `.release()`; `LearnHandle.wait()` maps to Learn plan's `RetrievalFuture.wait()`
