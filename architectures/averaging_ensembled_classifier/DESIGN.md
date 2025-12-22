# **Entelechy Vision Document: The Asynchronous Computational Organism**

## **Executive Vision**

Upon the full realization of its asynchronous execution engine, Entelechy will no longer feel like a machine learning framework you *run*—it will feel like a living computational organism you *collaborate with*. It will be a system that breathes with the rhythm of your data, instantly responding to urgent inference requests while patiently and efficiently processing deep learning tasks in the background.

Entelechy will manifest as a **concurrent, policy-driven execution surface** that transforms the developer experience from imperative scripting to declarative orchestration. You will interact with it not by writing loops that block and wait, but by issuing **intentional requests** that the system weaves into a global tapestry of computation, automatically prioritizing, scheduling, and executing them according to a transparent, inviolable set of rules.

---

## **The Developer Experience: Declarative Intent over Imperative Control**

### **The API Surface: A Conversation with the System**

The system will expose a fluent, asynchronous API that feels more like giving instructions to a capable assistant than programming a machine. The core of this experience is the **`WorkTicket`**—a persistent, globally outstanding promise representing a single piece of work in the system.

```python
# Creating a promise for real-time analysis
ticket = await engine.create_work_ticket(
    x_data=live_sensor_reading,
    act_callback=lambda probs: send_alert_if_anomaly(probs)
)

# Later, when ground truth arrives asynchronously
learn_ticket = await ticket.resolve(y_data=validated_label)

# The (x, y) pair now lives forever, training silently in the background
async for receipt in learn_ticket:
    print(f"Epoch {receipt.epoch_num}: Training complete")
```

This API pattern will make three powerful sensations immediately apparent:

1. **Immediacy:** Your `Act` callbacks fire the moment predictions are ready, with zero coupling to the training lifecycle.
2. **Persistence:** Your resolved `LearnTicket` endures across epochs, a perpetual entity that represents a fixed (x, y) datum in the model's evolving understanding.
3. **Control:** The `retire()` method on a `LearnTicket` gives you surgical precision—you can stop a specific data point from influencing future training without disrupting the entire system.

### **The Three Canonical Scenarios, Seamlessly Unified**

The same core abstractions will handle all three fundamental interaction patterns with elegant consistency:

**Scenario A: The Batch Trainer**
```python
# For maximum throughput on pre-labeled data
for x_batch, y_batch in dataset:
    ticket = await engine.create_work_ticket(x_batch)
    await ticket.resolve(y_batch)  # Resolves instantly, no callback needed
```
*Feel:* You are feeding a hungry learner. The system takes your data and queues it efficiently, training at maximum GPU throughput without ceremony.

**Scenario B: The Real-Time Analyst**
```python
# For event-driven systems where labels arrive later
ticket = await engine.create_work_ticket(
    x_data=trade_data,
    act_callback=execute_trade_decision
)
# ... seconds later ...
await ticket.resolve(y_data=trade_outcome)
```
*Feel:* The system operates on your timelines, not the other way around. Inference is instant and decoupled; learning is eventual and reliable.

**Scenario C: The Interactive Scientist**
```python
# For exploratory workflows with human-in-the-loop
ticket = await engine.request_inference(x_data)
prediction = await ticket.get_prediction()

if human_approves(prediction):
    learn_ticket = await ticket.resolve(y_data)
else:
    await ticket.resolve(None)  # Explicitly discard
```
*Feel:* You are in a dialogue with the model. You can inspect, approve, or reject its predictions, and the system respects your judgment with precision.

---

## **The System in Motion: Observability and Behavior**

### **The Global DAG: A Living Map of Computation**

At the heart of the system will be the **Global Dependency DAG**—a single, mutable `networkx.DiGraph` that serves as the absolute source of truth for all pending work. Through a simple monitoring API, you will be able to **inspect the living state of the system** at any moment:

```python
# Peek into the system's mind
dag_snapshot = engine.conductor.get_dag_snapshot()
ready_acts = dag_snapshot.get_ready_nodes_of_type("ActInstruction")
ready_learns = dag_snapshot.get_ready_nodes_of_type("LearnInstruction")

print(f"System Status: {len(ready_acts)} Acts waiting, {len(ready_learns)} Learns queued")
```

This will give you a **God's-eye view** of what the system is thinking: which jobs are waiting for dependencies, which are ready to run, and how the `Act > Learn` policy is being enforced in real-time.

### **The Priority Dispatcher: The System's Heartbeat**

The **`AsyncConductor`** will run its main loop silently in the background, but its behavior will be observable through logging and metrics. You will see it:

1. **Scan** the DAG for nodes with in-degree zero.
2. **Categorize** them into the high-priority `Act` queue and the low-priority `Learn` queue.
3. **Dispatch** an `Act` task the moment it appears, preempting any `Learn` task that might be running.
4. **Yield** to `Learn` tasks only when the `Act` queue is empty, demonstrating the system's unwavering commitment to latency-sensitive requests.

This will create a system that **feels alive**—constantly juggling priorities, making decisions, and keeping the GPU saturated with meaningful work.

### **The WorkBatch: The Symphony of Parallelism**

When you submit a `WorkBatch` of `N` tickets, you will witness the system's intelligence in action:

```python
# Submit a batch of 1000 independent learning items
tickets = [engine.create_work_ticket(x) for x in large_batch]
work_batch = await engine.submit_work_batch(tickets, y_labels)

# The system will:
# 1. Fan-out: Launch 1000 parallel LearnInstructions
# 2. Stabilize: Apply clipping at each of the log_K(N) reduction stages
# 3. Fan-in: Consolidate into a single AdamUpdate
# 4. Report: Notify you when the entire batch is committed
await work_batch.completion_event.wait()
```

The `SystemPlanner` will dynamically construct a **multi-stage reduction tree** in the DAG, and you will be able to watch this complex graph being built and executed through the monitoring API. This will feel like watching a maestro conduct an orchestra—each kernel a note, each reduction stage a phrase, all culminating in a harmonious update.

---

## **The Architectural Personality: Discipline, Transparency, and Grace**

### **The Principle of Jurisdictional Purity**

Every component will **feel like it knows its place**. There will be no confusion about which layer is responsible for what:
- The `Orchestrator` will feel like a **state manager**, never touching kernel code.
- The `Planner` will feel like an **architect**, never awaiting events.
- The `Conductor` will feel like a **dispatcher**, never questioning the plan.
- The `KernelExecutor` will feel like a **toolbox**, never knowing the strategy.

This purity will make the system feel **incredibly trustworthy**. When something goes wrong, you will know exactly where to look.

### **The Primacy of the Global Dependency DAG**

The DAG will be **sacred**. Every operation, from a single H2D copy to a complex multi-stage reduction, will be a node. You will never wonder, "Is this task done?"—you will simply inspect its in-degree in the DAG. The system will feel **perfectly transparent**, with no hidden state or implicit dependencies.

### **The Principle of Native Asynchronicity**

The system's embrace of `asyncio` will make it feel **modern and idiomatic**. It will integrate seamlessly with other async Python libraries (like async web frameworks or data pipelines). You will be able to compose Entelechy tasks with `asyncio.gather()`, `asyncio.wait()`, and other native primitives, making it feel like a natural extension of the Python ecosystem.

### **The Grace of Adaptive Memory**

The "Cache vs. Recompute" metabolic switch will operate **silently and gracefully**. When you load a massive dataset, the system will feel **resilient**, not fragile. It won't crash with an out-of-memory error; it will simply switch to recompute mode, trading time for space in a transparent act of self-preservation. You will observe this in logs, seeing messages like `[INFO] Switching to RECOMPUTE mode for unit_id=0x4a2f due to VRAM pressure`, and you will feel confident that the system is making intelligent decisions.

---

## **The Promise of the Future: A Platform for Fearless Experimentation**

After completing this design, Entelechy will not just be a tool for running experiments—it will be **a platform for inventing them**. The clean separation of concerns, the verifiable contracts, and the asynchronous, policy-driven execution model will create an environment where:

- **New architectures** can be added by simply implementing new `Instruction` types and registering them with the `SystemPlanner`.
- **New scheduling policies** can be swapped in by replacing the `PriorityDispatcher` logic in the `AsyncConductor`.
- **New hardware backends** can be supported by implementing new `Foundation` layer components.

The system will feel **extensible by design**, not by accident. It will have achieved the ultimate goal of the Entelechy philosophy: **true creative velocity born from profound structural integrity**. You will be able to conduct fearless and ambitious experiments, knowing that the system itself is a reliable, observable, and intelligent partner in your research.


# **Entelechy Technical Vision Document: The Asynchronous Execution Engine**

**Document Revision:** 1.0
**Target State:** Post-implementation of Asynchronous, Policy-Driven Execution Design
**Scope:** Host-side architectural implementation, Layer 1 through Layer 4

---

## **1.0 System Overview: The Four-Layer Asynchronous Architecture**

Upon full implementation, the Entelechy host system will be a strictly layered, asynchronous execution engine orchestrating GPU computation via a global, mutable dependency graph. The architecture abolishes synchronous batch processing in favor of a cooperative multitasking model built on Python's `asyncio`. The fundamental unit of work ceases to be a batch plan and becomes a **node** in a global `networkx.DiGraph`, representing an executable primitive instruction.

The system will be composed of four jurisdictional layers, each with a formally defined contract and zero overlap of responsibility:

*   **Layer 4 (Application Layer):** The `TrainingOrchestrator` manages the lifecycle of `WorkTicket` entities, stages incomplete user requests, and submits primitive planning requests to Layer 3.
*   **Layer 3 (Planning Layer):** The `SystemPlanner` is a stateless function that receives planning requests (e.g., "plan an Act for unit X") and performs atomic mutations on the Global Dependency DAG, inserting instruction nodes and dependency edges.
*   **Layer 2 (Execution Layer):** The `AsyncConductor` runs the master `asyncio` event loop. It continuously scans the Global DAG for nodes with in-degree zero, applies priority policy gating, and dispatches coroutine tasks (`asyncio.Task`) to execute ready instructions.
*   **Layer 1 (Foundation Layer):** The `KernelExecutor`, `BufferManager`, and `awaitable_cl_event` bridge provide the stateless, hardware-interfacing primitives. They are unaware of the DAG, policy, or application logic.

---

## **2.0 Core Data Structures: The Global Dependency DAG**

### **2.1 The Graph Object**

The Global Dependency DAG is a **single, centralized `networkx.DiGraph` instance** owned and exclusively mutated by the `SystemPlanner`. It is read concurrently by the `AsyncConductor` during its scanning loop. All mutations are assumed to be atomic with respect to the conductor's read operations; a simple lock (`asyncio.Lock`) will serialize Planner mutations and Conductor reads to prevent data races during graph traversal.

**Graph Properties:**
*   **Nodes:** Represent executable instructions or data placeholders. Each node is a dataclass instance with a unique identifier.
*   **Edges:** Represent hard dependencies. An edge `A -> B` signifies that node `B` cannot be executed until node `A` has completed and its side-effects are committed.

### **2.2 Node Type Taxonomy**

All nodes inherit from a base class `GraphNode` and are type-distinguished via a `node_type` field. The conductor's scanning logic uses this field to apply policy.

```python
@dataclass(frozen=True)
class GraphNode(abc.ABC):
    node_id: str
    priority: int  # 1=Act, 2=Learn, 3=Utility

@dataclass(frozen=True)
class ActNode(GraphNode):
    node_type: Literal["ActInstruction"]
    unit_id: str
    x_ref: BufferHandle
    callback_handle: Callable[[np.ndarray], None]
    # ... other params for kernel execution

@dataclass(frozen=True)
class LearnNode(GraphNode):
    node_type: Literal["LearnInstruction"]
    unit_id: str
    x_ref: BufferHandle
    y_ref: BufferHandle
    # Subgraph nodes will be created by planner; this is the root.

@dataclass(frozen=True)
class KernelBatchNode(GraphNode):
    node_type: Literal["KernelBatch"]
    signatures: List[KernelSignature]
    wait_for: List[str]  # List of node_ids
    # Represents a batch of kernels that can be enqueued together.

@dataclass(frozen=True)
class HostComputeNode(GraphNode):
    node_type: Literal["HostCompute"]
    compute_fn: Callable[[...], Any]
    wait_for: List[str]
    # For optimizer term calculation, loss reduction, etc.

@dataclass(frozen=True)
class ParameterUpdateNode(GraphNode):
    node_type: Literal["ApplyUpdateInstruction"]
    param_refs: Dict[str, BufferHandle]
    grad_refs: Dict[str, BufferHandle]
    wait_for: List[str]  # Depends on all LearnNodes and reduction tree.
```

### **2.3 Edge Semantics and Dependency Resolution**

*   **Hard Dependencies:** All edges are hard. A node is **ready** if and only if its in-degree is zero (all its dependencies have been removed from the graph by the conductor upon their completion).
*   **Dynamic Edge Creation:** When the `SystemPlanner` creates a `ParameterUpdateNode`, it queries the DAG to find all `LearnNode` IDs within the current `WorkBatch` and adds an edge from each to the `ParameterUpdateNode`. This is performed atomically during the planning phase.

---

## **3.0 Layer 4: The Application Layer (TrainingOrchestrator)**

### **3.1 Core Responsibilities**

The `TrainingOrchestrator` is the **state keeper for user-level intents**. It owns no execution logic, kernels, or memory. Its jurisdiction is the lifecycle of `WorkTicket` entities and the translation of high-level user goals (e.g., "train for 10 epochs") into sequences of primitive planning requests.

### **3.2 Key Internal State**

```python
class TrainingOrchestrator:
    def __init__(...):
        self._staging_area: Dict[str, WorkTicket] = {}
        self._learn_ticket_registry: Dict[str, LearnTicket] = {}
        self._next_unit_id: int = 0
        self._system_planner: SystemPlanner  # Injected dependency
        self._conductor: AsyncConductor      # Injected dependency
```

### **3.3 WorkTicket Lifecycle & State Machine**

A `WorkTicket` is a **stateful, non-async object** that represents a promise. It is created by the orchestrator and returned to the user. Its state transitions are triggered by user calls to its methods, which delegate orchestrator logic.

```
State: PENDING
    | (user calls resolve(y_data))
    v
State: RESOLVED_AS_LEARN
    |
    |-- Spawns LearnTicket
    |-- Calls SystemPlanner.plan_learn()
    v
State: CONSUMED (terminal)
```

*   **Pending:** The ticket exists in `_staging_area`. It holds `x_data` and an optional `act_callback`.
*   **Resolved:** The `resolve(y_data)` method atomically moves the ticket from `_staging_area` to `_learn_ticket_registry`, spawns the `LearnTicket`, and triggers the planner.
*   **Consumed:** After resolution, the `WorkTicket` object is logically dead. Future calls to its methods raise `RuntimeError`.

---

## **4.0 Layer 3: The Planning Layer (SystemPlanner)**

### **4.1 Core Contract**

The `SystemPlanner` is a **stateless, pure function** (or a class with no internal state). It receives high-level requests and performs **atomic, transactional mutations** on the Global DAG. It is the sole writer to the graph.

**Method Signatures:**
```python
class SystemPlanner:
    def plan_act(self, unit_id: str, x_ref: BufferHandle, callback: Callable, dag: nx.DiGraph) -> str:
        """Inserts: H2D Node -> Forward KernelBatch -> D2D Node -> ActNode"""
        # Algorithm:
        # 1. Create H2D node, add to dag.
        # 2. Create KernelBatch for forward pass (nodes 4-7), add edge from H2D.
        # 3. Create D2H node for final_probs, add edge from KernelBatch.
        # 4. Create ActNode (contains callback handle), add edge from D2H.
        # 5. Return node_id of ActNode for potential dependency chaining.

    def plan_learn(self, unit_id: str, x_ref: BufferHandle, y_ref: BufferHandle, dag: nx.DiGraph) -> List[str]:
        """Inserts a complete LearnInstruction subgraph."""
        # Algorithm:
        # 1. Call _build_learning_subgraph() to get list of nodes (H2D, Kernels for fwd/bwd, reduction nodes, HostCompute for Adam terms, ApplyUpdate).
        # 2. Add all nodes and edges to dag in a single transaction (hold lock).
        # 3. Return list of all node_ids in the subgraph (for WorkBatch chaining).

    def plan_work_batch(self, tickets: List[WorkTicket], dag: nx.DiGraph) -> str:
        """The core algorithm for parallel batch training."""
        # 1. For each ticket, call _build_learning_subgraph(), but DO NOT add ApplyUpdate nodes.
        # 2. Collect all Partial_Grad leaf node IDs from each subgraph.
        # 3. Call _render_aggregation_tree(partial_grad_nodes, dag) to insert the log_K(N) reduction subgraph.
        # 4. Create a single ParameterUpdateNode that depends on the reduction tree's final node AND all HostCompute nodes for Adam terms.
        # 5. Add all nodes to dag atomically.
        # 6. Return the node_id of the ParameterUpdateNode (the batch's sync point).
```

### **4.2 The Aggregation Tree Rendering Algorithm**

This is the heart of the `WorkBatch` capability. Given `N` partial gradient nodes, the planner constructs the reduction subgraph.

```python
def _render_aggregation_tree(self, partial_nodes: List[str], dag: nx.DiGraph):
    """Renders the log_K(N) reduction tree into the DAG."""
    N = len(partial_nodes)
    k = self.reduction_plan.k  # From hardware constants
    current_level_nodes = partial_nodes
    stage_idx = 0

    while len(current_level_nodes) > 1:
        next_level_nodes = []
        for group_idx in range(0, len(current_level_nodes), k):
            # Create a KernelBatchNode for this aggregation stage
            group_nodes = current_level_nodes[group_idx : group_idx + k]
            agg_node_id = f"agg_s{stage_idx}_g{group_idx}"

            # The KernelBatch will contain:
            # 1. aggregate_local_reduce (or register) for the sum
            # 2. clip_intermediate_grad for the stage_j threshold
            agg_node = KernelBatchNode(
                node_id=agg_node_id,
                priority=3,
                signatures=self._build_aggregation_signatures(group_nodes, stage_j=len(current_level_nodes) - 1),
                wait_for=group_nodes
            )
            dag.add_node(agg_node_id, data=agg_node)
            for dep_id in group_nodes:
                dag.add_edge(dep_id, agg_node_id)
            next_level_nodes.append(agg_node_id)

        current_level_nodes = next_level_nodes
        stage_idx += 1

    # Final node in current_level_nodes is the root of the reduction tree.
    return current_level_nodes[0]  # ID of final summed gradient node
```

---

## **5.0 Layer 2: The Execution Layer (AsyncConductor)**

### **5.1 The Master Event Loop**

The `AsyncConductor` runs a **single, immortal coroutine** that is the system's heartbeat. It is started by the orchestrator and runs until system shutdown.

```python
class AsyncConductor:
    async def run_main_loop(self):
        """The core asyncio loop. Runs forever."""
        while self._is_running:
            # Phase 1: Scan for ready nodes.
            ready_nodes = self._scan_dag_for_ready_nodes()

            # Phase 2: Policy-based dispatch.
            act_nodes = [n for n in ready_nodes if n.node_type == "ActInstruction"]
            learn_nodes = [n for n in ready_nodes if n.node_type == "LearnInstruction"]

            # Dispatch all ready Act nodes (highest priority).
            for node in act_nodes:
                task = asyncio.create_task(self._execute_act_node(node))
                self._active_tasks[node.node_id] = task

            # Dispatch ready Learn nodes only if no Acts are pending or running.
            if not act_nodes and not any(t for t in self._active_tasks.values() if t.node_type == "ActInstruction"):
                for node in learn_nodes[:self._max_parallel_learn]:  # Respect concurrency limit
                    task = asyncio.create_task(self._execute_learn_subgraph(node))
                    self._active_tasks[node.node_id] = task

            # Phase 3: Cooperative yield.
            await asyncio.sleep(0)  # Yield control to the event loop.
```

### **5.2 Node Execution Coroutines**

Each node type has a dedicated `async` executor method. These methods **await** the completion of their work and then **mutate the Global DAG** to remove the completed node.

```python
async def _execute_act_node(self, node: ActNode):
    """Executes an Act instruction subgraph."""
    # Node.data contains the full subgraph: H2D -> KernelBatch -> D2H -> ActCallback
    # 1. Await H2D copy.
    h2d_event = await self._foundation.enqueue_h2d(node.x_ref)
    # 2. Execute forward kernels.
    kernel_event = await self._foundation.execute_kernel_batch(node.kernel_batch, wait_for=[h2d_event])
    # 3. Await D2H copy.
    d2h_event = await self._foundation.enqueue_d2h(node.probs_ref, wait_for=[kernel_event])
    # 4. Execute callback on host.
    probs_data = await self._foundation.get_buffer_host_view(node.probs_ref, d2h_event)
    node.callback_handle(probs_data)

    # 5. Atomically remove node from DAG.
    async with self._dag_lock:
        self._dag.remove_node(node.node_id)
        # Removing node may free up dependents. No need to signal; next loop scan will see it.
```

---

## **6.0 Layer 1: The Foundation Layer**

### **6.1 The `awaitable_cl_event` Bridge**

This is the **critical integration point** between PyOpenCL and `asyncio`. It converts a blocking `cl.Event.wait()` into an `asyncio.Future` that resolves when the OpenCL event's callback fires.

```python
def awaitable_cl_event(pyopencl_event: cl.Event, loop: asyncio.AbstractEventLoop) -> Awaitable[cl.Event]:
    """Returns an awaitable that resolves when the PyOpenCL event completes."""
    future = loop.create_future()

    def _event_callback(status):
        # Called from OpenCL driver thread. Must schedule on asyncio thread.
        loop.call_soon_threadsafe(lambda: future.set_result(pyopencl_event))

    # Set the callback on the PyOpenCL event.
    pyopencl_event.set_callback(cl.command_execution_status.COMPLETE, _event_callback)
    return future
```

### **6.2 Memory Sovereignty and Versioning**

The `BufferManager`'s role remains unchanged: it is the sole allocator and provides opaque `BufferHandle`s. For the parallel batch training (`WorkBatch`), **versioning is enforced at the graph level**, not via buffer copying.

*   **Static Snapshot:** When the `SystemPlanner` creates a `WorkBatch`, all `LearnInstruction` subgraphs reference the **same** `BufferHandle` for model parameters (e.g., `shared_weights`). This parameter buffer is static for the duration of the batch.
*   **The Update Atomicity:** The `ParameterUpdateNode` has an edge from **every** `LearnNode` in the batch. This ensures the `KernelExecutor` will only launch the final `adam_update` kernel after all parallel executions are complete. The single update kernel call makes the parameter state transition **atomic** from the perspective of the DAG.

---

## **7.0 API Contracts and Developer Interaction**

### **7.1 TrainingOrchestrator Public API**

```python
class TrainingOrchestrator:
    async def create_work_ticket(self, x_data: np.ndarray, act_callback: Optional[Callable]) -> WorkTicket:
        """Creates a promise. If callback provided, immediately plans and dispatches Act subgraph."""

    async def submit_work_batch(self, tickets: List[WorkTicket], y_data: np.ndarray) -> WorkBatchHandle:
        """Plans a complete parallel batch training subgraph. Returns a handle to await its completion."""
        # Internally: Validates tickets, calls SystemPlanner.plan_work_batch(), returns a wrapper around the ParameterUpdateNode's ID.
```

### **7.2 Concurrency Guarantees**

*   **No Deadlocks:** The `AsyncConductor` only dispatches nodes with in-degree zero. Since the DAG is acyclic by construction (planner never creates cycles), and all nodes are eventually completed and removed, the system is deadlock-free.
*   **Fairness:** The priority dispatcher ensures starvation cannot occur. If the `Act` queue is continuously filled, `Learn` tasks will not run, but this is the intended policy. If the system is idle, `Learn` tasks are dispatched.
*   **Cancellation:** Cancelling a `WorkTicket` before resolution simply removes it from the Orchestrator's staging area. Cancelling a `LearnTicket` calls its `retire()` method, which adds a special "poison pill" node to the DAG that, when executed, cleans up the ticket's state and prevents further planning.

---

## **8.0 Example Execution Trace**

**Scenario:** User submits a `WorkBatch` of 2 tickets for a simple model.

1. **Orchestrator:** Receives call to `submit_work_batch([ticket_a, ticket_b], y_data)`. Tickets are resolved, moved to registry.
2. **Planner:** `plan_work_batch()` is called. It creates 2 `LearnNode` subgraphs (each with their own H2D, Kernel Batches).
3. **Planner:** It renders a `log_2(2)=1` stage aggregation tree: one `AggregateLocalReduce` node that depends on both `LearnNode` final gradient nodes.
4. **Planner:** It creates one `ParameterUpdateNode` that depends on the aggregation node. All nodes added to `GlobalDAG`.
5. **Conductor:** Scans DAG. Finds 2 `LearnNode`s, both ready. No `Act` nodes. Dispatches them as `asyncio.Task`s.
6. **Execution:** Both `LearnNode` coroutines run concurrently. They await their kernel batches.
7. **Completion:** LearnNode A finishes, its coroutine acquires `dag_lock`, removes node A. LearnNode B finishes, removes node B.
8. **Conductor:** On next scan, aggregation node now has in-degree zero. Dispatches it.
9. **Aggregation:** Aggregation coroutine awaits its kernel, completes, removes itself.
10. **Conductor:** `ParameterUpdateNode` now has in-degree zero. Dispatches final `adam_update` kernel.
11. **Finalization:** Update coroutine completes, removes the node. `WorkBatchHandle` signals completion to user.

---

## **9.0 Performance and Observability Metrics**

The system will expose the following metrics via a `metrics` object on the `TrainingOrchestrator`:

*   **`dag.nodes_ready`:** Gauge of nodes currently executable.
*   **`dag.nodes_total`:** Gauge of total nodes in the graph.
*   **`conductor.act_queue_depth`:** Number of ready `Act` nodes.
*   **`conductor.learn_queue_depth`:** Number of ready `Learn` nodes.
*   **`planner.work_batches_planned`:** Counter of `WorkBatch` subgraphs created.
*   **`executor.kernel_batch_time`:** Histogram of kernel batch execution times.

These metrics will be queryable in real-time to verify that the system is behaving according to its architectural contracts.
