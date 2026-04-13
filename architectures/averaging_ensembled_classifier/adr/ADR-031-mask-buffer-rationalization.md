# ADR-031: Mask Buffer Rationalization

**Status:** ACCEPTED
**Date:** 2026-04-08
**Deciders:** —
**Triggered by:** (1) Recognition that `hidden_mask` carries non-derivable information under precision-boundary configurations; (2) Category error of assigning precision roles to the discrete boolean `sample_mask` signal
**Depends on:** ADR-008 (Precision Configuration), ADR-020 (Mixed-Precision Execution Model), ADR-009 (Buffer Lifecycle)
**Constrains:** `kernels.cl.h` (Nodes 4, 5, 6, 7, 8, 9, 10, 17, 18), `PrecisionConfig`, host-side buffer allocation and packing
**Amends:** CONTRACT.md Article 8 (Lexicon), CONCEPT.md §2 (Primacy of Memory Strategy — FP8 quantization floor), CONCEPT.md §11 (Host Orchestrator)

---

## Decision Summary

| Decision Axis | Selected Option | Rationale |
|:--------------|:----------------|:----------|
| `hidden_mask` lifecycle | **Policy-controlled via `MaskStrategy`** | FLAG pattern (Strategy A) preserves unified kernel interface; Policy tier selects mode based on `PrecisionConfig` |
| `hidden_mask` mode selection | `explicit` when `storage_dtype != compute_dtype`; `recompute` otherwise | Precision boundary destroys derivative information for sub-floor activations; uniform dtype permits lossless derivation |
| `sample_mask` encoding | **Bitmask: 32 samples per `uint` word, LSB-first** | Eliminates precision-role category error; reduces allocation 32×; decouples from `PrecisionConfig` |
| `sample_mask` in Node 5 | **Added as new parameter** | Enables sample-level early exit; bitmask check is negligible cost (3 integer ALU ops) |
| `sample_mask` in Nodes 17/18 | **Retained (retyped to `uint *` bitmask)** | Early-exit optimization; correctness is guaranteed by upstream invariants but early exit avoids ~3 global loads per masked sample per output element |

---

## Part I: Hidden Mask Policy-Controlled Lifecycle

### 1.1 Problem Statement

The original design treated `hidden_mask` as a "derived cache" — a buffer whose contents could always be reconstructed from `hidden_activations` via `mask = activation > 0`. This framing is valid **only when `STORAGE_TYPE == COMPUTE_TYPE`**.

When they diverge, the mask carries information the kernel provably cannot recover:

| Scenario | Mask Information Content | Derivable from Stored Activations? |
|----------|--------------------------|-----------------------------------|
| ReLU, `STORAGE == COMPUTE` | Zero bits (fully derivable) | **Yes** |
| ReLU, `STORAGE < COMPUTE` | Derivative truth for sub-floor activations | **No** |
| ReLU + external gate (future) | Gate identity (ReLU vs. external) | **No** |
| Non-ReLU activation (future) | Derivative value | **No** |

**Concrete Example (FP8 E4M3):** Node 4 computes in FP32:
```
pre_relu = W*x + b          // e.g., 0.0005
post_relu = max(0, pre_relu) // 0.0005 — genuinely active
mask = (post_relu > 0) ? 1 : 0  // 1.0
```

Then stores both:
```
store_storage(hidden_activations, ..., 0.0005)  // → 0.0 in FP8 E4M3 (below 2⁻⁹ floor)
store_storage(hidden_mask, ..., 1.0)            // → 1.0 (exactly representable)
```

Any consuming kernel that derives the mask from `load_storage(hidden_activations) > 0` sees `0.0 > 0 = false` and incorrectly gates the gradient to zero. The mask preserves the compute-precision derivative truth in a format where the truth survives narrowing.

### 1.2 Solution: `MaskStrategy` in the Policy Tier

The Policy tier determines the mask lifecycle at plan-construction time based on the active `PrecisionConfig`:

```python
@dataclass(frozen=True)
class MaskStrategy:
    """Controls hidden_mask buffer lifecycle in the execution plan."""
    mode: Literal["recompute", "explicit"]
    # "recompute": Kernels derive mask from stored activations. No mask buffer allocated.
    # "explicit":  Node 4 produces mask buffer. Consuming kernels use it directly.
```

Selection logic (implemented as `PrecisionConfig.mask_strategy` property):

```python
def mask_strategy(self) -> MaskStrategy:
    if self.storage_dtype != self.compute_dtype:
        # Precision boundary destroys derivative information — mask required
        return MaskStrategy(mode="explicit")
    # Mask is fully derivable from stored activations
    return MaskStrategy(mode="recompute")
```

**Conservative behavior for wide-compute configurations.** The binary criterion `storage_dtype != compute_dtype` triggers `explicit` mode for configurations like `PrecisionConfig.mixed_f32_f64()` (FP32 storage, FP64 compute), where the quantization floor is vanishingly small (~1.4×10⁻⁴⁵). In practice, no neural network produces activations within a few ULP of zero in FP64 compute that would then be zeroed by FP32 storage. The explicit mask is therefore technically unnecessary in this configuration but architecturally harmless — the buffer cost is `total_batch_count × padded_hidden_count × sizeof(STORAGE_TYPE)` (FP32, the same footprint as the activation buffer) and the mask writes are branch-free. The binary selection criterion is preferred over a floor-magnitude threshold to avoid introducing a subjective constant and to maintain forward compatibility with activation functions where the floor may be more significant.

### 1.3 Kernel Interface: FLAG Pattern

Following the established precedent of Node 11's `src_scalar_FLAG_use_per_item_norm`, affected kernels receive a FLAG that selects the mask source.

**Node 4 (Producer):**
```c
__global STORAGE_TYPE *dest_buffer_GLOBAL_hidden_mask,  // may be stub
uint out_scalar_FLAG_produce_hidden_mask,               // 0 or 1
```

When `FLAG = 0`, the kernel skips mask writes. The host binds a single-element stub buffer.

**Canonical producer parameter block:**
```c
/**
 * @param dest_buffer_GLOBAL_hidden_mask The derivative mask for hidden layer activations.
 *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
 *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
 *        - Precision Role: "storage"
 *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
 *        - Validation Preconditions: [1] This buffer is written ONLY IF
 *          out_scalar_FLAG_produce_hidden_mask == 1. [2] If the flag is 1,
 *          the Host MUST provide a buffer matching the Calculability Proof.
 *          [3] If the flag is 0, the Host MAY provide a minimal 1-element stub
 *          buffer; the kernel SHALL NOT write to it.
 */
__global STORAGE_TYPE *dest_buffer_GLOBAL_hidden_mask,
```

**Nodes 5, 17, 18 (Consumers):**
```c
__global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,  // may be stub
uint src_scalar_FLAG_use_explicit_hidden_mask,                // 0 or 1
```

When `FLAG = 0`, the kernel derives the mask internally: `mask = load_storage(hidden_activations) > 0`.
When `FLAG = 1`, the kernel reads the mask directly: `mask = load_storage(hidden_mask)`.

**Canonical consumer parameter block:**
```c
/**
 * @param src_buffer_GLOBAL_hidden_mask The derivative mask from Node 4.
 *        - Tensor Shape: (src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count)
 *        - Padding Contract: {Type: CACHE, Formula: "Padded to alignment"}
 *        - Precision Role: "storage"
 *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count, src_scalar_NATURAL_padded_hidden_count]
 *        - Validation Preconditions: [1] This buffer is read ONLY IF
 *          src_scalar_FLAG_use_explicit_hidden_mask == 1. [2] If the flag is 1,
 *          the Host MUST provide a buffer matching the Calculability Proof.
 *          [3] If the flag is 0, the Host MAY provide a minimal 1-element stub
 *          buffer; the kernel SHALL NOT access it.
 */
__global const STORAGE_TYPE *src_buffer_GLOBAL_hidden_mask,
```

### 1.4 Plan-Level Expression

**Under `MaskStrategy(mode="recompute")`:**
```
Node 4 outputs: [hidden_activations]
Node 5 inputs:  [hidden_activations], FLAG=0, mask=stub
Node 17 inputs: [hidden_activations, ...], FLAG=0, mask=stub
Node 18 inputs: [hidden_activations, ...], FLAG=0, mask=stub

BufferDescriptor for hidden_mask: NOT ALLOCATED
```

**Under `MaskStrategy(mode="explicit")`:**
```
Node 4 outputs: [hidden_activations, hidden_mask]
Node 5 inputs:  [hidden_activations, hidden_mask], FLAG=1
Node 17 inputs: [hidden_activations, hidden_mask, ...], FLAG=1
Node 18 inputs: [hidden_activations, hidden_mask, ...], FLAG=1

BufferDescriptor for hidden_mask: {
    producing_node: "forward_pass",
    consumers: {"render_logits_chunk", "backprop_shared_weights_chunk",
                "backprop_shared_biases_chunk"},
    last_consumer: <last of Nodes 17/18 in the streaming loop>,
    role: BATCH_INTERMEDIATE,
    precision_role: "storage"
    # Storage role is safe because the mask's value domain {0.0, 1.0}
    # is exactly representable in every supported format (§1.5).
}
```

**Orthogonality with the Activation Cache/Recompute strategy.** `MaskStrategy` and the existing Cache/Recompute activation lifecycle strategy are independent Policy-tier decisions. All four combinations are valid:

| Activation Strategy | MaskStrategy | Act Phase | Learn Phase |
|:---|:---|:---|:---|
| Cache + explicit | Node 4 → both buffers persist | Nodes 5 reads mask | Nodes 17/18 read cached mask directly |
| Cache + recompute | Node 4 → activations only persist | Node 5 derives internally | Nodes 17/18 derive from cached activations |
| Recompute + explicit | Node 4 → both buffers freed between phases | Node 5 reads mask | Node 4 re-run → both buffers; Nodes 17/18 read regenerated mask |
| Recompute + recompute | Node 4 → activations only; freed between phases | Node 5 derives internally | Node 4 re-run → activations only; Nodes 17/18 derive internally |

Under Recompute + explicit, Node 4 re-executes during the Learn phase with `FLAG = 1`, regenerating both the activation and mask buffers from the stored input. The regenerated mask is bit-identical to the Act-phase mask because the input data and model parameters are unchanged. Both buffers are freed after the last consumer completes.

### 1.5 Architectural Alignment

**Principle §3 (Strategy A):** The divergence between modes is parametric, not structural. Same kernel, same interface, same memory access pattern — just one conditional branch on a FLAG scalar.

**Principle §2 (Primacy of Memory Strategy):** Under `recompute` mode, zero additional memory. Under `explicit` mode, the buffer cost is `total_batch_count × padded_hidden_count × sizeof(STORAGE_TYPE)`, proportional to the activation buffer. The Policy tier makes the optimal choice automatically.

**Exact-Representability Invariant.** The `hidden_mask` buffer uses `precision_role: "storage"` despite carrying derivative information because its value domain `{0.0, 1.0}` is exactly representable in every IEEE floating-point format supported by the architecture:

| Format | `0.0` encoding | `1.0` encoding | Survives store-load cycle? |
|:-------|:---------------|:---------------|:--------------------------|
| FP8 E4M3 | `0x00` | `0x38` (exp=7, mant=0 → 2⁰) | **Yes** |
| FP8 E5M2 | `0x00` | `0x3C` (exp=15, mant=0 → 2⁰) | **Yes** |
| FP16 | `0x0000` | `0x3C00` | **Yes** |
| FP32 | `0x00000000` | `0x3F800000` | **Yes** |
| FP64 | All-zero | `0x3FF0...0` | **Yes** |

Storage narrowing is therefore **lossless** for this buffer — the mask value survives the full `store_storage()` → `load_storage()` cycle without quantization error regardless of `STORAGE_TYPE`. This invariant is the formal prerequisite for the storage precision role assignment and for the correctness of the explicit-mode strategy. Should the architecture extend to non-binary activation functions (e.g., GELU, Swish) where derivative values span a continuous range, this invariant would not hold, and the mask would require `precision_role: "compute"`.

### 1.6 The Fundamental Distinction: `hidden_mask` vs. `sample_mask`

The reason `hidden_mask` is added to Nodes 17/18 while `sample_mask` is NOT added for correctness is a mathematical property, not a design preference:

**`hidden_mask` encodes information that CAN be destroyed by precision boundaries:**
```
Node 4 (COMPUTE_TYPE):
    pre_relu = W*x + b = 0.0005        // positive, genuinely active
    post_relu = max(0, 0.0005) = 0.0005
    mask = (0.0005 > 0) = 1            // TRUE — unit is active

Node 4 (store to STORAGE_TYPE):
    store_storage(hidden_activations, ..., 0.0005)  →  0.0  (FP8 E4M3 floor = 0.002)
    store_storage(hidden_mask, ..., 1.0)            →  1.0  (exact in all formats, §1.5)
```
A consumer that recomputes the mask from stored activation sees `0.0 > 0 = false` — the **wrong answer**. The explicit mask preserves the truth.

**`sample_mask` encodes information that CANNOT be destroyed by precision boundaries:**
```
Node 4 (COMPUTE_TYPE):
    pre_relu = W*x + b = 3.7           // some value for padding sample
    post_relu = max(0, 3.7) = 3.7
    mask_b = load_sample_mask(...) = 0  // padding sample
    masked = 3.7 * (COMPUTE_TYPE)0 = 0.0  // exactly zero

Node 4 (store to STORAGE_TYPE):
    store_storage(hidden_activations, ..., 0.0)  →  0.0  (exact in EVERY format)
```
Zero is perfectly representable in every IEEE floating-point encoding. A zeroed activation **stays zeroed** through every subsequent load, computation, and store.

**Scope of explicit-mode recovery.** The explicit mask recovers the binary *derivative indicator* — whether a hidden unit was active — but not the activation *magnitude*. This distinction produces an asymmetry across gradient paths: the shared-layer backpropagation (Nodes 17/18), which needs only the binary ReLU gate, is fully corrected; the module-layer gradient (Node 8), which reads the stored activation value directly for its outer-product computation, remains subject to quantization-floor sparsification. This asymmetry is fully analyzed in §3.2.

| Property | `hidden_mask` | `sample_mask` |
|----------|--------------|---------------|
| What it encodes | `activation > 0` at compute precision | Sample validity (boolean gate) |
| Effect on data | Sets derivative to 0 or 1 | Zeros the activation entirely |
| Survives storage narrowing? | **No** — sub-floor positives round to zero | **Yes** — zero is exact in all formats |
| Recoverable from stored data? | **Not always** | **Always** — zero is unambiguous |
| Role in Nodes 17/18 | **Correctness** — preserves non-derivable derivative truth | **Performance** — early exit avoids computing zero contributions |

---

## Part II: Sample Mask Bitmask Encoding

### 2.1 Problem Statement

The existing `sample_mask` buffer is declared as:
```c
__global const STORAGE_TYPE *src_buffer_GLOBAL_sample_mask
```

This commits a category error: `sample_mask` is a **discrete boolean signal** (valid/invalid), not a floating-point quantity subject to bandwidth-precision tradeoffs. Assigning it a `precision_role` of `"storage"` forces:

1. Allocation size to scale with `sizeof(STORAGE_TYPE)` — 1 byte under FP8, 4 bytes under FP32
2. Every consuming kernel to pay `load_storage()` conversion overhead
3. `PrecisionConfig` coupling for a buffer that has no precision semantics

The mask conveys 1 bit of information per sample. Encoding it as a float wastes ≥7 bits per sample under FP8 and ≥31 bits under FP32.

### 2.2 Solution: Bitmask Encoding

Replace the per-sample float buffer with a packed `uint` bitmask:

- **32 samples per `uint` word**, LSB-first
- Buffer size: `ceil(total_batch_count / 32) × 4` bytes (fixed, `PrecisionConfig`-independent)
- Bits beyond `total_batch_count` are zero-padded by the host

**Device-side access utility:**
```c
static inline uint load_sample_mask(
    __global const uint *mask_words, uint sample_index)
{
    return (mask_words[sample_index >> 5u] >> (sample_index & 31u)) & 1u;
}
```

This compiles to 3 integer ALU operations. No cache miss penalty — the same word serves 32 adjacent samples.

### 2.3 Per-Kernel Changes

**Per-kernel `sample_mask` changes:**

| Node | Kernel | Change |
|------|--------|--------|
| 4 | `forward_pass` | `STORAGE_TYPE *` → `uint *` |
| 5 | `render_logits_chunk` | **New parameter** (`uint *` bitmask, early exit) |
| 6 | `compute_probs_loss_cce_chunk` | `STORAGE_TYPE *` → `uint *` |
| 7 | `compute_probs_loss_bce_chunk` | `STORAGE_TYPE *` → `uint *` |
| 8 | `calculate_module_param_grads_chunk` | `STORAGE_TYPE *` → `uint *` |
| 9 | `backprop_error_to_hidden_chunk` | `STORAGE_TYPE *` → `uint *` |
| 10 | `calculate_chunk_temp_gradients` | `STORAGE_TYPE *` → `uint *` |
| 17 | `backprop_shared_weights_chunk` | `STORAGE_TYPE *` → `uint *` (early exit) |
| 18 | `backprop_shared_biases_chunk` | `STORAGE_TYPE *` → `uint *` (early exit) |

**Canonical parameter block:**
```c
/**
 * @param src_buffer_GLOBAL_sample_mask Bitmask defining the validity (1) or
 *        padding (0) status of each sample in the batch.
 *        - Tensor Shape: ((src_scalar_NATURAL_total_batch_count + 31) / 32)
 *          expressed as uint words, each encoding 32 samples LSB-first.
 *        - Padding Contract: {Type: UNPADDED, Formula: "Bits beyond
 *          total_batch_count are zero-padded by the host."}
 *        - Calculability Proof: [src_scalar_NATURAL_total_batch_count]
 *        - Validation Preconditions: Host shall allocate exactly
 *          [((src_scalar_NATURAL_total_batch_count + 31) / 32) * sizeof(uint)]
 *          bytes. Accessed via load_sample_mask(); no precision role applies.
 */
__global const uint *src_buffer_GLOBAL_sample_mask,
```

**Note:** No `Precision Role` — integer-typed buffers are exempt per CONTRACT.md Article 3.

### 2.4 Node 5 Extension

Node 5 currently computes a full `hidden_dim × class_dim` dot product for padding samples, producing logits that Nodes 6/7 immediately discard. With the bitmask, a sample-level skip check is negligible.

**Behavioral Invariant additions:**

> "Sample-Level Early Exit: When `load_sample_mask()` returns 0 for a sample, the kernel writes zero (via `store_storage()`) to the logit output for all classes in the current chunk and skips the dot product. This is an unconditional optimization — not FLAG-gated — because the bitmask is always available and the check is trivially cheap."

> "Hidden-Mask-Aware Sparsity Optimization: When `src_scalar_FLAG_use_explicit_hidden_mask == 1`, the existing sparsity-aware dot product reads the explicit mask buffer as a branch predicate to elide dot-product terms corresponding to ReLU-zeroed hidden units. When the flag is `0`, the implementation MAY derive the predicate from `load_storage(hidden_activations) > 0` or MAY omit the per-dimension sparsity check entirely — both approaches produce identical results for ReLU activations, since zero-activation dimensions contribute zero to the dot product regardless. The choice between recomputed predicate and unconditional evaluation is an Execution-tier concern. When a sample is already skipped by the sample-level early exit, the per-dimension sparsity check is never reached."

### 2.5 Host-Side Obligations

**Bitmask packing:**
```python
def pack_sample_mask(mask_bool: np.ndarray) -> np.ndarray:
    """Pack boolean mask into uint32 bitmask, 32 samples/word, LSB-first."""
    n = len(mask_bool)
    n_words = (n + 31) // 32
    padded = np.zeros(n_words * 32, dtype=np.uint8)
    padded[:n] = mask_bool.astype(np.uint8)
    words = np.zeros(n_words, dtype=np.uint32)
    for bit in range(32):
        words |= padded[bit::32].astype(np.uint32) << bit
    return words
```

**`effective_batch_size` computation (for Node 21):**
```python
def compute_effective_batch_size(packed_mask: np.ndarray) -> int:
    """Exact popcount over packed bitmask words."""
    return sum(bin(int(w)).count('1') for w in packed_mask)
```

**Buffer allocation:**
```python
n_words = (total_batch_count + 31) // 32
mask_buf = backend.allocate(n_words * 4)  # PrecisionConfig-independent
```

### 2.6 Nodes 17/18: `sample_mask` Retained for Early Exit

The masking effect in Nodes 17/18 is indelibly encoded in upstream data through two independent, precision-proof invariants. The `sample_mask` is therefore **not required for correctness** — but it provides a valuable **early-exit optimization** that avoids computing zero contributions for padding samples.

#### 2.6.1 Correctness Proof: `sample_mask` Is Mathematically Redundant

**Invariant 1: `summed_grad_h[b][h] = 0` for masked samples**

```
Node 9:   partial_grad_h[b][h] = f(probs, targets, weights) × load_sample_mask(b)
          load_sample_mask(b) = 0  (integer, exact)
          → partial_grad_h = 0.0   (exact in COMPUTE_TYPE)
          store_storage(..., 0.0)   → 0.0 in STORAGE_TYPE (exact)

Node 11:  load_storage(0.0) = 0.0  (exact)
          clip(0.0) = 0.0           (exact)
          store_storage(0.0) = 0.0  (exact)

Node 13:  load_storage(0.0) = 0.0  (exact)
          sum of zeros = 0.0        (exact)
          store_storage(0.0) = 0.0  (exact)

Node 16:  load_storage(0.0) = 0.0  (exact)
          Σ(zeros) = 0.0            (exact, written to COMPUTE_TYPE buffer)

→ summed_grad_h[b][h] = 0.0 exactly, regardless of PrecisionConfig ✓
```

**Invariant 2: `relu_derivative[b][h] = 0` for masked samples**

```
Node 4:   masked_activation = post_relu × (COMPUTE_TYPE)0 = 0.0
          store_storage(hidden_activations, ..., 0.0)  → 0.0 (exact in all formats)

Node 17:  (Recompute mode)
            load_storage(hidden_activations, ...) = 0.0
            relu_mask = (0.0 > 0) = false → derivative = 0.0  ✓

          (Explicit hidden_mask mode)
            load_storage(hidden_mask, ...) = 0.0  (Node 4 wrote 0.0 for masked samples)
            derivative = 0.0  ✓
```

**The Double Guarantee:**
```
contribution = input[b][i] × relu_derivative[b][h] × summed_grad_h[b][h]
                              ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^
                              = 0 (invariant 2)         = 0 (invariant 1)
```

Two independent factors are exactly zero. An implementation that omits the early exit and computes the full contribution for padding samples produces identical results.

#### 2.6.2 The Performance Case for Retention

Without the early exit, Node 17's inner loop for a masked sample `b` and output element `(i, h)` still executes:

1. **Load** `summed_grad_h[b][h]` from global memory (COMPUTE_TYPE)
2. **Load** `hidden_activations[b][h]` through `load_storage()` (potential FP8 LUT lookup)
3. **Load** `input[b][i]` through `load_storage()` (same)
4. **Compute** the comparison and two multiplications
5. **Add** zero to the accumulator

That is three global memory loads and several ALU operations per masked sample per output element. Node 17 produces `input_dim × hidden_dim` output elements, so the wasted work per masked sample is `3 × input_dim × hidden_dim` global loads plus `input_dim × hidden_dim` multiply-adds. Node 18 is lighter (`hidden_dim` elements) but the same principle applies.

With the bitmask early exit, the total cost is 3 integer ALU ops and one branch per masked sample — amortized further because consecutive work-items in a warp read the same `uint` word. The savings scale with padding fraction:

| Scenario | Batch Size | Padded To | Padding % | Node 17 Savings |
|----------|-----------|-----------|-----------|------------------|
| Iris (small) | 120 | 128 | 6.3% | Modest |
| Power-of-2 alignment | 33 | 64 | 48.4% | **Nearly half** |
| Streaming chunk tail | 1000, chunk=64 | last chunk: 40/64 | 37.5% | Significant |
| Data Tsunami | 65536 | 65536 | 0% | None |

The streaming model (Model B) makes this worse, not better: the batch is split into `num_batch_chunks` chunks, and the last chunk is often short. If `batch_size = 1000` and `chunk_size = 64`, the last chunk has 40 real samples and 24 padding samples — 37.5% wasted computation per streaming iteration.

The buffer dependency cost is negligible: `ceil(total_batch_count / 32) × 4` bytes. For 1000 samples, 128 bytes — smaller than a cache line. The entire bitmask fits in L1 for any practical batch size.

This is the same justification used for adding `sample_mask` to Node 5 (§2.4): not correctness, but early exit.

| Node | Mask Justification | Essential for Correctness? | Essential for Performance? |
|------|-------------------|---------------------------|---------------------------|
| 4 | Zeros activations early | **Yes** — downstream depends on it | Yes |
| 5 | Skips dot product for padding rows | No — logits = bias (zeroed by Nodes 6/7) | **Yes** — saves `H×C` ops |
| 6, 7 | Zeros prob and loss for padding | **Yes** — target term is mask-independent | Yes |
| 8, 9, 10 | Zeros gradient contribution | **Yes** — target term is mask-independent | Yes |
| 17, 18 | Skips outer product for padding rows | No — both factors are zero | **Yes** — saves `I×H` or `H` ops |

Nodes 5, 17, and 18 form a coherent group: the mask is present for early exit, not correctness.

#### 2.6.3 Behavioral Invariants (Nodes 17 and 18)

> *"Precision Boundary Conversion: storage-role inputs widened via load_storage(); compute-role gradient consumed directly; partial gradient outputs narrowed via store_storage(). Integer-typed sample_mask accessed via load_sample_mask(). Intra-workgroup reduction in LOCAL COMPUTE_TYPE scratch. All arithmetic exclusively in COMPUTE_TYPE.*
>
> *ReLU derivative source is controlled by `src_scalar_FLAG_use_explicit_hidden_mask`. When 0: mask is derived internally from stored activations (`mask = activation > 0`). When 1: mask is read from `src_buffer_GLOBAL_hidden_mask`. The Host MAY pass a minimal stub buffer when the flag is 0.*
>
> *Sample-Level Early Exit: When `load_sample_mask()` returns 0 for a sample, the kernel skips the entire contribution for that sample. This is a performance optimization — not a correctness requirement. Both upstream invariants (summed_grad_h = 0 from Node 9's masking, relu_derivative = 0 from Node 4's activation zeroing) independently guarantee zero contribution for masked samples regardless of whether the early exit is applied. Implementations that omit this optimization and compute the full contribution for padding samples produce identical results, differing only in performance."*

**Streaming Loop Dependency Summary:**

| Buffer | Present in Nodes 17/18? | Justification |
|--------|-------------------------|---------------|
| `input` | Yes | Primary data |
| `hidden_activations` | Yes | Primary data + recompute-mode derivative source |
| `hidden_mask` | Conditional (FLAG) | Non-derivable derivative truth when STORAGE < COMPUTE |
| `summed_grad_h` (slice) | Yes | Primary upstream gradient |
| `sample_mask` | **Yes** | Early-exit optimization — avoids computing provably-zero contributions for padding samples |

---

## Part III: Contract Amendments

### 3.1 CONTRACT.md Article 8 — Lexicon (Group 6)

**Replace `sample_mask` entry:**
> `sample_mask`: A bitmask defining the validity (`1`) or invalidity/padding (`0`) of each sample in a batch. Encoded as `uint` words with 32 samples per word, LSB-first. This buffer is integer-typed and exempt from the three-role precision model. Device-side access uses the `load_sample_mask()` utility function. The host computes `effective_batch_size` via popcount over the mask words.

**Replace `hidden_mask` entry:**
> `hidden_mask`: A derivative mask for the hidden layer activation function. When the mask strategy is `explicit`, this buffer is produced by the forward pass kernel (Node 4) at compute precision and stored at storage precision, preserving derivative information that may be lost during activation storage narrowing. The storage precision role is safe because the mask's value domain `{0.0, 1.0}` is exactly representable in every IEEE floating-point format supported by the architecture (see ADR-031 §1.5). When the mask strategy is `recompute`, consuming kernels derive the mask from stored activations internally. The mask strategy is a Policy-tier decision based on the active `PrecisionConfig`.

### 3.2 CONCEPT.md §2 — FP8 Quantization Floor

**Replace existing passage with:**
> **Storage narrowing, quantization floor, and mask strategy.** When `storage_dtype` is narrower than `compute_dtype`, the storage format introduces a quantization floor at its minimum positive subnormal. Activations that are positive at compute precision but below this floor are stored as zero, erasing the derivative information. The magnitude of the floor depends on the storage format:
>
> | Format | Min Positive Subnormal | Practical Impact |
> |--------|----------------------|------------------|
> | FP8 E4M3 | 2⁻⁹ ≈ 0.00195 | Significant — affects real activation distributions |
> | FP8 E5M2 | 2⁻¹⁶ ≈ 1.53×10⁻⁵ | Moderate — affects small activations |
> | FP16 | 2⁻²⁴ ≈ 5.96×10⁻⁸ | Negligible — below practical activation range |
> | FP32 (vs. FP64 compute) | 2⁻¹⁴⁹ ≈ 1.4×10⁻⁴⁵ | Effectively nonexistent |
>
> The Policy tier automatically selects the `explicit` mask strategy when `storage_dtype != compute_dtype`: Node 4 writes the `hidden_mask` buffer capturing the compute-precision derivative truth (1.0 for active units, 0.0 for ReLU-zeroed) before the activation undergoes storage narrowing. The mask's value domain `{0.0, 1.0}` is exactly representable in all supported formats (ADR-031 §1.5), so storage narrowing is lossless for the mask itself.
>
> Consuming kernels (Nodes 5, 17, 18) read this mask directly via the `src_scalar_FLAG_use_explicit_hidden_mask` flag. When `storage_dtype == compute_dtype` (e.g., `PrecisionConfig.float32()`), the Policy tier selects the `recompute` strategy: the mask is bit-identical to `activation > 0` anyway, so no mask buffer is allocated and consumers derive the mask internally.
>
> **Asymmetric gradient recovery.** The explicit mask recovers the binary *derivative indicator* — whether a hidden unit was active at compute precision — but not the activation *magnitude*. This produces an asymmetry across gradient paths:
>
> - **Shared-layer backpropagation (Nodes 17/18):** The ReLU derivative is a binary 0/1 gate in the chain rule (`dL/dW_shared = input × relu_deriv × grad_h`). The explicit mask provides exactly this gate, and gradient flow is fully corrected for sub-floor activations.
> - **Module-layer gradients (Node 8):** The outer-product computation reads the stored activation value directly (`dL/dW_mod = hidden × error`). When a sub-floor activation is stored as zero, its gradient contribution is zeroed regardless of the mask — the explicit mask preserves *whether* a unit was active, not *how active* it was. This produces an effect equivalent to stochastic sparsification and is acceptable for the storage role: it does not affect convergence for typical activation distributions, and its magnitude correlates with the quantization floor (significant for FP8 E4M3, negligible for FP16 and below).
>
> This asymmetry is inherent to storing activations at narrower-than-compute precision. The explicit mask mitigates the effect where it can (binary gates) and accepts it where it cannot (magnitude-dependent outer products).

### 3.3 CONCEPT.md §11 — Host Orchestrator

**Add under Memory Assessment & Chunk Definition:**
> **Sample mask packing.** The host packs the per-sample boolean mask into a `uint` bitmask buffer (32 samples per word, LSB-first) before device upload. The `effective_batch_size` scalar for Node 21 is derived via integer popcount over the packed words, yielding an exact count for any batch size up to 2³².

**Add under Activation Lifecycle & Streaming:**
> **Mask strategy selection.** The Policy tier evaluates `PrecisionConfig.mask_strategy` at plan-construction time and propagates the result as FLAG scalars on Nodes 4, 5, 17, and 18. Under `recompute` mode, the `hidden_mask` BufferDescriptor is omitted from the plan entirely — no allocation, no lifecycle tracking. Under `explicit` mode, the buffer is allocated and tracked as a `BATCH_INTERMEDIATE` with its `last_consumer` set to the final streaming loop iteration's Node 17 or 18 dispatch. The mask strategy is orthogonal to the activation Cache/Recompute strategy; all four combinations produce correct results (see ADR-031 §1.4).

---

## Part IV: Cross-Backend Implementation

| Backend | `load_sample_mask()` | `MaskStrategy` FLAG |
|---------|---------------------|---------------------|
| **OpenCL** | 3 integer ALU ops; `uint` is built-in | Standard `uint` scalar parameter |
| **Vulkan/GLSL** | Identical semantics; `uint` is core GLSL | `uint` push constant or UBO member |
| **CPU/C** | `uint32_t` shift/mask; `__builtin_popcount()` for host | Standard `uint32_t` argument |

Both abstractions are backend-neutral by construction — they use only integer arithmetic and type-generic scalar parameters.

---

## Migration Checklist

| Step | Scope | Description |
|------|-------|-------------|
| 1 | `precision_config.py` | ✅ Add `MaskStrategy` dataclass; add `mask_strategy` property to `PrecisionConfig` |
| 2 | `__init__.py` | ✅ Export `MaskStrategy` |
| 3 | `kernels.cl.h` | ✅ Add `dest_buffer_GLOBAL_hidden_mask` (with stub validation), `out_scalar_FLAG_produce_hidden_mask` to Node 4 |
| 4 | `kernels.cl.h` | ✅ Add `src_buffer_GLOBAL_hidden_mask` (with stub validation), `src_scalar_FLAG_use_explicit_hidden_mask` to Nodes 5, 17, 18 |
| 5 | `phase_1_act.cl.c` | ✅ Implement FLAG-conditional mask production (Node 4) and consumption with sparsity behavior (Node 5) |
| 6 | `phase_2_learn_D_backprop.cl.c` | ✅ Implement FLAG-conditional mask consumption (Nodes 17, 18) |
| 7 | `kernels.cl.h` | ✅ Add `load_sample_mask()` utility function |
| 8 | `kernels.cl.h` | ✅ Retype `sample_mask` to `uint *` in Nodes 4, 6, 7, 8, 9, 10, 17, 18 |
| 9 | `kernels.cl.h` | ✅ Add `sample_mask` parameter (`uint *` bitmask) to Node 5 |
| 10 | Kernel implementations | ✅ Update all `.cl.c` files for bitmask access pattern |
| 11 | `CONTRACT.md` | ✅ Update `hidden_mask` lexicon entry |
| 12 | `CONTRACT.md` | ✅ Update `sample_mask` lexicon entry |
| 13 | `CONCEPT.md` | ✅ Update FP8 quantization floor passage (including asymmetric gradient note) |
| 14 | `CONCEPT.md` | ✅ Update DAG edges for Node 5; add host packing and mask strategy notes to §11 |
| 15 | `DESIGN.md` | ✅ Add `mask_strategy` to PrecisionConfig section |
| 16 | Host layer | ✅ Implement `pack_sample_mask()` and popcount-based `effective_batch_size` |
| 17 | Per-backend sources | ✅ Update Vulkan `.glsl` and CPU `.c` implementations |
| 18 | Test fixtures | ✅ Update to provide packed bitmask inputs |

(✅ = completed in initial implementation)

---

## Rejected Alternatives

### Hidden Mask: Always Explicit
Allocating the mask under all configurations wastes memory when `storage_dtype == compute_dtype`. The FLAG pattern costs one scalar parameter and a trivial branch vs. `hidden_count × batch_count × sizeof(STORAGE_TYPE)` bytes.

### Hidden Mask: Always Recompute
Incorrect under precision-boundary configurations. Silent gradient corruption for sub-floor activations.

### Hidden Mask: Floor-Magnitude Threshold
Instead of the binary `storage_dtype != compute_dtype` criterion, use a numerical threshold on the quantization floor (e.g., enable explicit mode only when the floor exceeds 10⁻⁶). Rejected because it introduces a subjective constant that must be justified against diverse activation distributions. The binary criterion is simple, conservative, forward-compatible, and imposes zero cost under `recompute` mode. The overhead under `explicit` mode for configurations where the floor is negligible (e.g., `mixed_f32_f64`) is proportional to the activation buffer and architecturally harmless (see §1.2).

### Sample Mask: Keep Float Encoding
Maintains the category error. Forces `PrecisionConfig` coupling for a boolean signal. Wastes 31–63 bits per sample depending on storage dtype.

### Sample Mask: Byte Array
Uses 8× more memory than bitmask. Still wastes 7 bits per sample. No meaningful simplification — byte indexing is not simpler than `>> 5` and `& 31`.

### Sample Mask in Nodes 17/18: Remove for Purity
Removing `sample_mask` from Nodes 17/18 on the grounds that it is mathematically redundant would conflate correctness with computational cost. While the double-zero guarantee (§2.6.1) makes the mask unnecessary for producing correct results, omitting the early exit forces three global memory loads and arithmetic per masked sample per output element — wasted work proportional to `padding_fraction × input_dim × hidden_dim` per streaming chunk. The bitmask encoding makes the check nearly free (3 integer ALU ops, one coalesced word load per 32 samples), making the cost-benefit ratio decisive in favor of retention.

---

## References

- ADR-008: Precision Configuration
- ADR-020: Mixed-Precision Execution Model
- ADR-021: Kernel Precision-Role Migration
- ADR-025: FP8 Support
- CONCEPT.md §3: Modular, "Dumb" Kernels (Strategy A vs. Strategy B)
