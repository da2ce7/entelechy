### **System Contract: Host-Device Kernel Interface (Revision 1)**

#### **Preamble**

This document constitutes the definitive and inviolable contract governing all interactions across the Host-Device boundary. Its articles are not guidelines but fundamental laws of the system architecture. Adherence is mandatory and absolute. This contract supersedes all prior conventions and informal agreements.

---

### **Article 1: Foundational Axioms**

The architecture is immutably founded upon two axioms, which are the basis for all subsequent articles.

- **1.1. Axiom of Jurisdictional Separation.** A parameter's syntactic structure (**Name**) defines its machine-enforced contract. A parameter's semantic block (**Commentary**) defines its human-verifiable and logical contract. These two jurisdictions are distinct and exhaustive.
- **1.2. Axiom of Semantic Uniqueness.** Information encoded within the syntactic jurisdiction (Name) is prohibited from being duplicated within the semantic jurisdiction (Commentary), and vice versa. There shall exist no redundancy between the two.

### **Article 2: Parameter Lexical Mandate**

The structure of a parameter name is formally specified. This grammar is not a convention but a mandatory syntactical requirement for interface validity.

**2.1. Buffer Name Grammar**
A buffer identifier shall be constructed as:
`[Flow] :: "buffer" :: [MemoryScope] :: [ContextAndUsage]`

- **`[Flow]`**: `src_` | `dest_` | `update_` | `sync_`
- **`[MemoryScope]`**: `GLOBAL_` | `LOCAL_` | `CONST_`
- **`[ContextAndUsage]`**: A canonical identifier defined exclusively in **Appendix B: Canonical Lexicon**.

**2.2. Scalar Name Grammar**
A scalar identifier shall be constructed as:
`[Flow] :: "scalar" :: [NumberType] :: [ContextAndUsage]`

- **`[Flow]`**: `src_` | `dest_`
- **`[NumberType]`**: A mandatory prefix defining the parameter's abstract numerical domain.
- **`[ContextAndUsage]`**: A canonical identifier defined exclusively in **Appendix B: Canonical Lexicon**.

**2.3. Scalar `[NumberType]` Taxonomy**
The `[NumberType]` component defines the set of valid values for a scalar. Any value outside the specified domain constitutes a contract violation.

| Token      | Abstract Numerical Domain            | Core Constraint                                                      |
| :--------- | :----------------------------------- | :------------------------------------------------------------------- |
| `NATURAL_` | Natural Numbers (ℕ₀: {0, 1, 2, ...}) | Value must be non-negative. Defines counts, indices, and sizes.      |
| `INTEGER_` | Integers (ℤ: {..., -1, 0, 1, ...})   | Value may be negative. Defines offsets and signed quantities.        |
| `REAL_`    | Real Numbers (ℝ)                     | Value represents a continuous quantity. Defines thresholds and data. |
| `FLAG_`    | Boolean Set ({0, 1})                 | Value must be `0` or `1`. Defines logical switches.                  |

### **Article 3: Parameter Commentary Contract**

The `@param` block constitutes the complete logical specification for a parameter.

- **3.1. Buffer Commentary.** For any parameter of type Buffer, the commentary block **shall** fully specify the Tensor Contract, including its Shape, Padding, Calculability Proof, and Validation Preconditions. No other location may define these properties.
- **3.2. Scalar Commentary.** For any parameter of type Scalar, the commentary **shall** provide a single, declarative statement of its logical purpose.

---

### **Appendix A: Canonical Interface Instantiation**

The following formal notation illustrates the sole valid method for specifying a kernel interface in adherence to this contract. This is a prescriptive template, not an illustrative example.

```c
/**
 * @brief Performs [function_name] operation.
 *
 * @param src_buffer_GLOBAL_input_stream The primary data source for the computational unit.
 *        - Tensor Shape: (src_scalar_NATURAL_total_item_count)
 *        - Padding Contract: {Type: CACHE, Formula: Post-pad to 128-byte alignment}
 *        - Calculability Proof: [src_scalar_NATURAL_total_item_count]
 *        - Validation Preconditions: [src_scalar_NATURAL_item_offset + src_scalar_NATURAL_item_count <= src_scalar_NATURAL_total_item_count]
 *
 * @param src_buffer_CONST_lookup_table A read-only, constant-memory data resource.
 *        - Tensor Shape: (LUT_CAPACITY)
 *        - Padding Contract: None.
 *        - Calculability Proof: [Compile-time constant: LUT_CAPACITY]
 *        - Validation Preconditions: None.
 *
 * @param dest_buffer_GLOBAL_partial_results The sole collection resource for this unit's partial output.
 *        - Tensor Shape: (dest_scalar_NATURAL_total_chunks, RESULT_ELEMENTS_PER_CHUNK)
 *        - Padding Contract: None.
 *        - Calculability Proof: [dest_scalar_NATURAL_total_chunks, Compile-time constant: RESULT_ELEMENTS_PER_CHUNK]
 *        - Validation Preconditions: Host shall zero-initialize this buffer prior to dispatch.
 *
 * @param update_buffer_LOCAL_reduction_tile A work-group exclusive memory resource for intra-group reductions.
 *        - Tensor Shape: (WORK_GROUP_SIZE + BANK_PADDING)
 *        - Padding Contract: {Type: BANK, Formula: + BANK_PADDING}
 *        - Calculability Proof: [Launch-time parameter: get_local_size(0), Compile-time constant: BANK_PADDING]
 *        - Validation Preconditions: Host interaction is prohibited.
 *
 * @param sync_buffer_GLOBAL_atomic_counter A global resource for cross-group atomic synchronization.
 *        - Tensor Shape: (1)
 *        - Padding Contract: None.
 *        - Calculability Proof: [Implicit size: atomic_uint]
 *        - Validation Preconditions: Host shall initialize this resource to 0.
 *
 * @param src_scalar_NATURAL_item_offset Specifies the physical element offset for the read window.
 * @param src_scalar_NATURAL_item_count Specifies the logical element count for the read window.
 * @param src_scalar_NATURAL_total_item_count Specifies the total logical element count of the source stream.
 * @param src_scalar_REAL_processing_threshold Defines the real-valued threshold for the filtering operation.
 * @param dest_scalar_NATURAL_output_chunk_index Defines the logical index for placement of the output chunk.
 * @param dest_scalar_NATURAL_total_chunks Defines the total number of chunks in the decomposition of the output space.
 */
__kernel void illustrative_kernel_name(
    __global const SCALAR_TYPE* src_buffer_GLOBAL_input_stream,
    __constant const SCALAR_TYPE* src_buffer_CONST_lookup_table,
    __global SCALAR_TYPE* dest_buffer_GLOBAL_partial_results,
    __local SCALAR_TYPE* update_buffer_LOCAL_reduction_tile,
    __global atomic_uint* sync_buffer_GLOBAL_atomic_counter,

    uint src_scalar_NATURAL_item_offset,
    uint src_scalar_NATURAL_item_count,
    uint src_scalar_NATURAL_total_item_count,
    float src_scalar_REAL_processing_threshold,
    uint dest_scalar_NATURAL_output_chunk_index,
    uint dest_scalar_NATURAL_total_chunks
);
```

### **Appendix B: Canonical Lexicon for `[ContextAndUsage]`**

#### **1.0 Mandate**

This Lexicon establishes the sole and binding semantic definitions for the `[ContextAndUsage]` component of any parameter name. Usage of any term not explicitly defined herein is a violation of the contract. The semantics of these terms are fixed. Amendments to this Lexicon require formal review and ratification by the Architecture Governance Committee.

#### **2.0 Core Data Role Primitives**

| Term                 | Definition                                                                            |
| :------------------- | :------------------------------------------------------------------------------------ |
| `input`              | The initial, untransformed data set for a complete computation.                       |
| `weights`            | The set of learnable weight parameters for a model layer.                             |
| `biases`             | The set of learnable bias parameters for a model layer.                               |
| `hidden_activations` | The post-activation output tensor of an intermediate system layer.                    |
| `logits`             | The pre-activation, real-valued output tensor of the final system layer.              |
| `probs`              | The post-activation, normalized probability tensor of the final system layer.         |
| `grad`               | The gradient tensor derived from a specified parameter.                               |
| `sample_mask`        | A tensor whose elements define the validity (`1`) or padding (`0`) status of samples. |
| `temps`              | The set of learnable temperature parameters for logit scaling.                        |

#### **3.0 Decomposition Strategy Primitives**

| Term          | Definition                                                                                             |
| :------------ | :----------------------------------------------------------------------------------------------------- |
| `flat_tile`   | A unit of work derived from the flattening of a logical multi-dimensional grid to a 1D dispatch index. |
| `batch_chunk` | A contiguous 1D partition of the primary batch dimension.                                              |

#### **4.0 Scalar Context Modifiers**

| Type   | Term      | Function                                                                                                                                                     |
| :----- | :-------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prefix | `total_`  | Denotes the total logical cardinality of a dimension for the complete problem space.                                                                         |
| Prefix | `padded_` | Denotes the physical, in-memory cardinality of a dimension, inclusive of any padding.                                                                        |
| Prefix | `num_`    | Denotes the number of discrete partitions into which a dimension has been decomposed.                                                                        |
| Suffix | `_index`  | Denotes a **logical**, ordinal position within a conceptual sequence or grid, independent of physical memory representation. Defines a work unit's identity. |
| Suffix | `_offset` | Denotes a **physical** displacement, in elements, specifying a read/write start position. Used for direct memory address calculation.                        |
| Suffix | `_count`  | The number of elements to process, typically relative to a corresponding `_offset`.                                                                          |

#### **5.0 Domain and Utility Primitives**

| Domain             | Term            | Type            | Definition                                                                       |
| :----------------- | :-------------- | :-------------- | :------------------------------------------------------------------------------- |
| Specialized Layout | `_simd_major`   | Suffix          | Specifies a Struct-of-Arrays (SoA) memory layout aligned to SIMD vector width.   |
| Optimizer State    | `m1`            | Data Role       | The first moment vector.                                                         |
| Optimizer State    | `m2`            | Data Role       | The second moment vector.                                                        |
| Optimizer State    | `learning_rate` | Hyperparameter  | The optimizer step size.                                                         |
| Optimizer State    | `beta1`         | Hyperparameter  | The exponential decay rate for `m1`.                                             |
| Optimizer State    | `beta2`         | Hyperparameter  | The exponential decay rate for `m2`.                                             |
| Optimizer State    | `epsilon`       | Hyperparameter  | The term for preventing division by zero.                                        |
| Matrix Navigation  | `_leading_dim`  | Suffix (Scalar) | The physical stride, in elements, between the start of consecutive rows/columns. |
| Constraint Value   | `min_value`     | Scalar Context  | The inclusive minimum boundary for a value.                                      |
| Constraint Value   | `max_value`     | Scalar Context  | The inclusive maximum boundary for a value.                                      |
