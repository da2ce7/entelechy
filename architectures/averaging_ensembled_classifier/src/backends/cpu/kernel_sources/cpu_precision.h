/* cpu_precision.h — Two-axis multi-precision instantiation macros (ADR-008, ADR-023 §2.2).
 *
 * Included by .inc template files. The including .c file must define:
 *
 *   STORAGE_T           — storage-role type (float, _Float16)
 *   STORAGE_SUFFIX      — storage token suffix (fp32, fp16)
 *   STATE_T             — state-role type (float, _Float16)
 *   STATE_SUFFIX        — state token suffix (fp32, fp16)
 *   PRECISION_SUFFIX    — combined suffix for symbol names (s32x32, s16x16, s16x32)
 *
 * Provides:
 *   PREC_FN(name)       — paste suffix onto function name
 *   PREC_TY(name)       — paste suffix onto type name
 *   PREC_SIZEOF_STORAGE — sizeof(STORAGE_T) as integer constant
 *   PREC_SIZEOF_STATE   — sizeof(STATE_T) as integer constant
 *
 * Storage-role load/store (keyed on STORAGE_SUFFIX):
 *   simd_load_storage(p)   — SIMD load converting STORAGE_T* → simd_float
 *   simd_store_storage(p,v)— SIMD store converting simd_float → STORAGE_T*
 *   scalar_load_storage(p,i) — scalar load converting STORAGE_T[i] → float
 *   scalar_store_storage(p,i,v) — scalar store converting float → STORAGE_T[i]
 *
 * State-role load/store (keyed on STATE_SUFFIX):
 *   simd_load_state(p)     — SIMD load converting STATE_T* → simd_float
 *   simd_store_state(p,v)  — SIMD store converting simd_float → STATE_T*
 *   scalar_load_state(p,i) — scalar load converting STATE_T[i] → float
 *   scalar_store_state(p,i,v) — scalar store converting float → STATE_T[i]
 *
 * Compute-role buffers (always float*) use direct access — no wrapper needed.
 */
#ifndef CPU_PRECISION_H_MACROS
#define CPU_PRECISION_H_MACROS

/* Two-level token paste for macro expansion */
#define _PREC_CAT(a, b)       a##_##b
#define _PREC_CAT2(a, b)      _PREC_CAT(a, b)

#define PREC_FN(name)          _PREC_CAT2(name, PRECISION_SUFFIX)
#define PREC_TY(name)          _PREC_CAT2(name, PRECISION_SUFFIX)

#endif /* CPU_PRECISION_H_MACROS */

/* Per-inclusion SIMD dispatch (must be outside include guard — re-evaluated
 * each time STORAGE_SUFFIX/STATE_SUFFIX changes). */

/* ── Storage-role load/store — keyed on STORAGE_SUFFIX ── */
#undef simd_load_storage
#undef simd_store_storage
#undef scalar_load_storage
#undef scalar_store_storage
#undef PREC_SIZEOF_STORAGE

/* simd_load/store take a plain aligned pointer (matching original API) */
#define simd_load_storage(ptr)           _PREC_CAT2(simd_load_real, STORAGE_SUFFIX)(ptr)
#define simd_store_storage(ptr, val)     _PREC_CAT2(simd_store_real, STORAGE_SUFFIX)(ptr, val)

/* scalar_load/store take (ptr, index) and compute ptr+index internally */
#define scalar_load_storage(ptr, idx)    _PREC_CAT2(scalar_load_real, STORAGE_SUFFIX)((ptr) + (idx))
#define scalar_store_storage(ptr, idx, val) \
    _PREC_CAT2(scalar_store_real, STORAGE_SUFFIX)((ptr) + (idx), (val))
#define PREC_SIZEOF_STORAGE    ((uint)sizeof(STORAGE_T))

/* ── State-role load/store — keyed on STATE_SUFFIX ── */
#undef simd_load_state
#undef simd_store_state
#undef scalar_load_state
#undef scalar_store_state
#undef PREC_SIZEOF_STATE

#define simd_load_state(ptr)             _PREC_CAT2(simd_load_real, STATE_SUFFIX)(ptr)
#define simd_store_state(ptr, val)       _PREC_CAT2(simd_store_real, STATE_SUFFIX)(ptr, val)
#define scalar_load_state(ptr, idx)      _PREC_CAT2(scalar_load_real, STATE_SUFFIX)((ptr) + (idx))
#define scalar_store_state(ptr, idx, val) \
    _PREC_CAT2(scalar_store_real, STATE_SUFFIX)((ptr) + (idx), (val))
#define PREC_SIZEOF_STATE      ((uint)sizeof(STATE_T))
