/* cpu_precision.h — Three-axis multi-precision instantiation macros (ADR-008, ADR-023 §2.2, ADR-024 §4).
 *
 * Included by .inc template files. The including .c file must define:
 *
 *   STORAGE_T           — storage-role type (float, _Float16, double)
 *   STORAGE_SUFFIX      — storage token suffix (fp32, fp16, fp64)
 *   COMPUTE_T           — compute-role type (float, double)
 *   COMPUTE_SUFFIX      — compute token suffix (fp32, fp64)
 *   STATE_T             — state-role type (float, _Float16, double)
 *   STATE_SUFFIX        — state token suffix (fp32, fp16, fp64)
 *   PRECISION_SUFFIX    — combined suffix for symbol names (s32c32x32, s16c32x32, etc.)
 *
 * Provides:
 *   PREC_FN(name)       — paste suffix onto function name
 *   PREC_TY(name)       — paste suffix onto type name
 *   PREC_SIZEOF_STORAGE — sizeof(STORAGE_T) as integer constant
 *   PREC_SIZEOF_COMPUTE — sizeof(COMPUTE_T) as integer constant
 *   PREC_SIZEOF_STATE   — sizeof(STATE_T) as integer constant
 *
 * Storage-role load/store (keyed on STORAGE_SUFFIX):
 *   simd_load_storage(p)   — SIMD load converting STORAGE_T* → simd_float
 *   simd_store_storage(p,v)— SIMD store converting simd_float → STORAGE_T*
 *   scalar_load_storage(p,i) — scalar load converting STORAGE_T[i] → float/double
 *   scalar_store_storage(p,i,v) — scalar store converting float/double → STORAGE_T[i]
 *
 * State-role load/store (keyed on STATE_SUFFIX):
 *   simd_load_state(p)     — SIMD load converting STATE_T* → simd_float
 *   simd_store_state(p,v)  — SIMD store converting simd_float → STATE_T*
 *   scalar_load_state(p,i) — scalar load converting STATE_T[i] → float/double
 *   scalar_store_state(p,i,v) — scalar store converting float/double → STATE_T[i]
 *
 * Compute-role math function dispatch (keyed on COMPUTE_T via C11 _Generic):
 *   COMPUTE_EXP, COMPUTE_LOG, COMPUTE_SQRT, COMPUTE_FABS, COMPUTE_FMAX, COMPUTE_FMIN
 */
#ifndef CPU_PRECISION_H_MACROS
#define CPU_PRECISION_H_MACROS

/* Two-level token paste for macro expansion */
#define _PREC_CAT(a, b)       a##_##b
#define _PREC_CAT2(a, b)      _PREC_CAT(a, b)

#define PREC_FN(name)          _PREC_CAT2(name, PRECISION_SUFFIX)
#define PREC_TY(name)          _PREC_CAT2(name, PRECISION_SUFFIX)

/* _Float16 compute-role math wrappers (round-trip through float).
 * On x86, _Float16 arithmetic is emulated via float anyway; these
 * wrappers make the _Generic dispatch explicit. */
static inline _Float16 _compute_exp_f16(_Float16 x)  { return (_Float16)expf((float)x); }
static inline _Float16 _compute_log_f16(_Float16 x)  { return (_Float16)logf((float)x); }
static inline _Float16 _compute_sqrt_f16(_Float16 x) { return (_Float16)sqrtf((float)x); }
static inline _Float16 _compute_fabs_f16(_Float16 x) { return (_Float16)fabsf((float)x); }
static inline _Float16 _compute_fmax_f16(_Float16 x, _Float16 y) { return (_Float16)fmaxf((float)x, (float)y); }
static inline _Float16 _compute_fmin_f16(_Float16 x, _Float16 y) { return (_Float16)fminf((float)x, (float)y); }

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

/* ── Compute-role size and math function dispatch (ADR-024 §4.2) ── */
#undef PREC_SIZEOF_COMPUTE
#define PREC_SIZEOF_COMPUTE    ((uint)sizeof(COMPUTE_T))

/* Compute-role math function dispatch via C11 _Generic */
#undef COMPUTE_EXP
#undef COMPUTE_LOG
#undef COMPUTE_SQRT
#undef COMPUTE_FABS
#undef COMPUTE_FMAX
#undef COMPUTE_FMIN

#define COMPUTE_EXP(x)    _Generic((x), _Float16: _compute_exp_f16, float: expf, double: exp)(x)
#define COMPUTE_LOG(x)    _Generic((x), _Float16: _compute_log_f16, float: logf, double: log)(x)
#define COMPUTE_SQRT(x)   _Generic((x), _Float16: _compute_sqrt_f16, float: sqrtf, double: sqrt)(x)
#define COMPUTE_FABS(x)   _Generic((x), _Float16: _compute_fabs_f16, float: fabsf, double: fabs)(x)
#define COMPUTE_FMAX(x,y) _Generic((x), _Float16: _compute_fmax_f16, float: fmaxf, double: fmax)(x,y)
#define COMPUTE_FMIN(x,y) _Generic((x), _Float16: _compute_fmin_f16, float: fminf, double: fmin)(x,y)
