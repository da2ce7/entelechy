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

/* FP64 detection table for accumulation-precision macros (ADR-027).
 * Maps suffix tokens to 0/1. Used by the per-inclusion ACCUM_T logic. */
#define _PREC_IS_F64_fp16 0
#define _PREC_IS_F64_fp32 0
#define _PREC_IS_F64_fp64 1
/* Note: FP8 suffixes (fp8e4m3, fp8e5m2) are storage-only, so ACCUM_T
 * detection based on STATE_SUFFIX never sees them. */

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

/* ── Accumulation-role type and macros (ADR-027) ── */
/* ACCUM_T = max(COMPUTE_T, STATE_T). When STATE_T is wider than COMPUTE_T
 * (e.g., FP64 state + FP32 compute), EMA arithmetic uses STATE_T precision
 * to prevent erosion over unbounded training steps.
 *
 * When STATE_T <= COMPUTE_T (the common case), ACCUM_T == COMPUTE_T and
 * all casts are identity operations eliminated by the compiler. */
#undef ACCUM_T
#undef ACCUM_IS_WIDER_THAN_COMPUTE
#undef scalar_load_state_for_accum
#undef scalar_store_state_from_accum
#undef scalar_widen_to_accum
#undef scalar_narrow_from_accum

#if _PREC_CAT2(_PREC_IS_F64, STATE_SUFFIX) && !_PREC_CAT2(_PREC_IS_F64, COMPUTE_SUFFIX)
    /* STATE_T (double) > COMPUTE_T (float or _Float16) */
    #define ACCUM_T double
    #define ACCUM_IS_WIDER_THAN_COMPUTE 1

    #define scalar_load_state_for_accum(ptr, idx) ((ptr)[(idx)])
    #define scalar_store_state_from_accum(ptr, idx, val) ((ptr)[(idx)] = (val))
    #define scalar_widen_to_accum(val) ((double)(val))
    #define scalar_narrow_from_accum(val) ((COMPUTE_T)(val))
#else
    /* STATE_T <= COMPUTE_T (standard case) */
    #define ACCUM_T COMPUTE_T
    #define ACCUM_IS_WIDER_THAN_COMPUTE 0

    #define scalar_load_state_for_accum(ptr, idx) scalar_load_state((ptr), (idx))
    #define scalar_store_state_from_accum(ptr, idx, val) scalar_store_state((ptr), (idx), (val))
    #define scalar_widen_to_accum(val) (val)
    #define scalar_narrow_from_accum(val) (val)
#endif
