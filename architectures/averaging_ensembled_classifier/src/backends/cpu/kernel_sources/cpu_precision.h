/* cpu_precision.h — Multi-precision instantiation macros (ADR-008).
 *
 * Included by .inc template files. The including .c file must define:
 *
 *   REAL_T              — storage type (float, _Float16, double)
 *   PRECISION_SUFFIX    — token suffix (fp32, fp16, fp64)
 *
 * Provides:
 *   PREC_FN(name)       — paste suffix onto function name
 *   PREC_TY(name)       — paste suffix onto type name
 *   PREC_SIZEOF_REAL    — sizeof(REAL_T) as integer constant
 *   simd_load_real(p)   — SIMD load converting REAL_T* → simd_float
 *   simd_store_real(p,v)— SIMD store converting simd_float → REAL_T*
 *   scalar_load_real(p) — scalar load converting REAL_T* → float
 *   scalar_store_real(p,v) — scalar store converting float → REAL_T*
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
 * each time PRECISION_SUFFIX changes). */
#undef simd_load_real
#undef simd_store_real
#undef scalar_load_real
#undef scalar_store_real
#undef PREC_SIZEOF_REAL

#define simd_load_real         _PREC_CAT2(simd_load_real, PRECISION_SUFFIX)
#define simd_store_real        _PREC_CAT2(simd_store_real, PRECISION_SUFFIX)
#define scalar_load_real       _PREC_CAT2(scalar_load_real, PRECISION_SUFFIX)
#define scalar_store_real      _PREC_CAT2(scalar_store_real, PRECISION_SUFFIX)
#define PREC_SIZEOF_REAL       ((uint)sizeof(REAL_T))
