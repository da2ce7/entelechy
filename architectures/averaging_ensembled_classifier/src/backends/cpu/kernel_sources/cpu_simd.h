/* cpu_simd.h — Compile-time SIMD abstraction layer (CPU_BACKEND.md).
 *
 * Provides a uniform macro API across AVX-512, AVX2, SSE2, ARM NEON,
 * and a scalar fallback.  ISA selection is compile-time via -D / -m flags.
 */
#ifndef CPU_SIMD_H
#define CPU_SIMD_H

#include <stdint.h>

/* ================================================================
 * ISA detection cascade (priority: AVX-512 > AVX2 > SSE2 > NEON > scalar)
 * ================================================================ */

#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH       16
    #define SIMD_ALIGNMENT   64
    typedef __m512  simd_float;
    typedef __m512i simd_int;

    #define simd_zero()            _mm512_setzero_ps()
    #define simd_set1(x)           _mm512_set1_ps(x)
    #define simd_load(p)           _mm512_load_ps(p)
    #define simd_loadu(p)          _mm512_loadu_ps(p)
    #define simd_store(p, v)       _mm512_store_ps(p, v)
    #define simd_add(a, b)         _mm512_add_ps(a, b)
    #define simd_mul(a, b)         _mm512_mul_ps(a, b)
    #define simd_fmadd(a, b, c)    _mm512_fmadd_ps(a, b, c)
    #define simd_max(a, b)         _mm512_max_ps(a, b)
    #define simd_min(a, b)         _mm512_min_ps(a, b)
    #define simd_reduce_add(v)     _mm512_reduce_add_ps(v)

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        __mmask16 k = _mm512_cmp_ps_mask(mask, _mm512_setzero_ps(),
                                          _CMP_GT_OQ);
        return _mm512_mask_blend_ps(k, a, b);
    }

#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH       8
    #define SIMD_ALIGNMENT   32
    typedef __m256  simd_float;
    typedef __m256i simd_int;

    #define simd_zero()            _mm256_setzero_ps()
    #define simd_set1(x)           _mm256_set1_ps(x)
    #define simd_load(p)           _mm256_load_ps(p)
    #define simd_loadu(p)          _mm256_loadu_ps(p)
    #define simd_store(p, v)       _mm256_store_ps(p, v)
    #define simd_add(a, b)         _mm256_add_ps(a, b)
    #define simd_mul(a, b)         _mm256_mul_ps(a, b)
    #define simd_fmadd(a, b, c)    _mm256_fmadd_ps(a, b, c)
    #define simd_max(a, b)         _mm256_max_ps(a, b)
    #define simd_min(a, b)         _mm256_min_ps(a, b)

    static inline float simd_reduce_add(simd_float v) {
        __m128 hi  = _mm256_extractf128_ps(v, 1);
        __m128 lo  = _mm256_castps256_ps128(v);
        __m128 sum = _mm_add_ps(lo, hi);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        simd_float cmp = _mm256_cmp_ps(mask, _mm256_setzero_ps(),
                                        _CMP_GT_OQ);
        return _mm256_blendv_ps(a, b, cmp);
    }

#elif defined(__SSE2__)
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #define SIMD_WIDTH       4
    #define SIMD_ALIGNMENT   16
    typedef __m128  simd_float;
    typedef __m128i simd_int;

    #define simd_zero()            _mm_setzero_ps()
    #define simd_set1(x)           _mm_set1_ps(x)
    #define simd_load(p)           _mm_load_ps(p)
    #define simd_loadu(p)          _mm_loadu_ps(p)
    #define simd_store(p, v)       _mm_store_ps(p, v)
    #define simd_add(a, b)         _mm_add_ps(a, b)
    #define simd_mul(a, b)         _mm_mul_ps(a, b)
    #define simd_max(a, b)         _mm_max_ps(a, b)
    #define simd_min(a, b)         _mm_min_ps(a, b)

    /* SSE2 lacks native FMA — emulate via mul+add */
    #define simd_fmadd(a, b, c)    simd_add(simd_mul(a, b), c)

    static inline float simd_reduce_add(simd_float v) {
        __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1));
        __m128 sums = _mm_add_ps(v, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        /* SSE2: compare mask > 0, blend manually */
        simd_float cmp = _mm_cmpgt_ps(mask, _mm_setzero_ps());
        return _mm_or_ps(_mm_and_ps(cmp, b), _mm_andnot_ps(cmp, a));
    }

#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define SIMD_WIDTH       4
    #define SIMD_ALIGNMENT   16
    typedef float32x4_t simd_float;

    #define simd_zero()            vdupq_n_f32(0.0f)
    #define simd_set1(x)           vdupq_n_f32(x)
    #define simd_load(p)           vld1q_f32(p)
    #define simd_loadu(p)          vld1q_f32(p)
    #define simd_store(p, v)       vst1q_f32(p, v)
    #define simd_add(a, b)         vaddq_f32(a, b)
    #define simd_mul(a, b)         vmulq_f32(a, b)
    #define simd_fmadd(a, b, c)    vfmaq_f32(c, a, b)
    #define simd_max(a, b)         vmaxq_f32(a, b)
    #define simd_min(a, b)         vminq_f32(a, b)

    static inline float simd_reduce_add(simd_float v) {
        return vaddvq_f32(v);
    }

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        uint32x4_t cmp = vcgtq_f32(mask, vdupq_n_f32(0.0f));
        return vbslq_f32(cmp, b, a);
    }

#else
    /* Scalar fallback — SIMD_WIDTH=1, all ops are plain float arithmetic */
    #define SIMD_WIDTH       1
    #define SIMD_ALIGNMENT   4
    typedef float simd_float;

    #define simd_zero()            0.0f
    #define simd_set1(x)           (x)
    #define simd_load(p)           (*(p))
    #define simd_loadu(p)          (*(p))
    #define simd_store(p, v)       (*(p) = (v))
    #define simd_add(a, b)         ((a) + (b))
    #define simd_mul(a, b)         ((a) * (b))
    #define simd_fmadd(a, b, c)    ((a) * (b) + (c))
    #define simd_max(a, b)         ((a) > (b) ? (a) : (b))
    #define simd_min(a, b)         ((a) < (b) ? (a) : (b))
    #define simd_reduce_add(v)     (v)

    static inline simd_float simd_select(simd_float a, simd_float b,
                                         simd_float mask) {
        return mask > 0.0f ? b : a;
    }
#endif

/* ================================================================
 * Aligned allocation (SIMD_ALIGNMENT-guaranteed)
 * ================================================================ */
#include <stdlib.h>

static inline void* simd_alloc(size_t bytes) {
#if defined(_MSC_VER)
    return _aligned_malloc(bytes, SIMD_ALIGNMENT);
#else
    void* ptr = NULL;
    if (posix_memalign(&ptr, SIMD_ALIGNMENT, bytes) != 0)
        return NULL;
    return ptr;
#endif
}

static inline void simd_free(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ================================================================
 * Precision Load/Store Wrappers (ADR-008)
 *
 * Computation always uses float (FP32) via simd_float.  These
 * wrappers convert between the storage type and the FP32 compute
 * type on load and store.
 *
 * - fp32: identity (no conversion)
 * - fp16: widen _Float16 → float on load, narrow on store
 * - fp64: narrow double → float on load, widen on store
 *
 * The SIMD variants process SIMD_WIDTH elements.  The scalar
 * variants process a single element.
 *
 * FP16 SIMD uses F16C intrinsics on AVX2+/AVX-512.  On ISAs
 * without F16C (SSE2, NEON, scalar fallback), FP16 SIMD falls
 * back to scalar loop conversion.
 * ================================================================ */

/* --- FP32: identity ------------------------------------------- */

static inline simd_float simd_load_real_fp32(const float* p) {
    return simd_load(p);
}
static inline void simd_store_real_fp32(float* p, simd_float v) {
    simd_store(p, v);
}
static inline float scalar_load_real_fp32(const float* p) {
    return *p;
}
static inline void scalar_store_real_fp32(float* p, float v) {
    *p = v;
}

/* --- FP16: _Float16 ↔ float ---------------------------------- */

#if defined(__AVX512F__) && defined(__AVX512BW__)
/* AVX-512 + F16C: 16-wide convert */
static inline simd_float simd_load_real_fp16(const _Float16* p) {
    __m256i half_vec = _mm256_loadu_si256((const __m256i*)p);
    return _mm512_cvtph_ps(half_vec);
}
static inline void simd_store_real_fp16(_Float16* p, simd_float v) {
    __m256i half_vec = _mm512_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256((__m256i*)p, half_vec);
}
#elif defined(__AVX2__) && defined(__F16C__)
/* AVX2 + F16C: 8-wide convert */
static inline simd_float simd_load_real_fp16(const _Float16* p) {
    __m128i half_vec = _mm_loadu_si128((const __m128i*)p);
    return _mm256_cvtph_ps(half_vec);
}
static inline void simd_store_real_fp16(_Float16* p, simd_float v) {
    __m128i half_vec = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*)p, half_vec);
}
#elif defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
/* NEON: 4-wide convert via vcvt */
static inline simd_float simd_load_real_fp16(const _Float16* p) {
    float16x4_t half_vec = vld1_f16((const float16_t*)p);
    return vcvt_f32_f16(half_vec);
}
static inline void simd_store_real_fp16(_Float16* p, simd_float v) {
    float16x4_t half_vec = vcvt_f16_f32(v);
    vst1_f16((float16_t*)p, half_vec);
}
#else
/* Scalar / SSE2 fallback: element-wise conversion */
static inline simd_float simd_load_real_fp16(const _Float16* p) {
    float tmp[SIMD_WIDTH];
    for (int i = 0; i < SIMD_WIDTH; i++) tmp[i] = (float)p[i];
    return simd_load(tmp);
}
static inline void simd_store_real_fp16(_Float16* p, simd_float v) {
    float tmp[SIMD_WIDTH];
    simd_store(tmp, v);
    for (int i = 0; i < SIMD_WIDTH; i++) p[i] = (_Float16)tmp[i];
}
#endif

static inline float scalar_load_real_fp16(const _Float16* p) {
    return (float)*p;
}
static inline void scalar_store_real_fp16(_Float16* p, float v) {
    *p = (_Float16)v;
}

/* --- FP64: double ↔ float ------------------------------------- */

#if defined(__AVX512F__)
/* AVX-512: load 16 doubles (two 512-bit loads), narrow to 16 floats */
static inline simd_float simd_load_real_fp64(const double* p) {
    __m512d lo = _mm512_loadu_pd(p);
    __m512d hi = _mm512_loadu_pd(p + 8);
    __m256  lo_f = _mm512_cvtpd_ps(lo);
    __m256  hi_f = _mm512_cvtpd_ps(hi);
    return _mm512_insertf32x8(_mm512_castps256_ps512(lo_f), hi_f, 1);
}
static inline void simd_store_real_fp64(double* p, simd_float v) {
    __m256 lo_f = _mm512_castps512_ps256(v);
    __m256 hi_f = _mm512_extractf32x8_ps(v, 1);
    __m512d lo = _mm512_cvtps_pd(lo_f);
    __m512d hi = _mm512_cvtps_pd(hi_f);
    _mm512_storeu_pd(p, lo);
    _mm512_storeu_pd(p + 8, hi);
}
#elif defined(__AVX2__)
/* AVX2: load 8 doubles (two 256-bit loads), narrow to 8 floats */
static inline simd_float simd_load_real_fp64(const double* p) {
    __m256d lo = _mm256_loadu_pd(p);
    __m256d hi = _mm256_loadu_pd(p + 4);
    __m128  lo_f = _mm256_cvtpd_ps(lo);
    __m128  hi_f = _mm256_cvtpd_ps(hi);
    return _mm256_set_m128(hi_f, lo_f);
}
static inline void simd_store_real_fp64(double* p, simd_float v) {
    __m128 lo_f = _mm256_castps256_ps128(v);
    __m128 hi_f = _mm256_extractf128_ps(v, 1);
    __m256d lo = _mm256_cvtps_pd(lo_f);
    __m256d hi = _mm256_cvtps_pd(hi_f);
    _mm256_storeu_pd(p, lo);
    _mm256_storeu_pd(p + 4, hi);
}
#elif defined(__SSE2__)
/* SSE2: load 4 doubles (two 128-bit loads), narrow to 4 floats */
static inline simd_float simd_load_real_fp64(const double* p) {
    __m128d lo = _mm_loadu_pd(p);
    __m128d hi = _mm_loadu_pd(p + 2);
    __m128  lo_f = _mm_cvtpd_ps(lo);
    __m128  hi_f = _mm_cvtpd_ps(hi);
    return _mm_movelh_ps(lo_f, hi_f);
}
static inline void simd_store_real_fp64(double* p, simd_float v) {
    __m128d lo = _mm_cvtps_pd(v);
    __m128  hi_f = _mm_movehl_ps(v, v);
    __m128d hi = _mm_cvtps_pd(hi_f);
    _mm_storeu_pd(p, lo);
    _mm_storeu_pd(p + 2, hi);
}
#elif defined(__ARM_NEON)
/* NEON: load 4 doubles via scalar, convert to float32x4 */
static inline simd_float simd_load_real_fp64(const double* p) {
    float tmp[4] = {(float)p[0], (float)p[1], (float)p[2], (float)p[3]};
    return vld1q_f32(tmp);
}
static inline void simd_store_real_fp64(double* p, simd_float v) {
    float tmp[4];
    vst1q_f32(tmp, v);
    for (int i = 0; i < 4; i++) p[i] = (double)tmp[i];
}
#else
/* Scalar fallback */
static inline simd_float simd_load_real_fp64(const double* p) {
    return (float)*p;
}
static inline void simd_store_real_fp64(double* p, simd_float v) {
    *p = (double)v;
}
#endif

static inline double scalar_load_real_fp64(const double* p) {
    return *p;
}
static inline void scalar_store_real_fp64(double* p, double v) {
    *p = v;
}

#endif /* CPU_SIMD_H */
