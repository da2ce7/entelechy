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
        __m128 shuf = _mm_movehdup_ps(v);
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

#endif /* CPU_SIMD_H */
