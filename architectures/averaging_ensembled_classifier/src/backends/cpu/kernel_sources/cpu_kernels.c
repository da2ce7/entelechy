/* cpu_kernels.c — All CPU kernel implementations, instantiated for
 *                  14 three-axis precision variants (ADR-024 §4.1).
 *
 * Each .inc template is included once per precision variant, with
 * STORAGE_T/STORAGE_SUFFIX, COMPUTE_T/COMPUTE_SUFFIX, and
 * STATE_T/STATE_SUFFIX set via macros. cpu_precision.h provides
 * token-pasting and math-dispatch machinery.
 *
 * Algorithmic references: kernels/phase_*.cl.c
 */
#include "cpu_kernels.h"

/* ================================================================
 * All valid s{storage}c{compute}x{state} combinations (ADR-024 §4.1)
 *
 * Instantiation order: grouped by STORAGE_T, then by COMPUTE_T.
 * Blocks with _Float16 storage suppress float-conversion warnings.
 * ================================================================ */

/* --- storage=16, compute=16 -------------------------------------------- */

/* s16c16x16: FP16 storage, FP16 compute, FP16 state (uniform FP16) */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          _Float16
#define STATE_SUFFIX     fp16
#define PRECISION_SUFFIX s16c16x16
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c16x32: FP16 storage, FP16 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s16c16x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c16x64: FP16 storage, FP16 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s16c16x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=16, compute=32 -------------------------------------------- */

/* s16c32x16: FP16 storage, FP32 compute, FP16 state (was s16x16) */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          _Float16
#define STATE_SUFFIX     fp16
#define PRECISION_SUFFIX s16c32x16
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c32x32: FP16 storage, FP32 compute, FP32 state (was s16x32) */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s16c32x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c32x64: FP16 storage, FP32 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s16c32x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=16, compute=64 -------------------------------------------- */

/* s16c64x16: FP16 storage, FP64 compute, FP16 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          _Float16
#define STATE_SUFFIX     fp16
#define PRECISION_SUFFIX s16c64x16
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c64x32: FP16 storage, FP64 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s16c64x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s16c64x64: FP16 storage, FP64 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        _Float16
#define STORAGE_SUFFIX   fp16
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s16c64x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=32, compute=32 -------------------------------------------- */

/* s32c32x32: FP32 storage, FP32 compute, FP32 state (reference — was s32x32) */
#define STORAGE_T        float
#define STORAGE_SUFFIX   fp32
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s32c32x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX

/* s32c32x64: FP32 storage, FP32 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        float
#define STORAGE_SUFFIX   fp32
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s32c32x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=32, compute=64 -------------------------------------------- */

/* s32c64x32: FP32 storage, FP64 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        float
#define STORAGE_SUFFIX   fp32
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s32c64x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s32c64x64: FP32 storage, FP64 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        float
#define STORAGE_SUFFIX   fp32
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s32c64x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=64, compute=64 -------------------------------------------- */

/* s64c64x64: FP64 storage, FP64 compute, FP64 state (uniform FP64 reference) */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        double
#define STORAGE_SUFFIX   fp64
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s64c64x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* ================================================================
 * FP8 E4M3 storage variants (ADR-025 §5.3)
 *
 * Storage type is cpu_fp8_e4m3 (struct { uint8_t bits; }).
 * SIMD load/store dispatch goes through scalar-loop wrappers
 * defined in cpu_fp8.h (no native FP8 SIMD instructions).
 * ================================================================ */

/* --- storage=8e4m3, compute=32 ----------------------------------------- */

/* s8e4c32x32: E4M3 storage, FP32 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e4m3
#define STORAGE_SUFFIX   fp8e4m3
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s8e4c32x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s8e4c32x64: E4M3 storage, FP32 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e4m3
#define STORAGE_SUFFIX   fp8e4m3
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s8e4c32x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=8e4m3, compute=64 ----------------------------------------- */

/* s8e4c64x64: E4M3 storage, FP64 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e4m3
#define STORAGE_SUFFIX   fp8e4m3
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s8e4c64x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* ================================================================
 * FP8 E5M2 storage variants (ADR-025 §5.3)
 * ================================================================ */

/* --- storage=8e5m2, compute=32 ----------------------------------------- */

/* s8e5c32x32: E5M2 storage, FP32 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e5m2
#define STORAGE_SUFFIX   fp8e5m2
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s8e5c32x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s8e5c32x64: E5M2 storage, FP32 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e5m2
#define STORAGE_SUFFIX   fp8e5m2
#define COMPUTE_T        float
#define COMPUTE_SUFFIX   fp32
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s8e5c32x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* --- storage=8e5m2, compute=64 ----------------------------------------- */

/* s8e5c64x64: E5M2 storage, FP64 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e5m2
#define STORAGE_SUFFIX   fp8e5m2
#define COMPUTE_T        double
#define COMPUTE_SUFFIX   fp64
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s8e5c64x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* ================================================================
 * FP8 variants with FP16 compute — requires _Float16 (ADR-025 §5.3)
 *
 * Guarded by HAS_FLOAT16 (emitted by Meson -DHAS_FLOAT16=1).
 * ================================================================ */

#if defined(HAS_FLOAT16) && HAS_FLOAT16

/* s8e4c16x32: E4M3 storage, FP16 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e4m3
#define STORAGE_SUFFIX   fp8e4m3
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s8e4c16x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s8e4c16x64: E4M3 storage, FP16 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e4m3
#define STORAGE_SUFFIX   fp8e4m3
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s8e4c16x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s8e5c16x32: E5M2 storage, FP16 compute, FP32 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e5m2
#define STORAGE_SUFFIX   fp8e5m2
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          float
#define STATE_SUFFIX     fp32
#define PRECISION_SUFFIX s8e5c16x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* s8e5c16x64: E5M2 storage, FP16 compute, FP64 state */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T        cpu_fp8_e5m2
#define STORAGE_SUFFIX   fp8e5m2
#define COMPUTE_T        _Float16
#define COMPUTE_SUFFIX   fp16
#define STATE_T          double
#define STATE_SUFFIX     fp64
#define PRECISION_SUFFIX s8e5c16x64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef COMPUTE_T
#undef COMPUTE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

#endif /* HAS_FLOAT16 — FP8+FP16 compute variants */

/* ================================================================
 * Precision-agnostic exports
 * ================================================================ */
uint get_simd_width(void) {
    return SIMD_WIDTH;
}
