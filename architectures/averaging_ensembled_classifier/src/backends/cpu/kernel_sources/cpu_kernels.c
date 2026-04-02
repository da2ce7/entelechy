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
 * Precision-agnostic exports
 * ================================================================ */
uint get_simd_width(void) {
    return SIMD_WIDTH;
}
