/* cpu_kernels.c — All CPU kernel implementations, triple-instantiated
 *                  for fp16, fp32, fp64 (ADR-008).
 *
 * Each .inc template is included three times — once per precision —
 * with REAL_T and PRECISION_SUFFIX set accordingly.  The token-pasting
 * macros in cpu_precision.h produce the suffixed symbol names.
 *
 * Algorithmic references: kernels/phase_*.cl.c
 */
#include "cpu_kernels.h"

/* ================================================================
 * FP16 instantiation
 *
 * Computation is FP32; _Float16 governs storage only.  Suppress the
 * expected float <-> _Float16 conversion warnings.
 * ================================================================ */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define REAL_T _Float16
#define PRECISION_SUFFIX fp16
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef REAL_T
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* ================================================================
 * FP32 instantiation  (reference — full warnings enabled)
 * ================================================================ */
#define REAL_T float
#define PRECISION_SUFFIX fp32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef REAL_T
#undef PRECISION_SUFFIX

/* ================================================================
 * FP64 instantiation
 *
 * Computation is FP32; double governs storage only.  Suppress the
 * expected float <-> double promotion/conversion warnings.
 * ================================================================ */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#define REAL_T double
#define PRECISION_SUFFIX fp64
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef REAL_T
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* ================================================================
 * Precision-agnostic exports
 * ================================================================ */
uint get_simd_width(void) {
    return SIMD_WIDTH;
}
