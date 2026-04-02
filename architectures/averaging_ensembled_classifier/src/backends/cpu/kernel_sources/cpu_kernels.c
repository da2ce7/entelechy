/* cpu_kernels.c — All CPU kernel implementations, instantiated for
 *                  s16x16, s32x32, s16x32 (ADR-008, ADR-023 §4.2).
 *
 * Each .inc template is included three times — once per precision config —
 * with STORAGE_T/STORAGE_SUFFIX and STATE_T/STATE_SUFFIX set accordingly.
 * The token-pasting macros in cpu_precision.h produce the suffixed symbol names.
 *
 * Algorithmic references: kernels/phase_*.cl.c
 */
#include "cpu_kernels.h"

/* ================================================================
 * s16x16 instantiation (FP16 storage, FP16 state — uniform FP16)
 *
 * Computation is FP32; _Float16 governs storage and state.
 * Suppress the expected float <-> _Float16 conversion warnings.
 * ================================================================ */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T       _Float16
#define STORAGE_SUFFIX  fp16
#define STATE_T         _Float16
#define STATE_SUFFIX    fp16
#define PRECISION_SUFFIX s16x16
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX
#pragma GCC diagnostic pop

/* ================================================================
 * s32x32 instantiation (FP32 storage, FP32 state — uniform FP32, reference)
 * ================================================================ */
#define STORAGE_T       float
#define STORAGE_SUFFIX  fp32
#define STATE_T         float
#define STATE_SUFFIX    fp32
#define PRECISION_SUFFIX s32x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
#undef STATE_T
#undef STATE_SUFFIX
#undef PRECISION_SUFFIX

/* ================================================================
 * s16x32 instantiation (FP16 storage, FP32 state — mixed precision)
 *
 * Suppress float <-> _Float16 conversion warnings for storage loads/stores.
 * State remains FP32, so no narrowing on state-role buffers.
 * ================================================================ */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#define STORAGE_T       _Float16
#define STORAGE_SUFFIX  fp16
#define STATE_T         float
#define STATE_SUFFIX    fp32
#define PRECISION_SUFFIX s16x32
#include "cpu_precision.h"
#include "phase_1_act.inc"
#include "phase_2_learn_A_production.inc"
#include "phase_2_learn_B_processing.inc"
#include "phase_2_learn_C_reduction.inc"
#include "phase_2_learn_D_backprop.inc"
#include "phase_3_update.inc"
#undef STORAGE_T
#undef STORAGE_SUFFIX
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
