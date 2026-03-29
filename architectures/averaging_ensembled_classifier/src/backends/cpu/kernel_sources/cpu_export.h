/* cpu_export.h — Symbol visibility macro for libcpu_kernels (ADR-015). */
#ifndef CPU_EXPORT_H
#define CPU_EXPORT_H

#if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef CPU_KERNELS_BUILDING
        #define CPU_KERNELS_EXPORT __declspec(dllexport)
    #else
        #define CPU_KERNELS_EXPORT __declspec(dllimport)
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define CPU_KERNELS_EXPORT __attribute__((visibility("default")))
#else
    #define CPU_KERNELS_EXPORT
#endif

#endif /* CPU_EXPORT_H */
