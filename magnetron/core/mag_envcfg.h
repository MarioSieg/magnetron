/*
** +---------------------------------------------------------------------+
** | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#ifndef MAG_ENVCFG_H
#define MAG_ENVCFG_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_ENV_LOG_LEVEL "MAG_LOG_LEVEL"                                /* Global log verbosity */
#define MAG_ENV_CPU_SPECIALIZATION_LEVEL "MAG_CPU_SPECIALIZATION_LEVEL"  /* Pinned CPU specialization level */
#define MAG_ENV_CPU_INTRAOP_MIN_ELEMS "MAG_CPU_INTRAOP_MIN_ELEMS"        /* Override the element count at which intra-op multithreading starts */
#define MAG_ENV_JIT "MAG_JIT"                                            /* Enable/disable the pointwise fusion JIT */
#define MAG_ENV_JIT_CC "MAG_JIT_CC"                                      /* Host compiler used to build fused kernels */
#define MAG_ENV_JIT_CACHE_DIR "MAG_JIT_CACHE_DIR"                        /* Where compiled fused kernels are kept */
#define MAG_ENV_JIT_POISON "MAG_JIT_POISON"                              /* Fill elided fused values with NaN to check the backward value table */

#ifdef _WIN32
#define MAG_JIT_DEFAULT_CC "cl"
#define MAG_JIT_DEFAULT_CACHE_DIR "."
#else
#define MAG_JIT_DEFAULT_CC "cc"
#define MAG_JIT_DEFAULT_CACHE_DIR "/tmp"
#endif

extern MAG_COLDPROC MAG_EXPORT const char *mag_envcfg_raw(const char *name);
extern MAG_COLDPROC MAG_EXPORT void mag_envcfg_apply_log_level(void);
extern MAG_COLDPROC MAG_EXPORT int64_t mag_envcfg_cpu_intraop_min_elems(void); /* <0 when unset, meaning use the per-op table. */
extern MAG_COLDPROC MAG_EXPORT bool mag_envcfg_jit_enabled(void);
extern MAG_COLDPROC MAG_EXPORT bool mag_envcfg_jit_poison(void);

typedef enum mag_envcfg_cpu_specialization_t {
  MAG_ENVCFG_CPU_SPECIALIZATION_AUTO,     /* Autodetect with runtime cpu detection */
  MAG_ENVCFG_CPU_SPECIALIZATION_FALLBACK, /* Use baseline for portability */
  MAG_ENVCFG_CPU_SPECIALIZATION_PINNED    /* Pin specific */
} mag_envcfg_cpu_specialization_t;
extern MAG_COLDPROC MAG_EXPORT mag_envcfg_cpu_specialization_t mag_envcfg_cpu_specialization_level(const char **out_name);

#ifdef __cplusplus
}
#endif

#endif
