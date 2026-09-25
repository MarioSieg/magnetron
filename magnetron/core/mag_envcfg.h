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
#define MAG_ENV_CPU_THREADS "MAG_CPU_THREADS"                            /* Number of CPU worker threads (default: all virtual cores) */
#define MAG_ENV_NUMA_STRATEGY "MAG_NUMA_STRATEGY"                        /* NUMA thread pinning: disabled, distribute, isolate, numactl */
#define MAG_ENV_READAHEAD_MB "MAG_READ_AHEAD_MB"                          /* Mapped snapshot readahead window in MiB, 0=off, */
#define MAG_ENV_READAHEAD_RELEASE "MAG_READ_AHEAD_RELEASE"                /* Drop mapped snapshot pages after use: 0, 1, or auto*/

extern MAG_COLDPROC MAG_EXPORT const char *mag_envcfg_raw(const char *name);
extern MAG_COLDPROC MAG_EXPORT void mag_envcfg_apply_log_level(void);

typedef enum mag_envcfg_cpu_specialization_t {
  MAG_ENVCFG_CPU_SPECIALIZATION_AUTO,     /* Autodetect with runtime cpu detection */
  MAG_ENVCFG_CPU_SPECIALIZATION_FALLBACK, /* Use baseline for portability */
  MAG_ENVCFG_CPU_SPECIALIZATION_PINNED    /* Pin specific */
} mag_envcfg_cpu_specialization_t;
extern MAG_COLDPROC MAG_EXPORT mag_envcfg_cpu_specialization_t mag_envcfg_cpu_specialization_level(const char **out_name);
extern MAG_COLDPROC MAG_EXPORT uint32_t mag_envcfg_cpu_threads(uint32_t fallback);
extern MAG_COLDPROC MAG_EXPORT int mag_envcfg_numa_strategy(int fallback);
extern MAG_COLDPROC MAG_EXPORT uint64_t mag_envcfg_readahead_mb(uint64_t fallback);
extern MAG_COLDPROC MAG_EXPORT int mag_envcfg_readahead_release(void);

#ifdef __cplusplus
}
#endif

#endif
