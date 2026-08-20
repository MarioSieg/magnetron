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

extern MAG_COLDPROC MAG_EXPORT const char *mag_envcfg_raw(const char *name);
extern MAG_COLDPROC MAG_EXPORT void mag_envcfg_apply_log_level(void);

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
