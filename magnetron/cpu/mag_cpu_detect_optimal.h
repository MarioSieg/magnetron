/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#ifndef MAG_DETECT_OPTIMAL_H
#define MAG_DETECT_OPTIMAL_H

#include "mag_cpu_kernel_data.h"

#ifdef __cplusplus
extern "C" {
#endif

extern bool mag_blas_detect_optimal_specialization(const mag_context_t *ctx, mag_kernel_registry_t *kernels);

#ifdef __cplusplus
}
#endif

#endif
