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

#ifndef MAG_DETECT_OPTIMAL_H
#define MAG_DETECT_OPTIMAL_H

#include "mag_cpu_kernel_data.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_cpu_specialization_t {
  const char *name;
  uint64_t (*get_feature_bitset)(void);
  void (*inject_kernels)(mag_kernel_registry_t *reg);
} mag_cpu_specialization_t;

extern MAG_EXPORT const mag_cpu_specialization_t *mag_cpu_specializations(size_t *num);
extern MAG_EXPORT const char *mag_cpu_cap_name(uint32_t bit);
extern MAG_EXPORT size_t mag_cpu_format_caps(uint64_t caps, char *buf, size_t cap);
extern MAG_EXPORT const mag_cpu_specialization_t *mag_cpu_select_specialization(const mag_cpu_specialization_t *impls, size_t num, uint64_t host_caps);
extern bool mag_blas_detect_optimal_specialization(const mag_context_t *ctx, mag_kernel_registry_t *kernels);

#ifdef __cplusplus
}
#endif

#endif
