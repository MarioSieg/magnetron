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

#ifndef MAG_CPU_ACCELERATE_H
#define MAG_CPU_ACCELERATE_H

#include <core/mag_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

extern bool mag_accel_matmul_supported(const mag_tensor_t *x, const mag_tensor_t *y, const mag_tensor_t *r);
extern bool mag_accel_matmul(mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y);
extern bool mag_accel_sgemm_available(void);
extern void mag_accel_sgemm_ex(bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha, const float *a, int64_t lda, const float *b, int64_t ldb, float beta, float *c, int64_t ldc);

#ifdef __cplusplus
}
#endif

#endif
