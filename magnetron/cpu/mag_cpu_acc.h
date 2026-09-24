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

#ifdef __cplusplus
}
#endif

#endif
