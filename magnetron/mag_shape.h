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

#ifndef MAG_SHAPE_H
#define MAG_SHAPE_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_FMT_DIM_BUF_SIZE ((21+4)*MAG_MAX_DIMS)
extern void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank);
extern bool mag_solve_view_strides(int64_t (*out)[MAG_MAX_DIMS], const int64_t *osz, const int64_t *ost, int64_t ork, const int64_t *nsz, int64_t nrk);
extern void mag_infer_missing_dim(int64_t (*out)[MAG_MAX_DIMS], const int64_t *dims, int64_t rank, int64_t numel);
extern bool mag_compute_broadcast_shape(const mag_tensor_t *a, const mag_tensor_t *b, int64_t *dims, int64_t *rank);

#ifdef __cplusplus
}
#endif

#endif
