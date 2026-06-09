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

#ifndef MAG_TENSOR_COORDS_H
#define MAG_TENSOR_COORDS_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_coords_t {
  int64_t rank;
  int64_t shape[MAG_MAX_DIMS];
  int64_t strides[MAG_MAX_DIMS];
} mag_coords_t;

#define MAG_FMT_DIM_BUF_SIZE (8 + MAG_MAX_DIMS*sizeof("-9223372036854775808, "))

extern bool mag_coords_broadcast_shape(const mag_coords_t *x, const mag_coords_t *y, int64_t *dims, int64_t *rank);
extern bool mag_coords_broadcast_multi_shape(const mag_coords_t **x, size_t n, int64_t *dims, int64_t *rank);
extern bool mag_coords_shape_cmp(const mag_coords_t *x, const mag_coords_t *y);
extern bool mag_coords_strides_cmp(const mag_coords_t *x, const mag_coords_t *y);
extern bool mag_coords_can_broadcast(const mag_coords_t *x, const mag_coords_t *y);
extern bool mag_coords_transposed(const mag_coords_t *x);
extern bool mag_coords_permuted(const mag_coords_t *x);
extern bool mag_coords_contiguous(const mag_coords_t *x);
extern MAG_EXPORT void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank);
extern MAG_EXPORT bool mag_solve_view_strides(int64_t (*out)[MAG_MAX_DIMS], const int64_t *osz, const int64_t *ost, int64_t ork, const int64_t *nsz, int64_t nrk);
extern MAG_EXPORT bool mag_infer_missing_dim(int64_t (*out)[MAG_MAX_DIMS], const int64_t *dims, int64_t rank, int64_t numel);

typedef enum mag_mat_layout_type_t {
  MAG_MAT_LAYOUT_TYPE_PACKED,
  MAG_MAT_LAYOUT_TYPE_TRANSPOSED,
  MAG_MAT_LAYOUT_TYPE_OTHER
} mag_mat_layout_type_t;
extern MAG_EXPORT mag_mat_layout_type_t mag_mat_layout_detect(const mag_coords_t *coords, bool *out_batch_packed);

typedef enum mag_matmul_type_t {
  MAG_MATMUL_TYPE_INVALID,
  MAG_MATMUL_TYPE_DOT,
  MAG_MATMUL_TYPE_GEMV_VEC_MAT,
  MAG_MATMUL_TYPE_GEMV_MAT_VEC,
  MAG_MATMUL_TYPE_GEMM,
  MAG_MATMUL_TYPE_BMM
} mag_matmul_type_t;
extern MAG_EXPORT mag_matmul_type_t mag_matmul_type_detect(const mag_tensor_t *x, const mag_tensor_t *y);
extern MAG_EXPORT const char *mag_matmul_type_name(mag_matmul_type_t type);

#ifdef __cplusplus
}
#endif

#endif
