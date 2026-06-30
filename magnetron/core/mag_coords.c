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

#include "mag_tensor.h"

bool mag_coords_broadcast_shape(const mag_coords_t *x, const mag_coords_t *y, int64_t *dims, int64_t *rank) {
  int64_t ar = x->rank, br = y->rank;
  int64_t r = *rank = ar > br ? ar : br;
  for (int64_t i=0; i < r; ++i) {
    int64_t ra = ar-1-i >= 0 ? x->shape[ar-1-i] : 1;
    int64_t rb = br-1-i >= 0 ? y->shape[br-1-i] : 1;
    if (mag_unlikely(!(ra == rb || ra == 1 || rb == 1))) /* Incompatible shapes */
      return false;
    dims[r-1-i] = ra == 1 ? rb : ra;
  }
  return true;
}

bool mag_coords_broadcast_multi_shape(const mag_coords_t **x, size_t n, int64_t *dims, int64_t *rank) {
  mag_assert2(n > 0);
  mag_coords_t acc = **x;
  for (size_t i=1; i < n; ++i) {
    int64_t tmp_dims[MAG_MAX_DIMS];
    int64_t tmp_rank = 0;
    if (!mag_coords_broadcast_shape(&acc, x[i], tmp_dims, &tmp_rank))
      return false;
    acc.rank = tmp_rank;
    memcpy(acc.shape, tmp_dims, sizeof(int64_t)*tmp_rank);
  }
  *rank = acc.rank;
  memcpy(dims, acc.shape, sizeof(int64_t)*acc.rank);
  return true;
}

bool mag_coords_shape_cmp(const mag_coords_t *x, const mag_coords_t *y) {
  if (x->rank != y->rank) return false;
  for (int64_t i=0; i < x->rank; ++i)
    if (x->shape[i] != y->shape[i])
      return false;
  return true;
}

bool mag_coords_strides_cmp(const mag_coords_t *x, const mag_coords_t *y) {
  if (x->rank != y->rank) return false;
  for (int64_t i=0; i < x->rank; ++i)
    if (x->strides[i] != y->strides[i])
      return false;
  return true;
}

bool mag_coords_can_broadcast(const mag_coords_t *x, const mag_coords_t *y) {
  int64_t mr = mag_xmax(x->rank, y->rank);
  for (int64_t i=0; i < mr; ++i) {
    int64_t asz = i < x->rank ? x->shape[x->rank-1-i] : 1;
    int64_t bsz = i < y->rank ? y->shape[y->rank-1-i] : 1;
    if (asz != bsz && asz != 1 && bsz != 1)
      return false;
  }
  return true;
}

bool mag_coords_transposed(const mag_coords_t *x) {
  if (x->rank < 2) return false;
  for (int64_t i=0; i < x->rank-1; ++i) {
    int64_t s0 = x->strides[i];
    int64_t s1 = x->strides[i+1];
    if (s0 == 0 || s1 == 0) continue;
    if (s0 < s1) return true;
  }
  return false;
}

bool mag_coords_permuted(const mag_coords_t *x) {
  if (x->rank < 2) return false;
  for (int64_t i=0; i < x->rank-1; ++i) {
    int64_t s0 = x->strides[i];
    int64_t s1 = x->strides[i+1];
    if (s0 == 0 || s1 == 0) continue;
    if (s0 < s1) return true;
  }
  return false;
}

bool mag_coords_contiguous(const mag_coords_t *x) {
  int64_t y=1;
  int64_t i=x->rank-1, j;
  for (; i >= 0; --i) {
    j = x->shape[i];
    if (j == 1) continue;
    if (x->strides[i] != y) return false;
    y *= j;
  }
  return true;
}

void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank) {
  memset(*buf, 0, sizeof(*buf));
  char *p = *buf;
  *p++ = '(';
  for (int64_t i=0; i < rank; ++i) {
    p += snprintf(p, 21, "%" PRIi64, (*dims)[i]);
    if (i < rank-1) {
      *p++ = ',';
      *p++ = ' ';
    }
  }
  *p++ = ')';
  *p = '\0';
}

mag_status_t mag_solve_view_strides(
  mag_error_t *err,
  int64_t (*out_new_strides)[MAG_MAX_DIMS],
  const int64_t *old_shape,
  const int64_t *old_strides,
  int64_t old_rank,
  const int64_t *new_shape,
  int64_t new_rank
) {
  int64_t numel=1;
  for (int64_t i=0; i < old_rank; ++i) {
    if (mag_unlikely(mag_mulov64(numel, old_shape[i], &numel)))
      return mag_set_error(err, MAG_ERR_DIM,
        "view: source element count overflowed at dim %" PRIi64
        " (size %" PRIi64 ").",
        i, old_shape[i]);
  }
  int64_t oi = old_rank-1;
  int64_t ni = new_rank-1;
  while (oi >= 0 && ni >= 0) {
    if (new_shape[ni] == 1) {
      (*out_new_strides)[ni] = ni+1 < new_rank ? (*out_new_strides)[ni+1]*new_shape[ni+1] : 1;
      --ni;
      continue;
    }
    for (; oi >= 0 && old_shape[oi] == 1; --oi);
    if (mag_unlikely(oi < 0))
      return mag_set_error(err, MAG_ERR_STRIDES,
        "view: tensor memory layout is incompatible with the requested view; "
        "ran out of source dimensions while matching target dim %" PRIi64 ".",
        ni);
    if (new_shape[ni] == old_shape[oi]) {
      (*out_new_strides)[ni] = old_strides[oi];
      --ni;
      --oi;
      continue;
    }
    int64_t nc = new_shape[ni];
    int64_t oc = old_shape[oi];
    int64_t cs = old_strides[oi];
    int64_t nkf = ni;
    while (nc != oc) {
      if (nc < oc) {
        --ni;
        if (mag_unlikely(ni < 0))
          return mag_set_error(err, MAG_ERR_STRIDES,
            "view: cannot split source dim %" PRIi64
            " of size %" PRIi64 " into requested target shape.",
            oi, old_shape[oi]);
        if (mag_unlikely(mag_mulov64(nc, new_shape[ni], &nc)))
          return mag_set_error(err, MAG_ERR_DIM,
            "view: target chunk size overflowed while merging dim %" PRIi64
            " (size %" PRIi64 ").",
            ni, new_shape[ni]);
      } else {
        --oi;
        for (; oi >= 0 && old_shape[oi] == 1; --oi);
        if (mag_unlikely(oi < 0))
          return mag_set_error(err, MAG_ERR_STRIDES,
            "view: cannot merge target dims into source layout; "
            "ran out of source dimensions.");
        int64_t expected_stride;
        if (mag_unlikely(mag_mulov64(old_shape[oi + 1], old_strides[oi + 1], &expected_stride)))
          return mag_set_error(err, MAG_ERR_DIM,
            "view: expected contiguous stride computation overflowed at source dim %" PRIi64 ".",
            oi);
        if (mag_unlikely(old_strides[oi] != expected_stride))
          return mag_set_error(err, MAG_ERR_STRIDES,
            "view: source dims %" PRIi64 " and %" PRIi64
            " are not contiguous enough to merge "
            "(stride[%" PRIi64 "]=%" PRIi64 ", expected %" PRIi64 ").",
            oi, oi + 1,
            oi, old_strides[oi], expected_stride);
        if (mag_unlikely(mag_mulov64(oc, old_shape[oi], &oc)))
          return mag_set_error(err, MAG_ERR_DIM,
            "view: source chunk size overflowed while merging dim %" PRIi64
            " (size %" PRIi64 ").",
            oi, old_shape[oi]);
      }
    }
    int64_t stride = cs;
    for (int64_t k=ni; k <= nkf; ++k) {
      (*out_new_strides)[k] = stride;
      if (mag_unlikely(mag_mulov64(stride, new_shape[k], &stride)))
        return mag_set_error(err, MAG_ERR_DIM,
          "view: output stride computation overflowed at target dim %" PRIi64 ".",
          k);
    }
    --ni;
    --oi;
  }
  while (ni >= 0) {
    (*out_new_strides)[ni] = ni+1 < new_rank ? (*out_new_strides)[ni+1]*new_shape[ni+1] : 1;
    --ni;
  }
  for (; oi >= 0 && old_shape[oi] == 1; --oi);
  if (mag_unlikely(oi >= 0))
    return mag_set_error(err, MAG_ERR_STRIDES,
      "view: tensor memory layout is incompatible with the requested view; "
      "source dim %" PRIi64 " of size %" PRIi64 " remains unmatched.",
      oi, old_shape[oi]);
  return MAG_OK;
}

mag_status_t mag_infer_missing_dim(
  mag_error_t *err,
  int64_t (*out)[MAG_MAX_DIMS],
  const int64_t *dims,
  int64_t rank,
  int64_t numel
) {
  int64_t prod=1;
  int64_t infer=-1;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t ax = dims[i];
    if (ax == -1) {
      if (mag_unlikely(infer != -1))
        return mag_set_error(err, MAG_ERR_DIM,
          "view: only one dimension can be inferred, but found another -1 at dim %" PRIi64 ".",
          i);
      infer = i;
      (*out)[i] = 1;
    } else {
      if (mag_unlikely(ax <= 0))
        return mag_set_error(err, MAG_ERR_DIM,
          "view: invalid dimension at dim %" PRIi64
          " (size %" PRIi64 "); expected positive size or -1.",
          i, ax);
      (*out)[i] = ax;
      if (mag_unlikely(mag_mulov64(prod, ax, &prod)))
        return mag_set_error(err, MAG_ERR_DIM,
          "view: requested shape element count overflowed at dim %" PRIi64
          " (size %" PRIi64 ").",
          i, ax);
    }
  }
  if (infer >= 0) {
    if (mag_unlikely(!(prod != 0 && numel % prod == 0)))
      return mag_set_error(err, MAG_ERR_DIM,
        "view: cannot infer dimension at dim %" PRIi64
        " because tensor with %" PRIi64
        " elements is not divisible by known product %" PRIi64 ".",
        infer, numel, prod);
    (*out)[infer] = numel / prod;
  } else {
    if (mag_unlikely(prod != numel))
      return mag_set_error(err, MAG_ERR_DIM,
        "view: requested shape has %" PRIi64
        " elements, but input tensor has %" PRIi64 " elements.",
        prod, numel);
  }
  return MAG_OK;
}

mag_mat_layout_type_t mag_mat_layout_detect(const mag_coords_t *coords, bool *out_batch_packed) {
  int64_t ra = coords->rank;
  if (ra < 2) { *out_batch_packed = true; return MAG_MAT_LAYOUT_TYPE_PACKED; }
  int64_t rows = coords->shape[ra-2];
  int64_t cols = coords->shape[ra-1];
  int64_t srows = coords->strides[ra-2];
  int64_t scols = coords->strides[ra-1];
  mag_mat_layout_type_t layout = MAG_MAT_LAYOUT_TYPE_OTHER;
  if (scols == 1 && srows == cols) layout = MAG_MAT_LAYOUT_TYPE_PACKED;
  else if (srows == 1 && scols == rows) layout = MAG_MAT_LAYOUT_TYPE_TRANSPOSED;
  else { *out_batch_packed = false; return MAG_MAT_LAYOUT_TYPE_OTHER; }
  if (ra == 2) { *out_batch_packed = true; return layout; }
  int64_t prod = rows*cols;
  for (int64_t i = ra-3; i >= 0; --i) {
    if (coords->strides[i] != prod) { *out_batch_packed = false; return MAG_MAT_LAYOUT_TYPE_OTHER; }
    prod *= coords->shape[i];
  }
  *out_batch_packed = true;
  return layout;
}

mag_matmul_type_t mag_matmul_type_detect(const mag_tensor_t *x, const mag_tensor_t *y) {
  int64_t xra = x->coords.rank&255;
  int64_t yra = y->coords.rank&255;
  if (mag_unlikely(xra < 1 || yra < 1)) return MAG_MATMUL_TYPE_INVALID;
  switch ((xra<<8)|yra) {
    case ((1<<8)|1): return MAG_MATMUL_TYPE_DOT;
    case ((1<<8)|2): return MAG_MATMUL_TYPE_GEMV_VEC_MAT;
    case ((2<<8)|1): return MAG_MATMUL_TYPE_GEMV_MAT_VEC;
    case ((2<<8)|2): return MAG_MATMUL_TYPE_GEMM;
    default: {
      int64_t M = x->coords.shape[xra-2];
      int64_t N = y->coords.shape[yra-1];
      if (M == 1 && N == 1) return MAG_MATMUL_TYPE_BMM_DOT;
      if (M == 1) return MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT;
      if (N == 1) return MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC;
      return MAG_MATMUL_TYPE_BMM_GEMM;
    }
  }
}

bool mag_matmul_type_is_micro_kernel_contig(mag_matmul_type_t type, const mag_tensor_t *x, const mag_tensor_t *y) {
  switch (type) {
    case MAG_MATMUL_TYPE_DOT:
    case MAG_MATMUL_TYPE_BMM_DOT: {
      int64_t sx = x->coords.strides[0];
      int64_t sy = y->coords.strides[0];
      return sx == 1 && sy == 1;
    }
    case MAG_MATMUL_TYPE_GEMV_MAT_VEC:
    case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC: {
      int64_t K = x->coords.shape[1];
      int64_t sx0 = x->coords.strides[0];
      int64_t sx1 = x->coords.strides[1];
      int64_t sy = y->coords.strides[0];
      return sx0 == K && sx1 == 1 && sy == 1;
    }
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT:
    case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT: {
      int64_t sx = x->coords.strides[0];
      int64_t sy0 = y->coords.strides[0];
      int64_t sy1 = y->coords.strides[1];
      int64_t N = y->coords.shape[1];
      return sx == 1 && sy0 == N && sy1 == 1;
    }
    case MAG_MATMUL_TYPE_GEMM:
    case MAG_MATMUL_TYPE_BMM_GEMM: {
      int64_t K = x->coords.shape[1];
      int64_t N = y->coords.shape[1];
      int64_t sx0 = x->coords.strides[0];
      int64_t sx1 = x->coords.strides[1];
      int64_t sy0 = y->coords.strides[0];
      int64_t sy1 = y->coords.strides[1];
      return sx0 == K && sx1 == 1 && sy0 == N && sy1 == 1;
    }
    default: return false;
  }
}

const char *mag_matmul_type_name(mag_matmul_type_t type) {
  switch (type) {
    case MAG_MATMUL_TYPE_INVALID: return "invalid";
    case MAG_MATMUL_TYPE_DOT: return "DOT";
    case MAG_MATMUL_TYPE_GEMV_VEC_MAT: return "VGEM";
    case MAG_MATMUL_TYPE_GEMV_MAT_VEC: return "GEMV";
    case MAG_MATMUL_TYPE_GEMM: return "GEMM";
    case MAG_MATMUL_TYPE_BMM_DOT: return "BMM_DOT";
    case MAG_MATMUL_TYPE_BMM_GEMV_VEC_MAT: return "BMM_VGEM";
    case MAG_MATMUL_TYPE_BMM_GEMV_MAT_VEC: return "BMM_GEMV";
    case MAG_MATMUL_TYPE_BMM_GEMM: return "BMM_GEMM";
    default: return "unknown";
  }
}
