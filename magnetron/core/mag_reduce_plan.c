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

#include "mag_reduce_plan.h"

static int mag_cmp_axis(const void *a, const void *b) {
  int64_t da = *(const int64_t *)a;
  int64_t db = *(const int64_t *)b;
  return (da > db) - (da < db);
}

mag_status_t mag_reduce_plan_init(
  mag_error_t *err,
  mag_reduce_plan_t *plan,
  const mag_coords_t *coords,
  const int64_t *dims_in,
  int64_t rank_in,
  bool keepdim
) {
  memset(plan, 0, sizeof(*plan));
  int64_t xr = coords->rank;
  plan->nd = xr;
  plan->keepdim = keepdim;
  for (int64_t d=0; d < xr; ++d) {
    plan->in_shape[d] = coords->shape[d];
    plan->in_strides[d] = coords->strides[d];
  }
  int64_t ax[MAG_MAX_DIMS];
  int64_t rank = rank_in;
  if (mag_unlikely(!(dims_in != NULL || rank == 0))) {
    return mag_set_error(err, MAG_ERR_RANK, "reduce: dims must be non-NULL when rank is non-zero.");
  }
  if (mag_unlikely(!(rank >= 0 && rank <= MAG_MAX_DIMS))) {
    return mag_set_error(err, MAG_ERR_RANK, "reduce: rank must be in [0, %d], but got %" PRIi64 ".", MAG_MAX_DIMS, rank);
  }
  if (mag_unlikely(!(xr >= rank))) {
    return mag_set_error(err, MAG_ERR_RANK, "reduce: cannot reduce over %" PRIi64 " dimensions of a tensor with rank %" PRIi64 ".", rank, xr);
  }
  if (!dims_in && !rank) { /* canonicalize dims (global reduce, negatives, sort, unique) */
    rank = xr;
    for (int64_t i=0; i < rank; ++i) ax[i] = i;
  } else if (dims_in) {
    for (int64_t i=0; i < rank; ++i) {
      int64_t a = dims_in[i];
      if (a < 0) a += xr;
      if (mag_unlikely(!(0 <= a && a < xr))) {
        return mag_set_error(err, MAG_ERR_DIM, "reduce: axis %" PRIi64 " is out of bounds for tensor of rank %" PRIi64 ".", a, xr);
      }
      ax[i] = a;
    }
    qsort(ax, (size_t)rank, sizeof(int64_t), &mag_cmp_axis);
    int64_t r = 0;
    for (int64_t i=0; i < rank; ++i)
      if (!i || ax[i] != ax[i-1])
        ax[r++] = ax[i];
    rank = r;
  }
  int64_t prev=-1;
  for (int64_t i=0; i < rank; ++i) {
    int64_t a = ax[i];
    if (mag_unlikely(!(0 <= a && a < xr))) {
      return mag_set_error(err, MAG_ERR_DIM, "reduce: axis %" PRIi64 " is out of bounds for tensor of rank %" PRIi64 ".", a, xr);
    }
    if (mag_unlikely(!(a > prev))) {
      return mag_set_error(err, MAG_ERR_DIM, "reduce: axes must be strictly increasing and unique.");
    }
    prev = a;
  }
  plan->rank = rank;
  for (int64_t i=0; i < rank; ++i)
    plan->axes[i] = ax[i];
  /* build output shape (+ out_rank) */
  int64_t shape[MAG_MAX_DIMS] = {0};
  int64_t j=0, k=0;
  for (int64_t d=0; d < xr; ++d) {
    if (k < rank && ax[k] == d) {
      if (keepdim) shape[j++] = 1;
      ++k;
    } else shape[j++] = coords->shape[d];
  }
  plan->out_rank = keepdim ? xr : xr - rank;
  for (int64_t i=0; i < plan->out_rank; ++i)
    plan->out_shape[i] = shape[i];
  /* keep_axes */
  plan->nk = 0;
  for (int64_t d = 0; d < xr; ++d) {
    bool red = false;
    for (int64_t k2 = 0; k2 < rank; ++k2) {
      if (ax[k2] == d) {
        red = true;
        break;
      }
    }
    if (!red) plan->keep_axes[plan->nk++] = d;
  }
  /* red_sizes / strides / red_prod */
  plan->red_prod = 1;
  for (int64_t k2=0; k2 < rank; ++k2) {
    int64_t axd = ax[k2];
    int64_t sz = coords->shape[axd];
    plan->red_sizes[k2] = sz;
    plan->red_strides[k2] = coords->strides[axd];
    plan->red_prod *= sz;
  }
  return MAG_OK;
}
