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

#ifndef MAG_REDUCE_PLAN_H
#define MAG_REDUCE_PLAN_H

#include "mag_coords.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_reduce_plan_t {
    int64_t nd;
    int64_t in_shape[MAG_MAX_DIMS];
    int64_t in_strides[MAG_MAX_DIMS];
    int64_t rank;
    int64_t axes[MAG_MAX_DIMS]; /* Sorted unique reduction axes */
    bool keepdim;
    int64_t out_rank;
    int64_t out_shape[MAG_MAX_DIMS];
    int64_t nk;
    int64_t keep_axes[MAG_MAX_DIMS];
    int64_t red_sizes[MAG_MAX_DIMS];
    int64_t red_strides[MAG_MAX_DIMS];
    int64_t red_prod;
} mag_reduce_plan_t;

extern MAG_EXPORT mag_status_t mag_reduce_plan_init(
    mag_context_t *ctx,
    mag_reduce_plan_t *plan,
    const mag_coords_t *coords,
    const int64_t *dims_in,
    int64_t rank_in,
    bool keepdim
);

static MAG_CUDA_DEVICE inline int64_t mag_reduce_plan_to_offset(const mag_reduce_plan_t *plan, int64_t i) {
    int64_t rem = i;
    int64_t off = 0;
    for (int64_t k=plan->nk-1; k >= 0; --k) {
        int64_t ax = plan->keep_axes[k];
        int64_t sz = plan->in_shape[ax];
        int64_t idx = sz > 1 ? rem % sz : 0;
        if (sz > 1) rem /= sz;
        off += idx*plan->in_strides[ax];
    }
    return off;
}

#ifdef __cplusplus
}
#endif

#endif
