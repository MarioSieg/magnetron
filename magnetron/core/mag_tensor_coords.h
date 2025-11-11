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

#ifndef MAG_TENSOR_COORDS_H
#define MAG_TENSOR_COORDS_H

#include "mag_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_tensor_coords_t {
    int64_t rank;
    int64_t shape[MAG_MAX_DIMS];
    int64_t strides[MAG_MAX_DIMS];
} mag_tensor_coords_t;

static inline int64_t mag_coords_broadcast(mag_tensor_coords_t *r, const mag_tensor_coords_t *x, int64_t i) {
    const int64_t *rd = r->shape;
    const int64_t *rs = r->strides;
    const int64_t *xd = x->shape;
    const int64_t *xs = x->strides;
    int64_t ra = r->rank;
    int64_t delta = ra-- - x->rank;
    int64_t o = 0;
    for (int64_t k=ra; k >= 0; --k) {
        int64_t dim = rd[k];
        int64_t ax = i % dim;
        i /= dim;
        if (x == r) {
            o += ax*rs[k];
        } else {
            int64_t kd = k-delta;
            if (kd >= 0 && xd[kd] > 1)
                o += ax*xs[kd];
        }
    }
    return o;
}

static inline int64_t mag_coords_index_to_offset(const mag_tensor_coords_t *r, int64_t i) {
    const int64_t *rd = r->shape;
    const int64_t *rs = r->strides;
    int64_t ra = r->rank-1;
    int64_t o = 0;
    for (int64_t k=ra; k >= 0; --k) {
        int64_t dim = rd[k];
        int64_t ax = i % dim;
        i /= dim;
        o += ax*rs[k];
    }
    return o;
}

static inline int64_t mag_coords_index_repeat(mag_tensor_coords_t *r, const mag_tensor_coords_t *x, int64_t i) {
    const int64_t *rd = r->shape;
    const int64_t *rs = r->strides;
    const int64_t *xd = x->shape;
    int64_t rr = r->rank;
    int64_t rx = x->rank;
    int64_t delta = rx-- - rr;
    int64_t o = 0;
    for (int64_t k=rx; k >= 0; --k) {
        int64_t dim = xd[k];
        int64_t ax = i % dim;
        i /= dim;
        int64_t kd = k - delta;
        if (kd < 0) continue;
        o += ax % rd[kd] * rs[kd];
    }
    return o;
}

static inline bool mag_coords_broadcast_shape(const mag_tensor_coords_t *x, const mag_tensor_coords_t *y, int64_t *dims, int64_t *rank) {
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

static inline bool mag_coords_shape_cmp(const mag_tensor_coords_t *x, const mag_tensor_coords_t *y) {
    return memcmp(x->shape, y->shape, sizeof(x->shape)) == 0;
}

static inline bool mag_coords_strides_cmp(const mag_tensor_coords_t *x, const mag_tensor_coords_t *y) {
    return memcmp(x->strides, y->strides, sizeof(x->strides)) == 0;
}

static inline bool mag_coords_can_broadcast(const mag_tensor_coords_t *x, const mag_tensor_coords_t *y) {
    int64_t mr = mag_xmax(x->rank, y->rank);
    for (int64_t d=0; d < mr; ++d) {
        int64_t asz = d < x->rank ? x->shape[x->rank-1-d] : 1;
        int64_t bsz = d < y->rank ? y->shape[y->rank-1-d] : 1;
        if (asz != bsz && asz != 1 && bsz != 1)
            return false;
    }
    return true;
}

static inline bool mag_coords_transposed(const mag_tensor_coords_t *x) { return x->strides[0] > x->strides[1]; }

static inline bool mag_coords_permuted(const mag_tensor_coords_t *x) {
    for (int i=0; i < MAG_MAX_DIMS-1; ++i)
        if (x->strides[i] > x->strides[i+1])
            return true;
    return false;
}

static inline bool mag_coords_contiguous(const mag_tensor_coords_t *x) {
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


#define MAG_FMT_DIM_BUF_SIZE (8 + MAG_MAX_DIMS*sizeof("-9223372036854775808, "))
extern MAG_EXPORT void mag_fmt_shape(char (*buf)[MAG_FMT_DIM_BUF_SIZE], const int64_t (*dims)[MAG_MAX_DIMS], int64_t rank);
extern MAG_EXPORT bool mag_solve_view_strides(int64_t (*out)[MAG_MAX_DIMS], const int64_t *osz, const int64_t *ost, int64_t ork, const int64_t *nsz, int64_t nrk);
extern MAG_EXPORT bool mag_infer_missing_dim(int64_t (*out)[MAG_MAX_DIMS], const int64_t *dims, int64_t rank, int64_t numel);

#ifdef __cplusplus
}
#endif

#endif
