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

#ifndef MAG_COORDS_ITER_H
#define MAG_COORDS_ITER_H

#include "mag_coords.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mag_coords_iter_t {
    int64_t rank;
    int64_t shape[MAG_MAX_DIMS];
    int64_t strides[MAG_MAX_DIMS];
} mag_coords_iter_t;

static inline void mag_coords_iter_init(mag_coords_iter_t *ci, const mag_coords_t *co) {
    ci->rank = co->rank;
    for (uint32_t i=0; i < ci->rank; ++i) {
        mag_assert(co->shape[i] >= 0 && co->shape[i] < UINT32_MAX, "Dim out of range for coord iterator: %, must be within [0, UINT32_MAX)" PRIi64, co->shape[i]);
        mag_assert(co->strides[i] >= 0 && co->strides[i] < UINT32_MAX, "Dim out of range for coord iterator: %, must be within [0, UINT32_MAX)" PRIi64, co->shape[i]);
        ci->shape[i] = co->shape[i];
        ci->strides[i] = co->strides[i];
    }
}

static inline int64_t mag_coords_iter_to_offset(const mag_coords_iter_t *cr, int64_t i) {
    const int64_t *rd = cr->shape;
    const int64_t *rs = cr->strides;
    int64_t ra = cr->rank-1;
    int64_t o = 0;
    for (int64_t k=ra; k >= 0; --k) {
        int64_t di = rd[k];
        int64_t ax = i % di;
        i /= di;
        o += ax*rs[k];
    }
    return o;
}

static inline int64_t mag_coords_iter_broadcast(mag_coords_iter_t *cr, const mag_coords_iter_t *cx, int64_t i) {
    const int64_t *rd = cr->shape;
    const int64_t *xd = cx->shape;
    const int64_t *xs = cx->strides;
    int64_t ra = cr->rank;
    int64_t delta = ra-- - cx->rank;
    int64_t o = 0;
    for (int64_t k=ra; k >= 0; --k) {
        int64_t di = rd[k];
        int64_t ax = i % di;
        i /= di;
        int64_t kd = k-delta;
        if (kd >= 0 && xd[kd] > 1)
            o += ax*xs[kd];
    }
    return o;
}

static inline int64_t mag_coords_iter_repeat(mag_coords_iter_t *cr, const mag_coords_iter_t *cx, int64_t i) {
    const int64_t *rd = cr->shape;
    const int64_t *rs = cr->strides;
    const int64_t *xd = cx->shape;
    int64_t rr = cr->rank;
    int64_t rx = cx->rank;
    int64_t delta = rx-- - rr;
    int64_t o = 0;
    for (int64_t k=rx; k >= 0; --k) {
        int64_t di = xd[k];
        int64_t ax = i % di;
        i /= di;
        int64_t kd = k - delta;
        if (kd < 0) continue;
        o += ax % rd[kd] * rs[kd];
    }
    return o;
}

static inline void mag_coords_iter_offset2(
    const mag_coords_iter_t *cr,
    const mag_coords_iter_t *cx,
    int64_t i,
    int64_t *oir,
    int64_t *oix
) {
    const int64_t *rd = cr->shape;
    const int64_t *rs = cr->strides;
    const int64_t *xd = cx->shape;
    const int64_t *xs = cx->strides;
    int64_t rr = cr->rank;
    int64_t rx = cx->rank;
    int64_t dx = rr-rx;
    int64_t ir = 0;
    int64_t ix = 0;
    for (int64_t k = rr-1; k >= 0; --k) {
        int64_t dim = rd[k];
        int64_t ax = i % dim;
        i /= dim;
        ir += ax*rs[k];
        int64_t kx = k-dx;
        if (kx >= 0 && xd[kx] > 1)
            ix += ax*xs[kx];
    }
    *oir = ir;
    *oix = ix;
}

static inline void mag_coords_iter_offset3(
    mag_coords_iter_t *cr,
    const mag_coords_iter_t *cx,
    const mag_coords_iter_t *cy,
    int64_t i,
    int64_t *oir,
    int64_t *oix,
    int64_t *oiy
) {
    const int64_t *rd = cr->shape;
    const int64_t *rs = cr->strides;
    const int64_t *xd = cx->shape;
    const int64_t *xs = cx->strides;
    const int64_t *yd = cy->shape;
    const int64_t *ys = cy->strides;
    int64_t rr = cr->rank;
    int64_t rx = cx->rank;
    int64_t ry = cy->rank;
    int64_t dx = rr-rx;
    int64_t dy = rr-ry;
    int64_t ir = 0;
    int64_t ix = 0;
    int64_t iy = 0;
    for (int64_t k=rr-1; k >= 0; --k) {
        int64_t dim = rd[k];
        int64_t ax = i % dim;
        i /= dim;
        ir += ax*rs[k];
        int64_t kx = k-dx;
        if (kx >= 0 && xd[kx] > 1)
            ix += ax*xs[kx];
        int64_t ky = k-dy;
        if (ky >= 0 && yd[ky] > 1)
            iy += ax*ys[ky];
    }
    *oir = ir;
    *oix = ix;
    *oiy = iy;
}

#ifdef __cplusplus
}
#endif

#endif
