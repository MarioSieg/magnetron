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

#ifndef MAG_COORDS_ITER_H
#define MAG_COORDS_ITER_H

#include "mag_coords.h"
#include "mag_fastdivmod.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MAG_COORDS_ITER_INTEGRAL_TYPE /* Allows overriding the integral type for indexing. E.g: CUDA uses int instead of int64 for better perf */
#define MAG_COORDS_ITER_INTEGRAL_TYPE int64_t
#endif

typedef MAG_COORDS_ITER_INTEGRAL_TYPE mag_codim_t;

typedef struct mag_coords_iter_t {
  mag_codim_t rank;
  mag_codim_t shape[MAG_MAX_DIMS];
  mag_codim_t strides[MAG_MAX_DIMS];
} mag_coords_iter_t;

static inline void mag_coords_iter_init(mag_coords_iter_t *ci, const mag_coords_t *co) {
  ci->rank = co->rank;
  for (mag_codim_t k=0; k < ci->rank; ++k) {
    mag_codim_t dim = co->shape[k];
    mag_assert(dim >= 0, "coords_iter: dimension size must be >= 0 (got %" PRIi64 ").", (int64_t)dim);
    ci->shape[k] = dim;
    ci->strides[k] = co->strides[k];
  }
}

static MAG_CUDA_DEVICE inline mag_codim_t mag_coords_iter_offset_at(const mag_coords_iter_t *ci, const mag_codim_t *idx) {
  mag_codim_t o=0;
  for (mag_codim_t k=0; k < ci->rank; ++k)
    o += idx[k]*ci->strides[k];
  return o;
}

static MAG_CUDA_DEVICE inline mag_codim_t mag_coords_iter_to_offset(const mag_coords_iter_t *cr, mag_codim_t i) {
  const mag_codim_t *restrict rd = cr->shape;
  const mag_codim_t *restrict rs = cr->strides;
  mag_codim_t ra = cr->rank-1;
  mag_codim_t o = 0;
  for (mag_codim_t k=ra; k >= 0; --k) {
    mag_codim_t dim = rd[k];
    mag_codim_t ax;
    ax = i % dim;
    i /= dim;
    o += ax*rs[k];
  }
  return o;
}

static MAG_CUDA_DEVICE inline mag_codim_t mag_coords_iter_broadcast(mag_coords_iter_t *cr, const mag_coords_iter_t *cx, mag_codim_t i) {
  const mag_codim_t *restrict rd = cr->shape;
  const mag_codim_t *restrict xd = cx->shape;
  const mag_codim_t *restrict xs = cx->strides;
  mag_codim_t ra = cr->rank;
  mag_codim_t delta = ra-- - cx->rank;
  mag_codim_t o = 0;
  for (mag_codim_t k=ra; k >= 0; --k) {
    mag_codim_t dim = rd[k];
    mag_codim_t ax;
    ax = i % dim;
    i /= dim;
    mag_codim_t kd = k-delta;
    if (kd >= 0 && xd[kd] > 1)
      o += ax*xs[kd];
  }
  return o;
}

static MAG_CUDA_DEVICE inline mag_codim_t mag_coords_iter_repeat(mag_coords_iter_t *cr, const mag_coords_iter_t *cx, mag_codim_t i) {
  const mag_codim_t *restrict rd = cr->shape;
  const mag_codim_t *restrict rs = cr->strides;
  const mag_codim_t *restrict xd = cx->shape;
  mag_codim_t rr = cr->rank;
  mag_codim_t rx = cx->rank;
  mag_codim_t delta = rx-- - rr;
  mag_codim_t o = 0;
  for (mag_codim_t k=rx; k >= 0; --k) {
    mag_codim_t dim = xd[k];
    mag_codim_t ax;
    ax = i % dim;
    i /= dim;
    mag_codim_t kd = k - delta;
    if (kd < 0) continue;
    o += ax % rd[kd]*rs[kd];
  }
  return o;
}

static MAG_CUDA_DEVICE inline void mag_coords_iter_offset2(
  const mag_coords_iter_t *cr,
  const mag_coords_iter_t *cx,
  mag_codim_t i,
  mag_codim_t *oir,
  mag_codim_t *oix
) {
  const mag_codim_t *restrict rd = cr->shape;
  const mag_codim_t *restrict rs = cr->strides;
  const mag_codim_t *restrict xd = cx->shape;
  const mag_codim_t *restrict xs = cx->strides;
  mag_codim_t rr = cr->rank;
  mag_codim_t rx = cx->rank;
  mag_codim_t dx = rr-rx;
  mag_codim_t ir = 0;
  mag_codim_t ix = 0;
  for (mag_codim_t k=rr-1; k >= 0; --k) {
    mag_codim_t dim = rd[k];
    mag_codim_t ax;
    ax = i % dim;
    i /= dim;
    ir += ax*rs[k];
    mag_codim_t kx = k-dx;
    if (kx >= 0 && xd[kx] > 1)
      ix += ax*xs[kx];
  }
  *oir = ir;
  *oix = ix;
}

static MAG_CUDA_DEVICE inline void mag_coords_iter_offset3(
  mag_coords_iter_t *cr,
  const mag_coords_iter_t *cx,
  const mag_coords_iter_t *cy,
  mag_codim_t i,
  mag_codim_t *oir,
  mag_codim_t *oix,
  mag_codim_t *oiy
) {
  const mag_codim_t *restrict rd = cr->shape;
  const mag_codim_t *restrict rs = cr->strides;
  const mag_codim_t *restrict xd = cx->shape;
  const mag_codim_t *restrict xs = cx->strides;
  const mag_codim_t *restrict yd = cy->shape;
  const mag_codim_t *restrict ys = cy->strides;
  mag_codim_t rr = cr->rank;
  mag_codim_t rx = cx->rank;
  mag_codim_t ry = cy->rank;
  mag_codim_t dx = rr-rx;
  mag_codim_t dy = rr-ry;
  mag_codim_t ir = 0;
  mag_codim_t ix = 0;
  mag_codim_t iy = 0;
  for (mag_codim_t k=rr-1; k >= 0; --k) {
    mag_codim_t dim = rd[k];
    mag_codim_t ax;
    ax = i % dim;
    i /= dim;
    ir += ax*rs[k];
    mag_codim_t kx = k-dx;
    if (kx >= 0 && xd[kx] > 1)
      ix += ax*xs[kx];
    mag_codim_t ky = k-dy;
    if (ky >= 0 && yd[ky] > 1)
      iy += ax*ys[ky];
  }
  *oir = ir;
  *oix = ix;
  *oiy = iy;
}

static MAG_CUDA_DEVICE inline void mag_coords_iter_offset4(
  mag_coords_iter_t *cr,
  const mag_coords_iter_t *ca,
  const mag_coords_iter_t *cb,
  const mag_coords_iter_t *cc,
  mag_codim_t i,
  mag_codim_t *oir,
  mag_codim_t *oia,
  mag_codim_t *oib,
  mag_codim_t *oic
) {
  const mag_codim_t *restrict rd = cr->shape;
  const mag_codim_t *restrict rs = cr->strides;
  const mag_codim_t *restrict ad = ca->shape;
  const mag_codim_t *restrict as = ca->strides;
  const mag_codim_t *restrict bd = cb->shape;
  const mag_codim_t *restrict bs = cb->strides;
  const mag_codim_t *restrict cd = cc->shape;
  const mag_codim_t *restrict cs = cc->strides;
  mag_codim_t rr = cr->rank;
  mag_codim_t ra = ca->rank;
  mag_codim_t rb = cb->rank;
  mag_codim_t rc = cc->rank;
  mag_codim_t da = rr - ra;
  mag_codim_t db = rr - rb;
  mag_codim_t dc = rr - rc;
  mag_codim_t ir = 0;
  mag_codim_t ia = 0;
  mag_codim_t ib = 0;
  mag_codim_t ic = 0;
  for (mag_codim_t k=rr-1; k >= 0; --k) {
    mag_codim_t dim = rd[k];
    mag_codim_t ax = i % dim;
    i /= dim;
    ir += ax*rs[k];
    mag_codim_t ka = k - da;
    if (ka >= 0 && ad[ka] > 1)
      ia += ax*as[ka];
    mag_codim_t kb = k - db;
    if (kb >= 0 && bd[kb] > 1)
      ib += ax*bs[kb];
    mag_codim_t kc = k - dc;
    if (kc >= 0 && cd[kc] > 1)
      ic += ax*cs[kc];
  }
  *oir = ir;
  *oia = ia;
  *oib = ib;
  *oic = ic;
}

#ifdef __cplusplus
}
#endif

#endif
