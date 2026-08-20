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

#include "mag_cpu_vectorize_plan.h"

bool mag_unary_vectorization_plan_init(mag_unary_vectorization_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x) {
  int64_t rank = r->meta.coords.rank;
  if (x->meta.coords.rank != rank) return false;
  const int64_t *rs = r->meta.coords.shape;
  const int64_t *xs = x->meta.coords.shape;
  const int64_t *rt = r->meta.coords.strides;
  const int64_t *xt = x->meta.coords.strides;
  for (int64_t d=0; d < rank; ++d)
    if (xs[d] != rs[d]) return false;
  int64_t inner = 1;
  int64_t d = rank-1;
  for (; d >= 0; --d) {
    if (rs[d] == 1) continue;
    if (rt[d] != inner || xt[d] != inner) break;
    inner *= rs[d];
  }
  if (inner <= 1) return false;
  p->inner = inner;
  p->outer_rank = d+1;
  for (int64_t k=0; k <= d; ++k) {
    p->shape[k] = rs[k];
    p->rstr[k] = rs[k] == 1 ? 0 : rt[k];
    p->xstr[k] = rs[k] == 1 ? 0 : xt[k];
  }
  return true;
}

void mag_unary_vectorization_plan_step(const mag_unary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb) {
  int64_t ri=0, xi=0;
  for (int64_t k=p->outer_rank-1; k >= 0; --k) {
    int64_t c = o%p->shape[k];
    o /= p->shape[k];
    ri += c*p->rstr[k];
    xi += c*p->xstr[k];
  }
  *rb = ri;
  *xb = xi;
}

bool mag_binary_vectorization_plan_init(mag_binary_vectorization_plan_t *p, const mag_tensor_t *r, const mag_tensor_t *x, const mag_tensor_t *y) {
  if (!(mag_tensor_is_contiguous(r) && mag_tensor_is_contiguous(x) && mag_tensor_is_contiguous(y))) return false;
  bool x1 = x->meta.numel == 1;
  bool y1 = y->meta.numel == 1;
  if (x1 != y1) {
    const mag_tensor_t *big = x1 ? y : x;
    if (big->meta.numel != r->meta.numel) return false;
    p->inner = r->meta.numel;
    p->outer_rank = 0;
    p->x_const = x1;
    p->y_const = y1;
    return p->inner > 1;
  }
  int64_t rank = r->meta.coords.rank;
  if (x->meta.coords.rank != rank || y->meta.coords.rank != rank) return false;
  const int64_t *rs = r->meta.coords.shape;
  const int64_t *xs = x->meta.coords.shape;
  const int64_t *ys = y->meta.coords.shape;
  for (int64_t d=0; d < rank; ++d)
    if (!((xs[d] == rs[d] || xs[d] == 1) && (ys[d] == rs[d] || ys[d] == 1))) return false;
  bool xf=true, xc=true, yf=true, yc=true;
  int64_t inner = 1;
  int64_t d = rank-1;
  for (; d >= 0; --d) {
    bool nxf = xf && xs[d] == rs[d], nxc = xc && xs[d] == 1;
    bool nyf = yf && ys[d] == rs[d], nyc = yc && ys[d] == 1;
    if (!(nxf || nxc) || !(nyf || nyc)) break;
    xf = nxf; xc = nxc; yf = nyf; yc = nyc;
    inner *= rs[d];
  }
  if (inner <= 1) return false;
  p->inner = inner;
  p->x_const = xc;
  p->y_const = yc;
  p->outer_rank = d+1;
  for (int64_t k=0; k <= d; ++k) {
    p->shape[k] = rs[k];
    p->xstr[k] = xs[k] == 1 ? 0 : x->meta.coords.strides[k];
    p->ystr[k] = ys[k] == 1 ? 0 : y->meta.coords.strides[k];
  }
  return true;
}

void mag_binary_vectorization_plan_step(const mag_binary_vectorization_plan_t *p, int64_t o, int64_t *xb, int64_t *yb) {
  int64_t xi=0, yi=0;
  for (int64_t k=p->outer_rank-1; k >= 0; --k) {
    int64_t c = o%p->shape[k];
    o /= p->shape[k];
    xi += c*p->xstr[k];
    yi += c*p->ystr[k];
  }
  *xb = xi;
  *yb = yi;
}
