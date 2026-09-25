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
  int64_t xr = x->meta.coords.rank;
  if (xr > rank) return false;
  const int64_t *rs = r->meta.coords.shape;
  const int64_t *rt = r->meta.coords.strides;
  int64_t xs[MAG_MAX_DIMS], xt[MAG_MAX_DIMS];
  int64_t dx = rank-xr;
  for (int64_t d=0; d < rank; ++d) {
    xs[d] = d < dx ? 1 : x->meta.coords.shape[d-dx];
    xt[d] = d < dx ? 0 : x->meta.coords.strides[d-dx];
  }
  for (int64_t d=0; d < rank; ++d)
    if (!(xs[d] == rs[d] || xs[d] == 1)) return false;
  bool xf=true, xc=true;
  int64_t inner = 1;
  int64_t d = rank-1;
  for (; d >= 0; --d) {
    if (rs[d] == 1) continue;
    if (rt[d] != inner) break;
    bool xb = xs[d] == 1 || xt[d] == 0;
    bool nxf = xf && !xb && xt[d] == inner;
    bool nxc = xc && xb;
    if (!(nxf || nxc)) break;
    xf = nxf; xc = nxc;
    inner *= rs[d];
  }
  if (inner <= 1) return false;
  p->inner = inner;
  p->x_const = xc;
  p->outer_rank = d+1;
  for (int64_t k=0; k <= d; ++k) {
    p->shape[k] = rs[k];
    p->rstr[k] = rs[k] == 1 ? 0 : rt[k];
    p->xstr[k] = xs[k] == 1 ? 0 : xt[k];
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
  int64_t rank = r->meta.coords.rank;
  int64_t xr = x->meta.coords.rank;
  int64_t yr = y->meta.coords.rank;
  if (xr > rank || yr > rank) return false;
  const int64_t *rs = r->meta.coords.shape;
  const int64_t *rt = r->meta.coords.strides;
  int64_t xs[MAG_MAX_DIMS], ys[MAG_MAX_DIMS], xst[MAG_MAX_DIMS], yst[MAG_MAX_DIMS];
  int64_t dx = rank-xr, dy = rank-yr;
  for (int64_t d=0; d < rank; ++d) {
    xs[d] = d < dx ? 1 : x->meta.coords.shape[d-dx];
    xst[d] = d < dx ? 0 : x->meta.coords.strides[d-dx];
    ys[d] = d < dy ? 1 : y->meta.coords.shape[d-dy];
    yst[d] = d < dy ? 0 : y->meta.coords.strides[d-dy];
  }
  for (int64_t d=0; d < rank; ++d)
    if (!((xs[d] == rs[d] || xs[d] == 1) && (ys[d] == rs[d] || ys[d] == 1))) return false;
  bool xf=true, xc=true, yf=true, yc=true;
  int64_t inner = 1;
  int64_t d = rank-1;
  for (; d >= 0; --d) {
    if (rs[d] == 1) continue;
    if (rt[d] != inner) break;
    bool xb = xs[d] == 1 || xst[d] == 0;
    bool yb = ys[d] == 1 || yst[d] == 0;
    bool nxf = xf && !xb && xst[d] == inner, nxc = xc && xb;
    bool nyf = yf && !yb && yst[d] == inner, nyc = yc && yb;
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
    p->rstr[k] = rs[k] == 1 ? 0 : rt[k];
    p->xstr[k] = xs[k] == 1 ? 0 : xst[k];
    p->ystr[k] = ys[k] == 1 ? 0 : yst[k];
  }
  return true;
}

void mag_binary_vectorization_plan_step(const mag_binary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb, int64_t *yb) {
  int64_t ri=0, xi=0, yi=0;
  for (int64_t k=p->outer_rank-1; k >= 0; --k) {
    int64_t c = o%p->shape[k];
    o /= p->shape[k];
    ri += c*p->rstr[k];
    xi += c*p->xstr[k];
    yi += c*p->ystr[k];
  }
  *rb = ri;
  *xb = xi;
  *yb = yi;
}

static int64_t mag_stride_mag(int64_t s) { return s < 0 ? -s : s; }

bool mag_tile_plan_init(mag_tile_plan_t *p, const mag_tensor_t *r, const int64_t *rt, const int64_t *xt, const int64_t *yt, int64_t tile) {
  int64_t rank = r->meta.coords.rank;
  const int64_t *rs = r->meta.coords.shape;
  int64_t a = -1, b = -1;
  for (int64_t d=0; d < rank; ++d) {
    if (rs[d] <= 1) continue;
    if (a < 0 || mag_stride_mag(rt[d]) < mag_stride_mag(rt[a])) a = d;
  }
  if (a < 0) return false;
  for (int64_t d=0; d < rank; ++d) {
    if (rs[d] <= 1 || d == a) continue;
    if (mag_stride_mag(xt[d]) == 1 || (yt && mag_stride_mag(yt[d]) == 1)) { b = d; break; }
  }
  if (b < 0) {
    for (int64_t d=0; d < rank; ++d) {
      if (rs[d] <= 1 || d == a) continue;
      if (b < 0 || mag_stride_mag(rt[d]) < mag_stride_mag(rt[b])) b = d;
    }
  }
  if (b < 0) return false;
  p->rank = rank;
  p->a = a;
  p->b = b;
  p->tile = tile;
  int64_t n = 1, c = 1;
  for (int64_t d=rank-1; d >= 0; --d) {
    p->cstr[d] = c;
    c *= rs[d];
  }
  for (int64_t d=0; d < rank; ++d) {
    p->grid[d] = (d == a || d == b) ? (rs[d]+tile-1)/tile : rs[d];
    n *= p->grid[d];
  }
  p->ntiles = n;
  return true;
}

void mag_tile_plan_origin(const mag_tile_plan_t *p, int64_t i, int64_t *o) {
  for (int64_t d=p->rank-1; d >= 0; --d) {
    int64_t c = i%p->grid[d];
    i /= p->grid[d];
    o[d] = (d == p->a || d == p->b) ? c*p->tile : c;
  }
}
