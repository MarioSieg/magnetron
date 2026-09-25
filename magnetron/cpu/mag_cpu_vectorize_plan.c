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
  p->flags = 0;
  int64_t rank = r->meta.coords.rank;
  int64_t xr = x->meta.coords.rank;
  if (xr > rank) return false;
  const int64_t *rs = r->meta.coords.shape;
  const int64_t *rt = r->meta.coords.strides;
  int64_t xs[MAG_MAX_DIMS], xt[MAG_MAX_DIMS];
  int64_t ax = rank - xr;
  for (int64_t dim=0; dim < rank; ++dim) {
    xs[dim] = dim < ax ? 1 : x->meta.coords.shape[dim-ax];
    xt[dim] = dim < ax ? 0 : x->meta.coords.strides[dim-ax];
  }
  for (int64_t dim = 0; dim < rank; ++dim)
    if (!(xs[dim] == rs[dim] || xs[dim] == 1))
      return false;
  p->flags = MAG_VAX_CX|MAG_VAX_FX;
  int64_t inner = 1;
  int64_t dim = rank-1;
  for (; dim >= 0; --dim) {
    if (rs[dim] == 1) continue;
    if (rt[dim] != inner) break;
    mag_vectorize_flags_t nf = p->flags;
    nf &= xs[dim] == 1 || xt[dim] == 0 ? ~MAG_VAX_FX : ~MAG_VAX_CX;
    if (xt[dim] != inner) nf &= ~MAG_VAX_FX;
    if (!(nf & (MAG_VAX_FX|MAG_VAX_CX))) break;
    p->flags = nf;
    inner *= rs[dim];
  }
  if (inner <= 1) return false;
  p->inner = inner;
  p->outer_rank = dim+1;
  for (int64_t kax = 0; kax <= dim; ++kax) {
    p->shape[kax] = rs[kax];
    p->rstr[kax] = rs[kax] == 1 ? 0 : rt[kax];
    p->xstr[kax] = xs[kax] == 1 ? 0 : xt[kax];
  }
  return true;
}

void mag_unary_vectorization_plan_step(const mag_unary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb) {
  int64_t ri=0, xi=0;
  for (int64_t dim=p->outer_rank-1; dim >= 0; --dim) {
    int64_t ax = o%p->shape[dim];
    o /= p->shape[dim];
    ri += ax*p->rstr[dim];
    xi += ax*p->xstr[dim];
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
  int64_t xs[MAG_MAX_DIMS], ys[MAG_MAX_DIMS];
  int64_t xst[MAG_MAX_DIMS], yst[MAG_MAX_DIMS];
  int64_t dx = rank - xr;
  int64_t dy = rank - yr;
  for (int64_t dim=0; dim < rank; ++dim) {
    xs[dim] = dim < dx ? 1 : x->meta.coords.shape[dim-dx];
    xst[dim] = dim < dx ? 0 : x->meta.coords.strides[dim-dx];
    ys[dim] = dim < dy ? 1 : y->meta.coords.shape[dim-dy];
    yst[dim] = dim < dy ? 0 : y->meta.coords.strides[dim-dy];
  }
  for (int64_t dim=0; dim < rank; ++dim)
    if (!((xs[dim] == rs[dim] || xs[dim] == 1) && (ys[dim] == rs[dim] || ys[dim] == 1)))
      return false;
  mag_vectorize_flags_t flags = MAG_VAX_CX|MAG_VAX_CY|MAG_VAX_FX|MAG_VAX_FY;
  int64_t inner = 1;
  int64_t dim=rank-1;
  for (; dim >= 0; --dim) {
    if (rs[dim] == 1) continue;
    if (rt[dim] != inner) break;
    mag_vectorize_flags_t nf = flags;
    nf&=xs[dim] == 1 || xst[dim] == 0 ?~MAG_VAX_FX:~MAG_VAX_CX;
    nf&=ys[dim] == 1 || yst[dim] == 0 ?~MAG_VAX_FY:~MAG_VAX_CY;
    if (xst[dim] != inner) nf&=~MAG_VAX_FX;
    if (yst[dim] != inner) nf&=~MAG_VAX_FY;
    if (!(nf & (MAG_VAX_CX|MAG_VAX_FX)) || !(nf & (MAG_VAX_CY|MAG_VAX_FY))) break;
    flags = nf;
    inner *= rs[dim];
  }
  if (inner <= 1) return false;
  p->inner = inner;
  p->flags = flags;
  p->outer_rank = dim + 1;
  for (int64_t kax=0; kax <= dim; ++kax) {
    p->shape[kax] = rs[kax];
    p->rstr[kax] = rs[kax] == 1 ? 0 : rt[kax];
    p->xstr[kax] = xs[kax] == 1 ? 0 : xst[kax];
    p->ystr[kax] = ys[kax] == 1 ? 0 : yst[kax];
  }
  return true;
}

void mag_binary_vectorization_plan_step(const mag_binary_vectorization_plan_t *p, int64_t o, int64_t *rb, int64_t *xb, int64_t *yb) {
  int64_t ri=0, xi=0, yi=0;
  for (int64_t dim=p->outer_rank-1; dim >= 0; --dim) {
    int64_t ax = o%p->shape[dim];
    o /= p->shape[dim];
    ri += ax*p->rstr[dim];
    xi += ax*p->xstr[dim];
    yi += ax*p->ystr[dim];
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
    p->grid[d] = d == a || d == b ? (rs[d]+tile-1)/tile : rs[d];
    n *= p->grid[d];
  }
  p->ntiles = n;
  return true;
}

void mag_tile_plan_origin(const mag_tile_plan_t *p, int64_t i, int64_t *o) {
  for (int64_t d=p->rank-1; d >= 0; --d) {
    int64_t c = i%p->grid[d];
    i /= p->grid[d];
    o[d] = d == p->a || d == p->b ? c*p->tile : c;
  }
}
