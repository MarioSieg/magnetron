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

#include "mag_immapi.h"
#include "mag_alloc.h"

void mag_imm_start(mag_imm_t *i, mag_error_t *err) {
  i->err_sink = err;
  i->sticky_stat = MAG_OK;
  i->pool = i->inline_pool;
  i->len = 0;
  i->cap = MAG_IMM_INLINE_CAP;
}

mag_status_t mag_imm_end(mag_imm_t *i) {
  for (size_t k=0; k < i->len; ++k)
    mag_tensor_decref(i->pool[k]);
  if (i->pool != i->inline_pool)
    (*mag_alloc)(i->pool, 0, 0);
  i->pool = i->inline_pool;
  i->len = 0;
  i->cap = MAG_IMM_INLINE_CAP;
  return i->sticky_stat;
}

static void mag_imm_push(mag_imm_t *i, mag_tensor_t *t) {
  if (mag_unlikely(i->len == i->cap)) {
    size_t cap = i->cap<<1;
    mag_tensor_t **pool = (*mag_alloc)(NULL, cap*sizeof(*pool), 0);
    memcpy(pool, i->pool, i->len*sizeof(*pool));
    if (i->pool != i->inline_pool) (*mag_alloc)(i->pool, 0, 0);
    i->pool = pool;
    i->cap = cap;
  }
  i->pool[i->len++] = t;
}

static bool mag_imm_live(mag_imm_t *i, bool operands_ok) {
  if (mag_iserr(i->sticky_stat)) return false;
  if (mag_unlikely(!operands_ok)) {
    i->sticky_stat = mag_set_error(i->err_sink, MAG_ERR_PARAM, "immapi: NULL operand passed to an op on a healthy scope.");
    return false;
  }
  return true;
}

mag_tensor_t *mag_imm_track(mag_imm_t *i, mag_status_t sticky_stat, mag_tensor_t **pt) {
  mag_tensor_t *t = *pt;
  *pt = NULL;
  if (mag_iserr(sticky_stat)) {
    if (!mag_iserr(i->sticky_stat)) i->sticky_stat = sticky_stat;
    if (t) mag_tensor_decref(t);
    return NULL;
  }
  if (!t) return NULL;
  mag_imm_push(i, t);
  return t;
}

bool mag_imm_check(mag_imm_t *i, mag_status_t sticky_stat) {
  if (mag_iserr(sticky_stat) && !mag_iserr(i->sticky_stat)) i->sticky_stat = sticky_stat;
  return mag_imm_ok(i);
}

mag_tensor_t *mag_imm_adopt(mag_imm_t *i, mag_tensor_t *t) {
  if (t) mag_imm_push(i, t);
  return t;
}

mag_tensor_t *mag_imm_retain(mag_imm_t *i, mag_tensor_t *t) {
  if (!t) return NULL;
  mag_tensor_incref(t);
  mag_imm_push(i, t);
  return t;
}

void mag_imm_collect(mag_imm_t *i, size_t mark, mag_tensor_t *const *keep, size_t num_keep) {
  mag_assert2(mark <= i->len);
  size_t w = mark;
  for (size_t k=mark; k < i->len; ++k) {
    mag_tensor_t *t = i->pool[k];
    bool kept = false;
    for (size_t j=0; !kept && j < num_keep; ++j) kept = keep[j] == t;
    if (kept) i->pool[w++] = t;
    else mag_tensor_decref(t);
  }
  i->len = w;
}

bool mag_imm_yield(mag_imm_t *i, mag_tensor_t **out, mag_tensor_t *t) {
  if (!mag_imm_live(i, t != NULL)) return false;
  mag_tensor_incref(t);
  *out = t;
  return true;
}

mag_status_t mag_imm_finalize(mag_imm_t *i, mag_tensor_t **out, mag_tensor_t *t) {
  mag_imm_yield(i, out, t);
  return mag_imm_end(i);
}

#define mag_imm_exec(i, operands_ok, call) do { \
    if (!mag_imm_live((i), (operands_ok))) return NULL; \
    mag_tensor_t *r_ = NULL; \
    mag_status_t st_ = (call); \
    return mag_imm_track((i), st_, &r_); \
  } while (0)

#define mag_imm_def_unary(name) \
  mag_tensor_t *mag__##name(mag_imm_t *i, mag_tensor_t *x) { mag_imm_exec(i, x != NULL, mag_##name(i->err_sink, &r_, x)); }

#define mag_imm_def_binary(name) \
  mag_tensor_t *mag__##name(mag_imm_t *i, mag_tensor_t *x, mag_tensor_t *y) { mag_imm_exec(i, x && y, mag_##name(i->err_sink, &r_, x, y)); }

#define mag_imm_def_reduce(name) \
  mag_tensor_t *mag__##name(mag_imm_t *i, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) { mag_imm_exec(i, x != NULL, mag_##name(i->err_sink, &r_, x, dims, rank, keepdim)); }

mag_tensor_t *mag__scalar(mag_imm_t *i, mag_tensor_t *like, double v) {
  mag_imm_exec(i, like != NULL, mag_scalar(i->err_sink, &r_, like->ctx, like->meta.dtype, mag_scalar_from_float64(v), mag_tensor_device_id(like)));
}

mag_tensor_t *mag__scalar_of(mag_imm_t *i, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t v, mag_device_id_t device) {
  mag_imm_exec(i, true, mag_scalar(i->err_sink, &r_, ctx, type, v, device));
}

mag_tensor_t *mag__full(mag_imm_t *i, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t v, mag_device_id_t device) {
  mag_imm_exec(i, true, mag_full(i->err_sink, &r_, ctx, type, rank, shape, v, device));
}

mag_tensor_t *mag__full_like(mag_imm_t *i, mag_tensor_t *like, double v) {
  mag_imm_exec(i, like != NULL, mag_full_like(i->err_sink, &r_, like, mag_scalar_from_float64(v)));
}

mag_tensor_t *mag__zeros(mag_imm_t *i, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  mag_imm_exec(i, true, mag_zeros(i->err_sink, &r_, ctx, type, rank, shape, device));
}

mag_imm_def_unary(zeros_like)
mag_imm_def_unary(ones_like)
mag_imm_def_unary(empty_like)

mag_tensor_t *mag__arange(mag_imm_t *i, mag_context_t *ctx, mag_dtype_t type, int64_t start, int64_t end, int64_t step, mag_device_id_t device) {
  mag_imm_exec(i, true, mag_arange(i->err_sink, &r_, ctx, type, mag_scalar_from_int64(start), mag_scalar_from_int64(end), mag_scalar_from_int64(step), device));
}

mag_imm_def_unary(clone)
mag_imm_def_unary(contiguous)
mag_imm_def_unary(neg)
mag_imm_def_unary(abs)
mag_imm_def_unary(step)
mag_imm_def_unary(sqr)
mag_imm_def_unary(sqrt)
mag_imm_def_unary(rsqrt)
mag_imm_def_unary(rcp)
mag_imm_def_unary(exp)
mag_imm_def_unary(exp2)
mag_imm_def_unary(log)
mag_imm_def_unary(sin)
mag_imm_def_unary(cos)
mag_imm_def_unary(sinh)
mag_imm_def_unary(cosh)
mag_imm_def_unary(tanh)
mag_imm_def_unary(erf)
mag_imm_def_unary(sigmoid)
mag_imm_def_unary(i0)
mag_imm_def_unary(T)

mag_tensor_t *mag__cast(mag_imm_t *i, mag_tensor_t *x, mag_dtype_t type) {
  mag_imm_exec(i, x != NULL, mag_cast(i->err_sink, &r_, x, type));
}

mag_imm_def_binary(add)
mag_imm_def_binary(sub)
mag_imm_def_binary(mul)
mag_imm_def_binary(div)
mag_imm_def_binary(mod)
mag_imm_def_binary(floordiv)
mag_imm_def_binary(pow)
mag_imm_def_binary(min)
mag_imm_def_binary(max)
mag_imm_def_binary(eq)
mag_imm_def_binary(lt)
mag_imm_def_binary(le)
mag_imm_def_binary(gt)
mag_imm_def_binary(ge)
mag_imm_def_binary(and)
mag_imm_def_binary(matmul)
mag_imm_def_binary(outer)
mag_imm_def_binary(repeat_back)

typedef mag_tensor_t *(mag_imm_binop_t)(mag_imm_t *, mag_tensor_t *, mag_tensor_t *);

static mag_tensor_t *mag_imm_binop_scalar(mag_imm_t *i, mag_imm_binop_t *op, mag_tensor_t *x, double v, bool scalar_lhs) {
  if (!mag_imm_live(i, x != NULL)) return NULL;
  mag_tensor_t *c = mag__scalar(i, x, v);
  return scalar_lhs ? (*op)(i, c, x) : (*op)(i, x, c);
}

mag_tensor_t *mag__adds(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__add, x, v, false); }
mag_tensor_t *mag__subs(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__sub, x, v, false); }
mag_tensor_t *mag__rsubs(mag_imm_t *i, double v, mag_tensor_t *x) { return mag_imm_binop_scalar(i, &mag__sub, x, v, true); }
mag_tensor_t *mag__muls(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__mul, x, v, false); }
mag_tensor_t *mag__divs(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__div, x, v, false); }
mag_tensor_t *mag__eqs(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__eq, x, v, false); }
mag_tensor_t *mag__lts(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__lt, x, v, false); }
mag_tensor_t *mag__gts(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__gt, x, v, false); }
mag_tensor_t *mag__clamp_mins(mag_imm_t *i, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(i, &mag__max, x, v, false); }

mag_tensor_t *mag__where(mag_imm_t *i, mag_tensor_t *cond, mag_tensor_t *x, mag_tensor_t *y) {
  mag_imm_exec(i, cond && x && y, mag_where(i->err_sink, &r_, cond, x, y));
}

mag_tensor_t *mag__clamp(mag_imm_t *i, mag_tensor_t *x, mag_tensor_t *lo, mag_tensor_t *hi) {
  mag_imm_exec(i, x && lo && hi, mag_clamp(i->err_sink, &r_, x, lo, hi));
}

mag_imm_def_reduce(sum)
mag_imm_def_reduce(maxima)
mag_imm_def_reduce(minima)
mag_imm_def_reduce(any)
mag_imm_def_reduce(argmax)
mag_imm_def_reduce(argmin)

mag_tensor_t *mag__cusum(mag_imm_t *i, mag_tensor_t *x, int64_t dim) { mag_imm_exec(i, x != NULL, mag_cusum(i->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__cuprod(mag_imm_t *i, mag_tensor_t *x, int64_t dim) { mag_imm_exec(i, x != NULL, mag_cuprod(i->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__reshape(mag_imm_t *i, mag_tensor_t *x, const int64_t *dims, int64_t rank) { mag_imm_exec(i, x != NULL, mag_reshape(i->err_sink, &r_, x, dims, rank)); }
mag_tensor_t *mag__view(mag_imm_t *i, mag_tensor_t *x, const int64_t *dims, int64_t rank) { mag_imm_exec(i, x != NULL, mag_view(i->err_sink, &r_, x, dims, rank)); }
mag_tensor_t *mag__transpose(mag_imm_t *i, mag_tensor_t *x, int64_t dim1, int64_t dim2) { mag_imm_exec(i, x != NULL, mag_transpose(i->err_sink, &r_, x, dim1, dim2)); }
mag_tensor_t *mag__permute(mag_imm_t *i, mag_tensor_t *x, const int64_t *dims, int64_t rank) { mag_imm_exec(i, x != NULL, mag_permute(i->err_sink, &r_, x, dims, rank)); }
mag_tensor_t *mag__unsqueeze(mag_imm_t *i, mag_tensor_t *x, int64_t dim) { mag_imm_exec(i, x != NULL, mag_unsqueeze(i->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__squeeze_dim(mag_imm_t *i, mag_tensor_t *x, int64_t dim) { mag_imm_exec(i, x != NULL, mag_squeeze_dim(i->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__movedim(mag_imm_t *i, mag_tensor_t *x, int64_t src, int64_t dst) { mag_imm_exec(i, x != NULL, mag_movedim(i->err_sink, &r_, x, src, dst)); }
mag_tensor_t *mag__narrow(mag_imm_t *i, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length) { mag_imm_exec(i, x != NULL, mag_narrow(i->err_sink, &r_, x, dim, start, length)); }
mag_tensor_t *mag__flip(mag_imm_t *i, mag_tensor_t *x, const int64_t *dims, int64_t ndims) { mag_imm_exec(i, x != NULL, mag_flip(i->err_sink, &r_, x, dims, ndims)); }
mag_tensor_t *mag__expand(mag_imm_t *i, mag_tensor_t *x, int64_t rank, const int64_t *shape) { mag_imm_exec(i, x != NULL, mag_expand(i->err_sink, &r_, x, rank, shape)); }
mag_tensor_t *mag__broadcast(mag_imm_t *i, mag_tensor_t *x, int64_t rank, const int64_t *shape) { mag_imm_exec(i, x != NULL, mag_broadcast(i->err_sink, &r_, x, rank, shape)); }
mag_tensor_t *mag__repeat(mag_imm_t *i, mag_tensor_t *x, const int64_t *repeats, int64_t num_reps) { mag_imm_exec(i, x != NULL, mag_repeat(i->err_sink, &r_, x, repeats, num_reps)); }

mag_tensor_t *mag__pad(mag_imm_t *i, mag_tensor_t *x, const int64_t *pad, int64_t pad_len, mag_pad_mode_t mode, mag_scalar_t value) {
  mag_imm_exec(i, x != NULL, mag_pad(i->err_sink, &r_, x, pad, pad_len, mode, value));
}

mag_tensor_t *mag__strided_view(mag_imm_t *i, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
  mag_imm_exec(i, base != NULL, mag_strided_view(i->err_sink, &r_, base->ctx, base, rank, shape, strides, offset));
}

mag_tensor_t *mag__cat(mag_imm_t *i, mag_tensor_t **tensors, size_t count, int64_t dim) {
  bool all = tensors != NULL;
  for (size_t j=0; all && j < count; ++j) all = tensors[j] != NULL;
  mag_imm_exec(i, all, mag_cat(i->err_sink, &r_, tensors, count, dim));
}

bool mag__split(mag_imm_t *i, mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim) {
  for (int64_t j=0; j < num_splits; ++j) outs[j] = NULL;
  if (!mag_imm_live(i, x != NULL)) return false;
  if (!mag_imm_check(i, mag_split(i->err_sink, outs, num_splits, x, split_size, dim))) return false;
  for (int64_t j=0; j < num_splits; ++j) mag_imm_adopt(i, outs[j]);
  return true;
}

mag_tensor_t *mag__gather(mag_imm_t *i, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx) {
  mag_imm_exec(i, x && idx, mag_gather(i->err_sink, &r_, x, dim, idx));
}

mag_tensor_t *mag__scatter(mag_imm_t *i, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src) {
  mag_imm_exec(i, x && idx && src, mag_scatter(i->err_sink, &r_, x, dim, idx, src));
}

mag_tensor_t *mag__scatter_add(mag_imm_t *i, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src) {
  mag_imm_exec(i, x && idx && src, mag_scatter_add(i->err_sink, &r_, x, dim, idx, src));
}

bool mag__fill_(mag_imm_t *i, mag_tensor_t *x, mag_scalar_t v) {
  return mag_imm_live(i, x != NULL) && mag_imm_check(i, mag_fill_(i->err_sink, x, v));
}

bool mag__scatter_add_(mag_imm_t *i, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src) {
  return mag_imm_live(i, x && idx && src) && mag_imm_check(i, mag_scatter_add_(i->err_sink, x, dim, idx, src));
}

bool mag__index_add_(mag_imm_t *i, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src, double alpha) {
  return mag_imm_live(i, x && idx && src) && mag_imm_check(i, mag_index_add_(i->err_sink, x, dim, idx, src, alpha));
}

#undef mag_imm_def_reduce
#undef mag_imm_def_binary
#undef mag_imm_def_unary
#undef mag_imm_exec
