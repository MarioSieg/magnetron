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

void mag_imm_start(mag_imm_t *s, mag_error_t *err) {
  s->err_sink = err;
  s->sticky_stat = MAG_OK;
  s->pool = s->inline_pool;
  s->len = 0;
  s->cap = MAG_IMM_INLINE_CAP;
}

mag_status_t mag_imm_end(mag_imm_t *s) {
  for (size_t i=0; i < s->len; ++i)
    mag_tensor_decref(s->pool[i]);
  if (s->pool != s->inline_pool)
    (*mag_alloc)(s->pool, 0, 0);
  s->pool = s->inline_pool;
  s->len = 0;
  s->cap = MAG_IMM_INLINE_CAP;
  return s->sticky_stat;
}

static void mag_imm_push(mag_imm_t *s, mag_tensor_t *t) {
  if (mag_unlikely(s->len == s->cap)) {
    size_t cap = s->cap<<1;
    mag_tensor_t **pool = (*mag_alloc)(NULL, cap*sizeof(*pool), 0);
    memcpy(pool, s->pool, s->len*sizeof(*pool));
    if (s->pool != s->inline_pool) (*mag_alloc)(s->pool, 0, 0);
    s->pool = pool;
    s->cap = cap;
  }
  s->pool[s->len++] = t;
}

static bool mag_imm_live(mag_imm_t *s, bool operands_ok) {
  if (mag_iserr(s->sticky_stat)) return false;
  if (mag_unlikely(!operands_ok)) {
    s->sticky_stat = mag_set_error(s->err_sink, MAG_ERR_PARAM, "immapi: NULL operand passed to an op on a healthy scope.");
    return false;
  }
  return true;
}

mag_tensor_t *mag_imm_track(mag_imm_t *s, mag_status_t sticky_stat, mag_tensor_t **pt) {
  mag_tensor_t *t = *pt;
  *pt = NULL;
  if (mag_iserr(sticky_stat)) {
    if (!mag_iserr(s->sticky_stat)) s->sticky_stat = sticky_stat;
    if (t) mag_tensor_decref(t);
    return NULL;
  }
  if (!t) return NULL;
  mag_imm_push(s, t);
  return t;
}

bool mag_imm_check(mag_imm_t *s, mag_status_t sticky_stat) {
  if (mag_iserr(sticky_stat) && !mag_iserr(s->sticky_stat)) s->sticky_stat = sticky_stat;
  return mag_imm_ok(s);
}

mag_tensor_t *mag_imm_adopt(mag_imm_t *s, mag_tensor_t *t) {
  if (t) mag_imm_push(s, t);
  return t;
}

mag_tensor_t *mag_imm_retain(mag_imm_t *s, mag_tensor_t *t) {
  if (!t) return NULL;
  mag_tensor_incref(t);
  mag_imm_push(s, t);
  return t;
}

void mag_imm_collect(mag_imm_t *s, size_t mark, mag_tensor_t *const *keep, size_t num_keep) {
  mag_assert2(mark <= s->len);
  size_t w = mark;
  for (size_t i=mark; i < s->len; ++i) {
    mag_tensor_t *t = s->pool[i];
    bool kept = false;
    for (size_t k=0; !kept && k < num_keep; ++k) kept = keep[k] == t;
    if (kept) s->pool[w++] = t;
    else mag_tensor_decref(t);
  }
  s->len = w;
}

bool mag_imm_yield(mag_imm_t *s, mag_tensor_t **out, mag_tensor_t *t) {
  if (!mag_imm_live(s, t != NULL)) return false;
  mag_tensor_incref(t);
  *out = t;
  return true;
}

mag_status_t mag_imm_finalize(mag_imm_t *s, mag_tensor_t **out, mag_tensor_t *t) {
  mag_imm_yield(s, out, t);
  return mag_imm_end(s);
}

#define mag_imm_exec(s, operands_ok, call) do { \
    if (!mag_imm_live((s), (operands_ok))) return NULL; \
    mag_tensor_t *r_ = NULL; \
    mag_status_t st_ = (call); \
    return mag_imm_track((s), st_, &r_); \
  } while (0)

#define mag_imm_def_unary(name) \
  mag_tensor_t *mag__##name(mag_imm_t *s, mag_tensor_t *x) { mag_imm_exec(s, x != NULL, mag_##name(s->err_sink, &r_, x)); }

#define mag_imm_def_binary(name) \
  mag_tensor_t *mag__##name(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y) { mag_imm_exec(s, x && y, mag_##name(s->err_sink, &r_, x, y)); }

#define mag_imm_def_reduce(name) \
  mag_tensor_t *mag__##name(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim) { mag_imm_exec(s, x != NULL, mag_##name(s->err_sink, &r_, x, dims, rank, keepdim)); }

mag_tensor_t *mag__scalar(mag_imm_t *s, mag_tensor_t *like, double v) {
  mag_imm_exec(s, like != NULL, mag_scalar(s->err_sink, &r_, like->ctx, like->meta.dtype, mag_scalar_from_float64(v), mag_tensor_device_id(like)));
}

mag_tensor_t *mag__scalar_of(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t v, mag_device_id_t device) {
  mag_imm_exec(s, true, mag_scalar(s->err_sink, &r_, ctx, type, v, device));
}

mag_tensor_t *mag__full(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t v, mag_device_id_t device) {
  mag_imm_exec(s, true, mag_full(s->err_sink, &r_, ctx, type, rank, shape, v, device));
}

mag_tensor_t *mag__full_like(mag_imm_t *s, mag_tensor_t *like, double v) {
  mag_imm_exec(s, like != NULL, mag_full_like(s->err_sink, &r_, like, mag_scalar_from_float64(v)));
}

mag_tensor_t *mag__zeros(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device) {
  mag_imm_exec(s, true, mag_zeros(s->err_sink, &r_, ctx, type, rank, shape, device));
}

mag_imm_def_unary(zeros_like)
mag_imm_def_unary(ones_like)
mag_imm_def_unary(empty_like)

mag_tensor_t *mag__arange(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, int64_t start, int64_t end, int64_t step, mag_device_id_t device) {
  mag_imm_exec(s, true, mag_arange(s->err_sink, &r_, ctx, type, mag_scalar_from_int64(start), mag_scalar_from_int64(end), mag_scalar_from_int64(step), device));
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

mag_tensor_t *mag__cast(mag_imm_t *s, mag_tensor_t *x, mag_dtype_t type) {
  mag_imm_exec(s, x != NULL, mag_cast(s->err_sink, &r_, x, type));
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

static mag_tensor_t *mag_imm_binop_scalar(mag_imm_t *s, mag_imm_binop_t *op, mag_tensor_t *x, double v, bool scalar_lhs) {
  if (!mag_imm_live(s, x != NULL)) return NULL;
  mag_tensor_t *c = mag__scalar(s, x, v);
  return scalar_lhs ? (*op)(s, c, x) : (*op)(s, x, c);
}

mag_tensor_t *mag__adds(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__add, x, v, false); }
mag_tensor_t *mag__subs(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__sub, x, v, false); }
mag_tensor_t *mag__rsubs(mag_imm_t *s, double v, mag_tensor_t *x) { return mag_imm_binop_scalar(s, &mag__sub, x, v, true); }
mag_tensor_t *mag__muls(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__mul, x, v, false); }
mag_tensor_t *mag__divs(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__div, x, v, false); }
mag_tensor_t *mag__eqs(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__eq, x, v, false); }
mag_tensor_t *mag__lts(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__lt, x, v, false); }
mag_tensor_t *mag__gts(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__gt, x, v, false); }
mag_tensor_t *mag__clamp_mins(mag_imm_t *s, mag_tensor_t *x, double v) { return mag_imm_binop_scalar(s, &mag__max, x, v, false); }

mag_tensor_t *mag__where(mag_imm_t *s, mag_tensor_t *cond, mag_tensor_t *x, mag_tensor_t *y) {
  mag_imm_exec(s, cond && x && y, mag_where(s->err_sink, &r_, cond, x, y));
}

mag_tensor_t *mag__clamp(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *lo, mag_tensor_t *hi) {
  mag_imm_exec(s, x && lo && hi, mag_clamp(s->err_sink, &r_, x, lo, hi));
}

mag_imm_def_reduce(sum)
mag_imm_def_reduce(maxima)
mag_imm_def_reduce(minima)
mag_imm_def_reduce(any)
mag_imm_def_reduce(argmax)
mag_imm_def_reduce(argmin)

mag_tensor_t *mag__cusum(mag_imm_t *s, mag_tensor_t *x, int64_t dim) { mag_imm_exec(s, x != NULL, mag_cusum(s->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__cuprod(mag_imm_t *s, mag_tensor_t *x, int64_t dim) { mag_imm_exec(s, x != NULL, mag_cuprod(s->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__reshape(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank) { mag_imm_exec(s, x != NULL, mag_reshape(s->err_sink, &r_, x, dims, rank)); }
mag_tensor_t *mag__view(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank) { mag_imm_exec(s, x != NULL, mag_view(s->err_sink, &r_, x, dims, rank)); }
mag_tensor_t *mag__transpose(mag_imm_t *s, mag_tensor_t *x, int64_t dim1, int64_t dim2) { mag_imm_exec(s, x != NULL, mag_transpose(s->err_sink, &r_, x, dim1, dim2)); }
mag_tensor_t *mag__permute(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank) { mag_imm_exec(s, x != NULL, mag_permute(s->err_sink, &r_, x, dims, rank)); }
mag_tensor_t *mag__unsqueeze(mag_imm_t *s, mag_tensor_t *x, int64_t dim) { mag_imm_exec(s, x != NULL, mag_unsqueeze(s->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__squeeze_dim(mag_imm_t *s, mag_tensor_t *x, int64_t dim) { mag_imm_exec(s, x != NULL, mag_squeeze_dim(s->err_sink, &r_, x, dim)); }
mag_tensor_t *mag__movedim(mag_imm_t *s, mag_tensor_t *x, int64_t src, int64_t dst) { mag_imm_exec(s, x != NULL, mag_movedim(s->err_sink, &r_, x, src, dst)); }
mag_tensor_t *mag__narrow(mag_imm_t *s, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length) { mag_imm_exec(s, x != NULL, mag_narrow(s->err_sink, &r_, x, dim, start, length)); }
mag_tensor_t *mag__flip(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t ndims) { mag_imm_exec(s, x != NULL, mag_flip(s->err_sink, &r_, x, dims, ndims)); }
mag_tensor_t *mag__expand(mag_imm_t *s, mag_tensor_t *x, int64_t rank, const int64_t *shape) { mag_imm_exec(s, x != NULL, mag_expand(s->err_sink, &r_, x, rank, shape)); }
mag_tensor_t *mag__broadcast(mag_imm_t *s, mag_tensor_t *x, int64_t rank, const int64_t *shape) { mag_imm_exec(s, x != NULL, mag_broadcast(s->err_sink, &r_, x, rank, shape)); }
mag_tensor_t *mag__repeat(mag_imm_t *s, mag_tensor_t *x, const int64_t *repeats, int64_t num_reps) { mag_imm_exec(s, x != NULL, mag_repeat(s->err_sink, &r_, x, repeats, num_reps)); }

mag_tensor_t *mag__pad(mag_imm_t *s, mag_tensor_t *x, const int64_t *pad, int64_t pad_len, mag_pad_mode_t mode, mag_scalar_t value) {
  mag_imm_exec(s, x != NULL, mag_pad(s->err_sink, &r_, x, pad, pad_len, mode, value));
}

mag_tensor_t *mag__strided_view(mag_imm_t *s, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset) {
  mag_imm_exec(s, base != NULL, mag_strided_view(s->err_sink, &r_, base->ctx, base, rank, shape, strides, offset));
}

mag_tensor_t *mag__cat(mag_imm_t *s, mag_tensor_t **tensors, size_t count, int64_t dim) {
  bool all = tensors != NULL;
  for (size_t i=0; all && i < count; ++i) all = tensors[i] != NULL;
  mag_imm_exec(s, all, mag_cat(s->err_sink, &r_, tensors, count, dim));
}

bool mag__split(mag_imm_t *s, mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim) {
  for (int64_t i=0; i < num_splits; ++i) outs[i] = NULL;
  if (!mag_imm_live(s, x != NULL)) return false;
  if (!mag_imm_check(s, mag_split(s->err_sink, outs, num_splits, x, split_size, dim))) return false;
  for (int64_t i=0; i < num_splits; ++i) mag_imm_adopt(s, outs[i]);
  return true;
}

mag_tensor_t *mag__gather(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx) {
  mag_imm_exec(s, x && idx, mag_gather(s->err_sink, &r_, x, dim, idx));
}

mag_tensor_t *mag__scatter(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src) {
  mag_imm_exec(s, x && idx && src, mag_scatter(s->err_sink, &r_, x, dim, idx, src));
}

mag_tensor_t *mag__scatter_add(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src) {
  mag_imm_exec(s, x && idx && src, mag_scatter_add(s->err_sink, &r_, x, dim, idx, src));
}

bool mag__fill_(mag_imm_t *s, mag_tensor_t *x, mag_scalar_t v) {
  return mag_imm_live(s, x != NULL) && mag_imm_check(s, mag_fill_(s->err_sink, x, v));
}

bool mag__scatter_add_(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src) {
  return mag_imm_live(s, x && idx && src) && mag_imm_check(s, mag_scatter_add_(s->err_sink, x, dim, idx, src));
}

bool mag__index_add_(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src, double alpha) {
  return mag_imm_live(s, x && idx && src) && mag_imm_check(s, mag_index_add_(s->err_sink, x, dim, idx, src, alpha));
}

#undef mag_imm_def_reduce
#undef mag_imm_def_binary
#undef mag_imm_def_unary
#undef mag_imm_exec
