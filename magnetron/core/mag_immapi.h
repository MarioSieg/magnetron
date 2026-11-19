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

#ifndef MAG_IMMAPI_H
#define MAG_IMMAPI_H

#include "mag_tensor.h"

/*
** Ok so wtf is this second API 🫩?
** We compose many calculations within Magnetron's C core out of tensor ops (for example DSP stuff, or gradients).
** Tensors are ref counted and status/err must be handled which makes it very ugly and error prone to handle in C as there is no RAII.
** This immediate API simplified lifetime handling and stuff so we we it#s clean's clean and easy.
** Just make a mag_imm_t context and then use mag__* ops like a normal human.
*/

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_IMM_INLINE_CAP 64
typedef struct mag_imm_t {
  mag_error_t *err_sink;
  mag_status_t sticky_stat;
  mag_tensor_t **pool;
  size_t len;
  size_t cap;
  mag_tensor_t *inline_pool[MAG_IMM_INLINE_CAP];
} mag_imm_t;

extern void mag_imm_begin(mag_imm_t *s, mag_error_t *err);
extern mag_status_t mag_imm_end(mag_imm_t *s);
static inline bool mag_imm_ok(const mag_imm_t *s) { return !mag_iserr(s->sticky_stat); }
extern mag_tensor_t *mag_imm_track(mag_imm_t *s, mag_status_t stat, mag_tensor_t **pt);
extern bool mag_imm_check(mag_imm_t *s, mag_status_t stat);
extern mag_tensor_t *mag_imm_adopt(mag_imm_t *s, mag_tensor_t *t);
extern mag_tensor_t *mag_imm_retain(mag_imm_t *s, mag_tensor_t *t);
static inline size_t mag_imm_mark(const mag_imm_t *s) { return s->len; }
extern void mag_imm_collect(mag_imm_t *s, size_t mark, mag_tensor_t *const *keep, size_t num_keep);
extern bool mag_imm_yield(mag_imm_t *s, mag_tensor_t **out, mag_tensor_t *t);
extern mag_status_t mag_imm_finish(mag_imm_t *s, mag_tensor_t **out, mag_tensor_t *t);

extern mag_tensor_t *mag__scalar(mag_imm_t *s, mag_tensor_t *like, double v);
extern mag_tensor_t *mag__scalar_of(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, mag_scalar_t v, mag_device_id_t device);
extern mag_tensor_t *mag__full(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_scalar_t v, mag_device_id_t device);
extern mag_tensor_t *mag__full_like(mag_imm_t *s, mag_tensor_t *like, double v);
extern mag_tensor_t *mag__zeros(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, int64_t rank, const int64_t *shape, mag_device_id_t device);
extern mag_tensor_t *mag__zeros_like(mag_imm_t *s, mag_tensor_t *like);
extern mag_tensor_t *mag__ones_like(mag_imm_t *s, mag_tensor_t *like);
extern mag_tensor_t *mag__empty_like(mag_imm_t *s, mag_tensor_t *like);
extern mag_tensor_t *mag__arange(mag_imm_t *s, mag_context_t *ctx, mag_dtype_t type, int64_t start, int64_t end, int64_t step, mag_device_id_t device);
extern mag_tensor_t *mag__clone(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__contiguous(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__cast(mag_imm_t *s, mag_tensor_t *x, mag_dtype_t type);
extern mag_tensor_t *mag__neg(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__abs(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__step(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__sqr(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__sqrt(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__rsqrt(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__rcp(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__exp(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__exp2(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__log(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__sin(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__cos(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__sinh(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__cosh(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__tanh(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__erf(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__sigmoid(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__i0(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__add(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__sub(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__mul(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__div(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__mod(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__floordiv(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__pow(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__min(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__max(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__eq(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__lt(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__le(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__gt(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__ge(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__and(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__matmul(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__outer(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__adds(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__subs(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__rsubs(mag_imm_t *s, double v, mag_tensor_t *x);
extern mag_tensor_t *mag__muls(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__divs(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__eqs(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__lts(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__gts(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__clamp_mins(mag_imm_t *s, mag_tensor_t *x, double v);
extern mag_tensor_t *mag__where(mag_imm_t *s, mag_tensor_t *cond, mag_tensor_t *x, mag_tensor_t *y);
extern mag_tensor_t *mag__clamp(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *lo, mag_tensor_t *hi);
extern mag_tensor_t *mag__sum(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern mag_tensor_t *mag__maxima(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern mag_tensor_t *mag__minima(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern mag_tensor_t *mag__any(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank, bool keepdim);
extern mag_tensor_t *mag__cusum(mag_imm_t *s, mag_tensor_t *x, int64_t dim);
extern mag_tensor_t *mag__cuprod(mag_imm_t *s, mag_tensor_t *x, int64_t dim);
extern mag_tensor_t *mag__reshape(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern mag_tensor_t *mag__transpose(mag_imm_t *s, mag_tensor_t *x, int64_t dim1, int64_t dim2);
extern mag_tensor_t *mag__T(mag_imm_t *s, mag_tensor_t *x);
extern mag_tensor_t *mag__permute(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t rank);
extern mag_tensor_t *mag__unsqueeze(mag_imm_t *s, mag_tensor_t *x, int64_t dim);
extern mag_tensor_t *mag__squeeze_dim(mag_imm_t *s, mag_tensor_t *x, int64_t dim);
extern mag_tensor_t *mag__movedim(mag_imm_t *s, mag_tensor_t *x, int64_t src, int64_t dst);
extern mag_tensor_t *mag__narrow(mag_imm_t *s, mag_tensor_t *x, int64_t dim, int64_t start, int64_t length);
extern mag_tensor_t *mag__flip(mag_imm_t *s, mag_tensor_t *x, const int64_t *dims, int64_t ndims);
extern mag_tensor_t *mag__expand(mag_imm_t *s, mag_tensor_t *x, int64_t rank, const int64_t *shape);
extern mag_tensor_t *mag__repeat(mag_imm_t *s, mag_tensor_t *x, const int64_t *repeats, int64_t num_reps);
extern mag_tensor_t *mag__repeat_back(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *like);
extern mag_tensor_t *mag__pad(mag_imm_t *s, mag_tensor_t *x, const int64_t *pad, int64_t pad_len, mag_pad_mode_t mode, mag_scalar_t value);
extern mag_tensor_t *mag__strided_view(mag_imm_t *s, mag_tensor_t *base, int64_t rank, const int64_t *shape, const int64_t *strides, int64_t offset);
extern mag_tensor_t *mag__cat(mag_imm_t *s, mag_tensor_t **tensors, size_t count, int64_t dim);
extern bool mag__split(mag_imm_t *s, mag_tensor_t **outs, int64_t num_splits, mag_tensor_t *x, int64_t split_size, int64_t dim);
extern mag_tensor_t *mag__gather(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx);
extern mag_tensor_t *mag__scatter(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src);
extern mag_tensor_t *mag__scatter_add(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src);
extern bool mag__fill_(mag_imm_t *s, mag_tensor_t *x, mag_scalar_t v);
extern bool mag__scatter_add_(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src);
extern bool mag__index_add_(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *idx, mag_tensor_t *src, double alpha);

#ifdef __cplusplus
}
#endif

#endif
