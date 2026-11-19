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

#include "mag_op_grads.h"
#include "mag_op_dispatch.h"
#include "mag_alloc.h"
#include "mag_immapi.h"

/* Query if backward op reads input j. This is used to determine if the input tensor needs to be backed up */
bool mag_op_backward_reads_input(mag_opcode_t op, mag_tensor_t **in, uint32_t num_in, uint32_t j) {
  if (!(mag_op_trait(op)->flags&MAG_OP_FLAG_GRAD_READS_IN)) return false;
  bool other_grad = false;
  for (uint32_t k=0; k < num_in; ++k)
    if (k != j && in[k] && in[k]->meta.flags & MAG_TFLAG_REQUIRES_GRAD) other_grad = true;
  switch (op) {
    case MAG_OP_MUL: case MAG_OP_MATMUL: return other_grad;
    case MAG_OP_DIV: return j == 1 || other_grad;
    case MAG_OP_WHERE: return j == 0;
    case MAG_OP_MASKED_FILL: case MAG_OP_GATHER: case MAG_OP_EMBEDDING: case MAG_OP_REPEAT_INTERLEAVE: return j == 1;
    case MAG_OP_SCATTER: case MAG_OP_SCATTER_ADD: case MAG_OP_INDEX_ADD: return j == 2;
    default: return true;
  }
}

#define mag_grad_requires(t) (((t)->meta.flags&MAG_TFLAG_REQUIRES_GRAD))

static mag_tensor_t *mag_grad_saved_output(mag_imm_t *s, mag_au_state_t *node, mag_status_t (*forward)(mag_error_t *, mag_tensor_t **, mag_tensor_t *)) {
  if (!mag_imm_ok(s)) return NULL;
  if (mag_au_state_output_is_valid(node)) return mag_imm_retain(s, node->owner);
  mag_tensor_t *r = NULL;
  return mag_imm_track(s, (*forward)(s->err_sink, &r, node->in[0]), &r);
}

static mag_tensor_t *mag_grad_reduce_to(mag_imm_t *s, mag_tensor_t *g, mag_tensor_t *like) {
  if (!g || mag_tensor_is_shape_eq(g, like)) return g;
  return mag__repeat_back(s, g, like);
}

static void mag_grad_yield_reduced(mag_imm_t *s, mag_tensor_t **grads, uint32_t i, mag_tensor_t *g, mag_tensor_t *like) {
  mag_imm_yield(s, &grads[i], mag_grad_reduce_to(s, g, like));
}

static mag_tensor_t *mag_grad_keepdim(mag_imm_t *s, mag_au_state_t *node) {
  mag_tensor_t *grad = node->grad;
  const mag_reduce_plan_t *plan = node->params ? &node->params->reduction.red_plan : NULL;
  if (!plan || plan->keepdim || !plan->rank || !plan->out_rank) return grad;
  int64_t shape[MAG_MAX_DIMS];
  int64_t i=0;
  for (int64_t dim=0; dim < plan->nd; ++dim) {
    if (i < plan->rank && plan->axes[i] == dim) {
      shape[dim] = 1;
      ++i;
    } else shape[dim] = plan->in_shape[dim];
  }
  return mag__reshape(s, grad, shape, plan->nd);
}

mag_status_t mag_op_backward_clone(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_clone(err, grads, node->grad);
}

mag_status_t mag_op_backward_cast(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_cast(err, grads, node->grad, node->in[0]->meta.dtype);
}

mag_status_t mag_op_backward_mean(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *scale = mag__full_like(&s, x, (double)node->grad->meta.numel/(double)x->meta.numel);
  return mag_imm_finalize(&s, grads, mag__mul(&s, scale, mag_grad_keepdim(&s, node)));
}

mag_status_t mag_op_backward_sum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *ones = mag__full_like(&s, x, 1.0);
  return mag_imm_finalize(&s, grads, mag__mul(&s, ones, mag_grad_keepdim(&s, node)));
}

mag_status_t mag_op_backward_abs(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *sign = mag__subs(&s, mag__muls(&s, mag__step(&s, x), 2.0), 1.0);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, sign));
}

mag_status_t mag_op_backward_neg(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__muls(&s, node->grad, -1.0));
}

mag_status_t mag_op_backward_log(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_div(err, grads, node->grad, node->in[0]);
}

mag_status_t mag_op_backward_sqr(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__muls(&s, x, 2.0)));
}

mag_status_t mag_op_backward_sqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *y = mag_grad_saved_output(&s, node, &mag_sqrt);
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, mag__muls(&s, y, 2.0)));
}

mag_status_t mag_op_backward_sin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__cos(&s, node->in[0])));
}

mag_status_t mag_op_backward_cos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__neg(&s, mag__sin(&s, node->in[0]))));
}

mag_status_t mag_op_backward_exp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag_grad_saved_output(&s, node, &mag_exp)));
}

mag_status_t mag_op_backward_softmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *y = mag_grad_saved_output(&s, node, &mag_softmax);
  int64_t axis = node->in[0]->meta.coords.rank - 1;
  mag_tensor_t *dot = mag__sum(&s, mag__mul(&s, node->grad, y), &axis, 1, true);
  return mag_imm_finalize(&s, grads, mag__mul(&s, y, mag__sub(&s, node->grad, dot)));
}

mag_status_t mag_op_backward_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *y = mag_grad_saved_output(&s, node, &mag_sigmoid);
  mag_tensor_t *dv = mag__mul(&s, y, mag__rsubs(&s, 1.0, y));
  return mag_imm_finalize(&s, grads, mag__mul(&s, dv, node->grad));
}

mag_status_t mag_op_backward_silu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *sg = mag__sigmoid(&s, x);
  mag_tensor_t *p = mag__adds(&s, mag__mul(&s, x, mag__rsubs(&s, 1.0, sg)), 1.0);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__mul(&s, sg, p)));
}

mag_status_t mag_op_backward_tanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *y = mag_grad_saved_output(&s, node, &mag_tanh);
  mag_tensor_t *dv = mag__rsubs(&s, 1.0, mag__mul(&s, y, y));
  return mag_imm_finalize(&s, grads, mag__mul(&s, dv, node->grad));
}

mag_status_t mag_op_backward_relu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *y = mag_grad_saved_output(&s, node, &mag_relu);
  mag_tensor_t *dv = mag__cast(&s, mag__gts(&s, y, 0.0), node->in[0]->meta.dtype);
  return mag_imm_finalize(&s, grads, mag__mul(&s, dv, node->grad));
}

mag_status_t mag_op_backward_gelu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *cdf = mag__muls(&s, mag__adds(&s, mag__erf(&s, mag__muls(&s, x, 0.7071067811865476)), 1.0), 0.5);
  mag_tensor_t *pdf = mag__muls(&s, mag__exp(&s, mag__muls(&s, mag__sqr(&s, x), -0.5)), 0.3989422804014327);
  mag_tensor_t *dv = mag__add(&s, cdf, mag__mul(&s, x, pdf));
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, dv));
}

mag_status_t mag_op_backward_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 0, mag__clone(&s, node->grad), x);
  if (mag_grad_requires(y)) mag_grad_yield_reduced(&s, grads, 1, mag__clone(&s, node->grad), y);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_sub(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 0, mag__clone(&s, node->grad), x);
  if (mag_grad_requires(y)) mag_grad_yield_reduced(&s, grads, 1, mag__neg(&s, node->grad), y);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_cat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int64_t dim = node->params->cat.dim;
  int64_t offset = 0;
  mag_imm_t s;
  mag_imm_start(&s, err);
  for (uint32_t i=0; i < node->num_in && mag_imm_ok(&s); ++i) {
    mag_tensor_t *xi = node->in[i];
    int64_t len = xi->meta.coords.shape[dim];
    if (mag_grad_requires(xi))
      mag_imm_yield(&s, &grads[i], mag__clone(&s, mag__narrow(&s, node->grad, dim, offset, len)));
    offset += len;
  }
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 0, mag__mul(&s, node->grad, y), x);
  if (mag_grad_requires(y)) mag_grad_yield_reduced(&s, grads, 1, mag__mul(&s, x, node->grad), y);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_div(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 0, mag__div(&s, node->grad, y), x);
  if (mag_grad_requires(y)) {
    mag_tensor_t *g = mag__neg(&s, mag__div(&s, mag__mul(&s, node->grad, x), mag__mul(&s, y, y)));
    mag_grad_yield_reduced(&s, grads, 1, g, y);
  }
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_matmul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  bool need_x = mag_grad_requires(x);
  bool need_y = mag_grad_requires(y);
  bool x1 = x->meta.coords.rank == 1;
  bool y1 = y->meta.coords.rank == 1;
  int64_t rx = x1 ? 2 : x->meta.coords.rank;
  int64_t ry = y1 ? 2 : y->meta.coords.rank;
  int64_t rg = node->grad->meta.coords.rank + (x1 ? 1 : 0);
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *xa = x1 ? mag__unsqueeze(&s, x, 0) : x;
  mag_tensor_t *ya = y1 ? mag__unsqueeze(&s, y, 1) : y;
  mag_tensor_t *ga = x1 ? mag__unsqueeze(&s, node->grad, 0) : node->grad;
  if (y1) ga = mag__unsqueeze(&s, ga, rg);
  if (need_x) {
    mag_tensor_t *gx = mag__matmul(&s, ga, mag__transpose(&s, ya, ry-2, ry-1));
    if (x1) gx = mag__squeeze_dim(&s, gx, 0);
    mag_grad_yield_reduced(&s, grads, 0, gx, x);
  }
  if (need_y) {
    mag_tensor_t *gy = mag__matmul(&s, mag__transpose(&s, xa, rx-2, rx-1), ga);
    if (y1) gy = mag__squeeze_dim(&s, gy, -1);
    mag_grad_yield_reduced(&s, grads, 1, gy, y);
  }
  return mag_imm_end(&s);
}

static bool mag_strided_view_backward_fast(
  mag_imm_t *s,
  mag_tensor_t *base,
  mag_tensor_t *grad,
  int64_t rank,
  const int64_t *vshape,
  const int64_t *vstride,
  int64_t voffset,
  int64_t storel,
  mag_tensor_t **out_grad
) {
  *out_grad = NULL;
  const mag_coords_t *bc = &base->meta.coords;
  if (voffset != 0) return false;
  if (base->meta.numel != storel) return false;
  if (!mag_tensor_is_contiguous(base)) return false;
  int64_t vnumel = 1;
  for (int64_t k=0; k < rank; ++k) vnumel *= vshape[k];
  if (vnumel != storel) return false;
  bool vcont = true;
  int64_t prod=1;
  for (int64_t k=rank; k --> 0;) {
    if (vshape[k] == 1) continue;
    if (vstride[k] != prod) { vcont = false; break; }
    prod *= vshape[k];
  }
  if (vcont) {
    *out_grad = mag__reshape(s, mag__contiguous(s, grad), bc->shape, bc->rank);
    return true;
  }
  if (rank != bc->rank) return false;
  int64_t bstride[MAG_MAX_DIMS];
  int64_t acc = 1;
  for (int64_t i=rank; i --> 0;) {
    bstride[i] = acc;
    acc *= bc->shape[i];
  }
  int64_t perm[MAG_MAX_DIMS];
  bool used[MAG_MAX_DIMS] = {false};
  for (int64_t k=0; k < rank; ++k) {
    int64_t match=-1;
    for (int64_t m=0; m < rank; ++m) {
      if (used[m] || bc->shape[m] != vshape[k]) continue;
      if (vshape[k] != 1 && vstride[k] != bstride[m]) continue;
      match = m;
      break;
    }
    if (match < 0) return false;
    used[match] = true;
    perm[k] = match;
  }
  int64_t inv[MAG_MAX_DIMS];
  for (int64_t k=0; k < rank; ++k) inv[perm[k]] = k;
  *out_grad = mag__contiguous(s, mag__permute(s, mag__contiguous(s, grad), inv, rank));
  return true;
}

mag_status_t mag_op_backward_strided_view(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *base = node->in[0];
  mag_context_t *ctx = base->ctx;
  mag_device_id_t dev = mag_tensor_device_id(base);
  int64_t rank = node->params->strided.rank;
  const int64_t *vshape = node->params->strided.shape;
  const int64_t *vstride = node->params->strided.strides;
  int64_t voffset = node->params->strided.offset;
  int64_t storel = (int64_t)(base->storage->size/mag_type_trait(base->meta.dtype)->size);
  int64_t vnumel=1;
  for (int64_t k=0; k < rank; ++k) vnumel *= vshape[k];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *fast = NULL;
  if (mag_strided_view_backward_fast(&s, base, node->grad, rank, vshape, vstride, voffset, storel, &fast))
    return mag_imm_finalize(&s, grads, fast);
  mag_tensor_t *flat = mag__zeros(&s, ctx, node->grad->meta.dtype, 1, &storel, dev);
  mag_tensor_t *idx = mag__full(&s, ctx, MAG_DTYPE_INT64, rank, vshape, mag_scalar_from_int64(voffset), dev);
  size_t mark = mag_imm_mark(&s);
  for (int64_t k=0; k < rank && mag_imm_ok(&s); ++k) {
    if (vshape[k] <= 1 || vstride[k] == 0) continue;
    int64_t bshape[MAG_MAX_DIMS];
    for (int64_t dim=0; dim < rank; ++dim) bshape[dim] = dim == k ? vshape[k] : 1;
    mag_tensor_t *ar = mag__arange(&s, ctx, MAG_DTYPE_INT64, 0, vshape[k], 1, dev);
    mag_tensor_t *off = mag__mul(&s, ar, mag__scalar_of(&s, ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(vstride[k]), dev));
    idx = mag__add(&s, idx, mag__reshape(&s, off, bshape, rank));
    mag_imm_collect(&s, mark, idx, NULL);
  }
  mag_tensor_t *idxf = mag__reshape(&s, idx, &vnumel, 1);
  mag_tensor_t *gradf = mag__reshape(&s, mag__contiguous(&s, node->grad), &vnumel, 1);
  mag_tensor_t *scattered = mag__scatter_add(&s, flat, 0, idxf, gradf);
  mag_tensor_t *gx = mag__strided_view(&s, scattered, base->meta.coords.rank, base->meta.coords.shape, base->meta.coords.strides, base->meta.storage_offset);
  return mag_imm_finalize(&s, grads, mag__contiguous(&s, gx));
}

mag_status_t mag_op_backward_log2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, mag__muls(&s, node->in[0], 0.6931471805599453)));
}

mag_status_t mag_op_backward_log10(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, mag__muls(&s, node->in[0], 2.302585092994046)));
}

mag_status_t mag_op_backward_log1p(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, mag__adds(&s, node->in[0], 1.0)));
}

mag_status_t mag_op_backward_rcp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__neg(&s, mag__div(&s, node->grad, mag__mul(&s, x, x))));
}

mag_status_t mag_op_backward_rsqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *y = mag_grad_saved_output(&s, node, &mag_rsqrt);
  mag_tensor_t *dv = mag__muls(&s, mag__mul(&s, mag__mul(&s, y, y), y), -0.5);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, dv));
}

mag_status_t mag_op_backward_tan(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *c = mag__cos(&s, node->in[0]);
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, mag__mul(&s, c, c)));
}

mag_status_t mag_op_backward_sinh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__cosh(&s, node->in[0])));
}

mag_status_t mag_op_backward_cosh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__sinh(&s, node->in[0])));
}

static mag_tensor_t *mag_grad_sqrt_1_minus_xx(mag_imm_t *s, mag_tensor_t *x) {
  return mag__sqrt(s, mag__rsubs(s, 1.0, mag__mul(s, x, x)));
}

mag_status_t mag_op_backward_asin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, mag_grad_sqrt_1_minus_xx(&s, node->in[0])));
}

mag_status_t mag_op_backward_acos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__neg(&s, mag__div(&s, node->grad, mag_grad_sqrt_1_minus_xx(&s, node->in[0]))));
}

mag_status_t mag_op_backward_atan(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *denom = mag__add(&s, mag__scalar(&s, x, 1.0), mag__mul(&s, x, x));
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, denom));
}

mag_status_t mag_op_backward_asinh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *denom = mag__sqrt(&s, mag__adds(&s, mag__mul(&s, x, x), 1.0));
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, denom));
}

mag_status_t mag_op_backward_acosh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *denom = mag__sqrt(&s, mag__subs(&s, mag__mul(&s, x, x), 1.0));
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, denom));
}

mag_status_t mag_op_backward_atanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *denom = mag__rsubs(&s, 1.0, mag__mul(&s, x, x));
  return mag_imm_finalize(&s, grads, mag__div(&s, node->grad, denom));
}

mag_status_t mag_op_backward_exp2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *dv = mag__muls(&s, mag__exp2(&s, node->in[0]), 0.6931471805599453);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, dv));
}

mag_status_t mag_op_backward_expm1(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__exp(&s, node->in[0])));
}

static mag_tensor_t *mag_grad_erf_kernel(mag_imm_t *s, mag_tensor_t *x, double c) {
  return mag__muls(s, mag__exp(s, mag__neg(s, mag__mul(s, x, x))), c);
}

mag_status_t mag_op_backward_erf(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag_grad_erf_kernel(&s, node->in[0], 1.1283791670955126)));
}

mag_status_t mag_op_backward_erfc(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag_grad_erf_kernel(&s, node->in[0], -1.1283791670955126)));
}

mag_status_t mag_op_backward_hard_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *mask = mag__and(&s, mag__gts(&s, x, -3.0), mag__lts(&s, x, 3.0));
  mag_tensor_t *gs = mag__mul(&s, node->grad, mag__scalar(&s, x, 1.0/6.0));
  return mag_imm_finalize(&s, grads, mag__where(&s, mask, gs, mag__zeros_like(&s, gs)));
}

mag_status_t mag_op_backward_pow(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) {
    mag_tensor_t *xpym1 = mag__pow(&s, x, mag__sub(&s, y, mag__scalar(&s, x, 1.0)));
    mag_grad_yield_reduced(&s, grads, 0, mag__mul(&s, node->grad, mag__mul(&s, y, xpym1)), x);
  }
  if (mag_grad_requires(y)) {
    mag_tensor_t *t = mag__mul(&s, mag__pow(&s, x, y), mag__log(&s, x));
    mag_grad_yield_reduced(&s, grads, 1, mag__mul(&s, node->grad, t), y);
  }
  return mag_imm_end(&s);
}

static mag_tensor_t *mag_grad_minmax_weight(mag_imm_t *s, mag_tensor_t *a, mag_tensor_t *b, bool is_max) {
  mag_dtype_t dt = a->meta.dtype;
  mag_tensor_t *sel = mag__cast(s, is_max ? mag__gt(s, a, b) : mag__lt(s, a, b), dt);
  mag_tensor_t *tie = mag__muls(s, mag__cast(s, mag__eq(s, a, b), dt), 0.5);
  return mag__add(s, sel, tie);
}

static mag_status_t mag_grad_binary_minmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool is_max) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 0, mag__mul(&s, node->grad, mag_grad_minmax_weight(&s, x, y, is_max)), x);
  if (mag_grad_requires(y)) mag_grad_yield_reduced(&s, grads, 1, mag__mul(&s, node->grad, mag_grad_minmax_weight(&s, y, x, is_max)), y);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_min(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_binary_minmax(err, node, grads, false);
}

mag_status_t mag_op_backward_max(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_binary_minmax(err, node, grads, true);
}

static mag_tensor_t *mag_grad_masked(mag_imm_t *s, mag_tensor_t *mask, mag_tensor_t *grad) {
  return mag__where(s, mask, grad, mag__zeros_like(s, grad));
}

mag_status_t mag_op_backward_where(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *cond = node->in[0];
  mag_tensor_t *x = node->in[1];
  mag_tensor_t *y = node->in[2];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 1, mag_grad_masked(&s, cond, node->grad), x);
  if (mag_grad_requires(y)) mag_grad_yield_reduced(&s, grads, 2, mag__where(&s, cond, mag__zeros_like(&s, node->grad), node->grad), y);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_clamp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *lo = node->in[1];
  mag_tensor_t *hi = node->in[2];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) {
    mag_tensor_t *inside = mag__and(&s, mag__ge(&s, x, lo), mag__le(&s, x, hi));
    mag_grad_yield_reduced(&s, grads, 0, mag_grad_masked(&s, inside, node->grad), x);
  }
  if (mag_grad_requires(lo)) mag_grad_yield_reduced(&s, grads, 1, mag_grad_masked(&s, mag__lt(&s, x, lo), node->grad), lo);
  if (mag_grad_requires(hi)) mag_grad_yield_reduced(&s, grads, 2, mag_grad_masked(&s, mag__gt(&s, x, hi), node->grad), hi);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_tril(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int32_t diag = node->params->trilu.diag;
  return mag_tril(err, grads, node->grad, diag);
}

mag_status_t mag_op_backward_triu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int32_t diag = node->params->trilu.diag;
  return mag_triu(err, grads, node->grad, diag);
}

static mag_tensor_t *mag_grad_from_index_map(mag_imm_t *s, mag_tensor_t *x, mag_tensor_t *map, mag_tensor_t *grad) {
  int64_t n = x->meta.numel;
  int64_t m = grad->meta.numel;
  mag_tensor_t *flat = mag__zeros(s, x->ctx, grad->meta.dtype, 1, &n, mag_tensor_device_id(x));
  mag_tensor_t *mapf = mag__reshape(s, mag__contiguous(s, map), &m, 1);
  mag_tensor_t *gf = mag__reshape(s, mag__contiguous(s, grad), &m, 1);
  if (m > 0) mag__scatter_add_(s, flat, 0, mapf, gf);
  return mag__reshape(s, flat, x->meta.coords.shape, x->meta.coords.rank);
}

static mag_tensor_t *mag_grad_flat_index_of(mag_imm_t *s, mag_tensor_t *x, const int64_t *shape, int64_t rank) {
  mag_tensor_t *ar = mag__arange(s, x->ctx, MAG_DTYPE_INT64, 0, x->meta.numel, 1, mag_tensor_device_id(x));
  return mag__reshape(s, ar, shape, rank);
}

mag_status_t mag_op_backward_repeat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  int64_t rank = node->params->repeat.rank;
  const int64_t *in_shape = node->params->repeat.in_shape;
  const int64_t *out_shape = node->params->repeat.out_shape;
  int64_t reps[MAG_MAX_DIMS];
  for (int64_t d=0; d < rank; ++d) reps[d] = in_shape[d] ? out_shape[d]/in_shape[d] : 0;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *map = mag__repeat(&s, mag_grad_flat_index_of(&s, x, in_shape, rank), reps, rank);
  return mag_imm_finalize(&s, grads, mag_grad_from_index_map(&s, x, map, node->grad));
}

mag_status_t mag_op_backward_gather(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *idx = node->in[1];
  if (!mag_grad_requires(x)) return MAG_OK;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *gx = mag__zeros_like(&s, x);
  mag__scatter_add_(&s, gx, node->params->gather.dim, idx, node->grad);
  return mag_imm_finalize(&s, grads, gx);
}

mag_status_t mag_op_backward_embedding(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *w = node->in[0];
  mag_tensor_t *idx = node->in[1];
  if (!mag_grad_requires(w)) return MAG_OK;
  int64_t rows = w->meta.coords.shape[0];
  int64_t dim = 1;
  for (int64_t d=1; d < w->meta.coords.rank; ++d) dim *= w->meta.coords.shape[d];
  int64_t numel = idx->meta.numel;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *gw = mag__zeros(&s, w->ctx, w->meta.dtype, 2, (int64_t[2]){rows, dim}, mag_tensor_device_id(w));
  mag_tensor_t *g2 = mag__reshape(&s, node->grad, (int64_t[2]){numel, dim}, 2);
  mag__index_add_(&s, gw, 0, mag__reshape(&s, idx, &numel, 1), g2, 1.0);
  return mag_imm_finalize(&s, grads, mag__reshape(&s, gw, w->meta.coords.shape, w->meta.coords.rank));
}

mag_status_t mag_op_backward_masked_fill(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *mask = node->in[1];
  if (!mag_grad_requires(x)) return MAG_OK;
  return mag_masked_fill(err, &grads[0], node->grad, mask, mag_scalar_from_float64(0.0));
}

static mag_tensor_t *mag_grad_dispatch_like(mag_imm_t *s, mag_opcode_t op, mag_tensor_t *like, mag_tensor_t **ins, uint32_t num_ins, const mag_op_params_t *params) {
  mag_tensor_t *out = mag__empty_like(s, like);
  for (uint32_t i=0; i < num_ins; ++i) if (!ins[i]) return NULL;
  if (!mag_imm_ok(s) || !mag_imm_check(s, mag_dispatch(s->err_sink, op, false, ins, num_ins, &out, 1, params))) return NULL;
  return out;
}

static mag_status_t mag_conv_backward_common(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool transposed) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *w = node->in[1];
  mag_tensor_t *b = node->num_in > 2 ? node->in[2] : NULL;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *dy = mag__contiguous(&s, node->grad);
  if (mag_grad_requires(x))
    mag_imm_yield(&s, &grads[0], mag_grad_dispatch_like(&s, transposed ? MAG_OP_CONV : MAG_OP_CONV_T, x, (mag_tensor_t *[2]){dy, w}, 2, node->params));
  if (mag_grad_requires(w))
    mag_imm_yield(&s, &grads[1], mag_grad_dispatch_like(&s, MAG_OP_CONV_WGRAD, w, (mag_tensor_t *[2]){transposed ? dy : x, transposed ? x : dy}, 2, node->params));
  if (b && mag_grad_requires(b)) {
    int64_t dims[MAG_MAX_DIMS];
    int64_t nd = 0;
    dims[nd++] = 0;
    for (int64_t i=2; i < node->grad->meta.coords.rank; ++i) dims[nd++] = i;
    mag_imm_yield(&s, &grads[2], mag__sum(&s, dy, dims, nd, false));
  }
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_conv(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_conv_backward_common(err, node, grads, false);
}

mag_status_t mag_op_backward_convT(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_conv_backward_common(err, node, grads, true);
}

mag_status_t mag_op_backward_interpolate(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *dy = mag__contiguous(&s, node->grad);
  return mag_imm_finalize(&s, grads, mag_grad_dispatch_like(&s, MAG_OP_INTERPOLATE_BACK, node->in[0], &dy, 1, node->params));
}

static void mag_grad_reduce_axes(mag_au_state_t *node, int64_t *axes, int64_t *rank) {
  const mag_reduce_plan_t *plan = &node->params->reduction.red_plan;
  if (plan->rank) {
    for (int64_t i=0; i < plan->rank; ++i) axes[i] = plan->axes[i];
    *rank = plan->rank;
  } else {
    for (int64_t i=0; i < plan->nd; ++i) axes[i] = i;
    *rank = plan->nd;
  }
}

static mag_status_t mag_grad_reduce_minmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool is_max) {
  mag_tensor_t *x = node->in[0];
  int64_t axes[MAG_MAX_DIMS];
  int64_t rank = 0;
  mag_grad_reduce_axes(node, axes, &rank);
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *gk = mag_grad_keepdim(&s, node);
  mag_tensor_t *y = is_max ? mag__maxima(&s, x, axes, rank, true) : mag__minima(&s, x, axes, rank, true);
  mag_tensor_t *mask = mag__cast(&s, mag__eq(&s, x, y), x->meta.dtype);
  mag_tensor_t *w = mag__div(&s, mask, mag__sum(&s, mask, axes, rank, true));
  return mag_imm_finalize(&s, grads, mag__mul(&s, gk, w));
}

mag_status_t mag_op_backward_minima(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_reduce_minmax(err, node, grads, false);
}

mag_status_t mag_op_backward_maxima(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_reduce_minmax(err, node, grads, true);
}

mag_status_t mag_op_backward_prod(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_context_t *ctx = x->ctx;
  mag_device_id_t dev = mag_tensor_device_id(x);
  int64_t nd = x->meta.coords.rank;
  int64_t axes[MAG_MAX_DIMS];
  int64_t rank = 0;
  mag_grad_reduce_axes(node, axes, &rank);
  bool reduced[MAG_MAX_DIMS] = {false};
  for (int64_t i=0; i < rank; ++i) reduced[axes[i]] = true;
  int64_t perm[MAG_MAX_DIMS];
  int64_t np = 0;
  int64_t K = 1, R = 1;
  for (int64_t d=0; d < nd; ++d) if (!reduced[d]) { perm[np++] = d; K *= x->meta.coords.shape[d]; }
  for (int64_t d=0; d < nd; ++d) if (reduced[d]) { perm[np++] = d; R *= x->meta.coords.shape[d]; }
  int64_t inv[MAG_MAX_DIMS];
  for (int64_t i=0; i < nd; ++i) inv[perm[i]] = i;
  int64_t ones_shape[MAG_MAX_DIMS];
  for (int64_t i=0; i < nd; ++i) ones_shape[i] = 1;
  int64_t pshape[MAG_MAX_DIMS];
  for (int64_t i=0; i < nd; ++i) pshape[i] = x->meta.coords.shape[perm[i]];
  int64_t s2[2] = {K, R};
  int64_t sk[2] = {K, 1};
  int64_t one = 1;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *gk = mag_grad_keepdim(&s, node);
  if (gk && gk->meta.coords.rank != nd) gk = mag__reshape(&s, gk, ones_shape, nd);
  mag_tensor_t *g2 = mag__reshape(&s, mag__contiguous(&s, mag__permute(&s, gk, perm, nd)), sk, 2);
  mag_tensor_t *x2 = mag__reshape(&s, mag__contiguous(&s, mag__permute(&s, x, perm, nd)), s2, 2);
  mag_tensor_t *ones_col = mag__full(&s, ctx, x->meta.dtype, 2, sk, mag_scalar_from_float64(1.0), dev);
  mag_tensor_t *pe;
  if (R <= 1) {
    pe = mag__full(&s, ctx, x->meta.dtype, 2, s2, mag_scalar_from_float64(1.0), dev);
  } else {
    mag_tensor_t *excl_fwd = mag__cat(&s, (mag_tensor_t *[2]){ones_col, mag__narrow(&s, mag__cuprod(&s, x2, 1), 1, 0, R-1)}, 2, 1);
    mag_tensor_t *rev = mag__flip(&s, mag__cuprod(&s, mag__flip(&s, x2, &one, 1), 1), &one, 1);
    mag_tensor_t *excl_rev = mag__cat(&s, (mag_tensor_t *[2]){mag__narrow(&s, rev, 1, 1, R-1), ones_col}, 2, 1);
    pe = mag__mul(&s, excl_fwd, excl_rev);
  }
  mag_tensor_t *gxp = mag__reshape(&s, mag__mul(&s, g2, pe), pshape, nd);
  return mag_imm_finalize(&s, grads, mag__contiguous(&s, mag__permute(&s, gxp, inv, nd)));
}

static mag_tensor_t *mag_grad_scatter_selected(mag_imm_t *s, mag_tensor_t *x, int64_t dim, mag_tensor_t *indices, mag_tensor_t *grad) {
  mag_tensor_t *gx = mag__zeros_like(s, x);
  mag__scatter_add_(s, gx, dim, indices, mag__contiguous(s, grad));
  return gx;
}

static mag_tensor_t *mag_grad_track_selection(mag_imm_t *s, mag_status_t stat, mag_tensor_t *values, mag_tensor_t *indices) {
  if (!mag_imm_check(s, stat)) return NULL;
  mag_imm_adopt(s, values);
  return mag_imm_adopt(s, indices);
}

mag_status_t mag_op_backward_topk(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_status_t stat = mag_topk(err, &v, &i, x, node->params->topk.k, node->params->topk.dim, node->params->topk.largest, node->params->topk.sorted);
  mag_tensor_t *idx = mag_grad_track_selection(&s, stat, v, i);
  return mag_imm_finalize(&s, grads, mag_grad_scatter_selected(&s, x, node->params->topk.dim, idx, node->grad));
}

mag_status_t mag_op_backward_sort(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_status_t stat = mag_sort(err, &v, &i, x, node->params->sort.dim, node->params->sort.descending, node->params->sort.stable);
  mag_tensor_t *idx = mag_grad_track_selection(&s, stat, v, i);
  return mag_imm_finalize(&s, grads, mag_grad_scatter_selected(&s, x, node->params->sort.dim, idx, node->grad));
}

mag_status_t mag_op_backward_cumax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_status_t stat = mag_cumax(err, &v, &i, x, node->params->cumu.dim);
  mag_tensor_t *idx = mag_grad_track_selection(&s, stat, v, i);
  return mag_imm_finalize(&s, grads, mag_grad_scatter_selected(&s, x, node->params->cumu.dim, idx, node->grad));
}

mag_status_t mag_op_backward_cumin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_status_t stat = mag_cumin(err, &v, &i, x, node->params->cumu.dim);
  mag_tensor_t *idx = mag_grad_track_selection(&s, stat, v, i);
  return mag_imm_finalize(&s, grads, mag_grad_scatter_selected(&s, x, node->params->cumu.dim, idx, node->grad));
}

static mag_tensor_t *mag_grad_rev_cusum(mag_imm_t *s, mag_tensor_t *x, int64_t dim) {
  return mag__flip(s, mag__cusum(s, mag__flip(s, x, &dim, 1), dim), &dim, 1);
}

mag_status_t mag_op_backward_cusum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_imm_t s;
  mag_imm_start(&s, err);
  return mag_imm_finalize(&s, grads, mag__contiguous(&s, mag_grad_rev_cusum(&s, node->grad, node->params->cumu.dim)));
}

mag_status_t mag_op_backward_gelu_approx(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *x2 = mag__sqr(&s, x);
  mag_tensor_t *inner = mag__muls(&s, mag__mul(&s, x, mag__adds(&s, mag__muls(&s, x2, 0.044715), 1.0)), 0.7978845608028654);
  mag_tensor_t *th = mag__tanh(&s, inner);
  mag_tensor_t *omt2 = mag__rsubs(&s, 1.0, mag__sqr(&s, th));
  mag_tensor_t *dinner = mag__muls(&s, mag__adds(&s, mag__muls(&s, x2, 3.0*0.044715), 1.0), 0.7978845608028654);
  mag_tensor_t *term2 = mag__muls(&s, mag__mul(&s, mag__mul(&s, x, omt2), dinner), 0.5);
  mag_tensor_t *term1 = mag__muls(&s, mag__adds(&s, th, 1.0), 0.5);
  return mag_imm_finalize(&s, grads, mag__mul(&s, node->grad, mag__add(&s, term1, term2)));
}

mag_status_t mag_op_backward_pad(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  int64_t rank = node->params->pad.rank;
  const int64_t *pre = node->params->pad.pre_pad;
  const int64_t *post = node->params->pad.post_pad;
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (node->params->pad.mode == MAG_PAD_MODE_CONSTANT) {
    mag_tensor_t *cur = node->grad;
    for (int64_t d=0; d < rank; ++d)
      if (pre[d] || post[d]) cur = mag__narrow(&s, cur, d, pre[d], x->meta.coords.shape[d]);
    return mag_imm_finalize(&s, grads, mag__contiguous(&s, cur));
  }
  int64_t pad[2*MAG_MAX_DIMS];
  for (int64_t d=0; d < rank; ++d) {
    int64_t i = (rank-1-d)<<1;
    pad[i] = pre[d];
    pad[i+1] = post[d];
  }
  mag_tensor_t *map = mag__pad(&s, mag_grad_flat_index_of(&s, x, x->meta.coords.shape, rank), pad, 2*rank, node->params->pad.mode, mag_scalar_from_int64(0));
  return mag_imm_finalize(&s, grads, mag_grad_from_index_map(&s, x, map, node->grad));
}

mag_status_t mag_op_backward_repeat_interleave(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  if (mag_unlikely(node->num_in < 2))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: repeat_interleave node is missing its recorded repeat counts.");
  mag_tensor_t *counts = node->in[1];
  mag_context_t *ctx = x->ctx;
  mag_device_id_t dev = mag_tensor_device_id(x);
  bool flatten = node->params->repeat_interleave.flatten;
  int64_t dim = flatten ? 0 : node->params->repeat_interleave.dim;
  int64_t in_len = flatten ? x->meta.numel : x->meta.coords.shape[dim];
  int64_t out_len = node->params->repeat_interleave.out_shape[dim];
  int64_t shape_o1[2] = {out_len, 1};
  int64_t shape_1i[2] = {1, in_len};
  int64_t axis1 = 1;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *cnt = counts->meta.numel == in_len ? counts : mag__expand(&s, counts, 1, &in_len);
  mag_tensor_t *excl = mag__sub(&s, mag__cusum(&s, cnt, 0), cnt);
  mag_tensor_t *ar2 = mag__reshape(&s, mag__arange(&s, ctx, MAG_DTYPE_INT64, 0, out_len, 1, dev), shape_o1, 2);
  mag_tensor_t *ge = mag__cast(&s, mag__ge(&s, ar2, mag__reshape(&s, excl, shape_1i, 2)), MAG_DTYPE_INT64);
  mag_tensor_t *src = mag__sub(&s, mag__sum(&s, ge, &axis1, 1, false), mag__scalar_of(&s, ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(1), dev));
  mag_tensor_t *gc = mag__contiguous(&s, node->grad);
  if (flatten) {
    int64_t n = x->meta.numel;
    mag_tensor_t *gxf = mag__zeros(&s, ctx, x->meta.dtype, 1, &n, dev);
    mag_tensor_t *gf = mag__reshape(&s, gc, &out_len, 1);
    if (out_len > 0) mag__index_add_(&s, gxf, 0, src, gf, 1.0);
    return mag_imm_finalize(&s, grads, mag__reshape(&s, gxf, x->meta.coords.shape, x->meta.coords.rank));
  }
  mag_tensor_t *gx = mag__zeros_like(&s, x);
  if (out_len > 0) mag__index_add_(&s, gx, dim, src, gc, 1.0);
  return mag_imm_finalize(&s, grads, gx);
}

static mag_status_t mag_grad_scatter_common(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool overwrite) {
  mag_tensor_t *self = node->in[0];
  mag_tensor_t *src = node->in[1];
  mag_tensor_t *idx = node->in[2];
  int64_t dim = node->params->scatter.dim;
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(self)) {
    mag_tensor_t *g = overwrite ? mag__scatter(&s, node->grad, dim, idx, mag__zeros_like(&s, src)) : mag__clone(&s, node->grad);
    mag_imm_yield(&s, &grads[0], g);
  }
  if (mag_grad_requires(src)) mag_imm_yield(&s, &grads[1], mag__gather(&s, node->grad, dim, idx));
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_scatter(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_scatter_common(err, node, grads, true);
}

mag_status_t mag_op_backward_scatter_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_scatter_common(err, node, grads, false);
}

mag_status_t mag_op_backward_zero_unary(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_zeros_like(err, grads, node->in[0]);
}

mag_status_t mag_op_backward_zero_binary(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  for (uint32_t j=0; j < 2; ++j) {
    if (!mag_grad_requires(node->in[j])) continue;
    mag_status_t stat = mag_zeros_like(err, &grads[j], node->in[j]);
    if (mag_iserr(stat)) return stat;
  }
  return MAG_OK;
}

mag_status_t mag_op_backward_mod(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(x)) mag_grad_yield_reduced(&s, grads, 0, mag__clone(&s, node->grad), x);
  if (mag_grad_requires(y)) mag_grad_yield_reduced(&s, grads, 1, mag__neg(&s, mag__mul(&s, node->grad, mag__floordiv(&s, x, y))), y);
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_index_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *self = node->in[0];
  mag_tensor_t *src = node->in[1];
  mag_tensor_t *idx = node->in[2];
  int64_t dim = node->params->index_add.dim;
  double alpha = node->params->index_add.alpha;
  mag_imm_t s;
  mag_imm_start(&s, err);
  if (mag_grad_requires(self)) mag_imm_yield(&s, &grads[0], mag__clone(&s, node->grad));
  if (mag_grad_requires(src)) {
    int64_t rank = src->meta.coords.rank;
    int64_t shape1[MAG_MAX_DIMS];
    for (int64_t d=0; d < rank; ++d) shape1[d] = d == dim ? idx->meta.numel : 1;
    mag_tensor_t *idxe = mag__expand(&s, mag__reshape(&s, idx, shape1, rank), rank, src->meta.coords.shape);
    mag_tensor_t *g = mag__gather(&s, node->grad, dim, idxe);
    if (alpha != 1.0) g = mag__mul(&s, g, mag__scalar(&s, src, alpha));
    mag_imm_yield(&s, &grads[1], g);
  }
  return mag_imm_end(&s);
}

mag_status_t mag_op_backward_cuprod(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_context_t *ctx = x->ctx;
  mag_device_id_t dev = mag_tensor_device_id(x);
  int64_t dim = node->params->cumu.dim;
  int64_t nd = x->meta.coords.rank;
  mag_imm_t s;
  mag_imm_start(&s, err);
  mag_tensor_t *any = mag__any(&s, mag__eqs(&s, x, 0.0), NULL, 0, false);
  mag_scalar_t sc = mag_scalar_from_int64(0);
  if (mag_imm_ok(&s)) mag_imm_check(&s, mag_tensor_item(s.err_sink, any, &sc));
  if (!mag_imm_ok(&s)) return mag_imm_end(&s);
  if (!mag_scalar_as_int64(sc)) {
    mag_tensor_t *gy = mag__mul(&s, node->grad, mag__cuprod(&s, x, dim));
    return mag_imm_finalize(&s, grads, mag__contiguous(&s, mag__div(&s, mag_grad_rev_cusum(&s, gy, dim), x)));
  }
  int64_t perm[MAG_MAX_DIMS], inv[MAG_MAX_DIMS], pshape[MAG_MAX_DIMS];
  int64_t np = 0;
  for (int64_t d=0; d < nd; ++d) if (d != dim) perm[np++] = d;
  perm[np++] = dim;
  for (int64_t i=0; i < nd; ++i) inv[perm[i]] = i;
  for (int64_t i=0; i < nd; ++i) pshape[i] = x->meta.coords.shape[perm[i]];
  int64_t n = x->meta.coords.shape[dim];
  int64_t K = n ? x->meta.numel/n : 0;
  int64_t s2[2] = {K, n};
  int64_t sk[2] = {K, 1};
  mag_tensor_t *x2 = mag__reshape(&s, mag__contiguous(&s, mag__permute(&s, x, perm, nd)), s2, 2);
  mag_tensor_t *g2 = mag__reshape(&s, mag__contiguous(&s, mag__permute(&s, node->grad, perm, nd)), s2, 2);
  mag_tensor_t *ones_col = mag__full(&s, ctx, x->meta.dtype, 2, sk, mag_scalar_from_float64(1.0), dev);
  mag_tensor_t *excl_fwd = n > 1 ? mag__cat(&s, (mag_tensor_t *[2]){ones_col, mag__narrow(&s, mag__cuprod(&s, x2, 1), 1, 0, n-1)}, 2, 1) : ones_col;
  mag_tensor_t **cols = (*mag_alloc)(NULL, (size_t)(n > 0 ? n : 1)*sizeof(*cols), 0);
  for (int64_t i=0; i < n; ++i) cols[i] = NULL;
  for (int64_t i=n-1; i >= 0 && mag_imm_ok(&s); --i) {
    size_t mark = mag_imm_mark(&s);
    mag_tensor_t *gcol = mag__narrow(&s, g2, 1, i, 1);
    cols[i] = i == n-1 ? mag__contiguous(&s, gcol) : mag__add(&s, gcol, mag__mul(&s, mag__narrow(&s, x2, 1, i+1, 1), cols[i+1]));
    mag_imm_collect(&s, mark, cols[i], NULL);
  }
  mag_tensor_t *T = mag__cat(&s, cols, (size_t)n, 1);
  (*mag_alloc)(cols, 0, 0);
  mag_tensor_t *gxp = mag__reshape(&s, mag__mul(&s, excl_fwd, T), pshape, nd);
  return mag_imm_finalize(&s, grads, mag__contiguous(&s, mag__permute(&s, gxp, inv, nd)));
}
