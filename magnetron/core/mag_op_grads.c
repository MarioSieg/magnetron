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

/* Query if b ackward op reads input j. This is used to determine if the input tensor needs to be backed up */
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

static mag_status_t mag_grad_saved_output(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **out, mag_status_t (*forward)(mag_error_t *, mag_tensor_t **, mag_tensor_t *)) {
  if (mag_au_state_output_is_valid(node)) {
    mag_rc_incref(node->owner);
    *out = node->owner;
    return MAG_OK;
  }
  return (*forward)(err, out, node->in[0]);
}

mag_status_t mag_op_backward_clone(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_clone(err, grads, node->grad);
}

mag_status_t mag_op_backward_cast(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_cast(err, grads, node->grad, node->in[0]->meta.dtype);
}

static mag_status_t mag_op_backward_reduce_grad_keepdim(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **out) {
  mag_tensor_t *grad = node->grad;
  const mag_reduce_plan_t *plan = node->params ? &node->params->reduction.red_plan : NULL;
  if (!plan || plan->keepdim || !plan->rank || !plan->out_rank) {
    mag_rc_incref(grad);
    *out = grad;
    return MAG_OK;
  }
  int64_t shape[MAG_MAX_DIMS];
  int64_t i=0;
  for (int64_t dim=0; dim < plan->nd; ++dim) {
    if (i < plan->rank && plan->axes[i] == dim) {
      shape[dim] = 1;
      ++i;
    } else shape[dim] = plan->in_shape[dim];
  }
  return mag_reshape(err, out, grad, shape, plan->nd);
}

mag_status_t mag_op_backward_mean(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *scale = NULL;
  mag_tensor_t *grad = NULL;

  status = mag_op_backward_reduce_grad_keepdim(err, node, &grad);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_full_like(err, &scale, x, mag_scalar_from_float64((double)node->grad->meta.numel/(double)x->meta.numel));
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, scale, grad);

cleanup:
  if (scale) mag_rc_decref(scale);
  if (grad) mag_rc_decref(grad);
  return status;
}

mag_status_t mag_op_backward_sum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *ones = NULL;
  mag_tensor_t *grad = NULL;

  status = mag_op_backward_reduce_grad_keepdim(err, node, &grad);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_full_like(err, &ones, x, mag_scalar_from_float64(1.0));
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, ones, grad);

cleanup:
  if (ones) mag_rc_decref(ones);
  if (grad) mag_rc_decref(grad);
  return status;
}

mag_status_t mag_op_backward_abs(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_status_t stat = MAG_OK;
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *step = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *two = NULL;
  mag_tensor_t *step2 = NULL;
  mag_tensor_t *sign = NULL;
  stat = mag_step(err, &step, x);
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_scalar(err, &two, x->ctx, x->meta.dtype, mag_scalar_from_float64(2.0), mag_tensor_device_id(x));
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_mul(err, &step2, step, two);
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_sub(err, &sign, step2, one);
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_mul(err, grads, node->grad, sign);
cleanup:
  if (sign) mag_rc_decref(sign);
  if (step2) mag_rc_decref(step2);
  if (two) mag_rc_decref(two);
  if (one) mag_rc_decref(one);
  if (step) mag_rc_decref(step);
  return stat;
}

mag_status_t mag_op_backward_neg(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_status_t status = MAG_OK;
  mag_tensor_t *m1 = NULL;

  status = mag_scalar(err, &m1, node->grad->ctx, node->grad->meta.dtype, mag_scalar_from_float64(-1.0), mag_tensor_device_id(node->grad));
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, m1);
  if (mag_iserr(status))
    goto cleanup;

cleanup:
  if (m1) mag_rc_decref(m1);
  return status;
}

mag_status_t mag_op_backward_log(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  return mag_div(err, grads, node->grad, x);
}

mag_status_t mag_op_backward_sqr(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *two = NULL;
  mag_tensor_t *two_x = NULL;

  status = mag_scalar(err, &two, x->ctx, x->meta.dtype, mag_scalar_from_float64(2.0), mag_tensor_device_id(x));
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, &two_x, x, two);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, two_x);
  if (mag_iserr(status))
    goto cleanup;

cleanup:
  if (two_x) mag_rc_decref(two_x);
  if (two) mag_rc_decref(two);
  return status;
}

mag_status_t mag_op_backward_sqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, &mag_sqrt);
  if (mag_iserr(status)) return status;
  mag_tensor_t *two = NULL;
  mag_tensor_t *denom = NULL;
  status = mag_scalar(err, &two, y->ctx, y->meta.dtype, mag_scalar_from_float64(2.0), mag_tensor_device_id(y));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &denom, y, two);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, denom);
cleanup:
  mag_rc_decref(y);
  if (denom) mag_rc_decref(denom);
  if (two) mag_rc_decref(two);
  return status;
}

mag_status_t mag_op_backward_sin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *cos_x = NULL;

  status = mag_cos(err, &cos_x, x);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, cos_x);
  if (mag_iserr(status))
    goto cleanup;

cleanup:
  if (cos_x) mag_rc_decref(cos_x);
  return status;
}

mag_status_t mag_op_backward_cos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *sinx = NULL;
  mag_tensor_t *nsinx = NULL;

  status = mag_sin(err, &sinx, x);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_neg(err, &nsinx, sinx);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, nsinx);
  if (mag_iserr(status))
    goto cleanup;

cleanup:
  if (nsinx) mag_rc_decref(nsinx);
  if (sinx) mag_rc_decref(sinx);
  return status;
}

mag_status_t mag_op_backward_exp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, &mag_exp);
  if (mag_iserr(status)) return status;
  status = mag_mul(err, grads, node->grad, y);
  mag_rc_decref(y);
  return status;
}

mag_status_t mag_op_backward_softmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, &mag_softmax);
  if (mag_iserr(status)) return status;
  mag_tensor_t *tmp = NULL;
  mag_tensor_t *sum_tmp = NULL;
  mag_tensor_t *diff = NULL;
  status = mag_mul(err, &tmp, node->grad, y);
  if (mag_iserr(status)) goto cleanup;
  int64_t axis = y->meta.coords.rank - 1;
  status = mag_sum(err, &sum_tmp, tmp, &axis, 1, true);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &diff, node->grad, sum_tmp);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, y, diff);
cleanup:
  mag_rc_decref(y);
  if (diff) mag_rc_decref(diff);
  if (sum_tmp) mag_rc_decref(sum_tmp);
  if (tmp) mag_rc_decref(tmp);
  return status;
}

mag_status_t mag_op_backward_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, &mag_sigmoid);
  if (mag_iserr(status)) return status;
  mag_tensor_t *one = NULL;
  mag_tensor_t *omy = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_scalar(err, &one, y->ctx, y->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(y));
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &omy, one, y);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &dv, y, omy);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
cleanup:
  mag_rc_decref(y);
  if (dv) mag_rc_decref(dv);
  if (omy) mag_rc_decref(omy);
  if (one) mag_rc_decref(one);
  return status;
}

mag_status_t mag_op_backward_silu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_silu_dv(err, &dv, x);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_iserr(status))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

mag_status_t mag_op_backward_tanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, mag_tanh);
  if (mag_iserr(status)) return status;
  mag_tensor_t *one = NULL;
  mag_tensor_t *yy = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_scalar(err, &one, y->ctx, y->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(y));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &yy, y, y);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &dv, one, yy);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
cleanup:
  mag_rc_decref(y);
  if (dv) mag_rc_decref(dv);
  if (yy) mag_rc_decref(yy);
  if (one) mag_rc_decref(one);
  return status;
}

mag_status_t mag_op_backward_relu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, mag_relu);
  if (mag_iserr(status)) return status;
  mag_tensor_t *zero = NULL;
  mag_tensor_t *mask = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_scalar(err, &zero, y->ctx, y->meta.dtype, mag_scalar_from_float64(0.0), mag_tensor_device_id(y));
  if (mag_iserr(status)) goto cleanup;
  status = mag_gt(err, &mask, y, zero);
  if (mag_iserr(status)) goto cleanup;
  status = mag_cast(err, &dv, mask, y->meta.dtype);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
cleanup:
  mag_rc_decref(y);
  if (dv) mag_rc_decref(dv);
  if (mask) mag_rc_decref(mask);
  if (zero) mag_rc_decref(zero);
  return status;
}

mag_status_t mag_op_backward_gelu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_gelu_dv(err, &dv, x);
  if (mag_iserr(status))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_iserr(status))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

static mag_status_t mag_grad_reduce_to(mag_error_t *err, mag_tensor_t **io, mag_tensor_t *like);

mag_status_t mag_op_backward_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *g = NULL;

  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, &g, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, &g, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }

cleanup:
  if (g) mag_rc_decref(g);
  return status;
}

mag_status_t mag_op_backward_sub(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *g = NULL;

  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, &g, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_neg(err, &g, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }

cleanup:
  if (g) mag_rc_decref(g);
  return status;
}

mag_status_t mag_op_backward_cat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int64_t dim = node->params->cat.dim;
  mag_status_t status = MAG_OK;
  mag_tensor_t *slice = NULL, *g = NULL;
  int64_t offset = 0;
  for (uint32_t i=0; i < node->num_in; ++i) {
    mag_tensor_t *xi = node->in[i];
    int64_t len = xi->meta.coords.shape[dim];
    if (xi->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
      status = mag_narrow(err, &slice, node->grad, dim, offset, len);
      if (mag_iserr(status)) goto cleanup;
      status = mag_clone(err, &g, slice);
      mag_rc_decref(slice); slice = NULL;
      if (mag_iserr(status)) goto cleanup;
      grads[i] = g;
      g = NULL;
    }
    offset += len;
  }
cleanup:
  if (slice) mag_rc_decref(slice);
  if (g) mag_rc_decref(g);
  return status;
}


mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *g = NULL;

  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_mul(err, &g, node->grad, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_mul(err, &g, x, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }

cleanup:
  if (g) mag_rc_decref(g);
  return status;
}

mag_status_t mag_op_backward_div(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *g = NULL;
  mag_tensor_t *gx = NULL;
  mag_tensor_t *yy = NULL;
  mag_tensor_t *gxyy = NULL;

  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_div(err, &g, node->grad, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_mul(err, &gx, node->grad, x);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &yy, y, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_div(err, &gxyy, gx, yy);
    if (mag_iserr(status)) goto cleanup;
    status = mag_neg(err, &g, gxyy);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }

cleanup:
  if (g) mag_rc_decref(g);
  if (gxyy) mag_rc_decref(gxyy);
  if (yy) mag_rc_decref(yy);
  if (gx) mag_rc_decref(gx);
  return status;
}

static mag_status_t mag_grad_matmul_2d(mag_error_t *err, mag_tensor_t *g, mag_tensor_t *x, mag_tensor_t *y, bool need_x, bool need_y, mag_tensor_t **gx, mag_tensor_t **gy) {
  mag_status_t status = MAG_OK;
  mag_tensor_t *yT = NULL;
  mag_tensor_t *xT = NULL;
  int64_t rx = x->meta.coords.rank;
  int64_t ry = y->meta.coords.rank;
  *gx = NULL;
  *gy = NULL;
  if (need_x) {
    status = mag_transpose(err, &yT, y, ry-2, ry-1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_matmul(err, gx, g, yT);
    if (mag_iserr(status)) goto cleanup;
  }
  if (need_y) {
    status = mag_transpose(err, &xT, x, rx-2, rx-1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_matmul(err, gy, xT, g);
    if (mag_iserr(status)) goto cleanup;
  }
cleanup:
  if (xT) mag_rc_decref(xT);
  if (yT) mag_rc_decref(yT);
  return status;
}

mag_status_t mag_op_backward_matmul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xa = NULL;
  mag_tensor_t *ya = NULL;
  mag_tensor_t *ga = NULL;
  mag_tensor_t *ga2 = NULL;
  mag_tensor_t *gx = NULL;
  mag_tensor_t *gy = NULL;
  mag_tensor_t *g = NULL;
  bool need_x = x->meta.flags & MAG_TFLAG_REQUIRES_GRAD;
  bool need_y = y->meta.flags & MAG_TFLAG_REQUIRES_GRAD;
  bool x1 = x->meta.coords.rank == 1;
  bool y1 = y->meta.coords.rank == 1;
  if (x1) status = mag_unsqueeze(err, &xa, x, 0);
  else { mag_rc_incref(x); xa = x; }
  if (mag_iserr(status)) goto cleanup;
  if (y1) status = mag_unsqueeze(err, &ya, y, 1);
  else { mag_rc_incref(y); ya = y; }
  if (mag_iserr(status)) goto cleanup;
  if (x1) status = mag_unsqueeze(err, &ga, node->grad, 0);
  else { mag_rc_incref(node->grad); ga = node->grad; }
  if (mag_iserr(status)) goto cleanup;
  if (y1) status = mag_unsqueeze(err, &ga2, ga, ga->meta.coords.rank);
  else { mag_rc_incref(ga); ga2 = ga; }
  if (mag_iserr(status)) goto cleanup;
  status = mag_grad_matmul_2d(err, ga2, xa, ya, need_x, need_y, &gx, &gy);
  if (mag_iserr(status)) goto cleanup;
  if (need_x) {
    if (x1) status = mag_squeeze_dim(err, &g, gx, 0);
    else { mag_rc_incref(gx); g = gx; }
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (need_y) {
    if (y1) status = mag_squeeze_dim(err, &g, gy, gy->meta.coords.rank-1);
    else { mag_rc_incref(gy); g = gy; }
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (gy) mag_rc_decref(gy);
  if (gx) mag_rc_decref(gx);
  if (ga2) mag_rc_decref(ga2);
  if (ga) mag_rc_decref(ga);
  if (ya) mag_rc_decref(ya);
  if (xa) mag_rc_decref(xa);
  return status;
}

static mag_status_t mag_grad_reduce_to(mag_error_t *err, mag_tensor_t **io, mag_tensor_t *like) {
  if (mag_tensor_is_shape_eq(*io, like)) return MAG_OK;
  mag_tensor_t *r = NULL;
  mag_status_t status = mag_repeat_back(err, &r, *io, like);
  if (mag_iserr(status)) return status;
  mag_rc_decref(*io);
  *io = r;
  return MAG_OK;
}

static bool mag_strided_view_backward_fast(
  mag_error_t *err,
  mag_status_t *status,
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
  mag_tensor_t *gc = NULL;
  if (vcont) {
    *status = mag_contiguous(err, &gc, grad);
    if (mag_iserr(*status)) return true;
    *status = mag_reshape(err, out_grad, gc, bc->shape, bc->rank);
    mag_rc_decref(gc);
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
  *status = mag_contiguous(err, &gc, grad);
  if (mag_iserr(*status)) return true;
  mag_tensor_t *permuted = NULL;
  *status = mag_permute(err, &permuted, gc, inv, rank);
  mag_rc_decref(gc);
  if (mag_iserr(*status)) return true;
  *status = mag_contiguous(err, out_grad, permuted);
  mag_rc_decref(permuted);
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
  mag_status_t status = MAG_OK;
  mag_tensor_t *flat=NULL;
  mag_tensor_t *idx=NULL;
  mag_tensor_t *ar=NULL;
  mag_tensor_t *sc=NULL;
  mag_tensor_t *arm=NULL;
  mag_tensor_t *arb=NULL;
  mag_tensor_t *idx2=NULL;
  mag_tensor_t *idxf=NULL;
  mag_tensor_t *gc=NULL;
  mag_tensor_t *gradf=NULL;
  mag_tensor_t *scattered=NULL;
  mag_tensor_t *gx_view=NULL;
  mag_tensor_t *gx=NULL;
  int64_t storel = (int64_t)(base->storage->size/mag_type_trait(base->meta.dtype)->size);
  int64_t vnumel=1;
  for (int64_t k=0; k < rank; ++k) vnumel *= vshape[k];
  mag_tensor_t *fast = NULL;
  if (mag_strided_view_backward_fast(err, &status, base, node->grad, rank, vshape, vstride, voffset, storel, &fast)) {
    if (mag_iserr(status)) return status;
    grads[0] = fast;
    return MAG_OK;
  }
  status = mag_zeros(err, &flat, ctx, node->grad->meta.dtype, 1, &storel, dev);
  if (mag_iserr(status)) goto cleanup;
  status = mag_full(err, &idx, ctx, MAG_DTYPE_INT64, rank, vshape, mag_scalar_from_int64(voffset), dev);
  if (mag_iserr(status)) goto cleanup;
  for (int64_t k=0; k < rank; ++k) {
    if (vshape[k] <= 1 || vstride[k] == 0) continue;
    status = mag_arange(err, &ar, ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(0), mag_scalar_from_int64(vshape[k]), mag_scalar_from_int64(1), dev);
    if (mag_iserr(status)) goto cleanup;
    status = mag_scalar(err, &sc, ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(vstride[k]), dev);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &arm, ar, sc);
    if (mag_iserr(status)) goto cleanup;
    int64_t bshape[MAG_MAX_DIMS];
    for (int64_t dim=0; dim < rank; ++dim) bshape[dim] = dim == k ? vshape[k] : 1;
    status = mag_reshape(err, &arb, arm, bshape, rank);
    if (mag_iserr(status)) goto cleanup;
    status = mag_add(err, &idx2, idx, arb);
    if (mag_iserr(status)) goto cleanup;
    mag_rc_decref(idx); idx = idx2; idx2 = NULL;
    mag_rc_decref(ar); ar = NULL;
    mag_rc_decref(sc); sc = NULL;
    mag_rc_decref(arm); arm = NULL;
    mag_rc_decref(arb); arb = NULL;
  }
  status = mag_reshape(err, &idxf, idx, &vnumel, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &gc, node->grad);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &gradf, gc, &vnumel, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scatter_add(err, &scattered, flat, 0, idxf, gradf);
  if (mag_iserr(status)) goto cleanup;
  status = mag_strided_view(err, &gx_view, ctx, scattered, base->meta.coords.rank, base->meta.coords.shape, base->meta.coords.strides, base->meta.storage_offset);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &gx, gx_view);
  if (mag_iserr(status)) goto cleanup;
  grads[0] = gx;
  gx = NULL;

  cleanup:
  if (gx) mag_rc_decref(gx);
  if (gx_view) mag_rc_decref(gx_view);
  if (scattered) mag_rc_decref(scattered);
  if (gradf) mag_rc_decref(gradf);
  if (gc) mag_rc_decref(gc);
  if (idxf) mag_rc_decref(idxf);
  if (idx2) mag_rc_decref(idx2);
  if (arb) mag_rc_decref(arb);
  if (arm) mag_rc_decref(arm);
  if (sc) mag_rc_decref(sc);
  if (ar) mag_rc_decref(ar);
  if (idx) mag_rc_decref(idx);
  if (flat) mag_rc_decref(flat);
  return status;
}

mag_status_t mag_op_backward_log2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *c = NULL;
  mag_tensor_t *xc = NULL;
  status = mag_scalar(err, &c, x->ctx, x->meta.dtype, mag_scalar_from_float64(0.6931471805599453), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &xc, x, c);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, xc);
cleanup:
  if (xc) mag_rc_decref(xc);
  if (c) mag_rc_decref(c);
  return status;
}

mag_status_t mag_op_backward_log10(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *c = NULL;
  mag_tensor_t *xc = NULL;
  status = mag_scalar(err, &c, x->ctx, x->meta.dtype, mag_scalar_from_float64(2.302585092994046), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &xc, x, c);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, xc);
cleanup:
  if (xc) mag_rc_decref(xc);
  if (c) mag_rc_decref(c);
  return status;
}

mag_status_t mag_op_backward_log1p(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *one = NULL;
  mag_tensor_t *denom = NULL;
  status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_add(err, &denom, x, one);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, denom);
cleanup:
  if (denom) mag_rc_decref(denom);
  if (one) mag_rc_decref(one);
  return status;
}

mag_status_t mag_op_backward_rcp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *g = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, &g, node->grad, xx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_neg(err, grads, g);
cleanup:
  if (g) mag_rc_decref(g);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_rsqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *y = NULL;
  mag_status_t status = mag_grad_saved_output(err, node, &y, mag_rsqrt);
  if (mag_iserr(status)) return status;
  mag_tensor_t *yy = NULL;
  mag_tensor_t *yyy = NULL;
  mag_tensor_t *half = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_mul(err, &yy, y, y);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &yyy, yy, y);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &half, y->ctx, y->meta.dtype, mag_scalar_from_float64(-0.5), mag_tensor_device_id(y));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &dv, yyy, half);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, dv);
cleanup:
  mag_rc_decref(y);
  if (dv) mag_rc_decref(dv);
  if (half) mag_rc_decref(half);
  if (yyy) mag_rc_decref(yyy);
  if (yy) mag_rc_decref(yy);
  return status;
}

mag_status_t mag_op_backward_tan(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *c = NULL;
  mag_tensor_t *cc = NULL;
  status = mag_cos(err, &c, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &cc, c, c);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, cc);
cleanup:
  if (cc) mag_rc_decref(cc);
  if (c) mag_rc_decref(c);
  return status;
}

mag_status_t mag_op_backward_sinh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *ch = NULL;
  status = mag_cosh(err, &ch, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, ch);
cleanup:
  if (ch) mag_rc_decref(ch);
  return status;
}

mag_status_t mag_op_backward_cosh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *sh = NULL;
  status = mag_sinh(err, &sh, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, sh);
cleanup:
  if (sh) mag_rc_decref(sh);
  return status;
}

static mag_status_t mag_grad_sqrt_1_minus_xx(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x) {
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *d = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &d, one, xx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sqrt(err, out, d);
cleanup:
  if (d) mag_rc_decref(d);
  if (one) mag_rc_decref(one);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_asin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *s = NULL;
  status = mag_grad_sqrt_1_minus_xx(err, &s, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, s);
cleanup:
  if (s) mag_rc_decref(s);
  return status;
}

mag_status_t mag_op_backward_acos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *s = NULL;
  mag_tensor_t *d = NULL;
  status = mag_grad_sqrt_1_minus_xx(err, &s, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, &d, node->grad, s);
  if (mag_iserr(status)) goto cleanup;
  status = mag_neg(err, grads, d);
cleanup:
  if (d) mag_rc_decref(d);
  if (s) mag_rc_decref(s);
  return status;
}

mag_status_t mag_op_backward_atan(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *denom = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_add(err, &denom, one, xx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, denom);
cleanup:
  if (denom) mag_rc_decref(denom);
  if (one) mag_rc_decref(one);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_asinh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *d = NULL;
  mag_tensor_t *s = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_add(err, &d, xx, one);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sqrt(err, &s, d);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, s);
cleanup:
  if (s) mag_rc_decref(s);
  if (d) mag_rc_decref(d);
  if (one) mag_rc_decref(one);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_acosh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *d = NULL;
  mag_tensor_t *s = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &d, xx, one);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sqrt(err, &s, d);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, s);
cleanup:
  if (s) mag_rc_decref(s);
  if (d) mag_rc_decref(d);
  if (one) mag_rc_decref(one);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_atanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *denom = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &denom, one, xx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, grads, node->grad, denom);
cleanup:
  if (denom) mag_rc_decref(denom);
  if (one) mag_rc_decref(one);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_exp2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *y = NULL;
  mag_tensor_t *c = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_exp2(err, &y, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &c, x->ctx, x->meta.dtype, mag_scalar_from_float64(0.6931471805599453), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &dv, y, c);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, dv);
cleanup:
  if (dv) mag_rc_decref(dv);
  if (c) mag_rc_decref(c);
  if (y) mag_rc_decref(y);
  return status;
}

mag_status_t mag_op_backward_expm1(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *e = NULL;
  status = mag_exp(err, &e, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, e);
cleanup:
  if (e) mag_rc_decref(e);
  return status;
}

mag_status_t mag_op_backward_erf(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *nxx = NULL;
  mag_tensor_t *e = NULL;
  mag_tensor_t *c = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_neg(err, &nxx, xx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_exp(err, &e, nxx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &c, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.1283791670955126), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &dv, e, c);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, dv);
cleanup:
  if (dv) mag_rc_decref(dv);
  if (c) mag_rc_decref(c);
  if (e) mag_rc_decref(e);
  if (nxx) mag_rc_decref(nxx);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_erfc(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xx = NULL;
  mag_tensor_t *nxx = NULL;
  mag_tensor_t *e = NULL;
  mag_tensor_t *c = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_mul(err, &xx, x, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_neg(err, &nxx, xx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_exp(err, &e, nxx);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &c, x->ctx, x->meta.dtype, mag_scalar_from_float64(-1.1283791670955126), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &dv, e, c);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, dv);
cleanup:
  if (dv) mag_rc_decref(dv);
  if (c) mag_rc_decref(c);
  if (e) mag_rc_decref(e);
  if (nxx) mag_rc_decref(nxx);
  if (xx) mag_rc_decref(xx);
  return status;
}

mag_status_t mag_op_backward_hard_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *lo = NULL;
  mag_tensor_t *hi = NULL;
  mag_tensor_t *m1 = NULL;
  mag_tensor_t *m2 = NULL;
  mag_tensor_t *mask = NULL;
  mag_tensor_t *sixth = NULL;
  mag_tensor_t *gs = NULL;
  mag_tensor_t *z = NULL;
  status = mag_scalar(err, &lo, x->ctx, x->meta.dtype, mag_scalar_from_float64(-3.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &hi, x->ctx, x->meta.dtype, mag_scalar_from_float64(3.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_gt(err, &m1, x, lo);
  if (mag_iserr(status)) goto cleanup;
  status = mag_lt(err, &m2, x, hi);
  if (mag_iserr(status)) goto cleanup;
  status = mag_and(err, &mask, m1, m2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &sixth, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0/6.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &gs, node->grad, sixth);
  if (mag_iserr(status)) goto cleanup;
  status = mag_zeros_like(err, &z, gs);
  if (mag_iserr(status)) goto cleanup;
  status = mag_where(err, grads, mask, gs, z);
cleanup:
  if (z) mag_rc_decref(z);
  if (gs) mag_rc_decref(gs);
  if (sixth) mag_rc_decref(sixth);
  if (mask) mag_rc_decref(mask);
  if (m2) mag_rc_decref(m2);
  if (m1) mag_rc_decref(m1);
  if (hi) mag_rc_decref(hi);
  if (lo) mag_rc_decref(lo);
  return status;
}

mag_status_t mag_op_backward_pow(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *one = NULL;
  mag_tensor_t *ym1 = NULL;
  mag_tensor_t *xpym1 = NULL;
  mag_tensor_t *t = NULL;
  mag_tensor_t *gx = NULL;
  mag_tensor_t *xpy = NULL;
  mag_tensor_t *lnx = NULL;
  mag_tensor_t *t2 = NULL;
  mag_tensor_t *gy = NULL;
  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_scalar(err, &one, x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
    if (mag_iserr(status)) goto cleanup;
    status = mag_sub(err, &ym1, y, one);
    if (mag_iserr(status)) goto cleanup;
    status = mag_pow(err, &xpym1, x, ym1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &t, y, xpym1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &gx, node->grad, t);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &gx, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = gx;
    gx = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_pow(err, &xpy, x, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_log(err, &lnx, x);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &t2, xpy, lnx);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &gy, node->grad, t2);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &gy, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = gy;
    gy = NULL;
  }
cleanup:
  if (gy) mag_rc_decref(gy);
  if (t2) mag_rc_decref(t2);
  if (lnx) mag_rc_decref(lnx);
  if (xpy) mag_rc_decref(xpy);
  if (gx) mag_rc_decref(gx);
  if (t) mag_rc_decref(t);
  if (xpym1) mag_rc_decref(xpym1);
  if (ym1) mag_rc_decref(ym1);
  if (one) mag_rc_decref(one);
  return status;
}

static mag_status_t mag_grad_minmax_weight(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *a, mag_tensor_t *b, bool is_max) {
  mag_status_t status = MAG_OK;
  mag_tensor_t *sel = NULL;
  mag_tensor_t *selc = NULL;
  mag_tensor_t *eq = NULL;
  mag_tensor_t *eqc = NULL;
  mag_tensor_t *half = NULL;
  mag_tensor_t *eqh = NULL;
  status = is_max ? mag_gt(err, &sel, a, b) : mag_lt(err, &sel, a, b);
  if (mag_iserr(status)) goto cleanup;
  status = mag_cast(err, &selc, sel, a->meta.dtype);
  if (mag_iserr(status)) goto cleanup;
  status = mag_eq(err, &eq, a, b);
  if (mag_iserr(status)) goto cleanup;
  status = mag_cast(err, &eqc, eq, a->meta.dtype);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &half, a->ctx, a->meta.dtype, mag_scalar_from_float64(0.5), mag_tensor_device_id(a));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &eqh, eqc, half);
  if (mag_iserr(status)) goto cleanup;
  status = mag_add(err, out, selc, eqh);
cleanup:
  if (eqh) mag_rc_decref(eqh);
  if (half) mag_rc_decref(half);
  if (eqc) mag_rc_decref(eqc);
  if (eq) mag_rc_decref(eq);
  if (selc) mag_rc_decref(selc);
  if (sel) mag_rc_decref(sel);
  return status;
}

static mag_status_t mag_grad_binary_minmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool is_max) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *w = NULL;
  mag_tensor_t *g = NULL;
  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_grad_minmax_weight(err, &w, x, y, is_max);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &g, node->grad, w);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
    mag_rc_decref(w); w = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_grad_minmax_weight(err, &w, y, x, is_max);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &g, node->grad, w);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (w) mag_rc_decref(w);
  return status;
}

mag_status_t mag_op_backward_min(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_binary_minmax(err, node, grads, false);
}

mag_status_t mag_op_backward_max(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_grad_binary_minmax(err, node, grads, true);
}

mag_status_t mag_op_backward_where(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *cond = node->in[0];
  mag_tensor_t *x = node->in[1];
  mag_tensor_t *y = node->in[2];
  mag_status_t status = MAG_OK;
  mag_tensor_t *z = NULL;
  mag_tensor_t *g = NULL;
  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, cond, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
    mag_rc_decref(z); z = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, cond, z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[2] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (z) mag_rc_decref(z);
  return status;
}

mag_status_t mag_op_backward_clamp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *lo = node->in[1];
  mag_tensor_t *hi = node->in[2];
  mag_status_t status = MAG_OK;
  mag_tensor_t *a = NULL;
  mag_tensor_t *b = NULL;
  mag_tensor_t *mask = NULL;
  mag_tensor_t *z = NULL;
  mag_tensor_t *g = NULL;
  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_ge(err, &a, x, lo);
    if (mag_iserr(status)) goto cleanup;
    status = mag_le(err, &b, x, hi);
    if (mag_iserr(status)) goto cleanup;
    status = mag_and(err, &mask, a, b);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
    mag_rc_decref(a); a = NULL;
    mag_rc_decref(b); b = NULL;
    mag_rc_decref(mask); mask = NULL;
    mag_rc_decref(z); z = NULL;
  }
  if (lo->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_lt(err, &mask, x, lo);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, lo);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
    mag_rc_decref(mask); mask = NULL;
    mag_rc_decref(z); z = NULL;
  }
  if (hi->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_gt(err, &mask, x, hi);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, hi);
    if (mag_iserr(status)) goto cleanup;
    grads[2] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (z) mag_rc_decref(z);
  if (mask) mag_rc_decref(mask);
  if (b) mag_rc_decref(b);
  if (a) mag_rc_decref(a);
  return status;
}

mag_status_t mag_op_backward_tril(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int32_t diag = node->params->trilu.diag;
  return mag_tril(err, grads, node->grad, diag);
}

mag_status_t mag_op_backward_triu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int32_t diag = node->params->trilu.diag;
  return mag_triu(err, grads, node->grad, diag);
}

static mag_status_t mag_grad_from_index_map(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, mag_tensor_t *map, mag_tensor_t *grad) {
  mag_status_t status = MAG_OK;
  mag_context_t *ctx = x->ctx;
  mag_device_id_t dev = mag_tensor_device_id(x);
  int64_t n = x->meta.numel;
  int64_t m = grad->meta.numel;
  mag_tensor_t *flat = NULL;
  mag_tensor_t *mapc = NULL;
  mag_tensor_t *mapf = NULL;
  mag_tensor_t *gc = NULL;
  mag_tensor_t *gf = NULL;
  *out = NULL;
  status = mag_zeros(err, &flat, ctx, grad->meta.dtype, 1, &n, dev);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &mapc, map);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &mapf, mapc, &m, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &gc, grad);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &gf, gc, &m, 1);
  if (mag_iserr(status)) goto cleanup;
  if (m > 0) {
    status = mag_scatter_add_(err, flat, 0, mapf, gf);
    if (mag_iserr(status)) goto cleanup;
  }
  status = mag_reshape(err, out, flat, x->meta.coords.shape, x->meta.coords.rank);
cleanup:
  if (gf) mag_rc_decref(gf);
  if (gc) mag_rc_decref(gc);
  if (mapf) mag_rc_decref(mapf);
  if (mapc) mag_rc_decref(mapc);
  if (flat) mag_rc_decref(flat);
  return status;
}

static mag_status_t mag_grad_flat_index_of(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, const int64_t *shape, int64_t rank) {
  mag_status_t status = MAG_OK;
  mag_tensor_t *ar = NULL;
  *out = NULL;
  status = mag_arange(err, &ar, x->ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(0), mag_scalar_from_int64(x->meta.numel), mag_scalar_from_int64(1), mag_tensor_device_id(x));
  if (mag_iserr(status)) return status;
  status = mag_reshape(err, out, ar, shape, rank);
  mag_rc_decref(ar);
  return status;
}

mag_status_t mag_op_backward_repeat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *idx = NULL;
  mag_tensor_t *map = NULL;
  int64_t rank = node->params->repeat.rank;
  const int64_t *in_shape = node->params->repeat.in_shape;
  const int64_t *out_shape = node->params->repeat.out_shape;
  int64_t reps[MAG_MAX_DIMS];
  for (int64_t d=0; d < rank; ++d) reps[d] = in_shape[d] ? out_shape[d]/in_shape[d] : 0;
  status = mag_grad_flat_index_of(err, &idx, x, in_shape, rank);
  if (mag_iserr(status)) goto cleanup;
  status = mag_repeat(err, &map, idx, reps, rank);
  if (mag_iserr(status)) goto cleanup;
  status = mag_grad_from_index_map(err, grads, x, map, node->grad);
cleanup:
  if (map) mag_rc_decref(map);
  if (idx) mag_rc_decref(idx);
  return status;
}

mag_status_t mag_op_backward_gather(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *idx = node->in[1];
  if (!(x->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) return MAG_OK;
  int64_t dim = node->params->gather.dim;
  mag_tensor_t *gx = NULL;
  mag_status_t status = mag_zeros_like(err, &gx, x);
  if (mag_iserr(status)) return status;
  status = mag_scatter_add_(err, gx, dim, idx, node->grad);
  if (mag_iserr(status)) { mag_rc_decref(gx); return status; }
  grads[0] = gx;
  return MAG_OK;
}

mag_status_t mag_op_backward_embedding(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *w = node->in[0];
  mag_tensor_t *idx = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *gw = NULL;
  mag_tensor_t *g2 = NULL;
  mag_tensor_t *idx1 = NULL;
  if (!(w->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) return MAG_OK;
  int64_t rows = w->meta.coords.shape[0];
  int64_t dim = 1;
  for (int64_t d=1; d < w->meta.coords.rank; ++d) dim *= w->meta.coords.shape[d];
  int64_t numel = idx->meta.numel;
  status = mag_zeros(err, &gw, w->ctx, w->meta.dtype, 2, (int64_t[2]){rows, dim}, mag_tensor_device_id(w));
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &g2, node->grad, (int64_t[2]){numel, dim}, 2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &idx1, idx, &numel, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_index_add_(err, gw, 0, idx1, g2, 1.0);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &grads[0], gw, w->meta.coords.shape, w->meta.coords.rank);
cleanup:
  if (idx1) mag_rc_decref(idx1);
  if (g2) mag_rc_decref(g2);
  if (gw) mag_rc_decref(gw);
  return status;
}

mag_status_t mag_op_backward_masked_fill(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];    /* the filled operand */
  mag_tensor_t *mask = node->in[1]; /* boolean mask (non-differentiable) */
  if (!(x->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) return MAG_OK;
  /* out = where(mask, value, x)  =>  dx = where(mask, 0, grad): filled positions do not depend on x. */
  return mag_masked_fill(err, &grads[0], node->grad, mask, mag_scalar_from_float64(0.0));
}

static mag_status_t mag_conv_backward_common(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool transposed) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *w = node->in[1];
  mag_tensor_t *b = node->num_in > 2 ? node->in[2] : NULL;
  mag_status_t status = MAG_OK;
  mag_tensor_t *dy = NULL;
  mag_tensor_t *g = NULL;
  status = mag_contiguous(err, &dy, node->grad);
  if (mag_iserr(status)) goto cleanup;
  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_empty_like(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    mag_tensor_t *ins[2] = {dy, w};
    status = mag_dispatch(err, transposed ? MAG_OP_CONV : MAG_OP_CONV_T, false, ins, 2, &g, 1, node->params);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (w->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_empty_like(err, &g, w);
    if (mag_iserr(status)) goto cleanup;
    mag_tensor_t *ins[2] = {transposed ? dy : x, transposed ? x : dy};
    status = mag_dispatch(err, MAG_OP_CONV_WGRAD, false, ins, 2, &g, 1, node->params);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }
  if (b && b->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    int64_t dims[MAG_MAX_DIMS];
    int64_t nd = 0;
    dims[nd++] = 0;
    for (int64_t i=2; i < dy->meta.coords.rank; ++i) dims[nd++] = i;
    status = mag_sum(err, &g, dy, dims, nd, false);
    if (mag_iserr(status)) goto cleanup;
    grads[2] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (dy) mag_rc_decref(dy);
  return status;
}

mag_status_t mag_op_backward_conv(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_conv_backward_common(err, node, grads, false);
}

mag_status_t mag_op_backward_convT(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_conv_backward_common(err, node, grads, true);
}

mag_status_t mag_op_backward_interpolate(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *dy = NULL;
  mag_tensor_t *g = NULL;
  mag_status_t status = mag_contiguous(err, &dy, node->grad);
  if (mag_iserr(status)) goto cleanup;
  status = mag_empty_like(err, &g, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_dispatch(err, MAG_OP_INTERPOLATE_BACK, false, &dy, 1, &g, 1, node->params);
  if (mag_iserr(status)) goto cleanup;
  grads[0] = g;
  g = NULL;
cleanup:
  if (g) mag_rc_decref(g);
  if (dy) mag_rc_decref(dy);
  return status;
}

static mag_status_t mag_grad_reduce_axes(mag_au_state_t *node, int64_t *axes, int64_t *rank) {
  const mag_reduce_plan_t *plan = &node->params->reduction.red_plan;
  if (plan->rank) {
    for (int64_t i=0; i < plan->rank; ++i) axes[i] = plan->axes[i];
    *rank = plan->rank;
  } else {
    for (int64_t i=0; i < plan->nd; ++i) axes[i] = i;
    *rank = plan->nd;
  }
  return MAG_OK;
}

static mag_status_t mag_grad_reduce_minmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool is_max) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *gk = NULL;
  mag_tensor_t *y = NULL;
  mag_tensor_t *eq = NULL;
  mag_tensor_t *mask = NULL;
  mag_tensor_t *cnt = NULL;
  mag_tensor_t *w = NULL;
  int64_t axes[MAG_MAX_DIMS];
  int64_t rank = 0;
  mag_grad_reduce_axes(node, axes, &rank);
  status = mag_op_backward_reduce_grad_keepdim(err, node, &gk);
  if (mag_iserr(status)) goto cleanup;
  status = is_max ? mag_maxima(err, &y, x, axes, rank, true) : mag_minima(err, &y, x, axes, rank, true);
  if (mag_iserr(status)) goto cleanup;
  status = mag_eq(err, &eq, x, y);
  if (mag_iserr(status)) goto cleanup;
  status = mag_cast(err, &mask, eq, x->meta.dtype);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sum(err, &cnt, mask, axes, rank, true);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, &w, mask, cnt);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, gk, w);
cleanup:
  if (w) mag_rc_decref(w);
  if (cnt) mag_rc_decref(cnt);
  if (mask) mag_rc_decref(mask);
  if (eq) mag_rc_decref(eq);
  if (y) mag_rc_decref(y);
  if (gk) mag_rc_decref(gk);
  return status;
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
  mag_status_t status = MAG_OK;
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
  mag_tensor_t *gk = NULL, *gkr = NULL, *gp = NULL, *gpc = NULL, *g2 = NULL;
  mag_tensor_t *xp = NULL, *xc = NULL, *x2 = NULL;
  mag_tensor_t *fwd = NULL, *fwd_n = NULL, *ones_col = NULL, *excl_fwd = NULL;
  mag_tensor_t *xf = NULL, *rf = NULL, *rev = NULL, *rev_n = NULL, *excl_rev = NULL, *pe = NULL;
  mag_tensor_t *gx2 = NULL, *gxp = NULL, *gxi = NULL;
  status = mag_op_backward_reduce_grad_keepdim(err, node, &gk);
  if (mag_iserr(status)) goto cleanup;
  if (gk->meta.coords.rank != nd) {
    status = mag_reshape(err, &gkr, gk, ones_shape, nd);
    if (mag_iserr(status)) goto cleanup;
  } else { mag_rc_incref(gk); gkr = gk; }
  status = mag_permute(err, &gp, gkr, perm, nd);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &gpc, gp);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &g2, gpc, sk, 2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_permute(err, &xp, x, perm, nd);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &xc, xp);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &x2, xc, s2, 2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_full(err, &ones_col, ctx, x->meta.dtype, 2, sk, mag_scalar_from_float64(1.0), dev);
  if (mag_iserr(status)) goto cleanup;
  if (R <= 1) {
    status = mag_full(err, &pe, ctx, x->meta.dtype, 2, s2, mag_scalar_from_float64(1.0), dev);
    if (mag_iserr(status)) goto cleanup;
  } else {
    int64_t one = 1;
    status = mag_cuprod(err, &fwd, x2, 1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_narrow(err, &fwd_n, fwd, 1, 0, R-1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_cat(err, &excl_fwd, (mag_tensor_t *[2]){ones_col, fwd_n}, 2, 1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_flip(err, &xf, x2, &one, 1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_cuprod(err, &rf, xf, 1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_flip(err, &rev, rf, &one, 1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_narrow(err, &rev_n, rev, 1, 1, R-1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_cat(err, &excl_rev, (mag_tensor_t *[2]){rev_n, ones_col}, 2, 1);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &pe, excl_fwd, excl_rev);
    if (mag_iserr(status)) goto cleanup;
  }
  status = mag_mul(err, &gx2, g2, pe);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &gxp, gx2, pshape, nd);
  if (mag_iserr(status)) goto cleanup;
  status = mag_permute(err, &gxi, gxp, inv, nd);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, grads, gxi);
cleanup:
  if (gxi) mag_rc_decref(gxi);
  if (gxp) mag_rc_decref(gxp);
  if (gx2) mag_rc_decref(gx2);
  if (pe) mag_rc_decref(pe);
  if (excl_rev) mag_rc_decref(excl_rev);
  if (rev_n) mag_rc_decref(rev_n);
  if (rev) mag_rc_decref(rev);
  if (rf) mag_rc_decref(rf);
  if (xf) mag_rc_decref(xf);
  if (excl_fwd) mag_rc_decref(excl_fwd);
  if (ones_col) mag_rc_decref(ones_col);
  if (fwd_n) mag_rc_decref(fwd_n);
  if (fwd) mag_rc_decref(fwd);
  if (x2) mag_rc_decref(x2);
  if (xc) mag_rc_decref(xc);
  if (xp) mag_rc_decref(xp);
  if (g2) mag_rc_decref(g2);
  if (gpc) mag_rc_decref(gpc);
  if (gp) mag_rc_decref(gp);
  if (gkr) mag_rc_decref(gkr);
  if (gk) mag_rc_decref(gk);
  return status;
}

static mag_status_t mag_grad_scatter_selected(mag_error_t *err, mag_tensor_t **out, mag_tensor_t *x, int64_t dim, mag_tensor_t *indices, mag_tensor_t *grad) {
  mag_status_t status = MAG_OK;
  mag_tensor_t *gx = NULL;
  mag_tensor_t *gc = NULL;
  *out = NULL;
  status = mag_zeros_like(err, &gx, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &gc, grad);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scatter_add_(err, gx, dim, indices, gc);
  if (mag_iserr(status)) goto cleanup;
  *out = gx;
  gx = NULL;
cleanup:
  if (gc) mag_rc_decref(gc);
  if (gx) mag_rc_decref(gx);
  return status;
}

mag_status_t mag_op_backward_topk(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_status_t status = mag_topk(err, &v, &i, x, node->params->topk.k, node->params->topk.dim, node->params->topk.largest, node->params->topk.sorted);
  if (mag_iserr(status)) return status;
  status = mag_grad_scatter_selected(err, grads, x, node->params->topk.dim, i, node->grad);
  mag_rc_decref(v);
  mag_rc_decref(i);
  return status;
}

mag_status_t mag_op_backward_sort(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_status_t status = mag_sort(err, &v, &i, x, node->params->sort.dim, node->params->sort.descending, node->params->sort.stable);
  if (mag_iserr(status)) return status;
  status = mag_grad_scatter_selected(err, grads, x, node->params->sort.dim, i, node->grad);
  mag_rc_decref(v);
  mag_rc_decref(i);
  return status;
}

mag_status_t mag_op_backward_cumax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_status_t status = mag_cumax(err, &v, &i, x, node->params->cumu.dim);
  if (mag_iserr(status)) return status;
  status = mag_grad_scatter_selected(err, grads, x, node->params->cumu.dim, i, node->grad);
  mag_rc_decref(v);
  mag_rc_decref(i);
  return status;
}

mag_status_t mag_op_backward_cumin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *v = NULL, *i = NULL;
  mag_status_t status = mag_cumin(err, &v, &i, x, node->params->cumu.dim);
  if (mag_iserr(status)) return status;
  status = mag_grad_scatter_selected(err, grads, x, node->params->cumu.dim, i, node->grad);
  mag_rc_decref(v);
  mag_rc_decref(i);
  return status;
}

mag_status_t mag_op_backward_cusum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int64_t dim = node->params->cumu.dim;
  mag_status_t status = MAG_OK;
  mag_tensor_t *f = NULL;
  mag_tensor_t *c = NULL;
  mag_tensor_t *b = NULL;
  status = mag_flip(err, &f, node->grad, &dim, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_cusum(err, &c, f, dim);
  if (mag_iserr(status)) goto cleanup;
  status = mag_flip(err, &b, c, &dim, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, grads, b);
cleanup:
  if (b) mag_rc_decref(b);
  if (c) mag_rc_decref(c);
  if (f) mag_rc_decref(f);
  return status;
}

mag_status_t mag_op_backward_gelu_approx(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  enum { ONE, C, K, K3, HALF, X2, KX2, ONEP, U, INNER, TH, TH2, OMT2, K3X2, ONEP3, DINNER, XOMT2, TERM2U, TERM2, THP1, TERM1, DV, NUM };
  mag_tensor_t *t[NUM] = {NULL};
  #define CHK if (mag_iserr(status)) goto cleanup
  status = mag_scalar(err, &t[ONE], x->ctx, x->meta.dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x)); CHK;
  status = mag_scalar(err, &t[C], x->ctx, x->meta.dtype, mag_scalar_from_float64(0.7978845608028654), mag_tensor_device_id(x)); CHK;
  status = mag_scalar(err, &t[K], x->ctx, x->meta.dtype, mag_scalar_from_float64(0.044715), mag_tensor_device_id(x)); CHK;
  status = mag_scalar(err, &t[K3], x->ctx, x->meta.dtype, mag_scalar_from_float64(3.0*0.044715), mag_tensor_device_id(x)); CHK;
  status = mag_scalar(err, &t[HALF], x->ctx, x->meta.dtype, mag_scalar_from_float64(0.5), mag_tensor_device_id(x)); CHK;
  status = mag_sqr(err, &t[X2], x); CHK;
  status = mag_mul(err, &t[KX2], t[X2], t[K]); CHK;
  status = mag_add(err, &t[ONEP], t[KX2], t[ONE]); CHK;
  status = mag_mul(err, &t[U], x, t[ONEP]); CHK;
  status = mag_mul(err, &t[INNER], t[U], t[C]); CHK;
  status = mag_tanh(err, &t[TH], t[INNER]); CHK;
  status = mag_sqr(err, &t[TH2], t[TH]); CHK;
  status = mag_sub(err, &t[OMT2], t[ONE], t[TH2]); CHK;
  status = mag_mul(err, &t[K3X2], t[X2], t[K3]); CHK;
  status = mag_add(err, &t[ONEP3], t[K3X2], t[ONE]); CHK;
  status = mag_mul(err, &t[DINNER], t[ONEP3], t[C]); CHK;
  status = mag_mul(err, &t[XOMT2], x, t[OMT2]); CHK;
  status = mag_mul(err, &t[TERM2U], t[XOMT2], t[DINNER]); CHK;
  status = mag_mul(err, &t[TERM2], t[TERM2U], t[HALF]); CHK;
  status = mag_add(err, &t[THP1], t[TH], t[ONE]); CHK;
  status = mag_mul(err, &t[TERM1], t[THP1], t[HALF]); CHK;
  status = mag_add(err, &t[DV], t[TERM1], t[TERM2]); CHK;
  status = mag_mul(err, grads, node->grad, t[DV]);
  #undef CHK
cleanup:
  for (int i=0; i < NUM; ++i) if (t[i]) mag_rc_decref(t[i]);
  return status;
}

mag_status_t mag_op_backward_pad(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  int64_t rank = node->params->pad.rank;
  const int64_t *pre = node->params->pad.pre_pad;
  const int64_t *post = node->params->pad.post_pad;
  if (node->params->pad.mode == MAG_PAD_MODE_CONSTANT) {
    mag_tensor_t *cur = node->grad;
    mag_rc_incref(cur);
    for (int64_t d=0; d < rank; ++d) {
      if (!pre[d] && !post[d]) continue;
      mag_tensor_t *nxt = NULL;
      status = mag_narrow(err, &nxt, cur, d, pre[d], x->meta.coords.shape[d]);
      mag_rc_decref(cur);
      if (mag_iserr(status)) return status;
      cur = nxt;
    }
    status = mag_contiguous(err, grads, cur);
    mag_rc_decref(cur);
    return status;
  }
  mag_tensor_t *idx = NULL;
  mag_tensor_t *map = NULL;
  int64_t pad[2*MAG_MAX_DIMS];
  for (int64_t d=0; d < rank; ++d) {
    int64_t i = (rank-1-d)<<1;
    pad[i] = pre[d];
    pad[i+1] = post[d];
  }
  status = mag_grad_flat_index_of(err, &idx, x, x->meta.coords.shape, rank);
  if (mag_iserr(status)) goto cleanup;
  status = mag_pad(err, &map, idx, pad, 2*rank, node->params->pad.mode, mag_scalar_from_int64(0));
  if (mag_iserr(status)) goto cleanup;
  status = mag_grad_from_index_map(err, grads, x, map, node->grad);
cleanup:
  if (map) mag_rc_decref(map);
  if (idx) mag_rc_decref(idx);
  return status;
}

mag_status_t mag_op_backward_repeat_interleave(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  if (mag_unlikely(node->num_in < 2))
    return mag_set_error(err, MAG_ERR_AUTOGRAD, "autograd: repeat_interleave node is missing its recorded repeat counts.");
  mag_tensor_t *counts = node->in[1];
  mag_context_t *ctx = x->ctx;
  mag_device_id_t dev = mag_tensor_device_id(x);
  mag_status_t status = MAG_OK;
  bool flatten = node->params->repeat_interleave.flatten;
  int64_t dim = flatten ? 0 : node->params->repeat_interleave.dim;
  int64_t in_len = flatten ? x->meta.numel : x->meta.coords.shape[dim];
  int64_t out_len = node->params->repeat_interleave.out_shape[dim];
  int64_t shape_o1[2] = {out_len, 1};
  int64_t shape_1i[2] = {1, in_len};
  int64_t axis1 = 1;
  mag_tensor_t *cnt = NULL, *cs = NULL, *excl = NULL, *ar = NULL, *ar2 = NULL, *excl2 = NULL;
  mag_tensor_t *ge = NULL, *gi = NULL, *s = NULL, *one = NULL, *src = NULL;
  mag_tensor_t *gc = NULL, *gf = NULL, *gx = NULL, *gxf = NULL;
  if (counts->meta.numel == in_len) { mag_rc_incref(counts); cnt = counts; }
  else {
    status = mag_expand(err, &cnt, counts, 1, &in_len);
    if (mag_iserr(status)) goto cleanup;
  }
  status = mag_cusum(err, &cs, cnt, 0);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &excl, cs, cnt);
  if (mag_iserr(status)) goto cleanup;
  status = mag_arange(err, &ar, ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(0), mag_scalar_from_int64(out_len), mag_scalar_from_int64(1), dev);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &ar2, ar, shape_o1, 2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &excl2, excl, shape_1i, 2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_ge(err, &ge, ar2, excl2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_cast(err, &gi, ge, MAG_DTYPE_INT64);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sum(err, &s, gi, &axis1, 1, false);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &one, ctx, MAG_DTYPE_INT64, mag_scalar_from_int64(1), dev);
  if (mag_iserr(status)) goto cleanup;
  status = mag_sub(err, &src, s, one);
  if (mag_iserr(status)) goto cleanup;
  status = mag_contiguous(err, &gc, node->grad);
  if (mag_iserr(status)) goto cleanup;
  if (flatten) {
    int64_t n = x->meta.numel;
    status = mag_zeros(err, &gxf, ctx, x->meta.dtype, 1, &n, dev);
    if (mag_iserr(status)) goto cleanup;
    status = mag_reshape(err, &gf, gc, &out_len, 1);
    if (mag_iserr(status)) goto cleanup;
    if (out_len > 0) {
      status = mag_index_add_(err, gxf, 0, src, gf, 1.0);
      if (mag_iserr(status)) goto cleanup;
    }
    status = mag_reshape(err, grads, gxf, x->meta.coords.shape, x->meta.coords.rank);
  } else {
    status = mag_zeros_like(err, &gx, x);
    if (mag_iserr(status)) goto cleanup;
    if (out_len > 0) {
      status = mag_index_add_(err, gx, dim, src, gc, 1.0);
      if (mag_iserr(status)) goto cleanup;
    }
    grads[0] = gx;
    gx = NULL;
  }
cleanup:
  if (gxf) mag_rc_decref(gxf);
  if (gx) mag_rc_decref(gx);
  if (gf) mag_rc_decref(gf);
  if (gc) mag_rc_decref(gc);
  if (src) mag_rc_decref(src);
  if (one) mag_rc_decref(one);
  if (s) mag_rc_decref(s);
  if (gi) mag_rc_decref(gi);
  if (ge) mag_rc_decref(ge);
  if (excl2) mag_rc_decref(excl2);
  if (ar2) mag_rc_decref(ar2);
  if (ar) mag_rc_decref(ar);
  if (excl) mag_rc_decref(excl);
  if (cs) mag_rc_decref(cs);
  if (cnt) mag_rc_decref(cnt);
  return status;
}

static mag_status_t mag_grad_scatter_common(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads, bool overwrite) {
  mag_tensor_t *self = node->in[0];
  mag_tensor_t *src = node->in[1];
  mag_tensor_t *idx = node->in[2];
  int64_t dim = node->params->scatter.dim;
  mag_status_t status = MAG_OK;
  mag_tensor_t *z = NULL;
  if (self->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    if (overwrite) {
      status = mag_zeros_like(err, &z, src);
      if (mag_iserr(status)) goto cleanup;
      status = mag_scatter(err, &grads[0], node->grad, dim, idx, z);
    } else {
      status = mag_clone(err, &grads[0], node->grad);
    }
    if (mag_iserr(status)) goto cleanup;
  }
  if (src->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_gather(err, &grads[1], node->grad, dim, idx);
    if (mag_iserr(status)) goto cleanup;
  }
cleanup:
  if (z) mag_rc_decref(z);
  return status;
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
    if (!(node->in[j]->meta.flags & MAG_TFLAG_REQUIRES_GRAD)) continue;
    mag_status_t status = mag_zeros_like(err, &grads[j], node->in[j]);
    if (mag_iserr(status)) return status;
  }
  return MAG_OK;
}

mag_status_t mag_op_backward_mod(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *g = NULL;
  mag_tensor_t *q = NULL;
  mag_tensor_t *gq = NULL;
  if (x->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, &g, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
  }
  if (y->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_floordiv(err, &q, x, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_mul(err, &gq, node->grad, q);
    if (mag_iserr(status)) goto cleanup;
    status = mag_neg(err, &g, gq);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }
cleanup:
  if (gq) mag_rc_decref(gq);
  if (q) mag_rc_decref(q);
  if (g) mag_rc_decref(g);
  return status;
}

mag_status_t mag_op_backward_index_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *self = node->in[0];
  mag_tensor_t *src = node->in[1];
  mag_tensor_t *idx = node->in[2];
  int64_t dim = node->params->index_add.dim;
  double alpha = node->params->index_add.alpha;
  mag_status_t status = MAG_OK;
  mag_tensor_t *idxr = NULL, *idxe = NULL, *g = NULL, *a = NULL;
  if (self->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, &grads[0], node->grad);
    if (mag_iserr(status)) goto cleanup;
  }
  if (src->meta.flags & MAG_TFLAG_REQUIRES_GRAD) {
    int64_t rank = src->meta.coords.rank;
    int64_t shape1[MAG_MAX_DIMS];
    for (int64_t d=0; d < rank; ++d) shape1[d] = d == dim ? idx->meta.numel : 1;
    status = mag_reshape(err, &idxr, idx, shape1, rank);
    if (mag_iserr(status)) goto cleanup;
    status = mag_expand(err, &idxe, idxr, rank, src->meta.coords.shape);
    if (mag_iserr(status)) goto cleanup;
    status = mag_gather(err, &g, node->grad, dim, idxe);
    if (mag_iserr(status)) goto cleanup;
    if (alpha == 1.0) { grads[1] = g; g = NULL; }
    else {
      status = mag_scalar(err, &a, src->ctx, src->meta.dtype, mag_scalar_from_float64(alpha), mag_tensor_device_id(src));
      if (mag_iserr(status)) goto cleanup;
      status = mag_mul(err, &grads[1], g, a);
      if (mag_iserr(status)) goto cleanup;
    }
  }
cleanup:
  if (a) mag_rc_decref(a);
  if (g) mag_rc_decref(g);
  if (idxe) mag_rc_decref(idxe);
  if (idxr) mag_rc_decref(idxr);
  return status;
}

mag_status_t mag_op_backward_cuprod(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_context_t *ctx = x->ctx;
  mag_device_id_t dev = mag_tensor_device_id(x);
  int64_t dim = node->params->cumu.dim;
  int64_t nd = x->meta.coords.rank;
  mag_status_t status = MAG_OK;
  mag_tensor_t *zero = NULL, *eq = NULL, *any = NULL;
  status = mag_scalar(err, &zero, ctx, x->meta.dtype, mag_scalar_from_float64(0.0), dev);
  if (mag_iserr(status)) goto cleanup0;
  status = mag_eq(err, &eq, x, zero);
  if (mag_iserr(status)) goto cleanup0;
  status = mag_any(err, &any, eq, NULL, 0, false);
  if (mag_iserr(status)) goto cleanup0;
  mag_scalar_t sc;
  status = mag_tensor_item(err, any, &sc);
  if (mag_iserr(status)) goto cleanup0;
  bool has_zero = mag_scalar_as_int64(sc) != 0;
cleanup0:
  if (any) mag_rc_decref(any);
  if (eq) mag_rc_decref(eq);
  if (zero) mag_rc_decref(zero);
  if (mag_iserr(status)) return status;
  if (!has_zero) {
    mag_tensor_t *y = NULL, *gy = NULL, *f = NULL, *cs = NULL, *rcs = NULL, *q = NULL;
    status = mag_cuprod(err, &y, x, dim);
    if (mag_iserr(status)) goto cleanup1;
    status = mag_mul(err, &gy, node->grad, y);
    if (mag_iserr(status)) goto cleanup1;
    status = mag_flip(err, &f, gy, &dim, 1);
    if (mag_iserr(status)) goto cleanup1;
    status = mag_cusum(err, &cs, f, dim);
    if (mag_iserr(status)) goto cleanup1;
    status = mag_flip(err, &rcs, cs, &dim, 1);
    if (mag_iserr(status)) goto cleanup1;
    status = mag_div(err, &q, rcs, x);
    if (mag_iserr(status)) goto cleanup1;
    status = mag_contiguous(err, grads, q);
  cleanup1:
    if (q) mag_rc_decref(q);
    if (rcs) mag_rc_decref(rcs);
    if (cs) mag_rc_decref(cs);
    if (f) mag_rc_decref(f);
    if (gy) mag_rc_decref(gy);
    if (y) mag_rc_decref(y);
    return status;
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
  int64_t one = 1;
  mag_tensor_t *xp = NULL, *xc = NULL, *x2 = NULL, *gp = NULL, *gc = NULL, *g2 = NULL;
  mag_tensor_t *fwd = NULL, *fwd_n = NULL, *ones_col = NULL, *excl_fwd = NULL;
  mag_tensor_t **cols = NULL;
  mag_tensor_t *gcol = NULL, *xcol = NULL, *xt = NULL, *tsum = NULL, *T = NULL, *gx2 = NULL, *gxp = NULL, *gxi = NULL;
  status = mag_permute(err, &xp, x, perm, nd);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_contiguous(err, &xc, xp);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_reshape(err, &x2, xc, s2, 2);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_permute(err, &gp, node->grad, perm, nd);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_contiguous(err, &gc, gp);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_reshape(err, &g2, gc, s2, 2);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_full(err, &ones_col, ctx, x->meta.dtype, 2, sk, mag_scalar_from_float64(1.0), dev);
  if (mag_iserr(status)) goto cleanup2;
  if (n > 1) {
    status = mag_cuprod(err, &fwd, x2, 1);
    if (mag_iserr(status)) goto cleanup2;
    status = mag_narrow(err, &fwd_n, fwd, 1, 0, n-1);
    if (mag_iserr(status)) goto cleanup2;
    status = mag_cat(err, &excl_fwd, (mag_tensor_t *[2]){ones_col, fwd_n}, 2, 1);
    if (mag_iserr(status)) goto cleanup2;
  } else {
    mag_rc_incref(ones_col);
    excl_fwd = ones_col;
  }
  cols = (*mag_alloc)(NULL, (size_t)(n > 0 ? n : 1)*sizeof(*cols), 0);
  for (int64_t i=0; i < n; ++i) cols[i] = NULL;
  for (int64_t i=n-1; i >= 0; --i) {
    status = mag_narrow(err, &gcol, g2, 1, i, 1);
    if (mag_iserr(status)) goto cleanup2;
    if (i == n-1) {
      status = mag_contiguous(err, &cols[i], gcol);
      if (mag_iserr(status)) goto cleanup2;
    } else {
      status = mag_narrow(err, &xcol, x2, 1, i+1, 1);
      if (mag_iserr(status)) goto cleanup2;
      status = mag_mul(err, &xt, xcol, cols[i+1]);
      if (mag_iserr(status)) goto cleanup2;
      status = mag_add(err, &cols[i], gcol, xt);
      if (mag_iserr(status)) goto cleanup2;
      mag_rc_decref(xt); xt = NULL;
      mag_rc_decref(xcol); xcol = NULL;
    }
    mag_rc_decref(gcol); gcol = NULL;
  }
  status = mag_cat(err, &T, cols, (size_t)n, 1);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_mul(err, &gx2, excl_fwd, T);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_reshape(err, &gxp, gx2, pshape, nd);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_permute(err, &gxi, gxp, inv, nd);
  if (mag_iserr(status)) goto cleanup2;
  status = mag_contiguous(err, grads, gxi);
  (void)one;
  (void)tsum;
cleanup2:
  if (gxi) mag_rc_decref(gxi);
  if (gxp) mag_rc_decref(gxp);
  if (gx2) mag_rc_decref(gx2);
  if (T) mag_rc_decref(T);
  if (xt) mag_rc_decref(xt);
  if (xcol) mag_rc_decref(xcol);
  if (gcol) mag_rc_decref(gcol);
  if (cols) {
    for (int64_t i=0; i < n; ++i) if (cols[i]) mag_rc_decref(cols[i]);
    (*mag_alloc)(cols, 0, 0);
  }
  if (excl_fwd) mag_rc_decref(excl_fwd);
  if (ones_col) mag_rc_decref(ones_col);
  if (fwd_n) mag_rc_decref(fwd_n);
  if (fwd) mag_rc_decref(fwd);
  if (g2) mag_rc_decref(g2);
  if (gc) mag_rc_decref(gc);
  if (gp) mag_rc_decref(gp);
  if (x2) mag_rc_decref(x2);
  if (xc) mag_rc_decref(xc);
  if (xp) mag_rc_decref(xp);
  return status;
}
