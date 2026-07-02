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

mag_status_t mag_op_backward_clone(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_clone(err, grads, node->grad);
}

mag_status_t mag_op_backward_view(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  return mag_reshape(err, grads, node->grad, x->coords.shape, x->coords.rank);
}

mag_status_t mag_op_backward_transpose(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int64_t ax0 = node->params.transpose.original_axes[0];
  int64_t ax1 = node->params.transpose.original_axes[1];
  return mag_transpose(err, grads, node->grad, ax0, ax1);
}

mag_status_t mag_op_backward_mean(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *scale = NULL;

  status = mag_full_like(err, &scale, x, mag_scalar_from_float64(1.0 / (double)x->numel));
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, scale, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (scale) mag_rc_decref(scale);
  return status;
}

mag_status_t mag_op_backward_sum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *ones = NULL;

  status = mag_full_like(err, &ones, x, mag_scalar_from_float64(1.0));
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, ones, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (ones) mag_rc_decref(ones);
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
  stat = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_float64(2.0), mag_tensor_device_id(x));
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

  status = mag_scalar(err, &m1, node->grad->ctx, node->grad->dtype, mag_scalar_from_float64(-1.0), mag_tensor_device_id(node->grad));
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, m1);
  if (mag_unlikely(status != MAG_OK))
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

  status = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_float64(2.0), mag_tensor_device_id(x));
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, &two_x, x, two);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, two_x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (two_x) mag_rc_decref(two_x);
  if (two) mag_rc_decref(two);
  return status;
}

mag_status_t mag_op_backward_sqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *sqrt_x = NULL;
  mag_tensor_t *two = NULL;
  mag_tensor_t *denom = NULL;

  status = mag_sqrt(err, &sqrt_x, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_float64(2.0), mag_tensor_device_id(x));
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, &denom, sqrt_x, two);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_div(err, grads, node->grad, denom);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (denom) mag_rc_decref(denom);
  if (two) mag_rc_decref(two);
  if (sqrt_x) mag_rc_decref(sqrt_x);
  return status;
}

mag_status_t mag_op_backward_sin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *cos_x = NULL;

  status = mag_cos(err, &cos_x, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, cos_x);
  if (mag_unlikely(status != MAG_OK))
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
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_neg(err, &nsinx, sinx);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, nsinx);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (nsinx) mag_rc_decref(nsinx);
  if (sinx) mag_rc_decref(sinx);
  return status;
}

mag_status_t mag_op_backward_exp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *exp_x = NULL;

  status = mag_exp(err, &exp_x, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, node->grad, exp_x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (exp_x) mag_rc_decref(exp_x);
  return status;
}

mag_status_t mag_op_backward_softmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *y = NULL;
  mag_tensor_t *tmp = NULL;
  mag_tensor_t *sum_tmp = NULL;
  mag_tensor_t *diff = NULL;

  status = mag_softmax(err, &y, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, &tmp, node->grad, y);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_sum(err, &sum_tmp, tmp, NULL, 0, false);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_sub(err, &diff, node->grad, sum_tmp);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, y, diff);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (diff) mag_rc_decref(diff);
  if (sum_tmp) mag_rc_decref(sum_tmp);
  if (tmp) mag_rc_decref(tmp);
  if (y) mag_rc_decref(y);
  return status;
}

mag_status_t mag_op_backward_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_sigmoid_dv(err, &dv, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

mag_status_t mag_op_backward_silu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_silu_dv(err, &dv, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

mag_status_t mag_op_backward_tanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_tanh_dv(err, &dv, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

mag_status_t mag_op_backward_relu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_step(err, &dv, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

mag_status_t mag_op_backward_gelu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *dv = NULL;

  status = mag_gelu_dv(err, &dv, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_mul(err, grads, dv, node->grad);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;

cleanup:
  if (dv) mag_rc_decref(dv);
  return status;
}

mag_status_t mag_op_backward_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status;

  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, grads, node->grad);
    if (mag_unlikely(status != MAG_OK))
      return status;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    mag_tensor_t *grad = NULL;
    if (!mag_tensor_is_shape_eq(x, y)) {
      status = mag_repeat_back(err, &grad, node->grad, y);
      if (mag_unlikely(status != MAG_OK))
        return status;
    } else {
      status = mag_clone(err, &grad, node->grad);
      if (mag_unlikely(status != MAG_OK))
        return status;
    }
    grads[1] = grad;
  }
  return MAG_OK;
}

mag_status_t mag_op_backward_sub(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *mg = NULL;

  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_clone(err, grads, node->grad);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_neg(err, &mg, node->grad);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    if (!mag_tensor_is_shape_eq(x, y)) {
      mag_tensor_t *pmg = mg;
      mg = NULL;
      status = mag_repeat_back(err, &mg, pmg, y);
      mag_rc_decref(pmg);
      if (mag_unlikely(status != MAG_OK))
        goto cleanup;
    }
    grads[1] = mg;
    mg = NULL;
  }

cleanup:
  if (mg) mag_rc_decref(mg);
  return status;
}

mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *xg = NULL;

  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_mul(err, grads, node->grad, y);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_mul(err, &xg, x, node->grad);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    if (!mag_tensor_is_shape_eq(x, y)) {
      mag_tensor_t *pxg = xg;
      xg = NULL;
      status = mag_repeat_back(err, &xg, pxg, y);
      mag_rc_decref(pxg);
      if (mag_unlikely(status != MAG_OK))
        goto cleanup;
    }
    grads[1] = xg;
    xg = NULL;
  }

cleanup:
  if (xg) mag_rc_decref(xg);
  return status;
}

mag_status_t mag_op_backward_div(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *gx = NULL;
  mag_tensor_t *yy = NULL;
  mag_tensor_t *gxyy = NULL;
  mag_tensor_t *mgxyy = NULL;

  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_div(err, grads, node->grad, y);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_mul(err, &gx, node->grad, x);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    status = mag_mul(err, &yy, y, y);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    status = mag_div(err, &gxyy, gx, yy);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    status = mag_neg(err, &mgxyy, gxyy);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    if (!mag_tensor_is_shape_eq(x, y)) {
      mag_tensor_t *pmgxyy = mgxyy;
      mgxyy = NULL;
      status = mag_repeat_back(err, &mgxyy, pmgxyy, y);
      mag_rc_decref(pmgxyy);
      if (mag_unlikely(status != MAG_OK))
        goto cleanup;
    }
    grads[1] = mgxyy;
    mgxyy = NULL;
  }

cleanup:
  if (mgxyy) mag_rc_decref(mgxyy);
  if (gxyy) mag_rc_decref(gxyy);
  if (yy) mag_rc_decref(yy);
  if (gx) mag_rc_decref(gx);
  return status;
}

mag_status_t mag_op_backward_matmul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *yT = NULL;
  mag_tensor_t *xT = NULL;

  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_transpose(err, &yT, y, 0, 1);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    status = mag_matmul(err, grads, node->grad, yT);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    mag_rc_decref(yT);
    yT = NULL;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_transpose(err, &xT, x, 0, 1);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    status = mag_matmul(err, grads + 1, xT, node->grad);
    if (mag_unlikely(status != MAG_OK))
      goto cleanup;
    mag_rc_decref(xT);
    xT = NULL;
  }

cleanup:
  if (xT) mag_rc_decref(xT);
  if (yT) mag_rc_decref(yT);
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

mag_status_t mag_op_backward_log2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *c = NULL;
  mag_tensor_t *xc = NULL;
  status = mag_scalar(err, &c, x->ctx, x->dtype, mag_scalar_from_float64(0.6931471805599453), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &c, x->ctx, x->dtype, mag_scalar_from_float64(2.302585092994046), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  mag_tensor_t *x = node->in[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *y = NULL;
  mag_tensor_t *yx = NULL;
  mag_tensor_t *half = NULL;
  mag_tensor_t *dv = NULL;
  status = mag_rsqrt(err, &y, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_div(err, &yx, y, x);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &half, x->ctx, x->dtype, mag_scalar_from_float64(-0.5), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, &dv, yx, half);
  if (mag_iserr(status)) goto cleanup;
  status = mag_mul(err, grads, node->grad, dv);
cleanup:
  if (dv) mag_rc_decref(dv);
  if (half) mag_rc_decref(half);
  if (yx) mag_rc_decref(yx);
  if (y) mag_rc_decref(y);
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
  status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &c, x->ctx, x->dtype, mag_scalar_from_float64(0.6931471805599453), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &c, x->ctx, x->dtype, mag_scalar_from_float64(1.1283791670955126), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &c, x->ctx, x->dtype, mag_scalar_from_float64(-1.1283791670955126), mag_tensor_device_id(x));
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
  status = mag_scalar(err, &lo, x->ctx, x->dtype, mag_scalar_from_float64(-3.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &hi, x->ctx, x->dtype, mag_scalar_from_float64(3.0), mag_tensor_device_id(x));
  if (mag_iserr(status)) goto cleanup;
  status = mag_gt(err, &m1, x, lo);
  if (mag_iserr(status)) goto cleanup;
  status = mag_lt(err, &m2, x, hi);
  if (mag_iserr(status)) goto cleanup;
  status = mag_and(err, &mask, m1, m2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_scalar(err, &sixth, x->ctx, x->dtype, mag_scalar_from_float64(1.0/6.0), mag_tensor_device_id(x));
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
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_float64(1.0), mag_tensor_device_id(x));
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
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
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

mag_status_t mag_op_backward_min(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *mask = NULL;
  mag_tensor_t *z = NULL;
  mag_tensor_t *g = NULL;
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_le(err, &mask, x, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
    mag_rc_decref(mask); mask = NULL;
    mag_rc_decref(z); z = NULL;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_lt(err, &mask, y, x);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (z) mag_rc_decref(z);
  if (mask) mag_rc_decref(mask);
  return status;
}

mag_status_t mag_op_backward_max(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  mag_tensor_t *y = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *mask = NULL;
  mag_tensor_t *z = NULL;
  mag_tensor_t *g = NULL;
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_ge(err, &mask, x, y);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, x);
    if (mag_iserr(status)) goto cleanup;
    grads[0] = g;
    g = NULL;
    mag_rc_decref(mask); mask = NULL;
    mag_rc_decref(z); z = NULL;
  }
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
    status = mag_gt(err, &mask, y, x);
    if (mag_iserr(status)) goto cleanup;
    status = mag_zeros_like(err, &z, node->grad);
    if (mag_iserr(status)) goto cleanup;
    status = mag_where(err, &g, mask, node->grad, z);
    if (mag_iserr(status)) goto cleanup;
    status = mag_grad_reduce_to(err, &g, y);
    if (mag_iserr(status)) goto cleanup;
    grads[1] = g;
    g = NULL;
  }
cleanup:
  if (g) mag_rc_decref(g);
  if (z) mag_rc_decref(z);
  if (mask) mag_rc_decref(mask);
  return status;
}

mag_status_t mag_op_backward_where(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *cond = node->in[0];
  mag_tensor_t *x = node->in[1];
  mag_tensor_t *y = node->in[2];
  mag_status_t status = MAG_OK;
  mag_tensor_t *z = NULL;
  mag_tensor_t *g = NULL;
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
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
  if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
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
  if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
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
  if (lo->flags & MAG_TFLAG_REQUIRES_GRAD) {
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
  if (hi->flags & MAG_TFLAG_REQUIRES_GRAD) {
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
  int32_t diag = node->params.trilu.diag;
  return mag_tril(err, grads, node->grad, diag);
}

mag_status_t mag_op_backward_triu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int32_t diag = node->params.trilu.diag;
  return mag_triu(err, grads, node->grad, diag);
}

mag_status_t mag_op_backward_repeat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->in[0];
  return mag_repeat_back(err, grads, node->grad, x);
}

mag_status_t mag_op_backward_embedding(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *w = node->in[0];
  mag_tensor_t *idx = node->in[1];
  mag_status_t status = MAG_OK;
  mag_tensor_t *gw = NULL;
  mag_tensor_t *g2 = NULL;
  mag_tensor_t *idx1 = NULL;
  if (!(w->flags & MAG_TFLAG_REQUIRES_GRAD)) return MAG_OK;
  int64_t dim = w->coords.shape[w->coords.rank-1];
  int64_t numel = idx->numel;
  status = mag_zeros_like(err, &gw, w);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &g2, node->grad, (int64_t[2]){numel, dim}, 2);
  if (mag_iserr(status)) goto cleanup;
  status = mag_reshape(err, &idx1, idx, &numel, 1);
  if (mag_iserr(status)) goto cleanup;
  status = mag_index_add_(err, gw, 0, idx1, g2, 1.0);
  if (mag_iserr(status)) goto cleanup;
  grads[0] = gw;
  gw = NULL;
cleanup:
  if (idx1) mag_rc_decref(idx1);
  if (g2) mag_rc_decref(g2);
  if (gw) mag_rc_decref(gw);
  return status;
}
