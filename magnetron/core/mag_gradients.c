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

#include "mag_gradients.h"

mag_status_t mag_op_backward_clone(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  return mag_clone(err, grads, node->grad);
}

mag_status_t mag_op_backward_view(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->op_inputs[0];
  return mag_reshape(err, grads, node->grad, x->coords.shape, x->coords.rank);
}

mag_status_t mag_op_backward_transpose(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  int64_t ax0 = mag_op_attr_unwrap_int64(node->op_attrs[0]);
  int64_t ax1 = mag_op_attr_unwrap_int64(node->op_attrs[1]);
  return mag_transpose(err, grads, node->grad, ax0, ax1);
}

mag_status_t mag_op_backward_mean(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->op_inputs[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *scale = NULL;

  status = mag_full_like(err, &scale, x, mag_scalar_from_f64(1.0 / (double)x->numel));
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *ones = NULL;

  status = mag_full_like(err, &ones, x, mag_scalar_from_f64(1.0));
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *step = NULL;
  mag_tensor_t *one = NULL;
  mag_tensor_t *two = NULL;
  mag_tensor_t *step2 = NULL;
  mag_tensor_t *sign = NULL;
  stat = mag_step(err, &step, x);
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_f64(1.0), mag_tensor_device_id(x));
  if (mag_iserr(stat)) goto cleanup;
  stat = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_f64(2.0), mag_tensor_device_id(x));
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

  status = mag_scalar(err, &m1, node->grad->ctx, node->grad->dtype, mag_scalar_from_f64(-1.0), mag_tensor_device_id(node->grad));
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
  mag_tensor_t *x = node->op_inputs[0];
  return mag_div(err, grads, node->grad, x);
}

mag_status_t mag_op_backward_sqr(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
  mag_tensor_t *x = node->op_inputs[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *two = NULL;
  mag_tensor_t *two_x = NULL;

  status = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_f64(2.0), mag_tensor_device_id(x));
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_status_t status = MAG_OK;
  mag_tensor_t *sqrt_x = NULL;
  mag_tensor_t *two = NULL;
  mag_tensor_t *denom = NULL;

  status = mag_sqrt(err, &sqrt_x, x);
  if (mag_unlikely(status != MAG_OK))
    goto cleanup;
  status = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_f64(2.0), mag_tensor_device_id(x));
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *y = node->op_inputs[1];
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *y = node->op_inputs[1];
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *y = node->op_inputs[1];
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *y = node->op_inputs[1];
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
  mag_tensor_t *x = node->op_inputs[0];
  mag_tensor_t *y = node->op_inputs[1];
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
