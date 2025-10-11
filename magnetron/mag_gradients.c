/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#include "mag_gradients.h"

void mag_op_backward_clone(mag_au_state_t *node, mag_tensor_t **grads) {
    *grads = mag_clone(node->grad);
}

void mag_op_backward_view(mag_au_state_t *node, mag_tensor_t **grads) {
    *grads = mag_clone(node->grad);
}

void mag_op_backward_transpose(mag_au_state_t *node, mag_tensor_t **grads) {
    int64_t ax0 = mag_op_param_unpack_i64_or_panic(node->op_params[0]);
    int64_t ax1 = mag_op_param_unpack_i64_or_panic(node->op_params[1]);
    *grads = mag_transpose(node->grad, ax0, ax1);
}

void mag_op_backward_mean(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *scale = mag_tensor_full_like(x, (mag_e8m23_t)(1.0/(mag_e11m52_t)x->numel));
    *grads = mag_mul(scale, node->grad);
    mag_tensor_decref(scale);
}

void mag_op_backward_sum(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *ones = mag_tensor_full_like(x, 1.0f);
    *grads = mag_mul(ones, node->grad);
    mag_tensor_decref(ones);
}

void mag_op_backward_abs(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *step = mag_step(x);
    mag_tensor_t *one = mag_tensor_scalar(x->ctx, x->dtype, 1.0f);
    mag_tensor_t *two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t *step2 = mag_mul(step, two);
    mag_tensor_t *sign = mag_sub(step2, one);
    grads[0] = mag_mul(node->grad, sign);
    mag_tensor_decref(two);
    mag_tensor_decref(one);
    mag_tensor_decref(step);
    mag_tensor_decref(step2);
    mag_tensor_decref(sign);
}

void mag_op_backward_neg(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *m1 = mag_tensor_scalar(node->grad->ctx, node->grad->dtype, -1.f);
    grads[0] = mag_mul(node->grad, m1);
    mag_tensor_decref(m1);
}

void mag_op_backward_log(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    grads[0] = mag_div(node->grad, x);
}

void mag_op_backward_sqr(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t *two_x = mag_mul(x, two);
    grads[0] = mag_mul(node->grad, two_x);
    mag_tensor_decref(two);
    mag_tensor_decref(two_x);
}

void mag_op_backward_sqrt(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *sqrt_x = mag_sqrt(x);
    mag_tensor_t *two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t *denom = mag_mul(sqrt_x, two);
    grads[0] = mag_div(node->grad, denom);
    mag_tensor_decref(two);
    mag_tensor_decref(sqrt_x);
    mag_tensor_decref(denom);
}

void mag_op_backward_sin(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *cos_x = mag_cos(x);
    grads[0] = mag_mul(node->grad, cos_x);
    mag_tensor_decref(cos_x);
}

void mag_op_backward_cos(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *sinx = mag_sin(x);
    mag_tensor_t *nsinx = mag_neg(sinx);
    grads[0] = mag_mul(node->grad, nsinx);
    mag_tensor_decref(sinx);
    mag_tensor_decref(nsinx);
}

void mag_op_backward_exp(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *exp_x = mag_exp(x);
    grads[0] = mag_mul(node->grad, exp_x);
    mag_tensor_decref(exp_x);
}

void mag_op_backward_softmax(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = mag_softmax(x);
    mag_tensor_t *tmp = mag_mul(node->grad, y);
    mag_tensor_t *sum_tmp = mag_sum(tmp, NULL, 0, false);
    mag_tensor_t *diff = mag_sub(node->grad, sum_tmp);
    grads[0] = mag_mul(y, diff);
    mag_tensor_decref(tmp);
    mag_tensor_decref(sum_tmp);
    mag_tensor_decref(diff);
    mag_tensor_decref(y);
}

void mag_op_backward_sigmoid(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = mag_sigmoid_dv(x);
    grads[0] = mag_mul(dv, node->grad);
    mag_tensor_decref(dv);
}

void mag_op_backward_silu(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = mag_silu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

void mag_op_backward_tanh(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = mag_tanh_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

void mag_op_backward_relu(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *mask = mag_step(x);
    grads[0] = mag_mul(node->grad, mask);
    mag_tensor_decref(mask);
}

void mag_op_backward_gelu(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = mag_gelu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

void mag_op_backward_add(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *grad = node->grad;
        if (!mag_tensor_is_shape_eq(x, y)) {
            grad = mag_repeat_back(grad, y);
        } else {
            grad = mag_clone(grad); /* Output gradients must be a new allocated tensor, so we clone. */
        }
        grads[1] = grad;
    }
}

void mag_op_backward_sub(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *mg = mag_neg(node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pmg = mg;
            mg = mag_repeat_back(pmg, y);
            mag_tensor_decref(pmg);
        }
        grads[1] = mg;
    }
}

void mag_op_backward_mul(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_mul(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *xg = mag_mul(x, node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pxg = xg;
            xg = mag_repeat_back(pxg, y);
            mag_tensor_decref(pxg);
        }
        grads[1] = xg;
    }
}

void mag_op_backward_div(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_div(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *gx = mag_mul(node->grad, x);
        mag_tensor_t *yy = mag_mul(y, y);
        mag_tensor_t *gxyy = mag_div(gx, yy);
        mag_tensor_t *mgxyy = mag_neg(gxyy);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pmgxyy = mgxyy;
            mgxyy = mag_repeat_back(pmgxyy, y);
            mag_tensor_decref(pmgxyy);
        }
        grads[1] = mgxyy;
        mag_tensor_decref(gxyy);
        mag_tensor_decref(yy);
        mag_tensor_decref(gx);
    }
}

void mag_op_backward_matmul(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *yt = mag_transpose(y, 0, 1);
        grads[0] = mag_matmul(node->grad, yt);
        mag_tensor_decref(yt);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *xt = mag_transpose(x, 0, 1);
        grads[1] = mag_matmul(xt, node->grad);
        mag_tensor_decref(xt);
    }
}

