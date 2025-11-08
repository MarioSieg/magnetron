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

mag_status_t mag_op_backward_clone(mag_au_state_t *node, mag_tensor_t **grads) {
    return mag_clone(grads, node->grad);
}

mag_status_t mag_op_backward_view(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    return mag_reshape(grads, node->grad, x->shape, x->rank);
}

mag_status_t mag_op_backward_transpose(mag_au_state_t *node, mag_tensor_t **grads) {
    int64_t ax0 = mag_op_param_unpack_i64_or_panic(node->op_params[0]);
    int64_t ax1 = mag_op_param_unpack_i64_or_panic(node->op_params[1]);
    return mag_transpose(grads, node->grad, ax0, ax1);
}

mag_status_t mag_op_backward_mean(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_status_t stat;
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *scale;
    stat = mag_tensor_full_like(&scale, x, (mag_e8m23_t)(1.0/(mag_e11m52_t)x->numel));
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, scale, node->grad);
    mag_tensor_decref(scale);
    return stat;
}

mag_status_t mag_op_backward_sum(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_status_t stat;
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *ones;
    stat = mag_tensor_full_like(&ones, x, 1.0f);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, ones, node->grad);
    mag_tensor_decref(ones);
    return stat;
}

mag_status_t mag_op_backward_abs(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_status_t stat;
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *step = NULL;
    mag_tensor_t *one = NULL;
    mag_tensor_t *two = NULL;
    mag_tensor_t *step2 = NULL;
    mag_tensor_t *sign = NULL;
    stat = mag_step(&step, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_tensor_scalar(&one, x->ctx, x->dtype, 1.0f);
    if (mag_iserr(stat)) goto error;
    stat = mag_tensor_scalar(&two, x->ctx, x->dtype, 2.0f);
    if (mag_iserr(stat)) goto error;
    stat = mag_mul(&step2, step, two);
    if (mag_iserr(stat)) goto error;
    stat = mag_sub(&sign, step2, one);
    if (mag_iserr(stat)) goto error;
    stat = mag_mul(grads, node->grad, sign);
error:
    if (two) mag_tensor_decref(two);
    if (one) mag_tensor_decref(one);
    if (step) mag_tensor_decref(step);
    if (step2) mag_tensor_decref(step2);
    if (sign) mag_tensor_decref(sign);
    return stat;
}

mag_status_t mag_op_backward_neg(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *m1 = NULL;
    mag_status_t stat = mag_tensor_scalar(&m1, node->grad->ctx, node->grad->dtype, -1.f);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, node->grad, m1);
    mag_tensor_decref(m1);
    return stat;
}

mag_status_t mag_op_backward_log(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    return mag_div(grads, node->grad, x);
}

mag_status_t mag_op_backward_sqr(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *two;
    mag_status_t stat = mag_tensor_scalar(&two, x->ctx, x->dtype, 2.0f);
    if (mag_iserr(stat)) return stat;
    mag_tensor_t *two_x;
    stat = mag_mul(&two_x, x, two);
    if (mag_iserr(stat)) {
        mag_tensor_decref(two);
        return stat;
    }
    stat = mag_mul(grads, node->grad, two_x);
    mag_tensor_decref(two);
    mag_tensor_decref(two_x);
    return stat;
}

mag_status_t mag_op_backward_sqrt(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *sqrt_x;
    mag_status_t stat = mag_sqrt(&sqrt_x, x);
    if (mag_iserr(stat)) return stat;
    mag_tensor_t *two;
    stat = mag_tensor_scalar(&two, x->ctx, x->dtype, 2.0f);
    if (mag_iserr(stat)) {
        mag_tensor_decref(sqrt_x);
        return stat;
    }
    mag_tensor_t *denom;
    stat = mag_mul(&denom, sqrt_x, two);
    if (mag_iserr(stat)) {
        mag_tensor_decref(two);
        mag_tensor_decref(sqrt_x);
        return stat;
    }
    stat = mag_div(grads, node->grad, denom);
    mag_tensor_decref(two);
    mag_tensor_decref(sqrt_x);
    mag_tensor_decref(denom);
    return stat;
}

mag_status_t mag_op_backward_sin(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *cos_x;
    mag_status_t stat = mag_cos(&cos_x, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, node->grad, cos_x);
    mag_tensor_decref(cos_x);
    return stat;
}

mag_status_t mag_op_backward_cos(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *sinx;
    mag_status_t stat = mag_sin(&sinx, x);
    if (mag_iserr(stat)) return stat;
    mag_tensor_t *nsinx;
    stat = mag_neg(&nsinx, sinx);
    if (mag_iserr(stat)) {
        mag_tensor_decref(sinx);
        return stat;
    }
    stat = mag_mul(grads, node->grad, nsinx);
    mag_tensor_decref(sinx);
    mag_tensor_decref(nsinx);
    return stat;
}

mag_status_t mag_op_backward_exp(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *exp_x;
    mag_status_t stat = mag_exp(&exp_x, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, node->grad, exp_x);
    mag_tensor_decref(exp_x);
    return stat;
}

mag_status_t mag_op_backward_softmax(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y;
    mag_status_t stat = mag_softmax(&y, x);
    if (mag_iserr(stat)) return stat;
    mag_tensor_t *tmp;
    stat = mag_mul(&tmp, node->grad, y);
    if (mag_iserr(stat)) {
        mag_tensor_decref(y);
        return stat;
    }
    mag_tensor_t *sum_tmp;
    stat = mag_sum(&sum_tmp, tmp, NULL, 0, false);
    if (mag_iserr(stat)) {
        mag_tensor_decref(y);
        mag_tensor_decref(tmp);
        return stat;
    }
    mag_tensor_t *diff;
    stat = mag_sub(&diff, node->grad, sum_tmp);
    if (mag_iserr(stat)) {
        mag_tensor_decref(y);
        mag_tensor_decref(tmp);
        mag_tensor_decref(sum_tmp);
        return stat;
    }
    stat = mag_mul(grads, y, diff);
    mag_tensor_decref(tmp);
    mag_tensor_decref(sum_tmp);
    mag_tensor_decref(diff);
    mag_tensor_decref(y);
    return stat;
}

mag_status_t mag_op_backward_sigmoid(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv;
    mag_status_t stat = mag_sigmoid_dv(&dv, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, dv, node->grad);
    mag_tensor_decref(dv);
    return stat;
}

mag_status_t mag_op_backward_silu(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv;
    mag_status_t stat = mag_silu_dv(&dv, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, dv, node->grad);
    mag_tensor_decref(dv);
    return stat;
}

mag_status_t mag_op_backward_tanh(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv;
    mag_status_t stat = mag_tanh_dv(&dv, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, dv, node->grad);
    mag_tensor_decref(dv);
    return stat;
}

mag_status_t mag_op_backward_relu(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv;
    mag_status_t mask = mag_step(&dv, x);
    if (mag_iserr(mask)) return mask;
    mask = mag_mul(grads, dv, node->grad);
    mag_tensor_decref(dv);
    return mask;
}

mag_status_t mag_op_backward_gelu(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv;
    mag_status_t stat = mag_gelu_dv(&dv, x);
    if (mag_iserr(stat)) return stat;
    stat = mag_mul(grads, dv, node->grad);
    mag_tensor_decref(dv);
    return stat;
}

mag_status_t mag_op_backward_add(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    mag_status_t stat = MAG_STATUS_OK;
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        stat = mag_clone(grads, node->grad);
        if (mag_iserr(stat)) return stat;
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *grad = node->grad;
        stat = !mag_tensor_is_shape_eq(x, y) ? mag_repeat_back(&grad, grad, y) : mag_clone(&grad, grad);
        if (mag_iserr(stat)) return stat;
        grads[1] = grad;
    }
    return stat;
}

mag_status_t mag_op_backward_sub(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    mag_status_t stat = MAG_STATUS_OK;
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        stat = mag_clone(grads, node->grad);
        if (mag_iserr(stat)) return stat;
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *mg;
        stat = mag_neg(&mg, node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pmg = mg;
            stat = mag_repeat_back(&mg, pmg, y);
            if (mag_iserr(stat)) {
                mag_tensor_decref(pmg);
                return stat;
            }
            mag_tensor_decref(pmg);
        }
        grads[1] = mg;
    }
    return stat;
}

mag_status_t mag_op_backward_mul(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    mag_status_t stat = MAG_STATUS_OK;
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        stat = mag_mul(grads, node->grad, y);
        if (mag_iserr(stat)) return stat;
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *xg;
        stat = mag_mul(&xg, x, node->grad);
        if (mag_iserr(stat)) return stat;
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pxg = xg;
            stat = mag_repeat_back(&xg, pxg, y);
            if (mag_iserr(stat)) {
                mag_tensor_decref(pxg);
                return stat;
            }
            mag_tensor_decref(pxg);
        }
        grads[1] = xg;
    }
    return stat;
}

mag_status_t mag_op_backward_div(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    mag_status_t stat = MAG_STATUS_OK;
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        stat = mag_div(grads, node->grad, y);
        if (mag_iserr(stat)) return stat;
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *gx;
        stat = mag_mul(&gx, node->grad, x);
        if (mag_iserr(stat)) return stat;
        mag_tensor_t *yy;
        stat = mag_mul(&yy, y, y);
        if (mag_iserr(stat)) {
            mag_tensor_decref(gx);
            return stat;
        }
        mag_tensor_t *gxyy;
        stat = mag_div(&gxyy, gx, yy);
        if (mag_iserr(stat)) {
            mag_tensor_decref(gx);
            mag_tensor_decref(yy);
            return stat;
        }
        mag_tensor_t *mgxyy;
        stat = mag_neg(&mgxyy, gxyy);
        if (mag_iserr(stat)) {
            mag_tensor_decref(gx);
            mag_tensor_decref(yy);
            mag_tensor_decref(gxyy);
            return stat;
        }
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pmgxyy = mgxyy;
            stat = mag_repeat_back(&mgxyy, pmgxyy, y);
            if (mag_iserr(stat)) {
                mag_tensor_decref(pmgxyy);
                mag_tensor_decref(gxyy);
                mag_tensor_decref(yy);
                mag_tensor_decref(gx);
                return stat;
            }
            mag_tensor_decref(pmgxyy);
        }
        grads[1] = mgxyy;
        mag_tensor_decref(gxyy);
        mag_tensor_decref(yy);
        mag_tensor_decref(gx);
    }
    return stat;
}

mag_status_t mag_op_backward_matmul(mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];
    mag_status_t stat = MAG_STATUS_OK;
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *yT;
        stat = mag_transpose(&yT, y, 0, 1);
        if (mag_iserr(stat)) return stat;
        stat = mag_matmul(grads, node->grad, yT);
        mag_tensor_decref(yT);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *xT;
        stat = mag_transpose(&xT, x, 0, 1);
        if (mag_iserr(stat)) return stat;
        stat = mag_matmul(grads+1, xT, node->grad);
        mag_tensor_decref(xT);
    }
    return stat;
}

