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
    mag_tensor_t *scale;
    mag_try(mag_full_like(err, &scale, x, mag_scalar_from_f64(1.0 / (double)x->numel)));
    mag_try_do(mag_mul(err, grads, scale, node->grad), {
        mag_rc_decref(scale);
    });
    mag_rc_decref(scale);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_sum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *ones;
    mag_try(mag_full_like(err, &ones, x, mag_scalar_from_f64(1.0)));
    mag_try_do(mag_mul(err, grads, ones, node->grad), {
        mag_rc_decref(ones);
    });
    mag_rc_decref(ones);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_abs(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_status_t stat = MAG_STATUS_OK;
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *step = NULL;
    mag_tensor_t *one = NULL;
    mag_tensor_t *two = NULL;
    mag_tensor_t *step2 = NULL;
    mag_tensor_t *sign = NULL;
    stat = mag_step(err, &step, x);
    if (mag_iserr(stat)) goto error;
    stat = mag_scalar(err, &one, x->ctx, x->dtype, mag_scalar_from_f64(1.0));
    if (mag_iserr(stat)) goto error;
    stat = mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_f64(2.0));
    if (mag_iserr(stat)) goto error;
    stat = mag_mul(err, &step2, step, two);
    if (mag_iserr(stat)) goto error;
    stat = mag_sub(err, &sign, step2, one);
    if (mag_iserr(stat)) goto error;
    stat = mag_mul(err, grads, node->grad, sign);
error:
    if (sign) mag_rc_decref(sign);
    if (step2) mag_rc_decref(step2);
    if (two) mag_rc_decref(two);
    if (one) mag_rc_decref(one);
    if (step) mag_rc_decref(step);
    return stat;
}

mag_status_t mag_op_backward_neg(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *m1 = NULL;
    mag_try(mag_scalar(err, &m1, node->grad->ctx, node->grad->dtype, mag_scalar_from_f64(-1.0)));
    mag_try_do(mag_mul(err, grads, node->grad, m1), {
        mag_rc_decref(m1);
    });
    mag_rc_decref(m1);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_log(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    return mag_div(err, grads, node->grad, x);
}

mag_status_t mag_op_backward_sqr(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *two = NULL;
    mag_tensor_t *two_x = NULL;

    mag_try(mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_f64(2.0)));
    mag_try_do(mag_mul(err, &two_x, x, two), {
        mag_rc_decref(two);
    });
    mag_try_do(mag_mul(err, grads, node->grad, two_x), {
        mag_rc_decref(two_x);
        mag_rc_decref(two);
    });

    mag_rc_decref(two_x);
    mag_rc_decref(two);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_sqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *sqrt_x = NULL;
    mag_tensor_t *two = NULL;
    mag_tensor_t *denom = NULL;

    mag_try(mag_sqrt(err, &sqrt_x, x));
    mag_try_do(mag_scalar(err, &two, x->ctx, x->dtype, mag_scalar_from_f64(2.0)), {
        mag_rc_decref(sqrt_x);
    });
    mag_try_do(mag_mul(err, &denom, sqrt_x, two), {
        mag_rc_decref(two);
        mag_rc_decref(sqrt_x);
    });
    mag_try_do(mag_div(err, grads, node->grad, denom), {
        mag_rc_decref(denom);
        mag_rc_decref(two);
        mag_rc_decref(sqrt_x);
    });

    mag_rc_decref(denom);
    mag_rc_decref(two);
    mag_rc_decref(sqrt_x);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_sin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *cos_x = NULL;

    mag_try(mag_cos(err, &cos_x, x));
    mag_try_do(mag_mul(err, grads, node->grad, cos_x), {
        mag_rc_decref(cos_x);
    });
    mag_rc_decref(cos_x);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_cos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *sinx = NULL;
    mag_tensor_t *nsinx = NULL;

    mag_try(mag_sin(err, &sinx, x));
    mag_try_do(mag_neg(err, &nsinx, sinx), {
        mag_rc_decref(sinx);
    });
    mag_try_do(mag_mul(err, grads, node->grad, nsinx), {
        mag_rc_decref(nsinx);
        mag_rc_decref(sinx);
    });

    mag_rc_decref(nsinx);
    mag_rc_decref(sinx);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_exp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *exp_x = NULL;

    mag_try(mag_exp(err, &exp_x, x));
    mag_try_do(mag_mul(err, grads, node->grad, exp_x), {
        mag_rc_decref(exp_x);
    });
    mag_rc_decref(exp_x);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_softmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = NULL;
    mag_tensor_t *tmp = NULL;
    mag_tensor_t *sum_tmp = NULL;
    mag_tensor_t *diff = NULL;

    mag_try(mag_softmax(err, &y, x));
    mag_try_do(mag_mul(err, &tmp, node->grad, y), {
        mag_rc_decref(y);
    });
    mag_try_do(mag_sum(err, &sum_tmp, tmp, NULL, 0, false), {
        mag_rc_decref(tmp);
        mag_rc_decref(y);
    });
    mag_try_do(mag_sub(err, &diff, node->grad, sum_tmp), {
        mag_rc_decref(sum_tmp);
        mag_rc_decref(tmp);
        mag_rc_decref(y);
    });
    mag_try_do(mag_mul(err, grads, y, diff), {
        mag_rc_decref(diff);
        mag_rc_decref(sum_tmp);
        mag_rc_decref(tmp);
        mag_rc_decref(y);
    });

    mag_rc_decref(diff);
    mag_rc_decref(sum_tmp);
    mag_rc_decref(tmp);
    mag_rc_decref(y);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = NULL;

    mag_try(mag_sigmoid_dv(err, &dv, x));
    mag_try_do(mag_mul(err, grads, dv, node->grad), {
        mag_rc_decref(dv);
    });
    mag_rc_decref(dv);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_silu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = NULL;

    mag_try(mag_silu_dv(err, &dv, x));
    mag_try_do(mag_mul(err, grads, dv, node->grad), {
        mag_rc_decref(dv);
    });
    mag_rc_decref(dv);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_tanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = NULL;

    mag_try(mag_tanh_dv(err, &dv, x));
    mag_try_do(mag_mul(err, grads, dv, node->grad), {
        mag_rc_decref(dv);
    });
    mag_rc_decref(dv);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_relu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = NULL;

    mag_try(mag_step(err, &dv, x));
    mag_try_do(mag_mul(err, grads, dv, node->grad), {
        mag_rc_decref(dv);
    });
    mag_rc_decref(dv);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_gelu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *dv = NULL;

    mag_try(mag_gelu_dv(err, &dv, x));
    mag_try_do(mag_mul(err, grads, dv, node->grad), {
        mag_rc_decref(dv);
    });
    mag_rc_decref(dv);
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];

    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_try(mag_clone(err, grads, node->grad));
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *grad = NULL;
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_try(mag_repeat_back(err, &grad, node->grad, y));
        } else {
            mag_try(mag_clone(err, &grad, node->grad));
        }
        grads[1] = grad;
    }
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_sub(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];

    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_try(mag_clone(err, grads, node->grad));
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *mg = NULL;
        mag_try(mag_neg(err, &mg, node->grad));
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pmg = mg;
            mag_try_do(mag_repeat_back(err, &mg, pmg, y), {
                mag_rc_decref(pmg);
            });
            mag_rc_decref(pmg);
        }
        grads[1] = mg;
    }
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];

    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_try(mag_mul(err, grads, node->grad, y));
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *xg = NULL;
        mag_try(mag_mul(err, &xg, x, node->grad));
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pxg = xg;
            mag_try_do(mag_repeat_back(err, &xg, pxg, y), {
                mag_rc_decref(pxg);
            });
            mag_rc_decref(pxg);
        }
        grads[1] = xg;
    }
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_div(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];

    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_try(mag_div(err, grads, node->grad, y));
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *gx = NULL;
        mag_tensor_t *yy = NULL;
        mag_tensor_t *gxyy = NULL;
        mag_tensor_t *mgxyy = NULL;

        mag_try(mag_mul(err, &gx, node->grad, x));
        mag_try_do(mag_mul(err, &yy, y, y), {
            mag_rc_decref(gx);
        });
        mag_try_do(mag_div(err, &gxyy, gx, yy), {
            mag_rc_decref(yy);
            mag_rc_decref(gx);
        });
        mag_try_do(mag_neg(err, &mgxyy, gxyy), {
            mag_rc_decref(gxyy);
            mag_rc_decref(yy);
            mag_rc_decref(gx);
        });

        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t *pmgxyy = mgxyy;
            mag_try_do(mag_repeat_back(err, &mgxyy, pmgxyy, y), {
                mag_rc_decref(pmgxyy);
                mag_rc_decref(gxyy);
                mag_rc_decref(yy);
                mag_rc_decref(gx);
            });
            mag_rc_decref(pmgxyy);
        }

        grads[1] = mgxyy;
        mag_rc_decref(gxyy);
        mag_rc_decref(yy);
        mag_rc_decref(gx);
    }
    return MAG_STATUS_OK;
}

mag_status_t mag_op_backward_matmul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads) {
    mag_tensor_t *x = node->op_inputs[0];
    mag_tensor_t *y = node->op_inputs[1];

    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *yT;
        mag_try(mag_transpose(err, &yT, y, 0, 1));
        mag_try_do(mag_matmul(err, grads, node->grad, yT), {
            mag_rc_decref(yT);
        });
        mag_rc_decref(yT);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t *xT;
        mag_try(mag_transpose(err, &xT, x, 0, 1));
        mag_try_do(mag_matmul(err, grads + 1, xT, node->grad), {
            mag_rc_decref(xT);
        });
        mag_rc_decref(xT);
    }
    return MAG_STATUS_OK;
}
