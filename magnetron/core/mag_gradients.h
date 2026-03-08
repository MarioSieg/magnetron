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

#ifndef MAG_GRADIENTS_H
#define MAG_GRADIENTS_H

#include "mag_autodiff.h"

#ifdef __cplusplus
extern "C" {
#endif

mag_status_t mag_op_backward_clone(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_view(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_transpose(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_mean(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sum(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_abs(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_neg(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_log(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sqr(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_cos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_exp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_softmax(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_silu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_tanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_relu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_gelu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_add(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sub(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_div(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_matmul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);

#ifdef __cplusplus
}
#endif

#endif
