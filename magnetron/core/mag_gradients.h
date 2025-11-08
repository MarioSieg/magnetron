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

#ifndef MAG_GRADIENTS_H
#define MAG_GRADIENTS_H

#include "mag_autodiff.h"

#ifdef __cplusplus
extern "C" {
#endif

mag_status_t mag_op_backward_clone(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_view(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_transpose(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_mean(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sum(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_abs(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_neg(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_log(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sqr(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sqrt(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sin(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_cos(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_exp(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_softmax(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sigmoid(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_silu(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_tanh(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_relu(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_gelu(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_add(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sub(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_mul(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_div(mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_matmul(mag_au_state_t *node, mag_tensor_t **grads);

#ifdef __cplusplus
}
#endif

#endif
