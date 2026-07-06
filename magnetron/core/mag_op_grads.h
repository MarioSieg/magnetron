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
mag_status_t mag_op_backward_cast(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_view(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_transpose(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_permute(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_flip(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
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
mag_status_t mag_op_backward_cat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_slice(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_mul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_div(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_matmul(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_log2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_log10(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_log1p(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_rcp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_rsqrt(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_tan(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_sinh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_cosh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_asin(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_acos(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_atan(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_asinh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_acosh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_atanh(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_exp2(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_expm1(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_erf(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_erfc(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_hard_sigmoid(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_pow(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_min(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_max(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_where(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_clamp(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_tril(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_triu(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_repeat(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_gather(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);
mag_status_t mag_op_backward_embedding(mag_error_t *err, mag_au_state_t *node, mag_tensor_t **grads);

#ifdef __cplusplus
}
#endif

#endif
