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

#pragma once

#include "mag_cuda_prelude.cuh"

namespace mag {
  constexpr unsigned UNARY_BLOCK_SIZE = 256;

  [[nodiscard]] extern mag_status_t unary_op_clone(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_cast(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_abs(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sgn(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_neg(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_not(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_log(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_log10(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_log1p(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_log2(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sqr(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_rcp(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sqrt(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_rsqrt(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sin(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_cos(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_tan(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_asin(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_acos(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_atan(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sinh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_cosh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_tanh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_asinh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_acosh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_atanh(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_step(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_erf(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_erfc(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_exp(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_exp2(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_expm1(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_floor(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_ceil(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_round(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_trunc(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_softmax(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_softmax_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sigmoid(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_sigmoid_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_hard_sigmoid(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_silu(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_silu_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_tanh_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_relu(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_relu_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_gelu(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t unary_op_gelu_dv(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
}
