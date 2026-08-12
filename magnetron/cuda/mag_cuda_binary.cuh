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

#include "mag_cuda.cuh"

namespace mag {
  constexpr unsigned BINARY_BLOCK_SIZE = 256;

  [[nodiscard]] extern mag_status_t binary_op_add(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_sub(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_mul(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_div(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_floordiv(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_mod(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_pow(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_and(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_or(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_xor(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_shl(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_shr(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_eq(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_ne(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_le(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_ge(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_lt(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t binary_op_gt(mag_error_t *err, const mag_command_t &cmd);
}
