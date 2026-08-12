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
  constexpr unsigned FILL_BLOCK_SIZE = 256;

  [[nodiscard]] extern mag_status_t fill_op_fill(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_masked_fill(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_fill_rand_uniform(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_fill_rand_normal(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_rand_bernoulli(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_rand_perm(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_arange(mag_error_t *err, const mag_command_t &cmd);
  [[nodiscard]] extern mag_status_t fill_op_eye(mag_error_t *err, const mag_command_t &cmd);
}
