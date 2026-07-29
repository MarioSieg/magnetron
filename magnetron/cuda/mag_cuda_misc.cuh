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
  extern mag_status_t misc_op_one_hot(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_topk(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_tril(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_triu(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_multinomial(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_cat(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_pad(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_cusum(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_cuprod(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_cumax(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_cumin(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_repeat(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_repeat_interleave(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_index_add(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_scatter(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_scatter_add(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_matmul(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_repeat_back(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_gather(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_embedding(mag_error_t *err, const mag_command_t &cmd);
  extern mag_status_t misc_op_where(mag_error_t *err, const mag_command_t &cmd);
}
