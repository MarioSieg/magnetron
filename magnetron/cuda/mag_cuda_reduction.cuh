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
  constexpr unsigned REDUCTION_BLOCK_SIZE = 256;
  constexpr unsigned WIDE_REDUCTION_BLOCK_SIZE = 1024;
  constexpr int64_t WIDE_REDUCTION_OUTPUTS = 64;
  constexpr int64_t REDUCTION_SPLIT_MAX_OUTPUTS = 1024;
  constexpr int64_t REDUCTION_SPLIT_MIN_ELEMS = 8192;
  constexpr int64_t REDUCTION_SPLIT_ELEMS_PER_THREAD = 8;
  constexpr int64_t REDUCTION_MAX_SPLITS = 1024;

  [[nodiscard]] extern mag_status_t reduce_op_mean(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_minima(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_maxima(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_sum(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_prod(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_all(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_any(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_argmin(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t reduce_op_argmax(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
}