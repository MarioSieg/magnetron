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
  [[nodiscard]] extern mag_status_t interp_op_interpolate(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
  [[nodiscard]] extern mag_status_t interp_op_interpolate_back(mag_error_t *err, const mag_command_t &cmd, cudaStream_t stream);
}
