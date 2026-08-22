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
  constexpr bool sync_after_each_op = false;

  [[nodiscard]] extern mag_status_t submit_op(mag_error_t *err, mag_device_t *dvc, const mag_command_t *cmd);
}
