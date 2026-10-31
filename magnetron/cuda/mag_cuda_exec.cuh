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
  [[nodiscard]] extern mag_status_t submit_op(mag_error_t *err, mag_device_t *dvc, const mag_command_t *cmd);
  [[nodiscard]] extern mag_status_t sync_device(mag_error_t *err, mag_device_t *dvc);
  [[nodiscard]] extern mag_status_t wait_device(mag_error_t *err, mag_device_t *dvc, mag_sync_barrier_seq_t seq);
  [[nodiscard]] extern bool query_device(mag_device_t *dvc, mag_sync_barrier_seq_t seq);
}
