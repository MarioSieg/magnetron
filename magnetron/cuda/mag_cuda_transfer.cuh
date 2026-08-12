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
  [[nodiscard]] extern mag_status_t bidirectional_transfer(
    mag_error_t *err,
    mag_device_t *dvc,
    mag_transfer_dir_t dir,
    mag_tensor_t *src,
    mag_tensor_t *dst
  );
}
