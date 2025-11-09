/*
** +---------------------------------------------------------------------+
** | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
** | Licensed under the Apache License, Version 2.0                      |
** |                                                                     |
** | Website : https://mariosieg.com                                     |
** | GitHub  : https://github.com/MarioSieg                              |
** | License : https://www.apache.org/licenses/LICENSE-2.0               |
** +---------------------------------------------------------------------+
*/

#pragma once

#include <core/mag_backend.h>
#include <core/mag_context.h>
#include <core/mag_tensor.h>

#include <cuda.h>
#include <cuda_fp16.h>

extern "C" {
  mag_backend_decl_interface();
}

namespace mag {
  constexpr uint32_t MAG_CUDA_BACKEND_VERSION = mag_ver_encode(0, 1, 0);
}
