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

#include <atomic>

#include <core/mag_backend.h>
#include <core/mag_context.h>
#include <core/mag_tensor.h>
#include <core/mag_coords_iter.h>

#include <cuda.h>
#include <cuda_fp16.h>

extern "C" {
  mag_backend_decl_interface();
}

namespace mag {
  constexpr uint32_t MAG_CUDA_BACKEND_VERSION = mag_ver_encode(0, 1, 0);

  static inline std::atomic_uint64_t global_seed = 0;
  static inline std::atomic_uint64_t global_subseq = 0;

  template <typename T>
  concept is_floating_point = std::is_same_v<T, mag_e8m23_t> || std::is_same_v<T, half>;

  template <typename T>
  concept is_integer = std::is_same_v<T, int32_t> || std::is_same_v<T, uint8_t>; // TODO: own boolean type

  template <typename T>
  concept is_integral = std::is_same_v<T, int32_t> || std::is_same_v<T, uint8_t>;

  template <typename T>
  concept is_numeric = is_floating_point<T> || is_integral<T>;

  template <typename T>
  concept is_dtype = is_floating_point<T> || is_integral<T>;

  template <typename T>
  [[nodiscard]] T unpack_param(const mag_op_attr_t (&params)[MAG_MAX_OP_PARAMS], size_t i) {
    if constexpr (is_floating_point<T>) return static_cast<T>(mag_op_attr_unpack_e8m23_or_panic(params[i]));
    else return static_cast<T>(mag_op_attr_unpack_i64_or_panic(params[i]));
  }
}
