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

#include <atomic>

/* Use int instead of int64 for indexing iterators as int64 is much slower on CUA */
#define MAG_COORDS_ITER_INTEGRAL_TYPE int

#include <core/mag_backend.h>
#include <core/mag_context.h>
#include <core/mag_tensor.h>
#include <core/mag_coords_iter.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" {
  mag_backend_decl_interface();
}

namespace mag {
  constexpr uint32_t MAG_CUDA_BACKEND_VERSION = mag_ver_encode(0, 1, 0);

  static inline std::atomic_uint64_t global_seed = 0;
  static inline std::atomic_uint64_t global_subseq = 0;

  [[nodiscard]] inline int numel_i32(const mag_tensor_t *x) {
    int64_t numel = x->numel;
    assert(numel <= INT_MAX);
    return static_cast<int>(numel);
  }

  template <typename T>
  [[nodiscard]] __host__ __device__ __forceinline__ static T unpack_scalar(mag_scalar_t scalar) {
    switch (scalar.type) {
      case MAG_SCALAR_TYPE_F64: return static_cast<T>(scalar.value.float64);
      case MAG_SCALAR_TYPE_I64: return static_cast<T>(scalar.value.int64);
      default:
      case MAG_SCALAR_TYPE_U64: return static_cast<T>(scalar.value.uint64);
    }
  }
}
