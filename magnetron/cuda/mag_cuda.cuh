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

  template <typename I>
  struct coords_iter final {
    static_assert(std::is_integral_v<I> && std::is_signed_v<I>, "Index cardinal must be signed integral type");
    I rank;
    I shape[MAG_MAX_DIMS];
    I strides[MAG_MAX_DIMS];

    __host__ __device__ constexpr coords_iter() noexcept = default;
    __host__ __device__ constexpr explicit coords_iter(const mag_coords_t &co) noexcept {
      rank = static_cast<I>(co.rank);
      for (int64_t k=0; k < co.rank; ++k) {
        shape[k] = static_cast<I>(co.shape[k]);
        strides[k] = static_cast<I>(co.strides[k]);
      }
    }
    __host__ __device__ constexpr explicit coords_iter(const mag_tensor_t *tensor) noexcept : coords_iter{tensor->meta.coords} {}

    [[nodiscard]] __device__ __forceinline__ constexpr I operator()(I i) const noexcept {
        I o {};
        for (I k=rank-1; k >= 0; --k) {
          I dim = shape[k];
          I ax = i % dim;
          i /= dim;
          o += ax*strides[k];
        }
        return o;
      }

      [[nodiscard]] __device__ __forceinline__ constexpr I broadcast(const coords_iter &cx, I i) const noexcept {
        I delta = rank - cx.rank;
        I o {};
        for (I k=rank-1; k >= 0; --k) {
          I dim = shape[k];
          I ax = i % dim;
          i /= dim;
          I kd = k-delta;
          if (kd >= 0 && cx.shape[kd] > 1)
            o += ax*cx.strides[kd];
        }
        return o;
      }
  };

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
