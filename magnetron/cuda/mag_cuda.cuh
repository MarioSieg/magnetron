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
#include <stdexcept>
#include <string>

#define MAG_COORDS_ITER_INTEGRAL_TYPE int

#include <core/mag_backend.h>
#include <core/mag_context.h>
#include <core/mag_tensor.h>
#include <core/mag_coords_iter.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "mag_cuda_device.cuh"

extern "C" {
  mag_backend_decl_interface();
}

namespace mag {
  constexpr uint32_t MAG_CUDA_BACKEND_VERSION = mag_ver_encode(0, 1, 0);

#define mag_cu_check(E, expr, msg) \
  do { \
    CUresult ___res___ = (expr); \
    if (mag_unlikely(___res___ != CUDA_SUCCESS)) { \
      const char *___err___ = nullptr; \
      cuGetErrorString(___res___, &___err___); \
      return mag_set_error((E), MAG_ERR_BACKEND, \
        "cuda error in " #expr ": %s: %s", \
        (msg), ___err___ ? ___err___ : "unknown error"); \
    } \
  } while (0)

  #define mag_cu_rt_check(E, expr, msg) \
    do { \
      cudaError_t ___res___ = (expr); \
      if (mag_unlikely(___res___ != cudaSuccess)) \
        return mag_set_error((E), MAG_ERR_BACKEND, "cuda error in: " #expr ": %s: %s", (msg), cudaGetErrorString(___res___)); \
    } while (0)

  /*
  ** These must have external linkage: 'static inline' at namespace scope gives every translation
  ** unit its own private copy, so a store from one .cu is invisible to the readers inlined into
  ** the others. Plain 'inline' is the C++17 spelling for a single object shared across all TUs.
  */
  inline std::atomic_uint64_t global_seed = 0;
  inline std::atomic_uint64_t global_subseq = 0;

  inline std::atomic_bool global_async_alloc = false;

  [[nodiscard]] inline cudaError_t stream_alloc(void **p, size_t size, cudaStream_t stream) {
    if (global_async_alloc.load(std::memory_order_relaxed))
      return cudaMallocAsync(p, size, stream);
    return cudaMalloc(p, size);
  }

  [[nodiscard]] inline cudaError_t stream_free(void *p, cudaStream_t stream) {
    if (global_async_alloc.load(std::memory_order_relaxed))
      return cudaFreeAsync(p, stream);
    return cudaFree(p);
  }

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
