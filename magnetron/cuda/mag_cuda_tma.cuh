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

#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <mutex>

namespace mag::tma {
  [[nodiscard]] inline PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
    static std::once_flag once;
    static std::atomic<PFN_cuTensorMapEncodeTiled_v12000> fn = nullptr;
    std::call_once(once, [] {
      cudaDriverEntryPointQueryResult stat;
      PFN_cuTensorMapEncodeTiled_v12000 pfn = nullptr;
      cudaError_t res = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled",
        reinterpret_cast<void **>(&pfn),
        12000,
        cudaEnableDefault,
        &stat
      );
      if (mag_unlikely(res != cudaSuccess || stat != cudaDriverEntryPointSuccess)) pfn = nullptr;
      fn.store(pfn, std::memory_order_release);
    });
    return fn.load(std::memory_order_acquire);
  }

  template <typename T>
  [[nodiscard]] constexpr CUtensorMapDataType data_type() noexcept {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    else if constexpr (std::is_same_v<T, half>) return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    else return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  }

  template <typename T, size_t RANK>
  [[nodiscard]] inline bool encode_tiled(
    CUtensorMap &out,
    const void *base,
    const std::array<uint64_t, RANK> &dims,
    const std::array<uint64_t, RANK-1> &strides,
    const std::array<uint32_t, RANK> &box,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B
  ) noexcept {
    if (mag_unlikely(!base || (reinterpret_cast<uintptr_t>(base) & 15))) return false;
    for (uint64_t d : dims) if (mag_unlikely(d < 1 || d > (1ull<<32)-1)) return false;
    for (uint64_t s : strides) if (mag_unlikely((s & 15) || s >= (1ull<<40))) return false;
    for (uint32_t b : box) if (mag_unlikely(b < 1 || b > 256)) return false;
    auto *encode = encode_fn();
    if (mag_unlikely(!encode)) return false;
    std::array<uint32_t, RANK> elem_stride {};
    elem_stride.fill(1);
    CUresult rc = (*encode)(
      &out,
      data_type<T>(),
      static_cast<cuuint32_t>(RANK),
      const_cast<void *>(base),
      dims.data(),
      strides.data(),
      box.data(),
      elem_stride.data(),
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      l2,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    return rc == CUDA_SUCCESS;
  }

  template <typename T>
  [[nodiscard]] inline bool encode_operand_3d(
    CUtensorMap &out,
    const void *base,
    int64_t rows, int64_t cols, int64_t ld,
    int64_t batch, int64_t bstride,
    uint32_t box_cols, uint32_t box_rows,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B
  ) noexcept {
    if (batch <= 1) { batch = 1; bstride = rows*ld; }
    return encode_tiled<T, 3>(
      out,
      base,
      { static_cast<uint64_t>(cols), static_cast<uint64_t>(rows), static_cast<uint64_t>(batch) },
      { static_cast<uint64_t>(ld)*sizeof(T), static_cast<uint64_t>(bstride)*sizeof(T) },
      { box_cols, box_rows, 1u },
      swizzle,
      l2
    );
  }
}
