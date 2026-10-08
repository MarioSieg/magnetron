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
  struct gemm_desc final {
    mag_dtype_t dtype = MAG_DTYPE_BFLOAT16;
    const void *a = nullptr;
    const void *b = nullptr;
    void *c = nullptr;
    const void *bias = nullptr;
    int64_t M = 0, N = 0, K = 0;
    int64_t lda = 0, ldb = 0, ldc = 0;
    bool a_kmajor = true;
    bool b_kmajor = false;
    int64_t batch = 1;
    int64_t a_bstride = 0, b_bstride = 0, c_bstride = 0;
    int64_t k_batch = 1;
    int64_t a_kbstride = 0, b_kbstride = 0;
  };

  [[nodiscard]] extern bool gemm_supported(const gemm_desc &d) noexcept;

  [[nodiscard]] extern mag_status_t gemm(mag_error_t *err, const physical_device &dev, const gemm_desc &d, cudaStream_t stream);

  [[nodiscard]] extern bool tcgen05_disabled() noexcept;

#if defined(MAG_HAVE_CUDA_SM_100) || (defined(MAG_CUDA_SM) && MAG_CUDA_SM == 100)
  namespace sm_100 {
    [[nodiscard]] extern bool gemm_supported(const gemm_desc &d, uint32_t compute_capability) noexcept;
    [[nodiscard]] extern mag_status_t gemm(mag_error_t *err, const gemm_desc &d, cudaStream_t stream, uint32_t num_sms);
  }
#endif
}
