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

#include "mag_cuda_prelude.cuh"

namespace mag {
  bool can_use_i32_indexes(const mag_tensor_t *tensor) noexcept {
    constexpr int64_t lim = std::numeric_limits<int32_t>::max();
    if (tensor->meta.numel > lim) return false;
    const mag_coords_t &co = tensor->meta.coords;
    int64_t span = 0;
    for (int64_t k=0; k < co.rank; ++k) {
      int64_t s = co.strides[k];
      if (s < 0) s = -s; /* A negative stride covers the same distance, just the other way. */
      if (co.shape[k] > 0) span += (co.shape[k]-1)*s;
      if (span > lim) return false;
    }
    return true;
  }
}
