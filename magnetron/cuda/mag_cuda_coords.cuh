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

#include "mag_cuda.cuh"

namespace mag {
    struct tensor_coords final {
        int rank = 1;
        int64_t shape[MAG_MAX_DIMS] = {};
        int64_t strides[MAG_MAX_DIMS] = {};

        __host__ __device__ explicit tensor_coords(const mag_coords_t &co) noexcept {
            rank = static_cast<int>(co.rank);
            for (int i=0; i < rank; i++) {
                shape[i] = co.shape[i];
                strides[i] = co.strides[i];
            }
        }

        [[nodiscard]] __device__ __forceinline__ int64_t to_offset(int64_t i) const noexcept {
            const int64_t *rd = shape;
            const int64_t *rs = strides;
            int ra = rank-1;
            int64_t o = 0;
            for (int k=ra; k >= 0; --k) {
                int64_t di = rd[k];
                int64_t ax = i % di;
                i /= di;
                o += ax*rs[k];
            }
            return o;
        }

        [[nodiscard]] __device__ __forceinline__ int64_t broadcast(const tensor_coords &x, int64_t i) const noexcept {
            const int64_t *rd = shape;
            const int64_t *xd = x.shape;
            const int64_t *xs = x.strides;
            int ra = rank;
            int delta = ra-- - x.rank;
            int64_t o = 0;
            for (int k=ra; k >= 0; --k) {
                int64_t di = rd[k];
                int64_t ax = i % di;
                i /= di;
                int kd = k-delta;
                if (kd >= 0 && xd[kd] > 1)
                    o += ax*xs[kd];
            }
            return o;
        }
    };
}
