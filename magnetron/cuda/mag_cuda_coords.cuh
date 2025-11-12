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
        int shape[MAG_MAX_DIMS] = {};
        int strides[MAG_MAX_DIMS] = {};

        __host__ __device__ explicit tensor_coords(const mag_tensor_coords_t& coords) noexcept {
            rank = static_cast<int>(coords.rank);
            for (int i=0; i < rank; i++) {
                shape[i] = static_cast<int>(coords.shape[i]);
                strides[i] = static_cast<int>(coords.strides[i]);
            }
        }

        [[nodiscard]] __device__ __forceinline__ int to_offset(int i) const noexcept {
            const int *rd = shape;
            const int *rs = strides;
            int ra = rank-1;
            int o = 0;
            for (int k=ra; k >= 0; --k) {
                int di = rd[k];
                int ax = i % di;
                i /= di;
                o += ax*rs[k];
            }
            return o;
        }

        [[nodiscard]] __device__ __forceinline__ int broadcast(const tensor_coords &x, int i) const noexcept {
            const int *rd = shape;
            const int *xd = x.shape;
            const int *xs = x.strides;
            int ra = rank;
            int delta = ra-- - x.rank;
            int o = 0;
            for (int k=ra; k >= 0; --k) {
                int di = rd[k];
                int ax = i % di;
                i /= di;
                int kd = k-delta;
                if (kd >= 0 && xd[kd] > 1)
                    o += ax*xs[kd];
            }
            return o;
        }
    };
}
