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

#include "mag_cuda_matmul.cuh"

#include <core/mag_prng_philox4x32.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace mag {
    struct matmul_desc_t {
        int64_t rank;
        int64_t shape[MAG_MAX_DIMS];
        int64_t stride[MAG_MAX_DIMS];
    };

    static matmul_desc_t desc_from(const mag_tensor_t *t) {
        matmul_desc_t d{};
        d.rank = t->coords.rank;
        for (int64_t i=0; i < d.rank; ++i) {
            d.shape[i] = t->coords.shape[i];
            d.stride[i] = t->coords.strides[i];
        }
        return d;
    }

    __device__ __forceinline__ int64_t mag_d_offset_rmn(matmul_desc_t t, int64_t flat, int64_t i, int64_t j) {
        int64_t ra = t.rank;
        const int64_t *td = t.shape;
        const int64_t *ts = t.stride;
        if (ra <= 3) {
            switch (ra) {
                case 0: return 0;
                case 1: return flat*ts[0];
                case 2: return i*ts[0] + j*ts[1];
                case 3: return flat*ts[0] + i*ts[1] + j*ts[2];
            }
        }
        int64_t off = 0, rem = flat;
        for (int64_t d = ra-3; d >= 0; --d) {
            int64_t idx = rem % td[d];
            rem /= td[d];
            off += idx*ts[d];
        }
        off += i*ts[ra-2];
        off += j*ts[ra-1];
        return off;
    }

    template <typename T>
    __device__ __forceinline__ float d_mm_to_f32(T x) {
        if constexpr (std::is_same_v<T, float>) return x;
        else if constexpr (std::is_same_v<T, half>) return __half2float(x);
        else return __bfloat162float(x);
    }

    template <typename T>
    __device__ __forceinline__ T d_mm_from_f32(float v) {
        if constexpr (std::is_same_v<T, float>) return v;
        else if constexpr (std::is_same_v<T, half>) return __float2half(v);
        else return __float2bfloat16(v);
    }

    template <typename T>
    __device__ float d_load_x(const matmul_desc_t &x, const T *bx, int64_t xb_flat, int64_t i, int64_t k) {
        int64_t off = (x.rank == 1) ? mag_d_offset_rmn(x, k, 0, 0) : mag_d_offset_rmn(x, xb_flat, i, k);
        return d_mm_to_f32(bx[off]);
    }

    template <typename T>
    __device__ float d_load_y(const matmul_desc_t &y, const T *by, int64_t yb_flat, int64_t k, int64_t n) {
        int64_t off = (y.rank == 1) ? mag_d_offset_rmn(y, k, 0, 0) : mag_d_offset_rmn(y, yb_flat, k, n);
        return d_mm_to_f32(by[off]);
    }

    template <typename T>
    __device__ void d_store_r(const matmul_desc_t &r, T *br, int64_t rb_flat, int64_t i, int64_t n, float v) {
        if (r.rank == 0)
            br[0] = d_mm_from_f32<T>(v);
        else if (r.rank == 1) {
            int64_t off = mag_d_offset_rmn(r, n, 0, 0);
            br[off] = d_mm_from_f32<T>(v);
        } else {
            int64_t off = mag_d_offset_rmn(r, rb_flat, i, n);
            br[off] = d_mm_from_f32<T>(v);
        }
    }

    template <typename T>
    __global__ static void matmul_kernel(
        int64_t M, int64_t N, int64_t K, int64_t batch_total,
        int64_t bdx, int64_t bdy, int64_t bdr,
        matmul_desc_t dx, matmul_desc_t dy, matmul_desc_t dr,
        T *br, const T *bx, const T *by
    ) {
        int64_t tid = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t total = batch_total * M * N;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        for (int64_t w = tid; w < total; w += step) {
            int64_t mn = M * N;
            int64_t batch_idx = w / mn;
            int64_t rem = w % mn;
            int64_t i = rem / N;
            int64_t n = rem % N;
            int64_t idx_r[MAG_MAX_DIMS] = {0};
            {
                int64_t t = batch_idx;
                for (int64_t d = bdr-1; d >= 0; --d) {
                    idx_r[d] = t % dr.shape[d];
                    t /= dr.shape[d];
                }
            }
            int64_t xb_flat = 0;
            for (int64_t d = 0; d < bdx; ++d) {
                int64_t rd = bdr - bdx + d;
                int64_t idx = (dx.shape[d] == 1) ? 0 : idx_r[rd];
                xb_flat = xb_flat * dx.shape[d] + idx;
            }
            int64_t yb_flat = 0;
            for (int64_t d = 0; d < bdy; ++d) {
                int64_t rd = bdr - bdy + d;
                int64_t idx = (dy.shape[d] == 1) ? 0 : idx_r[rd];
                yb_flat = yb_flat * dy.shape[d] + idx;
            }
            int64_t rb_flat = 0;
            for (int64_t d = 0; d < bdr; ++d)
                rb_flat = rb_flat * dr.shape[d] + idx_r[d];
            float sum = 0.f;
            for (int64_t k = 0; k < K; ++k) {
                float ax = d_load_x(dx, bx, xb_flat, i, k);
                float byv = d_load_y(dy, by, yb_flat, k, n);
                sum += ax * byv;
            }
            d_store_r(dr, br, rb_flat, i, n, sum);
        }
    }

    template <typename T>
    static void launch_matmul(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        const mag_tensor_t *y = cmd.in[1];
        int64_t M = (x->coords.rank == 1) ? 1 : x->coords.shape[x->coords.rank - 2];
        int64_t N = (y->coords.rank == 1) ? 1 : y->coords.shape[y->coords.rank - 1];
        int64_t K = x->coords.shape[x->coords.rank - 1];
        int64_t bdr = (r->coords.rank > 2) ? (r->coords.rank - 2) : 0;
        int64_t batch_total = 1;
        for (int64_t d = 0; d < bdr; ++d) batch_total *= r->coords.shape[d];
        int64_t bdx = (x->coords.rank > 2) ? (x->coords.rank - 2) : 0;
        int64_t bdy = (y->coords.rank > 2) ? (y->coords.rank - 2) : 0;
        matmul_desc_t dx = desc_from(x);
        matmul_desc_t dy = desc_from(y);
        matmul_desc_t dr = desc_from(r);
        auto *br = reinterpret_cast<T *>(mag_tensor_data_ptr_mut(r));
        const auto *bx = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        const auto *by = reinterpret_cast<const T *>(mag_tensor_data_ptr(y));
        int64_t total = batch_total * M * N;
        int64_t blocks = (total + MATMUL_BLOCK_SIZE - 1) / MATMUL_BLOCK_SIZE;
        matmul_kernel<T><<<blocks, MATMUL_BLOCK_SIZE>>>(M, N, K, batch_total, bdx, bdy, bdr, dx, dy, dr, br, bx, by);
    }

    void misc_op_matmul(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_matmul<float>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_matmul<half>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_matmul<__nv_bfloat16>(cmd); break;
            default: mag_assert(false, "matmul: unsupported dtype");
        }
    }

}
