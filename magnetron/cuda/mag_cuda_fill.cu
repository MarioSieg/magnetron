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

#include "mag_cuda_fill.cuh"

#include <core/mag_prng_philox4x32.h>

namespace mag {
    template <typename T, const bool contig> requires is_dtype<T>
    __global__ static void fill_kernel(
        int64_t n,
        T *__restrict__ o,
        T v,
        [[maybe_unused]] mag_coords_iter_t rc
    ) {
        int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        if constexpr (contig) {
            if (ti >= n) return;
            o[ti] = v;
        } else {
            int64_t step = static_cast<int64_t>(blockDim.x)*gridDim.x;
            for (; ti < n; ti += step) {
                int64_t ri = mag_coords_iter_to_offset(&rc, ti);
                o[ri] = v;
            }
        }
    }

    template <typename T, const bool contig> requires is_dtype<T>
    __global__ static void masked_fill_kernel(
        int64_t n,
        T *__restrict__ o,
        const uint8_t *__restrict__ mask,
        T v,
        [[maybe_unused]] mag_coords_iter_t rc,
        mag_coords_iter_t mc
    ) {
        int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        if constexpr (contig) {
            for (; ti < n; ti += step) {
                int64_t mi = mag_coords_iter_broadcast(&rc, &mc, ti);
                if (mask[mi]) o[ti] = v;
            }
        } else {
            for (; ti < n; ti += step) {
                int64_t ri = mag_coords_iter_to_offset(&rc, ti);
                int64_t mi = mag_coords_iter_broadcast(&rc, &mc, ti);
                if (mask[mi]) o[ri] = v;
            }
        }
    }

    template <typename T> requires is_dtype<T>
    static void launch_fill_kernel(mag_tensor_t *r, const mag_command_t *cmd, const mag_tensor_t *mask = nullptr) {
        auto *o = static_cast<T *>(mag_tensor_get_data_ptr(r));
        auto v = unpack_param<T>(cmd->params, 0);
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        bool rc_cont = mag_tensor_is_contiguous(r);
        int64_t n = mag_tensor_get_numel(r);
        int64_t blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        if (mask) {
            auto *pm = static_cast<const uint8_t *>(mag_tensor_get_data_ptr(mask));
            mag_coords_iter_t mc;
            mag_coords_iter_init(&mc, &mask->coords);
            if (rc_cont) masked_fill_kernel<T, true><<<blocks, FILL_BLOCK_SIZE>>>(n, o, pm, v, rc, mc);
            else  masked_fill_kernel<T, false><<<blocks, FILL_BLOCK_SIZE>>>(n, o, pm, v, rc, mc);
        } else {
            if (rc_cont) fill_kernel<T, true><<<blocks, FILL_BLOCK_SIZE>>>(n, o, v, rc);
            else fill_kernel<T, false><<<blocks, FILL_BLOCK_SIZE>>>(n, o, v, rc);
        }
    }

    template <typename T, const bool is_cont, const bool normal>  requires is_numeric<T>
    __global__ static void fill_random_kernel(
        int64_t n,
        T *__restrict__ o,
        T p0,
        T p1,
        uint64_t seed,
        uint64_t subseq,
        [[maybe_unused]] mag_coords_iter_t rc
    ) {
        int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        int64_t nb = (n+3)>>2;
        if (ti >= nb) return;
        mag_philox4x32_stream_t stream;
        mag_philox4x32_stream_seed(&stream, seed, subseq + static_cast<uint64_t>(ti));
        for (int64_t b=ti; b < nb; b += step) {
            int64_t base = b<<2;
            mag_philox4x32_e8m23x4_t r;
            if constexpr (normal) r = mag_philox4x32_next_e8m23x4_normal(&stream, p0, p1);
            else r = mag_philox4x32_next_e8m23x4_uniform(&stream, p0, p1);
            int64_t mk = n-base;
            if (mk > 4) mk = 4;
            if constexpr (is_cont) {
                #pragma unroll
                for (int64_t k=0; k < mk; ++k)
                    o[base+k] = static_cast<T>(r.v[k]);
            } else {
                #pragma unroll
                for (int64_t k=0; k < mk; ++k) {
                    int64_t ri = mag_coords_iter_to_offset(&rc, base+k);
                    o[ri] = static_cast<T>(r.v[k]);
                }
            }
        }
    }

    template <typename T, const bool normal> requires is_dtype<T>
    static void launch_rand_fill_kernel(mag_tensor_t *r, const mag_command_t *cmd) {
        auto *o = static_cast<T *>(mag_tensor_get_data_ptr(r));
        auto p0 = unpack_param<T>(cmd->params, 0);
        auto p1 = unpack_param<T>(cmd->params, 1);
        int64_t n = mag_tensor_get_numel(r);
        int64_t blocks = (((n+3)>>2)+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        uint64_t seed = global_seed.load(std::memory_order_relaxed);
        uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
        if (mag_tensor_is_contiguous(r)) fill_random_kernel<T, true, normal><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p0, p1, seed, subseq, rc);
        else fill_random_kernel<T, false, normal><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p0, p1, seed, subseq, rc);
    }

    void fill_op_fill(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->in[0];
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_fill_kernel<mag_e8m23_t>(r, cmd); break;
            case MAG_DTYPE_E5M10: launch_fill_kernel<half>(r, cmd); break;
            case MAG_DTYPE_I32: launch_fill_kernel<int32_t>(r, cmd); break;
            case MAG_DTYPE_BOOL: launch_fill_kernel<uint8_t>(r, cmd); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_masked_fill(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->in[0];
        mag_tensor_t *mask = reinterpret_cast<mag_tensor_t *>(static_cast<uintptr_t>(mag_op_attr_unpack_i64_or_panic(cmd->params[0]))); // TODO: pass in cmd in why the fuck are these here
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_fill_kernel<mag_e8m23_t>(r, cmd, mask); break;
            case MAG_DTYPE_E5M10: launch_fill_kernel<half>(r, cmd, mask); break;
            case MAG_DTYPE_I32: launch_fill_kernel<int32_t>(r, cmd, mask); break;
            case MAG_DTYPE_BOOL: launch_fill_kernel<uint8_t>(r, cmd, mask); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_fill_rand_uniform(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->in[0];
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_rand_fill_kernel<mag_e8m23_t, false>(r, cmd); break;
            case MAG_DTYPE_E5M10: launch_rand_fill_kernel<half, false>(r, cmd); break;
            case MAG_DTYPE_I32: launch_rand_fill_kernel<int32_t, false>(r, cmd); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_fill_rand_normal(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->in[0];
        switch (r->dtype) {
            case MAG_DTYPE_E8M23: launch_rand_fill_kernel<mag_e8m23_t, true>(r, cmd); break;
            case MAG_DTYPE_E5M10: launch_rand_fill_kernel<half, true>(r, cmd); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }
}
