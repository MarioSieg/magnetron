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

#include "mag_cuda_fill.cuh"

#include <core/mag_prng_philox4x32.h>

#include <type_traits>

namespace mag {
    template <typename scalar_t, const bool contig>
    __global__ static void fill_kernel(
        int64_t n,
        scalar_t *__restrict__ o,
        scalar_t v,
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

    template <typename scalar_t, const bool contig>
    __global__ static void masked_fill_kernel(
        int64_t n,
        scalar_t *__restrict__ o,
        const uint8_t *__restrict__ mask,
        scalar_t v,
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

    template <typename scalar_t>
    static void launch_fill_kernel(mag_tensor_t *r, const mag_command_t &cmd, const mag_tensor_t *mask = nullptr) {
        auto *o = reinterpret_cast<scalar_t *>(mag_tensor_data_ptr_mut(r));
        auto v = unpack_param<scalar_t>(cmd.attrs, 0);
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        bool rc_cont = mag_tensor_is_contiguous(r);
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        if (mask) {
            const auto *pm = reinterpret_cast<const uint8_t *>(mag_tensor_data_ptr(mask));
            mag_coords_iter_t mc;
            mag_coords_iter_init(&mc, &mask->coords);
            if (rc_cont) masked_fill_kernel<scalar_t, true><<<blocks, FILL_BLOCK_SIZE>>>(n, o, pm, v, rc, mc);
            else  masked_fill_kernel<scalar_t, false><<<blocks, FILL_BLOCK_SIZE>>>(n, o, pm, v, rc, mc);
        } else {
            if (rc_cont) fill_kernel<scalar_t, true><<<blocks, FILL_BLOCK_SIZE>>>(n, o, v, rc);
            else fill_kernel<scalar_t, false><<<blocks, FILL_BLOCK_SIZE>>>(n, o, v, rc);
        }
    }

    template <typename scalar_t, const bool is_cont, const bool normal>
    __global__ static void fill_random_kernel(
        int64_t n,
        scalar_t *__restrict__ o,
        scalar_t p0,
        scalar_t p1,
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
            mag_philox4x32_float32x4_t r;
            if constexpr (normal) r = mag_philox4x32_next_float32x4_normal(&stream, p0, p1);
            else r = mag_philox4x32_next_float32x4_uniform(&stream, p0, p1);
            int64_t mk = n-base;
            if (mk > 4) mk = 4;
            if constexpr (is_cont) {
                #pragma unroll
                for (int64_t k=0; k < mk; ++k)
                    o[base+k] = static_cast<scalar_t>(r.v[k]);
            } else {
                #pragma unroll
                for (int64_t k=0; k < mk; ++k) {
                    int64_t ri = mag_coords_iter_to_offset(&rc, base+k);
                    o[ri] = static_cast<scalar_t>(r.v[k]);
                }
            }
        }
    }

    template <typename scalar_t, const bool normal>
    static void launch_rand_fill_kernel(mag_tensor_t *r, const mag_command_t &cmd) {
        auto *o = reinterpret_cast<scalar_t *>(mag_tensor_data_ptr_mut(r));
        auto p0 = unpack_param<scalar_t>(cmd.attrs, 0);
        auto p1 = unpack_param<scalar_t>(cmd.attrs, 1);
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (((n+3)>>2)+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        uint64_t seed = global_seed.load(std::memory_order_relaxed);
        uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
        if (mag_tensor_is_contiguous(r)) {
            fill_random_kernel<scalar_t, true, normal><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p0, p1, seed, subseq, rc);
        } else {
            fill_random_kernel<scalar_t, false, normal><<<blocks, FILL_BLOCK_SIZE>>>(n, o, p0, p1, seed, subseq, rc);
        }
    }

    void fill_op_fill(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_fill_kernel<float>(r, cmd); break;
            case MAG_DTYPE_FLOAT16: launch_fill_kernel<half>(r, cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_fill_kernel<__nv_bfloat16>(r, cmd); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_fill_kernel<uint8_t>(r, cmd); break;
            case MAG_DTYPE_INT8: launch_fill_kernel<int8_t>(r, cmd); break;
            case MAG_DTYPE_UINT16: launch_fill_kernel<uint16_t>(r, cmd); break;
            case MAG_DTYPE_INT16: launch_fill_kernel<int16_t>(r, cmd); break;
            case MAG_DTYPE_UINT32: launch_fill_kernel<uint32_t>(r, cmd); break;
            case MAG_DTYPE_INT32: launch_fill_kernel<int32_t>(r, cmd); break;
            case MAG_DTYPE_UINT64: launch_fill_kernel<uint64_t>(r, cmd); break;
            case MAG_DTYPE_INT64: launch_fill_kernel<int64_t>(r, cmd); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_masked_fill(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_tensor_t *mask = static_cast<mag_tensor_t *>(mag_op_attr_unwrap_ptr(cmd.attrs[0])); // TODO: pass in cmd in why the fuck are these here
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_fill_kernel<float>(r, cmd, mask); break;
            case MAG_DTYPE_FLOAT16: launch_fill_kernel<half>(r, cmd, mask); break;
            case MAG_DTYPE_BFLOAT16: launch_fill_kernel<__nv_bfloat16>(r, cmd, mask); break;
            case MAG_DTYPE_BOOLEAN:
            case MAG_DTYPE_UINT8: launch_fill_kernel<uint8_t>(r, cmd, mask); break;
            case MAG_DTYPE_INT8: launch_fill_kernel<int8_t>(r, cmd, mask); break;
            case MAG_DTYPE_UINT16: launch_fill_kernel<uint16_t>(r, cmd, mask); break;
            case MAG_DTYPE_INT16: launch_fill_kernel<int16_t>(r, cmd, mask); break;
            case MAG_DTYPE_UINT32: launch_fill_kernel<uint32_t>(r, cmd, mask); break;
            case MAG_DTYPE_INT32: launch_fill_kernel<int32_t>(r, cmd, mask); break;
            case MAG_DTYPE_UINT64: launch_fill_kernel<uint64_t>(r, cmd, mask); break;
            case MAG_DTYPE_INT64: launch_fill_kernel<int64_t>(r, cmd, mask); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_fill_rand_uniform(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_rand_fill_kernel<float, false>(r, cmd); break;
            case MAG_DTYPE_FLOAT16: launch_rand_fill_kernel<half, false>(r, cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_rand_fill_kernel<__nv_bfloat16, false>(r, cmd); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    void fill_op_fill_rand_normal(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_rand_fill_kernel<float, true>(r, cmd); break;
            case MAG_DTYPE_FLOAT16: launch_rand_fill_kernel<half, true>(r, cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_rand_fill_kernel<__nv_bfloat16, true>(r, cmd); break;
            default: mag_assert(false, "Unsupported data type in binary operation");
        }
    }

    __global__ static void bernoulli_kernel(
        int64_t n,
        uint8_t *__restrict__ o,
        float p,
        uint64_t seed,
        uint64_t subseq,
        mag_coords_iter_t rc,
        bool contig
    ) {
        int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        for (; ti < n; ti += step) {
            mag_philox4x32_stream_t stream;
            mag_philox4x32_stream_seed(&stream, seed, subseq + static_cast<uint64_t>(ti));
            float u = mag_philox4x32_next_float32(&stream);
            uint8_t v = (u < p) ? 1u : 0u;
            if (contig)
                o[ti] = v;
            else {
                int64_t ri = mag_coords_iter_to_offset(&rc, ti);
                o[ri] = v;
            }
        }
    }

    void fill_op_rand_bernoulli(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(r->dtype == MAG_DTYPE_BOOLEAN);
        float p = static_cast<float>(mag_op_attr_unwrap_float64(cmd.attrs[0]));
        auto *o = reinterpret_cast<uint8_t *>(mag_tensor_data_ptr_mut(r));
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        uint64_t seed = global_seed.load(std::memory_order_relaxed);
        uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
        if (mag_tensor_is_contiguous(r))
            bernoulli_kernel<<<blocks, FILL_BLOCK_SIZE>>>(n, o, p, seed, subseq, rc, true);
        else
            bernoulli_kernel<<<blocks, FILL_BLOCK_SIZE>>>(n, o, p, seed, subseq, rc, false);
    }

    template <typename T>
    __global__ static void rand_perm_single_thread_kernel(
        int64_t n,
        T *__restrict__ o,
        uint64_t seed,
        uint64_t subseq,
        mag_coords_iter_t rc,
        bool contig
    ) {
        if (blockIdx.x != 0 || threadIdx.x != 0) return;
        mag_philox4x32_stream_t stream;
        mag_philox4x32_stream_seed(&stream, seed, subseq);
        for (int64_t i=0; i < n; ++i) {
            int64_t ri = contig ? i : mag_coords_iter_to_offset(&rc, i);
            o[ri] = static_cast<T>(i);
        }
        for (int64_t i=0; i < n-1; ++i) {
            int64_t j = i + static_cast<int64_t>(mag_philox4x32_next_uint64(&stream) % static_cast<uint64_t>(n - i));
            int64_t off_i = contig ? i : mag_coords_iter_to_offset(&rc, i);
            int64_t off_j = contig ? j : mag_coords_iter_to_offset(&rc, j);
            T tmp = o[off_i];
            o[off_i] = o[off_j];
            o[off_j] = tmp;
        }
    }

    void fill_op_rand_perm(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        bool cont = mag_tensor_is_contiguous(r);
        int64_t n = mag_tensor_numel(r);
        uint64_t seed = global_seed.load(std::memory_order_relaxed);
        uint64_t subseq = global_subseq.fetch_add(1, std::memory_order_relaxed);
        void *raw = reinterpret_cast<void *>(mag_tensor_data_ptr_mut(r));
        switch (r->dtype) {
        case MAG_DTYPE_UINT8: rand_perm_single_thread_kernel<uint8_t><<<1, 1>>>(n, reinterpret_cast<uint8_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_INT8: rand_perm_single_thread_kernel<int8_t><<<1, 1>>>(n, reinterpret_cast<int8_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_UINT16: rand_perm_single_thread_kernel<uint16_t><<<1, 1>>>(n, reinterpret_cast<uint16_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_INT16: rand_perm_single_thread_kernel<int16_t><<<1, 1>>>(n, reinterpret_cast<int16_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_UINT32: rand_perm_single_thread_kernel<uint32_t><<<1, 1>>>(n, reinterpret_cast<uint32_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_INT32: rand_perm_single_thread_kernel<int32_t><<<1, 1>>>(n, reinterpret_cast<int32_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_UINT64: rand_perm_single_thread_kernel<uint64_t><<<1, 1>>>(n, reinterpret_cast<uint64_t *>(raw), seed, subseq, rc, cont); break;
        case MAG_DTYPE_INT64: rand_perm_single_thread_kernel<int64_t><<<1, 1>>>(n, reinterpret_cast<int64_t *>(raw), seed, subseq, rc, cont); break;
        default: mag_assert(false, "Unsupported dtype for rand_perm");
        }
    }

    template <typename scalar_t>
    __global__ static void arange_kernel(
        int64_t n,
        scalar_t *__restrict__ o,
        double start,
        double step,
        mag_coords_iter_t rc,
        bool contig
    ) {
        int64_t ti = static_cast<int64_t>(blockIdx.x)*static_cast<int64_t>(blockDim.x) + threadIdx.x;
        int64_t step_th = static_cast<int64_t>(blockDim.x)*static_cast<int64_t>(gridDim.x);
        for (; ti < n; ti += step_th) {
            double v = start + static_cast<double>(ti)*step;
            scalar_t out;
            if constexpr (std::is_same_v<scalar_t, float>)
                out = static_cast<scalar_t>(v);
            else if constexpr (std::is_same_v<scalar_t, half>)
                out = __float2half(static_cast<float>(v));
            else if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>)
                out = __float2bfloat16(static_cast<float>(v));
            else if constexpr (std::is_integral_v<scalar_t>)
                out = static_cast<scalar_t>(v);
            else
                out = static_cast<scalar_t>(v);
            if (contig)
                o[ti] = out;
            else {
                int64_t ri = mag_coords_iter_to_offset(&rc, ti);
                o[ri] = out;
            }
        }
    }

    template <typename scalar_t>
    static void launch_arange(mag_tensor_t *r, const mag_command_t &cmd) {
        auto *o = reinterpret_cast<scalar_t *>(mag_tensor_data_ptr_mut(r));
        double start = mag_op_attr_unwrap_float64(cmd.attrs[0]);
        double step = mag_op_attr_unwrap_float64(cmd.attrs[1]);
        int64_t n = mag_tensor_numel(r);
        int64_t blocks = (n+FILL_BLOCK_SIZE-1)/FILL_BLOCK_SIZE;
        mag_coords_iter_t rc;
        mag_coords_iter_init(&rc, &r->coords);
        if (mag_tensor_is_contiguous(r))
            arange_kernel<scalar_t><<<blocks, FILL_BLOCK_SIZE>>>(n, o, start, step, rc, true);
        else
            arange_kernel<scalar_t><<<blocks, FILL_BLOCK_SIZE>>>(n, o, start, step, rc, false);
    }

    void fill_op_arange(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_arange<float>(r, cmd); break;
            case MAG_DTYPE_FLOAT16: launch_arange<half>(r, cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_arange<__nv_bfloat16>(r, cmd); break;
            case MAG_DTYPE_UINT8: launch_arange<uint8_t>(r, cmd); break;
            case MAG_DTYPE_INT8: launch_arange<int8_t>(r, cmd); break;
            case MAG_DTYPE_UINT16: launch_arange<uint16_t>(r, cmd); break;
            case MAG_DTYPE_INT16: launch_arange<int16_t>(r, cmd); break;
            case MAG_DTYPE_UINT32: launch_arange<uint32_t>(r, cmd); break;
            case MAG_DTYPE_INT32: launch_arange<int32_t>(r, cmd); break;
            case MAG_DTYPE_UINT64: launch_arange<uint64_t>(r, cmd); break;
            case MAG_DTYPE_INT64: launch_arange<int64_t>(r, cmd); break;
            default: mag_assert(false, "Unsupported dtype for arange");
        }
    }
}
