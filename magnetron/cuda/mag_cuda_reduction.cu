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

#include "mag_cuda_reduction.cuh"

#include <core/mag_reduce_plan.h>

#include <cuda/std/limits>

namespace mag {
    template <typename scalar_in_t, typename acc_in_t> requires is_dtype<scalar_in_t> && std::is_arithmetic_v<acc_in_t>
    struct op_mean {
        using scalar_t = scalar_in_t;
        using acc_t = acc_in_t;
        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return acc_t{}; }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(scalar_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a + b; }
        [[nodiscard]] __device__ __forceinline__ scalar_t finalize(acc_t acc, int64_t red_prod) const {
            acc /= static_cast<acc_t>(red_prod);
            return static_cast<scalar_t>(acc);
        }
    };

    template <typename scalar_in_t, typename acc_in_t> requires is_dtype<scalar_in_t> && std::is_arithmetic_v<acc_in_t>
    struct op_sum {
        using scalar_t = scalar_in_t;
        using acc_t = acc_in_t;
        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return acc_t{}; }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(scalar_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a + b; }
        [[nodiscard]] __device__ __forceinline__ scalar_t finalize(acc_t acc, int64_t red_prod) const { return static_cast<scalar_t>(acc); }
    };

    template <typename scalar_in_t, typename acc_in_t> requires is_dtype<scalar_in_t> && std::is_arithmetic_v<acc_in_t>
    struct op_prod {
        using scalar_t = scalar_in_t;
        using acc_t = acc_in_t;
        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return acc_t{1.0}; }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(scalar_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a * b; }
        [[nodiscard]] __device__ __forceinline__ scalar_t finalize(acc_t acc, int64_t red_prod) const { return static_cast<scalar_t>(acc); }
    };

    template <typename scalar_in_t, typename acc_in_t> requires is_dtype<scalar_in_t> && std::is_arithmetic_v<acc_in_t>
    struct op_min {
        using scalar_t = scalar_in_t;
        using acc_t = acc_in_t;
        [[nodiscard]] __device__ __forceinline__ acc_t init() const {
            if constexpr (std::is_floating_point_v<acc_t>) return cuda::std::numeric_limits<acc_t>::infinity();
            else return cuda::std::numeric_limits<acc_t>::max();
        }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(scalar_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a < b ? a : b; }
        [[nodiscard]] __device__ __forceinline__ scalar_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const { return static_cast<scalar_t>(acc); }
    };

    template <typename scalar_in_t, typename acc_in_t> requires is_dtype<scalar_in_t> && std::is_arithmetic_v<acc_in_t>
    struct op_max {
        using scalar_t = scalar_in_t;
        using acc_t = acc_in_t;
        [[nodiscard]] __device__ __forceinline__ acc_t init() const {
            if constexpr (std::is_floating_point_v<acc_t>) return -cuda::std::numeric_limits<acc_t>::infinity();
            else return cuda::std::numeric_limits<acc_t>::lowest();
        }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(scalar_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a > b ? a : b; }
        [[nodiscard]] __device__ __forceinline__ scalar_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const { return static_cast<scalar_t>(acc); }
    };

    template <typename op_t>
    __global__ static void reduce_op_kernel(
        op_t op,
        int64_t n,
        typename op_t::scalar_t *__restrict__ o,
        const typename op_t::scalar_t *__restrict__ x,
        mag_reduce_plan_t plan
    ) {
        auto i = static_cast<int64_t>(blockIdx.x);
        if (i >= n) return;
        int64_t b = mag_reduce_plan_to_offset(&plan, i);
        typename op_t::acc_t acc = op.init(); /* Partial per thread accumulation */
        for (int64_t ri=threadIdx.x; ri < plan.red_prod; ri += blockDim.x) {
            int64_t t = ri;
            int64_t xi = b;
            #pragma unroll
            for (int64_t k=plan.rank-1; k >= 0; --k) {
                int64_t sz  = plan.red_sizes[k];
                int64_t j = t%sz;
                t /= sz;
                xi += j*plan.red_strides[k];
            }
            typename op_t::scalar_t v = x[xi];
            acc = op.reduce(acc, op.transform(v));
        }
        extern __shared__ uint8_t smem_raw[]; /* Block wide reduction */
        auto *smem = reinterpret_cast<typename op_t::acc_t *>(smem_raw);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (int stride=blockDim.x>>1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                smem[threadIdx.x] = op.reduce(smem[threadIdx.x], smem[threadIdx.x+stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) o[i] = op.finalize(smem[0], plan.red_prod);
    }

    template <typename op_t>
    static void launch_reduce_op(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        int64_t n = mag_tensor_get_numel(r);
        int64_t blocks = n;
        int64_t threads = REDUCTION_BLOCK_SIZE;
        auto *plan = static_cast<mag_reduce_plan_t *>(mag_op_attr_unwrap_ptr(cmd->attrs[0]));
        if (threads > plan->red_prod) threads = plan->red_prod;
        if (threads < 1) threads = 1;
        size_t shmem = sizeof(typename op_t::acc_t)*threads;
        auto *pr = static_cast<typename op_t::scalar_t *>(mag_tensor_get_data_ptr(r));
        auto *px = static_cast<const typename op_t::scalar_t *>(mag_tensor_get_data_ptr(x));
        reduce_op_kernel<op_t><<<blocks, threads, shmem>>>(op_t{}, n, pr, px, *plan);
    }

    template <template <typename, typename> typename op_t>
    static void impl_reduce_op(const mag_command_t *cmd) {
        mag_tensor_t *r = cmd->out[0];
        const mag_tensor_t *x = cmd->in[0];
        mag_assert2(r->dtype == x->dtype);
        switch (r->dtype) {
            case MAG_DTYPE_FLOAT32: launch_reduce_op<op_t<float, double>>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_reduce_op<op_t<half, float>>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for unary op");
        }
    }
    void reduce_op_mean(const mag_command_t *cmd) { impl_reduce_op<op_mean>(cmd); }
    void reduce_op_min(const mag_command_t *cmd) { impl_reduce_op<op_min>(cmd); }
    void reduce_op_max(const mag_command_t *cmd) { impl_reduce_op<op_max>(cmd); }
    void reduce_op_sum(const mag_command_t *cmd) { impl_reduce_op<op_sum>(cmd); }
    void reduce_op_prod(const mag_command_t *cmd) { impl_reduce_op<op_prod>(cmd); }
}