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

#include "mag_cuda_reduction.cuh"

#include <core/mag_reduce_plan.h>

#include <cuda/std/limits>
#include <type_traits>
#include <stdint.h>

namespace mag {
    template <typename T> [[nodiscard]] __device__ __forceinline__ T fn_min(T a, T b) { return a < b ? a : b; }
    template <> [[nodiscard]] __device__ __forceinline__ float fn_min<float>(float a, float b) { return fminf(a, b); }
    template <> [[nodiscard]] __device__ __forceinline__ double fn_min<double>(double a, double b) { return fmin(a, b); }

    template <typename T> [[nodiscard]] __device__ __forceinline__ T fn_max(T a, T b) { return a > b ? a : b; }
    template <> [[nodiscard]] __device__ __forceinline__ float fn_max<float>(float a, float b) { return fmaxf(a, b);}
    template <> [[nodiscard]] __device__ __forceinline__ double fn_max<double>(double a, double b) {return fmax(a, b); }

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_mean {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return acc_t{}; }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a + b; }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, int64_t red_prod) const {
            acc /= static_cast<acc_t>(red_prod);
            return static_cast<out_t>(acc);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_sum {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return acc_t{}; }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a + b; }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const {
            return static_cast<out_t>(acc);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_prod {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return static_cast<acc_t>(1); }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return a * b; }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const {
            return static_cast<out_t>(acc);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_min {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const {
            if constexpr (std::is_floating_point_v<acc_t>) return cuda::std::numeric_limits<acc_t>::infinity();
            else return cuda::std::numeric_limits<acc_t>::max();
        }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return fn_min<acc_t>(a, b); }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const {
            return static_cast<out_t>(acc);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_max {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const {
            if constexpr (std::is_floating_point_v<acc_t>) return -cuda::std::numeric_limits<acc_t>::infinity();
            else return cuda::std::numeric_limits<acc_t>::lowest();
        }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return fn_max<acc_t>(a, b); }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const {
            return static_cast<out_t>(acc);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_all {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return static_cast<acc_t>(1); }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x != in_t{}); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return static_cast<acc_t>(a && b); }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const {
            return static_cast<out_t>(acc != 0);
        }
    };

    template <typename scalar_in_t, typename scalar_out_t, typename acc_in_t>
    struct op_any {
        using in_t  = scalar_in_t;
        using out_t = scalar_out_t;
        using acc_t = acc_in_t;

        [[nodiscard]] __device__ __forceinline__ acc_t init() const { return static_cast<acc_t>(0); }
        [[nodiscard]] __device__ __forceinline__ acc_t transform(in_t x) const { return static_cast<acc_t>(x != in_t{}); }
        [[nodiscard]] __device__ __forceinline__ acc_t reduce(acc_t a, acc_t b) const { return static_cast<acc_t>(a || b); }
        [[nodiscard]] __device__ __forceinline__ out_t finalize(acc_t acc, [[maybe_unused]] int64_t red_prod) const {
            return static_cast<out_t>(acc != 0);
        }
    };

    template <typename op_t>
    __global__ static void reduce_op_kernel(
        op_t op,
        int64_t n,
        typename op_t::out_t *__restrict__ o,
        const typename op_t::in_t *__restrict__ x,
        mag_reduce_plan_t plan
    ) {
        int64_t oi = static_cast<int64_t>(blockIdx.x);
        if (oi >= n) return;
        const int64_t base = mag_reduce_plan_to_offset(&plan, oi);
        typename op_t::acc_t acc = op.init();
        for (int64_t ri = threadIdx.x; ri < plan.red_prod; ri += blockDim.x) {
            int64_t t = ri;
            int64_t xi = base;
            #pragma unroll
            for (int64_t k = plan.rank - 1; k >= 0; --k) {
                const int64_t sz = plan.red_sizes[k];
                const int64_t j  = t % sz;
                t /= sz;
                xi += j * plan.red_strides[k];
            }
            acc = op.reduce(acc, op.transform(x[xi]));
        }
        extern __shared__ uint8_t smem_raw[];
        auto *smem = reinterpret_cast<typename op_t::acc_t *>(smem_raw);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x>>1; stride > 0; stride >>= 1) {   // blockDim.x is guaranteed power-of-two by launcher
            if (threadIdx.x < stride)
                smem[threadIdx.x] = op.reduce(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
            o[oi] = op.finalize(smem[0], plan.red_prod);
    }

    template <typename op_t>
    static void launch_reduce_op(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        int64_t n = mag_tensor_numel(r);
        const auto *plan = static_cast<const mag_reduce_plan_t *>(mag_op_attr_unwrap_ptr(cmd.attrs[0]));
        int threads = REDUCTION_BLOCK_SIZE;
        if (threads < 1) threads = 1;
        if (plan->red_prod < threads)
            threads = static_cast<int>(mag_next_pow2_u32(static_cast<uint32_t>(plan->red_prod > 0 ? plan->red_prod : 1)));
        if (threads > REDUCTION_BLOCK_SIZE) threads = REDUCTION_BLOCK_SIZE;
        if (threads < 1) threads = 1;
        size_t shmem = sizeof(typename op_t::acc_t)*static_cast<size_t>(threads);
        auto *pr = reinterpret_cast<typename op_t::out_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const typename op_t::in_t *>(mag_tensor_data_ptr(x));
        reduce_op_kernel<op_t><<<static_cast<unsigned>(n), threads, shmem>>>(op_t{}, n, pr, px, *plan);
    }

    template <template <typename, typename, typename> typename op_t>
    static void impl_reduce_op_fp(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(r->dtype == x->dtype);
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_reduce_op<op_t<float, float, double>>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_reduce_op<op_t<half, half, float>>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_reduce_op<op_t<__nv_bfloat16, __nv_bfloat16, float>>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for floating reduction op");
        }
    }

    template <template <typename, typename, typename> typename op_t>
    static void impl_reduce_op_sumprod(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_reduce_op<op_t<float, float, double>>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_reduce_op<op_t<half, half, float>>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_reduce_op<op_t<__nv_bfloat16, __nv_bfloat16, float>>(cmd); break;
            case MAG_DTYPE_BOOLEAN: launch_reduce_op<op_t<uint8_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_UINT8: launch_reduce_op<op_t<uint8_t,  uint64_t, uint64_t>>(cmd); break;
            case MAG_DTYPE_INT8: launch_reduce_op<op_t<int8_t,   int64_t,  int64_t>>(cmd); break;
            case MAG_DTYPE_UINT16: launch_reduce_op<op_t<uint16_t, uint64_t, uint64_t>>(cmd); break;
            case MAG_DTYPE_INT16: launch_reduce_op<op_t<int16_t,  int64_t,  int64_t>>(cmd); break;
            case MAG_DTYPE_UINT32: launch_reduce_op<op_t<uint32_t, uint64_t, uint64_t>>(cmd); break;
            case MAG_DTYPE_INT32: launch_reduce_op<op_t<int32_t,  int64_t,  int64_t>>(cmd); break;
            case MAG_DTYPE_UINT64: launch_reduce_op<op_t<uint64_t, uint64_t, uint64_t>>(cmd); break;
            case MAG_DTYPE_INT64: launch_reduce_op<op_t<int64_t,  int64_t,  int64_t>>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for sum/prod reduction op");
        }
    }

    template <template <typename, typename, typename> typename op_t>
    static void impl_reduce_op_minmax(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(r->dtype == x->dtype);
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_reduce_op<op_t<float, float, float>>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_reduce_op<op_t<half, half, half>>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_reduce_op<op_t<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>(cmd); break;
            case MAG_DTYPE_BOOLEAN: launch_reduce_op<op_t<uint8_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_UINT8: launch_reduce_op<op_t<uint8_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_INT8: launch_reduce_op<op_t<int8_t, int8_t, int8_t>>(cmd); break;
            case MAG_DTYPE_UINT16: launch_reduce_op<op_t<uint16_t, uint16_t, uint16_t>>(cmd); break;
            case MAG_DTYPE_INT16: launch_reduce_op<op_t<int16_t, int16_t, int16_t>>(cmd); break;
            case MAG_DTYPE_UINT32: launch_reduce_op<op_t<uint32_t, uint32_t, uint32_t>>(cmd); break;
            case MAG_DTYPE_INT32: launch_reduce_op<op_t<int32_t, int32_t, int32_t>>(cmd); break;
            case MAG_DTYPE_UINT64: launch_reduce_op<op_t<uint64_t, uint64_t, uint64_t>>(cmd); break;
            case MAG_DTYPE_INT64: launch_reduce_op<op_t<int64_t, int64_t, int64_t>>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for min/max reduction op");
        }
    }

    template <template <typename, typename, typename> typename op_t>
    static void impl_reduce_op_logical(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(r->dtype == MAG_DTYPE_BOOLEAN);
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_reduce_op<op_t<float, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_reduce_op<op_t<half, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_reduce_op<op_t<__nv_bfloat16, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_BOOLEAN: launch_reduce_op<op_t<uint8_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_UINT8: launch_reduce_op<op_t<uint8_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_INT8: launch_reduce_op<op_t<int8_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_UINT16: launch_reduce_op<op_t<uint16_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_INT16: launch_reduce_op<op_t<int16_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_UINT32: launch_reduce_op<op_t<uint32_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_INT32: launch_reduce_op<op_t<int32_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_UINT64: launch_reduce_op<op_t<uint64_t, uint8_t, uint8_t>>(cmd); break;
            case MAG_DTYPE_INT64: launch_reduce_op<op_t<int64_t, uint8_t, uint8_t>>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for logical reduction op");
        }
    }

    void reduce_op_mean(const mag_command_t &cmd) { impl_reduce_op_fp<op_mean>(cmd); }
    void reduce_op_sum (const mag_command_t &cmd) { impl_reduce_op_sumprod<op_sum>(cmd); }
    void reduce_op_prod(const mag_command_t &cmd) { impl_reduce_op_sumprod<op_prod>(cmd); }
    void reduce_op_min (const mag_command_t &cmd) { impl_reduce_op_minmax<op_min>(cmd); }
    void reduce_op_max (const mag_command_t &cmd) { impl_reduce_op_minmax<op_max>(cmd); }
    void reduce_op_all (const mag_command_t &cmd) { impl_reduce_op_logical<op_all>(cmd); }
    void reduce_op_any (const mag_command_t &cmd) { impl_reduce_op_logical<op_any>(cmd); }

    struct arg_acc_f32 {
        float val;
        int64_t idx;
        bool set;
    };

    struct arg_acc_i64 {
        int64_t val;
        int64_t idx;
        bool set;
    };

    template <bool is_max>
    __device__ __forceinline__ arg_acc_f32 merge_arg_f32(arg_acc_f32 a, arg_acc_f32 b) {
        if (!a.set) return b;
        if (!b.set) return a;
        if constexpr (is_max) {
            if (b.val > a.val) return b;
            if (a.val > b.val) return a;
        } else {
            if (b.val < a.val) return b;
            if (a.val < b.val) return a;
        }
        return a.idx <= b.idx ? a : b;
    }

    template <bool is_max>
    __device__ __forceinline__ arg_acc_i64 merge_arg_i64(arg_acc_i64 a, arg_acc_i64 b) {
        if (!a.set) return b;
        if (!b.set) return a;
        if constexpr (is_max) {
            if (b.val > a.val) return b;
            if (a.val > b.val) return a;
        } else {
            if (b.val < a.val) return b;
            if (a.val < b.val) return a;
        }
        return a.idx <= b.idx ? a : b;
    }

    template <typename T>
    __device__ __forceinline__ float arg_to_cmp_float(T x) {
        if constexpr (std::is_same_v<T, float>) return x;
        else if constexpr (std::is_same_v<T, half>) return __half2float(x);
        else return __bfloat162float(x);
    }

    template <typename T, bool is_max>
    __device__ __forceinline__ arg_acc_f32 reduce_arg_acc_f32(arg_acc_f32 acc, T x, int64_t ri) {
        float xv = arg_to_cmp_float(x);
        if (!acc.set || (is_max ? xv > acc.val : xv < acc.val)) {
            acc.val = xv;
            acc.idx = ri;
            acc.set = true;
        }
        return acc;
    }

    template <typename T, bool is_max>
    __device__ __forceinline__ arg_acc_i64 reduce_arg_acc_i64(arg_acc_i64 acc, T x, int64_t ri) {
        int64_t xv = static_cast<int64_t>(x);
        if (!acc.set || (is_max ? xv > acc.val : xv < acc.val)) {
            acc.val = xv;
            acc.idx = ri;
            acc.set = true;
        }
        return acc;
    }

    template <typename T, bool is_max>
    __global__ static void reduce_arg_op_kernel(
        int64_t n,
        int64_t *__restrict__ o,
        const T *__restrict__ x,
        mag_reduce_plan_t plan
    ) {
        int64_t oi = static_cast<int64_t>(blockIdx.x);
        if (oi >= n) return;
        const int64_t base = mag_reduce_plan_to_offset(&plan, oi);
        arg_acc_f32 acc {};
        for (int64_t ri = threadIdx.x; ri < plan.red_prod; ri += blockDim.x) {
            int64_t t = ri;
            int64_t xi = base;
            #pragma unroll
            for (int64_t k = plan.rank - 1; k >= 0; --k) {
                const int64_t sz = plan.red_sizes[k];
                const int64_t j  = t % sz;
                t /= sz;
                xi += j * plan.red_strides[k];
            }
            acc = reduce_arg_acc_f32<T, is_max>(acc, x[xi], ri);
        }
        extern __shared__ uint8_t smem_raw[];
        auto *smem = reinterpret_cast<arg_acc_f32 *>(smem_raw);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x>>1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] = merge_arg_f32<is_max>(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
            o[oi] = smem[0].set ? smem[0].idx : 0;
    }

    template <typename T, bool is_max>
    __global__ static void reduce_arg_op_kernel_int(
        int64_t n,
        int64_t *__restrict__ o,
        const T *__restrict__ x,
        mag_reduce_plan_t plan
    ) {
        int64_t oi = static_cast<int64_t>(blockIdx.x);
        if (oi >= n) return;
        const int64_t base = mag_reduce_plan_to_offset(&plan, oi);
        arg_acc_i64 acc {};
        for (int64_t ri = threadIdx.x; ri < plan.red_prod; ri += blockDim.x) {
            int64_t t = ri;
            int64_t xi = base;
            #pragma unroll
            for (int64_t k = plan.rank - 1; k >= 0; --k) {
                const int64_t sz = plan.red_sizes[k];
                const int64_t j  = t % sz;
                t /= sz;
                xi += j * plan.red_strides[k];
            }
            acc = reduce_arg_acc_i64<T, is_max>(acc, x[xi], ri);
        }
        extern __shared__ uint8_t smem_raw[];
        auto *smem = reinterpret_cast<arg_acc_i64 *>(smem_raw);
        smem[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x>>1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] = merge_arg_i64<is_max>(smem[threadIdx.x], smem[threadIdx.x + stride]);
            __syncthreads();
        }
        if (threadIdx.x == 0)
            o[oi] = smem[0].set ? smem[0].idx : 0;
    }

    template <typename T, bool is_max>
    static void launch_reduce_arg_op(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        int64_t n = mag_tensor_numel(r);
        const auto *plan = static_cast<const mag_reduce_plan_t *>(mag_op_attr_unwrap_ptr(cmd.attrs[0]));
        int threads = REDUCTION_BLOCK_SIZE;
        if (threads < 1) threads = 1;
        if (plan->red_prod < threads)
            threads = static_cast<int>(mag_next_pow2_u32(static_cast<uint32_t>(plan->red_prod > 0 ? plan->red_prod : 1)));
        if (threads > REDUCTION_BLOCK_SIZE) threads = REDUCTION_BLOCK_SIZE;
        if (threads < 1) threads = 1;
        size_t shmem = sizeof(arg_acc_f32)*static_cast<size_t>(threads);
        auto *pr = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        reduce_arg_op_kernel<T, is_max><<<static_cast<unsigned>(n), threads, shmem>>>(n, pr, px, *plan);
    }

    template <typename T, bool is_max>
    static void launch_reduce_arg_op_int(const mag_command_t &cmd) {
        mag_tensor_t *r = cmd.out[0];
        const mag_tensor_t *x = cmd.in[0];
        int64_t n = mag_tensor_numel(r);
        const auto *plan = static_cast<const mag_reduce_plan_t *>(mag_op_attr_unwrap_ptr(cmd.attrs[0]));
        int threads = REDUCTION_BLOCK_SIZE;
        if (threads < 1) threads = 1;
        if (plan->red_prod < threads)
            threads = static_cast<int>(mag_next_pow2_u32(static_cast<uint32_t>(plan->red_prod > 0 ? plan->red_prod : 1)));
        if (threads > REDUCTION_BLOCK_SIZE) threads = REDUCTION_BLOCK_SIZE;
        if (threads < 1) threads = 1;
        size_t shmem = sizeof(arg_acc_i64)*static_cast<size_t>(threads);
        auto *pr = reinterpret_cast<int64_t *>(mag_tensor_data_ptr_mut(r));
        const auto *px = reinterpret_cast<const T *>(mag_tensor_data_ptr(x));
        reduce_arg_op_kernel_int<T, is_max><<<static_cast<unsigned>(n), threads, shmem>>>(n, pr, px, *plan);
    }

    template <bool is_max>
    static void impl_reduce_op_arg_fp(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(r->dtype == MAG_DTYPE_INT64);
        switch (x->dtype) {
            case MAG_DTYPE_FLOAT32: launch_reduce_arg_op<float, is_max>(cmd); break;
            case MAG_DTYPE_FLOAT16: launch_reduce_arg_op<half, is_max>(cmd); break;
            case MAG_DTYPE_BFLOAT16: launch_reduce_arg_op<__nv_bfloat16, is_max>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for argmin/argmax fp");
        }
    }

    template <bool is_max>
    static void impl_reduce_op_arg_int(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        mag_tensor_t *r = cmd.out[0];
        mag_assert2(r->dtype == MAG_DTYPE_INT64);
        switch (x->dtype) {
            case MAG_DTYPE_UINT8: launch_reduce_arg_op_int<uint8_t, is_max>(cmd); break;
            case MAG_DTYPE_INT8: launch_reduce_arg_op_int<int8_t, is_max>(cmd); break;
            case MAG_DTYPE_UINT16: launch_reduce_arg_op_int<uint16_t, is_max>(cmd); break;
            case MAG_DTYPE_INT16: launch_reduce_arg_op_int<int16_t, is_max>(cmd); break;
            case MAG_DTYPE_UINT32: launch_reduce_arg_op_int<uint32_t, is_max>(cmd); break;
            case MAG_DTYPE_INT32: launch_reduce_arg_op_int<int32_t, is_max>(cmd); break;
            case MAG_DTYPE_UINT64: launch_reduce_arg_op_int<uint64_t, is_max>(cmd); break;
            case MAG_DTYPE_INT64: launch_reduce_arg_op_int<int64_t, is_max>(cmd); break;
            default: mag_assert(false, "Unsupported dtype for argmin/argmax int");
        }
    }

    void reduce_op_argmin(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        if (mag_tensor_is_floating_point_typed(x))
            impl_reduce_op_arg_fp<false>(cmd);
        else
            impl_reduce_op_arg_int<false>(cmd);
    }

    void reduce_op_argmax(const mag_command_t &cmd) {
        const mag_tensor_t *x = cmd.in[0];
        if (mag_tensor_is_floating_point_typed(x))
            impl_reduce_op_arg_fp<true>(cmd);
        else
            impl_reduce_op_arg_int<true>(cmd);
    }
}
